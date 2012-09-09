/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Texas Instruments AM3517
 **
 **    Used with ICCARM and AARM.
 **
 **    (c) Copyright IAR Systems 2011
 **
 **    $Revision: #10 $
 **
 ***************************************************************************/

#ifndef __IOAM3517_H
#define __IOAM3517_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4f = 79 dec */
#error This file should only be compiled by ICCARM/AARM
#endif


#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    AM3517 SPECIAL FUNCTION REGISTERS
 **
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

/* CONTROL_IDCODE */
typedef struct {
 	__REG32 																: 1;
 	__REG32 TI_IDM													:11;
 	__REG32 HAWKEYE													:16;
 	__REG32 VERSION													: 4;
} __control_idcode_bits;

/* PRM_REVISION */
typedef struct {
  __REG32 REV								  	: 8;
  __REG32                 			:24;
} __prm_revision_bits;

/* PRM_SYSCONFIG */
typedef struct {
  __REG32 AUTOIDLE					  	: 1;
  __REG32                 			:31;
} __prm_sysconfig_bits;

/* PRM_IRQSTATUS_MPU */
typedef struct {
  __REG32 WKUP_ST 					  	: 1;
  __REG32                 			: 1;
  __REG32 EVGENON_ST				  	: 1;
  __REG32 EVGENOFF_ST				  	: 1;
  __REG32 TRANSITION_ST 		  	: 1;
  __REG32 CORE_DPLL_ST			  	: 1;
  __REG32 PERIPH_DPLL_ST		  	: 1;
  __REG32 MPU_DPLL_ST				  	: 1;
  __REG32                 			:17;
  __REG32 SND_PERIPH_DPLL_ST  	: 1;
  __REG32                 			: 6;
} __prm_irqstatus_mpu_bits;

/* PRM_IRQENABLE_MPU */
typedef struct {
  __REG32 WKUP_EN 					  			: 1;
  __REG32                 					: 1;
  __REG32 EVGENON_EN				  			: 1;
  __REG32 EVGENOFF_EN				  			: 1;
  __REG32 RANSITION_EN  		  			: 1;
  __REG32 CORE_DPLL_RECAL_EN  			: 1;
  __REG32 PERIPH_DPLL_RECAL_EN 			: 1;
  __REG32 MPU_DPLL_RECAL_EN	  			: 1;
  __REG32                 					:17;
  __REG32 SND_PERIPH_DPLL_RECAL_EN	: 1;
  __REG32                 					: 6;
} __prm_irqenable_mpu_bits;

/* RM_RSTST_MPU */
typedef struct {
  __REG32 GLOBALCOLD_RST		  			: 1;
  __REG32 GLOBALWARM_RST		  			: 1;
  __REG32                 					: 9;
  __REG32 EMULATION_MPU_RST	  			: 1;
  __REG32                 					:20;
} __rm_rstst_mpu_bits;

/* PM_WKDEP_MPU */
typedef struct {
  __REG32 EN_CORE						  			: 1;
  __REG32                 					: 4;
  __REG32 EN_DSS						  			: 1;
  __REG32                 					: 1;
  __REG32 EN_PER						  			: 1;
  __REG32                 					:24;
} __pm_wkdep_mpu_bits;

/* PM_EVGENCTRL_MPU */
typedef struct {
  __REG32 ENABLE						  			: 1;
  __REG32 ONLOADMODE				  			: 2;
  __REG32 OFFLOADMODE				  			: 2;
  __REG32                 					:27;
} __pm_evgenctrl_mpu_bits;

/* PM_PWSTCTRL_MPU */
typedef struct {
  __REG32 POWERSTATE				  			: 2;
  __REG32 LOGICL1CACHERETSTATE 			: 1;
  __REG32 MEMORYCHANGE				 			: 1;
  __REG32                 					: 4;
  __REG32 L2CACHERETSTATE		  			: 1;
  __REG32                 					: 7;
  __REG32 L2CACHEONSTATE		  			: 2;
  __REG32                 					:14;
} __pm_pwstctrl_mpu_bits;

/* PM_PWSTST_MPU */
typedef struct {
  __REG32 POWERSTATEST			  			: 2;
  __REG32                 					:18;
  __REG32 INTRANSITION			  			: 1;
  __REG32                 					:11;
} __pm_pwstst_mpu_bits;

/* PM_PREPWSTST_MPU */
typedef struct {
  __REG32 LASTPOWERSTATEENTERED					: 2;
  __REG32 LASTLOGICL1CACHESTATEENTERED	: 1;
  __REG32                 							: 3;
  __REG32 LASTL2CACHESTATEENTERED				: 2;
  __REG32                 							:24;
} __pm_prepwstst_mpu_bits;

/* RM_RSTST_CORE */
typedef struct {
  __REG32 GLOBALCOLD_RST	: 1;
  __REG32 GLOBALWARM_RST	: 1;
  __REG32                 :30;
} __rm_rstst_core_bits;

/* PM_WKEN1_CORE */
typedef struct {
  __REG32                 : 9;
  __REG32 EN_MCBSP1				: 1;
  __REG32 EN_MCBSP5				: 1;
  __REG32 EN_GPT10				: 1;
  __REG32 EN_GPT11				: 1;
  __REG32 EN_UART1				: 1;
  __REG32 EN_UART2				: 1;
  __REG32 EN_I2C1					: 1;
  __REG32 EN_I2C2					: 1;
  __REG32 EN_I2C3					: 1;
  __REG32 EN_MCSPI1 			: 1;
  __REG32 EN_MCSPI2				: 1;
  __REG32 EN_MCSPI3				: 1;
  __REG32 EN_MCSPI4				: 1;
  __REG32                 : 2;
  __REG32 EN_MMC1					: 1;
  __REG32 EN_MMC2					: 1;
  __REG32                 : 4;
  __REG32 EN_MMC3					: 1;
  __REG32                 : 1;
} __pm_wken1_core_bits;

/* PM_MPUGRPSEL1_CORE */
typedef struct {
  __REG32                 : 9;
  __REG32 GRPSEL_MCBSP1		: 1;
  __REG32 GRPSEL_MCBSP5		: 1;
  __REG32 GRPSEL_GPT10		: 1;
  __REG32 GRPSEL_GPT11		: 1;
  __REG32 GRPSEL_UART1		: 1;
  __REG32 GRPSEL_UART2		: 1;
  __REG32 GRPSEL_I2C1			: 1;
  __REG32 GRPSEL_I2C2			: 1;
  __REG32 GRPSEL_I2C3			: 1;
  __REG32 GRPSEL_MCSPI1		: 1;
  __REG32 GRPSEL_MCSPI2		: 1;
  __REG32 GRPSEL_MCSPI3		: 1;
  __REG32 GRPSEL_MCSPI4		: 1;
  __REG32                 : 2;
  __REG32 GRPSEL_MMC1			: 1;
  __REG32 GRPSEL_MMC2			: 1;
  __REG32                 : 4;
  __REG32 GRPSEL_MMC3     : 1;
  __REG32                 : 1;
} __pm_mpugrpsel1_core_bits;

/* PM_WKST1_CORE */
typedef struct {
  __REG32                 : 9;
  __REG32 ST_MCBSP1				: 1;
  __REG32 ST_MCBSP5				: 1;
  __REG32 ST_GPT10				: 1;
  __REG32 ST_GPT11				: 1;
  __REG32 ST_UART1				: 1;
  __REG32 ST_UART2				: 1;
  __REG32 ST_I2C1					: 1;
  __REG32 ST_I2C2					: 1;
  __REG32 ST_I2C3					: 1;
  __REG32 ST_MCSPI1				: 1;
  __REG32 ST_MCSPI2				: 1;
  __REG32 ST_MCSPI3				: 1;
  __REG32 ST_MCSPI4				: 1;
  __REG32                 : 2;
  __REG32 ST_MMC1					: 1;
  __REG32 ST_MMC2					: 1;
  __REG32                 : 4;
  __REG32 ST_MMC3		      : 1;
  __REG32                 : 1;
} __pm_wkst1_core_bits;

/* PM_WKST3_CORE */
typedef struct {
  __REG32                 : 2;
  __REG32 ST_USBTLL				: 1;
  __REG32                 :29;
} __pm_wkst3_core_bits;

/* PM_PWSTCTRL_CORE */
typedef struct {
  __REG32 POWERSTATE			: 2;
  __REG32 LOGICRETSTATE		: 1;
  __REG32 MEMORYCHANGE		: 1;
  __REG32                 : 4;
  __REG32 MEM1RETSTATE		: 1;
  __REG32 MEM2RETSTATE		: 1;
  __REG32                 : 6;
  __REG32 MEM1ONSTATE			: 2;
  __REG32 MEM2ONSTATE			: 2;
  __REG32                 :12;
} __pm_pwstctrl_core_bits;

/* PM_PWSTST_CORE */
typedef struct {
  __REG32 POWERSTATEST		: 2;
  __REG32                 :18;
  __REG32 INTRANSITION		: 1;
  __REG32                 :11;
}__pm_pwstst_core_bits;
	
/* PM_PREPWSTST_CORE */
typedef	struct {
  __REG32 LASTPOWERSTATEENTERED		: 2;
  __REG32 LASTLOGICSTATEENTERED		: 1;
  __REG32                 				: 1;
  __REG32 LASTMEM1STATEENTERED		: 2;
  __REG32 LASTMEM2STATEENTERED		: 2;
  __REG32                 				:24;
} __pm_prepwstst_core_bits;

/* PM_WKEN3_CORE */
typedef struct {
  __REG32 												: 2;
  __REG32 EN_USBTLL								: 1;
  __REG32                 				:29;
} __pm_wken3_core_bits;

/* PM_MPUGRPSEL3_CORE */
typedef struct {
  __REG32 												: 2;
  __REG32 GRPSEL_USBTLL						: 1;
  __REG32                 				:29;
} __pm_mpugrpsel3_core_bits;

/* RM_RSTST_SGX */
typedef struct {
  __REG32 GLOBALCOLD_RST					: 1;
  __REG32 GLOBALWARM_RST					: 1;
  __REG32                 				:30;
} __rm_rstst_sgx_bits;

/* PM_WKDEP_SGX */
typedef struct {
  __REG32                 				: 1;
  __REG32 EN_MPU									: 1;
  __REG32                 				: 2;
  __REG32 EN_WKUP									: 1;
  __REG32                 				:27;
} __pm_wkdep_sgx_bits;

/* PM_PWSTCTRL_SGX */
typedef struct {
  __REG32 POWERSTATE 	    				: 2;
  __REG32 LOGICRETSTATE						: 1;
  __REG32                 				: 5;
  __REG32 MEMRETSTATE							: 1;
  __REG32                 				: 7;
  __REG32 MEMONSTATE							: 2;
  __REG32                 				:14;
} __pm_pwstctrl_sgx_bits;

/* PM_PWSTST_SGX */
typedef struct {
  __REG32 POWERSTATEST    				: 2;
  __REG32                 				:18;
  __REG32 INTRANSITION						: 1;
  __REG32                 				:11;
} __pm_pwstst_sgx_bits;

/* PM_PREPWSTCTRL_SGX */
typedef struct {
  __REG32 LASTPOWERSTATEENTERED		: 2;
  __REG32                 				:30;
} __pm_prepwstctrl_sgx_bits;

/* PM_WKEN_WKUP */
typedef struct {
  __REG32 EN_GPT1									: 1;
  __REG32 EN_GPT12								: 1;
  __REG32                 				: 1;
  __REG32 EN_GPIO1								: 1;
  __REG32                 				:28;
} __pm_wken_wkup_bits;

/* PM_MPUGRPSEL_WKUP */
typedef struct {
  __REG32 GRPSEL_GPT1							: 1;
  __REG32 GRPSEL_GPT12						: 1;
  __REG32                 				: 1;
  __REG32 GRPSEL_GPIO1						: 1;
  __REG32                 				:28;
} __pm_mpugrpsel_wkup_bits;

/* PM_WKST_WKUP */
typedef struct {
  __REG32 ST_GPT1									: 1;
  __REG32 ST_GPT12								: 1;
  __REG32                 				: 1;
  __REG32 ST_GPIO1								: 1;
  __REG32                 				:28;
} __pm_wkst_wkup_bits;

/* PRM_CLKSEL */
typedef struct {
  __REG32 SYS_CLKIN_SEL						: 3;
  __REG32                 				:29;
} __prm_clksel_bits;

/* PRM_CLKOUT_CTRL */
typedef struct {
  __REG32                 				: 7;
  __REG32 CLKOUT_EN								: 1;
  __REG32                 				:24;
} __prm_clkout_ctrl_bits;

/* RM_RSTST_DSS */
typedef struct {
  __REG32 GLOBALCOLD_RST					: 1;
  __REG32 GLOBALWARM_RST					: 1;
  __REG32                 				:30;
} __rm_rstst_dss_bits;

/* PM_WKEN_DSS */
typedef struct {
  __REG32 EN_DSS									: 1;
  __REG32                 				:31;
} __pm_wken_dss_bits;

/* PM_WKDEP_DSS */
typedef struct {
  __REG32                 				: 1;
  __REG32 EN_MPU									: 1;
  __REG32                 				: 2;
  __REG32 EN_WKUP									: 1;
  __REG32                 				:27;
} __pm_wkdep_dss_bits;

/* PM_PWSTCTRL_DSS */
typedef struct {
  __REG32 POWERSTATE							: 2;
  __REG32 LOGICRETSTATE						: 1;
  __REG32                 				: 5;
  __REG32 MEMRETSTATE							: 1;
  __REG32                 				: 7;
  __REG32 MEMONSTATE							: 2;
  __REG32                 				:14;
} __pm_pwstctrl_dss_bits;

/* PM_PWSTST_DSS */
typedef struct {
  __REG32 POWERSTATEST						: 2;
  __REG32                 				:18;
  __REG32 INTRANSITION						: 1;
  __REG32                 				:11;
} __pm_pwstst_dss_bits;

/* PM_PREPWSTST_DSS */
typedef struct {
  __REG32 LASTPOWERSTATEENTERED		: 2;
  __REG32                 				:30;
} __pm_prepwstst_dss_bits;

/* RM_RSTST_PER */
typedef struct {
  __REG32 GLOBALCOLD_RST					: 1;
  __REG32 GLOBALWARM_RST					: 1;
  __REG32                 				:30;
} __rm_rstst_per_bits;

/* PM_WKEN_PER */
typedef struct {
  __REG32 EN_MCBSP2								: 1;
  __REG32 EN_MCBSP3								: 1;
  __REG32 EN_MCBSP4								: 1;
  __REG32 EN_GPT2									: 1;
  __REG32 EN_GPT3									: 1;
  __REG32 EN_GPT4									: 1;
  __REG32 EN_GPT5									: 1;
  __REG32 EN_GPT6									: 1;
  __REG32 EN_GPT7									: 1;
  __REG32 EN_GPT8									: 1;
  __REG32 EN_GPT9									: 1;
  __REG32 EN_UART3								: 1;
  __REG32 		 										: 1;
  __REG32 EN_GPIO2								: 1;
  __REG32 EN_GPIO3								: 1;
  __REG32 EN_GPIO4								: 1;
  __REG32 EN_GPIO5								: 1;
  __REG32 EN_GPIO6								: 1;
  __REG32        	        				:14;
} __pm_wken_per_bits;

/* PM_MPUGRPSEL_PER */
typedef struct {
  __REG32 GRPSEL_MCBSP2						: 1;
  __REG32 GRPSEL_MCBSP3						: 1;
  __REG32 GRPSEL_MCBSP4						: 1;
  __REG32 GRPSEL_GPT2							: 1;
  __REG32 GRPSEL_GPT3							: 1;
  __REG32 GRPSEL_GPT4							: 1;
  __REG32 GRPSEL_GPT5							: 1;
  __REG32 GRPSEL_GPT6							: 1;
  __REG32 GRPSEL_GPT7							: 1;
  __REG32 GRPSEL_GPT8							: 1;
  __REG32 GRPSEL_GPT9							: 1;
  __REG32 GRPSEL_UART3						: 1;
  __REG32 		 										: 1;
  __REG32 GRPSEL_GPIO2						: 1;
  __REG32 GRPSEL_GPIO3						: 1;
  __REG32 GRPSEL_GPIO4						: 1;
  __REG32 GRPSEL_GPIO5						: 1;
  __REG32 GRPSEL_GPIO6						: 1;
  __REG32        	        				:14;
} __pm_mpugrpsel_per_bits;

/* PM_WKST_PER */
typedef struct {
  __REG32 EN_MCBSP2								: 1;
  __REG32 EN_MCBSP3								: 1;
  __REG32 EN_MCBSP4								: 1;
  __REG32 ST_GPT2									: 1;
  __REG32 ST_GPT3									: 1;
  __REG32 ST_GPT4									: 1;
  __REG32 ST_GPT5									: 1;
  __REG32 ST_GPT6									: 1;
  __REG32 ST_GPT7									: 1;
  __REG32 ST_GPT8									: 1;
  __REG32 ST_GPT9									: 1;
  __REG32 ST_UART3								: 1;
  __REG32 		 										: 1;
  __REG32 ST_GPIO2								: 1;
  __REG32 ST_GPIO3								: 1;
  __REG32 ST_GPIO4								: 1;
  __REG32 ST_GPIO5								: 1;
  __REG32 ST_GPIO6								: 1;
  __REG32        	        				:14;
} __pm_wkst_per_bits;

/* PM_WKDEP_PER */
typedef struct {
  __REG32 EN_CORE									: 1;
  __REG32 EN_MPU									: 1;
  __REG32 												: 2;
  __REG32 EN_WKUP									: 1;
  __REG32        	        				:27;
} __pm_wkdep_per_bits;

/* PM_PWSTCTRL_PER */
typedef struct {
  __REG32 POWERSTATE							: 2;
  __REG32 LOGICRETSTATE						: 1;
  __REG32 												: 5;
  __REG32 MEMRETSTATE							: 1;
  __REG32        	        				: 7;
  __REG32 MEMONSTATE							: 2;
  __REG32        	        				:14;
} __pm_pwstctrl_per_bits;

/* PM_PWSTST_PER */
typedef struct {
  __REG32 POWERSTATEST						: 2;
  __REG32 												:18;
  __REG32 INTRANSITION						: 1;
  __REG32        	        				:11;
} __pm_pwstst_per_bits;

/* PM_PREPWSTST_PER */
typedef struct {
  __REG32 LASTPOWERSTATEENTERED		: 2;
  __REG32 LASTLOGICSTATEENTERED		: 1;
  __REG32        	        				:29;
} __pm_prepwstst_per_bits;

/* RM_RSTST_EMU */
typedef struct {
  __REG32 GLOBALCOLD_RST					: 1;
  __REG32 GLOBALWARM_RST					: 1;
  __REG32        	        				:30;
} __rm_rstst_emu_bits;

/* PRM_RSTCTRL */
typedef struct {
  __REG32        	        				: 1;
  __REG32 RST_GS									: 1;
  __REG32 RST_DPLL3								: 1;
  __REG32        	        				:29;
} __prm_rstctrl_bits;

/* PRM_RSTTIME */
typedef struct {
  __REG32 RSTTIME1								: 8;
  __REG32        	        				:24;
} __prm_rsttime_bits;

/* PRM_RSTST */
typedef struct {
  __REG32 GLOBAL_COLD_RST					: 1;
  __REG32 GLOBAL_SW_RST						: 1;
  __REG32        	        				: 1;
  __REG32 SECURITY_VIOL_RST				: 1;
  __REG32 MPU_WD_RST							: 1;
  __REG32 SECURE_WD_RST						: 1;
  __REG32 EXTERNAL_WARM_RST				: 1;
  __REG32        	        				: 2;
  __REG32 ICEPICK_RST							: 1;
  __REG32 ICECRUSHER_RST					: 1;
  __REG32        	        				:21;
} __prm_rstst_bits;

/* PRM_CLKSRC_CTRL */
typedef struct {
  __REG32 SYSCLKSEL								: 2;
  __REG32        	        				: 1;
  __REG32 AUTOEXTCLKMODE					: 2;
  __REG32        	        				: 1;
  __REG32 SYSCLKDIV								: 2;
  __REG32        	        				:24;
} __prm_clksrc_ctrl_bits;

/* PRM_OBS */
typedef struct {
  __REG32 OBS_BUS									:18;
  __REG32        	        				:14;
} __prm_obs_bits;

/* PRM_CLKSETUP */
typedef struct {
  __REG32 SETUP_TIME							:16;
  __REG32        	        				:16;
} __prm_clksetup_bits;

/* PRM_POLCTRL */
typedef struct {
  __REG32        	        				: 1;
  __REG32 CLKREQ_POL							: 1;
  __REG32 CLKOUT_POL							: 1;
  __REG32        	        				:29;
} __prm_polctrl_bits;

/* RM_RSTST_NEON */
typedef struct {
  __REG32 GLOBALCOLD_RST					: 1;
  __REG32 GLOBALWARM_RST					: 1;
  __REG32        	        				:30;
} __rm_rstst_neon_bits;

/* PM_WKDEP_NEON */
typedef struct {
  __REG32 												: 1;
  __REG32 EN_MPU									: 1;
  __REG32        	        				:30;
} __pm_wkdep_neon_bits;

/* PM_PWSTCTRL_NEON */
typedef struct {
  __REG32 POWERSTATE							: 2;
  __REG32 LOGICRETSTATE						: 1;
  __REG32        	        				:29;
} __pm_pwstctrl_neon_bits;

/* PM_PWSTST_NEON */
typedef struct {
  __REG32 POWERSTATEST						: 2;
  __REG32        	        				:18;
  __REG32 INTRANSITION						: 1;
  __REG32        	        				:11;
} __pm_pwstst_neon_bits;

/* PM_PREPWSTST_NEON */
typedef struct {
  __REG32 LASTPOWERSTATEENTERED		: 2;
  __REG32        	        				:30;
} __pm_prepwstst_neon_bits;

/* RM_RSTST_USBHOST */
typedef struct {
  __REG32 GLOBALCOLD_RST					: 1;
  __REG32 GLOBALWARM_RST					: 1;
  __REG32        	        				:30;
} __rm_rstst_usbhost_bits;

/* PM_WKEN_USBHOST */
typedef struct {
  __REG32 EN_USBHOST							: 1;
  __REG32        	        				:31;
} __pm_wken_usbhost_bits;

/* PM_MPUGRPSEL_USBHOST */
typedef struct {
  __REG32 GRPSEL_USBHOST					: 1;
  __REG32        	        				:31;
} __pm_mpugrpsel_usbhost_bits;

/* PM_WKST_USBHOST */
typedef struct {
  __REG32 ST_USBHOST							: 1;
  __REG32        	        				:31;
} __pm_wkst_usbhost_bits;

/* PM_WKDEP_USBHOST */
typedef struct {
  __REG32 EN_CORE									: 1;
  __REG32 EN_MPU									: 1;
  __REG32        	        				: 2;
  __REG32 EN_WKUP									: 1;
  __REG32        	        				:27;
} __pm_wkdep_usbhost_bits;

/* PM_PWSTCTRL_USBHOST */
typedef struct {
  __REG32 POWERSTATE							: 2;
  __REG32 LOGICRETSTATE						: 1;
  __REG32        	        				: 5;
  __REG32 MEMRETSTATE							: 1;
  __REG32        	        				: 7;
  __REG32 MEMONSTATE							: 2;
  __REG32        	        				:14;
} __pm_pwstctrl_usbhost_bits;

/* PM_PWSTST_USBHOST */
typedef struct {
  __REG32 POWERSTATEST						: 2;
  __REG32        	        				:18;
  __REG32 INTRANSITION						: 1;
  __REG32        	        				:11;
} __pm_pwstst_usbhost_bits;

/* PM_PREPWSTST_USBHOST */
typedef struct {
  __REG32 LASTPOWERSTATEENTERED		: 2;
  __REG32        	        				:30;
} __pm_prepwstst_usbhost_bits;

/* CM_REVISION */
typedef struct {
  __REG32 REV											: 8;
  __REG32        	        				:24;
} __cm_revision_bits;

/* CM_SYSCONFIG */
typedef struct {
  __REG32 AUTOIDLE								: 1;
  __REG32        	        				:31;
} __cm_sysconfig_bits;

/* CM_CLKEN_PLL_MPU */
typedef struct {
  __REG32 EN_MPU_DPLL							: 3;
  __REG32 EN_MPU_DPLL_DRIFTGUARD	: 1;
  __REG32 MPU_DPLL_FREQSEL				: 4;
  __REG32 MPU_DPLL_RAMPTIME				: 2;
  __REG32 EN_MPU_DPLL_LPMODE			: 1;
  __REG32        	        				:21;
} __cm_clken_pll_mpu_bits;

/* CM_IDLEST_MPU */
typedef struct {
  __REG32 ST_MPU									: 1;
  __REG32        	        				:31;
} __cm_idlest_mpu_bits;

/* CM_IDLEST_PLL_MPU */
typedef struct {
  __REG32 ST_MPU_CLK							: 1;
  __REG32        	        				:31;
} __cm_idlest_pll_mpu_bits;

/* CM_AUTOIDLE_PLL_MPU */
typedef struct {
  __REG32 AUTO_MPU_DPLL						: 1;
  __REG32        	        				:31;
} __cm_autoidle_pll_mpu_bits;

/* CM_CLKSEL1_PLL_MPU */
typedef struct {
  __REG32 MPU_DPLL_DIV						: 7;
  __REG32        	        				: 1;
  __REG32 MPU_DPLL_MULT						:11;
  __REG32 MPU_CLK_SRC 						: 3;
  __REG32        	        				:10;
} __cm_clksel1_pll_mpu_bits;

/* CM_CLKSEL2_PLL_MPU */
typedef struct {
  __REG32 MPU_DPLL_DIV						: 7;
  __REG32        	        				: 1;
  __REG32 MPU_DPLL_MULT						:11;
  __REG32 MPU_CLK_SRC 						: 3;
  __REG32        	        				:10;
} __cm_clksel2_pll_mpu_bits;

/* CM_CLKSTCTRL_MPU */
typedef struct {
  __REG32 CLKTRCTRL_MPU						: 2;
  __REG32        	        				:30;
} __cm_clkstctrl_mpu_bits;

/* CM_CLKSTST_MPU */
typedef struct {
  __REG32 CLKACTIVITY_MPU					: 1;
  __REG32        	        				:31;
} __cm_clkstst_mpu_bits;

/* CM_FCLKEN1_CORE */
typedef struct {
  __REG32        	        				: 9;
  __REG32 EN_MCBSP1								: 1;
  __REG32 EN_MCBSP5								: 1;
  __REG32 EN_GPT10								: 1;
  __REG32 EN_GPT11								: 1;
  __REG32 EN_UART1								: 1;
  __REG32 EN_UART2								: 1;
  __REG32 EN_I2C1 								: 1;
  __REG32 EN_I2C2									: 1;
  __REG32 EN_I2C3									: 1;
  __REG32 EN_MCSPI1								: 1;
  __REG32 EN_MCSPI2								: 1;
  __REG32 EN_MCSPI3								: 1;
  __REG32 EN_MCSPI4								: 1;
  __REG32 EN_HDQ									: 1;
  __REG32        	        				: 1;
  __REG32 EN_MMC1									: 1;
  __REG32 EN_MMC2									: 1;
  __REG32        	        				: 4;
  __REG32 EN_MMC3									: 1;
  __REG32        	        				: 1;
} __cm_fclken1_core_bits;

/* CM_FCLKEN3_CORE */
typedef struct {
  __REG32 EN_CPEFUSE							: 1;
  __REG32 EN_TS										: 1;
  __REG32 EN_USBTLL								: 1;
  __REG32        	        				:29;
} __cm_fclken3_core_bits;

/* CM_ICLKEN1_CORE */
typedef struct {
  __REG32        	        				: 1;
  __REG32 EN_SDRC									: 1;
  __REG32        	        				: 2;
  __REG32 EN_IPSS									: 1;
  __REG32        	        				: 1;
  __REG32 EN_SCMCTRL							: 1;
  __REG32        	        				: 2;
  __REG32 EN_MCBSP1								: 1;
  __REG32 EN_MCBSP5								: 1;
  __REG32 EN_GPT10								: 1;
  __REG32 EN_GPT11								: 1;
  __REG32 EN_UART1								: 1;
  __REG32 EN_UART2								: 1;
  __REG32 EN_I2C1 								: 1;
  __REG32 EN_I2C2 								: 1;
  __REG32 EN_I2C3									: 1;
  __REG32 EN_MCSPI1								: 1;
  __REG32 EN_MCSPI2								: 1;
  __REG32 EN_MCSPI3								: 1;
  __REG32 EN_MCSPI4								: 1;
  __REG32 EN_HDQ									: 1;
  __REG32 EN_UART4								: 1;
  __REG32 EN_MMC1									: 1;
  __REG32 EN_MMC2									: 1;
  __REG32        	        				: 4;
  __REG32 EN_MMC3									: 1;
  __REG32        	        				: 1;
} __cm_iclken1_core_bits;

/* CM_ICLKEN3_CORE */
typedef struct {
  __REG32 												: 2;
  __REG32 EN_USBTLL								: 1;
  __REG32        	        				:29;
} __cm_iclken3_core_bits;

/* CM_IDLEST1_CORE */
typedef struct {
  __REG32 												: 1;
  __REG32 ST_EMIF4								: 1;
  __REG32 ST_SDMA									: 1;
  __REG32        	        				: 2;
  __REG32 ST_IPSS_IDLE						: 1;
  __REG32 ST_SCMCTRL							: 1;
  __REG32        	        				: 2;
  __REG32 ST_MCBSP1								: 1;
  __REG32 ST_MCBSP5								: 1;
  __REG32 ST_GPT10								: 1;
  __REG32 ST_GPT11								: 1;
  __REG32 ST_UART1								: 1;
  __REG32 ST_UART2								: 1;
  __REG32 ST_I2C1 								: 1;
  __REG32 ST_I2C2									: 1;
  __REG32 ST_I2C3									: 1;
  __REG32 ST_MCSPI1								: 1;
  __REG32 ST_MCSPI2								: 1;
  __REG32 ST_MCSPI3								: 1;
  __REG32 ST_MCSPI4								: 1;
  __REG32 ST_HDQ									: 1;
  __REG32 ST_UART4								: 1;
  __REG32 ST_MMC1									: 1;
  __REG32 ST_MMC2									: 1;
  __REG32        	        				: 4;
  __REG32 ST_MMC3									: 1;
  __REG32        	        				: 1;
} __cm_idlest1_core_bits;

/* CM_IDLEST3_CORE */
typedef struct {
  __REG32 ST_CPEFUSE							: 1;
  __REG32 												: 1;
  __REG32 ST_USBTLL 							: 1;
  __REG32        	        				:29;
} __cm_idlest3_core_bits;

/* CM_AUTOIDLE1_CORE */
typedef struct {
  __REG32 												: 4;
  __REG32 AUTO_IPSS								: 1;
  __REG32        	        				: 1;
  __REG32 AUTO_SCMCTRL						: 1;
  __REG32        	        				: 2;
  __REG32 AUTO_MCBSP1							: 1;
  __REG32 AUTO_MCBSP5							: 1;
  __REG32 AUTO_GPT10							: 1;
  __REG32 AUTO_GPT11							: 1;
  __REG32 AUTO_UART1							: 1;
  __REG32 AUTO_UART2							: 1;
  __REG32 AUTO_I2C1								: 1;
  __REG32 AUTO_I2C2								: 1;
  __REG32 AUTO_I2C3								: 1;
  __REG32 AUTO_MCSPI1							: 1;
  __REG32 AUTO_MCSPI2							: 1;
  __REG32 AUTO_MCSPI3							: 1;
  __REG32 AUTO_MCSPI4							: 1;
  __REG32 AUTO_HDQ								: 1;
  __REG32 AUTO_UART4							: 1;
  __REG32 AUTO_MMC1								: 1;
  __REG32 AUTO_MMC2								: 1;
  __REG32        	        				: 4;
  __REG32 AUTO_MMC3								: 1;
  __REG32        	        				: 1;
} __cm_autoidle1_core_bits;

/* CM_AUTOIDLE3_CORE */
typedef struct {
  __REG32        	        				: 2;
  __REG32 AUTO_USBTLL							: 1;
  __REG32        	        				:29;
} __cm_autoidle3_core_bits;

/* CM_CLKSEL_CORE */
typedef struct {
  __REG32 CLKSEL_L3								: 2;
  __REG32 CLKSEL_L4								: 2;
  __REG32        	        				: 2;
  __REG32 CLKSEL_GPT10						: 1;
  __REG32 CLKSEL_GPT11						: 1;
  __REG32        	        				:24;
} __cm_clksel_core_bits;

/* CM_CLKSTCTRL_CORE */
typedef struct {
  __REG32 CLKTRCTRL_L3						: 2;
  __REG32 CLKTRCTRL_L4						: 2;
  __REG32        	        				:28;
} __cm_clkstctrl_core_bits;

/* CM_CLKSTST_CORE */
typedef struct {
  __REG32 CLKACTIVITY_L3					: 1;
  __REG32 CLKACTIVITY_L4					: 1;
  __REG32        	        				:30;
} __cm_clkstst_core_bits;

/* CM_FCLKEN_SGX */
typedef struct {
  __REG32 												: 1;
  __REG32 EN_SGX									: 1;
  __REG32        	        				:30;
} __cm_fclken_sgx_bits;

/* CM_ICLKEN_SGX */
typedef struct {
  __REG32 EN_SGX									: 1;
  __REG32        	        				:31;
} __cm_iclken_sgx_bits;

/* CM_IDLEST_SGX */
typedef struct {
  __REG32 ST_SGX									: 1;
  __REG32        	        				:31;
} __cm_idlest_sgx_bits;

/* CM_CLKSEL_SGX */
typedef struct {
  __REG32 CLKSEL_SGX							: 3;
  __REG32        	        				:29;
} __cm_clksel_sgx_bits;

/* CM_SLEEPDEP_SGX */
typedef struct {
  __REG32        	        				: 1;
  __REG32 EN_MPU									: 1;
  __REG32        	        				:30;
} __cm_sleepdep_sgx_bits;

/* CM_CLKSTCTRL_SGX */
typedef struct {
  __REG32 CLKTRCTRL_SGX						: 2;
  __REG32        	        				:30;
} __cm_clkstctrl_sgx_bits;

/* CM_CLKSTST_SGX */
typedef struct {
  __REG32 CLKACTIVITY_SGX					: 1;
  __REG32        	        				:31;
} __cm_clkstst_sgx_bits;

/* CM_FCLKEN_WKUP */
typedef struct {
  __REG32 EN_GPT1									: 1;
  __REG32        	        				: 2;
  __REG32 EN_GPIO1								: 1;
  __REG32        	        				: 1;
  __REG32 EN_WDT2 								: 1;
  __REG32        	        				:26;
} __cm_fclken_wkup_bits;

/* CM_ICLKEN_WKUP */
typedef struct {
  __REG32 EN_GPT1									: 1;
  __REG32 EN_GPT12        				: 1;
  __REG32 EN_32KSYNC       				: 1;
  __REG32 EN_GPIO1								: 1;
  __REG32 EN_WDT1 	      				: 1;
  __REG32 EN_WDT2 								: 1;
  __REG32        	        				:26;
} __cm_iclken_wkup_bits;

/* CM_IDLEST_WKUP */
typedef struct {
  __REG32 ST_GPT1									: 1;
  __REG32 ST_GPT12        				: 1;
  __REG32 ST_32KSYNC       				: 1;
  __REG32 ST_GPIO1								: 1;
  __REG32 ST_WDT1 	      				: 1;
  __REG32 ST_WDT2 								: 1;
  __REG32        	        				:26;
} __cm_idlest_wkup_bits;

/* CM_AUTOIDLE_WKUP */
typedef struct {
  __REG32 AUTO_GPT1								: 1;
  __REG32 AUTO_GPT12        			: 1;
  __REG32 AUTO_32KSYNC       			: 1;
  __REG32 AUTO_GPIO1							: 1;
  __REG32 AUTO_WDT1 	      			: 1;
  __REG32 AUTO_WDT2 							: 1;
  __REG32        	        				:26;
} __cm_autoidle_wkup_bits;

/* CM_CLKSEL_WKUP */
typedef struct {
  __REG32 CLKSEL_GPT1							: 1;
  __REG32 CLKSEL_RM	        			: 2;
  __REG32        	        				:29;
} __cm_clksel_wkup_bits;

/* CM_CLKEN_PLL */
typedef struct {
  __REG32 EN_CORE_DPLL							: 3;
  __REG32 EN_CORE_DPLL_DRIFTGUARD		: 1;
  __REG32 CORE_DPLL_FREQSEL					: 4;
  __REG32 CORE_DPLL_RAMPTIME				: 2;
  __REG32 EN_CORE_DPLL_LPMODE				: 1;
  __REG32        	        					: 1;
  __REG32 PWRDN_EMU_CORE						: 1;
  __REG32        	        					: 3;
  __REG32 EN_PERIPH_DPLL						: 3;
  __REG32 EN_PERIPH_DPLL_DRIFTGUARD	: 1;
  __REG32 PERIPH_DPLL_FREQSEL				: 4;
  __REG32 PERIPH_DPLL_RAMPTIME			: 2;
  __REG32 EN_PERIPH_DPLL_LPMODE			: 1;
  __REG32 PWRDN_96M									: 1;
  __REG32 PWRDN_TV									: 1;
  __REG32 PWRDN_DSS1								: 1;
  __REG32        	        					: 1;
  __REG32 PWRDN_EMU_PERIPH					: 1;
} __cm_clken_pll_bits;

/* CM_CLKEN2_PLL */
typedef struct {
  __REG32 EN_PERIPH2_DPLL						: 3;
  __REG32 EN_PERIPH2_DPLL_DRIFTGUARD: 1;
  __REG32 PERIPH2_DPLL_FREQSEL			: 4;
  __REG32 PERIPH2_DPLL_RAMPTIME			: 2;
  __REG32 EN_PERIPH2_DPLL_LPMODE		: 1;
  __REG32        	        					:21;
} __cm_clken2_pll_bits;

/* CM_IDLEST_CKGEN */
typedef struct {
  __REG32 ST_CORE_CLK								: 1;
  __REG32 ST_PERIPH_CLK							: 1;
  __REG32 ST_96M_CLK								: 1;
  __REG32 ST_48M_CLK								: 1;
  __REG32 ST_12M_CLK								: 1;
  __REG32 ST_54M_CLK								: 1;
  __REG32        	        					: 2;
  __REG32 ST_EMU_CORE_CLK						: 1;
  __REG32 ST_FUNC96M_CLK						: 1;
  __REG32 ST_TV_CLK									: 1;
  __REG32 ST_DSS1_CLK								: 1;
  __REG32        	        					: 1;
  __REG32 ST_EMU_PERIPH_CLK					: 1;
  __REG32        	        					:18;
} __cm_idlest_ckgen_bits;

/* CM_IDLEST2_CKGEN */
typedef struct {
  __REG32 ST_PERIPH2_CLK						: 1;
  __REG32 ST_120M_CLK								: 1;
  __REG32        	        					: 1;
  __REG32 ST_FUNC120M_CLK						: 1;
 	__REG32        	        					:28;
} __cm_idlest2_ckgen_bits;

/* CM_AUTOIDLE_PLL */
typedef struct {
  __REG32 AUTO_CORE_DPLL						: 3;
  __REG32 AUTO_PERIPH_DPLL 					: 3;
 __REG32        	        					:26;
} __cm_autoidle_pll_bits;

/* CM_AUTOIDLE2_PLL */
typedef struct {
 	__REG32 AUTO_PERIPH2_DPLL					: 3;
	__REG32        	        					:29;
} __cm_autoidle2_pll_bits;

/* CM_CLKSEL1_PLL */
typedef struct {
 	__REG32        	        					: 3;
 	__REG32 SOURCE_48M								: 1;
 	__REG32        	        					: 1;
 	__REG32 SOURCE_54M								: 1;
 	__REG32 SOURCE_96M								: 1;
 	__REG32        	        					: 1;
 	__REG32 CORE_DPLL_DIV							: 7;
 	__REG32        	        					: 1;
 	__REG32 CORE_DPLL_MULT						:11;
 	__REG32 CORE_DPLL_CLKOUT_DIV			: 5;
} __cm_clksel1_pll_bits;

/* CM_CLKSEL2_PLL */
typedef struct {
 	__REG32 PERIPH_DPLL_DIV						: 7;
 	__REG32        	        					: 1;
 	__REG32 PERIPH_DPLL_MULT					:11;
 	__REG32        	        					:13;
} __cm_clksel2_pll_bits;

/* CM_CLKSEL3_PLL */
typedef struct {
 	__REG32 DIV_96M										: 5;
 	__REG32        	        					:27;
} __cm_clksel3_pll_bits;

/* CM_CLKSEL4_PLL */
typedef struct {
 	__REG32 PERIPH2_DPLL_DIV					: 7;
 	__REG32        	        					: 1;
 	__REG32 PERIPH2_DPLL_MULT					:11;
 	__REG32        	        					:13;
} __cm_clksel4_pll_bits;

/* CM_CLKSEL5_PLL */
typedef struct {
 	__REG32 DIV_120M									: 5;
 	__REG32        	        					:27;
} __cm_clksel5_pll_bits;

/* CM_CLKOUT_CTRL */
typedef struct {
 	__REG32 CLKOUT2SOURCE							: 2;
 	__REG32        	        					: 1;
 	__REG32 CLKOUT2_DIV								: 3;
 	__REG32        	        					: 1;
 	__REG32 CLKOUT2_EN								: 1;
 	__REG32        	        					:24;
} __cm_clkout_ctrl_bits;

/* CM_FCLKEN_DSS */
typedef struct {
 	__REG32 EN_DSS1										: 1;
 	__REG32 EN_DSS2										: 1;
 	__REG32 EN_TV											: 1;
 	__REG32        	        					:29;
} __cm_fclken_dss_bits;

/* CM_ICLKEN_DSS */
typedef struct {
 	__REG32 EN_DSS										: 1;
 	__REG32        	        					:31;
} __cm_iclken_dss_bits;

/* CM_IDLEST_DSS */
typedef struct {
 	__REG32 ST_DSS_STDBY							: 1;
 	__REG32 ST_DSS_IDLE								: 1;
 	__REG32        	        					:30;
} __cm_idlest_dss_bits;

/* CM_AUTOIDLE_DSS */
typedef struct {
 	__REG32 AUTO_DSS									: 1;
 	__REG32        	        					:31;
} __cm_autoidle_dss_bits;

/* CM_CLKSEL_DSS */
typedef struct {
 	__REG32 CLKSEL_DSS1								: 5;
 	__REG32        	        					: 3;
 	__REG32 CLKSEL_TV									: 5;
 	__REG32        	        					:19;
} __cm_clksel_dss_bits;

/* CM_SLEEPDEP_DSS */
typedef struct {
 	__REG32 EN_CORE										: 1;
 	__REG32 EN_MPU										: 1;
 	__REG32        	        					:30;
} __cm_sleepdep_dss_bits;

/* CM_CLKSTCTRL_DSS */
typedef struct {
 	__REG32 CLKTRCTRL_DSS							: 2;
 	__REG32        	        					:30;
} __cm_clkstctrl_dss_bits;

/* CM_CLKSTST_DSS */
typedef struct {
 	__REG32 CLKACTIVITY_DSS						: 1;
 	__REG32        	        					:31;
} __cm_clkstst_dss_bits;

/* CM_FCLKEN_PER */
typedef struct {
 	__REG32 EN_MCBSP2									: 1;
 	__REG32 EN_MCBSP3									: 1;
 	__REG32 EN_MCBSP4									: 1;
 	__REG32 EN_GPT2										: 1;
 	__REG32 EN_GPT3										: 1;
 	__REG32 EN_GPT4										: 1;
 	__REG32 EN_GPT5										: 1;
 	__REG32 EN_GPT6										: 1;
 	__REG32 EN_GPT7										: 1;
 	__REG32 EN_GPT8										: 1;
 	__REG32 EN_GPT9										: 1;
 	__REG32 EN_UART3									: 1;
 	__REG32 EN_WDT3										: 1;
 	__REG32 EN_GPIO2									: 1;
 	__REG32 EN_GPIO3									: 1;
 	__REG32 EN_GPIO4									: 1;
 	__REG32 EN_GPIO5									: 1;
 	__REG32 EN_GPIO6									: 1;
 	__REG32        	        					:14;
} __cm_fclken_per_bits;

/* CM_IDLEST_PER */
typedef struct {
 	__REG32 ST_MCBSP2									: 1;
 	__REG32 ST_MCBSP3									: 1;
 	__REG32 ST_MCBSP4									: 1;
 	__REG32 ST_GPT2										: 1;
 	__REG32 ST_GPT3										: 1;
 	__REG32 ST_GPT4										: 1;
 	__REG32 ST_GPT5										: 1;
 	__REG32 ST_GPT6										: 1;
 	__REG32 ST_GPT7										: 1;
 	__REG32 ST_GPT8										: 1;
 	__REG32 ST_GPT9										: 1;
 	__REG32 ST_UART3									: 1;
 	__REG32 ST_WDT3										: 1;
 	__REG32 ST_GPIO2									: 1;
 	__REG32 ST_GPIO3									: 1;
 	__REG32 ST_GPIO4									: 1;
 	__REG32 ST_GPIO5									: 1;
 	__REG32 ST_GPIO6									: 1;
 	__REG32        	        					:14;
} __cm_idlest_per_bits;

/* CM_AUTOIDLE_PER */
typedef struct {
 	__REG32 AUTO_MCBSP2									: 1;
 	__REG32 AUTO_MCBSP3									: 1;
 	__REG32 AUTO_MCBSP4									: 1;
 	__REG32 AUTO_GPT2										: 1;
 	__REG32 AUTO_GPT3										: 1;
 	__REG32 AUTO_GPT4										: 1;
 	__REG32 AUTO_GPT5										: 1;
 	__REG32 AUTO_GPT6										: 1;
 	__REG32 AUTO_GPT7										: 1;
 	__REG32 AUTO_GPT8										: 1;
 	__REG32 AUTO_GPT9										: 1;
 	__REG32 AUTO_UART3									: 1;
 	__REG32 AUTO_WDT3										: 1;
 	__REG32 AUTO_GPIO2									: 1;
 	__REG32 AUTO_GPIO3									: 1;
 	__REG32 AUTO_GPIO4									: 1;
 	__REG32 AUTO_GPIO5									: 1;
 	__REG32 AUTO_GPIO6									: 1;
 	__REG32        	        						:14;
} __cm_autoidle_per_bits;

/* CM_CLKSEL_PER */
typedef struct {
 	__REG32 CLKSEL_GPT2									: 1;
 	__REG32 CLKSEL_GPT3									: 1;
 	__REG32 CLKSEL_GPT4									: 1;
 	__REG32 CLKSEL_GPT5									: 1;
 	__REG32 CLKSEL_GPT6									: 1;
 	__REG32 CLKSEL_GPT7 								: 1;
 	__REG32 CLKSEL_GPT8 								: 1;
 	__REG32 CLKSEL_GPT9									: 1;
 	__REG32        	        						:24;
} __cm_clksel_per_bits;

/* CM_SLEEPDEP_PER */
typedef struct {
 	__REG32 														: 1;
 	__REG32 EN_MPU											: 1;
 	__REG32        	        						:30;
} __cm_sleepdep_per_bits;

/* CM_CLKSTCTRL_PER */
typedef struct {
 	__REG32 CLKTRCTRL_PER								: 2;
 	__REG32        	        						:30;
} __cm_clkstctrl_per_bits;

/* CM_CLKSTST_PER */
typedef struct {
 	__REG32 CLKACTIVITY_PER							: 1;
 	__REG32        	        						:31;
} __cm_clkstst_per_bits;

/* CM_CLKSEL1_EMU */
typedef struct {
 	__REG32 MUX_CTRL										: 2;
 	__REG32 TRACE_MUX_CTRL							: 2;
 	__REG32 CLKSEL_ATCLK								: 2;
 	__REG32 CLKSEL_PCLKX2								: 2;
 	__REG32 CLKSEL_PCLK									: 3;
 	__REG32 CLKSEL_TRACECLK							: 3;
 	__REG32        	        						: 2;
 	__REG32 DIV_DPLL3										: 5;
 	__REG32        	        						: 3;
 	__REG32 DIV_DPLL4										: 5;
 	__REG32        	        						: 3;
} __cm_clksel1_emu_bits;

/* CM_CLKSTCTRL_EMU */
typedef struct {
 	__REG32 CLKTRCTRL_EMU								: 2;
 	__REG32        	        						:30;
} __cm_clkstctrl_emu_bits;

/* CM_CLKSTST_EMU */
typedef struct {
 	__REG32 CLKACTIVITY_EMU							: 1;
 	__REG32        	        						:31;
} __cm_clkstst_emu_bits;

/* CM_CLKSEL2_EMU */
typedef struct {
 	__REG32 CORE_DPLL_EMU_DIV						: 7;
 	__REG32        	        						: 1;
 	__REG32 CORE_DPLL_EMU_MULT					:11;
 	__REG32 OVERRIDE_ENABLE							: 1;
 	__REG32        	        						:12;
} __cm_clksel2_emu_bits;

/* CM_CLKSEL3_EMU */
typedef struct {
 	__REG32 PERIPH_DPLL_EMU_DIV					: 7;
 	__REG32        	        						: 1;
 	__REG32 PERIPH_DPLL_EMU_MULT				:11;
 	__REG32 OVERRIDE_ENABLE							: 1;
 	__REG32        	        						:12;
} __cm_clksel3_emu_bits;

/* CM_POLCTRL */
typedef struct {
 	__REG32 CLKOUT2_POL									: 1;
 	__REG32        	        						:31;
} __cm_polctrl_bits;

/* CM_IDLEST_NEON */
typedef struct {
 	__REG32 ST_NEON											: 1;
 	__REG32        	        						:31;
} __cm_idlest_neon_bits;

/* CM_CLKSTCTRL_NEON */
typedef struct {
 	__REG32 CLKTRCTRL_NEON							: 2;
 	__REG32        	        						:30;
} __cm_clkstctrl_neon_bits;

/* CM_FCLKEN_USBHOST */
typedef struct {
 	__REG32 EN_USBHOST1									: 1;
 	__REG32 EN_USBHOST2									: 1;
 	__REG32        	        						:30;
} __cm_fclken_usbhost_bits;

/* CM_ICLKEN_USBHOST */
typedef struct {
 	__REG32 EN_USBHOST									: 1;
 	__REG32        	        						:31;
} __cm_iclken_usbhost_bits;

/* CM_IDLEST_USBHOST */
typedef struct {
 	__REG32 ST_USBHOST_STDBY						: 1;
 	__REG32 ST_USBHOST_IDLE							: 1;
 	__REG32        	        						:30;
} __cm_idlest_usbhost_bits;

/* CM_AUTOIDLE_USBHOST */
typedef struct {
 	__REG32 AUTO_USBHOST								: 1;
 	__REG32        	        						:31;
} __cm_autoidle_usbhost_bits;

/* CM_SLEEPDEP_USBHOST */
typedef struct {
 	__REG32        	        						: 1;
 	__REG32 EN_MPU											: 1;
 	__REG32        	        						:30;
} __cm_sleepdep_usbhost_bits;

/* CM_CLKSTCTRL_USBHOST */
typedef struct {
 	__REG32 CLKTRCTRL_USBHOST						: 2;
 	__REG32        	        						:30;
} __cm_clkstctrl_usbhost_bits;

/* CM_CLKSTST_USBHOST */
typedef struct {
 	__REG32 CLKACTIVITY_USBHOST					: 2;
 	__REG32        	        						:30;
} __cm_clkstst_usbhost_bits;

/* L3_IA_AGENT_CONTROL */
typedef struct {
 	__REG32 CORE_RESET									: 1;
 	__REG32        	        						: 3;
 	__REG32 REJECT											: 1;
 	__REG32        	        						: 3;
 	__REG32 RESP_TIMEOUT								: 3;
 	__REG32        	        						: 5;
 	__REG32 BURST_TIMEOUT								: 3;
 	__REG32        	        						: 6;
 	__REG32 RESP_TIMEOUT_REP						: 1;
 	__REG32 BURST_TIMEOUT_REP						: 1;
 	__REG32 ALL_INBAND_ERROR_REP				: 1;
 	__REG32 INBAND_ERROR_PRIMARY_REP		: 1;
 	__REG32 INBAND_ERROR_SECONDARY_REP	: 1;
 	__REG32        	        						: 2;
} __l3_ia_agent_control_bits;

/* L3_IA_AGENT_STATUS */
typedef struct {
 	__REG32 CORE_RESET									: 1;
 	__REG32        	        						: 3;
 	__REG32 REQ_ACTIVE									: 1;
 	__REG32 RESP_WAITING								: 1;
 	__REG32 BURST												: 1;
 	__REG32 READEX											: 1;
 	__REG32 RESP_TIMEOUT								: 1;
 	__REG32        	        						: 3;
 	__REG32 TIMEBASE										: 4;
 	__REG32 BURST_TIMEOUT								: 1;
 	__REG32        	        						:11;
 	__REG32 INBAND_ERROR_PRIMARY_REP		: 1;
 	__REG32 INBAND_ERROR_SECONDARY_REP	: 1;
 	__REG32        	        						: 2;
} __l3_ia_agent_status_bits;

/* L3_IA_ERROR_LOG */
typedef struct {
 	__REG32 CMD													: 3;
 	__REG32        	        						: 5;
 	__REG32 INITID											: 8;
 	__REG32        	        						: 8;
 	__REG32 CODE												: 4;
 	__REG32        	        						: 2;
 	__REG32 SECONDARY										: 1;
 	__REG32 MULTI												: 1;
} __l3_ia_error_log_bits;

/* L3_TA_AGENT_CONTROL */
typedef struct {
 	__REG32 CORE_RESET									: 1;
 	__REG32        	        						: 3;
 	__REG32 REJECT											: 1;
 	__REG32        	        						: 3;
 	__REG32 REQ_TIMEOUT									: 3;
 	__REG32        	        						:13;
 	__REG32 SERROR_REP									: 1;
 	__REG32 REQ_TIMEOUT_REP							: 1;
 	__REG32        	        						: 6;
} __l3_ta_agent_control_bits;

/* L3_TA_AGENT_STATUS */
typedef struct {
 	__REG32 CORE_RESET									: 1;
 	__REG32        	        						: 3;
 	__REG32 REQ_WAITING									: 1;
 	__REG32 RESP_ACTIVE									: 1;
 	__REG32 BURST												: 1;
 	__REG32 READEX											: 1;
 	__REG32 REQ_TIMEOUT									: 1;
 	__REG32        	        						: 3;
 	__REG32 TIMEBASE										: 4;
 	__REG32 BURST_CLOSE									: 1;
 	__REG32        	        						: 7;
 	__REG32 SERROR											: 1;
 	__REG32        	        						: 7;
} __l3_ta_agent_status_bits;

/* L3_TA_ERROR_LOG */
typedef struct {
 	__REG32 CMD													: 3;
 	__REG32        	        						: 5;
 	__REG32 INITID											: 8;
 	__REG32 														: 8;
 	__REG32 CODE												: 4;
 	__REG32        	        						: 3;
 	__REG32 MULTI												: 1;
} __l3_ta_error_log_bits;

/* L3_RT_INITID_READBACK */
typedef struct {
 	__REG32 INITID											: 8;
 	__REG32        	        						:24;
} __l3_rt_initid_readback_bits;

/* L3_RT_NETWORK_CONTROL LOW */
typedef struct {
 	__REG32        	        						: 8;
 	__REG32 TIMEOUT_BASE								: 3;
 	__REG32        	        						:21;
} __l3_rt_network_control_l_bits;

/* L3_RT_NETWORK_CONTROL HIGH */
typedef struct {
 	__REG32        	        						:24;
 	__REG32 CLOCK_GATE_DISABLE					: 1;
 	__REG32        	        						: 7;
} __l3_rt_network_control_h_bits;

/* L3_PM_ERROR_LOG */
typedef struct {
 	__REG32 CMD													: 3;
 	__REG32        	        						: 1;
 	__REG32 REGION											: 3;
 	__REG32        	        						: 1;
 	__REG32 INITID											: 8;
 	__REG32 REQ_INFO										: 5;
 	__REG32        	        						: 3;
 	__REG32 CODE												: 4;
 	__REG32        	        						: 2;
 	__REG32 SECONDARY										: 1;
 	__REG32 MULTI												: 1;
} __l3_pm_error_log_bits;

/* L3_PM_CONTROL_RT */
typedef struct {
 	__REG32        	        						:24;
 	__REG32 ERROR_REP										: 1;
 	__REG32 ERROR_SECONDARY_REP					: 1;
 	__REG32        	        						: 6;
} __l3_pm_control_bits;

/* L3_PM_ERROR_CLEAR_SINGLE_RT */
typedef struct {
 	__REG32 CLEAR												: 1;
 	__REG32        	        						:31;
} __l3_pm_error_clear_single_bits;

/* L3_PM_REQ_INFO_PERMISSION_i */
typedef struct {
 	__REG32 REQ_INFO										:16;
 	__REG32        	        						:16;
} __l3_pm_req_info_permission_bits;

/* L3_PM_READ_PERMISSION_i */
typedef struct {
 	__REG32        	        						: 1;
 	__REG32 MPU													: 1;
 	__REG32        	        						: 1;
 	__REG32 SDMA												: 1;
 	__REG32 IPSS												: 1;
 	__REG32        	        						: 3;
 	__REG32 DISP_SS											: 1;
 	__REG32 USB_HS_HOST									: 1;
 	__REG32        	        						: 2;
 	__REG32 DAP													: 1;
 	__REG32        	        						: 1;
 	__REG32 SGX													: 1;
 	__REG32        	        						:17;
} __l3_pm_read_permission_bits;

/* L3_PM_ADDR_MATCH_k */
typedef struct {
 	__REG32 ADDR_SPACE									: 3;
 	__REG32 SIZE  	        						: 5;
 	__REG32        	        						: 1;
 	__REG32 LEVEL 											: 1;
 	__REG32 BASE_ADDR										:10;
 	__REG32        	        						:12;
} __l3_pm_addr_match_bits;

/* L3_SI_CONTROL */
typedef struct {
 	__REG32        	        						:24;
 	__REG32 CLOCK_GATE_DISABLE  				: 1;
 	__REG32        	        						: 7;
} __l3_si_control_bits;

/* L4_IA_AGENT_CONTROL_L */
typedef struct {
 	__REG32        	        						:24;
 	__REG32 MERROR_REP									: 1;
 	__REG32        	        						: 2;
 	__REG32 INBAND_ERROR_REP						: 1;
 	__REG32        	        						: 4;
} __l4_ia_agent_control_l_bits;

/* L4_IA_ERROR_LOG_L */
typedef struct {
 	__REG32        	        						:24;
 	__REG32 CODE												: 2;
 	__REG32        	        						: 5;
 	__REG32 MULTI           						: 1;
} __l4_ia_error_log_l_bits;

/* L4_TA_AGENT_CONTROL_L */
typedef struct {
 	__REG32 OCP_RESET										: 1;
 	__REG32        	        						: 7;
 	__REG32 REQ_TIMEOUT									: 3;
 	__REG32        	        						:13;
 	__REG32 SERROR_REP									: 1;
 	__REG32        	        						: 7;
} __l4_ta_agent_control_l_bits;

/* L4_TA_AGENT_CONTROL_H */
typedef struct {
 	__REG32        	        						: 8;
 	__REG32 EXT_CLOCK										: 1;
 	__REG32        	        						:23;
} __l4_ta_agent_control_h_bits;

/* L4_TA_AGENT_STATUS_L */
typedef struct {
 	__REG32        	        						: 8;
 	__REG32 REQ_TIMEOUT									: 1;
 	__REG32        	        						:23;
} __l4_ta_agent_status_l_bits;

/* L4_LA_INITIATOR_INFO_L */
typedef struct {
 	__REG32 SEGMENTS         						: 4;
 	__REG32                  						:12;
 	__REG32 NUMBER_REGIONS   						: 8;
 	__REG32 PROT_GROUPS      						: 4;
 	__REG32                  						: 4;
} __l4_la_initiator_info_l_bits;

/* L4_LA_INITIATOR_INFO_H */
typedef struct {
 	__REG32 ADDR_WIDTH       						: 5;
 	__REG32                  						: 3;
 	__REG32 BYTE_DATA_WIDTH_EXP 				: 3;
 	__REG32                      				: 1;
 	__REG32 CONNID_WIDTH         				: 3;
 	__REG32                      				: 1;
 	__REG32 THREADS         						: 3;
 	__REG32                  						:13;
} __l4_la_initiator_info_h_bits;

/* L4_LA_NETWORK_CONTROL_L */
typedef struct {
 	__REG32                  						: 8;
 	__REG32 TIMEOUT_BASE         				: 3;
 	__REG32                  						:21;
} __l4_la_network_control_l_bits;

/* L4_LA_NETWORK_CONTROL_H */
typedef struct {
 	__REG32                  						: 8;
 	__REG32 EXT_CLOCK           				: 1;
 	__REG32                      				:11;
 	__REG32 THREAD0_PRI          				: 1;
 	__REG32                     				: 3;
 	__REG32 CLOCK_GATE_DISABLE   				: 1;
 	__REG32                  						: 7;
} __l4_la_network_control_h_bits;

/* L4_AP_SEGMENT_i_L */
typedef struct {
 	__REG32 BASE												:24;
 	__REG32        	        						: 8;
} __l4_ap_segment_l_bits;

/* L4_AP_SEGMENT_i_H */
typedef struct {
 	__REG32 SIZE												: 5;
 	__REG32        	        						:27;
} __l4_ap_segment_h_bits;

/* L4_AP_PROT_GROUP_MEMBERS_k_L */
typedef struct {
 	__REG32 CONNID_BIT_VECTOR						:16;
 	__REG32        	        						:16;
} __l4_ap_prot_group_members_l_bits;

/* L4_AP_PROT_GROUP_ROLES_k_L */
typedef struct {
 	__REG32 ENABLE											:16;
 	__REG32        	        						:16;
} __l4_ap_prot_group_roles_l_bits;

/* L4_AP_REGION_l_L */
typedef struct {
 	__REG32 BASE												:24;
 	__REG32        	        						: 8;
} __l4_ap_region_l_bits;

/* L4_AP_REGION_l_H */
typedef struct {
 	__REG32 ENABLE											: 1;
 	__REG32 SIZE												: 5;
 	__REG32        	        						:11;
 	__REG32 BYTE_DATA_WIDTH_EXP					: 3;
 	__REG32 PROT_GROUP_ID								: 3;
 	__REG32        	        						: 1;
 	__REG32 SEGMENT_ID									: 4;
 	__REG32        	        						: 4;
} __l4_ap_region_h_bits;

/* SCM_CONTROL_REVISION */
typedef struct {
 	__REG32 REVISION										: 8;
 	__REG32        	        						:24;
} __scm_control_revision_bits;

/* SCM_CONTROL_SYSCONFIG */
typedef struct {
 	__REG32 AUTOIDLE										: 1;
 	__REG32 SOFTRESET										: 1;
 	__REG32 ENAWAKEUP										: 1;
 	__REG32 IDLEMODE										: 2;
 	__REG32        	        						:27;
} __scm_control_sysconfig_bits;

/* CONTROL_PADCONF_X */
typedef struct {
 	__REG32 MUXMODE0										: 3;
 	__REG32 PULLUDENABLE0								: 1;
 	__REG32 PULLTYPESELECT0							: 1;
 	__REG32        	        						: 3;
 	__REG32 INPUTENABLE0								: 1;
 	__REG32        	        						: 7;
 	__REG32 MUXMODE1										: 3;
 	__REG32 PULLUDENABLE1								: 1;
 	__REG32 PULLTYPESELECT1							: 1;
 	__REG32        	        						: 3;
 	__REG32 INPUTENABLE1								: 1;
 	__REG32        	        						: 7;
} __scm_control_padconf_bits;

/* CONTROL_GENERAL */
typedef struct {
 	__REG32        	        						: 2;
 	__REG32 WKUPCTRLCLOCKDIV						: 1;
 	__REG32        	        						:29;
} __scm_control_general_bits;

/* SCM_CONTROL_DEVCONF0 */
typedef struct {
 	__REG32 SENSDMAREQ0     						: 1;
 	__REG32 SENSDMAREQ1     						: 1;
 	__REG32 MCBSP1_CLKS     						: 1;
 	__REG32 MCBSP1_CLKR     						: 1;
 	__REG32 MCBSP1_FSR									: 1;
 	__REG32        	        						: 1;
 	__REG32 MCBSP2_CLKS									: 1;
 	__REG32        	        						:25;
} __scm_control_devconf0_bits;

/* SCM_CONTROL_MEM_DFTRW0 */
typedef struct {
 	__REG32 MEMORY0DFTREADCTRL					: 2;
 	__REG32 MEMORY0DFTWRITECTRL					: 2;
 	__REG32 MEMORY0DFTGLXCTRL						: 1;
 	__REG32 MEMORY1DFTREADCTRL					: 2;
 	__REG32 MEMORY1DFTWRITECTRL					: 2;
 	__REG32        	        						: 1;
 	__REG32 MEMORY2DFTREADCTRL					: 2;
 	__REG32 MEMORY2DFTWRITECTRL					: 2;
 	__REG32 MEMORY3DFTGLXCTRL						: 1;
 	__REG32 MEMORY4DFTGLXCTRL						: 1;
 	__REG32        	        						:16;
} __scm_control_mem_dftrw0_bits;

/* SCM_CONTROL_MEM_DFTRW1 */
typedef struct {
 	__REG32        	        						: 8;
 	__REG32 MEMORY6DFTGLXCTRL						: 1;
 	__REG32 MEMORY7DFTREADCTRL					: 2;
 	__REG32 MEMORY7DFTWRITECTRL					: 2;
 	__REG32        	        						: 5;
 	__REG32 MEMORY9DFTREADCTRL					: 2;
 	__REG32 MEMORY9DFTWRITECTRL					: 2;
 	__REG32 MEMORY10DFTREADCTRL					: 2;
 	__REG32 MEMORY10DFTWRITECTRL				: 2;
 	__REG32        	        						: 5;
 	__REG32 DFTREADWRITEENABLE					: 1;
} __scm_control_mem_dftrw1_bits;

/* SCM_CONTROL_MSUSPENDMUX_0 */
typedef struct {
 	__REG32        	        						:12;
 	__REG32 I2C1MSCTRL									: 3;
 	__REG32 I2C2MSCTRL									: 3;
 	__REG32 MCBSP1MSCTRL								: 3;
 	__REG32 MCBSP2MSCTRL								: 3;
 	__REG32        	        						: 8;
} __scm_control_msuspendmux_0_bits;

/* SCM_CONTROL_MSUSPENDMUX_1 */
typedef struct {
 	__REG32        	        						: 9;
 	__REG32 GPTM1MSCTRL									: 3;
 	__REG32 GPTM2MSCTRL									: 3;
 	__REG32 GPTM3MSCTRL 								: 3;
 	__REG32 GPTM4MSCTRL									: 3;
 	__REG32 GPTM5MSCTRL									: 3;
 	__REG32 GPTM6MSCTRL									: 3;
 	__REG32 GPTM7MSCTRL									: 3;
 	__REG32        	        						: 2;
} __scm_control_msuspendmux_1_bits;

/* CONTROL_MSUSPENDMUX_2 */
typedef struct {
 	__REG32 GPTM8MSCTRL     						: 3;
 	__REG32 GPTM9MSCTRL     						: 3;
 	__REG32 GPTM10MSCTRL								: 3;
 	__REG32 GPTM11MSCTRL								: 3;
 	__REG32 GPTM12MSCTRL								: 3;
 	__REG32 WD1MSCTRL										: 3;
 	__REG32 WD2MSCTRL										: 3;
 	__REG32 WD3MSCTRL										: 3;
 	__REG32        	        						: 3;
 	__REG32 SYNCTMMSCTRL								: 3;
 	__REG32        	        						: 2;
} __scm_control_msuspendmux_2_bits;

/* SCM_CONTROL_MSUSPENDMUX_4 */
typedef struct {
 	__REG32        	        						:27;
 	__REG32 DMAMSCTRL										: 3;
 	__REG32        	        						: 2;
} __scm_control_msuspendmux_4_bits;

/* SCM_CONTROL_MSUSPENDMUX_5 */
typedef struct {
 	__REG32 MCBSP3MSCTRL								: 3;
 	__REG32 MCBSP4MSCTRL								: 3;
 	__REG32 MCBSP5MSCTRL								: 3;
 	__REG32        	        						:12;
 	__REG32 I2C3MSCTRL									: 3;
 	__REG32        	        						: 8;
} __scm_control_msuspendmux_5_bits;

/* SCM_CONTROL_MSUSPENDMUX_6 */
typedef struct {
 	__REG32 HECCMSCTRL									: 3;
 	__REG32 CPGMACMSCTRL								: 3;
 	__REG32 USB20OTGMSCTRL							: 3;
 	__REG32        	        						:23;
} __scm_control_msuspendmux_6_bits;

/* CONTROL_DEVCONF1 */
typedef struct {
 	__REG32 MCBSP3_CLKS									: 1;
 	__REG32        	        						: 1;
 	__REG32 MCBSP4_CLKS									: 1;
 	__REG32        	        						: 1;
 	__REG32 MCBSP5_CLKS									: 1;
 	__REG32        	        						: 1;
 	__REG32 MMCSDIO2ADPCLKISEL					: 1;
 	__REG32 SENSDMAREQ2									: 1;
 	__REG32 SENSDMAREQ3									: 1;
 	__REG32 MPUFORCEWRNP								: 1;
 	__REG32        	        						: 1;
 	__REG32 TVACEN											: 1;
 	__REG32        	        						: 6;
 	__REG32 TVOUTBYPASS									: 1;
 	__REG32        	        						:13;
} __scm_control_devconf1_bits;

/* SCM_CONTROL_SEC_STATUS */
typedef struct {
 	__REG32 POWERONRESET								: 1;
 	__REG32 GLOBALWARMRESET							: 1;
 	__REG32        	        						: 2;
 	__REG32 RAMBISTSTARTED							: 1;
 	__REG32 MPUWKUPRST									: 1;
 	__REG32 COREWKUPRST     						: 1;
 	__REG32 EMUWKUPRST									: 1;
 	__REG32 PERWKUPRST									: 1;
 	__REG32           									: 1;
 	__REG32 DISPWKUPRST									: 1;
 	__REG32 SGXWKUPRST									: 1;
 	__REG32        	        						: 2;
 	__REG32 NEONWKUPRST									: 1;
 	__REG32 USBHOSTWKUPRST							: 1;
 	__REG32 COREBANK1ISDESTROYED				: 1;
 	__REG32 COREBANK2ISDESTROYED				: 1;
 	__REG32        	        						: 2;
 	__REG32 MPUL1ISDESTROYED						: 1;
 	__REG32        	        						: 1;
 	__REG32 MPUL2ISDESTROYED						: 1;
 	__REG32        	        						: 1;
 	__REG32 COREBANK1ISNOTACCESSIBLE		: 1;
 	__REG32 COREBANK2ISNOTACCESSIBLE		: 1;
 	__REG32        	        						: 2;
 	__REG32 MPUL1ISNOTACCESSIBLE				: 1;
 	__REG32        	        						: 1;
 	__REG32 MPUL2ISNOTACCESSIBLE				: 1;
 	__REG32        	        						: 1;
} __scm_control_sec_status_bits;

/* SCM_CONTROL_SEC_ERR_STATUS */
typedef struct {
 	__REG32 OCMROMFWERROR								: 1;
 	__REG32 OCMRAMFWERROR								: 1;
 	__REG32 GPMCFWERROR									: 1;
 	__REG32 SMSFUNCFWERROR							: 1;
 	__REG32 SMSFWERROR									: 1;
 	__REG32        	        						: 2;
 	__REG32 L4COREFWERROR								: 1;
 	__REG32 SYSDMAACCERROR							: 1;
 	__REG32        	        						: 1;
 	__REG32 DISPDMAACCERROR  						: 1;
 	__REG32 SECMODFWERROR								: 1;
 	__REG32 SMXAPERTFWERROR							: 1;
 	__REG32           									: 3;
 	__REG32 L4PERIPHFWERROR							: 1;
 	__REG32 L4EMUFWERROR								: 1;
 	__REG32        	        						:14;
} __scm_control_sec_err_status_bits;

/* SCM_CONTROL_SEC_ERR_STATUS_DEBUG */
typedef struct {
 	__REG32 OCMROMDBGFWERROR						: 1;
 	__REG32 OCMRAMDBGFWERROR						: 1;
 	__REG32 GPMCDBGFWERROR							: 1;
 	__REG32 SMSDBGFWERROR								: 1;
 	__REG32        	        						: 3;
 	__REG32 L4COREDBGFWERROR						: 1;
 	__REG32        	        						: 4;
 	__REG32 SMXAPERTDBGFWERROR					: 1;
 	__REG32        	        						: 3;
 	__REG32 L4PERIPHERALDBGFWERROR			: 1;
 	__REG32 L4EMUDBGFWERROR							: 1;
 	__REG32        	        						:14;
} __scm_control_sec_err_status_debug_bits;

/* SCM_CONTROL_STATUS */
typedef struct {
 	__REG32 SYS_BOOT										: 6;
 	__REG32        	        						: 2;
 	__REG32 DEVICETYPE									: 3;
 	__REG32        	        						:21;
} __scm_control_status_bits;

/* SCM_CONTROL_FUSE_EMAC_LSB */
typedef struct {
 	__REG32 FUSE_OPP_95_72							:24;
 	__REG32        	        						: 8;
} __scm_control_fuse_emac_lsb_bits;

/* SCM_CONTROL_FUSE_EMAC_MSB */
typedef struct {
 	__REG32 FUSE_OPP_119_96							:24;
 	__REG32        	        						: 8;
} __scm_control_fuse_emac_msb_bits;

/* SCM_CONTROL_FUSE_SR */
typedef struct {
 	__REG32 FUSE_SR1										: 8;
 	__REG32 FUSE_SR2										: 8;
 	__REG32        	        						:16;
} __scm_control_fuse_sr_bits;

/* SCM_CONTROL_DEBOBS_0 */
typedef struct {
 	__REG32 OBSMUX1											: 7;
 	__REG32        	        						: 9;
 	__REG32 OBSMUX0											: 7;
 	__REG32        	        						: 9;
} __scm_control_debobs_0_bits;

/* SCM_CONTROL_DEBOBS_1 */
typedef struct {
 	__REG32 OBSMUX3											: 7;
 	__REG32        	        						: 9;
 	__REG32 OBSMUX2											: 7;
 	__REG32        	        						: 9;
} __scm_control_debobs_1_bits;

/* SCM_CONTROL_DEBOBS_2 */
typedef struct {
 	__REG32 OBSMUX5											: 7;
 	__REG32        	        						: 9;
 	__REG32 OBSMUX4											: 7;
 	__REG32        	        						: 9;
} __scm_control_debobs_2_bits;

/* SCM_CONTROL_DEBOBS_3 */
typedef struct {
 	__REG32 OBSMUX7											: 7;
 	__REG32        	        						: 9;
 	__REG32 OBSMUX6											: 7;
 	__REG32        	        						: 9;
} __scm_control_debobs_3_bits;

/* SCM_CONTROL_DEBOBS_4 */
typedef struct {
 	__REG32 OBSMUX9											: 7;
 	__REG32        	        						: 9;
 	__REG32 OBSMUX8											: 7;
 	__REG32        	        						: 9;
} __scm_control_debobs_4_bits;

/* SCM_CONTROL_DEBOBS_5 */
typedef struct {
 	__REG32 OBSMUX11										: 7;
 	__REG32        	        						: 9;
 	__REG32 OBSMUX10										: 7;
 	__REG32        	        						: 9;
} __scm_control_debobs_5_bits;

/* SCM_CONTROL_DEBOBS_6 */
typedef struct {
 	__REG32 OBSMUX13										: 7;
 	__REG32        	        						: 9;
 	__REG32 OBSMUX12										: 7;
 	__REG32        	        						: 9;
} __scm_control_debobs_6_bits;

/* SCM_CONTROL_DEBOBS_7 */
typedef struct {
 	__REG32 OBSMUX15										: 7;
 	__REG32        	        						: 9;
 	__REG32 OBSMUX14										: 7;
 	__REG32        	        						: 9;
} __scm_control_debobs_7_bits;

/* SCM_CONTROL_DEBOBS_8 */
typedef struct {
 	__REG32 OBSMUX17										: 7;
 	__REG32        	        						: 9;
 	__REG32 OBSMUX16										: 7;
 	__REG32        	        						: 9;
} __scm_control_debobs_8_bits;

/* SCM_CONTROL_WKUP_CTRL */
typedef struct {
 	__REG32 MM_FSUSB1_TXEN_N_OUT_POLARITY_CTRL	: 1;
 	__REG32 MM_FSUSB2_TXEN_N_OUT_POLARITY_CTRL	: 1;
 	__REG32 MM_FSUSB3_TXEN_N_OUT_POLARITY_CTRL	: 1;
 	__REG32        	 					       						:29;
} __scm_control_wkup_ctrl_bits;

/* SCM_CONTROL_DSS_DPLL_SPREADING */
typedef struct {
 	__REG32 DSS_SPREADING_RATE					: 2;
 	__REG32 DSS_SPREADING_AMPLITUDE			: 2;
 	__REG32 DSS_SPREADING_ENABLE				: 1;
 	__REG32        	 					      		: 2;
 	__REG32 DSS_SPREADING_ENABLE_STATUS	: 1;
 	__REG32        	 					      		:24;
} __scm_control_dss_dpll_spreading_bits;

/* SCM_CONTROL_DSS_DPLL_SPREADING */
typedef struct {
 	__REG32 CORE_SPREADING_RATE					: 2;
 	__REG32 CORE_SPREADING_AMPLITUDE		: 2;
 	__REG32 CORE_SPREADING_ENABLE				: 1;
 	__REG32        	 					      		: 2;
 	__REG32 CORE_SPREADING_ENABLE_STATUS: 1;
 	__REG32        	 					      		:24;
} __scm_control_core_dpll_spreading_bits;

/* SCM_CONTROL_PER_DPLL_SPREADING */
typedef struct {
 	__REG32 PER_SPREADING_RATE					: 2;
 	__REG32 PER_SPREADING_AMPLITUDE			: 2;
 	__REG32 PER_SPREADING_ENABLE				: 1;
 	__REG32        	 					      		: 2;
 	__REG32 PER_SPREADING_ENABLE_STATUS	: 1;
 	__REG32        	 					      		:24;
} __scm_control_per_dpll_spreading_bits;

/* SCM_CONTROL_USBHOST_DPLL_SPREADING */
typedef struct {
 	__REG32 USBHOST_SPREADING_RATE					: 2;
 	__REG32 USBHOST_SPREADING_AMPLITUDE			: 2;
 	__REG32 USBHOST_SPREADING_ENABLE				: 1;
 	__REG32        	 					      				: 2;
 	__REG32 USBHOST_SPREADING_ENABLE_STATUS	: 1;
 	__REG32        	 					      				:24;
} __scm_control_usbhost_dpll_spreading_bits;

/* SCM_CONTROL_DPF_OCM_RAM_FW_ADDR_MATCH */
typedef struct {
 	__REG32 REGIONOCMRAMFWADDRMATCH					:20;
 	__REG32        	 					      				:12;
} __scm_control_dpf_ocm_ram_fw_addr_match_bits;

/* SCM_CONTROL_DPF_OCM_RAM_FW_REQINFO */
typedef struct {
 	__REG32 REGIONOCMRAMFWREQINFO						:16;
 	__REG32        	 					      				:16;
} __scm_control_dpf_ocm_ram_fw_reqinfo_bits;

/* SCM_CONTROL_DPF_OCM_RAM_FW_WR */
typedef struct {
 	__REG32 REGIONOCMRAMFWWR								:16;
 	__REG32        	 					      				:16;
} __scm_control_dpf_ocm_ram_fw_wr_bits;

/* SCM_CONTROL_DPF_REGION4_GPMC_FW_ADDR_MATCH */
typedef struct {
 	__REG32 REGION4GPMCFWADDRMATCH					:30;
 	__REG32        	 					      				: 2;
} __scm_control_dpf_region4_gpmc_fw_addr_match_bits;

/* SCM_CONTROL_DPF_REGION4_GPMC_FW_WR */
typedef struct {
 	__REG32 REGION4GPMCFWWR									:16;
 	__REG32        	 					      				:16;
} __scm_control_dpf_region4_gpmc_fw_wr_bits;

/* SCM_CONTROL_APE_FW_DEFAULT_SECURE_LOCK */
typedef struct {
 	__REG32 GPMCDEFAULTSECURELOCK						: 1;
 	__REG32 IPSSDEFAULTSECURELOCK						: 1;
 	__REG32 OCMRAMDEFAULTSECURELOCK					: 1;
 	__REG32 SMSDEFAULTSECURELOCK						: 1;
 	__REG32        	 					      				: 4;
 	__REG32 GPMCDEFAULTDEBUGLOCK						: 1;
 	__REG32 IPSSDEFAULTDEBUGLOCK						: 1;
 	__REG32 OCMRAMDEFAULTDEBUGLOCK					: 1;
 	__REG32 SMSDEFAULTDEBUGLOCK							: 1;
 	__REG32        	 					      				: 4;
 	__REG32 L4COREAPDEFAULTSECURELOCK				: 1;
 	__REG32 L4CRYPTODEFAULTSECURELOCK				: 1;
 	__REG32 L4DISPDEFAULTSECURELOCK					: 1;
 	__REG32        	 					      				: 1;
 	__REG32 L4TIMERDEFAULTSECURELOCK				: 1;
 	__REG32        	 					      				: 3;
 	__REG32 L4COREAPDEFAULTDEBUGLOCK				: 1;
 	__REG32 L4CRYPTODEFAULTDEBUGLOCK				: 1;
 	__REG32 L4DISPDEFAULTDEBUGLOCK					: 1;
 	__REG32        	 					      				: 1;
 	__REG32 L4TIMERDEFAULTDEBUGLOCK					: 1;
 	__REG32        	 					      				: 3;
} __scm_control_ape_fw_default_secure_lock_bits;

/* SCM_CONTROL_OCMROM_SECURE_DEBUG */
typedef struct {
 	__REG32 OCMROMSECUREDEBUG								: 1;
 	__REG32        	 					      				:31;
} __scm_control_ocmrom_secure_debug_bits;

/* SCM_CONTROL_EXT_SEC_CONTROL */
typedef struct {
 	__REG32 SECUREEXECINDICATOR							: 1;
 	__REG32 CCSECURITYDISABLE								: 1;
 	__REG32 I2CSENABLE											: 1;
 	__REG32        	 					      				:29;
} __scm_control_ext_sec_control_bits;

/* SCM_CONTROL_DEVCONF2 */
typedef struct {
 	__REG32 VPFE_PCLK_INVERT_EN							: 1;
 	__REG32 USBOTG_DATAPOLARITY							: 1;
 	__REG32 USBOTG_POWERDOWNOTG							: 1;
 	__REG32 USBOTG_PHY_PD										: 1;
 	__REG32 USBOTG_PHY_RESET								: 1;
 	__REG32 USBOTG_PHY_PLLON								: 1;
 	__REG32 USBOTG_VBUSSENSE								: 1;
 	__REG32 USBOTG_PWRCLKGOOD								: 1;
 	__REG32 USBOTG_SELINPUTCLKFREQ					: 4;
 	__REG32 USBOTG_VBUSDETECTEN							: 1;
 	__REG32 USBOTG_SESSENDEN								: 1;
 	__REG32 USBOTG_OTGMODE									: 2;
 	__REG32 FUSE_AVDAC_DISABLE							: 1;
 	__REG32        	 					      				: 1;
 	__REG32 FUNC_MODE_SEL										: 1;
 	__REG32 FUNC_SYS32K_SEL									: 1;
 	__REG32        	 					      				: 3;
 	__REG32 USBPHY_GPIO_MODE								: 1;
 	__REG32        	 					      				: 2;
 	__REG32 VDD_HVMODEOUT										: 1;
 	__REG32        	 					      				: 5;
} __scm_control_devconf2_bits;

/* SCM_CONTROL_DEVCONF3 */
typedef struct {
 	__REG32 DDR_CONFIG_TERMON								: 2;
 	__REG32 DDR_CONFIG_TERMOFF							: 2;
 	__REG32 																: 1;
 	__REG32 VTP_READY												: 1;
 	__REG32 DDR_VREF_TAP										: 2;
 	__REG32 DDR_VREF_EN											: 1;
 	__REG32 VTP_DRVSTRENGTH									: 3;
 	__REG32 DDR_CMOSEN											: 1;
 	__REG32 VTP_PWRSAVE											: 1;
 	__REG32 EMIF4A_FCLKEN										: 1;
 	__REG32 DDRPHY_CLK_EN										: 1;
 	__REG32        	 					      				:16;
} __scm_control_devconf3_bits;

/* SCM_CONTROL_CBA_PRIORITY */
typedef struct {
 	__REG32 CBA_VPFE_PRIORITY								: 3;
 	__REG32 																: 1;
 	__REG32 CBA_EMAC_PRIORITY								: 3;
 	__REG32 																: 1;
 	__REG32 CBA_USBM_PRIORITY								: 3;
 	__REG32 																: 1;
 	__REG32 CBA_USBCDMA_PRIORITY						: 3;
 	__REG32 																:17;
} __scm_control_cba_priority_bits;

/* SCM_CONTROL_LVL_INTR_CLEAR */
typedef struct {
 	__REG32 CPGMAC_C0_MISC_PULSE_CLR				: 1;
 	__REG32 CPGMAC_C0_RX_PULSE_CLR					: 1;
 	__REG32 CPGMAC_C0_RX_THRESH_PULSE_CLR		: 1;
 	__REG32 CPGMAC_C0_TX_PULSE_CLR					: 1;
 	__REG32 USB20OTGSS_USB_INT_CLR					: 1;
 	__REG32 VPFE_CCDC_VD0_INT_CLR						: 1;
 	__REG32 VPFE_CCDC_VD1_INT_CLR						: 1;
 	__REG32 VPFE_CCDC_VD2_INT_CLR						: 1;
 	__REG32 																:24;
} __scm_control_lvl_intr_clear_bits;

/* SCM_CONTROL_LVL_INTR_CLEAR */
typedef struct {
 	__REG32 USB20OTGSS_SW_RST								: 1;
 	__REG32 CPGMACSS_SW_RST									: 1;
 	__REG32 VPFE_VBUSP_SW_RST								: 1;
 	__REG32 HECC_SW_RST											: 1;
 	__REG32 VPFE_PCLK_SW_RST								: 1;
 	__REG32 																:27;
} __scm_control_ip_sw_reset_bits;

/* SCM_CONTROL_IPSS_CLK_CTRL */
typedef struct {
 	__REG32 USB20OTG_VBUSP_CLK_EN						: 1;
 	__REG32 CPGMAC_VBUSP_CLK_EN							: 1;
 	__REG32 VPFE_VBUSP_CLK_EN								: 1;
 	__REG32 HECC_VBUSP_CLK_EN								: 1;
 	__REG32 USB20OTG_VBUSP_CLK_EN_ACK				: 1;
 	__REG32 CPGMAC_VBUSP_CLK_EN_ACK					: 1;
 	__REG32 VPFE_VBUSP_CLK_EN_ACK						: 1;
 	__REG32 HECC_VBUSP_CLK_EN_ACK						: 1;
 	__REG32 USB20OTG_FUNC_CLK_EN						: 1;
 	__REG32 CPGMAC_FUNC_CLK_EN							: 1;
 	__REG32 VPFE_FUNC_CLK_EN								: 1;
 	__REG32 																:21;
} __scm_control_ipss_clk_ctrl_bits;

/* SCM_CONTROL_SEC_TAP */
typedef struct {
 	__REG32 MPUTAPENABLE										: 1;
 	__REG32 CPEFUSETAPENABLE								: 1;
 	__REG32 ETBTAPENABLE										: 1;
 	__REG32 CHIPLEVELTAPENABLE							: 1;
 	__REG32 EFUSETAPENABLE									: 1;
 	__REG32 SDTITAPENABLE										: 1;
 	__REG32 																: 2;
 	__REG32 SUBTAPCTRLDISABLE								: 1;
 	__REG32 																: 1;
 	__REG32 OCTTAPENABLE										: 1;
 	__REG32 																: 3;
 	__REG32 SR2TAPENABLE										: 1;
 	__REG32 																:17;
} __scm_control_sec_tap_bits;

/* SCM_CONTROL_SEC_EMU */
typedef struct {
 	__REG32 GENDBGENABLE										: 2;
 	__REG32 ETMSECPRIVDBGENABLE							: 1;
 	__REG32 ICESECPRIVDBGENABLE							: 1;
 	__REG32 																:28;
} __scm_control_sec_emu_bits;

/* SCM_CONTROL_WKUP_DEBOBS_0 */
typedef struct {
 	__REG32 OBSMUX0													: 5;
 	__REG32 																: 3;
 	__REG32 OBSMUX1													: 5;
 	__REG32 																: 3;
 	__REG32 OBSMUX2													: 5;
 	__REG32 																: 3;
 	__REG32 OBSMUX3													: 5;
 	__REG32 																: 3;
} __scm_control_wkup_debobs_0_bits;

/* SCM_CONTROL_WKUP_DEBOBS_1 */
typedef struct {
 	__REG32 OBSMUX4													: 5;
 	__REG32 																: 3;
 	__REG32 OBSMUX5													: 5;
 	__REG32 																: 3;
 	__REG32 OBSMUX6													: 5;
 	__REG32 																: 3;
 	__REG32 OBSMUX7													: 5;
 	__REG32 																: 3;
} __scm_control_wkup_debobs_1_bits;

/* SCM_CONTROL_WKUP_DEBOBS_2 */
typedef struct {
 	__REG32 OBSMUX8													: 5;
 	__REG32 																: 3;
 	__REG32 OBSMUX9													: 5;
 	__REG32 																: 3;
 	__REG32 OBSMUX10												: 5;
 	__REG32 																: 3;
 	__REG32 OBSMUX11												: 5;
 	__REG32 																: 3;
} __scm_control_wkup_debobs_2_bits;

/* SCM_CONTROL_WKUP_DEBOBS_3 */
typedef struct {
 	__REG32 OBSMUX12												: 5;
 	__REG32 																: 3;
 	__REG32 OBSMUX13												: 5;
 	__REG32 																: 3;
 	__REG32 OBSMUX14												: 5;
 	__REG32 																: 3;
 	__REG32 OBSMUX15												: 5;
 	__REG32 																: 3;
} __scm_control_wkup_debobs_3_bits;

/* SCM_CONTROL_WKUP_DEBOBS_4 */
typedef struct {
 	__REG32 OBSMUX16												: 5;
 	__REG32 																: 3;
 	__REG32 OBSMUX17												: 5;
 	__REG32 																:19;
} __scm_control_wkup_debobs_4_bits;

/* SCM_CONTROL_SEC_DAP */
typedef struct {
 	__REG32 FORCEDAPPUBUSERDEBUGEN					: 1;
 	__REG32 																: 1;
 	__REG32 FORCEDAPSECPUBLICDEBUGEN				: 1;
 	__REG32 FORCEDAPSECUSERDEBUGEN					: 1;
 	__REG32 																:28;
} __scm_control_sec_dap_bits;

/* DMA4_REVISION */
typedef struct {
 	__REG32 REV															: 8;
 	__REG32 																:24;
} __dma4_revision_bits;

/* DMA4_IRQSTATUS_Lj */
typedef struct {
 	__REG32 CH0															: 1;
 	__REG32 CH1															: 1;
 	__REG32 CH2															: 1;
 	__REG32 CH3															: 1;
 	__REG32 CH4															: 1;
 	__REG32 CH5															: 1;
 	__REG32 CH6															: 1;
 	__REG32 CH7															: 1;
 	__REG32 CH8															: 1;
 	__REG32 CH9															: 1;
 	__REG32 CH10														: 1;
 	__REG32 CH11														: 1;
 	__REG32 CH12														: 1;
 	__REG32 CH13														: 1;
 	__REG32 CH14														: 1;
 	__REG32 CH15														: 1;
 	__REG32 CH16														: 1;
 	__REG32 CH17														: 1;
 	__REG32 CH18														: 1;
 	__REG32 CH19														: 1;
 	__REG32 CH20														: 1;
 	__REG32 CH21														: 1;
 	__REG32 CH22														: 1;
 	__REG32 CH23														: 1;
 	__REG32 CH24														: 1;
 	__REG32 CH25														: 1;
 	__REG32 CH26														: 1;
 	__REG32 CH27		 												: 1;
 	__REG32 CH28 														: 1;
 	__REG32 CH29														: 1;
 	__REG32 CH30														: 1;
 	__REG32 CH31														: 1;
} __dma4_irqstatus_bits;

/* DMA4_SYSSTATUS */
typedef struct {
 	__REG32 RESETDONE												: 1;
 	__REG32 																:31;
} __dma4_sysstatus_bits;

/* DMA4_OCP_SYSCONFIG */
typedef struct {
 	__REG32 AUTOIDLE												: 1;
 	__REG32 SOFTRESET												: 1;
 	__REG32 																: 1;
 	__REG32 SIDLEMODE												: 2;
 	__REG32 EMUFREE													: 1;
 	__REG32 																: 2;
 	__REG32 CLOCKACTIVITY										: 2;
 	__REG32 																: 2;
 	__REG32 MIDLEMODE												: 2;
 	__REG32 																:18;
} __dma4_ocp_sysconfig_bits;

/* DMA4_CAPS_0 */
typedef struct {
 	__REG32 																:18;
 	__REG32 TRANSPARENT_BLT_CPBLTY					: 1;
 	__REG32 CONST_FILL_CPBLTY								: 1;
 	__REG32 																:12;
} __dma4_caps_0_bits;

/* DMA4_CAPS_2 */
typedef struct {
 	__REG32 SRC_CONST_ADRS_CPBLTY							: 1;
 	__REG32 SRC_POST_INCREMENT_ADRS_CPBLTY		: 1;
 	__REG32 SRC_SINGLE_INDEX_ADRS_CPBLTY			: 1;
 	__REG32 SRC_DOUBLE_INDEX_ADRS_CPBLTY			: 1;
 	__REG32 DST_CONST_ADRS_CPBLTY							: 1;
 	__REG32 DST_POST_INCRMNT_ADRS_CPBLTY			: 1;
 	__REG32 DST_SINGLE_INDEX_ADRS_CPBLTY			: 1;
 	__REG32 DST_DOUBLE_INDEX_ADRS_CPBLTY			: 1;
 	__REG32 SEPARATE_SRC_AND_DST_INDEX_CPBLTY	: 1;
 	__REG32 																	:23;
} __dma4_caps_2_bits;

/* DMA4_CAPS_3 */
typedef struct {
 	__REG32 ELMNT_SYNCHR_CPBLTY								: 1;
 	__REG32 FRAME_SYNCHR_CPBLTY								: 1;
 	__REG32 																	: 2;
 	__REG32 CHANNEL_INTERLEAVE_CPBLTY					: 1;
 	__REG32 CHANNEL_CHAINING_CPBLTY						: 1;
 	__REG32 PKT_SYNCHR_CPBLTY									: 1;
 	__REG32 BLOCK_SYNCHR_CPBLTY								: 1;
 	__REG32 																	:24;
} __dma4_caps_3_bits;

/* DMA4_CAPS_4 */
typedef struct {
 	__REG32 																			: 1;
 	__REG32 EVENT_DROP_INTERRUPT_CPBLTY						: 1;
 	__REG32 HALF_FRAME_INTERRUPT_CPBLTY						: 1;
 	__REG32 FRAME_INTERRUPT_CPBLTY								: 1;
 	__REG32 LAST_FRAME_INTERRUPT_CPBLTY						: 1;
 	__REG32 BLOCK_INTERRUPT_CPBLTY								: 1;
 	__REG32 SYNC_STATUS_CPBLTY										: 1;
 	__REG32 PKT_INTERRUPT_CPBLTY									: 1;
 	__REG32 TRANS_ERR_INTERRUPT_CPBLTY						: 1;
 	__REG32 																			: 1;
 	__REG32 SUPERVISOR_ERR_INTERRUPT_CPBLTY				: 1;
 	__REG32 MISALIGNED_ADRS_ERR_INTERRUPT_CPBLTY	: 1;
 	__REG32 DRAIN_END_INTERRUPT_CPBLTY						: 1;
 	__REG32 																			:19;
} __dma4_caps_4_bits;

/* DMA4_GCR */
typedef struct {
 	__REG32 MAX_CHANNEL_FIFO_DEPTH						: 8;
 	__REG32 																	: 4;
 	__REG32 HI_THREAD_RESERVED								: 2;
 	__REG32 HI_LO_FIFO_BUDGET									: 2;
 	__REG32 ARBITRATION_RATE									: 8;
 	__REG32 																	: 8;
} __dma4_gcr_bits;

/* DMA4_CCRi */
typedef struct {
 	__REG32 SYNCHRO_CONTROL										: 5;
 	__REG32 FS																: 1;
 	__REG32 READ_PRIORITY											: 1;
 	__REG32 ENABLE														: 1;
 	__REG32 SUSPEND_SENSITIVE									: 1;
 	__REG32 RD_ACTIVE													: 1;
 	__REG32 WR_ACTIVE													: 1;
 	__REG32 																	: 1;
 	__REG32 SRC_AMODE													: 2;
 	__REG32 DST_AMODE													: 2;
 	__REG32 CONST_FILL_ENABLE									: 1;
 	__REG32 TRANSPARENT_COPY_ENABLE						: 1;
 	__REG32 BS																: 1;
 	__REG32 SYNCHRO_CONTROL_UPPER							: 2;
 	__REG32 																	: 1;
 	__REG32 SUPERVISOR												: 1;
 	__REG32 PREFETCH													: 1;
 	__REG32 SEL_SRC_DST_SYNC									: 1;
 	__REG32 BUFFERING_DISABLE									: 1;
 	__REG32 WRITE_PRIORITY										: 1;
 	__REG32 																	: 5;
} __dma4_ccr_bits;

/* DMA4_CLNK_CTRLi */
typedef struct {
 	__REG32 NEXTLCH_ID												: 5;
 	__REG32 																	:10;
 	__REG32 ENABLE_LNK												: 1;
 	__REG32 																	:16;
} __dma4_clnk_ctrl_bits;

/* DMA4_CICRi */
typedef struct {
 	__REG32 																	: 1;
 	__REG32 DROP_IE														: 1;
 	__REG32 HALF_IE														: 1;
 	__REG32 FRAME_IE													: 1;
 	__REG32 LAST_IE														: 1;
 	__REG32 BLOCK_IE													: 1;
 	__REG32 																	: 1;
 	__REG32 PKT_IE														: 1;
 	__REG32 TRANS_ERR_IE											: 1;
 	__REG32 																	: 1;
 	__REG32 SUPERVISOR_ERR_IE									: 1;
 	__REG32 MISALIGNED_ERR_IE									: 1;
 	__REG32 DRAIN_IE													: 1;
 	__REG32 																	:19;
} __dma4_cicr_bits;

/* DMA4_CSRi */
typedef struct {
 	__REG32 																	: 1;
 	__REG32 DROP  														: 1;
 	__REG32 HALF															: 1;
 	__REG32 FRAME															: 1;
 	__REG32 LAST															: 1;
 	__REG32 BLOCK															: 1;
 	__REG32 SYNC															: 1;
 	__REG32 PKT																: 1;
 	__REG32 TRANS_ERR													: 1;
 	__REG32 																	: 1;
 	__REG32 SUPERVISOR_ERR										: 1;
 	__REG32 MISALIGNED_ERR										: 1;
 	__REG32 DRAIN_END													: 1;
 	__REG32 																	:19;
} __dma4_csr_bits;

/* DMA4_CSDPi */
typedef struct {
 	__REG32 DATA_TYPE													: 2;
 	__REG32 		  														: 4;
 	__REG32 SRC_PACKED												: 1;
 	__REG32 SRC_BURST_EN											: 2;
 	__REG32 																	: 4;
 	__REG32 DST_PACKED												: 1;
 	__REG32 DST_BURST_EN											: 2;
 	__REG32 WRITE_MODE												: 2;
 	__REG32 DST_ENDIAN_LOCK										: 1;
 	__REG32 DST_ENDIAN												: 1;
 	__REG32 SRC_ENDIAN_LOCK										: 1;
 	__REG32 SRC_ENDIAN												: 1;
 	__REG32 																	:10;
} __dma4_csdp_bits;

/* DMA4_CENi */
typedef struct {
 	__REG32 CHANNEL_ELMNT_NBR									:24;
 	__REG32 		  														: 8;
} __dma4_cen_bits;

/* DMA4_CFNi */
typedef struct {
 	__REG32 CHANNEL_FRAME_NBR									:16;
 	__REG32 		  														:16;
} __dma4_cfn_bits;

/* DMA4_CSEIi */
typedef struct {
 	__REG32 CHANNEL_SRC_ELMNT_INDEX						:16;
 	__REG32 		  														:16;
} __dma4_csei_bits;

/* DMA4_CDEIi */
typedef struct {
 	__REG32 CHANNEL_DST_ELMNT_INDEX						:16;
 	__REG32 		  														:16;
} __dma4_cdei_bits;

/* DMA4_CCENi */
typedef struct {
 	__REG32 CURRENT_ELMNT_NBR									:24;
 	__REG32 		  														: 8;
} __dma4_ccen_bits;

/* DMA4_CCFNi */
typedef struct {
 	__REG32 CURRENT_FRAME_NBR									:16;
 	__REG32 		  														:16;
} __dma4_ccfn_bits;

/* DMA4_COLORi */
typedef struct {
 	__REG32 CH_BLT_FRGRND_COLOR_OR_SOLID_COLOR_PTRN	:16;
 	__REG32 		  																	:16;
} __dma4_color_bits;

/* INTCPS_SYSCONFIG */
typedef struct {
 	__REG32 AUTOIDLE				: 1;
 	__REG32 SOFTRESET				: 1;
 	__REG32       	 				:30;
} __intcps_sysconfig_bits;

/* INTCPS_SYSSTATUS */
typedef struct {
 	__REG32 RESETDONE				: 1;
 	__REG32       	 				:31;
} __intcps_sysstatus_bits;

/* INTCPS_SIR_IRQ */
typedef struct {
 	__REG32 ACTIVEIRQ				: 7;
 	__REG32 SPURIOUSIRQFLAG	:25;
} __intcps_sir_irq_bits;

/* INTCPS_SIR_FIQ */
typedef struct {
 	__REG32 ACTIVEFIQ				: 7;
 	__REG32 SPURIOUSFIQFLAG	:25;
} __intcps_sir_fiq_bits;

/* INTCPS_CONTROL */
typedef struct {
 	__REG32 NEWIRQAGR				: 1;
 	__REG32 NEWFIQAGR				: 1;
 	__REG32 								:30;
} __intcps_control_bits;

/* INTCPS_PROTECTION */
typedef struct {
 	__REG32 PROTECTION			: 1;
 	__REG32 								:31;
} __intcps_protection_bits;

/* INTCPS_IDLE */
typedef struct {
 	__REG32 FUNCIDLE				: 1;
 	__REG32 TURBO						: 1;
 	__REG32 								:30;
} __intcps_idle_bits;

/* INTCPS_IRQ_PRIORITY */
typedef struct {
 	__REG32 IRQPRIORITY			: 6;
 	__REG32 SPURIOUSIRQFLAG	:26;
} __intcps_irq_priority_bits;

/* INTCPS_FIQ_PRIORITY */
typedef struct {
 	__REG32 FIQPRIORITY			: 6;
 	__REG32 SPURIOUSFIQFLAG	:26;
} __intcps_fiq_priority_bits;

/* INTCPS_THRESHOLD */
typedef struct {
 	__REG32 PRIORITYTHRESHOLD	: 8;
 	__REG32 									:24;
} __intcps_threshold_bits;

/* INTCPS_ITR0 */
typedef struct {
 	__REG32 ITR0		: 1;
 	__REG32 ITR1		: 1;
 	__REG32 ITR2		: 1;
 	__REG32 ITR3		: 1;
 	__REG32 ITR4		: 1;
 	__REG32 ITR5		: 1;
 	__REG32 ITR6		: 1;
 	__REG32 ITR7		: 1;
 	__REG32 ITR8		: 1;
 	__REG32 ITR9		: 1;
 	__REG32 ITR10		: 1;
 	__REG32 ITR11		: 1;
 	__REG32 ITR12		: 1;
 	__REG32 ITR13		: 1;
 	__REG32 ITR14		: 1;
 	__REG32 ITR15		: 1;
 	__REG32 ITR16		: 1;
 	__REG32 ITR17		: 1;
 	__REG32 ITR18		: 1;
 	__REG32 ITR19		: 1;
 	__REG32 ITR20		: 1;
 	__REG32 ITR21		: 1;
 	__REG32 ITR22		: 1;
 	__REG32 ITR23		: 1;
 	__REG32 ITR24		: 1;
 	__REG32 ITR25		: 1;
 	__REG32 ITR26		: 1;
 	__REG32 ITR27		: 1;
 	__REG32 ITR28		: 1;
 	__REG32 ITR29		: 1;
 	__REG32 ITR30		: 1;
 	__REG32 ITR31		: 1;
} __intcps_itr0_bits;

/* INTCPS_ITR1 */
typedef struct {
 	__REG32 ITR32		: 1;
 	__REG32 ITR33		: 1;
 	__REG32 ITR34		: 1;
 	__REG32 ITR35		: 1;
 	__REG32 ITR36		: 1;
 	__REG32 ITR37		: 1;
 	__REG32 ITR38		: 1;
 	__REG32 ITR39		: 1;
 	__REG32 ITR40		: 1;
 	__REG32 ITR41		: 1;
 	__REG32 ITR42		: 1;
 	__REG32 ITR43		: 1;
 	__REG32 ITR44		: 1;
 	__REG32 ITR45		: 1;
 	__REG32 ITR46		: 1;
 	__REG32 ITR47		: 1;
 	__REG32 ITR48		: 1;
 	__REG32 ITR49		: 1;
 	__REG32 ITR50		: 1;
 	__REG32 ITR51		: 1;
 	__REG32 ITR52		: 1;
 	__REG32 ITR53		: 1;
 	__REG32 ITR54		: 1;
 	__REG32 ITR55		: 1;
 	__REG32 ITR56		: 1;
 	__REG32 ITR57		: 1;
 	__REG32 ITR58		: 1;
 	__REG32 ITR59		: 1;
 	__REG32 ITR60		: 1;
 	__REG32 ITR61		: 1;
 	__REG32 ITR62		: 1;
 	__REG32 ITR63		: 1;
} __intcps_itr1_bits;

/* INTCPS_ITR2 */
typedef struct {
 	__REG32 ITR64		: 1;
 	__REG32 ITR65		: 1;
 	__REG32 ITR66		: 1;
 	__REG32 ITR67		: 1;
 	__REG32 ITR68		: 1;
 	__REG32 ITR69		: 1;
 	__REG32 ITR70		: 1;
 	__REG32 ITR71		: 1;
 	__REG32 ITR72		: 1;
 	__REG32 ITR73		: 1;
 	__REG32 ITR74		: 1;
 	__REG32 ITR75		: 1;
 	__REG32 ITR76		: 1;
 	__REG32 ITR77		: 1;
 	__REG32 ITR78		: 1;
 	__REG32 ITR79		: 1;
 	__REG32 ITR80		: 1;
 	__REG32 ITR81		: 1;
 	__REG32 ITR82		: 1;
 	__REG32 ITR83		: 1;
 	__REG32 ITR84		: 1;
 	__REG32 ITR85		: 1;
 	__REG32 ITR86		: 1;
 	__REG32 ITR87		: 1;
 	__REG32 ITR88		: 1;
 	__REG32 ITR89		: 1;
 	__REG32 ITR90		: 1;
 	__REG32 ITR91		: 1;
 	__REG32 ITR92		: 1;
 	__REG32 ITR93		: 1;
 	__REG32 ITR94		: 1;
 	__REG32 ITR95		: 1;
} __intcps_itr2_bits;

/* INTCPS_MIR0 */
typedef struct {
 	__REG32 MIR0		: 1;
 	__REG32 MIR1		: 1;
 	__REG32 MIR2		: 1;
 	__REG32 MIR3		: 1;
 	__REG32 MIR4		: 1;
 	__REG32 MIR5		: 1;
 	__REG32 MIR6		: 1;
 	__REG32 MIR7		: 1;
 	__REG32 MIR8		: 1;
 	__REG32 MIR9		: 1;
 	__REG32 MIR10		: 1;
 	__REG32 MIR11		: 1;
 	__REG32 MIR12		: 1;
 	__REG32 MIR13		: 1;
 	__REG32 MIR14		: 1;
 	__REG32 MIR15		: 1;
 	__REG32 MIR16		: 1;
 	__REG32 MIR17		: 1;
 	__REG32 MIR18		: 1;
 	__REG32 MIR19		: 1;
 	__REG32 MIR20		: 1;
 	__REG32 MIR21		: 1;
 	__REG32 MIR22		: 1;
 	__REG32 MIR23		: 1;
 	__REG32 MIR24		: 1;
 	__REG32 MIR25		: 1;
 	__REG32 MIR26		: 1;
 	__REG32 MIR27		: 1;
 	__REG32 MIR28		: 1;
 	__REG32 MIR29		: 1;
 	__REG32 MIR30		: 1;
 	__REG32 MIR31		: 1;
} __intcps_mir0_bits;

/* INTCPS_MIR1 */
typedef struct {
 	__REG32 MIR32		: 1;
 	__REG32 MIR33		: 1;
 	__REG32 MIR34		: 1;
 	__REG32 MIR35		: 1;
 	__REG32 MIR36		: 1;
 	__REG32 MIR37		: 1;
 	__REG32 MIR38		: 1;
 	__REG32 MIR39		: 1;
 	__REG32 MIR40		: 1;
 	__REG32 MIR41		: 1;
 	__REG32 MIR42		: 1;
 	__REG32 MIR43		: 1;
 	__REG32 MIR44		: 1;
 	__REG32 MIR45		: 1;
 	__REG32 MIR46		: 1;
 	__REG32 MIR47		: 1;
 	__REG32 MIR48		: 1;
 	__REG32 MIR49		: 1;
 	__REG32 MIR50		: 1;
 	__REG32 MIR51		: 1;
 	__REG32 MIR52		: 1;
 	__REG32 MIR53		: 1;
 	__REG32 MIR54		: 1;
 	__REG32 MIR55		: 1;
 	__REG32 MIR56		: 1;
 	__REG32 MIR57		: 1;
 	__REG32 MIR58		: 1;
 	__REG32 MIR59		: 1;
 	__REG32 MIR60		: 1;
 	__REG32 MIR61		: 1;
 	__REG32 MIR62		: 1;
 	__REG32 MIR63		: 1;
} __intcps_mir1_bits;

/* INTCPS_MIR2 */
typedef struct {
 	__REG32 MIR64		: 1;
 	__REG32 MIR65		: 1;
 	__REG32 MIR66		: 1;
 	__REG32 MIR67		: 1;
 	__REG32 MIR68		: 1;
 	__REG32 MIR69		: 1;
 	__REG32 MIR70		: 1;
 	__REG32 MIR71		: 1;
 	__REG32 MIR72		: 1;
 	__REG32 MIR73		: 1;
 	__REG32 MIR74		: 1;
 	__REG32 MIR75		: 1;
 	__REG32 MIR76		: 1;
 	__REG32 MIR77		: 1;
 	__REG32 MIR78		: 1;
 	__REG32 MIR79		: 1;
 	__REG32 MIR80		: 1;
 	__REG32 MIR81		: 1;
 	__REG32 MIR82		: 1;
 	__REG32 MIR83		: 1;
 	__REG32 MIR84		: 1;
 	__REG32 MIR85		: 1;
 	__REG32 MIR86		: 1;
 	__REG32 MIR87		: 1;
 	__REG32 MIR88		: 1;
 	__REG32 MIR89		: 1;
 	__REG32 MIR90		: 1;
 	__REG32 MIR91		: 1;
 	__REG32 MIR92		: 1;
 	__REG32 MIR93		: 1;
 	__REG32 MIR94		: 1;
 	__REG32 MIR95		: 1;
} __intcps_mir2_bits;

/* INTCPS_MIR_CLEAR0 */
typedef struct {
 	__REG32 MIRCLEAR0		: 1;
 	__REG32 MIRCLEAR1		: 1;
 	__REG32 MIRCLEAR2		: 1;
 	__REG32 MIRCLEAR3		: 1;
 	__REG32 MIRCLEAR4		: 1;
 	__REG32 MIRCLEAR5		: 1;
 	__REG32 MIRCLEAR6		: 1;
 	__REG32 MIRCLEAR7		: 1;
 	__REG32 MIRCLEAR8		: 1;
 	__REG32 MIRCLEAR9		: 1;
 	__REG32 MIRCLEAR10		: 1;
 	__REG32 MIRCLEAR11		: 1;
 	__REG32 MIRCLEAR12		: 1;
 	__REG32 MIRCLEAR13		: 1;
 	__REG32 MIRCLEAR14		: 1;
 	__REG32 MIRCLEAR15		: 1;
 	__REG32 MIRCLEAR16		: 1;
 	__REG32 MIRCLEAR17		: 1;
 	__REG32 MIRCLEAR18		: 1;
 	__REG32 MIRCLEAR19		: 1;
 	__REG32 MIRCLEAR20		: 1;
 	__REG32 MIRCLEAR21		: 1;
 	__REG32 MIRCLEAR22		: 1;
 	__REG32 MIRCLEAR23		: 1;
 	__REG32 MIRCLEAR24		: 1;
 	__REG32 MIRCLEAR25		: 1;
 	__REG32 MIRCLEAR26		: 1;
 	__REG32 MIRCLEAR27		: 1;
 	__REG32 MIRCLEAR28		: 1;
 	__REG32 MIRCLEAR29		: 1;
 	__REG32 MIRCLEAR30		: 1;
 	__REG32 MIRCLEAR31		: 1;
} __intcps_mir_clear0_bits;

/* INTCPS_MIR_CLEAR1 */
typedef struct {
 	__REG32 MIRCLEAR32		: 1;
 	__REG32 MIRCLEAR33		: 1;
 	__REG32 MIRCLEAR34		: 1;
 	__REG32 MIRCLEAR35		: 1;
 	__REG32 MIRCLEAR36		: 1;
 	__REG32 MIRCLEAR37		: 1;
 	__REG32 MIRCLEAR38		: 1;
 	__REG32 MIRCLEAR39		: 1;
 	__REG32 MIRCLEAR40		: 1;
 	__REG32 MIRCLEAR41		: 1;
 	__REG32 MIRCLEAR42		: 1;
 	__REG32 MIRCLEAR43		: 1;
 	__REG32 MIRCLEAR44		: 1;
 	__REG32 MIRCLEAR45		: 1;
 	__REG32 MIRCLEAR46		: 1;
 	__REG32 MIRCLEAR47		: 1;
 	__REG32 MIRCLEAR48		: 1;
 	__REG32 MIRCLEAR49		: 1;
 	__REG32 MIRCLEAR50		: 1;
 	__REG32 MIRCLEAR51		: 1;
 	__REG32 MIRCLEAR52		: 1;
 	__REG32 MIRCLEAR53		: 1;
 	__REG32 MIRCLEAR54		: 1;
 	__REG32 MIRCLEAR55		: 1;
 	__REG32 MIRCLEAR56		: 1;
 	__REG32 MIRCLEAR57		: 1;
 	__REG32 MIRCLEAR58		: 1;
 	__REG32 MIRCLEAR59		: 1;
 	__REG32 MIRCLEAR60		: 1;
 	__REG32 MIRCLEAR61		: 1;
 	__REG32 MIRCLEAR62		: 1;
 	__REG32 MIRCLEAR63		: 1;
} __intcps_mir_clear1_bits;

/* INTCPS_MIR_CLEAR2 */
typedef struct {
 	__REG32 MIRCLEAR64		: 1;
 	__REG32 MIRCLEAR65		: 1;
 	__REG32 MIRCLEAR66		: 1;
 	__REG32 MIRCLEAR67		: 1;
 	__REG32 MIRCLEAR68		: 1;
 	__REG32 MIRCLEAR69		: 1;
 	__REG32 MIRCLEAR70		: 1;
 	__REG32 MIRCLEAR71		: 1;
 	__REG32 MIRCLEAR72		: 1;
 	__REG32 MIRCLEAR73		: 1;
 	__REG32 MIRCLEAR74		: 1;
 	__REG32 MIRCLEAR75		: 1;
 	__REG32 MIRCLEAR76		: 1;
 	__REG32 MIRCLEAR77		: 1;
 	__REG32 MIRCLEAR78		: 1;
 	__REG32 MIRCLEAR79		: 1;
 	__REG32 MIRCLEAR80		: 1;
 	__REG32 MIRCLEAR81		: 1;
 	__REG32 MIRCLEAR82		: 1;
 	__REG32 MIRCLEAR83		: 1;
 	__REG32 MIRCLEAR84		: 1;
 	__REG32 MIRCLEAR85		: 1;
 	__REG32 MIRCLEAR86		: 1;
 	__REG32 MIRCLEAR87		: 1;
 	__REG32 MIRCLEAR88		: 1;
 	__REG32 MIRCLEAR89		: 1;
 	__REG32 MIRCLEAR90		: 1;
 	__REG32 MIRCLEAR91		: 1;
 	__REG32 MIRCLEAR92		: 1;
 	__REG32 MIRCLEAR93		: 1;
 	__REG32 MIRCLEAR94		: 1;
 	__REG32 MIRCLEAR95		: 1;
} __intcps_mir_clear2_bits;

/* INTCPS_MIR_SET0 */
typedef struct {
 	__REG32 MIRSET0		: 1;
 	__REG32 MIRSET1		: 1;
 	__REG32 MIRSET2		: 1;
 	__REG32 MIRSET3		: 1;
 	__REG32 MIRSET4		: 1;
 	__REG32 MIRSET5		: 1;
 	__REG32 MIRSET6		: 1;
 	__REG32 MIRSET7		: 1;
 	__REG32 MIRSET8		: 1;
 	__REG32 MIRSET9		: 1;
 	__REG32 MIRSET10		: 1;
 	__REG32 MIRSET11		: 1;
 	__REG32 MIRSET12		: 1;
 	__REG32 MIRSET13		: 1;
 	__REG32 MIRSET14		: 1;
 	__REG32 MIRSET15		: 1;
 	__REG32 MIRSET16		: 1;
 	__REG32 MIRSET17		: 1;
 	__REG32 MIRSET18		: 1;
 	__REG32 MIRSET19		: 1;
 	__REG32 MIRSET20		: 1;
 	__REG32 MIRSET21		: 1;
 	__REG32 MIRSET22		: 1;
 	__REG32 MIRSET23		: 1;
 	__REG32 MIRSET24		: 1;
 	__REG32 MIRSET25		: 1;
 	__REG32 MIRSET26		: 1;
 	__REG32 MIRSET27		: 1;
 	__REG32 MIRSET28		: 1;
 	__REG32 MIRSET29		: 1;
 	__REG32 MIRSET30		: 1;
 	__REG32 MIRSET31		: 1;
} __intcps_mir_set0_bits;

/* INTCPS_MIR_SET1 */
typedef struct {
 	__REG32 MIRSET32		: 1;
 	__REG32 MIRSET33		: 1;
 	__REG32 MIRSET34		: 1;
 	__REG32 MIRSET35		: 1;
 	__REG32 MIRSET36		: 1;
 	__REG32 MIRSET37		: 1;
 	__REG32 MIRSET38		: 1;
 	__REG32 MIRSET39		: 1;
 	__REG32 MIRSET40		: 1;
 	__REG32 MIRSET41		: 1;
 	__REG32 MIRSET42		: 1;
 	__REG32 MIRSET43		: 1;
 	__REG32 MIRSET44		: 1;
 	__REG32 MIRSET45		: 1;
 	__REG32 MIRSET46		: 1;
 	__REG32 MIRSET47		: 1;
 	__REG32 MIRSET48		: 1;
 	__REG32 MIRSET49		: 1;
 	__REG32 MIRSET50		: 1;
 	__REG32 MIRSET51		: 1;
 	__REG32 MIRSET52		: 1;
 	__REG32 MIRSET53		: 1;
 	__REG32 MIRSET54		: 1;
 	__REG32 MIRSET55		: 1;
 	__REG32 MIRSET56		: 1;
 	__REG32 MIRSET57		: 1;
 	__REG32 MIRSET58		: 1;
 	__REG32 MIRSET59		: 1;
 	__REG32 MIRSET60		: 1;
 	__REG32 MIRSET61		: 1;
 	__REG32 MIRSET62		: 1;
 	__REG32 MIRSET63		: 1;
} __intcps_mir_set1_bits;

/* INTCPS_MIR_SET2 */
typedef struct {
 	__REG32 MIRSET64		: 1;
 	__REG32 MIRSET65		: 1;
 	__REG32 MIRSET66		: 1;
 	__REG32 MIRSET67		: 1;
 	__REG32 MIRSET68		: 1;
 	__REG32 MIRSET69		: 1;
 	__REG32 MIRSET70		: 1;
 	__REG32 MIRSET71		: 1;
 	__REG32 MIRSET72		: 1;
 	__REG32 MIRSET73		: 1;
 	__REG32 MIRSET74		: 1;
 	__REG32 MIRSET75		: 1;
 	__REG32 MIRSET76		: 1;
 	__REG32 MIRSET77		: 1;
 	__REG32 MIRSET78		: 1;
 	__REG32 MIRSET79		: 1;
 	__REG32 MIRSET80		: 1;
 	__REG32 MIRSET81		: 1;
 	__REG32 MIRSET82		: 1;
 	__REG32 MIRSET83		: 1;
 	__REG32 MIRSET84		: 1;
 	__REG32 MIRSET85		: 1;
 	__REG32 MIRSET86		: 1;
 	__REG32 MIRSET87		: 1;
 	__REG32 MIRSET88		: 1;
 	__REG32 MIRSET89		: 1;
 	__REG32 MIRSET90		: 1;
 	__REG32 MIRSET91		: 1;
 	__REG32 MIRSET92		: 1;
 	__REG32 MIRSET93		: 1;
 	__REG32 MIRSET94		: 1;
 	__REG32 MIRSET95		: 1;
} __intcps_mir_set2_bits;

/* INTCPS_ISR_SET0 */
typedef struct {
 	__REG32 ISRSET0		: 1;
 	__REG32 ISRSET1		: 1;
 	__REG32 ISRSET2		: 1;
 	__REG32 ISRSET3		: 1;
 	__REG32 ISRSET4		: 1;
 	__REG32 ISRSET5		: 1;
 	__REG32 ISRSET6		: 1;
 	__REG32 ISRSET7		: 1;
 	__REG32 ISRSET8		: 1;
 	__REG32 ISRSET9		: 1;
 	__REG32 ISRSET10		: 1;
 	__REG32 ISRSET11		: 1;
 	__REG32 ISRSET12		: 1;
 	__REG32 ISRSET13		: 1;
 	__REG32 ISRSET14		: 1;
 	__REG32 ISRSET15		: 1;
 	__REG32 ISRSET16		: 1;
 	__REG32 ISRSET17		: 1;
 	__REG32 ISRSET18		: 1;
 	__REG32 ISRSET19		: 1;
 	__REG32 ISRSET20		: 1;
 	__REG32 ISRSET21		: 1;
 	__REG32 ISRSET22		: 1;
 	__REG32 ISRSET23		: 1;
 	__REG32 ISRSET24		: 1;
 	__REG32 ISRSET25		: 1;
 	__REG32 ISRSET26		: 1;
 	__REG32 ISRSET27		: 1;
 	__REG32 ISRSET28		: 1;
 	__REG32 ISRSET29		: 1;
 	__REG32 ISRSET30		: 1;
 	__REG32 ISRSET31		: 1;
} __intcps_isr_set0_bits;

/* INTCPS_ISR_SET1 */
typedef struct {
 	__REG32 ISRSET32		: 1;
 	__REG32 ISRSET33		: 1;
 	__REG32 ISRSET34		: 1;
 	__REG32 ISRSET35		: 1;
 	__REG32 ISRSET36		: 1;
 	__REG32 ISRSET37		: 1;
 	__REG32 ISRSET38		: 1;
 	__REG32 ISRSET39		: 1;
 	__REG32 ISRSET40		: 1;
 	__REG32 ISRSET41		: 1;
 	__REG32 ISRSET42		: 1;
 	__REG32 ISRSET43		: 1;
 	__REG32 ISRSET44		: 1;
 	__REG32 ISRSET45		: 1;
 	__REG32 ISRSET46		: 1;
 	__REG32 ISRSET47		: 1;
 	__REG32 ISRSET48		: 1;
 	__REG32 ISRSET49		: 1;
 	__REG32 ISRSET50		: 1;
 	__REG32 ISRSET51		: 1;
 	__REG32 ISRSET52		: 1;
 	__REG32 ISRSET53		: 1;
 	__REG32 ISRSET54		: 1;
 	__REG32 ISRSET55		: 1;
 	__REG32 ISRSET56		: 1;
 	__REG32 ISRSET57		: 1;
 	__REG32 ISRSET58		: 1;
 	__REG32 ISRSET59		: 1;
 	__REG32 ISRSET60		: 1;
 	__REG32 ISRSET61		: 1;
 	__REG32 ISRSET62		: 1;
 	__REG32 ISRSET63		: 1;
} __intcps_isr_set1_bits;

/* INTCPS_ISR_SET2 */
typedef struct {
 	__REG32 ISRSET64		: 1;
 	__REG32 ISRSET65		: 1;
 	__REG32 ISRSET66		: 1;
 	__REG32 ISRSET67		: 1;
 	__REG32 ISRSET68		: 1;
 	__REG32 ISRSET69		: 1;
 	__REG32 ISRSET70		: 1;
 	__REG32 ISRSET71		: 1;
 	__REG32 ISRSET72		: 1;
 	__REG32 ISRSET73		: 1;
 	__REG32 ISRSET74		: 1;
 	__REG32 ISRSET75		: 1;
 	__REG32 ISRSET76		: 1;
 	__REG32 ISRSET77		: 1;
 	__REG32 ISRSET78		: 1;
 	__REG32 ISRSET79		: 1;
 	__REG32 ISRSET80		: 1;
 	__REG32 ISRSET81		: 1;
 	__REG32 ISRSET82		: 1;
 	__REG32 ISRSET83		: 1;
 	__REG32 ISRSET84		: 1;
 	__REG32 ISRSET85		: 1;
 	__REG32 ISRSET86		: 1;
 	__REG32 ISRSET87		: 1;
 	__REG32 ISRSET88		: 1;
 	__REG32 ISRSET89		: 1;
 	__REG32 ISRSET90		: 1;
 	__REG32 ISRSET91		: 1;
 	__REG32 ISRSET92		: 1;
 	__REG32 ISRSET93		: 1;
 	__REG32 ISRSET94		: 1;
 	__REG32 ISRSET95		: 1;
} __intcps_isr_set2_bits;

/* INTCPS_ISR_CLEAR0 */
typedef struct {
 	__REG32 ISRCLEART0		: 1;
 	__REG32 ISRCLEART1		: 1;
 	__REG32 ISRCLEART2		: 1;
 	__REG32 ISRCLEART3		: 1;
 	__REG32 ISRCLEART4		: 1;
 	__REG32 ISRCLEART5		: 1;
 	__REG32 ISRCLEART6		: 1;
 	__REG32 ISRCLEART7		: 1;
 	__REG32 ISRCLEART8		: 1;
 	__REG32 ISRCLEART9		: 1;
 	__REG32 ISRCLEART10		: 1;
 	__REG32 ISRCLEART11		: 1;
 	__REG32 ISRCLEART12		: 1;
 	__REG32 ISRCLEART13		: 1;
 	__REG32 ISRCLEART14		: 1;
 	__REG32 ISRCLEART15		: 1;
 	__REG32 ISRCLEART16		: 1;
 	__REG32 ISRCLEART17		: 1;
 	__REG32 ISRCLEART18		: 1;
 	__REG32 ISRCLEART19		: 1;
 	__REG32 ISRCLEART20		: 1;
 	__REG32 ISRCLEART21		: 1;
 	__REG32 ISRCLEART22		: 1;
 	__REG32 ISRCLEART23		: 1;
 	__REG32 ISRCLEART24		: 1;
 	__REG32 ISRCLEART25		: 1;
 	__REG32 ISRCLEART26		: 1;
 	__REG32 ISRCLEART27		: 1;
 	__REG32 ISRCLEART28		: 1;
 	__REG32 ISRCLEART29		: 1;
 	__REG32 ISRCLEART30		: 1;
 	__REG32 ISRCLEART31		: 1;
} __intcps_isr_clear0_bits;

/* INTCPS_ISR_CLEAR1 */
typedef struct {
 	__REG32 ISRCLEAR32		: 1;
 	__REG32 ISRCLEAR33		: 1;
 	__REG32 ISRCLEAR34		: 1;
 	__REG32 ISRCLEAR35		: 1;
 	__REG32 ISRCLEAR36		: 1;
 	__REG32 ISRCLEAR37		: 1;
 	__REG32 ISRCLEAR38		: 1;
 	__REG32 ISRCLEAR39		: 1;
 	__REG32 ISRCLEAR40		: 1;
 	__REG32 ISRCLEAR41		: 1;
 	__REG32 ISRCLEAR42		: 1;
 	__REG32 ISRCLEAR43		: 1;
 	__REG32 ISRCLEAR44		: 1;
 	__REG32 ISRCLEAR45		: 1;
 	__REG32 ISRCLEAR46		: 1;
 	__REG32 ISRCLEAR47		: 1;
 	__REG32 ISRCLEAR48		: 1;
 	__REG32 ISRCLEAR49		: 1;
 	__REG32 ISRCLEAR50		: 1;
 	__REG32 ISRCLEAR51		: 1;
 	__REG32 ISRCLEAR52		: 1;
 	__REG32 ISRCLEAR53		: 1;
 	__REG32 ISRCLEAR54		: 1;
 	__REG32 ISRCLEAR55		: 1;
 	__REG32 ISRCLEAR56		: 1;
 	__REG32 ISRCLEAR57		: 1;
 	__REG32 ISRCLEAR58		: 1;
 	__REG32 ISRCLEAR59		: 1;
 	__REG32 ISRCLEAR60		: 1;
 	__REG32 ISRCLEAR61		: 1;
 	__REG32 ISRCLEAR62		: 1;
 	__REG32 ISRCLEAR63		: 1;
} __intcps_isr_clear1_bits;

/* INTCPS_ISR_CLEAR2 */
typedef struct {
 	__REG32 ISRCLEAR64		: 1;
 	__REG32 ISRCLEAR65		: 1;
 	__REG32 ISRCLEAR66		: 1;
 	__REG32 ISRCLEAR67		: 1;
 	__REG32 ISRCLEAR68		: 1;
 	__REG32 ISRCLEAR69		: 1;
 	__REG32 ISRCLEAR70		: 1;
 	__REG32 ISRCLEAR71		: 1;
 	__REG32 ISRCLEAR72		: 1;
 	__REG32 ISRCLEAR73		: 1;
 	__REG32 ISRCLEAR74		: 1;
 	__REG32 ISRCLEAR75		: 1;
 	__REG32 ISRCLEAR76		: 1;
 	__REG32 ISRCLEAR77		: 1;
 	__REG32 ISRCLEAR78		: 1;
 	__REG32 ISRCLEAR79		: 1;
 	__REG32 ISRCLEAR80		: 1;
 	__REG32 ISRCLEAR81		: 1;
 	__REG32 ISRCLEAR82		: 1;
 	__REG32 ISRCLEAR83		: 1;
 	__REG32 ISRCLEAR84		: 1;
 	__REG32 ISRCLEAR85		: 1;
 	__REG32 ISRCLEAR86		: 1;
 	__REG32 ISRCLEAR87		: 1;
 	__REG32 ISRCLEAR88		: 1;
 	__REG32 ISRCLEAR89		: 1;
 	__REG32 ISRCLEAR90		: 1;
 	__REG32 ISRCLEAR91		: 1;
 	__REG32 ISRCLEAR92		: 1;
 	__REG32 ISRCLEAR93		: 1;
 	__REG32 ISRCLEAR94		: 1;
 	__REG32 ISRCLEAR95		: 1;
} __intcps_isr_clear2_bits;

/* INTCPS_PENDING_IRQ0 */
typedef struct {
 	__REG32 PENDINGIRQ0		: 1;
 	__REG32 PENDINGIRQ1		: 1;
 	__REG32 PENDINGIRQ2		: 1;
 	__REG32 PENDINGIRQ3		: 1;
 	__REG32 PENDINGIRQ4		: 1;
 	__REG32 PENDINGIRQ5		: 1;
 	__REG32 PENDINGIRQ6		: 1;
 	__REG32 PENDINGIRQ7		: 1;
 	__REG32 PENDINGIRQ8		: 1;
 	__REG32 PENDINGIRQ9		: 1;
 	__REG32 PENDINGIRQ10		: 1;
 	__REG32 PENDINGIRQ11		: 1;
 	__REG32 PENDINGIRQ12		: 1;
 	__REG32 PENDINGIRQ13		: 1;
 	__REG32 PENDINGIRQ14		: 1;
 	__REG32 PENDINGIRQ15		: 1;
 	__REG32 PENDINGIRQ16		: 1;
 	__REG32 PENDINGIRQ17		: 1;
 	__REG32 PENDINGIRQ18		: 1;
 	__REG32 PENDINGIRQ19		: 1;
 	__REG32 PENDINGIRQ20		: 1;
 	__REG32 PENDINGIRQ21		: 1;
 	__REG32 PENDINGIRQ22		: 1;
 	__REG32 PENDINGIRQ23		: 1;
 	__REG32 PENDINGIRQ24		: 1;
 	__REG32 PENDINGIRQ25		: 1;
 	__REG32 PENDINGIRQ26		: 1;
 	__REG32 PENDINGIRQ27		: 1;
 	__REG32 PENDINGIRQ28		: 1;
 	__REG32 PENDINGIRQ29		: 1;
 	__REG32 PENDINGIRQ30		: 1;
 	__REG32 PENDINGIRQ31		: 1;
} __intcps_pending_irq0_bits;

/* INTCPS_PENDING_IRQ1 */
typedef struct {
 	__REG32 PENDINGIRQ32		: 1;
 	__REG32 PENDINGIRQ33		: 1;
 	__REG32 PENDINGIRQ34		: 1;
 	__REG32 PENDINGIRQ35		: 1;
 	__REG32 PENDINGIRQ36		: 1;
 	__REG32 PENDINGIRQ37		: 1;
 	__REG32 PENDINGIRQ38		: 1;
 	__REG32 PENDINGIRQ39		: 1;
 	__REG32 PENDINGIRQ40		: 1;
 	__REG32 PENDINGIRQ41		: 1;
 	__REG32 PENDINGIRQ42		: 1;
 	__REG32 PENDINGIRQ43		: 1;
 	__REG32 PENDINGIRQ44		: 1;
 	__REG32 PENDINGIRQ45		: 1;
 	__REG32 PENDINGIRQ46		: 1;
 	__REG32 PENDINGIRQ47		: 1;
 	__REG32 PENDINGIRQ48		: 1;
 	__REG32 PENDINGIRQ49		: 1;
 	__REG32 PENDINGIRQ50		: 1;
 	__REG32 PENDINGIRQ51		: 1;
 	__REG32 PENDINGIRQ52		: 1;
 	__REG32 PENDINGIRQ53		: 1;
 	__REG32 PENDINGIRQ54		: 1;
 	__REG32 PENDINGIRQ55		: 1;
 	__REG32 PENDINGIRQ56		: 1;
 	__REG32 PENDINGIRQ57		: 1;
 	__REG32 PENDINGIRQ58		: 1;
 	__REG32 PENDINGIRQ59		: 1;
 	__REG32 PENDINGIRQ60		: 1;
 	__REG32 PENDINGIRQ61		: 1;
 	__REG32 PENDINGIRQ62		: 1;
 	__REG32 PENDINGIRQ63		: 1;
} __intcps_pending_irq1_bits;

/* INTCPS_PENDING_IRQ2 */
typedef struct {
 	__REG32 PENDINGIRQ64		: 1;
 	__REG32 PENDINGIRQ65		: 1;
 	__REG32 PENDINGIRQ66		: 1;
 	__REG32 PENDINGIRQ67		: 1;
 	__REG32 PENDINGIRQ68		: 1;
 	__REG32 PENDINGIRQ69		: 1;
 	__REG32 PENDINGIRQ70		: 1;
 	__REG32 PENDINGIRQ71		: 1;
 	__REG32 PENDINGIRQ72		: 1;
 	__REG32 PENDINGIRQ73		: 1;
 	__REG32 PENDINGIRQ74		: 1;
 	__REG32 PENDINGIRQ75		: 1;
 	__REG32 PENDINGIRQ76		: 1;
 	__REG32 PENDINGIRQ77		: 1;
 	__REG32 PENDINGIRQ78		: 1;
 	__REG32 PENDINGIRQ79		: 1;
 	__REG32 PENDINGIRQ80		: 1;
 	__REG32 PENDINGIRQ81		: 1;
 	__REG32 PENDINGIRQ82		: 1;
 	__REG32 PENDINGIRQ83		: 1;
 	__REG32 PENDINGIRQ84		: 1;
 	__REG32 PENDINGIRQ85		: 1;
 	__REG32 PENDINGIRQ86		: 1;
 	__REG32 PENDINGIRQ87		: 1;
 	__REG32 PENDINGIRQ88		: 1;
 	__REG32 PENDINGIRQ89		: 1;
 	__REG32 PENDINGIRQ90		: 1;
 	__REG32 PENDINGIRQ91		: 1;
 	__REG32 PENDINGIRQ92		: 1;
 	__REG32 PENDINGIRQ93		: 1;
 	__REG32 PENDINGIRQ94		: 1;
 	__REG32 PENDINGIRQ95		: 1;
} __intcps_pending_irq2_bits;

/* INTCPS_PENDING_FIQ0 */
typedef struct {
 	__REG32 PENDINGFIQ0		: 1;
 	__REG32 PENDINGFIQ1		: 1;
 	__REG32 PENDINGFIQ2		: 1;
 	__REG32 PENDINGFIQ3		: 1;
 	__REG32 PENDINGFIQ4		: 1;
 	__REG32 PENDINGFIQ5		: 1;
 	__REG32 PENDINGFIQ6		: 1;
 	__REG32 PENDINGFIQ7		: 1;
 	__REG32 PENDINGFIQ8		: 1;
 	__REG32 PENDINGFIQ9		: 1;
 	__REG32 PENDINGFIQ10		: 1;
 	__REG32 PENDINGFIQ11		: 1;
 	__REG32 PENDINGFIQ12		: 1;
 	__REG32 PENDINGFIQ13		: 1;
 	__REG32 PENDINGFIQ14		: 1;
 	__REG32 PENDINGFIQ15		: 1;
 	__REG32 PENDINGFIQ16		: 1;
 	__REG32 PENDINGFIQ17		: 1;
 	__REG32 PENDINGFIQ18		: 1;
 	__REG32 PENDINGFIQ19		: 1;
 	__REG32 PENDINGFIQ20		: 1;
 	__REG32 PENDINGFIQ21		: 1;
 	__REG32 PENDINGFIQ22		: 1;
 	__REG32 PENDINGFIQ23		: 1;
 	__REG32 PENDINGFIQ24		: 1;
 	__REG32 PENDINGFIQ25		: 1;
 	__REG32 PENDINGFIQ26		: 1;
 	__REG32 PENDINGFIQ27		: 1;
 	__REG32 PENDINGFIQ28		: 1;
 	__REG32 PENDINGFIQ29		: 1;
 	__REG32 PENDINGFIQ30		: 1;
 	__REG32 PENDINGFIQ31		: 1;
} __intcps_pending_fiq0_bits;

/* INTCPS_PENDING_FIQ1 */
typedef struct {
 	__REG32 PENDINGFIQ32		: 1;
 	__REG32 PENDINGFIQ33		: 1;
 	__REG32 PENDINGFIQ34		: 1;
 	__REG32 PENDINGFIQ35		: 1;
 	__REG32 PENDINGFIQ36		: 1;
 	__REG32 PENDINGFIQ37		: 1;
 	__REG32 PENDINGFIQ38		: 1;
 	__REG32 PENDINGFIQ39		: 1;
 	__REG32 PENDINGFIQ40		: 1;
 	__REG32 PENDINGFIQ41		: 1;
 	__REG32 PENDINGFIQ42		: 1;
 	__REG32 PENDINGFIQ43		: 1;
 	__REG32 PENDINGFIQ44		: 1;
 	__REG32 PENDINGFIQ45		: 1;
 	__REG32 PENDINGFIQ46		: 1;
 	__REG32 PENDINGFIQ47		: 1;
 	__REG32 PENDINGFIQ48		: 1;
 	__REG32 PENDINGFIQ49		: 1;
 	__REG32 PENDINGFIQ50		: 1;
 	__REG32 PENDINGFIQ51		: 1;
 	__REG32 PENDINGFIQ52		: 1;
 	__REG32 PENDINGFIQ53		: 1;
 	__REG32 PENDINGFIQ54		: 1;
 	__REG32 PENDINGFIQ55		: 1;
 	__REG32 PENDINGFIQ56		: 1;
 	__REG32 PENDINGFIQ57		: 1;
 	__REG32 PENDINGFIQ58		: 1;
 	__REG32 PENDINGFIQ59		: 1;
 	__REG32 PENDINGFIQ60		: 1;
 	__REG32 PENDINGFIQ61		: 1;
 	__REG32 PENDINGFIQ62		: 1;
 	__REG32 PENDINGFIQ63		: 1;
} __intcps_pending_fiq1_bits;

/* INTCPS_PENDING_FIQ2 */
typedef struct {
 	__REG32 PENDINGFIQ64		: 1;
 	__REG32 PENDINGFIQ65		: 1;
 	__REG32 PENDINGFIQ66		: 1;
 	__REG32 PENDINGFIQ67		: 1;
 	__REG32 PENDINGFIQ68		: 1;
 	__REG32 PENDINGFIQ69		: 1;
 	__REG32 PENDINGFIQ70		: 1;
 	__REG32 PENDINGFIQ71		: 1;
 	__REG32 PENDINGFIQ72		: 1;
 	__REG32 PENDINGFIQ73		: 1;
 	__REG32 PENDINGFIQ74		: 1;
 	__REG32 PENDINGFIQ75		: 1;
 	__REG32 PENDINGFIQ76		: 1;
 	__REG32 PENDINGFIQ77		: 1;
 	__REG32 PENDINGFIQ78		: 1;
 	__REG32 PENDINGFIQ79		: 1;
 	__REG32 PENDINGFIQ80		: 1;
 	__REG32 PENDINGFIQ81		: 1;
 	__REG32 PENDINGFIQ82		: 1;
 	__REG32 PENDINGFIQ83		: 1;
 	__REG32 PENDINGFIQ84		: 1;
 	__REG32 PENDINGFIQ85		: 1;
 	__REG32 PENDINGFIQ86		: 1;
 	__REG32 PENDINGFIQ87		: 1;
 	__REG32 PENDINGFIQ88		: 1;
 	__REG32 PENDINGFIQ89		: 1;
 	__REG32 PENDINGFIQ90		: 1;
 	__REG32 PENDINGFIQ91		: 1;
 	__REG32 PENDINGFIQ92		: 1;
 	__REG32 PENDINGFIQ93		: 1;
 	__REG32 PENDINGFIQ94		: 1;
 	__REG32 PENDINGFIQ95		: 1;
} __intcps_pending_fiq2_bits;

/* INTCPS_ILRm */
typedef struct {
 	__REG32 FIQNIRQ					: 1;
 	__REG32 								: 1;
 	__REG32 PRIORITY				: 6;
 	__REG32 								:24;
} __intcps_ilr_bits;

/* INTC_INIT_REGISTER1 */
typedef struct {
 	__REG32 INIT1						: 1;
 	__REG32 								:31;
} __intc_init_register1_bits;

/* INTC_INIT_REGISTER2 */
typedef struct {
 	__REG32 								: 1;
 	__REG32 INIT2						: 1;
 	__REG32 								:30;
} __intc_init_register2_bits;

/* GPMC_SYSCONFIG */
typedef struct {
 	__REG32 AUTOIDLE				: 1;
 	__REG32 SOFTRESET				: 1;
 	__REG32 								: 1;
 	__REG32 IDLEMODE				: 2;
 	__REG32 								:27;
} __gpmc_sysconfig_bits;

/* GPMC_SYSSTATUS */
typedef struct {
 	__REG32 RESETDONE				: 1;
 	__REG32 								:31;
} __gpmc_sysstatus_bits;

/* GPMC_IRQSTATUS */
typedef struct {
 	__REG32 FIFOEVENTSTATUS						: 1;
 	__REG32 TERMINALCOUNTSTATUS				: 1;
 	__REG32 													: 6;
 	__REG32 WAIT0EDGEDETECTIONSTATUS	: 1;
 	__REG32 WAIT1EDGEDETECTIONSTATUS	: 1;
 	__REG32 WAIT2EDGEDETECTIONSTATUS	: 1;
 	__REG32 WAIT3EDGEDETECTIONSTATUS	: 1;
 	__REG32 													:20;
} __gpmc_irqstatus_bits;

/* GPMC_IRQENABLE */
typedef struct {
 	__REG32 FIFOEVENTENABLE						: 1;
 	__REG32 TERMINALCOUNTEVENTENABLE	: 1;
 	__REG32 													: 6;
 	__REG32 WAIT0EDGEDETECTIONENABLE	: 1;
 	__REG32 WAIT1EDGEDETECTIONENABLE	: 1;
 	__REG32 WAIT2EDGEDETECTIONENABLE	: 1;
 	__REG32 WAIT3EDGEDETECTIONENABLE	: 1;
 	__REG32 													:20;
} __gpmc_irqenable_bits;

/* GPMC_TIMEOUT_CONTROL */
typedef struct {
 	__REG32 TIMEOUTENABLE							: 1;
 	__REG32 													: 3;
 	__REG32 TIMEOUTSTARTVALUE					: 9;
 	__REG32 													:19;
} __gpmc_timeout_control_bits;

/* GPMC_ERR_ADDRESS */
typedef struct {
 	__REG32 ILLEGALADD								:31;
 	__REG32 													: 1;
} __gpmc_err_address_bits;

/* GPMC_ERR_TYPE */
typedef struct {
 	__REG32 ERRORVALID								: 1;
 	__REG32 													: 1;
 	__REG32 ERRORTIMEOUT							: 1;
 	__REG32 ERRORNOTSUPPMCMD					: 1;
 	__REG32 ERRORNOTSUPPADD						: 1;
 	__REG32 													: 3;
 	__REG32 ILLEGALMCMD								: 3;
 	__REG32 													:21;
} __gpmc_err_type_bits;

/* GPMC_CONFIG */
typedef struct {
 	__REG32 NANDFORCEPOSTEDWRITE			: 1;
 	__REG32 LIMITEDADDRESS						: 1;
 	__REG32 													: 2;
 	__REG32 WRITEPROTECT							: 1;
 	__REG32 													: 3;
 	__REG32 WAIT0PINPOLARITY					: 1;
 	__REG32 WAIT1PINPOLARITY					: 1;
 	__REG32 WAIT2PINPOLARITY					: 1;
 	__REG32 WAIT3PINPOLARITY					: 1;
 	__REG32 													:20;
} __gpmc_config_bits;

/* GPMC_STATUS */
typedef struct {
 	__REG32 EMPTYWRITEBUFFERSTATUS		: 1;
 	__REG32 													: 7;
 	__REG32 WAIT0STATUS								: 1;
 	__REG32 WAIT1STATUS								: 1;
 	__REG32 WAIT2STATUS								: 1;
 	__REG32 WAIT3STATUS								: 1;
 	__REG32 													:20;
} __gpmc_status_bits;

/* GPMC_CONFIG1_i */
typedef struct {
 	__REG32 GPMCFCLKDIVIDER						: 2;
 	__REG32 													: 2;
 	__REG32 TIMEPARAGRANULARITY				: 1;
 	__REG32 													: 4;
 	__REG32 MUXADDDATA								: 1;
 	__REG32 DEVICETYPE								: 2;
 	__REG32 DEVICESIZE								: 2;
 	__REG32 													: 2;
 	__REG32 WAITPINSELECT							: 2;
 	__REG32 WAITMONITORINGTIME				: 2;
 	__REG32 													: 1;
 	__REG32 WAITWRITEMONITORING				: 1;
 	__REG32 WAITREADMONITORING				: 1;
 	__REG32 ATTACHEDDEVICEPAGELENGTH	: 2;
 	__REG32 CLKACTIVATIONTIME					: 2;
 	__REG32 WRITETYPE									: 1;
 	__REG32 WRITEMULTIPLE							: 1;
 	__REG32 READTYPE									: 1;
 	__REG32 READMULTIPLE							: 1;
 	__REG32 WRAPBURST									: 1;
} __gpmc_config1_bits;

/* GPMC_CONFIG2_i */
typedef struct {
 	__REG32 CSONTIME									: 4;
 	__REG32 													: 3;
 	__REG32 CSEXTRADELAY							: 1;
 	__REG32 CSRDOFFTIME								: 5;
 	__REG32 													: 3;
 	__REG32 CSWROFFTIME								: 5;
 	__REG32 													:11;
} __gpmc_config2_bits;

/* GPMC_CONFIG3_i */
typedef struct {
 	__REG32 ADVONTIME									: 4;
 	__REG32 													: 3;
 	__REG32 ADVEXTRADELAY							: 1;
 	__REG32 ADVRDOFFTIME							: 5;
 	__REG32 													: 3;
 	__REG32 ADVWROFFTIME							: 5;
 	__REG32 													:11;
} __gpmc_config3_bits;

/* GPMC_CONFIG4_i */
typedef struct {
 	__REG32 OEONTIME									: 4;
 	__REG32 													: 3;
 	__REG32 OEEXTRADELAY							: 1;
 	__REG32 OEOFFTIME									: 5;
 	__REG32 													: 3;
 	__REG32 WEONTIME									: 4;
 	__REG32 													: 3;
 	__REG32 WEEXTRADELAY							: 1;
 	__REG32 WEOFFTIME									: 5;
 	__REG32 													: 3;
} __gpmc_config4_bits;

/* GPMC_CONFIG5_i */
typedef struct {
 	__REG32 RDCYCLETIME								: 5;
 	__REG32 													: 3;
 	__REG32 WRCYCLETIME								: 5;
 	__REG32 													: 3;
 	__REG32 RDACCESSTIME							: 5;
 	__REG32 													: 3;
 	__REG32 PAGEBURSTACCESSTIME				: 4;
 	__REG32 													: 4;
} __gpmc_config5_bits;

/* GPMC_CONFIG6_i */
typedef struct {
 	__REG32 BUSTURNAROUND							: 4;
 	__REG32 													: 2;
 	__REG32 CYCLE2CYCLEDIFFCSEN				: 1;
 	__REG32 CYCLE2CYCLESAMECSEN				: 1;
 	__REG32 CYCLE2CYCLEDELAY					: 4;
 	__REG32 													: 4;
 	__REG32 WRDATAONADMUXBUS					: 4;
 	__REG32 													: 4;
 	__REG32 WRACCESSTIME							: 5;
 	__REG32 													: 3;
} __gpmc_config6_bits;

/* GPMC_CONFIG7_i */
typedef struct {
 	__REG32 BASEADDRESS								: 6;
 	__REG32 CSVALID										: 1;
 	__REG32 													: 1;
 	__REG32 MASKADDRESS								: 4;
 	__REG32 													:20;
} __gpmc_config7_bits;

/* GPMC_PREFETCH_CONFIG1 */
typedef struct {
 	__REG32 ACCESSMODE								: 1;
 	__REG32 													: 1;
 	__REG32 DMAMODE										: 1;
 	__REG32 SYNCHROMODE								: 1;
 	__REG32 WAITPINSELECTOR						: 2;
 	__REG32 													: 1;
 	__REG32 ENABLEENGINE							: 1;
 	__REG32 FIFOTHRESHOLD							: 7;
 	__REG32 													: 1;
 	__REG32 PFPWWEIGHTEDPRIO					: 4;
 	__REG32 													: 3;
 	__REG32 PFPWENROUNDROBIN					: 1;
 	__REG32 ENGINECSSELECTOR					: 3;
 	__REG32 ENABLEOPTIMIZEDACCESS			: 1;
 	__REG32 CYCLEOPTIMIZATION					: 3;
 	__REG32 													: 1;
} __gpmc_prefetch_config1_bits;

/* GPMC_PREFETCH_CONFIG2 */
typedef struct {
 	__REG32 TRANSFERCOUNT							:14;
 	__REG32 													:18;
} __gpmc_prefetch_config2_bits;

/* GPMC_PREFETCH_CONTROL */
typedef struct {
 	__REG32 STARTENGINE								: 1;
 	__REG32 													:31;
} __gpmc_prefetch_control_bits;

/* GPMC_PREFETCH_STATUS */
typedef struct {
 	__REG32 COUNTVALUE								:14;
 	__REG32 													: 2;
 	__REG32 FIFOTHRESHOLDSTATUS				: 1;
 	__REG32 													: 7;
 	__REG32 FIFOPOINTER								: 7;
 	__REG32 													: 1;
} __gpmc_prefetch_status_bits;

/* GPMC_ECC_CONFIG */
typedef struct {
 	__REG32 ECCENABLE									: 1;
 	__REG32 ECCCS											: 3;
 	__REG32 ECCTOPSECTOR							: 3;
 	__REG32 ECC16B										: 1;
 	__REG32 ECCWRAPMODE								: 4;
 	__REG32 ECCBCHT8									: 1;
 	__REG32 													: 3;
 	__REG32 ECCALGORITHM							: 1;
 	__REG32 													:15;
} __gpmc_ecc_config_bits;

/* GPMC_ECC_CONTROL */
typedef struct {
 	__REG32 ECCPOINTER								: 4;
 	__REG32 													: 4;
 	__REG32 ECCCLEAR									: 1;
 	__REG32 													:23;
} __gpmc_ecc_control_bits;

/* GPMC_ECC_SIZE_CONFIG */
typedef struct {
 	__REG32 ECC1RESULTSIZE						: 1;
 	__REG32 ECC2RESULTSIZE						: 1;
 	__REG32 ECC3RESULTSIZE						: 1;
 	__REG32 ECC4RESULTSIZE						: 1;
 	__REG32 ECC5RESULTSIZE						: 1;
 	__REG32 ECC6RESULTSIZE						: 1;
 	__REG32 ECC7RESULTSIZE						: 1;
 	__REG32 ECC8RESULTSIZE						: 1;
 	__REG32 ECC9RESULTSIZE						: 1;
 	__REG32 													: 3;
 	__REG32 ECCSIZE0									: 8;
 	__REG32 													: 2;
 	__REG32 ECCSIZE1									: 8;
 	__REG32 													: 2;
} __gpmc_ecc_size_config_bits;

/* GPMC_ECCj_RESULT */
typedef struct {
 	__REG32 P1e												: 1;
 	__REG32 P2e												: 1;
 	__REG32 P4e												: 1;
 	__REG32 P8e												: 1;
 	__REG32 P16e											: 1;
 	__REG32 P32e											: 1;
 	__REG32 P64e											: 1;
 	__REG32 P128e											: 1;
 	__REG32 P256e											: 1;
 	__REG32 P512e											: 1;
 	__REG32 P1024e										: 1;
 	__REG32 P2048e										: 1;
 	__REG32 													: 4;
 	__REG32 P1o												: 1;
 	__REG32 P2o												: 1;
 	__REG32 P4o												: 1;
 	__REG32 P8o												: 1;
 	__REG32 P16o											: 1;
 	__REG32 P32o											: 1;
 	__REG32 P64o											: 1;
 	__REG32 P128o											: 1;
 	__REG32 P256o											: 1;
 	__REG32 P512o											: 1;
 	__REG32 P1024o										: 1;
 	__REG32 P2048o										: 1;
 	__REG32 													: 4;
} __gpmc_ecc_result_bits;

/* GPMC_BCH_RESULT3_i */
typedef struct {
 	__REG32 BCH_RESULT_3							: 8;
 	__REG32 													:24;
} __gpmc_bch_result3_bits;

/* GPMC_BCH_SWDATA */
typedef struct {
 	__REG32 BCH_DATA									:16;
 	__REG32 													:16;
} __gpmc_bch_swdata_bits;

/* EMIF_MOD_ID_REV */
typedef struct {
 	__REG32 REG_MINOR_REVISION				: 6;
 	__REG32 													: 2;
 	__REG32 REG_MAJOR_REVISION				: 4;
 	__REG32 REG_RTL_VERSION						: 4;
 	__REG32 REG_MODULE_ID							:12;
 	__REG32 													: 2;
 	__REG32 REG_SCHEME								: 2;
} __emif_mod_id_rev_bits;

/* EMIF_STATUS */
typedef struct {
 	__REG32 													: 2;
 	__REG32 REG_PHY_DLL_READY					: 1;
 	__REG32 													:26;
 	__REG32 REG_FAST_INIT							: 1;
 	__REG32 REG_DUAL_CLK_MODE					: 1;
 	__REG32 REG_BE										: 1;
} __emif_status_bits;

/* EMIF_SDRAM_CONFIG */
typedef struct {
 	__REG32 REG_PAGESIZE							: 3;
 	__REG32 REG_EBANK									: 1;
 	__REG32 REG_IBANK									: 3;
 	__REG32 REG_ROWSIZE								: 3;
 	__REG32 REG_CL										: 4;
 	__REG32 REG_NARROW_MODE						: 2;
 	__REG32 													: 2;
 	__REG32 REG_SDRAM_DRIVE						: 2;
 	__REG32 REG_DDR_DISABLE_DLL				: 1;
 	__REG32 													: 2;
 	__REG32 REG_DDR2_DDQS							: 1;
 	__REG32 REG_DDR_TERM							: 3;
 	__REG32 REG_IBANK_POS							: 2;
 	__REG32 REG_SDRAM_TYPE						: 3;
} __emif_sdram_config_bits;

/* EMIF_SDRAM_REF_CTRL */
typedef struct {
 	__REG32 REG_REFRESH_RATE					:16;
 	__REG32 													: 8;
 	__REG32 REG_PASR									: 3;
 	__REG32 													: 4;
 	__REG32 REG_INITREF_DIS						: 1;
} __emif_sdram_ref_ctrl_bits;

/* EMIF_SDRAM_REF_CTRL_SHDW */
typedef struct {
 	__REG32 REG_REFRESH_RATE_SHDW			:16;
 	__REG32 													:16;
} __emif_sdram_ref_ctrl_shdw_bits;

/* EMIF_SDRAM_TIM_1 */
typedef struct {
 	__REG32 REG_T_WTR									: 3;
 	__REG32 REG_T_RRD									: 3;
 	__REG32 REG_T_RC									: 6;
 	__REG32 REG_T_RAS									: 5;
 	__REG32 REG_T_WR									: 4;
 	__REG32 REG_T_RCD									: 4;
 	__REG32 REG_T_RP									: 4;
 	__REG32 													: 3;
} __emif_sdram_tim_1_bits;

/* EMIF_SDRAM_TIM_1_SHDW */
typedef struct {
 	__REG32 REG_T_WTR_SHDW						: 3;
 	__REG32 REG_T_RRD_SHDW						: 3;
 	__REG32 REG_T_RC_SHDW							: 6;
 	__REG32 REG_T_RAS_SHDW						: 5;
 	__REG32 REG_T_WR_SHDW							: 4;
 	__REG32 REG_T_RCD_SHDW						: 4;
 	__REG32 REG_T_RP_SHDW							: 4;
 	__REG32 													: 3;
} __emif_sdram_tim_1_shdw_bits;

/* EMIF_SDRAM_TIM_2 */
typedef struct {
 	__REG32 REG_T_CKE									: 3;
 	__REG32 REG_T_RTP									: 3;
 	__REG32 REG_T_XSRD								:10;
 	__REG32 REG_T_XSNR								: 9;
 	__REG32 REG_T_ODT									: 3;
 	__REG32 REG_T_XP									: 3;
 	__REG32 													: 1;
} __emif_sdram_tim_2_bits;

/* EMIF_SDRAM_TIM_2_SHDW */
typedef struct {
 	__REG32 REG_T_CKE_SHDW						: 3;
 	__REG32 REG_T_RTP_SHDW						: 3;
 	__REG32 REG_T_XSRD_SHDW						:10;
 	__REG32 REG_T_XSNR_SHDW						: 9;
 	__REG32 REG_T_ODT_SHDW						: 3;
 	__REG32 REG_T_XP_SHDW							: 3;
 	__REG32 													: 1;
} __emif_sdram_tim_2_shdw_bits;

/* EMIF_SDRAM_TIM_3 */
typedef struct {
 	__REG32 REG_T_RAS_MAX							: 4;
 	__REG32 REG_T_RFC									: 9;
 	__REG32 													:19;
} __emif_sdram_tim_3_bits;

/* EMIF_SDRAM_TIM_3_SHDW */
typedef struct {
 	__REG32 REG_T_RAS_MAX_SHDW				: 4;
 	__REG32 REG_T_RFC_SHDW						: 9;
 	__REG32 													:19;
} __emif_sdram_tim_3_shdw_bits;

/* EMIF_PWR_MGMT_CTRL */
typedef struct {
 	__REG32 REG_PM_TIM								: 4;
 	__REG32 													: 4;
 	__REG32 REG_LP_MODE								: 2;
 	__REG32 REG_DPD_EN								: 1;
 	__REG32 													:19;
 	__REG32 REG_IDLEMODE							: 2;
} __emif_pwr_mgmt_ctrl_bits;

/* EMIF_PWR_MGMT_CTRL_SHDW */
typedef struct {
 	__REG32 REG_PM_TIM_SHDW						: 4;
 	__REG32 													:28;
} __emif_pwr_mgmt_ctrl_shdw_bits;

/* EMIF_OCP_CONFIG */
typedef struct {
 	__REG32 REG_PR_OLD_COUNT					: 8;
 	__REG32 													:24;
} __emif_ocp_config_bits;

/* EMIF_OCP_CFG_VAL_1 */
typedef struct {
 	__REG32 REG_CMD_FIFO_DEPTH				: 8;
 	__REG32 REG_WR_FIFO_DEPTH					: 8;
 	__REG32 													:14;
 	__REG32 REG_SYS_BUS_WIDTH					: 2;
} __emif_ocp_cfg_val_1_bits;

/* EMIF_OCP_CFG_VAL_2 */
typedef struct {
 	__REG32 REG_RCMD_FIFO_DEPTH				: 8;
 	__REG32 REG_RSD_FIFO_DEPTH				: 8;
 	__REG32 REG_RREG_FIFO_DEPTH				: 8;
 	__REG32 													: 8;
} __emif_ocp_cfg_val_2_bits;

/* EMIF_IODFT_TLGC */
typedef struct {
 	__REG32 REG_TM										: 1;
 	__REG32 REG_PC										: 3;
 	__REG32 REG_MC										: 2;
 	__REG32 													: 2;
 	__REG32 REG_MMS										: 1;
 	__REG32 													: 1;
 	__REG32 REG_RESET_PHY							: 1;
 	__REG32 													: 1;
 	__REG32 REG_OPG_LD								: 1;
 	__REG32 REG_ACT_CAP_EN						: 1;
 	__REG32 REG_MT										: 1;
 	__REG32 													: 1;
 	__REG32 REG_TLEC									:16;
} __emif_iodft_tlgc_shdw_bits;

/* EMIF_IODFT_CTRL_MISR_RSLT */
typedef struct {
 	__REG32 REG_CTL_TLMR							:11;
 	__REG32 													: 5;
 	__REG32 REG_DQM_TLMR							:10;
 	__REG32 													: 6;
} __emif_iodft_ctrl_misr_rslt_bits;

/* EMIF_IODFT_ADDR_MISR_RSLT */
typedef struct {
 	__REG32 REG_ADDR_TLMR							:21;
 	__REG32 													:11;
} __emif_iodft_addr_misr_rslt_bits;

/* EMIF_IODFT_DATA_MISR_RSLT_3 */
typedef struct {
 	__REG32 REG_DATA_TLMR_66_64				: 3;
 	__REG32 													:29;
} __emif_iodft_data_misr_rslt_3_bits;

/* EMIF_PERF_CNT_CFG */
typedef struct {
 	__REG32 REG_CNTR1_CFG							: 4;
 	__REG32 													:10;
 	__REG32 REG_CNTR1_REGION_EN				: 1;
 	__REG32 REG_CNTR1_MCONNID_EN			: 1;
 	__REG32 REG_CNTR2_CFG							: 4;
 	__REG32 													:10;
 	__REG32 REG_CNTR2_REGION_EN				: 1;
 	__REG32 REG_CNTR2_MCONNID_EN			: 1;
} __emif_perf_cnt_cfg_bits;

/* EMIF_PERF_CNT_SEL */
typedef struct {
 	__REG32 REG_REGION_SEL1						: 2;
 	__REG32 													: 6;
 	__REG32 REG_MCONNID1							: 8;
 	__REG32 REG_REGION_SEL2						: 2;
 	__REG32 													: 6;
 	__REG32 REG_MCONNID2							: 8;
} __emif_perf_cnt_sel_bits;

/* EMIF_IRQ_EOI */
typedef struct {
 	__REG32 REG_EOI										: 1;
 	__REG32 													:31;
} __emif_irq_eoi_bits;

/* EMIF_IRQSTATUS_RAW_SYS */
typedef struct {
 	__REG32 REG_RAW_SYS								: 1;
 	__REG32 													:31;
} __emif_irqstatus_raw_sys_bits;

/* EMIF_IRQSTATUS_SYS */
typedef struct {
 	__REG32 REG_ENABLED_SYS						: 1;
 	__REG32 													:31;
} __emif_irqstatus_sys_bits;

/* EMIF_IRQENABLE_SET_SYS */
typedef struct {
 	__REG32 REG_EN_SYS								: 1;
 	__REG32 													:31;
} __emif_irqenable_set_sys_bits;

/* EMIF_IRQENABLE_CLR_SYS */
typedef struct {
 	__REG32 REG_DIS_SYS								: 1;
 	__REG32 													:31;
} __emif_irqenable_clr_sys_bits;

/* EMIF_OCP_ERR_LOG */
typedef struct {
 	__REG32 REG_MCONNID								:11;
 	__REG32 REG_MBURSTSEQ							: 3;
 	__REG32 REG_MADDRSPACE						: 2;
 	__REG32 													:16;
} __emif_ocp_err_log_bits;

/* EMIF_DDR_PHY_CTRL_1 */
typedef struct {
 	__REG32 READ_LATENCY							: 3;
 	__REG32 													: 3;
 	__REG32 CONFIG_PWRDN_DISABLE			: 1;
 	__REG32 CONFIG_EXT_STRBEN					: 1;
 	__REG32 DDR_16B_MODE_PWRSAVE			: 1;
 	__REG32 													: 3;
 	__REG32 CONFIG_DLL_MODE						: 3;
 	__REG32 CONFIG_VTP_DYNAMIC_UPDATE	: 1;
 	__REG32 													: 7;
 	__REG32 TESTIN_LB_CK_SELECT				: 1;
 	__REG32 													: 8;
} __emif_ddr_phy_ctrl_1_bits;

/* EMIF_DDR_PHY_CTRL_1_SHDW */
typedef struct {
 	__REG32 REG_READ_LATENCY_SHDW			: 3;
 	__REG32 REG_DDR_PHY_CTRL_1_SHDW		:29;
} __emif_ddr_phy_ctrl_1_shdw_bits;

/* EMIF_DDR_PHY_CTRL_2 */
typedef struct {
 	__REG32 CONFIG_TX_STRB_DATA_ALIGN	: 1;
 	__REG32 CONFIG_RX_DLL_BYPASS			: 1;
 	__REG32                  					:30;
} __emif_ddr_phy_ctrl_2_bits;

/* SMS_SYSCONFIG */
typedef struct {
 	__REG32 AUTOIDLE									: 1;
 	__REG32 SOFTRESET									: 1;
 	__REG32                  					: 1;
 	__REG32 SIDLEMODE									: 2;
 	__REG32                  					:27;
} __sms_sysconfig_bits;

/* SMS_SYSSTATUS */
typedef struct {
 	__REG32 RESETDONE									: 1;
 	__REG32                  					:31;
} __sms_sysstatus_bits;

/* SMS_RG_RDPERMi */
/* SMS_RG_WRPERMi */
typedef struct {
 	__REG32 CONNIDVECTOR							:16;
 	__REG32                  					:16;
} __sms_rg_rdperm_bits;

/* SMS_SECURITY_CONTROL */
typedef struct {
 	__REG32 SECURITYCONTROLREGLOCK		: 1;
 	__REG32 FIREWALLLOCK							: 1;
 	__REG32 ERRORREGSLOCK							: 1;
 	__REG32 SOFTRESETLOCK							: 1;
 	__REG32 REGION1REGSLOCK						: 1;
 	__REG32 ARBITRATIONREGSLOCK				: 1;
 	__REG32                  					:10;
 	__REG32 ROTCTXT0LOCK							: 1;
 	__REG32 ROTCTXT1LOCK							: 1;
 	__REG32 ROTCTXT2LOCK							: 1;
 	__REG32 ROTCTXT3LOCK							: 1;
 	__REG32 ROTCTXT4LOCK							: 1;
 	__REG32 ROTCTXT5LOCK							: 1;
 	__REG32 ROTCTXT6LOCK							: 1;
 	__REG32 ROTCTXT7LOCK							: 1;
 	__REG32 ROTCTXT8LOCK							: 1;
 	__REG32 ROTCTXT9LOCK							: 1;
 	__REG32 ROTCTXT10LOCK							: 1;
 	__REG32 ROTCTXT11LOCK 						: 1;
 	__REG32                  					: 4;
} __sms_security_control_bits;

/* SMS_CLASS_ARBITER0 */
typedef struct {
 	__REG32                  					: 6;
 	__REG32 HIGHPRIOVECTOR						: 2;
 	__REG32                  					:12;
 	__REG32 EXTENDEDGRANT							: 4;
 	__REG32                  					: 6;
 	__REG32 BURST_COMPLETE						: 2;
} __sms_class_arbiter0_bits;

/* SMS_CLASS_ARBITER1 */
typedef struct {
 	__REG32 HIGHPRIOVECTOR						: 2;
 	__REG32                  					: 6;
 	__REG32 EXTENDEDGRANT							: 4;
 	__REG32                  					:12;
 	__REG32 BURST_COMPLETE						: 2;
 	__REG32                  					: 6;
} __sms_class_arbiter1_bits;

/* SMS_CLASS_ARBITER2 */
typedef struct {
 	__REG32                  					: 2;
 	__REG32 HIGHPRIOVECTOR						: 4;
 	__REG32                  					: 6;
 	__REG32 EXTENDEDGRANT							: 8;
 	__REG32                  					: 6;
 	__REG32 BURST_COMPLETE						: 4;
 	__REG32                  					: 2;
} __sms_class_arbiter2_bits;

/* SMS_INTERCLASS_ARBITER */
typedef struct {
 	__REG32 CLASS1PRIO								: 8;
 	__REG32                  					: 8;
 	__REG32 CLASS2PRIO								: 8;
 	__REG32                  					: 8;
} __sms_interclass_arbiter_bits;

/* SMS_CLASS_ROTATIONm */
typedef struct {
 	__REG32 NOFSERVICES								: 5;
 	__REG32                  					:27;
} __sms_class_rotation_bits;

/* SMS_ERR_TYPE */
typedef struct {
 	__REG32 ERRORVALID								: 1;
 	__REG32 ERRORSECURITY							: 1;
 	__REG32 ERRORSECREG								: 1;
 	__REG32 ERRORSECOVERLAP						: 1;
 	__REG32                  					: 4;
 	__REG32 ILLEGALCMD								: 1;
 	__REG32 UNEXPECTEDREQ							: 1;
 	__REG32 UNEXPECTEDADD							: 1;
 	__REG32                  					: 5;
 	__REG32 ERRORCONNID								: 4;
 	__REG32 ERRORMCMD									: 3;
 	__REG32                  					: 1;
 	__REG32 ERRORREGIONID							: 3;
 	__REG32                  					: 5;
} __sms_err_type_bits;

/* SMS_POW_CTRL */
typedef struct {
 	__REG32 IDLEDELAY									: 8;
 	__REG32                  					:24;
} __sms_pow_ctrl_bits;

/* SMS_ROT_CONTROLn */
typedef struct {
 	__REG32 PS												: 2;
 	__REG32                  					: 2;
 	__REG32 PW												: 3;
 	__REG32                  					: 1;
 	__REG32 PH												: 3;
 	__REG32                  					:21;
} __sms_rot_control_bits;

/* SMS_ROT_SIZEn */
typedef struct {
 	__REG32 IMAGEWIDTH								:11;
 	__REG32                  					: 5;
 	__REG32 IMAGEHEIGHT								:11;
 	__REG32                  					: 5;
} __sms_rot_size_bits;

/* SMS_ROT_PHYSICAL_BAn */
typedef struct {
 	__REG32 PHYSICALBA								:31;
 	__REG32                  					: 1;
} __sms_rot_physical_ba_bits;

/* VPFE_PCR */
typedef struct {
 	__REG32 ENABLE										: 1;
 	__REG32 BUSY											: 1;
 	__REG32                  					:30;
} __vpfe_pcr_bits;

/* VPFE_SYN_MODE */
typedef struct {
 	__REG32                  					: 2;
 	__REG32 VDPOL											: 1;
 	__REG32 HDPOL											: 1;
 	__REG32 FLDPOL										: 1;
 	__REG32 EXWEN											: 1;
 	__REG32 DATAPOL										: 1;
 	__REG32 FLDMODE										: 1;
 	__REG32 DATSIZ										: 3;
 	__REG32 PACK8											: 1;
 	__REG32 INPMOD										: 2;
 	__REG32 LPF												: 1;
 	__REG32 FLDSTAT										: 1;
 	__REG32 VDHDEN										: 1;
 	__REG32 WEN												: 1;
 	__REG32                  					:14;
} __vpfe_syn_mode_bits;

/* VPFE_HORZ_INFO */
typedef struct {
 	__REG32 NPH												:15;
 	__REG32                  					: 1;
 	__REG32 SPH												:15;
 	__REG32                  					: 1;
} __vpfe_horz_info_bits;

/* VPFE_VERT_START */
typedef struct {
 	__REG32 SLV1											:15;
 	__REG32                  					: 1;
 	__REG32 SLV0											:15;
 	__REG32                  					: 1;
} __vpfe_vert_start_bits;

/* VPFE_VERT_LINES */
typedef struct {
 	__REG32 NLV												:15;
 	__REG32                  					:17;
} __vpfe_vert_lines_bits;

/* VPFE_CULLING */
typedef struct {
 	__REG32 CULV											: 8;
 	__REG32                  					: 8;
 	__REG32 CULHODD										: 8;
 	__REG32 CULHEVN										: 8;
} __vpfe_culling_bits;

/* VPFE_HSIZE_OFF */
typedef struct {
 	__REG32 LNOFST										:16;
 	__REG32                  					:16;
} __vpfe_hsize_off_bits;

/* VPFE_SDOFST */
typedef struct {
 	__REG32 LOFTS3										: 3;
 	__REG32 LOFTS2										: 3;
 	__REG32 LOFTS1										: 3;
 	__REG32 LOFTS0										: 3;
 	__REG32 FOFST											: 2;
 	__REG32 FIINV											: 1;
 	__REG32                  					:17;
} __vpfe_sdofst_bits;

/* VPFE_CLAMP */
typedef struct {
 	__REG32 OBGAIN										: 5;
 	__REG32 													: 5;
 	__REG32 OBST											:15;
 	__REG32 OBSLN											: 3;
 	__REG32 OBSLEN										: 3;
 	__REG32 CLAMPEN										: 1;
} __vpfe_clamp_bits;

/* VPFE_DCSUB */
typedef struct {
 	__REG32 DCSUB											:14;
 	__REG32 													:18;
} __vpfe_dcsub_bits;

/* VPFE_COLPTN */
typedef struct {
 	__REG32 CP0LPC0										: 2;
 	__REG32 CP0LPC1										: 2;
 	__REG32 CP0LPC2										: 2;
 	__REG32 CP0LPC3										: 2;
 	__REG32 CP1LPC0										: 2;
 	__REG32 CP1LPC1										: 2;
 	__REG32 CP1LPC2										: 2;
 	__REG32 CP1LPC3										: 2;
 	__REG32 CP2LPC0										: 2;
 	__REG32 CP2LPC1										: 2;
 	__REG32 CP2LPC2										: 2;
 	__REG32 CP2LPC3										: 2;
 	__REG32 CP3LPC0										: 2;
 	__REG32 CP3LPC1										: 2;
 	__REG32 CP3LPC2										: 2;
 	__REG32 CP3LPC3										: 2;
} __vpfe_colptn_bits;

/* VPFE_BLKCMP */
typedef struct {
 	__REG32 B_MG											: 8;
 	__REG32 GB_G											: 8;
 	__REG32 GR_CY											: 8;
 	__REG32 R_YE											: 8;
} __vpfe_blkcmp_bits;

/* VPFE_VDINT */
typedef struct {
 	__REG32 VDINT1										:15;
 	__REG32 													: 1;
 	__REG32 VDINT0										:15;
 	__REG32       										: 1;
} __vpfe_vdint_bits;

/* VPFE_ALAW */
typedef struct {
 	__REG32 GWDI											: 3;
 	__REG32 CCDTBL										: 1;
 	__REG32       										:28;
} __vpfe_alaw_bits;

/* VPFE_REC656IF */
typedef struct {
 	__REG32 R656ON										: 1;
 	__REG32 ECCFVH										: 1;
 	__REG32       										:30;
} __vpfe_rec656if_bits;

/* VPFE_CCDCCFG */
typedef struct {
 	__REG32       										: 4;
 	__REG32 YCINSWP										: 1;
 	__REG32 BW656 										: 1;
 	__REG32 FIDMD 										: 2;
 	__REG32 WENLOG 										: 1;
 	__REG32       										: 2;
 	__REG32 Y8POS 										: 1;
 	__REG32 BSWD	 										: 1;
 	__REG32 MSBINVI										: 1;
 	__REG32 			 										: 1;
 	__REG32 VDLC	 										: 1;
 	__REG32       										:16;
} __vpfe_ccdccfg_bits;

/* VPFE_DMA_CNTL */
typedef struct {
 	__REG32 CPRIORITY									: 3;
 	__REG32       										:28;
 	__REG32 OVERFLOW									: 1;
} __vpfe_dma_cntl_bits;

/* DSS_REVISIONNUMBER */
typedef struct {
 	__REG32 REV												: 8;
 	__REG32       										:24;
} __dss_revisionnumber_bits;

/* DSS_SYSCONFIG */
typedef struct {
 	__REG32 AUTOIDLE									: 1;
 	__REG32 SOFTRESET									: 1;
 	__REG32       										:30;
} __dss_sysconfig_bits;

/* DSS_SYSSTATUS */
typedef struct {
 	__REG32 RESETDONE									: 1;
 	__REG32       										:31;
} __dss_sysstatus_bits;

/* DSS_IRQSTATUS */
typedef struct {
 	__REG32 DISPC_IRQ									: 1;
 	__REG32 DSI_IRQ										: 1;
 	__REG32       										:30;
} __dss_irqstatus_bits;

/* DSS_CONTROL */
typedef struct {
 	__REG32 DISPC_CLK_SWITCH					: 1;
 	__REG32 DSI_CLK_SWITCH						: 1;
 	__REG32 VENC_CLOCK_MODE						: 1;
 	__REG32 VENC_CLOCK_4X_ENABLE			: 1;
 	__REG32 DAC_DEMEN									: 1;
 	__REG32 DAC_POWERDN_BGZ						: 1;
 	__REG32 VENC_OUT_SEL							: 1;
 	__REG32       										:25;
} __dss_control_bits;

/* DSS_SDI_CONTROL */
typedef struct {
 	__REG32 SDI_BWSEL									: 2;
 	__REG32 SDI_PRSEL									: 2;
 	__REG32 													: 7;
 	__REG32 SDI_AUTOSTDBY							: 1;
 	__REG32 SDI_RBITS									: 2;
 	__REG32 SDI_PHYLPMODE							: 1;
 	__REG32 SDI_PDIV									: 5;
 	__REG32       										:12;
} __dss_sdi_control_bits;

/* DSS_PLL_CONTROL */
typedef struct {
 	__REG32 SDI_PLL_IDLE							: 1;
 	__REG32 SDI_PLL_REGM							:10;
 	__REG32 SDI_PLL_REGN							: 6;
 	__REG32 SDI_PLL_STOPMODE					: 1;
 	__REG32 SDI_PLL_SYSRESET					: 1;
 	__REG32 SDI_PLL_HIGHFREQ					: 1;
 	__REG32 SDI_PLL_LOWCURRSTBY				: 1;
 	__REG32 SDI_PLL_PLLLPMODE					: 1;
 	__REG32 SDI_PLL_FREQSEL						: 4;
 	__REG32 SDI_PLL_LOCKSEL						: 2;
 	__REG32 SDI_PLL_GOBIT							: 1;
 	__REG32       										: 3;
} __dss_pll_control_bits;

/* DSS_SDI_STATUS */
typedef struct {
 	__REG32 DSS_DISPC_CLK1_STATUS			: 1;
 	__REG32 DSI_PLL_CLK1_STATUS				: 1;
 	__REG32 SDI_RESET_DONE						: 1;
 	__REG32 SDI_ERROR									: 1;
 	__REG32 SDI_PLL_RECAL							: 1;
 	__REG32 SDI_PLL_LOCK							: 1;
 	__REG32 SDI_PLL_BUSYFLAG					: 1;
 	__REG32 DSS_DSI_CLK1_STATUS				: 1;
 	__REG32 DSI_PLL_CLK2_STATUS				: 1;
 	__REG32       										:23;
} __dss_sdi_status_bits;

/* DISPC_REVISION */
typedef struct {
 	__REG32 REV												: 8;
 	__REG32       										:24;
} __dispc_revision_bits;

/* DISPC_SYSCONFIG */
typedef struct {
 	__REG32 AUTOIDLE									: 1;
 	__REG32 SOFTRESET									: 1;
 	__REG32 ENWAKEUP									: 1;
 	__REG32 SIDLEMODE									: 2;
 	__REG32       										: 3;
 	__REG32 CLOCKACTIVITY							: 2;
 	__REG32       										: 2;
 	__REG32 MIDLEMODE									: 2;
 	__REG32       										:18;
} __dispc_sysconfig_bits;

/* DISPC_SYSSTATUS */
typedef struct {
 	__REG32 RESETDONE									: 1;
 	__REG32       										:31;
} __dispc_sysstatus_bits;

/* DISPC_IRQSTATUS */
typedef struct {
 	__REG32 FRAMEDONE									: 1;
 	__REG32 VSYNC											: 1;
 	__REG32 EVSYNC_EVEN								: 1;
 	__REG32 EVSYNC_ODD								: 1;
 	__REG32 ACBIASCOUNTSTATUS					: 1;
 	__REG32 PROGRAMMEDLINENUMBER			: 1;
 	__REG32 GFXFIFOUNDERFLOW					: 1;
 	__REG32 GFXENDWINDOW							: 1;
 	__REG32 PALETTEGAMMALOADING				: 1;
 	__REG32 OCPERROR									: 1;
 	__REG32 VID1FIFOUNDERFLOW					: 1;
 	__REG32 VID1ENDWINDOW							: 1;
 	__REG32 VID2FIFOUNDERFLOW					: 1;
 	__REG32 VID2ENDWINDOW							: 1;
 	__REG32 SYNCLOST									: 1;
 	__REG32 SYNCLOSTDIGITAL						: 1;
 	__REG32 WAKEUP										: 1;
 	__REG32       										:15;
} __dispc_irqstatus_bits;

/* DISPC_CONTROL */
typedef struct {
 	__REG32 LCDENABLE												: 1;
 	__REG32 DIGITALENABLE										: 1;
 	__REG32 MONOCOLOR												: 1;
 	__REG32 STNTFT													: 1;
 	__REG32 M8B															: 1;
 	__REG32 GOLCD														: 1;
 	__REG32 GODIGITAL												: 1;
 	__REG32 STDITHERENABLE									: 1;
 	__REG32 TFTDATALINES										: 2;
 	__REG32 																: 1;
 	__REG32 STALLMODE												: 1;
 	__REG32 OVERLAYOPTIMIZATION							: 1;
 	__REG32 GPIN0														: 1;
 	__REG32 GPIN1														: 1;
 	__REG32 GPOUT0													: 1;
 	__REG32 GPOUT1													: 1;
 	__REG32 HT															: 3;
 	__REG32 TDMENABLE												: 1;
 	__REG32 TDMPARALLELMODE									: 2;
 	__REG32 TDMCYCLEFORMAT									: 2;
 	__REG32 TDMUNUSEDBITS										: 2;
 	__REG32 PCKFREEENABLE										: 1;
 	__REG32 LCDENABLESIGNAL									: 1;
 	__REG32 LCDENABLEPOL										: 1;
 	__REG32 SPATIALTEMPORALDITHERINGFRAMES	: 2;
} __dispc_control_bits;

/* DISPC_CONFIG */
typedef struct {
 	__REG32 PIXELGATED								: 1;
 	__REG32 LOADMODE									: 2;
 	__REG32 PALETTEGAMMATABLE					: 1;
 	__REG32 PIXELDATAGATED						: 1;
 	__REG32 PIXELCLOCKGATED						: 1;
 	__REG32 HSYNCGATED								: 1;
 	__REG32 VSYNCGATED								: 1;
 	__REG32 ACBIASGATED								: 1;
 	__REG32 FUNCGATED									: 1;
 	__REG32 TCKLCDENABLE							: 1;
 	__REG32 TCKLCDSELECTION						: 1;
 	__REG32 TCKDIGENABLE							: 1;
 	__REG32 TCKDIGSELECTION						: 1;
 	__REG32 FIFOMERGE									: 1;
 	__REG32 CPR												: 1;
 	__REG32 FIFOHANDCHECK							: 1;
 	__REG32 FIFOFILLING								: 1;
 	__REG32 LCDALPHABLENDERENABLE			: 1;
 	__REG32 TVALPHABLENDERENABLE			: 1;
 	__REG32       										:12;
} __dispc_config_bits;

/* DISPC_DEFAULT_COLOR_m */
typedef struct {
 	__REG32 DEFAULTCOLOR							:24;
 	__REG32       										: 8;
} __dispc_default_color_bits;

/* DISPC_TRANS_COLOR_m */
typedef struct {
 	__REG32 TRANSCOLORKEY							:24;
 	__REG32       										: 8;
} __dispc_trans_color_bits;

/* DISPC_LINE_STATUS */
typedef struct {
 	__REG32 LINENUMBER								:11;
 	__REG32       										:21;
} __dispc_line_status_bits;

/* DISPC_TIMING_H */
typedef struct {
 	__REG32 HSW												: 8;
 	__REG32 HFP												:12;
 	__REG32 HBP												:12;
} __dispc_timing_h_bits;

/* DISPC_TIMING_V */
typedef struct {
 	__REG32 VSW												: 8;
 	__REG32 VFP												:12;
 	__REG32 VBP												:12;
} __dispc_timing_v_bits;

/* DISPC_POL_FREQ */
typedef struct {
 	__REG32 ACB												: 8;
 	__REG32 ACBI											: 4;
 	__REG32 IVS												: 1;
 	__REG32 IHS												: 1;
 	__REG32 IPC												: 1;
 	__REG32 IEO												: 1;
 	__REG32 RF 												: 1;
 	__REG32 ONOFF											: 1;
 	__REG32 													:14;
} __dispc_pol_freq_bits;

/* DISPC_DIVISOR */
typedef struct {
 	__REG32 PCD												: 8;
 	__REG32 													: 8;
 	__REG32 LCD												: 8;
 	__REG32 													: 8;
} __dispc_divisor_bits;

/* DISPC_GLOBAL_ALPHA */
typedef struct {
 	__REG32 GFXGLOBALALPHA						: 8;
 	__REG32 													: 8;
 	__REG32 VID2GLOBALALPHA						: 8;
 	__REG32 													: 8;
} __dispc_global_alpha_bits;

/* DISPC_SIZE_DIG */
typedef struct {
 	__REG32 PPL												:11;
 	__REG32 													: 5;
 	__REG32 LPP												:11;
 	__REG32 													: 5;
} __dispc_size_dig_bits;

/* DISPC_GFX_POSITION */
typedef struct {
 	__REG32 GFXPOSX										:11;
 	__REG32 													: 5;
 	__REG32 GFXPOSY										:11;
 	__REG32 													: 5;
} __dispc_gfx_position_bits;

/* DISPC_GFX_SIZE */
typedef struct {
 	__REG32 GFXSIZEX									:11;
 	__REG32 													: 5;
 	__REG32 GFXSIZEY									:11;
 	__REG32 													: 5;
} __dispc_gfx_size_bits;

/* DISPC_GFX_ATTRIBUTES */
typedef struct {
 	__REG32 GFXENABLE									: 1;
 	__REG32 GFXFORMAT									: 4;
 	__REG32 GFXREPLICATIONENABLE			: 1;
 	__REG32 GFXBURSTSIZE							: 2;
 	__REG32 GFXCHANNELOUT							: 1;
 	__REG32 GFXNIBBLEMODE							: 1;
 	__REG32 GFXENDIANNESS							: 1;
 	__REG32 GFXFIFOPRELOAD						: 1;
 	__REG32 GFXROTATION								: 2;
 	__REG32 GFXARBITRATION						: 1;
 	__REG32 GFXSELFREFRESH						: 1;
 	__REG32 													:16;
} __dispc_gfx_attributes_bits;

/* DISPC_GFX_FIFO_THRESHOLD */
typedef struct {
 	__REG32 GFXFIFOLOWTHRESHOLD				:12;
 	__REG32 													: 4;
 	__REG32 GFXFIFOHIGHTHRESHOLD			:12;
 	__REG32 													: 4;
} __dispc_gfx_fifo_threshold_bits;

/* DISPC_GFX_FIFO_SIZE_STATUS */
typedef struct {
 	__REG32 GFXFIFOSIZE								:11;
 	__REG32 													:21;
} __dispc_gfx_fifo_size_status_bits;

/* DISPC_GFX_PIXEL_INC */
typedef struct {
 	__REG32 GFXPIXELINC								:16;
 	__REG32 													:16;
} __dispc_gfx_pixel_inc_bits;

/* DISPC_DATA_CYCLEk */
typedef struct {
 	__REG32 NBBITSPIXEL1							: 5;
 	__REG32 													: 3;
 	__REG32 BITALIGNMENTPIXEL1				: 4;
 	__REG32 													: 4;
 	__REG32 NBBITSPIXEL2							: 5;
 	__REG32 													: 3;
 	__REG32 BITALIGNMENTPIXEL2				: 4;
 	__REG32 													: 4;
} __dispc_data_cycle_bits;

/* DISPC_CPR_COEF_R */
typedef struct {
 	__REG32 RB												:10;
 	__REG32 													: 1;
 	__REG32 RG												:10;
 	__REG32 													: 1;
 	__REG32 RR												:10;
} __dispc_cpr_coef_r_bits;

/* DISPC_CPR_COEF_G */
typedef struct {
 	__REG32 GB												:10;
 	__REG32 													: 1;
 	__REG32 GG												:10;
 	__REG32 													: 1;
 	__REG32 GR												:10;
} __dispc_cpr_coef_g_bits;

/* DISPC_CPR_COEF_B */
typedef struct {
 	__REG32 BB												:10;
 	__REG32 													: 1;
 	__REG32 BG												:10;
 	__REG32 													: 1;
 	__REG32 BR												:10;
} __dispc_cpr_coef_b_bits;

/* DISPC_GFX_PRELOAD */
typedef struct {
 	__REG32 PRELOAD										:12;
 	__REG32 													:20;
} __dispc_gfx_preload_bits;

/* DISPC_VIDn_POSITION */
typedef struct {
 	__REG32 VIDPOSX										:11;
 	__REG32 													: 5;
 	__REG32 VIDPOSY										:11;
 	__REG32 													: 5;
} __dispc_vidn_position_bits;

/* DISPC_VIDn_SIZE */
typedef struct {
 	__REG32 VIDSIZEX									:11;
 	__REG32 													: 5;
 	__REG32 VIDSIZEY									:11;
 	__REG32 													: 5;
} __dispc_vidn_size_bits;

/* DISPC_VIDn_ATTRIBUTES */
typedef struct {
 	__REG32 VIDENABLE									: 1;
 	__REG32 VIDFORMAT									: 4;
 	__REG32 VIDRESIZEENABLE						: 2;
 	__REG32 VIDHRESIZECONF						: 1;
 	__REG32 VIDVRESIZECONF						: 1;
 	__REG32 VIDCOLORCONVENABLE				: 1;
 	__REG32 VIDREPLICATIONENABLE			: 1;
 	__REG32 VIDFULLRANGE							: 1;
 	__REG32 VIDROTATION								: 2;
 	__REG32 VIDBURSTSIZE							: 2;
 	__REG32 VIDCHANNELOUT							: 1;
 	__REG32 VIDENDIANNESS							: 1;
 	__REG32 VIDROWREPEATENABLE				: 1;
 	__REG32 VIDFIFOPRELOAD						: 1;
 	__REG32 VIDDMAOPTIMIZATION				: 1;
 	__REG32 VIDVERTICALTAPS						: 1;
 	__REG32 VIDLINEBUFFERSPLIT				: 1;
 	__REG32 VIDARBITRATION						: 1;
 	__REG32 VIDSELFREFRESH						: 1;
 	__REG32 													: 7;
} __dispc_vidn_attributes_bits;

/* DISPC_VIDn_ATTRIBUTES */
typedef struct {
 	__REG32 VIDFIFOLOWTHRESHOLD				:12;
 	__REG32 													: 4;
 	__REG32 VIDFIFOHIGHTHRESHOLD			:12;
 	__REG32 													: 4;
} __dispc_vidn_fifo_threshold_bits;

/* DISPC_VIDn_FIFO_SIZE_STATUS */
typedef struct {
 	__REG32 VIDFIFOSIZE								:11;
 	__REG32 													:21;
} __dispc_vidn_fifo_size_status_bits;

/* DISPC_VIDn_PIXEL_INC */
typedef struct {
 	__REG32 VIDPIXELINC								:16;
 	__REG32 													:16;
} __dispc_vidn_pixel_inc_bits;

/* DISPC_VIDn_FIR */
typedef struct {
 	__REG32 VIDFIRHINC								:13;
 	__REG32 													: 3;
 	__REG32 VIDFIRVINC								:13;
 	__REG32 													: 3;
} __dispc_vidn_fir_bits;

/* DISPC_VIDn_PICTURE_SIZE */
typedef struct {
 	__REG32 VIDORGSIZEX								:11;
 	__REG32 													: 5;
 	__REG32 VIDORGSIZEY								:11;
 	__REG32 													: 5;
} __dispc_vidn_picture_size_bits;

/* DISPC_VIDn_ACCUl */
typedef struct {
 	__REG32 VIDHORIZONTALACCU					:10;
 	__REG32 													: 6;
 	__REG32 VIDVERTICALACCU						:10;
 	__REG32 													: 6;
} __dispc_vidn_accu_bits;

/* DISPC_VIDn_FIR_COEF_Hi */
typedef struct {
 	__REG32 VIDFIRHC0									: 8;
 	__REG32 VIDFIRHC1									: 8;
 	__REG32 VIDFIRHC2									: 8;
 	__REG32 VIDFIRHC3									: 8;
} __dispc_vidn_fir_coef_h_bits;

/* DISPC_VIDn_FIR_COEF_HVi */
typedef struct {
 	__REG32 VIDFIRHC4									: 8;
 	__REG32 VIDFIRVC0									: 8;
 	__REG32 VIDFIRVC1									: 8;
 	__REG32 VIDFIRVC2									: 8;
} __dispc_vidn_fir_coef_hv_bits;

/* DISPC_VIDn_CONV_COEF0 */
typedef struct {
 	__REG32 RY												:11;
 	__REG32 													: 5;
 	__REG32 RCR												:11;
 	__REG32 			 										: 5;
} __dispc_vidn_conv_coef0_bits;

/* DISPC_VIDn_CONV_COEF1 */
typedef struct {
 	__REG32 RCB												:11;
 	__REG32 													: 5;
 	__REG32 GY												:11;
 	__REG32 			 										: 5;
} __dispc_vidn_conv_coef1_bits;

/* DISPC_VIDn_CONV_COEF2 */
typedef struct {
 	__REG32 GCR												:11;
 	__REG32 													: 5;
 	__REG32 GCB												:11;
 	__REG32 			 										: 5;
} __dispc_vidn_conv_coef2_bits;

/* DISPC_VIDn_CONV_COEF3 */
typedef struct {
 	__REG32 BY												:11;
 	__REG32 													: 5;
 	__REG32 BCR												:11;
 	__REG32 			 										: 5;
} __dispc_vidn_conv_coef3_bits;

/* DISPC_VIDn_CONV_COEF4 */
typedef struct {
 	__REG32 BCB												:11;
 	__REG32 													:21;
} __dispc_vidn_conv_coef4_bits;

/* DISPC_VIDn_FIR_COEF_Vi */
typedef struct {
 	__REG32 VIDFIRVC00								: 8;
 	__REG32 VIDFIRVC22								: 8;
 	__REG32 													:16;
} __dispc_vidn_fir_coef_v_bits;

/* DISPC_VIDn_PRELOAD */
typedef struct {
 	__REG32 PRELOAD										:12;
 	__REG32 													:20;
} __dispc_vidn_preload_bits;

/* RFBI_REVISION */
typedef struct {
 	__REG32 REV												: 8;
 	__REG32 													:24;
} __rfbi_revision_bits;

/* RFBI_SYSCONFIG */
typedef struct {
 	__REG32 AUTOIDLE									: 1;
 	__REG32 SOFTRESET									: 1;
 	__REG32 													: 1;
 	__REG32 SIDLEMODE									: 2;
 	__REG32 													:27;
} __rfbi_sysconfig_bits;

/* RFBI_SYSSTATUS */
typedef struct {
 	__REG32 RESETDONE									: 1;
 	__REG32 													: 7;
 	__REG32 BUSY											: 1;
 	__REG32 BUSYRFBIDATA							: 1;
 	__REG32 													:22;
} __rfbi_sysstatus_bits;

/* RFBI_CONTROL */
typedef struct {
 	__REG32 ENABLE										: 1;
 	__REG32 BYPASSMODE								: 1;
 	__REG32 CONFIGSELECT							: 2;
 	__REG32 ITE												: 1;
 	__REG32 HIGHTHRESHOLD							: 2;
 	__REG32 DISABLE_DMA_REQ						: 1;
 	__REG32 SMART_DMA_REQ							: 1;
 	__REG32 													:23;
} __rfbi_control_bits;

/* RFBI_LINE_NUMBER */
typedef struct {
 	__REG32 LINENUMBER								:11;
 	__REG32 													:21;
} __rfbi_line_number_bits;

/* RFBI_CMD */
typedef struct {
 	__REG32 CMD												:16;
 	__REG32 													:16;
} __rfbi_cmd_bits;

/* RFBI_PARAM */
typedef struct {
 	__REG32 PARAM											:16;
 	__REG32 													:16;
} __rfbi_param_bits;

/* RFBI_READ */
typedef struct {
 	__REG32 READ											:16;
 	__REG32 													:16;
} __rfbi_read_bits;

/* RFBI_STATUS */
typedef struct {
 	__REG32 STATUS										:16;
 	__REG32 													:16;
} __rfbi_status_bits;

/* RFBI_CONFIGi */
typedef struct {
 	__REG32 PARALLEL_MODE							: 2;
 	__REG32 TRIGGERMODE 							: 2;
 	__REG32 TIMEGRANULARITY						: 1;
 	__REG32 DATA_TYPE 								: 2;
 	__REG32 L4FORMAT		 							: 2;
 	__REG32 CYCLEFORMAT	 							: 2;
 	__REG32 UNUSEDBITS	 							: 2;
 	__REG32 													: 3;
 	__REG32 A0POLARITY	 							: 1;
 	__REG32 REPOLARITY	 							: 1;
 	__REG32 WEPOLARITY	 							: 1;
 	__REG32 CSPOLARITY	 							: 1;
 	__REG32 TE_VSYNC_POLARITY					: 1;
 	__REG32 HSYNCPOLARITY							: 1;
 	__REG32 													:10;
} __rfbi_config_bits;

/* RFBI_ONOFF_TIMEi */
typedef struct {
 	__REG32 CSONTIME									: 4;
 	__REG32 CSOFFTIME	 								: 6;
 	__REG32 WEONTIME									: 4;
 	__REG32 WEOFFTIME 								: 6;
 	__REG32 REONTIME		 							: 4;
 	__REG32 REOFFTIME		 							: 6;
 	__REG32 													: 2;
} __rfbi_onoff_time_bits;

/* RFBI_CYCLE_TIMEi */
typedef struct {
 	__REG32 WECYCLETIME								: 6;
 	__REG32 RECYCLETIME								: 6;
 	__REG32 CSPULSEWIDTH							: 6;
 	__REG32 RWENABLE	 								: 1;
 	__REG32 RRENABLE	 								: 1;
 	__REG32 WWENABLE	 								: 1;
 	__REG32 WRENABLE	 								: 1;
 	__REG32 ACCESSTIME	 							: 6;
 	__REG32 													: 4;
} __rfbi_cycle_time_bits;

/* RFBI_DATA_CYCLEx_i */
typedef struct {
 	__REG32 NBBITSPIXEL1							: 5;
 	__REG32 													: 3;
 	__REG32 BITALIGNMENTPIXEL1				: 4;
 	__REG32 					 								: 4;
 	__REG32 NBBITSPIXEL2							: 5;
 	__REG32 					 								: 3;
 	__REG32 BITALIGNMENTPIXEL2				: 4;
 	__REG32 													: 4;
} __rfbi_data_cycle_bits;

/* RFBI_VSYNC_WIDTH */
typedef struct {
 	__REG32 MINVSYNCPULSEWIDTH				:16;
 	__REG32 													:16;
} __rfbi_vsync_width_bits;

/* RFBI_HSYNC_WIDTH */
typedef struct {
 	__REG32 MINHSYNCPULSEWIDTH				:16;
 	__REG32 													:16;
} __rfbi_hsync_width_bits;

/* VENC_REV_ID */
typedef struct {
 	__REG32 REV_ID										: 8;
 	__REG32 													:24;
} __venc_rev_id_bits;

/* VENC_STATUS */
typedef struct {
 	__REG32 FSQ												: 3;
 	__REG32 CCO												: 1;
 	__REG32 CCE												: 1;
 	__REG32 													:27;
} __venc_status_bits;

/* VENC_F_CONTROL */
typedef struct {
 	__REG32 FMT												: 2;
 	__REG32 BCOLOR										: 3;
 	__REG32 RGBF											: 1;
 	__REG32 SVDS											: 2;
 	__REG32 RESET											: 1;
 	__REG32 													:23;
} __venc_f_control_bits;

/* VENC_VIDOUT_CTRL */
typedef struct {
 	__REG32 _27_54										: 1;
 	__REG32 													:31;
} __venc_vidout_ctrl_bits;

/* VENC_SYNC_CTRL */
typedef struct {
 	__REG32 													: 6;
 	__REG32 FID_POL										: 1;
 	__REG32 													: 1;
 	__REG32 HBLKM											: 2;
 	__REG32 VBLKM											: 2;
 	__REG32 NBLNKS										: 1;
 	__REG32 IGNP											: 1;
 	__REG32 ESAV											: 1;
 	__REG32 FREE											: 1;
 	__REG32 													:16;
} __venc_sync_ctrl_bits;

/* VENC_LLEN */
typedef struct {
 	__REG32 LLEN											:11;
 	__REG32 													:21;
} __venc_llen_bits;

/* VENC_FLENS */
typedef struct {
 	__REG32 FLENS											:11;
 	__REG32 													:21;
} __venc_flens_bits;

/* VENC_HFLTR_CTRL */
typedef struct {
 	__REG32 YINTP											: 1;
 	__REG32 CINTP											: 2;
 	__REG32 													:29;
} __venc_hfltr_ctrl_bits;

/* VENC_CC_CARR_WSS_CARR */
typedef struct {
 	__REG32 FCC												:16;
 	__REG32 FWSS											:16;
} __venc_cc_carr_wss_carr_bits;

/* VENC_C_PHASE */
typedef struct {
 	__REG32 CPHS											: 8;
 	__REG32 													:24;
} __venc_c_phase_bits;

/* VENC_GAIN_U */
typedef struct {
 	__REG32 GU  											: 9;
 	__REG32 													:23;
} __venc_gain_u_bits;

/* VENC_GAIN_V */
typedef struct {
 	__REG32 GV												: 9;
 	__REG32 													:23;
} __venc_gain_v_bits;

/* VENC_GAIN_Y */
typedef struct {
 	__REG32 GY												: 9;
 	__REG32 													:23;
} __venc_gain_y_bits;

/* VENC_BLACK_LEVEL */
typedef struct {
 	__REG32 BLACK											: 7;
 	__REG32 													:25;
} __venc_black_level_bits;

/* VENC_BLANK_LEVEL */
typedef struct {
 	__REG32 BLANK											: 7;
 	__REG32 													:25;
} __venc_blank_level_bits;

/* VENC_X_COLOR */
typedef struct {
 	__REG32 LCD												: 3;
 	__REG32 XCBW											: 2;
 	__REG32 													: 1;
 	__REG32 XCE												: 1;
 	__REG32 													:25;
} __venc_x_color_bits;

/* VENC_M_CONTROL */
typedef struct {
 	__REG32 FFRQ											: 1;
 	__REG32 PAL 											: 1;
 	__REG32 CBW												: 3;
 	__REG32 PALPHS										: 1;
 	__REG32 PALN  										: 1;
 	__REG32 PALI											: 1;
 	__REG32 													:24;
} __venc_m_control_bits;

/* VENC_BSTAMP_WSS_DATA */
typedef struct {
 	__REG32 BSTAP											: 7;
 	__REG32 SQP 											: 1;
 	__REG32 WSS_DATA									:20;
 	__REG32 													: 4;
} __venc_bstamp_wss_data_bits;

/* VENC_LINE21 */
typedef struct {
 	__REG32 L21O											:16;
 	__REG32 L21E 											:16;
} __venc_line21_bits;

/* VENC_LN_SEL */
typedef struct {
 	__REG32 SLINE											: 5;
 	__REG32     											:11;
 	__REG32 LN21_RUNIN								:10;
 	__REG32     											: 6;
} __venc_ln_sel_bits;

/* VENC_L21_WC_CTL */
typedef struct {
 	__REG32 L21EN											: 2;
 	__REG32     											: 6;
 	__REG32 LINE											: 5;
 	__REG32 EVEN_ODD_EN								: 2;
 	__REG32 INV												: 1;
 	__REG32     											:16;
} __venc_l21_wc_ctl_bits;

/* VENC_HTRIGGER_VTRIGGER */
typedef struct {
 	__REG32 HTRIG											:11;
 	__REG32     											: 5;
 	__REG32 VTRIG											:10;
 	__REG32     											: 6;
} __venc_htrigger_vtrigger_bits;

/* VENC_SAVID_EAVID */
typedef struct {
 	__REG32 SAVID											:11;
 	__REG32     											: 5;
 	__REG32 EAVID											:11;
 	__REG32     											: 5;
} __venc_savid_eavid_bits;

/* VENC_FLEN_FAL */
typedef struct {
 	__REG32 FLEN											:10;
 	__REG32     											: 6;
 	__REG32 FAL												: 9;
 	__REG32     											: 7;
} __venc_flen_fal_bits;

/* VENC_LAL_PHASE_RESET */
typedef struct {
 	__REG32 LAL												: 9;
 	__REG32     											: 7;
 	__REG32 SBLANK										: 1;
 	__REG32 PRES											: 2;
 	__REG32     											:13;
} __venc_lal_phase_reset_bits;

/* VENC_HS_INT_START_STOP_X */
typedef struct {
 	__REG32 HS_INT_START_X						:10;
 	__REG32     											: 6;
 	__REG32 HS_INT_STOP_X							:10;
 	__REG32     											: 6;
} __venc_hs_int_start_stop_x_bits;

/* VENC_HS_EXT_START_STOP_X */
typedef struct {
 	__REG32 HS_EXT_START_X						:10;
 	__REG32     											: 6;
 	__REG32 HS_EXT_STOP_X							:10;
 	__REG32     											: 6;
} __venc_hs_ext_start_stop_x_bits;

/* VENC_VS_INT_START_X */
typedef struct {
 	__REG32     											:16;
 	__REG32 VS_INT_START_X						:10;
 	__REG32     											: 6;
} __venc_vs_int_start_x_bits;

/* VENC_VS_INT_STOP_X_VS_INT_START_Y */
typedef struct {
 	__REG32 VS_INT_STOP_X							:10;
 	__REG32     											: 6;
 	__REG32 VS_INT_START_Y						:10;
 	__REG32     											: 6;
} __venc_vs_int_stop_x_vs_int_start_y_bits;

/* VENC_VS_INT_STOP_Y_VS_EXT_START_X */
typedef struct {
 	__REG32 VS_INT_STOP_Y							:10;
 	__REG32     											: 6;
 	__REG32 VS_EXT_START_X						:10;
 	__REG32     											: 6;
} __venc_vs_int_stop_y_vs_ext_start_x_bits;

/* VENC_VS_EXT_STOP_X_VS_EXT_START_Y */
typedef struct {
 	__REG32 VS_EXT_STOP_X							:10;
 	__REG32     											: 6;
 	__REG32 VS_EXT_START_Y						:10;
 	__REG32     											: 6;
} __venc_vs_ext_stop_x_vs_ext_start_y_bits;

/* VENC_VS_EXT_STOP_Y */
typedef struct {
 	__REG32 VS_EXT_STOP_Y							:10;
 	__REG32     											:22;
} __venc_vs_ext_stop_y_bits;

/* VENC_AVID_START_STOP_X */
typedef struct {
 	__REG32 AVID_START_X							:10;
 	__REG32     											: 6;
 	__REG32 AVID_STOP_X								:10;
 	__REG32     											: 6;
} __venc_avid_start_stop_x_bits;

/* VENC_AVID_START_STOP_Y */
typedef struct {
 	__REG32 AVID_START_Y							:10;
 	__REG32     											: 6;
 	__REG32 AVID_STOP_Y								:10;
 	__REG32     											: 6;
} __venc_avid_start_stop_y_bits;

/* VENC_FID_INT_START_X_FID_INT_START_Y */
typedef struct {
 	__REG32 FID_INT_START_X						:10;
 	__REG32     											: 6;
 	__REG32 FID_INT_START_Y						:10;
 	__REG32     											: 6;
} __venc_fid_int_start_x_fid_int_start_y_bits;

/* VENC_FID_INT_OFFSET_Y_FID_EXT_START_X */
typedef struct {
 	__REG32 FID_INT_OFFSET_Y					:10;
 	__REG32     											: 6;
 	__REG32 FID_EXT_START_X						:10;
 	__REG32     											: 6;
} __venc_fid_int_offset_y_fid_ext_start_x_bits;

/* VENC_FID_EXT_START_Y_FID_EXT_OFFSET_Y */
typedef struct {
 	__REG32 FID_EXT_START_Y						:10;
 	__REG32     											: 6;
 	__REG32 FID_EXT_OFFSET_Y					:10;
 	__REG32     											: 6;
} __venc_fid_ext_start_y_fid_ext_offset_y_bits;

/* VENC_TVDETGP_INT_START_STOP_X */
typedef struct {
 	__REG32 TVDETGP_INT_START_X				:10;
 	__REG32     											: 6;
 	__REG32 TVDETGP_INT_STOP_X				:10;
 	__REG32     											: 6;
} __venc_tvdetgp_int_start_stop_x_bits;

/* VENC_TVDETGP_INT_START_STOP_Y */
typedef struct {
 	__REG32 TVDETGP_INT_START_Y				:10;
 	__REG32     											: 6;
 	__REG32 TVDETGP_INT_STOP_Y				:10;
 	__REG32     											: 6;
} __venc_tvdetgp_int_start_stop_y_bits;

/* VENC_GEN_CTRL */
typedef struct {
 	__REG32 EN												: 1;
 	__REG32     											:15;
 	__REG32 TVDP											: 1;
 	__REG32 FEP 											: 1;
 	__REG32 FIP												: 1;
 	__REG32 AVIDP											: 1;
 	__REG32 VEP												: 1;
 	__REG32 HEP												: 1;
 	__REG32 VIP												: 1;
 	__REG32 HIP												: 1;
 	__REG32 CBAR											: 1;
 	__REG32 _656											: 1;
 	__REG32 MS												: 1;
 	__REG32     											: 5;
} __venc_gen_ctrl_bits;

/* VENC_OUTPUT_CONTROL */
typedef struct {
 	__REG32 LUMA_ENABLE								: 1;
 	__REG32 COMPOSITE_ENABLE					: 1;
 	__REG32 CHROMA_ENABLE							: 1;
 	__REG32 VIDEO_INVERT							: 1;
 	__REG32 TEST_MODE									: 1;
 	__REG32 LUMA_SOURCE								: 1;
 	__REG32 COMPOSITE_SOURCE					: 1;
 	__REG32 CHROMA_SOURCE							: 1;
 	__REG32     											: 8;
 	__REG32 LUMA_TEST									:10;
 	__REG32     											: 6;
} __venc_output_control_bits;

/* VENC_OUTPUT_TEST */
typedef struct {
 	__REG32 COMPOSITE_TEST						:10;
 	__REG32     											: 6;
 	__REG32 CHROMA_TEST								:10;
 	__REG32     											: 6;
} __venc_output_test_bits;

/* DSI_REVISION */
typedef struct {
 	__REG32 REV												: 8;
 	__REG32     											:24;
} __dsi_revision_bits;

/* DSI_SYSCONFIG */
typedef struct {
 	__REG32 AUTO_IDLE									: 1;
 	__REG32 SOFT_RESET								: 1;
 	__REG32 ENWAKEUP									: 1;
 	__REG32 SIDLEMODE									: 2;
 	__REG32     											: 3;
 	__REG32 CLOCKACTIVITY							: 2;
 	__REG32     											:22;
} __dsi_sysconfig_bits;

/* DSI_SYSSTATUS */
typedef struct {
 	__REG32 RESET_DONE								: 1;
 	__REG32     											:31;
} __dsi_sysstatus_bits;

/* DSI_IRQSTATUS */
typedef struct {
 	__REG32 VIRTUAL_CHANNEL0_IRQ			: 1;
 	__REG32 VIRTUAL_CHANNEL1_IRQ			: 1;
 	__REG32 VIRTUAL_CHANNEL2_IRQ			: 1;
 	__REG32 VIRTUAL_CHANNEL3_IRQ			: 1;
 	__REG32 WAKEUP_IRQ								: 1;
 	__REG32 RESYNCHRONIZATION_IRQ			: 1;
 	__REG32     											: 1;
 	__REG32 PLL_LOCK_IRQ							: 1;
 	__REG32 PLL_UNLOCK_IRQ						: 1;
 	__REG32 PLL_RECAL_IRQ							: 1;
 	__REG32 COMPLEXIO_ERR_IRQ					: 1;
 	__REG32     											: 3;
 	__REG32 HS_TX_TO_IRQ							: 1;
 	__REG32 LP_RX_TO_IRQ							: 1;
 	__REG32 TE_TRIGGER_IRQ						: 1;
 	__REG32 ACK_TRIGGER_IRQ						: 1;
 	__REG32 SYNC_LOST_IRQ							: 1;
 	__REG32 LDO_POWER_GOOD_IRQ				: 1;
 	__REG32 TA_TO_IRQ									: 1;
 	__REG32     											:11;
} __dsi_irqstatus_bits;

/* DSI_IRQSTATUS */
typedef struct {
 	__REG32     											: 4;
 	__REG32 WAKEUP_IRQ_EN							: 1;
 	__REG32 RESYNCHRONIZATION_IRQ_EN	: 1;
 	__REG32 													: 1;
 	__REG32 PLL_LOCK_IRQ_EN						: 1;
 	__REG32 PLL_UNLOCK_IRQ_EN					: 1;
 	__REG32 PLL_RECAL_IRQ_EN					: 1;
 	__REG32     											: 4;
 	__REG32 HS_TX_TO_IRQ_EN						: 1;
 	__REG32 LP_RX_TO_IRQ_EN						: 1;
 	__REG32 TE_TRIGGER_IRQ_EN					: 1;
 	__REG32 ACK_TRIGGER_IRQ_EN				: 1;
 	__REG32 SYNC_LOST_IRQ_EN					: 1;
 	__REG32 LDO_POWER_GOOD_IRQ_EN			: 1;
 	__REG32 TA_TO_IRQ_EN							: 1;
 	__REG32     											:11;
} __dsi_irqenable_bits;

/* DSI_CTRL */
typedef struct {
 	__REG32 IF_EN											: 1;
 	__REG32 CS_RX_EN									: 1;
 	__REG32 ECC_RX_EN									: 1;
 	__REG32 TX_FIFO_ARBITRATION				: 1;
 	__REG32 VP_CLK_RATIO							: 1;
 	__REG32 TRIGGER_RESET							: 1;
 	__REG32 VP_DATA_BUS_WIDTH					: 2;
 	__REG32 VP_CLK_POL								: 1;
 	__REG32 VP_DE_POL									: 1;
 	__REG32 VP_HSYNC_POL							: 1;
 	__REG32 VP_VSYNC_POL							: 1;
 	__REG32 LINE_BUFFER								: 2;
 	__REG32 TRIGGER_RESET_MODE				: 1;
 	__REG32 VP_VSYNC_START						: 1;
 	__REG32 VP_VSYNC_END							: 1;
 	__REG32 VP_HSYNC_START						: 1;
 	__REG32 VP_HSYNC_END							: 1;
 	__REG32 EOT_ENABLE								: 1;
 	__REG32 BLANKING_MODE							: 1;
 	__REG32 HFP_BLANKING_MODE					: 1;
 	__REG32 HBP_BLANKING_MODE					: 1;
 	__REG32 HSA_BLANKING_MODE					: 1;
 	__REG32 DCS_CMD_ENABLE						: 1;
 	__REG32 DCS_CMD_CODE							: 1;
 	__REG32 RGB565_ORDER							: 1;
 	__REG32     											: 5;
} __dsi_ctrl_bits;

/* DSI_COMPLEXIO_CFG1 */
typedef struct {
 	__REG32 CLOCK_POSITION						: 3;
 	__REG32 CLOCK_POL									: 1;
 	__REG32 DATA1_POSITION						: 3;
 	__REG32 DATA1_POL									: 1;
 	__REG32 DATA2_POSITION						: 3;
 	__REG32 DATA2_POL									: 1;
 	__REG32 													: 8;
 	__REG32 USE_LDO_EXTERNAL					: 1;
 	__REG32 LDO_POWER_GOOD_STATE			: 1;
 	__REG32 													: 3;
 	__REG32 PWR_STATUS								: 2;
 	__REG32 PWR_CMD										: 2;
 	__REG32 RESET_DONE								: 1;
 	__REG32 GOBIT											: 1;
 	__REG32 SHADOWING									: 1;
} __dsi_complexio_cfg_bits;

/* DSI_COMPLEXIO_IRQ_STATUS */
typedef struct {
 	__REG32 ERRSYNCESC1_IRQ						: 1;
 	__REG32 ERRSYNCESC2_IRQ						: 1;
 	__REG32 ERRSYNCESC3_IRQ						: 1;
 	__REG32 													: 2;
 	__REG32 ERRESC1_IRQ								: 1;
 	__REG32 ERRESC2_IRQ								: 1;
 	__REG32 ERRESC3_IRQ								: 1;
 	__REG32 													: 2;
 	__REG32 ERRCONTROL1_IRQ						: 1;
 	__REG32 ERRCONTROL2_IRQ						: 1;
 	__REG32 ERRCONTROL3_IRQ						: 1;
 	__REG32 													: 2;
 	__REG32 STATEULPS1_IRQ						: 1;
 	__REG32 STATEULPS2_IRQ						: 1;
 	__REG32 STATEULPS3_IRQ						: 1;
 	__REG32 													: 2;
 	__REG32 ERRCONTENTIONLP0_1_IRQ		: 1;
 	__REG32 ERRCONTENTIONLP1_1_IRQ		: 1;
 	__REG32 ERRCONTENTIONLP0_2_IRQ		: 1;
 	__REG32 ERRCONTENTIONLP1_2_IRQ		: 1;
 	__REG32 ERRCONTENTIONLP0_3_IRQ		: 1;
 	__REG32 ERRCONTENTIONLP1_3_IRQ		: 1;
 	__REG32 													: 4;
 	__REG32 ULPSACTIVENOT_ALL0_IRQ		: 1;
 	__REG32 ULPSACTIVENOT_ALL1_IRQ		: 1;
} __dsi_complexio_irq_status_bits;

/* DSI_COMPLEXIO_IRQ_ENABLE */
typedef struct {
 	__REG32 ERRSYNCESC1_IRQ_EN				: 1;
 	__REG32 ERRSYNCESC2_IRQ_EN				: 1;
 	__REG32 ERRSYNCESC3_IRQ_EN				: 1;
 	__REG32 													: 2;
 	__REG32 ERRESC1_IRQ_EN						: 1;
 	__REG32 ERRESC2_IRQ_EN						: 1;
 	__REG32 ERRESC3_IRQ_EN						: 1;
 	__REG32 													: 2;
 	__REG32 ERRCONTROL1_IRQ_EN				: 1;
 	__REG32 ERRCONTROL2_IRQ_EN				: 1;
 	__REG32 ERRCONTROL3_IRQ_EN				: 1;
 	__REG32 													: 2;
 	__REG32 STATEULPS1_IRQ_EN					: 1;
 	__REG32 STATEULPS2_IRQ_EN					: 1;
 	__REG32 STATEULPS3_IRQ_EN					: 1;
 	__REG32 													: 2;
 	__REG32 ERRCONTENTIONLP0_1_IRQ_EN	: 1;
 	__REG32 ERRCONTENTIONLP1_1_IRQ_EN	: 1;
 	__REG32 ERRCONTENTIONLP0_2_IRQ_EN	: 1;
 	__REG32 ERRCONTENTIONLP1_2_IRQ_EN	: 1;
 	__REG32 ERRCONTENTIONLP0_3_IRQ_EN	: 1;
 	__REG32 ERRCONTENTIONLP1_3_IRQ_EN	: 1;
 	__REG32 													: 4;
 	__REG32 ULPSACTIVENOT_ALL0_IRQ_EN	: 1;
 	__REG32 ULPSACTIVENOT_ALL1_IRQ_EN	: 1;
} __dsi_complexio_irq_enable_bits;

/* DSI_CLK_CTRL */
typedef struct {
 	__REG32 LP_CLK_DIVISOR						:13;
 	__REG32 DDR_CLK_ALWAYS_ON					: 1;
 	__REG32 CIO_CLK_ICG								: 1;
 	__REG32 LP_CLK_NULL_PACKET_ENABLE	: 1;
 	__REG32 LP_CLK_NULL_PACKET_SIZE		: 2;
 	__REG32 HS_AUTO_STOP_ENABLE				: 1;
 	__REG32 HS_MANUAL_STOP_CTRL				: 1;
 	__REG32 LP_CLK_ENABLE							: 1;
 	__REG32 LP_RX_SYNCHRO_ENABLE			: 1;
 	__REG32 													: 6;
 	__REG32 PLL_PWR_STATUS						: 2;
 	__REG32 PLL_PWR_CMD								: 2;
} __dsi_clk_ctrl_bits;

/* DSI_TIMING1 */
typedef struct {
 	__REG32 STOP_STATE_COUNTER_IO			:13;
 	__REG32 STOP_STATE_X4_IO					: 1;
 	__REG32 STOP_STATE_X16_IO					: 1;
 	__REG32 FORCE_TX_STOP_MODE_IO			: 1;
 	__REG32 TA_TO_COUNTER							:13;
 	__REG32 TA_TO_X8									: 1;
 	__REG32 TA_TO_X16									: 1;
 	__REG32 TA_TO											: 1;
} __dsi_timing1_bits;

/* DSI_TIMING2 */
typedef struct {
 	__REG32 LP_RX_TO_COUNTER					:13;
 	__REG32 LP_RX_TO_X4								: 1;
 	__REG32 LP_RX_TO_X16							: 1;
 	__REG32 LP_RX_TO									: 1;
 	__REG32 HS_TX_TO_COUNTER					:13;
 	__REG32 HS_TX_TO_X8								: 1;
 	__REG32 HS_TX_TO_X16							: 1;
 	__REG32 HS_TX_TO									: 1;
} __dsi_timing2_bits;

/* DSI_VM_TIMING1 */
typedef struct {
 	__REG32 HBP												:12;
 	__REG32 HFP												:12;
 	__REG32 HSA												: 8;
} __dsi_vm_timing1_bits;

/* DSI_VM_TIMING2 */
typedef struct {
 	__REG32 VBP												: 8;
 	__REG32 VFP												: 8;
 	__REG32 VSA												: 8;
 	__REG32 WINDOW_SYNC								: 4;
 	__REG32   												: 4;
} __dsi_vm_timing2_bits;

/* DSI_VM_TIMING3 */
typedef struct {
 	__REG32 VACT											:16;
 	__REG32 TL												:16;
} __dsi_vm_timing3_bits;

/* DSI_CLK_TIMING */
typedef struct {
 	__REG32 DDR_CLK_POST							: 8;
 	__REG32 DDR_CLK_PRE								: 8;
 	__REG32           								:16;
} __dsi_clk_timing_bits;

/* DSI_TX_FIFO_VC_SIZE */
typedef struct {
 	__REG32 VC0_FIFO_ADD							: 3;
 	__REG32           								: 1;
 	__REG32 VC0_FIFO_SIZE							: 4;
 	__REG32 VC1_FIFO_ADD							: 3;
 	__REG32           								: 1;
 	__REG32 VC1_FIFO_SIZE							: 4;
 	__REG32 VC2_FIFO_ADD							: 3;
 	__REG32           								: 1;
 	__REG32 VC2_FIFO_SIZE							: 4;
 	__REG32 VC3_FIFO_ADD							: 3;
 	__REG32           								: 1;
 	__REG32 VC3_FIFO_SIZE							: 4;
} __dsi_tx_fifo_vc_size_bits;

/* DSI_COMPLEXIO_CFG2 */
typedef struct {
 	__REG32 LANE1_ULPS_SIG1						: 1;
 	__REG32 LANE2_ULPS_SIG1						: 1;
 	__REG32 LANE3_ULPS_SIG1						: 1;
 	__REG32           								: 2;
 	__REG32 LANE1_ULPS_SIG2						: 1;
 	__REG32 LANE2_ULPS_SIG2						: 1;
 	__REG32 LANE3_ULPS_SIG2						: 1;
 	__REG32           								: 8;
 	__REG32 HS_BUSY										: 1;
 	__REG32 LP_BUSY										: 1;
 	__REG32           								:14;
} __dsi_complexio_cfg2_bits;

/* DSI_RX_FIFO_VC_FULLNESS */
typedef struct {
 	__REG32 VC0_FIFO_FULLNESS					: 8;
 	__REG32 VC1_FIFO_FULLNESS					: 8;
 	__REG32 VC2_FIFO_FULLNESS					: 8;
 	__REG32 VC3_FIFO_FULLNESS					: 8;
} __dsi_rx_fifo_vc_fullness_bits;

/* DSI_VM_TIMING4 */
typedef struct {
 	__REG32 HBP_HS_INTERLEAVING				: 8;
 	__REG32 HFP_HS_INTERLEAVING				: 8;
 	__REG32 HSA_HS_INTERLEAVING				: 8;
 	__REG32 													: 8;
} __dsi_vm_timing4_bits;

/* DSI_TX_FIFO_VC_EMPTINESS */
typedef struct {
 	__REG32 VC0_FIFO_EMPTINESS				: 8;
 	__REG32 VC1_FIFO_EMPTINESS				: 8;
 	__REG32 VC2_FIFO_EMPTINESS				: 8;
 	__REG32 VC3_FIFO_EMPTINESS				: 8;
} __dsi_tx_fifo_vc_emptiness_bits;

/* DSI_VM_TIMING5 */
typedef struct {
 	__REG32 HBP_LP_INTERLEAVING				: 8;
 	__REG32 HFP_LP_INTERLEAVING				: 8;
 	__REG32 HSA_LP_INTERLEAVING				: 8;
 	__REG32 													: 8;
} __dsi_vm_timing5_bits;

/* DSI_VM_TIMING6 */
typedef struct {
 	__REG32 BL_LP_INTERLEAVING				:16;
 	__REG32 BL_HS_INTERLEAVING				:16;
} __dsi_vm_timing6_bits;

/* DSI_VM_TIMING7 */
typedef struct {
 	__REG32 EXIT_HS_MODE_LATENCY			:16;
 	__REG32 ENTER_HS_MODE_LATENCY			:16;
} __dsi_vm_timing7_bits;

/* DSI_STOPCLK_TIMING */
typedef struct {
 	__REG32 DSI_STOPCLK_LATENCY				: 8;
 	__REG32 													:24;
} __dsi_stopclk_timing_bits;

/* DSI_VCn_CTRL */
typedef struct {
 	__REG32 VC_EN											: 1;
 	__REG32 SOURCE										: 1;
 	__REG32 BTA_SHORT_EN							: 1;
 	__REG32 BTA_LONG_EN								: 1;
 	__REG32 MODE											: 1;
 	__REG32 TX_FIFO_NOT_EMPTY					: 1;
 	__REG32 BTA_EN										: 1;
 	__REG32 CS_TX_EN									: 1;
 	__REG32 ECC_TX_EN									: 1;
 	__REG32 MODE_SPEED								: 1;
 	__REG32 													: 4;
 	__REG32 PP_BUSY										: 1;
 	__REG32 VC_BUSY										: 1;
 	__REG32 TX_FIFO_FULL							: 1;
 	__REG32 DMA_TX_THRESHOLD					: 3;
 	__REG32 RX_FIFO_NOT_EMPTY					: 1;
 	__REG32 DMA_TX_REQ_NB							: 3;
 	__REG32 DMA_RX_THRESHOLD					: 3;
 	__REG32 DMA_RX_REQ_NB							: 3;
 	__REG32 													: 2;
} __dsi_vcn_ctrl_bits;

/* DSI_VCn_TE */
typedef struct {
 	__REG32 TE_SIZE										:24;
 	__REG32 													: 6;
 	__REG32 TE_EN   									: 1;
 	__REG32 TE_START									: 1;
} __dsi_vcn_te_bits;

/* DSI_VCn_IRQSTATUS */
typedef struct {
 	__REG32 CS_IRQ										: 1;
 	__REG32 ECC_CORRECTION_IRQ				: 1;
 	__REG32 PACKET_SENT_IRQ						: 1;
 	__REG32 FIFO_TX_OVF_IRQ						: 1;
 	__REG32 FIFO_RX_OVF_IRQ						: 1;
 	__REG32 BTA_IRQ										: 1;
 	__REG32 ECC_NO_CORRECTION_IRQ			: 1;
 	__REG32 FIFO_TX_UDF_IRQ						: 1;
 	__REG32 PP_BUSY_CHANGE_IRQ				: 1;
 	__REG32 													:23;
} __dsi_vcn_irqstatus_bits;

/* DSI_VCn_IRQENABLE */
typedef struct {
 	__REG32 CS_IRQ_EN									: 1;
 	__REG32 ECC_CORRECTION_IRQ_EN			: 1;
 	__REG32 PACKET_SENT_IRQ_EN				: 1;
 	__REG32 FIFO_TX_OVF_IRQ_EN				: 1;
 	__REG32 FIFO_RX_OVF_IRQ_EN				: 1;
 	__REG32 BTA_IRQ_EN								: 1;
 	__REG32 ECC_NO_CORRECTION_IRQ_EN	: 1;
 	__REG32 FIFO_TX_UDF_IRQ_EN				: 1;
 	__REG32 PP_BUSY_CHANGE_IRQ_EN			: 1;
 	__REG32 													:23;
} __dsi_vcn_irqenable_bits;

/* DSI_PHY_CFG0 */
typedef struct {
 	__REG32 THS_EXIT									: 8;
 	__REG32 THS_TRAIL									: 8;
 	__REG32 THS_PREPARE_THS_ZERO			: 8;
 	__REG32 THS_PREPARE								: 8;
} __dsi_phy_cfg0_bits;

/* DSI_PHY_CFG1 */
typedef struct {
 	__REG32 TCLK_ZERO									: 8;
 	__REG32 TCLK_TRAIL								: 8;
 	__REG32 TLPX_HALF									: 7;
 	__REG32 													: 9;
} __dsi_phy_cfg1_bits;

/* DSI_PHY_CFG2 */
typedef struct {
 	__REG32 TCLK_PREPARE							: 8;
 	__REG32 													:24;
} __dsi_phy_cfg2_bits;

/* DSI_PHY_CFG5 */
typedef struct {
 	__REG32 													:26;
 	__REG32 RESETDONETXCLKESC2				: 1;
 	__REG32 RESETDONETXCLKESC1				: 1;
 	__REG32 RESETDONETXCLKESC0				: 1;
 	__REG32 RESETDONEPWRCLK						: 1;
 	__REG32 RESETDONESCPCLK						: 1;
 	__REG32 RESETDONETXBYTECLK				: 1;
} __dsi_phy_cfg5_bits;

/* DSI_PLL_CONTROL */
typedef struct {
 	__REG32 DSI_PLL_AUTOMODE					: 1;
 	__REG32 DSI_PLL_GATEMODE					: 1;
 	__REG32 DSI_PLL_HALTMODE					: 1;
 	__REG32 DSI_PLL_SYSRESET					: 1;
 	__REG32 DSI_HSDIV_SYSRESET				: 1;
 	__REG32 													:27;
} __dsi_pll_control_bits;

/* DSI_PLL_STATUS */
typedef struct {
 	__REG32 DSI_PLLCTRL_RESET_DONE		: 1;
 	__REG32 DSI_PLL_LOCK							: 1;
 	__REG32 DSI_PLL_RECAL							: 1;
 	__REG32 DSI_PLL_LOSSREF						: 1;
 	__REG32 DSI_PLL_LIMP							: 1;
 	__REG32 DSI_PLL_HIGHJITTER				: 1;
 	__REG32 DSI_PLL_BYPASS						: 1;
 	__REG32 DSS_CLOCK_ACK							: 1;
 	__REG32 DSIPROTO_CLOCK_ACK				: 1;
 	__REG32 DSI_BYPASSACKZ						: 1;
 	__REG32 													:22;
} __dsi_pll_status_bits;

/* DSI_PLL_GO */
typedef struct {
 	__REG32 DSI_PLL_GO								: 1;
 	__REG32 													:31;
} __dsi_pll_go_bits;

/* DSI_PLL_CONFIGURATION1 */
typedef struct {
 	__REG32 DSI_PLL_STOPMODE					: 1;
 	__REG32 DSI_PLL_REGN							: 7;
 	__REG32 DSI_PLL_REGM							:11;
 	__REG32 DSS_CLOCK_DIV							: 4;
 	__REG32 DSIPROTO_CLOCK_DIV				: 4;
 	__REG32 													: 5;
} __dsi_pll_configuration1_bits;

/* DSI_PLL_CONFIGURATION2 */
typedef struct {
 	__REG32 DSI_PLL_IDLE							: 1;
 	__REG32 DSI_PLL_FREQSEL						: 4;
 	__REG32 DSI_PLL_PLLLPMODE					: 1;
 	__REG32 DSI_PLL_LOWCURRSTBY				: 1;
 	__REG32 DSI_PLL_TIGHTPHASELOCK		: 1;
 	__REG32 DSI_PLL_DRIFTGUARDEN			: 1;
 	__REG32 DSI_PLL_LOCKSEL						: 2;
 	__REG32 DSI_PLL_CLKSEL						: 1;
 	__REG32 DSI_PLL_HIGHFREQ					: 1;
 	__REG32 DSI_PLL_REFEN							: 1;
 	__REG32 DSI_PHY_CLKINEN						: 1;
 	__REG32 DSI_BYPASSEN							: 1;
 	__REG32 DSS_CLOCK_EN							: 1;
 	__REG32 DSS_CLOCK_PWDN						: 1;
 	__REG32 DSI_PROTO_CLOCK_EN				: 1;
 	__REG32 DSI_PROTO_CLOCK_PWDN			: 1;
 	__REG32 DSI_HSDIVBYPASS						: 1;
 	__REG32 													:11;
} __dsi_pll_configuration2_bits;

/* GPTn_TIDR */
typedef struct {
 	__REG32 TID_REV										: 8;
 	__REG32 													:24;
} __gpt_tidr_bits;

/* GPTn_TIOCP_CFG */
typedef struct {
 	__REG32 AUTOIDLE									: 1;
 	__REG32 SOFTRESET									: 1;
 	__REG32 ENAWAKEUP									: 1;
 	__REG32 IDLEMODE									: 2;
 	__REG32 EMUFREE										: 1;
 	__REG32 													: 2;
 	__REG32 CLOCKACTIVITY							: 2;
 	__REG32 													:22;
} __gpt_tiocp_cfg_bits;

/* GPTn_TISTAT */
typedef struct {
 	__REG32 RESETDONE									: 1;
 	__REG32 													:31;
} __gpt_tistat_bits;

/* GPTn_TISR */
typedef struct {
 	__REG32 MAT_IT_FLAG								: 1;
 	__REG32 OVF_IT_FLAG								: 1;
 	__REG32 TCAR_IT_FLAG							: 1;
 	__REG32 													:29;
} __gpt_tisr_bits;

/* GPTn_TIER */
typedef struct {
 	__REG32 MAT_IT_ENA								: 1;
 	__REG32 OVF_IT_ENA								: 1;
 	__REG32 TCAR_IT_ENA								: 1;
 	__REG32 													:29;
} __gpt_tier_bits;

/* GPTn_TWER */
typedef struct {
 	__REG32 MAT_WUP_ENA								: 1;
 	__REG32 OVF_WUP_ENA								: 1;
 	__REG32 TCAR_WUP_ENA							: 1;
 	__REG32 													:29;
} __gpt_twer_bits;

/* GPTn_TCLR */
typedef struct {
 	__REG32 ST												: 1;
 	__REG32 AR												: 1;
 	__REG32 PTV												: 3;
 	__REG32 PRE												: 1;
 	__REG32 CE												: 1;
 	__REG32 SCPWM											: 1;
 	__REG32 TCM												: 2;
 	__REG32 TRG												: 2;
 	__REG32 PT												: 1;
 	__REG32 CAPT_MODE									: 1;
 	__REG32 GPO_CFG										: 1;
 	__REG32 													:17;
} __gpt_tclr_bits;

/* GPTn_TWPS */
typedef struct {
 	__REG32 W_PEND_TCLR								: 1;
 	__REG32 W_PEND_TCRR								: 1;
 	__REG32 W_PEND_TLDR								: 1;
 	__REG32 W_PEND_TTGR								: 1;
 	__REG32 W_PEND_TMAR								: 1;
 	__REG32 W_PEND_TPIR								: 1;
 	__REG32 W_PEND_TNIR								: 1;
 	__REG32 W_PEND_TCVR								: 1;
 	__REG32 W_PEND_TOCR								: 1;
 	__REG32 W_PEND_TOWR								: 1;
 	__REG32 													:22;
} __gpt_twps_bits;

/* GPTn_TSICR */
typedef struct {
 	__REG32 													: 1;
 	__REG32 SFT												: 1;
 	__REG32 POSTED										: 1;
 	__REG32 													:29;
} __gpt_tsicr_bits;

/* GPTn_TOCR */
typedef struct {
 	__REG32 OVF_COUNTER_VALUE					:24;
 	__REG32 													: 8;
} __gpt_tocr_bits;

/* GPTn_TOWR */
typedef struct {
 	__REG32 OVF_COUNTER_VALUE					:24;
 	__REG32 													: 8;
} __gpt_towr_bits;

/* WDTn_WIDR */
typedef struct {
 	__REG32 WD_REV										: 8;
 	__REG32 													:24;
} __wdt_widr_bits;

/* WDTn_SYSCONFIG */
typedef struct {
 	__REG32 AUTOIDLE									: 1;
 	__REG32 SOFTRESET									: 1;
 	__REG32 ENAWAKEUP									: 1;
 	__REG32 IDLEMODE									: 2;
 	__REG32 EMUFREE										: 1;
 	__REG32 													: 2;
 	__REG32 CLOCKACTIVITY							: 2;
 	__REG32 													:22;
} __wdt_sysconfig_bits;

/* WDTn_SYSSTATUS */
typedef struct {
 	__REG32 RESETDONE									: 1;
 	__REG32 													:31;
} __wdt_sysstatus_bits;

/* WDTn_WISR */
typedef struct {
 	__REG32 OVF_IT_FLAG								: 1;
 	__REG32 													:31;
} __wdt_wisr_bits;

/* WDTn_WIER */
typedef struct {
 	__REG32 OVF_IT_ENA								: 1;
 	__REG32 													:31;
} __wdt_wier_bits;

/* WDTn_WCLR */
typedef struct {
 	__REG32 													: 2;
 	__REG32 PTV												: 3;
 	__REG32 PRE												: 1;
 	__REG32 													:26;
} __wdt_wclr_bits;

/* WDTn_WWPS */
typedef struct {
 	__REG32 W_PEND_WCLR								: 1;
 	__REG32 W_PEND_WCRR								: 1;
 	__REG32 W_PEND_WLDR								: 1;
 	__REG32 W_PEND_WTGR								: 1;
 	__REG32 W_PEND_WSPR								: 1;
 	__REG32 													:27;
} __wdt_wwps_bits;

/* REG_32KSYNCNT_REV */
typedef struct {
 	__REG32 CID_REV										: 8;
 	__REG32 													:24;
} __reg_32ksyncnt_rev_bits;

/* REG_32KSYNCNT_SYSCONFIG */
typedef struct {
 	__REG32 													: 3;
 	__REG32 IDLEMODE									: 2;
 	__REG32 													:27;
} __reg_32ksyncnt_sysconfig_bits;

/* I2Cn_REV */
typedef struct {
 	__REG16 REV												: 8;
 	__REG16 													: 8;
} __i2c_rev_bits;

/* I2Cn_IE */
typedef struct {
 	__REG16 AL_IE											: 1;
 	__REG16 NACK_IE										: 1;
 	__REG16 ARDY_IE										: 1;
 	__REG16 RRDY_IE										: 1;
 	__REG16 XRDY_IE										: 1;
 	__REG16 GC_IE											: 1;
 	__REG16 STC_IE										: 1;
 	__REG16 AERR_IE										: 1;
 	__REG16 BF_IE											: 1;
 	__REG16 AAS_IE										: 1;
 	__REG16         									: 3;
 	__REG16 RDR_IE										: 1;
 	__REG16 XDR_IE										: 1;
 	__REG16 													: 1;
} __i2c_ie_bits;

/* I2Cn_STAT */
typedef struct {
 	__REG16 AL												: 1;
 	__REG16 NACK 											: 1;
 	__REG16 ARDY 											: 1;
 	__REG16 RRDY											: 1;
 	__REG16 XRDY											: 1;
 	__REG16 GC												: 1;
 	__REG16 STC												: 1;
 	__REG16 AERR											: 1;
 	__REG16 BF												: 1;
 	__REG16 AAS												: 1;
 	__REG16 XUDF											: 1;
 	__REG16 ROVR											: 1;
 	__REG16 BB												: 1;
 	__REG16 RDR												: 1;
 	__REG16 XDR												: 1;
 	__REG16 													: 1;
} __i2c_stat_bits;

/* I2Cn_WE */
typedef struct {
 	__REG16 AL_WE											: 1;
 	__REG16 NACK_WE										: 1;
 	__REG16 ARDY_WE										: 1;
 	__REG16 DRDY_WE										: 1;
 	__REG16 													: 1;
 	__REG16 GC_WE											: 1;
 	__REG16 STC_WE										: 1;
 	__REG16 													: 1;
 	__REG16 BF_WE											: 1;
 	__REG16 AAS_WE										: 1;
 	__REG16 													: 3;
 	__REG16 RDR_WE										: 1;
 	__REG16 XDR_WE										: 1;
 	__REG16 													: 1;
} __i2c_we_bits;

/* I2Cn_SYSS */
typedef struct {
 	__REG16 RDONE											: 1;
 	__REG16 													:15;
} __i2c_syss_bits;

/* I2Cn_BUF */
typedef struct {
 	__REG16 XTRSH											: 6;
 	__REG16 TXFIFO_CLR								: 1;
 	__REG16 XDMA_EN										: 1;
 	__REG16 RTRSH											: 6;
 	__REG16 RXFIFO_CLR								: 1;
 	__REG16 RDMA_EN										: 1;
} __i2c_buf_bits;

/* I2Cn_DATA */
typedef struct {
 	__REG16 DATA											: 8;
 	__REG16 													: 8;
} __i2c_data_bits;

/* I2Cn_SYSC */
typedef struct {
 	__REG16 AUTOIDLE									: 1;
 	__REG16 SRST											: 1;
 	__REG16 ENAWAKEUP									: 1;
 	__REG16 IDLEMODE									: 2;
 	__REG16 													: 3;
 	__REG16 CLOCKACTIVITY							: 2;
 	__REG16 													: 6;
} __i2c_sysc_bits;

/* I2Cn_CON */
typedef struct {
 	__REG16 STT												: 1;
 	__REG16 STP												: 1;
 	__REG16 													: 2;
 	__REG16 XOA3											: 1;
 	__REG16 XOA2											: 1;
 	__REG16 XOA1											: 1;
 	__REG16 XOA0											: 1;
 	__REG16 XSA												: 1;
 	__REG16 TRX												: 1;
 	__REG16 MST												: 1;
 	__REG16 STB												: 1;
 	__REG16 OPMODE										: 2;
 	__REG16 													: 1;
 	__REG16 I2C_EN										: 1;
} __i2c_con_bits;

/* I2Cn_OA0 */
typedef struct {
 	__REG16 OA												:10;
 	__REG16 													: 3;
 	__REG16 MCODE											: 3;
} __i2c_oa0_bits;

/* I2Cn_SA */
typedef struct {
 	__REG16 SA												:10;
 	__REG16 													: 6;
} __i2c_sa_bits;

/* I2Cn_PSC */
typedef struct {
 	__REG16 PSC												: 8;
 	__REG16 													: 8;
} __i2c_psc_bits;

/* I2Cn_SCLL */
typedef struct {
 	__REG16 SCLL											: 8;
 	__REG16 HSSCLL										: 8;
} __i2c_scll_bits;

/* I2Cn_SCLH */
typedef struct {
 	__REG16 SCLH											: 8;
 	__REG16 HSSCLH										: 8;
} __i2c_sclh_bits;

/* I2Cn_SYSTEST */
typedef struct {
 	__REG16 SDA_O											: 1;
 	__REG16 SDA_I											: 1;
 	__REG16 SCL_O											: 1;
 	__REG16 SCL_I											: 1;
 	__REG16 SCCBE_O										: 1;
 	__REG16 													: 6;
 	__REG16 SSB												: 1;
 	__REG16 TMODE											: 2;
 	__REG16 FREE											: 1;
 	__REG16 ST_EN											: 1;
} __i2c_systest_bits;

/* I2Cn_BUFSTAT */
typedef struct {
 	__REG16 TXSTAT										: 6;
 	__REG16 													: 2;
 	__REG16 RXSTAT										: 6;
 	__REG16 FIFODEPTH									: 2;
} __i2c_bufstat_bits;

/* I2Cn_OA1 */
typedef struct {
 	__REG16 OA1												:10;
 	__REG16 													: 6;
} __i2c_oa1_bits;

/* I2Cn_OA2 */
typedef struct {
 	__REG16 OA2												:10;
 	__REG16 													: 6;
} __i2c_oa2_bits;

/* I2Cn_OA3 */
typedef struct {
 	__REG16 OA3												:10;
 	__REG16 													: 6;
} __i2c_oa3_bits;

/* I2Cn_ACTOA */
typedef struct {
 	__REG16 OA0_ACT										: 1;
 	__REG16 OA1_ACT										: 1;
 	__REG16 OA2_ACT										: 1;
 	__REG16 OA3_ACT										: 1;
 	__REG16 													:12;
} __i2c_actoa_bits;

/* I2Cn_SBLOCK */
typedef struct {
 	__REG16 OA0_EN										: 1;
 	__REG16 OA1_EN										: 1;
 	__REG16 OA2_EN										: 1;
 	__REG16 OA3_EN										: 1;
 	__REG16 													:12;
} __i2c_sblock_bits;

/* MCSPIn_REVISION */
typedef struct {
 	__REG32 REV												: 8;
 	__REG32 													:24;
} __mcspi_revision_bits;

/* MCSPIn_SYSCONFIG */
typedef struct {
 	__REG32 AUTOIDLE									: 1;
 	__REG32 SOFTRESET									: 1;
 	__REG32 ENAWAKEUP									: 1;
 	__REG32 SIDLEMODE									: 2;
 	__REG32 													: 3;
 	__REG32 CLOCKACTIVITY							: 2;
 	__REG32 													:22;
} __mcspi_sysconfig_bits;

/* MCSPIn_SYSSTATUS */
typedef struct {
 	__REG32 RESETDONE									: 1;
 	__REG32 													:31;
} __mcspi_sysstatus_bits;

/* MCSPIn_IRQSTATUS */
typedef struct {
 	__REG32 TX0_EMPTY									: 1;
 	__REG32 TX0_UNDERFLOW							: 1;
 	__REG32 RX0_FULL									: 1;
 	__REG32 RX0_OVERFLOW							: 1;
 	__REG32 TX1_EMPTY									: 1;
 	__REG32 TX1_UNDERFLOW							: 1;
 	__REG32 RX1_FULL									: 1;
 	__REG32 													: 1;
 	__REG32 TX2_EMPTY									: 1;
 	__REG32 TX2_UNDERFLOW							: 1;
 	__REG32 RX2_FULL									: 1;
 	__REG32 													: 1;
 	__REG32 TX3_EMPTY									: 1;
 	__REG32 TX3_UNDERFLOW							: 1;
 	__REG32 RX3_FULL									: 1;
 	__REG32 													: 1;
 	__REG32 WKS												: 1;
 	__REG32 EOW												: 1;
 	__REG32 													:14;
} __mcspi_irqstatus_bits;

/* MCSPIn_IRQENABLE */
typedef struct {
 	__REG32 TX0_EMPTY_ENABLE					: 1;
 	__REG32 TX0_UNDERFLOW_ENABLE			: 1;
 	__REG32 RX0_FULL_ENABLE						: 1;
 	__REG32 RX0_OVERFLOW_ENABLE				: 1;
 	__REG32 TX1_EMPTY_ENABLE					: 1;
 	__REG32 TX1_UNDERFLOW_ENABLE			: 1;
 	__REG32 RX1_FULL_ENABLE						: 1;
 	__REG32 													: 1;
 	__REG32 TX2_EMPTY_ENABLE					: 1;
 	__REG32 TX2_UNDERFLOW_ENABLE			: 1;
 	__REG32 RX2_FULL_ENABLE						: 1;
 	__REG32 													: 1;
 	__REG32 TX3_EMPTY_ENABLE					: 1;
 	__REG32 TX3_UNDERFLOW_ENABLE			: 1;
 	__REG32 RX3_FULL_ENABLE						: 1;
 	__REG32 													: 1;
 	__REG32 WKE												: 1;
 	__REG32 EOWKE											: 1;
 	__REG32 													:14;
} __mcspi_irqenable_bits;

/* MCSPIn_WAKEUPENABLE */
typedef struct {
 	__REG32 WKEN											: 1;
 	__REG32 													:31;
} __mcspi_wakeupenable_bits;

/* MCSPIn_SYST */
typedef struct {
 	__REG32 SPIEN_0										: 1;
 	__REG32 SPIEN_1										: 1;
 	__REG32 PIEN_2										: 1;
 	__REG32 SPIEN_3										: 1;
 	__REG32 SPIDAT_0									: 1;
 	__REG32 SPIDAT_1									: 1;
 	__REG32 SPICLK										: 1;
 	__REG32 WAKD											: 1;
 	__REG32 SPIDATDIR0								: 1;
 	__REG32 SPIDATDIR1								: 1;
 	__REG32 SPIENDIR									: 1;
 	__REG32 SSB												: 1;
 	__REG32 													:20;
} __mcspi_syst_bits;

/* MCSPIn_MODULCTRL */
typedef struct {
 	__REG32 SINGLE										: 1;
 	__REG32 													: 1;
 	__REG32 MS												: 1;
 	__REG32 SYSTEM_TEST								: 1;
 	__REG32 													:28;
} __mcspi_modulctrl_bits;

/* MCSPIn_CHxCONF */
typedef struct {
 	__REG32 PHA												: 1;
 	__REG32 POL												: 1;
 	__REG32 CLKD											: 4;
 	__REG32 EPOL											: 1;
 	__REG32 WL												: 5;
 	__REG32 TRM												: 2;
 	__REG32 DMAW											: 1;
 	__REG32 DMAR											: 1;
 	__REG32 DPE0											: 1;
 	__REG32 DPE1											: 1;
 	__REG32 IS												: 1;
 	__REG32 TURBO											: 1;
 	__REG32 FORCE											: 1;
 	__REG32 													: 2;
 	__REG32 SBE												: 1;
 	__REG32 SBPOL											: 1;
 	__REG32 TCS												: 2;
 	__REG32 FFEW											: 1;
 	__REG32 FFER											: 1;
 	__REG32 CLKG											: 1;
 	__REG32 													: 2;
} __mcspi_chxconf_bits;

/* MCSPIn_CHxSTAT */
typedef struct {
 	__REG32 RXS												: 1;
 	__REG32 TXS												: 1;
 	__REG32 EOT												: 1;
 	__REG32 TXFFE											: 1;
 	__REG32 TXFFF											: 1;
 	__REG32 RXFFE											: 1;
 	__REG32 RXFFF											: 1;
 	__REG32 													:25;
} __mcspi_chxstat_bits;

/* MCSPIn_CHxCTRL */
typedef struct {
 	__REG32 EN												: 1;
 	__REG32 													: 7;
 	__REG32 EXTCLK										: 8;
 	__REG32 													:16;
} __mcspi_chxctrl_bits;

/* MCSPIn_XFERLEVEL */
typedef struct {
 	__REG32 AEL												: 6;
 	__REG32 													: 2;
 	__REG32 AFL												: 6;
 	__REG32 													: 2;
 	__REG32 WCNT											:16;
} __mcspi_xferlevel_bits;

/* HDQ_REVISION */
typedef struct {
 	__REG32 REVISION									: 8;
 	__REG32 													:24;
} __hdq_revision_bits;

/* HDQ_TX_DATA */
typedef struct {
 	__REG32 TX_DATA										: 8;
 	__REG32 													:24;
} __hdq_tx_data_bits;

/* HDQ_RX_DATA */
typedef struct {
 	__REG32 RX_DATA										: 8;
 	__REG32 													:24;
} __hdq_rx_data_bits;

/* HDQ_CTRL_STATUS */
typedef struct {
 	__REG32 MODE											: 1;
 	__REG32 DIR												: 1;
 	__REG32 INITIALIZATION						: 1;
 	__REG32 PRESENCEDETECT						: 1;
 	__REG32 GO												: 1;
 	__REG32 CLOCKENABLE								: 1;
 	__REG32 INTERRUPTMASK							: 1;
 	__REG32 _1_WIRE_SINGLE_BIT				: 1;
 	__REG32 													:24;
} __hdq_ctrl_status_bits;

/* HDQ_INT_STATUS */
typedef struct {
 	__REG32 TIMEOUT										: 1;
 	__REG32 RXCOMPLETE								: 1;
 	__REG32 TXCOMPLETE								: 1;
 	__REG32 													:29;
} __hdq_int_status_bits;

/* HDQ_SYSCONFIG */
typedef struct {
 	__REG32 AUTOIDLE									: 1;
 	__REG32 SOFTRESET									: 1;
 	__REG32 													:30;
} __hdq_sysconfig_bits;

/* HDQ_SYSSTATUS */
typedef struct {
 	__REG32 RESETDONE									: 1;
 	__REG32 													:31;
} __hdq_sysstatus_bits;

/* MCBSPLPn_SPCR2_REG */
typedef struct {
 	__REG32 XRST											: 1;
 	__REG32 XRDY											: 1;
 	__REG32 XEMPTY										: 1;
 	__REG32 XSYNCERR									: 1;
 	__REG32 XINTM											: 2;
 	__REG32 GRST											: 1;
 	__REG32 FRST											: 1;
 	__REG32 SOFT											: 1;
 	__REG32 FREE											: 1;
 	__REG32 													:22;
} __mcbsplp_spcr2_reg_bits;

/* MCBSPLPn_SPCR1_REG */
typedef struct {
 	__REG32 RRST											: 1;
 	__REG32 RRDY											: 1;
 	__REG32 RFULL											: 1;
 	__REG32 RSYNCERR									: 1;
 	__REG32 RINTM											: 2;
 	__REG32 													: 1;
 	__REG32 DXENA											: 1;
 	__REG32 													: 5;
 	__REG32 RJUST											: 2;
 	__REG32 ALB												: 1;
 	__REG32 													:16;
} __mcbsplp_spcr1_reg_bits;

/* MCBSPLPn_RCR2_REG */
typedef struct {
 	__REG32 RDATDLY										: 2;
 	__REG32    												: 1;
 	__REG32 RREVERSE									: 2;
 	__REG32 RWDLEN2										: 3;
 	__REG32 RFRLEN2										: 7;
 	__REG32 RPHASE										: 1;
 	__REG32 													:16;
} __mcbsplp_rcr2_reg_bits;

/* MCBSPLPn_RCR1_REG */
typedef struct {
 	__REG32    												: 5;
 	__REG32 RWDLEN1										: 3;
 	__REG32 RFRLEN1										: 7;
 	__REG32 													:17;
} __mcbsplp_rcr1_reg_bits;

/* MCBSPLPn_XCR2_REG */
typedef struct {
 	__REG32 XDATDLY										: 2;
 	__REG32    												: 1;
 	__REG32 XREVERSE									: 2;
 	__REG32 XWDLEN2										: 3;
 	__REG32 XFRLEN2										: 7;
 	__REG32 XPHASE										: 1;
 	__REG32 													:16;
} __mcbsplp_xcr2_reg_bits;

/* MCBSPLPn_XCR1_REG */
typedef struct {
 	__REG32    												: 5;
 	__REG32 XWDLEN1										: 3;
 	__REG32 XFRLEN1										: 7;
 	__REG32 													:17;
} __mcbsplp_xcr1_reg_bits;

/* MCBSPLPn_SRGR2_REG */
typedef struct {
 	__REG32 FPER											:12;
 	__REG32 FSGM											: 1;
 	__REG32 CLKSM											: 1;
 	__REG32 CLKSP											: 1;
 	__REG32 GSYNC											: 1;
 	__REG32 													:16;
} __mcbsplp_srgr2_reg_bits;

/* MCBSPLPn_SRGR1_REG */
typedef struct {
 	__REG32 CLKGDV										: 8;
 	__REG32 FWID											: 8;
 	__REG32 													:16;
} __mcbsplp_srgr1_reg_bits;

/* MCBSPLPn_MCR2_REG */
typedef struct {
 	__REG32 XMCM											: 2;
 	__REG32 													: 3;
 	__REG32 XPABLK										: 2;
 	__REG32 XPBBLK										: 2;
 	__REG32 XMCME											: 1;
 	__REG32 													:22;
} __mcbsplp_mcr2_reg_bits;

/* MCBSPLPn_MCR1_REG */
typedef struct {
 	__REG32 RMCM											: 1;
 	__REG32 													: 4;
 	__REG32 RPABLK										: 2;
 	__REG32 RPBBLK										: 2;
 	__REG32 RMCME											: 1;
 	__REG32 													:22;
} __mcbsplp_mcr1_reg_bits;

/* MCBSPLPn_RCERA_REG */
typedef struct {
 	__REG32 RCERA											:16;
 	__REG32 													:16;
} __mcbsplp_rcera_reg_bits;

/* MCBSPLPn_RCERB_REG */
typedef struct {
 	__REG32 RCERB											:16;
 	__REG32 													:16;
} __mcbsplp_rcerb_reg_bits;

/* MCBSPLPn_XCERA_REG */
typedef struct {
 	__REG32 XCERA											:16;
 	__REG32 													:16;
} __mcbsplp_xcera_reg_bits;

/* MCBSPLPn_XCERB_REG */
typedef struct {
 	__REG32 XCERB											:16;
 	__REG32 													:16;
} __mcbsplp_xcerb_reg_bits;

/* MCBSPLPn_PCR_REG */
typedef struct {
 	__REG32 CLKRP											: 1;
 	__REG32 CLKXP											: 1;
 	__REG32 FSRP											: 1;
 	__REG32 FSXP											: 1;
 	__REG32 DR_STAT										: 1;
 	__REG32 DX_STAT										: 1;
 	__REG32 CLKS_STAT									: 1;
 	__REG32 SCLKME										: 1;
 	__REG32 CLKRM											: 1;
 	__REG32 CLKXM											: 1;
 	__REG32 FSRM											: 1;
 	__REG32 FSXM											: 1;
 	__REG32 RIOEN											: 1;
 	__REG32 XIOEN											: 1;
 	__REG32 IDLE_EN										: 1;
 	__REG32 													:17;
} __mcbsplp_pcr_reg_bits;

/* MCBSPLPn_RCERC_REG */
typedef struct {
 	__REG32 RCERC											:16;
 	__REG32 													:16;
} __mcbsplp_rcerc_reg_bits;

/* MCBSPLPn_RCERD_REG */
typedef struct {
 	__REG32 RCERD											:16;
 	__REG32 													:16;
} __mcbsplp_rcerd_reg_bits;

/* MCBSPLPn_XCERC_REG */
typedef struct {
 	__REG32 XCERC											:16;
 	__REG32 													:16;
} __mcbsplp_xcerc_reg_bits;

/* MCBSPLPn_XCERD_REG */
typedef struct {
 	__REG32 XCERD											:16;
 	__REG32 													:16;
} __mcbsplp_xcerd_reg_bits;

/* MCBSPLPn_RCERE_REG */
typedef struct {
 	__REG32 RCERE											:16;
 	__REG32 													:16;
} __mcbsplp_rcere_reg_bits;

/* MCBSPLPn_RCERF_REG */
typedef struct {
 	__REG32 RCERF											:16;
 	__REG32 													:16;
} __mcbsplp_rcerf_reg_bits;

/* MCBSPLPn_XCERE_REG */
typedef struct {
 	__REG32 XCERE											:16;
 	__REG32 													:16;
} __mcbsplp_xcere_reg_bits;

/* MCBSPLPn_XCERF_REG */
typedef struct {
 	__REG32 XCERF											:16;
 	__REG32 													:16;
} __mcbsplp_xcerf_reg_bits;

/* MCBSPLPn_RCERG_REG */
typedef struct {
 	__REG32 RCERG											:16;
 	__REG32 													:16;
} __mcbsplp_rcerg_reg_bits;

/* MCBSPLPn_RCERH_REG */
typedef struct {
 	__REG32 RCERH											:16;
 	__REG32 													:16;
} __mcbsplp_rcerh_reg_bits;

/* MCBSPLPn_XCERG_REG */
typedef struct {
 	__REG32 XCERG											:16;
 	__REG32 													:16;
} __mcbsplp_xcerg_reg_bits;

/* MCBSPLPn_XCERH_REG */
typedef struct {
 	__REG32 XCERH											:16;
 	__REG32 													:16;
} __mcbsplp_xcerh_reg_bits;

/* MCBSPLPn_REV_REG */
typedef struct {
 	__REG32 REV												: 8;
 	__REG32 													:24;
} __mcbsplp_rev_reg_bits;

/* MCBSPLPn_SYSCONFIG_REG */
typedef struct {
 	__REG32 													: 1;
 	__REG32 SOFTRESET									: 1;
 	__REG32 ENAWAKEUP									: 1;
 	__REG32 SIDLEMODE									: 2;
 	__REG32 													: 3;
 	__REG32 CLOCKACTIVITY							: 2;
 	__REG32 													:22;
} __mcbsplp_sysconfig_reg_bits;

/* MCBSPLPn_THRSH2_REG */
typedef struct {
 	__REG32 XTHRESHOLD								:11;
 	__REG32 													:21;
} __mcbsplp_thrsh2_reg_bits;

/* MCBSPLPn_THRSH1_REG */
typedef struct {
 	__REG32 RTHRESHOLD								:11;
 	__REG32 													:21;
} __mcbsplp_thrsh1_reg_bits;

/* MCBSPLPn_IRQSTATUS_REG */
typedef struct {
 	__REG32 RSYNCERR									: 1;
 	__REG32 RFSR											: 1;
 	__REG32 REOF											: 1;
 	__REG32 RRDY											: 1;
 	__REG32 RUNDFLSTAT								: 1;
 	__REG32 ROVFLSTAT									: 1;
 	__REG32 													: 1;
 	__REG32 XSYNCERR									: 1;
 	__REG32 XFSX											: 1;
 	__REG32 XEOF											: 1;
 	__REG32 XRDY											: 1;
 	__REG32 XUNDFLSTAT								: 1;
 	__REG32 XOVFLSTAT									: 1;
 	__REG32 													: 1;
 	__REG32 XEMPTYEOF									: 1;
 	__REG32 													:17;
} __mcbsplp_irqstatus_reg_bits;

/* MCBSPLPn_IRQENABLE_REG */
typedef struct {
 	__REG32 RSYNCERREN								: 1;
 	__REG32 RFSREN										: 1;
 	__REG32 REOFEN										: 1;
 	__REG32 RRDYEN										: 1;
 	__REG32 RUNDFLEN									: 1;
 	__REG32 ROVFLEN										: 1;
 	__REG32 													: 1;
 	__REG32 XSYNCERREN								: 1;
 	__REG32 XFSXEN										: 1;
 	__REG32 XEOFEN										: 1;
 	__REG32 XRDYEN										: 1;
 	__REG32 XUNDFLEN									: 1;
 	__REG32 XOVFLEN										: 1;
 	__REG32 													: 1;
 	__REG32 XEMPTYEOFEN								: 1;
 	__REG32 													:17;
} __mcbsplp_irqenable_reg_bits;

/* MCBSPLPn_WAKEUPEN_REG */
typedef struct {
 	__REG32 RSYNCERREN								: 1;
 	__REG32 RFSREN										: 1;
 	__REG32 REOFEN										: 1;
 	__REG32 RRDYEN										: 1;
 	__REG32 													: 3;
 	__REG32 XSYNCERREN								: 1;
 	__REG32 XFSXEN										: 1;
 	__REG32 XEOFEN										: 1;
 	__REG32 XRDYEN										: 1;
 	__REG32 													: 3;
 	__REG32 XEMPTYEOFEN								: 1;
 	__REG32 													:17;
} __mcbsplp_wakeupen_reg_bits;

/* MCBSPLPn_XCCR_REG */
typedef struct {
 	__REG32 XDISABLE									: 1;
 	__REG32 													: 2;
 	__REG32 XDMAEN										: 1;
 	__REG32 													: 1;
 	__REG32 DLB												: 1;
 	__REG32 													: 5;
 	__REG32 XFULL_CYCLE								: 1;
 	__REG32 DXENDLY										: 2;
 	__REG32 PPCONNECT									: 1;
 	__REG32 EXTCLKGATE								: 1;
 	__REG32 													:16;
} __mcbsplp_xccr_reg_bits;

/* MCBSPLPn_RCCR_REG */
typedef struct {
 	__REG32 RDISABLE									: 1;
 	__REG32 													: 2;
 	__REG32 RDMAEN										: 1;
 	__REG32 													: 7;
 	__REG32 RFULL_CYCLE								: 1;
 	__REG32 													:20;
} __mcbsplp_rccr_reg_bits;

/* MCBSPLPn_XBUFFSTAT_REG */
typedef struct {
 	__REG32 XBUFFSTAT									:11;
 	__REG32 													:21;
} __mcbsplp_xbuffstat_reg_bits;

/* MCBSPLPn_RBUFFSTAT_REG */
typedef struct {
 	__REG32 RBUFFSTAT									:11;
 	__REG32 													:21;
} __mcbsplp_rbuffstat_reg_bits;

/* MCBSPLPn_SSELCR_REG */
typedef struct {
 	__REG32 ICH0ASSIGN								: 2;
 	__REG32 ICH1ASSIGN								: 2;
 	__REG32 OCH0ASSIGN								: 3;
 	__REG32 OCH1ASSIGN								: 3;
 	__REG32 SIDETONEEN								: 1;
 	__REG32 													:21;
} __mcbsplp_sselcr_reg_bits;

/* MCBSPLPn_STATUS_REG */
typedef struct {
 	__REG32 CLKMUXSTATUS							: 1;
 	__REG32 													:31;
} __mcbsplp_status_reg_bits;

/* STn_REV_REG */
typedef struct {
 	__REG32 REV												: 8;
 	__REG32 													:24;
} __st_rev_reg_bits;

/* STn_SYSCONFIG_REG */
typedef struct {
 	__REG32 AUTOIDLE									: 1;
 	__REG32 													:31;
} __st_sysconfig_reg_bits;

/* STn_IRQSTATUS_REG */
typedef struct {
 	__REG32 OVRRERROR									: 1;
 	__REG32 													:31;
} __st_irqstatus_reg_bits;

/* STn_IRQENABLE_REG */
typedef struct {
 	__REG32 OVRRERROREN								: 1;
 	__REG32 													:31;
} __st_irqenable_reg_bits;

/* STn_SGAINCR_REG */
typedef struct {
 	__REG32 CH0GAIN										:16;
 	__REG32 CH1GAIN										:16;
} __st_sgaincr_reg_bits;

/* STn_SFIRCR_REG */
typedef struct {
 	__REG32 FIRCOEFF									:16;
 	__REG32 													:16;
} __st_sfircr_reg_bits;

/* STn_SSELCR_REG */
typedef struct {
 	__REG32 SIDETONEEN								: 1;
 	__REG32 COEFFWREN									: 1;
 	__REG32 COEFFWRDONE								: 1;
 	__REG32 													:29;
} __st_sselcr_reg_bits;

/* MMCHSn_SYSCONFIG */
typedef struct {
 	__REG32 AUTOIDLE									: 1;
 	__REG32 SOFTRESET									: 1;
 	__REG32 ENAWAKEUP									: 1;
 	__REG32 SIDLEMODE									: 2;
 	__REG32 													: 3;
 	__REG32 CLOCKACTIVITY							: 2;
 	__REG32 													:22;
} __mmchs_sysconfig_bits;

/* MMCHSn_SYSSTATUS */
typedef struct {
 	__REG32 RESETDONE									: 1;
 	__REG32 													:31;
} __mmchs_sysstatus_bits;

/* MMCHSn_SYSTEST */
typedef struct {
 	__REG32 MCKD											: 1;
 	__REG32 CDIR											: 1;
 	__REG32 CDAT											: 1;
 	__REG32 DDIR											: 1;
 	__REG32 D0D												: 1;
 	__REG32 D1D 											: 1;
 	__REG32 D2D												: 1;
 	__REG32 D3D												: 1;
 	__REG32 D4D												: 1;
 	__REG32 D5D												: 1;
 	__REG32 D6D												: 1;
 	__REG32 D7D												: 1;
 	__REG32 SSB												: 1;
 	__REG32 WAKD											: 1;
 	__REG32 SDWP											: 1;
 	__REG32 SDCD											: 1;
 	__REG32 OBI												: 1;
 	__REG32 													:15;
} __mmchs_systest_bits;

/* MMCHSn_CON */
typedef struct {
 	__REG32 OD												: 1;
 	__REG32 INIT											: 1;
 	__REG32 HR												: 1;
 	__REG32 STR												: 1;
 	__REG32 MODE											: 1;
 	__REG32 DW8 											: 1;
 	__REG32 MIT												: 1;
 	__REG32 CDP												: 1;
 	__REG32 WPP												: 1;
 	__REG32 DVAL											: 2;
 	__REG32 CTPL											: 1;
 	__REG32 CEATA											: 1;
 	__REG32 OBIP											: 1;
 	__REG32 OBIE											: 1;
 	__REG32 PADEN											: 1;
 	__REG32 CLKEXTFREE								: 1;
 	__REG32 													:15;
} __mmchs_con_bits;

/* MMCHSn_PWCNT */
typedef struct {
 	__REG32 PWRCNT										:16;
 	__REG32       										:16;
} __mmchs_pwcnt_bits;

/* MMCHSn_BLK */
typedef struct {
 	__REG32 BLEN											:11;
 	__REG32 													: 5;
 	__REG32 NBLK											:16;
} __mmchs_blk_bits;

/* MMCHSn_CMD */
typedef struct {
 	__REG32 DE												: 1;
 	__REG32 BCE												: 1;
 	__REG32 ACEN											: 1;
 	__REG32 													: 1;
 	__REG32 DDIR											: 1;
 	__REG32 MSBS											: 1;
 	__REG32 													:10;
 	__REG32 RSP_TYPE									: 2;
 	__REG32 													: 1;
 	__REG32 CCCE											: 1;
 	__REG32 CICE											: 1;
 	__REG32 DP												: 1;
 	__REG32 CMD_TYPE									: 2;
 	__REG32 INDX											: 6;
 	__REG32 													: 2;
} __mmchs_cmd_bits;

/* MMCHSn_RSP10 */
typedef struct {
 	__REG32 RSP0											:16;
 	__REG32 RSP1											:16;
} __mmchs_rsp10_bits;

/* MMCHSn_RSP32 */
typedef struct {
 	__REG32 RSP2											:16;
 	__REG32 RSP3											:16;
} __mmchs_rsp32_bits;

/* MMCHSn_RSP54 */
typedef struct {
 	__REG32 RSP4											:16;
 	__REG32 RSP5											:16;
} __mmchs_rsp54_bits;

/* MMCHSn_RSP76 */
typedef struct {
 	__REG32 RSP6											:16;
 	__REG32 RSP7											:16;
} __mmchs_rsp76_bits;

/* MMCHSn_PSTATE */
typedef struct {
 	__REG32 CMDI											: 1;
 	__REG32 DATI											: 1;
 	__REG32 DLA												: 1;
 	__REG32 													: 5;
 	__REG32 WTA												: 1;
 	__REG32 RTA												: 1;
 	__REG32 BWE												: 1;
 	__REG32 BRE												: 1;
 	__REG32 													: 8;
 	__REG32 DLEV											: 4;
 	__REG32 CLEV											: 1;
 	__REG32 													: 7;
} __mmchs_pstate_bits;

/* MMCHSn_HCTL */
typedef struct {
 	__REG32 													: 1;
 	__REG32 DTW												: 1;
 	__REG32 													: 6;
 	__REG32 SDBP											: 1;
 	__REG32 SDVS											: 3;
 	__REG32 													: 4;
 	__REG32 SBGR											: 1;
 	__REG32 CR												: 1;
 	__REG32 RWC												: 1;
 	__REG32 IBG												: 1;
 	__REG32 													: 4;
 	__REG32 IWE												: 1;
 	__REG32 INS												: 1;
 	__REG32 REM												: 1;
 	__REG32 OBWE											: 1;
 	__REG32 													: 4;
} __mmchs_hctl_bits;

/* MMCHSn_SYSCTL */
typedef struct {
 	__REG32 ICE												: 1;
 	__REG32 ICS												: 1;
 	__REG32 CEN												: 1;
 	__REG32 													: 3;
 	__REG32 CLKD											:10;
 	__REG32 DTO												: 4;
 	__REG32 													: 4;
 	__REG32 SRA												: 1;
 	__REG32 SRC												: 1;
 	__REG32 SRD												: 1;
 	__REG32 													: 5;
} __mmchs_sysctl_bits;

/* MMCHSn_STAT */
typedef struct {
 	__REG32 CC												: 1;
 	__REG32 TC												: 1;
 	__REG32 BGE												: 1;
 	__REG32 													: 1;
 	__REG32 BWR												: 1;
 	__REG32 BRR												: 1;
 	__REG32 													: 2;
 	__REG32 CIRQ											: 1;
 	__REG32 OBI												: 1;
 	__REG32 													: 5;
 	__REG32 ERRI											: 1;
 	__REG32 CTO												: 1;
 	__REG32 CCRC											: 1;
 	__REG32 CEB												: 1;
 	__REG32 CIE												: 1;
 	__REG32 DTO												: 1;
 	__REG32 DCRC											: 1;
 	__REG32 DEB												: 1;
 	__REG32 													: 1;
 	__REG32 ACE												: 1;
 	__REG32 													: 3;
 	__REG32 CERR											: 1;
 	__REG32 BADA											: 1;
 	__REG32 													: 2;
} __mmchs_stat_bits;

/* MMCHSn_IE */
typedef struct {
 	__REG32 CC_ENABLE									: 1;
 	__REG32 TC_ENABLE									: 1;
 	__REG32 BGE_ENABLE								: 1;
 	__REG32 													: 1;
 	__REG32 BWR_ENABLE								: 1;
 	__REG32 BRR_ENABLE								: 1;
 	__REG32 													: 2;
 	__REG32 CIRQ_ENABLE								: 1;
 	__REG32 OBI_ENABLE								: 1;
 	__REG32 													: 6;
 	__REG32 CTO_ENABLE								: 1;
 	__REG32 CCRC_ENABLE								: 1;
 	__REG32 CEB_ENABLE								: 1;
 	__REG32 CIE_ENABLE								: 1;
 	__REG32 DTO_ENABLE								: 1;
 	__REG32 DCRC_ENABLE								: 1;
 	__REG32 DEB_ENABLE								: 1;
 	__REG32 													: 1;
 	__REG32 ACE_ENABLE								: 1;
 	__REG32 													: 3;
 	__REG32 CERR_ENABLE								: 1;
 	__REG32 BADA_ENABLE								: 1;
 	__REG32 													: 2;
} __mmchs_ie_bits;

/* MMCHSn_ISE */
typedef struct {
 	__REG32 CC_SIGEN									: 1;
 	__REG32 TC_SIGEN									: 1;
 	__REG32 BGE_SIGEN									: 1;
 	__REG32 													: 1;
 	__REG32 BWR_SIGEN									: 1;
 	__REG32 BRR_SIGEN									: 1;
 	__REG32 													: 2;
 	__REG32 CIRQ_SIGEN								: 1;
 	__REG32 OBI_SIGEN									: 1;
 	__REG32 													: 6;
 	__REG32 CTO_SIGEN									: 1;
 	__REG32 CCRC_SIGEN								: 1;
 	__REG32 CEB_SIGEN									: 1;
 	__REG32 CIE_SIGEN									: 1;
 	__REG32 DTO_SIGEN									: 1;
 	__REG32 DCRC_SIGEN								: 1;
 	__REG32 DEB_SIGEN									: 1;
 	__REG32 													: 1;
 	__REG32 ACE_SIGEN									: 1;
 	__REG32 													: 3;
 	__REG32 CERR_SIGEN								: 1;
 	__REG32 BADA_SIGEN								: 1;
 	__REG32 													: 2;
} __mmchs_ise_bits;

/* MMCHSn_AC12 */
typedef struct {
 	__REG32 ACNE											: 1;
 	__REG32 ACTO											: 1;
 	__REG32 ACCE											: 1;
 	__REG32 ACEB											: 1;
 	__REG32 ACIE											: 1;
 	__REG32 													: 2;
 	__REG32 CNI												: 1;
 	__REG32 													:24;
} __mmchs_ac12_bits;

/* MMCHSn_CAPA */
typedef struct {
 	__REG32 TCF												: 6;
 	__REG32 													: 1;
 	__REG32 TCU												: 1;
 	__REG32 BCF												: 6;
 	__REG32 													: 2;
 	__REG32 MBL												: 2;
 	__REG32 													: 3;
 	__REG32 HSS												: 1;
 	__REG32 DS												: 1;
 	__REG32 SRS												: 1;
 	__REG32 VS33											: 1;
 	__REG32 VS30											: 1;
 	__REG32 VS18											: 1;
 	__REG32 													: 5;
} __mmchs_capa_bits;

/* MMCHSn_CUR_CAPA */
typedef struct {
 	__REG32 CUR_3V3										: 8;
 	__REG32 CUR_3V0										: 8;
 	__REG32 CUR_1V8										: 8;
 	__REG32 													: 8;
} __mmchs_cur_capa_bits;

/* MMCHSn_REV */
typedef struct {
 	__REG32 SIS												: 1;
 	__REG32 													:15;
 	__REG32 SREV											: 8;
 	__REG32 VREV											: 8;
} __mmchs_rev_bits;

/* GPIOn_REVISION */
typedef struct {
 	__REG32 GPIOREVISION							: 8;
 	__REG32 													:24;
} __gpio_revision_bits;

/* GPIOn_SYSCONFIG */
typedef struct {
 	__REG32 AUTOIDLE									: 1;
 	__REG32 SOFTRESET									: 1;
 	__REG32 ENAWAKEUP									: 1;
 	__REG32 IDLEMODE									: 2;
 	__REG32 													:27;
} __gpio_sysconfig_bits;

/* GPIOn_SYSSTATUS */
typedef struct {
 	__REG32 RESETDONE									: 1;
 	__REG32 													:31;
} __gpio_sysstatus_bits;

/* GPIOn_IRQSTATUS */
typedef struct {
 	__REG32 IRQSTATUS0								: 1;
 	__REG32 IRQSTATUS1								: 1;
 	__REG32 IRQSTATUS2								: 1;
 	__REG32 IRQSTATUS3								: 1;
 	__REG32 IRQSTATUS4								: 1;
 	__REG32 IRQSTATUS5								: 1;
 	__REG32 IRQSTATUS6								: 1;
 	__REG32 IRQSTATUS7								: 1;
 	__REG32 IRQSTATUS8								: 1;
 	__REG32 IRQSTATUS9								: 1;
 	__REG32 IRQSTATUS10								: 1;
 	__REG32 IRQSTATUS11								: 1;
 	__REG32 IRQSTATUS12								: 1;
 	__REG32 IRQSTATUS13								: 1;
 	__REG32 IRQSTATUS14								: 1;
 	__REG32 IRQSTATUS15								: 1;
 	__REG32 IRQSTATUS16								: 1;
 	__REG32 IRQSTATUS17								: 1;
 	__REG32 IRQSTATUS18								: 1;
 	__REG32 IRQSTATUS19								: 1;
 	__REG32 IRQSTATUS20								: 1;
 	__REG32 IRQSTATUS21								: 1;
 	__REG32 IRQSTATUS22								: 1;
 	__REG32 IRQSTATUS23								: 1;
 	__REG32 IRQSTATUS24								: 1;
 	__REG32 IRQSTATUS25								: 1;
 	__REG32 IRQSTATUS26								: 1;
 	__REG32 IRQSTATUS27								: 1;
 	__REG32 IRQSTATUS28								: 1;
 	__REG32 IRQSTATUS29								: 1;
 	__REG32 IRQSTATUS30								: 1;
 	__REG32 IRQSTATUS31								: 1;
} __gpio_irqstatus_bits;

/* GPIOn_IRQENABLE */
typedef struct {
 	__REG32 IRQENABLE0								: 1;
 	__REG32 IRQENABLE1								: 1;
 	__REG32 IRQENABLE2								: 1;
 	__REG32 IRQENABLE3								: 1;
 	__REG32 IRQENABLE4								: 1;
 	__REG32 IRQENABLE5								: 1;
 	__REG32 IRQENABLE6								: 1;
 	__REG32 IRQENABLE7								: 1;
 	__REG32 IRQENABLE8								: 1;
 	__REG32 IRQENABLE9								: 1;
 	__REG32 IRQENABLE10								: 1;
 	__REG32 IRQENABLE11								: 1;
 	__REG32 IRQENABLE12								: 1;
 	__REG32 IRQENABLE13								: 1;
 	__REG32 IRQENABLE14								: 1;
 	__REG32 IRQENABLE15								: 1;
 	__REG32 IRQENABLE16								: 1;
 	__REG32 IRQENABLE17								: 1;
 	__REG32 IRQENABLE18								: 1;
 	__REG32 IRQENABLE19								: 1;
 	__REG32 IRQENABLE20								: 1;
 	__REG32 IRQENABLE21								: 1;
 	__REG32 IRQENABLE22								: 1;
 	__REG32 IRQENABLE23								: 1;
 	__REG32 IRQENABLE24								: 1;
 	__REG32 IRQENABLE25								: 1;
 	__REG32 IRQENABLE26								: 1;
 	__REG32 IRQENABLE27								: 1;
 	__REG32 IRQENABLE28								: 1;
 	__REG32 IRQENABLE29								: 1;
 	__REG32 IRQENABLE30								: 1;
 	__REG32 IRQENABLE31								: 1;
} __gpio_irqenable_bits;

/* GPIOn_WAKEUPENABLE */
typedef struct {
 	__REG32 WAKEUPEN0								: 1;
 	__REG32 WAKEUPEN1								: 1;
 	__REG32 WAKEUPEN2								: 1;
 	__REG32 WAKEUPEN3								: 1;
 	__REG32 WAKEUPEN4								: 1;
 	__REG32 WAKEUPEN5								: 1;
 	__REG32 WAKEUPEN6								: 1;
 	__REG32 WAKEUPEN7								: 1;
 	__REG32 WAKEUPEN8								: 1;
 	__REG32 WAKEUPEN9								: 1;
 	__REG32 WAKEUPEN10							: 1;
 	__REG32 WAKEUPEN11							: 1;
 	__REG32 WAKEUPEN12							: 1;
 	__REG32 WAKEUPEN13							: 1;
 	__REG32 WAKEUPEN14							: 1;
 	__REG32 WAKEUPEN15							: 1;
 	__REG32 WAKEUPEN16							: 1;
 	__REG32 WAKEUPEN17							: 1;
 	__REG32 WAKEUPEN18							: 1;
 	__REG32 WAKEUPEN19							: 1;
 	__REG32 WAKEUPEN20							: 1;
 	__REG32 WAKEUPEN21							: 1;
 	__REG32 WAKEUPEN22							: 1;
 	__REG32 WAKEUPEN23							: 1;
 	__REG32 WAKEUPEN24							: 1;
 	__REG32 WAKEUPEN25							: 1;
 	__REG32 WAKEUPEN26							: 1;
 	__REG32 WAKEUPEN27							: 1;
 	__REG32 WAKEUPEN28							: 1;
 	__REG32 WAKEUPEN29							: 1;
 	__REG32 WAKEUPEN30							: 1;
 	__REG32 WAKEUPEN31							: 1;
} __gpio_wakeupenable_bits;

/* GPIOn_CTRL */
typedef struct {
 	__REG32 DISABLEMODULE						: 1;
 	__REG32 GATINGRATIO							: 2;
 	__REG32 												:29;
} __gpio_ctrl_bits;

/* GPIOn_OE */
typedef struct {
 	__REG32 OUTPUTEN0								: 1;
 	__REG32 OUTPUTEN1								: 1;
 	__REG32 OUTPUTEN2								: 1;
 	__REG32 OUTPUTEN3								: 1;
 	__REG32 OUTPUTEN4								: 1;
 	__REG32 OUTPUTEN5								: 1;
 	__REG32 OUTPUTEN6								: 1;
 	__REG32 OUTPUTEN7								: 1;
 	__REG32 OUTPUTEN8								: 1;
 	__REG32 OUTPUTEN9								: 1;
 	__REG32 OUTPUTEN10							: 1;
 	__REG32 OUTPUTEN11							: 1;
 	__REG32 OUTPUTEN12							: 1;
 	__REG32 OUTPUTEN13							: 1;
 	__REG32 OUTPUTEN14							: 1;
 	__REG32 OUTPUTEN15							: 1;
 	__REG32 OUTPUTEN16							: 1;
 	__REG32 OUTPUTEN17							: 1;
 	__REG32 OUTPUTEN18							: 1;
 	__REG32 OUTPUTEN19							: 1;
 	__REG32 OUTPUTEN20							: 1;
 	__REG32 OUTPUTEN21							: 1;
 	__REG32 OUTPUTEN22							: 1;
 	__REG32 OUTPUTEN23							: 1;
 	__REG32 OUTPUTEN24							: 1;
 	__REG32 OUTPUTEN25							: 1;
 	__REG32 OUTPUTEN26							: 1;
 	__REG32 OUTPUTEN27							: 1;
 	__REG32 OUTPUTEN28							: 1;
 	__REG32 OUTPUTEN29							: 1;
 	__REG32 OUTPUTEN30							: 1;
 	__REG32 OUTPUTEN31							: 1;
} __gpio_oe_bits;

/* GPIOn_DATAIN */
typedef struct {
 	__REG32 DATAINPUT0							: 1;
 	__REG32 DATAINPUT1							: 1;
 	__REG32 DATAINPUT2							: 1;
 	__REG32 DATAINPUT3							: 1;
 	__REG32 DATAINPUT4							: 1;
 	__REG32 DATAINPUT5							: 1;
 	__REG32 DATAINPUT6							: 1;
 	__REG32 DATAINPUT7							: 1;
 	__REG32 DATAINPUT8							: 1;
 	__REG32 DATAINPUT9							: 1;
 	__REG32 DATAINPUT10							: 1;
 	__REG32 DATAINPUT11							: 1;
 	__REG32 DATAINPUT12							: 1;
 	__REG32 DATAINPUT13							: 1;
 	__REG32 DATAINPUT14							: 1;
 	__REG32 DATAINPUT15							: 1;
 	__REG32 DATAINPUT16							: 1;
 	__REG32 DATAINPUT17							: 1;
 	__REG32 DATAINPUT18							: 1;
 	__REG32 DATAINPUT19							: 1;
 	__REG32 DATAINPUT20							: 1;
 	__REG32 DATAINPUT21							: 1;
 	__REG32 DATAINPUT22							: 1;
 	__REG32 DATAINPUT23							: 1;
 	__REG32 DATAINPUT24							: 1;
 	__REG32 DATAINPUT25							: 1;
 	__REG32 DATAINPUT26							: 1;
 	__REG32 DATAINPUT27							: 1;
 	__REG32 DATAINPUT28							: 1;
 	__REG32 DATAINPUT29							: 1;
 	__REG32 DATAINPUT30							: 1;
 	__REG32 DATAINPUT31							: 1;
} __gpio_datain_bits;

/* GPIOn_DATAOUT */
typedef struct {
 	__REG32 DATAOUTPUT0							: 1;
 	__REG32 DATAOUTPUT1							: 1;
 	__REG32 DATAOUTPUT2							: 1;
 	__REG32 DATAOUTPUT3							: 1;
 	__REG32 DATAOUTPUT4							: 1;
 	__REG32 DATAOUTPUT5							: 1;
 	__REG32 DATAOUTPUT6							: 1;
 	__REG32 DATAOUTPUT7							: 1;
 	__REG32 DATAOUTPUT8							: 1;
 	__REG32 DATAOUTPUT9							: 1;
 	__REG32 DATAOUTPUT10						: 1;
 	__REG32 DATAOUTPUT11						: 1;
 	__REG32 DATAOUTPUT12						: 1;
 	__REG32 DATAOUTPUT13						: 1;
 	__REG32 DATAOUTPUT14						: 1;
 	__REG32 DATAOUTPUT15						: 1;
 	__REG32 DATAOUTPUT16						: 1;
 	__REG32 DATAOUTPUT17						: 1;
 	__REG32 DATAOUTPUT18						: 1;
 	__REG32 DATAOUTPUT19						: 1;
 	__REG32 DATAOUTPUT20						: 1;
 	__REG32 DATAOUTPUT21						: 1;
 	__REG32 DATAOUTPUT22						: 1;
 	__REG32 DATAOUTPUT23						: 1;
 	__REG32 DATAOUTPUT24						: 1;
 	__REG32 DATAOUTPUT25						: 1;
 	__REG32 DATAOUTPUT26						: 1;
 	__REG32 DATAOUTPUT27						: 1;
 	__REG32 DATAOUTPUT28						: 1;
 	__REG32 DATAOUTPUT29						: 1;
 	__REG32 DATAOUTPUT30						: 1;
 	__REG32 DATAOUTPUT31						: 1;
} __gpio_dataout_bits;

/* GPIO1_LEVELDETECT0 */
typedef struct {
 	__REG32 LOWLEVEL0							: 1;
 	__REG32 LOWLEVEL1							: 1;
 	__REG32 LOWLEVEL2							: 1;
 	__REG32 LOWLEVEL3							: 1;
 	__REG32 LOWLEVEL4							: 1;
 	__REG32 LOWLEVEL5							: 1;
 	__REG32 LOWLEVEL6							: 1;
 	__REG32 LOWLEVEL7							: 1;
 	__REG32 LOWLEVEL8							: 1;
 	__REG32 LOWLEVEL9							: 1;
 	__REG32 LOWLEVEL10						: 1;
 	__REG32 LOWLEVEL11						: 1;
 	__REG32 LOWLEVEL12						: 1;
 	__REG32 LOWLEVEL13						: 1;
 	__REG32 LOWLEVEL14						: 1;
 	__REG32 LOWLEVEL15						: 1;
 	__REG32 LOWLEVEL16						: 1;
 	__REG32 LOWLEVEL17						: 1;
 	__REG32 LOWLEVEL18						: 1;
 	__REG32 LOWLEVEL19						: 1;
 	__REG32 LOWLEVEL20						: 1;
 	__REG32 LOWLEVEL21						: 1;
 	__REG32 LOWLEVEL22						: 1;
 	__REG32 LOWLEVEL23						: 1;
 	__REG32 LOWLEVEL24						: 1;
 	__REG32 LOWLEVEL25						: 1;
 	__REG32 LOWLEVEL26						: 1;
 	__REG32 LOWLEVEL27						: 1;
 	__REG32 LOWLEVEL28						: 1;
 	__REG32 LOWLEVEL29						: 1;
 	__REG32 LOWLEVEL30						: 1;
 	__REG32 LOWLEVEL31						: 1;
} __gpio_leveldetect0_bits;

/* GPIO1_LEVELDETECT1 */
typedef struct {
 	__REG32 HIGHLEVEL0						: 1;
 	__REG32 HIGHLEVEL1						: 1;
 	__REG32 HIGHLEVEL2						: 1;
 	__REG32 HIGHLEVEL3						: 1;
 	__REG32 HIGHLEVEL4						: 1;
 	__REG32 HIGHLEVEL5						: 1;
 	__REG32 HIGHLEVEL6						: 1;
 	__REG32 HIGHLEVEL7						: 1;
 	__REG32 HIGHLEVEL8						: 1;
 	__REG32 HIGHLEVEL9						: 1;
 	__REG32 HIGHLEVEL10						: 1;
 	__REG32 HIGHLEVEL11						: 1;
 	__REG32 HIGHLEVEL12						: 1;
 	__REG32 HIGHLEVEL13						: 1;
 	__REG32 HIGHLEVEL14						: 1;
 	__REG32 HIGHLEVEL15						: 1;
 	__REG32 HIGHLEVEL16						: 1;
 	__REG32 HIGHLEVEL17						: 1;
 	__REG32 HIGHLEVEL18						: 1;
 	__REG32 HIGHLEVEL19						: 1;
 	__REG32 HIGHLEVEL20						: 1;
 	__REG32 HIGHLEVEL21						: 1;
 	__REG32 HIGHLEVEL22						: 1;
 	__REG32 HIGHLEVEL23						: 1;
 	__REG32 HIGHLEVEL24						: 1;
 	__REG32 HIGHLEVEL25						: 1;
 	__REG32 HIGHLEVEL26						: 1;
 	__REG32 HIGHLEVEL27						: 1;
 	__REG32 HIGHLEVEL28						: 1;
 	__REG32 HIGHLEVEL29						: 1;
 	__REG32 HIGHLEVEL30						: 1;
 	__REG32 HIGHLEVEL31						: 1;
} __gpio_leveldetect1_bits;

/* GPIOn_RISINGDETECT */
typedef struct {
 	__REG32 RISINGEDGE0						: 1;
 	__REG32 RISINGEDGE1						: 1;
 	__REG32 RISINGEDGE2						: 1;
 	__REG32 RISINGEDGE3						: 1;
 	__REG32 RISINGEDGE4						: 1;
 	__REG32 RISINGEDGE5						: 1;
 	__REG32 RISINGEDGE6						: 1;
 	__REG32 RISINGEDGE7						: 1;
 	__REG32 RISINGEDGE8						: 1;
 	__REG32 RISINGEDGE9						: 1;
 	__REG32 RISINGEDGE10					: 1;
 	__REG32 RISINGEDGE11					: 1;
 	__REG32 RISINGEDGE12					: 1;
 	__REG32 RISINGEDGE13					: 1;
 	__REG32 RISINGEDGE14					: 1;
 	__REG32 RISINGEDGE15					: 1;
 	__REG32 RISINGEDGE16					: 1;
 	__REG32 RISINGEDGE17					: 1;
 	__REG32 RISINGEDGE18					: 1;
 	__REG32 RISINGEDGE19					: 1;
 	__REG32 RISINGEDGE20					: 1;
 	__REG32 RISINGEDGE21					: 1;
 	__REG32 RISINGEDGE22					: 1;
 	__REG32 RISINGEDGE23					: 1;
 	__REG32 RISINGEDGE24					: 1;
 	__REG32 RISINGEDGE25					: 1;
 	__REG32 RISINGEDGE26					: 1;
 	__REG32 RISINGEDGE27					: 1;
 	__REG32 RISINGEDGE28					: 1;
 	__REG32 RISINGEDGE29					: 1;
 	__REG32 RISINGEDGE30					: 1;
 	__REG32 RISINGEDGE31					: 1;
} __gpio_risingdetect_bits;

/* GPIOn_FALLINGDETECT */
typedef struct {
 	__REG32 FALLINGEDGE0					: 1;
 	__REG32 FALLINGEDGE1					: 1;
 	__REG32 FALLINGEDGE2					: 1;
 	__REG32 FALLINGEDGE3					: 1;
 	__REG32 FALLINGEDGE4					: 1;
 	__REG32 FALLINGEDGE5					: 1;
 	__REG32 FALLINGEDGE6					: 1;
 	__REG32 FALLINGEDGE7					: 1;
 	__REG32 FALLINGEDGE8					: 1;
 	__REG32 FALLINGEDGE9					: 1;
 	__REG32 FALLINGEDGE10					: 1;
 	__REG32 FALLINGEDGE11					: 1;
 	__REG32 FALLINGEDGE12					: 1;
 	__REG32 FALLINGEDGE13					: 1;
 	__REG32 FALLINGEDGE14					: 1;
 	__REG32 FALLINGEDGE15					: 1;
 	__REG32 FALLINGEDGE16					: 1;
 	__REG32 FALLINGEDGE17					: 1;
 	__REG32 FALLINGEDGE18					: 1;
 	__REG32 FALLINGEDGE19					: 1;
 	__REG32 FALLINGEDGE20					: 1;
 	__REG32 FALLINGEDGE21					: 1;
 	__REG32 FALLINGEDGE22					: 1;
 	__REG32 FALLINGEDGE23					: 1;
 	__REG32 FALLINGEDGE24					: 1;
 	__REG32 FALLINGEDGE25					: 1;
 	__REG32 FALLINGEDGE26					: 1;
 	__REG32 FALLINGEDGE27					: 1;
 	__REG32 FALLINGEDGE28					: 1;
 	__REG32 FALLINGEDGE29					: 1;
 	__REG32 FALLINGEDGE30					: 1;
 	__REG32 FALLINGEDGE31					: 1;
} __gpio_fallingdetect_bits;

/* GPIOn_DEBOUNCENABLE */
typedef struct {
 	__REG32 DEBOUNCEEN0						: 1;
 	__REG32 DEBOUNCEEN1						: 1;
 	__REG32 DEBOUNCEEN2						: 1;
 	__REG32 DEBOUNCEEN3						: 1;
 	__REG32 DEBOUNCEEN4						: 1;
 	__REG32 DEBOUNCEEN5						: 1;
 	__REG32 DEBOUNCEEN6						: 1;
 	__REG32 DEBOUNCEEN7						: 1;
 	__REG32 DEBOUNCEEN8						: 1;
 	__REG32 DEBOUNCEEN9						: 1;
 	__REG32 DEBOUNCEEN10					: 1;
 	__REG32 DEBOUNCEEN11					: 1;
 	__REG32 DEBOUNCEEN12					: 1;
 	__REG32 DEBOUNCEEN13					: 1;
 	__REG32 DEBOUNCEEN14					: 1;
 	__REG32 DEBOUNCEEN15					: 1;
 	__REG32 DEBOUNCEEN16					: 1;
 	__REG32 DEBOUNCEEN17					: 1;
 	__REG32 DEBOUNCEEN18					: 1;
 	__REG32 DEBOUNCEEN19					: 1;
 	__REG32 DEBOUNCEEN20					: 1;
 	__REG32 DEBOUNCEEN21					: 1;
 	__REG32 DEBOUNCEEN22					: 1;
 	__REG32 DEBOUNCEEN23					: 1;
 	__REG32 DEBOUNCEEN24					: 1;
 	__REG32 DEBOUNCEEN25					: 1;
 	__REG32 DEBOUNCEEN26					: 1;
 	__REG32 DEBOUNCEEN27					: 1;
 	__REG32 DEBOUNCEEN28					: 1;
 	__REG32 DEBOUNCEEN29					: 1;
 	__REG32 DEBOUNCEEN30					: 1;
 	__REG32 DEBOUNCEEN31					: 1;
} __gpio_debouncenable_bits;

/* GPIOn_DEBOUNCINGTIME */
typedef struct {
 	__REG32 DEBOUNCEVAL						: 8;
 	__REG32 											:24;
} __gpio_debouncingtime_bits;

/* GPIOn_CLEARIRQENABLE */
typedef struct {
 	__REG32 CLEARIRQEN0						: 1;
 	__REG32 CLEARIRQEN1						: 1;
 	__REG32 CLEARIRQEN2						: 1;
 	__REG32 CLEARIRQEN3						: 1;
 	__REG32 CLEARIRQEN4						: 1;
 	__REG32 CLEARIRQEN5						: 1;
 	__REG32 CLEARIRQEN6						: 1;
 	__REG32 CLEARIRQEN7						: 1;
 	__REG32 CLEARIRQEN8						: 1;
 	__REG32 CLEARIRQEN9						: 1;
 	__REG32 CLEARIRQEN10					: 1;
 	__REG32 CLEARIRQEN11					: 1;
 	__REG32 CLEARIRQEN12					: 1;
 	__REG32 CLEARIRQEN13					: 1;
 	__REG32 CLEARIRQEN14					: 1;
 	__REG32 CLEARIRQEN15					: 1;
 	__REG32 CLEARIRQEN16					: 1;
 	__REG32 CLEARIRQEN17					: 1;
 	__REG32 CLEARIRQEN18					: 1;
 	__REG32 CLEARIRQEN19					: 1;
 	__REG32 CLEARIRQEN20					: 1;
 	__REG32 CLEARIRQEN21					: 1;
 	__REG32 CLEARIRQEN22					: 1;
 	__REG32 CLEARIRQEN23					: 1;
 	__REG32 CLEARIRQEN24					: 1;
 	__REG32 CLEARIRQEN25					: 1;
 	__REG32 CLEARIRQEN26					: 1;
 	__REG32 CLEARIRQEN27					: 1;
 	__REG32 CLEARIRQEN28					: 1;
 	__REG32 CLEARIRQEN29					: 1;
 	__REG32 CLEARIRQEN30					: 1;
 	__REG32 CLEARIRQEN31					: 1;
} __gpio_clearirqenable_bits;

/* GPIOn_SETIRQENABLE */
typedef struct {
 	__REG32 SETIRQEN0							: 1;
 	__REG32 SETIRQEN1							: 1;
 	__REG32 SETIRQEN2							: 1;
 	__REG32 SETIRQEN3							: 1;
 	__REG32 SETIRQEN4							: 1;
 	__REG32 SETIRQEN5							: 1;
 	__REG32 SETIRQEN6							: 1;
 	__REG32 SETIRQEN7							: 1;
 	__REG32 SETIRQEN8							: 1;
 	__REG32 SETIRQEN9							: 1;
 	__REG32 SETIRQEN10						: 1;
 	__REG32 SETIRQEN11						: 1;
 	__REG32 SETIRQEN12						: 1;
 	__REG32 SETIRQEN13						: 1;
 	__REG32 SETIRQEN14						: 1;
 	__REG32 SETIRQEN15						: 1;
 	__REG32 SETIRQEN16						: 1;
 	__REG32 SETIRQEN17						: 1;
 	__REG32 SETIRQEN18						: 1;
 	__REG32 SETIRQEN19						: 1;
 	__REG32 SETIRQEN20						: 1;
 	__REG32 SETIRQEN21						: 1;
 	__REG32 SETIRQEN22						: 1;
 	__REG32 SETIRQEN23						: 1;
 	__REG32 SETIRQEN24						: 1;
 	__REG32 SETIRQEN25						: 1;
 	__REG32 SETIRQEN26						: 1;
 	__REG32 SETIRQEN27						: 1;
 	__REG32 SETIRQEN28						: 1;
 	__REG32 SETIRQEN29						: 1;
 	__REG32 SETIRQEN30						: 1;
 	__REG32 SETIRQEN31						: 1;
} __gpio_setirqenable_bits;

/* GPIOn_CLEARWKUENA */
typedef struct {
 	__REG32 CLEARWAKEUPEN0						: 1;
 	__REG32 CLEARWAKEUPEN1						: 1;
 	__REG32 CLEARWAKEUPEN2						: 1;
 	__REG32 CLEARWAKEUPEN3						: 1;
 	__REG32 CLEARWAKEUPEN4						: 1;
 	__REG32 CLEARWAKEUPEN5						: 1;
 	__REG32 CLEARWAKEUPEN6						: 1;
 	__REG32 CLEARWAKEUPEN7						: 1;
 	__REG32 CLEARWAKEUPEN8						: 1;
 	__REG32 CLEARWAKEUPEN9						: 1;
 	__REG32 CLEARWAKEUPEN10						: 1;
 	__REG32 CLEARWAKEUPEN11						: 1;
 	__REG32 CLEARWAKEUPEN12						: 1;
 	__REG32 CLEARWAKEUPEN13						: 1;
 	__REG32 CLEARWAKEUPEN14						: 1;
 	__REG32 CLEARWAKEUPEN15						: 1;
 	__REG32 CLEARWAKEUPEN16						: 1;
 	__REG32 CLEARWAKEUPEN17						: 1;
 	__REG32 CLEARWAKEUPEN18						: 1;
 	__REG32 CLEARWAKEUPEN19						: 1;
 	__REG32 CLEARWAKEUPEN20						: 1;
 	__REG32 CLEARWAKEUPEN21						: 1;
 	__REG32 CLEARWAKEUPEN22						: 1;
 	__REG32 CLEARWAKEUPEN23						: 1;
 	__REG32 CLEARWAKEUPEN24						: 1;
 	__REG32 CLEARWAKEUPEN25						: 1;
 	__REG32 CLEARWAKEUPEN26						: 1;
 	__REG32 CLEARWAKEUPEN27						: 1;
 	__REG32 CLEARWAKEUPEN28						: 1;
 	__REG32 CLEARWAKEUPEN29						: 1;
 	__REG32 CLEARWAKEUPEN30						: 1;
 	__REG32 CLEARWAKEUPEN31						: 1;
} __gpio_clearwkuena_bits;

/* GPIOn_SETWKUENA */
typedef struct {
 	__REG32 SETWAKEUPEN0						: 1;
 	__REG32 SETWAKEUPEN1						: 1;
 	__REG32 SETWAKEUPEN2						: 1;
 	__REG32 SETWAKEUPEN3						: 1;
 	__REG32 SETWAKEUPEN4						: 1;
 	__REG32 SETWAKEUPEN5						: 1;
 	__REG32 SETWAKEUPEN6						: 1;
 	__REG32 SETWAKEUPEN7						: 1;
 	__REG32 SETWAKEUPEN8						: 1;
 	__REG32 SETWAKEUPEN9						: 1;
 	__REG32 SETWAKEUPEN10						: 1;
 	__REG32 SETWAKEUPEN11						: 1;
 	__REG32 SETWAKEUPEN12						: 1;
 	__REG32 SETWAKEUPEN13						: 1;
 	__REG32 SETWAKEUPEN14						: 1;
 	__REG32 SETWAKEUPEN15						: 1;
 	__REG32 SETWAKEUPEN16						: 1;
 	__REG32 SETWAKEUPEN17						: 1;
 	__REG32 SETWAKEUPEN18						: 1;
 	__REG32 SETWAKEUPEN19						: 1;
 	__REG32 SETWAKEUPEN20						: 1;
 	__REG32 SETWAKEUPEN21						: 1;
 	__REG32 SETWAKEUPEN22						: 1;
 	__REG32 SETWAKEUPEN23						: 1;
 	__REG32 SETWAKEUPEN24						: 1;
 	__REG32 SETWAKEUPEN25						: 1;
 	__REG32 SETWAKEUPEN26						: 1;
 	__REG32 SETWAKEUPEN27						: 1;
 	__REG32 SETWAKEUPEN28						: 1;
 	__REG32 SETWAKEUPEN29						: 1;
 	__REG32 SETWAKEUPEN30						: 1;
 	__REG32 SETWAKEUPEN31						: 1;
} __gpio_setwkuena_bits;

/* GPIOn_CLEARDATAOUT */
typedef struct {
 	__REG32 CLEARDATAOUT0						: 1;
 	__REG32 CLEARDATAOUT1						: 1;
 	__REG32 CLEARDATAOUT2						: 1;
 	__REG32 CLEARDATAOUT3						: 1;
 	__REG32 CLEARDATAOUT4						: 1;
 	__REG32 CLEARDATAOUT5						: 1;
 	__REG32 CLEARDATAOUT6						: 1;
 	__REG32 CLEARDATAOUT7						: 1;
 	__REG32 CLEARDATAOUT8						: 1;
 	__REG32 CLEARDATAOUT9						: 1;
 	__REG32 CLEARDATAOUT10					: 1;
 	__REG32 CLEARDATAOUT11					: 1;
 	__REG32 CLEARDATAOUT12					: 1;
 	__REG32 CLEARDATAOUT13					: 1;
 	__REG32 CLEARDATAOUT14					: 1;
 	__REG32 CLEARDATAOUT15					: 1;
 	__REG32 CLEARDATAOUT16					: 1;
 	__REG32 CLEARDATAOUT17					: 1;
 	__REG32 CLEARDATAOUT18					: 1;
 	__REG32 CLEARDATAOUT19					: 1;
 	__REG32 CLEARDATAOUT20					: 1;
 	__REG32 CLEARDATAOUT21					: 1;
 	__REG32 CLEARDATAOUT22					: 1;
 	__REG32 CLEARDATAOUT23					: 1;
 	__REG32 CLEARDATAOUT24					: 1;
 	__REG32 CLEARDATAOUT25					: 1;
 	__REG32 CLEARDATAOUT26					: 1;
 	__REG32 CLEARDATAOUT27					: 1;
 	__REG32 CLEARDATAOUT28					: 1;
 	__REG32 CLEARDATAOUT29					: 1;
 	__REG32 CLEARDATAOUT30					: 1;
 	__REG32 CLEARDATAOUT31					: 1;
} __gpio_cleardataout_bits;

/* GPIOn_SETDATAOUT */
typedef struct {
 	__REG32 SETDATAOUT0						: 1;
 	__REG32 SETDATAOUT1						: 1;
 	__REG32 SETDATAOUT2						: 1;
 	__REG32 SETDATAOUT3						: 1;
 	__REG32 SETDATAOUT4						: 1;
 	__REG32 SETDATAOUT5						: 1;
 	__REG32 SETDATAOUT6						: 1;
 	__REG32 SETDATAOUT7						: 1;
 	__REG32 SETDATAOUT8						: 1;
 	__REG32 SETDATAOUT9						: 1;
 	__REG32 SETDATAOUT10					: 1;
 	__REG32 SETDATAOUT11					: 1;
 	__REG32 SETDATAOUT12					: 1;
 	__REG32 SETDATAOUT13					: 1;
 	__REG32 SETDATAOUT14					: 1;
 	__REG32 SETDATAOUT15					: 1;
 	__REG32 SETDATAOUT16					: 1;
 	__REG32 SETDATAOUT17					: 1;
 	__REG32 SETDATAOUT18					: 1;
 	__REG32 SETDATAOUT19					: 1;
 	__REG32 SETDATAOUT20					: 1;
 	__REG32 SETDATAOUT21					: 1;
 	__REG32 SETDATAOUT22					: 1;
 	__REG32 SETDATAOUT23					: 1;
 	__REG32 SETDATAOUT24					: 1;
 	__REG32 SETDATAOUT25					: 1;
 	__REG32 SETDATAOUT26					: 1;
 	__REG32 SETDATAOUT27					: 1;
 	__REG32 SETDATAOUT28					: 1;
 	__REG32 SETDATAOUT29					: 1;
 	__REG32 SETDATAOUT30					: 1;
 	__REG32 SETDATAOUT31					: 1;
} __gpio_setdataout_bits;

/* EMAC_SRESET */
typedef struct {
 	__REG32 RESET									: 1;
 	__REG32 											:31;
} __emac_sreset_bits;

/* EMAC_INTCONTROL */
typedef struct {
 	__REG32 INTPRESCALE						:12;
 	__REG32 											: 4;
 	__REG32 C0RXPACEEN						: 1;
 	__REG32 C0TXPACEEN						: 1;
 	__REG32 C1RXPACEEN						: 1;
 	__REG32 C1TXPACEEN						: 1;
 	__REG32 C2RXPACEEN						: 1;
 	__REG32 C2TXPACEEN						: 1;
 	__REG32 											: 9;
 	__REG32 INTTEST								: 1;
} __emac_intcontrol_bits;

/* EMAC_CnRXTHRESHEN */
typedef struct {
 	__REG32 RXCH0THRESHEN					: 1;
 	__REG32 RXCH1THRESHEN					: 1;
 	__REG32 RXCH2THRESHEN					: 1;
 	__REG32 RXCH3THRESHEN					: 1;
 	__REG32 RXCH4THRESHEN					: 1;
 	__REG32 RXCH5THRESHEN					: 1;
 	__REG32 RXCH6THRESHEN					: 1;
 	__REG32 RXCH7THRESHEN					: 1;
 	__REG32 											:24;
} __emac_crxthreshen_bits;

/* EMAC_CnRXEN */
typedef struct {
 	__REG32 RXCH0EN								: 1;
 	__REG32 RXCH1EN								: 1;
 	__REG32 RXCH2EN								: 1;
 	__REG32 RXCH3EN								: 1;
 	__REG32 RXCH4EN								: 1;
 	__REG32 RXCH5EN								: 1;
 	__REG32 RXCH6EN								: 1;
 	__REG32 RXCH7EN								: 1;
 	__REG32 											:24;
} __emac_crxen_bits;

/* EMAC_CnTXEN */
typedef struct {
 	__REG32 RXCH0EN								: 1;
 	__REG32 RXCH1EN								: 1;
 	__REG32 RXCH2EN								: 1;
 	__REG32 RXCH3EN								: 1;
 	__REG32 RXCH4EN								: 1;
 	__REG32 RXCH5EN								: 1;
 	__REG32 RXCH6EN								: 1;
 	__REG32 RXCH7EN								: 1;
 	__REG32 											:24;
} __emac_ctxen_bits;

/* EMAC_CnMISCEN */
typedef struct {
 	__REG32 USERINT0EN						: 1;
 	__REG32 LINKINT0EN						: 1;
 	__REG32 HOSTPENDEN						: 1;
 	__REG32 STATPENDEN						: 1;
 	__REG32 											:28;
} __emac_cmiscen_bits;

/* EMAC_CnRXTHRESHSTAT */
typedef struct {
 	__REG32 RXCH0THRESHSTAT				: 1;
 	__REG32 RXCH1THRESHSTAT				: 1;
 	__REG32 RXCH2THRESHSTAT				: 1;
 	__REG32 RXCH3THRESHSTAT				: 1;
 	__REG32 RXCH4THRESHSTAT				: 1;
 	__REG32 RXCH5THRESHSTAT				: 1;
 	__REG32 RXCH6THRESHSTAT				: 1;
 	__REG32 RXCH7THRESHSTAT				: 1;
 	__REG32 											:24;
} __emac_crxthreshstat_bits;

/* EMAC_CnRXSTAT */
typedef struct {
 	__REG32 RXCH0STAT							: 1;
 	__REG32 RXCH1STAT							: 1;
 	__REG32 RXCH2STAT							: 1;
 	__REG32 RXCH3STAT							: 1;
 	__REG32 RXCH4STAT							: 1;
 	__REG32 RXCH5STAT							: 1;
 	__REG32 RXCH6STAT							: 1;
 	__REG32 RXCH7STAT							: 1;
 	__REG32 											:24;
} __emac_crxstat_bits;

/* EMAC_CnTXSTAT */
typedef struct {
 	__REG32 TXCH0STAT							: 1;
 	__REG32 TXCH1STAT							: 1;
 	__REG32 TXCH2STAT							: 1;
 	__REG32 TXCH3STAT							: 1;
 	__REG32 TXCH4STAT							: 1;
 	__REG32 TXCH5STAT							: 1;
 	__REG32 TXCH6STAT							: 1;
 	__REG32 TXCH7STAT							: 1;
 	__REG32 											:24;
} __emac_ctxstat_bits;

/* EMAC_CnMISCSTAT */
typedef struct {
 	__REG32 USERINT0STAT					: 1;
 	__REG32 LINKINT0STAT					: 1;
 	__REG32 HOSTPENDSTAT					: 1;
 	__REG32 STATPENDSTAT					: 1;
 	__REG32 											:28;
} __emac_cmiscstat_bits;

/* EMAC_CnRXIMAX */
typedef struct {
 	__REG32 RXIMAX								: 6;
 	__REG32 											:26;
} __emac_crximax_bits;

/* EMAC_CnTXIMAX */
typedef struct {
 	__REG32 TXIMAX								: 6;
 	__REG32 											:26;
} __emac_ctximax_bits;

/* MDIO_CONTROL */
typedef struct {
 	__REG32 CLKDIV								:16;
 	__REG32 											: 1;
 	__REG32 INT_TEST_ENABLE				: 1;
 	__REG32 FAULTENB							: 1;
 	__REG32 FAULT									: 1;
 	__REG32 PREAMBLE							: 1;
 	__REG32 											: 3;
 	__REG32 HIGHEST_USER_CHANNEL	: 5;
 	__REG32 											: 1;
 	__REG32 ENABLE								: 1;
 	__REG32 IDLE									: 1;
} __mdio_control_bits;

/* MDIO_LINKINTRAW */
typedef struct {
 	__REG32 USERPHY0							: 1;
 	__REG32 USERPHY1							: 1;
 	__REG32 											:30;
} __mdio_linkintraw_bits;

/* MDIO_USERINTRAW */
typedef struct {
 	__REG32 USERACCESS0						: 1;
 	__REG32 USERACCESS1						: 1;
 	__REG32 											:30;
} __mdio_userintraw_bits;

/* MDIO_USERACCESS0 */
typedef struct {
 	__REG32 DATA									:16;
 	__REG32 PHYADR								: 5;
 	__REG32 REGADR								: 5;
 	__REG32 											: 3;
 	__REG32 ACK										: 1;
 	__REG32 WRITE									: 1;
 	__REG32 GO										: 1;
} __mdio_useraccess_bits;

/* MDIO_USERPHYSEL0 */
typedef struct {
 	__REG32 PHYADRMON							: 5;
 	__REG32 											: 1;
 	__REG32 LINKINTENB						: 1;
 	__REG32 LINKSEL								: 1;
 	__REG32 											:24;
} __mdio_userphysel_bits;

/* EMAC_TXCONTROL */
typedef struct {
 	__REG32 TXEN									: 1;
 	__REG32 											:31;
} __emac_txcontrol_bits;

/* EMAC_TXTEARDOWN */
typedef struct {
 	__REG32 TXTDNCH								: 3;
 	__REG32 											:29;
} __emac_txteardown_bits;

/* EMAC_RXCONTROL */
typedef struct {
 	__REG32 RXEN									: 1;
 	__REG32 											:31;
} __emac_rxcontrol_bits;

/* EMAC_RXTEARDOWN */
typedef struct {
 	__REG32 RXTDNCH								: 3;
 	__REG32 											:29;
} __emac_rxteardown_bits;

/* EMAC_TXINTSTATRAW */
typedef struct {
 	__REG32 TX0PEND								: 1;
 	__REG32 TX1PEND								: 1;
 	__REG32 TX2PEND								: 1;
 	__REG32 TX3PEND								: 1;
 	__REG32 TX4PEND								: 1;
 	__REG32 TX5PEND								: 1;
 	__REG32 TX6PEND								: 1;
 	__REG32 TX7PEND								: 1;
 	__REG32 											:24;
} __emac_txintstatraw_bits;

/* EMAC_TXINTMASKSET */
typedef struct {
 	__REG32 TX0PENDMASK						: 1;
 	__REG32 TX1PENDMASK						: 1;
 	__REG32 TX2PENDMASK						: 1;
 	__REG32 TX3PENDMASK						: 1;
 	__REG32 TX4PENDMASK						: 1;
 	__REG32 TX5PENDMASK						: 1;
 	__REG32 TX6PENDMASK						: 1;
 	__REG32 TX7PENDMASK						: 1;
 	__REG32 											: 8;
 	__REG32 TX0PULSEMASK					: 1;
 	__REG32 TX1PULSEMASK					: 1;
 	__REG32 TX2PULSEMASK					: 1;
 	__REG32 TX3PULSEMASK					: 1;
 	__REG32 TX4PULSEMASK					: 1;
 	__REG32 TX5PULSEMASK					: 1;
 	__REG32 TX6PULSEMASK					: 1;
 	__REG32 TX7PULSEMASK					: 1;
 	__REG32 											: 8;
} __emac_txintmaskset_bits;

/* EMAC_MACINVECTOR */
typedef struct {
 	__REG32 RXPEND								: 8;
 	__REG32 RXTHRESHPEND					: 8;
 	__REG32 TXPEND								: 8;
 	__REG32 USERINT0							: 1;
 	__REG32 LINKINT0							: 1;
 	__REG32 HOSTPEND							: 1;
 	__REG32 STATPEND							: 1;
 	__REG32 											: 4;
} __emac_macinvector_bits;

/* EMAC_MACEOIVECTOR */
typedef struct {
 	__REG32 INTVECT								: 5;
 	__REG32 											:27;
} __emac_maceoivector_bits;

/* EMAC_RXINTSTATRAW */
typedef struct {
 	__REG32 RX0PEND								: 1;
 	__REG32 RX1PEND								: 1;
 	__REG32 RX2PEND								: 1;
 	__REG32 RX3PEND								: 1;
 	__REG32 RX4PEND								: 1;
 	__REG32 RX5PEND								: 1;
 	__REG32 RX6PEND								: 1;
 	__REG32 RX7PEND								: 1;
 	__REG32 RX0THRESHPEND					: 1;
 	__REG32 RX1THRESHPEND					: 1;
 	__REG32 RX2THRESHPEND					: 1;
 	__REG32 RX3THRESHPEND					: 1;
 	__REG32 RX4THRESHPEND					: 1;
 	__REG32 RX5THRESHPEND					: 1;
 	__REG32 RX6THRESHPEND					: 1;
 	__REG32 RX7THRESHPEND					: 1;
 	__REG32 											:16;
} __emac_rxintstatraw_bits;

/* EMAC_RXINTMASKSET */
typedef struct {
 	__REG32 RX0PENDMASK						: 1;
 	__REG32 RX1PENDMASK						: 1;
 	__REG32 RX2PENDMASK						: 1;
 	__REG32 RX3PENDMASK						: 1;
 	__REG32 RX4PENDMASK						: 1;
 	__REG32 RX5PENDMASK						: 1;
 	__REG32 RX6PENDMASK						: 1;
 	__REG32 RX7PENDMASK						: 1;
 	__REG32 RX0PENDTHRESHMASK			: 1;
 	__REG32 RX1THRESHPENDMASK			: 1;
 	__REG32 RX2THRESHPENDMASK			: 1;
 	__REG32 RX3THRESHPENDMASK			: 1;
 	__REG32 RX4THRESHPENDMASK			: 1;
 	__REG32 RX5THRESHPENDMASK			: 1;
 	__REG32 RX6THRESHPENDMASK			: 1;
 	__REG32 RX7THRESHPENDMASK			: 1;
 	__REG32 RX0PULSEMASK					: 1;
 	__REG32 RX1PULSEMASK					: 1;
 	__REG32 RX2PULSEMASK					: 1;
 	__REG32 RX3PULSEMASK					: 1;
 	__REG32 RX4PULSEMASK					: 1;
 	__REG32 RX5PULSEMASK					: 1;
 	__REG32 RX6PULSEMASK					: 1;
 	__REG32 RX7PULSEMASK					: 1;
 	__REG32 											: 8;
} __emac_rxintmaskset_bits;

/* EMAC_MACINTSTATRAW */
typedef struct {
 	__REG32 STATPEND							: 1;
 	__REG32 HOSTPEND							: 1;
 	__REG32 											:30;
} __emac_macintstatraw_bits;

/* EMAC_MACINTMASKSET */
typedef struct {
 	__REG32 STATMASK							: 1;
 	__REG32 HOSTMASK							: 1;
 	__REG32 											:30;
} __emac_macintmaskset_bits;

/* EMAC_RXMBPENABLE */
typedef struct {
 	__REG32 RXMULTCH							: 3;
 	__REG32 											: 2;
 	__REG32 RXMULTEN							: 1;
 	__REG32 											: 2;
 	__REG32 RXBROADCH							: 3;
 	__REG32 											: 2;
 	__REG32 RXBROADEN							: 1;
 	__REG32 											: 2;
 	__REG32 RXPROMCH							: 3;
 	__REG32 											: 2;
 	__REG32 RXCAFEN								: 1;
 	__REG32 RXCEFEN								: 1;
 	__REG32 RXCSFEN								: 1;
 	__REG32 RXCMFEN								: 1;
 	__REG32 											: 3;
 	__REG32 RXNOCHAIN							: 1;
 	__REG32 RXQOSEN								: 1;
 	__REG32 RXPASSCRC							: 1;
 	__REG32 											: 1;
} __emac_rxmbpenable_bits;

/* EMAC_RXUNICASTSET */
typedef struct {
 	__REG32 RXCH0EN								: 1;
 	__REG32 RXCH1EN								: 1;
 	__REG32 RXCH2EN								: 1;
 	__REG32 RXCH3EN								: 1;
 	__REG32 RXCH4EN								: 1;
 	__REG32 RXCH5EN								: 1;
 	__REG32 RXCH6EN								: 1;
 	__REG32 RXCH7EN								: 1;
 	__REG32 											:24;
} __emac_rxunicastset_bits;

/* EMAC_RXMAXLEN */
typedef struct {
 	__REG32 RXMAXLEN							:16;
 	__REG32 											:16;
} __emac_rxmaxlen_bits;

/* EMAC_RXBUFFEROFFSET */
typedef struct {
 	__REG32 RXBUFFEROFFSET				:16;
 	__REG32 											:16;
} __emac_rxbufferoffset_bits;

/* EMAC_RXFILTERLOWTHRESH */
typedef struct {
 	__REG32 RXFILTERTHRESH				: 8;
 	__REG32 											:24;
} __emac_rxfilterlowthresh_bits;

/* EMAC_RXnFLOWTHRESH */
typedef struct {
 	__REG32 RXFLOWTHRESH					: 8;
 	__REG32 											:24;
} __emac_rxflowthresh_bits;

/* EMAC_RXnFREEBUFFER */
typedef struct {
 	__REG32 RXFREEBUF							:16;
 	__REG32 											:16;
} __emac_rxfreebuffer_bits;

/* EMAC_MACCONTROL */
typedef struct {
 	__REG32 FULLDUPLEX						: 1;
 	__REG32 LOOPBACK							: 1;
 	__REG32 											: 1;
 	__REG32 RXBUFFERFLOWEN				: 1;
 	__REG32 TXFLOWEN							: 1;
 	__REG32 GMIIEN								: 1;
 	__REG32 TXPACE								: 1;
 	__REG32 											: 1;
 	__REG32 MEMTEST								: 1;
 	__REG32 TXPTYPE								: 1;
 	__REG32 TXSHORTGAPEN					: 1;
 	__REG32 CMDIDLE								: 1;
 	__REG32 											: 1;
 	__REG32 RXOWNERSHIP						: 1;
 	__REG32 RXOFFLENBLOCK					: 1;
 	__REG32 MACIFCTL_A						: 1;
 	__REG32 MACIFCTL_B						: 1;
 	__REG32 GIGFORCE							: 1;
 	__REG32 MACEXTEN							: 1;
 	__REG32 											:13;
} __emac_maccontrol_bits;

/* EMAC_MACSTATUS */
typedef struct {
 	__REG32 TXFLOWACT							: 1;
 	__REG32 RXFLOWACT							: 1;
 	__REG32 RXQOSACT							: 1;
 	__REG32 EXTFULLDUPLEXSEL			: 1;
 	__REG32 EXTGIGSEL							: 1;
 	__REG32 											: 3;
 	__REG32 RXERRCH								: 3;
 	__REG32 											: 1;
 	__REG32 RXERRCODE							: 4;
 	__REG32 TXERRCH								: 3;
 	__REG32 											: 1;
 	__REG32 TXERRCODE							: 4;
 	__REG32 											: 7;
 	__REG32 IDLE									: 1;
} __emac_macstatus_bits;

/* EMAC_EMCONTROL */
typedef struct {
 	__REG32 FREE									: 1;
 	__REG32 SOFT									: 1;
 	__REG32 											:30;
} __emac_emcontrol_bits;

/* EMAC_FIFOCONTROL */
typedef struct {
 	__REG32 TXCELLTHRESH					: 2;
 	__REG32 											:30;
} __emac_fifocontrol_bits;

/* EMAC_MACCONFIG */
typedef struct {
 	__REG32 MACCFIG								: 8;
 	__REG32 ADDRESSTYPE						: 8;
 	__REG32 RXCELLDEPTH						: 8;
 	__REG32 TXCELLDEPTH						: 8;
} __emac_macconfig_bits;

/* EMAC_SOFTRESET */
typedef struct {
 	__REG32 SOFTRESET							: 1;
 	__REG32 											:31;
} __emac_softreset_bits;

/* EMAC_MACSRCADDRLO */
typedef struct {
 	__REG32 MACSRCADDR1						: 8;
 	__REG32 MACSRCADDR0						: 8;
 	__REG32 											:16;
} __emac_macsrcaddrlo_bits;

/* EMAC_MACSRCADDRHI */
typedef struct {
 	__REG32 MACSRCADDR5						: 8;
 	__REG32 MACSRCADDR4						: 8;
 	__REG32 MACSRCADDR3						: 8;
 	__REG32 MACSRCADDR2						: 8;
} __emac_macsrcaddrhi_bits;

/* EMAC_BOFFTEST */
typedef struct {
 	__REG32 TXBACKOFF							:10;
 	__REG32 											: 2;
 	__REG32 COLLCOUNT							: 4;
 	__REG32 RNDNUM								:10;
 	__REG32 											: 6;
} __emac_bofftest_bits;

/* EMAC_TPACETEST */
typedef struct {
 	__REG32 PACEVAL								: 5;
 	__REG32 											:27;
} __emac_tpacetest_bits;

/* EMAC_RXPAUSE */
typedef struct {
 	__REG32 PAUSETIMER						:16;
 	__REG32 											:16;
} __emac_rxpause_bits;

/* EMAC_TXPAUSE */
typedef struct {
 	__REG32 PAUSETIMER						:16;
 	__REG32 											:16;
} __emac_txpause_bits;

/* EMAC_MACADDRLO */
typedef struct {
 	__REG32 MACADDR1							: 8;
 	__REG32 MACADDR0							: 8;
 	__REG32 CHANNEL								: 3;
 	__REG32 MATCHFILT							: 1;
 	__REG32 VALID									: 1;
 	__REG32 											:11;
} __emac_macaddrlo_bits;

/* EMAC_MACADDRHI */
typedef struct {
 	__REG32 MACADDR5							: 8;
 	__REG32 MACADDR4							: 8;
 	__REG32 MACADDR3							: 8;
 	__REG32 MACADDR2							: 8;
} __emac_macaddrhi_bits;

/* EMAC_MACINDEX */
typedef struct {
 	__REG32 MACINDEX							: 5;
 	__REG32 											:27;
} __emac_macindex_bits;

/* CANME */
typedef struct {
 	__REG32 ME0										: 1;
 	__REG32 ME1										: 1;
 	__REG32 ME2										: 1;
 	__REG32 ME3										: 1;
 	__REG32 ME4										: 1;
 	__REG32 ME5										: 1;
 	__REG32 ME6										: 1;
 	__REG32 ME7										: 1;
 	__REG32 ME8										: 1;
 	__REG32 ME9										: 1;
 	__REG32 ME10									: 1;
 	__REG32 ME11									: 1;
 	__REG32 ME12									: 1;
 	__REG32 ME13									: 1;
 	__REG32 ME14									: 1;
 	__REG32 ME15									: 1;
 	__REG32 ME16									: 1;
 	__REG32 ME17									: 1;
 	__REG32 ME18									: 1;
 	__REG32 ME19									: 1;
 	__REG32 ME20									: 1;
 	__REG32 ME21									: 1;
 	__REG32 ME22									: 1;
 	__REG32 ME23									: 1;
 	__REG32 ME24									: 1;
 	__REG32 ME25									: 1;
 	__REG32 ME26									: 1;
 	__REG32 ME27									: 1;
 	__REG32 ME28									: 1;
 	__REG32 ME29									: 1;
 	__REG32 ME30									: 1;
 	__REG32 ME31									: 1;
} __canme_bits;

/* CANMD */
typedef struct {
 	__REG32 MD0										: 1;
 	__REG32 MD1										: 1;
 	__REG32 MD2										: 1;
 	__REG32 MD3										: 1;
 	__REG32 MD4										: 1;
 	__REG32 MD5										: 1;
 	__REG32 MD6										: 1;
 	__REG32 MD7										: 1;
 	__REG32 MD8										: 1;
 	__REG32 MD9										: 1;
 	__REG32 MD10									: 1;
 	__REG32 MD11									: 1;
 	__REG32 MD12									: 1;
 	__REG32 MD13									: 1;
 	__REG32 MD14									: 1;
 	__REG32 MD15									: 1;
 	__REG32 MD16									: 1;
 	__REG32 MD17									: 1;
 	__REG32 MD18									: 1;
 	__REG32 MD19									: 1;
 	__REG32 MD20									: 1;
 	__REG32 MD21									: 1;
 	__REG32 MD22									: 1;
 	__REG32 MD23									: 1;
 	__REG32 MD24									: 1;
 	__REG32 MD25									: 1;
 	__REG32 MD26									: 1;
 	__REG32 MD27									: 1;
 	__REG32 MD28									: 1;
 	__REG32 MD29									: 1;
 	__REG32 MD30									: 1;
 	__REG32 MD31									: 1;
} __canmd_bits;

/* CANTRS */
typedef struct {
 	__REG32 TRS0									: 1;
 	__REG32 TRS1									: 1;
 	__REG32 TRS2									: 1;
 	__REG32 TRS3									: 1;
 	__REG32 TRS4									: 1;
 	__REG32 TRS5									: 1;
 	__REG32 TRS6									: 1;
 	__REG32 TRS7									: 1;
 	__REG32 TRS8									: 1;
 	__REG32 TRS9									: 1;
 	__REG32 TRS10									: 1;
 	__REG32 TRS11									: 1;
 	__REG32 TRS12									: 1;
 	__REG32 TRS13									: 1;
 	__REG32 TRS14									: 1;
 	__REG32 TRS15									: 1;
 	__REG32 TRS16									: 1;
 	__REG32 TRS17									: 1;
 	__REG32 TRS18									: 1;
 	__REG32 TRS19									: 1;
 	__REG32 TRS20									: 1;
 	__REG32 TRS21									: 1;
 	__REG32 TRS22									: 1;
 	__REG32 TRS23									: 1;
 	__REG32 TRS24									: 1;
 	__REG32 TRS25									: 1;
 	__REG32 TRS26									: 1;
 	__REG32 TRS27									: 1;
 	__REG32 TRS28									: 1;
 	__REG32 TRS29									: 1;
 	__REG32 TRS30									: 1;
 	__REG32 TRS31									: 1;
} __cantrs_bits;

/* CANTRR */
typedef struct {
 	__REG32 TRR0									: 1;
 	__REG32 TRR1									: 1;
 	__REG32 TRR2									: 1;
 	__REG32 TRR3									: 1;
 	__REG32 TRR4									: 1;
 	__REG32 TRR5									: 1;
 	__REG32 TRR6									: 1;
 	__REG32 TRR7									: 1;
 	__REG32 TRR8									: 1;
 	__REG32 TRR9									: 1;
 	__REG32 TRR10									: 1;
 	__REG32 TRR11									: 1;
 	__REG32 TRR12									: 1;
 	__REG32 TRR13									: 1;
 	__REG32 TRR14									: 1;
 	__REG32 TRR15									: 1;
 	__REG32 TRR16									: 1;
 	__REG32 TRR17									: 1;
 	__REG32 TRR18									: 1;
 	__REG32 TRR19									: 1;
 	__REG32 TRR20									: 1;
 	__REG32 TRR21									: 1;
 	__REG32 TRR22									: 1;
 	__REG32 TRR23									: 1;
 	__REG32 TRR24									: 1;
 	__REG32 TRR25									: 1;
 	__REG32 TRR26									: 1;
 	__REG32 TRR27									: 1;
 	__REG32 TRR28									: 1;
 	__REG32 TRR29									: 1;
 	__REG32 TRR30									: 1;
 	__REG32 TRR31									: 1;
} __cantrr_bits;

/* CANTA */
typedef struct {
 	__REG32 TA0										: 1;
 	__REG32 TA1										: 1;
 	__REG32 TA2										: 1;
 	__REG32 TA3										: 1;
 	__REG32 TA4										: 1;
 	__REG32 TA5										: 1;
 	__REG32 TA6										: 1;
 	__REG32 TA7										: 1;
 	__REG32 TA8										: 1;
 	__REG32 TA9										: 1;
 	__REG32 TA10									: 1;
 	__REG32 TA11									: 1;
 	__REG32 TA12									: 1;
 	__REG32 TA13									: 1;
 	__REG32 TA14									: 1;
 	__REG32 TA15									: 1;
 	__REG32 TA16									: 1;
 	__REG32 TA17									: 1;
 	__REG32 TA18									: 1;
 	__REG32 TA19									: 1;
 	__REG32 TA20									: 1;
 	__REG32 TA21									: 1;
 	__REG32 TA22									: 1;
 	__REG32 TA23									: 1;
 	__REG32 TA24									: 1;
 	__REG32 TA25									: 1;
 	__REG32 TA26									: 1;
 	__REG32 TA27									: 1;
 	__REG32 TA28									: 1;
 	__REG32 TA29									: 1;
 	__REG32 TA30									: 1;
 	__REG32 TA31									: 1;
} __canta_bits;

/* CANAA */
typedef struct {
 	__REG32 AA0										: 1;
 	__REG32 AA1										: 1;
 	__REG32 AA2										: 1;
 	__REG32 AA3										: 1;
 	__REG32 AA4										: 1;
 	__REG32 AA5										: 1;
 	__REG32 AA6										: 1;
 	__REG32 AA7										: 1;
 	__REG32 AA8										: 1;
 	__REG32 AA9										: 1;
 	__REG32 AA10									: 1;
 	__REG32 AA11									: 1;
 	__REG32 AA12									: 1;
 	__REG32 AA13									: 1;
 	__REG32 AA14									: 1;
 	__REG32 AA15									: 1;
 	__REG32 AA16									: 1;
 	__REG32 AA17									: 1;
 	__REG32 AA18									: 1;
 	__REG32 AA19									: 1;
 	__REG32 AA20									: 1;
 	__REG32 AA21									: 1;
 	__REG32 AA22									: 1;
 	__REG32 AA23									: 1;
 	__REG32 AA24									: 1;
 	__REG32 AA25									: 1;
 	__REG32 AA26									: 1;
 	__REG32 AA27									: 1;
 	__REG32 AA28									: 1;
 	__REG32 AA29									: 1;
 	__REG32 AA30									: 1;
 	__REG32 AA31									: 1;
} __canaa_bits;

/* CANRMP */
typedef struct {
 	__REG32 RMP0									: 1;
 	__REG32 RMP1									: 1;
 	__REG32 RMP2									: 1;
 	__REG32 RMP3									: 1;
 	__REG32 RMP4									: 1;
 	__REG32 RMP5									: 1;
 	__REG32 RMP6									: 1;
 	__REG32 RMP7									: 1;
 	__REG32 RMP8									: 1;
 	__REG32 RMP9									: 1;
 	__REG32 RMP10									: 1;
 	__REG32 RMP11									: 1;
 	__REG32 RMP12									: 1;
 	__REG32 RMP13									: 1;
 	__REG32 RMP14									: 1;
 	__REG32 RMP15									: 1;
 	__REG32 RMP16									: 1;
 	__REG32 RMP17									: 1;
 	__REG32 RMP18									: 1;
 	__REG32 RMP19									: 1;
 	__REG32 RMP20									: 1;
 	__REG32 RMP21									: 1;
 	__REG32 RMP22									: 1;
 	__REG32 RMP23									: 1;
 	__REG32 RMP24									: 1;
 	__REG32 RMP25									: 1;
 	__REG32 RMP26									: 1;
 	__REG32 RMP27									: 1;
 	__REG32 RMP28									: 1;
 	__REG32 RMP29									: 1;
 	__REG32 RMP30									: 1;
 	__REG32 RMP31									: 1;
} __canrmp_bits;

/* CANRML */
typedef struct {
 	__REG32 RML0									: 1;
 	__REG32 RML1									: 1;
 	__REG32 RML2									: 1;
 	__REG32 RML3									: 1;
 	__REG32 RML4									: 1;
 	__REG32 RML5									: 1;
 	__REG32 RML6									: 1;
 	__REG32 RML7									: 1;
 	__REG32 RML8									: 1;
 	__REG32 RML9									: 1;
 	__REG32 RML10									: 1;
 	__REG32 RML11									: 1;
 	__REG32 RML12									: 1;
 	__REG32 RML13									: 1;
 	__REG32 RML14									: 1;
 	__REG32 RML15									: 1;
 	__REG32 RML16									: 1;
 	__REG32 RML17									: 1;
 	__REG32 RML18									: 1;
 	__REG32 RML19									: 1;
 	__REG32 RML20									: 1;
 	__REG32 RML21									: 1;
 	__REG32 RML22									: 1;
 	__REG32 RML23									: 1;
 	__REG32 RML24									: 1;
 	__REG32 RML25									: 1;
 	__REG32 RML26									: 1;
 	__REG32 RML27									: 1;
 	__REG32 RML28									: 1;
 	__REG32 RML29									: 1;
 	__REG32 RML30									: 1;
 	__REG32 RML31									: 1;
} __canrml_bits;

/* CANRFP */
typedef struct {
 	__REG32 RFP0									: 1;
 	__REG32 RFP1									: 1;
 	__REG32 RFP2									: 1;
 	__REG32 RFP3									: 1;
 	__REG32 RFP4									: 1;
 	__REG32 RFP5									: 1;
 	__REG32 RFP6									: 1;
 	__REG32 RFP7									: 1;
 	__REG32 RFP8									: 1;
 	__REG32 RFP9									: 1;
 	__REG32 RFP10									: 1;
 	__REG32 RFP11									: 1;
 	__REG32 RFP12									: 1;
 	__REG32 RFP13									: 1;
 	__REG32 RFP14									: 1;
 	__REG32 RFP15									: 1;
 	__REG32 RFP16									: 1;
 	__REG32 RFP17									: 1;
 	__REG32 RFP18									: 1;
 	__REG32 RFP19									: 1;
 	__REG32 RFP20									: 1;
 	__REG32 RFP21									: 1;
 	__REG32 RFP22									: 1;
 	__REG32 RFP23									: 1;
 	__REG32 RFP24									: 1;
 	__REG32 RFP25									: 1;
 	__REG32 RFP26									: 1;
 	__REG32 RFP27									: 1;
 	__REG32 RFP28									: 1;
 	__REG32 RFP29									: 1;
 	__REG32 RFP30									: 1;
 	__REG32 RFP31									: 1;
} __canrfp_bits;

/* CANGAM */
typedef struct {
 	__REG32 GAM0									: 1;
 	__REG32 GAM1									: 1;
 	__REG32 GAM2									: 1;
 	__REG32 GAM3									: 1;
 	__REG32 GAM4									: 1;
 	__REG32 GAM5									: 1;
 	__REG32 GAM6									: 1;
 	__REG32 GAM7									: 1;
 	__REG32 GAM8									: 1;
 	__REG32 GAM9									: 1;
 	__REG32 GAM10									: 1;
 	__REG32 GAM11									: 1;
 	__REG32 GAM12									: 1;
 	__REG32 GAM13									: 1;
 	__REG32 GAM14									: 1;
 	__REG32 GAM15									: 1;
 	__REG32 GAM16									: 1;
 	__REG32 GAM17									: 1;
 	__REG32 GAM18									: 1;
 	__REG32 GAM19									: 1;
 	__REG32 GAM20									: 1;
 	__REG32 GAM21									: 1;
 	__REG32 GAM22									: 1;
 	__REG32 GAM23									: 1;
 	__REG32 GAM24									: 1;
 	__REG32 GAM25									: 1;
 	__REG32 GAM26									: 1;
 	__REG32 GAM27									: 1;
 	__REG32 GAM28									: 1;
 	__REG32 											: 2;
 	__REG32 AMI										: 1;
} __cangam_bits;

/* CANLAM */
typedef struct {
 	__REG32 LAM0									: 1;
 	__REG32 LAM1									: 1;
 	__REG32 LAM2									: 1;
 	__REG32 LAM3									: 1;
 	__REG32 LAM4									: 1;
 	__REG32 LAM5									: 1;
 	__REG32 LAM6									: 1;
 	__REG32 LAM7									: 1;
 	__REG32 LAM8									: 1;
 	__REG32 LAM9									: 1;
 	__REG32 LAM10									: 1;
 	__REG32 LAM11									: 1;
 	__REG32 LAM12									: 1;
 	__REG32 LAM13									: 1;
 	__REG32 LAM14									: 1;
 	__REG32 LAM15									: 1;
 	__REG32 LAM16									: 1;
 	__REG32 LAM17									: 1;
 	__REG32 LAM18									: 1;
 	__REG32 LAM19									: 1;
 	__REG32 LAM20									: 1;
 	__REG32 LAM21									: 1;
 	__REG32 LAM22									: 1;
 	__REG32 LAM23									: 1;
 	__REG32 LAM24									: 1;
 	__REG32 LAM25									: 1;
 	__REG32 LAM26									: 1;
 	__REG32 LAM27									: 1;
 	__REG32 LAM28									: 1;
 	__REG32 											: 2;
 	__REG32 LAMI									: 1;
} __canlam_bits;

/* CANMC */
typedef struct {
 	__REG32 MBNR									: 5;
 	__REG32 SRES									: 1;
 	__REG32 STM										: 1;
 	__REG32 ABO										: 1;
 	__REG32 CDR										: 1;
 	__REG32 WUBA									: 1;
 	__REG32 DBO										: 1;
 	__REG32 PDR										: 1;
 	__REG32 CCR										: 1;
 	__REG32 SCM										: 1;
 	__REG32 LNTM									: 1;
 	__REG32 LNTC									: 1;
 	__REG32 											:16;
} __canmc_bits;

/* CANBTC */
typedef struct {
 	__REG32 TSEG2									: 3;
 	__REG32 TSEG1									: 4;
 	__REG32 SAM										: 1;
 	__REG32 SJW										: 2;
 	__REG32 ERM										: 1;
 	__REG32 											: 5;
 	__REG32 BRP										: 8;
 	__REG32 											: 8;
} __canbtc_bits;

/* CANES */
typedef struct {
 	__REG32 TM										: 1;
 	__REG32 RM										: 1;
 	__REG32 											: 1;
 	__REG32 PDA										: 1;
 	__REG32 CCE										: 1;
 	__REG32 SMA										: 1;
 	__REG32 											:10;
 	__REG32 EW										: 1;
 	__REG32 EP										: 1;
 	__REG32 BO										: 1;
 	__REG32 ACKE									: 1;
 	__REG32 SE										: 1;
 	__REG32 CRCE									: 1;
 	__REG32 SA1										: 1;
 	__REG32 BE										: 1;
 	__REG32 FE										: 1;
 	__REG32 											:	7;
} __canes_bits;

/* CANTEC */
typedef struct {
 	__REG32 TEC										: 8;
 	__REG32 											:24;
} __cantec_bits;

/* CANREC */
typedef struct {
 	__REG32 REC										: 8;
 	__REG32 											:24;
} __canrec_bits;

/* CANGIFn */
typedef struct {
 	__REG32 MIV0									: 1;
 	__REG32 MIV1									: 1;
 	__REG32 MIV2									: 1;
 	__REG32 MIV3									: 1;
 	__REG32 MIV4									: 1;
 	__REG32    										: 3;
 	__REG32 WLIF									: 1;
 	__REG32 EPIF									: 1;
 	__REG32 BOIF									: 1;
 	__REG32 RMLIF									: 1;
 	__REG32 WUIF									: 1;
 	__REG32 WDIF									: 1;
 	__REG32 AAIF									: 1;
 	__REG32 GMIF									: 1;
 	__REG32 TCOIF									: 1;
 	__REG32 MAIF									: 1;
 	__REG32 											:14;
} __cangif_bits;

/* CANGIM */
typedef struct {
 	__REG32 I0EN									: 1;
 	__REG32 I1EN									: 1;
 	__REG32 SIL										: 1;
 	__REG32    										: 5;
 	__REG32 WLIM									: 1;
 	__REG32 EPIM									: 1;
 	__REG32 BOIM									: 1;
 	__REG32 RMLIM									: 1;
 	__REG32 WUIM									: 1;
 	__REG32 WDIM									: 1;
 	__REG32 AAIM									: 1;
 	__REG32    										: 1;
 	__REG32 TCOIM									: 1;
 	__REG32 MAIM									: 1;
 	__REG32    										:14;
} __cangim_bits;

/* CANMIM */
typedef struct {
 	__REG32 MIM0									: 1;
 	__REG32 MIM1									: 1;
 	__REG32 MIM2									: 1;
 	__REG32 MIM3									: 1;
 	__REG32 MIM4									: 1;
 	__REG32 MIM5									: 1;
 	__REG32 MIM6									: 1;
 	__REG32 MIM7									: 1;
 	__REG32 MIM8									: 1;
 	__REG32 MIM9									: 1;
 	__REG32 MIM10									: 1;
 	__REG32 MIM11									: 1;
 	__REG32 MIM12									: 1;
 	__REG32 MIM13									: 1;
 	__REG32 MIM14									: 1;
 	__REG32 MIM15									: 1;
 	__REG32 MIM16									: 1;
 	__REG32 MIM17									: 1;
 	__REG32 MIM18									: 1;
 	__REG32 MIM19									: 1;
 	__REG32 MIM20									: 1;
 	__REG32 MIM21									: 1;
 	__REG32 MIM22									: 1;
 	__REG32 MIM23									: 1;
 	__REG32 MIM24									: 1;
 	__REG32 MIM25									: 1;
 	__REG32 MIM26									: 1;
 	__REG32 MIM27									: 1;
 	__REG32 MIM28									: 1;
 	__REG32 MIM29									: 1;
 	__REG32 MIM30									: 1;
 	__REG32 MIM31									: 1;
} __canmim_bits;

/* CANMIL */
typedef struct {
 	__REG32 MIL0									: 1;
 	__REG32 MIL1									: 1;
 	__REG32 MIL2									: 1;
 	__REG32 MIL3									: 1;
 	__REG32 MIL4									: 1;
 	__REG32 MIL5									: 1;
 	__REG32 MIL6									: 1;
 	__REG32 MIL7									: 1;
 	__REG32 MIL8									: 1;
 	__REG32 MIL9									: 1;
 	__REG32 MIL10									: 1;
 	__REG32 MIL11									: 1;
 	__REG32 MIL12									: 1;
 	__REG32 MIL13									: 1;
 	__REG32 MIL14									: 1;
 	__REG32 MIL15									: 1;
 	__REG32 MIL16									: 1;
 	__REG32 MIL17									: 1;
 	__REG32 MIL18									: 1;
 	__REG32 MIL19									: 1;
 	__REG32 MIL20									: 1;
 	__REG32 MIL21									: 1;
 	__REG32 MIL22									: 1;
 	__REG32 MIL23									: 1;
 	__REG32 MIL24									: 1;
 	__REG32 MIL25									: 1;
 	__REG32 MIL26									: 1;
 	__REG32 MIL27									: 1;
 	__REG32 MIL28									: 1;
 	__REG32 MIL29									: 1;
 	__REG32 MIL30									: 1;
 	__REG32 MIL31									: 1;
} __canmil_bits;

/* CANOPC */
typedef struct {
 	__REG32 OPC0									: 1;
 	__REG32 OPC1									: 1;
 	__REG32 OPC2									: 1;
 	__REG32 OPC3									: 1;
 	__REG32 OPC4									: 1;
 	__REG32 OPC5									: 1;
 	__REG32 OPC6									: 1;
 	__REG32 OPC7									: 1;
 	__REG32 OPC8									: 1;
 	__REG32 OPC9									: 1;
 	__REG32 OPC10									: 1;
 	__REG32 OPC11									: 1;
 	__REG32 OPC12									: 1;
 	__REG32 OPC13									: 1;
 	__REG32 OPC14									: 1;
 	__REG32 OPC15									: 1;
 	__REG32 OPC16									: 1;
 	__REG32 OPC17									: 1;
 	__REG32 OPC18									: 1;
 	__REG32 OPC19									: 1;
 	__REG32 OPC20									: 1;
 	__REG32 OPC21									: 1;
 	__REG32 OPC22									: 1;
 	__REG32 OPC23									: 1;
 	__REG32 OPC24									: 1;
 	__REG32 OPC25									: 1;
 	__REG32 OPC26									: 1;
 	__REG32 OPC27									: 1;
 	__REG32 OPC28									: 1;
 	__REG32 OPC29									: 1;
 	__REG32 OPC30									: 1;
 	__REG32 OPC31									: 1;
} __canopc_bits;

/* CANTIOC */
typedef struct {
 	__REG32 TXIN									: 1;
 	__REG32 TXOUT									: 1;
 	__REG32 TXDIR									: 1;
 	__REG32 TXFUNC								: 1;
 	__REG32 											:28;
} __cantioc_bits;

/* CANRIOC */
typedef struct {
 	__REG32 RXIN									: 1;
 	__REG32 RXOUT									: 1;
 	__REG32 RXDIR									: 1;
 	__REG32 RXFUNC								: 1;
 	__REG32 											:28;
} __canrioc_bits;

/* CANLNT */
typedef struct {
 	__REG32 LNT0									: 1;
 	__REG32 LNT1									: 1;
 	__REG32 LNT2									: 1;
 	__REG32 LNT3									: 1;
 	__REG32 LNT4									: 1;
 	__REG32 LNT5									: 1;
 	__REG32 LNT6									: 1;
 	__REG32 LNT7									: 1;
 	__REG32 LNT8									: 1;
 	__REG32 LNT9									: 1;
 	__REG32 LNT10									: 1;
 	__REG32 LNT11									: 1;
 	__REG32 LNT12									: 1;
 	__REG32 LNT13									: 1;
 	__REG32 LNT14									: 1;
 	__REG32 LNT15									: 1;
 	__REG32 LNT16									: 1;
 	__REG32 LNT17									: 1;
 	__REG32 LNT18									: 1;
 	__REG32 LNT19									: 1;
 	__REG32 LNT20									: 1;
 	__REG32 LNT21									: 1;
 	__REG32 LNT22									: 1;
 	__REG32 LNT23									: 1;
 	__REG32 LNT24									: 1;
 	__REG32 LNT25									: 1;
 	__REG32 LNT26									: 1;
 	__REG32 LNT27									: 1;
 	__REG32 LNT28									: 1;
 	__REG32 LNT29									: 1;
 	__REG32 LNT30									: 1;
 	__REG32 LNT31									: 1;
} __canlnt_bits;

/* CANTOC */
typedef struct {
 	__REG32 TOC0									: 1;
 	__REG32 TOC1									: 1;
 	__REG32 TOC2									: 1;
 	__REG32 TOC3									: 1;
 	__REG32 TOC4									: 1;
 	__REG32 TOC5									: 1;
 	__REG32 TOC6									: 1;
 	__REG32 TOC7									: 1;
 	__REG32 TOC8									: 1;
 	__REG32 TOC9									: 1;
 	__REG32 TOC10									: 1;
 	__REG32 TOC11									: 1;
 	__REG32 TOC12									: 1;
 	__REG32 TOC13									: 1;
 	__REG32 TOC14									: 1;
 	__REG32 TOC15									: 1;
 	__REG32 TOC16									: 1;
 	__REG32 TOC17									: 1;
 	__REG32 TOC18									: 1;
 	__REG32 TOC19									: 1;
 	__REG32 TOC20									: 1;
 	__REG32 TOC21									: 1;
 	__REG32 TOC22									: 1;
 	__REG32 TOC23									: 1;
 	__REG32 TOC24									: 1;
 	__REG32 TOC25									: 1;
 	__REG32 TOC26									: 1;
 	__REG32 TOC27									: 1;
 	__REG32 TOC28									: 1;
 	__REG32 TOC29									: 1;
 	__REG32 TOC30									: 1;
 	__REG32 TOC31									: 1;
} __cantoc_bits;

/* CANTOS */
typedef struct {
 	__REG32 TOS0									: 1;
 	__REG32 TOS1									: 1;
 	__REG32 TOS2									: 1;
 	__REG32 TOS3									: 1;
 	__REG32 TOS4									: 1;
 	__REG32 TOS5									: 1;
 	__REG32 TOS6									: 1;
 	__REG32 TOS7									: 1;
 	__REG32 TOS8									: 1;
 	__REG32 TOS9									: 1;
 	__REG32 TOS10									: 1;
 	__REG32 TOS11									: 1;
 	__REG32 TOS12									: 1;
 	__REG32 TOS13									: 1;
 	__REG32 TOS14									: 1;
 	__REG32 TOS15									: 1;
 	__REG32 TOS16									: 1;
 	__REG32 TOS17									: 1;
 	__REG32 TOS18									: 1;
 	__REG32 TOS19									: 1;
 	__REG32 TOS20									: 1;
 	__REG32 TOS21									: 1;
 	__REG32 TOS22									: 1;
 	__REG32 TOS23									: 1;
 	__REG32 TOS24									: 1;
 	__REG32 TOS25									: 1;
 	__REG32 TOS26									: 1;
 	__REG32 TOS27									: 1;
 	__REG32 TOS28									: 1;
 	__REG32 TOS29									: 1;
 	__REG32 TOS30									: 1;
 	__REG32 TOS31									: 1;
} __cantos_bits;

/* UARTn_DLL_REG */
typedef union {
	/* UARTx_DLL_REG */
	struct {
 	__REG16 CLOCK_LSB							: 8;
 	__REG16 											: 8;
 	};
	/* UARTx_RHR_REG */
	struct {
 	__REG16 RHR										: 8;
 	__REG16 											: 8;
 	};
	/* UARTx_THR_REG */
	struct {
 	__REG16 THR										: 8;
 	__REG16 											: 8;
 	};
} __uart_dll_reg_bits;

/* UARTn_IER_REG */
typedef union {
	/* UARTx_IER_REG */
	struct {
 	__REG16 RHR_IT								: 1;
 	__REG16 THR_IT								: 1;
 	__REG16 LINE_STS_IT						: 1;
 	__REG16 MODEM_STS_IT					: 1;
 	__REG16 SLEEP_MODE						: 1;
 	__REG16 XOFF_IT								: 1;
 	__REG16 RTS_IT								: 1;
 	__REG16 CTS_IT								: 1;
 	__REG16 											: 8;
 	};
	/* UARTx_IrDA_IER_REG */
	struct {
 	__REG16 RHR_IT_I							: 1;
 	__REG16 THR_IT_I							: 1;
 	__REG16 LAST_RX_BYTE_IT				: 1;
 	__REG16 RX_OVERRUN_IT 				: 1;
 	__REG16 STS_FIFO_TRIG_IT			: 1;
 	__REG16 TX_STATUS_IT					: 1;
 	__REG16 LINE_STS_IT_I					: 1;
 	__REG16 EOF_IT								: 1;
 	__REG16 											: 8;
 	};
	/* UARTx_CIR_IER_REG */
	struct {
 	__REG16         							: 1;
 	__REG16 _THR_IT								: 1;
 	__REG16 											: 3;
 	__REG16 _TX_STATUS_IT  				: 1;
 	__REG16 											:10;
 	};
	/* UARTx_DLH_REG */
	struct {
 	__REG16 CLOCK_MSB							: 6;
 	__REG16 											:10;
 	};
} __uart_ier_reg_bits;

/* UARTn_FCR_REG */
typedef union {
	/* UARTx_FCR_REG */
	struct {
 	__REG16 FIFO_EN								: 1;
 	__REG16 RX_FIFO_CLEAR					: 1;
 	__REG16 TX_FIFO_CLEAR					: 1;
 	__REG16 DMA_MODE							: 1;
 	__REG16 TX_FIFO_TRIG					: 2;
 	__REG16 RX_FIFO_TRIG					: 2;
 	__REG16 											: 8;
 	};
	/* UARTx_IIR_REG */
	struct {
 	__REG16 IT_PENDING						: 1;
 	__REG16 IT_TYPE								: 5;
 	__REG16 FCR_MIRROR						: 2;
 	__REG16 											: 8;
 	};
	/* UARTx_IrDA_FCR_REG */
	struct {
 	__REG16 RHR_IT_I							: 1;
 	__REG16 THR_IT_I							: 1;
 	__REG16 LAST_RX_BYTE_IT				: 1;
 	__REG16 RX_OVERRUN_IT 				: 1;
 	__REG16 STS_FIFO_TRIG_IT			: 1;
 	__REG16 TX_STATUS_IT					: 1;
 	__REG16 LINE_STS_IT_I					: 1;
 	__REG16 EOF_IT								: 1;
 	__REG16 											: 8;
 	};
	/* UARTx_CIR_FCR_REG */
	struct {
 	__REG16         							: 1;
 	__REG16	THR_IT								: 1;
 	__REG16 											: 3;
 	__REG16 _TX_STATUS_IT  				: 1;
 	__REG16 											:10;
 	};
	/* UARTx_EFR_REG */
	struct {
 	__REG16 SW_FLOW_CONTROL				: 4;
 	__REG16	ENHANCED_EN						: 1;
 	__REG16 SPEC_CHAR							: 1;
 	__REG16 AUTO_RTS_EN						: 1;
 	__REG16 AUTO_CTS_EN	  				: 1;
 	__REG16 											: 8;
 	};
} __uart_fcr_reg_bits;

/* UARTn_LCR_REG */
typedef struct {
 	__REG16 CHAR_LENGTH						: 2;
 	__REG16 NB_STOP								: 1;
 	__REG16 PARITY_EN							: 1;
 	__REG16 PARITY_TYPE1					: 1;
 	__REG16 PARITY_TYPE2					: 1;
 	__REG16 BREAK_EN							: 1;
 	__REG16 DIV_EN								: 1;
 	__REG16 											: 8;
} __uart_lcr_reg_bits;

/* UARTn_MCR_REG */
typedef union {
	/* UARTx_MCR_REG */
	struct {
 	__REG16 DTR										: 1;
 	__REG16 RTS										: 1;
 	__REG16 RI_STS_CH							: 1;
 	__REG16 CD_STS_CH							: 1;
 	__REG16 LOOPBACK_EN						: 1;
 	__REG16 XON_EN								: 1;
 	__REG16 TCR_TLR								: 1;
 	__REG16 											: 9;
 	};
	/* UARTx_XON1_ADDR1_REG */
	struct {
 	__REG16 XON_WORD1							: 8;
 	__REG16 											: 8;
 	};
} __uart_mcr_reg_bits;

/* UARTn_LSR_REG */
typedef union {
	/* UARTx_LSR_REG */
	struct {
 	__REG16 RX_FIFO_E							: 1;
 	__REG16 RX_OE									: 1;
 	__REG16 RX_PE									: 1;
 	__REG16 RX_FE									: 1;
 	__REG16 RX_BI									: 1;
 	__REG16 TX_FIFO_E							: 1;
 	__REG16 TX_SR_E								: 1;
 	__REG16 RX_FIFO_STS						: 1;
 	__REG16 											: 8;
 	};
	/* UARTx_LSR_IrDA_REG */
	struct {
 	__REG16 _RX_FIFO_E						: 1;
 	__REG16 STS_FIFO_E						: 1;
 	__REG16 CRC										: 1;
 	__REG16 ABORT									: 1;
 	__REG16 FRAME_TOO_LONG				: 1;
 	__REG16 RX_LAST_BYTE					: 1;
 	__REG16 STS_FIFO_FUL					: 1;
 	__REG16 THR_EMPTY							: 1;
 	__REG16 											: 8;
 	};
	/* UARTx_LSR_CIR_REG */
	struct {
 	__REG16   										: 7;
 	__REG16 _THR_EMPTY						: 1;
 	__REG16 											: 8;
 	};
	/* UARTx_XON2_ADDR2_REG */
	struct {
 	__REG16 XON_WORD2							: 8;
 	__REG16 											: 8;
 	};
} __uart_lsr_reg_bits;

/* UARTn_XOFF1_REG */
typedef union {
	/* UARTx_XOFF1_REG */
	struct {
 	__REG16 XOFF_WORD1						: 8;
 	__REG16 											: 8;
 	};
	/* UARTx_TCR_REG */
	struct {
 	__REG16 RX_FIFO_TRIG_HALT			: 4;
 	__REG16 RX_FIFO_TRIG_START		: 4;
 	__REG16 											: 8;
 	};
	/* UARTx_MSR_REG */
	struct {
 	__REG16 CTS_STS								: 1;
 	__REG16 DSR_STS								: 1;
 	__REG16 RI_STS								: 1;
 	__REG16 DCD_STS								: 1;
 	__REG16 NCTS_STS							: 1;
 	__REG16 NDSR_STS							: 1;
 	__REG16 NRI_STS								: 1;
 	__REG16 NCD_STS								: 1;
 	__REG16 											: 8;
 	};
} __uart_xoff1_reg_bits;

/* UARTn_XOFF2_REG */
typedef union {
	/* UARTx_XOFF2_REG */
	struct {
 	__REG16 XOFF_WORD2						: 8;
 	__REG16 											: 8;
 	};
	/* UARTx_SPR_REG */
	struct {
 	__REG16 SPR_WORD							: 8;
 	__REG16 											: 8;
 	};
	/* UARTx_TLR_REG */
	struct {
 	__REG16 TX_FIFO_TRIG_DMA			: 4;
 	__REG16 RX_FIFO_TRIG_DMA			: 4;
 	__REG16 											: 8;
 	};
} __uart_xoff2_reg_bits;

/* UARTn_MDR1_REG */
typedef struct {
 	__REG16 MODE_SELECT						: 3;
 	__REG16 IR_SLEEP							: 1;
 	__REG16 SET_TXIR							: 1;
 	__REG16 SCT										: 1;
 	__REG16 SIP_MODE							: 1;
 	__REG16 FRAME_END_MODE				: 1;
 	__REG16 											: 8;
} __uart_mdr1_reg_bits;

/* UARTn_MDR2_REG */
typedef struct {
 	__REG16 IRTX_UNDERRUN					: 1;
 	__REG16 STS_FIFO_TRIG					: 2;
 	__REG16 UART_PULSE						: 1;
 	__REG16 CIR_PULSE_MODE				: 2;
 	__REG16 IRRXINVERT						: 1;
 	__REG16 											: 9;
} __uart_mdr2_reg_bits;

/* UARTn_TXFLL_REG */
typedef union {
	/* UARTx_TXFLL_REG */
	struct {
 	__REG16 TXFLL									: 8;
 	__REG16 											: 8;
 	};
	/* UARTx_SFLSR_REG */
	struct {
 	__REG16 											: 1;
 	__REG16 CRC_ERROR							: 1;
 	__REG16 ABORT_DETECT					: 1;
 	__REG16 FTL_ERROR							: 1;
 	__REG16 OE_ERROR							: 1;
 	__REG16 											:11;
 	};
} __uart_txfll_reg_bits;

/* UARTn_RESUME_REG */
typedef union {
	/* UARTx_RESUME_REG */
	struct {
 	__REG16 RESUME								: 8;
 	__REG16 											: 8;
 	};
	/* UARTx_TXFLH_REG */
	struct {
 	__REG16 TXFLH									: 5;
 	__REG16 											:11;
 	};
} __uart_resume_reg_bits;

/* UARTn_RXFLL_REG */
typedef union {
	/* UARTx_RXFLL_REG */
	struct {
 	__REG16 RXFLL									: 8;
 	__REG16 											: 8;
 	};
	/* UARTx_SFREGL_REG */
	struct {
 	__REG16 SFREGL								: 8;
 	__REG16 											: 8;
 	};
} __uart_rxfll_reg_bits;

/* UARTn_RXFLH_REG */
typedef union {
	/* UARTx_RXFLH_REG */
	struct {
 	__REG16 RXFLH									: 4;
 	__REG16 											:12;
 	};
	/* UARTx_SFREGH_REG */
	struct {
 	__REG16 SFREGH								: 4;
 	__REG16 											:12;
 	};
} __uart_rxflh_reg_bits;

/* UARTn_BLR_REG */
typedef union {
	/* UARTx_BLR_REG */
	struct {
 	__REG16 											: 6;
 	__REG16 XBOF_TYPE							: 1;
 	__REG16 STS_FIFO_RESET				: 1;
 	__REG16 											: 8;
 	};
	/* UARTx_UASR_REG */
	struct {
 	__REG16 SPEED									: 5;
 	__REG16 BIT_BY_CHAR						: 1;
 	__REG16 PARITY_TYPE						: 2;
 	__REG16 											: 8;
 	};
} __uart_blr_reg_bits;

/* UARTn_ACREG_REG */
typedef struct {
 	__REG16 EOT_EN								: 1;
 	__REG16 ABORT_EN							: 1;
 	__REG16 SCTX_EN								: 1;
 	__REG16 SEND_SIP							: 1;
 	__REG16 DIS_TX_UNDERRUN				: 1;
 	__REG16 DIS_IR_RX							: 1;
 	__REG16 SD_MOD								: 1;
 	__REG16 PULSE_TYPE						: 1;
 	__REG16 											: 8;
} __uart_acreg_reg_bits;

/* UARTn_SCR_REG */
typedef struct {
 	__REG16 DMA_MODE_CTL					: 1;
 	__REG16 DMA_MODE_2						: 2;
 	__REG16 TX_EMPTY_CTL_IT				: 1;
 	__REG16 RX_CTS_WU_EN					: 1;
 	__REG16 											: 1;
 	__REG16 TX_TRIG_GRANU1				: 1;
 	__REG16 RX_TRIG_GRANU1				: 1;
 	__REG16 											: 8;
} __uart_scr_reg_bits;

/* UARTn_SSR_REG */
typedef struct {
 	__REG16 TX_FIFO_FULL					: 1;
 	__REG16 RX_CTS_WU_STS					: 1;
 	__REG16 											:14;
} __uart_ssr_reg_bits;

/* UARTn_EBLR_REG */
typedef struct {
 	__REG16 EBLR									: 8;
 	__REG16 											: 8;
} __uart_eblr_reg_bits;

/* UARTn_SYSC_REG */
typedef struct {
 	__REG16 AUTOIDLE							: 1;
 	__REG16 SOFTRESET							: 1;
 	__REG16 ENAWAKEUP							: 1;
 	__REG16 IDLEMODE							: 2;
 	__REG16 											:11;
} __uart_sysc_reg_bits;

/* UARTn_SYSS_REG */
typedef struct {
 	__REG16 RESETDONE							: 1;
 	__REG16 											:15;
} __uart_syss_reg_bits;

/* UARTn_WER_REG */
typedef struct {
 	__REG16 EVENT_0_CTS_ACTIVITY	: 1;
 	__REG16 											: 1;
 	__REG16 EVENT_2_RI_ACTIVITY		: 1;
 	__REG16 											: 1;
 	__REG16 EVENT_4_RX_ACTIVITY		: 1;
 	__REG16 EVENT_5_RHR_INTERRUPT	: 1;
 	__REG16 EVENT_6_RLS_INTERRUPT	: 1;
 	__REG16 											: 9;
} __uart_wer_reg_bits;

/* UARTn_CFPS_REG */
typedef struct {
 	__REG16 CFPS									: 8;
 	__REG16 											: 8;
} __uart_cfps_reg_bits;

/* USBTLL_REVISION */
typedef struct {
 	__REG32 MINOR									: 4;
 	__REG32 MAJOR									: 4;
 	__REG32 											:24;
} __usbtll_revision_bits;

/* USBTLL_SYSCONFIG */
typedef struct {
 	__REG32 AUTOIDLE							: 1;
 	__REG32 SOFTRESET							: 1;
 	__REG32 ENAWAKEUP							: 1;
 	__REG32 SIDLEMODE							: 2;
 	__REG32 											: 3;
 	__REG32 CLOCKACTIVITY					: 1;
 	__REG32 											:23;
} __usbtll_sysconfig_bits;

/* USBTLL_SYSSTATUS */
typedef struct {
 	__REG32 RESETDONE							: 1;
 	__REG32 											:31;
} __usbtll_sysstatus_bits;

/* USBTLL_IRQSTATUS */
typedef struct {
 	__REG32 FCLK_START						: 1;
 	__REG32 FCLK_END							: 1;
 	__REG32 ACCESS_ERROR					: 1;
 	__REG32 											:29;
} __usbtll_irqstatus_bits;

/* USBTLL_IRQENABLE */
typedef struct {
 	__REG32 FCLK_START_EN					: 1;
 	__REG32 FCLK_END_EN						: 1;
 	__REG32 ACCESS_ERROR_EN				: 1;
 	__REG32 											:29;
} __usbtll_irqenable_bits;

/* TLL_SHARED_CONF */
typedef struct {
 	__REG32 FCLK_IS_ON						: 1;
 	__REG32 FCLK_REQ							: 1;
 	__REG32 USB_DIVRATIO					: 3;
 	__REG32 USB_180D_SDR_EN				: 1;
 	__REG32 USB_90D_DDR_EN				: 1;
 	__REG32 											:25;
} __tll_shared_conf_bits;

/* TLL_CHANNEL_CONF_i */
typedef struct {
 	__REG32 CHANEN								: 1;
 	__REG32 CHANMODE							: 2;
 	__REG32 UTMIISADEV						: 1;
 	__REG32 TLLATTACH							: 1;
 	__REG32 TLLCONNECT						: 1;
 	__REG32 TLLFULLSPEED					: 1;
 	__REG32 ULPIOUTCLKMODE				: 1;
 	__REG32 ULPIDDRMODE						: 1;
 	__REG32 UTMIAUTOIDLE					: 1;
 	__REG32 ULPIAUTOIDLE					: 1;
 	__REG32 ULPINOBITSTUFF				: 1;
 	__REG32 											: 3;
 	__REG32 CHRGVBUS							: 1;
 	__REG32 DRVVBUS								: 1;
 	__REG32 TESTEN								: 1;
 	__REG32 TESTTXEN							: 1;
 	__REG32 TESTTXDAT							: 1;
 	__REG32 TESTTXSE0							: 1;
 	__REG32 											: 3;
 	__REG32 FSLSMODE							: 4;
 	__REG32 FSLSLINESTATE					: 2;
 	__REG32 											: 2;
} __tll_channel_conf_bits;

/* ULPI_FUNCTION_CTRL_i */
typedef struct {
 	__REG8  XCVRSELECT						: 2;
 	__REG8  TERMSELECT						: 1;
 	__REG8  OPMODE								: 2;
 	__REG8  RESET									: 1;
 	__REG8  SUSPENDM							: 1;
 	__REG8  											: 1;
} __ulpi_function_ctrl_bits;

/* ULPI_INTERFACE_CTRL_i */
typedef struct {
 	__REG8  FSLSSERIALMODE_6PIN				: 1;
 	__REG8  FSLSSERIALMODE_3PIN				: 1;
 	__REG8  													: 1;
 	__REG8  CLOCKSUSPENDM							: 1;
 	__REG8  AUTORESUME								: 1;
 	__REG8  													: 2;
 	__REG8  INTERFACE_PROTECT_DISABLE	: 1;
} __ulpi_interface_ctrl_bits;

/* ULPI_OTG_CTRL_i */
typedef struct {
 	__REG8  IDPULLUP									: 1;
 	__REG8  DPPULLDOWN								: 1;
 	__REG8  DMPULLDOWN								: 1;
 	__REG8  DISCHRGVBUS								: 1;
 	__REG8  CHRGVBUS									: 1;
 	__REG8  DRVVBUS										: 1;
 	__REG8  													: 2;
} __ulpi_otg_ctrl_bits;

/* ULPI_USB_INT_EN_RISE_i */
typedef struct {
 	__REG8  HOSTDISCONNECT_RISE				: 1;
 	__REG8  VBUSVALID_RISE						: 1;
 	__REG8  SESSVALID_RISE						: 1;
 	__REG8  SESSEND_RISE							: 1;
 	__REG8  IDGND_RISE								: 1;
 	__REG8  													: 3;
} __ulpi_usb_int_en_rise_bits;

/* ULPI_USB_INT_EN_FALL_i */
typedef struct {
 	__REG8  HOSTDISCONNECT_FALL				: 1;
 	__REG8  VBUSVALID_FALL						: 1;
 	__REG8  SESSVALID_FALL						: 1;
 	__REG8  SESSEND_FALL							: 1;
 	__REG8  IDGND_FALL								: 1;
 	__REG8  													: 3;
} __ulpi_usb_int_en_fall_bits;

/* ULPI_USB_INT_STATUS_i */
typedef struct {
 	__REG8  HOSTDISCONNECT						: 1;
 	__REG8  VBUSVALID									: 1;
 	__REG8  SESSVALID									: 1;
 	__REG8  SESSEND										: 1;
 	__REG8  IDGND											: 1;
 	__REG8  													: 3;
} __ulpi_usb_int_status_bits;

/* ULPI_USB_INT_LATCH_i */
typedef struct {
 	__REG8  HOSTDISCONNECT_LATCH			: 1;
 	__REG8  VBUSVALID_LATCH						: 1;
 	__REG8  SESSVALID_LATCH						: 1;
 	__REG8  SESSEND_LATCH							: 1;
 	__REG8  IDGND_LATCH								: 1;
 	__REG8  													: 3;
} __ulpi_usb_int_latch_bits;

/* ULPI_DEBUG_i */
typedef struct {
 	__REG8  LINESTATE									: 2;
 	__REG8  													: 6;
} __ulpi_debug_bits;

/* ULPI_UTMI_VCONTROL_EN_i */
typedef struct {
 	__REG8  VC0_EN										: 1;
 	__REG8  VC1_EN										: 1;
 	__REG8  VC2_EN										: 1;
 	__REG8  VC3_EN										: 1;
 	__REG8  VC4_EN										: 1;
 	__REG8  VC5_EN										: 1;
 	__REG8  VC6_EN										: 1;
 	__REG8  VC7_EN										: 1;
} __ulpi_utmi_vcontrol_en_bits;

/* ULPI_UTMI_VCONTROL_LATCH_i */
typedef struct {
 	__REG8  VC0_CHANGE								: 1;
 	__REG8  VC1_CHANGE								: 1;
 	__REG8  VC2_CHANGE								: 1;
 	__REG8  VC3_CHANGE								: 1;
 	__REG8  VC4_CHANGE								: 1;
 	__REG8  VC5_CHANGE								: 1;
 	__REG8  VC6_CHANGE								: 1;
 	__REG8  VC7_CHANGE								: 1;
} __ulpi_utmi_vcontrol_latch_bits;

/* ULPI_VENDOR_INT_EN_i */
typedef struct {
 	__REG8  P2P_EN										: 1;
 	__REG8  													: 7;
} __ulpi_vendor_int_en_bits;

/* ULPI_VENDOR_INT_STATUS_i */
typedef struct {
 	__REG8  UTMI_SUSPENDM							: 1;
 	__REG8  													: 7;
} __ulpi_vendor_int_status_bits;

/* ULPI_VENDOR_INT_LATCH_i */
typedef struct {
 	__REG8  P2P_LATCH									: 1;
 	__REG8  													: 7;
} __ulpi_vendor_int_latch_bits;

/* UHH_REVISION */
typedef struct {
 	__REG32 MIN_REV								: 4;
 	__REG32 MAJ_REV								: 4;
 	__REG32 											:24;
} __uhh_revision_bits;

/* UHH_SYSCONFIG */
typedef struct {
 	__REG32 AUTOIDLE							: 1;
 	__REG32 SOFTRESET							: 1;
 	__REG32 ENAWAKEUP							: 1;
 	__REG32 SIDLEMODE							: 2;
 	__REG32 											: 3;
 	__REG32 CLOCKACTIVITY					: 1;
 	__REG32 											: 3;
 	__REG32 MIDLEMODE							: 2;
 	__REG32 											:18;
} __uhh_sysconfig_bits;

/* UHH_SYSSTATUS */
typedef struct {
 	__REG32 RESETDONE							: 1;
 	__REG32 OHCI_RESETDONE				: 1;
 	__REG32 EHCI_RESETDONE				: 1;
 	__REG32 											:29;
} __uhh_sysstatus_bits;

/* UHH_HOSTCONFIG */
typedef struct {
 	__REG32 P1_ULPI_BYPASS				: 1;
 	__REG32 											: 1;
 	__REG32 ENA_INCR4							: 1;
 	__REG32 ENA_INCR8							: 1;
 	__REG32 ENA_INCR16						: 1;
 	__REG32 ENA_INCR_ALIGN				: 1;
 	__REG32 											: 2;
 	__REG32 P1_CONNECT_STATUS			: 1;
 	__REG32 P2_CONNECT_STATUS			: 1;
 	__REG32 P3_CONNECT_STATUS			: 1;
 	__REG32 											:21;
} __uhh_hostconfig_bits;

/* UHH_DEBUG_CSR */
typedef struct {
 	__REG32 EHCI_FLADJ						: 6;
 	__REG32 EHCI_SIMULATION_MODE	: 1;
 	__REG32 OCHI_CNTSEL						: 1;
 	__REG32 											: 8;
 	__REG32 OHCI_GLOBALSUSPEND		: 1;
 	__REG32 OHCI_CCS_1						: 1;
 	__REG32 OHCI_CCS_2						: 1;
 	__REG32 OHCI_CCS_3						: 1;
 	__REG32 											:12;
} __uhh_debug_csr_bits;

/* HCREVISION */
typedef struct {
 	__REG32 REV										: 8;
 	__REG32 											:24;
} __hcrevision_bits;

/* HCCONTROL */
typedef struct {
 	__REG32 CBSR									: 2;
 	__REG32 PLE										: 1;
 	__REG32 IE										: 1;
 	__REG32 CLE										: 1;
 	__REG32 BLE										: 1;
 	__REG32 HCFS									: 2;
 	__REG32 IR										: 1;
 	__REG32 RWC										: 1;
 	__REG32 RWE										: 1;
 	__REG32 											:21;
} __hccontrol_bits;

/* HCCOMMANDSTATUS */
typedef struct {
 	__REG32 HCR										: 1;
 	__REG32 CLF										: 1;
 	__REG32 BLF										: 1;
 	__REG32 OCR										: 1;
 	__REG32 											:12;
 	__REG32 SOC										: 2;
 	__REG32 											:14;
} __hccommandstatus_bits;

/* HCINTERRUPTSTATUS */
typedef struct {
 	__REG32 SO										: 1;
 	__REG32 WDH										: 1;
 	__REG32 SF										: 1;
 	__REG32 RD										: 1;
 	__REG32 UE										: 1;
 	__REG32 FNO										: 1;
 	__REG32 RHSC									: 1;
 	__REG32 											:23;
 	__REG32 OC										: 1;
 	__REG32 											: 1;
} __hcinterruptstatus_bits;

/* HCINTERRUPTENABLE */
typedef struct {
 	__REG32 SO										: 1;
 	__REG32 WDH										: 1;
 	__REG32 SF										: 1;
 	__REG32 RD										: 1;
 	__REG32 UE										: 1;
 	__REG32 FNO										: 1;
 	__REG32 RHSC									: 1;
 	__REG32 											:23;
 	__REG32 OC										: 1;
 	__REG32 MIE										: 1;
} __hcinterruptenable_bits;

/* HCFMINTERVAL */
typedef struct {
 	__REG32 FI										:14;
 	__REG32 											: 2;
 	__REG32 FSMPS									:15;
 	__REG32 FIT										: 1;
} __hcfminterval_bits;

/* HCFMREMAINING */
typedef struct {
 	__REG32 FR										:14;
 	__REG32 											:17;
 	__REG32 FIT										: 1;
} __hcfmremaining_bits;

/* HCFMNUMBER */
typedef struct {
 	__REG32 FN										:16;
 	__REG32 											:16;
} __hcfmnumber_bits;

/* HCPERIODICSTART */
typedef struct {
 	__REG32 PS										:14;
 	__REG32 											:18;
} __hcperiodicstart_bits;

/* HCLSTHRESHOLD */
typedef struct {
 	__REG32 LST										:12;
 	__REG32 											:20;
} __hclsthreshold_bits;

/* HCRHDESCRIPTORA */
typedef struct {
 	__REG32 NDP										: 8;
 	__REG32 PSM										: 1;
 	__REG32 NPS										: 1;
 	__REG32 DT										: 1;
 	__REG32 											:13;
 	__REG32 POTPG     						: 8;
} __hcrhdescriptora_bits;

/* HCRHDESCRIPTORB */
typedef struct {
 	__REG32 DR										:16;
 	__REG32 PPCM									:16;
} __hcrhdescriptorb_bits;

/* HCRHSTATUS */
typedef struct {
 	__REG32 LPS										: 1;
 	__REG32 											:14;
 	__REG32 DRWE									: 1;
 	__REG32 LPSC									: 1;
 	__REG32 											:14;
 	__REG32 CRWE									: 1;
} __hcrhstatus_bits;

/* HCRHPORTSTATUS_n */
typedef struct {
 	__REG32 CCS_CPE								: 1;
 	__REG32 PES_SPE								: 1;
 	__REG32 PSS_SPS								: 1;
 	__REG32 											: 1;
 	__REG32 PRS_SPR								: 1;
 	__REG32 											: 3;
 	__REG32 PPS_SPP								: 1;
 	__REG32 LSDA_CPP							: 1;
 	__REG32 											: 6;
 	__REG32 CSC										: 1;
 	__REG32 PESC									: 1;
 	__REG32 PSSC									: 1;
 	__REG32 											: 1;
 	__REG32 PRSC									: 1;
 	__REG32 											:11;
} __hcrhportstatus_bits;

/* HCCAPBASE */
typedef struct {
 	__REG32 CAPLENGTH							: 8;
 	__REG32 											: 8;
 	__REG32 HCIVERSION						:16;
} __ehci_hccapbase_bits;

/* HCSPARAMS */
typedef struct {
 	__REG32 N_PORTS								: 4;
 	__REG32 PPC										: 1;
 	__REG32 											: 2;
 	__REG32 PRR										: 1;
 	__REG32 N_PCC									: 4;
 	__REG32 N_CC									: 4;
 	__REG32 P_INDICATOR						: 1;
 	__REG32 											:15;
} __ehci_hcsparams_bits;

/* HCCPARAMS */
typedef struct {
 	__REG32 BIT64AC								: 1;
 	__REG32 PFLF									: 1;
 	__REG32 ASPC									: 1;
 	__REG32 											: 1;
 	__REG32 IST										: 4;
 	__REG32 EECP									: 8;
 	__REG32 											:16;
} __ehci_hccparams_bits;

/* EHCI_USBCMD */
typedef struct {
 	__REG32 RS										: 1;
 	__REG32 HCR										: 1;
 	__REG32 FLS										: 2;
 	__REG32 PSE										: 1;
 	__REG32 ASE										: 1;
 	__REG32 IAAD									: 1;
 	__REG32 LHCR									: 1;
 	__REG32 ASPMC									: 2;
 	__REG32 											: 1;
 	__REG32 ASPME									: 1;
 	__REG32 											: 4;
 	__REG32 ITC										: 8;
 	__REG32 											: 8;
} __ehci_usbcmd_bits;

/* EHCI_USBSTS */
typedef struct {
 	__REG32 USBI									: 1;
 	__REG32 USBEI									: 1;
 	__REG32 PCD										: 1;
 	__REG32 FLR										: 1;
 	__REG32 HSE										: 1;
 	__REG32 IAA										: 1;
 	__REG32 											: 6;
 	__REG32 HCH										: 1;
 	__REG32 REC										: 1;
 	__REG32 PSS										: 1;
 	__REG32 ASS										: 1;
 	__REG32 											:16;
} __ehci_usbsts_bits;

/* EHCI_USBINTR */
typedef struct {
 	__REG32 USBIE									: 1;
 	__REG32 USBEIE								: 1;
 	__REG32 PCIE									: 1;
 	__REG32 FLRE									: 1;
 	__REG32 HSEE									: 1;
 	__REG32 IAAE									: 1;
 	__REG32 											:26;
} __ehci_usbintr_bits;

/* EHCI_FRINDEX */
typedef struct {
 	__REG32 FI										:14;
 	__REG32 											:18;
} __ehci_frindex_bits;

/* EHCI_CONFIGFLAG */
typedef struct {
 	__REG32 CF										: 1;
 	__REG32 											:31;
} __ehci_configflag_bits;

/* EHCI_PORTSC_i */
typedef struct {
 	__REG32 CCS										: 1;
 	__REG32 CSC										: 1;
 	__REG32 PED										: 1;
 	__REG32 PEDC									: 1;
 	__REG32 											: 2;
 	__REG32 FPR										: 1;
 	__REG32 SUS										: 1;
 	__REG32 PR										: 1;
 	__REG32 											: 1;
 	__REG32 LS										: 2;
 	__REG32 PP										: 1;
 	__REG32 PO										: 1;
 	__REG32 PIC										: 2;
 	__REG32 PTC										: 4;
 	__REG32 WCE										: 1;
 	__REG32 WDE										: 1;
 	__REG32 											:10;
} __ehci_portsc_bits;

/* EHCI_INSNREG00 */
typedef struct {
 	__REG32 EN										: 1;
 	__REG32 UFRAME_CNT						:13;
 	__REG32 											:18;
} __ehci_insnreg00_bits;

/* EHCI_INSNREG01 */
typedef struct {
 	__REG32 IN_THRESHOLD					:16;
 	__REG32 OUT_THRESHOLD					:16;
} __ehci_insnreg01_bits;

/* EHCI_INSNREG02 */
typedef struct {
 	__REG32 BUF_DEPTH							:12;
 	__REG32 											:20;
} __ehci_insnreg02_bits;

/* EHCI_INSNREG03 */
typedef struct {
 	__REG32 BRK_MEM_TRSF					: 1;
 	__REG32 											:31;
} __ehci_insnreg03_bits;

/* EHCI_INSNREG04 */
typedef struct {
 	__REG32 HCSPARAMS_WRE					: 1;
 	__REG32 HCCPARAMS_WRE					: 1;
 	__REG32 SHORT_PORT_ENUM				: 1;
 	__REG32 											: 1;
 	__REG32 NAK_FIX_DIS						: 1;
 	__REG32 											:27;
} __ehci_insnreg04_bits;

/* EHCI_INSNREG04 */
typedef union {
	/*EHCI_INSNREG05_UTMI*/
	struct {
 	__REG32 VSTATUS								: 8;
 	__REG32 VCONTROL							: 4;
 	__REG32 VCONTROLLOADM					: 1;
 	__REG32 VPORT									: 4;
 	__REG32 VBUSY									: 1;
 	__REG32 											:14;
 	};
	/*EHCI_INSNREG05_ULPI*/
	struct {
 	__REG32 WRDATA								: 8;
 	__REG32 EXTREGADD							: 8;
 	__REG32 REGADD  							: 6;
 	__REG32 OPSEL									: 2;
 	__REG32 PORTSEL								: 4;
 	__REG32 											: 3;
 	__REG32 CONTROL								: 1;
 	};
} __ehci_insnreg05_utmi_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */

/* Common declarations  ****************************************************/
/***************************************************************************
 **
 ** DID
 **
 ***************************************************************************/
__IO_REG32_BIT(CONTROL_IDCODE,      	0x4830A204,__READ       ,__control_idcode_bits);
__IO_REG32(		 DIE_ID_0,      				0x4830A218,__READ       );
__IO_REG32(		 DIE_ID_1,      				0x4830A21C,__READ       );
__IO_REG32(		 DIE_ID_2,      				0x4830A220,__READ       );
__IO_REG32(		 DIE_ID_3,      				0x4830A224,__READ       );

/***************************************************************************
 **
 ** CM
 **
 ***************************************************************************/
__IO_REG32_BIT(CM_REVISION,          	0x48004800,__READ       ,__cm_revision_bits);
__IO_REG32_BIT(CM_SYSCONFIG,         	0x48004810,__READ_WRITE ,__cm_sysconfig_bits);
__IO_REG32_BIT(CM_CLKEN_PLL_MPU,     	0x48004904,__READ_WRITE ,__cm_clken_pll_mpu_bits);
__IO_REG32_BIT(CM_IDLEST_MPU,     		0x48004920,__READ				,__cm_idlest_mpu_bits);
__IO_REG32_BIT(CM_IDLEST_PLL_MPU,     0x48004924,__READ				,__cm_idlest_pll_mpu_bits);
__IO_REG32_BIT(CM_AUTOIDLE_PLL_MPU,   0x48004934,__READ_WRITE ,__cm_autoidle_pll_mpu_bits);
__IO_REG32_BIT(CM_CLKSEL1_PLL_MPU,    0x48004940,__READ_WRITE ,__cm_clksel1_pll_mpu_bits);
__IO_REG32_BIT(CM_CLKSEL2_PLL_MPU,    0x48004944,__READ_WRITE ,__cm_clksel2_pll_mpu_bits);
__IO_REG32_BIT(CM_CLKSTCTRL_MPU,     	0x48004948,__READ_WRITE ,__cm_clkstctrl_mpu_bits);
__IO_REG32_BIT(CM_CLKSTST_MPU,     		0x4800494C,__READ				,__cm_clkstst_mpu_bits);
__IO_REG32_BIT(CM_FCLKEN1_CORE,     	0x48004A00,__READ_WRITE ,__cm_fclken1_core_bits);
__IO_REG32_BIT(CM_FCLKEN3_CORE,     	0x48004A08,__READ_WRITE ,__cm_fclken3_core_bits);
__IO_REG32_BIT(CM_ICLKEN1_CORE,     	0x48004A10,__READ_WRITE ,__cm_iclken1_core_bits);
//__IO_REG32_BIT(CM_ICLKEN2_CORE,     	0x48004A14,__READ_WRITE ,__cm_iclken2_core_bits);
__IO_REG32_BIT(CM_ICLKEN3_CORE,     	0x48004A18,__READ_WRITE ,__cm_iclken3_core_bits);
__IO_REG32_BIT(CM_IDLEST1_CORE,     	0x48004A20,__READ				,__cm_idlest1_core_bits);
//__IO_REG32_BIT(CM_IDLEST2_CORE,     	0x48004A24,__READ				,__cm_idlest2_core_bits);
__IO_REG32_BIT(CM_IDLEST3_CORE,     	0x48004A28,__READ				,__cm_idlest3_core_bits);
__IO_REG32_BIT(CM_AUTOIDLE1_CORE,     0x48004A30,__READ_WRITE ,__cm_autoidle1_core_bits);
//__IO_REG32_BIT(CM_AUTOIDLE2_CORE,     0x48004A34,__READ_WRITE ,__cm_autoidle2_core_bits);
__IO_REG32_BIT(CM_AUTOIDLE3_CORE,     0x48004A38,__READ_WRITE ,__cm_autoidle3_core_bits);
__IO_REG32_BIT(CM_CLKSEL_CORE,     		0x48004A40,__READ_WRITE ,__cm_clksel_core_bits);
__IO_REG32_BIT(CM_CLKSTCTRL_CORE,     0x48004A48,__READ_WRITE ,__cm_clkstctrl_core_bits);
__IO_REG32_BIT(CM_CLKSTST_CORE,     	0x48004A4C,__READ_WRITE ,__cm_clkstst_core_bits);
__IO_REG32_BIT(CM_FCLKEN_SGX,     		0x48004B00,__READ_WRITE ,__cm_fclken_sgx_bits);
__IO_REG32_BIT(CM_ICLKEN_SGX,     		0x48004B10,__READ_WRITE ,__cm_iclken_sgx_bits);
__IO_REG32_BIT(CM_IDLEST_SGX,     		0x48004B20,__READ				,__cm_idlest_sgx_bits);
__IO_REG32_BIT(CM_CLKSEL_SGX,     		0x48004B40,__READ_WRITE ,__cm_clksel_sgx_bits);
__IO_REG32_BIT(CM_SLEEPDEP_SGX,     	0x48004B44,__READ_WRITE ,__cm_sleepdep_sgx_bits);
__IO_REG32_BIT(CM_CLKSTCTRL_SGX,     	0x48004B48,__READ_WRITE ,__cm_clkstctrl_sgx_bits);
__IO_REG32_BIT(CM_CLKSTST_SGX,     		0x48004B4C,__READ_WRITE ,__cm_clkstst_sgx_bits);
__IO_REG32_BIT(CM_FCLKEN_WKUP,     		0x48004C00,__READ_WRITE ,__cm_fclken_wkup_bits);
__IO_REG32_BIT(CM_ICLKEN_WKUP,     		0x48004C10,__READ_WRITE ,__cm_iclken_wkup_bits);
__IO_REG32_BIT(CM_IDLEST_WKUP,     		0x48004C20,__READ				,__cm_idlest_wkup_bits);
__IO_REG32_BIT(CM_AUTOIDLE_WKUP,     	0x48004C30,__READ_WRITE ,__cm_autoidle_wkup_bits);
__IO_REG32_BIT(CM_CLKSEL_WKUP,     		0x48004C40,__READ_WRITE ,__cm_clksel_wkup_bits);
__IO_REG32_BIT(CM_CLKEN_PLL,     			0x48004D00,__READ_WRITE ,__cm_clken_pll_bits);
__IO_REG32_BIT(CM_CLKEN2_PLL,     		0x48004D04,__READ_WRITE ,__cm_clken2_pll_bits);
__IO_REG32_BIT(CM_IDLEST_CKGEN,     	0x48004D20,__READ				,__cm_idlest_ckgen_bits);
__IO_REG32_BIT(CM_IDLEST2_CKGEN,     	0x48004D24,__READ				,__cm_idlest2_ckgen_bits);
__IO_REG32_BIT(CM_AUTOIDLE_PLL,     	0x48004D30,__READ_WRITE ,__cm_autoidle_pll_bits);
__IO_REG32_BIT(CM_AUTOIDLE2_PLL,     	0x48004D34,__READ_WRITE ,__cm_autoidle2_pll_bits);
__IO_REG32_BIT(CM_CLKSEL1_PLL,     		0x48004D40,__READ_WRITE ,__cm_clksel1_pll_bits);
__IO_REG32_BIT(CM_CLKSEL2_PLL,     		0x48004D44,__READ_WRITE ,__cm_clksel2_pll_bits);
__IO_REG32_BIT(CM_CLKSEL3_PLL,     		0x48004D48,__READ_WRITE ,__cm_clksel3_pll_bits);
__IO_REG32_BIT(CM_CLKSEL4_PLL,     		0x48004D4C,__READ_WRITE ,__cm_clksel4_pll_bits);
__IO_REG32_BIT(CM_CLKSEL5_PLL,     		0x48004D50,__READ_WRITE ,__cm_clksel5_pll_bits);
__IO_REG32_BIT(CM_CLKOUT_CTRL,     		0x48004D70,__READ_WRITE ,__cm_clkout_ctrl_bits);
__IO_REG32_BIT(CM_FCLKEN_DSS,     		0x48004E00,__READ_WRITE ,__cm_fclken_dss_bits);
__IO_REG32_BIT(CM_ICLKEN_DSS,     		0x48004E10,__READ_WRITE ,__cm_iclken_dss_bits);
__IO_REG32_BIT(CM_IDLEST_DSS,     		0x48004E20,__READ				,__cm_idlest_dss_bits);
__IO_REG32_BIT(CM_AUTOIDLE_DSS,     	0x48004E30,__READ_WRITE ,__cm_autoidle_dss_bits);
__IO_REG32_BIT(CM_CLKSEL_DSS,     		0x48004E40,__READ_WRITE ,__cm_clksel_dss_bits);
__IO_REG32_BIT(CM_SLEEPDEP_DSS,     	0x48004E44,__READ_WRITE ,__cm_sleepdep_dss_bits);
__IO_REG32_BIT(CM_CLKSTCTRL_DSS,     	0x48004E48,__READ_WRITE ,__cm_clkstctrl_dss_bits);
__IO_REG32_BIT(CM_CLKSTST_DSS,     		0x48004E4C,__READ				,__cm_clkstst_dss_bits);
__IO_REG32_BIT(CM_FCLKEN_PER,     		0x48005000,__READ_WRITE ,__cm_fclken_per_bits);
__IO_REG32_BIT(CM_ICLKEN_PER,     		0x48005010,__READ_WRITE ,__cm_fclken_per_bits);
__IO_REG32_BIT(CM_IDLEST_PER,     		0x48005020,__READ				,__cm_idlest_per_bits);
__IO_REG32_BIT(CM_AUTOIDLE_PER,     	0x48005030,__READ_WRITE ,__cm_autoidle_per_bits);
__IO_REG32_BIT(CM_CLKSEL_PER,     		0x48005040,__READ_WRITE ,__cm_clksel_per_bits);
__IO_REG32_BIT(CM_SLEEPDEP_PER,     	0x48005044,__READ_WRITE ,__cm_sleepdep_per_bits);
__IO_REG32_BIT(CM_CLKSTCTRL_PER,     	0x48005048,__READ_WRITE ,__cm_clkstctrl_per_bits);
__IO_REG32_BIT(CM_CLKSTST_PER,     		0x4800504C,__READ				,__cm_clkstst_per_bits);
__IO_REG32_BIT(CM_CLKSEL1_EMU,     		0x48005140,__READ_WRITE ,__cm_clksel1_emu_bits);
__IO_REG32_BIT(CM_CLKSTCTRL_EMU,     	0x48005148,__READ_WRITE ,__cm_clkstctrl_emu_bits);
__IO_REG32_BIT(CM_CLKSTST_EMU,     		0x4800514C,__READ				,__cm_clkstst_emu_bits);
__IO_REG32_BIT(CM_CLKSEL2_EMU,     		0x48005150,__READ_WRITE ,__cm_clksel2_emu_bits);
__IO_REG32_BIT(CM_CLKSEL3_EMU,     		0x48005154,__READ_WRITE ,__cm_clksel3_emu_bits);
__IO_REG32_BIT(CM_POLCTRL,     				0x4800529C,__READ_WRITE ,__cm_polctrl_bits);
__IO_REG32_BIT(CM_IDLEST_NEON,     		0x48005320,__READ				,__cm_idlest_neon_bits);
__IO_REG32_BIT(CM_CLKSTCTRL_NEON,     0x48005348,__READ_WRITE ,__cm_clkstctrl_neon_bits);
__IO_REG32_BIT(CM_FCLKEN_USBHOST,     0x48005400,__READ_WRITE ,__cm_fclken_usbhost_bits);
__IO_REG32_BIT(CM_ICLKEN_USBHOST,     0x48005410,__READ_WRITE ,__cm_iclken_usbhost_bits);
__IO_REG32_BIT(CM_IDLEST_USBHOST,     0x48005420,__READ				,__cm_idlest_usbhost_bits);
__IO_REG32_BIT(CM_AUTOIDLE_USBHOST,   0x48005430,__READ_WRITE ,__cm_autoidle_usbhost_bits);
__IO_REG32_BIT(CM_SLEEPDEP_USBHOST,   0x48005444,__READ_WRITE ,__cm_sleepdep_usbhost_bits);
__IO_REG32_BIT(CM_CLKSTCTRL_USBHOST,  0x48005448,__READ_WRITE ,__cm_clkstctrl_usbhost_bits);
__IO_REG32_BIT(CM_CLKSTST_USBHOST,    0x4800544C,__READ				,__cm_clkstst_usbhost_bits);

/***************************************************************************
 **
 ** PR
 **
 ***************************************************************************/
__IO_REG32_BIT(PRM_REVISION,          0x48306804,__READ       ,__prm_revision_bits);
__IO_REG32_BIT(PRM_SYSCONFIG,         0x48306814,__READ_WRITE ,__prm_sysconfig_bits);
__IO_REG32_BIT(PRM_IRQSTATUS_MPU,     0x48306818,__READ_WRITE ,__prm_irqstatus_mpu_bits);
__IO_REG32_BIT(PRM_IRQENABLE_MPU,     0x4830681C,__READ_WRITE ,__prm_irqenable_mpu_bits);
__IO_REG32_BIT(RM_RSTST_MPU,     			0x48306958,__READ_WRITE ,__rm_rstst_mpu_bits);
__IO_REG32_BIT(PM_WKDEP_MPU,     			0x483069C8,__READ_WRITE ,__pm_wkdep_mpu_bits);
__IO_REG32_BIT(PM_EVGENCTRL_MPU,     	0x483069D4,__READ_WRITE ,__pm_evgenctrl_mpu_bits);
__IO_REG32(		 PM_EVGENONTIM_MPU,    	0x483069D8,__READ_WRITE );
__IO_REG32(		 PM_EVGENOFFTIM_MPU,    0x483069DC,__READ_WRITE );
__IO_REG32_BIT(PM_PWSTCTRL_MPU,     	0x483069E0,__READ_WRITE ,__pm_pwstctrl_mpu_bits);
__IO_REG32_BIT(PM_PWSTST_MPU,     		0x483069E4,__READ				,__pm_pwstst_mpu_bits);
__IO_REG32_BIT(PM_PREPWSTST_MPU,     	0x483069E8,__READ_WRITE ,__pm_prepwstst_mpu_bits);
__IO_REG32_BIT(RM_RSTST_CORE,     		0x48306A58,__READ_WRITE ,__rm_rstst_core_bits);
__IO_REG32_BIT(PM_WKEN1_CORE,     		0x48306AA0,__READ_WRITE ,__pm_wken1_core_bits);
__IO_REG32_BIT(PM_MPUGRPSEL1_CORE,    0x48306AA4,__READ_WRITE ,__pm_mpugrpsel1_core_bits);
__IO_REG32_BIT(PM_WKST1_CORE,     		0x48306AB0,__READ_WRITE ,__pm_wkst1_core_bits);
__IO_REG32_BIT(PM_WKST3_CORE,     		0x48306AB8,__READ_WRITE ,__pm_wkst3_core_bits);
__IO_REG32_BIT(PM_PWSTCTRL_CORE,     	0x48306AE0,__READ_WRITE ,__pm_pwstctrl_core_bits);
__IO_REG32_BIT(PM_PWSTST_CORE,     		0x48306AE4,__READ				,__pm_pwstst_core_bits);
__IO_REG32_BIT(PM_PREPWSTST_CORE,    	0x48306AE8,__READ_WRITE ,__pm_prepwstst_core_bits);
__IO_REG32_BIT(PM_WKEN3_CORE,     		0x48306AF0,__READ_WRITE ,__pm_wken3_core_bits);
__IO_REG32_BIT(PM_MPUGRPSEL3_CORE,    0x48306AF8,__READ_WRITE ,__pm_mpugrpsel3_core_bits);
__IO_REG32_BIT(RM_RSTST_SGX,    			0x48306B58,__READ_WRITE ,__rm_rstst_sgx_bits);
__IO_REG32_BIT(PM_WKDEP_SGX,    			0x48306BC8,__READ_WRITE ,__pm_wkdep_sgx_bits);
__IO_REG32_BIT(PM_PWSTCTRL_SGX,    		0x48306BE0,__READ_WRITE ,__pm_pwstctrl_sgx_bits);
__IO_REG32_BIT(PM_PWSTST_SGX,    			0x48306BE4,__READ				,__pm_pwstst_sgx_bits);
__IO_REG32_BIT(PM_PREPWSTCTRL_SGX,    0x48306BE8,__READ_WRITE ,__pm_prepwstctrl_sgx_bits);
__IO_REG32_BIT(PM_WKEN_WKUP,    			0x48306CA0,__READ_WRITE ,__pm_wken_wkup_bits);
__IO_REG32_BIT(PM_MPUGRPSEL_WKUP,    	0x48306CA4,__READ_WRITE ,__pm_mpugrpsel_wkup_bits);
__IO_REG32_BIT(PM_WKST_WKUP,    			0x48306CB0,__READ_WRITE ,__pm_wkst_wkup_bits);
__IO_REG32_BIT(PRM_CLKSEL,    				0x48306D40,__READ_WRITE ,__prm_clksel_bits);
__IO_REG32_BIT(PRM_CLKOUT_CTRL,    		0x48306D70,__READ_WRITE ,__prm_clkout_ctrl_bits);
__IO_REG32_BIT(RM_RSTST_DSS,    			0x48306E58,__READ_WRITE ,__rm_rstst_dss_bits);
__IO_REG32_BIT(PM_WKEN_DSS,    				0x48306EA0,__READ_WRITE ,__pm_wken_dss_bits);
__IO_REG32_BIT(PM_WKDEP_DSS,    			0x48306EC8,__READ_WRITE ,__pm_wkdep_dss_bits);
__IO_REG32_BIT(PM_PWSTCTRL_DSS,    		0x48306EE0,__READ_WRITE ,__pm_pwstctrl_dss_bits);
__IO_REG32_BIT(PM_PWSTST_DSS,    			0x48306EE4,__READ				,__pm_pwstst_dss_bits);
__IO_REG32_BIT(PM_PREPWSTST_DSS,    	0x48306EE8,__READ_WRITE ,__pm_prepwstst_dss_bits);
__IO_REG32_BIT(RM_RSTST_PER,    			0x48307058,__READ_WRITE ,__rm_rstst_per_bits);
__IO_REG32_BIT(PM_WKEN_PER,    				0x483070A0,__READ_WRITE ,__pm_wken_per_bits);
__IO_REG32_BIT(PM_MPUGRPSEL_PER,    	0x483070A4,__READ_WRITE ,__pm_mpugrpsel_per_bits);
__IO_REG32_BIT(PM_WKST_PER,    				0x483070B0,__READ_WRITE ,__pm_wkst_per_bits);
__IO_REG32_BIT(PM_WKDEP_PER,    			0x483070C8,__READ_WRITE ,__pm_wkdep_per_bits);
__IO_REG32_BIT(PM_PWSTCTRL_PER,    		0x483070E0,__READ_WRITE ,__pm_pwstctrl_per_bits);
__IO_REG32_BIT(PM_PWSTST_PER,    			0x483070E4,__READ				,__pm_pwstst_per_bits);
__IO_REG32_BIT(PM_PREPWSTST_PER,    	0x483070E8,__READ_WRITE ,__pm_prepwstst_per_bits);
__IO_REG32_BIT(RM_RSTST_EMU,    			0x48307158,__READ_WRITE ,__rm_rstst_emu_bits);
__IO_REG32_BIT(PRM_RSTCTRL,    				0x48307250,__READ_WRITE ,__prm_rstctrl_bits);
__IO_REG32_BIT(PRM_RSTTIME,    				0x48307254,__READ_WRITE ,__prm_rsttime_bits);
__IO_REG32_BIT(PRM_RSTST,    					0x48307258,__READ_WRITE ,__prm_rstst_bits);
__IO_REG32_BIT(PRM_CLKSRC_CTRL,    		0x48307270,__READ_WRITE ,__prm_clksrc_ctrl_bits);
__IO_REG32_BIT(PRM_OBS,    						0x48307280,__READ				,__prm_obs_bits);
__IO_REG32_BIT(PRM_CLKSETUP,    			0x48307298,__READ_WRITE ,__prm_clksetup_bits);
__IO_REG32_BIT(PRM_POLCTRL,    				0x4830729C,__READ_WRITE ,__prm_polctrl_bits);
__IO_REG32_BIT(RM_RSTST_NEON,    			0x48307358,__READ_WRITE ,__rm_rstst_neon_bits);
__IO_REG32_BIT(PM_WKDEP_NEON,    			0x483073C8,__READ_WRITE ,__pm_wkdep_neon_bits);
__IO_REG32_BIT(PM_PWSTCTRL_NEON,    	0x483073E0,__READ_WRITE ,__pm_pwstctrl_neon_bits);
__IO_REG32_BIT(PM_PWSTST_NEON,    		0x483073E4,__READ				,__pm_pwstst_neon_bits);
__IO_REG32_BIT(PM_PREPWSTST_NEON,    	0x483073E8,__READ_WRITE ,__pm_prepwstst_neon_bits);
__IO_REG32_BIT(RM_RSTST_USBHOST,    	0x48307458,__READ_WRITE ,__rm_rstst_usbhost_bits);
__IO_REG32_BIT(PM_WKEN_USBHOST,    		0x483074A0,__READ_WRITE ,__pm_wken_usbhost_bits);
__IO_REG32_BIT(PM_MPUGRPSEL_USBHOST,  0x483074A4,__READ_WRITE ,__pm_mpugrpsel_usbhost_bits);
__IO_REG32_BIT(PM_WKST_USBHOST,    		0x483074B0,__READ_WRITE ,__pm_wkst_usbhost_bits);
__IO_REG32_BIT(PM_WKDEP_USBHOST,    	0x483074C8,__READ_WRITE ,__pm_wkdep_usbhost_bits);
__IO_REG32_BIT(PM_PWSTCTRL_USBHOST,   0x483074E0,__READ_WRITE ,__pm_pwstctrl_usbhost_bits);
__IO_REG32_BIT(PM_PWSTST_USBHOST,    	0x483074E4,__READ				,__pm_pwstst_usbhost_bits);
__IO_REG32_BIT(PM_PREPWSTST_USBHOST,  0x483074E8,__READ_WRITE ,__pm_prepwstst_usbhost_bits);

/***************************************************************************
 **
 ** L3 IA
 **
 ***************************************************************************/
__IO_REG32_BIT(L3_IA_AGENT_CONTROL_MPUSS,  	0x68001420,__READ_WRITE ,__l3_ia_agent_control_bits);
__IO_REG32_BIT(L3_IA_AGENT_STATUS_MPUSS,   	0x68001428,__READ_WRITE ,__l3_ia_agent_status_bits);
__IO_REG32_BIT(L3_IA_ERROR_LOG_MPUSS,   		0x68001458,__READ_WRITE ,__l3_ia_error_log_bits);
__IO_REG32(		 L3_IA_ERROR_LOG_ADDR_MPUSS,	0x68001460,__READ				);
__IO_REG32_BIT(L3_IA_AGENT_CONTROL_SGX,  		0x68001C20,__READ_WRITE ,__l3_ia_agent_control_bits);
__IO_REG32_BIT(L3_IA_AGENT_STATUS_SGX,   		0x68001C28,__READ_WRITE ,__l3_ia_agent_status_bits);
__IO_REG32_BIT(L3_IA_ERROR_LOG_SGX,   			0x68001C58,__READ_WRITE ,__l3_ia_error_log_bits);
__IO_REG32(		 L3_IA_ERROR_LOG_ADDR_SGX, 		0x68001C60,__READ				);
__IO_REG32_BIT(L3_IA_AGENT_CONTROL_USB_HS,  0x68004020,__READ_WRITE ,__l3_ia_agent_control_bits);
__IO_REG32_BIT(L3_IA_AGENT_STATUS_USB_HS,   0x68004028,__READ_WRITE ,__l3_ia_agent_status_bits);
__IO_REG32_BIT(L3_IA_ERROR_LOG_USB_HS,   		0x68004058,__READ_WRITE ,__l3_ia_error_log_bits);
__IO_REG32(		 L3_IA_ERROR_LOG_ADDR_USB_HS,	0x68004060,__READ				);
__IO_REG32_BIT(L3_IA_AGENT_CONTROL_IPSS,  	0x68004420,__READ_WRITE ,__l3_ia_agent_control_bits);
__IO_REG32_BIT(L3_IA_AGENT_STATUS_IPSS,   	0x68004428,__READ_WRITE ,__l3_ia_agent_status_bits);
__IO_REG32_BIT(L3_IA_ERROR_LOG_IPSS,   			0x68004458,__READ_WRITE ,__l3_ia_error_log_bits);
__IO_REG32(		 L3_IA_ERROR_LOG_ADDR_IPSS, 	0x68004460,__READ				);
__IO_REG32_BIT(L3_IA_AGENT_CONTROL_DMA_RD,  0x68004C20,__READ_WRITE ,__l3_ia_agent_control_bits);
__IO_REG32_BIT(L3_IA_AGENT_STATUS_DMA_RD,  	0x68004C28,__READ_WRITE ,__l3_ia_agent_status_bits);
__IO_REG32_BIT(L3_IA_ERROR_LOG_DMA_RD,  		0x68004C58,__READ_WRITE ,__l3_ia_error_log_bits);
__IO_REG32(		 L3_IA_ERROR_LOG_ADDR_DMA_RD, 0x68004C60,__READ				);
__IO_REG32_BIT(L3_IA_AGENT_CONTROL_DMA_WR,  0x68005020,__READ_WRITE ,__l3_ia_agent_control_bits);
__IO_REG32_BIT(L3_IA_AGENT_STATUS_DMA_WR,  	0x68005028,__READ_WRITE ,__l3_ia_agent_status_bits);
__IO_REG32_BIT(L3_IA_ERROR_LOG_DMA_WR,  		0x68005058,__READ_WRITE ,__l3_ia_error_log_bits);
__IO_REG32(		 L3_IA_ERROR_LOG_ADDR_DMA_WR, 0x68005060,__READ				);
__IO_REG32_BIT(L3_IA_AGENT_CONTROL_DSS,  		0x68005420,__READ_WRITE ,__l3_ia_agent_control_bits);
__IO_REG32_BIT(L3_IA_AGENT_STATUS_DSS,  		0x68005428,__READ_WRITE ,__l3_ia_agent_status_bits);
__IO_REG32_BIT(L3_IA_ERROR_LOG_DSS,  				0x68005458,__READ_WRITE ,__l3_ia_error_log_bits);
__IO_REG32(		 L3_IA_ERROR_LOG_ADDR_DSS, 		0x68005460,__READ				);
__IO_REG32_BIT(L3_IA_AGENT_CONTROL_DAP,  		0x68005C20,__READ_WRITE ,__l3_ia_agent_control_bits);
__IO_REG32_BIT(L3_IA_AGENT_STATUS_DAP,  		0x68005C28,__READ_WRITE ,__l3_ia_agent_status_bits);
__IO_REG32_BIT(L3_IA_ERROR_LOG_DAP,  				0x68005C58,__READ_WRITE ,__l3_ia_error_log_bits);
__IO_REG32(		 L3_IA_ERROR_LOG_ADDR_DAP, 		0x68005C60,__READ				);

/***************************************************************************
 **
 ** L3 TA
 **
 ***************************************************************************/
__IO_REG32_BIT(L3_TA_AGENT_CONTROL_SMS,  		0x68002020,__READ_WRITE ,__l3_ta_agent_control_bits);
__IO_REG32_BIT(L3_TA_AGENT_STATUS_SMS,   		0x68002028,__READ				,__l3_ta_agent_status_bits);
__IO_REG32_BIT(L3_TA_ERROR_LOG_SMS,   			0x68002058,__READ_WRITE ,__l3_ta_error_log_bits);
__IO_REG32(		 L3_TA_ERROR_LOG_ADDR_SMS, 		0x68002060,__READ				);
__IO_REG32_BIT(L3_TA_AGENT_CONTROL_GPMC, 		0x68002420,__READ_WRITE ,__l3_ta_agent_control_bits);
__IO_REG32_BIT(L3_TA_AGENT_STATUS_GPMC,  		0x68002428,__READ				,__l3_ta_agent_status_bits);
__IO_REG32_BIT(L3_TA_ERROR_LOG_GPMC,  			0x68002458,__READ_WRITE ,__l3_ta_error_log_bits);
__IO_REG32(		 L3_TA_ERROR_LOG_ADDR_GPMC,		0x68002460,__READ				);
__IO_REG32_BIT(L3_TA_AGENT_CONTROL_OCM_RAM, 0x68002820,__READ_WRITE ,__l3_ta_agent_control_bits);
__IO_REG32_BIT(L3_TA_AGENT_STATUS_OCM_RAM,  0x68002828,__READ				,__l3_ta_agent_status_bits);
__IO_REG32_BIT(L3_TA_ERROR_LOG_OCM_RAM,  		0x68002858,__READ_WRITE ,__l3_ta_error_log_bits);
__IO_REG32(		 L3_TA_ERROR_LOG_ADDR_OCM_RAM,0x68002860,__READ				);
__IO_REG32_BIT(L3_TA_AGENT_CONTROL_OCM_ROM, 0x68002C20,__READ_WRITE ,__l3_ta_agent_control_bits);
__IO_REG32_BIT(L3_TA_AGENT_STATUS_OCM_ROM,  0x68002C28,__READ				,__l3_ta_agent_status_bits);
__IO_REG32_BIT(L3_TA_ERROR_LOG_OCM_ROM,   	0x68002C58,__READ_WRITE ,__l3_ta_error_log_bits);
__IO_REG32(		 L3_TA_ERROR_LOG_ADDR_OCM_ROM,0x68002C60,__READ				);
__IO_REG32_BIT(L3_TA_AGENT_CONTROL_IPSS, 	 	0x68006020,__READ_WRITE ,__l3_ta_agent_control_bits);
__IO_REG32_BIT(L3_TA_AGENT_STATUS_IPSS,   	0x68006028,__READ				,__l3_ta_agent_status_bits);
__IO_REG32_BIT(L3_TA_ERROR_LOG_IPSS,   			0x68006058,__READ_WRITE ,__l3_ta_error_log_bits);
__IO_REG32(		 L3_TA_ERROR_LOG_ADDR_IPSS, 	0x68006060,__READ				);
__IO_REG32_BIT(L3_TA_AGENT_CONTROL_SGX, 	 	0x68006420,__READ_WRITE ,__l3_ta_agent_control_bits);
__IO_REG32_BIT(L3_TA_AGENT_STATUS_SGX,   		0x68006428,__READ				,__l3_ta_agent_status_bits);
__IO_REG32_BIT(L3_TA_ERROR_LOG_SGX,   			0x68006458,__READ_WRITE ,__l3_ta_error_log_bits);
__IO_REG32(		 L3_TA_ERROR_LOG_ADDR_SGX, 		0x68006460,__READ				);
__IO_REG32_BIT(L3_TA_AGENT_CONTROL_L4_CORE, 0x68006820,__READ_WRITE ,__l3_ta_agent_control_bits);
__IO_REG32_BIT(L3_TA_AGENT_STATUS_L4_CORE,  0x68006828,__READ				,__l3_ta_agent_status_bits);
__IO_REG32_BIT(L3_TA_ERROR_LOG_L4_CORE,   	0x68006858,__READ_WRITE ,__l3_ta_error_log_bits);
__IO_REG32(		 L3_TA_ERROR_LOG_ADDR_L4_CORE,0x68006860,__READ				);
__IO_REG32_BIT(L3_TA_AGENT_CONTROL_L4_PER, 	0x68006C20,__READ_WRITE ,__l3_ta_agent_control_bits);
__IO_REG32_BIT(L3_TA_AGENT_STATUS_L4_PER, 	0x68006C28,__READ				,__l3_ta_agent_status_bits);
__IO_REG32_BIT(L3_TA_ERROR_LOG_L4_PER,   		0x68006C58,__READ_WRITE ,__l3_ta_error_log_bits);
__IO_REG32(		 L3_TA_ERROR_LOG_ADDR_L4_PER,	0x68006C60,__READ				);
__IO_REG32_BIT(L3_TA_AGENT_CONTROL_L4_EMU, 	0x68007020,__READ_WRITE ,__l3_ta_agent_control_bits);
__IO_REG32_BIT(L3_TA_AGENT_STATUS_L4_EMU, 	0x68007028,__READ				,__l3_ta_agent_status_bits);
__IO_REG32_BIT(L3_TA_ERROR_LOG_L4_EMU,   		0x68007058,__READ_WRITE ,__l3_ta_error_log_bits);
__IO_REG32(		 L3_TA_ERROR_LOG_ADDR_L4_EMU,	0x68007060,__READ				);

/***************************************************************************
 **
 ** L3 RT
 **
 ***************************************************************************/
__IO_REG32(		 L3_RT_NETWORK,  							0x68000014,__READ				);
__IO_REG32_BIT(L3_RT_INITID_READBACK,  			0x68000070,__READ				,__l3_rt_initid_readback_bits);
__IO_REG32_BIT(L3_RT_NETWORK_CONTROL_L,  		0x68000078,__READ_WRITE ,__l3_rt_network_control_l_bits);
__IO_REG32_BIT(L3_RT_NETWORK_CONTROL_H,  		0x6800007C,__READ_WRITE ,__l3_rt_network_control_h_bits);

/***************************************************************************
 **
 ** L3 PM
 **
 ***************************************************************************/
__IO_REG32_BIT(L3_PM_ERROR_LOG_RT,  								0x68010020,__READ_WRITE	,__l3_pm_error_log_bits);
__IO_REG32_BIT(L3_PM_CONTROL_RT,  									0x68010028,__READ_WRITE	,__l3_pm_control_bits);
__IO_REG32_BIT(L3_PM_ERROR_CLEAR_SINGLE_RT,  				0x68010030,__READ				,__l3_pm_error_clear_single_bits);
__IO_REG32_BIT(L3_PM_ERROR_CLEAR_MULTI_RT,  				0x68010038,__READ				,__l3_pm_error_clear_single_bits);
__IO_REG32_BIT(L3_PM_REQ_INFO_PERMISSION_0_RT,			0x68010048,__READ_WRITE	,__l3_pm_req_info_permission_bits);
__IO_REG32_BIT(L3_PM_REQ_INFO_PERMISSION_1_RT,			0x68010068,__READ_WRITE	,__l3_pm_req_info_permission_bits);
__IO_REG32_BIT(L3_PM_READ_PERMISSION_0_RT,  				0x68010050,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_READ_PERMISSION_1_RT,  				0x68010070,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_WRITE_PERMISSION_0_RT,  				0x68010058,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_WRITE_PERMISSION_1_RT,  				0x68010078,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_ADDR_MATCH_1_RT,  							0x68010060,__READ_WRITE	,__l3_pm_addr_match_bits);
__IO_REG32_BIT(L3_PM_ERROR_LOG_GPMC,  							0x68012420,__READ_WRITE	,__l3_pm_error_log_bits);
__IO_REG32_BIT(L3_PM_CONTROL_GPMC,  								0x68012428,__READ_WRITE	,__l3_pm_control_bits);
__IO_REG32_BIT(L3_PM_ERROR_CLEAR_SINGLE_GPMC,  			0x68012430,__READ				,__l3_pm_error_clear_single_bits);
__IO_REG32_BIT(L3_PM_ERROR_CLEAR_MULTI_GPMC,  			0x68012438,__READ				,__l3_pm_error_clear_single_bits);
__IO_REG32_BIT(L3_PM_REQ_INFO_PERMISSION_0_GPMC,		0x68012448,__READ_WRITE	,__l3_pm_req_info_permission_bits);
__IO_REG32_BIT(L3_PM_REQ_INFO_PERMISSION_1_GPMC,		0x68012468,__READ_WRITE	,__l3_pm_req_info_permission_bits);
__IO_REG32_BIT(L3_PM_REQ_INFO_PERMISSION_2_GPMC,		0x68012488,__READ_WRITE	,__l3_pm_req_info_permission_bits);
__IO_REG32_BIT(L3_PM_REQ_INFO_PERMISSION_3_GPMC,		0x680124A8,__READ_WRITE	,__l3_pm_req_info_permission_bits);
__IO_REG32_BIT(L3_PM_REQ_INFO_PERMISSION_4_GPMC,		0x680124C8,__READ_WRITE	,__l3_pm_req_info_permission_bits);
__IO_REG32_BIT(L3_PM_REQ_INFO_PERMISSION_5_GPMC,		0x680124E8,__READ_WRITE	,__l3_pm_req_info_permission_bits);
__IO_REG32_BIT(L3_PM_REQ_INFO_PERMISSION_6_GPMC,		0x68012508,__READ_WRITE	,__l3_pm_req_info_permission_bits);
__IO_REG32_BIT(L3_PM_REQ_INFO_PERMISSION_7_GPMC,		0x68012528,__READ_WRITE	,__l3_pm_req_info_permission_bits);
__IO_REG32_BIT(L3_PM_READ_PERMISSION_0_GPMC,  			0x68012450,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_READ_PERMISSION_1_GPMC,  			0x68012470,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_READ_PERMISSION_2_GPMC,  			0x68012490,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_READ_PERMISSION_3_GPMC,  			0x680124B0,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_READ_PERMISSION_4_GPMC,  			0x680124D0,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_READ_PERMISSION_5_GPMC,  			0x680124F0,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_READ_PERMISSION_6_GPMC,  			0x68012510,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_READ_PERMISSION_7_GPMC,  			0x68012530,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_WRITE_PERMISSION_0_GPMC,  			0x68012458,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_WRITE_PERMISSION_1_GPMC, 			0x68012478,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_WRITE_PERMISSION_2_GPMC, 			0x68012498,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_WRITE_PERMISSION_3_GPMC, 			0x680124B8,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_WRITE_PERMISSION_4_GPMC, 			0x680124D8,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_WRITE_PERMISSION_5_GPMC, 			0x680124F8,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_WRITE_PERMISSION_6_GPMC, 			0x68012518,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_WRITE_PERMISSION_7_GPMC, 			0x68012538,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_ADDR_MATCH_1_GPMC,							0x68012460,__READ_WRITE	,__l3_pm_addr_match_bits);
__IO_REG32_BIT(L3_PM_ADDR_MATCH_2_GPMC,							0x68012480,__READ_WRITE	,__l3_pm_addr_match_bits);
__IO_REG32_BIT(L3_PM_ADDR_MATCH_3_GPMC,							0x680124A0,__READ_WRITE	,__l3_pm_addr_match_bits);
__IO_REG32_BIT(L3_PM_ADDR_MATCH_4_GPMC,							0x680124C0,__READ_WRITE	,__l3_pm_addr_match_bits);
__IO_REG32_BIT(L3_PM_ADDR_MATCH_5_GPMC,							0x680124E0,__READ_WRITE	,__l3_pm_addr_match_bits);
__IO_REG32_BIT(L3_PM_ADDR_MATCH_6_GPMC,							0x68012500,__READ_WRITE	,__l3_pm_addr_match_bits);
__IO_REG32_BIT(L3_PM_ADDR_MATCH_7_GPMC,							0x68012520,__READ_WRITE	,__l3_pm_addr_match_bits);
__IO_REG32_BIT(L3_PM_ERROR_LOG_OCM_RAM,  						0x68012820,__READ_WRITE	,__l3_pm_error_log_bits);
__IO_REG32_BIT(L3_PM_CONTROL_OCM_RAM,  							0x68012828,__READ_WRITE	,__l3_pm_control_bits);
__IO_REG32_BIT(L3_PM_ERROR_CLEAR_SINGLE_OCM_RAM,  	0x68012830,__READ				,__l3_pm_error_clear_single_bits);
__IO_REG32_BIT(L3_PM_ERROR_CLEAR_MULTI_OCM_RAM,  		0x68012838,__READ				,__l3_pm_error_clear_single_bits);
__IO_REG32_BIT(L3_PM_REQ_INFO_PERMISSION_0_OCM_RAM,	0x68012848,__READ_WRITE	,__l3_pm_req_info_permission_bits);
__IO_REG32_BIT(L3_PM_REQ_INFO_PERMISSION_1_OCM_RAM,	0x68012868,__READ_WRITE	,__l3_pm_req_info_permission_bits);
__IO_REG32_BIT(L3_PM_REQ_INFO_PERMISSION_2_OCM_RAM,	0x68012888,__READ_WRITE	,__l3_pm_req_info_permission_bits);
__IO_REG32_BIT(L3_PM_REQ_INFO_PERMISSION_3_OCM_RAM,	0x680128A8,__READ_WRITE	,__l3_pm_req_info_permission_bits);
__IO_REG32_BIT(L3_PM_REQ_INFO_PERMISSION_4_OCM_RAM,	0x680128C8,__READ_WRITE	,__l3_pm_req_info_permission_bits);
__IO_REG32_BIT(L3_PM_REQ_INFO_PERMISSION_5_OCM_RAM,	0x680128E8,__READ_WRITE	,__l3_pm_req_info_permission_bits);
__IO_REG32_BIT(L3_PM_REQ_INFO_PERMISSION_6_OCM_RAM,	0x68012908,__READ_WRITE	,__l3_pm_req_info_permission_bits);
__IO_REG32_BIT(L3_PM_REQ_INFO_PERMISSION_7_OCM_RAM,	0x68012928,__READ_WRITE	,__l3_pm_req_info_permission_bits);
__IO_REG32_BIT(L3_PM_READ_PERMISSION_0_OCM_RAM,  		0x68012850,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_READ_PERMISSION_1_OCM_RAM,  		0x68012870,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_READ_PERMISSION_2_OCM_RAM,  		0x68012890,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_READ_PERMISSION_3_OCM_RAM,  		0x680128B0,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_READ_PERMISSION_4_OCM_RAM,  		0x680128D0,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_READ_PERMISSION_5_OCM_RAM,  		0x680128F0,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_READ_PERMISSION_6_OCM_RAM,  		0x68012910,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_READ_PERMISSION_7_OCM_RAM,  		0x68012930,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_WRITE_PERMISSION_0_OCM_RAM,  	0x68012858,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_WRITE_PERMISSION_1_OCM_RAM, 		0x68012878,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_WRITE_PERMISSION_2_OCM_RAM, 		0x68012898,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_WRITE_PERMISSION_3_OCM_RAM, 		0x680128B8,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_WRITE_PERMISSION_4_OCM_RAM, 		0x680128D8,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_WRITE_PERMISSION_5_OCM_RAM, 		0x680128F8,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_WRITE_PERMISSION_6_OCM_RAM, 		0x68012918,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_WRITE_PERMISSION_7_OCM_RAM, 		0x68012938,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_ADDR_MATCH_1_OCM_RAM,					0x68012860,__READ_WRITE	,__l3_pm_addr_match_bits);
__IO_REG32_BIT(L3_PM_ADDR_MATCH_2_OCM_RAM,					0x68012880,__READ_WRITE	,__l3_pm_addr_match_bits);
__IO_REG32_BIT(L3_PM_ADDR_MATCH_3_OCM_RAM,					0x680128A0,__READ_WRITE	,__l3_pm_addr_match_bits);
__IO_REG32_BIT(L3_PM_ADDR_MATCH_4_OCM_RAM,					0x680128C0,__READ_WRITE	,__l3_pm_addr_match_bits);
__IO_REG32_BIT(L3_PM_ADDR_MATCH_5_OCM_RAM,					0x680128E0,__READ_WRITE	,__l3_pm_addr_match_bits);
__IO_REG32_BIT(L3_PM_ADDR_MATCH_6_OCM_RAM,					0x68012900,__READ_WRITE	,__l3_pm_addr_match_bits);
__IO_REG32_BIT(L3_PM_ADDR_MATCH_7_OCM_RAM,					0x68012920,__READ_WRITE	,__l3_pm_addr_match_bits);
__IO_REG32_BIT(L3_PM_ERROR_LOG_OCM_ROM,  						0x68012C20,__READ_WRITE	,__l3_pm_error_log_bits);
__IO_REG32_BIT(L3_PM_CONTROL_OCM_ROM,  							0x68012C28,__READ_WRITE	,__l3_pm_control_bits);
__IO_REG32_BIT(L3_PM_ERROR_CLEAR_SINGLE_OCM_ROM,  	0x68012C30,__READ				,__l3_pm_error_clear_single_bits);
__IO_REG32_BIT(L3_PM_ERROR_CLEAR_MULTI_OCM_ROM,  		0x68012C38,__READ				,__l3_pm_error_clear_single_bits);
__IO_REG32_BIT(L3_PM_REQ_INFO_PERMISSION_0_OCM_ROM,	0x68012C48,__READ_WRITE	,__l3_pm_req_info_permission_bits);
__IO_REG32_BIT(L3_PM_REQ_INFO_PERMISSION_1_OCM_ROM,	0x68012C68,__READ_WRITE	,__l3_pm_req_info_permission_bits);
__IO_REG32_BIT(L3_PM_READ_PERMISSION_0_OCM_ROM,  		0x68012C50,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_READ_PERMISSION_1_OCM_ROM,  		0x68012C70,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_WRITE_PERMISSION_0_OCM_ROM,  	0x68012C58,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_WRITE_PERMISSION_1_OCM_ROM, 		0x68012C78,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_ADDR_MATCH_1_OCM_ROM,					0x68012C60,__READ_WRITE	,__l3_pm_addr_match_bits);
__IO_REG32_BIT(L3_PM_ERROR_LOG_IPSS, 		 						0x68014020,__READ_WRITE	,__l3_pm_error_log_bits);
__IO_REG32_BIT(L3_PM_CONTROL_IPSS,  								0x68014028,__READ_WRITE	,__l3_pm_control_bits);
__IO_REG32_BIT(L3_PM_ERROR_CLEAR_SINGLE_IPSS,  			0x68014030,__READ				,__l3_pm_error_clear_single_bits);
__IO_REG32_BIT(L3_PM_ERROR_CLEAR_MULTI_IPSS,  			0x68014038,__READ				,__l3_pm_error_clear_single_bits);
__IO_REG32_BIT(L3_PM_REQ_INFO_PERMISSION_0_IPSS,		0x68014048,__READ_WRITE	,__l3_pm_req_info_permission_bits);
__IO_REG32_BIT(L3_PM_REQ_INFO_PERMISSION_1_IPSS,		0x68014068,__READ_WRITE	,__l3_pm_req_info_permission_bits);
__IO_REG32_BIT(L3_PM_REQ_INFO_PERMISSION_2_IPSS,		0x68014088,__READ_WRITE	,__l3_pm_req_info_permission_bits);
__IO_REG32_BIT(L3_PM_REQ_INFO_PERMISSION_3_IPSS,		0x680140A8,__READ_WRITE	,__l3_pm_req_info_permission_bits);
__IO_REG32_BIT(L3_PM_READ_PERMISSION_0_IPSS,  			0x68014050,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_READ_PERMISSION_1_IPSS,  			0x68014070,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_READ_PERMISSION_2_IPSS,  			0x68014090,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_READ_PERMISSION_3_IPSS,  			0x680140B0,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_WRITE_PERMISSION_0_IPSS,		  	0x68014058,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_WRITE_PERMISSION_1_IPSS,		 		0x68014078,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_WRITE_PERMISSION_2_IPSS,		 		0x68014098,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_WRITE_PERMISSION_3_IPSS,		 		0x680140B8,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_ADDR_MATCH_1_IPSS,							0x68014060,__READ_WRITE	,__l3_pm_addr_match_bits);
__IO_REG32_BIT(L3_PM_ADDR_MATCH_2_IPSS,							0x68014080,__READ_WRITE	,__l3_pm_addr_match_bits);
__IO_REG32_BIT(L3_PM_ADDR_MATCH_3_IPSS,							0x680140A0,__READ_WRITE	,__l3_pm_addr_match_bits);

/***************************************************************************
 **
 ** L3 SI
 **
 ***************************************************************************/
__IO_REG32_BIT(L3_SI_CONTROL,  											0x68000424,__READ_WRITE	,__l3_si_control_bits);
__IO_REG32(		 L3_SI_FLAG_STATUS_0_L,								0x68000510,__READ				);
__IO_REG32(		 L3_SI_FLAG_STATUS_0_H,								0x68000514,__READ				);
__IO_REG32(		 L3_SI_FLAG_STATUS_1_L,								0x68000530,__READ				);

/***************************************************************************
 **
 ** L4 IA
 **
 ***************************************************************************/
__IO_REG32_BIT(L4_IA_AGENT_CONTROL_L_CORE,					0x48040820,__READ_WRITE	,__l4_ia_agent_control_l_bits);
__IO_REG32_BIT(L4_IA_AGENT_STATUS_L_CORE,						0x48040828,__READ_WRITE	,__l4_ia_agent_control_l_bits);
__IO_REG32_BIT(L4_IA_ERROR_LOG_L_CORE,  						0x48040858,__READ_WRITE	,__l4_ia_error_log_l_bits);
__IO_REG32_BIT(L4_IA_AGENT_CONTROL_L_PER,						0x49000820,__READ_WRITE	,__l4_ia_agent_control_l_bits);
__IO_REG32_BIT(L4_IA_AGENT_STATUS_L_PER,						0x49000828,__READ_WRITE	,__l4_ia_agent_control_l_bits);
__IO_REG32_BIT(L4_IA_ERROR_LOG_L_PER,  							0x49000858,__READ_WRITE	,__l4_ia_error_log_l_bits);
__IO_REG32_BIT(L4_IA_AGENT_CONTROL_L_EMU,						0x54006820,__READ_WRITE	,__l4_ia_agent_control_l_bits);
__IO_REG32_BIT(L4_IA_AGENT_STATUS_L_EMU,						0x54006828,__READ_WRITE	,__l4_ia_agent_control_l_bits);
__IO_REG32_BIT(L4_IA_ERROR_LOG_L_EMU,  							0x54006858,__READ_WRITE	,__l4_ia_error_log_l_bits);
__IO_REG32_BIT(L4_IA_AGENT_CONTROL_L_EMU_IA_DAP,		0x54008020,__READ_WRITE	,__l4_ia_agent_control_l_bits);
__IO_REG32_BIT(L4_IA_AGENT_STATUS_L_EMU_IA_DAP,			0x54008028,__READ_WRITE	,__l4_ia_agent_control_l_bits);
__IO_REG32_BIT(L4_IA_ERROR_LOG_L_EMU_IA_DAP,  			0x54008058,__READ_WRITE	,__l4_ia_error_log_l_bits);
__IO_REG32_BIT(L4_IA_AGENT_CONTROL_L_WKUP_IA_EMU,		0x48328820,__READ_WRITE	,__l4_ia_agent_control_l_bits);
__IO_REG32_BIT(L4_IA_AGENT_STATUS_L_WKUP_IA_EMU,		0x48328828,__READ_WRITE	,__l4_ia_agent_control_l_bits);
__IO_REG32_BIT(L4_IA_ERROR_LOG_L_WKUP_IA_EMU,  			0x48328858,__READ_WRITE	,__l4_ia_error_log_l_bits);
__IO_REG32_BIT(L4_IA_AGENT_CONTROL_L_WKUP_IA_CORE,	0x4832A020,__READ_WRITE	,__l4_ia_agent_control_l_bits);
__IO_REG32_BIT(L4_IA_AGENT_STATUS_L_WKUP_IA_CORE,		0x4832A028,__READ_WRITE	,__l4_ia_agent_control_l_bits);
__IO_REG32_BIT(L4_IA_ERROR_LOG_L_WKUP_IA_CORE,  		0x4832A058,__READ_WRITE	,__l4_ia_error_log_l_bits);

/***************************************************************************
 **
 ** L4 TA
 **
 ***************************************************************************/
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_CONTROL,			0x48003020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_CONTROL,			0x48003024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_CONTROL,			0x48003028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_CM,					0x48027020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_CM,					0x48027024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_CM,						0x48027028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_DISPLAY_SS,	0x48051020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_DISPLAY_SS,	0x48051024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_DISPLAY_SS,		0x48051028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_SDMA,				0x48057020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_SDMA,				0x48057024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_SDMA,					0x48057028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_I2C3,				0x48061020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_I2C3,				0x48061024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_I2C3,					0x48061028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_USB_TLL,			0x48063020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_USB_TLL,			0x48063024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_USB_TLL,			0x48063028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_USB_HS_Host,	0x48065020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_USB_HS_Host,	0x48065024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_USB_HS_Host,	0x48065028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_UART1,				0x4806B020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_UART1,				0x4806B024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_UART1,				0x4806B028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_UART2,				0x4806D020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_UART2,				0x4806D024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_UART2,				0x4806D028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_I2C1,				0x48071020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_I2C1,				0x48071024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_I2C1,					0x48071028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_I2C2,				0x48073000,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_I2C2,				0x48073024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_I2C2,					0x48073028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_MCBSP1,			0x48075020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_MCBSP1,			0x48075024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_MCBSP1,				0x48075028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_GPTIMER10,		0x48087020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_GPTIMER10,		0x48087024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_GPTIMER10,		0x48087028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_GPTIMER11,		0x48089020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_GPTIMER11,		0x48089024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_GPTIMER11,		0x48089028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_MCBSP5,			0x48097020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_MCBSP5,			0x48097024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_MCBSP5,				0x48097028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_MCSPI1,			0x48099020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_MCSPI1,			0x48099024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_MCSPI1,				0x48099028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_MCSPI2,			0x4809B020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_MCSPI2,			0x4809B024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_MCSPI2,				0x4809B028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_MMC1,				0x4809D020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_MMC1,				0x4809D024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_MMC1,					0x4809D028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_UART4,				0x4809F020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_UART4,				0x4809F024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_UART4,				0x4809F028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_MMC3,				0x480AE020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_MMC3,				0x480AE024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_MMC3,					0x480AE028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_MG,					0x480B1020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_MG,					0x480B1024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_MG,						0x480B1028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_HDQ1,				0x480B3020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_HDQ1,				0x480B3024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_HDQ1,					0x480B3028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_MMC2,				0x480B5020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_MMC2,				0x480B5024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_MMC2,					0x480B5028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_MCSPI3,			0x480B9020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_MCSPI3,			0x480B9024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_MCSPI3,				0x480B9028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_MCSPI4,			0x480BB020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_MCSPI4,			0x480BB024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_MCSPI4,				0x480BB028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_INTH,				0x480C8020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_INTH,				0x480C8024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_INTH,					0x480C8028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_WKUP,				0x48340020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_WKUP,				0x48340024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_WKUP,					0x48340028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_UART3,				0x49021020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_UART3,				0x49021024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_UART3,					0x49021028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_MCBSP2,				0x49023020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_MCBSP2,				0x49023024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_MCBSP2,				0x49023028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_MCBSP3,				0x49025020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_MCBSP3,				0x49025024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_MCBSP3,				0x49025028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_MCBSP4,				0x49027020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_MCBSP4,				0x49027024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_MCBSP4,				0x49027028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_MCBSP2_SIDETONE2,	0x49029020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_MCBSP2_SIDETONE2,	0x49029024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_MCBSP2_SIDETONE2,	0x49029028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_MCBSP2_SIDETONE3,	0x4902B020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_MCBSP2_SIDETONE3,	0x4902B024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_MCBSP2_SIDETONE3,	0x4902B028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_WDTIMER3,			0x49031020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_WDTIMER3,			0x49031024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_WDTIMER3,			0x49031028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_GPTIMER2,			0x49033020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_GPTIMER2,			0x49033024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_GPTIMER2,			0x49033028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_GPTIMER3,			0x49035020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_GPTIMER3,			0x49035024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_GPTIMER3,			0x49035028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_GPTIMER4,			0x49037020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_GPTIMER4,			0x49037024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_GPTIMER4,			0x49037028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_GPTIMER5,			0x49039020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_GPTIMER5,			0x49039024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_GPTIMER5,			0x49039028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_GPTIMER6,			0x4903B020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_GPTIMER6,			0x4903B024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_GPTIMER6,			0x4903B028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_GPTIMER7,			0x4903D020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_GPTIMER7,			0x4903D024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_GPTIMER7,			0x4903D028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_GPTIMER8,			0x4903F020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_GPTIMER8,			0x4903F024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_GPTIMER8,			0x4903F028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_GPTIMER9,			0x49041020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_GPTIMER9,			0x49041024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_GPTIMER9,			0x49041028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_GPIO2,				0x49051020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_GPIO2,				0x49051024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_GPIO2,					0x49051028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_GPIO3,				0x49053020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_GPIO3,				0x49053024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_GPIO3,					0x49053028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_GPIO4,				0x49055020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_GPIO4,				0x49055024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_GPIO4,					0x49055028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_GPIO5,				0x49057020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_GPIO5,				0x49057024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_GPIO5,					0x49057028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_GPIO6,				0x49059020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_GPIO6,				0x49059024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_GPIO6,					0x49059028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_EMU_TA_TESTCHIPLEVELTAP,	0x54005020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_EMU_TA_TESTCHIPLEVELTAP,	0x54005024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_EMU_TA_TESTCHIPLEVELTAP,	0x54005028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_EMU_TA_MPU,					0x54018020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_EMU_TA_MPU,					0x54018024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_EMU_TA_MPU,						0x54018028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_EMU_TA_TPUI,					0x5401A020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_EMU_TA_TPUI,					0x5401A024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_EMU_TA_TPUI,					0x5401A028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_EMU_TA_ETB,					0x5401C020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_EMU_TA_ETB,					0x5401C024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_EMU_TA_ETB,						0x5401C028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_EMU_TA_DAPCTL,				0x5401E020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_EMU_TA_DAPCTL,				0x5401E024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_EMU_TA_DAPCTL,				0x5401E028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_EMU_TA_SDTI,					0x5401F020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_EMU_TA_SDTI,					0x5401F024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_EMU_TA_SDTI,					0x5401F028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_EMU_TA_L4WKUP,				0x54730020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_EMU_TA_L4WKUP,				0x54730024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_EMU_TA_L4WKUP,				0x54730028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_WKUP_TA_GPTIMER12,		0x48305020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_WKUP_TA_GPTIMER12,		0x48305024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_WKUP_TA_GPTIMER12,		0x48305028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_WKUP_TA_PRM,					0x48309020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_WKUP_TA_PRM,					0x48309024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_WKUP_TA_PRM,					0x48309028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_WKUP_TA_WDTIMER1,		0x4830D020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_WKUP_TA_WDTIMER1,		0x4830D024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_WKUP_TA_WDTIMER1,			0x4830D028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_WKUP_TA_GPIO1,				0x48311020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_WKUP_TA_GPIO1,				0x48311024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_WKUP_TA_GPIO1,				0x48311028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_WKUP_TA_WDTIMER,			0x48315020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_WKUP_TA_WDTIMER,			0x48315024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_WKUP_TA_WDTIMER,			0x48315028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_WKUP_TA_GPTIMER1,		0x48319020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_WKUP_TA_GPTIMER1,		0x48319024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_WKUP_TA_GPTIMER1,			0x48319028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_WKUP_TA_SYNCTIMER32K,0x48321020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_WKUP_TA_SYNCTIMER32K,0x48321024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_WKUP_TA_SYNCTIMER32K,	0x48321028,__READ_WRITE	,__l4_ta_agent_status_l_bits);

/***************************************************************************
 **
 ** L4 LA
 **
 ***************************************************************************/
__IO_REG32(    L4_LA_NETWORK_H_CORE,			          0x48041014,__READ     	);
__IO_REG32_BIT(L4_LA_INITIATOR_INFO_L_CORE,	        0x48041018,__READ     	,__l4_la_initiator_info_l_bits);
__IO_REG32_BIT(L4_LA_INITIATOR_INFO_H_CORE,	        0x4804101C,__READ     	,__l4_la_initiator_info_h_bits);
__IO_REG32_BIT(L4_LA_NETWORK_CONTROL_L_CORE,  		  0x48041020,__READ_WRITE ,__l4_la_network_control_l_bits);
__IO_REG32_BIT(L4_LA_NETWORK_CONTROL_H_CORE,  		  0x48041024,__READ_WRITE ,__l4_la_network_control_h_bits);
__IO_REG32(    L4_LA_NETWORK_H_PER,			            0x49001014,__READ     	);
__IO_REG32_BIT(L4_LA_INITIATOR_INFO_L_PER,	        0x49001018,__READ     	,__l4_la_initiator_info_l_bits);
__IO_REG32_BIT(L4_LA_INITIATOR_INFO_H_PER,	        0x4900101C,__READ     	,__l4_la_initiator_info_h_bits);
__IO_REG32_BIT(L4_LA_NETWORK_CONTROL_L_PER,  		    0x49001020,__READ_WRITE ,__l4_la_network_control_l_bits);
__IO_REG32_BIT(L4_LA_NETWORK_CONTROL_H_PER,  		    0x49001024,__READ_WRITE ,__l4_la_network_control_h_bits);
__IO_REG32(    L4_LA_NETWORK_H_EMU,			            0x54007014,__READ     	);
__IO_REG32_BIT(L4_LA_INITIATOR_INFO_L_EMU,	        0x54007018,__READ     	,__l4_la_initiator_info_l_bits);
__IO_REG32_BIT(L4_LA_INITIATOR_INFO_H_EMU,	        0x5400701C,__READ     	,__l4_la_initiator_info_h_bits);
__IO_REG32_BIT(L4_LA_NETWORK_CONTROL_L_EMU,  		    0x54007020,__READ_WRITE ,__l4_la_network_control_l_bits);
__IO_REG32_BIT(L4_LA_NETWORK_CONTROL_H_EMU,  		    0x54007024,__READ_WRITE ,__l4_la_network_control_h_bits);
__IO_REG32(    L4_LA_NETWORK_H_WKUP,			          0x48329014,__READ     	);
__IO_REG32_BIT(L4_LA_INITIATOR_INFO_L_WKUP,	        0x48329018,__READ     	,__l4_la_initiator_info_l_bits);
__IO_REG32_BIT(L4_LA_INITIATOR_INFO_H_WKUP,	        0x4832901C,__READ     	,__l4_la_initiator_info_h_bits);
__IO_REG32_BIT(L4_LA_NETWORK_CONTROL_L_WKUP,  		  0x48329020,__READ_WRITE ,__l4_la_network_control_l_bits);
__IO_REG32_BIT(L4_LA_NETWORK_CONTROL_H_WKUP,  		  0x48329024,__READ_WRITE ,__l4_la_network_control_h_bits);

/***************************************************************************
 **
 ** L4 LA CORE AP
 **
 ***************************************************************************/
__IO_REG32_BIT(L4_AP_SEGMENT_0_L_CORE_AP,									0x48040100,__READ_WRITE	,__l4_ap_segment_l_bits);
__IO_REG32_BIT(L4_AP_SEGMENT_0_H_CORE_AP,									0x48040104,__READ_WRITE	,__l4_ap_segment_h_bits);
__IO_REG32_BIT(L4_AP_SEGMENT_1_L_CORE_AP,									0x48040108,__READ_WRITE	,__l4_ap_segment_l_bits);
__IO_REG32_BIT(L4_AP_SEGMENT_1_H_CORE_AP,									0x4804010C,__READ_WRITE	,__l4_ap_segment_h_bits);
__IO_REG32_BIT(L4_AP_SEGMENT_2_L_CORE_AP,									0x48040110,__READ_WRITE	,__l4_ap_segment_l_bits);
__IO_REG32_BIT(L4_AP_SEGMENT_2_H_CORE_AP,									0x48040114,__READ_WRITE	,__l4_ap_segment_h_bits);
__IO_REG32_BIT(L4_AP_SEGMENT_3_L_CORE_AP,									0x48040118,__READ_WRITE	,__l4_ap_segment_l_bits);
__IO_REG32_BIT(L4_AP_SEGMENT_3_H_CORE_AP,									0x4804011C,__READ_WRITE	,__l4_ap_segment_h_bits);
__IO_REG32_BIT(L4_AP_SEGMENT_4_L_CORE_AP,									0x48040120,__READ_WRITE	,__l4_ap_segment_l_bits);
__IO_REG32_BIT(L4_AP_SEGMENT_4_H_CORE_AP,									0x48040124,__READ_WRITE	,__l4_ap_segment_h_bits);
__IO_REG32_BIT(L4_AP_SEGMENT_5_L_CORE_AP,									0x48040128,__READ_WRITE	,__l4_ap_segment_l_bits);
__IO_REG32_BIT(L4_AP_SEGMENT_5_H_CORE_AP,									0x4804012C,__READ_WRITE	,__l4_ap_segment_h_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_MEMBERS_0_L_CORE_AP,			0x48040200,__READ				,__l4_ap_prot_group_members_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_MEMBERS_1_L_CORE_AP,			0x48040208,__READ				,__l4_ap_prot_group_members_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_MEMBERS_2_L_CORE_AP,			0x48040210,__READ				,__l4_ap_prot_group_members_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_MEMBERS_3_L_CORE_AP,			0x48040218,__READ				,__l4_ap_prot_group_members_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_MEMBERS_4_L_CORE_AP,			0x48040220,__READ				,__l4_ap_prot_group_members_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_MEMBERS_5_L_CORE_AP,			0x48040228,__READ				,__l4_ap_prot_group_members_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_MEMBERS_6_L_CORE_AP,			0x48040230,__READ				,__l4_ap_prot_group_members_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_MEMBERS_7_L_CORE_AP,			0x48040238,__READ				,__l4_ap_prot_group_members_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_ROLES_0_L_CORE_AP,				0x48040280,__READ				,__l4_ap_prot_group_roles_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_ROLES_1_L_CORE_AP,				0x48040288,__READ				,__l4_ap_prot_group_roles_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_ROLES_2_L_CORE_AP,				0x48040290,__READ				,__l4_ap_prot_group_roles_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_ROLES_3_L_CORE_AP,				0x48040298,__READ				,__l4_ap_prot_group_roles_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_ROLES_4_L_CORE_AP,				0x480402A0,__READ				,__l4_ap_prot_group_roles_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_ROLES_5_L_CORE_AP,				0x480402A8,__READ				,__l4_ap_prot_group_roles_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_ROLES_6_L_CORE_AP,				0x480402B0,__READ				,__l4_ap_prot_group_roles_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_ROLES_7_L_CORE_AP,				0x480402B8,__READ				,__l4_ap_prot_group_roles_l_bits);
__IO_REG32_BIT(L4_AP_REGION_0_L_CORE_AP,									0x48040300,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_0_H_CORE_AP,									0x48040304,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_1_L_CORE_AP,									0x48040308,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_1_H_CORE_AP,									0x4804030C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_2_L_CORE_AP,									0x48040310,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_2_H_CORE_AP,									0x48040314,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_3_L_CORE_AP,									0x48040318,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_3_H_CORE_AP,									0x4804031C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_4_L_CORE_AP,									0x48040320,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_4_H_CORE_AP,									0x48040324,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_5_L_CORE_AP,									0x48040328,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_5_H_CORE_AP,									0x4804032C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_6_L_CORE_AP,									0x48040330,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_6_H_CORE_AP,									0x48040334,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_7_L_CORE_AP,									0x48040338,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_7_H_CORE_AP,									0x4804033C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_8_L_CORE_AP,									0x48040340,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_8_H_CORE_AP,									0x48040344,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_9_L_CORE_AP,									0x48040348,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_9_H_CORE_AP,									0x4804034C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_10_L_CORE_AP,									0x48040350,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_10_H_CORE_AP,									0x48040354,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_11_L_CORE_AP,									0x48040358,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_11_H_CORE_AP,									0x4804035C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_12_L_CORE_AP,									0x48040360,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_12_H_CORE_AP,									0x48040364,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_13_L_CORE_AP,									0x48040368,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_13_H_CORE_AP,									0x4804036C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_14_L_CORE_AP,									0x48040370,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_14_H_CORE_AP,									0x48040374,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_15_L_CORE_AP,									0x48040378,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_15_H_CORE_AP,									0x4804037C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_16_L_CORE_AP,									0x48040380,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_16_H_CORE_AP,									0x48040384,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_17_L_CORE_AP,									0x48040388,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_17_H_CORE_AP,									0x4804038C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_18_L_CORE_AP,									0x48040390,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_18_H_CORE_AP,									0x48040394,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_19_L_CORE_AP,									0x48040398,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_19_H_CORE_AP,									0x4804039C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_20_L_CORE_AP,									0x480403A0,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_20_H_CORE_AP,									0x480403A4,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_21_L_CORE_AP,									0x480403A8,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_21_H_CORE_AP,									0x480403AC,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_22_L_CORE_AP,									0x480403B0,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_22_H_CORE_AP,									0x480403B4,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_23_L_CORE_AP,									0x480403B8,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_23_H_CORE_AP,									0x480403BC,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_24_L_CORE_AP,									0x480403C0,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_24_H_CORE_AP,									0x480403C4,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_25_L_CORE_AP,									0x480403C8,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_25_H_CORE_AP,									0x480403CC,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_26_L_CORE_AP,									0x480403D0,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_26_H_CORE_AP,									0x480403D4,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_27_L_CORE_AP,									0x480403D8,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_27_H_CORE_AP,									0x480403DC,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_28_L_CORE_AP,									0x480403E0,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_28_H_CORE_AP,									0x480403E4,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_29_L_CORE_AP,									0x480403E8,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_29_H_CORE_AP,									0x480403EC,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_30_L_CORE_AP,									0x480403F0,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_30_H_CORE_AP,									0x480403F4,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_31_L_CORE_AP,									0x480403F8,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_31_H_CORE_AP,									0x480403FC,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_32_L_CORE_AP,									0x48040400,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_32_H_CORE_AP,									0x48040404,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_33_L_CORE_AP,									0x48040408,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_33_H_CORE_AP,									0x4804040C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_34_L_CORE_AP,									0x48040410,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_34_H_CORE_AP,									0x48040414,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_35_L_CORE_AP,									0x48040418,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_35_H_CORE_AP,									0x4804041C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_36_L_CORE_AP,									0x48040420,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_36_H_CORE_AP,									0x48040424,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_37_L_CORE_AP,									0x48040428,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_37_H_CORE_AP,									0x4804042C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_38_L_CORE_AP,									0x48040430,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_38_H_CORE_AP,									0x48040434,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_39_L_CORE_AP,									0x48040438,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_39_H_CORE_AP,									0x4804043C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_40_L_CORE_AP,									0x48040440,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_40_H_CORE_AP,									0x48040444,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_41_L_CORE_AP,									0x48040448,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_41_H_CORE_AP,									0x4804044C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_42_L_CORE_AP,									0x48040450,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_42_H_CORE_AP,									0x48040454,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_43_L_CORE_AP,									0x48040458,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_43_H_CORE_AP,									0x4804045C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_44_L_CORE_AP,									0x48040460,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_44_H_CORE_AP,									0x48040464,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_45_L_CORE_AP,									0x48040468,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_45_H_CORE_AP,									0x4804046C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_46_L_CORE_AP,									0x48040470,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_46_H_CORE_AP,									0x48040474,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_47_L_CORE_AP,									0x48040478,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_47_H_CORE_AP,									0x4804047C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_48_L_CORE_AP,									0x48040480,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_48_H_CORE_AP,									0x48040484,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_49_L_CORE_AP,									0x48040488,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_49_H_CORE_AP,									0x4804048C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_50_L_CORE_AP,									0x48040490,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_50_H_CORE_AP,									0x48040494,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_51_L_CORE_AP,									0x48040498,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_51_H_CORE_AP,									0x4804049C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_52_L_CORE_AP,									0x480404A0,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_52_H_CORE_AP,									0x480404A4,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_53_L_CORE_AP,									0x480404A8,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_53_H_CORE_AP,									0x480404AC,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_54_L_CORE_AP,									0x480404B0,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_54_H_CORE_AP,									0x480404B4,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_55_L_CORE_AP,									0x480404B8,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_55_H_CORE_AP,									0x480404BC,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_56_L_CORE_AP,									0x480404C0,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_56_H_CORE_AP,									0x480404C4,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_57_L_CORE_AP,									0x480404C8,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_57_H_CORE_AP,									0x480404CC,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_58_L_CORE_AP,									0x480404D0,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_58_H_CORE_AP,									0x480404D4,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_59_L_CORE_AP,									0x480404D8,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_59_H_CORE_AP,									0x480404DC,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_60_L_CORE_AP,									0x480404E0,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_60_H_CORE_AP,									0x480404E4,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_61_L_CORE_AP,									0x480404E8,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_61_H_CORE_AP,									0x480404EC,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_62_L_CORE_AP,									0x480404F0,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_62_H_CORE_AP,									0x480404F4,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_63_L_CORE_AP,									0x480404F8,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_63_H_CORE_AP,									0x480404FC,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_64_L_CORE_AP,									0x48040500,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_64_H_CORE_AP,									0x48040504,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_65_L_CORE_AP,									0x48040508,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_65_H_CORE_AP,									0x4804050C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_66_L_CORE_AP,									0x48040510,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_66_H_CORE_AP,									0x48040514,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_67_L_CORE_AP,									0x48040518,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_67_H_CORE_AP,									0x4804051C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_68_L_CORE_AP,									0x48040520,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_68_H_CORE_AP,									0x48040524,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_69_L_CORE_AP,									0x48040528,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_69_H_CORE_AP,									0x4804052C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_70_L_CORE_AP,									0x48040530,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_70_H_CORE_AP,									0x48040534,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_71_L_CORE_AP,									0x48040538,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_71_H_CORE_AP,									0x4804053C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_72_L_CORE_AP,									0x48040540,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_72_H_CORE_AP,									0x48040544,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_73_L_CORE_AP,									0x48040548,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_73_H_CORE_AP,									0x4804054C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_74_L_CORE_AP,									0x48040550,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_74_H_CORE_AP,									0x48040554,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_75_L_CORE_AP,									0x48040558,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_75_H_CORE_AP,									0x4804055C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_76_L_CORE_AP,									0x48040560,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_76_H_CORE_AP,									0x48040564,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_77_L_CORE_AP,									0x48040568,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_77_H_CORE_AP,									0x4804056C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_78_L_CORE_AP,									0x48040570,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_78_H_CORE_AP,									0x48040574,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_79_L_CORE_AP,									0x48040578,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_79_H_CORE_AP,									0x4804057C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_80_L_CORE_AP,									0x48040580,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_80_H_CORE_AP,									0x48040584,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_81_L_CORE_AP,									0x48040588,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_81_H_CORE_AP,									0x4804058C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_82_L_CORE_AP,									0x48040590,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_82_H_CORE_AP,									0x48040594,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_83_L_CORE_AP,									0x48040598,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_83_H_CORE_AP,									0x4804059C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_84_L_CORE_AP,									0x480405A0,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_84_H_CORE_AP,									0x480405A4,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_85_L_CORE_AP,									0x480405A8,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_85_H_CORE_AP,									0x480405AC,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_86_L_CORE_AP,									0x480405B0,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_86_H_CORE_AP,									0x480405B4,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_87_L_CORE_AP,									0x480405B8,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_87_H_CORE_AP,									0x480405BC,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_88_L_CORE_AP,									0x480405C0,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_88_H_CORE_AP,									0x480405C4,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_89_L_CORE_AP,									0x480405C8,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_89_H_CORE_AP,									0x480405CC,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_90_L_CORE_AP,									0x480405D0,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_90_H_CORE_AP,									0x480405D4,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_91_L_CORE_AP,									0x480405D8,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_91_H_CORE_AP,									0x480405DC,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_92_L_CORE_AP,									0x480405E0,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_92_H_CORE_AP,									0x480405E4,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_93_L_CORE_AP,									0x480405E8,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_93_H_CORE_AP,									0x480405EC,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_94_L_CORE_AP,									0x480405F0,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_94_H_CORE_AP,									0x480405F4,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_95_L_CORE_AP,									0x480405F8,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_95_H_CORE_AP,									0x480405FC,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_96_L_CORE_AP,									0x48040600,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_96_H_CORE_AP,									0x48040604,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_97_L_CORE_AP,									0x48040608,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_97_H_CORE_AP,									0x4804060C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_98_L_CORE_AP,									0x48040610,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_98_H_CORE_AP,									0x48040614,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_99_L_CORE_AP,									0x48040618,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_99_H_CORE_AP,									0x4804061C,__READ_WRITE	,__l4_ap_region_h_bits);


/***************************************************************************
 **
 ** L4 LA PER AP
 **
 ***************************************************************************/
__IO_REG32_BIT(L4_AP_SEGMENT_0_L_PER_AP,									0x49000100,__READ_WRITE	,__l4_ap_segment_l_bits);
__IO_REG32_BIT(L4_AP_SEGMENT_0_H_PER_AP,									0x49000104,__READ_WRITE	,__l4_ap_segment_h_bits);
__IO_REG32_BIT(L4_AP_SEGMENT_1_L_PER_AP,									0x49000108,__READ_WRITE	,__l4_ap_segment_l_bits);
__IO_REG32_BIT(L4_AP_SEGMENT_1_H_PER_AP,									0x4900010C,__READ_WRITE	,__l4_ap_segment_h_bits);
__IO_REG32_BIT(L4_AP_SEGMENT_2_L_PER_AP,									0x49000110,__READ_WRITE	,__l4_ap_segment_l_bits);
__IO_REG32_BIT(L4_AP_SEGMENT_2_H_PER_AP,									0x49000114,__READ_WRITE	,__l4_ap_segment_h_bits);
__IO_REG32_BIT(L4_AP_SEGMENT_3_L_PER_AP,									0x49000118,__READ_WRITE	,__l4_ap_segment_l_bits);
__IO_REG32_BIT(L4_AP_SEGMENT_3_H_PER_AP,									0x4900011C,__READ_WRITE	,__l4_ap_segment_h_bits);
__IO_REG32_BIT(L4_AP_SEGMENT_4_L_PER_AP,									0x49000120,__READ_WRITE	,__l4_ap_segment_l_bits);
__IO_REG32_BIT(L4_AP_SEGMENT_4_H_PER_AP,									0x49000124,__READ_WRITE	,__l4_ap_segment_h_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_MEMBERS_0_L_PER_AP,				0x49000200,__READ				,__l4_ap_prot_group_members_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_MEMBERS_1_L_PER_AP,				0x49000208,__READ				,__l4_ap_prot_group_members_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_MEMBERS_2_L_PER_AP,				0x49000210,__READ				,__l4_ap_prot_group_members_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_MEMBERS_3_L_PER_AP,				0x49000218,__READ				,__l4_ap_prot_group_members_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_MEMBERS_4_L_PER_AP,				0x49000220,__READ				,__l4_ap_prot_group_members_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_MEMBERS_5_L_PER_AP,				0x49000228,__READ				,__l4_ap_prot_group_members_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_MEMBERS_6_L_PER_AP,				0x49000230,__READ				,__l4_ap_prot_group_members_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_MEMBERS_7_L_PER_AP,				0x49000238,__READ				,__l4_ap_prot_group_members_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_ROLES_0_L_PER_AP,					0x49000280,__READ				,__l4_ap_prot_group_roles_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_ROLES_1_L_PER_AP,					0x49000288,__READ				,__l4_ap_prot_group_roles_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_ROLES_2_L_PER_AP,					0x49000290,__READ				,__l4_ap_prot_group_roles_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_ROLES_3_L_PER_AP,					0x49000298,__READ				,__l4_ap_prot_group_roles_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_ROLES_4_L_PER_AP,					0x490002A0,__READ				,__l4_ap_prot_group_roles_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_ROLES_5_L_PER_AP,					0x490002A8,__READ				,__l4_ap_prot_group_roles_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_ROLES_6_L_PER_AP,					0x490002B0,__READ				,__l4_ap_prot_group_roles_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_ROLES_7_L_PER_AP,					0x490002B8,__READ				,__l4_ap_prot_group_roles_l_bits);
__IO_REG32_BIT(L4_AP_REGION_0_L_PER_AP,										0x49000300,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_0_H_PER_AP,										0x49000304,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_1_L_PER_AP,										0x49000308,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_1_H_PER_AP,										0x4900030C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_2_L_PER_AP,										0x49000310,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_2_H_PER_AP,										0x49000314,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_3_L_PER_AP,										0x49000318,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_3_H_PER_AP,										0x4900031C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_4_L_PER_AP,										0x49000320,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_4_H_PER_AP,										0x49000324,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_5_L_PER_AP,										0x49000328,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_5_H_PER_AP,										0x4900032C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_6_L_PER_AP,										0x49000330,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_6_H_PER_AP,										0x49000334,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_7_L_PER_AP,										0x49000338,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_7_H_PER_AP,										0x4900033C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_8_L_PER_AP,										0x49000340,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_8_H_PER_AP,										0x49000344,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_9_L_PER_AP,										0x49000348,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_9_H_PER_AP,										0x4900034C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_10_L_PER_AP,									0x49000350,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_10_H_PER_AP,									0x49000354,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_11_L_PER_AP,									0x49000358,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_11_H_PER_AP,									0x4900035C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_12_L_PER_AP,									0x49000360,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_12_H_PER_AP,									0x49000364,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_13_L_PER_AP,									0x49000368,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_13_H_PER_AP,									0x4900036C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_14_L_PER_AP,									0x49000370,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_14_H_PER_AP,									0x49000374,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_15_L_PER_AP,									0x49000378,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_15_H_PER_AP,									0x4900037C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_16_L_PER_AP,									0x49000380,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_16_H_PER_AP,									0x49000384,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_17_L_PER_AP,									0x49000388,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_17_H_PER_AP,									0x4900038C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_18_L_PER_AP,									0x49000390,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_18_H_PER_AP,									0x49000394,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_19_L_PER_AP,									0x49000398,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_19_H_PER_AP,									0x4900039C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_20_L_PER_AP,									0x490003A0,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_20_H_PER_AP,									0x490003A4,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_21_L_PER_AP,									0x490003A8,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_21_H_PER_AP,									0x490003AC,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_22_L_PER_AP,									0x490003B0,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_22_H_PER_AP,									0x490003B4,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_23_L_PER_AP,									0x490003B8,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_23_H_PER_AP,									0x490003BC,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_24_L_PER_AP,									0x490003C0,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_24_H_PER_AP,									0x490003C4,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_25_L_PER_AP,									0x490003C8,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_25_H_PER_AP,									0x490003CC,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_26_L_PER_AP,									0x490003D0,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_26_H_PER_AP,									0x490003D4,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_27_L_PER_AP,									0x490003D8,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_27_H_PER_AP,									0x490003DC,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_28_L_PER_AP,									0x490003E0,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_28_H_PER_AP,									0x490003E4,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_29_L_PER_AP,									0x490003E8,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_29_H_PER_AP,									0x490003EC,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_30_L_PER_AP,									0x490003F0,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_30_H_PER_AP,									0x490003F4,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_31_L_PER_AP,									0x490003F8,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_31_H_PER_AP,									0x490003FC,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_32_L_PER_AP,									0x49000400,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_32_H_PER_AP,									0x49000404,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_33_L_PER_AP,									0x49000408,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_33_H_PER_AP,									0x4900040C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_34_L_PER_AP,									0x49000410,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_34_H_PER_AP,									0x49000414,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_35_L_PER_AP,									0x49000418,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_35_H_PER_AP,									0x4900041C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_36_L_PER_AP,									0x49000420,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_36_H_PER_AP,									0x49000424,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_37_L_PER_AP,									0x49000428,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_37_H_PER_AP,									0x4900042C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_38_L_PER_AP,									0x49000430,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_38_H_PER_AP,									0x49000434,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_39_L_PER_AP,									0x49000438,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_39_H_PER_AP,									0x4900043C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_40_L_PER_AP,									0x49000440,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_40_H_PER_AP,									0x49000444,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_41_L_PER_AP,									0x49000448,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_41_H_PER_AP,									0x4900044C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_42_L_PER_AP,									0x49000450,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_42_H_PER_AP,									0x49000454,__READ_WRITE	,__l4_ap_region_h_bits);

/***************************************************************************
 **
 ** L4 LA EMU AP
 **
 ***************************************************************************/
__IO_REG32_BIT(L4_AP_SEGMENT_0_L_EMU_AP,									0x54006100,__READ_WRITE	,__l4_ap_segment_l_bits);
__IO_REG32_BIT(L4_AP_SEGMENT_0_H_EMU_AP,									0x54006104,__READ_WRITE	,__l4_ap_segment_h_bits);
__IO_REG32_BIT(L4_AP_SEGMENT_1_L_EMU_AP,									0x54006108,__READ_WRITE	,__l4_ap_segment_l_bits);
__IO_REG32_BIT(L4_AP_SEGMENT_1_H_EMU_AP,									0x5400610C,__READ_WRITE	,__l4_ap_segment_h_bits);
__IO_REG32_BIT(L4_AP_SEGMENT_2_L_EMU_AP,									0x54006110,__READ_WRITE	,__l4_ap_segment_l_bits);
__IO_REG32_BIT(L4_AP_SEGMENT_2_H_EMU_AP,									0x54006114,__READ_WRITE	,__l4_ap_segment_h_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_MEMBERS_0_L_EMU_AP,				0x54006200,__READ				,__l4_ap_prot_group_members_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_MEMBERS_1_L_EMU_AP,				0x54006208,__READ				,__l4_ap_prot_group_members_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_MEMBERS_2_L_EMU_AP,				0x54006210,__READ				,__l4_ap_prot_group_members_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_MEMBERS_3_L_EMU_AP,				0x54006218,__READ				,__l4_ap_prot_group_members_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_MEMBERS_4_L_EMU_AP,				0x54006220,__READ				,__l4_ap_prot_group_members_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_MEMBERS_5_L_EMU_AP,				0x54006228,__READ				,__l4_ap_prot_group_members_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_ROLES_0_L_EMU_AP,					0x54006280,__READ				,__l4_ap_prot_group_roles_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_ROLES_1_L_EMU_AP,					0x54006288,__READ				,__l4_ap_prot_group_roles_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_ROLES_2_L_EMU_AP,					0x54006290,__READ				,__l4_ap_prot_group_roles_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_ROLES_3_L_EMU_AP,					0x54006298,__READ				,__l4_ap_prot_group_roles_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_ROLES_4_L_EMU_AP,					0x540062A0,__READ				,__l4_ap_prot_group_roles_l_bits);
__IO_REG32_BIT(L4_AP_PROT_GROUP_ROLES_5_L_EMU_AP,					0x540062A8,__READ				,__l4_ap_prot_group_roles_l_bits);
__IO_REG32_BIT(L4_AP_REGION_0_L_EMU_AP,										0x54006300,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_0_H_EMU_AP,										0x54006304,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_1_L_EMU_AP,										0x54006308,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_1_H_EMU_AP,										0x5400630C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_2_L_EMU_AP,										0x54006310,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_2_H_EMU_AP,										0x54006314,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_3_L_EMU_AP,										0x54006318,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_3_H_EMU_AP,										0x5400631C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_4_L_EMU_AP,										0x54006320,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_4_H_EMU_AP,										0x54006324,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_5_L_EMU_AP,										0x54006328,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_5_H_EMU_AP,										0x5400632C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_6_L_EMU_AP,										0x54006330,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_6_H_EMU_AP,										0x54006334,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_7_L_EMU_AP,										0x54006338,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_7_H_EMU_AP,										0x5400633C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_8_L_EMU_AP,										0x54006340,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_8_H_EMU_AP,										0x54006344,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_9_L_EMU_AP,										0x54006348,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_9_H_EMU_AP,										0x5400634C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_10_L_EMU_AP,									0x54006350,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_10_H_EMU_AP,									0x54006354,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_11_L_EMU_AP,									0x54006358,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_11_H_EMU_AP,									0x5400635C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_12_L_EMU_AP,									0x54006360,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_12_H_EMU_AP,									0x54006364,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_13_L_EMU_AP,									0x54006368,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_13_H_EMU_AP,									0x5400636C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_14_L_EMU_AP,									0x54006370,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_14_H_EMU_AP,									0x54006374,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_15_L_EMU_AP,									0x54006378,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_15_H_EMU_AP,									0x5400637C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_16_L_EMU_AP,									0x54006380,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_16_H_EMU_AP,									0x54006384,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_17_L_EMU_AP,									0x54006388,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_17_H_EMU_AP,									0x5400638C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_18_L_EMU_AP,									0x54006390,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_18_H_EMU_AP,									0x54006394,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_19_L_EMU_AP,									0x54006398,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_19_H_EMU_AP,									0x5400639C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_20_L_EMU_AP,									0x540063A0,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_20_H_EMU_AP,									0x540063A4,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_21_L_EMU_AP,									0x540063A8,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_21_H_EMU_AP,									0x540063AC,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_22_L_EMU_AP,									0x540063B0,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_22_H_EMU_AP,									0x540063B4,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_23_L_EMU_AP,									0x540063B8,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_23_H_EMU_AP,									0x540063BC,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_24_L_EMU_AP,									0x540063C0,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_24_H_EMU_AP,									0x540063C4,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_25_L_EMU_AP,									0x540063C8,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_25_H_EMU_AP,									0x540063CC,__READ_WRITE	,__l4_ap_region_h_bits);

/***************************************************************************
 **
 ** L4 LA WKUP AP
 **
 ***************************************************************************/
__IO_REG32_BIT(L4_AP_SEGMENT_0_L_WKUP_AP,									0x48328100,__READ_WRITE	,__l4_ap_segment_l_bits);
__IO_REG32_BIT(L4_AP_SEGMENT_0_H_WKUP_AP,									0x48328104,__READ_WRITE	,__l4_ap_segment_h_bits);
__IO_REG32_BIT(L4_AP_SEGMENT_1_L_WKUP_AP,									0x48328108,__READ_WRITE	,__l4_ap_segment_l_bits);
__IO_REG32_BIT(L4_AP_SEGMENT_1_H_WKUP_AP,									0x4832810C,__READ_WRITE	,__l4_ap_segment_h_bits);
__IO_REG32_BIT(L4_AP_REGION_0_L_WKUP_AP,									0x48328300,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_0_H_WKUP_AP,									0x48328304,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_1_L_WKUP_AP,									0x48328308,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_1_H_WKUP_AP,									0x4832830C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_2_L_WKUP_AP,									0x48328310,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_2_H_WKUP_AP,									0x48328314,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_3_L_WKUP_AP,									0x48328318,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_3_H_WKUP_AP,									0x4832831C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_4_L_WKUP_AP,									0x48328320,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_4_H_WKUP_AP,									0x48328324,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_5_L_WKUP_AP,									0x48328328,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_5_H_WKUP_AP,									0x4832832C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_6_L_WKUP_AP,									0x48328330,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_6_H_WKUP_AP,									0x48328334,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_7_L_WKUP_AP,									0x48328338,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_7_H_WKUP_AP,									0x4832833C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_8_L_WKUP_AP,									0x48328340,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_8_H_WKUP_AP,									0x48328344,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_9_L_WKUP_AP,									0x48328348,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_9_H_WKUP_AP,									0x4832834C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_10_L_WKUP_AP,									0x48328350,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_10_H_WKUP_AP,									0x48328354,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_11_L_WKUP_AP,									0x48328358,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_11_H_WKUP_AP,									0x4832835C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_12_L_WKUP_AP,									0x48328360,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_12_H_WKUP_AP,									0x48328364,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_13_L_WKUP_AP,									0x48328368,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_13_H_WKUP_AP,									0x4832836C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_14_L_WKUP_AP,									0x48328370,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_14_H_WKUP_AP,									0x48328374,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_15_L_WKUP_AP,									0x48328378,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_15_H_WKUP_AP,									0x4832837C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_16_L_WKUP_AP,									0x48328380,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_16_H_WKUP_AP,									0x48328384,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_17_L_WKUP_AP,									0x48328388,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_17_H_WKUP_AP,									0x4832838C,__READ_WRITE	,__l4_ap_region_h_bits);
__IO_REG32_BIT(L4_AP_REGION_18_L_WKUP_AP,									0x48328390,__READ_WRITE	,__l4_ap_region_l_bits);
__IO_REG32_BIT(L4_AP_REGION_18_H_WKUP_AP,									0x48328394,__READ_WRITE	,__l4_ap_region_h_bits);

/***************************************************************************
 **
 ** SCM
 **
 ***************************************************************************/
__IO_REG32_BIT(SCM_CONTROL_REVISION,											0x48002000,__READ				,__scm_control_revision_bits);
__IO_REG32_BIT(SCM_CONTROL_SYSCONFIG,											0x48002010,__READ_WRITE	,__scm_control_sysconfig_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SDRC_D0,								0x48002030,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SDRC_D2,								0x48002034,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SDRC_D4,								0x48002038,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SDRC_D6,								0x4800203C,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SDRC_D8,								0x48002040,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SDRC_D10,							0x48002044,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SDRC_D12,							0x48002048,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SDRC_D14,							0x4800204C,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SDRC_D16,							0x48002050,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SDRC_D18,							0x48002054,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SDRC_D20,							0x48002058,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SDRC_D22,							0x4800205C,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SDRC_D24,							0x48002060,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SDRC_D26,							0x48002064,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SDRC_D28,							0x48002068,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SDRC_D30,							0x4800206C,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SDRC_CLK,							0x48002070,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SDRC_CKE0,							0x48002260,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SDRC_CKE1,							0x48002264,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SDRC_DQS1,							0x48002074,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SDRC_DQS3,							0x48002078,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_GPMC_A2,								0x4800207C,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_GPMC_A4,								0x48002080,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_GPMC_A6,								0x48002084,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_GPMC_A8,								0x48002088,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_GPMC_A10,							0x4800208C,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_GPMC_D1,								0x48002090,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_GPMC_D3,								0x48002094,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_GPMC_D5,								0x48002098,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_GPMC_D7,								0x4800209C,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_GPMC_D9,								0x480020A0,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_GPMC_D11,							0x480020A4,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_GPMC_D13,							0x480020A8,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_GPMC_D15,							0x480020AC,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_GPMC_NCS1,							0x480020B0,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_GPMC_NCS3,							0x480020B4,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_GPMC_NCS5,							0x480020B8,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_GPMC_NCS7,							0x480020BC,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_GPMC_NADV_ALE,					0x480020C0,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_GPMC_NWE,							0x480020C4,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_GPMC_NBE1,							0x480020C8,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_GPMC_WAIT0,						0x480020CC,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_GPMC_WAIT2,						0x480020D0,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_DSS_PCLK,							0x480020D4,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_DSS_VSYNC,							0x480020D8,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_DSS_DATA0,							0x480020DC,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_DSS_DATA2,							0x480020E0,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_DSS_DATA4,							0x480020E4,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_DSS_DATA6,							0x480020E8,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_DSS_DATA8,							0x480020EC,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_DSS_DATA10,						0x480020F0,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_DSS_DATA12,						0x480020F4,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_DSS_DATA14,						0x480020F8,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_DSS_DATA16,						0x480020FC,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_DSS_DATA18,						0x48002100,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_DSS_DATA20,						0x48002104,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_DSS_DATA22,						0x48002108,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_MCBSP2_FSX,						0x4800213C,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_MCBSP2_DR,							0x48002140,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_MMC1_CLK,							0x48002144,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_MMC1_DAT0,							0x48002148,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_MMC1_DAT2,							0x4800214C,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_MMC1_DAT4,							0x48002150,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_MMC1_DAT6,							0x48002154,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_MMC2_CLK,							0x48002158,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_MMC2_DAT0,							0x4800215C,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_MMC2_DAT2,							0x48002160,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_MMC2_DAT4,							0x48002164,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_MMC2_DAT6,							0x48002168,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_MCBSP3_DX,							0x4800216C,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_MCBSP3_CLKX,						0x48002170,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_UART2_CTS,							0x48002174,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_UART2_TX,							0x48002178,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_UART1_TX,							0x4800217C,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_UART1_CTS,							0x48002180,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_MCBSP4_CLKX,						0x48002184,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_MCBSP4_DX,							0x48002188,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_MCBSP1_CLKR,						0x4800218C,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_MCBSP1_DX,							0x48002190,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_MCBSP_CLKS,						0x48002194,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_MCBSP1_CLKX,						0x48002198,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_UART3_RTS_SD,					0x4800219C,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_UART3_TX_IRTX,					0x480021A0,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_I2C1_SCL,							0x480021B8,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_I2C1_SDA,							0x480021BC,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_I2C2_SDA,							0x480021C0,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_I2C3_SDA,							0x480021C4,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_MCSPI1_CLK,						0x480021C8,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_MCSPI1_SOMI,						0x480021CC,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_MCSPI1_CS1,						0x480021D0,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_MCSPI1_CS3,						0x480021D4,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_MCSPI2_SIMO,						0x480021D8,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_MCSPI2_CS0,						0x480021DC,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SYS_NIRQ,							0x480021E0,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_ETK_CLK,								0x480025D8,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_ETK_D0,								0x480025DC,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_ETK_D2,								0x480025E0,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_ETK_D4,								0x480025E4,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_ETK_D6,								0x480025E8,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_ETK_D8,								0x480025EC,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_ETK_D10,								0x480025F0,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_ETK_D12,								0x480025F4,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_ETK_D14,								0x480025F8,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_CCDC_PCLK,							0x480021E4,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_CCDC_HD,								0x480021E8,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_CCDC_WEN,							0x480021EC,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_CCDC_DATA1,						0x480021F0,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_CCDC_DATA3,						0x480021F4,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_CCDC_DATA5,						0x480021F8,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_CCDC_DATA7,						0x480021FC,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_RMII_MDIO_CLK,					0x48002200,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_RMII_RXD1,							0x48002204,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_RMII_RXER,							0x48002208,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_RMII_TXD1,							0x4800220C,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_RMII_50MHZ_CLK,				0x48002210,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_HECC1_TXD,							0x48002214,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SYS_BOOT7,							0x48002218,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SDRC_DQS1N,						0x4800221C,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SDRC_DQS3N,						0x48002220,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SDRC_STRBEN_DLY1,			0x48002224,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_OFF,										0x48002270,__READ_WRITE	,__scm_control_general_bits);
__IO_REG32_BIT(SCM_CONTROL_DEVCONF0,											0x48002274,__READ_WRITE	,__scm_control_devconf0_bits);
__IO_REG32_BIT(SCM_CONTROL_MEM_DFTRW0,										0x48002278,__READ_WRITE	,__scm_control_mem_dftrw0_bits);
__IO_REG32_BIT(SCM_CONTROL_MEM_DFTRW1,										0x4800227C,__READ_WRITE	,__scm_control_mem_dftrw1_bits);
__IO_REG32_BIT(SCM_CONTROL_MSUSPENDMUX_0,									0x48002290,__READ_WRITE	,__scm_control_msuspendmux_0_bits);
__IO_REG32_BIT(SCM_CONTROL_MSUSPENDMUX_1,									0x48002294,__READ_WRITE	,__scm_control_msuspendmux_1_bits);
__IO_REG32_BIT(SCM_CONTROL_MSUSPENDMUX_2,									0x48002298,__READ_WRITE	,__scm_control_msuspendmux_2_bits);
__IO_REG32_BIT(SCM_CONTROL_MSUSPENDMUX_4,									0x480022A0,__READ_WRITE	,__scm_control_msuspendmux_4_bits);
__IO_REG32_BIT(SCM_CONTROL_MSUSPENDMUX_5,									0x480022A4,__READ_WRITE	,__scm_control_msuspendmux_5_bits);
__IO_REG32_BIT(SCM_CONTROL_MSUSPENDMUX_6,									0x480022A8,__READ_WRITE	,__scm_control_msuspendmux_6_bits);
__IO_REG32_BIT(SCM_CONTROL_DEVCONF1,											0x480022D8,__READ_WRITE	,__scm_control_devconf1_bits);
__IO_REG32_BIT(SCM_CONTROL_SEC_STATUS,										0x480022E0,__READ_WRITE	,__scm_control_sec_status_bits);
__IO_REG32_BIT(SCM_CONTROL_SEC_ERR_STATUS,								0x480022E4,__READ_WRITE	,__scm_control_sec_err_status_bits);
__IO_REG32_BIT(SCM_CONTROL_SEC_ERR_STATUS_DEBUG,					0x480022E8,__READ_WRITE	,__scm_control_sec_err_status_debug_bits);
__IO_REG32_BIT(SCM_CONTROL_STATUS,												0x480022F0,__READ				,__scm_control_status_bits);
__IO_REG32(		 SCM_CONTROL_RPUB_KEY_H_0,									0x48002300,__READ				);
__IO_REG32(		 SCM_CONTROL_RPUB_KEY_H_1,									0x48002304,__READ				);
__IO_REG32(		 SCM_CONTROL_RPUB_KEY_H_2,									0x48002308,__READ				);
__IO_REG32(		 SCM_CONTROL_RPUB_KEY_H_3,									0x4800230C,__READ				);
__IO_REG32(		 SCM_CONTROL_RPUB_KEY_H_4,									0x48002310,__READ				);
__IO_REG32(		 SCM_CONTROL_USB_CONF_0,										0x48002370,__READ				);
__IO_REG32(		 SCM_CONTROL_USB_CONF_1,										0x48002374,__READ				);
__IO_REG32_BIT(SCM_CONTROL_FUSE_EMAC_LSB,									0x48002380,__READ				,__scm_control_fuse_emac_lsb_bits);
__IO_REG32_BIT(SCM_CONTROL_FUSE_EMAC_MSB,									0x48002384,__READ				,__scm_control_fuse_emac_msb_bits);
__IO_REG32_BIT(SCM_CONTROL_FUSE_SR,												0x480023A0,__READ_WRITE	,__scm_control_fuse_sr_bits);
__IO_REG32(		 SCM_CONTROL_CEK_0,													0x480023A4,__READ_WRITE	);
__IO_REG32(		 SCM_CONTROL_CEK_1,													0x480023A8,__READ_WRITE	);
__IO_REG32(		 SCM_CONTROL_CEK_2,													0x480023AC,__READ_WRITE	);
__IO_REG32(		 SCM_CONTROL_CEK_3,													0x480023B0,__READ_WRITE	);
__IO_REG32(		 SCM_CONTROL_MSV_0,													0x480023B4,__READ_WRITE	);
__IO_REG32(		 SCM_CONTROL_CEK_BCH_0,											0x480023B8,__READ_WRITE	);
__IO_REG32(		 SCM_CONTROL_CEK_BCH_1,											0x480023BC,__READ_WRITE	);
__IO_REG32(		 SCM_CONTROL_CEK_BCH_2,											0x480023C0,__READ_WRITE	);
__IO_REG32(		 SCM_CONTROL_CEK_BCH_3,											0x480023C4,__READ_WRITE	);
__IO_REG32(		 SCM_CONTROL_CEK_BCH_4,											0x480023C8,__READ_WRITE	);
__IO_REG32(		 SCM_CONTROL_MSV_BCH_0,											0x480023CC,__READ_WRITE	);
__IO_REG32(		 SCM_CONTROL_MSV_BCH_1,											0x480023D0,__READ_WRITE	);
__IO_REG32(		 SCM_CONTROL_SWRV_0,												0x480023D4,__READ_WRITE	);
__IO_REG32(		 SCM_CONTROL_SWRV_1,												0x480023D8,__READ_WRITE	);
__IO_REG32(		 SCM_CONTROL_SWRV_2,												0x480023DC,__READ_WRITE	);
__IO_REG32(		 SCM_CONTROL_SWRV_3,												0x480023E0,__READ_WRITE	);
__IO_REG32(		 SCM_CONTROL_SWRV_4,												0x480023E4,__READ_WRITE	);
__IO_REG32_BIT(SCM_CONTROL_DEBOBS_0,											0x48002420,__READ_WRITE	,__scm_control_debobs_0_bits);
__IO_REG32_BIT(SCM_CONTROL_DEBOBS_1,											0x48002424,__READ_WRITE	,__scm_control_debobs_1_bits);
__IO_REG32_BIT(SCM_CONTROL_DEBOBS_2,											0x48002428,__READ_WRITE	,__scm_control_debobs_2_bits);
__IO_REG32_BIT(SCM_CONTROL_DEBOBS_3,											0x4800242C,__READ_WRITE	,__scm_control_debobs_3_bits);
__IO_REG32_BIT(SCM_CONTROL_DEBOBS_4,											0x48002430,__READ_WRITE	,__scm_control_debobs_4_bits);
__IO_REG32_BIT(SCM_CONTROL_DEBOBS_5,											0x48002434,__READ_WRITE	,__scm_control_debobs_5_bits);
__IO_REG32_BIT(SCM_CONTROL_DEBOBS_6,											0x48002438,__READ_WRITE	,__scm_control_debobs_6_bits);
__IO_REG32_BIT(SCM_CONTROL_DEBOBS_7,											0x4800243C,__READ_WRITE	,__scm_control_debobs_7_bits);
__IO_REG32_BIT(SCM_CONTROL_DEBOBS_8,											0x48002440,__READ_WRITE	,__scm_control_debobs_8_bits);
__IO_REG32_BIT(SCM_CONTROL_WKUP_CTRL,											0x48002A5C,__READ_WRITE	,__scm_control_wkup_ctrl_bits);
__IO_REG32_BIT(SCM_CONTROL_DSS_DPLL_SPREADING,						0x48002450,__READ_WRITE	,__scm_control_dss_dpll_spreading_bits);
__IO_REG32_BIT(SCM_CONTROL_CORE_DPLL_SPREADING,						0x48002454,__READ_WRITE	,__scm_control_core_dpll_spreading_bits);
__IO_REG32_BIT(SCM_CONTROL_PER_DPLL_SPREADING,						0x48002458,__READ_WRITE	,__scm_control_per_dpll_spreading_bits);
__IO_REG32_BIT(SCM_CONTROL_USBHOST_DPLL_SPREADING,				0x4800245C,__READ_WRITE	,__scm_control_usbhost_dpll_spreading_bits);
__IO_REG32_BIT(SCM_CONTROL_DPF_OCM_RAM_FW_ADDR_MATCH,			0x48002498,__READ_WRITE	,__scm_control_dpf_ocm_ram_fw_addr_match_bits);
__IO_REG32_BIT(SCM_CONTROL_DPF_OCM_RAM_FW_REQINFO,				0x4800249C,__READ_WRITE	,__scm_control_dpf_ocm_ram_fw_reqinfo_bits);
__IO_REG32_BIT(SCM_CONTROL_DPF_OCM_RAM_FW_WR,							0x480024A0,__READ_WRITE	,__scm_control_dpf_ocm_ram_fw_wr_bits);
__IO_REG32_BIT(SCM_CONTROL_DPF_REGION4_GPMC_FW_ADDR_MATCH,0x480024A4,__READ_WRITE	,__scm_control_dpf_region4_gpmc_fw_addr_match_bits);
__IO_REG32(		 SCM_CONTROL_DPF_REGION4_GPMC_FW_REQINFO,		0x480024A8,__READ_WRITE	);
__IO_REG32_BIT(SCM_CONTROL_DPF_REGION4_GPMC_FW_WR,				0x480024AC,__READ_WRITE	,__scm_control_dpf_region4_gpmc_fw_wr_bits);
__IO_REG32_BIT(SCM_CONTROL_APE_FW_DEFAULT_SECURE_LOCK,		0x480024BC,__READ_WRITE	,__scm_control_ape_fw_default_secure_lock_bits);
__IO_REG32_BIT(SCM_CONTROL_OCMROM_SECURE_DEBUG,						0x480024C0,__READ_WRITE	,__scm_control_ocmrom_secure_debug_bits);
__IO_REG32_BIT(SCM_CONTROL_EXT_SEC_CONTROL,								0x480024D4,__READ_WRITE	,__scm_control_ext_sec_control_bits);
__IO_REG32_BIT(SCM_CONTROL_DEVCONF2,											0x48002580,__READ_WRITE	,__scm_control_devconf2_bits);
__IO_REG32_BIT(SCM_CONTROL_DEVCONF3,											0x48002584,__READ_WRITE	,__scm_control_devconf3_bits);
__IO_REG32_BIT(SCM_CONTROL_CBA_PRIORITY,									0x48002590,__READ_WRITE	,__scm_control_cba_priority_bits);
__IO_REG32_BIT(SCM_CONTROL_LVL_INTR_CLEAR,								0x48002594,__READ_WRITE	,__scm_control_lvl_intr_clear_bits);
__IO_REG32_BIT(SCM_CONTROL_IP_SW_RESET,										0x48002598,__READ_WRITE	,__scm_control_ip_sw_reset_bits);
__IO_REG32_BIT(SCM_CONTROL_IPSS_CLK_CTRL,									0x4800259C,__READ_WRITE	,__scm_control_ipss_clk_ctrl_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_I2C4_SCL,							0x48002A00,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SYS_32K,								0x48002A04,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SYS_NRESWARM,					0x48002A08,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SYS_BOOT1,							0x48002A0C,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SYS_BOOT3,							0x48002A10,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SYS_BOOT5,							0x48002A14,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SYS_OFF_MODE,					0x48002A18,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_JTAG_NTRST,						0x48002A1C,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_JTAG_TMS_TMSC,					0x48002A20,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_JTAG_EMU0,							0x48002A24,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_JTAG_RTCK,							0x48002A4C,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_JTAG_TDO,							0x48002A50,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_SEC_TAP,												0x48002A60,__READ_WRITE	,__scm_control_sec_tap_bits);
__IO_REG32_BIT(SCM_CONTROL_SEC_EMU,												0x48002A64,__READ_WRITE	,__scm_control_sec_emu_bits);
__IO_REG32_BIT(SCM_CONTROL_WKUP_DEBOBS_0,									0x48002A68,__READ_WRITE	,__scm_control_wkup_debobs_0_bits);
__IO_REG32_BIT(SCM_CONTROL_WKUP_DEBOBS_1,									0x48002A6C,__READ_WRITE	,__scm_control_wkup_debobs_1_bits);
__IO_REG32_BIT(SCM_CONTROL_WKUP_DEBOBS_2,									0x48002A70,__READ_WRITE	,__scm_control_wkup_debobs_2_bits);
__IO_REG32_BIT(SCM_CONTROL_WKUP_DEBOBS_3,									0x48002A74,__READ_WRITE	,__scm_control_wkup_debobs_3_bits);
__IO_REG32_BIT(SCM_CONTROL_WKUP_DEBOBS_4,									0x48002A78,__READ_WRITE	,__scm_control_wkup_debobs_4_bits);
__IO_REG32_BIT(SCM_CONTROL_SEC_DAP,												0x48002A7C,__READ_WRITE	,__scm_control_sec_dap_bits);

/***************************************************************************
 **
 ** DMA4 Common
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA4_REVISION,         0x48056000,__READ       ,__dma4_revision_bits);
__IO_REG32_BIT(DMA4_IRQSTATUS_L0,     0x48056008,__READ_WRITE ,__dma4_irqstatus_bits);
__IO_REG32_BIT(DMA4_IRQSTATUS_L1,     0x4805600C,__READ_WRITE ,__dma4_irqstatus_bits);
__IO_REG32_BIT(DMA4_IRQSTATUS_L2,     0x48056010,__READ_WRITE ,__dma4_irqstatus_bits);
__IO_REG32_BIT(DMA4_IRQSTATUS_L3,     0x48056014,__READ_WRITE ,__dma4_irqstatus_bits);
__IO_REG32_BIT(DMA4_IRQENABLE_L0,     0x48056018,__READ_WRITE ,__dma4_irqstatus_bits);
__IO_REG32_BIT(DMA4_IRQENABLE_L1,     0x4805601C,__READ_WRITE ,__dma4_irqstatus_bits);
__IO_REG32_BIT(DMA4_IRQENABLE_L2,     0x48056020,__READ_WRITE ,__dma4_irqstatus_bits);
__IO_REG32_BIT(DMA4_IRQENABLE_L3,     0x48056024,__READ_WRITE ,__dma4_irqstatus_bits);
__IO_REG32_BIT(DMA4_SYSSTATUS,     		0x48056028,__READ				,__dma4_sysstatus_bits);
__IO_REG32_BIT(DMA4_OCP_SYSCONFIG,    0x4805602C,__READ_WRITE ,__dma4_ocp_sysconfig_bits);
__IO_REG32_BIT(DMA4_CAPS_0,     			0x48056064,__READ				,__dma4_caps_0_bits);
__IO_REG32_BIT(DMA4_CAPS_2,     			0x4805606C,__READ				,__dma4_caps_2_bits);
__IO_REG32_BIT(DMA4_CAPS_3,     			0x48056070,__READ				,__dma4_caps_3_bits);
__IO_REG32_BIT(DMA4_CAPS_4,     			0x48056074,__READ				,__dma4_caps_4_bits);
__IO_REG32_BIT(DMA4_GCR,    					0x48056078,__READ_WRITE ,__dma4_gcr_bits);

/***************************************************************************
 **
 ** DMA4 CH0
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA4_CCR0,     				0x48056080,__READ_WRITE ,__dma4_ccr_bits);
__IO_REG32_BIT(DMA4_CLNK_CTRL0,     	0x48056084,__READ_WRITE ,__dma4_clnk_ctrl_bits);
__IO_REG32_BIT(DMA4_CICR0,     				0x48056088,__READ_WRITE ,__dma4_cicr_bits);
__IO_REG32_BIT(DMA4_CSR0,     				0x4805608C,__READ_WRITE ,__dma4_csr_bits);
__IO_REG32_BIT(DMA4_CSDP0,     				0x48056090,__READ_WRITE ,__dma4_csdp_bits);
__IO_REG32_BIT(DMA4_CEN0,     				0x48056094,__READ_WRITE ,__dma4_cen_bits);
__IO_REG32_BIT(DMA4_CFN0,     				0x48056098,__READ_WRITE ,__dma4_cfn_bits);
__IO_REG32(		 DMA4_CSSA0,     				0x4805609C,__READ_WRITE );
__IO_REG32(		 DMA4_CDSA0,     				0x480560A0,__READ_WRITE );
__IO_REG32_BIT(DMA4_CSEI0,     				0x480560A4,__READ_WRITE ,__dma4_csei_bits);
__IO_REG32(		 DMA4_CSFI0,     				0x480560A8,__READ_WRITE );
__IO_REG32_BIT(DMA4_CDEI0,     				0x480560AC,__READ_WRITE ,__dma4_cdei_bits);
__IO_REG32(		 DMA4_CDFI0,     				0x480560B0,__READ_WRITE );
__IO_REG32(		 DMA4_CSAC0,     				0x480560B4,__READ				);
__IO_REG32(		 DMA4_CDAC0,     				0x480560B8,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCEN0,     				0x480560BC,__READ				,__dma4_ccen_bits);
__IO_REG32_BIT(DMA4_CCFN0,     				0x480560C0,__READ				,__dma4_ccfn_bits);
__IO_REG32_BIT(DMA4_COLOR0,     			0x480560C4,__READ_WRITE ,__dma4_color_bits);

/***************************************************************************
 **
 ** DMA4 CH1
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA4_CCR1,     				0x480560E0,__READ_WRITE ,__dma4_ccr_bits);
__IO_REG32_BIT(DMA4_CLNK_CTRL1,     	0x480560E4,__READ_WRITE ,__dma4_clnk_ctrl_bits);
__IO_REG32_BIT(DMA4_CICR1,     				0x480560E8,__READ_WRITE ,__dma4_cicr_bits);
__IO_REG32_BIT(DMA4_CSR1,     				0x480560EC,__READ_WRITE ,__dma4_csr_bits);
__IO_REG32_BIT(DMA4_CSDP1,     				0x480560F0,__READ_WRITE ,__dma4_csdp_bits);
__IO_REG32_BIT(DMA4_CEN1,     				0x480560F4,__READ_WRITE ,__dma4_cen_bits);
__IO_REG32_BIT(DMA4_CFN1,     				0x480560F8,__READ_WRITE ,__dma4_cfn_bits);
__IO_REG32(		 DMA4_CSSA1,     				0x480560FC,__READ_WRITE );
__IO_REG32(		 DMA4_CDSA1,     				0x48056100,__READ_WRITE );
__IO_REG32_BIT(DMA4_CSEI1,     				0x48056104,__READ_WRITE ,__dma4_csei_bits);
__IO_REG32(		 DMA4_CSFI1,     				0x48056108,__READ_WRITE );
__IO_REG32_BIT(DMA4_CDEI1,     				0x4805610C,__READ_WRITE ,__dma4_cdei_bits);
__IO_REG32(		 DMA4_CDFI1,     				0x48056110,__READ_WRITE );
__IO_REG32(		 DMA4_CSAC1,     				0x48056114,__READ				);
__IO_REG32(		 DMA4_CDAC1,     				0x48056118,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCEN1,     				0x4805611C,__READ				,__dma4_ccen_bits);
__IO_REG32_BIT(DMA4_CCFN1,     				0x48056120,__READ				,__dma4_ccfn_bits);
__IO_REG32_BIT(DMA4_COLOR1,     			0x48056124,__READ_WRITE ,__dma4_color_bits);

/***************************************************************************
 **
 ** DMA4 CH2
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA4_CCR2,     				0x48056140,__READ_WRITE ,__dma4_ccr_bits);
__IO_REG32_BIT(DMA4_CLNK_CTRL2,     	0x48056144,__READ_WRITE ,__dma4_clnk_ctrl_bits);
__IO_REG32_BIT(DMA4_CICR2,     				0x48056148,__READ_WRITE ,__dma4_cicr_bits);
__IO_REG32_BIT(DMA4_CSR2,     				0x4805614C,__READ_WRITE ,__dma4_csr_bits);
__IO_REG32_BIT(DMA4_CSDP2,     				0x48056150,__READ_WRITE ,__dma4_csdp_bits);
__IO_REG32_BIT(DMA4_CEN2,     				0x48056154,__READ_WRITE ,__dma4_cen_bits);
__IO_REG32_BIT(DMA4_CFN2,     				0x48056158,__READ_WRITE ,__dma4_cfn_bits);
__IO_REG32(		 DMA4_CSSA2,     				0x4805615C,__READ_WRITE );
__IO_REG32(		 DMA4_CDSA2,     				0x48056160,__READ_WRITE );
__IO_REG32_BIT(DMA4_CSEI2,     				0x48056164,__READ_WRITE ,__dma4_csei_bits);
__IO_REG32(		 DMA4_CSFI2,     				0x48056168,__READ_WRITE );
__IO_REG32_BIT(DMA4_CDEI2,     				0x4805616C,__READ_WRITE ,__dma4_cdei_bits);
__IO_REG32(		 DMA4_CDFI2,     				0x48056170,__READ_WRITE );
__IO_REG32(		 DMA4_CSAC2,     				0x48056174,__READ				);
__IO_REG32(		 DMA4_CDAC2,     				0x48056178,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCEN2,     				0x4805617C,__READ				,__dma4_ccen_bits);
__IO_REG32_BIT(DMA4_CCFN2,     				0x48056180,__READ				,__dma4_ccfn_bits);
__IO_REG32_BIT(DMA4_COLOR2,     			0x48056184,__READ_WRITE ,__dma4_color_bits);

/***************************************************************************
 **
 ** DMA4 CH3
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA4_CCR3,     				0x480561A0,__READ_WRITE ,__dma4_ccr_bits);
__IO_REG32_BIT(DMA4_CLNK_CTRL3,     	0x480561A4,__READ_WRITE ,__dma4_clnk_ctrl_bits);
__IO_REG32_BIT(DMA4_CICR3,     				0x480561A8,__READ_WRITE ,__dma4_cicr_bits);
__IO_REG32_BIT(DMA4_CSR3,     				0x480561AC,__READ_WRITE ,__dma4_csr_bits);
__IO_REG32_BIT(DMA4_CSDP3,     				0x480561B0,__READ_WRITE ,__dma4_csdp_bits);
__IO_REG32_BIT(DMA4_CEN3,     				0x480561B4,__READ_WRITE ,__dma4_cen_bits);
__IO_REG32_BIT(DMA4_CFN3,     				0x480561B8,__READ_WRITE ,__dma4_cfn_bits);
__IO_REG32(		 DMA4_CSSA3,     				0x480561BC,__READ_WRITE );
__IO_REG32(		 DMA4_CDSA3,     				0x480561C0,__READ_WRITE );
__IO_REG32_BIT(DMA4_CSEI3,     				0x480561C4,__READ_WRITE ,__dma4_csei_bits);
__IO_REG32(		 DMA4_CSFI3,     				0x480561C8,__READ_WRITE );
__IO_REG32_BIT(DMA4_CDEI3,     				0x480561CC,__READ_WRITE ,__dma4_cdei_bits);
__IO_REG32(		 DMA4_CDFI3,     				0x480561D0,__READ_WRITE );
__IO_REG32(		 DMA4_CSAC3,     				0x480561D4,__READ				);
__IO_REG32(		 DMA4_CDAC3,     				0x480561D8,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCEN3,     				0x480561DC,__READ				,__dma4_ccen_bits);
__IO_REG32_BIT(DMA4_CCFN3,     				0x480561E0,__READ				,__dma4_ccfn_bits);
__IO_REG32_BIT(DMA4_COLOR3,     			0x480561E4,__READ_WRITE ,__dma4_color_bits);

/***************************************************************************
 **
 ** DMA4 CH4
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA4_CCR4,     				0x48056200,__READ_WRITE ,__dma4_ccr_bits);
__IO_REG32_BIT(DMA4_CLNK_CTRL4,     	0x48056204,__READ_WRITE ,__dma4_clnk_ctrl_bits);
__IO_REG32_BIT(DMA4_CICR4,     				0x48056208,__READ_WRITE ,__dma4_cicr_bits);
__IO_REG32_BIT(DMA4_CSR4,     				0x4805620C,__READ_WRITE ,__dma4_csr_bits);
__IO_REG32_BIT(DMA4_CSDP4,     				0x48056210,__READ_WRITE ,__dma4_csdp_bits);
__IO_REG32_BIT(DMA4_CEN4,     				0x48056214,__READ_WRITE ,__dma4_cen_bits);
__IO_REG32_BIT(DMA4_CFN4,     				0x48056218,__READ_WRITE ,__dma4_cfn_bits);
__IO_REG32(		 DMA4_CSSA4,     				0x4805621C,__READ_WRITE );
__IO_REG32(		 DMA4_CDSA4,     				0x48056220,__READ_WRITE );
__IO_REG32_BIT(DMA4_CSEI4,     				0x48056224,__READ_WRITE ,__dma4_csei_bits);
__IO_REG32(		 DMA4_CSFI4,     				0x48056228,__READ_WRITE );
__IO_REG32_BIT(DMA4_CDEI4,     				0x4805622C,__READ_WRITE ,__dma4_cdei_bits);
__IO_REG32(		 DMA4_CDFI4,     				0x48056230,__READ_WRITE );
__IO_REG32(		 DMA4_CSAC4,     				0x48056234,__READ				);
__IO_REG32(		 DMA4_CDAC4,     				0x48056238,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCEN4,     				0x4805623C,__READ				,__dma4_ccen_bits);
__IO_REG32_BIT(DMA4_CCFN4,     				0x48056240,__READ				,__dma4_ccfn_bits);
__IO_REG32_BIT(DMA4_COLOR4,     			0x48056244,__READ_WRITE ,__dma4_color_bits);

/***************************************************************************
 **
 ** DMA4 CH5
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA4_CCR5,     				0x48056260,__READ_WRITE ,__dma4_ccr_bits);
__IO_REG32_BIT(DMA4_CLNK_CTRL5,     	0x48056264,__READ_WRITE ,__dma4_clnk_ctrl_bits);
__IO_REG32_BIT(DMA4_CICR5,     				0x48056268,__READ_WRITE ,__dma4_cicr_bits);
__IO_REG32_BIT(DMA4_CSR5,     				0x4805626C,__READ_WRITE ,__dma4_csr_bits);
__IO_REG32_BIT(DMA4_CSDP5,     				0x48056270,__READ_WRITE ,__dma4_csdp_bits);
__IO_REG32_BIT(DMA4_CEN5,     				0x48056274,__READ_WRITE ,__dma4_cen_bits);
__IO_REG32_BIT(DMA4_CFN5,     				0x48056278,__READ_WRITE ,__dma4_cfn_bits);
__IO_REG32(		 DMA4_CSSA5,     				0x4805627C,__READ_WRITE );
__IO_REG32(		 DMA4_CDSA5,     				0x48056280,__READ_WRITE );
__IO_REG32_BIT(DMA4_CSEI5,     				0x48056284,__READ_WRITE ,__dma4_csei_bits);
__IO_REG32(		 DMA4_CSFI5,     				0x48056288,__READ_WRITE );
__IO_REG32_BIT(DMA4_CDEI5,     				0x4805628C,__READ_WRITE ,__dma4_cdei_bits);
__IO_REG32(		 DMA4_CDFI5,     				0x48056290,__READ_WRITE );
__IO_REG32(		 DMA4_CSAC5,     				0x48056294,__READ				);
__IO_REG32(		 DMA4_CDAC5,     				0x48056298,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCEN5,     				0x4805629C,__READ				,__dma4_ccen_bits);
__IO_REG32_BIT(DMA4_CCFN5,     				0x480562A0,__READ				,__dma4_ccfn_bits);
__IO_REG32_BIT(DMA4_COLOR5,     			0x480562A4,__READ_WRITE ,__dma4_color_bits);

/***************************************************************************
 **
 ** DMA4 CH6
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA4_CCR6,     				0x480562C0,__READ_WRITE ,__dma4_ccr_bits);
__IO_REG32_BIT(DMA4_CLNK_CTRL6,     	0x480562C4,__READ_WRITE ,__dma4_clnk_ctrl_bits);
__IO_REG32_BIT(DMA4_CICR6,     				0x480562C8,__READ_WRITE ,__dma4_cicr_bits);
__IO_REG32_BIT(DMA4_CSR6,     				0x480562CC,__READ_WRITE ,__dma4_csr_bits);
__IO_REG32_BIT(DMA4_CSDP6,     				0x480562D0,__READ_WRITE ,__dma4_csdp_bits);
__IO_REG32_BIT(DMA4_CEN6,     				0x480562D4,__READ_WRITE ,__dma4_cen_bits);
__IO_REG32_BIT(DMA4_CFN6,     				0x480562D8,__READ_WRITE ,__dma4_cfn_bits);
__IO_REG32(		 DMA4_CSSA6,     				0x480562DC,__READ_WRITE );
__IO_REG32(		 DMA4_CDSA6,     				0x480562E0,__READ_WRITE );
__IO_REG32_BIT(DMA4_CSEI6,     				0x480562E4,__READ_WRITE ,__dma4_csei_bits);
__IO_REG32(		 DMA4_CSFI6,     				0x480562E8,__READ_WRITE );
__IO_REG32_BIT(DMA4_CDEI6,     				0x480562EC,__READ_WRITE ,__dma4_cdei_bits);
__IO_REG32(		 DMA4_CDFI6,     				0x480562F0,__READ_WRITE );
__IO_REG32(		 DMA4_CSAC6,     				0x480562F4,__READ				);
__IO_REG32(		 DMA4_CDAC6,     				0x480562F8,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCEN6,     				0x480562FC,__READ				,__dma4_ccen_bits);
__IO_REG32_BIT(DMA4_CCFN6,     				0x48056300,__READ				,__dma4_ccfn_bits);
__IO_REG32_BIT(DMA4_COLOR6,     			0x48056304,__READ_WRITE ,__dma4_color_bits);

/***************************************************************************
 **
 ** DMA4 CH7
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA4_CCR7,     				0x48056320,__READ_WRITE ,__dma4_ccr_bits);
__IO_REG32_BIT(DMA4_CLNK_CTRL7,     	0x48056324,__READ_WRITE ,__dma4_clnk_ctrl_bits);
__IO_REG32_BIT(DMA4_CICR7,     				0x48056328,__READ_WRITE ,__dma4_cicr_bits);
__IO_REG32_BIT(DMA4_CSR7,     				0x4805632C,__READ_WRITE ,__dma4_csr_bits);
__IO_REG32_BIT(DMA4_CSDP7,     				0x48056330,__READ_WRITE ,__dma4_csdp_bits);
__IO_REG32_BIT(DMA4_CEN7,     				0x48056334,__READ_WRITE ,__dma4_cen_bits);
__IO_REG32_BIT(DMA4_CFN7,     				0x48056338,__READ_WRITE ,__dma4_cfn_bits);
__IO_REG32(		 DMA4_CSSA7,     				0x4805633C,__READ_WRITE );
__IO_REG32(		 DMA4_CDSA7,     				0x48056340,__READ_WRITE );
__IO_REG32_BIT(DMA4_CSEI7,     				0x48056344,__READ_WRITE ,__dma4_csei_bits);
__IO_REG32(		 DMA4_CSFI7,     				0x48056348,__READ_WRITE );
__IO_REG32_BIT(DMA4_CDEI7,     				0x4805634C,__READ_WRITE ,__dma4_cdei_bits);
__IO_REG32(		 DMA4_CDFI7,     				0x48056350,__READ_WRITE );
__IO_REG32(		 DMA4_CSAC7,     				0x48056354,__READ				);
__IO_REG32(		 DMA4_CDAC7,     				0x48056358,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCEN7,     				0x4805635C,__READ				,__dma4_ccen_bits);
__IO_REG32_BIT(DMA4_CCFN7,     				0x48056360,__READ				,__dma4_ccfn_bits);
__IO_REG32_BIT(DMA4_COLOR7,     			0x48056364,__READ_WRITE ,__dma4_color_bits);

/***************************************************************************
 **
 ** DMA4 CH8
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA4_CCR8,     				0x48056380,__READ_WRITE ,__dma4_ccr_bits);
__IO_REG32_BIT(DMA4_CLNK_CTRL8,     	0x48056384,__READ_WRITE ,__dma4_clnk_ctrl_bits);
__IO_REG32_BIT(DMA4_CICR8,     				0x48056388,__READ_WRITE ,__dma4_cicr_bits);
__IO_REG32_BIT(DMA4_CSR8,     				0x4805638C,__READ_WRITE ,__dma4_csr_bits);
__IO_REG32_BIT(DMA4_CSDP8,     				0x48056390,__READ_WRITE ,__dma4_csdp_bits);
__IO_REG32_BIT(DMA4_CEN8,     				0x48056394,__READ_WRITE ,__dma4_cen_bits);
__IO_REG32_BIT(DMA4_CFN8,     				0x48056398,__READ_WRITE ,__dma4_cfn_bits);
__IO_REG32(		 DMA4_CSSA8,     				0x4805639C,__READ_WRITE );
__IO_REG32(		 DMA4_CDSA8,     				0x480563A0,__READ_WRITE );
__IO_REG32_BIT(DMA4_CSEI8,     				0x480563A4,__READ_WRITE ,__dma4_csei_bits);
__IO_REG32(		 DMA4_CSFI8,     				0x480563A8,__READ_WRITE );
__IO_REG32_BIT(DMA4_CDEI8,     				0x480563AC,__READ_WRITE ,__dma4_cdei_bits);
__IO_REG32(		 DMA4_CDFI8,     				0x480563B0,__READ_WRITE );
__IO_REG32(		 DMA4_CSAC8,     				0x480563B4,__READ				);
__IO_REG32(		 DMA4_CDAC8,     				0x480563B8,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCEN8,     				0x480563BC,__READ				,__dma4_ccen_bits);
__IO_REG32_BIT(DMA4_CCFN8,     				0x480563C0,__READ				,__dma4_ccfn_bits);
__IO_REG32_BIT(DMA4_COLOR8,     			0x480563C4,__READ_WRITE ,__dma4_color_bits);

/***************************************************************************
 **
 ** DMA4 CH9
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA4_CCR9,     				0x480563E0,__READ_WRITE ,__dma4_ccr_bits);
__IO_REG32_BIT(DMA4_CLNK_CTRL9,     	0x480563E4,__READ_WRITE ,__dma4_clnk_ctrl_bits);
__IO_REG32_BIT(DMA4_CICR9,     				0x480563E8,__READ_WRITE ,__dma4_cicr_bits);
__IO_REG32_BIT(DMA4_CSR9,     				0x480563EC,__READ_WRITE ,__dma4_csr_bits);
__IO_REG32_BIT(DMA4_CSDP9,     				0x480563F0,__READ_WRITE ,__dma4_csdp_bits);
__IO_REG32_BIT(DMA4_CEN9,     				0x480563F4,__READ_WRITE ,__dma4_cen_bits);
__IO_REG32_BIT(DMA4_CFN9,     				0x480563F8,__READ_WRITE ,__dma4_cfn_bits);
__IO_REG32(		 DMA4_CSSA9,     				0x480563FC,__READ_WRITE );
__IO_REG32(		 DMA4_CDSA9,     				0x48056400,__READ_WRITE );
__IO_REG32_BIT(DMA4_CSEI9,     				0x48056404,__READ_WRITE ,__dma4_csei_bits);
__IO_REG32(		 DMA4_CSFI9,     				0x48056408,__READ_WRITE );
__IO_REG32_BIT(DMA4_CDEI9,     				0x4805640C,__READ_WRITE ,__dma4_cdei_bits);
__IO_REG32(		 DMA4_CDFI9,     				0x48056410,__READ_WRITE );
__IO_REG32(		 DMA4_CSAC9,     				0x48056414,__READ				);
__IO_REG32(		 DMA4_CDAC9,     				0x48056418,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCEN9,     				0x4805641C,__READ				,__dma4_ccen_bits);
__IO_REG32_BIT(DMA4_CCFN9,     				0x48056420,__READ				,__dma4_ccfn_bits);
__IO_REG32_BIT(DMA4_COLOR9,     			0x48056424,__READ_WRITE ,__dma4_color_bits);

/***************************************************************************
 **
 ** DMA4 CH10
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA4_CCR10,     				0x48056440,__READ_WRITE ,__dma4_ccr_bits);
__IO_REG32_BIT(DMA4_CLNK_CTRL10,     	0x48056444,__READ_WRITE ,__dma4_clnk_ctrl_bits);
__IO_REG32_BIT(DMA4_CICR10,    				0x48056448,__READ_WRITE ,__dma4_cicr_bits);
__IO_REG32_BIT(DMA4_CSR10,     				0x4805644C,__READ_WRITE ,__dma4_csr_bits);
__IO_REG32_BIT(DMA4_CSDP10,    				0x48056450,__READ_WRITE ,__dma4_csdp_bits);
__IO_REG32_BIT(DMA4_CEN10,     				0x48056454,__READ_WRITE ,__dma4_cen_bits);
__IO_REG32_BIT(DMA4_CFN10,     				0x48056458,__READ_WRITE ,__dma4_cfn_bits);
__IO_REG32(		 DMA4_CSSA10,    				0x4805645C,__READ_WRITE );
__IO_REG32(		 DMA4_CDSA10,    				0x48056460,__READ_WRITE );
__IO_REG32_BIT(DMA4_CSEI10,    				0x48056464,__READ_WRITE ,__dma4_csei_bits);
__IO_REG32(		 DMA4_CSFI10,    				0x48056468,__READ_WRITE );
__IO_REG32_BIT(DMA4_CDEI10,    				0x4805646C,__READ_WRITE ,__dma4_cdei_bits);
__IO_REG32(		 DMA4_CDFI10,    				0x48056470,__READ_WRITE );
__IO_REG32(		 DMA4_CSAC10,    				0x48056474,__READ				);
__IO_REG32(		 DMA4_CDAC10,    				0x48056478,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCEN10,    				0x4805647C,__READ				,__dma4_ccen_bits);
__IO_REG32_BIT(DMA4_CCFN10,    				0x48056480,__READ				,__dma4_ccfn_bits);
__IO_REG32_BIT(DMA4_COLOR10,     			0x48056484,__READ_WRITE ,__dma4_color_bits);

/***************************************************************************
 **
 ** DMA4 CH11
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA4_CCR11,     				0x480564A0,__READ_WRITE ,__dma4_ccr_bits);
__IO_REG32_BIT(DMA4_CLNK_CTRL11,     	0x480564A4,__READ_WRITE ,__dma4_clnk_ctrl_bits);
__IO_REG32_BIT(DMA4_CICR11,    				0x480564A8,__READ_WRITE ,__dma4_cicr_bits);
__IO_REG32_BIT(DMA4_CSR11,     				0x480564AC,__READ_WRITE ,__dma4_csr_bits);
__IO_REG32_BIT(DMA4_CSDP11,    				0x480564B0,__READ_WRITE ,__dma4_csdp_bits);
__IO_REG32_BIT(DMA4_CEN11,     				0x480564B4,__READ_WRITE ,__dma4_cen_bits);
__IO_REG32_BIT(DMA4_CFN11,     				0x480564B8,__READ_WRITE ,__dma4_cfn_bits);
__IO_REG32(		 DMA4_CSSA11,    				0x480564BC,__READ_WRITE );
__IO_REG32(		 DMA4_CDSA11,    				0x480564C0,__READ_WRITE );
__IO_REG32_BIT(DMA4_CSEI11,    				0x480564C4,__READ_WRITE ,__dma4_csei_bits);
__IO_REG32(		 DMA4_CSFI11,    				0x480564C8,__READ_WRITE );
__IO_REG32_BIT(DMA4_CDEI11,    				0x480564CC,__READ_WRITE ,__dma4_cdei_bits);
__IO_REG32(		 DMA4_CDFI11,    				0x480564D0,__READ_WRITE );
__IO_REG32(		 DMA4_CSAC11,    				0x480564D4,__READ				);
__IO_REG32(		 DMA4_CDAC11,    				0x480564D8,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCEN11,    				0x480564DC,__READ				,__dma4_ccen_bits);
__IO_REG32_BIT(DMA4_CCFN11,    				0x480564E0,__READ				,__dma4_ccfn_bits);
__IO_REG32_BIT(DMA4_COLOR11,     			0x480564E4,__READ_WRITE ,__dma4_color_bits);

/***************************************************************************
 **
 ** DMA4 CH12
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA4_CCR12,     				0x48056500,__READ_WRITE ,__dma4_ccr_bits);
__IO_REG32_BIT(DMA4_CLNK_CTRL12,     	0x48056504,__READ_WRITE ,__dma4_clnk_ctrl_bits);
__IO_REG32_BIT(DMA4_CICR12,    				0x48056508,__READ_WRITE ,__dma4_cicr_bits);
__IO_REG32_BIT(DMA4_CSR12,     				0x4805650C,__READ_WRITE ,__dma4_csr_bits);
__IO_REG32_BIT(DMA4_CSDP12,    				0x48056510,__READ_WRITE ,__dma4_csdp_bits);
__IO_REG32_BIT(DMA4_CEN12,     				0x48056514,__READ_WRITE ,__dma4_cen_bits);
__IO_REG32_BIT(DMA4_CFN12,     				0x48056518,__READ_WRITE ,__dma4_cfn_bits);
__IO_REG32(		 DMA4_CSSA12,    				0x4805651C,__READ_WRITE );
__IO_REG32(		 DMA4_CDSA12,    				0x48056520,__READ_WRITE );
__IO_REG32_BIT(DMA4_CSEI12,    				0x48056524,__READ_WRITE ,__dma4_csei_bits);
__IO_REG32(		 DMA4_CSFI12,    				0x48056528,__READ_WRITE );
__IO_REG32_BIT(DMA4_CDEI12,    				0x4805652C,__READ_WRITE ,__dma4_cdei_bits);
__IO_REG32(		 DMA4_CDFI12,    				0x48056530,__READ_WRITE );
__IO_REG32(		 DMA4_CSAC12,    				0x48056534,__READ				);
__IO_REG32(		 DMA4_CDAC12,    				0x48056538,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCEN12,    				0x4805653C,__READ				,__dma4_ccen_bits);
__IO_REG32_BIT(DMA4_CCFN12,    				0x48056540,__READ				,__dma4_ccfn_bits);
__IO_REG32_BIT(DMA4_COLOR12,     			0x48056544,__READ_WRITE ,__dma4_color_bits);

/***************************************************************************
 **
 ** DMA4 CH13
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA4_CCR13,     				0x48056560,__READ_WRITE ,__dma4_ccr_bits);
__IO_REG32_BIT(DMA4_CLNK_CTRL13,     	0x48056564,__READ_WRITE ,__dma4_clnk_ctrl_bits);
__IO_REG32_BIT(DMA4_CICR13,     			0x48056568,__READ_WRITE ,__dma4_cicr_bits);
__IO_REG32_BIT(DMA4_CSR13,     				0x4805656C,__READ_WRITE ,__dma4_csr_bits);
__IO_REG32_BIT(DMA4_CSDP13,     			0x48056570,__READ_WRITE ,__dma4_csdp_bits);
__IO_REG32_BIT(DMA4_CEN13,     				0x48056574,__READ_WRITE ,__dma4_cen_bits);
__IO_REG32_BIT(DMA4_CFN13,     				0x48056578,__READ_WRITE ,__dma4_cfn_bits);
__IO_REG32(		 DMA4_CSSA13,     			0x4805657C,__READ_WRITE );
__IO_REG32(		 DMA4_CDSA13,     			0x48056580,__READ_WRITE );
__IO_REG32_BIT(DMA4_CSEI13,     			0x48056584,__READ_WRITE ,__dma4_csei_bits);
__IO_REG32(		 DMA4_CSFI13,     			0x48056588,__READ_WRITE );
__IO_REG32_BIT(DMA4_CDEI13,     			0x4805658C,__READ_WRITE ,__dma4_cdei_bits);
__IO_REG32(		 DMA4_CDFI13,     			0x48056590,__READ_WRITE );
__IO_REG32(		 DMA4_CSAC13,     			0x48056594,__READ				);
__IO_REG32(		 DMA4_CDAC13,     			0x48056598,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCEN13,     			0x4805659C,__READ				,__dma4_ccen_bits);
__IO_REG32_BIT(DMA4_CCFN13,     			0x480565A0,__READ				,__dma4_ccfn_bits);
__IO_REG32_BIT(DMA4_COLOR13,     			0x480565A4,__READ_WRITE ,__dma4_color_bits);

/***************************************************************************
 **
 ** DMA4 CH14
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA4_CCR14,     				0x480565C0,__READ_WRITE ,__dma4_ccr_bits);
__IO_REG32_BIT(DMA4_CLNK_CTRL14,     	0x480565C4,__READ_WRITE ,__dma4_clnk_ctrl_bits);
__IO_REG32_BIT(DMA4_CICR14,    				0x480565C8,__READ_WRITE ,__dma4_cicr_bits);
__IO_REG32_BIT(DMA4_CSR14,     				0x480565CC,__READ_WRITE ,__dma4_csr_bits);
__IO_REG32_BIT(DMA4_CSDP14,    				0x480565D0,__READ_WRITE ,__dma4_csdp_bits);
__IO_REG32_BIT(DMA4_CEN14,     				0x480565D4,__READ_WRITE ,__dma4_cen_bits);
__IO_REG32_BIT(DMA4_CFN14,     				0x480565D8,__READ_WRITE ,__dma4_cfn_bits);
__IO_REG32(		 DMA4_CSSA14,    				0x480565DC,__READ_WRITE );
__IO_REG32(		 DMA4_CDSA14,    				0x480565E0,__READ_WRITE );
__IO_REG32_BIT(DMA4_CSEI14,    				0x480565E4,__READ_WRITE ,__dma4_csei_bits);
__IO_REG32(		 DMA4_CSFI14,    				0x480565E8,__READ_WRITE );
__IO_REG32_BIT(DMA4_CDEI14,    				0x480565EC,__READ_WRITE ,__dma4_cdei_bits);
__IO_REG32(		 DMA4_CDFI14,    				0x480565F0,__READ_WRITE );
__IO_REG32(		 DMA4_CSAC14,    				0x480565F4,__READ				);
__IO_REG32(		 DMA4_CDAC14,    				0x480565F8,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCEN14,    				0x480565FC,__READ				,__dma4_ccen_bits);
__IO_REG32_BIT(DMA4_CCFN14,    				0x48056600,__READ				,__dma4_ccfn_bits);
__IO_REG32_BIT(DMA4_COLOR14,     			0x48056604,__READ_WRITE ,__dma4_color_bits);

/***************************************************************************
 **
 ** DMA4 CH15
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA4_CCR15,     				0x48056620,__READ_WRITE ,__dma4_ccr_bits);
__IO_REG32_BIT(DMA4_CLNK_CTRL15,     	0x48056624,__READ_WRITE ,__dma4_clnk_ctrl_bits);
__IO_REG32_BIT(DMA4_CICR15,    				0x48056628,__READ_WRITE ,__dma4_cicr_bits);
__IO_REG32_BIT(DMA4_CSR15,     				0x4805662C,__READ_WRITE ,__dma4_csr_bits);
__IO_REG32_BIT(DMA4_CSDP15,    				0x48056630,__READ_WRITE ,__dma4_csdp_bits);
__IO_REG32_BIT(DMA4_CEN15,     				0x48056634,__READ_WRITE ,__dma4_cen_bits);
__IO_REG32_BIT(DMA4_CFN15,     				0x48056638,__READ_WRITE ,__dma4_cfn_bits);
__IO_REG32(		 DMA4_CSSA15,    				0x4805663C,__READ_WRITE );
__IO_REG32(		 DMA4_CDSA15,    				0x48056640,__READ_WRITE );
__IO_REG32_BIT(DMA4_CSEI15,    				0x48056644,__READ_WRITE ,__dma4_csei_bits);
__IO_REG32(		 DMA4_CSFI15,    				0x48056648,__READ_WRITE );
__IO_REG32_BIT(DMA4_CDEI15,    				0x4805664C,__READ_WRITE ,__dma4_cdei_bits);
__IO_REG32(		 DMA4_CDFI15,    				0x48056650,__READ_WRITE );
__IO_REG32(		 DMA4_CSAC15,    				0x48056654,__READ				);
__IO_REG32(		 DMA4_CDAC15,    				0x48056658,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCEN15,    				0x4805665C,__READ				,__dma4_ccen_bits);
__IO_REG32_BIT(DMA4_CCFN15,    				0x48056660,__READ				,__dma4_ccfn_bits);
__IO_REG32_BIT(DMA4_COLOR15,     			0x48056664,__READ_WRITE ,__dma4_color_bits);

/***************************************************************************
 **
 ** DMA4 CH16
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA4_CCR16,     				0x48056680,__READ_WRITE ,__dma4_ccr_bits);
__IO_REG32_BIT(DMA4_CLNK_CTRL16,     	0x48056684,__READ_WRITE ,__dma4_clnk_ctrl_bits);
__IO_REG32_BIT(DMA4_CICR16,    				0x48056688,__READ_WRITE ,__dma4_cicr_bits);
__IO_REG32_BIT(DMA4_CSR16,     				0x4805668C,__READ_WRITE ,__dma4_csr_bits);
__IO_REG32_BIT(DMA4_CSDP16,    				0x48056690,__READ_WRITE ,__dma4_csdp_bits);
__IO_REG32_BIT(DMA4_CEN16,     				0x48056694,__READ_WRITE ,__dma4_cen_bits);
__IO_REG32_BIT(DMA4_CFN16,     				0x48056698,__READ_WRITE ,__dma4_cfn_bits);
__IO_REG32(		 DMA4_CSSA16,    				0x4805669C,__READ_WRITE );
__IO_REG32(		 DMA4_CDSA16,    				0x480566A0,__READ_WRITE );
__IO_REG32_BIT(DMA4_CSEI16,    				0x480566A4,__READ_WRITE ,__dma4_csei_bits);
__IO_REG32(		 DMA4_CSFI16,    				0x480566A8,__READ_WRITE );
__IO_REG32_BIT(DMA4_CDEI16,    				0x480566AC,__READ_WRITE ,__dma4_cdei_bits);
__IO_REG32(		 DMA4_CDFI16,    				0x480566B0,__READ_WRITE );
__IO_REG32(		 DMA4_CSAC16,    				0x480566B4,__READ				);
__IO_REG32(		 DMA4_CDAC16,    				0x480566B8,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCEN16,    				0x480566BC,__READ				,__dma4_ccen_bits);
__IO_REG32_BIT(DMA4_CCFN16,    				0x480566C0,__READ				,__dma4_ccfn_bits);
__IO_REG32_BIT(DMA4_COLOR16,     			0x480566C4,__READ_WRITE ,__dma4_color_bits);

/***************************************************************************
 **
 ** DMA4 CH17
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA4_CCR17,     				0x480566E0,__READ_WRITE ,__dma4_ccr_bits);
__IO_REG32_BIT(DMA4_CLNK_CTRL17,     	0x480566E4,__READ_WRITE ,__dma4_clnk_ctrl_bits);
__IO_REG32_BIT(DMA4_CICR17,    				0x480566E8,__READ_WRITE ,__dma4_cicr_bits);
__IO_REG32_BIT(DMA4_CSR17,     				0x480566EC,__READ_WRITE ,__dma4_csr_bits);
__IO_REG32_BIT(DMA4_CSDP17,    				0x480566F0,__READ_WRITE ,__dma4_csdp_bits);
__IO_REG32_BIT(DMA4_CEN17,     				0x480566F4,__READ_WRITE ,__dma4_cen_bits);
__IO_REG32_BIT(DMA4_CFN17,     				0x480566F8,__READ_WRITE ,__dma4_cfn_bits);
__IO_REG32(		 DMA4_CSSA17,    				0x480566FC,__READ_WRITE );
__IO_REG32(		 DMA4_CDSA17,    				0x48056700,__READ_WRITE );
__IO_REG32_BIT(DMA4_CSEI17,    				0x48056704,__READ_WRITE ,__dma4_csei_bits);
__IO_REG32(		 DMA4_CSFI17,    				0x48056708,__READ_WRITE );
__IO_REG32_BIT(DMA4_CDEI17,    				0x4805670C,__READ_WRITE ,__dma4_cdei_bits);
__IO_REG32(		 DMA4_CDFI17,    				0x48056710,__READ_WRITE );
__IO_REG32(		 DMA4_CSAC17,    				0x48056714,__READ				);
__IO_REG32(		 DMA4_CDAC17,    				0x48056718,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCEN17,    				0x4805671C,__READ				,__dma4_ccen_bits);
__IO_REG32_BIT(DMA4_CCFN17,    				0x48056720,__READ				,__dma4_ccfn_bits);
__IO_REG32_BIT(DMA4_COLOR17,     			0x48056724,__READ_WRITE ,__dma4_color_bits);

/***************************************************************************
 **
 ** DMA4 CH18
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA4_CCR18,     				0x48056740,__READ_WRITE ,__dma4_ccr_bits);
__IO_REG32_BIT(DMA4_CLNK_CTRL18,     	0x48056744,__READ_WRITE ,__dma4_clnk_ctrl_bits);
__IO_REG32_BIT(DMA4_CICR18,     			0x48056748,__READ_WRITE ,__dma4_cicr_bits);
__IO_REG32_BIT(DMA4_CSR18,     				0x4805674C,__READ_WRITE ,__dma4_csr_bits);
__IO_REG32_BIT(DMA4_CSDP18,     			0x48056750,__READ_WRITE ,__dma4_csdp_bits);
__IO_REG32_BIT(DMA4_CEN18,     				0x48056754,__READ_WRITE ,__dma4_cen_bits);
__IO_REG32_BIT(DMA4_CFN18,     				0x48056758,__READ_WRITE ,__dma4_cfn_bits);
__IO_REG32(		 DMA4_CSSA18,    				0x4805675C,__READ_WRITE );
__IO_REG32(		 DMA4_CDSA18,    				0x48056760,__READ_WRITE );
__IO_REG32_BIT(DMA4_CSEI18,    				0x48056764,__READ_WRITE ,__dma4_csei_bits);
__IO_REG32(		 DMA4_CSFI18,    				0x48056768,__READ_WRITE );
__IO_REG32_BIT(DMA4_CDEI18,    				0x4805676C,__READ_WRITE ,__dma4_cdei_bits);
__IO_REG32(		 DMA4_CDFI18,    				0x48056770,__READ_WRITE );
__IO_REG32(		 DMA4_CSAC18,    				0x48056774,__READ				);
__IO_REG32(		 DMA4_CDAC18,    				0x48056778,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCEN18,    				0x4805677C,__READ				,__dma4_ccen_bits);
__IO_REG32_BIT(DMA4_CCFN18,    				0x48056780,__READ				,__dma4_ccfn_bits);
__IO_REG32_BIT(DMA4_COLOR18,     			0x48056784,__READ_WRITE ,__dma4_color_bits);

/***************************************************************************
 **
 ** DMA4 CH19
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA4_CCR19,     				0x480567A0,__READ_WRITE ,__dma4_ccr_bits);
__IO_REG32_BIT(DMA4_CLNK_CTRL19,     	0x480567A4,__READ_WRITE ,__dma4_clnk_ctrl_bits);
__IO_REG32_BIT(DMA4_CICR19,     			0x480567A8,__READ_WRITE ,__dma4_cicr_bits);
__IO_REG32_BIT(DMA4_CSR19,     				0x480567AC,__READ_WRITE ,__dma4_csr_bits);
__IO_REG32_BIT(DMA4_CSDP19,     			0x480567B0,__READ_WRITE ,__dma4_csdp_bits);
__IO_REG32_BIT(DMA4_CEN19,     				0x480567B4,__READ_WRITE ,__dma4_cen_bits);
__IO_REG32_BIT(DMA4_CFN19,     				0x480567B8,__READ_WRITE ,__dma4_cfn_bits);
__IO_REG32(		 DMA4_CSSA19,    				0x480567BC,__READ_WRITE );
__IO_REG32(		 DMA4_CDSA19,    				0x480567C0,__READ_WRITE );
__IO_REG32_BIT(DMA4_CSEI19,    				0x480567C4,__READ_WRITE ,__dma4_csei_bits);
__IO_REG32(		 DMA4_CSFI19,    				0x480567C8,__READ_WRITE );
__IO_REG32_BIT(DMA4_CDEI19,    				0x480567CC,__READ_WRITE ,__dma4_cdei_bits);
__IO_REG32(		 DMA4_CDFI19,    				0x480567D0,__READ_WRITE );
__IO_REG32(		 DMA4_CSAC19,    				0x480567D4,__READ				);
__IO_REG32(		 DMA4_CDAC19,    				0x480567D8,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCEN19,    				0x480567DC,__READ				,__dma4_ccen_bits);
__IO_REG32_BIT(DMA4_CCFN19,    				0x480567E0,__READ				,__dma4_ccfn_bits);
__IO_REG32_BIT(DMA4_COLOR19,     			0x480567E4,__READ_WRITE ,__dma4_color_bits);

/***************************************************************************
 **
 ** DMA4 CH20
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA4_CCR20,     				0x48056800,__READ_WRITE ,__dma4_ccr_bits);
__IO_REG32_BIT(DMA4_CLNK_CTRL20,     	0x48056804,__READ_WRITE ,__dma4_clnk_ctrl_bits);
__IO_REG32_BIT(DMA4_CICR20,     			0x48056808,__READ_WRITE ,__dma4_cicr_bits);
__IO_REG32_BIT(DMA4_CSR20,     				0x4805680C,__READ_WRITE ,__dma4_csr_bits);
__IO_REG32_BIT(DMA4_CSDP20,     			0x48056810,__READ_WRITE ,__dma4_csdp_bits);
__IO_REG32_BIT(DMA4_CEN20,     				0x48056814,__READ_WRITE ,__dma4_cen_bits);
__IO_REG32_BIT(DMA4_CFN20,     				0x48056818,__READ_WRITE ,__dma4_cfn_bits);
__IO_REG32(		 DMA4_CSSA20,    				0x4805681C,__READ_WRITE );
__IO_REG32(		 DMA4_CDSA20,    				0x48056820,__READ_WRITE );
__IO_REG32_BIT(DMA4_CSEI20,    				0x48056824,__READ_WRITE ,__dma4_csei_bits);
__IO_REG32(		 DMA4_CSFI20,    				0x48056828,__READ_WRITE );
__IO_REG32_BIT(DMA4_CDEI20,    				0x4805682C,__READ_WRITE ,__dma4_cdei_bits);
__IO_REG32(		 DMA4_CDFI20,    				0x48056830,__READ_WRITE );
__IO_REG32(		 DMA4_CSAC20,    				0x48056834,__READ				);
__IO_REG32(		 DMA4_CDAC20,    				0x48056838,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCEN20,    				0x4805683C,__READ				,__dma4_ccen_bits);
__IO_REG32_BIT(DMA4_CCFN20,    				0x48056840,__READ				,__dma4_ccfn_bits);
__IO_REG32_BIT(DMA4_COLOR20,     			0x48056844,__READ_WRITE ,__dma4_color_bits);

/***************************************************************************
 **
 ** DMA4 CH21
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA4_CCR21,     				0x48056860,__READ_WRITE ,__dma4_ccr_bits);
__IO_REG32_BIT(DMA4_CLNK_CTRL21,     	0x48056864,__READ_WRITE ,__dma4_clnk_ctrl_bits);
__IO_REG32_BIT(DMA4_CICR21,     			0x48056868,__READ_WRITE ,__dma4_cicr_bits);
__IO_REG32_BIT(DMA4_CSR21,     				0x4805686C,__READ_WRITE ,__dma4_csr_bits);
__IO_REG32_BIT(DMA4_CSDP21,     			0x48056870,__READ_WRITE ,__dma4_csdp_bits);
__IO_REG32_BIT(DMA4_CEN21,     				0x48056874,__READ_WRITE ,__dma4_cen_bits);
__IO_REG32_BIT(DMA4_CFN21,     				0x48056878,__READ_WRITE ,__dma4_cfn_bits);
__IO_REG32(		 DMA4_CSSA21,    				0x4805687C,__READ_WRITE );
__IO_REG32(		 DMA4_CDSA21,    				0x48056880,__READ_WRITE );
__IO_REG32_BIT(DMA4_CSEI21,    				0x48056884,__READ_WRITE ,__dma4_csei_bits);
__IO_REG32(		 DMA4_CSFI21,    				0x48056888,__READ_WRITE );
__IO_REG32_BIT(DMA4_CDEI21,    				0x4805688C,__READ_WRITE ,__dma4_cdei_bits);
__IO_REG32(		 DMA4_CDFI21,    				0x48056890,__READ_WRITE );
__IO_REG32(		 DMA4_CSAC21,    				0x48056894,__READ				);
__IO_REG32(		 DMA4_CDAC21,    				0x48056898,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCEN21,    				0x4805689C,__READ				,__dma4_ccen_bits);
__IO_REG32_BIT(DMA4_CCFN21,    				0x480568A0,__READ				,__dma4_ccfn_bits);
__IO_REG32_BIT(DMA4_COLOR21,     			0x480568A4,__READ_WRITE ,__dma4_color_bits);

/***************************************************************************
 **
 ** DMA4 CH22
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA4_CCR22,     				0x480568C0,__READ_WRITE ,__dma4_ccr_bits);
__IO_REG32_BIT(DMA4_CLNK_CTRL22,     	0x480568C4,__READ_WRITE ,__dma4_clnk_ctrl_bits);
__IO_REG32_BIT(DMA4_CICR22,    				0x480568C8,__READ_WRITE ,__dma4_cicr_bits);
__IO_REG32_BIT(DMA4_CSR22,     				0x480568CC,__READ_WRITE ,__dma4_csr_bits);
__IO_REG32_BIT(DMA4_CSDP22,    				0x480568D0,__READ_WRITE ,__dma4_csdp_bits);
__IO_REG32_BIT(DMA4_CEN22,     				0x480568D4,__READ_WRITE ,__dma4_cen_bits);
__IO_REG32_BIT(DMA4_CFN22,     				0x480568D8,__READ_WRITE ,__dma4_cfn_bits);
__IO_REG32(		 DMA4_CSSA22,    				0x480568DC,__READ_WRITE );
__IO_REG32(		 DMA4_CDSA22,    				0x480568E0,__READ_WRITE );
__IO_REG32_BIT(DMA4_CSEI22,    				0x480568E4,__READ_WRITE ,__dma4_csei_bits);
__IO_REG32(		 DMA4_CSFI22,    				0x480568E8,__READ_WRITE );
__IO_REG32_BIT(DMA4_CDEI22,    				0x480568EC,__READ_WRITE ,__dma4_cdei_bits);
__IO_REG32(		 DMA4_CDFI22,    				0x480568F0,__READ_WRITE );
__IO_REG32(		 DMA4_CSAC22,    				0x480568F4,__READ				);
__IO_REG32(		 DMA4_CDAC22,    				0x480568F8,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCEN22,    				0x480568FC,__READ				,__dma4_ccen_bits);
__IO_REG32_BIT(DMA4_CCFN22,    				0x48056900,__READ				,__dma4_ccfn_bits);
__IO_REG32_BIT(DMA4_COLOR22,     			0x48056904,__READ_WRITE ,__dma4_color_bits);

/***************************************************************************
 **
 ** DMA4 CH23
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA4_CCR23,     				0x48056920,__READ_WRITE ,__dma4_ccr_bits);
__IO_REG32_BIT(DMA4_CLNK_CTRL23,     	0x48056924,__READ_WRITE ,__dma4_clnk_ctrl_bits);
__IO_REG32_BIT(DMA4_CICR23,    				0x48056928,__READ_WRITE ,__dma4_cicr_bits);
__IO_REG32_BIT(DMA4_CSR23,     				0x4805692C,__READ_WRITE ,__dma4_csr_bits);
__IO_REG32_BIT(DMA4_CSDP23,    				0x48056930,__READ_WRITE ,__dma4_csdp_bits);
__IO_REG32_BIT(DMA4_CEN23,     				0x48056934,__READ_WRITE ,__dma4_cen_bits);
__IO_REG32_BIT(DMA4_CFN23,     				0x48056938,__READ_WRITE ,__dma4_cfn_bits);
__IO_REG32(		 DMA4_CSSA23,    				0x4805693C,__READ_WRITE );
__IO_REG32(		 DMA4_CDSA23,    				0x48056940,__READ_WRITE );
__IO_REG32_BIT(DMA4_CSEI23,    				0x48056944,__READ_WRITE ,__dma4_csei_bits);
__IO_REG32(		 DMA4_CSFI23,    				0x48056948,__READ_WRITE );
__IO_REG32_BIT(DMA4_CDEI23,    				0x4805694C,__READ_WRITE ,__dma4_cdei_bits);
__IO_REG32(		 DMA4_CDFI23,    				0x48056950,__READ_WRITE );
__IO_REG32(		 DMA4_CSAC23,    				0x48056954,__READ				);
__IO_REG32(		 DMA4_CDAC23,    				0x48056958,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCEN23,    				0x4805695C,__READ				,__dma4_ccen_bits);
__IO_REG32_BIT(DMA4_CCFN23,    				0x48056960,__READ				,__dma4_ccfn_bits);
__IO_REG32_BIT(DMA4_COLOR23,     			0x48056964,__READ_WRITE ,__dma4_color_bits);

/***************************************************************************
 **
 ** DMA4 CH24
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA4_CCR24,     				0x48056980,__READ_WRITE ,__dma4_ccr_bits);
__IO_REG32_BIT(DMA4_CLNK_CTRL24,     	0x48056984,__READ_WRITE ,__dma4_clnk_ctrl_bits);
__IO_REG32_BIT(DMA4_CICR24,    				0x48056988,__READ_WRITE ,__dma4_cicr_bits);
__IO_REG32_BIT(DMA4_CSR24,     				0x4805698C,__READ_WRITE ,__dma4_csr_bits);
__IO_REG32_BIT(DMA4_CSDP24,    				0x48056990,__READ_WRITE ,__dma4_csdp_bits);
__IO_REG32_BIT(DMA4_CEN24,     				0x48056994,__READ_WRITE ,__dma4_cen_bits);
__IO_REG32_BIT(DMA4_CFN24,     				0x48056998,__READ_WRITE ,__dma4_cfn_bits);
__IO_REG32(		 DMA4_CSSA24,    				0x4805699C,__READ_WRITE );
__IO_REG32(		 DMA4_CDSA24,    				0x480569A0,__READ_WRITE );
__IO_REG32_BIT(DMA4_CSEI24,    				0x480569A4,__READ_WRITE ,__dma4_csei_bits);
__IO_REG32(		 DMA4_CSFI24,    				0x480569A8,__READ_WRITE );
__IO_REG32_BIT(DMA4_CDEI24,    				0x480569AC,__READ_WRITE ,__dma4_cdei_bits);
__IO_REG32(		 DMA4_CDFI24,    				0x480569B0,__READ_WRITE );
__IO_REG32(		 DMA4_CSAC24,    				0x480569B4,__READ				);
__IO_REG32(		 DMA4_CDAC24,    				0x480569B8,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCEN24,    				0x480569BC,__READ				,__dma4_ccen_bits);
__IO_REG32_BIT(DMA4_CCFN24,    				0x480569C0,__READ				,__dma4_ccfn_bits);
__IO_REG32_BIT(DMA4_COLOR24,     			0x480569C4,__READ_WRITE ,__dma4_color_bits);

/***************************************************************************
 **
 ** DMA4 CH25
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA4_CCR25,     				0x480569E0,__READ_WRITE ,__dma4_ccr_bits);
__IO_REG32_BIT(DMA4_CLNK_CTRL25,     	0x480569E4,__READ_WRITE ,__dma4_clnk_ctrl_bits);
__IO_REG32_BIT(DMA4_CICR25,    				0x480569E8,__READ_WRITE ,__dma4_cicr_bits);
__IO_REG32_BIT(DMA4_CSR25,     				0x480569EC,__READ_WRITE ,__dma4_csr_bits);
__IO_REG32_BIT(DMA4_CSDP25,    				0x480569F0,__READ_WRITE ,__dma4_csdp_bits);
__IO_REG32_BIT(DMA4_CEN25,     				0x480569F4,__READ_WRITE ,__dma4_cen_bits);
__IO_REG32_BIT(DMA4_CFN25,     				0x480569F8,__READ_WRITE ,__dma4_cfn_bits);
__IO_REG32(		 DMA4_CSSA25,    				0x480569FC,__READ_WRITE );
__IO_REG32(		 DMA4_CDSA25,    				0x48056A00,__READ_WRITE );
__IO_REG32_BIT(DMA4_CSEI25,    				0x48056A04,__READ_WRITE ,__dma4_csei_bits);
__IO_REG32(		 DMA4_CSFI25,    				0x48056A08,__READ_WRITE );
__IO_REG32_BIT(DMA4_CDEI25,    				0x48056A0C,__READ_WRITE ,__dma4_cdei_bits);
__IO_REG32(		 DMA4_CDFI25,    				0x48056A10,__READ_WRITE );
__IO_REG32(		 DMA4_CSAC25,    				0x48056A14,__READ				);
__IO_REG32(		 DMA4_CDAC25,    				0x48056A18,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCEN25,    				0x48056A1C,__READ				,__dma4_ccen_bits);
__IO_REG32_BIT(DMA4_CCFN25,    				0x48056A20,__READ				,__dma4_ccfn_bits);
__IO_REG32_BIT(DMA4_COLOR25,     			0x48056A24,__READ_WRITE ,__dma4_color_bits);

/***************************************************************************
 **
 ** DMA4 CH26
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA4_CCR26,     				0x48056A40,__READ_WRITE ,__dma4_ccr_bits);
__IO_REG32_BIT(DMA4_CLNK_CTRL26,     	0x48056A44,__READ_WRITE ,__dma4_clnk_ctrl_bits);
__IO_REG32_BIT(DMA4_CICR26,     			0x48056A48,__READ_WRITE ,__dma4_cicr_bits);
__IO_REG32_BIT(DMA4_CSR26,     				0x48056A4C,__READ_WRITE ,__dma4_csr_bits);
__IO_REG32_BIT(DMA4_CSDP26,     			0x48056A50,__READ_WRITE ,__dma4_csdp_bits);
__IO_REG32_BIT(DMA4_CEN26,     				0x48056A54,__READ_WRITE ,__dma4_cen_bits);
__IO_REG32_BIT(DMA4_CFN26,     				0x48056A58,__READ_WRITE ,__dma4_cfn_bits);
__IO_REG32(		 DMA4_CSSA26,    				0x48056A5C,__READ_WRITE );
__IO_REG32(		 DMA4_CDSA26,    				0x48056A60,__READ_WRITE );
__IO_REG32_BIT(DMA4_CSEI26,    				0x48056A64,__READ_WRITE ,__dma4_csei_bits);
__IO_REG32(		 DMA4_CSFI26,    				0x48056A68,__READ_WRITE );
__IO_REG32_BIT(DMA4_CDEI26,    				0x48056A6C,__READ_WRITE ,__dma4_cdei_bits);
__IO_REG32(		 DMA4_CDFI26,    				0x48056A70,__READ_WRITE );
__IO_REG32(		 DMA4_CSAC26,    				0x48056A74,__READ				);
__IO_REG32(		 DMA4_CDAC26,    				0x48056A78,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCEN26,    				0x48056A7C,__READ				,__dma4_ccen_bits);
__IO_REG32_BIT(DMA4_CCFN26,    				0x48056A80,__READ				,__dma4_ccfn_bits);
__IO_REG32_BIT(DMA4_COLOR26,     			0x48056A84,__READ_WRITE ,__dma4_color_bits);

/***************************************************************************
 **
 ** DMA4 CH27
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA4_CCR27,     				0x48056AA0,__READ_WRITE ,__dma4_ccr_bits);
__IO_REG32_BIT(DMA4_CLNK_CTRL27,     	0x48056AA4,__READ_WRITE ,__dma4_clnk_ctrl_bits);
__IO_REG32_BIT(DMA4_CICR27,     			0x48056AA8,__READ_WRITE ,__dma4_cicr_bits);
__IO_REG32_BIT(DMA4_CSR27,     				0x48056AAC,__READ_WRITE ,__dma4_csr_bits);
__IO_REG32_BIT(DMA4_CSDP27,     			0x48056AB0,__READ_WRITE ,__dma4_csdp_bits);
__IO_REG32_BIT(DMA4_CEN27,     				0x48056AB4,__READ_WRITE ,__dma4_cen_bits);
__IO_REG32_BIT(DMA4_CFN27,     				0x48056AB8,__READ_WRITE ,__dma4_cfn_bits);
__IO_REG32(		 DMA4_CSSA27,    				0x48056ABC,__READ_WRITE );
__IO_REG32(		 DMA4_CDSA27,    				0x48056AC0,__READ_WRITE );
__IO_REG32_BIT(DMA4_CSEI27,    				0x48056AC4,__READ_WRITE ,__dma4_csei_bits);
__IO_REG32(		 DMA4_CSFI27,    				0x48056AC8,__READ_WRITE );
__IO_REG32_BIT(DMA4_CDEI27,    				0x48056ACC,__READ_WRITE ,__dma4_cdei_bits);
__IO_REG32(		 DMA4_CDFI27,    				0x48056AD0,__READ_WRITE );
__IO_REG32(		 DMA4_CSAC27,    				0x48056AD4,__READ				);
__IO_REG32(		 DMA4_CDAC27,    				0x48056AD8,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCEN27,    				0x48056ADC,__READ				,__dma4_ccen_bits);
__IO_REG32_BIT(DMA4_CCFN27,    				0x48056AE0,__READ				,__dma4_ccfn_bits);
__IO_REG32_BIT(DMA4_COLOR27,     			0x48056AE4,__READ_WRITE ,__dma4_color_bits);

/***************************************************************************
 **
 ** DMA4 CH28
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA4_CCR28,     				0x48056B00,__READ_WRITE ,__dma4_ccr_bits);
__IO_REG32_BIT(DMA4_CLNK_CTRL28,     	0x48056B04,__READ_WRITE ,__dma4_clnk_ctrl_bits);
__IO_REG32_BIT(DMA4_CICR28,     			0x48056B08,__READ_WRITE ,__dma4_cicr_bits);
__IO_REG32_BIT(DMA4_CSR28,     				0x48056B0C,__READ_WRITE ,__dma4_csr_bits);
__IO_REG32_BIT(DMA4_CSDP28,     			0x48056B10,__READ_WRITE ,__dma4_csdp_bits);
__IO_REG32_BIT(DMA4_CEN28,     				0x48056B14,__READ_WRITE ,__dma4_cen_bits);
__IO_REG32_BIT(DMA4_CFN28,     				0x48056B18,__READ_WRITE ,__dma4_cfn_bits);
__IO_REG32(		 DMA4_CSSA28,    				0x48056B1C,__READ_WRITE );
__IO_REG32(		 DMA4_CDSA28,    				0x48056B20,__READ_WRITE );
__IO_REG32_BIT(DMA4_CSEI28,    				0x48056B24,__READ_WRITE ,__dma4_csei_bits);
__IO_REG32(		 DMA4_CSFI28,    				0x48056B28,__READ_WRITE );
__IO_REG32_BIT(DMA4_CDEI28,    				0x48056B2C,__READ_WRITE ,__dma4_cdei_bits);
__IO_REG32(		 DMA4_CDFI28,    				0x48056B30,__READ_WRITE );
__IO_REG32(		 DMA4_CSAC28,    				0x48056B34,__READ				);
__IO_REG32(		 DMA4_CDAC28,    				0x48056B38,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCEN28,    				0x48056B3C,__READ				,__dma4_ccen_bits);
__IO_REG32_BIT(DMA4_CCFN28,    				0x48056B40,__READ				,__dma4_ccfn_bits);
__IO_REG32_BIT(DMA4_COLOR28,     			0x48056B44,__READ_WRITE ,__dma4_color_bits);

/***************************************************************************
 **
 ** DMA4 CH29
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA4_CCR29,     				0x48056B60,__READ_WRITE ,__dma4_ccr_bits);
__IO_REG32_BIT(DMA4_CLNK_CTRL29,     	0x48056B64,__READ_WRITE ,__dma4_clnk_ctrl_bits);
__IO_REG32_BIT(DMA4_CICR29,     			0x48056B68,__READ_WRITE ,__dma4_cicr_bits);
__IO_REG32_BIT(DMA4_CSR29,     				0x48056B6C,__READ_WRITE ,__dma4_csr_bits);
__IO_REG32_BIT(DMA4_CSDP29,     			0x48056B70,__READ_WRITE ,__dma4_csdp_bits);
__IO_REG32_BIT(DMA4_CEN29,     				0x48056B74,__READ_WRITE ,__dma4_cen_bits);
__IO_REG32_BIT(DMA4_CFN29,     				0x48056B78,__READ_WRITE ,__dma4_cfn_bits);
__IO_REG32(		 DMA4_CSSA29,    				0x48056B7C,__READ_WRITE );
__IO_REG32(		 DMA4_CDSA29,    				0x48056B80,__READ_WRITE );
__IO_REG32_BIT(DMA4_CSEI29,    				0x48056B84,__READ_WRITE ,__dma4_csei_bits);
__IO_REG32(		 DMA4_CSFI29,    				0x48056B88,__READ_WRITE );
__IO_REG32_BIT(DMA4_CDEI29,    				0x48056B8C,__READ_WRITE ,__dma4_cdei_bits);
__IO_REG32(		 DMA4_CDFI29,    				0x48056B90,__READ_WRITE );
__IO_REG32(		 DMA4_CSAC29,    				0x48056B94,__READ				);
__IO_REG32(		 DMA4_CDAC29,    				0x48056B98,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCEN29,    				0x48056B9C,__READ				,__dma4_ccen_bits);
__IO_REG32_BIT(DMA4_CCFN29,    				0x48056BA0,__READ				,__dma4_ccfn_bits);
__IO_REG32_BIT(DMA4_COLOR29,     			0x48056BA4,__READ_WRITE ,__dma4_color_bits);

/***************************************************************************
 **
 ** DMA4 CH30
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA4_CCR30,     				0x48056BC0,__READ_WRITE ,__dma4_ccr_bits);
__IO_REG32_BIT(DMA4_CLNK_CTRL30,     	0x48056BC4,__READ_WRITE ,__dma4_clnk_ctrl_bits);
__IO_REG32_BIT(DMA4_CICR30,    				0x48056BC8,__READ_WRITE ,__dma4_cicr_bits);
__IO_REG32_BIT(DMA4_CSR30,     				0x48056BCC,__READ_WRITE ,__dma4_csr_bits);
__IO_REG32_BIT(DMA4_CSDP30,    				0x48056BD0,__READ_WRITE ,__dma4_csdp_bits);
__IO_REG32_BIT(DMA4_CEN30,     				0x48056BD4,__READ_WRITE ,__dma4_cen_bits);
__IO_REG32_BIT(DMA4_CFN30,     				0x48056BD8,__READ_WRITE ,__dma4_cfn_bits);
__IO_REG32(		 DMA4_CSSA30,    				0x48056BDC,__READ_WRITE );
__IO_REG32(		 DMA4_CDSA30,    				0x48056BE0,__READ_WRITE );
__IO_REG32_BIT(DMA4_CSEI30,    				0x48056BE4,__READ_WRITE ,__dma4_csei_bits);
__IO_REG32(		 DMA4_CSFI30,    				0x48056BE8,__READ_WRITE );
__IO_REG32_BIT(DMA4_CDEI30,    				0x48056BEC,__READ_WRITE ,__dma4_cdei_bits);
__IO_REG32(		 DMA4_CDFI30,    				0x48056BF0,__READ_WRITE );
__IO_REG32(		 DMA4_CSAC30,    				0x48056BF4,__READ				);
__IO_REG32(		 DMA4_CDAC30,    				0x48056BF8,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCEN30,    				0x48056BFC,__READ				,__dma4_ccen_bits);
__IO_REG32_BIT(DMA4_CCFN30,    				0x48056C00,__READ				,__dma4_ccfn_bits);
__IO_REG32_BIT(DMA4_COLOR30,     			0x48056C04,__READ_WRITE ,__dma4_color_bits);

/***************************************************************************
 **
 ** DMA4 CH31
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA4_CCR31,     				0x48056C20,__READ_WRITE ,__dma4_ccr_bits);
__IO_REG32_BIT(DMA4_CLNK_CTRL31,     	0x48056C24,__READ_WRITE ,__dma4_clnk_ctrl_bits);
__IO_REG32_BIT(DMA4_CICR31,    				0x48056C28,__READ_WRITE ,__dma4_cicr_bits);
__IO_REG32_BIT(DMA4_CSR31,     				0x48056C2C,__READ_WRITE ,__dma4_csr_bits);
__IO_REG32_BIT(DMA4_CSDP31,    				0x48056C30,__READ_WRITE ,__dma4_csdp_bits);
__IO_REG32_BIT(DMA4_CEN31,     				0x48056C34,__READ_WRITE ,__dma4_cen_bits);
__IO_REG32_BIT(DMA4_CFN31,     				0x48056C38,__READ_WRITE ,__dma4_cfn_bits);
__IO_REG32(		 DMA4_CSSA31,    				0x48056C3C,__READ_WRITE );
__IO_REG32(		 DMA4_CDSA31,    				0x48056C40,__READ_WRITE );
__IO_REG32_BIT(DMA4_CSEI31,    				0x48056C44,__READ_WRITE ,__dma4_csei_bits);
__IO_REG32(		 DMA4_CSFI31,    				0x48056C48,__READ_WRITE );
__IO_REG32_BIT(DMA4_CDEI31,    				0x48056C4C,__READ_WRITE ,__dma4_cdei_bits);
__IO_REG32(		 DMA4_CDFI31,    				0x48056C50,__READ_WRITE );
__IO_REG32(		 DMA4_CSAC31,    				0x48056C54,__READ				);
__IO_REG32(		 DMA4_CDAC31,    				0x48056C58,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCEN31,    				0x48056C5C,__READ				,__dma4_ccen_bits);
__IO_REG32_BIT(DMA4_CCFN31,    				0x48056C60,__READ				,__dma4_ccfn_bits);
__IO_REG32_BIT(DMA4_COLOR31,     			0x48056C64,__READ_WRITE ,__dma4_color_bits);

/***************************************************************************
 **
 ** MPU INTC
 **
 ***************************************************************************/
__IO_REG32_BIT(INTCPS_SYSCONFIG,     	0x48200010,__READ_WRITE ,__intcps_sysconfig_bits);
__IO_REG32_BIT(INTCPS_SYSSTATUS,     	0x48200014,__READ				,__intcps_sysstatus_bits);
__IO_REG32_BIT(INTCPS_SIR_IRQ,     		0x48200040,__READ				,__intcps_sir_irq_bits);
__IO_REG32_BIT(INTCPS_SIR_FIQ,     		0x48200044,__READ				,__intcps_sir_fiq_bits);
__IO_REG32_BIT(INTCPS_CONTROL,     		0x48200048,__READ_WRITE ,__intcps_control_bits);
__IO_REG32_BIT(INTCPS_PROTECTION,     0x4820004C,__READ_WRITE ,__intcps_protection_bits);
__IO_REG32_BIT(INTCPS_IDLE,     			0x48200050,__READ_WRITE ,__intcps_idle_bits);
__IO_REG32_BIT(INTCPS_IRQ_PRIORITY,   0x48200060,__READ_WRITE ,__intcps_irq_priority_bits);
__IO_REG32_BIT(INTCPS_FIQ_PRIORITY,   0x48200064,__READ_WRITE ,__intcps_fiq_priority_bits);
__IO_REG32_BIT(INTCPS_THRESHOLD,     	0x48200068,__READ_WRITE ,__intcps_threshold_bits);                                 
__IO_REG32_BIT(INTCPS_ITR0,     			0x48200080,__READ				,__intcps_itr0_bits);
__IO_REG32_BIT(INTCPS_MIR0,     			0x48200084,__READ_WRITE ,__intcps_mir0_bits);
__IO_REG32_BIT(INTCPS_MIR_CLEAR0,     0x48200088,__WRITE 			,__intcps_mir_clear0_bits);
__IO_REG32_BIT(INTCPS_MIR_SET0,     	0x4820008C,__WRITE 			,__intcps_mir_set0_bits);
__IO_REG32_BIT(INTCPS_ISR_SET0,     	0x48200090,__READ_WRITE ,__intcps_isr_set0_bits);
__IO_REG32_BIT(INTCPS_ISR_CLEAR0,     0x48200094,__WRITE 			,__intcps_isr_clear0_bits);
__IO_REG32_BIT(INTCPS_PENDING_IRQ0,   0x48200098,__READ				,__intcps_pending_irq0_bits);
__IO_REG32_BIT(INTCPS_PENDING_FIQ0,   0x4820009C,__READ				,__intcps_pending_fiq0_bits);
__IO_REG32_BIT(INTCPS_ITR1,     			0x482000A0,__READ				,__intcps_itr1_bits);
__IO_REG32_BIT(INTCPS_MIR1,     			0x482000A4,__READ_WRITE ,__intcps_mir1_bits);
__IO_REG32_BIT(INTCPS_MIR_CLEAR1,     0x482000A8,__WRITE 			,__intcps_mir_clear1_bits);
__IO_REG32_BIT(INTCPS_MIR_SET1,     	0x482000AC,__WRITE 			,__intcps_mir_set1_bits);
__IO_REG32_BIT(INTCPS_ISR_SET1,     	0x482000B0,__READ_WRITE ,__intcps_isr_set1_bits);
__IO_REG32_BIT(INTCPS_ISR_CLEAR1,     0x482000B4,__WRITE 			,__intcps_isr_clear1_bits);
__IO_REG32_BIT(INTCPS_PENDING_IRQ1,   0x482000B8,__READ				,__intcps_pending_irq1_bits);
__IO_REG32_BIT(INTCPS_PENDING_FIQ1,   0x482000BC,__READ				,__intcps_pending_fiq1_bits);
__IO_REG32_BIT(INTCPS_ITR2,     			0x482000C0,__READ				,__intcps_itr2_bits);
__IO_REG32_BIT(INTCPS_MIR2,     			0x482000C4,__READ_WRITE ,__intcps_mir2_bits);
__IO_REG32_BIT(INTCPS_MIR_CLEAR2,     0x482000C8,__WRITE 			,__intcps_mir_clear2_bits);
__IO_REG32_BIT(INTCPS_MIR_SET2,     	0x482000CC,__WRITE 			,__intcps_mir_set2_bits);
__IO_REG32_BIT(INTCPS_ISR_SET2,     	0x482000D0,__READ_WRITE ,__intcps_isr_set2_bits);
__IO_REG32_BIT(INTCPS_ISR_CLEAR2,     0x482000D4,__WRITE 			,__intcps_isr_clear2_bits);
__IO_REG32_BIT(INTCPS_PENDING_IRQ2,   0x482000D8,__READ				,__intcps_pending_irq2_bits);
__IO_REG32_BIT(INTCPS_PENDING_FIQ2,   0x482000DC,__READ				,__intcps_pending_fiq2_bits);
__IO_REG32_BIT(INTCPS_ILR0,     			0x48200100,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR1,     			0x48200104,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR2,     			0x48200108,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR3,     			0x4820010C,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR4,     			0x48200110,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR5,     			0x48200114,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR6,     			0x48200118,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR7,     			0x4820011C,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR8,     			0x48200120,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR9,     			0x48200124,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR10,     			0x48200128,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR11,     			0x4820012C,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR12,     			0x48200130,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR13,     			0x48200134,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR14,     			0x48200138,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR15,     			0x4820013C,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR16,     			0x48200140,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR17,     			0x48200144,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR18,     			0x48200148,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR19,     			0x4820014C,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR20,     			0x48200150,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR21,     			0x48200154,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR22,     			0x48200158,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR23,     			0x4820015C,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR24,     			0x48200160,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR25,     			0x48200164,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR26,     			0x48200168,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR27,     			0x4820016C,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR28,     			0x48200170,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR29,     			0x48200174,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR30,     			0x48200178,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR31,     			0x4820017C,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR32,     			0x48200180,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR33,     			0x48200184,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR34,     			0x48200188,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR35,     			0x4820018C,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR36,     			0x48200190,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR37,     			0x48200194,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR38,     			0x48200198,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR39,     			0x4820019C,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR40,     			0x482001A0,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR41,     			0x482001A4,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR42,     			0x482001A8,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR43,     			0x482001AC,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR44,     			0x482001B0,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR45,     			0x482001B4,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR46,     			0x482001B8,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR47,     			0x482001BC,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR48,     			0x482001C0,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR49,     			0x482001C4,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR50,     			0x482001C8,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR51,     			0x482001CC,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR52,     			0x482001D0,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR53,     			0x482001D4,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR54,     			0x482001D8,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR55,     			0x482001DC,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR56,     			0x482001E0,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR57,     			0x482001E4,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR58,     			0x482001E8,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR59,     			0x482001EC,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR60,     			0x482001F0,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR61,     			0x482001F4,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR62,     			0x482001F8,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR63,     			0x482001FC,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR64,     			0x48200200,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR65,     			0x48200204,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR66,     			0x48200208,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR67,     			0x4820020C,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR68,     			0x48200210,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR69,     			0x48200214,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR70,     			0x48200218,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR71,     			0x4820021C,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR72,     			0x48200220,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR73,     			0x48200224,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR74,     			0x48200228,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR75,     			0x4820022C,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR76,     			0x48200230,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR77,     			0x48200234,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR78,     			0x48200238,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR79,     			0x4820023C,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR80,     			0x48200240,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR81,     			0x48200244,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR82,     			0x48200248,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR83,     			0x4820024C,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR84,     			0x48200250,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR85,     			0x48200254,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR86,     			0x48200258,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR87,     			0x4820025C,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR88,     			0x48200260,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR89,     			0x48200264,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR90,     			0x48200268,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR91,     			0x4820026C,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR92,     			0x48200270,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR93,     			0x48200274,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR94,     			0x48200278,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTCPS_ILR95,     			0x4820027C,__READ_WRITE ,__intcps_ilr_bits);
__IO_REG32_BIT(INTC_INIT_REGISTER1,   0x480C7010,__READ_WRITE ,__intc_init_register1_bits);
__IO_REG32_BIT(INTC_INIT_REGISTER2,   0x480C7050,__READ_WRITE ,__intc_init_register2_bits);

/***************************************************************************
 **
 ** GPMC
 **
 ***************************************************************************/
__IO_REG32_BIT(GPMC_SYSCONFIG,     		0x6E000010,__READ_WRITE ,__gpmc_sysconfig_bits);
__IO_REG32_BIT(GPMC_SYSSTATUS,     		0x6E000014,__READ				,__gpmc_sysstatus_bits);
__IO_REG32_BIT(GPMC_IRQSTATUS,     		0x6E000018,__READ_WRITE ,__gpmc_irqstatus_bits);
__IO_REG32_BIT(GPMC_IRQENABLE,     		0x6E00001C,__READ_WRITE ,__gpmc_irqenable_bits);
__IO_REG32_BIT(GPMC_TIMEOUT_CONTROL,  0x6E000040,__READ_WRITE ,__gpmc_timeout_control_bits);
__IO_REG32_BIT(GPMC_ERR_ADDRESS,     	0x6E000044,__READ_WRITE ,__gpmc_err_address_bits);
__IO_REG32_BIT(GPMC_ERR_TYPE,     		0x6E000048,__READ_WRITE ,__gpmc_err_type_bits);
__IO_REG32_BIT(GPMC_CONFIG,     			0x6E000050,__READ_WRITE ,__gpmc_config_bits);
__IO_REG32_BIT(GPMC_STATUS,     			0x6E000054,__READ_WRITE ,__gpmc_status_bits);
__IO_REG32_BIT(GPMC_CONFIG1_0,     		0x6E000060,__READ_WRITE ,__gpmc_config1_bits);
__IO_REG32_BIT(GPMC_CONFIG2_0,     		0x6E000064,__READ_WRITE ,__gpmc_config2_bits);
__IO_REG32_BIT(GPMC_CONFIG3_0,     		0x6E000068,__READ_WRITE ,__gpmc_config3_bits);
__IO_REG32_BIT(GPMC_CONFIG4_0,     		0x6E00006C,__READ_WRITE ,__gpmc_config4_bits);
__IO_REG32_BIT(GPMC_CONFIG5_0,     		0x6E000070,__READ_WRITE ,__gpmc_config5_bits);
__IO_REG32_BIT(GPMC_CONFIG6_0,     		0x6E000074,__READ_WRITE ,__gpmc_config6_bits);
__IO_REG32_BIT(GPMC_CONFIG7_0,     		0x6E000078,__READ_WRITE ,__gpmc_config7_bits);
__IO_REG32(		 GPMC_NAND_COMMAND_0,   0x6E00007C,__WRITE 			);
__IO_REG32(		 GPMC_NAND_ADDRESS_0,   0x6E000080,__WRITE 			);
__IO_REG32(		 GPMC_NAND_DATA_0,   		0x6E000084,__READ_WRITE );
__IO_REG32_BIT(GPMC_CONFIG1_1,     		0x6E000090,__READ_WRITE ,__gpmc_config1_bits);
__IO_REG32_BIT(GPMC_CONFIG2_1,     		0x6E000094,__READ_WRITE ,__gpmc_config2_bits);
__IO_REG32_BIT(GPMC_CONFIG3_1,     		0x6E000098,__READ_WRITE ,__gpmc_config3_bits);
__IO_REG32_BIT(GPMC_CONFIG4_1,     		0x6E00009C,__READ_WRITE ,__gpmc_config4_bits);
__IO_REG32_BIT(GPMC_CONFIG5_1,     		0x6E0000A0,__READ_WRITE ,__gpmc_config5_bits);
__IO_REG32_BIT(GPMC_CONFIG6_1,     		0x6E0000A4,__READ_WRITE ,__gpmc_config6_bits);
__IO_REG32_BIT(GPMC_CONFIG7_1,     		0x6E0000A8,__READ_WRITE ,__gpmc_config7_bits);
__IO_REG32(		 GPMC_NAND_COMMAND_1,   0x6E0000AC,__WRITE 			);
__IO_REG32(		 GPMC_NAND_ADDRESS_1,   0x6E0000B0,__WRITE 			);
__IO_REG32(		 GPMC_NAND_DATA_1,   		0x6E0000B4,__READ_WRITE );
__IO_REG32_BIT(GPMC_CONFIG1_2,     		0x6E0000C0,__READ_WRITE ,__gpmc_config1_bits);
__IO_REG32_BIT(GPMC_CONFIG2_2,     		0x6E0000C4,__READ_WRITE ,__gpmc_config2_bits);
__IO_REG32_BIT(GPMC_CONFIG3_2,     		0x6E0000C8,__READ_WRITE ,__gpmc_config3_bits);
__IO_REG32_BIT(GPMC_CONFIG4_2,     		0x6E0000CC,__READ_WRITE ,__gpmc_config4_bits);
__IO_REG32_BIT(GPMC_CONFIG5_2,     		0x6E0000D0,__READ_WRITE ,__gpmc_config5_bits);
__IO_REG32_BIT(GPMC_CONFIG6_2,     		0x6E0000D4,__READ_WRITE ,__gpmc_config6_bits);
__IO_REG32_BIT(GPMC_CONFIG7_2,     		0x6E0000D8,__READ_WRITE ,__gpmc_config7_bits);
__IO_REG32(		 GPMC_NAND_COMMAND_2,   0x6E0000DC,__WRITE 			);
__IO_REG32(		 GPMC_NAND_ADDRESS_2,   0x6E0000E0,__WRITE 			);
__IO_REG32(		 GPMC_NAND_DATA_2,   		0x6E0000E4,__READ_WRITE );
__IO_REG32_BIT(GPMC_CONFIG1_3,     		0x6E0000F0,__READ_WRITE ,__gpmc_config1_bits);
__IO_REG32_BIT(GPMC_CONFIG2_3,     		0x6E0000F4,__READ_WRITE ,__gpmc_config2_bits);
__IO_REG32_BIT(GPMC_CONFIG3_3,     		0x6E0000F8,__READ_WRITE ,__gpmc_config3_bits);
__IO_REG32_BIT(GPMC_CONFIG4_3,     		0x6E0000FC,__READ_WRITE ,__gpmc_config4_bits);
__IO_REG32_BIT(GPMC_CONFIG5_3,     		0x6E000100,__READ_WRITE ,__gpmc_config5_bits);
__IO_REG32_BIT(GPMC_CONFIG6_3,     		0x6E000104,__READ_WRITE ,__gpmc_config6_bits);
__IO_REG32_BIT(GPMC_CONFIG7_3,     		0x6E000108,__READ_WRITE ,__gpmc_config7_bits);
__IO_REG32(		 GPMC_NAND_COMMAND_3,   0x6E00010C,__WRITE 			);
__IO_REG32(		 GPMC_NAND_ADDRESS_3,   0x6E000110,__WRITE 			);
__IO_REG32(		 GPMC_NAND_DATA_3,   		0x6E000114,__READ_WRITE );
__IO_REG32_BIT(GPMC_CONFIG1_4,     		0x6E000120,__READ_WRITE ,__gpmc_config1_bits);
__IO_REG32_BIT(GPMC_CONFIG2_4,     		0x6E000124,__READ_WRITE ,__gpmc_config2_bits);
__IO_REG32_BIT(GPMC_CONFIG3_4,     		0x6E000128,__READ_WRITE ,__gpmc_config3_bits);
__IO_REG32_BIT(GPMC_CONFIG4_4,     		0x6E00012C,__READ_WRITE ,__gpmc_config4_bits);
__IO_REG32_BIT(GPMC_CONFIG5_4,     		0x6E000130,__READ_WRITE ,__gpmc_config5_bits);
__IO_REG32_BIT(GPMC_CONFIG6_4,     		0x6E000134,__READ_WRITE ,__gpmc_config6_bits);
__IO_REG32_BIT(GPMC_CONFIG7_4,     		0x6E000138,__READ_WRITE ,__gpmc_config7_bits);
__IO_REG32(		 GPMC_NAND_COMMAND_4,   0x6E00013C,__WRITE 			);
__IO_REG32(		 GPMC_NAND_ADDRESS_4,   0x6E000140,__WRITE 			);
__IO_REG32(		 GPMC_NAND_DATA_4,   		0x6E000144,__READ_WRITE );
__IO_REG32_BIT(GPMC_CONFIG1_5,     		0x6E000150,__READ_WRITE ,__gpmc_config1_bits);
__IO_REG32_BIT(GPMC_CONFIG2_5,     		0x6E000154,__READ_WRITE ,__gpmc_config2_bits);
__IO_REG32_BIT(GPMC_CONFIG3_5,     		0x6E000158,__READ_WRITE ,__gpmc_config3_bits);
__IO_REG32_BIT(GPMC_CONFIG4_5,     		0x6E00015C,__READ_WRITE ,__gpmc_config4_bits);
__IO_REG32_BIT(GPMC_CONFIG5_5,     		0x6E000160,__READ_WRITE ,__gpmc_config5_bits);
__IO_REG32_BIT(GPMC_CONFIG6_5,     		0x6E000164,__READ_WRITE ,__gpmc_config6_bits);
__IO_REG32_BIT(GPMC_CONFIG7_5,     		0x6E000168,__READ_WRITE ,__gpmc_config7_bits);
__IO_REG32(		 GPMC_NAND_COMMAND_5,   0x6E00016C,__WRITE 			);
__IO_REG32(		 GPMC_NAND_ADDRESS_5,   0x6E000170,__WRITE 			);
__IO_REG32(		 GPMC_NAND_DATA_5,   		0x6E000174,__READ_WRITE );
__IO_REG32_BIT(GPMC_CONFIG1_6,     		0x6E000180,__READ_WRITE ,__gpmc_config1_bits);
__IO_REG32_BIT(GPMC_CONFIG2_6,     		0x6E000184,__READ_WRITE ,__gpmc_config2_bits);
__IO_REG32_BIT(GPMC_CONFIG3_6,     		0x6E000188,__READ_WRITE ,__gpmc_config3_bits);
__IO_REG32_BIT(GPMC_CONFIG4_6,     		0x6E00018C,__READ_WRITE ,__gpmc_config4_bits);
__IO_REG32_BIT(GPMC_CONFIG5_6,     		0x6E000190,__READ_WRITE ,__gpmc_config5_bits);
__IO_REG32_BIT(GPMC_CONFIG6_6,     		0x6E000194,__READ_WRITE ,__gpmc_config6_bits);
__IO_REG32_BIT(GPMC_CONFIG7_6,     		0x6E000198,__READ_WRITE ,__gpmc_config7_bits);
__IO_REG32(		 GPMC_NAND_COMMAND_6,   0x6E00019C,__WRITE 			);
__IO_REG32(		 GPMC_NAND_ADDRESS_6,   0x6E0001A0,__WRITE 			);
__IO_REG32(		 GPMC_NAND_DATA_6,   		0x6E0001A4,__READ_WRITE );
__IO_REG32_BIT(GPMC_CONFIG1_7,     		0x6E0001B0,__READ_WRITE ,__gpmc_config1_bits);
__IO_REG32_BIT(GPMC_CONFIG2_7,     		0x6E0001B4,__READ_WRITE ,__gpmc_config2_bits);
__IO_REG32_BIT(GPMC_CONFIG3_7,     		0x6E0001B8,__READ_WRITE ,__gpmc_config3_bits);
__IO_REG32_BIT(GPMC_CONFIG4_7,     		0x6E0001BC,__READ_WRITE ,__gpmc_config4_bits);
__IO_REG32_BIT(GPMC_CONFIG5_7,     		0x6E0001C0,__READ_WRITE ,__gpmc_config5_bits);
__IO_REG32_BIT(GPMC_CONFIG6_7,     		0x6E0001C4,__READ_WRITE ,__gpmc_config6_bits);
__IO_REG32_BIT(GPMC_CONFIG7_7,     		0x6E0001C8,__READ_WRITE ,__gpmc_config7_bits);
__IO_REG32(		 GPMC_NAND_COMMAND_7,   0x6E0001CC,__WRITE 			);
__IO_REG32(		 GPMC_NAND_ADDRESS_7,   0x6E0001D0,__WRITE 			);
__IO_REG32(		 GPMC_NAND_DATA_7,   		0x6E0001D4,__READ_WRITE );
__IO_REG32_BIT(GPMC_PREFETCH_CONFIG1, 0x6E0001E0,__READ_WRITE ,__gpmc_prefetch_config1_bits);
__IO_REG32_BIT(GPMC_PREFETCH_CONFIG2, 0x6E0001E4,__READ_WRITE ,__gpmc_prefetch_config2_bits);
__IO_REG32_BIT(GPMC_PREFETCH_CONTROL, 0x6E0001EC,__READ_WRITE ,__gpmc_prefetch_control_bits);
__IO_REG32_BIT(GPMC_PREFETCH_STATUS,  0x6E0001F0,__READ_WRITE ,__gpmc_prefetch_status_bits);
__IO_REG32_BIT(GPMC_ECC_CONFIG,     	0x6E0001F4,__READ_WRITE ,__gpmc_ecc_config_bits);
__IO_REG32_BIT(GPMC_ECC_CONTROL,     	0x6E0001F8,__READ_WRITE ,__gpmc_ecc_control_bits);
__IO_REG32_BIT(GPMC_ECC_SIZE_CONFIG,  0x6E0001FC,__READ_WRITE ,__gpmc_ecc_size_config_bits);
__IO_REG32_BIT(GPMC_ECC1_RESULT,     	0x6E000200,__READ_WRITE ,__gpmc_ecc_result_bits);
__IO_REG32_BIT(GPMC_ECC2_RESULT,     	0x6E000204,__READ_WRITE ,__gpmc_ecc_result_bits);
__IO_REG32_BIT(GPMC_ECC3_RESULT,     	0x6E000208,__READ_WRITE ,__gpmc_ecc_result_bits);
__IO_REG32_BIT(GPMC_ECC4_RESULT,     	0x6E00020C,__READ_WRITE ,__gpmc_ecc_result_bits);
__IO_REG32_BIT(GPMC_ECC5_RESULT,     	0x6E000210,__READ_WRITE ,__gpmc_ecc_result_bits);
__IO_REG32_BIT(GPMC_ECC6_RESULT,     	0x6E000214,__READ_WRITE ,__gpmc_ecc_result_bits);
__IO_REG32_BIT(GPMC_ECC7_RESULT,     	0x6E000218,__READ_WRITE ,__gpmc_ecc_result_bits);
__IO_REG32_BIT(GPMC_ECC8_RESULT,     	0x6E00021C,__READ_WRITE ,__gpmc_ecc_result_bits);
__IO_REG32_BIT(GPMC_ECC9_RESULT,     	0x6E000220,__READ_WRITE ,__gpmc_ecc_result_bits);
__IO_REG32(		 GPMC_BCH_RESULT0_0,    0x6E000240,__READ_WRITE );
__IO_REG32(		 GPMC_BCH_RESULT1_0,    0x6E000244,__READ_WRITE );
__IO_REG32(		 GPMC_BCH_RESULT2_0,    0x6E00024C,__READ_WRITE );
__IO_REG32_BIT(GPMC_BCH_RESULT3_0,    0x6E000248,__READ_WRITE ,__gpmc_bch_result3_bits);
__IO_REG32(		 GPMC_BCH_RESULT0_1,    0x6E000250,__READ_WRITE );
__IO_REG32(		 GPMC_BCH_RESULT1_1,    0x6E000254,__READ_WRITE );
__IO_REG32(		 GPMC_BCH_RESULT2_1,    0x6E00025C,__READ_WRITE );
__IO_REG32_BIT(GPMC_BCH_RESULT3_1,    0x6E000258,__READ_WRITE ,__gpmc_bch_result3_bits);
__IO_REG32(		 GPMC_BCH_RESULT0_2,    0x6E000260,__READ_WRITE );
__IO_REG32(		 GPMC_BCH_RESULT1_2,    0x6E000264,__READ_WRITE );
__IO_REG32(		 GPMC_BCH_RESULT2_2,    0x6E00026C,__READ_WRITE );
__IO_REG32_BIT(GPMC_BCH_RESULT3_2,    0x6E000268,__READ_WRITE ,__gpmc_bch_result3_bits);
__IO_REG32(		 GPMC_BCH_RESULT0_3,    0x6E000270,__READ_WRITE );
__IO_REG32(		 GPMC_BCH_RESULT1_3,    0x6E000274,__READ_WRITE );
__IO_REG32(		 GPMC_BCH_RESULT2_3,    0x6E00027C,__READ_WRITE );
__IO_REG32_BIT(GPMC_BCH_RESULT3_3,    0x6E000278,__READ_WRITE ,__gpmc_bch_result3_bits);
__IO_REG32(		 GPMC_BCH_RESULT0_4,    0x6E000280,__READ_WRITE );
__IO_REG32(		 GPMC_BCH_RESULT1_4,    0x6E000284,__READ_WRITE );
__IO_REG32(		 GPMC_BCH_RESULT2_4,    0x6E00028C,__READ_WRITE );
__IO_REG32_BIT(GPMC_BCH_RESULT3_4,    0x6E000288,__READ_WRITE ,__gpmc_bch_result3_bits);
__IO_REG32(		 GPMC_BCH_RESULT0_5,    0x6E000290,__READ_WRITE );
__IO_REG32(		 GPMC_BCH_RESULT1_5,    0x6E000294,__READ_WRITE );
__IO_REG32(		 GPMC_BCH_RESULT2_5,    0x6E00029C,__READ_WRITE );
__IO_REG32_BIT(GPMC_BCH_RESULT3_5,    0x6E000298,__READ_WRITE ,__gpmc_bch_result3_bits);
__IO_REG32(		 GPMC_BCH_RESULT0_6,    0x6E0002A0,__READ_WRITE );
__IO_REG32(		 GPMC_BCH_RESULT1_6,    0x6E0002A4,__READ_WRITE );
__IO_REG32(		 GPMC_BCH_RESULT2_6,    0x6E0002AC,__READ_WRITE );
__IO_REG32_BIT(GPMC_BCH_RESULT3_6,    0x6E0002A8,__READ_WRITE ,__gpmc_bch_result3_bits);
__IO_REG32(		 GPMC_BCH_RESULT0_7,    0x6E0002B0,__READ_WRITE );
__IO_REG32(		 GPMC_BCH_RESULT1_7,    0x6E0002B4,__READ_WRITE );
__IO_REG32(		 GPMC_BCH_RESULT2_7,    0x6E0002BC,__READ_WRITE );
__IO_REG32_BIT(GPMC_BCH_RESULT3_7,    0x6E0002B8,__READ_WRITE ,__gpmc_bch_result3_bits);
__IO_REG32_BIT(GPMC_BCH_SWDATA,    		0x6E0002D0,__READ_WRITE ,__gpmc_bch_swdata_bits);

/***************************************************************************
 **
 ** EMIF
 **
 ***************************************************************************/
__IO_REG32_BIT(EMIF_MOD_ID_REV,       			0x6D000000,__READ				,__emif_mod_id_rev_bits);
__IO_REG32_BIT(EMIF_STATUS,       					0x6D000004,__READ				,__emif_status_bits);
__IO_REG32_BIT(EMIF_SDRAM_CONFIG,     			0x6D000008,__READ_WRITE ,__emif_sdram_config_bits);
__IO_REG32_BIT(EMIF_SDRAM_REF_CTRL,   			0x6D000010,__READ_WRITE ,__emif_sdram_ref_ctrl_bits);
__IO_REG32_BIT(EMIF_SDRAM_REF_CTRL_SHDW,		0x6D000014,__READ_WRITE ,__emif_sdram_ref_ctrl_shdw_bits);
__IO_REG32_BIT(EMIF_SDRAM_TIM_1,       			0x6D000018,__READ_WRITE ,__emif_sdram_tim_1_bits);
__IO_REG32_BIT(EMIF_SDRAM_TIM_1_SHDW,     	0x6D00001C,__READ_WRITE ,__emif_sdram_tim_1_shdw_bits);
__IO_REG32_BIT(EMIF_SDRAM_TIM_2,       			0x6D000020,__READ_WRITE ,__emif_sdram_tim_2_bits);
__IO_REG32_BIT(EMIF_SDRAM_TIM_2_SHDW,     	0x6D000024,__READ_WRITE ,__emif_sdram_tim_2_shdw_bits);
__IO_REG32_BIT(EMIF_SDRAM_TIM_3,       			0x6D000028,__READ_WRITE ,__emif_sdram_tim_3_bits);
__IO_REG32_BIT(EMIF_SDRAM_TIM_3_SHDW,     	0x6D00002C,__READ_WRITE ,__emif_sdram_tim_3_shdw_bits);
__IO_REG32_BIT(EMIF_PWR_MGMT_CTRL,       		0x6D000038,__READ_WRITE ,__emif_pwr_mgmt_ctrl_bits);
__IO_REG32_BIT(EMIF_PWR_MGMT_CTRL_SHDW,   	0x6D00003C,__READ_WRITE ,__emif_pwr_mgmt_ctrl_shdw_bits);
__IO_REG32_BIT(EMIF_OCP_CONFIG,       			0x6D000054,__READ_WRITE ,__emif_ocp_config_bits);
__IO_REG32_BIT(EMIF_OCP_CFG_VAL_1,   				0x6D000058,__READ				,__emif_ocp_cfg_val_1_bits);
__IO_REG32_BIT(EMIF_OCP_CFG_VAL_2,       		0x6D00005C,__READ				,__emif_ocp_cfg_val_2_bits);
__IO_REG32_BIT(EMIF_IODFT_TLGC,   					0x6D000060,__READ_WRITE ,__emif_iodft_tlgc_shdw_bits);
__IO_REG32_BIT(EMIF_IODFT_CTRL_MISR_RSLT, 	0x6D000064,__READ				,__emif_iodft_ctrl_misr_rslt_bits);
__IO_REG32_BIT(EMIF_IODFT_ADDR_MISR_RSLT, 	0x6D000068,__READ				,__emif_iodft_addr_misr_rslt_bits);
__IO_REG32(		 EMIF_IODFT_DATA_MISR_RSLT_1, 0x6D00006C,__READ				);
__IO_REG32(		 EMIF_IODFT_DATA_MISR_RSLT_2, 0x6D000070,__READ				);
__IO_REG32_BIT(EMIF_IODFT_DATA_MISR_RSLT_3, 0x6D000074,__READ				,__emif_iodft_data_misr_rslt_3_bits);
__IO_REG32(		 EMIF_PERF_CNT_1, 						0x6D000080,__READ				);
__IO_REG32(		 EMIF_PERF_CNT_2, 						0x6D000084,__READ				);
__IO_REG32_BIT(EMIF_PERF_CNT_CFG, 					0x6D000088,__READ_WRITE ,__emif_perf_cnt_cfg_bits);
__IO_REG32_BIT(EMIF_PERF_CNT_SEL, 					0x6D00008C,__READ_WRITE ,__emif_perf_cnt_sel_bits);
__IO_REG32(		 EMIF_PERF_CNT_TIM, 					0x6D000090,__READ				);
__IO_REG32_BIT(EMIF_IRQ_EOI, 								0x6D0000A0,__READ_WRITE ,__emif_irq_eoi_bits);
__IO_REG32_BIT(EMIF_IRQSTATUS_RAW_SYS, 			0x6D0000A4,__READ_WRITE ,__emif_irqstatus_raw_sys_bits);
__IO_REG32_BIT(EMIF_IRQSTATUS_SYS, 					0x6D0000AC,__READ_WRITE ,__emif_irqstatus_sys_bits);
__IO_REG32_BIT(EMIF_IRQENABLE_SET_SYS, 			0x6D0000B4,__READ_WRITE ,__emif_irqenable_set_sys_bits);
__IO_REG32_BIT(EMIF_IRQENABLE_CLR_SYS, 			0x6D0000BC,__READ_WRITE ,__emif_irqenable_clr_sys_bits);
__IO_REG32_BIT(EMIF_OCP_ERR_LOG, 						0x6D0000D0,__READ				,__emif_ocp_err_log_bits);
__IO_REG32_BIT(EMIF_DDR_PHY_CTRL_1, 				0x6D0000E4,__READ_WRITE ,__emif_ddr_phy_ctrl_1_bits);
__IO_REG32_BIT(EMIF_DDR_PHY_CTRL_1_SHDW, 		0x6D0000E8,__READ_WRITE ,__emif_ddr_phy_ctrl_1_shdw_bits);
__IO_REG32_BIT(EMIF_DDR_PHY_CTRL_2, 				0x6D0000EC,__READ_WRITE ,__emif_ddr_phy_ctrl_2_bits);

/***************************************************************************
 **
 ** SMS
 **
 ***************************************************************************/
__IO_REG32_BIT(SMS_SYSCONFIG,     					0x6C000010,__READ_WRITE ,__sms_sysconfig_bits);
__IO_REG32_BIT(SMS_SYSSTATUS,     					0x6C000014,__READ				,__sms_sysstatus_bits);
__IO_REG32(		 SMS_RG_ATT0,     						0x6C000048,__READ_WRITE );
__IO_REG32_BIT(SMS_RG_RDPERM0,     					0x6C000050,__READ_WRITE ,__sms_rg_rdperm_bits);
__IO_REG32_BIT(SMS_RG_WRPERM0,     					0x6C000058,__READ_WRITE ,__sms_rg_rdperm_bits);
__IO_REG32(		 SMS_RG_START1,     					0x6C000060,__READ_WRITE );
__IO_REG32(	 	 SMS_RG_END1,     						0x6C000064,__READ_WRITE );
__IO_REG32(		 SMS_RG_ATT1,     						0x6C000068,__READ_WRITE );
__IO_REG32_BIT(SMS_RG_RDPERM1,     					0x6C000070,__READ_WRITE ,__sms_rg_rdperm_bits);
__IO_REG32_BIT(SMS_RG_WRPERM1,     					0x6C000078,__READ_WRITE ,__sms_rg_rdperm_bits);
__IO_REG32(		 SMS_RG_START2,     					0x6C000080,__READ_WRITE );
__IO_REG32(		 SMS_RG_END2,     						0x6C000084,__READ_WRITE );
__IO_REG32(		 SMS_RG_ATT2,     						0x6C000088,__READ_WRITE );
__IO_REG32_BIT(SMS_RG_RDPERM2,     					0x6C000090,__READ_WRITE ,__sms_rg_rdperm_bits);
__IO_REG32_BIT(SMS_RG_WRPERM2,     					0x6C000098,__READ_WRITE ,__sms_rg_rdperm_bits);
__IO_REG32(		 SMS_RG_START3,     					0x6C0000A0,__READ_WRITE );
__IO_REG32(		 SMS_RG_END3,     						0x6C0000A4,__READ_WRITE );
__IO_REG32(		 SMS_RG_ATT3,     						0x6C0000A8,__READ_WRITE );
__IO_REG32_BIT(SMS_RG_RDPERM3,     					0x6C0000B0,__READ_WRITE ,__sms_rg_rdperm_bits);
__IO_REG32_BIT(SMS_RG_WRPERM3,     					0x6C0000B8,__READ_WRITE ,__sms_rg_rdperm_bits);
__IO_REG32(		 SMS_RG_START4,     					0x6C0000C0,__READ_WRITE );
__IO_REG32(		 SMS_RG_END4,     						0x6C0000C4,__READ_WRITE );
__IO_REG32(		 SMS_RG_ATT4,     						0x6C0000C8,__READ_WRITE );
__IO_REG32_BIT(SMS_RG_RDPERM4,     					0x6C0000D0,__READ_WRITE ,__sms_rg_rdperm_bits);
__IO_REG32_BIT(SMS_RG_WRPERM4,     					0x6C0000D8,__READ_WRITE ,__sms_rg_rdperm_bits);
__IO_REG32(		 SMS_RG_START5,     					0x6C0000E0,__READ_WRITE );
__IO_REG32(		 SMS_RG_END5,     						0x6C0000E4,__READ_WRITE );
__IO_REG32(		 SMS_RG_ATT5,     						0x6C0000E8,__READ_WRITE );
__IO_REG32_BIT(SMS_RG_RDPERM5,     					0x6C0000F0,__READ_WRITE ,__sms_rg_rdperm_bits);
__IO_REG32_BIT(SMS_RG_WRPERM5,     					0x6C0000F8,__READ_WRITE ,__sms_rg_rdperm_bits);
__IO_REG32(		 SMS_RG_START6,     					0x6C000100,__READ_WRITE );
__IO_REG32(		 SMS_RG_END6,     						0x6C000104,__READ_WRITE );
__IO_REG32(		 SMS_RG_ATT6,     						0x6C000108,__READ_WRITE );
__IO_REG32_BIT(SMS_RG_RDPERM6,     					0x6C000110,__READ_WRITE ,__sms_rg_rdperm_bits);
__IO_REG32_BIT(SMS_RG_WRPERM6,     					0x6C000118,__READ_WRITE ,__sms_rg_rdperm_bits);
__IO_REG32(		 SMS_RG_START7,     					0x6C000120,__READ_WRITE );
__IO_REG32(		 SMS_RG_END7,     						0x6C000124,__READ_WRITE );
__IO_REG32(		 SMS_RG_ATT7,     						0x6C000128,__READ_WRITE );
__IO_REG32_BIT(SMS_RG_RDPERM7,     					0x6C000130,__READ_WRITE ,__sms_rg_rdperm_bits);
__IO_REG32_BIT(SMS_RG_WRPERM7,     					0x6C000138,__READ_WRITE ,__sms_rg_rdperm_bits);
__IO_REG32_BIT(SMS_SECURITY_CONTROL,     		0x6C000140,__READ_WRITE ,__sms_security_control_bits);
__IO_REG32_BIT(SMS_CLASS_ARBITER0,     			0x6C000150,__READ_WRITE ,__sms_class_arbiter0_bits);
__IO_REG32_BIT(SMS_CLASS_ARBITER1,     			0x6C000154,__READ_WRITE ,__sms_class_arbiter1_bits);
__IO_REG32_BIT(SMS_CLASS_ARBITER2,     			0x6C000158,__READ_WRITE ,__sms_class_arbiter2_bits);
__IO_REG32_BIT(SMS_INTERCLASS_ARBITER,     	0x6C000160,__READ_WRITE ,__sms_interclass_arbiter_bits);
__IO_REG32_BIT(SMS_CLASS_ROTATION0,     		0x6C000164,__READ_WRITE ,__sms_class_rotation_bits);
__IO_REG32_BIT(SMS_CLASS_ROTATION1,     		0x6C000168,__READ_WRITE ,__sms_class_rotation_bits);
__IO_REG32_BIT(SMS_CLASS_ROTATION2,     		0x6C00016C,__READ_WRITE ,__sms_class_rotation_bits);
__IO_REG32(		 SMS_ERR_ADDR,     						0x6C000170,__READ				);
__IO_REG32_BIT(SMS_ERR_TYPE,     						0x6C000174,__READ_WRITE ,__sms_err_type_bits);
__IO_REG32_BIT(SMS_POW_CTRL,     						0x6C000178,__READ_WRITE ,__sms_pow_ctrl_bits);
__IO_REG32_BIT(SMS_ROT_CONTROL0,     				0x6C000180,__READ_WRITE ,__sms_rot_control_bits);
__IO_REG32_BIT(SMS_ROT_SIZE0,     					0x6C000184,__READ_WRITE ,__sms_rot_size_bits);
__IO_REG32_BIT(SMS_ROT_PHYSICAL_BA0,     		0x6C000188,__READ_WRITE ,__sms_rot_physical_ba_bits);
__IO_REG32_BIT(SMS_ROT_CONTROL1,     				0x6C000190,__READ_WRITE ,__sms_rot_control_bits);
__IO_REG32_BIT(SMS_ROT_SIZE1,     					0x6C000194,__READ_WRITE ,__sms_rot_size_bits);
__IO_REG32_BIT(SMS_ROT_PHYSICAL_BA1,     		0x6C000198,__READ_WRITE ,__sms_rot_physical_ba_bits);
__IO_REG32_BIT(SMS_ROT_CONTROL2,     				0x6C0001A0,__READ_WRITE ,__sms_rot_control_bits);
__IO_REG32_BIT(SMS_ROT_SIZE2,     					0x6C0001A4,__READ_WRITE ,__sms_rot_size_bits);
__IO_REG32_BIT(SMS_ROT_PHYSICAL_BA2,     		0x6C0001A8,__READ_WRITE ,__sms_rot_physical_ba_bits);
__IO_REG32_BIT(SMS_ROT_CONTROL3,     				0x6C0001B0,__READ_WRITE ,__sms_rot_control_bits);
__IO_REG32_BIT(SMS_ROT_SIZE3,     					0x6C0001B4,__READ_WRITE ,__sms_rot_size_bits);
__IO_REG32_BIT(SMS_ROT_PHYSICAL_BA3,     		0x6C0001B8,__READ_WRITE ,__sms_rot_physical_ba_bits);
__IO_REG32_BIT(SMS_ROT_CONTROL4,     				0x6C0001C0,__READ_WRITE ,__sms_rot_control_bits);
__IO_REG32_BIT(SMS_ROT_SIZE4,     					0x6C0001C4,__READ_WRITE ,__sms_rot_size_bits);
__IO_REG32_BIT(SMS_ROT_PHYSICAL_BA4,     		0x6C0001C8,__READ_WRITE ,__sms_rot_physical_ba_bits);
__IO_REG32_BIT(SMS_ROT_CONTROL5,     				0x6C0001D0,__READ_WRITE ,__sms_rot_control_bits);
__IO_REG32_BIT(SMS_ROT_SIZE5,     					0x6C0001D4,__READ_WRITE ,__sms_rot_size_bits);
__IO_REG32_BIT(SMS_ROT_PHYSICAL_BA5,     		0x6C0001D8,__READ_WRITE ,__sms_rot_physical_ba_bits);
__IO_REG32_BIT(SMS_ROT_CONTROL6,     				0x6C0001E0,__READ_WRITE ,__sms_rot_control_bits);
__IO_REG32_BIT(SMS_ROT_SIZE6,     					0x6C0001E4,__READ_WRITE ,__sms_rot_size_bits);
__IO_REG32_BIT(SMS_ROT_PHYSICAL_BA6,     		0x6C0001E8,__READ_WRITE ,__sms_rot_physical_ba_bits);
__IO_REG32_BIT(SMS_ROT_CONTROL7,     				0x6C0001F0,__READ_WRITE ,__sms_rot_control_bits);
__IO_REG32_BIT(SMS_ROT_SIZE7,     					0x6C0001F4,__READ_WRITE ,__sms_rot_size_bits);
__IO_REG32_BIT(SMS_ROT_PHYSICAL_BA7,     		0x6C0001F8,__READ_WRITE ,__sms_rot_physical_ba_bits);
__IO_REG32_BIT(SMS_ROT_CONTROL8,     				0x6C000200,__READ_WRITE ,__sms_rot_control_bits);
__IO_REG32_BIT(SMS_ROT_SIZE8,     					0x6C000204,__READ_WRITE ,__sms_rot_size_bits);
__IO_REG32_BIT(SMS_ROT_PHYSICAL_BA8,     		0x6C000208,__READ_WRITE ,__sms_rot_physical_ba_bits);
__IO_REG32_BIT(SMS_ROT_CONTROL9,     				0x6C000210,__READ_WRITE ,__sms_rot_control_bits);
__IO_REG32_BIT(SMS_ROT_SIZE9,     					0x6C000214,__READ_WRITE ,__sms_rot_size_bits);
__IO_REG32_BIT(SMS_ROT_PHYSICAL_BA9,     		0x6C000218,__READ_WRITE ,__sms_rot_physical_ba_bits);
__IO_REG32_BIT(SMS_ROT_CONTROL10,    				0x6C000220,__READ_WRITE ,__sms_rot_control_bits);
__IO_REG32_BIT(SMS_ROT_SIZE10,     					0x6C000224,__READ_WRITE ,__sms_rot_size_bits);
__IO_REG32_BIT(SMS_ROT_PHYSICAL_BA10,    		0x6C000228,__READ_WRITE ,__sms_rot_physical_ba_bits);
__IO_REG32_BIT(SMS_ROT_CONTROL11,    				0x6C000230,__READ_WRITE ,__sms_rot_control_bits);
__IO_REG32_BIT(SMS_ROT_SIZE11,    					0x6C000234,__READ_WRITE ,__sms_rot_size_bits);
__IO_REG32_BIT(SMS_ROT_PHYSICAL_BA11,    		0x6C000238,__READ_WRITE ,__sms_rot_physical_ba_bits);

/***************************************************************************
 **
 ** VPFE
 **
 ***************************************************************************/
__IO_REG32(		 VPFE_PID,     								0x5C060000,__READ				);
__IO_REG32_BIT(VPFE_PCR,     								0x5C060004,__READ_WRITE ,__vpfe_pcr_bits);
__IO_REG32_BIT(VPFE_SYN_MODE,     					0x5C060008,__READ_WRITE ,__vpfe_syn_mode_bits);
__IO_REG32_BIT(VPFE_HORZ_INFO,     					0x5C060014,__READ_WRITE ,__vpfe_horz_info_bits);
__IO_REG32_BIT(VPFE_VERT_START,     				0x5C060018,__READ_WRITE ,__vpfe_vert_start_bits);
__IO_REG32_BIT(VPFE_VERT_LINES,     				0x5C06001C,__READ_WRITE ,__vpfe_vert_lines_bits);
__IO_REG32_BIT(VPFE_CULLING,     						0x5C060020,__READ_WRITE ,__vpfe_culling_bits);
__IO_REG32_BIT(VPFE_HSIZE_OFF,     					0x5C060024,__READ_WRITE ,__vpfe_hsize_off_bits);
__IO_REG32_BIT(VPFE_SDOFST,     						0x5C060028,__READ_WRITE ,__vpfe_sdofst_bits);
__IO_REG32(		 VPFE_SDR_ADDR,     					0x5C06002C,__READ_WRITE );
__IO_REG32_BIT(VPFE_CLAMP,     							0x5C060030,__READ_WRITE ,__vpfe_clamp_bits);
__IO_REG32_BIT(VPFE_DCSUB,     							0x5C060034,__READ_WRITE ,__vpfe_dcsub_bits);
__IO_REG32_BIT(VPFE_COLPTN,     						0x5C060038,__READ_WRITE ,__vpfe_colptn_bits);
__IO_REG32_BIT(VPFE_BLKCMP,     						0x5C06003C,__READ_WRITE ,__vpfe_blkcmp_bits);
__IO_REG32_BIT(VPFE_VDINT,     							0x5C060048,__READ_WRITE ,__vpfe_vdint_bits);
__IO_REG32_BIT(VPFE_ALAW,     							0x5C06004C,__READ_WRITE ,__vpfe_alaw_bits);
__IO_REG32_BIT(VPFE_REC656IF,     					0x5C060050,__READ_WRITE ,__vpfe_rec656if_bits);
__IO_REG32_BIT(VPFE_CCDCCFG,     						0x5C060054,__READ_WRITE ,__vpfe_ccdccfg_bits);
__IO_REG32_BIT(VPFE_DMA_CNTL,     					0x5C060098,__READ_WRITE ,__vpfe_dma_cntl_bits);

/***************************************************************************
 **
 ** DSS
 **
 ***************************************************************************/
__IO_REG32_BIT(DSS_REVISIONNUMBER,     			0x48050000,__READ				,__dss_revisionnumber_bits);
__IO_REG32_BIT(DSS_SYSCONFIG,     					0x48050010,__READ_WRITE ,__dss_sysconfig_bits);
__IO_REG32_BIT(DSS_SYSSTATUS,     					0x48050014,__READ				,__dss_sysstatus_bits);
__IO_REG32_BIT(DSS_IRQSTATUS,     					0x48050018,__READ				,__dss_irqstatus_bits);
__IO_REG32_BIT(DSS_CONTROL,     						0x48050040,__READ_WRITE ,__dss_control_bits);
__IO_REG32_BIT(DSS_SDI_CONTROL,     				0x48050044,__READ_WRITE ,__dss_sdi_control_bits);
__IO_REG32_BIT(DSS_PLL_CONTROL,     				0x48050048,__READ_WRITE ,__dss_pll_control_bits);
__IO_REG32_BIT(DSS_SDI_STATUS,     					0x4805005C,__READ				,__dss_sdi_status_bits);

/***************************************************************************
 **
 ** DISPC
 **
 ***************************************************************************/
__IO_REG32_BIT(DISPC_REVISION,     					0x48050400,__READ				,__dispc_revision_bits);
__IO_REG32_BIT(DISPC_SYSCONFIG,     				0x48050410,__READ_WRITE ,__dispc_sysconfig_bits);
__IO_REG32_BIT(DISPC_SYSSTATUS,     				0x48050414,__READ				,__dispc_sysstatus_bits);
__IO_REG32_BIT(DISPC_IRQSTATUS,     				0x48050418,__READ_WRITE ,__dispc_irqstatus_bits);
__IO_REG32_BIT(DISPC_IRQENABLE,     				0x4805041C,__READ_WRITE ,__dispc_irqstatus_bits);
__IO_REG32_BIT(DISPC_CONTROL,     					0x48050440,__READ_WRITE ,__dispc_control_bits);
__IO_REG32_BIT(DISPC_CONFIG,     						0x48050444,__READ_WRITE ,__dispc_config_bits);
__IO_REG32_BIT(DISPC_DEFAULT_COLOR_0,     	0x4805044C,__READ_WRITE ,__dispc_default_color_bits);
__IO_REG32_BIT(DISPC_DEFAULT_COLOR_1,     	0x48050450,__READ_WRITE ,__dispc_default_color_bits);
__IO_REG32_BIT(DISPC_TRANS_COLOR_0,     		0x48050454,__READ_WRITE ,__dispc_trans_color_bits);
__IO_REG32_BIT(DISPC_TRANS_COLOR_1,     		0x48050458,__READ_WRITE ,__dispc_trans_color_bits);
__IO_REG32_BIT(DISPC_LINE_STATUS,     			0x4805045C,__READ				,__dispc_line_status_bits);
__IO_REG32_BIT(DISPC_LINE_NUMBER,     			0x48050460,__READ_WRITE ,__dispc_line_status_bits);
__IO_REG32_BIT(DISPC_TIMING_H,     					0x48050464,__READ_WRITE ,__dispc_timing_h_bits);
__IO_REG32_BIT(DISPC_TIMING_V,     					0x48050468,__READ_WRITE ,__dispc_timing_v_bits);
__IO_REG32_BIT(DISPC_POL_FREQ,     					0x4805046C,__READ_WRITE ,__dispc_pol_freq_bits);
__IO_REG32_BIT(DISPC_DIVISOR,     					0x48050470,__READ_WRITE ,__dispc_divisor_bits);
__IO_REG32_BIT(DISPC_GLOBAL_ALPHA,     			0x48050474,__READ_WRITE ,__dispc_global_alpha_bits);
__IO_REG32_BIT(DISPC_SIZE_DIG,     					0x48050478,__READ_WRITE ,__dispc_size_dig_bits);
__IO_REG32_BIT(DISPC_SIZE_LCD,     					0x4805047C,__READ_WRITE ,__dispc_size_dig_bits);
__IO_REG32(		 DISPC_GFX_BA0,     					0x48050480,__READ_WRITE );
__IO_REG32(		 DISPC_GFX_BA1,     					0x48050484,__READ_WRITE );
__IO_REG32_BIT(DISPC_GFX_POSITION,     			0x48050488,__READ_WRITE ,__dispc_gfx_position_bits);
__IO_REG32_BIT(DISPC_GFX_SIZE,     					0x4805048C,__READ_WRITE ,__dispc_gfx_size_bits);
__IO_REG32_BIT(DISPC_GFX_ATTRIBUTES,     		0x480504A0,__READ_WRITE ,__dispc_gfx_attributes_bits);
__IO_REG32_BIT(DISPC_GFX_FIFO_THRESHOLD,    0x480504A4,__READ_WRITE ,__dispc_gfx_fifo_threshold_bits);
__IO_REG32_BIT(DISPC_GFX_FIFO_SIZE_STATUS,  0x480504A8,__READ				,__dispc_gfx_fifo_size_status_bits);
__IO_REG32(		 DISPC_GFX_ROW_INC,    				0x480504AC,__READ_WRITE );
__IO_REG32_BIT(DISPC_GFX_PIXEL_INC,    			0x480504B0,__READ_WRITE ,__dispc_gfx_pixel_inc_bits);
__IO_REG32(		 DISPC_GFX_WINDOW_SKIP,    		0x480504B4,__READ_WRITE );
__IO_REG32(		 DISPC_GFX_TABLE_BA,    			0x480504B8,__READ_WRITE );
__IO_REG32_BIT(DISPC_DATA_CYCLE0,    				0x480505D4,__READ_WRITE ,__dispc_data_cycle_bits);
__IO_REG32_BIT(DISPC_DATA_CYCLE1,    				0x480505D8,__READ_WRITE ,__dispc_data_cycle_bits);
__IO_REG32_BIT(DISPC_DATA_CYCLE2,    				0x480505DC,__READ_WRITE ,__dispc_data_cycle_bits);
__IO_REG32_BIT(DISPC_CPR_COEF_R,    				0x48050620,__READ_WRITE ,__dispc_cpr_coef_r_bits);
__IO_REG32_BIT(DISPC_CPR_COEF_G,    				0x48050624,__READ_WRITE ,__dispc_cpr_coef_g_bits);
__IO_REG32_BIT(DISPC_CPR_COEF_B,    				0x48050628,__READ_WRITE ,__dispc_cpr_coef_b_bits);
__IO_REG32_BIT(DISPC_GFX_PRELOAD,    				0x4805062C,__READ_WRITE ,__dispc_gfx_preload_bits);

/***************************************************************************
 **
 ** DISPC VID1
 **
 ***************************************************************************/
__IO_REG32(		 DISPC_VID1_BA0,    					0x480504BC,__READ_WRITE );
__IO_REG32(		 DISPC_VID1_BA1,    					0x480504C0,__READ_WRITE );
__IO_REG32_BIT(DISPC_VID1_POSITION0,   			0x480504C4,__READ_WRITE ,__dispc_vidn_position_bits);
__IO_REG32_BIT(DISPC_VID1_SIZE,    					0x480504C8,__READ_WRITE ,__dispc_vidn_size_bits);
__IO_REG32_BIT(DISPC_VID1_ATTRIBUTES,    		0x480504CC,__READ_WRITE ,__dispc_vidn_attributes_bits);
__IO_REG32_BIT(DISPC_VID1_FIFO_THRESHOLD,   0x480504D0,__READ_WRITE ,__dispc_vidn_fifo_threshold_bits);
__IO_REG32_BIT(DISPC_VIDn_FIFO_SIZE_STATUS, 0x480504D4,__READ				,__dispc_vidn_fifo_size_status_bits);
__IO_REG32(		 DISPC_VID1_ROW_INC,    			0x480504D8,__READ_WRITE );
__IO_REG32_BIT(DISPC_VID1_PIXEL_INC,    		0x480504DC,__READ_WRITE ,__dispc_vidn_pixel_inc_bits);
__IO_REG32_BIT(DISPC_VID1_FIR,    					0x480504E0,__READ_WRITE ,__dispc_vidn_fir_bits);
__IO_REG32_BIT(DISPC_VID1_PICTURE_SIZE,    	0x480504E4,__READ_WRITE ,__dispc_vidn_picture_size_bits);
__IO_REG32_BIT(DISPC_VID1_ACCU0,    				0x480504E8,__READ_WRITE ,__dispc_vidn_accu_bits);
__IO_REG32_BIT(DISPC_VID1_ACCU1,    				0x480504EC,__READ_WRITE ,__dispc_vidn_accu_bits);
__IO_REG32_BIT(DISPC_VID1_FIR_COEF_H0,    	0x480504F0,__READ_WRITE ,__dispc_vidn_fir_coef_h_bits);
__IO_REG32_BIT(DISPC_VID1_FIR_COEF_HV0,    	0x480504F4,__READ_WRITE ,__dispc_vidn_fir_coef_hv_bits);
__IO_REG32_BIT(DISPC_VID1_FIR_COEF_H1,    	0x480504F8,__READ_WRITE ,__dispc_vidn_fir_coef_h_bits);
__IO_REG32_BIT(DISPC_VID1_FIR_COEF_HV1,    	0x480504FC,__READ_WRITE ,__dispc_vidn_fir_coef_hv_bits);
__IO_REG32_BIT(DISPC_VID1_FIR_COEF_H2,    	0x48050500,__READ_WRITE ,__dispc_vidn_fir_coef_h_bits);
__IO_REG32_BIT(DISPC_VID1_FIR_COEF_HV2,    	0x48050504,__READ_WRITE ,__dispc_vidn_fir_coef_hv_bits);
__IO_REG32_BIT(DISPC_VID1_FIR_COEF_H3,    	0x48050508,__READ_WRITE ,__dispc_vidn_fir_coef_h_bits);
__IO_REG32_BIT(DISPC_VID1_FIR_COEF_HV3,    	0x4805050C,__READ_WRITE ,__dispc_vidn_fir_coef_hv_bits);
__IO_REG32_BIT(DISPC_VID1_FIR_COEF_H4,    	0x48050510,__READ_WRITE ,__dispc_vidn_fir_coef_h_bits);
__IO_REG32_BIT(DISPC_VID1_FIR_COEF_HV4,    	0x48050514,__READ_WRITE ,__dispc_vidn_fir_coef_hv_bits);
__IO_REG32_BIT(DISPC_VID1_FIR_COEF_H5,    	0x48050518,__READ_WRITE ,__dispc_vidn_fir_coef_h_bits);
__IO_REG32_BIT(DISPC_VID1_FIR_COEF_HV5,    	0x4805051C,__READ_WRITE ,__dispc_vidn_fir_coef_hv_bits);
__IO_REG32_BIT(DISPC_VID1_FIR_COEF_H6,    	0x48050520,__READ_WRITE ,__dispc_vidn_fir_coef_h_bits);
__IO_REG32_BIT(DISPC_VID1_FIR_COEF_HV6,    	0x48050524,__READ_WRITE ,__dispc_vidn_fir_coef_hv_bits);
__IO_REG32_BIT(DISPC_VID1_FIR_COEF_H7,    	0x48050528,__READ_WRITE ,__dispc_vidn_fir_coef_h_bits);
__IO_REG32_BIT(DISPC_VID1_FIR_COEF_HV7,    	0x4805052C,__READ_WRITE ,__dispc_vidn_fir_coef_hv_bits);
__IO_REG32_BIT(DISPC_VID1_CONV_COEF0,    		0x48050530,__READ_WRITE ,__dispc_vidn_conv_coef0_bits);
__IO_REG32_BIT(DISPC_VID1_CONV_COEF1,    		0x48050534,__READ_WRITE ,__dispc_vidn_conv_coef1_bits);
__IO_REG32_BIT(DISPC_VID1_CONV_COEF2,    		0x48050538,__READ_WRITE ,__dispc_vidn_conv_coef2_bits);
__IO_REG32_BIT(DISPC_VID1_CONV_COEF3,    		0x4805053C,__READ_WRITE ,__dispc_vidn_conv_coef3_bits);
__IO_REG32_BIT(DISPC_VID1_CONV_COEF4,    		0x48050540,__READ_WRITE ,__dispc_vidn_conv_coef4_bits);
__IO_REG32_BIT(DISPC_VID1_FIR_COEF_V0,    	0x480505E0,__READ_WRITE ,__dispc_vidn_fir_coef_v_bits);
__IO_REG32_BIT(DISPC_VID1_FIR_COEF_V1,    	0x480505E4,__READ_WRITE ,__dispc_vidn_fir_coef_v_bits);
__IO_REG32_BIT(DISPC_VID1_FIR_COEF_V2,    	0x480505E8,__READ_WRITE ,__dispc_vidn_fir_coef_v_bits);
__IO_REG32_BIT(DISPC_VID1_FIR_COEF_V3,    	0x480505EC,__READ_WRITE ,__dispc_vidn_fir_coef_v_bits);
__IO_REG32_BIT(DISPC_VID1_FIR_COEF_V4,    	0x480505F0,__READ_WRITE ,__dispc_vidn_fir_coef_v_bits);
__IO_REG32_BIT(DISPC_VID1_FIR_COEF_V5,    	0x480505F4,__READ_WRITE ,__dispc_vidn_fir_coef_v_bits);
__IO_REG32_BIT(DISPC_VID1_FIR_COEF_V6,    	0x480505F8,__READ_WRITE ,__dispc_vidn_fir_coef_v_bits);
__IO_REG32_BIT(DISPC_VID1_FIR_COEF_V7,    	0x480505FC,__READ_WRITE ,__dispc_vidn_fir_coef_v_bits);
__IO_REG32_BIT(DISPC_VID1_PRELOAD,    			0x48050630,__READ_WRITE ,__dispc_vidn_preload_bits);

/***************************************************************************
 **
 ** DISPC VID2
 **
 ***************************************************************************/
__IO_REG32(		 DISPC_VID2_BA0,    					0x4805054C,__READ_WRITE );
__IO_REG32(		 DISPC_VID2_BA1,    					0x48050550,__READ_WRITE );
__IO_REG32_BIT(DISPC_VID2_POSITION0,   			0x48050554,__READ_WRITE ,__dispc_vidn_position_bits);
__IO_REG32_BIT(DISPC_VID2_SIZE,    					0x48050558,__READ_WRITE ,__dispc_vidn_size_bits);
__IO_REG32_BIT(DISPC_VID2_ATTRIBUTES,    		0x4805055C,__READ_WRITE ,__dispc_vidn_attributes_bits);
__IO_REG32_BIT(DISPC_VID2_FIFO_THRESHOLD,   0x48050560,__READ_WRITE ,__dispc_vidn_fifo_threshold_bits);
__IO_REG32_BIT(DISPC_VID2_FIFO_SIZE_STATUS, 0x48050564,__READ				,__dispc_vidn_fifo_size_status_bits);
__IO_REG32(		 DISPC_VID2_ROW_INC,    			0x48050568,__READ_WRITE );
__IO_REG32_BIT(DISPC_VID2_PIXEL_INC,    		0x4805056C,__READ_WRITE ,__dispc_vidn_pixel_inc_bits);
__IO_REG32_BIT(DISPC_VID2_FIR,    					0x48050570,__READ_WRITE ,__dispc_vidn_fir_bits);
__IO_REG32_BIT(DISPC_VID2_PICTURE_SIZE,    	0x48050574,__READ_WRITE ,__dispc_vidn_picture_size_bits);
__IO_REG32_BIT(DISPC_VID2_ACCU0,    				0x48050578,__READ_WRITE ,__dispc_vidn_accu_bits);
__IO_REG32_BIT(DISPC_VID2_ACCU1,    				0x4805057C,__READ_WRITE ,__dispc_vidn_accu_bits);
__IO_REG32_BIT(DISPC_VID2_FIR_COEF_H0,    	0x48050580,__READ_WRITE ,__dispc_vidn_fir_coef_h_bits);
__IO_REG32_BIT(DISPC_VID2_FIR_COEF_HV0,    	0x48050584,__READ_WRITE ,__dispc_vidn_fir_coef_hv_bits);
__IO_REG32_BIT(DISPC_VID2_FIR_COEF_H1,    	0x48050588,__READ_WRITE ,__dispc_vidn_fir_coef_h_bits);
__IO_REG32_BIT(DISPC_VID2_FIR_COEF_HV1,    	0x4805058C,__READ_WRITE ,__dispc_vidn_fir_coef_hv_bits);
__IO_REG32_BIT(DISPC_VID2_FIR_COEF_H2,    	0x48050590,__READ_WRITE ,__dispc_vidn_fir_coef_h_bits);
__IO_REG32_BIT(DISPC_VID2_FIR_COEF_HV2,    	0x48050594,__READ_WRITE ,__dispc_vidn_fir_coef_hv_bits);
__IO_REG32_BIT(DISPC_VID2_FIR_COEF_H3,    	0x48050598,__READ_WRITE ,__dispc_vidn_fir_coef_h_bits);
__IO_REG32_BIT(DISPC_VID2_FIR_COEF_HV3,    	0x4805059C,__READ_WRITE ,__dispc_vidn_fir_coef_hv_bits);
__IO_REG32_BIT(DISPC_VID2_FIR_COEF_H4,    	0x480505A0,__READ_WRITE ,__dispc_vidn_fir_coef_h_bits);
__IO_REG32_BIT(DISPC_VID2_FIR_COEF_HV4,    	0x480505A4,__READ_WRITE ,__dispc_vidn_fir_coef_hv_bits);
__IO_REG32_BIT(DISPC_VID2_FIR_COEF_H5,    	0x480505A8,__READ_WRITE ,__dispc_vidn_fir_coef_h_bits);
__IO_REG32_BIT(DISPC_VID2_FIR_COEF_HV5,    	0x480505AC,__READ_WRITE ,__dispc_vidn_fir_coef_hv_bits);
__IO_REG32_BIT(DISPC_VID2_FIR_COEF_H6,    	0x480505B0,__READ_WRITE ,__dispc_vidn_fir_coef_h_bits);
__IO_REG32_BIT(DISPC_VID2_FIR_COEF_HV6,    	0x480505B4,__READ_WRITE ,__dispc_vidn_fir_coef_hv_bits);
__IO_REG32_BIT(DISPC_VID2_FIR_COEF_H7,    	0x480505B8,__READ_WRITE ,__dispc_vidn_fir_coef_h_bits);
__IO_REG32_BIT(DISPC_VID2_FIR_COEF_HV7,    	0x480505BC,__READ_WRITE ,__dispc_vidn_fir_coef_hv_bits);
__IO_REG32_BIT(DISPC_VID2_CONV_COEF0,    		0x480505C0,__READ_WRITE ,__dispc_vidn_conv_coef0_bits);
__IO_REG32_BIT(DISPC_VID2_CONV_COEF1,    		0x480505C4,__READ_WRITE ,__dispc_vidn_conv_coef1_bits);
__IO_REG32_BIT(DISPC_VID2_CONV_COEF2,    		0x480505C8,__READ_WRITE ,__dispc_vidn_conv_coef2_bits);
__IO_REG32_BIT(DISPC_VID2_CONV_COEF3,    		0x480505CC,__READ_WRITE ,__dispc_vidn_conv_coef3_bits);
__IO_REG32_BIT(DISPC_VID2_CONV_COEF4,    		0x480505D0,__READ_WRITE ,__dispc_vidn_conv_coef4_bits);
__IO_REG32_BIT(DISPC_VID2_FIR_COEF_V0,    	0x48050670,__READ_WRITE ,__dispc_vidn_fir_coef_v_bits);
__IO_REG32_BIT(DISPC_VID2_FIR_COEF_V1,    	0x48050674,__READ_WRITE ,__dispc_vidn_fir_coef_v_bits);
__IO_REG32_BIT(DISPC_VID2_FIR_COEF_V2,    	0x48050678,__READ_WRITE ,__dispc_vidn_fir_coef_v_bits);
__IO_REG32_BIT(DISPC_VID2_FIR_COEF_V3,    	0x4805067C,__READ_WRITE ,__dispc_vidn_fir_coef_v_bits);
__IO_REG32_BIT(DISPC_VID2_FIR_COEF_V4,    	0x48050680,__READ_WRITE ,__dispc_vidn_fir_coef_v_bits);
__IO_REG32_BIT(DISPC_VID2_FIR_COEF_V5,    	0x48050684,__READ_WRITE ,__dispc_vidn_fir_coef_v_bits);
__IO_REG32_BIT(DISPC_VID2_FIR_COEF_V6,    	0x48050688,__READ_WRITE ,__dispc_vidn_fir_coef_v_bits);
__IO_REG32_BIT(DISPC_VID2_FIR_COEF_V7,    	0x4805068C,__READ_WRITE ,__dispc_vidn_fir_coef_v_bits);
__IO_REG32_BIT(DISPC_VID2_PRELOAD,    			0x48050634,__READ_WRITE ,__dispc_vidn_preload_bits);

/***************************************************************************
 **
 ** RFBI
 **
 ***************************************************************************/
__IO_REG32_BIT(RFBI_REVISION,    						0x48050800,__READ				,__rfbi_revision_bits);
__IO_REG32_BIT(RFBI_SYSCONFIG,    					0x48050810,__READ_WRITE ,__rfbi_sysconfig_bits);
__IO_REG32_BIT(RFBI_SYSSTATUS,    					0x48050814,__READ				,__rfbi_sysstatus_bits);
__IO_REG32_BIT(RFBI_CONTROL,    						0x48050840,__READ_WRITE ,__rfbi_control_bits);
__IO_REG32(		 RFBI_PIXEL_CNT,    					0x48050844,__READ_WRITE );
__IO_REG32_BIT(RFBI_LINE_NUMBER,    				0x48050848,__READ_WRITE ,__rfbi_line_number_bits);
__IO_REG32_BIT(RFBI_CMD,    								0x4805084C,__WRITE 			,__rfbi_cmd_bits);
__IO_REG32_BIT(RFBI_PARAM,    							0x48050850,__WRITE 			,__rfbi_param_bits);
__IO_REG32(		 RFBI_DATA,    								0x48050854,__WRITE 			);
__IO_REG32_BIT(RFBI_READ,    								0x48050858,__READ_WRITE ,__rfbi_read_bits);
__IO_REG32_BIT(RFBI_STATUS,    							0x4805085C,__READ_WRITE ,__rfbi_status_bits);
__IO_REG32_BIT(RFBI_CONFIG0,    						0x48050860,__READ_WRITE ,__rfbi_config_bits);
__IO_REG32_BIT(RFBI_ONOFF_TIME0,    				0x48050864,__READ_WRITE ,__rfbi_onoff_time_bits);
__IO_REG32_BIT(RFBI_CYCLE_TIME0,   					0x48050868,__READ_WRITE ,__rfbi_cycle_time_bits);
__IO_REG32_BIT(RFBI_DATA_CYCLE1_0,    			0x4805086C,__READ_WRITE ,__rfbi_data_cycle_bits);
__IO_REG32_BIT(RFBI_DATA_CYCLE2_0,    			0x48050870,__READ_WRITE ,__rfbi_data_cycle_bits);
__IO_REG32_BIT(RFBI_DATA_CYCLE3_0,    			0x48050874,__READ_WRITE ,__rfbi_data_cycle_bits);
__IO_REG32_BIT(RFBI_CONFIG1,    						0x48050878,__READ_WRITE ,__rfbi_config_bits);
__IO_REG32_BIT(RFBI_ONOFF_TIME1,    				0x4805087C,__READ_WRITE ,__rfbi_onoff_time_bits);
__IO_REG32_BIT(RFBI_CYCLE_TIME1,   					0x48050880,__READ_WRITE ,__rfbi_cycle_time_bits);
__IO_REG32_BIT(RFBI_DATA_CYCLE1_1,    			0x48050884,__READ_WRITE ,__rfbi_data_cycle_bits);
__IO_REG32_BIT(RFBI_DATA_CYCLE2_1,    			0x48050888,__READ_WRITE ,__rfbi_data_cycle_bits);
__IO_REG32_BIT(RFBI_DATA_CYCLE3_1,    			0x4805088C,__READ_WRITE ,__rfbi_data_cycle_bits);
__IO_REG32_BIT(RFBI_VSYNC_WIDTH,    				0x48050890,__READ_WRITE ,__rfbi_vsync_width_bits);
__IO_REG32_BIT(RFBI_HSYNC_WIDTH,    				0x48050894,__READ_WRITE ,__rfbi_hsync_width_bits);

/***************************************************************************
 **
 ** VENC
 **
 ***************************************************************************/
__IO_REG32_BIT(VENC_REV_ID,    							0x48050C00,__READ				,__venc_rev_id_bits);
__IO_REG32_BIT(VENC_STATUS,    							0x48050C04,__READ				,__venc_status_bits);
__IO_REG32_BIT(VENC_F_CONTROL,    					0x48050C08,__READ_WRITE ,__venc_f_control_bits);
__IO_REG32_BIT(VENC_VIDOUT_CTRL,    				0x48050C10,__READ_WRITE ,__venc_vidout_ctrl_bits);
__IO_REG32_BIT(VENC_SYNC_CTRL,    					0x48050C14,__READ_WRITE ,__venc_sync_ctrl_bits);
__IO_REG32_BIT(VENC_LLEN,    								0x48050C1C,__READ_WRITE ,__venc_llen_bits);
__IO_REG32_BIT(VENC_FLENS,    							0x48050C20,__READ_WRITE ,__venc_flens_bits);
__IO_REG32_BIT(VENC_HFLTR_CTRL,    					0x48050C24,__READ_WRITE ,__venc_hfltr_ctrl_bits);
__IO_REG32_BIT(VENC_CC_CARR_WSS_CARR,    		0x48050C28,__READ_WRITE ,__venc_cc_carr_wss_carr_bits);
__IO_REG32_BIT(VENC_C_PHASE,    						0x48050C2C,__READ_WRITE ,__venc_c_phase_bits);
__IO_REG32_BIT(VENC_GAIN_U,    							0x48050C30,__READ_WRITE ,__venc_gain_u_bits);
__IO_REG32_BIT(VENC_GAIN_V,    							0x48050C34,__READ_WRITE ,__venc_gain_v_bits);
__IO_REG32_BIT(VENC_GAIN_Y,    							0x48050C38,__READ_WRITE ,__venc_gain_y_bits);
__IO_REG32_BIT(VENC_BLACK_LEVEL,    				0x48050C3C,__READ_WRITE ,__venc_black_level_bits);
__IO_REG32_BIT(VENC_BLANK_LEVEL,    				0x48050C40,__READ_WRITE ,__venc_blank_level_bits);
__IO_REG32_BIT(VENC_X_COLOR,    						0x48050C44,__READ_WRITE ,__venc_x_color_bits);
__IO_REG32_BIT(VENC_M_CONTROL,    					0x48050C48,__READ_WRITE ,__venc_m_control_bits);
__IO_REG32_BIT(VENC_BSTAMP_WSS_DATA,    		0x48050C4C,__READ_WRITE ,__venc_bstamp_wss_data_bits);
__IO_REG32(		 VENC_S_CARR,    							0x48050C50,__READ_WRITE );
__IO_REG32_BIT(VENC_LINE21,    							0x48050C54,__READ_WRITE ,__venc_line21_bits);
__IO_REG32_BIT(VENC_LN_SEL,    							0x48050C58,__READ_WRITE ,__venc_ln_sel_bits);
__IO_REG32_BIT(VENC_L21_WC_CTL,    					0x48050C5C,__READ_WRITE ,__venc_l21_wc_ctl_bits);
__IO_REG32_BIT(VENC_HTRIGGER_VTRIGGER,    	0x48050C60,__READ_WRITE ,__venc_htrigger_vtrigger_bits);
__IO_REG32_BIT(VENC_SAVID_EAVID,    				0x48050C64,__READ_WRITE ,__venc_savid_eavid_bits);
__IO_REG32_BIT(VENC_FLEN_FAL,    						0x48050C68,__READ_WRITE ,__venc_flen_fal_bits);
__IO_REG32_BIT(VENC_LAL_PHASE_RESET,    		0x48050C6C,__READ_WRITE ,__venc_lal_phase_reset_bits);
__IO_REG32_BIT(VENC_HS_INT_START_STOP_X,    0x48050C70,__READ_WRITE ,__venc_hs_int_start_stop_x_bits);
__IO_REG32_BIT(VENC_HS_EXT_START_STOP_X,    0x48050C74,__READ_WRITE ,__venc_hs_ext_start_stop_x_bits);
__IO_REG32_BIT(VENC_VS_INT_START_X,    			0x48050C78,__READ_WRITE ,__venc_vs_int_start_x_bits);
__IO_REG32_BIT(VENC_VS_INT_STOP_X_VS_INT_START_Y,			0x48050C7C,__READ_WRITE ,__venc_vs_int_stop_x_vs_int_start_y_bits);
__IO_REG32_BIT(VENC_VS_INT_STOP_Y_VS_EXT_START_X, 		0x48050C80,__READ_WRITE ,__venc_vs_int_stop_y_vs_ext_start_x_bits);
__IO_REG32_BIT(VENC_VS_EXT_STOP_X_VS_EXT_START_Y,		 	0x48050C84,__READ_WRITE ,__venc_vs_ext_stop_x_vs_ext_start_y_bits);
__IO_REG32_BIT(VENC_VS_EXT_STOP_Y,    			0x48050C88,__READ_WRITE ,__venc_vs_ext_stop_y_bits);
__IO_REG32_BIT(VENC_AVID_START_STOP_X,    	0x48050C90,__READ_WRITE ,__venc_avid_start_stop_x_bits);
__IO_REG32_BIT(VENC_AVID_START_STOP_Y,			0x48050C94,__READ_WRITE ,__venc_avid_start_stop_y_bits);
__IO_REG32_BIT(VENC_FID_INT_START_X_FID_INT_START_Y, 	0x48050CA0,__READ_WRITE ,__venc_fid_int_start_x_fid_int_start_y_bits);
__IO_REG32_BIT(VENC_FID_INT_OFFSET_Y_FID_EXT_START_X,	0x48050CA4,__READ_WRITE ,__venc_fid_int_offset_y_fid_ext_start_x_bits);
__IO_REG32_BIT(VENC_FID_EXT_START_Y_FID_EXT_OFFSET_Y,	0x48050CA8,__READ_WRITE ,__venc_fid_ext_start_y_fid_ext_offset_y_bits);
__IO_REG32_BIT(VENC_TVDETGP_INT_START_STOP_X,   			0x48050CB0,__READ_WRITE ,__venc_tvdetgp_int_start_stop_x_bits);
__IO_REG32_BIT(VENC_TVDETGP_INT_START_STOP_Y,    			0x48050CB4,__READ_WRITE ,__venc_tvdetgp_int_start_stop_y_bits);
__IO_REG32_BIT(VENC_GEN_CTRL,    						0x48050CB8,__READ_WRITE ,__venc_gen_ctrl_bits);
__IO_REG32_BIT(VENC_OUTPUT_CONTROL,    			0x48050CC4,__READ_WRITE ,__venc_output_control_bits);
__IO_REG32_BIT(VENC_OUTPUT_TEST,    				0x48050CC8,__READ_WRITE ,__venc_output_test_bits);

/***************************************************************************
 **
 ** DSI
 **
 ***************************************************************************/
__IO_REG32_BIT(DSI_REVISION,    						0x4804FC00,__READ				,__dsi_revision_bits);
__IO_REG32_BIT(DSI_SYSCONFIG,    						0x4804FC10,__READ_WRITE ,__dsi_sysconfig_bits);
__IO_REG32_BIT(DSI_SYSSTATUS,    						0x4804FC14,__READ				,__dsi_sysstatus_bits);
__IO_REG32_BIT(DSI_IRQSTATUS,    						0x4804FC18,__READ_WRITE ,__dsi_irqstatus_bits);
__IO_REG32_BIT(DSI_IRQENABLE,    						0x4804FC1C,__READ_WRITE ,__dsi_irqenable_bits);
__IO_REG32_BIT(DSI_CTRL,    								0x4804FC40,__READ_WRITE ,__dsi_ctrl_bits);
__IO_REG32_BIT(DSI_COMPLEXIO_CFG1,    			0x4804FC48,__READ_WRITE ,__dsi_complexio_cfg_bits);
__IO_REG32_BIT(DSI_COMPLEXIO_IRQ_STATUS,    0x4804FC4C,__READ_WRITE ,__dsi_complexio_irq_status_bits);
__IO_REG32_BIT(DSI_COMPLEXIO_IRQ_ENABLE,    0x4804FC50,__READ_WRITE ,__dsi_complexio_irq_enable_bits);
__IO_REG32_BIT(DSI_CLK_CTRL,    						0x4804FC54,__READ_WRITE ,__dsi_clk_ctrl_bits);
__IO_REG32_BIT(DSI_TIMING1,    							0x4804FC58,__READ_WRITE ,__dsi_timing1_bits);
__IO_REG32_BIT(DSI_TIMING2,    							0x4804FC5C,__READ_WRITE ,__dsi_timing2_bits);
__IO_REG32_BIT(DSI_VM_TIMING1,    					0x4804FC60,__READ_WRITE ,__dsi_vm_timing1_bits);
__IO_REG32_BIT(DSI_VM_TIMING2,    					0x4804FC64,__READ_WRITE ,__dsi_vm_timing2_bits);
__IO_REG32_BIT(DSI_VM_TIMING3,    					0x4804FC68,__READ_WRITE ,__dsi_vm_timing3_bits);
__IO_REG32_BIT(DSI_CLK_TIMING,    					0x4804FC6C,__READ_WRITE ,__dsi_clk_timing_bits);
__IO_REG32_BIT(DSI_TX_FIFO_VC_SIZE,    			0x4804FC70,__READ_WRITE ,__dsi_tx_fifo_vc_size_bits);
__IO_REG32_BIT(DSI_RX_FIFO_VC_SIZE,    			0x4804FC74,__READ_WRITE ,__dsi_tx_fifo_vc_size_bits);
__IO_REG32_BIT(DSI_COMPLEXIO_CFG2,    			0x4804FC78,__READ_WRITE ,__dsi_complexio_cfg2_bits);
__IO_REG32_BIT(DSI_RX_FIFO_VC_FULLNESS,    	0x4804FC7C,__READ				,__dsi_rx_fifo_vc_fullness_bits);
__IO_REG32_BIT(DSI_VM_TIMING4,    					0x4804FC80,__READ_WRITE ,__dsi_vm_timing4_bits);
__IO_REG32_BIT(DSI_TX_FIFO_VC_EMPTINESS,    0x4804FC84,__READ_WRITE ,__dsi_tx_fifo_vc_emptiness_bits);
__IO_REG32_BIT(DSI_VM_TIMING5,    					0x4804FC88,__READ_WRITE ,__dsi_vm_timing5_bits);
__IO_REG32_BIT(DSI_VM_TIMING6,    					0x4804FC8C,__READ_WRITE ,__dsi_vm_timing6_bits);
__IO_REG32_BIT(DSI_VM_TIMING7,    					0x4804FC90,__READ_WRITE ,__dsi_vm_timing7_bits);
__IO_REG32_BIT(DSI_STOPCLK_TIMING,    			0x4804FC94,__READ_WRITE ,__dsi_stopclk_timing_bits);
__IO_REG32_BIT(DSI_VC0_CTRL,    						0x4804FD00,__READ_WRITE ,__dsi_vcn_ctrl_bits);
__IO_REG32_BIT(DSI_VC0_TE,    							0x4804FD04,__READ_WRITE ,__dsi_vcn_te_bits);
__IO_REG32(		 DSI_VC0_LONG_PACKET_HEADER,  0x4804FD08,__WRITE 			);
__IO_REG32(		 DSI_VC0_LONG_PACKET_PAYLOAD, 0x4804FD0C,__WRITE 			);
__IO_REG32(		 DSI_VC0_SHORT_PACKET_HEADER, 0x4804FD10,__READ_WRITE );
__IO_REG32_BIT(DSI_VC0_IRQSTATUS, 					0x4804FD18,__READ_WRITE ,__dsi_vcn_irqstatus_bits);
__IO_REG32_BIT(DSI_VC0_IRQENABLE, 					0x4804FD1C,__READ_WRITE ,__dsi_vcn_irqenable_bits);
__IO_REG32_BIT(DSI_VC1_CTRL,    						0x4804FD20,__READ_WRITE ,__dsi_vcn_ctrl_bits);
__IO_REG32_BIT(DSI_VC1_TE,    							0x4804FD24,__READ_WRITE ,__dsi_vcn_te_bits);
__IO_REG32(		 DSI_VC1_LONG_PACKET_HEADER,  0x4804FD28,__WRITE 			);
__IO_REG32(		 DSI_VC1_LONG_PACKET_PAYLOAD, 0x4804FD2C,__WRITE 			);
__IO_REG32(		 DSI_VC1_SHORT_PACKET_HEADER, 0x4804FD30,__READ_WRITE );
__IO_REG32_BIT(DSI_VC1_IRQSTATUS, 					0x4804FD38,__READ_WRITE ,__dsi_vcn_irqstatus_bits);
__IO_REG32_BIT(DSI_VC1_IRQENABLE, 					0x4804FD3C,__READ_WRITE ,__dsi_vcn_irqenable_bits);
__IO_REG32_BIT(DSI_VC2_CTRL,    						0x4804FD40,__READ_WRITE ,__dsi_vcn_ctrl_bits);
__IO_REG32_BIT(DSI_VC2_TE,    							0x4804FD44,__READ_WRITE ,__dsi_vcn_te_bits);
__IO_REG32(		 DSI_VC2_LONG_PACKET_HEADER,  0x4804FD48,__WRITE 			);
__IO_REG32(		 DSI_VC2_LONG_PACKET_PAYLOAD, 0x4804FD4C,__WRITE 			);
__IO_REG32(		 DSI_VC2_SHORT_PACKET_HEADER, 0x4804FD50,__READ_WRITE );
__IO_REG32_BIT(DSI_VC2_IRQSTATUS, 					0x4804FD58,__READ_WRITE ,__dsi_vcn_irqstatus_bits);
__IO_REG32_BIT(DSI_VC2_IRQENABLE, 					0x4804FD5C,__READ_WRITE ,__dsi_vcn_irqenable_bits);
__IO_REG32_BIT(DSI_VC3_CTRL,    						0x4804FD60,__READ_WRITE ,__dsi_vcn_ctrl_bits);
__IO_REG32_BIT(DSI_VC3_TE,    							0x4804FD64,__READ_WRITE ,__dsi_vcn_te_bits);
__IO_REG32(		 DSI_VC3_LONG_PACKET_HEADER,  0x4804FD68,__WRITE 			);
__IO_REG32(		 DSI_VC3_LONG_PACKET_PAYLOAD, 0x4804FD6C,__WRITE 			);
__IO_REG32(		 DSI_VC3_SHORT_PACKET_HEADER, 0x4804FD70,__READ_WRITE );
__IO_REG32_BIT(DSI_VC3_IRQSTATUS, 					0x4804FD78,__READ_WRITE ,__dsi_vcn_irqstatus_bits);
__IO_REG32_BIT(DSI_VC3_IRQENABLE, 					0x4804FD7C,__READ_WRITE ,__dsi_vcn_irqenable_bits);

/***************************************************************************
 **
 ** DSI_PHY
 **
 ***************************************************************************/
__IO_REG32_BIT(DSI_PHY_CFG0,    						0x4804FE00,__READ_WRITE	,__dsi_phy_cfg0_bits);
__IO_REG32_BIT(DSI_PHY_CFG1,    						0x4804FE04,__READ_WRITE	,__dsi_phy_cfg1_bits);
__IO_REG32_BIT(DSI_PHY_CFG2,    						0x4804FE08,__READ_WRITE	,__dsi_phy_cfg2_bits);
__IO_REG32_BIT(DSI_PHY_CFG5,    						0x4804FE14,__READ				,__dsi_phy_cfg5_bits);

/***************************************************************************
 **
 ** DSI PLL
 **
 ***************************************************************************/
__IO_REG32_BIT(DSI_PLL_CONTROL,    					0x4804FF00,__READ_WRITE	,__dsi_pll_control_bits);
__IO_REG32_BIT(DSI_PLL_STATUS,    					0x4804FF04,__READ				,__dsi_pll_status_bits);
__IO_REG32_BIT(DSI_PLL_GO,    							0x4804FF08,__READ_WRITE	,__dsi_pll_go_bits);
__IO_REG32_BIT(DSI_PLL_CONFIGURATION1,    	0x4804FF0C,__READ_WRITE	,__dsi_pll_configuration1_bits);
__IO_REG32_BIT(DSI_PLL_CONFIGURATION2,    	0x4804FF10,__READ_WRITE	,__dsi_pll_configuration2_bits);

/***************************************************************************
 **
 ** GPT1
 **
 ***************************************************************************/
__IO_REG32_BIT(GPT1_TIDR,    								0x48318000,__READ				,__gpt_tidr_bits);
__IO_REG32_BIT(GPT1_TIOCP_CFG,    					0x48318010,__READ_WRITE	,__gpt_tiocp_cfg_bits);
__IO_REG32_BIT(GPT1_TISTAT,    							0x48318014,__READ				,__gpt_tistat_bits);
__IO_REG32_BIT(GPT1_TISR,    								0x48318018,__READ_WRITE	,__gpt_tisr_bits);
__IO_REG32_BIT(GPT1_TIER,    								0x4831801C,__READ_WRITE	,__gpt_tier_bits);
__IO_REG32_BIT(GPT1_TWER,    								0x48318020,__READ_WRITE	,__gpt_twer_bits);
__IO_REG32_BIT(GPT1_TCLR,    								0x48318024,__READ_WRITE	,__gpt_tclr_bits);
__IO_REG32(		 GPT1_TCRR,    								0x48318028,__READ_WRITE	);
__IO_REG32(		 GPT1_TLDR,    								0x4831802C,__READ_WRITE	);
__IO_REG32(		 GPT1_TTGR,    								0x48318030,__READ_WRITE	);
__IO_REG32_BIT(GPT1_TWPS,    								0x48318034,__READ				,__gpt_twps_bits);
__IO_REG32(		 GPT1_TMAR,    								0x48318038,__READ_WRITE	);
__IO_REG32(		 GPT1_TCAR1,    							0x4831803C,__READ				);
__IO_REG32_BIT(GPT1_TSICR,    							0x48318040,__READ_WRITE	,__gpt_tsicr_bits);
__IO_REG32(		 GPT1_TCAR2,    							0x48318044,__READ				);
__IO_REG32(		 GPT1_TPIR,    								0x48318048,__READ_WRITE	);
__IO_REG32(		 GPT1_TNIR,    								0x4831804C,__READ_WRITE	);
__IO_REG32(		 GPT1_TCVR,    								0x48318050,__READ_WRITE	);
__IO_REG32_BIT(GPT1_TOCR,    								0x48318054,__READ_WRITE	,__gpt_tocr_bits);
__IO_REG32_BIT(GPT1_TOWR,    								0x48318058,__READ_WRITE	,__gpt_towr_bits);

/***************************************************************************
 **
 ** GPT2
 **
 ***************************************************************************/
__IO_REG32_BIT(GPT2_TIDR,    								0x49032000,__READ				,__gpt_tidr_bits);
__IO_REG32_BIT(GPT2_TIOCP_CFG,    					0x49032010,__READ_WRITE	,__gpt_tiocp_cfg_bits);
__IO_REG32_BIT(GPT2_TISTAT,    							0x49032014,__READ				,__gpt_tistat_bits);
__IO_REG32_BIT(GPT2_TISR,    								0x49032018,__READ_WRITE	,__gpt_tisr_bits);
__IO_REG32_BIT(GPT2_TIER,    								0x4903201C,__READ_WRITE	,__gpt_tier_bits);
__IO_REG32_BIT(GPT2_TWER,    								0x49032020,__READ_WRITE	,__gpt_twer_bits);
__IO_REG32_BIT(GPT2_TCLR,    								0x49032024,__READ_WRITE	,__gpt_tclr_bits);
__IO_REG32(		 GPT2_TCRR,    								0x49032028,__READ_WRITE	);
__IO_REG32(		 GPT2_TLDR,    								0x4903202C,__READ_WRITE	);
__IO_REG32(		 GPT2_TTGR,    								0x49032030,__READ_WRITE	);
__IO_REG32_BIT(GPT2_TWPS,    								0x49032034,__READ				,__gpt_twps_bits);
__IO_REG32(		 GPT2_TMAR,    								0x49032038,__READ_WRITE	);
__IO_REG32(		 GPT2_TCAR1,    							0x4903203C,__READ				);
__IO_REG32_BIT(GPT2_TSICR,    							0x49032040,__READ_WRITE	,__gpt_tsicr_bits);
__IO_REG32(		 GPT2_TCAR2,    							0x49032044,__READ				);
__IO_REG32(		 GPT2_TPIR,    								0x49032048,__READ_WRITE	);
__IO_REG32(		 GPT2_TNIR,    								0x4903204C,__READ_WRITE	);
__IO_REG32(		 GPT2_TCVR,    								0x49032050,__READ_WRITE	);
__IO_REG32_BIT(GPT2_TOCR,    								0x49032054,__READ_WRITE	,__gpt_tocr_bits);
__IO_REG32_BIT(GPT2_TOWR,    								0x49032058,__READ_WRITE	,__gpt_towr_bits);

/***************************************************************************
 **
 ** GPT3
 **
 ***************************************************************************/
__IO_REG32_BIT(GPT3_TIDR,    								0x49034000,__READ				,__gpt_tidr_bits);
__IO_REG32_BIT(GPT3_TIOCP_CFG,    					0x49034010,__READ_WRITE	,__gpt_tiocp_cfg_bits);
__IO_REG32_BIT(GPT3_TISTAT,    							0x49034014,__READ				,__gpt_tistat_bits);
__IO_REG32_BIT(GPT3_TISR,    								0x49034018,__READ_WRITE	,__gpt_tisr_bits);
__IO_REG32_BIT(GPT3_TIER,    								0x4903401C,__READ_WRITE	,__gpt_tier_bits);
__IO_REG32_BIT(GPT3_TWER,    								0x49034020,__READ_WRITE	,__gpt_twer_bits);
__IO_REG32_BIT(GPT3_TCLR,    								0x49034024,__READ_WRITE	,__gpt_tclr_bits);
__IO_REG32(		 GPT3_TCRR,    								0x49034028,__READ_WRITE	);
__IO_REG32(		 GPT3_TLDR,    								0x4903402C,__READ_WRITE	);
__IO_REG32(		 GPT3_TTGR,    								0x49034030,__READ_WRITE	);
__IO_REG32_BIT(GPT3_TWPS,    								0x49034034,__READ				,__gpt_twps_bits);
__IO_REG32(		 GPT3_TMAR,    								0x49034038,__READ_WRITE	);
__IO_REG32(		 GPT3_TCAR1,    							0x4903403C,__READ				);
__IO_REG32_BIT(GPT3_TSICR,    							0x49034040,__READ_WRITE	,__gpt_tsicr_bits);
__IO_REG32(		 GPT3_TCAR2,    							0x49034044,__READ				);

/***************************************************************************
 **
 ** GPT4
 **
 ***************************************************************************/
__IO_REG32_BIT(GPT4_TIDR,    								0x49036000,__READ				,__gpt_tidr_bits);
__IO_REG32_BIT(GPT4_TIOCP_CFG,    					0x49036010,__READ_WRITE	,__gpt_tiocp_cfg_bits);
__IO_REG32_BIT(GPT4_TISTAT,    							0x49036014,__READ				,__gpt_tistat_bits);
__IO_REG32_BIT(GPT4_TISR,    								0x49036018,__READ_WRITE	,__gpt_tisr_bits);
__IO_REG32_BIT(GPT4_TIER,    								0x4903601C,__READ_WRITE	,__gpt_tier_bits);
__IO_REG32_BIT(GPT4_TWER,    								0x49036020,__READ_WRITE	,__gpt_twer_bits);
__IO_REG32_BIT(GPT4_TCLR,    								0x49036024,__READ_WRITE	,__gpt_tclr_bits);
__IO_REG32(		 GPT4_TCRR,    								0x49036028,__READ_WRITE	);
__IO_REG32(		 GPT4_TLDR,    								0x4903602C,__READ_WRITE	);
__IO_REG32(		 GPT4_TTGR,    								0x49036030,__READ_WRITE	);
__IO_REG32_BIT(GPT4_TWPS,    								0x49036034,__READ				,__gpt_twps_bits);
__IO_REG32(		 GPT4_TMAR,    								0x49036038,__READ_WRITE	);
__IO_REG32(		 GPT4_TCAR1,    							0x4903603C,__READ				);
__IO_REG32_BIT(GPT4_TSICR,    							0x49036040,__READ_WRITE	,__gpt_tsicr_bits);
__IO_REG32(		 GPT4_TCAR2,    							0x49036044,__READ				);

/***************************************************************************
 **
 ** GPT5
 **
 ***************************************************************************/
__IO_REG32_BIT(GPT5_TIDR,    								0x49038000,__READ				,__gpt_tidr_bits);
__IO_REG32_BIT(GPT5_TIOCP_CFG,    					0x49038010,__READ_WRITE	,__gpt_tiocp_cfg_bits);
__IO_REG32_BIT(GPT5_TISTAT,    							0x49038014,__READ				,__gpt_tistat_bits);
__IO_REG32_BIT(GPT5_TISR,    								0x49038018,__READ_WRITE	,__gpt_tisr_bits);
__IO_REG32_BIT(GPT5_TIER,    								0x4903801C,__READ_WRITE	,__gpt_tier_bits);
__IO_REG32_BIT(GPT5_TWER,    								0x49038020,__READ_WRITE	,__gpt_twer_bits);
__IO_REG32_BIT(GPT5_TCLR,    								0x49038024,__READ_WRITE	,__gpt_tclr_bits);
__IO_REG32(		 GPT5_TCRR,    								0x49038028,__READ_WRITE	);
__IO_REG32(		 GPT5_TLDR,    								0x4903802C,__READ_WRITE	);
__IO_REG32(		 GPT5_TTGR,    								0x49038030,__READ_WRITE	);
__IO_REG32_BIT(GPT5_TWPS,    								0x49038034,__READ				,__gpt_twps_bits);
__IO_REG32(		 GPT5_TMAR,    								0x49038038,__READ_WRITE	);
__IO_REG32(		 GPT5_TCAR1,    							0x4903803C,__READ				);
__IO_REG32_BIT(GPT5_TSICR,    							0x49038040,__READ_WRITE	,__gpt_tsicr_bits);
__IO_REG32(		 GPT5_TCAR2,    							0x49038044,__READ				);

/***************************************************************************
 **
 ** GPT6
 **
 ***************************************************************************/
__IO_REG32_BIT(GPT6_TIDR,    								0x4903A000,__READ				,__gpt_tidr_bits);
__IO_REG32_BIT(GPT6_TIOCP_CFG,    					0x4903A010,__READ_WRITE	,__gpt_tiocp_cfg_bits);
__IO_REG32_BIT(GPT6_TISTAT,    							0x4903A014,__READ				,__gpt_tistat_bits);
__IO_REG32_BIT(GPT6_TISR,    								0x4903A018,__READ_WRITE	,__gpt_tisr_bits);
__IO_REG32_BIT(GPT6_TIER,    								0x4903A01C,__READ_WRITE	,__gpt_tier_bits);
__IO_REG32_BIT(GPT6_TWER,    								0x4903A020,__READ_WRITE	,__gpt_twer_bits);
__IO_REG32_BIT(GPT6_TCLR,    								0x4903A024,__READ_WRITE	,__gpt_tclr_bits);
__IO_REG32(		 GPT6_TCRR,    								0x4903A028,__READ_WRITE	);
__IO_REG32(		 GPT6_TLDR,    								0x4903A02C,__READ_WRITE	);
__IO_REG32(		 GPT6_TTGR,    								0x4903A030,__READ_WRITE	);
__IO_REG32_BIT(GPT6_TWPS,    								0x4903A034,__READ				,__gpt_twps_bits);
__IO_REG32(		 GPT6_TMAR,    								0x4903A038,__READ_WRITE	);
__IO_REG32(		 GPT6_TCAR1,    							0x4903A03C,__READ				);
__IO_REG32_BIT(GPT6_TSICR,    							0x4903A040,__READ_WRITE	,__gpt_tsicr_bits);
__IO_REG32(		 GPT6_TCAR2,    							0x4903A044,__READ				);

/***************************************************************************
 **
 ** GPT7
 **
 ***************************************************************************/
__IO_REG32_BIT(GPT7_TIDR,    								0x4903C000,__READ				,__gpt_tidr_bits);
__IO_REG32_BIT(GPT7_TIOCP_CFG,    					0x4903C010,__READ_WRITE	,__gpt_tiocp_cfg_bits);
__IO_REG32_BIT(GPT7_TISTAT,    							0x4903C014,__READ				,__gpt_tistat_bits);
__IO_REG32_BIT(GPT7_TISR,    								0x4903C018,__READ_WRITE	,__gpt_tisr_bits);
__IO_REG32_BIT(GPT7_TIER,    								0x4903C01C,__READ_WRITE	,__gpt_tier_bits);
__IO_REG32_BIT(GPT7_TWER,    								0x4903C020,__READ_WRITE	,__gpt_twer_bits);
__IO_REG32_BIT(GPT7_TCLR,    								0x4903C024,__READ_WRITE	,__gpt_tclr_bits);
__IO_REG32(		 GPT7_TCRR,    								0x4903C028,__READ_WRITE	);
__IO_REG32(		 GPT7_TLDR,    								0x4903C02C,__READ_WRITE	);
__IO_REG32(		 GPT7_TTGR,    								0x4903C030,__READ_WRITE	);
__IO_REG32_BIT(GPT7_TWPS,    								0x4903C034,__READ				,__gpt_twps_bits);
__IO_REG32(		 GPT7_TMAR,    								0x4903C038,__READ_WRITE	);
__IO_REG32(		 GPT7_TCAR1,    							0x4903C03C,__READ				);
__IO_REG32_BIT(GPT7_TSICR,    							0x4903C040,__READ_WRITE	,__gpt_tsicr_bits);
__IO_REG32(		 GPT7_TCAR2,    							0x4903C044,__READ				);

/***************************************************************************
 **
 ** GPT8
 **
 ***************************************************************************/
__IO_REG32_BIT(GPT8_TIDR,    								0x4903E000,__READ				,__gpt_tidr_bits);
__IO_REG32_BIT(GPT8_TIOCP_CFG,    					0x4903E010,__READ_WRITE	,__gpt_tiocp_cfg_bits);
__IO_REG32_BIT(GPT8_TISTAT,    							0x4903E014,__READ				,__gpt_tistat_bits);
__IO_REG32_BIT(GPT8_TISR,    								0x4903E018,__READ_WRITE	,__gpt_tisr_bits);
__IO_REG32_BIT(GPT8_TIER,    								0x4903E01C,__READ_WRITE	,__gpt_tier_bits);
__IO_REG32_BIT(GPT8_TWER,    								0x4903E020,__READ_WRITE	,__gpt_twer_bits);
__IO_REG32_BIT(GPT8_TCLR,    								0x4903E024,__READ_WRITE	,__gpt_tclr_bits);
__IO_REG32(		 GPT8_TCRR,    								0x4903E028,__READ_WRITE	);
__IO_REG32(		 GPT8_TLDR,    								0x4903E02C,__READ_WRITE	);
__IO_REG32(		 GPT8_TTGR,    								0x4903E030,__READ_WRITE	);
__IO_REG32_BIT(GPT8_TWPS,    								0x4903E034,__READ				,__gpt_twps_bits);
__IO_REG32(		 GPT8_TMAR,    								0x4903E038,__READ_WRITE	);
__IO_REG32(		 GPT8_TCAR1,    							0x4903E03C,__READ				);
__IO_REG32_BIT(GPT8_TSICR,    							0x4903E040,__READ_WRITE	,__gpt_tsicr_bits);
__IO_REG32(		 GPT8_TCAR2,    							0x4903E044,__READ				);

/***************************************************************************
 **
 ** GPT9
 **
 ***************************************************************************/
__IO_REG32_BIT(GPT9_TIDR,    								0x49040000,__READ				,__gpt_tidr_bits);
__IO_REG32_BIT(GPT9_TIOCP_CFG,    					0x49040010,__READ_WRITE	,__gpt_tiocp_cfg_bits);
__IO_REG32_BIT(GPT9_TISTAT,    							0x49040014,__READ				,__gpt_tistat_bits);
__IO_REG32_BIT(GPT9_TISR,    								0x49040018,__READ_WRITE	,__gpt_tisr_bits);
__IO_REG32_BIT(GPT9_TIER,    								0x4904001C,__READ_WRITE	,__gpt_tier_bits);
__IO_REG32_BIT(GPT9_TWER,    								0x49040020,__READ_WRITE	,__gpt_twer_bits);
__IO_REG32_BIT(GPT9_TCLR,    								0x49040024,__READ_WRITE	,__gpt_tclr_bits);
__IO_REG32(		 GPT9_TCRR,    								0x49040028,__READ_WRITE	);
__IO_REG32(		 GPT9_TLDR,    								0x4904002C,__READ_WRITE	);
__IO_REG32(		 GPT9_TTGR,    								0x49040030,__READ_WRITE	);
__IO_REG32_BIT(GPT9_TWPS,    								0x49040034,__READ				,__gpt_twps_bits);
__IO_REG32(		 GPT9_TMAR,    								0x49040038,__READ_WRITE	);
__IO_REG32(		 GPT9_TCAR1,    							0x4904003C,__READ				);
__IO_REG32_BIT(GPT9_TSICR,    							0x49040040,__READ_WRITE	,__gpt_tsicr_bits);
__IO_REG32(		 GPT9_TCAR2,    							0x49040044,__READ				);

/***************************************************************************
 **
 ** GPT10
 **
 ***************************************************************************/
__IO_REG32_BIT(GPT10_TIDR,    							0x48086000,__READ				,__gpt_tidr_bits);
__IO_REG32_BIT(GPT10_TIOCP_CFG,    					0x48086010,__READ_WRITE	,__gpt_tiocp_cfg_bits);
__IO_REG32_BIT(GPT10_TISTAT,    						0x48086014,__READ				,__gpt_tistat_bits);
__IO_REG32_BIT(GPT10_TISR,    							0x48086018,__READ_WRITE	,__gpt_tisr_bits);
__IO_REG32_BIT(GPT10_TIER,    							0x4808601C,__READ_WRITE	,__gpt_tier_bits);
__IO_REG32_BIT(GPT10_TWER,    							0x48086020,__READ_WRITE	,__gpt_twer_bits);
__IO_REG32_BIT(GPT10_TCLR,    							0x48086024,__READ_WRITE	,__gpt_tclr_bits);
__IO_REG32(		 GPT10_TCRR,    							0x48086028,__READ_WRITE	);
__IO_REG32(		 GPT10_TLDR,    							0x4808602C,__READ_WRITE	);
__IO_REG32(		 GPT10_TTGR,    							0x48086030,__READ_WRITE	);
__IO_REG32_BIT(GPT10_TWPS,    							0x48086034,__READ				,__gpt_twps_bits);
__IO_REG32(		 GPT10_TMAR,    							0x48086038,__READ_WRITE	);
__IO_REG32(		 GPT10_TCAR1,    							0x4808603C,__READ				);
__IO_REG32_BIT(GPT10_TSICR,    							0x48086040,__READ_WRITE	,__gpt_tsicr_bits);
__IO_REG32(		 GPT10_TCAR2,    							0x48086044,__READ				);
__IO_REG32(		 GPT10_TPIR,    							0x48086048,__READ_WRITE	);
__IO_REG32(		 GPT10_TNIR,    							0x4808604C,__READ_WRITE	);
__IO_REG32(		 GPT10_TCVR,    							0x48086050,__READ_WRITE	);
__IO_REG32_BIT(GPT10_TOCR,    							0x48086054,__READ_WRITE	,__gpt_tocr_bits);
__IO_REG32_BIT(GPT10_TOWR,    							0x48086058,__READ_WRITE	,__gpt_towr_bits);

/***************************************************************************
 **
 ** GPT11
 **
 ***************************************************************************/
__IO_REG32_BIT(GPT11_TIDR,    							0x48088000,__READ				,__gpt_tidr_bits);
__IO_REG32_BIT(GPT11_TIOCP_CFG,    					0x48088010,__READ_WRITE	,__gpt_tiocp_cfg_bits);
__IO_REG32_BIT(GPT11_TISTAT,    						0x48088014,__READ				,__gpt_tistat_bits);
__IO_REG32_BIT(GPT11_TISR,    							0x48088018,__READ_WRITE	,__gpt_tisr_bits);
__IO_REG32_BIT(GPT11_TIER,    							0x4808801C,__READ_WRITE	,__gpt_tier_bits);
__IO_REG32_BIT(GPT11_TWER,    							0x48088020,__READ_WRITE	,__gpt_twer_bits);
__IO_REG32_BIT(GPT11_TCLR,    							0x48088024,__READ_WRITE	,__gpt_tclr_bits);
__IO_REG32(		 GPT11_TCRR,    							0x48088028,__READ_WRITE	);
__IO_REG32(		 GPT11_TLDR,    							0x4808802C,__READ_WRITE	);
__IO_REG32(		 GPT11_TTGR,    							0x48088030,__READ_WRITE	);
__IO_REG32_BIT(GPT11_TWPS,    							0x48088034,__READ				,__gpt_twps_bits);
__IO_REG32(		 GPT11_TMAR,    							0x48088038,__READ_WRITE	);
__IO_REG32(		 GPT11_TCAR1,    							0x4808803C,__READ				);
__IO_REG32_BIT(GPT11_TSICR,    							0x48088040,__READ_WRITE	,__gpt_tsicr_bits);
__IO_REG32(		 GPT11_TCAR2,    							0x48088044,__READ				);

/***************************************************************************
 **
 ** WDT2
 **
 ***************************************************************************/
__IO_REG32_BIT(WDT2_WIDR,    								0x48314000,__READ				,__wdt_widr_bits);
__IO_REG32_BIT(WDT2_SYSCONFIG,    					0x48314010,__READ_WRITE	,__wdt_sysconfig_bits);
__IO_REG32_BIT(WDT2_SYSSTATUS,    					0x48314014,__READ				,__wdt_sysstatus_bits);
__IO_REG32_BIT(WDT2_WISR,    								0x48314018,__READ_WRITE	,__wdt_wisr_bits);
__IO_REG32_BIT(WDT2_WIER,    								0x4831401C,__READ_WRITE	,__wdt_wier_bits);
__IO_REG32_BIT(WDT2_WCLR,    								0x48314024,__READ_WRITE	,__wdt_wclr_bits);
__IO_REG32(		 WDT2_WCRR,    								0x48314028,__READ_WRITE	);
__IO_REG32(	 	 WDT2_WLDR,    								0x4831402C,__READ_WRITE	);
__IO_REG32(		 WDT2_WTGR,    								0x48314030,__READ_WRITE	);
__IO_REG32_BIT(WDT2_WWPS,    								0x48314034,__READ				,__wdt_wwps_bits);
__IO_REG32(		 WDT2_WSPR,    								0x48314048,__READ_WRITE	);

/***************************************************************************
 **
 ** WDT3
 **
 ***************************************************************************/
__IO_REG32_BIT(WDT3_WIDR,    								0x49030000,__READ				,__wdt_widr_bits);
__IO_REG32_BIT(WDT3_SYSCONFIG,    					0x49030010,__READ_WRITE	,__wdt_sysconfig_bits);
__IO_REG32_BIT(WDT3_SYSSTATUS,    					0x49030114,__READ				,__wdt_sysstatus_bits);
__IO_REG32_BIT(WDT3_WISR,    								0x49030018,__READ_WRITE	,__wdt_wisr_bits);
__IO_REG32_BIT(WDT3_WIER,    								0x4903001C,__READ_WRITE	,__wdt_wier_bits);
__IO_REG32_BIT(WDT3_WCLR,    								0x49030024,__READ_WRITE	,__wdt_wclr_bits);
__IO_REG32(		 WDT3_WCRR,    								0x49030028,__READ_WRITE	);
__IO_REG32(	 	 WDT3_WLDR,    								0x4903002C,__READ_WRITE	);
__IO_REG32(		 WDT3_WTGR,    								0x49030030,__READ_WRITE	);
__IO_REG32_BIT(WDT3_WWPS,    								0x49030034,__READ				,__wdt_wwps_bits);
__IO_REG32(		 WDT3_WSPR,    								0x49030048,__READ_WRITE	);

/***************************************************************************
 **
 ** ST 32KHz
 **
 ***************************************************************************/
__IO_REG32_BIT(REG_32KSYNCNT_REV,    				0x48320000,__READ				,__reg_32ksyncnt_rev_bits);
__IO_REG32_BIT(REG_32KSYNCNT_SYSCONFIG,    	0x48320004,__READ_WRITE	,__reg_32ksyncnt_sysconfig_bits);
__IO_REG32(		 REG_32KSYNCNT_CR,    				0x48320010,__READ				);

/***************************************************************************
 **
 ** I2C1
 **
 ***************************************************************************/
__IO_REG16_BIT(I2C1_REV,    								0x48070000,__READ				,__i2c_rev_bits);
__IO_REG16_BIT(I2C1_IE,    									0x48070004,__READ_WRITE	,__i2c_ie_bits);
__IO_REG16_BIT(I2C1_STAT,    								0x48070008,__READ_WRITE	,__i2c_stat_bits);
__IO_REG16_BIT(I2C1_WE,    									0x4807000C,__READ_WRITE	,__i2c_we_bits);
__IO_REG16_BIT(I2C1_SYSS,    								0x48070010,__READ				,__i2c_syss_bits);
__IO_REG16_BIT(I2C1_BUF,    								0x48070014,__READ_WRITE	,__i2c_buf_bits);
__IO_REG16(    I2C1_CNT,   									0x48070018,__READ_WRITE	);
__IO_REG16_BIT(I2C1_DATA,    								0x4807001C,__READ_WRITE	,__i2c_data_bits);
__IO_REG16_BIT(I2C1_SYSC,    								0x48070020,__READ_WRITE	,__i2c_sysc_bits);
__IO_REG16_BIT(I2C1_CON,    								0x48070024,__READ_WRITE	,__i2c_con_bits);
__IO_REG16_BIT(I2C1_OA0,    								0x48070028,__READ_WRITE	,__i2c_oa0_bits);
__IO_REG16_BIT(I2C1_SA,    									0x4807002C,__READ_WRITE	,__i2c_sa_bits);
__IO_REG16_BIT(I2C1_PSC,    								0x48070030,__READ_WRITE	,__i2c_psc_bits);
__IO_REG16_BIT(I2C1_SCLL,    								0x48070034,__READ_WRITE	,__i2c_scll_bits);
__IO_REG16_BIT(I2C1_SCLH,    								0x48070038,__READ_WRITE	,__i2c_sclh_bits);
__IO_REG16_BIT(I2C1_SYSTEST,    						0x4807003C,__READ_WRITE	,__i2c_systest_bits);
__IO_REG16_BIT(I2C1_BUFSTAT,    						0x48070040,__READ				,__i2c_bufstat_bits);
__IO_REG16_BIT(I2C1_OA1,   									0x48070044,__READ_WRITE	,__i2c_oa1_bits);
__IO_REG16_BIT(I2C1_OA2,    								0x48070048,__READ_WRITE	,__i2c_oa2_bits);
__IO_REG16_BIT(I2C1_OA3,   									0x4807004C,__READ_WRITE	,__i2c_oa3_bits);
__IO_REG16_BIT(I2C1_ACTOA,   								0x48070050,__READ				,__i2c_actoa_bits);
__IO_REG16_BIT(I2C1_SBLOCK,  								0x48070054,__READ_WRITE	,__i2c_sblock_bits);

/***************************************************************************
 **
 ** I2C2
 **
 ***************************************************************************/
__IO_REG16_BIT(I2C2_REV,    								0x48072000,__READ				,__i2c_rev_bits);
__IO_REG16_BIT(I2C2_IE,    									0x48072004,__READ_WRITE	,__i2c_ie_bits);
__IO_REG16_BIT(I2C2_STAT,    								0x48072008,__READ_WRITE	,__i2c_stat_bits);
__IO_REG16_BIT(I2C2_WE,    									0x4807200C,__READ_WRITE	,__i2c_we_bits);
__IO_REG16_BIT(I2C2_SYSS,    								0x48072010,__READ				,__i2c_syss_bits);
__IO_REG16_BIT(I2C2_BUF,    								0x48072014,__READ_WRITE	,__i2c_buf_bits);
__IO_REG16(    I2C2_CNT,   									0x48072018,__READ_WRITE	);
__IO_REG16_BIT(I2C2_DATA,    								0x4807201C,__READ_WRITE	,__i2c_data_bits);
__IO_REG16_BIT(I2C2_SYSC,    								0x48072020,__READ_WRITE	,__i2c_sysc_bits);
__IO_REG16_BIT(I2C2_CON,    								0x48072024,__READ_WRITE	,__i2c_con_bits);
__IO_REG16_BIT(I2C2_OA0,    								0x48072028,__READ_WRITE	,__i2c_oa0_bits);
__IO_REG16_BIT(I2C2_SA,    									0x4807202C,__READ_WRITE	,__i2c_sa_bits);
__IO_REG16_BIT(I2C2_PSC,    								0x48072030,__READ_WRITE	,__i2c_psc_bits);
__IO_REG16_BIT(I2C2_SCLL,    								0x48072034,__READ_WRITE	,__i2c_scll_bits);
__IO_REG16_BIT(I2C2_SCLH,    								0x48072038,__READ_WRITE	,__i2c_sclh_bits);
__IO_REG16_BIT(I2C2_SYSTEST,    						0x4807203C,__READ_WRITE	,__i2c_systest_bits);
__IO_REG16_BIT(I2C2_BUFSTAT,    						0x48072040,__READ				,__i2c_bufstat_bits);
__IO_REG16_BIT(I2C2_OA1,   									0x48072044,__READ_WRITE	,__i2c_oa1_bits);
__IO_REG16_BIT(I2C2_OA2,    								0x48072048,__READ_WRITE	,__i2c_oa2_bits);
__IO_REG16_BIT(I2C2_OA3,   									0x4807204C,__READ_WRITE	,__i2c_oa3_bits);
__IO_REG16_BIT(I2C2_ACTOA,   								0x48072050,__READ				,__i2c_actoa_bits);
__IO_REG16_BIT(I2C2_SBLOCK,  								0x48072054,__READ_WRITE	,__i2c_sblock_bits);

/***************************************************************************
 **
 ** I2C3
 **
 ***************************************************************************/
__IO_REG16_BIT(I2C3_REV,    								0x48060000,__READ				,__i2c_rev_bits);
__IO_REG16_BIT(I2C3_IE,    									0x48060004,__READ_WRITE	,__i2c_ie_bits);
__IO_REG16_BIT(I2C3_STAT,    								0x48060008,__READ_WRITE	,__i2c_stat_bits);
__IO_REG16_BIT(I2C3_WE,    									0x4806000C,__READ_WRITE	,__i2c_we_bits);
__IO_REG16_BIT(I2C3_SYSS,    								0x48060010,__READ				,__i2c_syss_bits);
__IO_REG16_BIT(I2C3_BUF,    								0x48060014,__READ_WRITE	,__i2c_buf_bits);
__IO_REG16(    I2C3_CNT,   									0x48060018,__READ_WRITE	);
__IO_REG16_BIT(I2C3_DATA,    								0x4806001C,__READ_WRITE	,__i2c_data_bits);
__IO_REG16_BIT(I2C3_SYSC,    								0x48060020,__READ_WRITE	,__i2c_sysc_bits);
__IO_REG16_BIT(I2C3_CON,    								0x48060024,__READ_WRITE	,__i2c_con_bits);
__IO_REG16_BIT(I2C3_OA0,    								0x48060028,__READ_WRITE	,__i2c_oa0_bits);
__IO_REG16_BIT(I2C3_SA,    									0x4806002C,__READ_WRITE	,__i2c_sa_bits);
__IO_REG16_BIT(I2C3_PSC,    								0x48060030,__READ_WRITE	,__i2c_psc_bits);
__IO_REG16_BIT(I2C3_SCLL,    								0x48060034,__READ_WRITE	,__i2c_scll_bits);
__IO_REG16_BIT(I2C3_SCLH,    								0x48060038,__READ_WRITE	,__i2c_sclh_bits);
__IO_REG16_BIT(I2C3_SYSTEST,    						0x4806003C,__READ_WRITE	,__i2c_systest_bits);
__IO_REG16_BIT(I2C3_BUFSTAT,    						0x48060040,__READ				,__i2c_bufstat_bits);
__IO_REG16_BIT(I2C3_OA1,   									0x48060044,__READ_WRITE	,__i2c_oa1_bits);
__IO_REG16_BIT(I2C3_OA2,    								0x48060048,__READ_WRITE	,__i2c_oa2_bits);
__IO_REG16_BIT(I2C3_OA3,   									0x4806004C,__READ_WRITE	,__i2c_oa3_bits);
__IO_REG16_BIT(I2C3_ACTOA,   								0x48060050,__READ				,__i2c_actoa_bits);
__IO_REG16_BIT(I2C3_SBLOCK,  								0x48060054,__READ_WRITE	,__i2c_sblock_bits);

/***************************************************************************
 **
 ** McSPI1
 **
 ***************************************************************************/
__IO_REG32_BIT(MCSPI1_REVISION,    					0x48098000,__READ				,__mcspi_revision_bits);
__IO_REG32_BIT(MCSPI1_SYSCONFIG,    				0x48098010,__READ_WRITE	,__mcspi_sysconfig_bits);
__IO_REG32_BIT(MCSPI1_SYSSTATUS,    				0x48098014,__READ				,__mcspi_sysstatus_bits);
__IO_REG32_BIT(MCSPI1_IRQSTATUS,    				0x48098018,__READ_WRITE	,__mcspi_irqstatus_bits);
__IO_REG32_BIT(MCSPI1_IRQENABLE,    				0x4809801C,__READ_WRITE	,__mcspi_irqenable_bits);
__IO_REG32_BIT(MCSPI1_WAKEUPENABLE,    			0x48098020,__READ_WRITE	,__mcspi_wakeupenable_bits);
__IO_REG32_BIT(MCSPI1_SYST,    							0x48098024,__READ_WRITE	,__mcspi_syst_bits);
__IO_REG32_BIT(MCSPI1_MODULCTRL,    				0x48098028,__READ_WRITE	,__mcspi_modulctrl_bits);
__IO_REG32_BIT(MCSPI1_CH0CONF,    					0x4809802C,__READ_WRITE	,__mcspi_chxconf_bits);
__IO_REG32_BIT(MCSPI1_CH0STAT,    					0x48098030,__READ				,__mcspi_chxstat_bits);
__IO_REG32_BIT(MCSPI1_CH0CTRL,    					0x48098034,__READ_WRITE	,__mcspi_chxctrl_bits);
__IO_REG32(		 MCSPI1_TX0,    							0x48098038,__READ_WRITE	);
__IO_REG32(		 MCSPI1_RX0,    							0x4809803C,__READ				);
__IO_REG32_BIT(MCSPI1_CH1CONF,    					0x48098040,__READ_WRITE	,__mcspi_chxconf_bits);
__IO_REG32_BIT(MCSPI1_CH1STAT,    					0x48098044,__READ				,__mcspi_chxstat_bits);
__IO_REG32_BIT(MCSPI1_CH1CTRL,    					0x48098048,__READ_WRITE	,__mcspi_chxctrl_bits);
__IO_REG32(		 MCSPI1_TX1,    							0x4809804C,__READ_WRITE	);
__IO_REG32(		 MCSPI1_RX1,    							0x48098050,__READ				);
__IO_REG32_BIT(MCSPI1_CH2CONF,    					0x48098054,__READ_WRITE	,__mcspi_chxconf_bits);
__IO_REG32_BIT(MCSPI1_CH2STAT,    					0x48098058,__READ				,__mcspi_chxstat_bits);
__IO_REG32_BIT(MCSPI1_CH2CTRL,    					0x4809805C,__READ_WRITE	,__mcspi_chxctrl_bits);
__IO_REG32(		 MCSPI1_TX2,    							0x48098060,__READ_WRITE	);
__IO_REG32(		 MCSPI1_RX2,    							0x48098064,__READ				);
__IO_REG32_BIT(MCSPI1_CH3CONF,    					0x48098068,__READ_WRITE	,__mcspi_chxconf_bits);
__IO_REG32_BIT(MCSPI1_CH3STAT,    					0x4809806C,__READ				,__mcspi_chxstat_bits);
__IO_REG32_BIT(MCSPI1_CH3CTRL,    					0x48098070,__READ_WRITE	,__mcspi_chxctrl_bits);
__IO_REG32(		 MCSPI1_TX3,    							0x48098074,__READ_WRITE	);
__IO_REG32(		 MCSPI1_RX3,    							0x48098078,__READ				);
__IO_REG32_BIT(MCSPI1_XFERLEVEL,    				0x4809807C,__READ_WRITE	,__mcspi_xferlevel_bits);

/***************************************************************************
 **
 ** McSPI2
 **
 ***************************************************************************/
__IO_REG32_BIT(MCSPI2_REVISION,    					0x4809A000,__READ				,__mcspi_revision_bits);
__IO_REG32_BIT(MCSPI2_SYSCONFIG,    				0x4809A010,__READ_WRITE	,__mcspi_sysconfig_bits);
__IO_REG32_BIT(MCSPI2_SYSSTATUS,    				0x4809A014,__READ				,__mcspi_sysstatus_bits);
__IO_REG32_BIT(MCSPI2_IRQSTATUS,    				0x4809A018,__READ_WRITE	,__mcspi_irqstatus_bits);
__IO_REG32_BIT(MCSPI2_IRQENABLE,    				0x4809A01C,__READ_WRITE	,__mcspi_irqenable_bits);
__IO_REG32_BIT(MCSPI2_WAKEUPENABLE,    			0x4809A020,__READ_WRITE	,__mcspi_wakeupenable_bits);
__IO_REG32_BIT(MCSPI2_SYST,    							0x4809A024,__READ_WRITE	,__mcspi_syst_bits);
__IO_REG32_BIT(MCSPI2_MODULCTRL,    				0x4809A028,__READ_WRITE	,__mcspi_modulctrl_bits);
__IO_REG32_BIT(MCSPI2_CH0CONF,    					0x4809A02C,__READ_WRITE	,__mcspi_chxconf_bits);
__IO_REG32_BIT(MCSPI2_CH0STAT,    					0x4809A030,__READ				,__mcspi_chxstat_bits);
__IO_REG32_BIT(MCSPI2_CH0CTRL,    					0x4809A034,__READ_WRITE	,__mcspi_chxctrl_bits);
__IO_REG32(		 MCSPI2_TX0,    							0x4809A038,__READ_WRITE	);
__IO_REG32(		 MCSPI2_RX0,    							0x4809A03C,__READ				);
__IO_REG32_BIT(MCSPI2_CH1CONF,    					0x4809A040,__READ_WRITE	,__mcspi_chxconf_bits);
__IO_REG32_BIT(MCSPI2_CH1STAT,    					0x4809A044,__READ				,__mcspi_chxstat_bits);
__IO_REG32_BIT(MCSPI2_CH1CTRL,    					0x4809A048,__READ_WRITE	,__mcspi_chxctrl_bits);
__IO_REG32(		 MCSPI2_TX1,    							0x4809A04C,__READ_WRITE	);
__IO_REG32(		 MCSPI2_RX1,    							0x4809A050,__READ				);
__IO_REG32_BIT(MCSPI2_XFERLEVEL,    				0x4809A07C,__READ_WRITE	,__mcspi_xferlevel_bits);

/***************************************************************************
 **
 ** McSPI3
 **
 ***************************************************************************/
__IO_REG32_BIT(MCSPI3_REVISION,    					0x480B8000,__READ				,__mcspi_revision_bits);
__IO_REG32_BIT(MCSPI3_SYSCONFIG,    				0x480B8010,__READ_WRITE	,__mcspi_sysconfig_bits);
__IO_REG32_BIT(MCSPI3_SYSSTATUS,    				0x480B8014,__READ				,__mcspi_sysstatus_bits);
__IO_REG32_BIT(MCSPI3_IRQSTATUS,    				0x480B8018,__READ_WRITE	,__mcspi_irqstatus_bits);
__IO_REG32_BIT(MCSPI3_IRQENABLE,    				0x480B801C,__READ_WRITE	,__mcspi_irqenable_bits);
__IO_REG32_BIT(MCSPI3_WAKEUPENABLE,    			0x480B8020,__READ_WRITE	,__mcspi_wakeupenable_bits);
__IO_REG32_BIT(MCSPI3_SYST,    							0x480B8024,__READ_WRITE	,__mcspi_syst_bits);
__IO_REG32_BIT(MCSPI3_MODULCTRL,    				0x480B8028,__READ_WRITE	,__mcspi_modulctrl_bits);
__IO_REG32_BIT(MCSPI3_CH0CONF,    					0x480B802C,__READ_WRITE	,__mcspi_chxconf_bits);
__IO_REG32_BIT(MCSPI3_CH0STAT,    					0x480B8030,__READ				,__mcspi_chxstat_bits);
__IO_REG32_BIT(MCSPI3_CH0CTRL,    					0x480B8034,__READ_WRITE	,__mcspi_chxctrl_bits);
__IO_REG32(		 MCSPI3_TX0,    							0x480B8038,__READ_WRITE	);
__IO_REG32(		 MCSPI3_RX0,    							0x480B803C,__READ				);
__IO_REG32_BIT(MCSPI3_CH1CONF,    					0x480B8040,__READ_WRITE	,__mcspi_chxconf_bits);
__IO_REG32_BIT(MCSPI3_CH1STAT,    					0x480B8044,__READ				,__mcspi_chxstat_bits);
__IO_REG32_BIT(MCSPI3_CH1CTRL,    					0x480B8048,__READ_WRITE	,__mcspi_chxctrl_bits);
__IO_REG32(		 MCSPI3_TX1,    							0x480B804C,__READ_WRITE	);
__IO_REG32(		 MCSPI3_RX1,    							0x480B8050,__READ				);
__IO_REG32_BIT(MCSPI3_XFERLEVEL,    				0x480B807C,__READ_WRITE	,__mcspi_xferlevel_bits);

/***************************************************************************
 **
 ** McSPI4
 **
 ***************************************************************************/
__IO_REG32_BIT(MCSPI4_REVISION,    					0x480BA000,__READ				,__mcspi_revision_bits);
__IO_REG32_BIT(MCSPI4_SYSCONFIG,    				0x480BA010,__READ_WRITE	,__mcspi_sysconfig_bits);
__IO_REG32_BIT(MCSPI4_SYSSTATUS,    				0x480BA014,__READ				,__mcspi_sysstatus_bits);
__IO_REG32_BIT(MCSPI4_IRQSTATUS,    				0x480BA018,__READ_WRITE	,__mcspi_irqstatus_bits);
__IO_REG32_BIT(MCSPI4_IRQENABLE,    				0x480BA01C,__READ_WRITE	,__mcspi_irqenable_bits);
__IO_REG32_BIT(MCSPI4_WAKEUPENABLE,    			0x480BA020,__READ_WRITE	,__mcspi_wakeupenable_bits);
__IO_REG32_BIT(MCSPI4_SYST,    							0x480BA024,__READ_WRITE	,__mcspi_syst_bits);
__IO_REG32_BIT(MCSPI4_MODULCTRL,    				0x480BA028,__READ_WRITE	,__mcspi_modulctrl_bits);
__IO_REG32_BIT(MCSPI4_CH0CONF,    					0x480BA02C,__READ_WRITE	,__mcspi_chxconf_bits);
__IO_REG32_BIT(MCSPI4_CH0STAT,    					0x480BA030,__READ				,__mcspi_chxstat_bits);
__IO_REG32_BIT(MCSPI4_CH0CTRL,    					0x480BA034,__READ_WRITE	,__mcspi_chxctrl_bits);
__IO_REG32(		 MCSPI4_TX0,    							0x480BA038,__READ_WRITE	);
__IO_REG32(		 MCSPI4_RX0,    							0x480BA03C,__READ				);
__IO_REG32_BIT(MCSPI4_XFERLEVEL,    				0x480BA07C,__READ_WRITE	,__mcspi_xferlevel_bits);

/***************************************************************************
 **
 ** HDQ
 **
 ***************************************************************************/
__IO_REG32_BIT(HDQ_REVISION,    						0x480B2000,__READ				,__hdq_revision_bits);
__IO_REG32_BIT(HDQ_TX_DATA,    							0x480B2004,__READ_WRITE	,__hdq_tx_data_bits);
__IO_REG32_BIT(HDQ_RX_DATA,    							0x480B2008,__READ				,__hdq_rx_data_bits);
__IO_REG32_BIT(HDQ_CTRL_STATUS,    					0x480B200C,__READ_WRITE	,__hdq_ctrl_status_bits);
__IO_REG32_BIT(HDQ_INT_STATUS,    					0x480B2010,__READ				,__hdq_int_status_bits);
__IO_REG32_BIT(HDQ_SYSCONFIG,    						0x480B2014,__READ_WRITE	,__hdq_sysconfig_bits);
__IO_REG32_BIT(HDQ_SYSSTATUS,    						0x480B2018,__READ				,__hdq_sysstatus_bits);

/***************************************************************************
 **
 ** McBSP1
 **
 ***************************************************************************/
__IO_REG32(		 MCBSPLP1_DRR_REG,    				0x48074000,__READ				);
__IO_REG32(		 MCBSPLP1_DXR_REG,    				0x48074008,__WRITE			);
__IO_REG32_BIT(MCBSPLP1_SPCR2_REG,    			0x48074010,__READ_WRITE	,__mcbsplp_spcr2_reg_bits);
__IO_REG32_BIT(MCBSPLP1_SPCR1_REG,    			0x48074014,__READ_WRITE	,__mcbsplp_spcr1_reg_bits);
__IO_REG32_BIT(MCBSPLP1_RCR2_REG,    				0x48074018,__READ_WRITE	,__mcbsplp_rcr2_reg_bits);
__IO_REG32_BIT(MCBSPLP1_RCR1_REG,    				0x4807401C,__READ_WRITE	,__mcbsplp_rcr1_reg_bits);
__IO_REG32_BIT(MCBSPLP1_XCR2_REG,    				0x48074020,__READ_WRITE	,__mcbsplp_xcr2_reg_bits);
__IO_REG32_BIT(MCBSPLP1_XCR1_REG,    				0x48074024,__READ_WRITE	,__mcbsplp_xcr1_reg_bits);
__IO_REG32_BIT(MCBSPLP1_SRGR2_REG,    			0x48074028,__READ_WRITE	,__mcbsplp_srgr2_reg_bits);
__IO_REG32_BIT(MCBSPLP1_SRGR1_REG,    			0x4807402C,__READ_WRITE	,__mcbsplp_srgr1_reg_bits);
__IO_REG32_BIT(MCBSPLP1_MCR2_REG,    				0x48074030,__READ_WRITE	,__mcbsplp_mcr2_reg_bits);
__IO_REG32_BIT(MCBSPLP1_MCR1_REG,    				0x48074034,__READ_WRITE	,__mcbsplp_mcr1_reg_bits);
__IO_REG32_BIT(MCBSPLP1_RCERA_REG,    			0x48074038,__READ_WRITE	,__mcbsplp_rcera_reg_bits);
__IO_REG32_BIT(MCBSPLP1_RCERB_REG,    			0x4807403C,__READ_WRITE	,__mcbsplp_rcerb_reg_bits);
__IO_REG32_BIT(MCBSPLP1_XCERA_REG,    			0x48074040,__READ_WRITE	,__mcbsplp_xcera_reg_bits);
__IO_REG32_BIT(MCBSPLP1_XCERB_REG,    			0x48074044,__READ_WRITE	,__mcbsplp_xcerb_reg_bits);
__IO_REG32_BIT(MCBSPLP1_PCR_REG,    				0x48074048,__READ_WRITE	,__mcbsplp_pcr_reg_bits);
__IO_REG32_BIT(MCBSPLP1_RCERC_REG,    			0x4807404C,__READ_WRITE	,__mcbsplp_rcerc_reg_bits);
__IO_REG32_BIT(MCBSPLP1_RCERD_REG,    			0x48074050,__READ_WRITE	,__mcbsplp_rcerd_reg_bits);
__IO_REG32_BIT(MCBSPLP1_XCERC_REG,    			0x48074054,__READ_WRITE	,__mcbsplp_xcerc_reg_bits);
__IO_REG32_BIT(MCBSPLP1_XCERD_REG,    			0x48074058,__READ_WRITE	,__mcbsplp_xcerd_reg_bits);
__IO_REG32_BIT(MCBSPLP1_RCERE_REG,    			0x4807405C,__READ_WRITE	,__mcbsplp_rcere_reg_bits);
__IO_REG32_BIT(MCBSPLP1_RCERF_REG,    			0x48074060,__READ_WRITE	,__mcbsplp_rcerf_reg_bits);
__IO_REG32_BIT(MCBSPLP1_XCERE_REG,    			0x48074064,__READ_WRITE	,__mcbsplp_xcere_reg_bits);
__IO_REG32_BIT(MCBSPLP1_XCERF_REG,    			0x48074068,__READ_WRITE	,__mcbsplp_xcerf_reg_bits);
__IO_REG32_BIT(MCBSPLP1_RCERG_REG,    			0x4807406C,__READ_WRITE	,__mcbsplp_rcerg_reg_bits);
__IO_REG32_BIT(MCBSPLP1_RCERH_REG,    			0x48074070,__READ_WRITE	,__mcbsplp_rcerh_reg_bits);
__IO_REG32_BIT(MCBSPLP1_XCERG_REG,    			0x48074074,__READ_WRITE	,__mcbsplp_xcerg_reg_bits);
__IO_REG32_BIT(MCBSPLP1_XCERH_REG,    			0x48074078,__READ_WRITE	,__mcbsplp_xcerh_reg_bits);
__IO_REG32_BIT(MCBSPLP1_REV_REG,    				0x4807407C,__READ				,__mcbsplp_rev_reg_bits);
__IO_REG32(		 MCBSPLP1_RINTCLR_REG,    		0x48074080,__READ_WRITE	);
__IO_REG32(		 MCBSPLP1_XINTCLR_REG,    		0x48074084,__READ_WRITE	);
__IO_REG32(		 MCBSPLP1_ROVFLCLR_REG,    		0x48074088,__READ_WRITE	);
__IO_REG32_BIT(MCBSPLP1_SYSCONFIG_REG,    	0x4807408C,__READ_WRITE	,__mcbsplp_sysconfig_reg_bits);
__IO_REG32_BIT(MCBSPLP1_THRSH2_REG,    			0x48074090,__READ_WRITE	,__mcbsplp_thrsh2_reg_bits);
__IO_REG32_BIT(MCBSPLP1_THRSH1_REG,    			0x48074094,__READ_WRITE	,__mcbsplp_thrsh1_reg_bits);
__IO_REG32_BIT(MCBSPLP1_IRQSTATUS_REG,    	0x480740A0,__READ_WRITE	,__mcbsplp_irqstatus_reg_bits);
__IO_REG32_BIT(MCBSPLP1_IRQENABLE_REG,    	0x480740A4,__READ_WRITE	,__mcbsplp_irqenable_reg_bits);
__IO_REG32_BIT(MCBSPLP1_WAKEUPEN_REG,    		0x480740A8,__READ_WRITE	,__mcbsplp_wakeupen_reg_bits);
__IO_REG32_BIT(MCBSPLP1_XCCR_REG,    				0x480740AC,__READ_WRITE	,__mcbsplp_xccr_reg_bits);
__IO_REG32_BIT(MCBSPLP1_RCCR_REG,    				0x480740B0,__READ_WRITE	,__mcbsplp_rccr_reg_bits);
__IO_REG32_BIT(MCBSPLP1_XBUFFSTAT_REG,    	0x480740B4,__READ				,__mcbsplp_xbuffstat_reg_bits);
__IO_REG32_BIT(MCBSPLP1_RBUFFSTAT_REG,    	0x480740B8,__READ				,__mcbsplp_rbuffstat_reg_bits);
__IO_REG32_BIT(MCBSPLP1_SSELCR_REG,    			0x480740BC,__READ_WRITE	,__mcbsplp_sselcr_reg_bits);
__IO_REG32_BIT(MCBSPLP1_STATUS_REG,    			0x480740C0,__READ_WRITE	,__mcbsplp_status_reg_bits);

/***************************************************************************
 **
 ** McBSP5
 **
 ***************************************************************************/
__IO_REG32(		 MCBSPLP5_DRR_REG,    				0x48096000,__READ				);
__IO_REG32(		 MCBSPLP5_DXR_REG,    				0x48096008,__WRITE			);
__IO_REG32_BIT(MCBSPLP5_SPCR2_REG,    			0x48096010,__READ_WRITE	,__mcbsplp_spcr2_reg_bits);
__IO_REG32_BIT(MCBSPLP5_SPCR1_REG,    			0x48096014,__READ_WRITE	,__mcbsplp_spcr1_reg_bits);
__IO_REG32_BIT(MCBSPLP5_RCR2_REG,    				0x48096018,__READ_WRITE	,__mcbsplp_rcr2_reg_bits);
__IO_REG32_BIT(MCBSPLP5_RCR1_REG,    				0x4809601C,__READ_WRITE	,__mcbsplp_rcr1_reg_bits);
__IO_REG32_BIT(MCBSPLP5_XCR2_REG,    				0x48096020,__READ_WRITE	,__mcbsplp_xcr2_reg_bits);
__IO_REG32_BIT(MCBSPLP5_XCR1_REG,    				0x48096024,__READ_WRITE	,__mcbsplp_xcr1_reg_bits);
__IO_REG32_BIT(MCBSPLP5_SRGR2_REG,    			0x48096028,__READ_WRITE	,__mcbsplp_srgr2_reg_bits);
__IO_REG32_BIT(MCBSPLP5_SRGR1_REG,    			0x4809602C,__READ_WRITE	,__mcbsplp_srgr1_reg_bits);
__IO_REG32_BIT(MCBSPLP5_MCR2_REG,    				0x48096030,__READ_WRITE	,__mcbsplp_mcr2_reg_bits);
__IO_REG32_BIT(MCBSPLP5_MCR1_REG,    				0x48096034,__READ_WRITE	,__mcbsplp_mcr1_reg_bits);
__IO_REG32_BIT(MCBSPLP5_RCERA_REG,    			0x48096038,__READ_WRITE	,__mcbsplp_rcera_reg_bits);
__IO_REG32_BIT(MCBSPLP5_RCERB_REG,    			0x4809603C,__READ_WRITE	,__mcbsplp_rcerb_reg_bits);
__IO_REG32_BIT(MCBSPLP5_XCERA_REG,    			0x48096040,__READ_WRITE	,__mcbsplp_xcera_reg_bits);
__IO_REG32_BIT(MCBSPLP5_XCERB_REG,    			0x48096044,__READ_WRITE	,__mcbsplp_xcerb_reg_bits);
__IO_REG32_BIT(MCBSPLP5_PCR_REG,    				0x48096048,__READ_WRITE	,__mcbsplp_pcr_reg_bits);
__IO_REG32_BIT(MCBSPLP5_RCERC_REG,    			0x4809604C,__READ_WRITE	,__mcbsplp_rcerc_reg_bits);
__IO_REG32_BIT(MCBSPLP5_RCERD_REG,    			0x48096050,__READ_WRITE	,__mcbsplp_rcerd_reg_bits);
__IO_REG32_BIT(MCBSPLP5_XCERC_REG,    			0x48096054,__READ_WRITE	,__mcbsplp_xcerc_reg_bits);
__IO_REG32_BIT(MCBSPLP5_XCERD_REG,    			0x48096058,__READ_WRITE	,__mcbsplp_xcerd_reg_bits);
__IO_REG32_BIT(MCBSPLP5_RCERE_REG,    			0x4809605C,__READ_WRITE	,__mcbsplp_rcere_reg_bits);
__IO_REG32_BIT(MCBSPLP5_RCERF_REG,    			0x48096060,__READ_WRITE	,__mcbsplp_rcerf_reg_bits);
__IO_REG32_BIT(MCBSPLP5_XCERE_REG,    			0x48096064,__READ_WRITE	,__mcbsplp_xcere_reg_bits);
__IO_REG32_BIT(MCBSPLP5_XCERF_REG,    			0x48096068,__READ_WRITE	,__mcbsplp_xcerf_reg_bits);
__IO_REG32_BIT(MCBSPLP5_RCERG_REG,    			0x4809606C,__READ_WRITE	,__mcbsplp_rcerg_reg_bits);
__IO_REG32_BIT(MCBSPLP5_RCERH_REG,    			0x48096070,__READ_WRITE	,__mcbsplp_rcerh_reg_bits);
__IO_REG32_BIT(MCBSPLP5_XCERG_REG,    			0x48096074,__READ_WRITE	,__mcbsplp_xcerg_reg_bits);
__IO_REG32_BIT(MCBSPLP5_XCERH_REG,    			0x48096078,__READ_WRITE	,__mcbsplp_xcerh_reg_bits);
__IO_REG32_BIT(MCBSPLP5_REV_REG,    				0x4809607C,__READ				,__mcbsplp_rev_reg_bits);
__IO_REG32(		 MCBSPLP5_RINTCLR_REG,    		0x48096080,__READ_WRITE	);
__IO_REG32(		 MCBSPLP5_XINTCLR_REG,    		0x48096084,__READ_WRITE	);
__IO_REG32(		 MCBSPLP5_ROVFLCLR_REG,    		0x48096088,__READ_WRITE	);
__IO_REG32_BIT(MCBSPLP5_SYSCONFIG_REG,    	0x4809608C,__READ_WRITE	,__mcbsplp_sysconfig_reg_bits);
__IO_REG32_BIT(MCBSPLP5_THRSH2_REG,    			0x48096090,__READ_WRITE	,__mcbsplp_thrsh2_reg_bits);
__IO_REG32_BIT(MCBSPLP5_THRSH1_REG,    			0x48096094,__READ_WRITE	,__mcbsplp_thrsh1_reg_bits);
__IO_REG32_BIT(MCBSPLP5_IRQSTATUS_REG,    	0x480960A0,__READ_WRITE	,__mcbsplp_irqstatus_reg_bits);
__IO_REG32_BIT(MCBSPLP5_IRQENABLE_REG,    	0x480960A4,__READ_WRITE	,__mcbsplp_irqenable_reg_bits);
__IO_REG32_BIT(MCBSPLP5_WAKEUPEN_REG,    		0x480960A8,__READ_WRITE	,__mcbsplp_wakeupen_reg_bits);
__IO_REG32_BIT(MCBSPLP5_XCCR_REG,    				0x480960AC,__READ_WRITE	,__mcbsplp_xccr_reg_bits);
__IO_REG32_BIT(MCBSPLP5_RCCR_REG,    				0x480960B0,__READ_WRITE	,__mcbsplp_rccr_reg_bits);
__IO_REG32_BIT(MCBSPLP5_XBUFFSTAT_REG,    	0x480960B4,__READ				,__mcbsplp_xbuffstat_reg_bits);
__IO_REG32_BIT(MCBSPLP5_RBUFFSTAT_REG,    	0x480960B8,__READ				,__mcbsplp_rbuffstat_reg_bits);
__IO_REG32_BIT(MCBSPLP5_SSELCR_REG,    			0x480960BC,__READ_WRITE	,__mcbsplp_sselcr_reg_bits);
__IO_REG32_BIT(MCBSPLP5_STATUS_REG,    			0x480960C0,__READ_WRITE	,__mcbsplp_status_reg_bits);

/***************************************************************************
 **
 ** McBSP2
 **
 ***************************************************************************/
__IO_REG32(		 MCBSPLP2_DRR_REG,    				0x49022000,__READ				);
__IO_REG32(		 MCBSPLP2_DXR_REG,    				0x49022008,__WRITE			);
__IO_REG32_BIT(MCBSPLP2_SPCR2_REG,    			0x49022010,__READ_WRITE	,__mcbsplp_spcr2_reg_bits);
__IO_REG32_BIT(MCBSPLP2_SPCR1_REG,    			0x49022014,__READ_WRITE	,__mcbsplp_spcr1_reg_bits);
__IO_REG32_BIT(MCBSPLP2_RCR2_REG,    				0x49022018,__READ_WRITE	,__mcbsplp_rcr2_reg_bits);
__IO_REG32_BIT(MCBSPLP2_RCR1_REG,    				0x4902201C,__READ_WRITE	,__mcbsplp_rcr1_reg_bits);
__IO_REG32_BIT(MCBSPLP2_XCR2_REG,    				0x49022020,__READ_WRITE	,__mcbsplp_xcr2_reg_bits);
__IO_REG32_BIT(MCBSPLP2_XCR1_REG,    				0x49022024,__READ_WRITE	,__mcbsplp_xcr1_reg_bits);
__IO_REG32_BIT(MCBSPLP2_SRGR2_REG,    			0x49022028,__READ_WRITE	,__mcbsplp_srgr2_reg_bits);
__IO_REG32_BIT(MCBSPLP2_SRGR1_REG,    			0x4902202C,__READ_WRITE	,__mcbsplp_srgr1_reg_bits);
__IO_REG32_BIT(MCBSPLP2_MCR2_REG,    				0x49022030,__READ_WRITE	,__mcbsplp_mcr2_reg_bits);
__IO_REG32_BIT(MCBSPLP2_MCR1_REG,    				0x49022034,__READ_WRITE	,__mcbsplp_mcr1_reg_bits);
__IO_REG32_BIT(MCBSPLP2_RCERA_REG,    			0x49022038,__READ_WRITE	,__mcbsplp_rcera_reg_bits);
__IO_REG32_BIT(MCBSPLP2_RCERB_REG,    			0x4902203C,__READ_WRITE	,__mcbsplp_rcerb_reg_bits);
__IO_REG32_BIT(MCBSPLP2_XCERA_REG,    			0x49022040,__READ_WRITE	,__mcbsplp_xcera_reg_bits);
__IO_REG32_BIT(MCBSPLP2_XCERB_REG,    			0x49022044,__READ_WRITE	,__mcbsplp_xcerb_reg_bits);
__IO_REG32_BIT(MCBSPLP2_PCR_REG,    				0x49022048,__READ_WRITE	,__mcbsplp_pcr_reg_bits);
__IO_REG32_BIT(MCBSPLP2_RCERC_REG,    			0x4902204C,__READ_WRITE	,__mcbsplp_rcerc_reg_bits);
__IO_REG32_BIT(MCBSPLP2_RCERD_REG,    			0x49022050,__READ_WRITE	,__mcbsplp_rcerd_reg_bits);
__IO_REG32_BIT(MCBSPLP2_XCERC_REG,    			0x49022054,__READ_WRITE	,__mcbsplp_xcerc_reg_bits);
__IO_REG32_BIT(MCBSPLP2_XCERD_REG,    			0x49022058,__READ_WRITE	,__mcbsplp_xcerd_reg_bits);
__IO_REG32_BIT(MCBSPLP2_RCERE_REG,    			0x4902205C,__READ_WRITE	,__mcbsplp_rcere_reg_bits);
__IO_REG32_BIT(MCBSPLP2_RCERF_REG,    			0x49022060,__READ_WRITE	,__mcbsplp_rcerf_reg_bits);
__IO_REG32_BIT(MCBSPLP2_XCERE_REG,    			0x49022064,__READ_WRITE	,__mcbsplp_xcere_reg_bits);
__IO_REG32_BIT(MCBSPLP2_XCERF_REG,    			0x49022068,__READ_WRITE	,__mcbsplp_xcerf_reg_bits);
__IO_REG32_BIT(MCBSPLP2_RCERG_REG,    			0x4902206C,__READ_WRITE	,__mcbsplp_rcerg_reg_bits);
__IO_REG32_BIT(MCBSPLP2_RCERH_REG,    			0x49022070,__READ_WRITE	,__mcbsplp_rcerh_reg_bits);
__IO_REG32_BIT(MCBSPLP2_XCERG_REG,    			0x49022074,__READ_WRITE	,__mcbsplp_xcerg_reg_bits);
__IO_REG32_BIT(MCBSPLP2_XCERH_REG,    			0x49022078,__READ_WRITE	,__mcbsplp_xcerh_reg_bits);
__IO_REG32_BIT(MCBSPLP2_REV_REG,    				0x4902207C,__READ				,__mcbsplp_rev_reg_bits);
__IO_REG32(		 MCBSPLP2_RINTCLR_REG,    		0x49022080,__READ_WRITE	);
__IO_REG32(		 MCBSPLP2_XINTCLR_REG,    		0x49022084,__READ_WRITE	);
__IO_REG32(		 MCBSPLP2_ROVFLCLR_REG,    		0x49022088,__READ_WRITE	);
__IO_REG32_BIT(MCBSPLP2_SYSCONFIG_REG,    	0x4902208C,__READ_WRITE	,__mcbsplp_sysconfig_reg_bits);
__IO_REG32_BIT(MCBSPLP2_THRSH2_REG,    			0x49022090,__READ_WRITE	,__mcbsplp_thrsh2_reg_bits);
__IO_REG32_BIT(MCBSPLP2_THRSH1_REG,    			0x49022094,__READ_WRITE	,__mcbsplp_thrsh1_reg_bits);
__IO_REG32_BIT(MCBSPLP2_IRQSTATUS_REG,    	0x490220A0,__READ_WRITE	,__mcbsplp_irqstatus_reg_bits);
__IO_REG32_BIT(MCBSPLP2_IRQENABLE_REG,    	0x490220A4,__READ_WRITE	,__mcbsplp_irqenable_reg_bits);
__IO_REG32_BIT(MCBSPLP2_WAKEUPEN_REG,    		0x490220A8,__READ_WRITE	,__mcbsplp_wakeupen_reg_bits);
__IO_REG32_BIT(MCBSPLP2_XCCR_REG,    				0x490220AC,__READ_WRITE	,__mcbsplp_xccr_reg_bits);
__IO_REG32_BIT(MCBSPLP2_RCCR_REG,    				0x490220B0,__READ_WRITE	,__mcbsplp_rccr_reg_bits);
__IO_REG32_BIT(MCBSPLP2_XBUFFSTAT_REG,    	0x490220B4,__READ				,__mcbsplp_xbuffstat_reg_bits);
__IO_REG32_BIT(MCBSPLP2_RBUFFSTAT_REG,    	0x490220B8,__READ				,__mcbsplp_rbuffstat_reg_bits);
__IO_REG32_BIT(MCBSPLP2_SSELCR_REG,    			0x490220BC,__READ_WRITE	,__mcbsplp_sselcr_reg_bits);
__IO_REG32_BIT(MCBSPLP2_STATUS_REG,    			0x490220C0,__READ_WRITE	,__mcbsplp_status_reg_bits);
__IO_REG32_BIT(ST2_REV_REG,    							0x49028000,__READ				,__st_rev_reg_bits);
__IO_REG32_BIT(ST2_SYSCONFIG_REG,    				0x49028010,__READ_WRITE	,__st_sysconfig_reg_bits);
__IO_REG32_BIT(ST2_IRQSTATUS_REG,    				0x49028018,__READ_WRITE	,__st_irqstatus_reg_bits);
__IO_REG32_BIT(ST2_IRQENABLE_REG,    				0x4902801C,__READ_WRITE	,__st_irqenable_reg_bits);
__IO_REG32_BIT(ST2_SGAINCR_REG,    					0x49028024,__READ_WRITE	,__st_sgaincr_reg_bits);
__IO_REG32_BIT(ST2_SFIRCR_REG,    					0x49028028,__READ_WRITE	,__st_sfircr_reg_bits);
__IO_REG32_BIT(ST2_SSELCR_REG,    					0x4902802C,__READ_WRITE	,__st_sselcr_reg_bits);

/***************************************************************************
 **
 ** McBSP3
 **
 ***************************************************************************/
__IO_REG32(		 MCBSPLP3_DRR_REG,    				0x49024000,__READ				);
__IO_REG32(		 MCBSPLP3_DXR_REG,    				0x49024008,__WRITE			);
__IO_REG32_BIT(MCBSPLP3_SPCR2_REG,    			0x49024010,__READ_WRITE	,__mcbsplp_spcr2_reg_bits);
__IO_REG32_BIT(MCBSPLP3_SPCR1_REG,    			0x49024014,__READ_WRITE	,__mcbsplp_spcr1_reg_bits);
__IO_REG32_BIT(MCBSPLP3_RCR2_REG,    				0x49024018,__READ_WRITE	,__mcbsplp_rcr2_reg_bits);
__IO_REG32_BIT(MCBSPLP3_RCR1_REG,    				0x4902401C,__READ_WRITE	,__mcbsplp_rcr1_reg_bits);
__IO_REG32_BIT(MCBSPLP3_XCR2_REG,    				0x49024020,__READ_WRITE	,__mcbsplp_xcr2_reg_bits);
__IO_REG32_BIT(MCBSPLP3_XCR1_REG,    				0x49024024,__READ_WRITE	,__mcbsplp_xcr1_reg_bits);
__IO_REG32_BIT(MCBSPLP3_SRGR2_REG,    			0x49024028,__READ_WRITE	,__mcbsplp_srgr2_reg_bits);
__IO_REG32_BIT(MCBSPLP3_SRGR1_REG,    			0x4902402C,__READ_WRITE	,__mcbsplp_srgr1_reg_bits);
__IO_REG32_BIT(MCBSPLP3_MCR2_REG,    				0x49024030,__READ_WRITE	,__mcbsplp_mcr2_reg_bits);
__IO_REG32_BIT(MCBSPLP3_MCR1_REG,    				0x49024034,__READ_WRITE	,__mcbsplp_mcr1_reg_bits);
__IO_REG32_BIT(MCBSPLP3_RCERA_REG,    			0x49024038,__READ_WRITE	,__mcbsplp_rcera_reg_bits);
__IO_REG32_BIT(MCBSPLP3_RCERB_REG,    			0x4902403C,__READ_WRITE	,__mcbsplp_rcerb_reg_bits);
__IO_REG32_BIT(MCBSPLP3_XCERA_REG,    			0x49024040,__READ_WRITE	,__mcbsplp_xcera_reg_bits);
__IO_REG32_BIT(MCBSPLP3_XCERB_REG,    			0x49024044,__READ_WRITE	,__mcbsplp_xcerb_reg_bits);
__IO_REG32_BIT(MCBSPLP3_PCR_REG,    				0x49024048,__READ_WRITE	,__mcbsplp_pcr_reg_bits);
__IO_REG32_BIT(MCBSPLP3_RCERC_REG,    			0x4902404C,__READ_WRITE	,__mcbsplp_rcerc_reg_bits);
__IO_REG32_BIT(MCBSPLP3_RCERD_REG,    			0x49024050,__READ_WRITE	,__mcbsplp_rcerd_reg_bits);
__IO_REG32_BIT(MCBSPLP3_XCERC_REG,    			0x49024054,__READ_WRITE	,__mcbsplp_xcerc_reg_bits);
__IO_REG32_BIT(MCBSPLP3_XCERD_REG,    			0x49024058,__READ_WRITE	,__mcbsplp_xcerd_reg_bits);
__IO_REG32_BIT(MCBSPLP3_RCERE_REG,    			0x4902405C,__READ_WRITE	,__mcbsplp_rcere_reg_bits);
__IO_REG32_BIT(MCBSPLP3_RCERF_REG,    			0x49024060,__READ_WRITE	,__mcbsplp_rcerf_reg_bits);
__IO_REG32_BIT(MCBSPLP3_XCERE_REG,    			0x49024064,__READ_WRITE	,__mcbsplp_xcere_reg_bits);
__IO_REG32_BIT(MCBSPLP3_XCERF_REG,    			0x49024068,__READ_WRITE	,__mcbsplp_xcerf_reg_bits);
__IO_REG32_BIT(MCBSPLP3_RCERG_REG,    			0x4902406C,__READ_WRITE	,__mcbsplp_rcerg_reg_bits);
__IO_REG32_BIT(MCBSPLP3_RCERH_REG,    			0x49024070,__READ_WRITE	,__mcbsplp_rcerh_reg_bits);
__IO_REG32_BIT(MCBSPLP3_XCERG_REG,    			0x49024074,__READ_WRITE	,__mcbsplp_xcerg_reg_bits);
__IO_REG32_BIT(MCBSPLP3_XCERH_REG,    			0x49024078,__READ_WRITE	,__mcbsplp_xcerh_reg_bits);
__IO_REG32_BIT(MCBSPLP3_REV_REG,    				0x4902407C,__READ				,__mcbsplp_rev_reg_bits);
__IO_REG32(		 MCBSPLP3_RINTCLR_REG,    		0x49024080,__READ_WRITE	);
__IO_REG32(		 MCBSPLP3_XINTCLR_REG,    		0x49024084,__READ_WRITE	);
__IO_REG32(		 MCBSPLP3_ROVFLCLR_REG,    		0x49024088,__READ_WRITE	);
__IO_REG32_BIT(MCBSPLP3_SYSCONFIG_REG,    	0x4902408C,__READ_WRITE	,__mcbsplp_sysconfig_reg_bits);
__IO_REG32_BIT(MCBSPLP3_THRSH2_REG,    			0x49024090,__READ_WRITE	,__mcbsplp_thrsh2_reg_bits);
__IO_REG32_BIT(MCBSPLP3_THRSH1_REG,    			0x49024094,__READ_WRITE	,__mcbsplp_thrsh1_reg_bits);
__IO_REG32_BIT(MCBSPLP3_IRQSTATUS_REG,    	0x490240A0,__READ_WRITE	,__mcbsplp_irqstatus_reg_bits);
__IO_REG32_BIT(MCBSPLP3_IRQENABLE_REG,    	0x490240A4,__READ_WRITE	,__mcbsplp_irqenable_reg_bits);
__IO_REG32_BIT(MCBSPLP3_WAKEUPEN_REG,    		0x490240A8,__READ_WRITE	,__mcbsplp_wakeupen_reg_bits);
__IO_REG32_BIT(MCBSPLP3_XCCR_REG,    				0x490240AC,__READ_WRITE	,__mcbsplp_xccr_reg_bits);
__IO_REG32_BIT(MCBSPLP3_RCCR_REG,    				0x490240B0,__READ_WRITE	,__mcbsplp_rccr_reg_bits);
__IO_REG32_BIT(MCBSPLP3_XBUFFSTAT_REG,    	0x490240B4,__READ				,__mcbsplp_xbuffstat_reg_bits);
__IO_REG32_BIT(MCBSPLP3_RBUFFSTAT_REG,    	0x490240B8,__READ				,__mcbsplp_rbuffstat_reg_bits);
__IO_REG32_BIT(MCBSPLP3_SSELCR_REG,    			0x490240BC,__READ_WRITE	,__mcbsplp_sselcr_reg_bits);
__IO_REG32_BIT(MCBSPLP3_STATUS_REG,    			0x490240C0,__READ_WRITE	,__mcbsplp_status_reg_bits);
__IO_REG32_BIT(ST3_REV_REG,    							0x4902A000,__READ				,__st_rev_reg_bits);
__IO_REG32_BIT(ST3_SYSCONFIG_REG,    				0x4902A010,__READ_WRITE	,__st_sysconfig_reg_bits);
__IO_REG32_BIT(ST3_IRQSTATUS_REG,    				0x4902A018,__READ_WRITE	,__st_irqstatus_reg_bits);
__IO_REG32_BIT(ST3_IRQENABLE_REG,    				0x4902A01C,__READ_WRITE	,__st_irqenable_reg_bits);
__IO_REG32_BIT(ST3_SGAINCR_REG,    					0x4902A024,__READ_WRITE	,__st_sgaincr_reg_bits);
__IO_REG32_BIT(ST3_SFIRCR_REG,    					0x4902A028,__READ_WRITE	,__st_sfircr_reg_bits);
__IO_REG32_BIT(ST3_SSELCR_REG,    					0x4902A02C,__READ_WRITE	,__st_sselcr_reg_bits);

/***************************************************************************
 **
 ** McBSP4
 **
 ***************************************************************************/
__IO_REG32(		 MCBSPLP4_DRR_REG,    				0x49026000,__READ				);
__IO_REG32(		 MCBSPLP4_DXR_REG,    				0x49026008,__WRITE			);
__IO_REG32_BIT(MCBSPLP4_SPCR2_REG,    			0x49026010,__READ_WRITE	,__mcbsplp_spcr2_reg_bits);
__IO_REG32_BIT(MCBSPLP4_SPCR1_REG,    			0x49026014,__READ_WRITE	,__mcbsplp_spcr1_reg_bits);
__IO_REG32_BIT(MCBSPLP4_RCR2_REG,    				0x49026018,__READ_WRITE	,__mcbsplp_rcr2_reg_bits);
__IO_REG32_BIT(MCBSPLP4_RCR1_REG,    				0x4902601C,__READ_WRITE	,__mcbsplp_rcr1_reg_bits);
__IO_REG32_BIT(MCBSPLP4_XCR2_REG,    				0x49026020,__READ_WRITE	,__mcbsplp_xcr2_reg_bits);
__IO_REG32_BIT(MCBSPLP4_XCR1_REG,    				0x49026024,__READ_WRITE	,__mcbsplp_xcr1_reg_bits);
__IO_REG32_BIT(MCBSPLP4_SRGR2_REG,    			0x49026028,__READ_WRITE	,__mcbsplp_srgr2_reg_bits);
__IO_REG32_BIT(MCBSPLP4_SRGR1_REG,    			0x4902602C,__READ_WRITE	,__mcbsplp_srgr1_reg_bits);
__IO_REG32_BIT(MCBSPLP4_MCR2_REG,    				0x49026030,__READ_WRITE	,__mcbsplp_mcr2_reg_bits);
__IO_REG32_BIT(MCBSPLP4_MCR1_REG,    				0x49026034,__READ_WRITE	,__mcbsplp_mcr1_reg_bits);
__IO_REG32_BIT(MCBSPLP4_RCERA_REG,    			0x49026038,__READ_WRITE	,__mcbsplp_rcera_reg_bits);
__IO_REG32_BIT(MCBSPLP4_RCERB_REG,    			0x4902603C,__READ_WRITE	,__mcbsplp_rcerb_reg_bits);
__IO_REG32_BIT(MCBSPLP4_XCERA_REG,    			0x49026040,__READ_WRITE	,__mcbsplp_xcera_reg_bits);
__IO_REG32_BIT(MCBSPLP4_XCERB_REG,    			0x49026044,__READ_WRITE	,__mcbsplp_xcerb_reg_bits);
__IO_REG32_BIT(MCBSPLP4_PCR_REG,    				0x49026048,__READ_WRITE	,__mcbsplp_pcr_reg_bits);
__IO_REG32_BIT(MCBSPLP4_RCERC_REG,    			0x4902604C,__READ_WRITE	,__mcbsplp_rcerc_reg_bits);
__IO_REG32_BIT(MCBSPLP4_RCERD_REG,    			0x49026050,__READ_WRITE	,__mcbsplp_rcerd_reg_bits);
__IO_REG32_BIT(MCBSPLP4_XCERC_REG,    			0x49026054,__READ_WRITE	,__mcbsplp_xcerc_reg_bits);
__IO_REG32_BIT(MCBSPLP4_XCERD_REG,    			0x49026058,__READ_WRITE	,__mcbsplp_xcerd_reg_bits);
__IO_REG32_BIT(MCBSPLP4_RCERE_REG,    			0x4902605C,__READ_WRITE	,__mcbsplp_rcere_reg_bits);
__IO_REG32_BIT(MCBSPLP4_RCERF_REG,    			0x49026060,__READ_WRITE	,__mcbsplp_rcerf_reg_bits);
__IO_REG32_BIT(MCBSPLP4_XCERE_REG,    			0x49026064,__READ_WRITE	,__mcbsplp_xcere_reg_bits);
__IO_REG32_BIT(MCBSPLP4_XCERF_REG,    			0x49026068,__READ_WRITE	,__mcbsplp_xcerf_reg_bits);
__IO_REG32_BIT(MCBSPLP4_RCERG_REG,    			0x4902606C,__READ_WRITE	,__mcbsplp_rcerg_reg_bits);
__IO_REG32_BIT(MCBSPLP4_RCERH_REG,    			0x49026070,__READ_WRITE	,__mcbsplp_rcerh_reg_bits);
__IO_REG32_BIT(MCBSPLP4_XCERG_REG,    			0x49026074,__READ_WRITE	,__mcbsplp_xcerg_reg_bits);
__IO_REG32_BIT(MCBSPLP4_XCERH_REG,    			0x49026078,__READ_WRITE	,__mcbsplp_xcerh_reg_bits);
__IO_REG32_BIT(MCBSPLP4_REV_REG,    				0x4902607C,__READ				,__mcbsplp_rev_reg_bits);
__IO_REG32(		 MCBSPLP4_RINTCLR_REG,    		0x49026080,__READ_WRITE	);
__IO_REG32(		 MCBSPLP4_XINTCLR_REG,    		0x49026084,__READ_WRITE	);
__IO_REG32(		 MCBSPLP4_ROVFLCLR_REG,    		0x49026088,__READ_WRITE	);
__IO_REG32_BIT(MCBSPLP4_SYSCONFIG_REG,    	0x4902608C,__READ_WRITE	,__mcbsplp_sysconfig_reg_bits);
__IO_REG32_BIT(MCBSPLP4_THRSH2_REG,    			0x49026090,__READ_WRITE	,__mcbsplp_thrsh2_reg_bits);
__IO_REG32_BIT(MCBSPLP4_THRSH1_REG,    			0x49026094,__READ_WRITE	,__mcbsplp_thrsh1_reg_bits);
__IO_REG32_BIT(MCBSPLP4_IRQSTATUS_REG,    	0x490260A0,__READ_WRITE	,__mcbsplp_irqstatus_reg_bits);
__IO_REG32_BIT(MCBSPLP4_IRQENABLE_REG,    	0x490260A4,__READ_WRITE	,__mcbsplp_irqenable_reg_bits);
__IO_REG32_BIT(MCBSPLP4_WAKEUPEN_REG,    		0x490260A8,__READ_WRITE	,__mcbsplp_wakeupen_reg_bits);
__IO_REG32_BIT(MCBSPLP4_XCCR_REG,    				0x490260AC,__READ_WRITE	,__mcbsplp_xccr_reg_bits);
__IO_REG32_BIT(MCBSPLP4_RCCR_REG,    				0x490260B0,__READ_WRITE	,__mcbsplp_rccr_reg_bits);
__IO_REG32_BIT(MCBSPLP4_XBUFFSTAT_REG,    	0x490260B4,__READ				,__mcbsplp_xbuffstat_reg_bits);
__IO_REG32_BIT(MCBSPLP4_RBUFFSTAT_REG,    	0x490260B8,__READ				,__mcbsplp_rbuffstat_reg_bits);
__IO_REG32_BIT(MCBSPLP4_SSELCR_REG,    			0x490260BC,__READ_WRITE	,__mcbsplp_sselcr_reg_bits);
__IO_REG32_BIT(MCBSPLP4_STATUS_REG,    			0x490260C0,__READ_WRITE	,__mcbsplp_status_reg_bits);

/***************************************************************************
 **
 ** MMCHS1
 **
 ***************************************************************************/
__IO_REG32_BIT(MMCHS1_SYSCONFIG,    				0x4809C010,__READ_WRITE	,__mmchs_sysconfig_bits);
__IO_REG32_BIT(MMCHS1_SYSSTATUS,    				0x4809C014,__READ				,__mmchs_sysstatus_bits);
__IO_REG32(		 MMCHS1_CSRE,    							0x4809C024,__READ_WRITE	);
__IO_REG32_BIT(MMCHS1_SYSTEST,    					0x4809C028,__READ_WRITE	,__mmchs_systest_bits);
__IO_REG32_BIT(MMCHS1_CON,    							0x4809C02C,__READ_WRITE	,__mmchs_con_bits);
__IO_REG32_BIT(MMCHS1_PWCNT,    						0x4809C030,__READ_WRITE	,__mmchs_pwcnt_bits);
__IO_REG32_BIT(MMCHS1_BLK,    							0x4809C104,__READ_WRITE	,__mmchs_blk_bits);
__IO_REG32(		 MMCHS1_ARG,    							0x4809C108,__READ_WRITE	);
__IO_REG32_BIT(MMCHS1_CMD,    							0x4809C10C,__READ_WRITE	,__mmchs_cmd_bits);
__IO_REG32_BIT(MMCHS1_RSP10,    						0x4809C110,__READ				,__mmchs_rsp10_bits);
__IO_REG32_BIT(MMCHS1_RSP32,    						0x4809C114,__READ				,__mmchs_rsp32_bits);
__IO_REG32_BIT(MMCHS1_RSP54,    						0x4809C118,__READ				,__mmchs_rsp54_bits);
__IO_REG32_BIT(MMCHS1_RSP76,    						0x4809C11C,__READ				,__mmchs_rsp76_bits);
__IO_REG32(		 MMCHS1_DATA,    							0x4809C120,__READ_WRITE	);
__IO_REG32_BIT(MMCHS1_PSTATE,    						0x4809C124,__READ				,__mmchs_pstate_bits);
__IO_REG32_BIT(MMCHS1_HCTL,    							0x4809C128,__READ_WRITE	,__mmchs_hctl_bits);
__IO_REG32_BIT(MMCHS1_SYSCTL,    						0x4809C12C,__READ_WRITE	,__mmchs_sysctl_bits);
__IO_REG32_BIT(MMCHS1_STAT,    							0x4809C130,__READ_WRITE	,__mmchs_stat_bits);
__IO_REG32_BIT(MMCHS1_IE,    								0x4809C134,__READ_WRITE	,__mmchs_ie_bits);
__IO_REG32_BIT(MMCHS1_ISE,    							0x4809C138,__READ_WRITE	,__mmchs_ise_bits);
__IO_REG32_BIT(MMCHS1_AC12,    							0x4809C13C,__READ				,__mmchs_ac12_bits);
__IO_REG32_BIT(MMCHS1_CAPA,    							0x4809C140,__READ_WRITE	,__mmchs_capa_bits);
__IO_REG32_BIT(MMCHS1_CUR_CAPA,    					0x4809C144,__READ_WRITE	,__mmchs_cur_capa_bits);
__IO_REG32_BIT(MMCHS1_REV,    							0x4809C1FC,__READ				,__mmchs_rev_bits);

/***************************************************************************
 **
 ** MMCHS2
 **
 ***************************************************************************/
__IO_REG32_BIT(MMCHS2_SYSCONFIG,    				0x480B4010,__READ_WRITE	,__mmchs_sysconfig_bits);
__IO_REG32_BIT(MMCHS2_SYSSTATUS,    				0x480B4014,__READ				,__mmchs_sysstatus_bits);
__IO_REG32(		 MMCHS2_CSRE,    							0x480B4024,__READ_WRITE	);
__IO_REG32_BIT(MMCHS2_SYSTEST,    					0x480B4028,__READ_WRITE	,__mmchs_systest_bits);
__IO_REG32_BIT(MMCHS2_CON,    							0x480B402C,__READ_WRITE	,__mmchs_con_bits);
__IO_REG32_BIT(MMCHS2_PWCNT,    						0x480B4030,__READ_WRITE	,__mmchs_pwcnt_bits);
__IO_REG32_BIT(MMCHS2_BLK,    							0x480B4104,__READ_WRITE	,__mmchs_blk_bits);
__IO_REG32(		 MMCHS2_ARG,    							0x480B4108,__READ_WRITE	);
__IO_REG32_BIT(MMCHS2_CMD,    							0x480B410C,__READ_WRITE	,__mmchs_cmd_bits);
__IO_REG32_BIT(MMCHS2_RSP10,    						0x480B4110,__READ				,__mmchs_rsp10_bits);
__IO_REG32_BIT(MMCHS2_RSP32,    						0x480B4114,__READ				,__mmchs_rsp32_bits);
__IO_REG32_BIT(MMCHS2_RSP54,    						0x480B4118,__READ				,__mmchs_rsp54_bits);
__IO_REG32_BIT(MMCHS2_RSP76,    						0x480B411C,__READ				,__mmchs_rsp76_bits);
__IO_REG32(		 MMCHS2_DATA,    							0x480B4120,__READ_WRITE	);
__IO_REG32_BIT(MMCHS2_PSTATE,    						0x480B4124,__READ				,__mmchs_pstate_bits);
__IO_REG32_BIT(MMCHS2_HCTL,    							0x480B4128,__READ_WRITE	,__mmchs_hctl_bits);
__IO_REG32_BIT(MMCHS2_SYSCTL,    						0x480B412C,__READ_WRITE	,__mmchs_sysctl_bits);
__IO_REG32_BIT(MMCHS2_STAT,    							0x480B4130,__READ_WRITE	,__mmchs_stat_bits);
__IO_REG32_BIT(MMCHS2_IE,    								0x480B4134,__READ_WRITE	,__mmchs_ie_bits);
__IO_REG32_BIT(MMCHS2_ISE,    							0x480B4138,__READ_WRITE	,__mmchs_ise_bits);
__IO_REG32_BIT(MMCHS2_AC12,    							0x480B413C,__READ				,__mmchs_ac12_bits);
__IO_REG32_BIT(MMCHS2_CAPA,    							0x480B4140,__READ_WRITE	,__mmchs_capa_bits);
__IO_REG32_BIT(MMCHS2_CUR_CAPA,    					0x480B4144,__READ_WRITE	,__mmchs_cur_capa_bits);
__IO_REG32_BIT(MMCHS2_REV,    							0x480B41FC,__READ				,__mmchs_rev_bits);

/***************************************************************************
 **
 ** MMCHS3
 **
 ***************************************************************************/
__IO_REG32_BIT(MMCHS3_SYSCONFIG,    				0x480AD010,__READ_WRITE	,__mmchs_sysconfig_bits);
__IO_REG32_BIT(MMCHS3_SYSSTATUS,    				0x480AD014,__READ				,__mmchs_sysstatus_bits);
__IO_REG32(		 MMCHS3_CSRE,    							0x480AD024,__READ_WRITE	);
__IO_REG32_BIT(MMCHS3_SYSTEST,    					0x480AD028,__READ_WRITE	,__mmchs_systest_bits);
__IO_REG32_BIT(MMCHS3_CON,    							0x480AD02C,__READ_WRITE	,__mmchs_con_bits);
__IO_REG32_BIT(MMCHS3_PWCNT,    						0x480AD030,__READ_WRITE	,__mmchs_pwcnt_bits);
__IO_REG32_BIT(MMCHS3_BLK,    							0x480AD104,__READ_WRITE	,__mmchs_blk_bits);
__IO_REG32(		 MMCHS3_ARG,    							0x480AD108,__READ_WRITE	);
__IO_REG32_BIT(MMCHS3_CMD,    							0x480AD10C,__READ_WRITE	,__mmchs_cmd_bits);
__IO_REG32_BIT(MMCHS3_RSP10,    						0x480AD110,__READ				,__mmchs_rsp10_bits);
__IO_REG32_BIT(MMCHS3_RSP32,    						0x480AD114,__READ				,__mmchs_rsp32_bits);
__IO_REG32_BIT(MMCHS3_RSP54,    						0x480AD118,__READ				,__mmchs_rsp54_bits);
__IO_REG32_BIT(MMCHS3_RSP76,    						0x480AD11C,__READ				,__mmchs_rsp76_bits);
__IO_REG32(		 MMCHS3_DATA,    							0x480AD120,__READ_WRITE	);
__IO_REG32_BIT(MMCHS3_PSTATE,    						0x480AD124,__READ				,__mmchs_pstate_bits);
__IO_REG32_BIT(MMCHS3_HCTL,    							0x480AD128,__READ_WRITE	,__mmchs_hctl_bits);
__IO_REG32_BIT(MMCHS3_SYSCTL,    						0x480AD12C,__READ_WRITE	,__mmchs_sysctl_bits);
__IO_REG32_BIT(MMCHS3_STAT,    							0x480AD130,__READ_WRITE	,__mmchs_stat_bits);
__IO_REG32_BIT(MMCHS3_IE,    								0x480AD134,__READ_WRITE	,__mmchs_ie_bits);
__IO_REG32_BIT(MMCHS3_ISE,    							0x480AD138,__READ_WRITE	,__mmchs_ise_bits);
__IO_REG32_BIT(MMCHS3_AC12,    							0x480AD13C,__READ				,__mmchs_ac12_bits);
__IO_REG32_BIT(MMCHS3_CAPA,    							0x480AD140,__READ_WRITE	,__mmchs_capa_bits);
__IO_REG32_BIT(MMCHS3_CUR_CAPA,    					0x480AD144,__READ_WRITE	,__mmchs_cur_capa_bits);
__IO_REG32_BIT(MMCHS3_REV,    							0x480AD1FC,__READ				,__mmchs_rev_bits);

/***************************************************************************
 **
 ** GPIO1
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO1_REVISION,    					0x48310000,__READ				,__gpio_revision_bits);
__IO_REG32_BIT(GPIO1_SYSCONFIG,    					0x48310010,__READ_WRITE	,__gpio_sysconfig_bits);
__IO_REG32_BIT(GPIO1_SYSSTATUS,    					0x48310014,__READ				,__gpio_sysstatus_bits);
__IO_REG32_BIT(GPIO1_IRQSTATUS1,    				0x48310018,__READ_WRITE	,__gpio_irqstatus_bits);
__IO_REG32_BIT(GPIO1_IRQENABLE1,    				0x4831001C,__READ_WRITE	,__gpio_irqenable_bits);
__IO_REG32_BIT(GPIO1_WAKEUPENABLE,    			0x48310020,__READ_WRITE	,__gpio_wakeupenable_bits);
__IO_REG32_BIT(GPIO1_IRQSTATUS2,	    			0x48310028,__READ_WRITE	,__gpio_irqstatus_bits);
__IO_REG32_BIT(GPIO1_IRQENABLE2,    				0x4831002C,__READ_WRITE	,__gpio_irqenable_bits);
__IO_REG32_BIT(GPIO1_CTRL,    							0x48310030,__READ_WRITE	,__gpio_ctrl_bits);
__IO_REG32_BIT(GPIO1_OE,    								0x48310034,__READ_WRITE	,__gpio_oe_bits);
__IO_REG32_BIT(GPIO1_DATAIN,    						0x48310038,__READ				,__gpio_datain_bits);
__IO_REG32_BIT(GPIO1_DATAOUT,    						0x4831003C,__READ_WRITE	,__gpio_dataout_bits);
__IO_REG32_BIT(GPIO1_LEVELDETECT0,    			0x48310040,__READ_WRITE	,__gpio_leveldetect0_bits);
__IO_REG32_BIT(GPIO1_LEVELDETECT1,    			0x48310044,__READ_WRITE	,__gpio_leveldetect1_bits);
__IO_REG32_BIT(GPIO1_RISINGDETECT,    			0x48310048,__READ_WRITE	,__gpio_risingdetect_bits);
__IO_REG32_BIT(GPIO1_FALLINGDETECT,    			0x4831004C,__READ_WRITE	,__gpio_fallingdetect_bits);
__IO_REG32_BIT(GPIO1_DEBOUNCENABLE,    			0x48310050,__READ_WRITE	,__gpio_debouncenable_bits);
__IO_REG32_BIT(GPIO1_DEBOUNCINGTIME,    		0x48310054,__READ_WRITE	,__gpio_debouncingtime_bits);
__IO_REG32_BIT(GPIO1_CLEARIRQENABLE1,    		0x48310060,__READ_WRITE	,__gpio_clearirqenable_bits);
__IO_REG32_BIT(GPIO1_SETIRQENABLE1,    			0x48310064,__READ_WRITE	,__gpio_setirqenable_bits);
__IO_REG32_BIT(GPIO1_CLEARIRQENABLE2,    		0x48310070,__READ_WRITE	,__gpio_clearirqenable_bits);
__IO_REG32_BIT(GPIO1_SETIRQENABLE2,    			0x48310074,__READ_WRITE	,__gpio_setirqenable_bits);
__IO_REG32_BIT(GPIO1_CLEARWKUENA,    				0x48310080,__READ_WRITE	,__gpio_clearwkuena_bits);
__IO_REG32_BIT(GPIO1_SETWKUENA,    					0x48310084,__READ_WRITE	,__gpio_setwkuena_bits);
__IO_REG32_BIT(GPIO1_CLEARDATAOUT,    			0x48310090,__READ_WRITE	,__gpio_cleardataout_bits);
__IO_REG32_BIT(GPIO1_SETDATAOUT,    				0x48310094,__READ_WRITE	,__gpio_setdataout_bits);

/***************************************************************************
 **
 ** GPIO2
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO2_REVISION,    					0x49050000,__READ				,__gpio_revision_bits);
__IO_REG32_BIT(GPIO2_SYSCONFIG,    					0x49050010,__READ_WRITE	,__gpio_sysconfig_bits);
__IO_REG32_BIT(GPIO2_SYSSTATUS,    					0x49050014,__READ				,__gpio_sysstatus_bits);
__IO_REG32_BIT(GPIO2_IRQSTATUS1,    				0x49050018,__READ_WRITE	,__gpio_irqstatus_bits);
__IO_REG32_BIT(GPIO2_IRQENABLE1,    				0x4905001C,__READ_WRITE	,__gpio_irqenable_bits);
__IO_REG32_BIT(GPIO2_WAKEUPENABLE,    			0x49050020,__READ_WRITE	,__gpio_wakeupenable_bits);
__IO_REG32_BIT(GPIO2_IRQSTATUS2,	    			0x49050028,__READ_WRITE	,__gpio_irqstatus_bits);
__IO_REG32_BIT(GPIO2_IRQENABLE2,    				0x4905002C,__READ_WRITE	,__gpio_irqenable_bits);
__IO_REG32_BIT(GPIO2_CTRL,    							0x49050030,__READ_WRITE	,__gpio_ctrl_bits);
__IO_REG32_BIT(GPIO2_OE,    								0x49050034,__READ_WRITE	,__gpio_oe_bits);
__IO_REG32_BIT(GPIO2_DATAIN,    						0x49050038,__READ				,__gpio_datain_bits);
__IO_REG32_BIT(GPIO2_DATAOUT,    						0x4905003C,__READ_WRITE	,__gpio_dataout_bits);
__IO_REG32_BIT(GPIO2_LEVELDETECT0,    			0x49050040,__READ_WRITE	,__gpio_leveldetect0_bits);
__IO_REG32_BIT(GPIO2_LEVELDETECT1,    			0x49050044,__READ_WRITE	,__gpio_leveldetect1_bits);
__IO_REG32_BIT(GPIO2_RISINGDETECT,    			0x49050048,__READ_WRITE	,__gpio_risingdetect_bits);
__IO_REG32_BIT(GPIO2_FALLINGDETECT,    			0x4905004C,__READ_WRITE	,__gpio_fallingdetect_bits);
__IO_REG32_BIT(GPIO2_DEBOUNCENABLE,    			0x49050050,__READ_WRITE	,__gpio_debouncenable_bits);
__IO_REG32_BIT(GPIO2_DEBOUNCINGTIME,    		0x49050054,__READ_WRITE	,__gpio_debouncingtime_bits);
__IO_REG32_BIT(GPIO2_CLEARIRQENABLE1,    		0x49050060,__READ_WRITE	,__gpio_clearirqenable_bits);
__IO_REG32_BIT(GPIO2_SETIRQENABLE1,    			0x49050064,__READ_WRITE	,__gpio_setirqenable_bits);
__IO_REG32_BIT(GPIO2_CLEARIRQENABLE2,    		0x49050070,__READ_WRITE	,__gpio_clearirqenable_bits);
__IO_REG32_BIT(GPIO2_SETIRQENABLE2,    			0x49050074,__READ_WRITE	,__gpio_setirqenable_bits);
__IO_REG32_BIT(GPIO2_CLEARWKUENA,    				0x49050080,__READ_WRITE	,__gpio_clearwkuena_bits);
__IO_REG32_BIT(GPIO2_SETWKUENA,    					0x49050084,__READ_WRITE	,__gpio_setwkuena_bits);
__IO_REG32_BIT(GPIO2_CLEARDATAOUT,    			0x49050090,__READ_WRITE	,__gpio_cleardataout_bits);
__IO_REG32_BIT(GPIO2_SETDATAOUT,    				0x49050094,__READ_WRITE	,__gpio_setdataout_bits);

/***************************************************************************
 **
 ** GPIO3
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO3_REVISION,    					0x49052000,__READ				,__gpio_revision_bits);
__IO_REG32_BIT(GPIO3_SYSCONFIG,    					0x49052010,__READ_WRITE	,__gpio_sysconfig_bits);
__IO_REG32_BIT(GPIO3_SYSSTATUS,    					0x49052014,__READ				,__gpio_sysstatus_bits);
__IO_REG32_BIT(GPIO3_IRQSTATUS1,    				0x49052018,__READ_WRITE	,__gpio_irqstatus_bits);
__IO_REG32_BIT(GPIO3_IRQENABLE1,    				0x4905201C,__READ_WRITE	,__gpio_irqenable_bits);
__IO_REG32_BIT(GPIO3_WAKEUPENABLE,    			0x49052020,__READ_WRITE	,__gpio_wakeupenable_bits);
__IO_REG32_BIT(GPIO3_IRQSTATUS2,	    			0x49052028,__READ_WRITE	,__gpio_irqstatus_bits);
__IO_REG32_BIT(GPIO3_IRQENABLE2,    				0x4905202C,__READ_WRITE	,__gpio_irqenable_bits);
__IO_REG32_BIT(GPIO3_CTRL,    							0x49052030,__READ_WRITE	,__gpio_ctrl_bits);
__IO_REG32_BIT(GPIO3_OE,    								0x49052034,__READ_WRITE	,__gpio_oe_bits);
__IO_REG32_BIT(GPIO3_DATAIN,    						0x49052038,__READ				,__gpio_datain_bits);
__IO_REG32_BIT(GPIO3_DATAOUT,    						0x4905203C,__READ_WRITE	,__gpio_dataout_bits);
__IO_REG32_BIT(GPIO3_LEVELDETECT0,    			0x49052040,__READ_WRITE	,__gpio_leveldetect0_bits);
__IO_REG32_BIT(GPIO3_LEVELDETECT1,    			0x49052044,__READ_WRITE	,__gpio_leveldetect1_bits);
__IO_REG32_BIT(GPIO3_RISINGDETECT,    			0x49052048,__READ_WRITE	,__gpio_risingdetect_bits);
__IO_REG32_BIT(GPIO3_FALLINGDETECT,    			0x4905204C,__READ_WRITE	,__gpio_fallingdetect_bits);
__IO_REG32_BIT(GPIO3_DEBOUNCENABLE,    			0x49052050,__READ_WRITE	,__gpio_debouncenable_bits);
__IO_REG32_BIT(GPIO3_DEBOUNCINGTIME,    		0x49052054,__READ_WRITE	,__gpio_debouncingtime_bits);
__IO_REG32_BIT(GPIO3_CLEARIRQENABLE1,    		0x49052060,__READ_WRITE	,__gpio_clearirqenable_bits);
__IO_REG32_BIT(GPIO3_SETIRQENABLE1,    			0x49052064,__READ_WRITE	,__gpio_setirqenable_bits);
__IO_REG32_BIT(GPIO3_CLEARIRQENABLE2,    		0x49052070,__READ_WRITE	,__gpio_clearirqenable_bits);
__IO_REG32_BIT(GPIO3_SETIRQENABLE2,    			0x49052074,__READ_WRITE	,__gpio_setirqenable_bits);
__IO_REG32_BIT(GPIO3_CLEARWKUENA,    				0x49052080,__READ_WRITE	,__gpio_clearwkuena_bits);
__IO_REG32_BIT(GPIO3_SETWKUENA,    					0x49052084,__READ_WRITE	,__gpio_setwkuena_bits);
__IO_REG32_BIT(GPIO3_CLEARDATAOUT,    			0x49052090,__READ_WRITE	,__gpio_cleardataout_bits);
__IO_REG32_BIT(GPIO3_SETDATAOUT,    				0x49052094,__READ_WRITE	,__gpio_setdataout_bits);

/***************************************************************************
 **
 ** GPIO4
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO4_REVISION,    					0x49054000,__READ				,__gpio_revision_bits);
__IO_REG32_BIT(GPIO4_SYSCONFIG,    					0x49054010,__READ_WRITE	,__gpio_sysconfig_bits);
__IO_REG32_BIT(GPIO4_SYSSTATUS,    					0x49054014,__READ				,__gpio_sysstatus_bits);
__IO_REG32_BIT(GPIO4_IRQSTATUS1,    				0x49054018,__READ_WRITE	,__gpio_irqstatus_bits);
__IO_REG32_BIT(GPIO4_IRQENABLE1,    				0x4905401C,__READ_WRITE	,__gpio_irqenable_bits);
__IO_REG32_BIT(GPIO4_WAKEUPENABLE,    			0x49054020,__READ_WRITE	,__gpio_wakeupenable_bits);
__IO_REG32_BIT(GPIO4_IRQSTATUS2,	    			0x49054028,__READ_WRITE	,__gpio_irqstatus_bits);
__IO_REG32_BIT(GPIO4_IRQENABLE2,    				0x4905402C,__READ_WRITE	,__gpio_irqenable_bits);
__IO_REG32_BIT(GPIO4_CTRL,    							0x49054030,__READ_WRITE	,__gpio_ctrl_bits);
__IO_REG32_BIT(GPIO4_OE,    								0x49054034,__READ_WRITE	,__gpio_oe_bits);
__IO_REG32_BIT(GPIO4_DATAIN,    						0x49054038,__READ				,__gpio_datain_bits);
__IO_REG32_BIT(GPIO4_DATAOUT,    						0x4905403C,__READ_WRITE	,__gpio_dataout_bits);
__IO_REG32_BIT(GPIO4_LEVELDETECT0,    			0x49054040,__READ_WRITE	,__gpio_leveldetect0_bits);
__IO_REG32_BIT(GPIO4_LEVELDETECT1,    			0x49054044,__READ_WRITE	,__gpio_leveldetect1_bits);
__IO_REG32_BIT(GPIO4_RISINGDETECT,    			0x49054048,__READ_WRITE	,__gpio_risingdetect_bits);
__IO_REG32_BIT(GPIO4_FALLINGDETECT,    			0x4905404C,__READ_WRITE	,__gpio_fallingdetect_bits);
__IO_REG32_BIT(GPIO4_DEBOUNCENABLE,    			0x49054050,__READ_WRITE	,__gpio_debouncenable_bits);
__IO_REG32_BIT(GPIO4_DEBOUNCINGTIME,    		0x49054054,__READ_WRITE	,__gpio_debouncingtime_bits);
__IO_REG32_BIT(GPIO4_CLEARIRQENABLE1,    		0x49054060,__READ_WRITE	,__gpio_clearirqenable_bits);
__IO_REG32_BIT(GPIO4_SETIRQENABLE1,    			0x49054064,__READ_WRITE	,__gpio_setirqenable_bits);
__IO_REG32_BIT(GPIO4_CLEARIRQENABLE2,    		0x49054070,__READ_WRITE	,__gpio_clearirqenable_bits);
__IO_REG32_BIT(GPIO4_SETIRQENABLE2,    			0x49054074,__READ_WRITE	,__gpio_setirqenable_bits);
__IO_REG32_BIT(GPIO4_CLEARWKUENA,    				0x49054080,__READ_WRITE	,__gpio_clearwkuena_bits);
__IO_REG32_BIT(GPIO4_SETWKUENA,    					0x49054084,__READ_WRITE	,__gpio_setwkuena_bits);
__IO_REG32_BIT(GPIO4_CLEARDATAOUT,    			0x49054090,__READ_WRITE	,__gpio_cleardataout_bits);
__IO_REG32_BIT(GPIO4_SETDATAOUT,    				0x49054094,__READ_WRITE	,__gpio_setdataout_bits);

/***************************************************************************
 **
 ** GPIO5
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO5_REVISION,    					0x49056000,__READ				,__gpio_revision_bits);
__IO_REG32_BIT(GPIO5_SYSCONFIG,    					0x49056010,__READ_WRITE	,__gpio_sysconfig_bits);
__IO_REG32_BIT(GPIO5_SYSSTATUS,    					0x49056014,__READ				,__gpio_sysstatus_bits);
__IO_REG32_BIT(GPIO5_IRQSTATUS1,    				0x49056018,__READ_WRITE	,__gpio_irqstatus_bits);
__IO_REG32_BIT(GPIO5_IRQENABLE1,    				0x4905601C,__READ_WRITE	,__gpio_irqenable_bits);
__IO_REG32_BIT(GPIO5_WAKEUPENABLE,    			0x49056020,__READ_WRITE	,__gpio_wakeupenable_bits);
__IO_REG32_BIT(GPIO5_IRQSTATUS2,	    			0x49056028,__READ_WRITE	,__gpio_irqstatus_bits);
__IO_REG32_BIT(GPIO5_IRQENABLE2,    				0x4905602C,__READ_WRITE	,__gpio_irqenable_bits);
__IO_REG32_BIT(GPIO5_CTRL,    							0x49056030,__READ_WRITE	,__gpio_ctrl_bits);
__IO_REG32_BIT(GPIO5_OE,    								0x49056034,__READ_WRITE	,__gpio_oe_bits);
__IO_REG32_BIT(GPIO5_DATAIN,    						0x49056038,__READ				,__gpio_datain_bits);
__IO_REG32_BIT(GPIO5_DATAOUT,    						0x4905603C,__READ_WRITE	,__gpio_dataout_bits);
__IO_REG32_BIT(GPIO5_LEVELDETECT0,    			0x49056040,__READ_WRITE	,__gpio_leveldetect0_bits);
__IO_REG32_BIT(GPIO5_LEVELDETECT1,    			0x49056044,__READ_WRITE	,__gpio_leveldetect1_bits);
__IO_REG32_BIT(GPIO5_RISINGDETECT,    			0x49056048,__READ_WRITE	,__gpio_risingdetect_bits);
__IO_REG32_BIT(GPIO5_FALLINGDETECT,    			0x4905604C,__READ_WRITE	,__gpio_fallingdetect_bits);
__IO_REG32_BIT(GPIO5_DEBOUNCENABLE,    			0x49056050,__READ_WRITE	,__gpio_debouncenable_bits);
__IO_REG32_BIT(GPIO5_DEBOUNCINGTIME,    		0x49056054,__READ_WRITE	,__gpio_debouncingtime_bits);
__IO_REG32_BIT(GPIO5_CLEARIRQENABLE1,    		0x49056060,__READ_WRITE	,__gpio_clearirqenable_bits);
__IO_REG32_BIT(GPIO5_SETIRQENABLE1,    			0x49056064,__READ_WRITE	,__gpio_setirqenable_bits);
__IO_REG32_BIT(GPIO5_CLEARIRQENABLE2,    		0x49056070,__READ_WRITE	,__gpio_clearirqenable_bits);
__IO_REG32_BIT(GPIO5_SETIRQENABLE2,    			0x49056074,__READ_WRITE	,__gpio_setirqenable_bits);
__IO_REG32_BIT(GPIO5_CLEARWKUENA,    				0x49056080,__READ_WRITE	,__gpio_clearwkuena_bits);
__IO_REG32_BIT(GPIO5_SETWKUENA,    					0x49056084,__READ_WRITE	,__gpio_setwkuena_bits);
__IO_REG32_BIT(GPIO5_CLEARDATAOUT,    			0x49056090,__READ_WRITE	,__gpio_cleardataout_bits);
__IO_REG32_BIT(GPIO5_SETDATAOUT,    				0x49056094,__READ_WRITE	,__gpio_setdataout_bits);

/***************************************************************************
 **
 ** GPIO6
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO6_REVISION,    					0x49058000,__READ				,__gpio_revision_bits);
__IO_REG32_BIT(GPIO6_SYSCONFIG,    					0x49058010,__READ_WRITE	,__gpio_sysconfig_bits);
__IO_REG32_BIT(GPIO6_SYSSTATUS,    					0x49058014,__READ				,__gpio_sysstatus_bits);
__IO_REG32_BIT(GPIO6_IRQSTATUS1,    				0x49058018,__READ_WRITE	,__gpio_irqstatus_bits);
__IO_REG32_BIT(GPIO6_IRQENABLE1,    				0x4905801C,__READ_WRITE	,__gpio_irqenable_bits);
__IO_REG32_BIT(GPIO6_WAKEUPENABLE,    			0x49058020,__READ_WRITE	,__gpio_wakeupenable_bits);
__IO_REG32_BIT(GPIO6_IRQSTATUS2,	    			0x49058028,__READ_WRITE	,__gpio_irqstatus_bits);
__IO_REG32_BIT(GPIO6_IRQENABLE2,    				0x4905802C,__READ_WRITE	,__gpio_irqenable_bits);
__IO_REG32_BIT(GPIO6_CTRL,    							0x49058030,__READ_WRITE	,__gpio_ctrl_bits);
__IO_REG32_BIT(GPIO6_OE,    								0x49058034,__READ_WRITE	,__gpio_oe_bits);
__IO_REG32_BIT(GPIO6_DATAIN,    						0x49058038,__READ				,__gpio_datain_bits);
__IO_REG32_BIT(GPIO6_DATAOUT,    						0x4905803C,__READ_WRITE	,__gpio_dataout_bits);
__IO_REG32_BIT(GPIO6_LEVELDETECT0,    			0x49058040,__READ_WRITE	,__gpio_leveldetect0_bits);
__IO_REG32_BIT(GPIO6_LEVELDETECT1,    			0x49058044,__READ_WRITE	,__gpio_leveldetect1_bits);
__IO_REG32_BIT(GPIO6_RISINGDETECT,    			0x49058048,__READ_WRITE	,__gpio_risingdetect_bits);
__IO_REG32_BIT(GPIO6_FALLINGDETECT,    			0x4905804C,__READ_WRITE	,__gpio_fallingdetect_bits);
__IO_REG32_BIT(GPIO6_DEBOUNCENABLE,    			0x49058050,__READ_WRITE	,__gpio_debouncenable_bits);
__IO_REG32_BIT(GPIO6_DEBOUNCINGTIME,    		0x49058054,__READ_WRITE	,__gpio_debouncingtime_bits);
__IO_REG32_BIT(GPIO6_CLEARIRQENABLE1,    		0x49058060,__READ_WRITE	,__gpio_clearirqenable_bits);
__IO_REG32_BIT(GPIO6_SETIRQENABLE1,    			0x49058064,__READ_WRITE	,__gpio_setirqenable_bits);
__IO_REG32_BIT(GPIO6_CLEARIRQENABLE2,    		0x49058070,__READ_WRITE	,__gpio_clearirqenable_bits);
__IO_REG32_BIT(GPIO6_SETIRQENABLE2,    			0x49058074,__READ_WRITE	,__gpio_setirqenable_bits);
__IO_REG32_BIT(GPIO6_CLEARWKUENA,    				0x49058080,__READ_WRITE	,__gpio_clearwkuena_bits);
__IO_REG32_BIT(GPIO6_SETWKUENA,    					0x49058084,__READ_WRITE	,__gpio_setwkuena_bits);
__IO_REG32_BIT(GPIO6_CLEARDATAOUT,    			0x49058090,__READ_WRITE	,__gpio_cleardataout_bits);
__IO_REG32_BIT(GPIO6_SETDATAOUT,    				0x49058094,__READ_WRITE	,__gpio_setdataout_bits);

/***************************************************************************
 **
 ** EMAC
 **
 ***************************************************************************/
__IO_REG32(		 EMAC_REVID,    							0x5C000000,__READ				);
__IO_REG32_BIT(EMAC_SRESET,    							0x5C000004,__READ_WRITE	,__emac_sreset_bits);
__IO_REG32_BIT(EMAC_INTCONTROL,    					0x5C00000C,__READ_WRITE	,__emac_intcontrol_bits);
__IO_REG32_BIT(EMAC_C0RXTHRESHEN,    				0x5C000010,__READ_WRITE	,__emac_crxthreshen_bits);
__IO_REG32_BIT(EMAC_C0RXEN,    							0x5C000014,__READ_WRITE	,__emac_crxen_bits);
__IO_REG32_BIT(EMAC_C0TXEN,    							0x5C000018,__READ_WRITE	,__emac_ctxen_bits);
__IO_REG32_BIT(EMAC_C0MISCEN,    						0x5C00001C,__READ_WRITE	,__emac_cmiscen_bits);
__IO_REG32_BIT(EMAC_C1RXTHRESHEN,    				0x5C000020,__READ_WRITE	,__emac_crxthreshen_bits);
__IO_REG32_BIT(EMAC_C1RXEN,    							0x5C000024,__READ_WRITE	,__emac_crxen_bits);
__IO_REG32_BIT(EMAC_C1TXEN,    							0x5C000028,__READ_WRITE	,__emac_ctxen_bits);
__IO_REG32_BIT(EMAC_C1MISCEN,    						0x5C00002C,__READ_WRITE	,__emac_cmiscen_bits);
__IO_REG32_BIT(EMAC_C2RXTHRESHEN,    				0x5C000030,__READ_WRITE	,__emac_crxthreshen_bits);
__IO_REG32_BIT(EMAC_C2RXEN,    							0x5C000034,__READ_WRITE	,__emac_crxen_bits);
__IO_REG32_BIT(EMAC_C2TXEN,    							0x5C000038,__READ_WRITE	,__emac_ctxen_bits);
__IO_REG32_BIT(EMAC_C2MISCEN,    						0x5C00003C,__READ_WRITE	,__emac_cmiscen_bits);
__IO_REG32_BIT(EMAC_C0RXTHRESHSTAT,    			0x5C000040,__READ				,__emac_crxthreshstat_bits);
__IO_REG32_BIT(EMAC_C0RXSTAT,    						0x5C000044,__READ				,__emac_crxstat_bits);
__IO_REG32_BIT(EMAC_C0TXSTAT,    						0x5C000048,__READ				,__emac_ctxstat_bits);
__IO_REG32_BIT(EMAC_C0MISCSTAT,    					0x5C00004C,__READ				,__emac_cmiscstat_bits);
__IO_REG32_BIT(EMAC_C1RXTHRESHSTAT,    			0x5C000050,__READ				,__emac_crxthreshstat_bits);
__IO_REG32_BIT(EMAC_C1RXSTAT,    						0x5C000054,__READ				,__emac_crxstat_bits);
__IO_REG32_BIT(EMAC_C1TXSTAT,    						0x5C000058,__READ				,__emac_ctxstat_bits);
__IO_REG32_BIT(EMAC_C1MISCSTAT,    					0x5C00005C,__READ				,__emac_cmiscstat_bits);
__IO_REG32_BIT(EMAC_C2RXTHRESHSTAT,    			0x5C000060,__READ				,__emac_crxthreshstat_bits);
__IO_REG32_BIT(EMAC_C2RXSTAT,    						0x5C000064,__READ				,__emac_crxstat_bits);
__IO_REG32_BIT(EMAC_C2TXSTAT,    						0x5C000068,__READ				,__emac_ctxstat_bits);
__IO_REG32_BIT(EMAC_C2MISCSTAT,    					0x5C00006C,__READ				,__emac_cmiscstat_bits);
__IO_REG32_BIT(EMAC_C0RXIMAX,    						0x5C000070,__READ_WRITE	,__emac_crximax_bits);
__IO_REG32_BIT(EMAC_C0TXIMAX,    						0x5C000074,__READ_WRITE	,__emac_ctximax_bits);
__IO_REG32_BIT(EMAC_C1RXIMAX,    						0x5C000078,__READ_WRITE	,__emac_crximax_bits);
__IO_REG32_BIT(EMAC_C1TXIMAX,    						0x5C00007C,__READ_WRITE	,__emac_ctximax_bits);
__IO_REG32_BIT(EMAC_C2RXIMAX,    						0x5C000080,__READ_WRITE	,__emac_crximax_bits);
__IO_REG32_BIT(EMAC_C2TXIMAX,    						0x5C000084,__READ_WRITE	,__emac_ctximax_bits);
__IO_REG32(	   EMAC_TXREVID,    						0x5C010000,__READ				);
__IO_REG32_BIT(EMAC_TXCONTROL,    					0x5C010004,__READ_WRITE	,__emac_txcontrol_bits);
__IO_REG32_BIT(EMAC_TXTEARDOWN,    					0x5C010008,__READ_WRITE	,__emac_txteardown_bits);
__IO_REG32(		 EMAC_RXREVID,    						0x5C010010,__READ				);
__IO_REG32_BIT(EMAC_RXCONTROL,    					0x5C010014,__READ_WRITE	,__emac_rxcontrol_bits);
__IO_REG32_BIT(EMAC_RXTEARDOWN,    					0x5C010018,__READ_WRITE	,__emac_rxteardown_bits);
__IO_REG32_BIT(EMAC_TXINTSTATRAW,    				0x5C010080,__READ_WRITE	,__emac_txintstatraw_bits);
__IO_REG32_BIT(EMAC_TXINTSTATMASKED,    		0x5C010084,__READ_WRITE	,__emac_txintstatraw_bits);
__IO_REG32_BIT(EMAC_TXINTMASKSET,    				0x5C010088,__READ_WRITE	,__emac_txintmaskset_bits);
__IO_REG32_BIT(EMAC_TXINTMASKCLEAR,    			0x5C01008C,__READ_WRITE	,__emac_txintmaskset_bits);
__IO_REG32_BIT(EMAC_MACINVECTOR,    				0x5C010090,__READ				,__emac_macinvector_bits);
__IO_REG32_BIT(EMAC_MACEOIVECTOR,    				0x5C010094,__READ_WRITE	,__emac_maceoivector_bits);
__IO_REG32_BIT(EMAC_RXINTSTATRAW,    				0x5C0100A0,__READ				,__emac_rxintstatraw_bits);
__IO_REG32_BIT(EMAC_RXINTSTATMASKED,    		0x5C0100A4,__READ				,__emac_rxintstatraw_bits);
__IO_REG32_BIT(EMAC_RXINTMASKSET,    				0x5C0100A8,__READ_WRITE	,__emac_rxintmaskset_bits);
__IO_REG32_BIT(EMAC_RXINTMASKCLEAR,    			0x5C0100AC,__READ_WRITE	,__emac_rxintmaskset_bits);
__IO_REG32_BIT(EMAC_MACINTSTATRAW,    			0x5C0100B0,__READ_WRITE	,__emac_macintstatraw_bits);
__IO_REG32_BIT(EMAC_MACINTSTATMASKED,    		0x5C0100B4,__READ_WRITE	,__emac_macintstatraw_bits);
__IO_REG32_BIT(EMAC_MACINTMASKSET,    			0x5C0100B8,__READ_WRITE	,__emac_macintmaskset_bits);
__IO_REG32_BIT(EMAC_MACINTMASKCLEAR,    		0x5C0100BC,__READ_WRITE	,__emac_macintmaskset_bits);
__IO_REG32_BIT(EMAC_RXMBPENABLE,    				0x5C010100,__READ_WRITE	,__emac_rxmbpenable_bits);
__IO_REG32_BIT(EMAC_RXUNICASTSET,    				0x5C010104,__READ_WRITE	,__emac_rxunicastset_bits);
__IO_REG32_BIT(EMAC_RXUNICASTCLEAR,    			0x5C010108,__READ_WRITE	,__emac_rxunicastset_bits);
__IO_REG32_BIT(EMAC_RXMAXLEN,    						0x5C01010C,__READ_WRITE	,__emac_rxmaxlen_bits);
__IO_REG32_BIT(EMAC_RXBUFFEROFFSET,    			0x5C010110,__READ_WRITE	,__emac_rxbufferoffset_bits);
__IO_REG32_BIT(EMAC_RXFILTERLOWTHRESH,    	0x5C010114,__READ_WRITE	,__emac_rxfilterlowthresh_bits);
__IO_REG32_BIT(EMAC_RX0FLOWTHRESH,    			0x5C010120,__READ_WRITE	,__emac_rxflowthresh_bits);
__IO_REG32_BIT(EMAC_RX1FLOWTHRESH,    			0x5C010124,__READ_WRITE	,__emac_rxflowthresh_bits);
__IO_REG32_BIT(EMAC_RX2FLOWTHRESH,    			0x5C010128,__READ_WRITE	,__emac_rxflowthresh_bits);
__IO_REG32_BIT(EMAC_RX3FLOWTHRESH,    			0x5C01012C,__READ_WRITE	,__emac_rxflowthresh_bits);
__IO_REG32_BIT(EMAC_RX4FLOWTHRESH,    			0x5C010130,__READ_WRITE	,__emac_rxflowthresh_bits);
__IO_REG32_BIT(EMAC_RX5FLOWTHRESH,    			0x5C010134,__READ_WRITE	,__emac_rxflowthresh_bits);
__IO_REG32_BIT(EMAC_RX6FLOWTHRESH,    			0x5C010138,__READ_WRITE	,__emac_rxflowthresh_bits);
__IO_REG32_BIT(EMAC_RX7FLOWTHRESH,    			0x5C01013C,__READ_WRITE	,__emac_rxflowthresh_bits);
__IO_REG32_BIT(EMAC_RX0FREEBUFFER,    			0x5C010140,__READ_WRITE	,__emac_rxfreebuffer_bits);
__IO_REG32_BIT(EMAC_RX1FREEBUFFER,    			0x5C010144,__READ_WRITE	,__emac_rxfreebuffer_bits);
__IO_REG32_BIT(EMAC_RX2FREEBUFFER,    			0x5C010148,__READ_WRITE	,__emac_rxfreebuffer_bits);
__IO_REG32_BIT(EMAC_RX3FREEBUFFER,    			0x5C01014C,__READ_WRITE	,__emac_rxfreebuffer_bits);
__IO_REG32_BIT(EMAC_RX4FREEBUFFER,    			0x5C010150,__READ_WRITE	,__emac_rxfreebuffer_bits);
__IO_REG32_BIT(EMAC_RX5FREEBUFFER,    			0x5C010154,__READ_WRITE	,__emac_rxfreebuffer_bits);
__IO_REG32_BIT(EMAC_RX6FREEBUFFER,    			0x5C010158,__READ_WRITE	,__emac_rxfreebuffer_bits);
__IO_REG32_BIT(EMAC_RX7FREEBUFFER,    			0x5C01015C,__READ_WRITE	,__emac_rxfreebuffer_bits);
__IO_REG32_BIT(EMAC_MACCONTROL,    					0x5C010160,__READ_WRITE	,__emac_maccontrol_bits);
__IO_REG32_BIT(EMAC_MACSTATUS,    					0x5C010164,__READ				,__emac_macstatus_bits);
__IO_REG32_BIT(EMAC_EMCONTROL,    					0x5C010168,__READ_WRITE	,__emac_emcontrol_bits);
__IO_REG32_BIT(EMAC_FIFOCONTROL,    				0x5C01016C,__READ_WRITE	,__emac_fifocontrol_bits);
__IO_REG32_BIT(EMAC_MACCONFIG,    					0x5C010170,__READ_WRITE	,__emac_macconfig_bits);
__IO_REG32_BIT(EMAC_SOFTRESET,    					0x5C010174,__READ_WRITE	,__emac_softreset_bits);
__IO_REG32_BIT(EMAC_MACSRCADDRLO,    				0x5C0101D0,__READ_WRITE	,__emac_macsrcaddrlo_bits);
__IO_REG32_BIT(EMAC_MACSRCADDRHI,    				0x5C0101D4,__READ_WRITE	,__emac_macsrcaddrhi_bits);
__IO_REG32(		 EMAC_MACHASH1,    						0x5C0101D8,__READ_WRITE	);
__IO_REG32(		 EMAC_MACHASH2,    						0x5C0101DC,__READ_WRITE	);
__IO_REG32_BIT(EMAC_BOFFTEST,    						0x5C0101E0,__READ_WRITE	,__emac_bofftest_bits);
__IO_REG32_BIT(EMAC_TPACETEST,    					0x5C0101E4,__READ				,__emac_tpacetest_bits);
__IO_REG32_BIT(EMAC_RXPAUSE,    						0x5C0101E8,__READ				,__emac_rxpause_bits);
__IO_REG32_BIT(EMAC_TXPAUSE,    						0x5C0101EC,__READ				,__emac_txpause_bits);
__IO_REG32_BIT(EMAC_MACADDRLO,    					0x5C010500,__READ_WRITE	,__emac_macaddrlo_bits);
__IO_REG32_BIT(EMAC_MACADDRHI,    					0x5C010504,__READ_WRITE	,__emac_macaddrhi_bits);
__IO_REG32_BIT(EMAC_MACINDEX,    						0x5C010508,__READ_WRITE	,__emac_macindex_bits);
__IO_REG32(		 EMAC_TX0HDP,    							0x5C010600,__READ_WRITE	);
__IO_REG32(		 EMAC_TX1HDP,    							0x5C010604,__READ_WRITE	);
__IO_REG32(		 EMAC_TX2HDP,    							0x5C010608,__READ_WRITE	);
__IO_REG32(		 EMAC_TX3HDP,    							0x5C01060C,__READ_WRITE	);
__IO_REG32(		 EMAC_TX4HDP,    							0x5C010610,__READ_WRITE	);
__IO_REG32(		 EMAC_TX5HDP,    							0x5C010614,__READ_WRITE	);
__IO_REG32(		 EMAC_TX6HDP,    							0x5C010618,__READ_WRITE	);
__IO_REG32(		 EMAC_TX7HDP,    							0x5C01061C,__READ_WRITE	);
__IO_REG32(		 EMAC_RX0HDP,    							0x5C010620,__READ_WRITE	);
__IO_REG32(		 EMAC_RX1HDP,    							0x5C010624,__READ_WRITE	);
__IO_REG32(		 EMAC_RX2HDP,    							0x5C010628,__READ_WRITE	);
__IO_REG32(		 EMAC_RX3HDP,    							0x5C01062C,__READ_WRITE	);
__IO_REG32(		 EMAC_RX4HDP,    							0x5C010630,__READ_WRITE	);
__IO_REG32(		 EMAC_RX5HDP,    							0x5C010634,__READ_WRITE	);
__IO_REG32(		 EMAC_RX6HDP,    							0x5C010638,__READ_WRITE	);
__IO_REG32(		 EMAC_RX7HDP,    							0x5C01063C,__READ_WRITE	);
__IO_REG32(		 EMAC_TX0CP,    							0x5C010640,__READ_WRITE	);
__IO_REG32(		 EMAC_TX1CP,    							0x5C010644,__READ_WRITE	);
__IO_REG32(		 EMAC_TX2CP,    							0x5C010648,__READ_WRITE	);
__IO_REG32(		 EMAC_TX3CP,    							0x5C01064C,__READ_WRITE	);
__IO_REG32(		 EMAC_TX4CP,    							0x5C010650,__READ_WRITE	);
__IO_REG32(		 EMAC_TX5CP,    							0x5C010654,__READ_WRITE	);
__IO_REG32(		 EMAC_TX6CP,    							0x5C010658,__READ_WRITE	);
__IO_REG32(		 EMAC_TX7CP,    							0x5C01065C,__READ_WRITE	);
__IO_REG32(		 EMAC_RX0CP,    							0x5C010660,__READ_WRITE	);
__IO_REG32(		 EMAC_RX1CP,    							0x5C010664,__READ_WRITE	);
__IO_REG32(		 EMAC_RX2CP,    							0x5C010668,__READ_WRITE	);
__IO_REG32(		 EMAC_RX3CP,    							0x5C01066C,__READ_WRITE	);
__IO_REG32(		 EMAC_RX4CP,    							0x5C010670,__READ_WRITE	);
__IO_REG32(		 EMAC_RX5CP,    							0x5C010674,__READ_WRITE	);
__IO_REG32(		 EMAC_RX6CP,    							0x5C010678,__READ_WRITE	);
__IO_REG32(		 EMAC_RX7CP,    							0x5C01067C,__READ_WRITE	);
__IO_REG32(		 EMAC_RX_FIFO_BASE,    				0x5C010300,__READ_WRITE	);
__IO_REG32(		 EMAC_TX_FIFO_BASE,    				0x5C010400,__READ_WRITE	);
__IO_REG32(		 EMAC_STATERAM_TEST_BASE,    	0x5C010700,__READ_WRITE	);
__IO_REG32(		 EMAC_RXGOODFRAMES,    				0x5C010200,__READ_WRITE	);
__IO_REG32(		 EMAC_RXBCASTFRAMES,    			0x5C010204,__READ_WRITE	);
__IO_REG32(		 EMAC_RXMCASTFRAMES,    			0x5C010208,__READ_WRITE	);
__IO_REG32(		 EMAC_RXPAUSEFRAMES,    			0x5C01020C,__READ_WRITE	);
__IO_REG32(		 EMAC_RXCRCERRORS,    				0x5C010210,__READ_WRITE	);
__IO_REG32(		 EMAC_RXALIGNCODEERRORS,    	0x5C010214,__READ_WRITE	);
__IO_REG32(		 EMAC_RXOVERSIZED,    				0x5C010218,__READ_WRITE	);
__IO_REG32(		 EMAC_RXJABBER,    						0x5C01021C,__READ_WRITE	);
__IO_REG32(		 EMAC_RXUNDERSIZED,    				0x5C010220,__READ_WRITE	);
__IO_REG32(		 EMAC_RXFRAGMENTS,    				0x5C010224,__READ_WRITE	);
__IO_REG32(		 EMAC_RXFILTERED,    					0x5C010228,__READ_WRITE	);
__IO_REG32(		 EMAC_RXQOSFILTERED,    			0x5C01022C,__READ_WRITE	);
__IO_REG32(		 EMAC_RXOCTETS,    						0x5C010230,__READ_WRITE	);
__IO_REG32(		 EMAC_TXGOODFRAMES,    				0x5C010234,__READ_WRITE	);
__IO_REG32(		 EMAC_TXBCASTFRAMES,    			0x5C010238,__READ_WRITE	);
__IO_REG32(		 EMAC_TXMCASTFRAMES,    			0x5C01023C,__READ_WRITE	);
__IO_REG32(		 EMAC_TXPAUSEFRAMES,    			0x5C010240,__READ_WRITE	);
__IO_REG32(		 EMAC_TXDEFERRED,    					0x5C010244,__READ_WRITE	);
__IO_REG32(		 EMAC_TXCOLLISION,    				0x5C010248,__READ_WRITE	);
__IO_REG32(		 EMAC_TXSINGLECOLL,    				0x5C01024C,__READ_WRITE	);
__IO_REG32(		 EMAC_TXMULTICOLL,    				0x5C010250,__READ_WRITE	);
__IO_REG32(		 EMAC_TXEXCESSIVECOLL,    		0x5C010254,__READ_WRITE	);
__IO_REG32(		 EMAC_TXLATECOLL,    					0x5C010258,__READ_WRITE	);
__IO_REG32(		 EMAC_TXUNDERRUN,    					0x5C01025C,__READ_WRITE	);
__IO_REG32(		 EMAC_TXCARRIERSENSE,    			0x5C010260,__READ_WRITE	);
__IO_REG32(		 EMAC_TXOCTETS,    						0x5C010264,__READ_WRITE	);
__IO_REG32(		 EMAC_FRAME64,    						0x5C010268,__READ_WRITE	);
__IO_REG32(		 EMAC_FRAME65T127,    				0x5C01026C,__READ_WRITE	);
__IO_REG32(		 EMAC_FRAME128T255,    				0x5C010270,__READ_WRITE	);
__IO_REG32(		 EMAC_FRAME256T511,    				0x5C010274,__READ_WRITE	);
__IO_REG32(		 EMAC_FRAME512T1023,    			0x5C010278,__READ_WRITE	);
__IO_REG32(		 EMAC_FRAME1024TUP,    				0x5C01027C,__READ_WRITE	);
__IO_REG32(		 EMAC_NETOCTETS,    					0x5C010280,__READ_WRITE	);
__IO_REG32(		 EMAC_RXSOFOVERRUNS,    			0x5C010284,__READ_WRITE	);
__IO_REG32(		 EMAC_RXMOFOVERRUNS,    			0x5C010288,__READ_WRITE	);
__IO_REG32(		 EMAC_RXDMAOVERRUNS,    			0x5C01028C,__READ_WRITE	);

/***************************************************************************
 **
 ** MDIO
 **
 ***************************************************************************/
__IO_REG32(		 MDIO_REVID,    							0x5C030000,__READ				);
__IO_REG32_BIT(MDIO_CONTROL,    						0x5C030004,__READ_WRITE	,__mdio_control_bits);
__IO_REG32(		 MDIO_ALIVE,    							0x5C030008,__READ_WRITE	);
__IO_REG32(		 MDIO_LINK,    								0x5C03000C,__READ				);
__IO_REG32_BIT(MDIO_LINKINTRAW,    					0x5C030010,__READ_WRITE	,__mdio_linkintraw_bits);
__IO_REG32_BIT(MDIO_LINKINTMASKED,    			0x5C030014,__READ_WRITE	,__mdio_linkintraw_bits);
__IO_REG32_BIT(MDIO_USERINTRAW,    					0x5C030020,__READ_WRITE	,__mdio_userintraw_bits);
__IO_REG32_BIT(MDIO_USERINTMASKED,    			0x5C030024,__READ_WRITE	,__mdio_userintraw_bits);
__IO_REG32_BIT(MDIO_USERINTMASKSET,    			0x5C030028,__READ_WRITE	,__mdio_userintraw_bits);
__IO_REG32_BIT(MDIO_USERINTMASKCLEAR,    		0x5C03002C,__READ_WRITE	,__mdio_userintraw_bits);
__IO_REG32_BIT(MDIO_USERACCESS0,    				0x5C030080,__READ_WRITE	,__mdio_useraccess_bits);
__IO_REG32_BIT(MDIO_USERPHYSEL0,    				0x5C030084,__READ_WRITE	,__mdio_userphysel_bits);
__IO_REG32_BIT(MDIO_USERACCESS1,    				0x5C030088,__READ_WRITE	,__mdio_useraccess_bits);
__IO_REG32_BIT(MDIO_USERPHYSEL1,    				0x5C03008C,__READ_WRITE	,__mdio_userphysel_bits);

/***************************************************************************
 **
 ** CAN
 **
 ***************************************************************************/
__IO_REG32_BIT(CANME,    										0x5c050000,__READ_WRITE	,__canme_bits);
__IO_REG32_BIT(CANMD,    										0x5c050004,__READ_WRITE	,__canmd_bits);
__IO_REG32_BIT(CANTRS,    									0x5c050008,__READ_WRITE	,__cantrs_bits);
__IO_REG32_BIT(CANTRR,    									0x5c05000C,__READ_WRITE	,__cantrr_bits);
__IO_REG32_BIT(CANTA,    										0x5c050010,__READ_WRITE	,__canta_bits);
__IO_REG32_BIT(CANAA,    										0x5c050014,__READ_WRITE	,__canaa_bits);
__IO_REG32_BIT(CANRMP,    									0x5c050018,__READ_WRITE	,__canrmp_bits);
__IO_REG32_BIT(CANRML,    									0x5c05001C,__READ_WRITE	,__canrml_bits);
__IO_REG32_BIT(CANRFP,    									0x5c050020,__READ_WRITE	,__canrfp_bits);
__IO_REG32_BIT(CANGAM,    									0x5c050024,__READ_WRITE	,__cangam_bits);
__IO_REG32_BIT(CANMC,    										0x5c050028,__READ_WRITE	,__canmc_bits);
__IO_REG32_BIT(CANBTC,    									0x5c05002C,__READ_WRITE	,__canbtc_bits);
__IO_REG32_BIT(CANES,    										0x5c050030,__READ_WRITE	,__canes_bits);
__IO_REG32_BIT(CANTEC,    									0x5c050034,__READ				,__cantec_bits);
__IO_REG32_BIT(CANREC,    									0x5c050038,__READ				,__canrec_bits);
__IO_REG32_BIT(CANGIF0,    									0x5c05003C,__READ_WRITE	,__cangif_bits);
__IO_REG32_BIT(CANGIM,    									0x5c050040,__READ_WRITE	,__cangim_bits);
__IO_REG32_BIT(CANGIF1,    									0x5c050044,__READ_WRITE	,__cangif_bits);
__IO_REG32_BIT(CANMIM,    									0x5c050048,__READ_WRITE	,__canmim_bits);
__IO_REG32_BIT(CANMIL,    									0x5c05004C,__READ_WRITE	,__canmil_bits);
__IO_REG32_BIT(CANOPC,    									0x5c050050,__READ_WRITE	,__canopc_bits);
__IO_REG32_BIT(CANTIOC,    									0x5c050054,__READ_WRITE	,__cantioc_bits);
__IO_REG32_BIT(CANRIOC,    									0x5c050058,__READ_WRITE	,__canrioc_bits);
__IO_REG32_BIT(CANLNT,    									0x5c05005C,__READ_WRITE	,__canlnt_bits);
__IO_REG32_BIT(CANTOC,    									0x5c050060,__READ_WRITE	,__cantoc_bits);
__IO_REG32_BIT(CANTOS,    									0x5c050064,__READ_WRITE	,__cantos_bits);
__IO_REG32_BIT(CANLAM ,    									0x5c053000,__READ_WRITE	,__canlam_bits);
__IO_REG32(		 CANMOTS0,    								0x5c053080,__READ_WRITE	);
__IO_REG32(		 CANMOTS1,    								0x5c053084,__READ_WRITE	);
__IO_REG32(		 CANMOTS2,    								0x5c053088,__READ_WRITE	);
__IO_REG32(		 CANMOTS3,    								0x5c05308C,__READ_WRITE	);
__IO_REG32(		 CANMOTS4,    								0x5c053090,__READ_WRITE	);
__IO_REG32(		 CANMOTS5,    								0x5c053094,__READ_WRITE	);
__IO_REG32(		 CANMOTS6,    								0x5c053098,__READ_WRITE	);
__IO_REG32(		 CANMOTS7,    								0x5c05309C,__READ_WRITE	);
__IO_REG32(		 CANMOTS8,    								0x5c0530A0,__READ_WRITE	);
__IO_REG32(		 CANMOTS9,    								0x5c0530A4,__READ_WRITE	);
__IO_REG32(		 CANMOTS10,    								0x5c0530A8,__READ_WRITE	);
__IO_REG32(		 CANMOTS11,    								0x5c0530AC,__READ_WRITE	);
__IO_REG32(		 CANMOTS12,    								0x5c0530B0,__READ_WRITE	);
__IO_REG32(		 CANMOTS13,    								0x5c0530B4,__READ_WRITE	);
__IO_REG32(		 CANMOTS14,    								0x5c0530B8,__READ_WRITE	);
__IO_REG32(		 CANMOTS15,    								0x5c0530BC,__READ_WRITE	);
__IO_REG32(		 CANMOTS16,    								0x5c0530C0,__READ_WRITE	);
__IO_REG32(		 CANMOTS17,    								0x5c0530C4,__READ_WRITE	);
__IO_REG32(		 CANMOTS18,    								0x5c0530C8,__READ_WRITE	);
__IO_REG32(		 CANMOTS19,    								0x5c0530CC,__READ_WRITE	);
__IO_REG32(		 CANMOTS20,    								0x5c0530D0,__READ_WRITE	);
__IO_REG32(		 CANMOTS21,    								0x5c0530D4,__READ_WRITE	);
__IO_REG32(		 CANMOTS22,    								0x5c0530D8,__READ_WRITE	);
__IO_REG32(		 CANMOTS23,    								0x5c0530DC,__READ_WRITE	);
__IO_REG32(		 CANMOTS24,    								0x5c0530E0,__READ_WRITE	);
__IO_REG32(		 CANMOTS25,    								0x5c0530E4,__READ_WRITE	);
__IO_REG32(		 CANMOTS26,    								0x5c0530E8,__READ_WRITE	);
__IO_REG32(		 CANMOTS27,    								0x5c0530EC,__READ_WRITE	);
__IO_REG32(		 CANMOTS28,    								0x5c0530F0,__READ_WRITE	);
__IO_REG32(		 CANMOTS29,    								0x5c0530F4,__READ_WRITE	);
__IO_REG32(		 CANMOTS30,    								0x5c0530F8,__READ_WRITE	);
__IO_REG32(		 CANMOTS31,    								0x5c0530FC,__READ_WRITE	);
__IO_REG32(		 CANMOTO0,    								0x5c053100,__READ_WRITE	);
__IO_REG32(		 CANMOTO1,    								0x5c053104,__READ_WRITE	);
__IO_REG32(		 CANMOTO2,    								0x5c053108,__READ_WRITE	);
__IO_REG32(		 CANMOTO3,    								0x5c05310C,__READ_WRITE	);
__IO_REG32(		 CANMOTO4,    								0x5c053110,__READ_WRITE	);
__IO_REG32(		 CANMOTO5,    								0x5c053114,__READ_WRITE	);
__IO_REG32(		 CANMOTO6,    								0x5c053118,__READ_WRITE	);
__IO_REG32(		 CANMOTO7,    								0x5c05311C,__READ_WRITE	);
__IO_REG32(		 CANMOTO8,    								0x5c053120,__READ_WRITE	);
__IO_REG32(		 CANMOTO9,    								0x5c053124,__READ_WRITE	);
__IO_REG32(		 CANMOTO10,    								0x5c053128,__READ_WRITE	);
__IO_REG32(		 CANMOTO11,    								0x5c05312C,__READ_WRITE	);
__IO_REG32(		 CANMOTO12,    								0x5c053130,__READ_WRITE	);
__IO_REG32(		 CANMOTO13,    								0x5c053134,__READ_WRITE	);
__IO_REG32(		 CANMOTO14,    								0x5c053138,__READ_WRITE	);
__IO_REG32(		 CANMOTO15,    								0x5c05313C,__READ_WRITE	);
__IO_REG32(		 CANMOTO16,    								0x5c053140,__READ_WRITE	);
__IO_REG32(		 CANMOTO17,    								0x5c053144,__READ_WRITE	);
__IO_REG32(		 CANMOTO18,    								0x5c053148,__READ_WRITE	);
__IO_REG32(		 CANMOTO19,    								0x5c05314C,__READ_WRITE	);
__IO_REG32(		 CANMOTO20,    								0x5c053150,__READ_WRITE	);
__IO_REG32(		 CANMOTO21,    								0x5c053154,__READ_WRITE	);
__IO_REG32(		 CANMOTO22,    								0x5c053158,__READ_WRITE	);
__IO_REG32(		 CANMOTO23,    								0x5c05315C,__READ_WRITE	);
__IO_REG32(		 CANMOTO24,    								0x5c053160,__READ_WRITE	);
__IO_REG32(		 CANMOTO25,    								0x5c053164,__READ_WRITE	);
__IO_REG32(		 CANMOTO26,    								0x5c053168,__READ_WRITE	);
__IO_REG32(		 CANMOTO27,    								0x5c05316C,__READ_WRITE	);
__IO_REG32(		 CANMOTO28,    								0x5c053170,__READ_WRITE	);
__IO_REG32(		 CANMOTO29,    								0x5c053174,__READ_WRITE	);
__IO_REG32(		 CANMOTO30,    								0x5c053178,__READ_WRITE	);
__IO_REG32(		 CANMOTO31,    								0x5c05317C,__READ_WRITE	);

/***************************************************************************
 **
 ** UART1
 **
 ***************************************************************************/
__IO_REG16_BIT(UART1_DLL_REG,    						0x4806A000,__READ_WRITE	,__uart_dll_reg_bits);
#define 			 UART1_RHR_REG							UART1_DLL_REG
#define 			 UART1_RHR_REG_bit					UART1_DLL_REG_bit
#define 			 UART1_THR_REG							UART1_DLL_REG
#define 			 UART1_THR_REG_bit					UART1_DLL_REG_bit
__IO_REG16_BIT(UART1_IER_REG,    						0x4806A004,__READ_WRITE	,__uart_ier_reg_bits);
#define 			 UART1_DLH_REG							UART1_IER_REG
#define 			 UART1_DLH_REG_bit					UART1_IER_REG_bit
__IO_REG16_BIT(UART1_FCR_REG,    						0x4806A008,__READ_WRITE	,__uart_fcr_reg_bits);
#define 			 UART1_IIR_REG							UART1_FCR_REG
#define 			 UART1_IIR_REG_bit					UART1_FCR_REG_bit
#define 			 UART1_EFR_REG							UART1_FCR_REG
#define 			 UART1_EFR_REG_bit					UART1_FCR_REG_bit
__IO_REG16_BIT(UART1_LCR_REG,    						0x4806A00C,__READ_WRITE	,__uart_lcr_reg_bits);
__IO_REG16_BIT(UART1_MCR_REG,    						0x4806A010,__READ_WRITE	,__uart_mcr_reg_bits);
#define 			 UART1_XON1_ADDR1_REG				UART1_MCR_REG
#define 			 UART1_XON1_ADDR1_REG_bit		UART1_MCR_REG_bit
__IO_REG16_BIT(UART1_LSR_REG,    						0x4806A014,__READ_WRITE	,__uart_lsr_reg_bits);
#define 			 UART1_XON2_ADDR2_REG				UART1_LSR_REG
#define 			 UART1_XON2_ADDR2_REG_bit		UART1_LSR_REG_bit
__IO_REG16_BIT(UART1_XOFF1_REG,    					0x4806A018,__READ_WRITE	,__uart_xoff1_reg_bits);
#define 			 UART1_TCR_REG							UART1_XOFF1_REG
#define 			 UART1_TCR_REG_bit					UART1_XOFF1_REG_bit
#define 			 UART1_MSR_REG							UART1_XOFF1_REG
#define 			 UART1_MSR_REG_bit					UART1_XOFF1_REG_bit
__IO_REG16_BIT(UART1_XOFF2_REG,    					0x4806A01C,__READ_WRITE	,__uart_xoff2_reg_bits);
#define 			 UART1_SPR_REG							UART1_XOFF2_REG
#define 			 UART1_SPR_REG_bit					UART1_XOFF2_REG_bit
#define 			 UART1_TLR_REG							UART1_XOFF2_REG
#define 			 UART1_TLR_REG_bit					UART1_XOFF2_REG_bit
__IO_REG16_BIT(UART1_MDR1_REG,    					0x4806A020,__READ_WRITE	,__uart_mdr1_reg_bits);
__IO_REG16_BIT(UART1_MDR2_REG,    					0x4806A024,__READ_WRITE	,__uart_mdr2_reg_bits);
__IO_REG16_BIT(UART1_TXFLL_REG,    					0x4806A028,__READ_WRITE	,__uart_txfll_reg_bits);
#define 			 UART1_SFLSR_REG						UART1_TXFLL_REG
#define 			 UART1_SFLSR_REG_bit				UART1_TXFLL_REG_bit
__IO_REG16_BIT(UART1_RESUME_REG,    				0x4806A02C,__READ_WRITE	,__uart_resume_reg_bits);
#define 			 UART1_TXFLH_REG						UART1_RESUME_REG
#define 			 UART1_TXFLH_REG_bit				UART1_RESUME_REG_bit
__IO_REG16_BIT(UART1_RXFLL_REG,    					0x4806A030,__READ_WRITE	,__uart_rxfll_reg_bits);
#define 			 UART1_SFREGL_REG						UART1_RXFLL_REG
#define 			 UART1_SFREGL_REG_bit				UART1_RXFLL_REG_bit
__IO_REG16_BIT(UART1_RXFLH_REG,    					0x4806A034,__READ_WRITE	,__uart_rxflh_reg_bits);
#define 			 UART1_SFREGH_REG						UART1_RXFLH_REG
#define 			 UART1_SFREGH_REG_bit				UART1_RXFLH_REG_bit
__IO_REG16_BIT(UART1_BLR_REG,    						0x4806A038,__READ_WRITE	,__uart_blr_reg_bits);
#define 			 UART1_UASR_REG							UART1_BLR_REG
#define 			 UART1_UASR_REG_bit					UART1_BLR_REG_bit
__IO_REG16_BIT(UART1_SCR_REG,    						0x4806A040,__READ_WRITE	,__uart_scr_reg_bits);
__IO_REG16_BIT(UART1_SSR_REG,    						0x4806A044,__READ_WRITE	,__uart_ssr_reg_bits);
__IO_REG16_BIT(UART1_EBLR_REG,    					0x4806A048,__READ_WRITE	,__uart_eblr_reg_bits);
__IO_REG16_BIT(UART1_SYSC_REG,    					0x4806A054,__READ_WRITE	,__uart_sysc_reg_bits);
__IO_REG16_BIT(UART1_SYSS_REG,    					0x4806A058,__READ_WRITE	,__uart_syss_reg_bits);
__IO_REG16_BIT(UART1_WER_REG,    						0x4806A05C,__READ_WRITE	,__uart_wer_reg_bits);
__IO_REG16_BIT(UART1_CFPS_REG,    					0x4806A060,__READ_WRITE	,__uart_cfps_reg_bits);

/***************************************************************************
 **
 ** UART2
 **
 ***************************************************************************/
__IO_REG16_BIT(UART2_DLL_REG,    						0x4806C000,__READ_WRITE	,__uart_dll_reg_bits);
#define 			 UART2_RHR_REG							UART2_DLL_REG
#define 			 UART2_RHR_REG_bit					UART2_DLL_REG_bit
#define 			 UART2_THR_REG							UART2_DLL_REG
#define 			 UART2_THR_REG_bit					UART2_DLL_REG_bit
__IO_REG16_BIT(UART2_IER_REG,    						0x4806C004,__READ_WRITE	,__uart_ier_reg_bits);
#define 			 UART2_DLH_REG							UART2_IER_REG
#define 			 UART2_DLH_REG_bit					UART2_IER_REG_bit
__IO_REG16_BIT(UART2_FCR_REG,    						0x4806C008,__READ_WRITE	,__uart_fcr_reg_bits);
#define 			 UART2_IIR_REG							UART2_FCR_REG
#define 			 UART2_IIR_REG_bit					UART2_FCR_REG_bit
#define 			 UART2_EFR_REG							UART2_FCR_REG
#define 			 UART2_EFR_REG_bit					UART2_FCR_REG_bit
__IO_REG16_BIT(UART2_LCR_REG,    						0x4806C00C,__READ_WRITE	,__uart_lcr_reg_bits);
__IO_REG16_BIT(UART2_MCR_REG,    						0x4806C010,__READ_WRITE	,__uart_mcr_reg_bits);
#define 			 UART2_XON1_ADDR1_REG				UART2_MCR_REG
#define 			 UART2_XON1_ADDR1_REG_bit		UART2_MCR_REG_bit
__IO_REG16_BIT(UART2_LSR_REG,    						0x4806C014,__READ_WRITE	,__uart_lsr_reg_bits);
#define 			 UART2_XON2_ADDR2_REG				UART2_LSR_REG
#define 			 UART2_XON2_ADDR2_REG_bit		UART2_LSR_REG_bit
__IO_REG16_BIT(UART2_XOFF1_REG,    					0x4806C018,__READ_WRITE	,__uart_xoff1_reg_bits);
#define 			 UART2_TCR_REG							UART2_XOFF1_REG
#define 			 UART2_TCR_REG_bit					UART2_XOFF1_REG_bit
#define 			 UART2_MSR_REG							UART2_XOFF1_REG
#define 			 UART2_MSR_REG_bit					UART2_XOFF1_REG_bit
__IO_REG16_BIT(UART2_XOFF2_REG,    					0x4806C01C,__READ_WRITE	,__uart_xoff2_reg_bits);
#define 			 UART2_SPR_REG							UART2_XOFF2_REG
#define 			 UART2_SPR_REG_bit					UART2_XOFF2_REG_bit
#define 			 UART2_TLR_REG							UART2_XOFF2_REG
#define 			 UART2_TLR_REG_bit					UART2_XOFF2_REG_bit
__IO_REG16_BIT(UART2_MDR1_REG,    					0x4806C020,__READ_WRITE	,__uart_mdr1_reg_bits);
__IO_REG16_BIT(UART2_MDR2_REG,    					0x4806C024,__READ_WRITE	,__uart_mdr2_reg_bits);
__IO_REG16_BIT(UART2_TXFLL_REG,    					0x4806C028,__READ_WRITE	,__uart_txfll_reg_bits);
#define 			 UART2_SFLSR_REG						UART2_TXFLL_REG
#define 			 UART2_SFLSR_REG_bit				UART2_TXFLL_REG_bit
__IO_REG16_BIT(UART2_RESUME_REG,    				0x4806C02C,__READ_WRITE	,__uart_resume_reg_bits);
#define 			 UART2_TXFLH_REG						UART2_RESUME_REG
#define 			 UART2_TXFLH_REG_bit				UART2_RESUME_REG_bit
__IO_REG16_BIT(UART2_RXFLL_REG,    					0x4806C030,__READ_WRITE	,__uart_rxfll_reg_bits);
#define 			 UART2_SFREGL_REG						UART2_RXFLL_REG
#define 			 UART2_SFREGL_REG_bit				UART2_RXFLL_REG_bit
__IO_REG16_BIT(UART2_RXFLH_REG,    					0x4806C034,__READ_WRITE	,__uart_rxflh_reg_bits);
#define 			 UART2_SFREGH_REG						UART2_RXFLH_REG
#define 			 UART2_SFREGH_REG_bit				UART2_RXFLH_REG_bit
__IO_REG16_BIT(UART2_BLR_REG,    						0x4806C038,__READ_WRITE	,__uart_blr_reg_bits);
#define 			 UART2_UASR_REG							UART2_BLR_REG
#define 			 UART2_UASR_REG_bit					UART2_BLR_REG_bit
__IO_REG16_BIT(UART2_SCR_REG,    						0x4806C040,__READ_WRITE	,__uart_scr_reg_bits);
__IO_REG16_BIT(UART2_SSR_REG,    						0x4806C044,__READ_WRITE	,__uart_ssr_reg_bits);
__IO_REG16_BIT(UART2_EBLR_REG,    					0x4806C048,__READ_WRITE	,__uart_eblr_reg_bits);
__IO_REG16_BIT(UART2_SYSC_REG,    					0x4806C054,__READ_WRITE	,__uart_sysc_reg_bits);
__IO_REG16_BIT(UART2_SYSS_REG,    					0x4806C058,__READ_WRITE	,__uart_syss_reg_bits);
__IO_REG16_BIT(UART2_WER_REG,    						0x4806C05C,__READ_WRITE	,__uart_wer_reg_bits);
__IO_REG16_BIT(UART2_CFPS_REG,    					0x4806C060,__READ_WRITE	,__uart_cfps_reg_bits);

/***************************************************************************
 **
 ** UART4
 **
 ***************************************************************************/
__IO_REG16_BIT(UART4_DLL_REG,    						0x4809E000,__READ_WRITE	,__uart_dll_reg_bits);
#define 			 UART4_RHR_REG							UART4_DLL_REG
#define 			 UART4_RHR_REG_bit					UART4_DLL_REG_bit
#define 			 UART4_THR_REG							UART4_DLL_REG
#define 			 UART4_THR_REG_bit					UART4_DLL_REG_bit
__IO_REG16_BIT(UART4_IER_REG,    						0x4809E004,__READ_WRITE	,__uart_ier_reg_bits);
#define 			 UART4_DLH_REG							UART4_IER_REG
#define 			 UART4_DLH_REG_bit					UART4_IER_REG_bit
__IO_REG16_BIT(UART4_FCR_REG,    						0x4809E008,__READ_WRITE	,__uart_fcr_reg_bits);
#define 			 UART4_IIR_REG							UART4_FCR_REG
#define 			 UART4_IIR_REG_bit					UART4_FCR_REG_bit
#define 			 UART4_EFR_REG							UART4_FCR_REG
#define 			 UART4_EFR_REG_bit					UART4_FCR_REG_bit
__IO_REG16_BIT(UART4_LCR_REG,    						0x4809E00C,__READ_WRITE	,__uart_lcr_reg_bits);
__IO_REG16_BIT(UART4_MCR_REG,    						0x4809E010,__READ_WRITE	,__uart_mcr_reg_bits);
#define 			 UART4_XON1_ADDR1_REG				UART4_MCR_REG
#define 			 UART4_XON1_ADDR1_REG_bit		UART4_MCR_REG_bit
__IO_REG16_BIT(UART4_LSR_REG,    						0x4809E014,__READ_WRITE	,__uart_lsr_reg_bits);
#define 			 UART4_XON2_ADDR2_REG				UART4_LSR_REG
#define 			 UART4_XON2_ADDR2_REG_bit		UART4_LSR_REG_bit
__IO_REG16_BIT(UART4_XOFF1_REG,    					0x4809E018,__READ_WRITE	,__uart_xoff1_reg_bits);
#define 			 UART4_TCR_REG							UART4_XOFF1_REG
#define 			 UART4_TCR_REG_bit					UART4_XOFF1_REG_bit
#define 			 UART4_MSR_REG							UART4_XOFF1_REG
#define 			 UART4_MSR_REG_bit					UART4_XOFF1_REG_bit
__IO_REG16_BIT(UART4_XOFF2_REG,    					0x4809E01C,__READ_WRITE	,__uart_xoff2_reg_bits);
#define 			 UART4_SPR_REG							UART4_XOFF2_REG
#define 			 UART4_SPR_REG_bit					UART4_XOFF2_REG_bit
#define 			 UART4_TLR_REG							UART4_XOFF2_REG
#define 			 UART4_TLR_REG_bit					UART4_XOFF2_REG_bit
__IO_REG16_BIT(UART4_MDR1_REG,    					0x4809E020,__READ_WRITE	,__uart_mdr1_reg_bits);
__IO_REG16_BIT(UART4_MDR2_REG,    					0x4809E024,__READ_WRITE	,__uart_mdr2_reg_bits);
__IO_REG16_BIT(UART4_TXFLL_REG,    					0x4809E028,__READ_WRITE	,__uart_txfll_reg_bits);
#define 			 UART4_SFLSR_REG						UART4_TXFLL_REG
#define 			 UART4_SFLSR_REG_bit				UART4_TXFLL_REG_bit
__IO_REG16_BIT(UART4_RESUME_REG,    				0x4809E02C,__READ_WRITE	,__uart_resume_reg_bits);
#define 			 UART4_TXFLH_REG						UART4_RESUME_REG
#define 			 UART4_TXFLH_REG_bit				UART4_RESUME_REG_bit
__IO_REG16_BIT(UART4_RXFLL_REG,    					0x4809E030,__READ_WRITE	,__uart_rxfll_reg_bits);
#define 			 UART4_SFREGL_REG						UART4_RXFLL_REG_REG
#define 			 UART4_SFREGL_REG_bit				UART4_RXFLL_REG_REG_bit
__IO_REG16_BIT(UART4_RXFLH_REG,    					0x4809E034,__READ_WRITE	,__uart_rxflh_reg_bits);
#define 			 UART4_SFREGH_REG						UART4_RXFLH_REG
#define 			 UART4_SFREGH_REG_bit				UART4_RXFLH_REG_bit
__IO_REG16_BIT(UART4_BLR_REG,    						0x4809E038,__READ_WRITE	,__uart_blr_reg_bits);
#define 			 UART4_UASR_REG							UART4_BLR_REG
#define 			 UART4_UASR_REG_bit					UART4_BLR_REG_bit
__IO_REG16_BIT(UART4_SCR_REG,    						0x4809E040,__READ_WRITE	,__uart_scr_reg_bits);
__IO_REG16_BIT(UART4_SSR_REG,    						0x4809E044,__READ_WRITE	,__uart_ssr_reg_bits);
__IO_REG16_BIT(UART4_EBLR_REG,    					0x4809E048,__READ_WRITE	,__uart_eblr_reg_bits);
__IO_REG16_BIT(UART4_SYSC_REG,    					0x4809E054,__READ_WRITE	,__uart_sysc_reg_bits);
__IO_REG16_BIT(UART4_SYSS_REG,    					0x4809E058,__READ_WRITE	,__uart_syss_reg_bits);
__IO_REG16_BIT(UART4_WER_REG,    						0x4809E05C,__READ_WRITE	,__uart_wer_reg_bits);
__IO_REG16_BIT(UART4_CFPS_REG,    					0x4809E060,__READ_WRITE	,__uart_cfps_reg_bits);

/***************************************************************************
 **
 ** UART4
 **
 ***************************************************************************/
__IO_REG16_BIT(UART3_DLL_REG,    						0x49020000,__READ_WRITE	,__uart_dll_reg_bits);
#define 			 UART3_RHR_REG							UART3_DLL_REG
#define 			 UART3_RHR_REG_bit					UART3_DLL_REG_bit
#define 			 UART3_THR_REG							UART3_DLL_REG
#define 			 UART3_THR_REG_bit					UART3_DLL_REG_bit
__IO_REG16_BIT(UART3_IER_REG,    						0x49020004,__READ_WRITE	,__uart_ier_reg_bits);
#define 			 UART3_DLH_REG							UART3_IER_REG
#define 			 UART3_DLH_REG_bit					UART3_IER_REG_bit
#define 			 UART3_IrDA_IER_REG					UART3_IER_REG
#define 			 UART3_IrDA_IER_REG_bit			UART3_IER_REG_bit
#define 			 UART3_CIR_IER_REG					UART3_IER_REG
#define 			 UART3_CIR_IER_REG_bit			UART3_IER_REG_bit
__IO_REG16_BIT(UART3_FCR_REG,    						0x49020008,__READ_WRITE	,__uart_fcr_reg_bits);
#define 			 UART3_IIR_REG							UART3_FCR_REG
#define 			 UART3_IIR_REG_bit					UART3_FCR_REG_bit
#define 			 UART3_IrDA_FCR_REG					UART3_FCR_REG
#define 			 UART3_IrDA_FCR_REG_bit			UART3_FCR_REG_bit
#define 			 UART3_CIR_FCR_REG					UART3_FCR_REG
#define 			 UART3_CIR_FCR_REG_bit			UART3_FCR_REG_bit
#define 			 UART3_EFR_REG							UART3_FCR_REG
#define 			 UART3_EFR_REG_bit					UART3_FCR_REG_bit
__IO_REG16_BIT(UART3_LCR_REG,    						0x4902000C,__READ_WRITE	,__uart_lcr_reg_bits);
__IO_REG16_BIT(UART3_MCR_REG,    						0x49020010,__READ_WRITE	,__uart_mcr_reg_bits);
#define 			 UART3_XON1_ADDR1_REG				UART3_MCR_REG
#define 			 UART3_XON1_ADDR1_REG_bit		UART3_MCR_REG_bit
__IO_REG16_BIT(UART3_LSR_REG,    						0x49020014,__READ_WRITE	,__uart_lsr_reg_bits);
#define 			 UART3_XON2_ADDR2_REG				UART3_LSR_REG
#define 			 UART3_XON2_ADDR2_REG_bit		UART3_LSR_REG_bit
#define 			 UART3_LSR_IrDA_REG					UART3_LSR_REG
#define 			 UART3_LSR_IrDA_REG_bit			UART3_LSR_REG_bit
#define 			 UART3_LSR_CIR_REG					UART3_LSR_REG
#define 			 UART3_LSR_CIR_REG_bit			UART3_LSR_REG_bit
__IO_REG16_BIT(UART3_XOFF1_REG,    					0x49020018,__READ_WRITE	,__uart_xoff1_reg_bits);
#define 			 UART3_TCR_REG							UART3_XOFF1_REG
#define 			 UART3_TCR_REG_bit					UART3_XOFF1_REG_bit
#define 			 UART3_MSR_REG							UART3_XOFF1_REG
#define 			 UART3_MSR_REG_bit					UART3_XOFF1_REG_bit
__IO_REG16_BIT(UART3_XOFF2_REG,    					0x4902001C,__READ_WRITE	,__uart_xoff2_reg_bits);
#define 			 UART3_SPR_REG							UART3_XOFF2_REG
#define 			 UART3_SPR_REG_bit					UART3_XOFF2_REG_bit
#define 			 UART3_TLR_REG							UART3_XOFF2_REG
#define 			 UART3_TLR_REG_bit					UART3_XOFF2_REG_bit
__IO_REG16_BIT(UART3_MDR1_REG,    					0x49020020,__READ_WRITE	,__uart_mdr1_reg_bits);
__IO_REG16_BIT(UART3_MDR2_REG,    					0x49020024,__READ_WRITE	,__uart_mdr2_reg_bits);
__IO_REG16_BIT(UART3_TXFLL_REG,    					0x49020028,__READ_WRITE	,__uart_txfll_reg_bits);
#define 			 UART3_SFLSR_REG						UART3_TXFLL_REG
#define 			 UART3_SFLSR_REG_bit				UART3_TXFLL_REG_bit
__IO_REG16_BIT(UART3_RESUME_REG,    				0x4902002C,__READ_WRITE	,__uart_resume_reg_bits);
#define 			 UART3_TXFLH_REG						UART3_RESUME_REG
#define 			 UART3_TXFLH_REG_bit				UART3_RESUME_REG_bit
__IO_REG16_BIT(UART3_RXFLL_REG,    					0x49020030,__READ_WRITE	,__uart_rxfll_reg_bits);
#define 			 UART3_SFREGL_REG						UART3_RXFLL_REG_REG
#define 			 UART3_SFREGL_REG_bit				UART3_RXFLL_REG_REG_bit
__IO_REG16_BIT(UART3_RXFLH_REG,    					0x49020034,__READ_WRITE	,__uart_rxflh_reg_bits);
#define 			 UART3_SFREGH_REG						UART3_RXFLH_REG
#define 			 UART3_SFREGH_REG_bit				UART3_RXFLH_REG_bit
__IO_REG16_BIT(UART3_BLR_REG,    						0x49020038,__READ_WRITE	,__uart_blr_reg_bits);
#define 			 UART3_UASR_REG							UART3_BLR_REG
#define 			 UART3_UASR_REG_bit					UART3_BLR_REG_bit
__IO_REG16_BIT(UART3_ACREG_REG,    					0x4902003C,__READ_WRITE	,__uart_acreg_reg_bits);
__IO_REG16_BIT(UART3_SCR_REG,    						0x49020040,__READ_WRITE	,__uart_scr_reg_bits);
__IO_REG16_BIT(UART3_SSR_REG,    						0x49020044,__READ_WRITE	,__uart_ssr_reg_bits);
__IO_REG16_BIT(UART3_EBLR_REG,    					0x49020048,__READ_WRITE	,__uart_eblr_reg_bits);
__IO_REG16_BIT(UART3_SYSC_REG,    					0x49020054,__READ_WRITE	,__uart_sysc_reg_bits);
__IO_REG16_BIT(UART3_SYSS_REG,    					0x49020058,__READ_WRITE	,__uart_syss_reg_bits);
__IO_REG16_BIT(UART3_WER_REG,    						0x4902005C,__READ_WRITE	,__uart_wer_reg_bits);
__IO_REG16_BIT(UART3_CFPS_REG,    					0x49020060,__READ_WRITE	,__uart_cfps_reg_bits);

/***************************************************************************
 **
 ** USBTLL
 **
 ***************************************************************************/
__IO_REG32_BIT(USBTLL_REVISION,    					0x48062000,__READ				,__usbtll_revision_bits);
__IO_REG32_BIT(USBTLL_SYSCONFIG,    				0x48062010,__READ_WRITE	,__usbtll_sysconfig_bits);
__IO_REG32_BIT(USBTLL_SYSSTATUS,    				0x48062014,__READ				,__usbtll_sysstatus_bits);
__IO_REG32_BIT(USBTLL_IRQSTATUS,    				0x48062018,__READ_WRITE	,__usbtll_irqstatus_bits);
__IO_REG32_BIT(USBTLL_IRQENABLE,    				0x4806201C,__READ_WRITE	,__usbtll_irqenable_bits);
__IO_REG32_BIT(TLL_SHARED_CONF,    					0x48062030,__READ_WRITE	,__tll_shared_conf_bits);
__IO_REG32_BIT(TLL_CHANNEL_CONF_0,    			0x48062040,__READ_WRITE	,__tll_channel_conf_bits);
__IO_REG32_BIT(TLL_CHANNEL_CONF_1,    			0x48062044,__READ_WRITE	,__tll_channel_conf_bits);
__IO_REG32_BIT(TLL_CHANNEL_CONF_2,    			0x48062048,__READ_WRITE	,__tll_channel_conf_bits);

/***************************************************************************
 **
 ** ULPI0
 **
 ***************************************************************************/
__IO_REG8(		 ULPI_VENDOR_ID_LO_0,    			0x48062800,__READ				);
__IO_REG8(		 ULPI_VENDOR_ID_HI_0,    			0x48062801,__READ				);
__IO_REG8(		 ULPI_PRODUCT_ID_LO_0,    		0x48062802,__READ				);
__IO_REG8(		 ULPI_PRODUCT_ID_HI_0,    		0x48062803,__READ				);
__IO_REG8_BIT( ULPI_FUNCTION_CTRL_0,    		0x48062804,__READ_WRITE	,__ulpi_function_ctrl_bits);
__IO_REG8_BIT( ULPI_FUNCTION_CTRL_SET_0,    0x48062805,__READ_WRITE	,__ulpi_function_ctrl_bits);
__IO_REG8_BIT( ULPI_FUNCTION_CTRL_CLR_0,    0x48062806,__READ_WRITE	,__ulpi_function_ctrl_bits);
__IO_REG8_BIT( ULPI_INTERFACE_CTRL_0,    		0x48062807,__READ_WRITE	,__ulpi_interface_ctrl_bits);
__IO_REG8_BIT( ULPI_INTERFACE_CTRL_SET_0,   0x48062808,__READ_WRITE	,__ulpi_interface_ctrl_bits);
__IO_REG8_BIT( ULPI_INTERFACE_CTRL_CLR_0,   0x48062809,__READ_WRITE	,__ulpi_interface_ctrl_bits);
__IO_REG8_BIT( ULPI_OTG_CTRL_0,    					0x4806280A,__READ_WRITE	,__ulpi_otg_ctrl_bits);
__IO_REG8_BIT( ULPI_OTG_CTRL_SET_0,    			0x4806280B,__READ_WRITE	,__ulpi_otg_ctrl_bits);
__IO_REG8_BIT( ULPI_OTG_CTRL_CLR_0,    			0x4806280C,__READ_WRITE	,__ulpi_otg_ctrl_bits);
__IO_REG8_BIT( ULPI_USB_INT_EN_RISE_0,    	0x4806280D,__READ_WRITE	,__ulpi_usb_int_en_rise_bits);
__IO_REG8_BIT( ULPI_USB_INT_EN_RISE_SET_0,  0x4806280E,__READ_WRITE	,__ulpi_usb_int_en_rise_bits);
__IO_REG8_BIT( ULPI_USB_INT_EN_RISE_CLR_0,  0x4806280F,__READ_WRITE	,__ulpi_usb_int_en_rise_bits);
__IO_REG8_BIT( ULPI_USB_INT_EN_FALL_0,    	0x48062810,__READ_WRITE	,__ulpi_usb_int_en_fall_bits);
__IO_REG8_BIT( ULPI_USB_INT_EN_FALL_SET_0,  0x48062811,__READ_WRITE	,__ulpi_usb_int_en_fall_bits);
__IO_REG8_BIT( ULPI_USB_INT_EN_FALL_CLR_0,  0x48062812,__READ_WRITE	,__ulpi_usb_int_en_fall_bits);
__IO_REG8_BIT( ULPI_USB_INT_STATUS_0,  			0x48062813,__READ				,__ulpi_usb_int_status_bits);
__IO_REG8_BIT( ULPI_USB_INT_LATCH_0,  			0x48062814,__READ				,__ulpi_usb_int_latch_bits);
__IO_REG8_BIT( ULPI_DEBUG_0,  							0x48062815,__READ				,__ulpi_debug_bits);
__IO_REG8(		 ULPI_SCRATCH_REGISTER_0,  		0x48062816,__READ_WRITE	);
__IO_REG8(		 ULPI_SCRATCH_REGISTER_SET_0, 0x48062817,__READ_WRITE	);
__IO_REG8(		 ULPI_SCRATCH_REGISTER_CLR_0, 0x48062818,__READ_WRITE	);
__IO_REG8(		 ULPI_EXTENDED_SET_ACCESS_0,  0x4806282F,__READ_WRITE	);
__IO_REG8_BIT( ULPI_UTMI_VCONTROL_EN_0,  		0x48062830,__READ_WRITE	,__ulpi_utmi_vcontrol_en_bits);
__IO_REG8_BIT( ULPI_UTMI_VCONTROL_EN_SET_0, 0x48062831,__READ_WRITE	,__ulpi_utmi_vcontrol_en_bits);
__IO_REG8_BIT( ULPI_UTMI_VCONTROL_EN_CLR_0, 0x48062832,__READ_WRITE	,__ulpi_utmi_vcontrol_en_bits);
__IO_REG8(		 ULPI_UTMI_VCONTROL_STATUS_0, 0x48062833,__READ_WRITE	);
__IO_REG8_BIT( ULPI_UTMI_VCONTROL_LATCH_0,  0x48062834,__READ				,__ulpi_utmi_vcontrol_latch_bits);
__IO_REG8(		 ULPI_UTMI_VSTATUS_0,  				0x48062835,__READ_WRITE	);
__IO_REG8(		 ULPI_UTMI_VSTATUS_SET_0,  		0x48062836,__READ_WRITE	);
__IO_REG8(		 ULPI_UTMI_VSTATUS_CLR_0,  		0x48062837,__READ_WRITE	);
__IO_REG8(		 ULPI_USB_INT_LATCH_NOCLR_0,  0x48062838,__READ				);
__IO_REG8_BIT( ULPI_VENDOR_INT_EN_0,  			0x4806283B,__READ_WRITE	,__ulpi_vendor_int_en_bits);
__IO_REG8_BIT( ULPI_VENDOR_INT_EN_SET_0,  	0x4806283C,__READ_WRITE	,__ulpi_vendor_int_en_bits);
__IO_REG8_BIT( ULPI_VENDOR_INT_EN_CLR_0,  	0x4806283D,__READ_WRITE	,__ulpi_vendor_int_en_bits);
__IO_REG8_BIT( ULPI_VENDOR_INT_STATUS_0,  	0x4806283E,__READ				,__ulpi_vendor_int_status_bits);
__IO_REG8_BIT( ULPI_VENDOR_INT_LATCH_0,  		0x4806283F,__READ				,__ulpi_vendor_int_latch_bits);

/***************************************************************************
 **
 ** UHH
 **
 ***************************************************************************/
__IO_REG32_BIT(UHH_REVISION,    						0x48064000,__READ				,__uhh_revision_bits);
__IO_REG32_BIT(UHH_SYSCONFIG,    						0x48064010,__READ_WRITE	,__uhh_sysconfig_bits);
__IO_REG32_BIT(UHH_SYSSTATUS,    						0x48064014,__READ				,__uhh_sysstatus_bits);
__IO_REG32_BIT(UHH_HOSTCONFIG,    					0x48064040,__READ_WRITE	,__uhh_hostconfig_bits);
__IO_REG32_BIT(UHH_DEBUG_CSR,    						0x48064044,__READ_WRITE	,__uhh_debug_csr_bits);

/***************************************************************************
 **
 ** OHCI
 **
 ***************************************************************************/
__IO_REG32_BIT(HCREVISION,    							0x48064400,__READ				,__hcrevision_bits);
__IO_REG32_BIT(HCCONTROL,    								0x48064404,__READ_WRITE	,__hccontrol_bits);
__IO_REG32_BIT(HCCOMMANDSTATUS,    					0x48064408,__READ_WRITE	,__hccommandstatus_bits);
__IO_REG32_BIT(HCINTERRUPTSTATUS,    				0x4806440C,__READ_WRITE	,__hcinterruptstatus_bits);
__IO_REG32_BIT(HCINTERRUPTENABLE,    				0x48064410,__READ_WRITE	,__hcinterruptenable_bits);
__IO_REG32_BIT(HCINTERRUPTDISABLE,    			0x48064414,__READ_WRITE	,__hcinterruptenable_bits);
__IO_REG32(		 HCHCCA,    									0x48064418,__READ_WRITE	);
__IO_REG32(		 HCPERIODCURRENTED,    				0x4806441C,__READ				);
__IO_REG32(		 HCCONTROLHEADED,    					0x48064420,__READ_WRITE	);
__IO_REG32(		 HCCONTROLCURRENTED,    			0x48064424,__READ_WRITE	);
__IO_REG32(	 	 HCBULKHEADED,    						0x48064428,__READ_WRITE	);
__IO_REG32(		 HCBULKCURRENTED,    					0x4806442C,__READ_WRITE	);
__IO_REG32(		 HCDONEHEAD,    							0x48064430,__READ				);
__IO_REG32_BIT(HCFMINTERVAL,    						0x48064434,__READ_WRITE	,__hcfminterval_bits);
__IO_REG32_BIT(HCFMREMAINING,    						0x48064438,__READ				,__hcfmremaining_bits);
__IO_REG32_BIT(HCFMNUMBER,    							0x4806443C,__READ				,__hcfmnumber_bits);
__IO_REG32_BIT(HCPERIODICSTART,    					0x48064440,__READ_WRITE	,__hcperiodicstart_bits);
__IO_REG32_BIT(HCLSTHRESHOLD,    						0x48064444,__READ_WRITE	,__hclsthreshold_bits);
__IO_REG32_BIT(HCRHDESCRIPTORA,    					0x48064448,__READ_WRITE	,__hcrhdescriptora_bits);
__IO_REG32_BIT(HCRHDESCRIPTORB,    					0x4806444C,__READ_WRITE	,__hcrhdescriptorb_bits);
__IO_REG32_BIT(HCRHSTATUS,    							0x48064450,__READ_WRITE	,__hcrhstatus_bits);
__IO_REG32_BIT(HCRHPORTSTATUS_1,    				0x48064454,__READ_WRITE	,__hcrhportstatus_bits);
__IO_REG32_BIT(HCRHPORTSTATUS_2,    				0x48064458,__READ_WRITE	,__hcrhportstatus_bits);
__IO_REG32_BIT(HCRHPORTSTATUS_3,    				0x4806445C,__READ_WRITE	,__hcrhportstatus_bits);

/***************************************************************************
 **
 ** EHCI
 **
 ***************************************************************************/
__IO_REG32_BIT(EHCI_HCCAPBASE,    					0x48064800,__READ				,__ehci_hccapbase_bits);
__IO_REG32_BIT(EHCI_HCSPARAMS,    					0x48064804,__READ				,__ehci_hcsparams_bits);
__IO_REG32_BIT(EHCI_HCCPARAMS,    					0x48064808,__READ				,__ehci_hccparams_bits);
__IO_REG32_BIT(EHCI_USBCMD,    							0x48064810,__READ_WRITE	,__ehci_usbcmd_bits);
__IO_REG32_BIT(EHCI_USBSTS,    							0x48064814,__READ_WRITE	,__ehci_usbsts_bits);
__IO_REG32_BIT(EHCI_USBINTR,    						0x48064818,__READ_WRITE	,__ehci_usbintr_bits);
__IO_REG32_BIT(EHCI_FRINDEX,    						0x4806481C,__READ_WRITE	,__ehci_frindex_bits);
__IO_REG32(		 EHCI_CTRLDSSEGMENT,    			0x48064820,__READ				);
__IO_REG32(		 EHCI_PERIODICLISTBASE,    		0x48064824,__READ_WRITE	);
__IO_REG32(		 EHCI_ASYNCLISTADDR,    			0x48064828,__READ_WRITE	);
__IO_REG32_BIT(EHCI_CONFIGFLAG,    					0x48064850,__READ_WRITE	,__ehci_configflag_bits);
__IO_REG32_BIT(EHCI_PORTSC_0,    						0x48064854,__READ_WRITE	,__ehci_portsc_bits);
__IO_REG32_BIT(EHCI_PORTSC_1,    						0x48064858,__READ_WRITE	,__ehci_portsc_bits);
__IO_REG32_BIT(EHCI_PORTSC_2,    						0x4806485C,__READ_WRITE	,__ehci_portsc_bits);
__IO_REG32_BIT(EHCI_INSNREG00,    					0x48064890,__READ_WRITE	,__ehci_insnreg00_bits);
__IO_REG32_BIT(EHCI_INSNREG01,    					0x48064894,__READ_WRITE	,__ehci_insnreg01_bits);
__IO_REG32_BIT(EHCI_INSNREG02,    					0x48064898,__READ_WRITE	,__ehci_insnreg02_bits);
__IO_REG32_BIT(EHCI_INSNREG03,    					0x4806489C,__READ_WRITE	,__ehci_insnreg03_bits);
__IO_REG32_BIT(EHCI_INSNREG04,    					0x480648A0,__READ_WRITE	,__ehci_insnreg04_bits);
__IO_REG32_BIT(EHCI_INSNREG05_UTMI,    			0x480648A4,__READ_WRITE	,__ehci_insnreg05_utmi_bits);
#define	EHCI_INSNREG05_ULPI						EHCI_INSNREG05_UTMI
#define	EHCI_INSNREG05_ULPI_bit				EHCI_INSNREG05_UTMI_bit

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
#define RESETV  0x00  /* Reset                           */
#define UNDEFV  0x04  /* Undefined instruction           */
#define SWIV    0x08  /* Software interrupt              */
#define PABORTV 0x0c  /* Prefetch abort                  */
#define DABORTV 0x10  /* Data abort                      */
#define IRQV    0x18  /* Normal interrupt                */
#define FIQV    0x1c  /* Fast interrupt                  */

/***************************************************************************
 **
 **  AINT interrupt channels
 **
 ***************************************************************************/
#define AINT_COMMTX             0
#define AINT_COMMRX             1
#define AINT_NINT               2
#define AINT_PRU_EVTOUT0        3
#define AINT_PRU_EVTOUT1        4
#define AINT_PRU_EVTOUT2        5
#define AINT_PRU_EVTOUT3        6
#define AINT_PRU_EVTOUT4        7
#define AINT_PRU_EVTOUT5        8
#define AINT_PRU_EVTOUT6        9
#define AINT_PRU_EVTOUT7        10
#define AINT_EDMA3_0_CC0_INT0   11
#define AINT_EDMA3_0_CC0_ERRINT 12
#define AINT_EDMA3_0_TC0_ERRINT 13
#define AINT_EMIFA_INT          14
#define AINT_IIC0_INT           15
#define AINT_MMCSD0_INT0        16
#define AINT_MMCSD0_INT1        17
#define AINT_PSC0_ALLINT        18
#define AINT_RTC_IRQS           19
#define AINT_SPI0_INT           20
#define AINT_T64P0_TINT12       21
#define AINT_T64P0_TINT34       22
#define AINT_T64P1_TINT12       23
#define AINT_T64P1_TINT34       24
#define AINT_UART0_INT          25
#define AINT_PROTERR            27
#define SYSCFG_CHIPINT0         28
#define SYSCFG_CHIPINT1         29
#define SYSCFG_CHIPINT2         30
#define SYSCFG_CHIPINT3         31
#define AINT_EDMA3_0_TC1_ERRINT 32
#define AINT_EMAC_C0RXTHRESH    33
#define AINT_EMAC_C0RX          34
#define AINT_EMAC_C0TX          35
#define AINT_EMAC_C0MISC        36
#define AINT_EMAC_C1RXTHRESH    37
#define AINT_EMAC_C1RX          38
#define AINT_EMAC_C1TX          39
#define AINT_EMAC_C1MISC        40
#define AINT_DDR2_MEMERR        41
#define AINT_GPIO_B0INT         42
#define AINT_GPIO_B1INT         43
#define AINT_GPIO_B2INT         44
#define AINT_GPIO_B3INT         45
#define AINT_GPIO_B4INT         46
#define AINT_GPIO_B5INT         47
#define AINT_GPIO_B6INT         48
#define AINT_GPIO_B7INT         49
#define AINT_GPIO_B8INT         50
#define AINT_IIC1_INT           51
#define AINT_LCDC_INT           52
#define AINT_UART_INT1          53
#define AINT_MCASP_INT          54
#define AINT_PSC1_ALLINT        55
#define AINT_SPI1_INT           56
#define AINT_UHPI_ARMINT        57
#define AINT_USB0_INT           58
#define AINT_USB1_HCINT         59
#define AINT_USB1_RWAKEUP       60
#define AINT_UART2_INT          61
#define AINT_EHRPWM0            63
#define AINT_EHRPWM0TZ          64
#define AINT_EHRPWM1            65
#define AINT_EHRPWM1TZ          66
#define AINT_SATA_INT           67
#define AINT_T64P2_ALL          68
#define AINT_ECAP0              69
#define AINT_ECAP1              70
#define AINT_ECAP2              71
#define AINT_MMCSD1_INT0        72
#define AINT_MMCSD1_INT1        73
#define AINT_T64P0_CMPINT0      74
#define AINT_T64P0_CMPINT1      75
#define AINT_T64P0_CMPINT2      76
#define AINT_T64P0_CMPINT3      77
#define AINT_T64P0_CMPINT4      78
#define AINT_T64P0_CMPINT5      79
#define AINT_T64P0_CMPINT6      80
#define AINT_T64P0_CMPINT7      81
#define AINT_T64P1_CMPINT0      82
#define AINT_T64P1_CMPINT1      83
#define AINT_T64P1_CMPINT2      84
#define AINT_T64P1_CMPINT3      85
#define AINT_T64P1_CMPINT4      86
#define AINT_T64P1_CMPINT5      87
#define AINT_T64P1_CMPINT6      88
#define AINT_T64P1_CMPINT7      89
#define AINT_ARMCLKSTOPREQ      90
#define AINT_uPP_ALLINT         91
#define AINT_VPIF_ALLINT        92
#define AINT_EDMA3_1_CC0_INT0   93
#define AINT_EDMA3_1_CC0_ERRINT 94
#define AINT_EDMA3_1_TC0_ERRINT 95
#define AINT_T64P3_ALL          96
#define AINT_MCBSP0_RINT        97
#define AINT_MCBSP0_XINT        98
#define AINT_MCBSP1_RINT        99
#define AINT_MCBSP1_XINT        100

#endif    /* __IOAM3517_H */
