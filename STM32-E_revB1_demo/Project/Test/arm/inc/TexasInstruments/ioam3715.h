/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Texas Instruments AM3715
 **
 **    Used with ICCARM and AARM.
 **
 **    (c) Copyright IAR Systems 2011
 **
 **    $Revision: 46019 $
 **
 ***************************************************************************/

#ifndef __IOAM3715_H
#define __IOAM3715_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4f = 79 dec */
#error This file should only be compiled by ICCARM/AARM
#endif


#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    AM3715 SPECIAL FUNCTION REGISTERS
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

/* CM_FCLKEN_IVA2 */
typedef struct {
  __REG32 EN_IVA2 					  	  : 1;
  __REG32                 			  :31;
} __cm_fclken_iva2_bits;

/* CM_CLKEN_PLL_IVA2 */
typedef struct {
  __REG32 EN_IVA2_DPLL			  	  : 3;
  __REG32 EN_IVA2_DPLL_DRIFTGUARD : 1;
  __REG32             			  	  : 6;
  __REG32 EN_IVA2_DPLL_LPMODE 	  : 1;
  __REG32                 			  :21;
} __cm_clken_pll_iva2_bits;

/* CM_IDLEST_IVA2 */
typedef struct {
  __REG32 ST_IVA2     			  	  : 1;
  __REG32                 			  :31;
} __cm_idlest_iva2_bits;

/* CM_IDLEST_PLL_IVA2 */
typedef struct {
  __REG32 ST_IVA2_CLK  			  	  : 1;
  __REG32                 			  :31;
} __cm_idlest_pll_iva2_bits;

/* CM_AUTOIDLE_PLL_IVA2 */
typedef struct {
  __REG32 AUTO_IVA2_DPLL		  	  : 3;
  __REG32                 			  :29;
} __cm_autoidle_pll_iva2_bits;

/* CM_CLKSEL1_PLL_IVA2 */
typedef struct {
  __REG32 IVA2_DPLL_DIV 		  	  : 7;
  __REG32                		  	  : 1;
  __REG32 IVA2_DPLL_MULT 		  	  :11;
  __REG32 IVA2_CLK_SRC  		  	  : 3;
  __REG32                 			  :10;
} __cm_clksel1_pll_iva2_bits;

/* CM_CLKSEL2_PLL_IVA2 */
typedef struct {
  __REG32 IVA2_DPLL_CLKOUT_DIV	  : 5;
  __REG32                 			  :27;
} __cm_clksel2_pll_iva2_bits;

/* CM_CLKSTCTRL_IVA2 */
typedef struct {
  __REG32 CLKTRCTRL_IVA2      	  : 2;
  __REG32                 			  :30;
} __cm_clkstctrl_iva2_bits;

/* CM_CLKSTST_IVA2 */
typedef struct {
  __REG32 CLKACTIVITY_IVA2     	  : 1;
  __REG32                 			  :31;
} __cm_clkstst_iva2_bits;

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
  __REG32                 				: 6;
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
  __REG32 AUTO_MPU_DPLL						: 3;
  __REG32        	        				:29;
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
  __REG32 MPU_DPLL_DIV						: 5;
  __REG32        	        				:27;
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
  __REG32           							: 1;
  __REG32 EN_TS										: 1;
  __REG32 EN_USBTLL								: 1;
  __REG32        	        				:29;
} __cm_fclken3_core_bits;

/* CM_ICLKEN1_CORE */
typedef struct {
  __REG32        	        				: 1;
  __REG32 EN_SDRC									: 1;
  __REG32        	        				: 2;
  __REG32 EN_HSOTGUSB							: 1;
  __REG32        	        				: 1;
  __REG32 EN_SCMCTRL							: 1;
  __REG32 EN_MAILBOXES						: 1;
  __REG32        	        				: 1;
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
  __REG32         								: 1;
  __REG32 EN_MMC1									: 1;
  __REG32 EN_MMC2									: 1;
  __REG32        	        				: 3;
  __REG32 EN_ICR									: 1;
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
  __REG32 ST_SDRC 								: 1;
  __REG32 ST_SDMA									: 1;
  __REG32        	        				: 1;
  __REG32 ST_HSOTGUSB_STDBY				: 1;
  __REG32 ST_HSOTGUSB_IDLE				: 1;
  __REG32 ST_SCMCTRL							: 1;
  __REG32 ST_MAILBOXES						: 1;
  __REG32        	        				: 1;
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
  __REG32         								: 1;
  __REG32 ST_MMC1									: 1;
  __REG32 ST_MMC2									: 1;
  __REG32        	        				: 3;
  __REG32 ST_ICR									: 1;
  __REG32 ST_MMC3									: 1;
  __REG32        	        				: 1;
} __cm_idlest1_core_bits;

/* CM_IDLEST3_CORE */
typedef struct {
  __REG32 												: 2;
  __REG32 ST_USBTLL 							: 1;
  __REG32        	        				:29;
} __cm_idlest3_core_bits;

/* CM_AUTOIDLE1_CORE */
typedef struct {
  __REG32 												: 4;
  __REG32 AUTO_HSOTGUSB						: 1;
  __REG32        	        				: 1;
  __REG32 AUTO_SCMCTRL						: 1;
  __REG32 AUTO_MAILBOXES  				: 1;
  __REG32        	        				: 1;
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
  __REG32           							: 1;
  __REG32 AUTO_MMC1								: 1;
  __REG32 AUTO_MMC2								: 1;
  __REG32        	        				: 3;
  __REG32 AUTO_ICR                : 1;
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
  __REG32             						: 4;
  __REG32 CLKSEL_96M    					: 2;
  __REG32        	        				:18;
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
  __REG32 EN_SR1  								: 1;
  __REG32 EN_SR2  								: 1;
  __REG32        	        				:24;
} __cm_fclken_wkup_bits;

/* CM_ICLKEN_WKUP */
typedef struct {
  __REG32 EN_GPT1									: 1;
  __REG32                 				: 1;
  __REG32 EN_32KSYNC       				: 1;
  __REG32 EN_GPIO1								: 1;
  __REG32          	      				: 1;
  __REG32 EN_WDT2 								: 1;
  __REG32        	        				:26;
} __cm_iclken_wkup_bits;

/* CM_IDLEST_WKUP */
typedef struct {
  __REG32 ST_GPT1									: 1;
  __REG32                 				: 1;
  __REG32 ST_32KSYNC       				: 1;
  __REG32 ST_GPIO1								: 1;
  __REG32          	      				: 1;
  __REG32 ST_WDT2 								: 1;
  __REG32 ST_SR1  								: 1;
  __REG32 ST_SR2  								: 1;
  __REG32        	        				:24;
} __cm_idlest_wkup_bits;

/* CM_AUTOIDLE_WKUP */
typedef struct {
  __REG32 AUTO_GPT1								: 1;
  __REG32                   			: 1;
  __REG32 AUTO_32KSYNC       			: 1;
  __REG32 AUTO_GPIO1							: 1;
  __REG32           	      			: 1;
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
  __REG32                 					: 6;
  __REG32 EN_CORE_DPLL_LPMODE				: 1;
  __REG32        	        					: 1;
  __REG32 PWRDN_EMU_CORE						: 1;
  __REG32        	        					: 3;
  __REG32 EN_PERIPH_DPLL						: 3;
  __REG32 EN_PERIPH_DPLL_DRIFTGUARD	: 1;
  __REG32                   				: 7;
  __REG32 PWRDN_96M									: 1;
  __REG32 PWRDN_TV									: 1;
  __REG32 PWRDN_DSS1								: 1;
  __REG32 PWRDN_CAM       					: 1;
  __REG32 PWRDN_EMU_PERIPH					: 1;
} __cm_clken_pll_bits;

/* CM_CLKEN2_PLL */
typedef struct {
  __REG32 EN_PERIPH2_DPLL						: 3;
  __REG32 EN_PERIPH2_DPLL_DRIFTGUARD: 1;
  __REG32                     			: 6;
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
  __REG32 ST_CAM_CLK      					: 1;
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
 	__REG32 PERIPH_DPLL_MULT					:12;
 	__REG32                 					: 1;
 	__REG32 DCO_SEL         					: 3;
 	__REG32 SD_DIV          					: 8;
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
 	__REG32 CLKSEL_DSS1								: 6;
 	__REG32        	        					: 2;
 	__REG32 CLKSEL_TV									: 6;
 	__REG32        	        					:18;
} __cm_clksel_dss_bits;

/* CM_SLEEPDEP_DSS */
typedef struct {
 	__REG32 EN_CORE										: 1;
 	__REG32 EN_MPU										: 1;
 	__REG32 EN_IVA2										: 1;
 	__REG32        	        					:29;
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

/* CM_FCLKEN_CAM */
typedef struct {
 	__REG32 EN_CAM        						: 1;
 	__REG32 EN_CSI2        						: 1;
 	__REG32        	        					:30;
} __cm_fclken_cam_bits;

/* CM_ICLKEN_CAM */
typedef struct {
 	__REG32 EN_CAM        						: 1;
 	__REG32        	        					:31;
} __cm_iclken_cam_bits;

/* CM_IDLEST_CAM */
typedef struct {
 	__REG32 ST_CAM        						: 1;
 	__REG32        	        					:31;
} __cm_idlest_cam_bits;

/* CM_AUTOIDLE_CAM */
typedef struct {
 	__REG32 AUTO_CAM       						: 1;
 	__REG32        	        					:31;
} __cm_autoidle_cam_bits;

/* CM_CLKSEL_CAM */
typedef struct {
 	__REG32 CLKSEL_CAM     						: 6;
 	__REG32        	        					:26;
} __cm_clksel_cam_bits;

/* CM_SLEEPDEP_CAM */
typedef struct {
 	__REG32                						: 1;
 	__REG32 EN_MPU         						: 1;
 	__REG32        	        					:30;
} __cm_sleepdep_cam_bits;

/* CM_CLKSTCTRL_CAM */
typedef struct {
 	__REG32 CLKTRCTRL_CAM  						: 2;
 	__REG32        	        					:30;
} __cm_clkstctrl_cam_bits;

/* CM_CLKSTST_CAM */
typedef struct {
 	__REG32 CLKACTIVITY_CAM						: 1;
 	__REG32        	        					:31;
} __cm_clkstst_cam_bits;

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
 	__REG32 EN_UART4									: 1;
 	__REG32        	        					:13;
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
 	__REG32 ST_UART4									: 1;
 	__REG32        	        					:13;
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
 	__REG32 AUTO_UART4									: 1;
 	__REG32        	        						:13;
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
 	__REG32 EN_IVA2											: 1;
 	__REG32        	        						:29;
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
 	__REG32 DIV_DPLL3										: 6;
 	__REG32        	        						: 2;
 	__REG32 DIV_DPLL4										: 6;
 	__REG32        	        						: 2;
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
 	__REG32 EN_IVA2											: 1;
 	__REG32        	        						:29;
} __cm_sleepdep_usbhost_bits;

/* CM_CLKSTCTRL_USBHOST */
typedef struct {
 	__REG32 CLKTRCTRL_USBHOST						: 2;
 	__REG32        	        						:30;
} __cm_clkstctrl_usbhost_bits;

/* CM_CLKSTST_USBHOST */
typedef struct {
 	__REG32 CLKACTIVITY_USBHOST					: 1;
 	__REG32        	        						:31;
} __cm_clkstst_usbhost_bits;

/* RM_RSTCTRL_IVA2 */
typedef struct {
 	__REG32 RST1_IVA2         					: 1;
 	__REG32 RST2_IVA2         					: 1;
 	__REG32 RST3_IVA2         					: 1;
 	__REG32        	        						:29;
} __rm_rstctrl_iva2_bits;

/* RM_RSTST_IVA2 */
typedef struct {
 	__REG32 GLOBALCOLD_RST     					: 1;
 	__REG32 GLOBALWARM_RST     					: 1;
 	__REG32 DOMAINWKUP_RST     					: 1;
 	__REG32 COREDOMAINWKUP_RST 					: 1;
 	__REG32                    					: 4;
 	__REG32 IVA2_SW_RST1       					: 1;
 	__REG32 IVA2_SW_RST2       					: 1;
 	__REG32 IVA2_SW_RST3       					: 1;
 	__REG32 EMULATION_IVA2_RST 					: 1;
 	__REG32 EMULATION_VIDEO_HWA_RST 		: 1;
 	__REG32 EMULATION_SEQ_RST  					: 1;
 	__REG32        	        						:18;
} __rm_rstst_iva2_bits;

/* PM_WKDEP_IVA2 */
typedef struct {
 	__REG32 EN_CORE            					: 1;
 	__REG32 EN_MPU             					: 1;
 	__REG32                    					: 2;
 	__REG32 EN_WKUP           					: 1;
 	__REG32 EN_DSS             					: 1;
 	__REG32                    					: 1;
 	__REG32 EN_PER             					: 1;
 	__REG32        	        						:24;
} __pm_wkdep_iva2_bits;

/* PM_PWSTCTRL_IVA2 */
typedef struct {
 	__REG32 POWERSTATE         					: 2;
 	__REG32 LOGICRETSTATE      					: 1;
 	__REG32 MEMORYCHANGE      					: 1;
 	__REG32                    					: 4;
 	__REG32 SHAREDL1CACHEFLATRETSTATE		: 1;
 	__REG32 L1FLATMEMRETSTATE       		: 1;
 	__REG32 SHAREDL2CACHEFLATRETSTATE		: 1;
 	__REG32 L2FLATMEMRETSTATE       		: 1;
 	__REG32        	        						: 4;
 	__REG32 SHAREDL1CACHEFLATONSTATE 		: 2;
 	__REG32 L1FLATMEMONSTATE         		: 2;
 	__REG32 SHAREDL2CACHEFLATONSTATE 		: 2;
 	__REG32 L2FLATMEMONSTATE         		: 2;
 	__REG32                          		: 8;
} __pm_pwstctrl_iva2_bits;

/* PM_PWSTST_IVA2 */
typedef struct {
 	__REG32 POWERSTATEST       					: 2;
 	__REG32 LOGICSTATEST      					: 1;
 	__REG32                   					: 1;
 	__REG32 SHAREDL1CACHEFLATSTATEST 		: 2;
 	__REG32 L1FLATMEMSTATEST        		: 2;
 	__REG32 SHAREDL2CACHEFLATSTATEST 		: 2;
 	__REG32 L2FLATMEMSTATEST        		: 2;
 	__REG32        	        						: 8;
 	__REG32 INTRANSITION             		: 1;
 	__REG32                          		:11;
} __pm_pwstst_iva2_bits;

/* PM_PREPWSTST_IVA2 */
typedef struct {
 	__REG32 LASTPOWERSTATEENTERED 			      : 2;
 	__REG32 LASTLOGICSTATEENTERED 			      : 1;
 	__REG32                   					      : 1;
 	__REG32 LASTSHAREDL1CACHEFLATSTATEENTERED : 2;
 	__REG32 LASTL1FLATMEMSTATEENTERED		      : 2;
 	__REG32 LASTSHAREDL2CACHEFLATSTATEENTERED : 2;
 	__REG32 LASTL2FLATMEMSTATEENTERED		      : 2;
 	__REG32                          		      :20;
} __pm_prepwstst_iva2_bits;

/* PRM_IRQSTATUS_IVA2 */
typedef struct {
 	__REG32 WKUP_ST                     : 1;
 	__REG32 FORCEWKUP_ST     			      : 1;
 	__REG32 IVA2_DPLL_ST 					      : 1;
 	__REG32                    		      :29;
} __prm_irqstatus_iva2_bits;

/* PRM_IRQENABLE_IVA2 */
typedef struct {
 	__REG32 WKUP_EN                     : 1;
 	__REG32 FORCEWKUP_EN     			      : 1;
 	__REG32 IVA2_DPLL_RECAL_EN 			    : 1;
 	__REG32                    		      :29;
} __prm_irqenable_iva2_bits;

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
  __REG32 IVA2_DPLL_ST			  	: 1;
  __REG32 IO_ST     				  	: 1;
  __REG32 VP1_OPPCHANGEDONE_ST	: 1;
  __REG32 VP1_MINVDD_ST			  	: 1;
  __REG32 VP1_MAXVDD_ST			  	: 1;
  __REG32 VP1_NOSMPSACK_ST    	: 1;
  __REG32 VP1_EQVALUE_ST		  	: 1;
  __REG32 VP1_TRANXDONE_ST	  	: 1;
  __REG32 VP2_OPPCHANGEDONE_ST	: 1;
  __REG32 VP2_MINVDD_ST			  	: 1;
  __REG32 VP2_MAXVDD_ST			  	: 1;
  __REG32 VP2_NOSMPSACK_ST	  	: 1;
  __REG32 VP2_EQVALUE_ST  	  	: 1;
  __REG32 VP2_TRANXDONE_ST	  	: 1;
  __REG32 VC_SAERR_ST     	  	: 1;
  __REG32 VC_RAERR_ST     	  	: 1;
  __REG32 VC_TIMEOUTERR_ST 	  	: 1;
  __REG32 SND_PERIPH_DPLL_ST  	: 1;
  __REG32 ABB_LDO_TRANXDONE_ST	: 1;
  __REG32 VC_VP1_ACK_ST    	  	: 1;
  __REG32 VC_BYPASS_ACK_ST 	  	: 1;
  __REG32                 			: 3;
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
  __REG32 IVA2_DPLL_RECAL_EN  			: 1;
  __REG32 IO_EN           	  			: 1;
  __REG32 VP1_OPPCHANGEDONE_EN 			: 1;
  __REG32 VP1_MINVDD_EN   	  			: 1;
  __REG32 VP1_MAXVDD_EN       			: 1;
  __REG32 VP1_NOSMPSACK_EN	  			: 1;
  __REG32 VP1_EQVALUE_EN  	  			: 1;
  __REG32 VP1_TRANXDONE_EN 	  			: 1;
  __REG32 VP2_OPPCHANGEDONE_EN			: 1;
  __REG32 VP2_MINVDD_EN   	  			: 1;
  __REG32 VP2_MAXVDD_EN   	  			: 1;
  __REG32 VP2_NOSMPSACK_EN 	  			: 1;
  __REG32 VP2_EQVALUE_EN   	  			: 1;
  __REG32 VP2_TRANXDONE_EN 	  			: 1;
  __REG32 VC_SAERR_EN     	  			: 1;
  __REG32 VC_RAERR_EN     	  			: 1;
  __REG32 VC_TIMEOUTERR_EN 	  			: 1;
  __REG32 SND_PERIPH_DPLL_RECAL_EN	: 1;
  __REG32 ABB_LDO_TRANXDONE_EN    	: 1;
  __REG32 VC_VP1_ACK_EN           	: 1;
  __REG32 VC_BYPASS_ACK_EN        	: 1;
  __REG32                 					: 3;
} __prm_irqenable_mpu_bits;

/* RM_RSTST_MPU */
typedef struct {
  __REG32 GLOBALCOLD_RST		  			: 1;
  __REG32 GLOBALWARM_RST		  			: 1;
  __REG32 DOMAINWKUP_RST		  			: 1;
  __REG32 COREDOMAINWKUP_RST  			: 1;
  __REG32                 					: 7;
  __REG32 EMULATION_MPU_RST	  			: 1;
  __REG32                 					:20;
} __rm_rstst_mpu_bits;

/* PM_WKDEP_MPU */
typedef struct {
  __REG32 EN_CORE						  			: 1;
  __REG32                 					: 1;
  __REG32 EN_IVA2						  			: 1;
  __REG32                 					: 2;
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
  __REG32 LOGICL1CACHESTATEST  			: 1;
  __REG32             			  			: 3;
  __REG32 L2CACHESTATEST		  			: 2;
  __REG32                 					:12;
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
  __REG32 DOMAINWKUP_RST	: 1;
  __REG32                 :29;
} __rm_rstst_core_bits;

/* PM_WKEN1_CORE */
typedef struct {
  __REG32                 : 4;
  __REG32 EN_HSOTGUSB 		: 1;
  __REG32         				: 4;
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
  __REG32                 : 4;
  __REG32 GRPSEL_HSOTGUSB : 1;
  __REG32             		: 4;
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

/* PM_IVA2GRPSEL1_CORE */
typedef struct {
  __REG32                 : 4;
  __REG32 GRPSEL_HSOTGUSB : 1;
  __REG32             		: 4;
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
} __pm_iva2grpsel1_core_bits;

/* PM_WKST1_CORE */
typedef struct {
  __REG32                 : 4;
  __REG32 ST_HSOTGUSB			: 1;
  __REG32         				: 4;
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
  __REG32 SAVEANDRESTORE	: 1;
  __REG32                 : 3;
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
  __REG32 LOGICSTATEST		: 1;
  __REG32             		: 1;
  __REG32 MEM1STATEST 		: 2;
  __REG32 MEM2STATEST 		: 2;
  __REG32                 :12;
  __REG32 INTRANSITION		: 1;
  __REG32                 :11;
} __pm_pwstst_core_bits;

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

/* PM_IVA2GRPSEL3_CORE */
typedef struct {
  __REG32 												: 2;
  __REG32 GRPSEL_USBTLL						: 1;
  __REG32                 				:29;
} __pm_iva2grpsel3_core_bits;

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
  __REG32 DOMAINWKUP_RST					: 1;
  __REG32 COREDOMAINWKUP_RST			: 1;
  __REG32                 				:28;
} __rm_rstst_sgx_bits;

/* PM_WKDEP_SGX */
typedef struct {
  __REG32                 				: 1;
  __REG32 EN_MPU									: 1;
  __REG32 EN_IVA2									: 1;
  __REG32                 				: 1;
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
  __REG32                 				: 2;
  __REG32 EN_GPIO1								: 1;
  __REG32                 				: 2;
  __REG32 EN_SR1   								: 1;
  __REG32 EN_SR2   								: 1;
  __REG32 EN_IO   								: 1;
  __REG32          								: 7;
  __REG32 EN_IO_CHAIN							: 1;
  __REG32                 				:15;
} __pm_wken_wkup_bits;

/* PM_MPUGRPSEL_WKUP */
typedef struct {
  __REG32 GRPSEL_GPT1							: 1;
  __REG32                 				: 2;
  __REG32 GRPSEL_GPIO1						: 1;
  __REG32                 				: 2;
  __REG32 GRPSEL_SR1  						: 1;
  __REG32 GRPSEL_SR2  						: 1;
  __REG32 GRPSEL_IO   						: 1;
  __REG32                 				:23;
} __pm_mpugrpsel_wkup_bits;

/* PM_IVA2GRPSEL_WKUP */
typedef struct {
  __REG32 GRPSEL_GPT1							: 1;
  __REG32                 				: 2;
  __REG32 GRPSEL_GPIO1						: 1;
  __REG32                 				: 2;
  __REG32 GRPSEL_SR1  						: 1;
  __REG32 GRPSEL_SR2  						: 1;
  __REG32 GRPSEL_IO   						: 1;
  __REG32                 				:23;
} __pm_iva2grpsel_wkup_bits;

/* PM_WKST_WKUP */
typedef struct {
  __REG32 ST_GPT1									: 1;
  __REG32                 				: 2;
  __REG32 ST_GPIO1								: 1;
  __REG32                 				: 2;
  __REG32 ST_SR1  								: 1;
  __REG32 ST_SR2  								: 1;
  __REG32 ST_IO    								: 1;
  __REG32                 				: 7;
  __REG32 ST_IO_CHAIN							: 1;
  __REG32                 				:15;
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
  __REG32 DOMAINWKUP_RST					: 1;
  __REG32 COREDOMAINWKUP_RST  		: 1;
  __REG32                 				:28;
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
  __REG32 EN_IVA2									: 1;
  __REG32                 				: 1;
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

/* RM_RSTST_CAM */
typedef struct {
  __REG32 GLOBALCOLD_RST      		: 1;
  __REG32 GLOBALWARM_RST      		: 1;
  __REG32 DOMAINWKUP_RST      		: 1;
  __REG32 COREDOMAINWKUP_RST   		: 1;
  __REG32                 				:28;
} __rm_rstst_cam_bits;

/* PM_WKDEP_CAM */
typedef struct {
  __REG32                     		: 1;
  __REG32 EN_MPU              		: 1;
  __REG32 EN_IVA2             		: 1;
  __REG32                      		: 1;
  __REG32 EN_WKUP             		: 1;
  __REG32                 				:27;
} __pm_wkdep_cam_bits;

/* PM_PWSTCTRL_CAM */
typedef struct {
  __REG32 POWERSTATE          		: 2;
  __REG32 LOGICRETSTATE        		: 1;
  __REG32                     		: 5;
  __REG32 MEMRETSTATE          		: 1;
  __REG32                      		: 7;
  __REG32 MEMONSTATE           		: 2;
  __REG32                 				:14;
} __pm_pwstctrl_cam_bits;

/* PM_PWSTST_CAM */
typedef struct {
  __REG32 POWERSTATEST        		: 2;
  __REG32                     		:18;
  __REG32 INTRANSITION         		: 1;
  __REG32                 				:11;
} __pm_pwstst_cam_bits;

/* PM_PREPWSTST_CAM */
typedef struct {
  __REG32 LASTPOWERSTATEENTERED		: 2;
  __REG32                 				:30;
} __pm_prepwstst_cam_bits;

/* RM_RSTST_PER */
typedef struct {
  __REG32 GLOBALCOLD_RST					: 1;
  __REG32 GLOBALWARM_RST					: 1;
  __REG32 DOMAINWKUP_RST					: 1;
  __REG32 COREDOMAINWKUP_RST			: 1;
  __REG32                 				:28;
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
  __REG32 EN_UART4								: 1;
  __REG32        	        				:13;
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
  __REG32 GRPSEL_UART4						: 1;
  __REG32        	        				:13;
} __pm_mpugrpsel_per_bits;

/* PM_IVA2GRPSEL_PER */
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
  __REG32 GRPSEL_UART4						: 1;
  __REG32        	        				:13;
} __pm_iva2grpsel_per_bits;

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
  __REG32 ST_UART4								: 1;
  __REG32        	        				:13;
} __pm_wkst_per_bits;

/* PM_WKDEP_PER */
typedef struct {
  __REG32 EN_CORE									: 1;
  __REG32 EN_MPU									: 1;
  __REG32 EN_IVA2									: 1;
  __REG32 												: 1;
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
  __REG32 LOGICSTATEST						: 1;
  __REG32 												:17;
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
  __REG32 DOMAINWKUP_RST					: 1;
  __REG32        	        				:29;
} __rm_rstst_emu_bits;

/* PM_PWSTST_EMU */
typedef struct {
  __REG32 POWERSTATEST  					: 2;
  __REG32               					:18;
  __REG32 INTRANSITION  					: 1;
  __REG32        	        				:11;
} __pm_pwstst_emu_bits;

/* PRM_VC_SMPS_SA */
typedef struct {
  __REG32 SA0           					: 7;
  __REG32               					: 9;
  __REG32 SA1           					: 7;
  __REG32        	        				: 9;
} __prm_vc_smps_sa_bits;

/* PRM_VC_SMPS_VOL_RA */
typedef struct {
  __REG32 VOLRA0         					: 8;
  __REG32               					: 8;
  __REG32 VOLRA1         					: 8;
  __REG32        	        				: 8;
} __prm_vc_smps_vol_ra_bits;

/* PRM_VC_SMPS_CMD_RA */
typedef struct {
  __REG32 CMDRA0         					: 8;
  __REG32               					: 8;
  __REG32 CMDRA1         					: 8;
  __REG32        	        				: 8;
} __prm_vc_smps_cmd_ra_bits;

/* PRM_VC_CMD_VAL_0 */
/* PRM_VC_CMD_VAL_1 */
typedef struct {
  __REG32 OFF           					: 8;
  __REG32 RET           					: 8;
  __REG32 ONLP          					: 8;
  __REG32 ON     	        				: 8;
} __prm_vc_cmd_val_bits;

/* PRM_VC_CH_CONF */
typedef struct {
  __REG32 SA0           					: 1;
  __REG32 RAV0           					: 1;
  __REG32 RAC0          					: 1;
  __REG32 RACEN0 	        				: 1;
  __REG32 CMD0   	        				: 1;
  __REG32       	        				:11;
  __REG32 SA1           					: 1;
  __REG32 RAV1           					: 1;
  __REG32 RAC1          					: 1;
  __REG32 RACEN1 	        				: 1;
  __REG32 CMD1   	        				: 1;
  __REG32       	        				:11;
} __prm_vc_ch_conf_bits;

/* PRM_VC_I2C_CFG */
typedef struct {
  __REG32 MCODE          					: 3;
  __REG32 HSEN           					: 1;
  __REG32 SREN          					: 1;
  __REG32 HSMASTER        				: 1;
  __REG32       	        				:26;
} __prm_vc_i2c_cfg_bits;

/* PRM_VC_BYPASS_VAL */
typedef struct {
  __REG32 SLAVEADDR       				: 7;
  __REG32                					: 1;
  __REG32 REGADDR        					: 8;
  __REG32 DATA            				: 8;
  __REG32 VALID            				: 1;
  __REG32       	        				: 7;
} __prm_vc_bypass_val_bits;

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
  __REG32 RSTTIME2								: 5;
  __REG32        	        				:19;
} __prm_rsttime_bits;

/* PRM_RSTST */
typedef struct {
  __REG32 GLOBAL_COLD_RST					  : 1;
  __REG32 GLOBAL_SW_RST						  : 1;
  __REG32        	        				  : 2;
  __REG32 MPU_WD_RST							  : 1;
  __REG32             						  : 1;
  __REG32 EXTERNAL_WARM_RST				  : 1;
  __REG32 VDD1_VOLTAGE_MANAGER_RST  : 1;
  __REG32 VDD2_VOLTAGE_MANAGER_RST  : 1;
  __REG32 ICEPICK_RST							  : 1;
  __REG32 ICECRUSHER_RST					  : 1;
  __REG32        	        				  :21;
} __prm_rstst_bits;

/* PRM_VOLTCTRL */
typedef struct {
  __REG32 AUTO_SLEEP    					  : 1;
  __REG32 AUTO_RET    						  : 1;
  __REG32 AUTO_OFF    						  : 1;
  __REG32 SEL_OFF     						  : 1;
  __REG32 SEL_VMODE  							  : 1;
  __REG32        	        				  :27;
} __prm_voltctrl_bits;

/* PRM_SRAM_PCHARGE */
typedef struct {
  __REG32 PCHARGE_TIME   					  : 8;
  __REG32        	        				  :24;
} __prm_sram_pcharge_bits;

/* PRM_CLKSRC_CTRL */
typedef struct {
  __REG32 SYSCLKSEL								: 2;
  __REG32        	        				: 1;
  __REG32 AUTOEXTCLKMODE					: 2;
  __REG32        	        				: 1;
  __REG32 SYSCLKDIV								: 2;
  __REG32 DPLL4_CLKINP_DIV				: 1;
  __REG32        	        				:23;
} __prm_clksrc_ctrl_bits;

/* PRM_OBS */
typedef struct {
  __REG32 OBS_BUS									:18;
  __REG32        	        				:14;
} __prm_obs_bits;

/* PRM_VOLTSETUP1 */
typedef struct {
  __REG32 SETUPTIME1							:16;
  __REG32 SETUPTIME2							:16;
} __prm_voltsetup1_bits;

/* PRM_VOLTOFFSET */
typedef struct {
  __REG32 OFFSET_TIME							:16;
  __REG32           							:16;
} __prm_voltoffset_bits;

/* PRM_CLKSETUP */
typedef struct {
  __REG32 SETUP_TIME							:16;
  __REG32        	        				:16;
} __prm_clksetup_bits;

/* PRM_POLCTRL */
typedef struct {
  __REG32 EXTVOL_POL       				: 1;
  __REG32 CLKREQ_POL							: 1;
  __REG32 CLKOUT_POL							: 1;
  __REG32 OFFMODE_POL 						: 1;
  __REG32        	        				:28;
} __prm_polctrl_bits;

/* PRM_VOLTSETUP2 */
typedef struct {
  __REG32 OFFMODESETUPTIME 				:16;
  __REG32        	        				:16;
} __prm_voltsetup2_bits;

/* PRM_VP1_CONFIG */
/* PRM_VP2_CONFIG */
typedef struct {
  __REG32 VPENABLE         				: 1;
  __REG32 FORCEUPDATE     				: 1;
  __REG32 INITVDD         				: 1;
  __REG32 TIMEOUTEN        				: 1;
  __REG32                 				: 4;
  __REG32 INITVOLTAGE      				: 8;
  __REG32 ERRORGAIN        				: 8;
  __REG32 ERROROFFSET      				: 8;
} __prm_vp_config_bits;

/* PRM_VP1_VSTEPMIN */
/* PRM_VP2_VSTEPMIN */
typedef struct {
  __REG32 VSTEPMIN        				: 8;
  __REG32 SMPSWAITTIMEMIN  				:16;
  __REG32                 				: 8;
} __prm_vp_vstepmin_bits;

/* PRM_VP1_VSTEPMAX */
/* PRM_VP2_VSTEPMAX */
typedef struct {
  __REG32 VSTEPMAX        				: 8;
  __REG32 SMPSWAITTIMEMAX  				:16;
  __REG32                 				: 8;
} __prm_vp_vstepmax_bits;

/* PRM_VP1_VLIMITTO */
/* PRM_VP2_VLIMITTO */
typedef struct {
  __REG32 TIMEOUT         				:16;
  __REG32 VDDMIN          				: 8;
  __REG32 VDDMAX          				: 8;
} __prm_vp_vlimitto_bits;

/* PRM_VP1_VOLTAGE */
/* PRM_VP2_VOLTAGE */
typedef struct {
  __REG32 VPVOLTAGE        				: 8;
  __REG32                 				:24;
} __prm_vp_voltage_bits;

/* PRM_VP1_STATUS */
/* PRM_VP2_STATUS */
typedef struct {
  __REG32 VPINIDLE        				: 1;
  __REG32                 				:31;
} __prm_vp_status_bits;

/* PRM_LDO_ABB_SETUP */
typedef struct {
  __REG32 OPP_SEL         				: 2;
  __REG32 OPP_CHANGE       				: 1;
  __REG32 SR2_STATUS       				: 2;
  __REG32                  				: 1;
  __REG32 SR2_IN_TRANSITION				: 1;
  __REG32                 				:25;
} __prm_ldo_abb_setup_bits;

/* PRM_LDO_ABB_CTRL */
typedef struct {
  __REG32 SR2EN           				: 1;
  __REG32                  				: 1;
  __REG32 ACTIVE_FBB_SEL   				: 1;
  __REG32                  				: 5;
  __REG32 SR2_WTCNT_VALUE 				: 8;
  __REG32                 				:16;
} __prm_ldo_abb_ctrl_bits;

/* RM_RSTST_NEON */
typedef struct {
  __REG32 GLOBALCOLD_RST					: 1;
  __REG32 GLOBALWARM_RST					: 1;
  __REG32 DOMAINWKUP_RST					: 1;
  __REG32 COREDOMAINWKUP_RST			: 1;
  __REG32        	        				:28;
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
  __REG32 DOMAINWKUP_RST					: 1;
  __REG32 COREDOMAINWKUP_RST  		: 1;
  __REG32        	        				:28;
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

/* PM_IVA2GRPSEL_USBHOST */
typedef struct {
  __REG32 GRPSEL_USBHOST					: 1;
  __REG32        	        				:31;
} __pm_iva2grpsel_usbhost_bits;

/* PM_WKST_USBHOST */
typedef struct {
  __REG32 ST_USBHOST							: 1;
  __REG32        	        				:31;
} __pm_wkst_usbhost_bits;

/* PM_WKDEP_USBHOST */
typedef struct {
  __REG32 EN_CORE									: 1;
  __REG32 EN_MPU									: 1;
  __REG32 EN_IVA2									: 1;
  __REG32        	        				: 1;
  __REG32 EN_WKUP									: 1;
  __REG32        	        				:27;
} __pm_wkdep_usbhost_bits;

/* PM_PWSTCTRL_USBHOST */
typedef struct {
  __REG32 POWERSTATE							: 2;
  __REG32 LOGICRETSTATE						: 1;
  __REG32        	        				: 1;
  __REG32 SAVEANDRESTORE  				: 1;
  __REG32        	        				: 3;
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

/* SR1_SRCONFIG */
/* SR2_SRCONFIG */
typedef struct {
  __REG32 SENPENABLE          		: 1;
  __REG32 SENNENABLE          		: 1;
  __REG32                     		: 6;
  __REG32 MINMAXAVGENABLE     		: 1;
  __REG32 ERRORGENERATORENABLE		: 1;
  __REG32 SENENABLE           		: 1;
  __REG32 SRENABLE             		: 1;
  __REG32 SRCLKLENGTH         		:10;
  __REG32 ACCUMDATA           		:10;
} __sr_srconfig_bits;

/* SR1_SRSTATUS */
/* SR2_SRSTATUS */
typedef struct {
  __REG32 MINMAXAVGACCUMVALID 		: 1;
  __REG32 ERRORGENERATORVALID  		: 1;
  __REG32 MINMAXAVGVALID       		: 1;
  __REG32 AVGERRVALID         		: 1;
  __REG32                      		:28;
} __sr_srstatus_bits;

/* SR1_SENVAL */
/* SR2_SENVAL */
typedef struct {
  __REG32 SENNVAL              		:16;
  __REG32 SENPVAL              		:16;
} __sr_senval_bits;

/* SR1_SENMIN */
/* SR2_SENMIN */
typedef struct {
  __REG32 SENNMIN              		:16;
  __REG32 SENPMIN              		:16;
} __sr_senmin_bits;

/* SR1_SENMAX */
/* SR2_SENMAX */
typedef struct {
  __REG32 SENNMAX              		:16;
  __REG32 SENPMAX                 :16;
} __sr_senmax_bits;

/* SR1_SENAVG */
/* SR2_SENAVG */
typedef struct {
  __REG32 SENNAVG              		:16;
  __REG32 SENPAVG                 :16;
} __sr_senavg_bits;

/* SR1_AVGWEIGHT */
/* SR2_AVGWEIGHT */
typedef struct {
  __REG32 SENNAVGWEIGHT        		: 2;
  __REG32 SENPAVGWEIGHT        		: 2;
  __REG32                     		:28;
} __sr_avgweight_bits;

/* SR1_NVALUERECIPROCAL */
/* SR2_NVALUERECIPROCAL */
typedef struct {
  __REG32 RNSENN              		: 8;
  __REG32 RNSENP              		: 8;
  __REG32 SENNGAIN            		: 4;
  __REG32 SENPGAIN            		: 4;
  __REG32                     		: 8;
} __sr_nvaluereciprocal_bits;

/* SR1_IRQSTATUS_RAW */
/* SR2_IRQSTATUS_RAW */
typedef struct {
  __REG32 MCUDISABLEACKINTSTATRAW	: 1;
  __REG32 MCUBOUNDSINTSTATRAW   	: 1;
  __REG32 MCUVALIDINTSTATRAW    	: 1;
  __REG32 MCUACCUMINTSTATRAW    	: 1;
  __REG32                     		:28;
} __sr_irqstatus_raw_bits;

/* SR1_IRQSTATUS */
/* SR2_IRQSTATUS */
typedef struct {
  __REG32 MCUDISABLEACKINTSTATENA	: 1;
  __REG32 MCUBOUNDSINTSTATENA   	: 1;
  __REG32 MCUVALIDINTSTATENA    	: 1;
  __REG32 MCUACCUMINTSTATENA    	: 1;
  __REG32                     		:28;
} __sr_irqstatus_bits;

/* SR1_IRQENABLE_SET */
/* SR2_IRQENABLE_SET */
typedef struct {
  __REG32 MCUDISABLEACKINTENASET	: 1;
  __REG32 MCUBOUNDSINTENASET    	: 1;
  __REG32 MCUVALIDINTENASET     	: 1;
  __REG32 MCUACCUMINTENASET     	: 1;
  __REG32                     		:28;
} __sr_irqenable_set_bits;

/* SR1_IRQENABLE_CLR */
/* SR2_IRQENABLE_CLR */
typedef struct {
  __REG32 MCUDISABLEACKINTENACLR	: 1;
  __REG32 MCUBOUNDSINTENACLR    	: 1;
  __REG32 MCUVALIDINTENACLR     	: 1;
  __REG32 MCUACCUMINTENACLR     	: 1;
  __REG32                     		:28;
} __sr_irqenable_clr_bits;

/* SR1_SENERROR_REG */
/* SR2_SENERROR_REG */
typedef struct {
  __REG32 SENERROR              	: 8;
  __REG32 AVGERROR              	: 8;
  __REG32                     		:16;
} __sr_senerror_reg_bits;

/* SR1_ERRCONFIG */
/* SR2_ERRCONFIG */
typedef struct {
  __REG32 ERRMINLIMIT            	: 8;
  __REG32 ERRMAXLIMIT            	: 8;
  __REG32 ERRWEIGHT             	: 3;
  __REG32                        	: 3;
  __REG32 VPBOUNDSINTENABLE      	: 1;
  __REG32 VPBOUNDSINTSTATENA    	: 1;
  __REG32 IDLEMODE              	: 2;
  __REG32 WAKEUPENABLE           	: 1;
  __REG32                     		: 5;
} __sr_errconfig_bits;

/* ISP_REVISION */
typedef struct {
  __REG32 REV                 : 8;
  __REG32                  		:24;
} __isp_revision_bits;

/* ISP_SYSCONFIG */
typedef struct {
  __REG32 AUTO_IDLE           : 1;
  __REG32 SOFT_RESET          : 1;
  __REG32                     :10;
  __REG32 MIDLE_MODE          : 2;
  __REG32                  		:18;
} __isp_sysconfig_bits;

/* ISP_SYSSTATUS */
typedef struct {
  __REG32 RESET_DONE          : 1;
  __REG32                  		:31;
} __isp_sysstatus_bits;

/* ISP_IRQ0ENABLE */
/* ISP_IRQ0STATUS */
typedef struct {
  __REG32 CSI2A_IRQ                   : 1;
  __REG32 CSI2C_IRQ                   : 1;
  __REG32                             : 1;
  __REG32 CSIB_LCM_IRQ                : 1;
  __REG32 CSIB_LC0_IRQ                : 1;
  __REG32 CSIB_LC1_IRQ                : 1;
  __REG32 CSIB_LC2_IRQ                : 1;
  __REG32 CSIB_LC3_IRQ                : 1;
  __REG32 CCDC_VD0_IRQ                : 1;
  __REG32 CCDC_VD1_IRQ                : 1;
  __REG32 CCDC_VD2_IRQ                : 1;
  __REG32 CCDC_ERR_IRQ                : 1;
  __REG32 H3A_AF_DONE_IRQ             : 1;
  __REG32 H3A_AWB_DONE_IRQ            : 1;
  __REG32                             : 2;
  __REG32 HIST_DONE_IRQ               : 1;
  __REG32 CCDC_LSC_DONE               : 1;
  __REG32 CCDC_LSC_PREFETCH_COMPLETED : 1;
  __REG32 CCDC_LSC_PREFETCH_ERROR     : 1;
  __REG32 PRV_DONE_IRQ                : 1;
  __REG32 CBUFF_IRQ                   : 1;
  __REG32                             : 2;
  __REG32 RSZ_DONE_IRQ                : 1;
  __REG32 OVF_IRQ                     : 1;
  __REG32                             : 2;
  __REG32 MMU_ERR_IRQ                 : 1;
  __REG32 OCP_ERR_IRQ                 : 1;
  __REG32                             : 1;
  __REG32 HS_VS_IRQ                   : 1;
} __isp_irqxenable_bits;

/* TCTRL_GRESET_LENGTH */
typedef struct {
  __REG32 LENGTH              :24;
  __REG32                     : 8;
} __tctrl_greset_length_bits;

/* TCTRL_PSTRB_REPLAY */
typedef struct {
  __REG32 DELAY               :25;
  __REG32 COUNTER             : 7;
} __tctrl_pstrb_replay_bits;

/* ISP_CTRL */
typedef struct {
  __REG32 PAR_SER_CLK_SEL     : 2;
  __REG32 PAR_BRIDGE          : 2;
  __REG32 PAR_CLK_POL         : 1;
  __REG32                     : 1;
  __REG32 SHIFT               : 2;
  __REG32 CCDC_CLK_EN         : 1;
  __REG32 CBUFF_AUTOGATING    : 1;
  __REG32 H3A_CLK_EN          : 1;
  __REG32 HIST_CLK_EN         : 1;
  __REG32 PRV_CLK_EN          : 1;
  __REG32 RSZ_CLK_EN          : 1;
  __REG32 SYNC_DETECT         : 2;
  __REG32 CCDC_RAM_EN         : 1;
  __REG32 PREV_RAM_EN         : 1;
  __REG32 SBL_RD_RAM_EN       : 1;
  __REG32 SBL_WR1_RAM_EN      : 1;
  __REG32 SBL_WR0_RAM_EN      : 1;
  __REG32 SBL_AUTOIDLE        : 1;
  __REG32 CBUFF0_BCF_CTRL     : 2;
  __REG32 CBUFF1_BCF_CTRL     : 2;
  __REG32 SBL_SHARED_WPORTC   : 1;
  __REG32 SBL_SHARED_RPORTA   : 1;
  __REG32 SBL_SHARED_RPORTB   : 1;
  __REG32 CCDC_WEN_POL        : 1;
  __REG32 JPEG_FLUSH          : 1;
  __REG32 FLUSH               : 1;
} __isp_ctrl_bits;

/* TCTRL_CTRL */
typedef struct {
  __REG32 DIVA                : 5;
  __REG32 DIVB                : 5;
  __REG32 DIVC                : 9;
  __REG32                     : 2;
  __REG32 SHUTEN              : 1;
  __REG32 PSTRBEN             : 1;
  __REG32 STRBEN              : 1;
  __REG32 SHUTPOL             : 1;
  __REG32                     : 1;
  __REG32 STRBPSTRBPOL        : 1;
  __REG32 INSEL               : 2;
  __REG32 GRESETEN            : 1;
  __REG32 GRESETPOL           : 1;
  __REG32 GRESETDIR           : 1;
} __tctrl_ctrl_bits;

/* TCTRL_FRAME */
typedef struct {
  __REG32 SHUT                : 6;
  __REG32 PSTRB               : 6;
  __REG32 STRB                : 6;
  __REG32                     : 1;
  __REG32 CCP2B_EOL_ENABLE    : 1;
  __REG32                     :12;
} __tctrl_frame_bits;

/* TCTRL_PSTRB_DELAY */
/* TCTRL_STRB_DELAY */
/* TCTRL_SHUT_DELAY */
typedef struct {
  __REG32 DELAY               :25;
  __REG32                     : 7;
} __tctrl_pstrb_delay_bits;

/* TCTRL_PSTRB_LENGTH */
/* TCTRL_STRB_LENGTH */
/* TCTRL_STRB_LENGTH */
typedef struct {
  __REG32 LENGTH              :24;
  __REG32                     : 8;
} __tctrl_pstrb_length_bits;

/* CBUFF_REVISION */
typedef struct {
  __REG32 REV                 : 8;
  __REG32                     :24;
} __cbuff_revision_bits;

/* CBUFF_IRQSTATUS */
/* CBUFF_IRQENABLE */
typedef struct {
  __REG32 IRQ_CBUFF0_READY    : 1;
  __REG32 IRQ_CBUFF0_INVALID  : 1;
  __REG32 IRQ_CBUFF0_OVR      : 1;
  __REG32 IRQ_CBUFF1_READY    : 1;
  __REG32 IRQ_CBUFF1_INVALID  : 1;
  __REG32 IRQ_CBUFF1_OVR      : 1;
  __REG32                     :26;
} __cbuff_irqstatus_bits;

/* CBUFFx_CTRL */
typedef struct {
  __REG32 ENABLE              : 1;
  __REG32 RWMODE              : 1;
  __REG32 DONE                : 1;
  __REG32 ALLOW_NW_EQ_CPUW    : 1;
  __REG32 BCF                 : 4;
  __REG32 WCOUNT              : 2;
  __REG32                     :22;
} __cbuffx_ctrl_bits;

/* CBUFFx_STATUS */
typedef struct {
  __REG32 CPUW                : 4;
  __REG32                     : 4;
  __REG32 CW                  : 4;
  __REG32                     : 4;
  __REG32 NW                  : 4;
  __REG32                     :12;
} __cbuffx_status_bits;

/* CBUFFx_WINDOWSIZE */
typedef struct {
  __REG32 SIZE                :24;
  __REG32                     : 8;
} __cbuffx_windowsize_bits;

/* CBUFFx_THRESHOLD */
typedef struct {
  __REG32 THRESHOLD           :24;
  __REG32                     : 8;
} __cbuffx_threshold_bits;

/* CBUFF_VRFB_CTRL */
typedef struct {
  __REG32 ENABLE0             : 1;
  __REG32 BASE0               : 4;
  __REG32 WIDTH0              : 2;
  __REG32 ORIENTATION0        : 2;
  __REG32                     : 1;
  __REG32 ENABLE1             : 1;
  __REG32 BASE1               : 4;
  __REG32 WIDTH1              : 2;
  __REG32 ORIENTATION1        : 2;
  __REG32                     : 1;
  __REG32 ENABLE2             : 1;
  __REG32 BASE2               : 4;
  __REG32 WIDTH2              : 2;
  __REG32 ORIENTATION2        : 2;
  __REG32                     : 3;
} __cbuff_vrfb_ctrl_bits;

/* CCP2_REVISION */
typedef struct {
  __REG32 REV                 : 8;
  __REG32                     :24;
} __ccp2_revision_bits;

/* CCP2_SYSCONFIG */
typedef struct {
  __REG32 AUTO_IDLE           : 1;
  __REG32 SOFT_RESET          : 1;
  __REG32                     :10;
  __REG32 MSTANDBY_MODE       : 2;
  __REG32                     :18;
} __ccp2_sysconfig_bits;

/* CCP2_SYSSTATUS */
typedef struct {
  __REG32 RESET_DONE          : 1;
  __REG32                     :31;
} __ccp2_sysstatus_bits;

/* CCP2_LC01_IRQENABLE */
typedef struct {
  __REG32 LC0_SSC_IRQ         : 1;
  __REG32 LC0_FSC_IRQ         : 1;
  __REG32 LC0_FW_IRQ          : 1;
  __REG32 LC0_FSP_IRQ         : 1;
  __REG32 LC0_CRC_IRQ         : 1;
  __REG32 LC0_FIFO_OVF_IRQ    : 1;
  __REG32                     : 1;
  __REG32 LC0_COUNT_IRQ       : 1;
  __REG32 LC0_FE_IRQ          : 1;
  __REG32 LC0_LS_IRQ          : 1;
  __REG32 LC0_LE_IRQ          : 1;
  __REG32 LC0_FS_IRQ          : 1;
  __REG32                     : 4;
  __REG32 LC1_SSC_IRQ         : 1;
  __REG32 LC1_FSC_IRQ         : 1;
  __REG32 LC1_FW_IRQ          : 1;
  __REG32 LC1_FSP_IRQ         : 1;
  __REG32 LC1_CRC_IRQ         : 1;
  __REG32 LC1_FIFO_OVF_IRQ    : 1;
  __REG32                     : 1;
  __REG32 LC1_COUNT_IRQ       : 1;
  __REG32 LC1_FE_IRQ          : 1;
  __REG32 LC1_LS_IRQ          : 1;
  __REG32 LC1_LE_IRQ          : 1;
  __REG32 LC1_FS_IRQ          : 1;
  __REG32                     : 4;
} __ccp2_lc01_irqenable_bits;

/* CCP2_LC23_IRQENABLE */
typedef struct {
  __REG32 LC2_SSC_IRQ         : 1;
  __REG32 LC2_FSC_IRQ         : 1;
  __REG32 LC2_FW_IRQ          : 1;
  __REG32 LC2_FSP_IRQ         : 1;
  __REG32 LC2_CRC_IRQ         : 1;
  __REG32 LC2_FIFO_OVF_IRQ    : 1;
  __REG32                     : 1;
  __REG32 LC2_COUNT_IRQ       : 1;
  __REG32 LC2_FE_IRQ          : 1;
  __REG32 LC2_LS_IRQ          : 1;
  __REG32 LC2_LE_IRQ          : 1;
  __REG32 LC2_FS_IRQ          : 1;
  __REG32                     : 4;
  __REG32 LC3_SSC_IRQ         : 1;
  __REG32 LC3_FSC_IRQ         : 1;
  __REG32 LC3_FW_IRQ          : 1;
  __REG32 LC3_FSP_IRQ         : 1;
  __REG32 LC3_CRC_IRQ         : 1;
  __REG32 LC3_FIFO_OVF_IRQ    : 1;
  __REG32                     : 1;
  __REG32 LC3_COUNT_IRQ       : 1;
  __REG32 LC3_FE_IRQ          : 1;
  __REG32 LC3_LS_IRQ          : 1;
  __REG32 LC3_LE_IRQ          : 1;
  __REG32 LC3_FS_IRQ          : 1;
  __REG32                     : 4;
} __ccp2_lc23_irqenable_bits;

/* CCP2_LCM_IRQENABLE */
typedef struct {
  __REG32 LCM_EOF             : 1;
  __REG32 LCM_OCPERROR        : 1;
  __REG32                     :30;
} __ccp2_lcm_irqenable_bits;

/* CCP2_CTRL */
typedef struct {
  __REG32 IF_EN               : 1;
  __REG32 PHY_SEL             : 1;
  __REG32 IO_OUT_SEL          : 1;
  __REG32 FRAME               : 1;
  __REG32 MODE                : 1;
  __REG32 BURST               : 3;
  __REG32                     : 1;
  __REG32 VP_CLK_FORCE_ON     : 1;
  __REG32 INV                 : 1;
  __REG32 VP_ONLY_EN          : 1;
  __REG32 VP_CLK_POL          : 1;
  __REG32 DBG_EN              : 1;
  __REG32 POSTED              : 1;
  __REG32 FRACDIV             :17;
} __ccp2_ctrl_bits;

/* CCP2_GNQ */
typedef struct {
  __REG32 NBCHANNELS          : 2;
  __REG32 FIFODEPTH           : 3;
  __REG32 OCPREADPORT         : 1;
  __REG32                     :26;
} __ccp2_gnq_bits;

/* CCP2_CTRL1 */
typedef struct {
  __REG32 BLANKING            : 2;
  __REG32                     :14;
  __REG32 LEVL                : 7;
  __REG32                     : 1;
  __REG32 LEVH                : 7;
  __REG32                     : 1;
} __ccp2_ctrl1_bits;

/* CCP2_LCx_CTRL */
typedef struct {
  __REG32 CHAN_EN             : 1;
  __REG32 REGION_EN           : 1;
  __REG32 FORMAT              : 6;
  __REG32 ALPHA               : 8;
  __REG32 COUNT_UNLOCK        : 1;
  __REG32 PING_PONG           : 1;
  __REG32 DPCM_PRED           : 1;
  __REG32 CRC_EN              : 1;
  __REG32                     : 4;
  __REG32 COUNT               : 8;
} __ccp2_lcx_ctrl_bits;

/* CCP2_LCx_CODE */
typedef struct {
  __REG32 LSC                 : 4;
  __REG32 LEC                 : 4;
  __REG32 FSC                 : 4;
  __REG32 FEC                 : 4;
  __REG32 CHAN_ID             : 4;
  __REG32                     :12;
} __ccp2_lcx_code_bits;

/* CCP2_LCx_STAT_START */
typedef struct {
  __REG32 _SOF                :12;
  __REG32                     : 4;
  __REG32 _EOF                :12;
  __REG32                     : 4;
} __ccp2_lcx_stat_start_bits;

/* CCP2_LCx_STAT_SIZE */
typedef struct {
  __REG32 _SOF                :12;
  __REG32                     : 4;
  __REG32 _EOF                :12;
  __REG32                     : 4;
} __ccp2_lcx_stat_size_bits;

/* CCP2_LCx_DAT_START */
typedef struct {
  __REG32                     :16;
  __REG32 VERT                :12;
  __REG32                     : 4;
} __ccp2_lcx_dat_start_bits;

/* CCP2_LCx_DAT_SIZE */
typedef struct {
  __REG32                     :16;
  __REG32 VERT                :12;
  __REG32                     : 4;
} __ccp2_lcx_dat_size_bits;

/* CCP2_LCM_CTRL */
typedef struct {
  __REG32 CHAN_EN             : 1;
  __REG32                     : 1;
  __REG32 DST_PORT            : 1;
  __REG32 READ_THROTTLE       : 2;
  __REG32 BURST_SIZE          : 3;
  __REG32                     : 8;
  __REG32 SRC_FORMAT          : 3;
  __REG32                     : 4;
  __REG32 SRC_PACK            : 1;
  __REG32 DST_FORMAT          : 3;
  __REG32                     : 4;
  __REG32 DST_PACK            : 1;
} __ccp2_lcm_ctrl_bits;

/* CCP2_LCM_VSIZE */
typedef struct {
  __REG32                     :16;
  __REG32 COUNT               :13;
  __REG32                     : 3;
} __ccp2_lcm_vsize_bits;

/* CCP2_LCM_HSIZE */
typedef struct {
  __REG32 SKIP                :13;
  __REG32                     : 3;
  __REG32 COUNT               :13;
  __REG32                     : 3;
} __ccp2_lcm_hsize_bits;

/* CCP2_LCM_PREFETCH */
typedef struct {
  __REG32 HWORDS              :15;
  __REG32                     :17;
} __ccp2_lcm_prefetch_bits;

/* CCDC_PID */
typedef struct {
  __REG32 PREV                : 8;
  __REG32 CID                 : 8;
  __REG32 TID                 : 8;
  __REG32                     : 8;
} __ccdc_pid_bits;

/* CCDC_PCR */
typedef struct {
  __REG32 ENABLE              : 1;
  __REG32 BUSY                : 1;
  __REG32                     :30;
} __ccdc_pcr_bits;

/*CCDC_SYN_MODE */
typedef struct {
  __REG32 VDHDOUT             : 1;
  __REG32 FLDOUT              : 1;
  __REG32 VDPOL               : 1;
  __REG32 HDPOL               : 1;
  __REG32 FLDPOL              : 1;
  __REG32 EXWEN               : 1;
  __REG32 DATAPOL             : 1;
  __REG32 FLDMODE             : 1;
  __REG32 DATSIZ              : 3;
  __REG32 PACK8               : 1;
  __REG32 INPMOD              : 2;
  __REG32 LPF                 : 1;
  __REG32 FLDSTAT             : 1;
  __REG32 VDHDEN              : 1;
  __REG32 WEN                 : 1;
  __REG32 VP2SDR              : 1;
  __REG32 SDR2RSZ             : 1;
  __REG32                     :12;
} __ccdc_syn_mode_bits;

/*CCDC_HD_VD_WID */
typedef struct {
  __REG32 VDW                 :12;
  __REG32                     : 4;
  __REG32 HDW                 :12;
  __REG32                     : 4;
} __ccdc_hd_vd_wid_bits;

/* CCDC_PIX_LINES */
typedef struct {
  __REG32 HLPRF               :16;
  __REG32 PPLN                :16;
} __ccdc_pix_lines_bits;

/* CCDC_HORZ_INFO */
typedef struct {
  __REG32 NPH                 :15;
  __REG32                     : 1;
  __REG32 SPH                 :15;
  __REG32                     : 1;
} __ccdc_horz_info_bits;

/* CCDC_VERT_START */
typedef struct {
  __REG32 SLV1                :15;
  __REG32                     : 1;
  __REG32 SLV0                :15;
  __REG32                     : 1;
} __ccdc_vert_start_bits;

/* CCDC_VERT_LINES */
typedef struct {
  __REG32 NLV                 :15;
  __REG32                     :17;
} __ccdc_vert_lines_bits;

/* CCDC_CULLING */
typedef struct {
  __REG32 CULV                : 8;
  __REG32                     : 8;
  __REG32 CULHODD             : 8;
  __REG32 CULHEVN             : 8;
} __ccdc_culling_bits;

/* CCDC_HSIZE_OFF */
typedef struct {
  __REG32 LNOFST              :16;
  __REG32                     :16;
} __ccdc_hsize_off_bits;

/* CCDC_SDOFST */
typedef struct {
  __REG32 LOFST3              : 3;
  __REG32 LOFST2              : 3;
  __REG32 LOFST1              : 3;
  __REG32 LOFST0              : 3;
  __REG32 FOFST               : 2;
  __REG32 FIINV               : 1;
  __REG32                     :17;
} __ccdc_sdofst_bits;

/* CCDC_CLAMP */
typedef struct {
  __REG32 OBGAIN              : 5;
  __REG32                     : 5;
  __REG32 OBST                :15;
  __REG32 OBSLN               : 3;
  __REG32 OBSLEN              : 3;
  __REG32 CLAMPEN             : 1;
} __ccdc_clamp_bits;

/* CCDC_DCSUB */
typedef struct {
  __REG32 DCSUB               :14;
  __REG32                     :18;
} __ccdc_dcsub_bits;

/* CCDC_COLPTN */
typedef struct {
  __REG32 CP0PLC0             : 2;
  __REG32 CP0PLC1             : 2;
  __REG32 CP0PLC2             : 2;
  __REG32 CP0PLC3             : 2;
  __REG32 CP1PLC0             : 2;
  __REG32 CP1PLC1             : 2;
  __REG32 CP1PLC2             : 2;
  __REG32 CP1PLC3             : 2;
  __REG32 CP2PLC0             : 2;
  __REG32 CP2PLC1             : 2;
  __REG32 CP2PLC2             : 2;
  __REG32 CP2PLC3             : 2;
  __REG32 CP3PLC0             : 2;
  __REG32 CP3PLC1             : 2;
  __REG32 CP3PLC2             : 2;
  __REG32 CP3PLC3             : 2;
} __ccdc_colptn_bits;

/* CCDC_BLKCMP */
typedef struct {
  __REG32 B_MG                : 8;
  __REG32 GB_G                : 8;
  __REG32 GR_CY               : 8;
  __REG32 R_YE                : 8;
} __ccdc_blkcmp_bits;

/* CCDC_FPC */
typedef struct {
  __REG32 FPNUM               :15;
  __REG32 FPCEN               : 1;
  __REG32 FPERR               : 1;
  __REG32                     :15;
} __ccdc_fpc_bits;

/* CCDC_VDINT */
typedef struct {
  __REG32 VDINT1              :15;
  __REG32                     : 1;
  __REG32 VDINT0              :15;
  __REG32                     : 1;
} __ccdc_vdint_bits;

/* CCDC_ALAW */
typedef struct {
  __REG32 GWDI                : 3;
  __REG32 CCDTBL              : 1;
  __REG32                     :28;
} __ccdc_alaw_bits;

/* CCDC_REC656IF */
typedef struct {
  __REG32 R656ON              : 1;
  __REG32 ECCFVH              : 1;
  __REG32                     :30;
} __ccdc_rec656if_bits;

/* CCDC_CFG */
typedef struct {
  __REG32                     : 5;
  __REG32 BW656               : 1;
  __REG32 FIDMD               : 2;
  __REG32 WENLOG              : 1;
  __REG32                     : 2;
  __REG32 Y8POS               : 1;
  __REG32 BSWD                : 1;
  __REG32 MSBINVI             : 1;
  __REG32                     : 1;
  __REG32 VDLC                : 1;
  __REG32                     :16;
} __ccdc_cfg_bits;

/* CCDC_FMTCFG */
typedef struct {
  __REG32 FMTEN               : 1;
  __REG32 LNALT               : 1;
  __REG32 LNUM                : 2;
  __REG32 PLEN_ODD            : 4;
  __REG32 PLEN_EVEN           : 4;
  __REG32 VPIN                : 3;
  __REG32 VPEN                : 1;
  __REG32 VPIF_FRQ            : 6;
  __REG32                     :10;
} __ccdc_fmtcfg_bits;

/* CCDC_FMT_HORZ */
typedef struct {
  __REG32 FMTLNH              :13;
  __REG32                     : 3;
  __REG32 FMTSPH              :13;
  __REG32                     : 3;
} __ccdc_fmt_horz_bits;

/* CCDC_FMT_VERT */
typedef struct {
  __REG32 FMTLNV              :13;
  __REG32                     : 3;
  __REG32 FMTSLV              :13;
  __REG32                     : 3;
} __ccdc_fmt_vert_bits;

/* CCDC_FMT_ADDR_x */
typedef struct {
  __REG32 INIT                :13;
  __REG32                     :11;
  __REG32 LINE                : 2;
  __REG32                     : 6;
} __ccdc_fmt_addr_x_bits;

/* CCDC_PRGEVEN0 */
typedef struct {
  __REG32 EVEN0               : 4;
  __REG32 EVEN1               : 4;
  __REG32 EVEN2               : 4;
  __REG32 EVEN3               : 4;
  __REG32 EVEN4               : 4;
  __REG32 EVEN5               : 4;
  __REG32 EVEN6               : 4;
  __REG32 EVEN7               : 4;
} __ccdc_prgeven0_bits;

/* CCDC_PRGEVEN1 */
typedef struct {
  __REG32 EVEN8               : 4;
  __REG32 EVEN9               : 4;
  __REG32 EVEN10              : 4;
  __REG32 EVEN11              : 4;
  __REG32 EVEN12              : 4;
  __REG32 EVEN13              : 4;
  __REG32 EVEN14              : 4;
  __REG32 EVEN15              : 4;
} __ccdc_prgeven1_bits;

/* CCDC_PRGODD0 */
typedef struct {
  __REG32 ODD0                : 4;
  __REG32 ODD1                : 4;
  __REG32 ODD2                : 4;
  __REG32 ODD3                : 4;
  __REG32 ODD4                : 4;
  __REG32 ODD5                : 4;
  __REG32 ODD6                : 4;
  __REG32 ODD7                : 4;
} __ccdc_prgodd0_bits;

/* CCDC_PRGODD1 */
typedef struct {
  __REG32 ODD8                : 4;
  __REG32 ODD9                : 4;
  __REG32 ODD10               : 4;
  __REG32 ODD11               : 4;
  __REG32 ODD12               : 4;
  __REG32 ODD13               : 4;
  __REG32 ODD14               : 4;
  __REG32 ODD15               : 4;
} __ccdc_prgodd1_bits;

/* CCDC_VP_OUT */
typedef struct {
  __REG32 HORZ_ST             : 4;
  __REG32 HORZ_NUM            :13;
  __REG32 VERT_NUM            :14;
  __REG32                     : 1;
} __ccdc_vp_out_bits;

/* CCDC_LSC_CONFIG */
typedef struct {
  __REG32 ENABLE              : 1;
  __REG32 GAIN_FORMAT         : 3;
  __REG32                     : 2;
  __REG32 AFTER_REFORMATTER   : 1;
  __REG32 BUSY                : 1;
  __REG32 GAIN_MODE_N         : 3;
  __REG32                     : 1;
  __REG32 GAIN_MODE_M         : 3;
  __REG32                     :17;
} __ccdc_lsc_config_bits;

/* CCDC_LSC_INITIAL */
typedef struct {
  __REG32 X                   : 6;
  __REG32                     :10;
  __REG32 Y                   : 6;
  __REG32                     :10;
} __ccdc_lsc_initial_bits;

/* CCDC_LSC_TABLE_OFFSET */
typedef struct {
  __REG32 OFFSET              :16;
  __REG32                     :16;
} __ccdc_lsc_table_offset_bits;

/* HIST_PID */
typedef struct {
  __REG32 PREV                : 8;
  __REG32 CID                 : 8;
  __REG32 TID                 : 8;
  __REG32                     : 8;
} __hist_pid_bits;

/* HIST_PCR */
typedef struct {
  __REG32 ENABLE              : 1;
  __REG32 BUSY                : 1;
  __REG32                     :30;
} __hist_pcr_bits;

/* HIST_CNT */
typedef struct {
  __REG32 SHIFT               : 3;
  __REG32 SOURCE              : 1;
  __REG32 BINS                : 2;
  __REG32 CFA                 : 1;
  __REG32 CLR                 : 1;
  __REG32 DATSIZ              : 1;
  __REG32                     :23;
} __hist_cnt_bits;

/* HIST_WB_GAIN */
typedef struct {
  __REG32 WG03                : 8;
  __REG32 WG02                : 8;
  __REG32 WG01                : 8;
  __REG32 WG00                : 8;
} __hist_wb_gain_bits;

/* HIST_Rx_HORZ */
typedef struct {
  __REG32 HEND                :14;
  __REG32                     : 2;
  __REG32 HSTART              :14;
  __REG32                     : 2;
} __hist_rx_horz_bits;

/* HIST_Rx_VERT */
typedef struct {
  __REG32 VEND                :14;
  __REG32                     : 2;
  __REG32 VSTART              :14;
  __REG32                     : 2;
} __hist_rx_vert_bits;

/* HIST_ADDR */
typedef struct {
  __REG32 ADDR                :10;
  __REG32                     :22;
} __hist_addr_bits;

/* HIST_DATA */
typedef struct {
  __REG32 RDATA               :20;
  __REG32                     :12;
} __hist_data_bits;

/* HIST_RADD_OFF */
typedef struct {
  __REG32 RDATA               :16;
  __REG32                     :16;
} __hist_radd_off_bits;

/* HIST_H_V_INFO */
typedef struct {
  __REG32 VSIZE               :14;
  __REG32                     : 2;
  __REG32 HSIZE               :14;
  __REG32                     : 2;
} __hist_h_v_info_bits;

/* H3A_PID */
typedef struct {
  __REG32 PREV                : 8;
  __REG32 CID                 : 8;
  __REG32 TID                 : 8;
  __REG32                     : 8;
} __h3a_pid_bits;

/* H3A_PCR */
typedef struct {
  __REG32 AF_EN               : 1;
  __REG32 AF_ALAW_EN          : 1;
  __REG32 AF_MED_EN           : 1;
  __REG32 MED_TH              : 8;
  __REG32 RGBPOS              : 3;
  __REG32 FVMODE              : 1;
  __REG32 BUSYAF              : 1;
  __REG32 AEW_EN              : 1;
  __REG32 AEW_ALAW_EN         : 1;
  __REG32 BUSYAEAWB           : 1;
  __REG32                     : 3;
  __REG32 AVE2LMT             :10;
} __h3a_pcr_bits;

/* H3A_AFPAX1 */
typedef struct {
  __REG32 PAXH                : 7;
  __REG32                     : 9;
  __REG32 PAXW                : 7;
  __REG32                     : 9;
} __h3a_afpax1_bits;

/* H3A_AFPAX2 */
typedef struct {
  __REG32 PAXHC               : 6;
  __REG32 PAXVC               : 7;
  __REG32 AFINCV              : 4;
  __REG32                     :15;
} __h3a_afpax2_bits;

/* H3A_AFPAXSTART */
typedef struct {
  __REG32 PAXSV               :12;
  __REG32                     : 4;
  __REG32 PAXSH               :12;
  __REG32                     : 4;
} __h3a_afpaxstart_bits;

/* H3A_AFIIRSH */
typedef struct {
  __REG32 IIRSH               :12;
  __REG32                     :20;
} __h3a_afiirsh_bits;


/* H3A_AFCOEF010 */
/* H3A_AFCOEF110 */
typedef struct {
  __REG32 COEFF0              :12;
  __REG32                     : 4;
  __REG32 COEFF1              :12;
  __REG32                     : 4;
} __h3a_afcoefx10_bits;

/* H3A_AFCOEF032 */
/* H3A_AFCOEF132 */
typedef struct {
  __REG32 COEFF2              :12;
  __REG32                     : 4;
  __REG32 COEFF3              :12;
  __REG32                     : 4;
} __h3a_afcoefx32_bits;

/* H3A_AFCOEF054 */
/* H3A_AFCOEF154 */
typedef struct {
  __REG32 COEFF4              :12;
  __REG32                     : 4;
  __REG32 COEFF5              :12;
  __REG32                     : 4;
} __h3a_afcoefx54_bits;

/* H3A_AFCOEF076 */
/* H3A_AFCOEF176 */
typedef struct {
  __REG32 COEFF6              :12;
  __REG32                     : 4;
  __REG32 COEFF7              :12;
  __REG32                     : 4;
} __h3a_afcoefx76_bits;

/* H3A_AFCOEF098 */
/* H3A_AFCOEF198 */
typedef struct {
  __REG32 COEFF8              :12;
  __REG32                     : 4;
  __REG32 COEFF9              :12;
  __REG32                     : 4;
} __h3a_afcoefx98_bits;

/* H3A_AFCOEF0010 */
/* H3A_AFCOEF1010 */
typedef struct {
  __REG32 COEFF10             :12;
  __REG32                     :20;
} __h3a_afcoefx010_bits;

/* H3A_AEWWIN1 */
typedef struct {
  __REG32 WINHC               : 6;
  __REG32 WINVC               : 7;
  __REG32 WINW                : 7;
  __REG32                     : 4;
  __REG32 WINH                : 7;
  __REG32                     : 1;
} __h3a_aewwin1_bits;

/* H3A_AEWINSTART */
typedef struct {
  __REG32 WINSH               :12;
  __REG32                     : 4;
  __REG32 WINSV               :12;
  __REG32                     : 4;
} __h3a_aewinstart_bits;

/* H3A_AEWINBLK */
typedef struct {
  __REG32 WINH                : 7;
  __REG32                     : 9;
  __REG32 WINSV               :12;
  __REG32                     : 4;
} __h3a_aewinblk_bits;

/* H3A_AEWSUBWIN */
typedef struct {
  __REG32 AEWINCH             : 4;
  __REG32                     : 4;
  __REG32 AEWINCV             : 4;
  __REG32                     :20;
} __h3a_aewsubwin_bits;

/* PRV_PID */
typedef struct {
  __REG32 PREV                : 8;
  __REG32 CID                 : 8;
  __REG32 TID                 : 8;
  __REG32                     : 8;
} __prv_pid_bits;

/* PRV_PCR */
typedef struct {
  __REG32 ENABLE              : 1;
  __REG32 BUSY                : 1;
  __REG32 SOURCE              : 1;
  __REG32 ONESHOT             : 1;
  __REG32 WIDTH               : 1;
  __REG32 INVALAW             : 1;
  __REG32 DRKFEN              : 1;
  __REG32 DRKFCAP             : 1;
  __REG32 HMEDEN              : 1;
  __REG32 NFEN                : 1;
  __REG32 CFAEN               : 1;
  __REG32 CFAFMT              : 4;
  __REG32 YNENHEN             : 1;
  __REG32 SUPEN               : 1;
  __REG32 YCPOS               : 2;
  __REG32 RSZPORT             : 1;
  __REG32 SDRPORT             : 1;
  __REG32 SCOMP_EN            : 1;
  __REG32 SCOMP_SFT           : 3;
  __REG32                     : 1;
  __REG32 GAMMA_BYPASS        : 1;
  __REG32 DCOREN              : 1;
  __REG32 DCOR_METHOD         : 1;
  __REG32                     : 2;
  __REG32 DRK_FAIL            : 1;
} __prv_pcr_bits;

/* PRV_HORZ_INFO */
typedef struct {
  __REG32 EPH                 :14;
  __REG32                     : 2;
  __REG32 SPH                 :14;
  __REG32                     : 2;
} __prv_horz_info_bits;

/* PRV_VERT_INFO */
typedef struct {
  __REG32 ELV                 :14;
  __REG32                     : 2;
  __REG32 SLV                 :14;
  __REG32                     : 2;
} __prv_vert_info_bits;

/* PRV_RADR_OFFSET */
/* PRV_DRKF_OFFSET */
/* PRV_WADD_OFFSET */
typedef struct {
  __REG32 OFFSET              :16;
  __REG32                     :16;
} __prv_radr_offset_bits;

/* PRV_AVE */
typedef struct {
  __REG32 COUNT               : 2;
  __REG32 EVENDIST            : 2;
  __REG32 ODDDIST             : 2;
  __REG32                     :26;
} __prv_ave_bits;

/* PRV_HMED */
typedef struct {
  __REG32 THRESHOLD           : 8;
  __REG32 EVENDIST            : 1;
  __REG32 ODDDIST             : 1;
  __REG32                     :22;
} __prv_hmed_bits;

/* PRV_NF */
typedef struct {
  __REG32 SPR                 : 2;
  __REG32                     :30;
} __prv_nf_bits;

/* PRV_WB_DGAIN */
typedef struct {
  __REG32 DGAIN               :10;
  __REG32                     :22;
} __prv_wb_dgain_bits;

/* PRV_WBGAIN */
typedef struct {
  __REG32 COEF0               : 8;
  __REG32 COEF1               : 8;
  __REG32 COEF2               : 8;
  __REG32 COEF3               : 8;
} __prv_wbgain_bits;

/* PRV_WBSEL */
typedef struct {
  __REG32 N0_0                : 2;
  __REG32 N0_1                : 2;
  __REG32 N0_2                : 2;
  __REG32 N0_3                : 2;
  __REG32 N1_0                : 2;
  __REG32 N1_1                : 2;
  __REG32 N1_2                : 2;
  __REG32 N1_3                : 2;
  __REG32 N2_0                : 2;
  __REG32 N2_1                : 2;
  __REG32 N2_2                : 2;
  __REG32 N2_3                : 2;
  __REG32 N3_0                : 2;
  __REG32 N3_1                : 2;
  __REG32 N3_2                : 2;
  __REG32 N3_3                : 2;
} __prv_wbsel_bits;

/* PRV_CFA */
typedef struct {
  __REG32 GRADTH_HOR          : 8;
  __REG32 GRADTH_VER          : 8;
  __REG32                     :16;
} __prv_cfa_bits;

/* PRV_BLKADJOFF */
typedef struct {
  __REG32 B                   : 8;
  __REG32 G                   : 8;
  __REG32 R                   : 8;
  __REG32                     : 8;
} __prv_blkadjoff_bits;

/* PRV_RGB_MAT1 */
typedef struct {
  __REG32 MTX_RR              :12;
  __REG32                     : 4;
  __REG32 MTX_GR              :12;
  __REG32                     : 4;
} __prv_rgb_mat1_bits;

/* PRV_RGB_MAT2 */
typedef struct {
  __REG32 MTX_BR              :12;
  __REG32                     : 4;
  __REG32 MTX_RG              :12;
  __REG32                     : 4;
} __prv_rgb_mat2_bits;

/* PRV_RGB_MAT3 */
typedef struct {
  __REG32 MTX_GG              :12;
  __REG32                     : 4;
  __REG32 MTX_BG              :12;
  __REG32                     : 4;
} __prv_rgb_mat3_bits;

/* PRV_RGB_MAT4 */
typedef struct {
  __REG32 MTX_RB              :12;
  __REG32                     : 4;
  __REG32 MTX_GB              :12;
  __REG32                     : 4;
} __prv_rgb_mat4_bits;

/* PRV_RGB_MAT5 */
typedef struct {
  __REG32 MTX_BB              :12;
  __REG32                     :20;
} __prv_rgb_mat5_bits;

/* PRV_RGB_OFF1 */
typedef struct {
  __REG32 MTX_OFFG            :10;
  __REG32                     : 6;
  __REG32 MTX_OFFR            :10;
  __REG32                     : 6;
} __prv_rgb_off1_bits;

/* PRV_RGB_OFF2 */
typedef struct {
  __REG32 MTX_OFFB            :10;
  __REG32                     :22;
} __prv_rgb_off2_bits;

/* PRV_CSC0 */
typedef struct {
  __REG32 CSCRY               :10;
  __REG32 CSCGY               :10;
  __REG32 CSCBY               :10;
  __REG32                     : 2;
} __prv_csc0_bits;

/* PRV_CSC1 */
typedef struct {
  __REG32 CSCRCB              :10;
  __REG32 CSCGCB              :10;
  __REG32 CSCBCB              :10;
  __REG32                     : 2;
} __prv_csc1_bits;

/* PRV_CSC2 */
typedef struct {
  __REG32 CSCRCR              :10;
  __REG32 CSCGCR              :10;
  __REG32 CSCBCR              :10;
  __REG32                     : 2;
} __prv_csc2_bits;

/* PRV_CSC_OFFSET */
typedef struct {
  __REG32 OFSTCR              : 8;
  __REG32 OFSTCB              : 8;
  __REG32 YOFST               : 8;
  __REG32                     : 8;
} __prv_csc_offset_bits;

/* PRV_CNT_BRT */
typedef struct {
  __REG32 BRT                 : 8;
  __REG32 CNT                 : 8;
  __REG32                     :16;
} __prv_cnt_brt_bits;

/* PRV_CSUP */
typedef struct {
  __REG32 CSUPG               : 8;
  __REG32 CSUPTH              : 8;
  __REG32 HPYF                : 1;
  __REG32                     :15;
} __prv_csup_bits;

/* PRV_SETUP_YC */
typedef struct {
  __REG32 MINC                : 8;
  __REG32 MAXC                : 8;
  __REG32 MINY                : 8;
  __REG32 MAXY                : 8;
} __prv_setup_yc_bits;

/* PRV_SET_TBL_ADDR */
typedef struct {
  __REG32 ADDR                :13;
  __REG32                     :19;
} __prv_set_tbl_addr_bits;

/* PRV_SET_TBL_DATA */
typedef struct {
  __REG32 DATA                :20;
  __REG32                     :12;
} __prv_set_tbl_data_bits;

/* PRV_CDC_THRx */
typedef struct {
  __REG32 DETECT              :10;
  __REG32                     : 6;
  __REG32 CORRECT             :10;
  __REG32                     : 6;
} __prv_cdc_thrx_bits;

/* RSZ_PID */
typedef struct {
  __REG32 PREV                : 8;
  __REG32 CID                 : 8;
  __REG32 TID                 : 8;
  __REG32                     : 8;
} __rsz_pid_bits;

/* RSZ_PCR */
typedef struct {
  __REG32 ENABLE              : 1;
  __REG32 BUSY                : 1;
  __REG32 ONESHOT             : 1;
  __REG32                     :29;
} __rsz_pcr_bits;

/* RSZ_CNT */
typedef struct {
  __REG32 HRSZ                :10;
  __REG32 VRSZ                :10;
  __REG32 HSTPH               : 3;
  __REG32 VSTPH               : 3;
  __REG32 YCPOS               : 1;
  __REG32 INPTYP              : 1;
  __REG32 INPSRC              : 1;
  __REG32 CBILIN              : 1;
  __REG32                     : 2;
} __rsz_cnt_bits;

/* RSZ_OUT_SIZE */
typedef struct {
  __REG32 HORZ                :13;
  __REG32                     : 3;
  __REG32 VERT                :12;
  __REG32                     : 4;
} __rsz_out_size_bits;

/* RSZ_IN_START */
typedef struct {
  __REG32 HORZ_ST             :13;
  __REG32                     : 3;
  __REG32 VERT_ST             :13;
  __REG32                     : 3;
} __rsz_in_start_bits;

/* RSZ_IN_SIZE */
typedef struct {
  __REG32 HORZ                :13;
  __REG32                     : 3;
  __REG32 VERT                :13;
  __REG32                     : 3;
} __rsz_in_size_bits;

/* RSZ_SDR_INOFF */
typedef struct {
  __REG32 OFFSET              :16;
  __REG32                     :16;
} __rsz_sdr_inoff_bits;

/* RSZ_HFILT10 */
typedef struct {
  __REG32 COEF0               :10;
  __REG32                     : 6;
  __REG32 COEF1               :10;
  __REG32                     : 6;
} __rsz_filt10_bits;

/* RSZ_HFILT32 */
typedef struct {
  __REG32 COEF2               :10;
  __REG32                     : 6;
  __REG32 COEF3               :10;
  __REG32                     : 6;
} __rsz_filt32_bits;

/* RSZ_HFILT54 */
typedef struct {
  __REG32 COEF4               :10;
  __REG32                     : 6;
  __REG32 COEF5               :10;
  __REG32                     : 6;
} __rsz_filt54_bits;

/* RSZ_HFILT76 */
typedef struct {
  __REG32 COEF6               :10;
  __REG32                     : 6;
  __REG32 COEF7               :10;
  __REG32                     : 6;
} __rsz_filt76_bits;

/* RSZ_HFILT98 */
typedef struct {
  __REG32 COEF8               :10;
  __REG32                     : 6;
  __REG32 COEF9               :10;
  __REG32                     : 6;
} __rsz_filt98_bits;

/* RSZ_HFILT1110 */
typedef struct {
  __REG32 COEF10              :10;
  __REG32                     : 6;
  __REG32 COEF11              :10;
  __REG32                     : 6;
} __rsz_filt1110_bits;

/* RSZ_HFILT1312 */
typedef struct {
  __REG32 COEF12              :10;
  __REG32                     : 6;
  __REG32 COEF13              :10;
  __REG32                     : 6;
} __rsz_filt1312_bits;

/* RSZ_HFILT1514 */
typedef struct {
  __REG32 COEF14              :10;
  __REG32                     : 6;
  __REG32 COEF15              :10;
  __REG32                     : 6;
} __rsz_filt1514_bits;

/* RSZ_HFILT1716 */
typedef struct {
  __REG32 COEF16              :10;
  __REG32                     : 6;
  __REG32 COEF17              :10;
  __REG32                     : 6;
} __rsz_filt1716_bits;

/* RSZ_HFILT1918 */
typedef struct {
  __REG32 COEF18              :10;
  __REG32                     : 6;
  __REG32 COEF19              :10;
  __REG32                     : 6;
} __rsz_filt1918_bits;

/* RSZ_HFILT2120 */
typedef struct {
  __REG32 COEF20              :10;
  __REG32                     : 6;
  __REG32 COEF21              :10;
  __REG32                     : 6;
} __rsz_filt2120_bits;

/* RSZ_HFILT2322 */
typedef struct {
  __REG32 COEF22              :10;
  __REG32                     : 6;
  __REG32 COEF23              :10;
  __REG32                     : 6;
} __rsz_filt2322_bits;

/* RSZ_HFILT2524 */
typedef struct {
  __REG32 COEF24              :10;
  __REG32                     : 6;
  __REG32 COEF25              :10;
  __REG32                     : 6;
} __rsz_filt2524_bits;

/* RSZ_HFILT2726 */
typedef struct {
  __REG32 COEF26              :10;
  __REG32                     : 6;
  __REG32 COEF27              :10;
  __REG32                     : 6;
} __rsz_filt2726_bits;

/* RSZ_HFILT2928 */
typedef struct {
  __REG32 COEF28              :10;
  __REG32                     : 6;
  __REG32 COEF29              :10;
  __REG32                     : 6;
} __rsz_filt2928_bits;

/* RSZ_HFILT3130 */
typedef struct {
  __REG32 COEF30              :10;
  __REG32                     : 6;
  __REG32 COEF31              :10;
  __REG32                     : 6;
} __rsz_filt3130_bits;

/* RSZ_YENH */
typedef struct {
  __REG32 CORE                : 8;
  __REG32 SLOP                : 4;
  __REG32 GAIN                : 4;
  __REG32 ALGO                : 2;
  __REG32                     :14;
} __rsz_yenh_bits;

/* SBL_PID */
typedef struct {
  __REG32 PREV                : 8;
  __REG32 CID                 : 8;
  __REG32 TID                 : 8;
  __REG32                     : 8;
} __sbl_pid_bits;

/* SBL_PCR */
typedef struct {
  __REG32                           :16;
  __REG32 H3A_AEAWB_WBL_OVF         : 1;
  __REG32 H3A_AF_WBL_OVF            : 1;
  __REG32 RSZ4_WBL_OVF              : 1;
  __REG32 RSZ3_WBL_OVF              : 1;
  __REG32 RSZ2_WBL_OVF              : 1;
  __REG32 RSZ1_WBL_OVF              : 1;
  __REG32 PRV_WBL_OVF               : 1;
  __REG32 CCDC_WBL_OVF              : 1;
  __REG32 CCDCPRV_2_RSZ_OVF         : 1;
  __REG32 CSI2A_WBL_OVF             : 1;
  __REG32 CSI1_CCP2B_CSI2C_WBL_OVF  : 1;
  __REG32                           : 5;
} __sbl_pcr_bits;

/* SBL_GLB_REG_0 */
/* SBL_GLB_REG_1 */
/* SBL_GLB_REG_2 */
/* SBL_GLB_REG_3 */
/* SBL_GLB_REG_4 */
/* SBL_GLB_REG_5 */
/* SBL_GLB_REG_6 */
/* SBL_GLB_REG_7 */
typedef struct {
  __REG32 VALID               : 1;
  __REG32 DIRECTION           : 1;
  __REG32 SRC_DST_M           : 5;
  __REG32 SRC_DST_ID          : 2;
  __REG32                     :23;
} __sbl_glb_reg_x_bits;

/* SBL_CCDC_WR_0 */
/* SBL_CCDC_WR_1 */
/* SBL_CCDC_WR_2 */
/* SBL_CCDC_WR_3 */
typedef struct {
  __REG32 ADDR                :20;
  __REG32 DATA_SENT           : 1;
  __REG32 DATA_READY          : 1;
  __REG32 BYTE_CNT            : 8;
  __REG32                     : 2;
} __sbl_ccdc_wr_x_bits;

/* SBL_CCDC_FP_RD_0 */
/* SBL_CCDC_FP_RD_1 */
typedef struct {
  __REG32 ADDR                :20;
  __REG32 BYTE_CNT            : 8;
  __REG32 DATA_AVL            : 1;
  __REG32 DATA_WAIT           : 1;
  __REG32 VALID               : 1;
  __REG32                     : 1;
} __sbl_ccdc_fp_rd_x_bits;

/* SBL_PRV_RD_0 */
/* SBL_PRV_RD_1 */
/* SBL_PRV_RD_2 */
/* SBL_PRV_RD_3 */
typedef struct {
  __REG32 ADDR                :20;
  __REG32 BYTE_CNT            : 8;
  __REG32 DATA_AVL            : 1;
  __REG32 DATA_WAIT           : 1;
  __REG32 VALID               : 1;
  __REG32                     : 1;
} __sbl_prv_rd_x_bits;

/* SBL_PRV_WR_0 */
/* SBL_PRV_WR_1 */
/* SBL_PRV_WR_2 */
/* SBL_PRV_WR_3 */
typedef struct {
  __REG32 ADDR                :20;
  __REG32 DATA_SENT           : 1;
  __REG32 DATA_READY          : 1;
  __REG32 BYTE_CNT            : 8;
  __REG32                     : 2;
} __sbl_prv_wr_x_bits;

/* SBL_PRV_DK_RD_0 */
/* SBL_PRV_DK_RD_1 */
/* SBL_PRV_DK_RD_2 */
/* SBL_PRV_DK_RD_3 */
typedef struct {
  __REG32 ADDR                :20;
  __REG32 BYTE_CNT            : 8;
  __REG32 DATA_AVL            : 1;
  __REG32 DATA_WAIT           : 1;
  __REG32 VALID               : 1;
  __REG32                     : 1;
} __sbl_prv_dk_rd_x_bits;

/* SBL_RSZ_RD_0 */
/* SBL_RSZ_RD_1 */
/* SBL_RSZ_RD_2 */
/* SBL_RSZ_RD_3 */
typedef struct {
  __REG32 ADDR                :20;
  __REG32 BYTE_CNT            : 8;
  __REG32 DATA_AVL            : 1;
  __REG32 DATA_WAIT           : 1;
  __REG32 VALID               : 1;
  __REG32                     : 1;
} __sbl_rsz_rd_x_bits;

/* SBL_RSZ1_WR_0 */
/* SBL_RSZ1_WR_1 */
/* SBL_RSZ1_WR_2 */
/* SBL_RSZ1_WR_3 */
/* SBL_RSZ2_WR_0 */
/* SBL_RSZ2_WR_1 */
/* SBL_RSZ2_WR_2 */
/* SBL_RSZ2_WR_3 */
/* SBL_RSZ3_WR_0 */
/* SBL_RSZ3_WR_1 */
/* SBL_RSZ3_WR_2 */
/* SBL_RSZ3_WR_3 */
/* SBL_RSZ4_WR_0 */
/* SBL_RSZ4_WR_1 */
/* SBL_RSZ4_WR_2 */
/* SBL_RSZ4_WR_3 */
typedef struct {
  __REG32 ADDR                :20;
  __REG32 DATA_SENT           : 1;
  __REG32 DATA_READY          : 1;
  __REG32 BYTE_CNT            : 8;
  __REG32                     : 2;
} __sbl_rszx_wr_y_bits;

/* SBL_HIST_RD_0 */
/* SBL_HIST_RD_1 */
typedef struct {
  __REG32 ADDR                :20;
  __REG32 BYTE_CNT            : 8;
  __REG32 DATA_AVL            : 1;
  __REG32 DATA_WAIT           : 1;
  __REG32 VALID               : 1;
  __REG32                     : 1;
} __sbl_hist_rd_x_bits;

/* SBL_H3A_AF_WR_0 */
/* SBL_H3A_AF_WR_1 */
typedef struct {
  __REG32 ADDR                :20;
  __REG32 DATA_SENT           : 1;
  __REG32 DATA_READY          : 1;
  __REG32 BYTE_CNT            : 8;
  __REG32                     : 2;
} __sbl_h3a_af_wr_x_bits;

/* SBL_H3A_AEAWB_WR_0 */
/* SBL_H3A_AEAWB_WR_1 */
typedef struct {
  __REG32 ADDR                :20;
  __REG32 DATA_SENT           : 1;
  __REG32 DATA_READY          : 1;
  __REG32 BYTE_CNT            : 8;
  __REG32                     : 2;
} __sbl_h3a_aeawb_wr_x_bits;

/* SBL_CSIA_WR_0 */
/* SBL_CSIA_WR_1 */
/* SBL_CSIA_WR_2 */
/* SBL_CSIA_WR_3 */
/* SBL_CSIB_WR_0 */
/* SBL_CSIB_WR_1 */
/* SBL_CSIB_WR_2 */
/* SBL_CSIB_WR_3 */
typedef struct {
  __REG32 ADDR                :20;
  __REG32 DATA_SENT           : 1;
  __REG32 DATA_READY          : 1;
  __REG32 BYTE_CNT            : 8;
  __REG32                     : 2;
} __sbl_csi_wr_x_bits;

/* SBL_SDR_REQ_EXP */
typedef struct {
  __REG32 HIST_EXP            :10;
  __REG32 RSZ_EXP             :10;
  __REG32 PRV_EXP             :10;
  __REG32                     : 2;
} __sbl_sdr_req_exp_bits;

/* CSI2_REVISION */
typedef struct {
  __REG32 REV                 : 8;
  __REG32                     :24;
} __csi2_revision_bits;

/* CSI2_SYSCONFIG */
typedef struct {
  __REG32 AUTO_IDLE           : 1;
  __REG32 SOFT_RESET          : 1;
  __REG32                     :10;
  __REG32 MSTANDBY_MODE       : 2;
  __REG32                     :18;
} __csi2_sysconfig_bits;

/* CSI2_SYSSTATUS */
typedef struct {
  __REG32 RESET_DONE          : 1;
  __REG32                     :31;
} __csi2_sysstatus_bits;

/* CSI2_IRQSTATUS */
typedef struct {
  __REG32 CONTEXT0              : 1;
  __REG32 CONTEXT1              : 1;
  __REG32 CONTEXT2              : 1;
  __REG32 CONTEXT3              : 1;
  __REG32 CONTEXT4              : 1;
  __REG32 CONTEXT5              : 1;
  __REG32 CONTEXT6              : 1;
  __REG32 CONTEXT7              : 1;
  __REG32 FIFO_OVF_IRQ          : 1;
  __REG32 COMPLEXIO1_ERR_IRQ    : 1;
  __REG32                       : 1;
  __REG32 ECC_NO_CORRECTION_IRQ : 1;
  __REG32 ECC_CORRECTION_IRQ    : 1;
  __REG32 SHORT_PACKET_IRQ      : 1;
  __REG32 OCP_ERR_IRQ           : 1;
  __REG32                       :17;
} __csi2_irqstatus_bits;

/* CSI2_CTRL */
typedef struct {
  __REG32 IF_EN                 : 1;
  __REG32                       : 1;
  __REG32 ECC_EN                : 1;
  __REG32 FRAME                 : 1;
  __REG32 ENDIANNESS            : 1;
  __REG32                       : 2;
  __REG32 DBG_EN                : 1;
  __REG32 VP_OUT_CTRL           : 2;
  __REG32                       : 1;
  __REG32 VP_ONLY_EN            : 1;
  __REG32                       : 3;
  __REG32 VP_CLK_EN             : 1;
  __REG32                       :16;
} __csi2_ctrl_bits;

/* CSI2_GNQ */
typedef struct {
  __REG32 NBCONTEXTS            : 2;
  __REG32 FIFODEPTH             : 4;
  __REG32                       :26;
} __csi2_gnq_bits;

/* CSI2_COMPLEXIO_CFG1 */
typedef struct {
  __REG32 CLOCK_POSITION        : 3;
  __REG32 CLOCK_POL             : 1;
  __REG32 DATA1_POSITION        : 3;
  __REG32 DATA1_POL             : 1;
  __REG32 DATA2_POSITION        : 3;
  __REG32 DATA2_POL             : 1;
  __REG32                       :12;
  __REG32 PWR_AUTO              : 1;
  __REG32 PWR_STATUS            : 2;
  __REG32 PWR_CMD               : 2;
  __REG32 RESET_DONE            : 1;
  __REG32 RESET_CTRL            : 1;
  __REG32                       : 1;
} __csi2_complexio_cfg1_bits;

/* CSI2_COMPLEXIO1_IRQSTATUS */
typedef struct {
  __REG32 ERRSOTHS1             : 1;
  __REG32 ERRSOTHS2             : 1;
  __REG32 ERRSOTHS3             : 1;
  __REG32 ERRSOTHS4             : 1;
  __REG32 ERRSOTHS5             : 1;
  __REG32 ERRSOTSYNCHS1         : 1;
  __REG32 ERRSOTSYNCHS2         : 1;
  __REG32 ERRSOTSYNCHS3         : 1;
  __REG32 ERRSOTSYNCHS4         : 1;
  __REG32 ERRSOTSYNCHS5         : 1;
  __REG32 ERRESC1               : 1;
  __REG32 ERRESC2               : 1;
  __REG32 ERRESC3               : 1;
  __REG32 ERRESC4               : 1;
  __REG32 ERRESC5               : 1;
  __REG32 ERRCONTROL1           : 1;
  __REG32 ERRCONTROL2           : 1;
  __REG32 ERRCONTROL3           : 1;
  __REG32 ERRCONTROL4           : 1;
  __REG32 ERRCONTROL5           : 1;
  __REG32 STATEULPM1            : 1;
  __REG32 STATEULPM2            : 1;
  __REG32 STATEULPM3            : 1;
  __REG32 STATEULPM4            : 1;
  __REG32 STATEULPM5            : 1;
  __REG32 STATEALLULPMENTER     : 1;
  __REG32 STATEALLULPMEXIT      : 1;
  __REG32                       : 5;
} __csi2_complexio1_irqstatus_bits;

/* CSI2_SHORT_PACKET */
typedef struct {
  __REG32 SHORT_PACKET          :24;
  __REG32                       : 8;
} __csi2_short_packet_bits;

/* CSI2_TIMING */
typedef struct {
  __REG32 STOP_STATE_COUNTER_IO1  :13;
  __REG32 STOP_STATE_X4_IO1       : 1;
  __REG32 STOP_STATE_X16_IO1      : 1;
  __REG32 FORCE_RX_MODE_IO1       : 1;
  __REG32                         :16;
} __csi2_timing_bits;

/* CSI2_CTx_CTRL1 */
typedef struct {
  __REG32 CTX_EN                : 1;
  __REG32 LINE_MODULO           : 1;
  __REG32 VP_FORCE              : 1;
  __REG32 PING_PONG             : 1;
  __REG32 COUNT_UNLOCK          : 1;
  __REG32 CS_EN                 : 1;
  __REG32 EOL_EN                : 1;
  __REG32 EOF_EN                : 1;
  __REG32 COUNT                 : 8;
  __REG32 FEC_NUMBER            : 8;
  __REG32 TRANSCODE             : 4;
  __REG32                       : 2;
  __REG32 GENERIC               : 1;
  __REG32 BYTESWAP              : 1;
} __csi2_ctx_ctrl1_bits;

/* CSI2_CTx_CTRL2 */
typedef struct {
  __REG32 FORMAT                :10;
  __REG32 DPCM_PRED             : 1;
  __REG32 VIRTUAL_ID            : 2;
  __REG32 USER_DEF_MAPPING      : 2;
  __REG32                       : 1;
  __REG32 FRAME                 :16;
} __csi2_ctx_ctrl2_bits;

/* CSI2_CTx_DAT_OFST */
typedef struct {
  __REG32                       : 5;
  __REG32 OFST                  :12;
  __REG32                       :15;
} __csi2_ctx_dat_ofst_bits;

/* CSI2_CTx_IRQENABLE */
typedef struct {
  __REG32 FS_IRQ                : 1;
  __REG32 FE_IRQ                : 1;
  __REG32 LS_IRQ                : 1;
  __REG32 LE_IRQ                : 1;
  __REG32                       : 1;
  __REG32 CS_IRQ                : 1;
  __REG32 FRAME_NUMBER_IRQ      : 1;
  __REG32 LINE_NUMBER_IRQ       : 1;
  __REG32 ECC_CORRECTION_IRQ    : 1;
  __REG32                       :23;
} __csi2_ctx_irqenable_bits;

/* CSI2_CTx_CTRL3 */
typedef struct {
  __REG32 LINE_NUMBER           :16;
  __REG32 ALPHA                 :14;
  __REG32                       : 2;
} __csi2_ctx_ctrl3_bits;

/* CSI2_CTx_TRANSCODEH */
typedef struct {
  __REG32 HSKIP                 :13;
  __REG32                       : 3;
  __REG32 HCOUNT                :13;
  __REG32                       : 3;
} __csi2_ctx_transcodeh_bits;

/* CSI2_CTx_TRANSCODEV */
typedef struct {
  __REG32 VSKIP                 :13;
  __REG32                       : 3;
  __REG32 VCOUNT                :13;
  __REG32                       : 3;
} __csi2_ctx_transcodev_bits;

/* CSIPHY_REG0 */
typedef struct {
  __REG32 THS_SETTLE            : 8;
  __REG32 THS_TERM              : 8;
  __REG32                       : 8;
  __REG32 HSCLOCKCONFIG         : 1;
  __REG32                       : 7;
} __csiphy_reg0_bits;

/* CSIPHY_REG1 */
typedef struct {
  __REG32 TCLK_SETTLE                 : 8;
  __REG32 TCLK_MISS                   : 2;
  __REG32 DPHY_HS_SYNC_PATTERN        : 8;
  __REG32 TCLK_TERM                   : 7;
  __REG32 CLOCK_MISS_DETECTOR_STATUS  : 1;
  __REG32                             : 2;
  __REG32 RESETDONERXBYTECLK          : 1;
  __REG32 RESETDONECSI2_96M_FCLK      : 1;
  __REG32                             : 2;
} __csiphy_reg1_bits;

/* CSIPHY_REG2 */
typedef struct {
  __REG32 CCP2_SYNC_PATTERN       :24;
  __REG32 TRIGGER_CMD_RXTRIGESC3  : 2;
  __REG32 TRIGGER_CMD_RXTRIGESC2  : 2;
  __REG32 TRIGGER_CMD_RXTRIGESC1  : 2;
  __REG32 TRIGGER_CMD_RXTRIGESC0  : 2;
} __csiphy_reg2_bits;

/* DSS_REVISIONNUMBER */
typedef struct {
  __REG32 REV                 : 8;
  __REG32                     :24;
} __dss_revisionnumber_bits;

/* DSS_SYSCONFIG */
typedef struct {
  __REG32 AUTOIDLE            : 1;
  __REG32 SOFTRESET           : 1;
  __REG32                     :30;
} __dss_sysconfig_bits;

/* DSS_SYSSTATUS */
typedef struct {
  __REG32 RESETDONE           : 1;
  __REG32                     :31;
} __dss_sysstatus_bits;

/* DSS_IRQSTATUS */
typedef struct {
  __REG32 DISPC_IRQ           : 1;
  __REG32 DSI_IRQ             : 1;
  __REG32                     :30;
} __dss_irqstatus_bits;

/* DSS_CONTROL */
typedef struct {
  __REG32 DISPC_CLK_SWITCH      : 1;
  __REG32 DSI_CLK_SWITCH        : 1;
  __REG32 VENC_CLOCK_MODE       : 1;
  __REG32 VENC_CLOCK_4X_ENABLE  : 1;
  __REG32 DAC_DEMEN             : 1;
  __REG32 DAC_POWERDN_BGZ       : 1;
  __REG32 VENC_OUT_SEL          : 1;
  __REG32                       :25;
} __dss_control_bits;

/* DSS_CLK_STATUS */
typedef struct {
  __REG32 DSS_DISPC_CLK1_STATUS : 1;
  __REG32 DSI_PLL_CLK1_STATUS   : 1;
  __REG32                       : 5;
  __REG32 DSS_DSI_CLK1_STATUS   : 1;
  __REG32 DSI_PLL_CLK2_STATUS   : 1;
  __REG32                       :23;
} __dss_clk_status_bits;

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
 	__REG32               						:12;
 	__REG32 PREMULTIPLYALPHA  				: 1;
 	__REG32 													: 3;
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
 	__REG32 GU   											: 9;
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

/* DSI_IRQENABLE */
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

/* DSI_PHY_REGISTER0 */
typedef struct {
 	__REG32 REG_THSEXIT   						: 8;
 	__REG32 REG_THSTRAIL							: 8;
 	__REG32 REG_THSPRPR_THSZERO 			: 8;
 	__REG32 REG_THSPREPARE  					: 8;
} __dsi_phy_register0_bits;

/* DSI_PHY_REGISTER1 */
typedef struct {
 	__REG32 REG_TCLKZERO							: 8;
 	__REG32 REG_TCLKTRAIL							: 8;
 	__REG32 REG_TLPXBY2								: 5;
 	__REG32 													: 3;
  __REG32 REG_TTAGET								: 3;
  __REG32 REG_TTASURE								: 2;
  __REG32 REG_TTAGO 								: 3;
} __dsi_phy_register1_bits;

/* DSI_PHY_REGISTER2 */
typedef struct {
 	__REG32 REG_TCLKPREPARE						: 8;
 	__REG32 													: 3;
 	__REG32 REGULPMTX     						: 5;
 	__REG32 OVRRDULPMTX    						: 1;
 	__REG32                						: 7;
 	__REG32 HSSYNCPATTERN 						: 8;
} __dsi_phy_register2_bits;

/* DSI_PHY_REGISTER3 */
typedef struct {
 	__REG32 REG_TXTRIGGERESC0					: 8;
 	__REG32 REG_TXTRIGGERESC1					: 8;
 	__REG32 REG_TXTRIGGERESC2					: 8;
 	__REG32 REG_TXTRIGGERESC3					: 8;
} __dsi_phy_register3_bits;

/* DSI_PHY_REGISTER4 */
typedef struct {
 	__REG32 REG_RXTRIGGERESC0					: 8;
 	__REG32 REG_RXTRIGGERESC1					: 8;
 	__REG32 REG_RXTRIGGERESC2					: 8;
 	__REG32 REG_RXTRIGGERESC3					: 8;
} __dsi_phy_register4_bits;

/* DSI_PHY_REGISTER5 */
typedef struct {
 	__REG32 													:24;
 	__REG32 RESETDONETXCLKESC0 				: 1;
 	__REG32 RESETDONETXCLKESC1				: 1;
 	__REG32 RESETDONETXCLKESC2				: 1;
 	__REG32                   				: 2;
 	__REG32 RESETDONEPWRCLK						: 1;
 	__REG32 RESETDONESCPCLK						: 1;
 	__REG32 RESETDONETXBYTECLK				: 1;
} __dsi_phy_register5_bits;

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
 	__REG32                						: 4;
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

/* OCP_HWINFO */
typedef struct {
 	__REG32 SYS_BUS_WIDTH							: 2;
 	__REG32 MEM_BUS_WIDTH 						: 1;
 	__REG32 													:29;
} __ocp_hwinfo_bits;

/* OCP_SYSCONFIG */
typedef struct {
 	__REG32              							: 2;
 	__REG32 IDLE_MODE   							: 2;
 	__REG32 STANDBY_MODE  						: 2;
 	__REG32 													:26;
} __ocp_sysconfig_bits;

/* OCP_IRQSTATUS_RAW_0 */
typedef struct {
 	__REG32 INIT_MINTERRUPT_RAW  			: 1;
	__REG32 													:31;
} __ocp_irqstatus_raw_0_bits;

/* OCP_IRQSTATUS_RAW_1 */
typedef struct {
 	__REG32 TARGET_SINTERRUPT_RAW			: 1;
	__REG32 													:31;
} __ocp_irqstatus_raw_1_bits;

/* OCP_IRQSTATUS_RAW_2 */
typedef struct {
 	__REG32 THALIA_IRQ_RAW      			: 1;
	__REG32 													:31;
} __ocp_irqstatus_raw_2_bits;

/* OCP_IRQSTATUS_0 */
typedef struct {
 	__REG32 INIT_MINTERRUPT_STATUS		: 1;
	__REG32 													:31;
} __ocp_irqstatus_0_bits;

/* OCP_IRQSTATUS_1 */
typedef struct {
 	__REG32 TARGET_SINTERRUPT_STATUS  : 1;
	__REG32 													:31;
} __ocp_irqstatus_1_bits;

/* OCP_IRQSTATUS_2 */
typedef struct {
 	__REG32 THALIA_IRQ_STATUS         : 1;
	__REG32 													:31;
} __ocp_irqstatus_2_bits;

/* OCP_IRQENABLE_SET_0 */
typedef struct {
 	__REG32 INIT_MINTERRUPT_ENABLE    : 1;
	__REG32 													:31;
} __ocp_irqenable_set_0_bits;

/* OCP_IRQENABLE_SET_1 */
typedef struct {
 	__REG32 TARGET_SINTERRUPT_ENABLE  : 1;
	__REG32 													:31;
} __ocp_irqenable_set_1_bits;

/* OCP_IRQENABLE_SET_2 */
typedef struct {
 	__REG32 THALIA_IRQ_ENABLE         : 1;
	__REG32 													:31;
} __ocp_irqenable_set_2_bits;

/* OCP_IRQENABLE_CLR_0 */
typedef struct {
 	__REG32 INIT_MINTERRUPT_DISABLE   : 1;
	__REG32 													:31;
} __ocp_irqenable_clr_0_bits;

/* OCP_IRQENABLE_CLR_1 */
typedef struct {
 	__REG32 TARGET_SINTERRUPT_DISABLE : 1;
	__REG32 													:31;
} __ocp_irqenable_clr_1_bits;

/* OCP_IRQENABLE_CLR_2 */
typedef struct {
 	__REG32 THALIA_IRQ_DISABLE        : 1;
	__REG32 													:31;
} __ocp_irqenable_clr_2_bits;

/* OCP_PAGE_CONFIG */
typedef struct {
 	__REG32 MEM_PAGE_SIZE             : 2;
 	__REG32 MEM_PAGE_CHECK_EN         : 1;
 	__REG32 OCP_PAGE_SIZE             : 2;
	__REG32 													:27;
} __ocp_page_config_bits;

/* OCP_INTERRUPT_EVENT */
typedef struct {
 	__REG32 INIT_RESP_UNEXPECTED        : 1;
 	__REG32 INIT_RESP_UNUSED_TAG        : 1;
 	__REG32 INIT_RESP_ERROR             : 1;
 	__REG32 INIT_PAGE_CROSS_ERROR       : 1;
 	__REG32 INIT_READ_TAG_FIFO_OVERRUN  : 1;
 	__REG32 INIT_MEM_REQ_FIFO_OVERRUN   : 1;
 	__REG32                             : 2;
 	__REG32 TARGET_RESP_FIFO_FULL       : 1;
 	__REG32 TARGET_CMD_FIFO_FULL        : 1;
 	__REG32 TARGET_INVALID_OCP_CMD      : 1;
	__REG32 													  :21;
} __ocp_interrupt_event_bits;

/* OCP_DEBUG_CONFIG */
typedef struct {
 	__REG32 FORCE_TARGET_IDLE         : 2;
 	__REG32 FORCE_INIT_IDLE           : 2;
 	__REG32 FORCE_PASS_DATA           : 1;
 	__REG32 SELECT_INIT_IDLE          : 1;
 	__REG32                           :25;
 	__REG32 THALIA_INT_BYPASS         : 1;
} __ocp_debug_config_bits;

/* OCP_DEBUG_STATUS */
typedef struct {
 	__REG32 TARGET_MCONNECT           : 2;
 	__REG32 TARGET_SCONNECT           : 1;
 	__REG32 TARGET_SIDLEREQ           : 1;
 	__REG32 TARGET_SDISCACK           : 2;
 	__REG32 TARGET_SIDLEAC            : 2;
 	__REG32 INIT_MCONNECT             : 2;
 	__REG32 INIT_SCONNECT0            : 1;
 	__REG32 INIT_SCONNECT1            : 1;
 	__REG32 INIT_SCONNECT2            : 1;
 	__REG32 INIT_MDISCACK             : 1;
 	__REG32 INIT_MDISCREQ             : 2;
 	__REG32 INIT_MWAIT                : 1;
 	__REG32 INIT_MSTANDBY             : 1;
 	__REG32 TARGET_CMD_OUT            : 3;
 	__REG32 WHICH_TARGET_REGISTER     : 5;
 	__REG32 RESP_ERROR                : 1;
 	__REG32 CMD_FIFO_FULL             : 1;
 	__REG32 RESP_FIFO_FULL            : 1;
 	__REG32 TARGET_IDLE               : 1;
 	__REG32 CMD_RESP_DEBUG_STATE      : 1;
 	__REG32 CMD_DEBUG_STATE           : 1;
} __ocp_debug_status_bits;

/* L3_IA_COMPONENT */
typedef struct {
 	__REG32 REV       							  :16;
 	__REG32 CODE       								:16;
} __l3_ia_component_bits;

/* L3_IA_CORE_L */
typedef struct {
 	__REG32 REV_CODE   								:16;
 	__REG32 CORE_CODE  								:16;
} __l3_ia_core_l_bits;

/* L3_IA_CORE_H */
typedef struct {
 	__REG32 VENDOR_CODE								:16;
 	__REG32           								:16;
} __l3_ia_core_h_bits;

/* L3_IA_AGENT_CONTROL */
typedef struct {
 	__REG32 CORE_RESET									: 1;
 	__REG32        	        						: 3;
 	__REG32 REJECT											: 1;
 	__REG32        	        						: 3;
 	__REG32 RESP_TIMEOUT								: 3;
 	__REG32        	        						: 5;
 	__REG32 BURST_TIMEOUT								: 3;
 	__REG32        	        						: 5;
 	__REG32 MERROR_REP      						: 1;
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
 	__REG32        	        						: 7;
 	__REG32 MERROR                  		: 1;
 	__REG32                         		: 3;
 	__REG32 INBAND_ERROR_PRIMARY    		: 1;
 	__REG32 INBAND_ERROR_SECONDARY    	: 1;
 	__REG32        	        						: 2;
} __l3_ia_agent_status_bits;

/* L3_IA_ERROR_LOG_L */
typedef struct {
 	__REG32 CMD													: 3;
 	__REG32        	        						: 5;
 	__REG32 INITID											: 8;
 	__REG32        	        						: 8;
 	__REG32 CODE												: 4;
 	__REG32        	        						: 2;
 	__REG32 SECONDARY										: 1;
 	__REG32 MULTI												: 1;
} __l3_ia_error_log_l_bits;

/* L3_IA_ERROR_LOG_H */
typedef struct {
 	__REG32 REQ_INFO  									:16;
 	__REG32        	        						:16;
} __l3_ia_error_log_h_bits;

/* L3_TA_COMPONENT */
typedef struct {
 	__REG32 REV       									:16;
 	__REG32 CODE  	        						:16;
} __l3_ta_component_bits;

/* L3_TA_CORE_L */
typedef struct {
 	__REG32 REV_CODE   									:16;
 	__REG32 CORE_CODE        						:16;
} __l3_ta_core_l_bits;

/* L3_TA_CORE_H */
typedef struct {
 	__REG32 VEND_CODE  									:16;
 	__REG32       	        						:16;
} __l3_ta_core_h_bits;

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

/* L3_TA_ERROR_LOG_L */
typedef struct {
 	__REG32 CMD													: 3;
 	__REG32        	        						: 5;
 	__REG32 INITID											: 8;
 	__REG32 														: 8;
 	__REG32 CODE												: 4;
 	__REG32        	        						: 3;
 	__REG32 MULTI												: 1;
} __l3_ta_error_log_l_bits;

/* L3_TA_ERROR_LOG_H */
typedef struct {
 	__REG32 REQ_INFO										:10;
 	__REG32        	        						:22;
} __l3_ta_error_log_h_bits;

/* L3_RT_COMPONENT */
typedef struct {
 	__REG32 REV     										:16;
 	__REG32 CODE  	        						:16;
} __l3_rt_component_bits;

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
 	__REG32 IVA2_DMA        						: 1;
 	__REG32 SDMA												: 1;
 	__REG32 USB_HS_OTG									: 1;
 	__REG32 SAD2D     									: 1;
 	__REG32        	        						: 2;
 	__REG32 DISPSS  										: 1;
 	__REG32 USB_HS_HOST									: 1;
 	__REG32 IVA2_MMU  									: 1;
 	__REG32 CAM       									: 1;
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

/* L4_IA_COMPONENT_L */
typedef struct {
 	__REG32 REV               					:16;
 	__REG32 CODE                				:16;
} __l4_ia_component_l_bits;

/* L4_IA_COMPONENT_H */
typedef struct {
 	__REG32 VENDOR_CODE        					:16;
 	__REG32                     				:16;
} __l4_ia_component_h_bits;

/* L4_IA_CORE_L */
typedef struct {
 	__REG32 CORE_REV          					:16;
 	__REG32 CORE_CODE           				:16;
} __l4_ia_core_l_bits;

/* L4_IA_CORE_H */
typedef struct {
 	__REG32 VENDOR_CODE        					:16;
 	__REG32                     				:16;
} __l4_ia_core_h_bits;

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

/* L4_TA_COMPONENT_L */
typedef struct {
 	__REG32 REV               					:16;
 	__REG32 CODE                				:16;
} __l4_ta_component_l_bits;

/* L4_TA_CORE_L */
typedef struct {
 	__REG32 CORE_REV           					:16;
 	__REG32 CORE_CODE           				:16;
} __l4_ta_core_l_bits;

/* L4_TA_CORE_H */
typedef struct {
 	__REG32 VENDOR_CODE       					:16;
 	__REG32                      				:16;
} __l4_ta_core_h_bits;

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

/* L4_LA_COMPONENT_L */
typedef struct {
 	__REG32 REV              						:16;
 	__REG32 CODE             						:16;
} __l4_la_component_l_bits;

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

/* L4_AP_COMPONENT_L */
typedef struct {
 	__REG32 REV 												:16;
 	__REG32 CODE  	        						:16;
} __l4_ap_component_l_bits;

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

/* GPMC_REVISION */
typedef struct {
 	__REG32 REV     				: 8;
 	__REG32 								:24;
} __gpmc_revision_bits;

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

/* SMS_REVISION */
typedef struct {
 	__REG32 REV     									: 8;
 	__REG32 													:24;
} __sms_revision_bits;

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
 	__REG32             							: 2;
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

/* SDRC_REVISION */
typedef struct {
 	__REG32 REV       								: 8;
 	__REG32                  					:24;
} __sdrc_revision_bits;

/* SDRC_SYSCONFIG */
typedef struct {
 	__REG32            								: 1;
 	__REG32 SOFTRESET  								: 1;
 	__REG32           								: 1;
 	__REG32 IDLEMODE  								: 2;
 	__REG32           								: 3;
 	__REG32 NOMEMORYMRS								: 1;
 	__REG32                  					:23;
} __sdrc_sysconfig_bits;

/* SDRC_SYSSTATUS */
typedef struct {
 	__REG32 RESETDONE  								: 1;
 	__REG32                  					:31;
} __sdrc_sysstatus_bits;

/* SDRC_CS_CFG */
typedef struct {
 	__REG32 CS1STARTHIGH							: 4;
 	__REG32           								: 4;
 	__REG32 CS1STARTLOW 							: 2;
 	__REG32                  					:22;
} __sdrc_cs_cfg_bits;

/* SDRC_SHARING */
typedef struct {
 	__REG32             							: 8;
 	__REG32 SDRCTRISTATE							: 1;
 	__REG32 CS0MUXCFG   							: 3;
 	__REG32 CS1MUXCFG   							: 3;
 	__REG32           								:15;
 	__REG32 LOCK         							: 1;
 	__REG32                  					: 1;
} __sdrc_sharing_bits;

/* SDRC_ERR_TYPE */
typedef struct {
 	__REG32 ERRORVALID  							: 1;
 	__REG32 ERRORDPD     							: 1;
 	__REG32 ERRORADD    							: 2;
 	__REG32 ERRORMCMD   							: 3;
 	__REG32              							: 1;
 	__REG32 ERRORCONNID 							: 4;
 	__REG32           								:20;
} __sdrc_err_type_bits;

/* SDRC_DLLA_CTRL */
typedef struct {
 	__REG32             							: 2;
 	__REG32 LOCKDLL      							: 1;
 	__REG32 ENADLL      							: 1;
 	__REG32 DLLIDLE      							: 1;
 	__REG32 DLLMODEONIDLEREQ  				: 2;
 	__REG32              							: 9;
 	__REG32 MODEFIXEDDELAYINITLAT 		: 8;
 	__REG32 FIXEDDELAY             		: 8;
} __sdrc_dlla_ctrl_bits;

/* SDRC_DLLA_STATUS */
typedef struct {
 	__REG32             							: 2;
 	__REG32 LOCKSTATUS   							: 1;
 	__REG32              							:29;
} __sdrc_dlla_status_bits;

/* SDRC_POWER_REG */
typedef struct {
 	__REG32 PAGEPOLICY   							: 1;
 	__REG32              							: 1;
 	__REG32 PWDENA      							: 1;
 	__REG32 EXTCLKDIS    							: 1;
 	__REG32 CLKCTRL     							: 2;
 	__REG32 SRFRONIDLEREQ							: 1;
 	__REG32 SRFRONRESET  							: 1;
 	__REG32 AUTOCOUNT   							:16;
 	__REG32             							: 2;
 	__REG32 WAKEUPPROC   							: 1;
 	__REG32              							: 5;
} __sdrc_power_reg_bits;

typedef union{
    /* SDRC_MCFG_x */
    struct {
      __REG32 RAMTYPE      							: 2;
      __REG32 DDRTYPE     							: 1;
      __REG32 DEEPPD      							: 1;
      __REG32 B32NOT16    							: 1;
      __REG32              							: 1;
      __REG32 BANKALLOCATION						: 2;
      __REG32 RAMSIZE     							:10;
      __REG32             							: 1;
      __REG32 ADDRMUXLEGACY							: 1;
      __REG32 CASWIDTH    					    : 3;
      __REG32             					    : 1;
      __REG32 RASWIDTH    					    : 3;
      __REG32                    				: 3;
      __REG32 LOCKSTATUS    						: 1;
      __REG32              							: 1;
  };

  /* SDRC_MCFG_FIXED_x */
  struct {
    __REG32 RAMTYPE      							: 2;
    __REG32 DDRTYPE     							: 1;
    __REG32 DEEPPD      							: 1;
    __REG32 B32NOT16    							: 1;
    __REG32              							: 1;
    __REG32 BANKALLOCATION						: 2;
    __REG32 RAMSIZE     							:10;
    __REG32             							: 1;
    __REG32 ADDRMUXLEGACY							: 1;
    __REG32 ADDRMUX    						    : 5;
    __REG32             					    : 5;
    __REG32 LOCKSTATUS    						: 1;
    __REG32              							: 1;
  } FIXED;
} __sdrc_mcfg_bits;

/* SDRC_MR_p */
typedef struct {
 	__REG32 BL          							: 3;
 	__REG32 SIL          							: 1;
 	__REG32 CASL        							: 3;
 	__REG32 ZERO_0      							: 2;
 	__REG32 WBST        							: 1;
 	__REG32 ZERO_1        						: 2;
 	__REG32              							:20;
} __sdrc_mr_bits;

/* SDRC_EMR2_p */
typedef struct {
 	__REG32 PASR        							: 3;
 	__REG32 TCSR         							: 2;
 	__REG32 DS          							: 3;
 	__REG32 ZERO        							: 4;
 	__REG32              							:20;
} __sdrc_emr2_bits;

/* SDRC_ACTIM_CTRLA_p */
typedef struct {
 	__REG32 TDAL        							: 5;
 	__REG32              							: 1;
 	__REG32 TDPL         							: 3;
 	__REG32 TRRD        							: 3;
 	__REG32 TRCD        							: 3;
 	__REG32 TRP         							: 3;
 	__REG32 TRAS        							: 4;
 	__REG32 TRC         							: 5;
 	__REG32 TRFC         							: 5;
} __sdrc_actim_ctrla_bits;

/* SDRC_ACTIM_CTRLB_p */
typedef struct {
 	__REG32 TXSR        							: 8;
 	__REG32 TXP         							: 3;
 	__REG32              							: 1;
 	__REG32 TCKE        							: 3;
 	__REG32             							: 1;
 	__REG32 TWTR        							: 2;
 	__REG32              							:14;
} __sdrc_actim_ctrlb_bits;

/* SDRC_RFR_CTRL_p */
typedef struct {
 	__REG32 ARE         							: 2;
 	__REG32              							: 6;
 	__REG32 ARCV        							:16;
 	__REG32             							: 8;
} __sdrc_rfr_ctrl_bits;

/* SDRC_MANUAL_p */
typedef struct {
 	__REG32 CMDCODE      							: 4;
 	__REG32              							:12;
 	__REG32 CMDPARAM    							:16;
} __sdrc_manual_bits;

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
 	__REG32 																: 2;
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
 	__REG32 LINK_LIST_CPBLTY_TYPE123				: 1;
 	__REG32 LINK_LIST_CPBLTY_TYPE4					: 1;
 	__REG32 																:10;
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
 	__REG32                           						: 1;
 	__REG32 EOSB_INTERRUPT_CPBLTY     						: 1;
 	__REG32 																			:17;
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
 	__REG32         													: 1;
 	__REG32 SUPER_BLOCK_IE										: 1;
 	__REG32 																	:17;
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
 	__REG32           												: 1;
 	__REG32 SUPER_BLOCK 											: 1;
 	__REG32 																	:17;
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

/* DMA4_CDPi */
typedef struct {
 	__REG32 DEST_VALID                    : 2;
 	__REG32 SRC_VALID                     : 2;
 	__REG32 NEXT_DESCRIPTOR_TYPE          : 3;
 	__REG32 PAUSE_LINK_LIST               : 1;
 	__REG32 TRANSFER_MODE                 : 2;
 	__REG32 FAST                          : 1;
 	__REG32 		  											  :21;
} __dma4_cdp_bits;

/* DMA4_CCDNi */
typedef struct {
 	__REG32 CURRENT_DESCRIPTOR_NBR        :16;
 	__REG32 		  											  :16;
} __dma4_ccdn_bits;

/* INTCPS_REVISION */
typedef struct {
 	__REG32 REV     				: 8;
 	__REG32       	 				:24;
} __intcps_revision_bits;

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

/* INTC_SYSCONFIG */
typedef struct {
 	__REG32 AUTOIDLE				: 1;
 	__REG32 SOFTRESET				: 1;
 	__REG32 								:30;
} __intc_sysconfig_bits;

/* INTC_IDLE */
typedef struct {
 	__REG32 FUNCIDLE				: 1;
 	__REG32 TURBO						: 1;
 	__REG32 								:30;
} __intc_idle_bits;

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

/* SCM_CONTROL_PADCONF_X */
typedef struct {
 	__REG32 MUXMODE0										: 3;
 	__REG32 PULLUDENABLE0								: 1;
 	__REG32 PULLTYPESELECT0							: 1;
 	__REG32        	        						: 3;
 	__REG32 INPUTENABLE0								: 1;
 	__REG32 OFFENABLE0  								: 1;
 	__REG32 OFFOUTENABLE0								: 1;
 	__REG32 OFFOUTVALUE0								: 1;
 	__REG32 OFFPULLUDENABLE0  					: 1;
 	__REG32 OFFPULLTYPESELECT0 					: 1;
 	__REG32 WAKEUPENABLE0     					: 1;
 	__REG32 WAKEUPEVENT0      					: 1;
 	__REG32 MUXMODE1										: 3;
 	__REG32 PULLUDENABLE1								: 1;
 	__REG32 PULLTYPESELECT1							: 1;
 	__REG32        	        						: 3;
 	__REG32 INPUTENABLE1								: 1;
 	__REG32 OFFENABLE1  								: 1;
 	__REG32 OFFOUTENABLE1								: 1;
 	__REG32 OFFOUTVALUE1								: 1;
 	__REG32 OFFPULLUDENABLE1  					: 1;
 	__REG32 OFFPULLTYPESELECT1 					: 1;
 	__REG32 WAKEUPENABLE1     					: 1;
 	__REG32 WAKEUPEVENT1      					: 1;
} __scm_control_padconf_bits;

/* SCM_CONTROL_GENERAL */
typedef struct {
 	__REG32 FORCEOFFMODEEN  						: 1;
 	__REG32 STARTSAVE       						: 1;
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
 	__REG32        	        						:18;
 	__REG32 SPARE     									: 7;
} __scm_control_devconf0_bits;

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
 	__REG32             								: 6;
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

/* SCM_CONTROL_PROT_CTRL */
typedef struct {
 	__REG32             								: 5;
 	__REG32 OBSERVABILITYDISABLE  			: 1;
 	__REG32        	        						:26;
} __scm_control_prot_ctrl_bits;

/* SCM_CONTROL_DEVCONF1 */
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
 	__REG32 CARKITHSUSB0DATA0AUTOEN 		: 1;
 	__REG32 CARKITHSUSB0DATA1AUTOEN			: 1;
 	__REG32 SENSDMAREQ4									: 1;
 	__REG32 SENSDMAREQ5									: 1;
 	__REG32 SENSDMAREQ6									: 1;
 	__REG32        	        						: 8;
} __scm_control_devconf1_bits;

/* SCM_CONTROL_PROT_ERR_STATUS */
typedef struct {
 	__REG32 OCMROMFWERROR								: 1;
 	__REG32 OCMRAMFWERROR								: 1;
 	__REG32 GPMCFWERROR									: 1;
 	__REG32 SMSFUNCFWERROR							: 1;
 	__REG32 SMSFWERROR									: 1;
 	__REG32 MAD2DFWERROR								: 1;
 	__REG32 IVA2FWERROR		  						: 1;
 	__REG32 L4COREFWERROR								: 1;
 	__REG32 SYSDMAACCERROR							: 1;
 	__REG32 CAMERADMAACCERROR						: 1;
 	__REG32 DISPDMAACCERROR  						: 1;
 	__REG32             								: 1;
 	__REG32 SMXAPERTFWERROR							: 1;
 	__REG32           									: 2;
 	__REG32 D2DFWERROR    							: 1;
 	__REG32 L4PERIPHFWERROR							: 1;
 	__REG32 L4EMUFWERROR								: 1;
 	__REG32        	        						:14;
} __scm_control_prot_err_status_bits;

/* SCM_CONTROL_PROT_ERR_STATUS_DEBUG */
typedef struct {
 	__REG32 OCMROMDBGFWERROR						: 1;
 	__REG32 OCMRAMDBGFWERROR						: 1;
 	__REG32 GPMCDBGFWERROR							: 1;
 	__REG32 SMSDBGFWERROR								: 1;
 	__REG32        	        						: 1;
 	__REG32 MAD2D2DBGFWERROR						: 1;
 	__REG32 IVA2DBGFWERROR		  				: 1;
 	__REG32 L4COREDBGFWERROR						: 1;
 	__REG32        	        						: 4;
 	__REG32 SMXAPERTDBGFWERROR					: 1;
 	__REG32        	        						: 3;
 	__REG32 L4PERIPHERALDBGFWERROR			: 1;
 	__REG32 L4EMUDBGFWERROR							: 1;
 	__REG32        	        						:14;
} __scm_control_prot_err_status_debug_bits;

/* SCM_CONTROL_STATUS */
typedef struct {
 	__REG32 SYS_BOOT										: 6;
 	__REG32        	        						: 2;
 	__REG32 DEVICETYPE									: 3;
 	__REG32        	        						:21;
} __scm_control_status_bits;

/* SCM_CONTROL_GENERAL_PURPOSE_STATUS */
typedef struct {
 	__REG32 SAVEDONE										: 1;
 	__REG32        	        						:31;
} __scm_control_general_purpose_status_bits;

/* SCM_CONTROL_USB_CONF_0 */
typedef struct {
 	__REG32 USB_VENDOR_ID 							:16;
 	__REG32 USB_PROD_ID     						:16;
} __scm_control_usb_conf_0_bits;

/* SCM_CONTROL_FUSE_OPP1G_VDD1 */
typedef struct {
 	__REG32 FUSE_OPP1G_VDD1							:24;
 	__REG32                  						: 8;
} __scm_control_fuse_opp1g_vdd1_bits;

/* SCM_CONTROL_FUSE_OPP50_VDD1 */
typedef struct {
 	__REG32 FUSE_OPP50_VDD1							:24;
 	__REG32                  						: 8;
} __scm_control_fuse_opp50_vdd1_bits;

/* SCM_CONTROL_FUSE_OPP100_VDD1 */
typedef struct {
 	__REG32 FUSE_OPP100_VDD1						:24;
 	__REG32                  						: 8;
} __scm_control_fuse_opp100_vdd1_bits;

/* SCM_CONTROL_FUSE_OPP130_VDD1 */
typedef struct {
 	__REG32 FUSE_OPP130_VDD1						:24;
 	__REG32                  						: 8;
} __scm_control_fuse_opp130_vdd1_bits;

/* SCM_CONTROL_FUSE_OPP50_VDD2 */
typedef struct {
 	__REG32 FUSE_OPP50_VDD2 						:24;
 	__REG32                  						: 8;
} __scm_control_fuse_opp50_vdd2_bits;

/* SCM_CONTROL_FUSE_OPP100_VDD2 */
typedef struct {
 	__REG32 FUSE_OPP100_VDD2 						:24;
 	__REG32                  						: 8;
} __scm_control_fuse_opp100_vdd2_bits;

/* SCM_CONTROL_FUSE_SR */
typedef struct {
 	__REG32 FUSE_SR1										: 8;
 	__REG32 FUSE_SR2										: 8;
 	__REG32        	        						:16;
} __scm_control_fuse_sr_bits;

/* SCM_CONTROL_IVA2_BOOTMOD */
typedef struct {
 	__REG32 BOOTMODE										: 4;
	__REG32        	        						:28;
} __scm_control_iva2_bootmod_bits;

/* SCM_CONTROL_PROG_IO2 */
typedef struct {
 	__REG32 PRG_MCSPI1_CS1_LB						: 1;
 	__REG32 PRG_MCSPI1_MIN_CFG_LB				: 1;
 	__REG32 PRG_HDQ_SC      						: 2;
 	__REG32 PRG_HDQ_LB      						: 2;
 	__REG32 PRG_CHASSIS_JTAG_LB_STR 		: 1;
 	__REG32 PRG_I2C3_PULLUPRESX      		: 1;
 	__REG32 PRG_I2C3_FS              		: 2;
 	__REG32 PRG_I2C2_FS              		: 2;
 	__REG32 PRG_I2C1_HS              		: 2;
 	__REG32 PRG_CHASSIS_PRCM_LB     		: 1;
 	__REG32 PRG_CHASSIS_DMA_LB       		: 2;
 	__REG32 PRG_CHASSIS_DMA_SC       		: 2;
 	__REG32 PRG_CHASSIS_INT_LB       		: 2;
 	__REG32 PRG_CHASSIS_INT_SC       		: 2;
 	__REG32 PRG_CHASSIS_CLOCK_LB     		: 2;
 	__REG32 PRG_CHASSIS_CLOCK_SC     		: 2;
 	__REG32 PRG_CHASSIS_AD2D_LB      		: 1;
 	__REG32 PRG_ETK_CUT1_LB         		: 1;
 	__REG32 PRG_ETK_CUT2_LB         		: 1;
 	__REG32 PRG_CLKOUT2_LB           		: 1;
 	__REG32 PRG_MCBSP2_LB           		: 1;
} __scm_control_prog_io2_bits;

/* SCM_CONTROL_MEM_RTA_CTRL */
typedef struct {
 	__REG32 HD_MEM_RTA_SEL  						: 1;
 	__REG32                     				:31;
} __scm_control_mem_rta_ctrl_bits;

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

/* SCM_CONTROL_PROG_IO0 */
typedef struct {
 	__REG32        	        						: 1;
 	__REG32 PRG_SDMMC_PUSTRENGTH				: 1;
 	__REG32 PRG_GPMC_NWP_LB     				: 1;
 	__REG32 PRG_GPMC_NBE1_LB    				: 1;
 	__REG32 PRG_GPMC_NBE0_CLE_LB				: 1;
 	__REG32 PRG_GMPC_CLK_LB     				: 1;
 	__REG32 PRG_GPMC_NCS7_LB    				: 1;
 	__REG32 PRG_GPMC_NCS6_LB      			: 1;
 	__REG32 PRG_GPMC_NCS5_LB    				: 1;
 	__REG32 PRG_GPMC_NCS4_LB    				: 1;
 	__REG32 PRG_GPMC_NCS3_LB    				: 1;
 	__REG32 PRG_GPMC_NCS2_LB    				: 1;
 	__REG32 PRG_GPMC_NCS1_LB    				: 1;
 	__REG32 PRG_GPMC_NCS0_LB    				: 1;
 	__REG32 PRG_GPMC_D8_D15_LB  				: 1;
 	__REG32 PRG_GPMC_MIN_CFG_LB 				: 1;
 	__REG32 PRG_GPMC_A11_LB      				: 1;
 	__REG32 PRG_GPMC_A10_LB      				: 1;
 	__REG32 PRG_GPMC_A9_LB      				: 1;
 	__REG32 PRG_GPMC_A8_LB      				: 1;
 	__REG32 PRG_GPMC_A7_LB      				: 1;
 	__REG32 PRG_GPMC_A6_LB      				: 1;
 	__REG32 PRG_GPMC_A5_LB      				: 1;
 	__REG32 PRG_GPMC_A4_LB      				: 1;
 	__REG32 PRG_GPMC_A3_LB      				: 1;
 	__REG32 PRG_GPMC_A2_LB      				: 1;
 	__REG32 PRG_GPMC_A1_LB      				: 1;
 	__REG32 SDRC_NCS1            				: 1;
 	__REG32 SDRC_NCS0            				: 1;
 	__REG32 SDRC_ADDRCTR         				: 1;
 	__REG32 SDRC_HIGHDATA        				: 1;
 	__REG32 SDRC_LOWDATA        				: 1;
} __scm_control_prog_io0_bits;

/* SCM_CONTROL_PROG_IO1 */
typedef struct {
 	__REG32 PRG_I2C2_PULLUPRESX 				: 1;
 	__REG32 PRG_UART2_LB        				: 1;
 	__REG32 PRG_MCSPI2_LB        				: 1;
 	__REG32 PRG_MCSPI1_CS3_LB    				: 1;
 	__REG32 PRG_MCSPI1_CS2_LB   				: 1;
 	__REG32 PRG_MCBSP4_LB        				: 1;
 	__REG32 PRG_MCBSP3_LB       				: 1;
 	__REG32 PRG_MCBSP1_LB         			: 1;
 	__REG32 PRG_MCBSP_CLKS_LB   				: 1;
 	__REG32 PRG_MCBSP1_DUPLEX_LB 				: 1;
 	__REG32 PRG_UART3_LB        				: 2;
 	__REG32 PRG_UART3_SC        				: 2;
 	__REG32 PRG_UART1_LB        				: 1;
 	__REG32 PRG_SDMMC2_EXT_LB    				: 1;
 	__REG32 PRG_SDMMC2_MIN_CFG_LB				: 1;
 	__REG32 PRG_HSUSB0_LB        				: 1;
 	__REG32 PRG_HSUSB0_CLK_LB    				: 1;
 	__REG32 PRG_I2C1_PULLUPRESX  				: 1;
 	__REG32 PRG_SDMMC1_SPEEDCTRL 				: 1;
 	__REG32 PRG_CAM_DATA_LB      				: 1;
 	__REG32 PRG_CAM_SIDEBAND_LB  				: 1;
 	__REG32 PRG_DISP_DATA_LB     				: 1;
 	__REG32 PRG_DISP_DSI_LB      				: 2;
 	__REG32 PRG_DISP_DSI_SC      				: 2;
 	__REG32 PRG_DISP_SIDEBAND_LB 				: 1;
 	__REG32 PRG_GPMC_WAIT3_LB    				: 1;
 	__REG32 PRG_GPMC_WAIT2_LB    				: 1;
 	__REG32 PRG_GPMC_WAIT1_LB    				: 1;
} __scm_control_prog_io1_bits;

/* SCM_CONTROL_DSS_DPLL_SPREADING */
typedef struct {
 	__REG32                       			: 4;
 	__REG32 DSS_SPREADING_ENABLE				: 1;
 	__REG32        	 					      		: 2;
 	__REG32 DSS_SPREADING_ENABLE_STATUS	: 1;
 	__REG32 Q_DSS_SPREADING_SIDE      	: 1;
 	__REG32        	 					      		:23;
} __scm_control_dss_dpll_spreading_bits;

/* SCM_CONTROL_CORE_DPLL_SPREADING */
typedef struct {
 	__REG32                   					  : 4;
 	__REG32 CORE_SPREADING_ENABLE				  : 1;
 	__REG32        	 					      		  : 2;
 	__REG32 CORE_SPREADING_ENABLE_STATUS  : 1;
 	__REG32 Q_CORE_SPREADING_SIDE         : 1;
 	__REG32        	 					      		  :23;
} __scm_control_core_dpll_spreading_bits;

/* SCM_CONTROL_PER_DPLL_SPREADING */
typedef struct {
 	__REG32                   					: 4;
 	__REG32 PER_SPREADING_ENABLE				: 1;
 	__REG32        	 					      		: 2;
 	__REG32 PER_SPREADING_ENABLE_STATUS	: 1;
 	__REG32 Q_PER_SPREADING_SIDE      	: 1;
 	__REG32        	 					      		:23;
} __scm_control_per_dpll_spreading_bits;

/* SCM_CONTROL_USBHOST_DPLL_SPREADING */
typedef struct {
 	__REG32                         				: 4;
 	__REG32 USBHOST_SPREADING_ENABLE				: 1;
 	__REG32        	 					      				: 2;
 	__REG32 USBHOST_SPREADING_ENABLE_STATUS	: 1;
 	__REG32 Q_USBHOST_SPREADING_SIDE      	: 1;
 	__REG32        	 					      				:23;
} __scm_control_usbhost_dpll_spreading_bits;

/* SCM_CONTROL_SDRC_SHARING */
typedef struct {
 	__REG32 SDRCSHARING             				:30;
 	__REG32 SDRCSHARINGLOCK         				: 1;
 	__REG32        	 					      				: 1;
} __scm_control_sdrc_sharing_bits;

/* SCM_CONTROL_SDRC_MCFG0 */
typedef struct {
 	__REG32 SDRCMCFG0                				:30;
 	__REG32 SDRCMCFG0LOCK            				: 1;
 	__REG32        	 					      				: 1;
} __scm_control_sdrc_mcfg0_bits;

/* SCM_CONTROL_SDRC_MCFG1 */
typedef struct {
 	__REG32 SDRCMCFG1                				:30;
 	__REG32 SDRCMCFG1LOCK            				: 1;
 	__REG32        	 					      				: 1;
} __scm_control_sdrc_mcfg1_bits;

/* SCM_CONTROL_MODEM_FW_CONFIGURATION_LOCK */
typedef struct {
 	__REG32 FWCONFIGURATIONLOCK      				: 1;
 	__REG32        	 					      				:31;
} __scm_control_modem_fw_configuration_lock_bits;

/* SCM_CONTROL_MODEM_MEMORY_RESOURCES_CONF */
typedef struct {
 	__REG32 MODEMGPMCRESERVEDBASEADDR				:12;
 	__REG32 MODEMGPMCRESERVEDS1SIZE 				: 5;
 	__REG32 MODEMGPMCRESERVEDS2SIZE 				: 5;
 	__REG32 MODEMSMSMEMORYSIZE        			: 5;
 	__REG32 MODEMSTACKMEMORYSIZE    				: 4;
 	__REG32 CMDWTOCMRAMSWITCH       				: 1;
} __scm_control_modem_memory_resources_conf_bits;

/* SCM_CONTROL_MODEM_GPMC_DT_FW_REQ_INFO */
typedef struct {
 	__REG32 MODEMGPMCDTFWREQINFO    				:16;
 	__REG32                          				:16;
} __scm_control_modem_gpmc_dt_fw_req_info_bits;

/* SCM_CONTROL_MODEM_GPMC_DT_FW_RD */
typedef struct {
 	__REG32 MODEMGPMCDTFWRD         				:16;
 	__REG32                          				:16;
} __scm_control_modem_gpmc_dt_fw_rd_bits;

/* SCM_CONTROL_MODEM_GPMC_DT_FW_WR */
typedef struct {
 	__REG32 MODEMGPMCDTFWWR         				:16;
 	__REG32                          				:16;
} __scm_control_modem_gpmc_dt_fw_wr_bits;

/* SCM_CONTROL_MODEM_GPMC_BOOT_CODE */
typedef struct {
 	__REG32 GPMCBOOTCODESIZE         				: 5;
 	__REG32 GPMCBOOTCODEWRITEPROTECTED			: 1;
 	__REG32                          				:26;
} __scm_control_modem_gpmc_boot_code_bits;

/* SCM_CONTROL_MODEM_SMS_RG_RDPERM1 */
typedef struct {
 	__REG32 SMSRGRDPERM1             				:16;
	__REG32                          				:16;
} __scm_control_modem_sms_rg_rdperm1_bits;

/* SCM_CONTROL_MODEM_SMS_RG_WRPERM1 */
typedef struct {
 	__REG32 SMSRGWRPERM1             				:16;
	__REG32                          				:16;
} __scm_control_modem_sms_rg_wrperm1_bits;

/* SCM_CONTROL_MODEM_D2D_FW_DEBUG_MODE */
typedef struct {
 	__REG32 D2DFWDEBUGMODE           				: 1;
	__REG32                          				:31;
} __scm_control_modem_d2d_fw_debug_mode_bits ;

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

/* SCM_CONTROL_DPF_REGION4_GPMC_FW_REQINFO */
typedef struct {
 	__REG32 REGION4GPMCFWREQINFO  					:16;
 	__REG32        	 					      				:16;
} __scm_control_dpf_region4_gpmc_fw_reqinfo_bits;

/* SCM_CONTROL_DPF_REGION4_GPMC_FW_WR */
typedef struct {
 	__REG32 REGION4GPMCFWWR									:16;
 	__REG32        	 					      				:16;
} __scm_control_dpf_region4_gpmc_fw_wr_bits;

/* SCM_CONTROL_DPF_REGION1_IVA2_FW_ADDR_MATCH */
typedef struct {
 	__REG32 REGION1IVA2FWADDRMATCH					:24;
 	__REG32        	 					      				: 8;
} __scm_control_dpf_region4_iva2_fw_addr_match_bits;

/* SCM_CONTROL_DPF_REGION1_IVA2_FW_REQINFO */
typedef struct {
 	__REG32 REGION1IVA2FWREQINFO  					:16;
 	__REG32        	 					      				:16;
} __scm_control_dpf_region4_iva2_fw_reqinfo_bits;

/* SCM_CONTROL_DPF_REGION1_IVA2_FW_WR */
typedef struct {
 	__REG32 REGION1IVA2FWWR       					:13;
 	__REG32        	 					      				:19;
} __scm_control_dpf_region4_iva2_fw_wr_bits;

/* SCM_CONTROL_PBIAS_LITE */
typedef struct {
 	__REG32 PBIASLITEVMODE0       					: 1;
 	__REG32 PBIASLITEPWRDNZ0       					: 1;
 	__REG32                       					: 1;
 	__REG32 PBIASLITEVMODEERROR0   					: 1;
 	__REG32        	 					      				: 3;
 	__REG32 PBIASLITESUPPLYHIGH0   					: 1;
 	__REG32 PBIASLITEVMODE1       					: 1;
 	__REG32 PBIASLITEPWRDNZ1       					: 1;
 	__REG32                       					: 1;
 	__REG32 PBIASLITEVMODEERROR1   					: 1;
 	__REG32        	 					      				: 3;
 	__REG32 PBIASLITESUPPLYHIGH1   					: 1;
 	__REG32                        					:16;
} __scm_control_pbias_lite_bits;

/* SCM_CONTROL_TEMP_SENSOR */
typedef struct {
 	__REG32 TEMP                   					: 8;
 	__REG32 EOCZ                   					: 1;
 	__REG32 SOC                   					: 1;
 	__REG32 CONTCONV               					: 1;
 	__REG32                        					:21;
} __scm_control_temp_sensor_bits;

/* SCM_CONTROL_DPF_MAD2D_FW_ADDR_MATCH */
typedef struct {
 	__REG32 REGIONMAD2DFWADDRMATCH 					:28;
 	__REG32                        					: 4;
} __scm_control_dpf_mad2d_fw_addr_match_bits;

/* SCM_CONTROL_DPF_MAD2D_FW_REQINFO */
typedef struct {
 	__REG32 MAD2DFWREQINFO         					:16;
 	__REG32                        					:16;
} __scm_control_dpf_mad2d_fw_reqinfo_bits;

/* SCM_CONTROL_DPF_MAD2D_FW_WR */
typedef struct {
 	__REG32 REGIONMAD2DFWWR       					:16;
 	__REG32                        					:16;
} __scm_control_dpf_mad2d_fw_wr_bits;

/* SCM_CONTROL_DSS_DPLL_SPREADING_FREQ */
typedef struct {
 	__REG32 R_DSS_MOD_FREQ_MANT    					: 7;
 	__REG32 R_DSS_MOD_FREQ_EXP    					: 3;
 	__REG32 R_DSS_DELTA_M_FRACT    					:18;
 	__REG32 R_DSS_DELTA_M_INT     					: 2;
 	__REG32                        					: 2;
} __scm_control_dss_dpll_spreading_freq_bits;

/* SCM_CONTROL_CORE_DPLL_SPREADING_FREQ */
typedef struct {
 	__REG32 R_CORE_MOD_FREQ_MANT   					: 7;
 	__REG32 R_CORE_MOD_FREQ_EXP    					: 3;
 	__REG32 R_CORE_DELTA_M_FRACT   					:18;
 	__REG32 R_CORE_DELTA_M_INT     					: 2;
 	__REG32                        					: 2;
} __scm_control_core_dpll_spreading_freq_bits;

/* SCM_CONTROL_PER_DPLL_SPREADING_FREQ */
typedef struct {
 	__REG32 R_PER_MOD_FREQ_MANT   					: 7;
 	__REG32 R_PER_MOD_FREQ_EXP    					: 3;
 	__REG32 R_PER_DELTA_M_FRACT   					:18;
 	__REG32 R_PER_DELTA_M_INT     					: 2;
 	__REG32                        					: 2;
} __scm_control_per_dpll_spreading_freq_bits;

/* SCM_CONTROL_USBHOST_DPLL_SPREADING_FREQ */
typedef struct {
 	__REG32 R_USBHOST_MOD_FREQ_MANT   			: 7;
 	__REG32 R_USBHOST_MOD_FREQ_EXP    			: 3;
 	__REG32 R_USBHOST_DELTA_M_FRACT   			:18;
 	__REG32 R_USBHOST_DELTA_M_INT     			: 2;
 	__REG32                        					: 2;
} __scm_control_usbhost_dpll_spreading_freq_bits;

/* SCM_CONTROL_AVDAC1 */
typedef struct {
 	__REG32                            			:16;
 	__REG32 AVDAC1_COMP_EN0           			: 1;
 	__REG32 AVDAC1_COMP_EN1           			: 1;
 	__REG32 AVDAC1_COMP_EN2           			: 1;
 	__REG32 AVDAC1_COMP_EN3           			: 1;
 	__REG32                        					:12;
} __scm_control_avdac1_bits;

/* SCM_CONTROL_AVDAC2 */
typedef struct {
 	__REG32                            			:16;
 	__REG32 AVDAC2_COMP_EN0           			: 1;
 	__REG32 AVDAC2_COMP_EN1           			: 1;
 	__REG32 AVDAC2_COMP_EN2           			: 1;
 	__REG32 AVDAC2_COMP_EN3           			: 1;
 	__REG32                        					:12;
} __scm_control_avdac2_bits;

/* SCM_CONTROL_CAMERA_PHY_CTRL */
typedef struct {
 	__REG32 R_CONTROL_CAMERA2_PHY_CAMMODE 	: 2;
 	__REG32 R_CONTROL_CAMERA1_PHY_CAMMODE 	: 2;
 	__REG32 R_CONTROL_CSI1_RX_SEL      			: 1;
 	__REG32                        					:27;
} __scm_control_camera_phy_ctrl_bits;

/* SCM_CONTROL_WKUP_CTRL */
typedef struct {
 	__REG32 MM_FSUSB1_TXEN_N_OUT_POLARITY_CTRL	: 1;
 	__REG32 MM_FSUSB2_TXEN_N_OUT_POLARITY_CTRL	: 1;
 	__REG32 MM_FSUSB3_TXEN_N_OUT_POLARITY_CTRL	: 1;
 	__REG32                                   	: 2;
 	__REG32 GPIO_1_IN_SEL_SAD2D_NRESWARM_IN_SEL	: 1;
 	__REG32 GPIO_IO_PWRDNZ                    	: 1;
 	__REG32        	 					       						:25;
} __scm_control_wkup_ctrl_bits;

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
 	__REG32 																:18;
 	__REG32 WKUPOBSERVABILITYDISABLE  			: 1;
} __scm_control_wkup_debobs_4_bits;

/* SCM_CONTROL_PROG_IO_WKUP1 */
typedef struct {
 	__REG32 																: 3;
 	__REG32 PRG_SR_LB												: 2;
 	__REG32 PRG_SR_PULLUPRESX								: 1;
 	__REG32 PRG_GPIO_128_LB									: 2;
 	__REG32 PRG_GPIO_128_SC									: 2;
 	__REG32 PRG_CHASSIS_PRCM_LB_WKUP				: 1;
 	__REG32 PRG_CLKOUT1_LB          				: 2;
 	__REG32 PRG_CLKOUT1_SC          				: 2;
 	__REG32 PRG_OFFMODE_LB          				: 2;
 	__REG32 PRG_OFFMODE_SC          				: 2;
 	__REG32 PRG_SYSBOOT_LB          				: 1;
 	__REG32 PRG_NIRQ_LB             				: 2;
 	__REG32 PRG_NIRQ_SC             				: 2;
 	__REG32 PRG_CLKREQ_LB           				: 2;
 	__REG32 PRG_CLKREQ_SC           				: 2;
 	__REG32 PRG_32K_LB               				: 2;
 	__REG32 PRG_32K_SC               				: 2;
} __scm_control_prog_io_wkup1_bits;

/* SCM_CONTROL_BGAPTS_WKUP */
typedef struct {
 	__REG32 TMPSOFF     										: 1;
 	__REG32         												: 1;
 	__REG32 BGROFF          								: 1;
 	__REG32                          				:29;
} __scm_control_bgapts_wkup_bits;

/* SCM_CONTROL_VBBLDO_SW_CTRL */
typedef struct {
 	__REG32 AIPOFF       										: 1;
 	__REG32 LDOBYPASSZ											: 1;
 	__REG32 LDOBYPASSZ_MUX_SEL							: 1;
 	__REG32 BBSEL             							: 1;
 	__REG32 BBSEL_MUX_SEL      							: 1;
 	__REG32 NOCAP             							: 1;
 	__REG32 NOCAP_MUX_SEL      							: 1;
 	__REG32                          				:25;
} __scm_control_vbbldo_sw_ctrl_bits;

/* MAILBOX_REVISION */
typedef struct {
 	__REG32 REV          						  : 8;
 	__REG32                           :24;
} __mailbox_revision_bits;

/* MAILBOX_SYSCONFIG */
typedef struct {
 	__REG32 AUTOIDLE      						: 1;
 	__REG32 SOFTRESET      						: 1;
 	__REG32                           : 1;
 	__REG32 SIDLEMODE      						: 2;
 	__REG32               						: 3;
 	__REG32 CLOCKACTIVITY  						: 1;
 	__REG32                           :23;
} __mailbox_sysconfig_bits;

/* MAILBOX_SYSSTATUS */
typedef struct {
 	__REG32 RESETDONE      						: 1;
 	__REG32                           :31;
} __mailbox_sysstatus_bits;

/* MAILBOX_FIFOSTATUS_m */
typedef struct {
 	__REG32 FIFOFULLMB     						: 1;
 	__REG32                           :31;
} __mailbox_fifostatus_bits;

/* MAILBOX_MSGSTATUS_m */
typedef struct {
 	__REG32 NBOFMSGMB     						: 3;
 	__REG32                           :29;
} __mailbox_msgstatus_bits;

/* MAILBOX_IRQSTATUS_u */
typedef struct {
 	__REG32 NEWMSGSTATUSUUMB0 				: 1;
 	__REG32 NOTFULLSTATUSUUMB0 				: 1;
 	__REG32 NEWMSGSTATUSUUMB1 				: 1;
 	__REG32 NOTFULLSTATUSUUMB1 				: 1;
 	__REG32                           :28;
} __mailbox_irqstatus_bits;

/* MAILBOX_IRQENABLE_u */
typedef struct {
 	__REG32 NEWMSGENABLEUUMB0 				: 1;
 	__REG32 NOTFULLENABLEUUMB0 				: 1;
 	__REG32 NEWMSGENABLEUUMB1 				: 1;
 	__REG32 NOTFULLENABLEUUMB1 				: 1;
 	__REG32                           :28;
} __mailbox_irqenable_bits;

/* MMU_REVISION */
typedef struct {
 	__REG32 REV            				: 8;
 	__REG32                       :24;
} __mmu_revision_bits;

/* MMU_SYSCONFIG */
typedef struct {
 	__REG32 AUTOIDLE       				: 1;
 	__REG32 SOFTRESET     				: 1;
 	__REG32                				: 1;
 	__REG32 IDLEMODE      				: 2;
 	__REG32               				: 3;
 	__REG32 CLOCKACTIVITY 				: 2;
 	__REG32                       :22;
} __mmu_sysconfig_bits;

/* MMU_SYSSTATUS */
typedef struct {
 	__REG32 RESETDONE      				: 1;
 	__REG32                       :31;
} __mmu_sysstatus_bits;

/* MMU_IRQSTATUS */
typedef struct {
 	__REG32 TLBMISS        				: 1;
 	__REG32 TRANSLATIONFAULT			: 1;
 	__REG32 EMUMISS        				: 1;
 	__REG32 TABLEWALKFAULT 				: 1;
 	__REG32 MULTIHITFAULT 				: 1;
 	__REG32                       :27;
} __mmu_irqstatus_bits;

/* MMU_WALKING_ST */
typedef struct {
 	__REG32 TWLRUNNING    				: 1;
 	__REG32                       :31;
} __mmu_walking_st_bits;

/* MMU_CNTL */
typedef struct {
 	__REG32               				: 1;
 	__REG32 MMUENABLE     				: 1;
 	__REG32 TWLENABLE     				: 1;
 	__REG32 EMUTLBUPDATE   				: 1;
 	__REG32                       :28;
} __mmu_cntl_bits;

/* MMU_LOCK */
typedef struct {
 	__REG32               				: 4;
 	__REG32 CURRENTVICTIM 				: 5;
 	__REG32                				: 1;
 	__REG32 BASEVALUE      				: 5;
 	__REG32                       :17;
} __mmu_lock_bits;

/* MMU_LD_TLB */
typedef struct {
 	__REG32 LDTLBITEM      				: 1;
 	__REG32                       :31;
} __mmu_ld_tlb_bits;

/* MMU_CAM */
typedef struct {
 	__REG32 PAGESIZE      				: 2;
 	__REG32 V             				: 1;
 	__REG32 P             				: 1;
 	__REG32                   		: 8;
 	__REG32 VATAG         				:20;
} __mmu_cam_bits;

/* MMU_RAM */
typedef struct {
 	__REG32               				: 6;
 	__REG32 MIXED                 : 1;
 	__REG32 ELEMENTSIZE    				: 2;
 	__REG32 ENDIANNESS    				: 1;
 	__REG32                   		: 2;
 	__REG32 PHYSICALADDRESS				:20;
} __mmu_ram_bits;

/* MMU_GFLUSH */
typedef struct {
 	__REG32 GLOBALFLUSH           : 1;
 	__REG32                   		:31;
} __mmu_gflush_bits;

/* MMU_FLUSH_ENTRY */
typedef struct {
 	__REG32 FLUSHENTRY            : 1;
 	__REG32                   		:31;
} __mmu_flush_entry_bits;

/* MMU_FLUSH_ENTRY */
typedef struct {
 	__REG32 FLUSHENTRY            : 1;
 	__REG32                   		:31;
} __mmu_emu_fault_ad_bits;

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

/* GPT1_TWPS */
/* GPT2_TWPS */
/* GPT10_TWPS */
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

/* GPT3_TWPS */ 
/* GPT4_TWPS */ 
/* GPT5_TWPS */ 
/* GPT6_TWPS */ 
/* GPT7_TWPS */ 
/* GPT8_TWPS */ 
/* GPT9_TWPS */ 
/* GPT11_TWPS */ 
typedef struct {
 	__REG32 W_PEND_TCLR								: 1;
 	__REG32 W_PEND_TCRR								: 1;
 	__REG32 W_PEND_TLDR								: 1;
 	__REG32 W_PEND_TTGR								: 1;
 	__REG32 W_PEND_TMAR								: 1;
 	__REG32 													:27;
} __gpt3_twps_bits;

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
 	__REG16 XUDF_IE										: 1;
 	__REG16 ROVR_IE										: 1;
 	__REG16         									: 1;
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
 	__REG16 XUDF_WE										: 1;
 	__REG16 ROVR_WE										: 1;
 	__REG16 													: 1;
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
 	__REG16 SDA_O_FUNC  							: 1;
 	__REG16 SDA_I_FUNC								: 1;
 	__REG16 SCL_O_FUNC								: 1;
 	__REG16 SCL_I_FUNC								: 1;
 	__REG16 													: 2;
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

/* UARTn_IER_REG */
typedef union {
	/* UARTx_IER_REG */
	struct {
 	__REG8  RHR_IT								: 1;
 	__REG8  THR_IT								: 1;
 	__REG8  LINE_STS_IT					  : 1;
 	__REG8  MODEM_STS_IT					: 1;
 	__REG8  SLEEP_MODE						: 1;
 	__REG8  XOFF_IT							  : 1;
 	__REG8  RTS_IT								: 1;
 	__REG8  CTS_IT								: 1;
 	};
	/* UARTx_IrDA_IER_REG */
	struct {
 	__REG8  RHR_IT  							: 1;
 	__REG8  THR_IT  							: 1;
 	__REG8  LAST_RX_BYTE_IT				: 1;
 	__REG8  RX_OVERRUN_IT 				: 1;
 	__REG8  STS_FIFO_TRIG_IT			: 1;
 	__REG8  TX_STATUS_IT					: 1;
 	__REG8  LINE_STS_IT_I					: 1;
 	__REG8  EOF_IT								: 1;
 	} IrDA;
	/* UARTx_CIR_IER_REG */
	struct {
 	__REG8  RHR_IT  							: 1;
 	__REG8  THR_IT  							: 1;
 	__REG8  RX_STOP_IT  					: 1;
 	__REG8  RX_OVERRUN_IT					: 1;
 	__REG8              					: 1;
 	__REG8  TX_STATUS_IT  				: 1;
 	__REG8  											: 2;
 	} CIR;
} __uart_ier_reg_bits;

/* UARTn_FCR_REG */
typedef union {
	/* UARTx_FCR_REG */
	struct {
 	__REG8  FIFO_EN								: 1;
 	__REG8  RX_FIFO_CLEAR					: 1;
 	__REG8  TX_FIFO_CLEAR					: 1;
 	__REG8  DMA_MODE							: 1;
 	__REG8  TX_FIFO_TRIG					: 2;
 	__REG8  RX_FIFO_TRIG					: 2;
 	};
	/* UARTx_IIR_REG */
	struct {
 	__REG8  IT_PENDING						: 1;
 	__REG8  IT_TYPE								: 5;
 	__REG8  FCR_MIRROR						: 2;
 	};
	/* UARTx_IrDA_IIR_REG */
	struct {
 	__REG8  RHR_IT  							: 1;
 	__REG8  THR_IT  							: 1;
 	__REG8  RX_FIFO_LB_IT   			: 1;
 	__REG8  RX_OE_IT       				: 1;
 	__REG8  STS_FIFO_IT     			: 1;
 	__REG8  TX_STATUS_IT					: 1;
 	__REG8  LINE_STS_IT 					: 1;
 	__REG8  EOF_IT								: 1;
 	} IrDA;
	/* UARTx_CIR_IIR_REG */
	struct {
 	__REG8  RHR_IT  							: 1;
 	__REG8  THR_IT								: 1;
 	__REG8  RX_STOP_IT  					: 1;
 	__REG8  RX_OE_IT    					: 1;
 	__REG8              					: 1;
 	__REG8  TX_STATUS_IT					: 1;
 	__REG8  											: 2;
 	} CIR;
	/* UARTx_EFR_REG */
	struct {
 	__REG8  SW_FLOW_CONTROL				: 4;
 	__REG8  ENHANCED_EN						: 1;
 	__REG8  SPEC_CHAR							: 1;
 	__REG8  AUTO_RTS_EN						: 1;
 	__REG8  AUTO_CTS_EN	  				: 1;
 	};
} __uart_fcr_reg_bits;

/* UARTn_LCR_REG */
typedef struct {
 	__REG8  CHAR_LENGTH						: 2;
 	__REG8  NB_STOP								: 1;
 	__REG8  PARITY_EN							: 1;
 	__REG8  PARITY_TYPE1					: 1;
 	__REG8  PARITY_TYPE2					: 1;
 	__REG8  BREAK_EN							: 1;
 	__REG8  DIV_EN								: 1;
} __uart_lcr_reg_bits;

/* UARTn_MCR_REG */
typedef struct {
 	__REG8  DTR										: 1;
 	__REG8  RTS										: 1;
 	__REG8  RI_STS_CH							: 1;
 	__REG8  CD_STS_CH							: 1;
 	__REG8  LOOPBACK_EN						: 1;
 	__REG8  XON_EN								: 1;
 	__REG8  TCR_TLR								: 1;
 	__REG8  											: 1;
} __uart_mcr_reg_bits;

/* UARTn_LSR_REG */
typedef union {
	/* UARTx_LSR_REG */
	struct {
 	__REG8  RX_FIFO_E							: 1;
 	__REG8  RX_OE									: 1;
 	__REG8  RX_PE									: 1;
 	__REG8  RX_FE									: 1;
 	__REG8  RX_BI									: 1;
 	__REG8  TX_FIFO_E							: 1;
 	__REG8  TX_SR_E								: 1;
 	__REG8  RX_FIFO_STS						: 1;
 	};
	/* UARTx_LSR_IrDA_REG */
	struct {
 	__REG8  RX_FIFO_E 						: 1;
 	__REG8  STS_FIFO_E						: 1;
 	__REG8  CRC										: 1;
 	__REG8  ABORT									: 1;
 	__REG8  FRAME_TOO_LONG				: 1;
 	__REG8  RX_LAST_BYTE					: 1;
 	__REG8  STS_FIFO_FUL					: 1;
 	__REG8  THR_EMPTY							: 1;
 	} IrDA;
	/* UARTx_LSR_CIR_REG */
	struct {
 	__REG8    										: 7;
 	__REG8  THR_EMPTY 						: 1;
 	} CIR;
} __uart_lsr_reg_bits;

/* UARTx_TCR_REG */
typedef union {
	/* UARTx_TCR_REG */
	struct {
 	__REG8  RX_FIFO_TRIG_HALT			: 4;
 	__REG8  RX_FIFO_TRIG_START		: 4;
 	};
	/* UARTx_MSR_REG */
	struct {
 	__REG8  CTS_STS								: 1;
 	__REG8  DSR_STS								: 1;
 	__REG8  RI_STS								: 1;
 	__REG8  DCD_STS								: 1;
 	__REG8  NCTS_STS							: 1;
 	__REG8  NDSR_STS							: 1;
 	__REG8  NRI_STS								: 1;
 	__REG8  NCD_STS								: 1;
 	};
} __uart_tcr_reg_bits;

/* UARTx_TLR_REG */
typedef struct {
 	__REG8  TX_FIFO_TRIG_DMA			: 4;
 	__REG8  RX_FIFO_TRIG_DMA			: 4;
} __uart_tlr_reg_bits;

/* UARTn_MDR1_REG */
typedef struct {
 	__REG8  MODE_SELECT						: 3;
 	__REG8  IR_SLEEP							: 1;
 	__REG8  SET_TXIR							: 1;
 	__REG8  SCT										: 1;
 	__REG8  SIP_MODE							: 1;
 	__REG8  FRAME_END_MODE				: 1;
} __uart_mdr1_reg_bits;

/* UARTn_MDR2_REG */
typedef struct {
 	__REG8  IRTX_UNDERRUN					: 1;
 	__REG8  STS_FIFO_TRIG					: 2;
 	__REG8  UART_PULSE						: 1;
 	__REG8  CIR_PULSE_MODE				: 2;
 	__REG8  IRRXINVERT						: 1;
 	__REG8  SET_TXIR_ALT					: 1;
} __uart_mdr2_reg_bits;

/* UARTx_SFLSR_REG */
typedef struct {
 	__REG8  											: 1;
 	__REG8  CRC_ERROR							: 1;
 	__REG8  ABORT_DETECT					: 1;
 	__REG8  FTL_ERROR							: 1;
 	__REG8  OE_ERROR							: 1;
 	__REG8  											: 3;
} __uart_sflsr_reg_bits;

/* UARTx_TXFLH_REG */
typedef struct {
 	__REG8  TXFLH									: 5;
 	__REG8  											: 3;
} __uart_txflh_reg_bits;

/* UARTx_SFREGH_REG */
typedef union {
	/* UARTx_RXFLH_REG */
	struct {
 	__REG8  RXFLH									: 4;
 	__REG8  											: 4;
 	};
	/* UARTx_SFREGH_REG */
	struct {
 	__REG8  SFREGH								: 4;
 	__REG8  											: 4;
 	};
} __uart_sfregh_reg_bits;

/* UARTn_BLR_REG */
typedef union {
	/* UARTx_BLR_REG */
	struct {
 	__REG8  											: 6;
 	__REG8  XBOF_TYPE							: 1;
 	__REG8  STS_FIFO_RESET				: 1;
 	};
	/* UARTx_UASR_REG */
	struct {
 	__REG8  SPEED									: 5;
 	__REG8  BIT_BY_CHAR						: 1;
 	__REG8  PARITY_TYPE						: 2;
 	};
} __uart_uasr_reg_bits;

/* UARTn_ACREG_REG */
typedef struct {
 	__REG8  EOT_EN								: 1;
 	__REG8  ABORT_EN							: 1;
 	__REG8  SCTX_EN								: 1;
 	__REG8  SEND_SIP							: 1;
 	__REG8  DIS_TX_UNDERRUN				: 1;
 	__REG8  DIS_IR_RX							: 1;
 	__REG8  SD_MOD								: 1;
 	__REG8  PULSE_TYPE						: 1;
} __uart_acreg_reg_bits;

/* UARTn_SCR_REG */
typedef struct {
 	__REG8  DMA_MODE_CTL					: 1;
 	__REG8  DMA_MODE_2						: 2;
 	__REG8  TX_EMPTY_CTL_IT				: 1;
 	__REG8  RX_CTS_WU_EN					: 1;
 	__REG8  											: 1;
 	__REG8  TX_TRIG_GRANU1				: 1;
 	__REG8  RX_TRIG_GRANU1				: 1;
} __uart_scr_reg_bits;

/* UARTn_SSR_REG */
typedef struct {
 	__REG8  TX_FIFO_FULL					: 1;
 	__REG8  RX_CTS_DSR_WU_STS			: 1;
 	__REG8  DMA_COUNTER_RST 			: 1;
 	__REG8  											: 5;
} __uart_ssr_reg_bits;

/* UARTn_MVR_REG */
typedef struct {
 	__REG8  MINOR_REV							: 4;
 	__REG8  MAJOR_REV							: 4;
} __uart_mvr_reg_bits;

/* UARTn_SYSC_REG */
typedef struct {
 	__REG8  AUTOIDLE							: 1;
 	__REG8  SOFTRESET							: 1;
 	__REG8  ENAWAKEUP							: 1;
 	__REG8  IDLEMODE							: 2;
 	__REG8  											: 3;
} __uart_sysc_reg_bits;

/* UARTn_SYSS_REG */
typedef struct {
 	__REG8  RESETDONE							: 1;
 	__REG8  											: 7;
} __uart_syss_reg_bits;

/* UARTn_WER_REG */
typedef struct {
 	__REG8  EVENT_0_CTS_ACTIVITY	: 1;
 	__REG8  											: 1;
 	__REG8  EVENT_2_RI_ACTIVITY		: 1;
 	__REG8  											: 1;
 	__REG8  EVENT_4_RX_ACTIVITY		: 1;
 	__REG8  EVENT_5_RHR_INTERRUPT	: 1;
 	__REG8  EVENT_6_RLS_INTERRUPT	: 1;
 	__REG8  EVENT_7_TX_WAKEUP_EN	: 1;
} __uart_wer_reg_bits;

/* UARTn_IER2_REG */
typedef struct {
 	__REG8  EN_RXFIFO_EMPTY				: 1;
 	__REG8  EN_TXFIFO_EMPTY				: 1;
 	__REG8  											: 6;
} __uart_ier2_reg_bits;

/* UARTn_ISR2_REG */
typedef struct {
 	__REG8  RXFIFO_EMPTY_STS			: 1;
 	__REG8  TXFIFO_EMPTY_STS			: 1;
 	__REG8  											: 6;
} __uart_isr2_reg_bits;

/* UARTn_MDR3_REG */
typedef struct {
 	__REG8  DISABLE_CIR_RX_DEMOD	: 1;
 	__REG8  											: 7;
} __uart_mdr3_reg_bits;

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

/* OTG_REVISION */
typedef struct {
 	__REG32 OTG_REVISION  						: 8;
 	__REG32 													:24;
} __otg_revision_bits;

/* OTG_SYSCONFIG */
typedef struct {
 	__REG32 AUTOIDLE      						: 1;
 	__REG32 SOFTRESET      						: 1;
 	__REG32 ENABLEWAKEUP   						: 1;
 	__REG32 SIDLEMODE      						: 2;
 	__REG32               						: 7;
 	__REG32 MIDLEMODE      						: 2;
 	__REG32 													:18;
} __otg_sysconfig_bits;

/* OTG_SYSSTATUS */
typedef struct {
 	__REG32 RESETDONE      						: 1;
 	__REG32 													:31;
} __otg_sysstatus_bits;

/* OTG_INTERFSEL */
typedef struct {
 	__REG32 PHYSEL        						: 2;
 	__REG32 													:30;
} __otg_interfsel_bits;

/* OTG_SIMENABLE */
typedef struct {
 	__REG32 TM1           						: 1;
 	__REG32 													:31;
} __otg_simenable_bits;

/* OTG_FORCESTDBY */
typedef struct {
 	__REG32 ENABLEFORCE    						: 1;
 	__REG32 													:31;
} __otg_forcestdby_bits;

/* OTG_BIGENDIAN */
typedef struct {
 	__REG32 BIG_ENDIAN    						: 1;
 	__REG32 													:31;
} __otg_bigendian_bits;

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
 	__REG32 AUTOPPD_ON_OVERCUR_EN	: 1;
 	__REG32 ENA_INCR4							: 1;
 	__REG32 ENA_INCR8							: 1;
 	__REG32 ENA_INCR16						: 1;
 	__REG32 ENA_INCR_ALIGN				: 1;
 	__REG32 											: 2;
 	__REG32 P1_CONNECT_STATUS			: 1;
 	__REG32 P2_CONNECT_STATUS			: 1;
 	__REG32 P3_CONNECT_STATUS			: 1;
 	__REG32 P2_ULPI_BYPASS   			: 1;
 	__REG32 P3_ULPI_BYPASS   			: 1;
 	__REG32 											:19;
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
 	__REG32 OCPM  								: 1;
 	__REG32 NOCP  								: 1;
 	__REG32 											:11;
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
 	__REG32     											:16;
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

/* CONTROL_IDCODE */
typedef struct {
 	__REG32 																: 1;
 	__REG32 TI_IDM													:11;
 	__REG32 HAWKEYE													:16;
 	__REG32 VERSION													: 4;
} __control_idcode_bits;

/* CONTROL_PRODUCTION_ID_2 */
typedef struct {
 	__REG32 DEVICE_TYPE											: 8;
 	__REG32 TI_IDM													:24;
} __control_production_id_2_bits;

/* SDTI_REVISION */
typedef struct {
 	__REG32 SDTI_REV0 					    : 4;
 	__REG32 SDTI_REV1 							: 4;
 	__REG32       									:24;
} __sdti_revision_bits;

/* SDTI_SYSCONFIG */
typedef struct {
 	__REG32 AUTOIDLE   					    : 1;
 	__REG32 SOFTRESET 							: 1;
 	__REG32       									:30;
} __sdti_sysconfig_bits;

/* SDTI_SYSSTATUS */
typedef struct {
 	__REG32 RESETDONE  					    : 1;
 	__REG32           					    : 7;
 	__REG32 FIFOEMPTY 							: 1;
 	__REG32       									:23;
} __sdti_sysstatus_bits;

/* SDTI_WINCTRL */
typedef struct {
 	__REG32 CPU1TRACEEN					    : 1;
 	__REG32 CPU2TRACEEN					    : 1;
 	__REG32 DEBUGGERTRACEEN 		    : 1;
 	__REG32           					    :25;
 	__REG32 CURRENTOWNER						: 1;
 	__REG32 DEBUGGEROVERRIDE				: 1;
 	__REG32 OWNERSHIP       				: 2;
} __sdti_winctrl_bits;

/* SDTI_SCONFIG */
typedef struct {
 	__REG32 SDTISCLKRATE  			    : 4;
 	__REG32 SINGLEEDGE					    : 1;
 	__REG32 TXDSIZE          		    : 2;
 	__REG32           					    :25;
} __sdti_sconfig_bits;

/* SDTI_TESTCTRL */
typedef struct {
 	__REG32 TESTMODE      			    : 1;
 	__REG32 TESTPATTERNSEL			    : 2;
 	__REG32 SIMPLEPATSEL     		    : 1;
 	__REG32           					    :28;
} __sdti_testctrl_bits;

/* INT_MODE_CTRL_REG */
typedef struct {
 	__REG32 ITM           			    : 1;
 	__REG32           					    :31;
} __int_mode_ctrl_reg_bits;

/* INT_OUTPUT_REG */
typedef struct {
 	__REG32 OUTBITSELECT   			    :12;
 	__REG32 INTEGEN        			    : 1;
 	__REG32               			    : 3;
 	__REG32 NUMOUTPUTS    			    :12;
 	__REG32           					    : 4;
} __int_output_reg_bits;

/* INT_INPUT_REG */
typedef struct {
 	__REG32 INBITSELECT   			    :12;
 	__REG32 VALUE         			    : 1;
 	__REG32               			    : 3;
 	__REG32 NUMINPUTS     			    :12;
 	__REG32           					    : 4;
} __int_input_reg_bits;

/* CLAIM_TAG_SET_REG */
typedef struct {
 	__REG32 CLAIMTAGSIZE_SET		    : 1;
 	__REG32           					    :31;
} __claim_tag_set_reg_bits;

/* CLAIM_TAG_CLEAR_REG */
typedef struct {
 	__REG32 CLAINTAGVALUE_CLEAR     : 1;
 	__REG32           					    :31;
} __claim_tag_clear_reg_bits;

/* LOCK_STATUS_REG */
typedef struct {
 	__REG32 LOCKIMPLEMENTED         : 1;
 	__REG32 LOCKSTATUS              : 1;
 	__REG32 EIGTBITLOCK             : 1;
 	__REG32           					    :29;
} __lock_status_reg_bits;

/* AUTHENTICATION_STATUS */
typedef struct {
 	__REG32 NONSECURE_INVASIVE_DEBUGSTATUS      : 2;
 	__REG32 NONSECURE_NONINVASIVE_DEBUGSTATUS   : 2;
 	__REG32 SECURE_INVASIVE_DEBUGSTATUS         : 2;
 	__REG32 SECURE_NONINVASIVE_DEBUGSTATUS      : 2;
 	__REG32                       					    :24;
} __authentication_status_bits;

/* DEVICE_ID */
typedef struct {
 	__REG32 FIFODEPTH               : 8;
 	__REG32 TIMESTAMP               : 1;
 	__REG32            					    :23;
} __device_id_bits;

/* DEVICE_TYPE_REG */
typedef struct {
 	__REG32 DEVICETYPE              : 8;
 	__REG32            					    :24;
} __device_type_reg_bits;

/* PERIPHERAL_ID4 */
typedef struct {
 	__REG32 JEP106CONTCODE          : 4;
 	__REG32 FOURKBCOUNT             : 4;
 	__REG32            					    :24;
} __peripheral_id4_bits;

/* PERIPHERAL_ID0 */
typedef struct {
 	__REG32 PARTNUMBER0             : 8;
 	__REG32            					    :24;
} __peripheral_id0_bits;

/* PERIPHERAL_ID1 */
typedef struct {
 	__REG32 PARTNUMBER1             : 4;
 	__REG32 JEP106IDCODE            : 4;
 	__REG32            					    :24;
} __peripheral_id1_bits;

/* PERIPHERAL_ID2 */
typedef struct {
 	__REG32 JEP106IDCODE            : 3;
 	__REG32 JEDEC                   : 1;
 	__REG32 REVISION                : 4;
 	__REG32            					    :24;
} __peripheral_id2_bits;

/* PERIPHERAL_ID3 */
typedef struct {
 	__REG32 CUSTOMMODIFIED          : 4;
 	__REG32 REVAND                  : 4;
 	__REG32            					    :24;
} __peripheral_id3_bits;

/* COMPONENT_ID0 */
typedef struct {
 	__REG32 COMPONENTID0            : 8;
 	__REG32            					    :24;
} __component_id0_bits;

/* COMPONENT_ID1 */
typedef struct {
 	__REG32 COMPONENTID1            : 8;
 	__REG32            					    :24;
} __component_id1_bits;

/* COMPONENT_ID2 */
typedef struct {
 	__REG32 COMPONENTID2            : 8;
 	__REG32            					    :24;
} __component_id2_bits;

/* COMPONENT_ID3 */
typedef struct {
 	__REG32 COMPONENTID3            : 8;
 	__REG32            					    :24;
} __component_id3_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */

/* Common declarations  ****************************************************/
/***************************************************************************
 **
 ** CM
 **
 ***************************************************************************/
__IO_REG32_BIT(CM_FCLKEN_IVA2,        0x48004000,__READ_WRITE ,__cm_fclken_iva2_bits);
__IO_REG32_BIT(CM_CLKEN_PLL_IVA2,     0x48004004,__READ_WRITE ,__cm_clken_pll_iva2_bits);
__IO_REG32_BIT(CM_IDLEST_IVA2,        0x48004020,__READ       ,__cm_idlest_iva2_bits);
__IO_REG32_BIT(CM_IDLEST_PLL_IVA2,    0x48004024,__READ       ,__cm_idlest_pll_iva2_bits);
__IO_REG32_BIT(CM_AUTOIDLE_PLL_IVA2,  0x48004034,__READ_WRITE ,__cm_autoidle_pll_iva2_bits);
__IO_REG32_BIT(CM_CLKSEL1_PLL_IVA2,   0x48004040,__READ_WRITE ,__cm_clksel1_pll_iva2_bits);
__IO_REG32_BIT(CM_CLKSEL2_PLL_IVA2,   0x48004044,__READ_WRITE ,__cm_clksel2_pll_iva2_bits);
__IO_REG32_BIT(CM_CLKSTCTRL_IVA2,     0x48004048,__READ_WRITE ,__cm_clkstctrl_iva2_bits);
__IO_REG32_BIT(CM_CLKSTST_IVA2,       0x4800404C,__READ       ,__cm_clkstst_iva2_bits);
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
__IO_REG32_BIT(CM_CLKSTST_CORE,     	0x48004A4C,__READ       ,__cm_clkstst_core_bits);
__IO_REG32_BIT(CM_FCLKEN_SGX,     		0x48004B00,__READ_WRITE ,__cm_fclken_sgx_bits);
__IO_REG32_BIT(CM_ICLKEN_SGX,     		0x48004B10,__READ_WRITE ,__cm_iclken_sgx_bits);
__IO_REG32_BIT(CM_IDLEST_SGX,     		0x48004B20,__READ				,__cm_idlest_sgx_bits);
__IO_REG32_BIT(CM_CLKSEL_SGX,     		0x48004B40,__READ_WRITE ,__cm_clksel_sgx_bits);
__IO_REG32_BIT(CM_SLEEPDEP_SGX,     	0x48004B44,__READ_WRITE ,__cm_sleepdep_sgx_bits);
__IO_REG32_BIT(CM_CLKSTCTRL_SGX,     	0x48004B48,__READ_WRITE ,__cm_clkstctrl_sgx_bits);
__IO_REG32_BIT(CM_CLKSTST_SGX,     		0x48004B4C,__READ       ,__cm_clkstst_sgx_bits);
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
__IO_REG32_BIT(CM_FCLKEN_CAM,     	  0x48004F00,__READ_WRITE ,__cm_fclken_cam_bits);
__IO_REG32_BIT(CM_ICLKEN_CAM,     	  0x48004F10,__READ_WRITE ,__cm_iclken_cam_bits);
__IO_REG32_BIT(CM_IDLEST_CAM,     	  0x48004F20,__READ       ,__cm_idlest_cam_bits);
__IO_REG32_BIT(CM_AUTOIDLE_CAM,       0x48004F30,__READ_WRITE ,__cm_autoidle_cam_bits);
__IO_REG32_BIT(CM_CLKSEL_CAM,     	  0x48004F40,__READ_WRITE ,__cm_clksel_cam_bits);
__IO_REG32_BIT(CM_SLEEPDEP_CAM,       0x48004F44,__READ_WRITE ,__cm_sleepdep_cam_bits);
__IO_REG32_BIT(CM_CLKSTCTRL_CAM,      0x48004F48,__READ_WRITE ,__cm_clkstctrl_cam_bits);
__IO_REG32_BIT(CM_CLKSTST_CAM,     	  0x48004F4C,__READ       ,__cm_clkstst_cam_bits);
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
__IO_REG32_BIT(RM_RSTCTRL_IVA2,       0x48306050,__READ_WRITE ,__rm_rstctrl_iva2_bits);
__IO_REG32_BIT(RM_RSTST_IVA2,         0x48306058,__READ_WRITE ,__rm_rstst_iva2_bits);
__IO_REG32_BIT(PM_WKDEP_IVA2,         0x483060C8,__READ_WRITE ,__pm_wkdep_iva2_bits);
__IO_REG32_BIT(PM_PWSTCTRL_IVA2,      0x483060E0,__READ_WRITE ,__pm_pwstctrl_iva2_bits);
__IO_REG32_BIT(PM_PWSTST_IVA2,        0x483060E4,__READ       ,__pm_pwstst_iva2_bits);
__IO_REG32_BIT(PM_PREPWSTST_IVA2,     0x483060E8,__READ_WRITE ,__pm_prepwstst_iva2_bits);
__IO_REG32_BIT(PRM_IRQSTATUS_IVA2,    0x483060F8,__READ_WRITE ,__prm_irqstatus_iva2_bits);
__IO_REG32_BIT(PRM_IRQENABLE_IVA2,    0x483060FC,__READ_WRITE ,__prm_irqenable_iva2_bits);
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
__IO_REG32_BIT(PM_IVA2GRPSEL1_CORE,   0x48306AA8,__READ_WRITE ,__pm_iva2grpsel1_core_bits);
__IO_REG32_BIT(PM_WKST1_CORE,     		0x48306AB0,__READ_WRITE ,__pm_wkst1_core_bits);
__IO_REG32_BIT(PM_WKST3_CORE,     		0x48306AB8,__READ_WRITE ,__pm_wkst3_core_bits);
__IO_REG32_BIT(PM_PWSTCTRL_CORE,     	0x48306AE0,__READ_WRITE ,__pm_pwstctrl_core_bits);
__IO_REG32_BIT(PM_PWSTST_CORE,     		0x48306AE4,__READ				,__pm_pwstst_core_bits);
__IO_REG32_BIT(PM_PREPWSTST_CORE,    	0x48306AE8,__READ_WRITE ,__pm_prepwstst_core_bits);
__IO_REG32_BIT(PM_WKEN3_CORE,     		0x48306AF0,__READ_WRITE ,__pm_wken3_core_bits);
__IO_REG32_BIT(PM_IVA2GRPSEL3_CORE,   0x48306AF4,__READ_WRITE ,__pm_iva2grpsel3_core_bits);
__IO_REG32_BIT(PM_MPUGRPSEL3_CORE,    0x48306AF8,__READ_WRITE ,__pm_mpugrpsel3_core_bits);
__IO_REG32_BIT(RM_RSTST_SGX,    			0x48306B58,__READ_WRITE ,__rm_rstst_sgx_bits);
__IO_REG32_BIT(PM_WKDEP_SGX,    			0x48306BC8,__READ_WRITE ,__pm_wkdep_sgx_bits);
__IO_REG32_BIT(PM_PWSTCTRL_SGX,    		0x48306BE0,__READ_WRITE ,__pm_pwstctrl_sgx_bits);
__IO_REG32_BIT(PM_PWSTST_SGX,    			0x48306BE4,__READ				,__pm_pwstst_sgx_bits);
__IO_REG32_BIT(PM_PREPWSTCTRL_SGX,    0x48306BE8,__READ_WRITE ,__pm_prepwstctrl_sgx_bits);
__IO_REG32_BIT(PM_WKEN_WKUP,    			0x48306CA0,__READ_WRITE ,__pm_wken_wkup_bits);
__IO_REG32_BIT(PM_MPUGRPSEL_WKUP,    	0x48306CA4,__READ_WRITE ,__pm_mpugrpsel_wkup_bits);
__IO_REG32_BIT(PM_IVA2GRPSEL_WKUP,   	0x48306CA8,__READ_WRITE ,__pm_iva2grpsel_wkup_bits);
__IO_REG32_BIT(PM_WKST_WKUP,    			0x48306CB0,__READ_WRITE ,__pm_wkst_wkup_bits);
__IO_REG32_BIT(PRM_CLKSEL,    				0x48306D40,__READ_WRITE ,__prm_clksel_bits);
__IO_REG32_BIT(PRM_CLKOUT_CTRL,    		0x48306D70,__READ_WRITE ,__prm_clkout_ctrl_bits);
__IO_REG32_BIT(RM_RSTST_DSS,    			0x48306E58,__READ_WRITE ,__rm_rstst_dss_bits);
__IO_REG32_BIT(PM_WKEN_DSS,    				0x48306EA0,__READ_WRITE ,__pm_wken_dss_bits);
__IO_REG32_BIT(PM_WKDEP_DSS,    			0x48306EC8,__READ_WRITE ,__pm_wkdep_dss_bits);
__IO_REG32_BIT(PM_PWSTCTRL_DSS,    		0x48306EE0,__READ_WRITE ,__pm_pwstctrl_dss_bits);
__IO_REG32_BIT(PM_PWSTST_DSS,    			0x48306EE4,__READ				,__pm_pwstst_dss_bits);
__IO_REG32_BIT(PM_PREPWSTST_DSS,    	0x48306EE8,__READ_WRITE ,__pm_prepwstst_dss_bits);
__IO_REG32_BIT(RM_RSTST_CAM,         	0x48306F58,__READ_WRITE ,__rm_rstst_cam_bits);
__IO_REG32_BIT(PM_WKDEP_CAM,         	0x48306FC8,__READ_WRITE ,__pm_wkdep_cam_bits);
__IO_REG32_BIT(PM_PWSTCTRL_CAM,       0x48306FE0,__READ_WRITE ,__pm_pwstctrl_cam_bits);
__IO_REG32_BIT(PM_PWSTST_CAM,        	0x48306FE4,__READ       ,__pm_pwstst_cam_bits);
__IO_REG32_BIT(PM_PREPWSTST_CAM,     	0x48306FE8,__READ_WRITE ,__pm_prepwstst_cam_bits);
__IO_REG32_BIT(RM_RSTST_PER,    			0x48307058,__READ_WRITE ,__rm_rstst_per_bits);
__IO_REG32_BIT(PM_WKEN_PER,    				0x483070A0,__READ_WRITE ,__pm_wken_per_bits);
__IO_REG32_BIT(PM_MPUGRPSEL_PER,    	0x483070A4,__READ_WRITE ,__pm_mpugrpsel_per_bits);
__IO_REG32_BIT(PM_IVA2GRPSEL_PER,    	0x483070A8,__READ_WRITE ,__pm_iva2grpsel_per_bits);
__IO_REG32_BIT(PM_WKST_PER,    				0x483070B0,__READ_WRITE ,__pm_wkst_per_bits);
__IO_REG32_BIT(PM_WKDEP_PER,    			0x483070C8,__READ_WRITE ,__pm_wkdep_per_bits);
__IO_REG32_BIT(PM_PWSTCTRL_PER,    		0x483070E0,__READ_WRITE ,__pm_pwstctrl_per_bits);
__IO_REG32_BIT(PM_PWSTST_PER,    			0x483070E4,__READ				,__pm_pwstst_per_bits);
__IO_REG32_BIT(PM_PREPWSTST_PER,    	0x483070E8,__READ_WRITE ,__pm_prepwstst_per_bits);
__IO_REG32_BIT(RM_RSTST_EMU,    			0x48307158,__READ_WRITE ,__rm_rstst_emu_bits);
__IO_REG32_BIT(PM_PWSTST_EMU,    			0x483071E4,__READ_WRITE ,__pm_pwstst_emu_bits);
__IO_REG32_BIT(PRM_VC_SMPS_SA,    		0x48307220,__READ_WRITE ,__prm_vc_smps_sa_bits);
__IO_REG32_BIT(PRM_VC_SMPS_VOL_RA, 		0x48307224,__READ_WRITE ,__prm_vc_smps_vol_ra_bits);
__IO_REG32_BIT(PRM_VC_SMPS_CMD_RA, 		0x48307228,__READ_WRITE ,__prm_vc_smps_cmd_ra_bits);
__IO_REG32_BIT(PRM_VC_CMD_VAL_0,   		0x4830722C,__READ_WRITE ,__prm_vc_cmd_val_bits);
__IO_REG32_BIT(PRM_VC_CMD_VAL_1,  		0x48307230,__READ_WRITE ,__prm_vc_cmd_val_bits);
__IO_REG32_BIT(PRM_VC_CH_CONF,    		0x48307234,__READ_WRITE ,__prm_vc_ch_conf_bits);
__IO_REG32_BIT(PRM_VC_I2C_CFG,    		0x48307238,__READ_WRITE ,__prm_vc_i2c_cfg_bits);
__IO_REG32_BIT(PRM_VC_BYPASS_VAL, 		0x4830723C,__READ_WRITE ,__prm_vc_bypass_val_bits);
__IO_REG32_BIT(PRM_RSTCTRL,    				0x48307250,__READ_WRITE ,__prm_rstctrl_bits);
__IO_REG32_BIT(PRM_RSTTIME,    				0x48307254,__READ_WRITE ,__prm_rsttime_bits);
__IO_REG32_BIT(PRM_RSTST,    					0x48307258,__READ_WRITE ,__prm_rstst_bits);
__IO_REG32_BIT(PRM_VOLTCTRL, 					0x48307260,__READ_WRITE ,__prm_voltctrl_bits);
__IO_REG32_BIT(PRM_SRAM_PCHARGE, 	 	  0x48307264,__READ_WRITE ,__prm_sram_pcharge_bits);
__IO_REG32_BIT(PRM_CLKSRC_CTRL,    		0x48307270,__READ_WRITE ,__prm_clksrc_ctrl_bits);
__IO_REG32_BIT(PRM_OBS,    						0x48307280,__READ				,__prm_obs_bits);
__IO_REG32_BIT(PRM_VOLTSETUP1,   			0x48307290,__READ_WRITE ,__prm_voltsetup1_bits);
__IO_REG32_BIT(PRM_VOLTOFFSET,   			0x48307294,__READ_WRITE ,__prm_voltoffset_bits);
__IO_REG32_BIT(PRM_CLKSETUP,    			0x48307298,__READ_WRITE ,__prm_clksetup_bits);
__IO_REG32_BIT(PRM_POLCTRL,    				0x4830729C,__READ_WRITE ,__prm_polctrl_bits);
__IO_REG32_BIT(PRM_VOLTSETUP2, 				0x483072A0,__READ_WRITE ,__prm_voltsetup2_bits);
__IO_REG32_BIT(PRM_VP1_CONFIG, 				0x483072B0,__READ_WRITE ,__prm_vp_config_bits);
__IO_REG32_BIT(PRM_VP1_VSTEPMIN, 	  	0x483072B4,__READ_WRITE ,__prm_vp_vstepmin_bits);
__IO_REG32_BIT(PRM_VP1_VSTEPMAX, 			0x483072B8,__READ_WRITE ,__prm_vp_vstepmax_bits);
__IO_REG32_BIT(PRM_VP1_VLIMITTO, 		  0x483072BC,__READ_WRITE ,__prm_vp_vlimitto_bits);
__IO_REG32_BIT(PRM_VP1_VOLTAGE, 			0x483072C0,__READ       ,__prm_vp_voltage_bits);
__IO_REG32_BIT(PRM_VP1_STATUS, 				0x483072C4,__READ       ,__prm_vp_status_bits);
__IO_REG32_BIT(PRM_VP2_CONFIG, 				0x483072D0,__READ_WRITE ,__prm_vp_config_bits);
__IO_REG32_BIT(PRM_VP2_VSTEPMIN, 			0x483072D4,__READ_WRITE ,__prm_vp_vstepmin_bits);
__IO_REG32_BIT(PRM_VP2_VSTEPMAX, 			0x483072D8,__READ_WRITE ,__prm_vp_vstepmax_bits);
__IO_REG32_BIT(PRM_VP2_VLIMITTO, 			0x483072DC,__READ_WRITE ,__prm_vp_vlimitto_bits);
__IO_REG32_BIT(PRM_VP2_VOLTAGE, 			0x483072E0,__READ       ,__prm_vp_voltage_bits);
__IO_REG32_BIT(PRM_VP2_STATUS, 				0x483072E4,__READ       ,__prm_vp_status_bits);
__IO_REG32_BIT(PRM_LDO_ABB_SETUP, 		0x483072F0,__READ_WRITE ,__prm_ldo_abb_setup_bits);
__IO_REG32_BIT(PRM_LDO_ABB_CTRL, 			0x483072F4,__READ_WRITE ,__prm_ldo_abb_ctrl_bits);
__IO_REG32_BIT(RM_RSTST_NEON,    			0x48307358,__READ_WRITE ,__rm_rstst_neon_bits);
__IO_REG32_BIT(PM_WKDEP_NEON,    			0x483073C8,__READ_WRITE ,__pm_wkdep_neon_bits);
__IO_REG32_BIT(PM_PWSTCTRL_NEON,    	0x483073E0,__READ_WRITE ,__pm_pwstctrl_neon_bits);
__IO_REG32_BIT(PM_PWSTST_NEON,    		0x483073E4,__READ				,__pm_pwstst_neon_bits);
__IO_REG32_BIT(PM_PREPWSTST_NEON,    	0x483073E8,__READ_WRITE ,__pm_prepwstst_neon_bits);
__IO_REG32_BIT(RM_RSTST_USBHOST,    	0x48307458,__READ_WRITE ,__rm_rstst_usbhost_bits);
__IO_REG32_BIT(PM_WKEN_USBHOST,    		0x483074A0,__READ_WRITE ,__pm_wken_usbhost_bits);
__IO_REG32_BIT(PM_MPUGRPSEL_USBHOST,  0x483074A4,__READ_WRITE ,__pm_mpugrpsel_usbhost_bits);
__IO_REG32_BIT(PM_IVA2GRPSEL_USBHOST, 0x483074A8,__READ_WRITE ,__pm_iva2grpsel_usbhost_bits);
__IO_REG32_BIT(PM_WKST_USBHOST,    		0x483074B0,__READ_WRITE ,__pm_wkst_usbhost_bits);
__IO_REG32_BIT(PM_WKDEP_USBHOST,    	0x483074C8,__READ_WRITE ,__pm_wkdep_usbhost_bits);
__IO_REG32_BIT(PM_PWSTCTRL_USBHOST,   0x483074E0,__READ_WRITE ,__pm_pwstctrl_usbhost_bits);
__IO_REG32_BIT(PM_PWSTST_USBHOST,    	0x483074E4,__READ				,__pm_pwstst_usbhost_bits);
__IO_REG32_BIT(PM_PREPWSTST_USBHOST,  0x483074E8,__READ_WRITE ,__pm_prepwstst_usbhost_bits);

/***************************************************************************
 **
 ** SR1
 **
 ***************************************************************************/
__IO_REG32_BIT(SR1_SRCONFIG,          0x480C9000,__READ_WRITE ,__sr_srconfig_bits);
__IO_REG32_BIT(SR1_SRSTATUS,          0x480C9004,__READ       ,__sr_srstatus_bits);
__IO_REG32_BIT(SR1_SENVAL,            0x480C9008,__READ       ,__sr_senval_bits);
__IO_REG32_BIT(SR1_SENMIN,            0x480C900C,__READ       ,__sr_senmin_bits);
__IO_REG32_BIT(SR1_SENMAX,            0x480C9010,__READ       ,__sr_senmax_bits);
__IO_REG32_BIT(SR1_SENAVG,            0x480C9014,__READ       ,__sr_senavg_bits);
__IO_REG32_BIT(SR1_AVGWEIGHT,         0x480C9018,__READ_WRITE ,__sr_avgweight_bits);
__IO_REG32_BIT(SR1_NVALUERECIPROCAL,  0x480C901C,__READ_WRITE ,__sr_nvaluereciprocal_bits);
__IO_REG32_BIT(SR1_IRQSTATUS_RAW,     0x480C9024,__READ_WRITE ,__sr_irqstatus_raw_bits);
__IO_REG32_BIT(SR1_IRQSTATUS,         0x480C9028,__READ_WRITE ,__sr_irqstatus_bits);
__IO_REG32_BIT(SR1_IRQENABLE_SET,     0x480C902C,__READ_WRITE ,__sr_irqenable_set_bits);
__IO_REG32_BIT(SR1_IRQENABLE_CLR,     0x480C9030,__READ_WRITE ,__sr_irqenable_clr_bits);
__IO_REG32_BIT(SR1_SENERROR_REG,      0x480C9034,__READ       ,__sr_senerror_reg_bits);
__IO_REG32_BIT(SR1_ERRCONFIG,         0x480C9038,__READ_WRITE ,__sr_errconfig_bits);

/***************************************************************************
 **
 ** SR2
 **
 ***************************************************************************/
__IO_REG32_BIT(SR2_SRCONFIG,          0x480CB000,__READ_WRITE ,__sr_srconfig_bits);
__IO_REG32_BIT(SR2_SRSTATUS,          0x480CB004,__READ       ,__sr_srstatus_bits);
__IO_REG32_BIT(SR2_SENVAL,            0x480CB008,__READ       ,__sr_senval_bits);
__IO_REG32_BIT(SR2_SENMIN,            0x480CB00C,__READ       ,__sr_senmin_bits);
__IO_REG32_BIT(SR2_SENMAX,            0x480CB010,__READ       ,__sr_senmax_bits);
__IO_REG32_BIT(SR2_SENAVG,            0x480CB014,__READ       ,__sr_senavg_bits);
__IO_REG32_BIT(SR2_AVGWEIGHT,         0x480CB018,__READ_WRITE ,__sr_avgweight_bits);
__IO_REG32_BIT(SR2_NVALUERECIPROCAL,  0x480CB01C,__READ_WRITE ,__sr_nvaluereciprocal_bits);
__IO_REG32_BIT(SR2_IRQSTATUS_RAW,     0x480CB024,__READ_WRITE ,__sr_irqstatus_raw_bits);
__IO_REG32_BIT(SR2_IRQSTATUS,         0x480CB028,__READ_WRITE ,__sr_irqstatus_bits);
__IO_REG32_BIT(SR2_IRQENABLE_SET,     0x480CB02C,__READ_WRITE ,__sr_irqenable_set_bits);
__IO_REG32_BIT(SR2_IRQENABLE_CLR,     0x480CB030,__READ_WRITE ,__sr_irqenable_clr_bits);
__IO_REG32_BIT(SR2_SENERROR_REG,      0x480CB034,__READ       ,__sr_senerror_reg_bits);
__IO_REG32_BIT(SR2_ERRCONFIG,         0x480CB038,__READ_WRITE ,__sr_errconfig_bits);

/***************************************************************************
 **
 ** Camera ISP
 **
 ***************************************************************************/
__IO_REG32_BIT(ISP_REVISION,          0x480BC000,__READ       ,__isp_revision_bits);
__IO_REG32_BIT(ISP_SYSCONFIG,         0x480BC004,__READ_WRITE ,__isp_sysconfig_bits);
__IO_REG32_BIT(ISP_SYSSTATUS,         0x480BC008,__READ       ,__isp_sysstatus_bits);
__IO_REG32_BIT(ISP_IRQ0ENABLE,        0x480BC00C,__READ_WRITE ,__isp_irqxenable_bits);
__IO_REG32_BIT(ISP_IRQ0STATUS,        0x480BC010,__READ_WRITE ,__isp_irqxenable_bits);
__IO_REG32_BIT(ISP_IRQ1ENABLE,        0x480BC014,__READ_WRITE ,__isp_irqxenable_bits);
__IO_REG32_BIT(ISP_IRQ1STATUS,        0x480BC018,__READ_WRITE ,__isp_irqxenable_bits);
__IO_REG32_BIT(TCTRL_GRESET_LENGTH,   0x480BC030,__READ_WRITE ,__tctrl_greset_length_bits);
__IO_REG32_BIT(TCTRL_PSTRB_REPLAY,    0x480BC034,__READ_WRITE ,__tctrl_pstrb_replay_bits);
__IO_REG32_BIT(ISP_CTRL,              0x480BC040,__READ_WRITE ,__isp_ctrl_bits);
__IO_REG32_BIT(TCTRL_CTRL,            0x480BC050,__READ_WRITE ,__tctrl_ctrl_bits);
__IO_REG32_BIT(TCTRL_FRAME,           0x480BC054,__READ_WRITE ,__tctrl_frame_bits);
__IO_REG32_BIT(TCTRL_PSTRB_DELAY,     0x480BC058,__READ_WRITE ,__tctrl_pstrb_delay_bits);
__IO_REG32_BIT(TCTRL_STRB_DELAY,      0x480BC05C,__READ_WRITE ,__tctrl_pstrb_delay_bits);
__IO_REG32_BIT(TCTRL_SHUT_DELAY,      0x480BC060,__READ_WRITE ,__tctrl_pstrb_delay_bits);
__IO_REG32_BIT(TCTRL_PSTRB_LENGTH,    0x480BC064,__READ_WRITE ,__tctrl_pstrb_length_bits);
__IO_REG32_BIT(TCTRL_STRB_LENGTH,     0x480BC068,__READ_WRITE ,__tctrl_pstrb_length_bits);
__IO_REG32_BIT(TCTRL_SHUT_LENGTH,     0x480BC06C,__READ_WRITE ,__tctrl_pstrb_length_bits);

/***************************************************************************
 **
 ** Camera CBUFF
 **
 ***************************************************************************/
__IO_REG32_BIT(CBUFF_REVISION,        0x480BC100,__READ       ,__cbuff_revision_bits);
__IO_REG32_BIT(CBUFF_IRQSTATUS,       0x480BC118,__READ_WRITE ,__cbuff_irqstatus_bits);
__IO_REG32_BIT(CBUFF_IRQENABLE,       0x480BC11C,__READ_WRITE ,__cbuff_irqstatus_bits);
__IO_REG32_BIT(CBUFF0_CTRL,           0x480BC120,__READ_WRITE ,__cbuffx_ctrl_bits);
__IO_REG32_BIT(CBUFF1_CTRL,           0x480BC124,__READ_WRITE ,__cbuffx_ctrl_bits);
__IO_REG32_BIT(CBUFF0_STATUS,         0x480BC130,__READ       ,__cbuffx_status_bits);
__IO_REG32_BIT(CBUFF1_STATUS,         0x480BC134,__READ       ,__cbuffx_status_bits);
__IO_REG32(    CBUFF0_START,          0x480BC140,__READ_WRITE );
__IO_REG32(    CBUFF1_START,          0x480BC144,__READ_WRITE );
__IO_REG32(    CBUFF0_END,            0x480BC150,__READ_WRITE );
__IO_REG32(    CBUFF1_END,            0x480BC154,__READ_WRITE );
__IO_REG32_BIT(CBUFF0_WINDOWSIZE,     0x480BC160,__READ_WRITE ,__cbuffx_windowsize_bits);
__IO_REG32_BIT(CBUFF1_WINDOWSIZE,     0x480BC164,__READ_WRITE ,__cbuffx_windowsize_bits);
__IO_REG32_BIT(CBUFF0_THRESHOLD,      0x480BC170,__READ_WRITE ,__cbuffx_threshold_bits);
__IO_REG32_BIT(CBUFF1_THRESHOLD,      0x480BC174,__READ_WRITE ,__cbuffx_threshold_bits);
__IO_REG32_BIT(CBUFF_VRFB_CTRL,       0x480BC1C0,__READ_WRITE ,__cbuff_vrfb_ctrl_bits);

/***************************************************************************
 **
 ** Camera CCP2
 **
 ***************************************************************************/
__IO_REG32_BIT(CCP2_REVISION,           0x480BC400,__READ       ,__ccp2_revision_bits);
__IO_REG32_BIT(CCP2_SYSCONFIG,          0x480BC404,__READ_WRITE ,__ccp2_sysconfig_bits);
__IO_REG32_BIT(CCP2_SYSSTATUS,          0x480BC408,__READ       ,__ccp2_sysstatus_bits);
__IO_REG32_BIT(CCP2_LC01_IRQENABLE,     0x480BC40C,__READ_WRITE ,__ccp2_lc01_irqenable_bits);
__IO_REG32_BIT(CCP2_LC01_IRQSTATUS,     0x480BC410,__READ_WRITE ,__ccp2_lc01_irqenable_bits);
__IO_REG32_BIT(CCP2_LC23_IRQENABLE,     0x480BC414,__READ_WRITE ,__ccp2_lc23_irqenable_bits);
__IO_REG32_BIT(CCP2_LC23_IRQSTATUS,     0x480BC418,__READ_WRITE ,__ccp2_lc23_irqenable_bits);
__IO_REG32_BIT(CCP2_LCM_IRQENABLE,      0x480BC42C,__READ_WRITE ,__ccp2_lcm_irqenable_bits);
__IO_REG32_BIT(CCP2_LCM_IRQSTATUS,      0x480BC430,__READ_WRITE ,__ccp2_lcm_irqenable_bits);
__IO_REG32_BIT(CCP2_CTRL,               0x480BC440,__READ_WRITE ,__ccp2_ctrl_bits);
__IO_REG32(    CCP2_DBG,                0x480BC444,__WRITE      );
__IO_REG32_BIT(CCP2_GNQ,                0x480BC448,__READ       ,__ccp2_gnq_bits);
__IO_REG32_BIT(CCP2_CTRL1,              0x480BC44C,__READ_WRITE ,__ccp2_ctrl1_bits);
__IO_REG32_BIT(CCP2_LC0_CTRL,           0x480BC450,__READ_WRITE ,__ccp2_lcx_ctrl_bits);
__IO_REG32_BIT(CCP2_LC0_CODE,           0x480BC454,__READ_WRITE ,__ccp2_lcx_code_bits);
__IO_REG32_BIT(CCP2_LC0_STAT_START,     0x480BC458,__READ_WRITE ,__ccp2_lcx_stat_start_bits);
__IO_REG32_BIT(CCP2_LC0_STAT_SIZE,      0x480BC45C,__READ_WRITE ,__ccp2_lcx_stat_size_bits);
__IO_REG32(    CCP2_LC0_SOF_ADDR,       0x480BC460,__READ_WRITE );
__IO_REG32(    CCP2_LC0_EOF_ADDR,       0x480BC464,__READ_WRITE );
__IO_REG32_BIT(CCP2_LC0_DAT_START,      0x480BC468,__READ_WRITE ,__ccp2_lcx_dat_start_bits);
__IO_REG32_BIT(CCP2_LC0_DAT_SIZE,       0x480BC46C,__READ_WRITE ,__ccp2_lcx_dat_size_bits);
__IO_REG32(    CCP2_LC0_DAT_PING_ADDR,  0x480BC470,__READ_WRITE );
__IO_REG32(    CCP2_LC0_DAT_PONG_ADDR,  0x480BC474,__READ_WRITE );
__IO_REG32(    CCP2_LC0_DAT_OFST,       0x480BC478,__READ_WRITE );
__IO_REG32_BIT(CCP2_LC1_CODE,           0x480BC480,__READ_WRITE ,__ccp2_lcx_code_bits);
__IO_REG32_BIT(CCP2_LC1_CTRL,           0x480BC484,__READ_WRITE ,__ccp2_lcx_ctrl_bits);
__IO_REG32_BIT(CCP2_LC1_STAT_START,     0x480BC488,__READ_WRITE ,__ccp2_lcx_stat_start_bits);
__IO_REG32_BIT(CCP2_LC1_STAT_SIZE,      0x480BC48C,__READ_WRITE ,__ccp2_lcx_stat_size_bits);
__IO_REG32(    CCP2_LC1_SOF_ADDR,       0x480BC490,__READ_WRITE );
__IO_REG32(    CCP2_LC1_EOF_ADDR,       0x480BC494,__READ_WRITE );
__IO_REG32_BIT(CCP2_LC1_DAT_START,      0x480BC498,__READ_WRITE ,__ccp2_lcx_dat_start_bits);
__IO_REG32_BIT(CCP2_LC1_DAT_SIZE,       0x480BC49C,__READ_WRITE ,__ccp2_lcx_dat_size_bits);
__IO_REG32(    CCP2_LC1_DAT_PING_ADDR,  0x480BC4A0,__READ_WRITE );
__IO_REG32(    CCP2_LC1_DAT_PONG_ADDR,  0x480BC4A4,__READ_WRITE );
__IO_REG32(    CCP2_LC1_DAT_OFST,       0x480BC4A8,__READ_WRITE );
__IO_REG32_BIT(CCP2_LC2_CODE,           0x480BC4B0,__READ_WRITE ,__ccp2_lcx_code_bits);
__IO_REG32_BIT(CCP2_LC2_CTRL,           0x480BC4B4,__READ_WRITE ,__ccp2_lcx_ctrl_bits);
__IO_REG32_BIT(CCP2_LC2_STAT_START,     0x480BC4B8,__READ_WRITE ,__ccp2_lcx_stat_start_bits);
__IO_REG32_BIT(CCP2_LC2_STAT_SIZE,      0x480BC4BC,__READ_WRITE ,__ccp2_lcx_stat_size_bits);
__IO_REG32(    CCP2_LC2_SOF_ADDR,       0x480BC4C0,__READ_WRITE );
__IO_REG32(    CCP2_LC2_EOF_ADDR,       0x480BC4C4,__READ_WRITE );
__IO_REG32_BIT(CCP2_LC2_DAT_START,      0x480BC4C8,__READ_WRITE ,__ccp2_lcx_dat_start_bits);
__IO_REG32_BIT(CCP2_LC2_DAT_SIZE,       0x480BC4CC,__READ_WRITE ,__ccp2_lcx_dat_size_bits);
__IO_REG32(    CCP2_LC2_DAT_PING_ADDR,  0x480BC4D0,__READ_WRITE );
__IO_REG32(    CCP2_LC2_DAT_PONG_ADDR,  0x480BC4D4,__READ_WRITE );
__IO_REG32(    CCP2_LC2_DAT_OFST,       0x480BC4D8,__READ_WRITE );
__IO_REG32_BIT(CCP2_LC3_CODE,           0x480BC4E0,__READ_WRITE ,__ccp2_lcx_code_bits);
__IO_REG32_BIT(CCP2_LC3_CTRL,           0x480BC4E4,__READ_WRITE ,__ccp2_lcx_ctrl_bits);
__IO_REG32_BIT(CCP2_LC3_STAT_START,     0x480BC4E8,__READ_WRITE ,__ccp2_lcx_stat_start_bits);
__IO_REG32_BIT(CCP2_LC3_STAT_SIZE,      0x480BC4EC,__READ_WRITE ,__ccp2_lcx_stat_size_bits);
__IO_REG32(    CCP2_LC3_SOF_ADDR,       0x480BC4F0,__READ_WRITE );
__IO_REG32(    CCP2_LC3_EOF_ADDR,       0x480BC4F4,__READ_WRITE );
__IO_REG32_BIT(CCP2_LC3_DAT_START,      0x480BC4F8,__READ_WRITE ,__ccp2_lcx_dat_start_bits);
__IO_REG32_BIT(CCP2_LC3_DAT_SIZE,       0x480BC4FC,__READ_WRITE ,__ccp2_lcx_dat_size_bits);
__IO_REG32(    CCP2_LC3_DAT_PING_ADDR,  0x480BC500,__READ_WRITE );
__IO_REG32(    CCP2_LC3_DAT_PONG_ADDR,  0x480BC504,__READ_WRITE );
__IO_REG32(    CCP2_LC3_DAT_OFST,       0x480BC508,__READ_WRITE );
__IO_REG32_BIT(CCP2_LCM_CTRL,           0x480BC5D0,__READ_WRITE ,__ccp2_lcm_ctrl_bits);
__IO_REG32_BIT(CCP2_LCM_VSIZE,          0x480BC5D4,__READ_WRITE ,__ccp2_lcm_vsize_bits);
__IO_REG32_BIT(CCP2_LCM_HSIZE,          0x480BC5D8,__READ_WRITE ,__ccp2_lcm_hsize_bits);
__IO_REG32_BIT(CCP2_LCM_PREFETCH,       0x480BC5DC,__READ_WRITE ,__ccp2_lcm_prefetch_bits);
__IO_REG32(    CCP2_LCM_SRC_ADDR,       0x480BC5E0,__READ_WRITE );
__IO_REG32(    CCP2_LCM_SRC_OFST,       0x480BC5E4,__READ_WRITE );
__IO_REG32(    CCP2_LCM_DST_ADDR,       0x480BC5E8,__READ_WRITE );
__IO_REG32(    CCP2_LCM_DST_OFST,       0x480BC5EC,__READ_WRITE );

/***************************************************************************
 **
 ** Camera CCDC
 **
 ***************************************************************************/
__IO_REG32_BIT(CCDC_PID,                0x480BC600,__READ       ,__ccdc_pid_bits);
__IO_REG32_BIT(CCDC_PCR,                0x480BC604,__READ_WRITE ,__ccdc_pcr_bits);
__IO_REG32_BIT(CCDC_SYN_MODE,           0x480BC608,__READ_WRITE ,__ccdc_syn_mode_bits);
__IO_REG32_BIT(CCDC_HD_VD_WID,          0x480BC60C,__READ_WRITE ,__ccdc_hd_vd_wid_bits);
__IO_REG32_BIT(CCDC_PIX_LINES,          0x480BC610,__READ_WRITE ,__ccdc_pix_lines_bits);
__IO_REG32_BIT(CCDC_HORZ_INFO,          0x480BC614,__READ_WRITE ,__ccdc_horz_info_bits);
__IO_REG32_BIT(CCDC_VERT_START,         0x480BC618,__READ_WRITE ,__ccdc_vert_start_bits);
__IO_REG32_BIT(CCDC_VERT_LINES,         0x480BC61C,__READ_WRITE ,__ccdc_vert_lines_bits);
__IO_REG32_BIT(CCDC_CULLING,            0x480BC620,__READ_WRITE ,__ccdc_culling_bits);
__IO_REG32_BIT(CCDC_HSIZE_OFF,          0x480BC624,__READ_WRITE ,__ccdc_hsize_off_bits);
__IO_REG32_BIT(CCDC_SDOFST,             0x480BC628,__READ_WRITE ,__ccdc_sdofst_bits);
__IO_REG32(    CCDC_SDR_ADDR,           0x480BC62C,__READ_WRITE );
__IO_REG32_BIT(CCDC_CLAMP,              0x480BC630,__READ_WRITE ,__ccdc_clamp_bits);
__IO_REG32_BIT(CCDC_DCSUB,              0x480BC634,__READ_WRITE ,__ccdc_dcsub_bits);
__IO_REG32_BIT(CCDC_COLPTN,             0x480BC638,__READ_WRITE ,__ccdc_colptn_bits);
__IO_REG32_BIT(CCDC_BLKCMP,             0x480BC63C,__READ_WRITE ,__ccdc_blkcmp_bits);
__IO_REG32_BIT(CCDC_FPC,                0x480BC640,__READ_WRITE ,__ccdc_fpc_bits);
__IO_REG32(    CCDC_FPC_ADDR,           0x480BC644,__READ_WRITE );
__IO_REG32_BIT(CCDC_VDINT,              0x480BC648,__READ_WRITE ,__ccdc_vdint_bits);
__IO_REG32_BIT(CCDC_ALAW,               0x480BC64C,__READ_WRITE ,__ccdc_alaw_bits);
__IO_REG32_BIT(CCDC_REC656IF,           0x480BC650,__READ_WRITE ,__ccdc_rec656if_bits);
__IO_REG32_BIT(CCDC_CFG,                0x480BC654,__READ_WRITE ,__ccdc_cfg_bits);
__IO_REG32_BIT(CCDC_FMTCFG,             0x480BC658,__READ_WRITE ,__ccdc_fmtcfg_bits);
__IO_REG32_BIT(CCDC_FMT_HORZ,           0x480BC65C,__READ_WRITE ,__ccdc_fmt_horz_bits);
__IO_REG32_BIT(CCDC_FMT_VERT,           0x480BC660,__READ_WRITE ,__ccdc_fmt_vert_bits);
__IO_REG32_BIT(CCDC_FMT_ADDR_0,         0x480BC664,__READ_WRITE ,__ccdc_fmt_addr_x_bits);
__IO_REG32_BIT(CCDC_FMT_ADDR_1,         0x480BC668,__READ_WRITE ,__ccdc_fmt_addr_x_bits);
__IO_REG32_BIT(CCDC_FMT_ADDR_2,         0x480BC66C,__READ_WRITE ,__ccdc_fmt_addr_x_bits);
__IO_REG32_BIT(CCDC_FMT_ADDR_3,         0x480BC670,__READ_WRITE ,__ccdc_fmt_addr_x_bits);
__IO_REG32_BIT(CCDC_FMT_ADDR_4,         0x480BC674,__READ_WRITE ,__ccdc_fmt_addr_x_bits);
__IO_REG32_BIT(CCDC_FMT_ADDR_5,         0x480BC678,__READ_WRITE ,__ccdc_fmt_addr_x_bits);
__IO_REG32_BIT(CCDC_FMT_ADDR_6,         0x480BC67C,__READ_WRITE ,__ccdc_fmt_addr_x_bits);
__IO_REG32_BIT(CCDC_FMT_ADDR_7,         0x480BC680,__READ_WRITE ,__ccdc_fmt_addr_x_bits);
__IO_REG32_BIT(CCDC_PRGEVEN0,           0x480BC684,__READ_WRITE ,__ccdc_prgeven0_bits);
__IO_REG32_BIT(CCDC_PRGEVEN1,           0x480BC688,__READ_WRITE ,__ccdc_prgeven1_bits);
__IO_REG32_BIT(CCDC_PRGODD0,            0x480BC68C,__READ_WRITE ,__ccdc_prgodd0_bits);
__IO_REG32_BIT(CCDC_PRGODD1,            0x480BC690,__READ_WRITE ,__ccdc_prgodd1_bits);
__IO_REG32_BIT(CCDC_VP_OUT,             0x480BC694,__READ_WRITE ,__ccdc_vp_out_bits);
__IO_REG32_BIT(CCDC_LSC_CONFIG,         0x480BC698,__READ_WRITE ,__ccdc_lsc_config_bits);
__IO_REG32_BIT(CCDC_LSC_INITIAL,        0x480BC69C,__READ_WRITE ,__ccdc_lsc_initial_bits);
__IO_REG32(    CCDC_LSC_TABLE_BASE,     0x480BC6A0,__READ_WRITE );
__IO_REG32_BIT(CCDC_LSC_TABLE_OFFSET,   0x480BC6A4,__READ_WRITE ,__ccdc_lsc_table_offset_bits);

/***************************************************************************
 **
 ** Camera HIST
 **
 ***************************************************************************/
__IO_REG32_BIT(HIST_PID,                0x480BCA00,__READ       ,__hist_pid_bits);
__IO_REG32_BIT(HIST_PCR,                0x480BCA04,__READ_WRITE ,__hist_pcr_bits);
__IO_REG32_BIT(HIST_CNT,                0x480BCA08,__READ_WRITE ,__hist_cnt_bits);
__IO_REG32_BIT(HIST_WB_GAIN,            0x480BCA0C,__READ_WRITE ,__hist_wb_gain_bits);
__IO_REG32_BIT(HIST_R0_HORZ,            0x480BCA10,__READ_WRITE ,__hist_rx_horz_bits);
__IO_REG32_BIT(HIST_R0_VERT,            0x480BCA14,__READ_WRITE ,__hist_rx_vert_bits);
__IO_REG32_BIT(HIST_R1_HORZ,            0x480BCA18,__READ_WRITE ,__hist_rx_horz_bits);
__IO_REG32_BIT(HIST_R1_VERT,            0x480BCA1C,__READ_WRITE ,__hist_rx_vert_bits);
__IO_REG32_BIT(HIST_R2_HORZ,            0x480BCA20,__READ_WRITE ,__hist_rx_horz_bits);
__IO_REG32_BIT(HIST_R2_VERT,            0x480BCA24,__READ_WRITE ,__hist_rx_vert_bits);
__IO_REG32_BIT(HIST_R3_HORZ,            0x480BCA28,__READ_WRITE ,__hist_rx_horz_bits);
__IO_REG32_BIT(HIST_R3_VERT,            0x480BCA2C,__READ_WRITE ,__hist_rx_vert_bits);
__IO_REG32_BIT(HIST_ADDR,               0x480BCA30,__READ_WRITE ,__hist_addr_bits);
__IO_REG32_BIT(HIST_DATA,               0x480BCA34,__READ_WRITE ,__hist_data_bits);
__IO_REG32(    HIST_RADD,               0x480BCA38,__READ_WRITE );
__IO_REG32_BIT(HIST_RADD_OFF,           0x480BCA3C,__READ_WRITE ,__hist_radd_off_bits);
__IO_REG32_BIT(HIST_H_V_INFO,           0x480BCA40,__READ_WRITE ,__hist_h_v_info_bits);

/***************************************************************************
 **
 ** Camera H3A
 **
 ***************************************************************************/
__IO_REG32_BIT(H3A_PID,                 0x480BCC00,__READ       ,__h3a_pid_bits);
__IO_REG32_BIT(H3A_PCR,                 0x480BCC04,__READ_WRITE ,__h3a_pcr_bits);
__IO_REG32_BIT(H3A_AFPAX1,              0x480BCC08,__READ_WRITE ,__h3a_afpax1_bits);
__IO_REG32_BIT(H3A_AFPAX2,              0x480BCC0C,__READ_WRITE ,__h3a_afpax2_bits);
__IO_REG32_BIT(H3A_AFPAXSTART,          0x480BCC10,__READ_WRITE ,__h3a_afpaxstart_bits);
__IO_REG32_BIT(H3A_AFIIRSH,             0x480BCC14,__READ_WRITE ,__h3a_afiirsh_bits);
__IO_REG32(    H3A_AFBUFST,             0x480BCC18,__READ_WRITE );
__IO_REG32_BIT(H3A_AFCOEF010,           0x480BCC1C,__READ_WRITE ,__h3a_afcoefx10_bits);
__IO_REG32_BIT(H3A_AFCOEF032,           0x480BCC20,__READ_WRITE ,__h3a_afcoefx32_bits);
__IO_REG32_BIT(H3A_AFCOEF054,           0x480BCC24,__READ_WRITE ,__h3a_afcoefx54_bits);
__IO_REG32_BIT(H3A_AFCOEF076,           0x480BCC28,__READ_WRITE ,__h3a_afcoefx76_bits);
__IO_REG32_BIT(H3A_AFCOEF098,           0x480BCC2C,__READ_WRITE ,__h3a_afcoefx98_bits);
__IO_REG32_BIT(H3A_AFCOEF0010,          0x480BCC30,__READ_WRITE ,__h3a_afcoefx010_bits);
__IO_REG32_BIT(H3A_AFCOEF110,           0x480BCC34,__READ_WRITE ,__h3a_afcoefx10_bits);
__IO_REG32_BIT(H3A_AFCOEF132,           0x480BCC38,__READ_WRITE ,__h3a_afcoefx32_bits);
__IO_REG32_BIT(H3A_AFCOEF154,           0x480BCC3C,__READ_WRITE ,__h3a_afcoefx54_bits);
__IO_REG32_BIT(H3A_AFCOEF176,           0x480BCC40,__READ_WRITE ,__h3a_afcoefx76_bits);
__IO_REG32_BIT(H3A_AFCOEF198,           0x480BCC44,__READ_WRITE ,__h3a_afcoefx98_bits);
__IO_REG32_BIT(H3A_AFCOEF1010,          0x480BCC48,__READ_WRITE ,__h3a_afcoefx010_bits);
__IO_REG32_BIT(H3A_AEWWIN1,             0x480BCC4C,__READ_WRITE ,__h3a_aewwin1_bits);
__IO_REG32_BIT(H3A_AEWINSTART,          0x480BCC50,__READ_WRITE ,__h3a_aewinstart_bits);
__IO_REG32_BIT(H3A_AEWINBLK,            0x480BCC54,__READ_WRITE ,__h3a_aewinblk_bits);
__IO_REG32_BIT(H3A_AEWSUBWIN,           0x480BCC58,__READ_WRITE ,__h3a_aewsubwin_bits);
__IO_REG32(    H3A_AEWBUFST,            0x480BCC5C,__READ_WRITE );

/***************************************************************************
 **
 ** Camera PREVIEW
 **
 ***************************************************************************/
__IO_REG32_BIT(PRV_PID,                 0x480BCE00,__READ       ,__prv_pid_bits);
__IO_REG32_BIT(PRV_PCR,                 0x480BCE04,__READ_WRITE ,__prv_pcr_bits);
__IO_REG32_BIT(PRV_HORZ_INFO,           0x480BCE08,__READ_WRITE ,__prv_horz_info_bits);
__IO_REG32_BIT(PRV_VERT_INFO,           0x480BCE0C,__READ_WRITE ,__prv_vert_info_bits);
__IO_REG32(    PRV_RSDR_ADDR,           0x480BCE10,__READ_WRITE );
__IO_REG32_BIT(PRV_RADR_OFFSET,         0x480BCE14,__READ_WRITE ,__prv_radr_offset_bits);
__IO_REG32(    PRV_DSDR_ADDR,           0x480BCE18,__READ_WRITE );
__IO_REG32_BIT(PRV_DRKF_OFFSET,         0x480BCE1C,__READ_WRITE ,__prv_radr_offset_bits);
__IO_REG32(    PRV_WSDR_ADDR,           0x480BCE20,__READ_WRITE );
__IO_REG32_BIT(PRV_WADD_OFFSET,         0x480BCE24,__READ_WRITE ,__prv_radr_offset_bits);
__IO_REG32_BIT(PRV_AVE,                 0x480BCE28,__READ_WRITE ,__prv_ave_bits);
__IO_REG32_BIT(PRV_HMED,                0x480BCE2C,__READ_WRITE ,__prv_hmed_bits);
__IO_REG32_BIT(PRV_NF,                  0x480BCE30,__READ_WRITE ,__prv_nf_bits);
__IO_REG32_BIT(PRV_WB_DGAIN,            0x480BCE34,__READ_WRITE ,__prv_wb_dgain_bits);
__IO_REG32_BIT(PRV_WBGAIN,              0x480BCE38,__READ_WRITE ,__prv_wbgain_bits);
__IO_REG32_BIT(PRV_WBSEL,               0x480BCE3C,__READ_WRITE ,__prv_wbsel_bits);
__IO_REG32_BIT(PRV_CFA,                 0x480BCE40,__READ_WRITE ,__prv_cfa_bits);
__IO_REG32_BIT(PRV_BLKADJOFF,           0x480BCE44,__READ_WRITE ,__prv_blkadjoff_bits);
__IO_REG32_BIT(PRV_RGB_MAT1,            0x480BCE48,__READ_WRITE ,__prv_rgb_mat1_bits);
__IO_REG32_BIT(PRV_RGB_MAT2,            0x480BCE4C,__READ_WRITE ,__prv_rgb_mat2_bits);
__IO_REG32_BIT(PRV_RGB_MAT3,            0x480BCE50,__READ_WRITE ,__prv_rgb_mat3_bits);
__IO_REG32_BIT(PRV_RGB_MAT4,            0x480BCE54,__READ_WRITE ,__prv_rgb_mat4_bits);
__IO_REG32_BIT(PRV_RGB_MAT5,            0x480BCE58,__READ_WRITE ,__prv_rgb_mat5_bits);
__IO_REG32_BIT(PRV_RGB_OFF1,            0x480BCE5C,__READ_WRITE ,__prv_rgb_off1_bits);
__IO_REG32_BIT(PRV_RGB_OFF2,            0x480BCE60,__READ_WRITE ,__prv_rgb_off2_bits);
__IO_REG32_BIT(PRV_CSC0,                0x480BCE64,__READ_WRITE ,__prv_csc0_bits);
__IO_REG32_BIT(PRV_CSC1,                0x480BCE68,__READ_WRITE ,__prv_csc1_bits);
__IO_REG32_BIT(PRV_CSC2,                0x480BCE6C,__READ_WRITE ,__prv_csc2_bits);
__IO_REG32_BIT(PRV_CSC_OFFSET,          0x480BCE70,__READ_WRITE ,__prv_csc_offset_bits);
__IO_REG32_BIT(PRV_CNT_BRT,             0x480BCE74,__READ_WRITE ,__prv_cnt_brt_bits);
__IO_REG32_BIT(PRV_CSUP,                0x480BCE78,__READ_WRITE ,__prv_csup_bits);
__IO_REG32_BIT(PRV_SETUP_YC,            0x480BCE7C,__READ_WRITE ,__prv_setup_yc_bits);
__IO_REG32_BIT(PRV_SET_TBL_ADDR,        0x480BCE80,__READ_WRITE ,__prv_set_tbl_addr_bits);
__IO_REG32_BIT(PRV_SET_TBL_DATA,        0x480BCE84,__READ_WRITE ,__prv_set_tbl_data_bits);
__IO_REG32_BIT(PRV_CDC_THR0,            0x480BCE90,__READ_WRITE ,__prv_cdc_thrx_bits);
__IO_REG32_BIT(PRV_CDC_THR1,            0x480BCE94,__READ_WRITE ,__prv_cdc_thrx_bits);
__IO_REG32_BIT(PRV_CDC_THR2,            0x480BCE98,__READ_WRITE ,__prv_cdc_thrx_bits);
__IO_REG32_BIT(PRV_CDC_THR3,            0x480BCE9C,__READ_WRITE ,__prv_cdc_thrx_bits);

/***************************************************************************
 **
 ** Camera RESIZER
 **
 ***************************************************************************/
__IO_REG32_BIT(RSZ_PID,                 0x480BD000,__READ       ,__rsz_pid_bits);
__IO_REG32_BIT(RSZ_PCR,                 0x480BD004,__READ_WRITE ,__rsz_pcr_bits);
__IO_REG32_BIT(RSZ_CNT,                 0x480BD008,__READ_WRITE ,__rsz_cnt_bits);
__IO_REG32_BIT(RSZ_OUT_SIZE,            0x480BD00C,__READ_WRITE ,__rsz_out_size_bits);
__IO_REG32_BIT(RSZ_IN_START,            0x480BD010,__READ_WRITE ,__rsz_in_start_bits);
__IO_REG32_BIT(RSZ_IN_SIZE,             0x480BD014,__READ_WRITE ,__rsz_in_size_bits);
__IO_REG32(    RSZ_SDR_INADD,           0x480BD018,__READ_WRITE );
__IO_REG32_BIT(RSZ_SDR_INOFF,           0x480BD01C,__READ_WRITE ,__rsz_sdr_inoff_bits);
__IO_REG32(    RSZ_SDR_OUTADD,          0x480BD020,__READ_WRITE );
__IO_REG32_BIT(RSZ_SDR_OUTOFF,          0x480BD024,__READ_WRITE ,__rsz_sdr_inoff_bits);
__IO_REG32_BIT(RSZ_HFILT10,             0x480BD028,__READ_WRITE ,__rsz_filt10_bits);
__IO_REG32_BIT(RSZ_HFILT32,             0x480BD02C,__READ_WRITE ,__rsz_filt32_bits);
__IO_REG32_BIT(RSZ_HFILT54,             0x480BD030,__READ_WRITE ,__rsz_filt54_bits);
__IO_REG32_BIT(RSZ_HFILT76,             0x480BD034,__READ_WRITE ,__rsz_filt76_bits);
__IO_REG32_BIT(RSZ_HFILT98,             0x480BD038,__READ_WRITE ,__rsz_filt98_bits);
__IO_REG32_BIT(RSZ_HFILT1110,           0x480BD03C,__READ_WRITE ,__rsz_filt1110_bits);
__IO_REG32_BIT(RSZ_HFILT1312,           0x480BD040,__READ_WRITE ,__rsz_filt1312_bits);
__IO_REG32_BIT(RSZ_HFILT1514,           0x480BD044,__READ_WRITE ,__rsz_filt1514_bits);
__IO_REG32_BIT(RSZ_HFILT1716,           0x480BD048,__READ_WRITE ,__rsz_filt1716_bits);
__IO_REG32_BIT(RSZ_HFILT1918,           0x480BD04C,__READ_WRITE ,__rsz_filt1918_bits);
__IO_REG32_BIT(RSZ_HFILT2120,           0x480BD050,__READ_WRITE ,__rsz_filt2120_bits);
__IO_REG32_BIT(RSZ_HFILT2322,           0x480BD054,__READ_WRITE ,__rsz_filt2322_bits);
__IO_REG32_BIT(RSZ_HFILT2524,           0x480BD058,__READ_WRITE ,__rsz_filt2524_bits);
__IO_REG32_BIT(RSZ_HFILT2726,           0x480BD05C,__READ_WRITE ,__rsz_filt2726_bits);
__IO_REG32_BIT(RSZ_HFILT2928,           0x480BD060,__READ_WRITE ,__rsz_filt2928_bits);
__IO_REG32_BIT(RSZ_HFILT3130,           0x480BD064,__READ_WRITE ,__rsz_filt3130_bits);
__IO_REG32_BIT(RSZ_VFILT10,             0x480BD068,__READ_WRITE ,__rsz_filt10_bits);
__IO_REG32_BIT(RSZ_VFILT32,             0x480BD06C,__READ_WRITE ,__rsz_filt32_bits);
__IO_REG32_BIT(RSZ_VFILT54,             0x480BD070,__READ_WRITE ,__rsz_filt54_bits);
__IO_REG32_BIT(RSZ_VFILT76,             0x480BD074,__READ_WRITE ,__rsz_filt76_bits);
__IO_REG32_BIT(RSZ_VFILT98,             0x480BD078,__READ_WRITE ,__rsz_filt98_bits);
__IO_REG32_BIT(RSZ_VFILT1110,           0x480BD07C,__READ_WRITE ,__rsz_filt1110_bits);
__IO_REG32_BIT(RSZ_VFILT1312,           0x480BD080,__READ_WRITE ,__rsz_filt1312_bits);
__IO_REG32_BIT(RSZ_VFILT1514,           0x480BD084,__READ_WRITE ,__rsz_filt1514_bits);
__IO_REG32_BIT(RSZ_VFILT1716,           0x480BD088,__READ_WRITE ,__rsz_filt1716_bits);
__IO_REG32_BIT(RSZ_VFILT1918,           0x480BD08C,__READ_WRITE ,__rsz_filt1918_bits);
__IO_REG32_BIT(RSZ_VFILT2120,           0x480BD090,__READ_WRITE ,__rsz_filt2120_bits);
__IO_REG32_BIT(RSZ_VFILT2322,           0x480BD094,__READ_WRITE ,__rsz_filt2322_bits);
__IO_REG32_BIT(RSZ_VFILT2524,           0x480BD098,__READ_WRITE ,__rsz_filt2524_bits);
__IO_REG32_BIT(RSZ_VFILT2726,           0x480BD09C,__READ_WRITE ,__rsz_filt2726_bits);
__IO_REG32_BIT(RSZ_VFILT2928,           0x480BD0A0,__READ_WRITE ,__rsz_filt2928_bits);
__IO_REG32_BIT(RSZ_VFILT3130,           0x480BD0A4,__READ_WRITE ,__rsz_filt3130_bits);
__IO_REG32_BIT(RSZ_YENH,                0x480BD0A8,__READ_WRITE ,__rsz_yenh_bits);

/***************************************************************************
 **
 ** Camera SBL
 **
 ***************************************************************************/
__IO_REG32_BIT(SBL_PID,                 0x480BD200,__READ       ,__sbl_pid_bits);
__IO_REG32_BIT(SBL_PCR,                 0x480BD204,__READ_WRITE ,__sbl_pcr_bits);
__IO_REG32_BIT(SBL_GLB_REG_0,           0x480BD208,__READ       ,__sbl_glb_reg_x_bits);
__IO_REG32_BIT(SBL_GLB_REG_1,           0x480BD20C,__READ       ,__sbl_glb_reg_x_bits);
__IO_REG32_BIT(SBL_GLB_REG_2,           0x480BD210,__READ       ,__sbl_glb_reg_x_bits);
__IO_REG32_BIT(SBL_GLB_REG_3,           0x480BD214,__READ       ,__sbl_glb_reg_x_bits);
__IO_REG32_BIT(SBL_GLB_REG_4,           0x480BD218,__READ       ,__sbl_glb_reg_x_bits);
__IO_REG32_BIT(SBL_GLB_REG_5,           0x480BD21C,__READ       ,__sbl_glb_reg_x_bits);
__IO_REG32_BIT(SBL_GLB_REG_6,           0x480BD220,__READ       ,__sbl_glb_reg_x_bits);
__IO_REG32_BIT(SBL_GLB_REG_7,           0x480BD224,__READ       ,__sbl_glb_reg_x_bits);
__IO_REG32_BIT(SBL_CCDC_WR_0,           0x480BD228,__READ       ,__sbl_ccdc_wr_x_bits);
__IO_REG32_BIT(SBL_CCDC_WR_1,           0x480BD22C,__READ       ,__sbl_ccdc_wr_x_bits);
__IO_REG32_BIT(SBL_CCDC_WR_2,           0x480BD230,__READ       ,__sbl_ccdc_wr_x_bits);
__IO_REG32_BIT(SBL_CCDC_WR_3,           0x480BD234,__READ       ,__sbl_ccdc_wr_x_bits);
__IO_REG32_BIT(SBL_CCDC_FP_RD_0,        0x480BD238,__READ       ,__sbl_ccdc_fp_rd_x_bits);
__IO_REG32_BIT(SBL_CCDC_FP_RD_1,        0x480BD23C,__READ       ,__sbl_ccdc_fp_rd_x_bits);
__IO_REG32_BIT(SBL_PRV_RD_0,            0x480BD240,__READ       ,__sbl_prv_rd_x_bits);
__IO_REG32_BIT(SBL_PRV_RD_1,            0x480BD244,__READ       ,__sbl_prv_rd_x_bits);
__IO_REG32_BIT(SBL_PRV_RD_2,            0x480BD248,__READ       ,__sbl_prv_rd_x_bits);
__IO_REG32_BIT(SBL_PRV_RD_3,            0x480BD24C,__READ       ,__sbl_prv_rd_x_bits);
__IO_REG32_BIT(SBL_PRV_WR_0,            0x480BD250,__READ       ,__sbl_prv_wr_x_bits);
__IO_REG32_BIT(SBL_PRV_WR_1,            0x480BD254,__READ       ,__sbl_prv_wr_x_bits);
__IO_REG32_BIT(SBL_PRV_WR_2,            0x480BD258,__READ       ,__sbl_prv_wr_x_bits);
__IO_REG32_BIT(SBL_PRV_WR_3,            0x480BD25C,__READ       ,__sbl_prv_wr_x_bits);
__IO_REG32_BIT(SBL_PRV_DK_RD_0,         0x480BD260,__READ       ,__sbl_prv_dk_rd_x_bits);
__IO_REG32_BIT(SBL_PRV_DK_RD_1,         0x480BD264,__READ       ,__sbl_prv_dk_rd_x_bits);
__IO_REG32_BIT(SBL_PRV_DK_RD_2,         0x480BD268,__READ       ,__sbl_prv_dk_rd_x_bits);
__IO_REG32_BIT(SBL_PRV_DK_RD_3,         0x480BD26C,__READ       ,__sbl_prv_dk_rd_x_bits);
__IO_REG32_BIT(SBL_RSZ_RD_0,            0x480BD270,__READ       ,__sbl_rsz_rd_x_bits);
__IO_REG32_BIT(SBL_RSZ_RD_1,            0x480BD274,__READ       ,__sbl_rsz_rd_x_bits);
__IO_REG32_BIT(SBL_RSZ_RD_2,            0x480BD278,__READ       ,__sbl_rsz_rd_x_bits);
__IO_REG32_BIT(SBL_RSZ_RD_3,            0x480BD27C,__READ       ,__sbl_rsz_rd_x_bits);
__IO_REG32_BIT(SBL_RSZ1_WR_0,           0x480BD280,__READ       ,__sbl_rszx_wr_y_bits);
__IO_REG32_BIT(SBL_RSZ1_WR_1,           0x480BD284,__READ       ,__sbl_rszx_wr_y_bits);
__IO_REG32_BIT(SBL_RSZ1_WR_2,           0x480BD288,__READ       ,__sbl_rszx_wr_y_bits);
__IO_REG32_BIT(SBL_RSZ1_WR_3,           0x480BD28C,__READ       ,__sbl_rszx_wr_y_bits);
__IO_REG32_BIT(SBL_RSZ2_WR_0,           0x480BD290,__READ       ,__sbl_rszx_wr_y_bits);
__IO_REG32_BIT(SBL_RSZ2_WR_1,           0x480BD294,__READ       ,__sbl_rszx_wr_y_bits);
__IO_REG32_BIT(SBL_RSZ2_WR_2,           0x480BD298,__READ       ,__sbl_rszx_wr_y_bits);
__IO_REG32_BIT(SBL_RSZ2_WR_3,           0x480BD29C,__READ       ,__sbl_rszx_wr_y_bits);
__IO_REG32_BIT(SBL_RSZ3_WR_0,           0x480BD2A0,__READ       ,__sbl_rszx_wr_y_bits);
__IO_REG32_BIT(SBL_RSZ3_WR_1,           0x480BD2A4,__READ       ,__sbl_rszx_wr_y_bits);
__IO_REG32_BIT(SBL_RSZ3_WR_2,           0x480BD2A8,__READ       ,__sbl_rszx_wr_y_bits);
__IO_REG32_BIT(SBL_RSZ3_WR_3,           0x480BD2AC,__READ       ,__sbl_rszx_wr_y_bits);
__IO_REG32_BIT(SBL_RSZ4_WR_0,           0x480BD2B0,__READ       ,__sbl_rszx_wr_y_bits);
__IO_REG32_BIT(SBL_RSZ4_WR_1,           0x480BD2B4,__READ       ,__sbl_rszx_wr_y_bits);
__IO_REG32_BIT(SBL_RSZ4_WR_2,           0x480BD2B8,__READ       ,__sbl_rszx_wr_y_bits);
__IO_REG32_BIT(SBL_RSZ4_WR_3,           0x480BD2BC,__READ       ,__sbl_rszx_wr_y_bits);
__IO_REG32_BIT(SBL_HIST_RD_0,           0x480BD2C0,__READ       ,__sbl_hist_rd_x_bits);
__IO_REG32_BIT(SBL_HIST_RD_1,           0x480BD2C4,__READ       ,__sbl_hist_rd_x_bits);
__IO_REG32_BIT(SBL_H3A_AF_WR_0,         0x480BD2C8,__READ       ,__sbl_h3a_af_wr_x_bits);
__IO_REG32_BIT(SBL_H3A_AF_WR_1,         0x480BD2CC,__READ       ,__sbl_h3a_af_wr_x_bits);
__IO_REG32_BIT(SBL_H3A_AEAWB_WR_0,      0x480BD2D0,__READ       ,__sbl_h3a_aeawb_wr_x_bits);
__IO_REG32_BIT(SBL_H3A_AEAWB_WR_1,      0x480BD2D4,__READ       ,__sbl_h3a_aeawb_wr_x_bits);
__IO_REG32_BIT(SBL_CSIA_WR_0,           0x480BD2D8,__READ       ,__sbl_csi_wr_x_bits);
__IO_REG32_BIT(SBL_CSIA_WR_1,           0x480BD2DC,__READ       ,__sbl_csi_wr_x_bits);
__IO_REG32_BIT(SBL_CSIA_WR_2,           0x480BD2E0,__READ       ,__sbl_csi_wr_x_bits);
__IO_REG32_BIT(SBL_CSIA_WR_3,           0x480BD2E4,__READ       ,__sbl_csi_wr_x_bits);
__IO_REG32_BIT(SBL_CSIB_WR_0,           0x480BD2E8,__READ       ,__sbl_csi_wr_x_bits);
__IO_REG32_BIT(SBL_CSIB_WR_1,           0x480BD2EC,__READ       ,__sbl_csi_wr_x_bits);
__IO_REG32_BIT(SBL_CSIB_WR_2,           0x480BD2F0,__READ       ,__sbl_csi_wr_x_bits);
__IO_REG32_BIT(SBL_CSIB_WR_3,           0x480BD2F4,__READ       ,__sbl_csi_wr_x_bits);
__IO_REG32_BIT(SBL_SDR_REQ_EXP,         0x480BD2F8,__READ_WRITE ,__sbl_sdr_req_exp_bits);

/***************************************************************************
 **
 ** Camera CSI2A
 **
 ***************************************************************************/
__IO_REG32_BIT(CSI2A_REVISION,              0x480BD800,__READ       ,__csi2_revision_bits);
__IO_REG32_BIT(CSI2A_SYSCONFIG,             0x480BD810,__READ_WRITE ,__csi2_sysconfig_bits);
__IO_REG32_BIT(CSI2A_SYSSTATUS,             0x480BD814,__READ       ,__csi2_sysstatus_bits);
__IO_REG32_BIT(CSI2A_IRQSTATUS,             0x480BD818,__READ_WRITE ,__csi2_irqstatus_bits);
__IO_REG32_BIT(CSI2A_IRQENABLE,             0x480BD81C,__READ_WRITE ,__csi2_irqstatus_bits);
__IO_REG32_BIT(CSI2A_CTRL,                  0x480BD840,__READ_WRITE ,__csi2_ctrl_bits);
__IO_REG32(    CSI2A_DBG_H,                 0x480BD844,__WRITE      );
__IO_REG32_BIT(CSI2A_GNQ,                   0x480BD848,__READ       ,__csi2_gnq_bits);
__IO_REG32_BIT(CSI2A_COMPLEXIO_CFG1,        0x480BD850,__READ_WRITE ,__csi2_complexio_cfg1_bits);
__IO_REG32_BIT(CSI2A_COMPLEXIO1_IRQSTATUS,  0x480BD854,__READ_WRITE ,__csi2_complexio1_irqstatus_bits);
__IO_REG32_BIT(CSI2A_SHORT_PACKET,          0x480BD85C,__READ       ,__csi2_short_packet_bits);
__IO_REG32_BIT(CSI2A_COMPLEXIO1_IRQENABLE,  0x480BD860,__READ_WRITE ,__csi2_complexio1_irqstatus_bits);
__IO_REG32(    CSI2A_DBG_P,                 0x480BD868,__WRITE      );
__IO_REG32_BIT(CSI2A_TIMING,                0x480BD86C,__READ_WRITE ,__csi2_timing_bits);
__IO_REG32_BIT(CSI2A_CT0_CTRL1,             0x480BD870,__READ_WRITE ,__csi2_ctx_ctrl1_bits);
__IO_REG32_BIT(CSI2A_CT0_CTRL2,             0x480BD874,__READ_WRITE ,__csi2_ctx_ctrl2_bits);
__IO_REG32_BIT(CSI2A_CT0_DAT_OFST,          0x480BD878,__READ_WRITE ,__csi2_ctx_dat_ofst_bits);
__IO_REG32(    CSI2A_CT0_DAT_PING_ADDR,     0x480BD87C,__READ_WRITE );
__IO_REG32(    CSI2A_CT0_DAT_PONG_ADDR,     0x480BD880,__READ_WRITE );
__IO_REG32_BIT(CSI2A_CT0_IRQENABLE,         0x480BD884,__READ_WRITE ,__csi2_ctx_irqenable_bits);
__IO_REG32_BIT(CSI2A_CT0_IRQSTATUS,         0x480BD888,__READ_WRITE ,__csi2_ctx_irqenable_bits);
__IO_REG32_BIT(CSI2A_CT0_CTRL3,             0x480BD88C,__READ_WRITE ,__csi2_ctx_ctrl3_bits);
__IO_REG32_BIT(CSI2A_CT1_CTRL1,             0x480BD890,__READ_WRITE ,__csi2_ctx_ctrl1_bits);
__IO_REG32_BIT(CSI2A_CT1_CTRL2,             0x480BD894,__READ_WRITE ,__csi2_ctx_ctrl2_bits);
__IO_REG32_BIT(CSI2A_CT1_DAT_OFST,          0x480BD898,__READ_WRITE ,__csi2_ctx_dat_ofst_bits);
__IO_REG32(    CSI2A_CT1_DAT_PING_ADDR,     0x480BD89C,__READ_WRITE );
__IO_REG32(    CSI2A_CT1_DAT_PONG_ADDR,     0x480BD8A0,__READ_WRITE );
__IO_REG32_BIT(CSI2A_CT1_IRQENABLE,         0x480BD8A4,__READ_WRITE ,__csi2_ctx_irqenable_bits);
__IO_REG32_BIT(CSI2A_CT1_IRQSTATUS,         0x480BD8A8,__READ_WRITE ,__csi2_ctx_irqenable_bits);
__IO_REG32_BIT(CSI2A_CT1_CTRL3,             0x480BD8AC,__READ_WRITE ,__csi2_ctx_ctrl3_bits);
__IO_REG32_BIT(CSI2A_CT2_CTRL1,             0x480BD8B0,__READ_WRITE ,__csi2_ctx_ctrl1_bits);
__IO_REG32_BIT(CSI2A_CT2_CTRL2,             0x480BD8B4,__READ_WRITE ,__csi2_ctx_ctrl2_bits);
__IO_REG32_BIT(CSI2A_CT2_DAT_OFST,          0x480BD8B8,__READ_WRITE ,__csi2_ctx_dat_ofst_bits);
__IO_REG32(    CSI2A_CT2_DAT_PING_ADDR,     0x480BD8BC,__READ_WRITE );
__IO_REG32(    CSI2A_CT2_DAT_PONG_ADDR,     0x480BD8C0,__READ_WRITE );
__IO_REG32_BIT(CSI2A_CT2_IRQENABLE,         0x480BD8C4,__READ_WRITE ,__csi2_ctx_irqenable_bits);
__IO_REG32_BIT(CSI2A_CT2_IRQSTATUS,         0x480BD8C8,__READ_WRITE ,__csi2_ctx_irqenable_bits);
__IO_REG32_BIT(CSI2A_CT2_CTRL3,             0x480BD8CC,__READ_WRITE ,__csi2_ctx_ctrl3_bits);
__IO_REG32_BIT(CSI2A_CT3_CTRL1,             0x480BD8D0,__READ_WRITE ,__csi2_ctx_ctrl1_bits);
__IO_REG32_BIT(CSI2A_CT3_CTRL2,             0x480BD8D4,__READ_WRITE ,__csi2_ctx_ctrl2_bits);
__IO_REG32_BIT(CSI2A_CT3_DAT_OFST,          0x480BD8D8,__READ_WRITE ,__csi2_ctx_dat_ofst_bits);
__IO_REG32(    CSI2A_CT3_DAT_PING_ADDR,     0x480BD8DC,__READ_WRITE );
__IO_REG32(    CSI2A_CT3_DAT_PONG_ADDR,     0x480BD8E0,__READ_WRITE );
__IO_REG32_BIT(CSI2A_CT3_IRQENABLE,         0x480BD8E4,__READ_WRITE ,__csi2_ctx_irqenable_bits);
__IO_REG32_BIT(CSI2A_CT3_IRQSTATUS,         0x480BD8E8,__READ_WRITE ,__csi2_ctx_irqenable_bits);
__IO_REG32_BIT(CSI2A_CT3_CTRL3,             0x480BD8EC,__READ_WRITE ,__csi2_ctx_ctrl3_bits);
__IO_REG32_BIT(CSI2A_CT4_CTRL1,             0x480BD8F0,__READ_WRITE ,__csi2_ctx_ctrl1_bits);
__IO_REG32_BIT(CSI2A_CT4_CTRL2,             0x480BD8F4,__READ_WRITE ,__csi2_ctx_ctrl2_bits);
__IO_REG32_BIT(CSI2A_CT4_DAT_OFST,          0x480BD8F8,__READ_WRITE ,__csi2_ctx_dat_ofst_bits);
__IO_REG32(    CSI2A_CT4_DAT_PING_ADDR,     0x480BD8FC,__READ_WRITE );
__IO_REG32(    CSI2A_CT4_DAT_PONG_ADDR,     0x480BD900,__READ_WRITE );
__IO_REG32_BIT(CSI2A_CT4_IRQENABLE,         0x480BD904,__READ_WRITE ,__csi2_ctx_irqenable_bits);
__IO_REG32_BIT(CSI2A_CT4_IRQSTATUS,         0x480BD908,__READ_WRITE ,__csi2_ctx_irqenable_bits);
__IO_REG32_BIT(CSI2A_CT4_CTRL3,             0x480BD90C,__READ_WRITE ,__csi2_ctx_ctrl3_bits);
__IO_REG32_BIT(CSI2A_CT5_CTRL1,             0x480BD910,__READ_WRITE ,__csi2_ctx_ctrl1_bits);
__IO_REG32_BIT(CSI2A_CT5_CTRL2,             0x480BD914,__READ_WRITE ,__csi2_ctx_ctrl2_bits);
__IO_REG32_BIT(CSI2A_CT5_DAT_OFST,          0x480BD918,__READ_WRITE ,__csi2_ctx_dat_ofst_bits);
__IO_REG32(    CSI2A_CT5_DAT_PING_ADDR,     0x480BD91C,__READ_WRITE );
__IO_REG32(    CSI2A_CT5_DAT_PONG_ADDR,     0x480BD920,__READ_WRITE );
__IO_REG32_BIT(CSI2A_CT5_IRQENABLE,         0x480BD924,__READ_WRITE ,__csi2_ctx_irqenable_bits);
__IO_REG32_BIT(CSI2A_CT5_IRQSTATUS,         0x480BD928,__READ_WRITE ,__csi2_ctx_irqenable_bits);
__IO_REG32_BIT(CSI2A_CT5_CTRL3,             0x480BD92C,__READ_WRITE ,__csi2_ctx_ctrl3_bits);
__IO_REG32_BIT(CSI2A_CT6_CTRL1,             0x480BD930,__READ_WRITE ,__csi2_ctx_ctrl1_bits);
__IO_REG32_BIT(CSI2A_CT6_CTRL2,             0x480BD934,__READ_WRITE ,__csi2_ctx_ctrl2_bits);
__IO_REG32_BIT(CSI2A_CT6_DAT_OFST,          0x480BD938,__READ_WRITE ,__csi2_ctx_dat_ofst_bits);
__IO_REG32(    CSI2A_CT6_DAT_PING_ADDR,     0x480BD93C,__READ_WRITE );
__IO_REG32(    CSI2A_CT6_DAT_PONG_ADDR,     0x480BD940,__READ_WRITE );
__IO_REG32_BIT(CSI2A_CT6_IRQENABLE,         0x480BD944,__READ_WRITE ,__csi2_ctx_irqenable_bits);
__IO_REG32_BIT(CSI2A_CT6_IRQSTATUS,         0x480BD948,__READ_WRITE ,__csi2_ctx_irqenable_bits);
__IO_REG32_BIT(CSI2A_CT6_CTRL3,             0x480BD94C,__READ_WRITE ,__csi2_ctx_ctrl3_bits);
__IO_REG32_BIT(CSI2A_CT7_CTRL1,             0x480BD950,__READ_WRITE ,__csi2_ctx_ctrl1_bits);
__IO_REG32_BIT(CSI2A_CT7_CTRL2,             0x480BD954,__READ_WRITE ,__csi2_ctx_ctrl2_bits);
__IO_REG32_BIT(CSI2A_CT7_DAT_OFST,          0x480BD958,__READ_WRITE ,__csi2_ctx_dat_ofst_bits);
__IO_REG32(    CSI2A_CT7_DAT_PING_ADDR,     0x480BD95C,__READ_WRITE );
__IO_REG32(    CSI2A_CT7_DAT_PONG_ADDR,     0x480BD960,__READ_WRITE );
__IO_REG32_BIT(CSI2A_CT7_IRQENABLE,         0x480BD964,__READ_WRITE ,__csi2_ctx_irqenable_bits);
__IO_REG32_BIT(CSI2A_CT7_IRQSTATUS,         0x480BD968,__READ_WRITE ,__csi2_ctx_irqenable_bits);
__IO_REG32_BIT(CSI2A_CT7_CTRL3,             0x480BD96C,__READ_WRITE ,__csi2_ctx_ctrl3_bits);
__IO_REG32_BIT(CSI2A_CT0_TRANSCODEH,        0x480BD9C0,__READ_WRITE ,__csi2_ctx_transcodeh_bits);
__IO_REG32_BIT(CSI2A_CT0_TRANSCODEV,        0x480BD9C4,__READ_WRITE ,__csi2_ctx_transcodev_bits);
__IO_REG32_BIT(CSI2A_CT1_TRANSCODEH,        0x480BD9C8,__READ_WRITE ,__csi2_ctx_transcodeh_bits);
__IO_REG32_BIT(CSI2A_CT1_TRANSCODEV,        0x480BD9CC,__READ_WRITE ,__csi2_ctx_transcodev_bits);
__IO_REG32_BIT(CSI2A_CT2_TRANSCODEH,        0x480BD9D0,__READ_WRITE ,__csi2_ctx_transcodeh_bits);
__IO_REG32_BIT(CSI2A_CT2_TRANSCODEV,        0x480BD9D4,__READ_WRITE ,__csi2_ctx_transcodev_bits);
__IO_REG32_BIT(CSI2A_CT3_TRANSCODEH,        0x480BD9D8,__READ_WRITE ,__csi2_ctx_transcodeh_bits);
__IO_REG32_BIT(CSI2A_CT3_TRANSCODEV,        0x480BD9DC,__READ_WRITE ,__csi2_ctx_transcodev_bits);
__IO_REG32_BIT(CSI2A_CT4_TRANSCODEH,        0x480BD9E0,__READ_WRITE ,__csi2_ctx_transcodeh_bits);
__IO_REG32_BIT(CSI2A_CT4_TRANSCODEV,        0x480BD9E4,__READ_WRITE ,__csi2_ctx_transcodev_bits);
__IO_REG32_BIT(CSI2A_CT5_TRANSCODEH,        0x480BD9E8,__READ_WRITE ,__csi2_ctx_transcodeh_bits);
__IO_REG32_BIT(CSI2A_CT5_TRANSCODEV,        0x480BD9EC,__READ_WRITE ,__csi2_ctx_transcodev_bits);
__IO_REG32_BIT(CSI2A_CT6_TRANSCODEH,        0x480BD9F0,__READ_WRITE ,__csi2_ctx_transcodeh_bits);
__IO_REG32_BIT(CSI2A_CT6_TRANSCODEV,        0x480BD9F4,__READ_WRITE ,__csi2_ctx_transcodev_bits);
__IO_REG32_BIT(CSI2A_CT7_TRANSCODEH,        0x480BD9F8,__READ_WRITE ,__csi2_ctx_transcodeh_bits);
__IO_REG32_BIT(CSI2A_CT7_TRANSCODEV,        0x480BD9FC,__READ_WRITE ,__csi2_ctx_transcodev_bits);

/***************************************************************************
 **
 ** Camera CSI2C
 **
 ***************************************************************************/
__IO_REG32_BIT(CSI2C_REVISION,              0x480BDC00,__READ       ,__csi2_revision_bits);
__IO_REG32_BIT(CSI2C_SYSCONFIG,             0x480BDC10,__READ_WRITE ,__csi2_sysconfig_bits);
__IO_REG32_BIT(CSI2C_SYSSTATUS,             0x480BDC14,__READ       ,__csi2_sysstatus_bits);
__IO_REG32_BIT(CSI2C_IRQSTATUS,             0x480BDC18,__READ_WRITE ,__csi2_irqstatus_bits);
__IO_REG32_BIT(CSI2C_IRQENABLE,             0x480BDC1C,__READ_WRITE ,__csi2_irqstatus_bits);
__IO_REG32_BIT(CSI2C_CTRL,                  0x480BDC40,__READ_WRITE ,__csi2_ctrl_bits);
__IO_REG32(    CSI2C_DBG_H,                 0x480BDC44,__WRITE      );
__IO_REG32_BIT(CSI2C_GNQ,                   0x480BDC48,__READ       ,__csi2_gnq_bits);
__IO_REG32_BIT(CSI2C_COMPLEXIO_CFG1,        0x480BDC50,__READ_WRITE ,__csi2_complexio_cfg1_bits);
__IO_REG32_BIT(CSI2C_COMPLEXIO1_IRQSTATUS,  0x480BDC54,__READ_WRITE ,__csi2_complexio1_irqstatus_bits);
__IO_REG32_BIT(CSI2C_SHORT_PACKET,          0x480BDC5C,__READ       ,__csi2_short_packet_bits);
__IO_REG32_BIT(CSI2C_COMPLEXIO1_IRQENABLE,  0x480BDC60,__READ_WRITE ,__csi2_complexio1_irqstatus_bits);
__IO_REG32(    CSI2C_DBG_P,                 0x480BDC68,__WRITE      );
__IO_REG32_BIT(CSI2C_TIMING,                0x480BDC6C,__READ_WRITE ,__csi2_timing_bits);
__IO_REG32_BIT(CSI2C_CT0_CTRL1,             0x480BDC70,__READ_WRITE ,__csi2_ctx_ctrl1_bits);
__IO_REG32_BIT(CSI2C_CT0_CTRL2,             0x480BDC74,__READ_WRITE ,__csi2_ctx_ctrl2_bits);
__IO_REG32_BIT(CSI2C_CT0_DAT_OFST,          0x480BDC78,__READ_WRITE ,__csi2_ctx_dat_ofst_bits);
__IO_REG32(    CSI2C_CT0_DAT_PING_ADDR,     0x480BDC7C,__READ_WRITE );
__IO_REG32(    CSI2C_CT0_DAT_PONG_ADDR,     0x480BDC80,__READ_WRITE );
__IO_REG32_BIT(CSI2C_CT0_IRQENABLE,         0x480BDC84,__READ_WRITE ,__csi2_ctx_irqenable_bits);
__IO_REG32_BIT(CSI2C_CT0_IRQSTATUS,         0x480BDC88,__READ_WRITE ,__csi2_ctx_irqenable_bits);
__IO_REG32_BIT(CSI2C_CT0_CTRL3,             0x480BDC8C,__READ_WRITE ,__csi2_ctx_ctrl3_bits);
__IO_REG32_BIT(CSI2C_CT1_CTRL1,             0x480BDC90,__READ_WRITE ,__csi2_ctx_ctrl1_bits);
__IO_REG32_BIT(CSI2C_CT1_CTRL2,             0x480BDC94,__READ_WRITE ,__csi2_ctx_ctrl2_bits);
__IO_REG32_BIT(CSI2C_CT1_DAT_OFST,          0x480BDC98,__READ_WRITE ,__csi2_ctx_dat_ofst_bits);
__IO_REG32(    CSI2C_CT1_DAT_PING_ADDR,     0x480BDC9C,__READ_WRITE );
__IO_REG32(    CSI2C_CT1_DAT_PONG_ADDR,     0x480BDCA0,__READ_WRITE );
__IO_REG32_BIT(CSI2C_CT1_IRQENABLE,         0x480BDCA4,__READ_WRITE ,__csi2_ctx_irqenable_bits);
__IO_REG32_BIT(CSI2C_CT1_IRQSTATUS,         0x480BDCA8,__READ_WRITE ,__csi2_ctx_irqenable_bits);
__IO_REG32_BIT(CSI2C_CT1_CTRL3,             0x480BDCAC,__READ_WRITE ,__csi2_ctx_ctrl3_bits);
__IO_REG32_BIT(CSI2C_CT2_CTRL1,             0x480BDCB0,__READ_WRITE ,__csi2_ctx_ctrl1_bits);
__IO_REG32_BIT(CSI2C_CT2_CTRL2,             0x480BDCB4,__READ_WRITE ,__csi2_ctx_ctrl2_bits);
__IO_REG32_BIT(CSI2C_CT2_DAT_OFST,          0x480BDCB8,__READ_WRITE ,__csi2_ctx_dat_ofst_bits);
__IO_REG32(    CSI2C_CT2_DAT_PING_ADDR,     0x480BDCBC,__READ_WRITE );
__IO_REG32(    CSI2C_CT2_DAT_PONG_ADDR,     0x480BDCC0,__READ_WRITE );
__IO_REG32_BIT(CSI2C_CT2_IRQENABLE,         0x480BDCC4,__READ_WRITE ,__csi2_ctx_irqenable_bits);
__IO_REG32_BIT(CSI2C_CT2_IRQSTATUS,         0x480BDCC8,__READ_WRITE ,__csi2_ctx_irqenable_bits);
__IO_REG32_BIT(CSI2C_CT2_CTRL3,             0x480BDCCC,__READ_WRITE ,__csi2_ctx_ctrl3_bits);
__IO_REG32_BIT(CSI2C_CT3_CTRL1,             0x480BDCD0,__READ_WRITE ,__csi2_ctx_ctrl1_bits);
__IO_REG32_BIT(CSI2C_CT3_CTRL2,             0x480BDCD4,__READ_WRITE ,__csi2_ctx_ctrl2_bits);
__IO_REG32_BIT(CSI2C_CT3_DAT_OFST,          0x480BDCD8,__READ_WRITE ,__csi2_ctx_dat_ofst_bits);
__IO_REG32(    CSI2C_CT3_DAT_PING_ADDR,     0x480BDCDC,__READ_WRITE );
__IO_REG32(    CSI2C_CT3_DAT_PONG_ADDR,     0x480BDCE0,__READ_WRITE );
__IO_REG32_BIT(CSI2C_CT3_IRQENABLE,         0x480BDCE4,__READ_WRITE ,__csi2_ctx_irqenable_bits);
__IO_REG32_BIT(CSI2C_CT3_IRQSTATUS,         0x480BDCE8,__READ_WRITE ,__csi2_ctx_irqenable_bits);
__IO_REG32_BIT(CSI2C_CT3_CTRL3,             0x480BDCEC,__READ_WRITE ,__csi2_ctx_ctrl3_bits);
__IO_REG32_BIT(CSI2C_CT4_CTRL1,             0x480BDCF0,__READ_WRITE ,__csi2_ctx_ctrl1_bits);
__IO_REG32_BIT(CSI2C_CT4_CTRL2,             0x480BDCF4,__READ_WRITE ,__csi2_ctx_ctrl2_bits);
__IO_REG32_BIT(CSI2C_CT4_DAT_OFST,          0x480BDCF8,__READ_WRITE ,__csi2_ctx_dat_ofst_bits);
__IO_REG32(    CSI2C_CT4_DAT_PING_ADDR,     0x480BDCFC,__READ_WRITE );
__IO_REG32(    CSI2C_CT4_DAT_PONG_ADDR,     0x480BDD00,__READ_WRITE );
__IO_REG32_BIT(CSI2C_CT4_IRQENABLE,         0x480BDD04,__READ_WRITE ,__csi2_ctx_irqenable_bits);
__IO_REG32_BIT(CSI2C_CT4_IRQSTATUS,         0x480BDD08,__READ_WRITE ,__csi2_ctx_irqenable_bits);
__IO_REG32_BIT(CSI2C_CT4_CTRL3,             0x480BDD0C,__READ_WRITE ,__csi2_ctx_ctrl3_bits);
__IO_REG32_BIT(CSI2C_CT5_CTRL1,             0x480BDD10,__READ_WRITE ,__csi2_ctx_ctrl1_bits);
__IO_REG32_BIT(CSI2C_CT5_CTRL2,             0x480BDD14,__READ_WRITE ,__csi2_ctx_ctrl2_bits);
__IO_REG32_BIT(CSI2C_CT5_DAT_OFST,          0x480BDD18,__READ_WRITE ,__csi2_ctx_dat_ofst_bits);
__IO_REG32(    CSI2C_CT5_DAT_PING_ADDR,     0x480BDD1C,__READ_WRITE );
__IO_REG32(    CSI2C_CT5_DAT_PONG_ADDR,     0x480BDD20,__READ_WRITE );
__IO_REG32_BIT(CSI2C_CT5_IRQENABLE,         0x480BDD24,__READ_WRITE ,__csi2_ctx_irqenable_bits);
__IO_REG32_BIT(CSI2C_CT5_IRQSTATUS,         0x480BDD28,__READ_WRITE ,__csi2_ctx_irqenable_bits);
__IO_REG32_BIT(CSI2C_CT5_CTRL3,             0x480BDD2C,__READ_WRITE ,__csi2_ctx_ctrl3_bits);
__IO_REG32_BIT(CSI2C_CT6_CTRL1,             0x480BDD30,__READ_WRITE ,__csi2_ctx_ctrl1_bits);
__IO_REG32_BIT(CSI2C_CT6_CTRL2,             0x480BDD34,__READ_WRITE ,__csi2_ctx_ctrl2_bits);
__IO_REG32_BIT(CSI2C_CT6_DAT_OFST,          0x480BDD38,__READ_WRITE ,__csi2_ctx_dat_ofst_bits);
__IO_REG32(    CSI2C_CT6_DAT_PING_ADDR,     0x480BDD3C,__READ_WRITE );
__IO_REG32(    CSI2C_CT6_DAT_PONG_ADDR,     0x480BDD40,__READ_WRITE );
__IO_REG32_BIT(CSI2C_CT6_IRQENABLE,         0x480BDD44,__READ_WRITE ,__csi2_ctx_irqenable_bits);
__IO_REG32_BIT(CSI2C_CT6_IRQSTATUS,         0x480BDD48,__READ_WRITE ,__csi2_ctx_irqenable_bits);
__IO_REG32_BIT(CSI2C_CT6_CTRL3,             0x480BDD4C,__READ_WRITE ,__csi2_ctx_ctrl3_bits);
__IO_REG32_BIT(CSI2C_CT7_CTRL1,             0x480BDD50,__READ_WRITE ,__csi2_ctx_ctrl1_bits);
__IO_REG32_BIT(CSI2C_CT7_CTRL2,             0x480BDD54,__READ_WRITE ,__csi2_ctx_ctrl2_bits);
__IO_REG32_BIT(CSI2C_CT7_DAT_OFST,          0x480BDD58,__READ_WRITE ,__csi2_ctx_dat_ofst_bits);
__IO_REG32(    CSI2C_CT7_DAT_PING_ADDR,     0x480BDD5C,__READ_WRITE );
__IO_REG32(    CSI2C_CT7_DAT_PONG_ADDR,     0x480BDD60,__READ_WRITE );
__IO_REG32_BIT(CSI2C_CT7_IRQENABLE,         0x480BDD64,__READ_WRITE ,__csi2_ctx_irqenable_bits);
__IO_REG32_BIT(CSI2C_CT7_IRQSTATUS,         0x480BDD68,__READ_WRITE ,__csi2_ctx_irqenable_bits);
__IO_REG32_BIT(CSI2C_CT7_CTRL3,             0x480BDD6C,__READ_WRITE ,__csi2_ctx_ctrl3_bits);
__IO_REG32_BIT(CSI2C_CT0_TRANSCODEH,        0x480BDDC0,__READ_WRITE ,__csi2_ctx_transcodeh_bits);
__IO_REG32_BIT(CSI2C_CT0_TRANSCODEV,        0x480BDDC4,__READ_WRITE ,__csi2_ctx_transcodev_bits);
__IO_REG32_BIT(CSI2C_CT1_TRANSCODEH,        0x480BDDC8,__READ_WRITE ,__csi2_ctx_transcodeh_bits);
__IO_REG32_BIT(CSI2C_CT1_TRANSCODEV,        0x480BDDCC,__READ_WRITE ,__csi2_ctx_transcodev_bits);
__IO_REG32_BIT(CSI2C_CT2_TRANSCODEH,        0x480BDDD0,__READ_WRITE ,__csi2_ctx_transcodeh_bits);
__IO_REG32_BIT(CSI2C_CT2_TRANSCODEV,        0x480BDDD4,__READ_WRITE ,__csi2_ctx_transcodev_bits);
__IO_REG32_BIT(CSI2C_CT3_TRANSCODEH,        0x480BDDD8,__READ_WRITE ,__csi2_ctx_transcodeh_bits);
__IO_REG32_BIT(CSI2C_CT3_TRANSCODEV,        0x480BDDDC,__READ_WRITE ,__csi2_ctx_transcodev_bits);
__IO_REG32_BIT(CSI2C_CT4_TRANSCODEH,        0x480BDDE0,__READ_WRITE ,__csi2_ctx_transcodeh_bits);
__IO_REG32_BIT(CSI2C_CT4_TRANSCODEV,        0x480BDDE4,__READ_WRITE ,__csi2_ctx_transcodev_bits);
__IO_REG32_BIT(CSI2C_CT5_TRANSCODEH,        0x480BDDE8,__READ_WRITE ,__csi2_ctx_transcodeh_bits);
__IO_REG32_BIT(CSI2C_CT5_TRANSCODEV,        0x480BDDEC,__READ_WRITE ,__csi2_ctx_transcodev_bits);
__IO_REG32_BIT(CSI2C_CT6_TRANSCODEH,        0x480BDDF0,__READ_WRITE ,__csi2_ctx_transcodeh_bits);
__IO_REG32_BIT(CSI2C_CT6_TRANSCODEV,        0x480BDDF4,__READ_WRITE ,__csi2_ctx_transcodev_bits);
__IO_REG32_BIT(CSI2C_CT7_TRANSCODEH,        0x480BDDF8,__READ_WRITE ,__csi2_ctx_transcodeh_bits);
__IO_REG32_BIT(CSI2C_CT7_TRANSCODEV,        0x480BDDFC,__READ_WRITE ,__csi2_ctx_transcodev_bits);

/***************************************************************************
 **
 ** Camera CSIPHY1
 **
 ***************************************************************************/
__IO_REG32_BIT(CSIPHY1_REG0,                0x480BDD70,__READ_WRITE ,__csiphy_reg0_bits);
__IO_REG32_BIT(CSIPHY1_REG1,                0x480BDD74,__READ_WRITE ,__csiphy_reg1_bits);
__IO_REG32_BIT(CSIPHY1_REG2,                0x480BDD78,__READ_WRITE ,__csiphy_reg2_bits);

/***************************************************************************
 **
 ** Camera CSIPHY2
 **
 ***************************************************************************/
__IO_REG32_BIT(CSIPHY2_REG0,                0x480BD970,__READ_WRITE ,__csiphy_reg0_bits);
__IO_REG32_BIT(CSIPHY2_REG1,                0x480BD974,__READ_WRITE ,__csiphy_reg1_bits);
__IO_REG32_BIT(CSIPHY2_REG2,                0x480BD978,__READ_WRITE ,__csiphy_reg2_bits);

/***************************************************************************
 **
 ** DSS
 **
 ***************************************************************************/
__IO_REG32_BIT(DSS_REVISIONNUMBER,          0x48050000,__READ       ,__dss_revisionnumber_bits);
__IO_REG32_BIT(DSS_SYSCONFIG,               0x48050010,__READ_WRITE ,__dss_sysconfig_bits);
__IO_REG32_BIT(DSS_SYSSTATUS,               0x48050014,__READ       ,__dss_sysstatus_bits);
__IO_REG32_BIT(DSS_IRQSTATUS,               0x48050018,__READ       ,__dss_irqstatus_bits);
__IO_REG32_BIT(DSS_CONTROL,                 0x48050040,__READ_WRITE ,__dss_control_bits);
__IO_REG32_BIT(DSS_CLK_STATUS,              0x4805005C,__READ       ,__dss_clk_status_bits);

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
__IO_REG32_BIT(DSI_PHY_REGISTER0,						0x4804FE00,__READ_WRITE	,__dsi_phy_register0_bits);
__IO_REG32_BIT(DSI_PHY_REGISTER1,						0x4804FE04,__READ_WRITE	,__dsi_phy_register1_bits);
__IO_REG32_BIT(DSI_PHY_REGISTER2,						0x4804FE08,__READ_WRITE	,__dsi_phy_register2_bits);
__IO_REG32_BIT(DSI_PHY_REGISTER3,						0x4804FE0C,__READ_WRITE	,__dsi_phy_register3_bits);
__IO_REG32_BIT(DSI_PHY_REGISTER4,						0x4804FE10,__READ_WRITE	,__dsi_phy_register4_bits);
__IO_REG32_BIT(DSI_PHY_REGISTER5,						0x4804FE14,__READ				,__dsi_phy_register5_bits);

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
 ** SGX
 **
 ***************************************************************************/
__IO_REG32(    OCP_REVISION,    						0x5000FE00,__READ				);
__IO_REG32_BIT(OCP_HWINFO,    	  					0x5000FE04,__READ				,__ocp_hwinfo_bits);
__IO_REG32_BIT(OCP_SYSCONFIG,    						0x5000FE10,__READ_WRITE ,__ocp_sysconfig_bits);
__IO_REG32_BIT(OCP_IRQSTATUS_RAW_0, 				0x5000FE24,__READ_WRITE ,__ocp_irqstatus_raw_0_bits);
__IO_REG32_BIT(OCP_IRQSTATUS_RAW_1,  				0x5000FE28,__READ_WRITE ,__ocp_irqstatus_raw_1_bits);
__IO_REG32_BIT(OCP_IRQSTATUS_RAW_2,  				0x5000FE2C,__READ_WRITE ,__ocp_irqstatus_raw_2_bits);
__IO_REG32_BIT(OCP_IRQSTATUS_0,  			    	0x5000FE30,__READ_WRITE ,__ocp_irqstatus_0_bits);
__IO_REG32_BIT(OCP_IRQSTATUS_1,  			    	0x5000FE34,__READ_WRITE ,__ocp_irqstatus_1_bits);
__IO_REG32_BIT(OCP_IRQSTATUS_2,  			    	0x5000FE38,__READ_WRITE ,__ocp_irqstatus_2_bits);
__IO_REG32_BIT(OCP_IRQENABLE_SET_0,		    	0x5000FE3C,__READ_WRITE ,__ocp_irqenable_set_0_bits);
__IO_REG32_BIT(OCP_IRQENABLE_SET_1,		    	0x5000FE40,__READ_WRITE ,__ocp_irqenable_set_1_bits);
__IO_REG32_BIT(OCP_IRQENABLE_SET_2,		    	0x5000FE44,__READ_WRITE ,__ocp_irqenable_set_2_bits);
__IO_REG32_BIT(OCP_IRQENABLE_CLR_0,		    	0x5000FE48,__READ_WRITE ,__ocp_irqenable_clr_0_bits);
__IO_REG32_BIT(OCP_IRQENABLE_CLR_1,		    	0x5000FE4C,__READ_WRITE ,__ocp_irqenable_clr_1_bits);
__IO_REG32_BIT(OCP_IRQENABLE_CLR_2,		    	0x5000FE50,__READ_WRITE ,__ocp_irqenable_clr_2_bits);
__IO_REG32_BIT(OCP_PAGE_CONFIG,		        	0x5000FF00,__READ_WRITE ,__ocp_page_config_bits);
__IO_REG32_BIT(OCP_INTERRUPT_EVENT,		      0x5000FF04,__READ_WRITE ,__ocp_interrupt_event_bits);
__IO_REG32_BIT(OCP_DEBUG_CONFIG,		      	0x5000FF08,__READ_WRITE ,__ocp_debug_config_bits);
__IO_REG32_BIT(OCP_DEBUG_STATUS,	        	0x5000FF0C,__READ_WRITE ,__ocp_debug_status_bits);

/***************************************************************************
 **
 ** L3 IA
 **
 ***************************************************************************/
__IO_REG32_BIT(L3_IA_COMPONENT_MPUSS,  	    0x68001400,__READ       ,__l3_ia_component_bits);
__IO_REG32_BIT(L3_IA_CORE_L_MPUSS,      	  0x68001418,__READ       ,__l3_ia_core_l_bits);
__IO_REG32_BIT(L3_IA_CORE_H_MPUSS,      	  0x6800141C,__READ       ,__l3_ia_core_h_bits);
__IO_REG32_BIT(L3_IA_AGENT_CONTROL_MPUSS,  	0x68001420,__READ_WRITE ,__l3_ia_agent_control_bits);
__IO_REG32_BIT(L3_IA_AGENT_STATUS_MPUSS,   	0x68001428,__READ_WRITE ,__l3_ia_agent_status_bits);
__IO_REG32_BIT(L3_IA_ERROR_LOG_L_MPUSS,   	0x68001458,__READ_WRITE ,__l3_ia_error_log_l_bits);
__IO_REG32_BIT(L3_IA_ERROR_LOG_H_MPUSS,   	0x6800145C,__READ       ,__l3_ia_error_log_h_bits);
__IO_REG32(		 L3_IA_ERROR_LOG_ADDR_MPUSS,	0x68001460,__READ				);
__IO_REG32_BIT(L3_IA_COMPONENT_IVA2_2,      0x68001800,__READ       ,__l3_ia_component_bits);
__IO_REG32_BIT(L3_IA_CORE_L_IVA2_2,     	  0x68001818,__READ       ,__l3_ia_core_l_bits);
__IO_REG32_BIT(L3_IA_CORE_H_IVA2_2,     	  0x6800181C,__READ       ,__l3_ia_core_h_bits);
__IO_REG32_BIT(L3_IA_AGENT_CONTROL_IVA2_2, 	0x68001820,__READ_WRITE ,__l3_ia_agent_control_bits);
__IO_REG32_BIT(L3_IA_AGENT_STATUS_IVA2_2,  	0x68001828,__READ_WRITE ,__l3_ia_agent_status_bits);
__IO_REG32_BIT(L3_IA_ERROR_LOG_L_IVA2_2,  	0x68001858,__READ_WRITE ,__l3_ia_error_log_l_bits);
__IO_REG32_BIT(L3_IA_ERROR_LOG_H_IVA2_2,  	0x6800185C,__READ       ,__l3_ia_error_log_h_bits);
__IO_REG32(		 L3_IA_ERROR_LOG_ADDR_IVA2_2,	0x68001860,__READ				);
__IO_REG32_BIT(L3_IA_COMPONENT_SGX,    	    0x68001C00,__READ       ,__l3_ia_component_bits);
__IO_REG32_BIT(L3_IA_CORE_L_SGX,        	  0x68001C18,__READ       ,__l3_ia_core_l_bits);
__IO_REG32_BIT(L3_IA_CORE_H_SGX,        	  0x68001C1C,__READ       ,__l3_ia_core_h_bits);
__IO_REG32_BIT(L3_IA_AGENT_CONTROL_SGX,  		0x68001C20,__READ_WRITE ,__l3_ia_agent_control_bits);
__IO_REG32_BIT(L3_IA_AGENT_STATUS_SGX,   		0x68001C28,__READ_WRITE ,__l3_ia_agent_status_bits);
__IO_REG32_BIT(L3_IA_ERROR_LOG_L_SGX,   		0x68001C58,__READ_WRITE ,__l3_ia_error_log_l_bits);
__IO_REG32_BIT(L3_IA_ERROR_LOG_H_SGX,   		0x68001C5C,__READ       ,__l3_ia_error_log_h_bits);
__IO_REG32(		 L3_IA_ERROR_LOG_ADDR_SGX, 		0x68001C60,__READ				);
__IO_REG32_BIT(L3_IA_COMPONENT_SAD2D,  	    0x68003000,__READ       ,__l3_ia_component_bits);
__IO_REG32_BIT(L3_IA_CORE_L_SAD2D,      	  0x68003018,__READ       ,__l3_ia_core_l_bits);
__IO_REG32_BIT(L3_IA_CORE_H_SAD2D,      	  0x6800301C,__READ       ,__l3_ia_core_h_bits);
__IO_REG32_BIT(L3_IA_AGENT_CONTROL_SAD2D, 	0x68003020,__READ_WRITE ,__l3_ia_agent_control_bits);
__IO_REG32_BIT(L3_IA_AGENT_STATUS_SAD2D, 		0x68003028,__READ_WRITE ,__l3_ia_agent_status_bits);
__IO_REG32_BIT(L3_IA_ERROR_LOG_L_SAD2D,  		0x68003058,__READ_WRITE ,__l3_ia_error_log_l_bits);
__IO_REG32_BIT(L3_IA_ERROR_LOG_H_SAD2D,  		0x6800305C,__READ       ,__l3_ia_error_log_h_bits);
__IO_REG32(		 L3_IA_ERROR_LOG_ADDR_SAD2D,	0x68003060,__READ				);
__IO_REG32_BIT(L3_IA_COMPONENT_USB_HS_Host, 	    0x68004000,__READ       ,__l3_ia_component_bits);
__IO_REG32_BIT(L3_IA_CORE_L_USB_HS_Host,    	    0x68004018,__READ       ,__l3_ia_core_l_bits);
__IO_REG32_BIT(L3_IA_CORE_H_USB_HS_Host,     	    0x6800401C,__READ       ,__l3_ia_core_h_bits);
__IO_REG32_BIT(L3_IA_AGENT_CONTROL_USB_HS_Host,   0x68004020,__READ_WRITE ,__l3_ia_agent_control_bits);
__IO_REG32_BIT(L3_IA_AGENT_STATUS_USB_HS_Host,    0x68004028,__READ_WRITE ,__l3_ia_agent_status_bits);
__IO_REG32_BIT(L3_IA_ERROR_LOG_L_USB_HS_Host,     0x68004058,__READ_WRITE ,__l3_ia_error_log_l_bits);
__IO_REG32_BIT(L3_IA_ERROR_LOG_H_USB_HS_Host,     0x6800405C,__READ       ,__l3_ia_error_log_h_bits);
__IO_REG32(		 L3_IA_ERROR_LOG_ADDR_USB_HS_Host,  0x68004060,__READ				);
__IO_REG32_BIT(L3_IA_COMPONENT_USB_HS_OTG, 	      0x68004400,__READ       ,__l3_ia_component_bits);
__IO_REG32_BIT(L3_IA_CORE_L_USB_HS_OTG,     	    0x68004418,__READ       ,__l3_ia_core_l_bits);
__IO_REG32_BIT(L3_IA_CORE_H_USB_HS_OTG,    	      0x6800441C,__READ       ,__l3_ia_core_h_bits);
__IO_REG32_BIT(L3_IA_AGENT_CONTROL_USB_HS_OTG,    0x68004420,__READ_WRITE ,__l3_ia_agent_control_bits);
__IO_REG32_BIT(L3_IA_AGENT_STATUS_USB_HS_OTG,     0x68004428,__READ_WRITE ,__l3_ia_agent_status_bits);
__IO_REG32_BIT(L3_IA_ERROR_LOG_L_USB_HS_OTG,      0x68004458,__READ_WRITE ,__l3_ia_error_log_l_bits);
__IO_REG32_BIT(L3_IA_ERROR_LOG_H_USB_HS_OTG,      0x6800445C,__READ       ,__l3_ia_error_log_h_bits);
__IO_REG32(		 L3_IA_ERROR_LOG_ADDR_USB_HS_OTG,   0x68004460,__READ				);
__IO_REG32_BIT(L3_IA_COMPONENT_DMA_RD, 	    0x68004C00,__READ       ,__l3_ia_component_bits);
__IO_REG32_BIT(L3_IA_CORE_L_DMA_RD,     	  0x68004C18,__READ       ,__l3_ia_core_l_bits);
__IO_REG32_BIT(L3_IA_CORE_H_DMA_RD,     	  0x68004C1C,__READ       ,__l3_ia_core_h_bits);
__IO_REG32_BIT(L3_IA_AGENT_CONTROL_DMA_RD,  0x68004C20,__READ_WRITE ,__l3_ia_agent_control_bits);
__IO_REG32_BIT(L3_IA_AGENT_STATUS_DMA_RD,  	0x68004C28,__READ_WRITE ,__l3_ia_agent_status_bits);
__IO_REG32_BIT(L3_IA_ERROR_LOG_L_DMA_RD,  	0x68004C58,__READ_WRITE ,__l3_ia_error_log_l_bits);
__IO_REG32_BIT(L3_IA_ERROR_LOG_H_DMA_RD,  	0x68004C5C,__READ       ,__l3_ia_error_log_h_bits);
__IO_REG32(		 L3_IA_ERROR_LOG_ADDR_DMA_RD, 0x68004C60,__READ				);
__IO_REG32_BIT(L3_IA_COMPONENT_DMA_WR, 	    0x68005000,__READ       ,__l3_ia_component_bits);
__IO_REG32_BIT(L3_IA_CORE_L_DMA_WR,      	  0x68005018,__READ       ,__l3_ia_core_l_bits);
__IO_REG32_BIT(L3_IA_CORE_H_DMA_WR,      	  0x6800501C,__READ       ,__l3_ia_core_h_bits);
__IO_REG32_BIT(L3_IA_AGENT_CONTROL_DMA_WR,  0x68005020,__READ_WRITE ,__l3_ia_agent_control_bits);
__IO_REG32_BIT(L3_IA_AGENT_STATUS_DMA_WR,  	0x68005028,__READ_WRITE ,__l3_ia_agent_status_bits);
__IO_REG32_BIT(L3_IA_ERROR_LOG_L_DMA_WR,  	0x68005058,__READ_WRITE ,__l3_ia_error_log_l_bits);
__IO_REG32_BIT(L3_IA_ERROR_LOG_H_DMA_WR,  	0x6800505C,__READ       ,__l3_ia_error_log_h_bits);
__IO_REG32(		 L3_IA_ERROR_LOG_ADDR_DMA_WR, 0x68005060,__READ				);
__IO_REG32_BIT(L3_IA_COMPONENT_DSS,   	    0x68005400,__READ       ,__l3_ia_component_bits);
__IO_REG32_BIT(L3_IA_CORE_L_DSS,      	    0x68005418,__READ       ,__l3_ia_core_l_bits);
__IO_REG32_BIT(L3_IA_CORE_H_DSS,      	    0x6800541C,__READ       ,__l3_ia_core_h_bits);
__IO_REG32_BIT(L3_IA_AGENT_CONTROL_DSS,  		0x68005420,__READ_WRITE ,__l3_ia_agent_control_bits);
__IO_REG32_BIT(L3_IA_AGENT_STATUS_DSS,  		0x68005428,__READ_WRITE ,__l3_ia_agent_status_bits);
__IO_REG32_BIT(L3_IA_ERROR_LOG_L_DSS, 			0x68005458,__READ_WRITE ,__l3_ia_error_log_l_bits);
__IO_REG32_BIT(L3_IA_ERROR_LOG_H_DSS, 			0x6800545C,__READ       ,__l3_ia_error_log_h_bits);
__IO_REG32(		 L3_IA_ERROR_LOG_ADDR_DSS, 		0x68005460,__READ				);
__IO_REG32_BIT(L3_IA_COMPONENT_CAM,  	      0x68005800,__READ       ,__l3_ia_component_bits);
__IO_REG32_BIT(L3_IA_CORE_L_CAM,      	    0x68005818,__READ       ,__l3_ia_core_l_bits);
__IO_REG32_BIT(L3_IA_CORE_H_CAM,      	    0x6800581C,__READ       ,__l3_ia_core_h_bits);
__IO_REG32_BIT(L3_IA_AGENT_CONTROL_CAM,  	  0x68005820,__READ_WRITE ,__l3_ia_agent_control_bits);
__IO_REG32_BIT(L3_IA_AGENT_STATUS_CAM,   	  0x68005828,__READ_WRITE ,__l3_ia_agent_status_bits);
__IO_REG32_BIT(L3_IA_ERROR_LOG_L_CAM,   		0x68005858,__READ_WRITE ,__l3_ia_error_log_l_bits);
__IO_REG32_BIT(L3_IA_ERROR_LOG_H_CAM,   		0x6800585C,__READ       ,__l3_ia_error_log_h_bits);
__IO_REG32(		 L3_IA_ERROR_LOG_ADDR_CAM,	  0x68005860,__READ				);
__IO_REG32_BIT(L3_IA_COMPONENT_DAP,  	      0x68005C00,__READ       ,__l3_ia_component_bits);
__IO_REG32_BIT(L3_IA_CORE_L_DAP,      	    0x68005C18,__READ       ,__l3_ia_core_l_bits);
__IO_REG32_BIT(L3_IA_CORE_H_DAP,      	    0x68005C1C,__READ       ,__l3_ia_core_h_bits);
__IO_REG32_BIT(L3_IA_AGENT_CONTROL_DAP,  		0x68005C20,__READ_WRITE ,__l3_ia_agent_control_bits);
__IO_REG32_BIT(L3_IA_AGENT_STATUS_DAP,  		0x68005C28,__READ_WRITE ,__l3_ia_agent_status_bits);
__IO_REG32_BIT(L3_IA_ERROR_LOG_L_DAP, 			0x68005C58,__READ_WRITE ,__l3_ia_error_log_l_bits);
__IO_REG32_BIT(L3_IA_ERROR_LOG_H_DAP, 			0x68005C5C,__READ       ,__l3_ia_error_log_h_bits);
__IO_REG32(		 L3_IA_ERROR_LOG_ADDR_DAP, 		0x68005C60,__READ				);

/***************************************************************************
 **
 ** L3 TA
 **
 ***************************************************************************/
__IO_REG32_BIT(L3_TA_COMPONENT_SMS,  	      0x68002000,__READ       ,__l3_ta_component_bits);
__IO_REG32_BIT(L3_TA_CORE_L_SMS,      	    0x68002018,__READ       ,__l3_ta_core_l_bits);
__IO_REG32_BIT(L3_TA_CORE_H_SMS,      	    0x6800201C,__READ       ,__l3_ta_core_h_bits);
__IO_REG32_BIT(L3_TA_AGENT_CONTROL_SMS,  		0x68002020,__READ_WRITE ,__l3_ta_agent_control_bits);
__IO_REG32_BIT(L3_TA_AGENT_STATUS_SMS,   		0x68002028,__READ				,__l3_ta_agent_status_bits);
__IO_REG32_BIT(L3_TA_ERROR_LOG_L_SMS,   		0x68002058,__READ_WRITE ,__l3_ta_error_log_l_bits);
__IO_REG32_BIT(L3_TA_ERROR_LOG_H_SMS,   		0x6800205C,__READ       ,__l3_ta_error_log_h_bits);
__IO_REG32(		 L3_TA_ERROR_LOG_ADDR_SMS, 		0x68002060,__READ				);
__IO_REG32_BIT(L3_TA_COMPONENT_GPMC,  	    0x68002400,__READ       ,__l3_ta_component_bits);
__IO_REG32_BIT(L3_TA_CORE_L_GPMC,      	    0x68002418,__READ       ,__l3_ta_core_l_bits);
__IO_REG32_BIT(L3_TA_CORE_H_GPMC,      	    0x6800241C,__READ       ,__l3_ta_core_h_bits);
__IO_REG32_BIT(L3_TA_AGENT_CONTROL_GPMC, 		0x68002420,__READ_WRITE ,__l3_ta_agent_control_bits);
__IO_REG32_BIT(L3_TA_AGENT_STATUS_GPMC,  		0x68002428,__READ				,__l3_ta_agent_status_bits);
__IO_REG32_BIT(L3_TA_ERROR_LOG_L_GPMC,  		0x68002458,__READ_WRITE ,__l3_ta_error_log_l_bits);
__IO_REG32_BIT(L3_TA_ERROR_LOG_H_GPMC,  		0x6800245C,__READ       ,__l3_ta_error_log_h_bits);
__IO_REG32(		 L3_TA_ERROR_LOG_ADDR_GPMC,		0x68002460,__READ				);
__IO_REG32_BIT(L3_TA_COMPONENT_OCM_RAM,     0x68002800,__READ       ,__l3_ta_component_bits);
__IO_REG32_BIT(L3_TA_CORE_L_OCM_RAM,        0x68002818,__READ       ,__l3_ta_core_l_bits);
__IO_REG32_BIT(L3_TA_CORE_H_OCM_RAM,        0x6800281C,__READ       ,__l3_ta_core_h_bits);
__IO_REG32_BIT(L3_TA_AGENT_CONTROL_OCM_RAM, 0x68002820,__READ_WRITE ,__l3_ta_agent_control_bits);
__IO_REG32_BIT(L3_TA_AGENT_STATUS_OCM_RAM,  0x68002828,__READ				,__l3_ta_agent_status_bits);
__IO_REG32_BIT(L3_TA_ERROR_LOG_L_OCM_RAM,  	0x68002858,__READ_WRITE ,__l3_ta_error_log_l_bits);
__IO_REG32_BIT(L3_TA_ERROR_LOG_H_OCM_RAM,  	0x6800285C,__READ       ,__l3_ta_error_log_h_bits);
__IO_REG32(		 L3_TA_ERROR_LOG_ADDR_OCM_RAM,0x68002860,__READ				);
__IO_REG32_BIT(L3_TA_COMPONENT_OCM_ROM,  	  0x68002C00,__READ       ,__l3_ta_component_bits);
__IO_REG32_BIT(L3_TA_CORE_L_OCM_ROM,        0x68002C18,__READ       ,__l3_ta_core_l_bits);
__IO_REG32_BIT(L3_TA_CORE_H_OCM_ROM,        0x68002C1C,__READ       ,__l3_ta_core_h_bits);
__IO_REG32_BIT(L3_TA_AGENT_CONTROL_OCM_ROM, 0x68002C20,__READ_WRITE ,__l3_ta_agent_control_bits);
__IO_REG32_BIT(L3_TA_AGENT_STATUS_OCM_ROM,  0x68002C28,__READ				,__l3_ta_agent_status_bits);
__IO_REG32_BIT(L3_TA_ERROR_LOG_L_OCM_ROM,  	0x68002C58,__READ_WRITE ,__l3_ta_error_log_l_bits);
__IO_REG32_BIT(L3_TA_ERROR_LOG_H_OCM_ROM,  	0x68002C5C,__READ       ,__l3_ta_error_log_h_bits);
__IO_REG32(		 L3_TA_ERROR_LOG_ADDR_OCM_ROM,0x68002C60,__READ				);
__IO_REG32_BIT(L3_TA_COMPONENT_MAD2D,  	    0x68003400,__READ       ,__l3_ta_component_bits);
__IO_REG32_BIT(L3_TA_CORE_L_MAD2D,      	  0x68003418,__READ       ,__l3_ta_core_l_bits);
__IO_REG32_BIT(L3_TA_CORE_H_MAD2D,      	  0x6800341C,__READ       ,__l3_ta_core_h_bits);
__IO_REG32_BIT(L3_TA_AGENT_CONTROL_MAD2D,  	0x68003420,__READ_WRITE ,__l3_ta_agent_control_bits);
__IO_REG32_BIT(L3_TA_AGENT_STATUS_MAD2D,   	0x68003428,__READ				,__l3_ta_agent_status_bits);
__IO_REG32_BIT(L3_TA_ERROR_LOG_L_MAD2D,   	0x68003458,__READ_WRITE ,__l3_ta_error_log_l_bits);
__IO_REG32_BIT(L3_TA_ERROR_LOG_H_MAD2D,   	0x6800345C,__READ       ,__l3_ta_error_log_h_bits);
__IO_REG32(		 L3_TA_ERROR_LOG_ADDR_MAD2D, 	0x68003460,__READ				);
__IO_REG32_BIT(L3_TA_COMPONENT_IVA2_2,  	  0x68006000,__READ       ,__l3_ta_component_bits);
__IO_REG32_BIT(L3_TA_CORE_L_IVA2_2,      	  0x68006018,__READ       ,__l3_ta_core_l_bits);
__IO_REG32_BIT(L3_TA_CORE_H_IVA2_2,      	  0x6800601C,__READ       ,__l3_ta_core_h_bits);
__IO_REG32_BIT(L3_TA_AGENT_CONTROL_IVA2_2, 	0x68006020,__READ_WRITE ,__l3_ta_agent_control_bits);
__IO_REG32_BIT(L3_TA_AGENT_STATUS_IVA2_2,   0x68006028,__READ				,__l3_ta_agent_status_bits);
__IO_REG32_BIT(L3_TA_ERROR_LOG_L_IVA2_2,   	0x68006058,__READ_WRITE ,__l3_ta_error_log_l_bits);
__IO_REG32_BIT(L3_TA_ERROR_LOG_H_IVA2_2,   	0x6800605C,__READ       ,__l3_ta_error_log_h_bits);
__IO_REG32(		 L3_TA_ERROR_LOG_ADDR_IVA2_2,	0x68006060,__READ				);
__IO_REG32_BIT(L3_TA_COMPONENT_SGX,  	      0x68006400,__READ       ,__l3_ta_component_bits);
__IO_REG32_BIT(L3_TA_CORE_L_SGX,      	    0x68006418,__READ       ,__l3_ta_core_l_bits);
__IO_REG32_BIT(L3_TA_CORE_H_SGX,      	    0x6800641C,__READ       ,__l3_ta_core_h_bits);
__IO_REG32_BIT(L3_TA_AGENT_CONTROL_SGX, 	 	0x68006420,__READ_WRITE ,__l3_ta_agent_control_bits);
__IO_REG32_BIT(L3_TA_AGENT_STATUS_SGX,   		0x68006428,__READ				,__l3_ta_agent_status_bits);
__IO_REG32_BIT(L3_TA_ERROR_LOG_L_SGX,   		0x68006458,__READ_WRITE ,__l3_ta_error_log_l_bits);
__IO_REG32_BIT(L3_TA_ERROR_LOG_H_SGX,   		0x6800645C,__READ       ,__l3_ta_error_log_h_bits);
__IO_REG32(		 L3_TA_ERROR_LOG_ADDR_SGX, 		0x68006460,__READ				);
__IO_REG32_BIT(L3_TA_COMPONENT_L4_CORE,     0x68006800,__READ       ,__l3_ta_component_bits);
__IO_REG32_BIT(L3_TA_CORE_L_L4_CORE,        0x68006818,__READ       ,__l3_ta_core_l_bits);
__IO_REG32_BIT(L3_TA_CORE_H_L4_CORE,        0x6800681C,__READ       ,__l3_ta_core_h_bits);
__IO_REG32_BIT(L3_TA_AGENT_CONTROL_L4_CORE, 0x68006820,__READ_WRITE ,__l3_ta_agent_control_bits);
__IO_REG32_BIT(L3_TA_AGENT_STATUS_L4_CORE,  0x68006828,__READ				,__l3_ta_agent_status_bits);
__IO_REG32_BIT(L3_TA_ERROR_LOG_L_L4_CORE,  	0x68006858,__READ_WRITE ,__l3_ta_error_log_l_bits);
__IO_REG32_BIT(L3_TA_ERROR_LOG_H_L4_CORE,  	0x6800685C,__READ       ,__l3_ta_error_log_h_bits);
__IO_REG32(		 L3_TA_ERROR_LOG_ADDR_L4_CORE,0x68006860,__READ				);
__IO_REG32_BIT(L3_TA_COMPONENT_L4_PER,      0x68006C00,__READ       ,__l3_ta_component_bits);
__IO_REG32_BIT(L3_TA_CORE_L_L4_PER,         0x68006C18,__READ       ,__l3_ta_core_l_bits);
__IO_REG32_BIT(L3_TA_CORE_H_L4_PER,         0x68006C1C,__READ       ,__l3_ta_core_h_bits);
__IO_REG32_BIT(L3_TA_AGENT_CONTROL_L4_PER, 	0x68006C20,__READ_WRITE ,__l3_ta_agent_control_bits);
__IO_REG32_BIT(L3_TA_AGENT_STATUS_L4_PER, 	0x68006C28,__READ				,__l3_ta_agent_status_bits);
__IO_REG32_BIT(L3_TA_ERROR_LOG_L_L4_PER,   	0x68006C58,__READ_WRITE ,__l3_ta_error_log_l_bits);
__IO_REG32_BIT(L3_TA_ERROR_LOG_H_L4_PER,   	0x68006C5C,__READ       ,__l3_ta_error_log_h_bits);
__IO_REG32(		 L3_TA_ERROR_LOG_ADDR_L4_PER,	0x68006C60,__READ				);
__IO_REG32_BIT(L3_TA_COMPONENT_L4_EMU,      0x68007000,__READ       ,__l3_ta_component_bits);
__IO_REG32_BIT(L3_TA_CORE_L_L4_EMU,         0x68007018,__READ       ,__l3_ta_core_l_bits);
__IO_REG32_BIT(L3_TA_CORE_H_L4_EMU,         0x6800701C,__READ       ,__l3_ta_core_h_bits);
__IO_REG32_BIT(L3_TA_AGENT_CONTROL_L4_EMU, 	0x68007020,__READ_WRITE ,__l3_ta_agent_control_bits);
__IO_REG32_BIT(L3_TA_AGENT_STATUS_L4_EMU, 	0x68007028,__READ				,__l3_ta_agent_status_bits);
__IO_REG32_BIT(L3_TA_ERROR_LOG_L_L4_EMU,   	0x68007058,__READ_WRITE ,__l3_ta_error_log_l_bits);
__IO_REG32_BIT(L3_TA_ERROR_LOG_H_L4_EMU,   	0x6800705C,__READ       ,__l3_ta_error_log_h_bits);
__IO_REG32(		 L3_TA_ERROR_LOG_ADDR_L4_EMU,	0x68007060,__READ				);

/***************************************************************************
 **
 ** L3 RT
 **
 ***************************************************************************/
__IO_REG32_BIT(L3_RT_COMPONENT,  						0x68000000,__READ				,__l3_rt_component_bits);
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
__IO_REG32_BIT(L3_PM_ERROR_LOG_MAD2D,  						  0x68013020,__READ_WRITE	,__l3_pm_error_log_bits);
__IO_REG32_BIT(L3_PM_CONTROL_MAD2D,  							  0x68013028,__READ_WRITE	,__l3_pm_control_bits);
__IO_REG32_BIT(L3_PM_ERROR_CLEAR_SINGLE_MAD2D,  	  0x68013030,__READ				,__l3_pm_error_clear_single_bits);
__IO_REG32_BIT(L3_PM_ERROR_CLEAR_MULTI_MAD2D,  		  0x68013038,__READ				,__l3_pm_error_clear_single_bits);
__IO_REG32_BIT(L3_PM_REQ_INFO_PERMISSION_0_MAD2D,	  0x68013048,__READ_WRITE	,__l3_pm_req_info_permission_bits);
__IO_REG32_BIT(L3_PM_REQ_INFO_PERMISSION_1_MAD2D,	  0x68013068,__READ_WRITE	,__l3_pm_req_info_permission_bits);
__IO_REG32_BIT(L3_PM_REQ_INFO_PERMISSION_2_MAD2D,	  0x68013088,__READ_WRITE	,__l3_pm_req_info_permission_bits);
__IO_REG32_BIT(L3_PM_REQ_INFO_PERMISSION_3_MAD2D,	  0x680130A8,__READ_WRITE	,__l3_pm_req_info_permission_bits);
__IO_REG32_BIT(L3_PM_REQ_INFO_PERMISSION_4_MAD2D,	  0x680130C8,__READ_WRITE	,__l3_pm_req_info_permission_bits);
__IO_REG32_BIT(L3_PM_REQ_INFO_PERMISSION_5_MAD2D,	  0x680130E8,__READ_WRITE	,__l3_pm_req_info_permission_bits);
__IO_REG32_BIT(L3_PM_REQ_INFO_PERMISSION_6_MAD2D,	  0x68013108,__READ_WRITE	,__l3_pm_req_info_permission_bits);
__IO_REG32_BIT(L3_PM_REQ_INFO_PERMISSION_7_MAD2D,	  0x68013128,__READ_WRITE	,__l3_pm_req_info_permission_bits);
__IO_REG32_BIT(L3_PM_READ_PERMISSION_0_MAD2D,  		  0x68013050,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_READ_PERMISSION_1_MAD2D,  		  0x68013070,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_READ_PERMISSION_2_MAD2D,  		  0x68013090,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_READ_PERMISSION_3_MAD2D,  		  0x680130B0,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_READ_PERMISSION_4_MAD2D,  		  0x680130D0,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_READ_PERMISSION_5_MAD2D,  		  0x680130F0,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_READ_PERMISSION_6_MAD2D,  		  0x68013110,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_READ_PERMISSION_7_MAD2D,  		  0x68013130,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_WRITE_PERMISSION_0_MAD2D,  	  0x68013058,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_WRITE_PERMISSION_1_MAD2D, 		  0x68013078,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_WRITE_PERMISSION_2_MAD2D, 		  0x68013098,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_WRITE_PERMISSION_3_MAD2D, 		  0x680130B8,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_WRITE_PERMISSION_4_MAD2D, 		  0x680130D8,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_WRITE_PERMISSION_5_MAD2D, 		  0x680130F8,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_WRITE_PERMISSION_6_MAD2D, 		  0x68013118,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_WRITE_PERMISSION_7_MAD2D, 		  0x68013138,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_ADDR_MATCH_1_MAD2D,					  0x68013060,__READ_WRITE	,__l3_pm_addr_match_bits);
__IO_REG32_BIT(L3_PM_ADDR_MATCH_2_MAD2D,					  0x68013080,__READ_WRITE	,__l3_pm_addr_match_bits);
__IO_REG32_BIT(L3_PM_ADDR_MATCH_3_MAD2D,					  0x680130A0,__READ_WRITE	,__l3_pm_addr_match_bits);
__IO_REG32_BIT(L3_PM_ADDR_MATCH_4_MAD2D,					  0x680130C0,__READ_WRITE	,__l3_pm_addr_match_bits);
__IO_REG32_BIT(L3_PM_ADDR_MATCH_5_MAD2D,					  0x680130E0,__READ_WRITE	,__l3_pm_addr_match_bits);
__IO_REG32_BIT(L3_PM_ADDR_MATCH_6_MAD2D,					  0x68013100,__READ_WRITE	,__l3_pm_addr_match_bits);
__IO_REG32_BIT(L3_PM_ADDR_MATCH_7_MAD2D,					  0x68013120,__READ_WRITE	,__l3_pm_addr_match_bits);
__IO_REG32_BIT(L3_PM_ERROR_LOG_IVA2_2, 		 					0x68014020,__READ_WRITE	,__l3_pm_error_log_bits);
__IO_REG32_BIT(L3_PM_CONTROL_IVA2_2,  							0x68014028,__READ_WRITE	,__l3_pm_control_bits);
__IO_REG32_BIT(L3_PM_ERROR_CLEAR_SINGLE_IVA2_2,  		0x68014030,__READ				,__l3_pm_error_clear_single_bits);
__IO_REG32_BIT(L3_PM_ERROR_CLEAR_MULTI_IVA2_2,  		0x68014038,__READ				,__l3_pm_error_clear_single_bits);
__IO_REG32_BIT(L3_PM_REQ_INFO_PERMISSION_0_IVA2_2,	0x68014048,__READ_WRITE	,__l3_pm_req_info_permission_bits);
__IO_REG32_BIT(L3_PM_REQ_INFO_PERMISSION_1_IVA2_2,	0x68014068,__READ_WRITE	,__l3_pm_req_info_permission_bits);
__IO_REG32_BIT(L3_PM_REQ_INFO_PERMISSION_2_IVA2_2,	0x68014088,__READ_WRITE	,__l3_pm_req_info_permission_bits);
__IO_REG32_BIT(L3_PM_REQ_INFO_PERMISSION_3_IVA2_2,	0x680140A8,__READ_WRITE	,__l3_pm_req_info_permission_bits);
__IO_REG32_BIT(L3_PM_READ_PERMISSION_0_IVA2_2,  		0x68014050,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_READ_PERMISSION_1_IVA2_2,  		0x68014070,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_READ_PERMISSION_2_IVA2_2,  		0x68014090,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_READ_PERMISSION_3_IVA2_2,  		0x680140B0,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_WRITE_PERMISSION_0_IVA2_2,		  0x68014058,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_WRITE_PERMISSION_1_IVA2_2,		 	0x68014078,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_WRITE_PERMISSION_2_IVA2_2,		 	0x68014098,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_WRITE_PERMISSION_3_IVA2_2,		 	0x680140B8,__READ_WRITE	,__l3_pm_read_permission_bits);
__IO_REG32_BIT(L3_PM_ADDR_MATCH_1_IVA2_2,						0x68014060,__READ_WRITE	,__l3_pm_addr_match_bits);
__IO_REG32_BIT(L3_PM_ADDR_MATCH_2_IVA2_2,						0x68014080,__READ_WRITE	,__l3_pm_addr_match_bits);
__IO_REG32_BIT(L3_PM_ADDR_MATCH_3_IVA2_2,						0x680140A0,__READ_WRITE	,__l3_pm_addr_match_bits);

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
__IO_REG32_BIT(L4_IA_COMPONENT_L_CORE,	    				0x48040800,__READ     	,__l4_ia_component_l_bits);
__IO_REG32_BIT(L4_IA_COMPONENT_H_CORE,	    				0x48040804,__READ     	,__l4_ia_component_h_bits);
__IO_REG32_BIT(L4_IA_CORE_L_CORE,         					0x48040818,__READ_WRITE	,__l4_ia_core_l_bits);
__IO_REG32_BIT(L4_IA_CORE_H_CORE,         					0x4804081C,__READ_WRITE	,__l4_ia_core_h_bits);
__IO_REG32_BIT(L4_IA_AGENT_CONTROL_L_CORE,					0x48040820,__READ_WRITE	,__l4_ia_agent_control_l_bits);
__IO_REG32_BIT(L4_IA_AGENT_STATUS_L_CORE,						0x48040828,__READ_WRITE	,__l4_ia_agent_control_l_bits);
__IO_REG32_BIT(L4_IA_ERROR_LOG_L_CORE,  						0x48040858,__READ_WRITE	,__l4_ia_error_log_l_bits);
__IO_REG32_BIT(L4_IA_COMPONENT_L_PER,	    				  0x49000800,__READ     	,__l4_ia_component_l_bits);
__IO_REG32_BIT(L4_IA_COMPONENT_H_PER,	    				  0x49000804,__READ     	,__l4_ia_component_h_bits);
__IO_REG32_BIT(L4_IA_CORE_L_PER,         					  0x49000818,__READ_WRITE	,__l4_ia_core_l_bits);
__IO_REG32_BIT(L4_IA_CORE_H_PER,         					  0x4900081C,__READ_WRITE	,__l4_ia_core_h_bits);
__IO_REG32_BIT(L4_IA_AGENT_CONTROL_L_PER,						0x49000820,__READ_WRITE	,__l4_ia_agent_control_l_bits);
__IO_REG32_BIT(L4_IA_AGENT_STATUS_L_PER,						0x49000828,__READ_WRITE	,__l4_ia_agent_control_l_bits);
__IO_REG32_BIT(L4_IA_ERROR_LOG_L_PER,  							0x49000858,__READ_WRITE	,__l4_ia_error_log_l_bits);
__IO_REG32_BIT(L4_IA_COMPONENT_L_EMU,	    				  0x54006800,__READ     	,__l4_ia_component_l_bits);
__IO_REG32_BIT(L4_IA_COMPONENT_H_EMU,	    				  0x54006804,__READ     	,__l4_ia_component_h_bits);
__IO_REG32_BIT(L4_IA_CORE_L_EMU,         					  0x54006818,__READ_WRITE	,__l4_ia_core_l_bits);
__IO_REG32_BIT(L4_IA_CORE_H_EMU,         					  0x5400681C,__READ_WRITE	,__l4_ia_core_h_bits);
__IO_REG32_BIT(L4_IA_AGENT_CONTROL_L_EMU,						0x54006820,__READ_WRITE	,__l4_ia_agent_control_l_bits);
__IO_REG32_BIT(L4_IA_AGENT_STATUS_L_EMU,						0x54006828,__READ_WRITE	,__l4_ia_agent_control_l_bits);
__IO_REG32_BIT(L4_IA_ERROR_LOG_L_EMU,  							0x54006858,__READ_WRITE	,__l4_ia_error_log_l_bits);
__IO_REG32_BIT(L4_IA_COMPONENT_L_EMU_IA_DAP,	    	0x54008000,__READ     	,__l4_ia_component_l_bits);
__IO_REG32_BIT(L4_IA_COMPONENT_H_EMU_IA_DAP,	    	0x54008004,__READ     	,__l4_ia_component_h_bits);
__IO_REG32_BIT(L4_IA_CORE_L_EMU_IA_DAP,         		0x54008018,__READ_WRITE	,__l4_ia_core_l_bits);
__IO_REG32_BIT(L4_IA_CORE_H_EMU_IA_DAP,         		0x5400801C,__READ_WRITE	,__l4_ia_core_h_bits);
__IO_REG32_BIT(L4_IA_AGENT_CONTROL_L_EMU_IA_DAP,		0x54008020,__READ_WRITE	,__l4_ia_agent_control_l_bits);
__IO_REG32_BIT(L4_IA_AGENT_STATUS_L_EMU_IA_DAP,			0x54008028,__READ_WRITE	,__l4_ia_agent_control_l_bits);
__IO_REG32_BIT(L4_IA_ERROR_LOG_L_EMU_IA_DAP,  			0x54008058,__READ_WRITE	,__l4_ia_error_log_l_bits);
__IO_REG32_BIT(L4_IA_COMPONENT_L_WKUP_IA_EMU,	 			0x48328800,__READ     	,__l4_ia_component_l_bits);
__IO_REG32_BIT(L4_IA_COMPONENT_H_WKUP_IA_EMU,	 			0x48328804,__READ     	,__l4_ia_component_h_bits);
__IO_REG32_BIT(L4_IA_CORE_L_WKUP_IA_EMU,       			0x48328818,__READ_WRITE	,__l4_ia_core_l_bits);
__IO_REG32_BIT(L4_IA_CORE_H_WKUP_IA_EMU,       			0x4832881C,__READ_WRITE	,__l4_ia_core_h_bits);
__IO_REG32_BIT(L4_IA_AGENT_CONTROL_L_WKUP_IA_EMU,		0x48328820,__READ_WRITE	,__l4_ia_agent_control_l_bits);
__IO_REG32_BIT(L4_IA_AGENT_STATUS_L_WKUP_IA_EMU,		0x48328828,__READ_WRITE	,__l4_ia_agent_control_l_bits);
__IO_REG32_BIT(L4_IA_ERROR_LOG_L_WKUP_IA_EMU,  			0x48328858,__READ_WRITE	,__l4_ia_error_log_l_bits);
__IO_REG32_BIT(L4_IA_COMPONENT_L_WKUP_IA_CORE,	    0x4832A000,__READ     	,__l4_ia_component_l_bits);
__IO_REG32_BIT(L4_IA_COMPONENT_H_WKUP_IA_CORE,	    0x4832A004,__READ     	,__l4_ia_component_h_bits);
__IO_REG32_BIT(L4_IA_CORE_L_WKUP_IA_CORE,         	0x4832A018,__READ_WRITE	,__l4_ia_core_l_bits);
__IO_REG32_BIT(L4_IA_CORE_H_WKUP_IA_CORE,         	0x4832A01C,__READ_WRITE	,__l4_ia_core_h_bits);
__IO_REG32_BIT(L4_IA_AGENT_CONTROL_L_WKUP_IA_CORE,	0x4832A020,__READ_WRITE	,__l4_ia_agent_control_l_bits);
__IO_REG32_BIT(L4_IA_AGENT_STATUS_L_WKUP_IA_CORE,		0x4832A028,__READ_WRITE	,__l4_ia_agent_control_l_bits);
__IO_REG32_BIT(L4_IA_ERROR_LOG_L_WKUP_IA_CORE,  		0x4832A058,__READ_WRITE	,__l4_ia_error_log_l_bits);

/***************************************************************************
 **
 ** L4 TA
 **
 ***************************************************************************/
__IO_REG32_BIT(L4_TA_COMPONENT_L_CORE_TA_CONTROL,			    0x48003000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_CORE_TA_CONTROL,			        0x48003018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_CORE_TA_CONTROL,			        0x4800301C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_CONTROL,			0x48003020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_CONTROL,			0x48003024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_CONTROL,			0x48003028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_CORE_TA_CM,			        0x48027000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_CORE_TA_CM,			              0x48027018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_CORE_TA_CM,			              0x4802701C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_CM,					0x48027020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_CM,					0x48027024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_CM,						0x48027028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_CORE_TA_DISPLAY_SS,			0x48051000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_CORE_TA_DISPLAY_SS,			      0x48051018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_CORE_TA_DISPLAY_SS,			      0x4805101C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_DISPLAY_SS,	0x48051020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_DISPLAY_SS,	0x48051024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_DISPLAY_SS,		0x48051028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_CORE_TA_SDMA,			      0x48057000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_CORE_TA_SDMA,			            0x48057018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_CORE_TA_SDMA,			            0x4805701C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_SDMA,				0x48057020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_SDMA,				0x48057024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_SDMA,					0x48057028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_CORE_TA_I2C3,			      0x48061000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_CORE_TA_I2C3,			            0x48061018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_CORE_TA_I2C3,			            0x4806101C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_I2C3,				0x48061020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_I2C3,				0x48061024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_I2C3,					0x48061028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_CORE_TA_USB_TLL,		      0x48063000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_CORE_TA_USB_TLL,			        0x48063018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_CORE_TA_USB_TLL,			        0x4806301C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_USB_TLL,			0x48063020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_USB_TLL,			0x48063024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_USB_TLL,			0x48063028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_CORE_TA_USB_Host,	      0x48065000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_CORE_TA_USB_Host,			        0x48065018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_CORE_TA_USB_Host,			        0x4806501C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_USB_HS_Host,	0x48065020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_USB_HS_Host,	0x48065024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_USB_HS_Host,	0x48065028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_CORE_TA_UART1,	          0x4806B000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_CORE_TA_UART1,			          0x4806B018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_CORE_TA_UART1,			          0x4806B01C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_UART1,				0x4806B020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_UART1,				0x4806B024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_UART1,				0x4806B028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_CORE_TA_UART2,	          0x4806D000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_CORE_TA_UART2,			          0x4806D018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_CORE_TA_UART2,			          0x4806D01C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_UART2,				0x4806D020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_UART2,				0x4806D024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_UART2,				0x4806D028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_CORE_TA_I2C1,	          0x48071000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_CORE_TA_I2C1,			            0x48071018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_CORE_TA_I2C1,			            0x4807101C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_I2C1,				0x48071020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_I2C1,				0x48071024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_I2C1,					0x48071028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_CORE_TA_I2C2,	          0x48073000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_CORE_TA_I2C2,			            0x48073018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_CORE_TA_I2C2,			            0x4807301C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_I2C2,				0x48073020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_I2C2,				0x48073024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_I2C2,					0x48073028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_CORE_TA_MCBSP1,	        0x48075000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_CORE_TA_MCBSP1,			          0x48075018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_CORE_TA_MCBSP1,			          0x4807501C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_MCBSP1,			0x48075020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_MCBSP1,			0x48075024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_MCBSP1,				0x48075028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_CORE_TA_GPTIMER10,	      0x48087000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_CORE_TA_GPTIMER10,			      0x48087018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_CORE_TA_GPTIMER10,			      0x4808701C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_GPTIMER10,		0x48087020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_GPTIMER10,		0x48087024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_GPTIMER10,		0x48087028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_CORE_TA_GPTIMER11,	      0x48089000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_CORE_TA_GPTIMER11,			      0x48089018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_CORE_TA_GPTIMER11,			      0x4808901C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_GPTIMER11,		0x48089020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_GPTIMER11,		0x48089024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_GPTIMER11,		0x48089028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_CORE_TA_MAILBOX, 	      0x48095000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_CORE_TA_MAILBOX,			        0x48095018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_CORE_TA_MAILBOX,			        0x4809501C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_MAILBOX,		  0x48095020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_MAILBOX,		  0x48095024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_MAILBOX,		  0x48095028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_CORE_TA_MCBSP5, 	        0x48097000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_CORE_TA_MCBSP5,			          0x48097018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_CORE_TA_MCBSP5,			          0x4809701C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_MCBSP5,			0x48097020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_MCBSP5,			0x48097024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_MCBSP5,				0x48097028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_CORE_TA_MCSPI1, 	        0x48099000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_CORE_TA_MCSPI1,			          0x48099018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_CORE_TA_MCSPI1,			          0x4809901C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_MCSPI1,			0x48099020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_MCSPI1,			0x48099024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_MCSPI1,				0x48099028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_CORE_TA_MCSPI2, 	        0x4809B000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_CORE_TA_MCSPI2,			          0x4809B018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_CORE_TA_MCSPI2,			          0x4809B01C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_MCSPI2,			0x4809B020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_MCSPI2,			0x4809B024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_MCSPI2,				0x4809B028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_CORE_TA_MMCHS1, 	        0x4809D000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_CORE_TA_MMCHS1,			          0x4809D018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_CORE_TA_MMCHS1,			          0x4809D01C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_MMCHS1,			0x4809D020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_MMCHS1,			0x4809D024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_MMCHS1,				0x4809D028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_CORE_TA_USB_HS_OTG,      0x480AC000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_CORE_TA_USB_HS_OTG,			      0x480AC018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_CORE_TA_USB_HS_OTG,			      0x480AC01C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_USB_HS_OTG,	0x480AC020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_USB_HS_OTG,	0x480AC024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_USB_HS_OTG,		0x480AC028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_CORE_TA_MMCHS3, 	        0x480AE000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_CORE_TA_MMCHS3,			          0x480AE018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_CORE_TA_MMCHS3,			          0x480AE01C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_MMCHS3,			0x480AE020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_MMCHS3,			0x480AE024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_MMCHS3,				0x480AE028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_CORE_TA_HDQ1W, 	        0x480B3000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_CORE_TA_HDQ1W,			          0x480B3018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_CORE_TA_HDQ1W,			          0x480B301C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_HDQ1W,				0x480B3020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_HDQ1W,				0x480B3024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_HDQ1W,				0x480B3028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_CORE_TA_MMCHS2, 	        0x480B5000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_CORE_TA_MMCHS2,			          0x480B5018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_CORE_TA_MMCHS2,			          0x480B501C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_MMCHS2,			0x480B5020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_MMCHS2,			0x480B5024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_MMCHS2,				0x480B5028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_CORE_TA_ICR, 	          0x480B7000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_CORE_TA_ICR,			            0x480B7018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_CORE_TA_ICR,			            0x480B701C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_ICR,			    0x480B7020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_ICR,			    0x480B7024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_ICR,				  0x480B7028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_CORE_TA_MCSPI3, 	        0x480B9000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_CORE_TA_MCSPI3,			          0x480B9018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_CORE_TA_MCSPI3,			          0x480B901C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_MCSPI3,			0x480B9020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_MCSPI3,			0x480B9024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_MCSPI3,				0x480B9028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_CORE_TA_MCSPI4, 	        0x480BB000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_CORE_TA_MCSPI4,			          0x480BB018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_CORE_TA_MCSPI4,			          0x480BB01C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_MCSPI4,			0x480BB020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_MCSPI4,			0x480BB024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_MCSPI4,				0x480BB028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_CORE_TA_CAMERA, 	        0x480C0000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_CORE_TA_CAMERA,			          0x480C0018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_CORE_TA_CAMERA,			          0x480C001C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_CAMERA,			0x480C0020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_CAMERA,			0x480C0024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_CAMERA,				0x480C0028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_CORE_TA_INTH, 	          0x480C8000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_CORE_TA_INTH,			            0x480C8018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_CORE_TA_INTH,			            0x480C801C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_INTH,				0x480C8020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_INTH,				0x480C8024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_INTH,					0x480C8028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_CORE_TA_SR1, 	          0x480CA000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_CORE_TA_SR1,			            0x480CA018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_CORE_TA_SR1,			            0x480CA01C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_SR1, 				0x480CA020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_SR1,	  			0x480CA024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_SR1,		  		0x480CA028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_CORE_TA_SR2, 	          0x480CC000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_CORE_TA_SR2,			            0x480CC018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_CORE_TA_SR2,			            0x480CC01C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_SR2, 				0x480CC020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_SR2,	  			0x480CC024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_SR2,		  		0x480CC028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_CORE_TA_ICR_MODEM, 	    0x480CE000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_CORE_TA_ICR_MODEM,			      0x480CE018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_CORE_TA_ICR_MODEM,			      0x480CE01C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_ICR_MODEM,	  0x480CE020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_ICR_MODEM,	  0x480CE024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_ICR_MODEM,	  0x480CE028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_CORE_TA_WKUP, 	          0x48340000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_CORE_TA_WKUP,			            0x48340018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_CORE_TA_WKUP,			            0x4834001C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_CORE_TA_WKUP,				0x48340020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_CORE_TA_WKUP,				0x48340024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_CORE_TA_WKUP,					0x48340028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_PER_TA_UART3, 	          0x49021000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_PER_TA_UART3,			            0x49021018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_PER_TA_UART3,			            0x4902101C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_UART3,				0x49021020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_UART3,				0x49021024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_UART3,					0x49021028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_PER_TA_MCBSP2, 	        0x49023000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_PER_TA_MCBSP2,			          0x49023018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_PER_TA_MCBSP2,			          0x4902301C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_MCBSP2,				0x49023020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_MCBSP2,				0x49023024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_MCBSP2,				0x49023028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_PER_TA_MCBSP3, 	        0x49025000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_PER_TA_MCBSP3,			          0x49025018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_PER_TA_MCBSP3,			          0x4902501C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_MCBSP3,				0x49025020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_MCBSP3,				0x49025024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_MCBSP3,				0x49025028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_PER_TA_MCBSP4, 	        0x49027000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_PER_TA_MCBSP4,			          0x49027018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_PER_TA_MCBSP4,			          0x4902701C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_MCBSP4,				0x49027020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_MCBSP4,				0x49027024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_MCBSP4,				0x49027028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_PER_TA_MCBSP2_SIDETONE, 	    0x49029000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_PER_TA_MCBSP2_SIDETONE,			      0x49029018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_PER_TA_MCBSP2_SIDETONE,			      0x4902901C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_MCBSP2_SIDETONE,	0x49029020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_MCBSP2_SIDETONE,	0x49029024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_MCBSP2_SIDETONE,	  0x49029028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_PER_TA_MCBSP3_SIDETONE, 	    0x4902B000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_PER_TA_MCBSP3_SIDETONE,			      0x4902B018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_PER_TA_MCBSP3_SIDETONE,			      0x4902B01C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_MCBSP3_SIDETONE,	0x4902B020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_MCBSP3_SIDETONE,	0x4902B024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_MCBSP3_SIDETONE,	  0x4902B028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_PER_TA_WDTIMER3, 	      0x49031000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_PER_TA_WDTIMER3,			        0x49031018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_PER_TA_WDTIMER3,			        0x4903101C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_WDTIMER3,			0x49031020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_WDTIMER3,			0x49031024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_WDTIMER3,			0x49031028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_PER_TA_GPTIMER2, 	      0x49033000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_PER_TA_GPTIMER2,			        0x49033018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_PER_TA_GPTIMER2,			        0x4903301C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_GPTIMER2,			0x49033020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_GPTIMER2,			0x49033024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_GPTIMER2,			0x49033028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_PER_TA_GPTIMER3, 	      0x49035000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_PER_TA_GPTIMER3,			        0x49035018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_PER_TA_GPTIMER3,			        0x4903501C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_GPTIMER3,			0x49035020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_GPTIMER3,			0x49035024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_GPTIMER3,			0x49035028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_PER_TA_GPTIMER4, 	      0x49037000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_PER_TA_GPTIMER4,			        0x49037018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_PER_TA_GPTIMER4,			        0x4903701C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_GPTIMER4,			0x49037020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_GPTIMER4,			0x49037024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_GPTIMER4,			0x49037028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_PER_TA_GPTIMER5, 	      0x49039000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_PER_TA_GPTIMER5,			        0x49039018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_PER_TA_GPTIMER5,			        0x4903901C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_GPTIMER5,			0x49039020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_GPTIMER5,			0x49039024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_GPTIMER5,			0x49039028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_PER_TA_GPTIMER6, 	      0x4903B000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_PER_TA_GPTIMER6,			        0x4903B018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_PER_TA_GPTIMER6,			        0x4903B01C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_GPTIMER6,			0x4903B020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_GPTIMER6,			0x4903B024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_GPTIMER6,			0x4903B028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_PER_TA_GPTIMER7, 	      0x4903D000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_PER_TA_GPTIMER7,			        0x4903D018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_PER_TA_GPTIMER7,			        0x4903D01C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_GPTIMER7,			0x4903D020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_GPTIMER7,			0x4903D024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_GPTIMER7,			0x4903D028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_PER_TA_GPTIMER8, 	      0x4903F000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_PER_TA_GPTIMER8,			        0x4903F018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_PER_TA_GPTIMER8,			        0x4903F01C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_GPTIMER8,			0x4903F020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_GPTIMER8,			0x4903F024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_GPTIMER8,			0x4903F028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_PER_TA_GPTIMER9, 	      0x49041000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_PER_TA_GPTIMER9,			        0x49041018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_PER_TA_GPTIMER9,			        0x4904101C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_GPTIMER9,			0x49041020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_GPTIMER9,			0x49041024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_GPTIMER9,			0x49041028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_PER_TA_UART4, 	          0x49043000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_PER_TA_UART4,			            0x49043018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_PER_TA_UART4,			            0x4904301C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_UART4,		    0x49043020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_UART4,		    0x49043024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_UART4,			    0x49043028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_PER_TA_GPIO2, 	          0x49051000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_PER_TA_GPIO2,			            0x49051018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_PER_TA_GPIO2,			            0x4905101C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_GPIO2,				0x49051020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_GPIO2,				0x49051024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_GPIO2,					0x49051028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_PER_TA_GPIO3, 	          0x49053000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_PER_TA_GPIO3,			            0x49053018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_PER_TA_GPIO3,			            0x4905301C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_GPIO3,				0x49053020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_GPIO3,				0x49053024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_GPIO3,					0x49053028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_PER_TA_GPIO4, 	          0x49055000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_PER_TA_GPIO4,			            0x49055018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_PER_TA_GPIO4,			            0x4905501C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_GPIO4,				0x49055020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_GPIO4,				0x49055024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_GPIO4,					0x49055028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_PER_TA_GPIO5, 	          0x49057000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_PER_TA_GPIO5,			            0x49057018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_PER_TA_GPIO5,			            0x4905701C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_GPIO5,				0x49057020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_GPIO5,				0x49057024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_GPIO5,					0x49057028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_PER_TA_GPIO6, 	          0x49059000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_PER_TA_GPIO6,			            0x49059018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_PER_TA_GPIO6,			            0x4905901C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_PER_TA_GPIO6,				0x49059020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_PER_TA_GPIO6,				0x49059024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_PER_TA_GPIO6,					0x49059028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_EMU_TA_TEST_TAP, 	      0x54005000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_EMU_TA_TEST_TAP,			        0x54005018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_EMU_TA_TEST_TAP,			        0x5400501C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_EMU_TA_TEST_TAP,	    0x54005020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_EMU_TA_TEST_TAP,	    0x54005024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_EMU_TA_TEST_TAP,	    0x54005028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_EMU_TA_MPU, 	            0x54018000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_EMU_TA_MPU,			              0x54018018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_EMU_TA_MPU,			              0x5401801C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_EMU_TA_MPU,					0x54018020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_EMU_TA_MPU,					0x54018024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_EMU_TA_MPU,						0x54018028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_EMU_TA_TPUI, 	          0x5401A000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_EMU_TA_TPUI,			            0x5401A018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_EMU_TA_TPUI,			            0x5401A01C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_EMU_TA_TPUI,					0x5401A020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_EMU_TA_TPUI,					0x5401A024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_EMU_TA_TPUI,					0x5401A028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_EMU_TA_ETB, 	            0x5401C000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_EMU_TA_ETB,			              0x5401C018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_EMU_TA_ETB,			              0x5401C01C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_EMU_TA_ETB,					0x5401C020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_EMU_TA_ETB,					0x5401C024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_EMU_TA_ETB,						0x5401C028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_EMU_TA_DAP, 	            0x5401E000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_EMU_TA_DAP,			              0x5401E018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_EMU_TA_DAP,			              0x5401E01C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_EMU_TA_DAP,	  			0x5401E020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_EMU_TA_DAP,		  		0x5401E024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_EMU_TA_DAP,				    0x5401E028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_EMU_TA_SDTI, 	          0x5401F000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_EMU_TA_SDTI,			            0x5401F018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_EMU_TA_SDTI,			            0x5401F01C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_EMU_TA_SDTI,					0x5401F020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_EMU_TA_SDTI,					0x5401F024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_EMU_TA_SDTI,					0x5401F028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_EMU_TA_L4WKUP, 	        0x54730000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_EMU_TA_L4WKUP,			          0x54730018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_EMU_TA_L4WKUP,			          0x5473001C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_EMU_TA_L4WKUP,				0x54730020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_EMU_TA_L4WKUP,				0x54730024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_EMU_TA_L4WKUP,				0x54730028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_WKUP_TA_PRM, 	          0x48309000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_WKUP_TA_PRM,			            0x48309018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_WKUP_TA_PRM,			            0x4830901C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_WKUP_TA_PRM,					0x48309020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_WKUP_TA_PRM,					0x48309024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_WKUP_TA_PRM,					0x48309028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_WKUP_TA_GPIO1, 	        0x48311000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_WKUP_TA_GPIO1,			          0x48311018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_WKUP_TA_GPIO1,			          0x4831101C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_WKUP_TA_GPIO1,				0x48311020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_WKUP_TA_GPIO1,				0x48311024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_WKUP_TA_GPIO1,				0x48311028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_WKUP_TA_WDTIMER2, 	      0x48315000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_WKUP_TA_WDTIMER2,			        0x48315018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_WKUP_TA_WDTIMER2,			        0x4831501C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_WKUP_TA_WDTIMER2,		0x48315020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_WKUP_TA_WDTIMER2,		0x48315024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_WKUP_TA_WDTIMER2,			0x48315028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_WKUP_TA_GPTIMER1, 	      0x48319000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_WKUP_TA_GPTIMER1,			        0x48319018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_WKUP_TA_GPTIMER1,			        0x4831901C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_WKUP_TA_GPTIMER1,		0x48319020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_WKUP_TA_GPTIMER1,		0x48319024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_WKUP_TA_GPTIMER1,			0x48319028,__READ_WRITE	,__l4_ta_agent_status_l_bits);
__IO_REG32_BIT(L4_TA_COMPONENT_L_WKUP_TA_SYNCTIMER32K, 	  0x48321000,__READ     	,__l4_ta_component_l_bits);
__IO_REG32_BIT(L4_TA_CORE_L_WKUP_TA_SYNCTIMER32K,			    0x48321018,__READ     	,__l4_ta_core_l_bits);
__IO_REG32_BIT(L4_TA_CORE_H_WKUP_TA_SYNCTIMER32K,			    0x4832101C,__READ     	,__l4_ta_core_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_L_WKUP_TA_SYNCTIMER32K,0x48321020,__READ_WRITE	,__l4_ta_agent_control_l_bits);
__IO_REG32_BIT(L4_TA_AGENT_CONTROL_H_WKUP_TA_SYNCTIMER32K,0x48321024,__READ_WRITE	,__l4_ta_agent_control_h_bits);
__IO_REG32_BIT(L4_TA_AGENT_STATUS_L_WKUP_TA_SYNCTIMER32K,	0x48321028,__READ_WRITE	,__l4_ta_agent_status_l_bits);

/***************************************************************************
 **
 ** L4 LA
 **
 ***************************************************************************/
__IO_REG32_BIT(L4_LA_COMPONENT_L_CORE, 	            0x48041000,__READ     	,__l4_la_component_l_bits);
__IO_REG32(    L4_LA_NETWORK_H_CORE,			          0x48041014,__READ     	);
__IO_REG32_BIT(L4_LA_INITIATOR_INFO_L_CORE,	        0x48041018,__READ     	,__l4_la_initiator_info_l_bits);
__IO_REG32_BIT(L4_LA_INITIATOR_INFO_H_CORE,	        0x4804101C,__READ     	,__l4_la_initiator_info_h_bits);
__IO_REG32_BIT(L4_LA_NETWORK_CONTROL_L_CORE,  		  0x48041020,__READ_WRITE ,__l4_la_network_control_l_bits);
__IO_REG32_BIT(L4_LA_NETWORK_CONTROL_H_CORE,  		  0x48041024,__READ_WRITE ,__l4_la_network_control_h_bits);
__IO_REG32_BIT(L4_LA_COMPONENT_L_PER, 	            0x49001000,__READ     	,__l4_la_component_l_bits);
__IO_REG32(    L4_LA_NETWORK_H_PER,			            0x49001014,__READ     	);
__IO_REG32_BIT(L4_LA_INITIATOR_INFO_L_PER,	        0x49001018,__READ     	,__l4_la_initiator_info_l_bits);
__IO_REG32_BIT(L4_LA_INITIATOR_INFO_H_PER,	        0x4900101C,__READ     	,__l4_la_initiator_info_h_bits);
__IO_REG32_BIT(L4_LA_NETWORK_CONTROL_L_PER,  		    0x49001020,__READ_WRITE ,__l4_la_network_control_l_bits);
__IO_REG32_BIT(L4_LA_NETWORK_CONTROL_H_PER,  		    0x49001024,__READ_WRITE ,__l4_la_network_control_h_bits);
__IO_REG32_BIT(L4_LA_COMPONENT_L_EMU, 	            0x54007000,__READ     	,__l4_la_component_l_bits);
__IO_REG32(    L4_LA_NETWORK_H_EMU,			            0x54007014,__READ     	);
__IO_REG32_BIT(L4_LA_INITIATOR_INFO_L_EMU,	        0x54007018,__READ     	,__l4_la_initiator_info_l_bits);
__IO_REG32_BIT(L4_LA_INITIATOR_INFO_H_EMU,	        0x5400701C,__READ     	,__l4_la_initiator_info_h_bits);
__IO_REG32_BIT(L4_LA_NETWORK_CONTROL_L_EMU,  		    0x54007020,__READ_WRITE ,__l4_la_network_control_l_bits);
__IO_REG32_BIT(L4_LA_NETWORK_CONTROL_H_EMU,  		    0x54007024,__READ_WRITE ,__l4_la_network_control_h_bits);
__IO_REG32_BIT(L4_LA_COMPONENT_L_WKUP, 	            0x48329000,__READ     	,__l4_la_component_l_bits);
__IO_REG32(    L4_LA_NETWORK_H_WKUP,			          0x48329014,__READ     	);
__IO_REG32_BIT(L4_LA_INITIATOR_INFO_L_WKUP,	        0x48329018,__READ     	,__l4_la_initiator_info_l_bits);
__IO_REG32_BIT(L4_LA_INITIATOR_INFO_H_WKUP,	        0x4832901C,__READ     	,__l4_la_initiator_info_h_bits);
__IO_REG32_BIT(L4_LA_NETWORK_CONTROL_L_WKUP,  		  0x48329020,__READ_WRITE ,__l4_la_network_control_l_bits);
__IO_REG32_BIT(L4_LA_NETWORK_CONTROL_H_WKUP,  		  0x48329024,__READ_WRITE ,__l4_la_network_control_h_bits);

/***************************************************************************
 **
 ** L4 AP CORE
 **
 ***************************************************************************/
__IO_REG32_BIT(L4_AP_COMPONENT_L_CORE_AP,									0x48040000,__READ       ,__l4_ap_component_l_bits);
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
 ** L4 AP PER
 **
 ***************************************************************************/
__IO_REG32_BIT(L4_AP_COMPONENT_L_PER_AP,									0x49000000,__READ       ,__l4_ap_component_l_bits);
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
 ** L4 AP EMU
 **
 ***************************************************************************/
__IO_REG32_BIT(L4_AP_COMPONENT_L_EMU_AP,									0x54006000,__READ       ,__l4_ap_component_l_bits);
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
 ** L4 AP WKUP
 **
 ***************************************************************************/
__IO_REG32_BIT(L4_AP_COMPONENT_L_WKUP_AP,									0x48328000,__READ       ,__l4_ap_component_l_bits);
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
 ** GPMC
 **
 ***************************************************************************/
__IO_REG32_BIT(GPMC_REVISION,     		0x6E000000,__READ       ,__gpmc_revision_bits);
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
 ** SMS
 **
 ***************************************************************************/
__IO_REG32_BIT(SMS_REVISION,       					0x6C000000,__READ				,__sms_revision_bits);
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
 ** SDRC
 **
 ***************************************************************************/
__IO_REG32_BIT(SDRC_REVISION,      					0x6D000000,__READ				,__sdrc_revision_bits);
__IO_REG32_BIT(SDRC_SYSCONFIG,     					0x6D000010,__READ_WRITE ,__sdrc_sysconfig_bits);
__IO_REG32_BIT(SDRC_SYSSTATUS,     					0x6D000014,__READ       ,__sdrc_sysstatus_bits);
__IO_REG32_BIT(SDRC_CS_CFG,     					  0x6D000040,__READ_WRITE ,__sdrc_cs_cfg_bits);
__IO_REG32_BIT(SDRC_SHARING,     					  0x6D000044,__READ_WRITE ,__sdrc_sharing_bits);
__IO_REG32(    SDRC_ERR_ADDR,     					0x6D000048,__READ       );
__IO_REG32_BIT(SDRC_ERR_TYPE,     					0x6D00004C,__READ_WRITE ,__sdrc_err_type_bits);
__IO_REG32_BIT(SDRC_DLLA_CTRL,     					0x6D000060,__READ_WRITE ,__sdrc_dlla_ctrl_bits);
__IO_REG32_BIT(SDRC_DLLA_STATUS,     				0x6D000064,__READ       ,__sdrc_dlla_status_bits);
__IO_REG32_BIT(SDRC_POWER_REG,     					0x6D000070,__READ_WRITE ,__sdrc_power_reg_bits);
__IO_REG32_BIT(SDRC_MCFG_0,     					  0x6D000080,__READ_WRITE ,__sdrc_mcfg_bits);
#define SDRC_MCFG_FIXED_0       SDRC_MCFG_0
#define SDRC_MCFG_FIXED_0_bit   SDRC_MCFG_0_bit.FIXED
__IO_REG32_BIT(SDRC_MR_0,     					    0x6D000084,__READ_WRITE ,__sdrc_mr_bits);
__IO_REG32_BIT(SDRC_EMR2_0,     		  			0x6D00008C,__READ_WRITE ,__sdrc_emr2_bits);
__IO_REG32_BIT(SDRC_ACTIM_CTRLA_0, 					0x6D00009C,__READ_WRITE ,__sdrc_actim_ctrla_bits);
__IO_REG32_BIT(SDRC_ACTIM_CTRLB_0, 					0x6D0000A0,__READ_WRITE ,__sdrc_actim_ctrlb_bits);
__IO_REG32_BIT(SDRC_RFR_CTRL_0,    					0x6D0000A4,__READ_WRITE ,__sdrc_rfr_ctrl_bits);
__IO_REG32_BIT(SDRC_MANUAL_0,     					0x6D0000A8,__READ_WRITE ,__sdrc_manual_bits);
__IO_REG32_BIT(SDRC_MCFG_1,     					  0x6D0000B0,__READ_WRITE ,__sdrc_mcfg_bits);
#define SDRC_MCFG_FIXED_1       SDRC_MCFG_1
#define SDRC_MCFG_FIXED_1_bit   SDRC_MCFG_1_bit.FIXED
__IO_REG32_BIT(SDRC_MR_1,     					    0x6D0000B4,__READ_WRITE ,__sdrc_mr_bits);
__IO_REG32_BIT(SDRC_EMR2_1,     		  			0x6D0000BC,__READ_WRITE ,__sdrc_emr2_bits);
__IO_REG32_BIT(SDRC_ACTIM_CTRLA_1, 					0x6D0000C4,__READ_WRITE ,__sdrc_actim_ctrla_bits);
__IO_REG32_BIT(SDRC_ACTIM_CTRLB_1, 					0x6D0000C8,__READ_WRITE ,__sdrc_actim_ctrlb_bits);
__IO_REG32_BIT(SDRC_RFR_CTRL_1,    					0x6D0000D4,__READ_WRITE ,__sdrc_rfr_ctrl_bits);
__IO_REG32_BIT(SDRC_MANUAL_1,     					0x6D0000D8,__READ_WRITE ,__sdrc_manual_bits);

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
__IO_REG32_BIT(DMA4_CDP0,       			0x480560D0,__READ_WRITE ,__dma4_cdp_bits);
__IO_REG32(    DMA4_CNDP0,       			0x480560D4,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCDN0,       			0x480560D8,__READ_WRITE ,__dma4_ccdn_bits);

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
__IO_REG32_BIT(DMA4_CDP1,       			0x48056130,__READ_WRITE ,__dma4_cdp_bits);
__IO_REG32(    DMA4_CNDP1,       			0x48056134,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCDN1,       			0x48056138,__READ_WRITE ,__dma4_ccdn_bits);

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
__IO_REG32_BIT(DMA4_CDP2,       			0x48056190,__READ_WRITE ,__dma4_cdp_bits);
__IO_REG32(    DMA4_CNDP2,       			0x48056194,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCDN2,       			0x48056198,__READ_WRITE ,__dma4_ccdn_bits);

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
__IO_REG32_BIT(DMA4_CDP3,       			0x480561F0,__READ_WRITE ,__dma4_cdp_bits);
__IO_REG32(    DMA4_CNDP3,       			0x480561F4,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCDN3,       			0x480561F8,__READ_WRITE ,__dma4_ccdn_bits);

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
__IO_REG32_BIT(DMA4_CDP4,       			0x48056250,__READ_WRITE ,__dma4_cdp_bits);
__IO_REG32(    DMA4_CNDP4,       			0x48056254,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCDN4,       			0x48056258,__READ_WRITE ,__dma4_ccdn_bits);

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
__IO_REG32_BIT(DMA4_CDP5,       			0x480562B0,__READ_WRITE ,__dma4_cdp_bits);
__IO_REG32(    DMA4_CNDP5,       			0x480562B4,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCDN5,       			0x480562B8,__READ_WRITE ,__dma4_ccdn_bits);

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
__IO_REG32_BIT(DMA4_CDP6,       			0x48056310,__READ_WRITE ,__dma4_cdp_bits);
__IO_REG32(    DMA4_CNDP6,       			0x48056314,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCDN6,       			0x48056318,__READ_WRITE ,__dma4_ccdn_bits);

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
__IO_REG32_BIT(DMA4_CDP7,       			0x48056370,__READ_WRITE ,__dma4_cdp_bits);
__IO_REG32(    DMA4_CNDP7,       			0x48056374,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCDN7,       			0x48056378,__READ_WRITE ,__dma4_ccdn_bits);

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
__IO_REG32_BIT(DMA4_CDP8,       			0x480563D0,__READ_WRITE ,__dma4_cdp_bits);
__IO_REG32(    DMA4_CNDP8,       			0x480563D4,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCDN8,       			0x480563D8,__READ_WRITE ,__dma4_ccdn_bits);

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
__IO_REG32_BIT(DMA4_CDP9,       			0x48056430,__READ_WRITE ,__dma4_cdp_bits);
__IO_REG32(    DMA4_CNDP9,       			0x48056434,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCDN9,       			0x48056438,__READ_WRITE ,__dma4_ccdn_bits);

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
__IO_REG32_BIT(DMA4_CDP10,       			0x48056490,__READ_WRITE ,__dma4_cdp_bits);
__IO_REG32(    DMA4_CNDP10,      			0x48056494,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCDN10,      			0x48056498,__READ_WRITE ,__dma4_ccdn_bits);

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
__IO_REG32_BIT(DMA4_CDP11,       			0x480564F0,__READ_WRITE ,__dma4_cdp_bits);
__IO_REG32(    DMA4_CNDP11,      			0x480564F4,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCDN11,      			0x480564F8,__READ_WRITE ,__dma4_ccdn_bits);

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
__IO_REG32_BIT(DMA4_CDP12,       			0x48056550,__READ_WRITE ,__dma4_cdp_bits);
__IO_REG32(    DMA4_CNDP12,      			0x48056554,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCDN12,      			0x48056558,__READ_WRITE ,__dma4_ccdn_bits);

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
__IO_REG32_BIT(DMA4_CDP13,       			0x480565B0,__READ_WRITE ,__dma4_cdp_bits);
__IO_REG32(    DMA4_CNDP13,      			0x480565B4,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCDN13,      			0x480565B8,__READ_WRITE ,__dma4_ccdn_bits);

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
__IO_REG32_BIT(DMA4_CDP14,       			0x48056610,__READ_WRITE ,__dma4_cdp_bits);
__IO_REG32(    DMA4_CNDP14,      			0x48056614,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCDN14,      			0x48056618,__READ_WRITE ,__dma4_ccdn_bits);

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
__IO_REG32_BIT(DMA4_CDP15,       			0x48056670,__READ_WRITE ,__dma4_cdp_bits);
__IO_REG32(    DMA4_CNDP15,      			0x48056674,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCDN15,      			0x48056678,__READ_WRITE ,__dma4_ccdn_bits);

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
__IO_REG32_BIT(DMA4_CDP16,       			0x480566D0,__READ_WRITE ,__dma4_cdp_bits);
__IO_REG32(    DMA4_CNDP16,      			0x480566D4,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCDN16,      			0x480566D8,__READ_WRITE ,__dma4_ccdn_bits);

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
__IO_REG32_BIT(DMA4_CDP17,       			0x48056730,__READ_WRITE ,__dma4_cdp_bits);
__IO_REG32(    DMA4_CNDP17,      			0x48056734,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCDN17,      			0x48056738,__READ_WRITE ,__dma4_ccdn_bits);

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
__IO_REG32_BIT(DMA4_CDP18,       			0x48056790,__READ_WRITE ,__dma4_cdp_bits);
__IO_REG32(    DMA4_CNDP18,      			0x48056794,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCDN18,      			0x48056798,__READ_WRITE ,__dma4_ccdn_bits);

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
__IO_REG32_BIT(DMA4_CDP19,       			0x480567F0,__READ_WRITE ,__dma4_cdp_bits);
__IO_REG32(    DMA4_CNDP19,      			0x480567F4,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCDN19,      			0x480567F8,__READ_WRITE ,__dma4_ccdn_bits);

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
__IO_REG32_BIT(DMA4_CDP20,       			0x48056850,__READ_WRITE ,__dma4_cdp_bits);
__IO_REG32(    DMA4_CNDP20,      			0x48056854,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCDN20,      			0x48056858,__READ_WRITE ,__dma4_ccdn_bits);

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
__IO_REG32_BIT(DMA4_CDP21,       			0x480568B0,__READ_WRITE ,__dma4_cdp_bits);
__IO_REG32(    DMA4_CNDP21,      			0x480568B4,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCDN21,      			0x480568B8,__READ_WRITE ,__dma4_ccdn_bits);

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
__IO_REG32_BIT(DMA4_CDP22,       			0x48056910,__READ_WRITE ,__dma4_cdp_bits);
__IO_REG32(    DMA4_CNDP22,      			0x48056914,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCDN22,      			0x48056918,__READ_WRITE ,__dma4_ccdn_bits);

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
__IO_REG32_BIT(DMA4_CDP23,       			0x48056970,__READ_WRITE ,__dma4_cdp_bits);
__IO_REG32(    DMA4_CNDP23,      			0x48056974,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCDN23,      			0x48056978,__READ_WRITE ,__dma4_ccdn_bits);

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
__IO_REG32_BIT(DMA4_CDP24,       			0x480569D0,__READ_WRITE ,__dma4_cdp_bits);
__IO_REG32(    DMA4_CNDP24,      			0x480569D4,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCDN24,      			0x480569D8,__READ_WRITE ,__dma4_ccdn_bits);

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
__IO_REG32_BIT(DMA4_CDP25,       			0x48056A30,__READ_WRITE ,__dma4_cdp_bits);
__IO_REG32(    DMA4_CNDP25,      			0x48056A34,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCDN25,      			0x48056A38,__READ_WRITE ,__dma4_ccdn_bits);

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
__IO_REG32_BIT(DMA4_CDP26,       			0x48056A90,__READ_WRITE ,__dma4_cdp_bits);
__IO_REG32(    DMA4_CNDP26,      			0x48056A94,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCDN26,      			0x48056A98,__READ_WRITE ,__dma4_ccdn_bits);

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
__IO_REG32_BIT(DMA4_CDP27,       			0x48056AF0,__READ_WRITE ,__dma4_cdp_bits);
__IO_REG32(    DMA4_CNDP27,      			0x48056AF4,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCDN27,      			0x48056AF8,__READ_WRITE ,__dma4_ccdn_bits);

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
__IO_REG32_BIT(DMA4_CDP28,       			0x48056B50,__READ_WRITE ,__dma4_cdp_bits);
__IO_REG32(    DMA4_CNDP28,      			0x48056B54,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCDN28,      			0x48056B58,__READ_WRITE ,__dma4_ccdn_bits);

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
__IO_REG32_BIT(DMA4_CDP29,       			0x48056BB0,__READ_WRITE ,__dma4_cdp_bits);
__IO_REG32(    DMA4_CNDP29,      			0x48056BB4,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCDN29,      			0x48056BB8,__READ_WRITE ,__dma4_ccdn_bits);

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
__IO_REG32_BIT(DMA4_CDP30,       			0x48056C10,__READ_WRITE ,__dma4_cdp_bits);
__IO_REG32(    DMA4_CNDP30,      			0x48056C14,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCDN30,      			0x48056C18,__READ_WRITE ,__dma4_ccdn_bits);

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
__IO_REG32_BIT(DMA4_CDP31,       			0x48056C70,__READ_WRITE ,__dma4_cdp_bits);
__IO_REG32(    DMA4_CNDP31,      			0x48056C74,__READ_WRITE );
__IO_REG32_BIT(DMA4_CCDN31,      			0x48056C78,__READ_WRITE ,__dma4_ccdn_bits);

/***************************************************************************
 **
 ** MPU INTC
 **
 ***************************************************************************/
__IO_REG32_BIT(INTCPS_REVISION,     	0x48200000,__READ       ,__intcps_revision_bits);
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
__IO_REG32_BIT(INTC_SYSCONFIG,        0x480C7010,__READ_WRITE ,__intc_sysconfig_bits);
__IO_REG32_BIT(INTC_IDLE,             0x480C7050,__READ_WRITE ,__intc_idle_bits);

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
__IO_REG32_BIT(SCM_CONTROL_PADCONF_CAM_HS,		    				0x4800210C,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_CAM_XCLKA,		  				0x48002110,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_CAM_FLD,		  	  			0x48002114,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_CAM_D1,	  	  				0x48002118,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_CAM_D3,	  	  				0x4800211C,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_CAM_D5,	  	  				0x48002120,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_CAM_D7,	  	  				0x48002124,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_CAM_D9,	  	  				0x48002128,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_CAM_D11,   	  				0x4800212C,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_CAM_WEN,	  	  				0x48002130,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_CSI2_DX0,  	  				0x48002134,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_CSI2_DX1,  	  				0x48002138,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_MCBSP2_FSX,						0x4800213C,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_MCBSP2_DR,							0x48002140,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_MMC1_CLK,							0x48002144,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_MMC1_DAT0,							0x48002148,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_MMC1_DAT2,							0x4800214C,__READ_WRITE	,__scm_control_padconf_bits);
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
__IO_REG32_BIT(SCM_CONTROL_PADCONF_HSUSB0_STP,		  			0x480021A4,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_HSUSB0_NXT,		  			0x480021A8,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_HSUSB0_DATA1,		  		0x480021AC,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_HSUSB0_DATA3,		  		0x480021B0,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_HSUSB0_DATA5,		  		0x480021B4,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_HSUSB0_DATA7,		  		0x480021B8,__READ_WRITE	,__scm_control_padconf_bits);
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
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SAD2D_MCAD0,						0x480021E4,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SAD2D_MCAD2,						0x480021E8,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SAD2D_MCAD4,						0x480021EC,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SAD2D_MCAD6,						0x480021F0,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SAD2D_MCAD8,						0x480021F4,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SAD2D_MCAD10,					0x480021F8,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SAD2D_MCAD12,					0x480021FC,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SAD2D_MCAD14,					0x48002200,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SAD2D_MCAD16,					0x48002204,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SAD2D_MCAD18,					0x48002208,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SAD2D_MCAD20,					0x4800220C,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SAD2D_MCAD22,					0x48002210,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SAD2D_MCAD24,					0x48002214,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SAD2D_MCAD26,					0x48002218,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SAD2D_MCAD28,					0x4800221C,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SAD2D_MCAD30,					0x48002220,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SAD2D_MCAD32,					0x48002224,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SAD2D_MCAD34,					0x48002228,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SAD2D_MCAD36,					0x4800222C,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SAD2D_NRESPWRON,				0x48002230,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SAD2D_ARMNIRQ,					0x48002234,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SAD2D_SPINT,				  	0x48002238,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SAD2D_DMAREQ0,					0x4800223C,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SAD2D_DMAREQ2,					0x48002240,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SAD2D_NTRST,			  		0x48002244,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SAD2D_TDO,			    		0x48002248,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SAD2D_TCK,		    			0x4800224C,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SAD2D_MSTDBY,					0x48002250,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SAD2D_IDLEACK,					0x48002254,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SAD2D_SWRITE,					0x48002258,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SAD2D_SREAD,					  0x4800225C,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SAD2D_SBUSFLAG,				0x48002260,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SDRC_CKE1,							0x48002264,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_OFF,										0x48002270,__READ_WRITE	,__scm_control_general_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SDRC_BA0,							0x480025A0,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SDRC_A0, 							0x480025A4,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SDRC_A2, 							0x480025A8,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SDRC_A4, 							0x480025AC,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SDRC_A6, 							0x480025B0,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SDRC_A8, 							0x480025B4,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SDRC_A10, 							0x480025B8,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SDRC_A12, 							0x480025BC,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SDRC_A14, 							0x480025C0,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SDRC_NCS1, 						0x480025C4,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SDRC_NRAS, 						0x480025C8,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SDRC_NWE, 							0x480025CC,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SDRC_DM1, 							0x480025D0,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_SDRC_DM3, 							0x480025D4,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_ETK_CLK,								0x480025D8,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_ETK_D0,								0x480025DC,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_ETK_D2,								0x480025E0,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_ETK_D4,								0x480025E4,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_ETK_D6,								0x480025E8,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_ETK_D8,								0x480025EC,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_ETK_D10,								0x480025F0,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_ETK_D12,								0x480025F4,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_ETK_D14,								0x480025F8,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_WKUP_I2C4_SCL,				  0x48002A00,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_WKUP_SYS_32K,				  0x48002A04,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_WKUP_SYS_NRESWARM,		  0x48002A08,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_WKUP_SYS_BOOT1,			  0x48002A0C,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_WKUP_SYS_BOOT3,			  0x48002A10,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_WKUP_SYS_BOOT5,			  0x48002A14,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_WKUP_SYS_OFF_MODE,		  0x48002A18,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_WKUP_JTAG_NTRST,				0x48002A1C,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_WKUP_JTAG_TMS_TMSC,		0x48002A20,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_WKUP_JTAG_EMU0,				0x48002A24,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_WKUP_CHASSIS_SWAKEUP,	0x48002A4C,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_WKUP_JTAG_TDO,					0x48002A50,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_WKUP_GPIO127,					0x48002A54,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_PADCONF_WKUP_GPIO128,					0x48002A58,__READ_WRITE	,__scm_control_padconf_bits);
__IO_REG32_BIT(SCM_CONTROL_DEVCONF0,											0x48002274,__READ_WRITE	,__scm_control_devconf0_bits);
__IO_REG32_BIT(SCM_CONTROL_MSUSPENDMUX_0,									0x48002290,__READ_WRITE	,__scm_control_msuspendmux_0_bits);
__IO_REG32_BIT(SCM_CONTROL_MSUSPENDMUX_1,									0x48002294,__READ_WRITE	,__scm_control_msuspendmux_1_bits);
__IO_REG32_BIT(SCM_CONTROL_MSUSPENDMUX_2,									0x48002298,__READ_WRITE	,__scm_control_msuspendmux_2_bits);
__IO_REG32_BIT(SCM_CONTROL_MSUSPENDMUX_4,									0x480022A0,__READ_WRITE	,__scm_control_msuspendmux_4_bits);
__IO_REG32_BIT(SCM_CONTROL_MSUSPENDMUX_5,									0x480022A4,__READ_WRITE	,__scm_control_msuspendmux_5_bits);
__IO_REG32_BIT(SCM_CONTROL_PROT_CTRL,				    					0x480022B0,__READ_WRITE	,__scm_control_prot_ctrl_bits);
__IO_REG32_BIT(SCM_CONTROL_DEVCONF1,											0x480022D8,__READ_WRITE	,__scm_control_devconf1_bits);
__IO_REG32_BIT(SCM_CONTROL_PROT_ERR_STATUS,								0x480022E4,__READ_WRITE	,__scm_control_prot_err_status_bits);
__IO_REG32_BIT(SCM_CONTROL_PROT_ERR_STATUS_DEBUG,					0x480022E8,__READ_WRITE	,__scm_control_prot_err_status_debug_bits);
__IO_REG32_BIT(SCM_CONTROL_STATUS,												0x480022F0,__READ				,__scm_control_status_bits);
__IO_REG32_BIT(SCM_CONTROL_GENERAL_PURPOSE_STATUS,				0x480022F4,__READ				,__scm_control_general_purpose_status_bits);
__IO_REG32(		 SCM_CONTROL_RPUB_KEY_H_0,									0x48002300,__READ				);
__IO_REG32(		 SCM_CONTROL_RPUB_KEY_H_1,									0x48002304,__READ				);
__IO_REG32(		 SCM_CONTROL_RPUB_KEY_H_2,									0x48002308,__READ				);
__IO_REG32(		 SCM_CONTROL_RPUB_KEY_H_3,									0x4800230C,__READ				);
__IO_REG32(		 SCM_CONTROL_RPUB_KEY_H_4,									0x48002310,__READ				);
__IO_REG32_BIT(SCM_CONTROL_USB_CONF_0,										0x48002370,__READ				, __scm_control_usb_conf_0_bits);
__IO_REG32(		 SCM_CONTROL_USB_CONF_1,										0x48002374,__READ				);
__IO_REG32_BIT(SCM_CONTROL_FUSE_OPP1G_VDD1, 							0x48002380,__READ				,__scm_control_fuse_opp1g_vdd1_bits);
__IO_REG32_BIT(SCM_CONTROL_FUSE_OPP50_VDD1,								0x48002384,__READ				,__scm_control_fuse_opp50_vdd1_bits);
__IO_REG32_BIT(SCM_CONTROL_FUSE_OPP100_VDD1,							0x48002388,__READ				,__scm_control_fuse_opp100_vdd1_bits);
__IO_REG32_BIT(SCM_CONTROL_FUSE_OPP130_VDD1,							0x48002390,__READ				,__scm_control_fuse_opp130_vdd1_bits);
__IO_REG32_BIT(SCM_CONTROL_FUSE_OPP50_VDD2, 							0x48002398,__READ				,__scm_control_fuse_opp50_vdd2_bits);
__IO_REG32_BIT(SCM_CONTROL_FUSE_OPP100_VDD2, 							0x4800239C,__READ				,__scm_control_fuse_opp100_vdd2_bits);
__IO_REG32_BIT(SCM_CONTROL_FUSE_SR,												0x480023A0,__READ     	,__scm_control_fuse_sr_bits);
__IO_REG32(    SCM_CONTROL_IVA2_BOOTADDR,									0x48002400,__READ_WRITE	);
__IO_REG32_BIT(SCM_CONTROL_IVA2_BOOTMOD,									0x48002404,__READ_WRITE	,__scm_control_iva2_bootmod_bits);
__IO_REG32_BIT(SCM_CONTROL_PROG_IO2,				    					0x48002408,__READ_WRITE	,__scm_control_prog_io2_bits);
__IO_REG32_BIT(SCM_CONTROL_MEM_RTA_CTRL,				 					0x4800240C,__READ_WRITE	,__scm_control_mem_rta_ctrl_bits);
__IO_REG32_BIT(SCM_CONTROL_DEBOBS_0,											0x48002420,__READ_WRITE	,__scm_control_debobs_0_bits);
__IO_REG32_BIT(SCM_CONTROL_DEBOBS_1,											0x48002424,__READ_WRITE	,__scm_control_debobs_1_bits);
__IO_REG32_BIT(SCM_CONTROL_DEBOBS_2,											0x48002428,__READ_WRITE	,__scm_control_debobs_2_bits);
__IO_REG32_BIT(SCM_CONTROL_DEBOBS_3,											0x4800242C,__READ_WRITE	,__scm_control_debobs_3_bits);
__IO_REG32_BIT(SCM_CONTROL_DEBOBS_4,											0x48002430,__READ_WRITE	,__scm_control_debobs_4_bits);
__IO_REG32_BIT(SCM_CONTROL_DEBOBS_5,											0x48002434,__READ_WRITE	,__scm_control_debobs_5_bits);
__IO_REG32_BIT(SCM_CONTROL_DEBOBS_6,											0x48002438,__READ_WRITE	,__scm_control_debobs_6_bits);
__IO_REG32_BIT(SCM_CONTROL_DEBOBS_7,											0x4800243C,__READ_WRITE	,__scm_control_debobs_7_bits);
__IO_REG32_BIT(SCM_CONTROL_DEBOBS_8,											0x48002440,__READ_WRITE	,__scm_control_debobs_8_bits);
__IO_REG32_BIT(SCM_CONTROL_PROG_IO0,											0x48002444,__READ_WRITE	,__scm_control_prog_io0_bits);
__IO_REG32_BIT(SCM_CONTROL_PROG_IO1,											0x48002448,__READ_WRITE	,__scm_control_prog_io1_bits);
__IO_REG32_BIT(SCM_CONTROL_DSS_DPLL_SPREADING,  					0x48002450,__READ_WRITE	,__scm_control_dss_dpll_spreading_bits);
__IO_REG32_BIT(SCM_CONTROL_CORE_DPLL_SPREADING,  					0x48002454,__READ_WRITE	,__scm_control_core_dpll_spreading_bits);
__IO_REG32_BIT(SCM_CONTROL_PER_DPLL_SPREADING,  					0x48002458,__READ_WRITE	,__scm_control_per_dpll_spreading_bits);
__IO_REG32_BIT(SCM_CONTROL_USBHOST_DPLL_SPREADING,  			0x4800245C,__READ_WRITE	,__scm_control_usbhost_dpll_spreading_bits);
__IO_REG32_BIT(SCM_CONTROL_SDRC_SHARING,  		      			0x48002460,__READ_WRITE	,__scm_control_sdrc_sharing_bits);
__IO_REG32_BIT(SCM_CONTROL_SDRC_MCFG0,    		      			0x48002464,__READ_WRITE	,__scm_control_sdrc_mcfg0_bits);
__IO_REG32_BIT(SCM_CONTROL_SDRC_MCFG1,    		      			0x48002468,__READ_WRITE	,__scm_control_sdrc_mcfg1_bits);
__IO_REG32_BIT(SCM_CONTROL_MODEM_FW_CONFIGURATION_LOCK,		0x4800246C,__READ_WRITE	,__scm_control_modem_fw_configuration_lock_bits);
__IO_REG32_BIT(SCM_CONTROL_MODEM_MEMORY_RESOURCES_CONF,  	0x48002470,__READ_WRITE	,__scm_control_modem_memory_resources_conf_bits);
__IO_REG32_BIT(SCM_CONTROL_MODEM_GPMC_DT_FW_REQ_INFO,  	  0x48002474,__READ_WRITE	,__scm_control_modem_gpmc_dt_fw_req_info_bits);
__IO_REG32_BIT(SCM_CONTROL_MODEM_GPMC_DT_FW_RD,  	        0x48002478,__READ_WRITE	,__scm_control_modem_gpmc_dt_fw_rd_bits);
__IO_REG32_BIT(SCM_CONTROL_MODEM_GPMC_DT_FW_WR,  	        0x4800247C,__READ_WRITE	,__scm_control_modem_gpmc_dt_fw_wr_bits);
__IO_REG32_BIT(SCM_CONTROL_MODEM_GPMC_BOOT_CODE,  	      0x48002480,__READ_WRITE	,__scm_control_modem_gpmc_boot_code_bits);
__IO_REG32(    SCM_CONTROL_MODEM_SMS_RG_ATT1,  	          0x48002484,__READ_WRITE	);
__IO_REG32_BIT(SCM_CONTROL_MODEM_SMS_RG_RDPERM1,  	      0x48002488,__READ_WRITE	,__scm_control_modem_sms_rg_rdperm1_bits);
__IO_REG32_BIT(SCM_CONTROL_MODEM_SMS_RG_WRPERM1,          0x4800248C,__READ_WRITE	,__scm_control_modem_sms_rg_wrperm1_bits);
__IO_REG32_BIT(SCM_CONTROL_MODEM_D2D_FW_DEBUG_MODE,       0x48002490,__READ_WRITE	,__scm_control_modem_d2d_fw_debug_mode_bits);
__IO_REG32_BIT(SCM_CONTROL_DPF_OCM_RAM_FW_ADDR_MATCH,     0x48002498,__READ_WRITE	,__scm_control_dpf_ocm_ram_fw_addr_match_bits);
__IO_REG32_BIT(SCM_CONTROL_DPF_OCM_RAM_FW_REQINFO,        0x4800249C,__READ_WRITE	,__scm_control_dpf_ocm_ram_fw_reqinfo_bits);
__IO_REG32_BIT(SCM_CONTROL_DPF_OCM_RAM_FW_WR,             0x480024A0,__READ_WRITE	,__scm_control_dpf_ocm_ram_fw_wr_bits);
__IO_REG32_BIT(SCM_CONTROL_DPF_REGION4_GPMC_FW_ADDR_MATCH,0x480024A4,__READ_WRITE	,__scm_control_dpf_region4_gpmc_fw_addr_match_bits);
__IO_REG32_BIT(SCM_CONTROL_DPF_REGION4_GPMC_FW_REQINFO,		0x480024A8,__READ_WRITE	,__scm_control_dpf_region4_gpmc_fw_reqinfo_bits);
__IO_REG32_BIT(SCM_CONTROL_DPF_REGION4_GPMC_FW_WR,				0x480024AC,__READ_WRITE	,__scm_control_dpf_region4_gpmc_fw_wr_bits);
__IO_REG32_BIT(SCM_CONTROL_DPF_REGION4_IVA2_FW_ADDR_MATCH,0x480024B0,__READ_WRITE	,__scm_control_dpf_region4_iva2_fw_addr_match_bits);
__IO_REG32_BIT(SCM_CONTROL_DPF_REGION4_IVA2_FW_REQINFO,		0x480024B4,__READ_WRITE	,__scm_control_dpf_region4_iva2_fw_reqinfo_bits);
__IO_REG32_BIT(SCM_CONTROL_DPF_REGION4_IVA2_FW_WR,				0x480024B8,__READ_WRITE	,__scm_control_dpf_region4_iva2_fw_wr_bits);
__IO_REG32_BIT(SCM_CONTROL_PBIAS_LITE,				            0x48002520,__READ_WRITE	,__scm_control_pbias_lite_bits);
__IO_REG32_BIT(SCM_CONTROL_TEMP_SENSOR,		                0x48002524,__READ_WRITE	,__scm_control_temp_sensor_bits);
__IO_REG32_BIT(SCM_CONTROL_DPF_MAD2D_FW_ADDR_MATCH,			  0x48002538,__READ_WRITE	,__scm_control_dpf_mad2d_fw_addr_match_bits);
__IO_REG32_BIT(SCM_CONTROL_DPF_MAD2D_FW_REQINFO,			    0x4800253C,__READ_WRITE	,__scm_control_dpf_mad2d_fw_reqinfo_bits);
__IO_REG32_BIT(SCM_CONTROL_DPF_MAD2D_FW_WR,		            0x48002540,__READ_WRITE	,__scm_control_dpf_mad2d_fw_wr_bits);
__IO_REG32_BIT(SCM_CONTROL_DSS_DPLL_SPREADING_FREQ,	      0x48002544,__READ_WRITE	,__scm_control_dss_dpll_spreading_freq_bits);
__IO_REG32_BIT(SCM_CONTROL_CORE_DPLL_SPREADING_FREQ,		  0x48002548,__READ_WRITE	,__scm_control_core_dpll_spreading_freq_bits);
__IO_REG32_BIT(SCM_CONTROL_PER_DPLL_SPREADING_FREQ,		    0x4800254C,__READ_WRITE	,__scm_control_per_dpll_spreading_freq_bits);
__IO_REG32_BIT(SCM_CONTROL_USBHOST_DPLL_SPREADING_FREQ,		0x48002550,__READ_WRITE	,__scm_control_usbhost_dpll_spreading_freq_bits);
__IO_REG32_BIT(SCM_CONTROL_AVDAC1,		    		            0x48002554,__READ_WRITE	,__scm_control_avdac1_bits);
__IO_REG32_BIT(SCM_CONTROL_AVDAC2,		    		            0x48002558,__READ_WRITE	,__scm_control_avdac2_bits);
__IO_REG32_BIT(SCM_CONTROL_CAMERA_PHY_CTRL,		            0x48002560,__READ_WRITE	,__scm_control_camera_phy_ctrl_bits);
#define CONTROL_SAVE_RESTORE_MEM_BASE_ADDR    (0x48002910)
__IO_REG32_BIT(SCM_CONTROL_WKUP_CTRL,											0x48002A5C,__READ_WRITE	,__scm_control_wkup_ctrl_bits);
__IO_REG32_BIT(SCM_CONTROL_WKUP_DEBOBS_0,									0x48002A68,__READ_WRITE	,__scm_control_wkup_debobs_0_bits);
__IO_REG32_BIT(SCM_CONTROL_WKUP_DEBOBS_1,									0x48002A6C,__READ_WRITE	,__scm_control_wkup_debobs_1_bits);
__IO_REG32_BIT(SCM_CONTROL_WKUP_DEBOBS_2,									0x48002A70,__READ_WRITE	,__scm_control_wkup_debobs_2_bits);
__IO_REG32_BIT(SCM_CONTROL_WKUP_DEBOBS_3,									0x48002A74,__READ_WRITE	,__scm_control_wkup_debobs_3_bits);
__IO_REG32_BIT(SCM_CONTROL_WKUP_DEBOBS_4,									0x48002A78,__READ_WRITE	,__scm_control_wkup_debobs_4_bits);
__IO_REG32_BIT(SCM_CONTROL_PROG_IO_WKUP1,									0x48002A80,__READ_WRITE	,__scm_control_prog_io_wkup1_bits);
__IO_REG32_BIT(SCM_CONTROL_BGAPTS_WKUP,									  0x48002A84,__READ_WRITE	,__scm_control_bgapts_wkup_bits);
__IO_REG32_BIT(SCM_CONTROL_VBBLDO_SW_CTRL,			     			0x48002A90,__READ_WRITE	,__scm_control_vbbldo_sw_ctrl_bits);

/***************************************************************************
 **
 ** MAILBOX
 **
 ***************************************************************************/
__IO_REG32_BIT(MAILBOX_REVISION,        0x48094000,__READ     	,__mailbox_revision_bits);
__IO_REG32_BIT(MAILBOX_SYSCONFIG,       0x48094010,__READ_WRITE	,__mailbox_sysconfig_bits);
__IO_REG32_BIT(MAILBOX_SYSSTATUS,       0x48094014,__READ     	,__mailbox_sysstatus_bits);
__IO_REG32(    MAILBOX_MESSAGE_0,       0x48094040,__READ_WRITE	);
__IO_REG32(    MAILBOX_MESSAGE_1,       0x48094044,__READ_WRITE	);
__IO_REG32_BIT(MAILBOX_FIFOSTATUS_0,    0x48094080,__READ       ,__mailbox_fifostatus_bits);
__IO_REG32_BIT(MAILBOX_FIFOSTATUS_1,    0x48094084,__READ       ,__mailbox_fifostatus_bits);
__IO_REG32_BIT(MAILBOX_MSGSTATUS_0,     0x480940C0,__READ     	,__mailbox_msgstatus_bits);
__IO_REG32_BIT(MAILBOX_MSGSTATUS_1,     0x480940C4,__READ     	,__mailbox_msgstatus_bits);
__IO_REG32_BIT(MAILBOX_IRQSTATUS_0,     0x48094100,__READ_WRITE	,__mailbox_irqstatus_bits);
__IO_REG32_BIT(MAILBOX_IRQENABLE_0,     0x48094104,__READ_WRITE	,__mailbox_irqenable_bits);
__IO_REG32_BIT(MAILBOX_IRQSTATUS_1,     0x48094108,__READ_WRITE	,__mailbox_irqstatus_bits);
__IO_REG32_BIT(MAILBOX_IRQENABLE_1,     0x4809410C,__READ_WRITE	,__mailbox_irqenable_bits);

/***************************************************************************
 **
 ** MMU1
 **
 ***************************************************************************/
__IO_REG32_BIT(MMU1_REVISION,           0x480BD400,__READ     	,__mmu_revision_bits);
__IO_REG32_BIT(MMU1_SYSCONFIG,          0x480BD410,__READ_WRITE	,__mmu_sysconfig_bits);
__IO_REG32_BIT(MMU1_SYSSTATUS,          0x480BD414,__READ       ,__mmu_sysstatus_bits);
__IO_REG32_BIT(MMU1_IRQSTATUS,          0x480BD418,__READ_WRITE	,__mmu_irqstatus_bits);
__IO_REG32_BIT(MMU1_IRQENABLE,          0x480BD41C,__READ_WRITE	,__mmu_irqstatus_bits);
__IO_REG32_BIT(MMU1_WALKING_ST,         0x480BD440,__READ       ,__mmu_walking_st_bits);
__IO_REG32_BIT(MMU1_CNTL,               0x480BD444,__READ_WRITE	,__mmu_cntl_bits);
__IO_REG32(    MMU1_FAULT_AD,           0x480BD448,__READ       );
__IO_REG32(    MMU1_TTB,                0x480BD44C,__READ_WRITE	);
__IO_REG32_BIT(MMU1_LOCK,               0x480BD450,__READ_WRITE	,__mmu_lock_bits);
__IO_REG32_BIT(MMU1_LD_TLB,             0x480BD454,__READ_WRITE	,__mmu_ld_tlb_bits);
__IO_REG32_BIT(MMU1_CAM,                0x480BD458,__READ_WRITE	,__mmu_cam_bits);
__IO_REG32_BIT(MMU1_RAM,                0x480BD45C,__READ_WRITE	,__mmu_ram_bits);
__IO_REG32_BIT(MMU1_GFLUSH,             0x480BD460,__READ_WRITE	,__mmu_gflush_bits);
__IO_REG32_BIT(MMU1_FLUSH_ENTRY,        0x480BD464,__READ_WRITE	,__mmu_flush_entry_bits);
__IO_REG32_BIT(MMU1_READ_CAM,           0x480BD468,__READ       ,__mmu_cam_bits);
__IO_REG32_BIT(MMU1_READ_RAM,           0x480BD46C,__READ       ,__mmu_ram_bits);
__IO_REG32(    MMU1_EMU_FAULT_AD,       0x480BD470,__READ       );

/***************************************************************************
 **
 ** MMU2
 **
 ***************************************************************************/
__IO_REG32_BIT(MMU2_REVISION,           0x5D000000,__READ     	,__mmu_revision_bits);
__IO_REG32_BIT(MMU2_SYSCONFIG,          0x5D000010,__READ_WRITE	,__mmu_sysconfig_bits);
__IO_REG32_BIT(MMU2_SYSSTATUS,          0x5D000014,__READ       ,__mmu_sysstatus_bits);
__IO_REG32_BIT(MMU2_IRQSTATUS,          0x5D000018,__READ_WRITE	,__mmu_irqstatus_bits);
__IO_REG32_BIT(MMU2_IRQENABLE,          0x5D00001C,__READ_WRITE	,__mmu_irqstatus_bits);
__IO_REG32_BIT(MMU2_WALKING_ST,         0x5D000040,__READ       ,__mmu_walking_st_bits);
__IO_REG32_BIT(MMU2_CNTL,               0x5D000044,__READ_WRITE	,__mmu_cntl_bits);
__IO_REG32(    MMU2_FAULT_AD,           0x5D000048,__READ       );
__IO_REG32(    MMU2_TTB,                0x5D00004C,__READ_WRITE	);
__IO_REG32_BIT(MMU2_LOCK,               0x5D000050,__READ_WRITE	,__mmu_lock_bits);
__IO_REG32_BIT(MMU2_LD_TLB,             0x5D000054,__READ_WRITE	,__mmu_ld_tlb_bits);
__IO_REG32_BIT(MMU2_CAM,                0x5D000058,__READ_WRITE	,__mmu_cam_bits);
__IO_REG32_BIT(MMU2_RAM,                0x5D00005C,__READ_WRITE	,__mmu_ram_bits);
__IO_REG32_BIT(MMU2_GFLUSH,             0x5D000060,__READ_WRITE	,__mmu_gflush_bits);
__IO_REG32_BIT(MMU2_FLUSH_ENTRY,        0x5D000064,__READ_WRITE	,__mmu_flush_entry_bits);
__IO_REG32_BIT(MMU2_READ_CAM,           0x5D000068,__READ       ,__mmu_cam_bits);
__IO_REG32_BIT(MMU2_READ_RAM,           0x5D00006C,__READ       ,__mmu_ram_bits);
__IO_REG32(    MMU2_EMU_FAULT_AD,       0x5D000070,__READ       );

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
__IO_REG32_BIT(GPT3_TWPS,    								0x49034034,__READ				,__gpt3_twps_bits);
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
__IO_REG32_BIT(GPT4_TWPS,    								0x49036034,__READ				,__gpt3_twps_bits);
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
__IO_REG32_BIT(GPT5_TWPS,    								0x49038034,__READ				,__gpt3_twps_bits);
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
__IO_REG32_BIT(GPT6_TWPS,    								0x4903A034,__READ				,__gpt3_twps_bits);
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
__IO_REG32_BIT(GPT7_TWPS,    								0x4903C034,__READ				,__gpt3_twps_bits);
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
__IO_REG32_BIT(GPT8_TWPS,    								0x4903E034,__READ				,__gpt3_twps_bits);
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
__IO_REG32_BIT(GPT9_TWPS,    								0x49040034,__READ				,__gpt3_twps_bits);
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
__IO_REG32_BIT(GPT11_TWPS,    							0x48088034,__READ				,__gpt3_twps_bits);
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
 ** UART1
 **
 ***************************************************************************/
__IO_REG8(     UART1_DLL_REG,    						0x4806A000,__READ_WRITE	);
#define 			 UART1_RHR_REG							UART1_DLL_REG
#define 			 UART1_THR_REG							UART1_DLL_REG
__IO_REG8_BIT( UART1_IER_REG,    						0x4806A004,__READ_WRITE	,__uart_ier_reg_bits);
#define 			 UART1_DLH_REG							UART1_IER_REG
__IO_REG8_BIT( UART1_FCR_REG,    						0x4806A008,__READ_WRITE	,__uart_fcr_reg_bits);
#define 			 UART1_IIR_REG							UART1_FCR_REG
#define 			 UART1_IIR_REG_bit					UART1_FCR_REG_bit
#define 			 UART1_EFR_REG							UART1_FCR_REG
#define 			 UART1_EFR_REG_bit					UART1_FCR_REG_bit
__IO_REG8_BIT( UART1_LCR_REG,    						0x4806A00C,__READ_WRITE	,__uart_lcr_reg_bits);
__IO_REG8_BIT( UART1_MCR_REG,    						0x4806A010,__READ_WRITE	,__uart_mcr_reg_bits);
#define 			 UART1_XON1_ADDR1_REG				UART1_MCR_REG
__IO_REG8_BIT( UART1_LSR_REG,    						0x4806A014,__READ_WRITE	,__uart_lsr_reg_bits);
#define 			 UART1_XON2_ADDR2_REG				UART1_LSR_REG
__IO_REG8_BIT( UART1_TCR_REG,     					0x4806A018,__READ_WRITE	,__uart_tcr_reg_bits);
#define 			 UART1_XOFF1_REG						UART1_TCR_REG
#define 			 UART1_MSR_REG							UART1_TCR_REG
#define 			 UART1_MSR_REG_bit					UART1_TCR_REG_bit
__IO_REG8_BIT( UART1_TLR_REG,    					  0x4806A01C,__READ_WRITE	,__uart_tlr_reg_bits);
#define 			 UART1_SPR_REG							UART1_TLR_REG
#define 			 UART1_XOFF2_REG						UART1_TLR_REG
__IO_REG8_BIT( UART1_MDR1_REG,    					0x4806A020,__READ_WRITE	,__uart_mdr1_reg_bits);
__IO_REG8_BIT( UART1_MDR2_REG,    					0x4806A024,__READ_WRITE	,__uart_mdr2_reg_bits);
__IO_REG8_BIT( UART1_UASR_REG,   						0x4806A038,__READ_WRITE	,__uart_uasr_reg_bits);
__IO_REG8_BIT( UART1_SCR_REG,    						0x4806A040,__READ_WRITE	,__uart_scr_reg_bits);
__IO_REG8_BIT( UART1_SSR_REG,    						0x4806A044,__READ_WRITE	,__uart_ssr_reg_bits);
__IO_REG8_BIT( UART1_MVR_REG,     					0x4806A050,__READ       ,__uart_mvr_reg_bits);
__IO_REG8_BIT( UART1_SYSC_REG,    					0x4806A054,__READ_WRITE	,__uart_sysc_reg_bits);
__IO_REG8_BIT( UART1_SYSS_REG,    					0x4806A058,__READ     	,__uart_syss_reg_bits);
__IO_REG8_BIT( UART1_WER_REG,    						0x4806A05C,__READ_WRITE	,__uart_wer_reg_bits);
__IO_REG8(     UART1_RXFIFO_LVL_REG, 				0x4806A064,__READ     	);
__IO_REG8(     UART1_TXFIFO_LVL_REG,    		0x4806A068,__READ     	);
__IO_REG8_BIT( UART1_IER2_REG,   						0x4806A06C,__READ_WRITE	,__uart_ier2_reg_bits);
__IO_REG8_BIT( UART1_ISR2_REG,   						0x4806A070,__READ_WRITE	,__uart_isr2_reg_bits);
__IO_REG8_BIT( UART1_MDR3_REG,   						0x4806A080,__READ_WRITE	,__uart_mdr3_reg_bits);

/***************************************************************************
 **
 ** UART2
 **
 ***************************************************************************/
__IO_REG8(     UART2_DLL_REG,    						0x4806C000,__READ_WRITE	);
#define 			 UART2_RHR_REG							UART2_DLL_REG
#define 			 UART2_THR_REG							UART2_DLL_REG
__IO_REG8_BIT( UART2_IER_REG,    						0x4806C004,__READ_WRITE	,__uart_ier_reg_bits);
#define 			 UART2_DLH_REG							UART2_IER_REG
__IO_REG8_BIT( UART2_FCR_REG,    						0x4806C008,__READ_WRITE	,__uart_fcr_reg_bits);
#define 			 UART2_IIR_REG							UART2_FCR_REG
#define 			 UART2_IIR_REG_bit					UART2_FCR_REG_bit
#define 			 UART2_EFR_REG							UART2_FCR_REG
#define 			 UART2_EFR_REG_bit					UART2_FCR_REG_bit
__IO_REG8_BIT( UART2_LCR_REG,    						0x4806C00C,__READ_WRITE	,__uart_lcr_reg_bits);
__IO_REG8_BIT( UART2_MCR_REG,    						0x4806C010,__READ_WRITE	,__uart_mcr_reg_bits);
#define 			 UART2_XON1_ADDR1_REG				UART2_MCR_REG
__IO_REG8_BIT( UART2_LSR_REG,    						0x4806C014,__READ_WRITE	,__uart_lsr_reg_bits);
#define 			 UART2_XON2_ADDR2_REG				UART2_LSR_REG
__IO_REG8_BIT( UART2_TCR_REG,    					  0x4806C018,__READ_WRITE	,__uart_tcr_reg_bits);
#define 			 UART2_XOFF1_REG						UART2_TCR_REG
#define 			 UART2_MSR_REG							UART2_TCR_REG
#define 			 UART2_MSR_REG_bit					UART2_TCR_REG_bit
__IO_REG8_BIT( UART2_TLR_REG,    					  0x4806C01C,__READ_WRITE	,__uart_tlr_reg_bits);
#define 			 UART2_SPR_REG							UART2_TLR_REG
#define 			 UART2_XOFF2_REG						UART2_TLR_REG
__IO_REG8_BIT( UART2_MDR1_REG,    					0x4806C020,__READ_WRITE	,__uart_mdr1_reg_bits);
__IO_REG8_BIT( UART2_MDR2_REG,    					0x4806C024,__READ_WRITE	,__uart_mdr2_reg_bits);
__IO_REG8_BIT( UART2_UASR_REG,   						0x4806C038,__READ_WRITE	,__uart_uasr_reg_bits);
__IO_REG8_BIT( UART2_SCR_REG,    						0x4806C040,__READ_WRITE	,__uart_scr_reg_bits);
__IO_REG8_BIT( UART2_SSR_REG,    						0x4806C044,__READ_WRITE	,__uart_ssr_reg_bits);
__IO_REG8_BIT( UART2_MVR_REG,     					0x4806C050,__READ     	,__uart_mvr_reg_bits);
__IO_REG8_BIT( UART2_SYSC_REG,    					0x4806C054,__READ_WRITE	,__uart_sysc_reg_bits);
__IO_REG8_BIT( UART2_SYSS_REG,    					0x4806C058,__READ     	,__uart_syss_reg_bits);
__IO_REG8_BIT( UART2_WER_REG,    						0x4806C05C,__READ_WRITE	,__uart_wer_reg_bits);
__IO_REG8(     UART2_RXFIFO_LVL_REG, 				0x4806C064,__READ     	);
__IO_REG8(     UART2_TXFIFO_LVL_REG,    		0x4806C068,__READ     	);
__IO_REG8_BIT( UART2_IER2_REG,   						0x4806C06C,__READ_WRITE	,__uart_ier2_reg_bits);
__IO_REG8_BIT( UART2_ISR2_REG,   						0x4806C070,__READ_WRITE	,__uart_isr2_reg_bits);
__IO_REG8_BIT( UART2_MDR3_REG,   						0x4806C080,__READ_WRITE	,__uart_mdr3_reg_bits);

/***************************************************************************
 **
 ** UART4
 **
 ***************************************************************************/
__IO_REG8(     UART4_DLL_REG,    						0x49042000,__READ_WRITE	);
#define 			 UART4_RHR_REG							UART4_DLL_REG
#define 			 UART4_THR_REG							UART4_DLL_REG
__IO_REG8_BIT( UART4_IER_REG,    						0x49042004,__READ_WRITE	,__uart_ier_reg_bits);
#define 			 UART4_DLH_REG							UART4_IER_REG
__IO_REG8_BIT( UART4_FCR_REG,    						0x49042008,__READ_WRITE	,__uart_fcr_reg_bits);
#define 			 UART4_IIR_REG							UART4_FCR_REG
#define 			 UART4_IIR_REG_bit					UART4_FCR_REG_bit
#define 			 UART4_EFR_REG							UART4_FCR_REG
#define 			 UART4_EFR_REG_bit					UART4_FCR_REG_bit
__IO_REG8_BIT( UART4_LCR_REG,    						0x4904200C,__READ_WRITE	,__uart_lcr_reg_bits);
__IO_REG8_BIT( UART4_MCR_REG,    						0x49042010,__READ_WRITE	,__uart_mcr_reg_bits);
#define 			 UART4_XON1_ADDR1_REG				UART4_MCR_REG
__IO_REG8_BIT( UART4_LSR_REG,    						0x49042014,__READ_WRITE	,__uart_lsr_reg_bits);
#define 			 UART4_XON2_ADDR2_REG				UART4_LSR_REG
__IO_REG8_BIT( UART4_TCR_REG,    					  0x49042018,__READ_WRITE	,__uart_tcr_reg_bits);
#define 			 UART4_XOFF1_REG						UART4_TCR_REG
#define 			 UART4_MSR_REG							UART4_TCR_REG
#define 			 UART4_MSR_REG_bit					UART4_TCR_REG_bit
__IO_REG8_BIT( UART4_TLR_REG,    					  0x4904201C,__READ_WRITE	,__uart_tlr_reg_bits);
#define 			 UART4_SPR_REG							UART4_TLR_REG
#define 			 UART4_XOFF2_REG						UART4_TLR_REG
__IO_REG8_BIT( UART4_MDR1_REG,    					0x49042020,__READ_WRITE	,__uart_mdr1_reg_bits);
__IO_REG8_BIT( UART4_MDR2_REG,    					0x49042024,__READ_WRITE	,__uart_mdr2_reg_bits);
__IO_REG8_BIT( UART4_UASR_REG,   						0x49042038,__READ_WRITE	,__uart_uasr_reg_bits);
__IO_REG8_BIT( UART4_SCR_REG,    						0x49042040,__READ_WRITE	,__uart_scr_reg_bits);
__IO_REG8_BIT( UART4_SSR_REG,    						0x49042044,__READ_WRITE	,__uart_ssr_reg_bits);
__IO_REG8_BIT( UART4_MVR_REG,     					0x49042050,__READ       ,__uart_mvr_reg_bits);
__IO_REG8_BIT( UART4_SYSC_REG,    					0x49042054,__READ_WRITE	,__uart_sysc_reg_bits);
__IO_REG8_BIT( UART4_SYSS_REG,    					0x49042058,__READ     	,__uart_syss_reg_bits);
__IO_REG8_BIT( UART4_WER_REG,    						0x4904205C,__READ_WRITE	,__uart_wer_reg_bits);
__IO_REG8(     UART4_RXFIFO_LVL_REG, 				0x49042064,__READ     	);
__IO_REG8(     UART4_TXFIFO_LVL_REG,    		0x49042068,__READ     	);
__IO_REG8_BIT( UART4_IER2_REG,   						0x4904206C,__READ_WRITE	,__uart_ier2_reg_bits);
__IO_REG8_BIT( UART4_ISR2_REG,   						0x49042070,__READ_WRITE	,__uart_isr2_reg_bits);
__IO_REG8_BIT( UART4_MDR3_REG,   						0x49042080,__READ_WRITE	,__uart_mdr3_reg_bits);

/***************************************************************************
 **
 ** UART3
 **
 ***************************************************************************/
__IO_REG8(     UART3_DLL_REG,    						0x49020000,__READ_WRITE	);
#define 			 UART3_RHR_REG							UART3_DLL_REG
#define 			 UART3_THR_REG							UART3_DLL_REG
__IO_REG8_BIT( UART3_IER_REG,    						0x49020004,__READ_WRITE	,__uart_ier_reg_bits);
#define 			 UART3_DLH_REG							UART3_IER_REG
#define 			 UART3_IrDA_IER_REG					UART3_IER_REG
#define 			 UART3_IrDA_IER_REG_bit			UART3_IER_REG_bit.IrDA
#define 			 UART3_CIR_IER_REG					UART3_IER_REG
#define 			 UART3_CIR_IER_REG_bit			UART3_IER_REG_bit.CIR
__IO_REG8_BIT( UART3_FCR_REG,    						0x49020008,__READ_WRITE	,__uart_fcr_reg_bits);
#define 			 UART3_IIR_REG							UART3_FCR_REG
#define 			 UART3_IIR_REG_bit					UART3_FCR_REG_bit
#define 			 UART3_IrDA_IIR_REG					UART3_FCR_REG
#define 			 UART3_IrDA_IIR_REG_bit			UART3_FCR_REG_bit.IrDA
#define 			 UART3_CIR_IIR_REG					UART3_FCR_REG
#define 			 UART3_CIR_IIR_REG_bit			UART3_FCR_REG_bit.CIR
#define 			 UART3_EFR_REG							UART3_FCR_REG
#define 			 UART3_EFR_REG_bit					UART3_FCR_REG_bit
__IO_REG8_BIT( UART3_LCR_REG,    						0x4902000C,__READ_WRITE	,__uart_lcr_reg_bits);
__IO_REG8_BIT( UART3_MCR_REG,    						0x49020010,__READ_WRITE	,__uart_mcr_reg_bits);
#define 			 UART3_XON1_ADDR1_REG				UART3_MCR_REG
__IO_REG8_BIT( UART3_LSR_REG,    						0x49020014,__READ_WRITE	,__uart_lsr_reg_bits);
#define 			 UART3_XON2_ADDR2_REG				UART3_LSR_REG
#define 			 UART3_LSR_IrDA_REG					UART3_LSR_REG
#define 			 UART3_LSR_IrDA_REG_bit			UART3_LSR_REG_bit.IrDA
#define 			 UART3_LSR_CIR_REG					UART3_LSR_REG
#define 			 UART3_LSR_CIR_REG_bit			UART3_LSR_REG_bit.CIR
__IO_REG8_BIT( UART3_TCR_REG,    					  0x49020018,__READ_WRITE	,__uart_tcr_reg_bits);
#define 			 UART3_XOFF1_REG						UART3_TCR_REG
#define 			 UART3_MSR_REG							UART3_TCR_REG
#define 			 UART3_MSR_REG_bit					UART3_TCR_REG_bit
__IO_REG8_BIT( UART3_TLR_REG,    					  0x4902001C,__READ_WRITE	,__uart_tlr_reg_bits);
#define 			 UART3_SPR_REG							UART3_TLR_REG
#define 			 UART3_XOFF2_REG						UART3_TLR_REG
__IO_REG8_BIT( UART3_MDR1_REG,    					0x49020020,__READ_WRITE	,__uart_mdr1_reg_bits);
__IO_REG8_BIT( UART3_MDR2_REG,    					0x49020024,__READ_WRITE	,__uart_mdr2_reg_bits);
__IO_REG8_BIT( UART3_SFLSR_REG,    					0x49020028,__READ_WRITE	,__uart_sflsr_reg_bits);
#define 			 UART3_TXFLL_REG            UART3_SFLSR_REG
__IO_REG8_BIT( UART3_TXFLH_REG,    				  0x4902002C,__READ_WRITE	,__uart_txflh_reg_bits);
#define 			 UART3_RESUME_REG						UART3_TXFLH_REG
__IO_REG8(     UART3_SFREGL_REG,  				  0x49020030,__READ_WRITE	);
#define 			 UART3_RXFLL_REG_REG        UART3_SFREGL_REG
__IO_REG8_BIT( UART3_SFREGH_REG,  					0x49020034,__READ_WRITE	,__uart_sfregh_reg_bits);
#define 			 UART3_RXFLH_REG            UART3_SFREGH_REG			
#define 			 UART3_RXFLH_REG_bit        UART3_SFREGH_REG_bit	
__IO_REG8_BIT( UART3_UASR_REG,   						0x49020038,__READ_WRITE	,__uart_uasr_reg_bits);
#define 			 UART3_BLR_REG              UART3_UASR_REG			
#define 			 UART3_BLR_REG_bit          UART3_UASR_REG_bit	
__IO_REG8_BIT( UART3_ACREG_REG,    					0x4902003C,__READ_WRITE	,__uart_acreg_reg_bits);
__IO_REG8_BIT( UART3_SCR_REG,    						0x49020040,__READ_WRITE	,__uart_scr_reg_bits);
__IO_REG8_BIT( UART3_SSR_REG,    						0x49020044,__READ_WRITE	,__uart_ssr_reg_bits);
__IO_REG8(     UART3_EBLR_REG,    					0x49020048,__READ_WRITE	);
__IO_REG8_BIT( UART3_MVR_REG,     					0x49020050,__READ       ,__uart_mvr_reg_bits);
__IO_REG8_BIT( UART3_SYSC_REG,    					0x49020054,__READ_WRITE	,__uart_sysc_reg_bits);
__IO_REG8_BIT( UART3_SYSS_REG,    					0x49020058,__READ     	,__uart_syss_reg_bits);
__IO_REG8_BIT( UART3_WER_REG,    						0x4902005C,__READ_WRITE	,__uart_wer_reg_bits);
__IO_REG8(     UART3_CFPS_REG,    					0x49020060,__READ_WRITE	);
__IO_REG8(     UART3_RXFIFO_LVL_REG, 				0x49020064,__READ     	);
__IO_REG8(     UART3_TXFIFO_LVL_REG,    		0x49020068,__READ     	);
__IO_REG8_BIT( UART3_IER2_REG,   						0x4902006C,__READ_WRITE	,__uart_ier2_reg_bits);
__IO_REG8_BIT( UART3_ISR2_REG,   						0x49020070,__READ_WRITE	,__uart_isr2_reg_bits);
__IO_REG8_BIT( UART3_MDR3_REG,   						0x49020080,__READ_WRITE	,__uart_mdr3_reg_bits);

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
__IO_REG32_BIT(MCBSPLP1_STATUS_REG,    			0x480740C0,__READ       ,__mcbsplp_status_reg_bits);

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
__IO_REG32_BIT(MCBSPLP5_STATUS_REG,    			0x480960C0,__READ       ,__mcbsplp_status_reg_bits);

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
__IO_REG32_BIT(MCBSPLP2_STATUS_REG,    			0x490220C0,__READ     	,__mcbsplp_status_reg_bits);
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
__IO_REG32_BIT(MCBSPLP3_STATUS_REG,    			0x490240C0,__READ       ,__mcbsplp_status_reg_bits);
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
__IO_REG32_BIT(MCBSPLP4_STATUS_REG,    			0x490260C0,__READ       ,__mcbsplp_status_reg_bits);

/***************************************************************************
 **
 ** OTG Controller
 **
 ***************************************************************************/
__IO_REG32_BIT(OTG_REVISION,      					0x480AB400,__READ				,__otg_revision_bits);
__IO_REG32_BIT(OTG_SYSCONFIG,       				0x480AB404,__READ_WRITE	,__otg_sysconfig_bits);
__IO_REG32_BIT(OTG_SYSSTATUS,       				0x480AB408,__READ       ,__otg_sysstatus_bits);
__IO_REG32_BIT(OTG_INTERFSEL,       				0x480AB40C,__READ_WRITE	,__otg_interfsel_bits);
__IO_REG32_BIT(OTG_SIMENABLE,       				0x480AB410,__READ_WRITE	,__otg_simenable_bits);
__IO_REG32_BIT(OTG_FORCESTDBY,       				0x480AB414,__READ_WRITE	,__otg_forcestdby_bits);
__IO_REG32_BIT(OTG_BIGENDIAN,       				0x480AB418,__READ_WRITE	,__otg_bigendian_bits);

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
__IO_REG8_BIT( ULPI_USB_INT_LATCH_NOCLR_0,  0x48062838,__READ				,__ulpi_usb_int_latch_bits);
__IO_REG8_BIT( ULPI_VENDOR_INT_EN_0,  			0x4806283B,__READ_WRITE	,__ulpi_vendor_int_en_bits);
__IO_REG8_BIT( ULPI_VENDOR_INT_EN_SET_0,  	0x4806283C,__READ_WRITE	,__ulpi_vendor_int_en_bits);
__IO_REG8_BIT( ULPI_VENDOR_INT_EN_CLR_0,  	0x4806283D,__READ_WRITE	,__ulpi_vendor_int_en_bits);
__IO_REG8_BIT( ULPI_VENDOR_INT_STATUS_0,  	0x4806283E,__READ				,__ulpi_vendor_int_status_bits);
__IO_REG8_BIT( ULPI_VENDOR_INT_LATCH_0,  		0x4806283F,__READ				,__ulpi_vendor_int_latch_bits);
__IO_REG8(		 ULPI_VENDOR_ID_LO_1,    			0x48062900,__READ				);
__IO_REG8(		 ULPI_VENDOR_ID_HI_1,    			0x48062901,__READ				);
__IO_REG8(		 ULPI_PRODUCT_ID_LO_1,    		0x48062902,__READ				);
__IO_REG8(		 ULPI_PRODUCT_ID_HI_1,    		0x48062903,__READ				);
__IO_REG8_BIT( ULPI_FUNCTION_CTRL_1,    		0x48062904,__READ_WRITE	,__ulpi_function_ctrl_bits);
__IO_REG8_BIT( ULPI_FUNCTION_CTRL_SET_1,    0x48062905,__READ_WRITE	,__ulpi_function_ctrl_bits);
__IO_REG8_BIT( ULPI_FUNCTION_CTRL_CLR_1,    0x48062906,__READ_WRITE	,__ulpi_function_ctrl_bits);
__IO_REG8_BIT( ULPI_INTERFACE_CTRL_1,    		0x48062907,__READ_WRITE	,__ulpi_interface_ctrl_bits);
__IO_REG8_BIT( ULPI_INTERFACE_CTRL_SET_1,   0x48062908,__READ_WRITE	,__ulpi_interface_ctrl_bits);
__IO_REG8_BIT( ULPI_INTERFACE_CTRL_CLR_1,   0x48062909,__READ_WRITE	,__ulpi_interface_ctrl_bits);
__IO_REG8_BIT( ULPI_OTG_CTRL_1,    					0x4806290A,__READ_WRITE	,__ulpi_otg_ctrl_bits);
__IO_REG8_BIT( ULPI_OTG_CTRL_SET_1,    			0x4806290B,__READ_WRITE	,__ulpi_otg_ctrl_bits);
__IO_REG8_BIT( ULPI_OTG_CTRL_CLR_1,    			0x4806290C,__READ_WRITE	,__ulpi_otg_ctrl_bits);
__IO_REG8_BIT( ULPI_USB_INT_EN_RISE_1,    	0x4806290D,__READ_WRITE	,__ulpi_usb_int_en_rise_bits);
__IO_REG8_BIT( ULPI_USB_INT_EN_RISE_SET_1,  0x4806290E,__READ_WRITE	,__ulpi_usb_int_en_rise_bits);
__IO_REG8_BIT( ULPI_USB_INT_EN_RISE_CLR_1,  0x4806290F,__READ_WRITE	,__ulpi_usb_int_en_rise_bits);
__IO_REG8_BIT( ULPI_USB_INT_EN_FALL_1,    	0x48062910,__READ_WRITE	,__ulpi_usb_int_en_fall_bits);
__IO_REG8_BIT( ULPI_USB_INT_EN_FALL_SET_1,  0x48062911,__READ_WRITE	,__ulpi_usb_int_en_fall_bits);
__IO_REG8_BIT( ULPI_USB_INT_EN_FALL_CLR_1,  0x48062912,__READ_WRITE	,__ulpi_usb_int_en_fall_bits);
__IO_REG8_BIT( ULPI_USB_INT_STATUS_1,  			0x48062913,__READ				,__ulpi_usb_int_status_bits);
__IO_REG8_BIT( ULPI_USB_INT_LATCH_1,  			0x48062914,__READ				,__ulpi_usb_int_latch_bits);
__IO_REG8_BIT( ULPI_DEBUG_1,  							0x48062915,__READ				,__ulpi_debug_bits);
__IO_REG8(		 ULPI_SCRATCH_REGISTER_1,  		0x48062916,__READ_WRITE	);
__IO_REG8(		 ULPI_SCRATCH_REGISTER_SET_1, 0x48062917,__READ_WRITE	);
__IO_REG8(		 ULPI_SCRATCH_REGISTER_CLR_1, 0x48062918,__READ_WRITE	);
__IO_REG8(		 ULPI_EXTENDED_SET_ACCESS_1,  0x4806292F,__READ_WRITE	);
__IO_REG8_BIT( ULPI_UTMI_VCONTROL_EN_1,  		0x48062930,__READ_WRITE	,__ulpi_utmi_vcontrol_en_bits);
__IO_REG8_BIT( ULPI_UTMI_VCONTROL_EN_SET_1, 0x48062931,__READ_WRITE	,__ulpi_utmi_vcontrol_en_bits);
__IO_REG8_BIT( ULPI_UTMI_VCONTROL_EN_CLR_1, 0x48062932,__READ_WRITE	,__ulpi_utmi_vcontrol_en_bits);
__IO_REG8(		 ULPI_UTMI_VCONTROL_STATUS_1, 0x48062933,__READ_WRITE	);
__IO_REG8_BIT( ULPI_UTMI_VCONTROL_LATCH_1,  0x48062934,__READ				,__ulpi_utmi_vcontrol_latch_bits);
__IO_REG8(		 ULPI_UTMI_VSTATUS_1,  				0x48062935,__READ_WRITE	);
__IO_REG8(		 ULPI_UTMI_VSTATUS_SET_1,  		0x48062936,__READ_WRITE	);
__IO_REG8(		 ULPI_UTMI_VSTATUS_CLR_1,  		0x48062937,__READ_WRITE	);
__IO_REG8_BIT( ULPI_USB_INT_LATCH_NOCLR_1,  0x48062938,__READ				,__ulpi_usb_int_latch_bits);
__IO_REG8_BIT( ULPI_VENDOR_INT_EN_1,  			0x4806293B,__READ_WRITE	,__ulpi_vendor_int_en_bits);
__IO_REG8_BIT( ULPI_VENDOR_INT_EN_SET_1,  	0x4806293C,__READ_WRITE	,__ulpi_vendor_int_en_bits);
__IO_REG8_BIT( ULPI_VENDOR_INT_EN_CLR_1,  	0x4806293D,__READ_WRITE	,__ulpi_vendor_int_en_bits);
__IO_REG8_BIT( ULPI_VENDOR_INT_STATUS_1,  	0x4806293E,__READ				,__ulpi_vendor_int_status_bits);
__IO_REG8_BIT( ULPI_VENDOR_INT_LATCH_1,  		0x4806293F,__READ				,__ulpi_vendor_int_latch_bits);
__IO_REG8(		 ULPI_VENDOR_ID_LO_2,    			0x48062A00,__READ				);
__IO_REG8(		 ULPI_VENDOR_ID_HI_2,    			0x48062A01,__READ				);
__IO_REG8(		 ULPI_PRODUCT_ID_LO_2,    		0x48062A02,__READ				);
__IO_REG8(		 ULPI_PRODUCT_ID_HI_2,    		0x48062A03,__READ				);
__IO_REG8_BIT( ULPI_FUNCTION_CTRL_2,    		0x48062A04,__READ_WRITE	,__ulpi_function_ctrl_bits);
__IO_REG8_BIT( ULPI_FUNCTION_CTRL_SET_2,    0x48062A05,__READ_WRITE	,__ulpi_function_ctrl_bits);
__IO_REG8_BIT( ULPI_FUNCTION_CTRL_CLR_2,    0x48062A06,__READ_WRITE	,__ulpi_function_ctrl_bits);
__IO_REG8_BIT( ULPI_INTERFACE_CTRL_2,    		0x48062A07,__READ_WRITE	,__ulpi_interface_ctrl_bits);
__IO_REG8_BIT( ULPI_INTERFACE_CTRL_SET_2,   0x48062A08,__READ_WRITE	,__ulpi_interface_ctrl_bits);
__IO_REG8_BIT( ULPI_INTERFACE_CTRL_CLR_2,   0x48062A09,__READ_WRITE	,__ulpi_interface_ctrl_bits);
__IO_REG8_BIT( ULPI_OTG_CTRL_2,    					0x48062A0A,__READ_WRITE	,__ulpi_otg_ctrl_bits);
__IO_REG8_BIT( ULPI_OTG_CTRL_SET_2,    			0x48062A0B,__READ_WRITE	,__ulpi_otg_ctrl_bits);
__IO_REG8_BIT( ULPI_OTG_CTRL_CLR_2,    			0x48062A0C,__READ_WRITE	,__ulpi_otg_ctrl_bits);
__IO_REG8_BIT( ULPI_USB_INT_EN_RISE_2,    	0x48062A0D,__READ_WRITE	,__ulpi_usb_int_en_rise_bits);
__IO_REG8_BIT( ULPI_USB_INT_EN_RISE_SET_2,  0x48062A0E,__READ_WRITE	,__ulpi_usb_int_en_rise_bits);
__IO_REG8_BIT( ULPI_USB_INT_EN_RISE_CLR_2,  0x48062A0F,__READ_WRITE	,__ulpi_usb_int_en_rise_bits);
__IO_REG8_BIT( ULPI_USB_INT_EN_FALL_2,    	0x48062A10,__READ_WRITE	,__ulpi_usb_int_en_fall_bits);
__IO_REG8_BIT( ULPI_USB_INT_EN_FALL_SET_2,  0x48062A11,__READ_WRITE	,__ulpi_usb_int_en_fall_bits);
__IO_REG8_BIT( ULPI_USB_INT_EN_FALL_CLR_2,  0x48062A12,__READ_WRITE	,__ulpi_usb_int_en_fall_bits);
__IO_REG8_BIT( ULPI_USB_INT_STATUS_2,  			0x48062A13,__READ				,__ulpi_usb_int_status_bits);
__IO_REG8_BIT( ULPI_USB_INT_LATCH_2,  			0x48062A14,__READ				,__ulpi_usb_int_latch_bits);
__IO_REG8_BIT( ULPI_DEBUG_2,  							0x48062A15,__READ				,__ulpi_debug_bits);
__IO_REG8(		 ULPI_SCRATCH_REGISTER_2,  		0x48062A16,__READ_WRITE	);
__IO_REG8(		 ULPI_SCRATCH_REGISTER_SET_2, 0x48062A17,__READ_WRITE	);
__IO_REG8(		 ULPI_SCRATCH_REGISTER_CLR_2, 0x48062A18,__READ_WRITE	);
__IO_REG8(		 ULPI_EXTENDED_SET_ACCESS_2,  0x48062A2F,__READ_WRITE	);
__IO_REG8_BIT( ULPI_UTMI_VCONTROL_EN_2,  		0x48062A30,__READ_WRITE	,__ulpi_utmi_vcontrol_en_bits);
__IO_REG8_BIT( ULPI_UTMI_VCONTROL_EN_SET_2, 0x48062A31,__READ_WRITE	,__ulpi_utmi_vcontrol_en_bits);
__IO_REG8_BIT( ULPI_UTMI_VCONTROL_EN_CLR_2, 0x48062A32,__READ_WRITE	,__ulpi_utmi_vcontrol_en_bits);
__IO_REG8(		 ULPI_UTMI_VCONTROL_STATUS_2, 0x48062A33,__READ_WRITE	);
__IO_REG8_BIT( ULPI_UTMI_VCONTROL_LATCH_2,  0x48062A34,__READ				,__ulpi_utmi_vcontrol_latch_bits);
__IO_REG8(		 ULPI_UTMI_VSTATUS_2,  				0x48062A35,__READ_WRITE	);
__IO_REG8(		 ULPI_UTMI_VSTATUS_SET_2,  		0x48062A36,__READ_WRITE	);
__IO_REG8(		 ULPI_UTMI_VSTATUS_CLR_2,  		0x48062A37,__READ_WRITE	);
__IO_REG8_BIT( ULPI_USB_INT_LATCH_NOCLR_2,  0x48062A38,__READ				,__ulpi_usb_int_latch_bits);
__IO_REG8_BIT( ULPI_VENDOR_INT_EN_2,  			0x48062A3B,__READ_WRITE	,__ulpi_vendor_int_en_bits);
__IO_REG8_BIT( ULPI_VENDOR_INT_EN_SET_2,  	0x48062A3C,__READ_WRITE	,__ulpi_vendor_int_en_bits);
__IO_REG8_BIT( ULPI_VENDOR_INT_EN_CLR_2,  	0x48062A3D,__READ_WRITE	,__ulpi_vendor_int_en_bits);
__IO_REG8_BIT( ULPI_VENDOR_INT_STATUS_2,  	0x48062A3E,__READ				,__ulpi_vendor_int_status_bits);
__IO_REG8_BIT( ULPI_VENDOR_INT_LATCH_2,  		0x48062A3F,__READ				,__ulpi_vendor_int_latch_bits);

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
 ** DID
 **
 ***************************************************************************/
__IO_REG32_BIT(CONTROL_IDCODE,      	  0x4830A204,__READ       ,__control_idcode_bits);
__IO_REG32_BIT(CONTROL_PRODUCTION_ID_2, 0x4830A210,__READ       ,__control_production_id_2_bits);
__IO_REG32(		 DIE_ID_0,      				  0x4830A218,__READ       );
__IO_REG32(		 DIE_ID_1,      				  0x4830A21C,__READ       );
__IO_REG32(		 DIE_ID_2,      				  0x4830A220,__READ       );
__IO_REG32(		 DIE_ID_3,      				  0x4830A224,__READ       );

/***************************************************************************
 **
 ** DID
 **
 ***************************************************************************/
__IO_REG32_BIT(SDTI_REVISION,       	  0x54500000,__READ       ,__sdti_revision_bits);
__IO_REG32_BIT(SDTI_SYSCONFIG,          0x54500010,__READ_WRITE ,__sdti_sysconfig_bits);
__IO_REG32_BIT(SDTI_SYSSTATUS,       	  0x54500014,__READ       ,__sdti_sysstatus_bits);
__IO_REG32_BIT(SDTI_WINCTRL,        	  0x54500024,__READ_WRITE ,__sdti_winctrl_bits);
__IO_REG32_BIT(SDTI_SCONFIG,        	  0x54500028,__READ_WRITE ,__sdti_sconfig_bits);
__IO_REG32_BIT(SDTI_TESTCTRL,        	  0x5450002C,__READ_WRITE ,__sdti_testctrl_bits);
__IO_REG32_BIT(INT_MODE_CTRL_REG,    	  0x54500F00,__READ_WRITE ,__int_mode_ctrl_reg_bits);
__IO_REG32_BIT(INT_OUTPUT_REG,          0x54500F04,__READ_WRITE ,__int_output_reg_bits);
__IO_REG32_BIT(INT_INPUT_REG,        	  0x54500F08,__READ_WRITE ,__int_input_reg_bits);
__IO_REG32_BIT(CLAIM_TAG_SET_REG,       0x54500FA0,__READ_WRITE ,__claim_tag_set_reg_bits);
__IO_REG32_BIT(CLAIM_TAG_CLEAR_REG,  	  0x54500FA4,__READ_WRITE ,__claim_tag_clear_reg_bits);
__IO_REG32(    LOCK_ACCESS_REG,      	  0x54500FB0,__READ_WRITE );
__IO_REG32_BIT(LOCK_STATUS_REG,     	  0x54500FB4,__READ       ,__lock_status_reg_bits);
__IO_REG32_BIT(AUTHENTICATION_STATUS,   0x54500FB8,__READ       ,__authentication_status_bits);
__IO_REG32_BIT(DEVICE_ID,            	  0x54500FC8,__READ       ,__device_id_bits);
__IO_REG32_BIT(DEVICE_TYPE_REG,      	  0x54500FCC,__READ       ,__device_type_reg_bits);
__IO_REG32_BIT(PERIPHERAL_ID4,      	  0x54500FD0,__READ       ,__peripheral_id4_bits);
__IO_REG32_BIT(PERIPHERAL_ID0,      	  0x54500FE0,__READ       ,__peripheral_id0_bits);
__IO_REG32_BIT(PERIPHERAL_ID1,      	  0x54500FE4,__READ       ,__peripheral_id1_bits);
__IO_REG32_BIT(PERIPHERAL_ID2,      	  0x54500FE8,__READ       ,__peripheral_id2_bits);
__IO_REG32_BIT(PERIPHERAL_ID3,      	  0x54500FEC,__READ       ,__peripheral_id3_bits);
__IO_REG32_BIT(COMPONENT_ID0,       	  0x54500FF0,__READ       ,__component_id0_bits);
__IO_REG32_BIT(COMPONENT_ID1,       	  0x54500FF4,__READ       ,__component_id1_bits);
__IO_REG32_BIT(COMPONENT_ID2,       	  0x54500FF8,__READ       ,__component_id2_bits);
__IO_REG32_BIT(COMPONENT_ID3,       	  0x54500FFC,__READ       ,__component_id3_bits);








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
 ** Interrupt Mapping to the MPU Subsystem
 **
 ***************************************************************************/
#define INT_EMUINT                    0   /* MPU emulation */
#define INT_COMMTX                    1   /* MPU emulation */
#define INT_COMMRX                    2   /* MPU emulation */
#define INT_BENCH                     3   /* MPU emulation */
#define INT_MCBSP2_ST_IRQ             4   /* Sidetone MCBSP2 overflow */
#define INT_MCBSP3_ST_IRQ             5   /* Sidetone MCBSP3 overflow */
#define INT_SYS_NIRQ                  7   /* External source (active low) */
#define INT_SMX_DBG_IRQ               9   /* L3 Interconnect error for debug */
#define INT_SMX_APP_IRQ              10   /* L3 Interconnect error for application */
#define INT_PRCM_MPU_IRQ             11   /* PRCM module IRQ */
#define INT_SDMA_IRQ_0               12   /* System DMA request 0 */
#define INT_SDMA_IRQ_1               13   /* System DMA request 1 */
#define INT_SDMA_IRQ_2               14   /* System DMA request 2 */
#define INT_SDMA_IRQ_3               15   /* System DMA request 3 */
#define INT_MCBSP1_IRQ               16   /* McBSP module 1 IRQ */
#define INT_MCBSP2_IRQ               17   /* McBSP module 2 IRQ */
#define INT_SR1_IRQ                  18   /* SmartReflex™ 1 */
#define INT_SR2_IRQ                  19   /* SmartReflex™ 2 */
#define INT_GPMC_IRQ                 20   /* General-purpose memory controller module */
#define INT_SGX_IRQ                  21   /* 2D/3D graphics module */
#define INT_MCBSP3_IRQ               22   /* McBSP module 3 */
#define INT_MCBSP4_IRQ               23   /* McBSP module 4 */
#define INT_CAM_IRQ0                 24   /* Camera interface request 0 */
#define INT_DSS_IRQ                  25   /* Display subsystem module */
#define INT_MAIL_U0_MPU_IRQ          26   /* Mailbox user 0 request */
#define INT_MCBSP5_IRQ               27   /* McBSP module 5 */
#define INT_IVA2_MMU_IRQ             28   /* IVA2 MMU */
#define INT_GPIO1_MPU_IRQ            29   /* GPIO module 1 */
#define INT_GPIO2_MPU_IRQ            30   /* GPIO module 2 */
#define INT_GPIO3_MPU_IRQ            31   /* GPIO module 3 */
#define INT_GPIO4_MPU_IRQ            32   /* GPIO module 4 */
#define INT_GPIO5_MPU_IRQ            33   /* GPIO module 5 */
#define INT_GPIO6_MPU_IRQ            34   /* GPIO module 6 */
#define INT_WDT3_IRQ                 36   /* Watchdog timer module 3 overflow */
#define INT_GPT1_IRQ                 37   /* General-purpose timer module 1 */
#define INT_GPT2_IRQ                 38   /* General-purpose timer module 2 */
#define INT_GPT3_IRQ                 39   /* General-purpose timer module 3 */
#define INT_GPT4_IRQ                 40   /* General-purpose timer module 4 */
#define INT_GPT5_IRQ                 41   /* General-purpose timer module 5 */
#define INT_GPT6_IRQ                 42   /* General-purpose timer module 6 */
#define INT_GPT7_IRQ                 43   /* General-purpose timer module 7 */
#define INT_GPT8_IRQ                 44   /* General-purpose timer module 8 */
#define INT_GPT9_IRQ                 45   /* General-purpose timer module 9 */
#define INT_GPT10_IRQ                46   /* General-purpose timer module 10 */
#define INT_GPT11_IRQ                47   /* General-purpose timer module 11 */
#define INT_SPI4_IRQ                 48   /* McSPI module 4 */
#define INT_MCBSP4_IRQ_TX            54   /* McBSP module 4 transmit */
#define INT_MCBSP4_IRQ_RX            55   /* McBSP module 4 receive */
#define INT_I2C1_IRQ                 56   /* I2C module 1 */
#define INT_I2C2_IRQ                 57   /* I2C module 2 */
#define INT_HDQ_IRQ                  58   /* HDQ/1-Wire */
#define INT_MCBSP1_IRQ_TX            59   /* McBSP module 1 transmit */
#define INT_MCBSP1_IRQ_RX            60   /* McBSP module 1 receive */
#define INT_I2C3_IRQ                 61   /* I2C module 3 */
#define INT_MCBSP2_IRQ_TX            62   /* McBSP module 2 transmit */
#define INT_MCBSP2_IRQ_RX            63   /* McBSP module 2 receive */
#define INT_SPI1_IRQ                 65   /* McSPI module 1 */
#define INT_SPI2_IRQ                 66   /* McSPI module 2 */
#define INT_UART1_IRQ                72   /* UART module 1 */
#define INT_UART2_IRQ                73   /* UART module 2 */
#define INT_UART3_IRQ                74   /* UART module 3 (also infrared) */
#define INT_PBIAS_IRQ                75   /* Merged interrupt for PBIASlite1 and 2 */
#define INT_OHCI_IRQ                 76   /* OHCI controller HSUSB MP Host Interrupt */
#define INT_EHCI_IRQ                 77   /* EHCI controller HSUSB MP Host Interrupt */
#define INT_TLL_IRQ                  78   /* HSUSB MP TLL Interrupt */
#define INT_UART4_IRQ                80   /* UART module 4 */
#define INT_MCBSP5_IRQ_TX            81   /* McBSP module 5 transmit */
#define INT_MCBSP5_IRQ_RX            82   /* McBSP module 5 receive */
#define INT_MMC1_IRQ                 83   /* MMC/SD module 1 */
#define INT_MMC2_IRQ                 86   /* MMC/SD module 2 */
#define INT_MPU_ICR_IRQ              87   /* MPU ICR interrupt */
#define INT_D2DFRINT                 88   /* From 3G coprocessor hardware when used in stacked modem configuration */
#define INT_MCBSP3_IRQ_TX            89   /* McBSP module 3 transmit */
#define INT_MCBSP3_IRQ_RX            90   /* McBSP module 3 receive */
#define INT_SPI3_IRQ                 91   /* McSPI module 3 */
#define INT_HSUSB_MC_NINT            92   /* High-Speed USB OTG controller */
#define INT_HSUSB_DMA_NINT           93   /* High-Speed USB OTG DMA controller */
#define INT_MMC3_IRQ                 94   /* MMC/SD module 3 */

#endif    /* __IOAM3715_H */
