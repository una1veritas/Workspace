#ifndef _tmpa900cm_h_
#define _tmpa900cm_h_
/* ************************************************************************ */
/*
 * ------------------------------------------------------------------------
 *   Application : 
 *   Micon : TMPA900CMXBG
 *   Copyright(C) TOSHIBA CORPORATION 2008 All rights reserved
 * ------------------------------------------------------------------------
 */

/*! \file tmpa900cm.h
	\brief Header file of SFR difinition

	\author TOSHIBA CORPORATION

	\date 2009/01/06 New Create
 */
/* ************************************************************************ */

/* ************************************************************************ */
/*
 * --------------------------------------------------------------------------
 *   Header Include Area
 * --------------------------------------------------------------------------
 */
#include	"cmn_type.h"


/*
 * --------------------------------------------------------------------------
 *   Macro Define
 * --------------------------------------------------------------------------
 */
#define MADDRESS_ACCESS		1	/* use address for UDC2AB	*/


#define	IO_0_Base				(0xF0000000)
#define	SysCtrl_Base			(IO_0_Base+0x00000)
#define	WDT_Base				(IO_0_Base+0x10000)
#define	PMC_Base				(IO_0_Base+0x20000)
#define	RTC_MLD_Base			(IO_0_Base+0x30000)
#define	TMR16_01_Base			(IO_0_Base+0x40000)
#define	TMR16_23_Base			(IO_0_Base+0x41000)
#define	TMR16_45_Base			(IO_0_Base+0x42000)
#define	PLLCG_Base				(IO_0_Base+0x50000)
#define	TSI_Base				(IO_0_Base+0x60000)
#define	I2C0_Base				(IO_0_Base+0x70000)
#define	I2C1_Base				(IO_0_Base+0x71000)
#define	ADC_Base				(IO_0_Base+0x80000)
#define	OFD_Base				(IO_0_Base+0x90000)
#define	EBI_Base				(IO_0_Base+0xA0000)
#define	LCDCOP_Base				(IO_0_Base+0xB0000)
/* ---- */
#define	IO_1_Base				(0xF0800000)
#define	IO_1_GPIO_Base			(IO_1_Base+0x00000)
#define	GPIO_A_Base				(IO_1_GPIO_Base+0x0000)
#define	GPIO_B_Base				(IO_1_GPIO_Base+0x1000)
#define	GPIO_C_Base				(IO_1_GPIO_Base+0x2000)
#define	GPIO_D_Base				(IO_1_GPIO_Base+0x3000)
#define	GPIO_F_Base				(IO_1_GPIO_Base+0x5000)
#define	GPIO_G_Base				(IO_1_GPIO_Base+0x6000)
#define	GPIO_J_Base				(IO_1_GPIO_Base+0x8000)
#define	GPIO_K_Base				(IO_1_GPIO_Base+0x9000)
#define	GPIO_L_Base				(IO_1_GPIO_Base+0xA000)
#define	GPIO_M_Base				(IO_1_GPIO_Base+0xB000)
#define	GPIO_N_Base				(IO_1_GPIO_Base+0xC000)
#define	GPIO_R_Base				(IO_1_GPIO_Base+0xE000)
#define	GPIO_T_Base				(IO_1_GPIO_Base+0xF000)
#define	GPIO_U_Base				(IO_1_GPIO_Base+0x4000)
#define	GPIO_V_Base				(IO_1_GPIO_Base+0x7000)
/* ---- */
#define	IO_3_Base				(0xF2000000)
#define	UART0_Base				(IO_3_Base+0x00000)
#define	UART1_Base				(IO_3_Base+0x01000)
#define	SSP0_Base				(IO_3_Base+0x02000)
#define	SSP1_Base				(IO_3_Base+0x03000)
#define	UART2_Base				(IO_3_Base+0x04000)
#define	NDFC_Base				(IO_3_Base+0x10000)
#define	CMSI_Base				(IO_3_Base+0x20000)
#define	I2S_Base				(IO_3_Base+0x40000)
#define	LCDDA_Base				(IO_3_Base+0x50000)
/* ---- */
#define	IO_4_Base				(0xF4000000)
#define	INTC_Base				(IO_4_Base+0x000000)
#define	DMAC_Base				(IO_4_Base+0x100000)
#define	LCDC_Base				(IO_4_Base+0x200000)
#define	MPMC0_DMC_Base			(IO_4_Base+0x300000)
#define	MPMC0_SMC_Base			(IO_4_Base+0x301000)
#define	MPMC1_DMC_Base			(IO_4_Base+0x310000)
#define	MPMC1_SMC_Base			(IO_4_Base+0x311000)
#define	UDC2_Base				(IO_4_Base+0x400000)
#define	USBH_Base				(IO_4_Base+0x500000)

/* ======================================================================== */

/* ------------------------------------------------------------------------ */
/* System Controller		 : 0xF0000000	*/
#define	Remap					*((volatile UINT32_t *)(SysCtrl_Base+0x0004))	/* Reset memory map (REMAP) */

/* ------------------------------------------------------------------------ */
/* Clock Controller			 : 0xF0050000	*/
#define	SYSCR0					*((volatile UINT32_t *)(PLLCG_Base+0x0000))	/* System Control Register 0 */
#define	SYSCR1					*((volatile UINT32_t *)(PLLCG_Base+0x0004))	/* System Control Register 1 */
#define	SYSCR2					*((volatile UINT32_t *)(PLLCG_Base+0x0008))	/* System Control Register 2 */
#define	SYSCR3					*((volatile UINT32_t *)(PLLCG_Base+0x000C))	/* System Control Register 3 */
#define	SYSCR4					*((volatile UINT32_t *)(PLLCG_Base+0x0010))	/* System Control Register 4 */
#define	SYSCR5					*((volatile UINT32_t *)(PLLCG_Base+0x0014))	/* System Control Register 5 */
#define	SYSCR6					*((volatile UINT32_t *)(PLLCG_Base+0x0018))	/* System Control Register 6 */
#define	SYSCR7					*((volatile UINT32_t *)(PLLCG_Base+0x001C))	/* System Control Register 7 */
#define	SYSCR8					*((volatile UINT32_t *)(PLLCG_Base+0x0020))	/* System Control Register 8 */

#define	CLKCR5					*((volatile UINT32_t *)(PLLCG_Base+0x0054))	/* Clock Control Register 5 */

/* ------------------------------------------------------------------------ */
/* INTC						 : 0xF4000000	*/
#define	VICIRQSTATUS			*((volatile UINT32_t *)(INTC_Base+0x0000))	/* IRQ Status Register */
#define	VICFIQSTATUS			*((volatile UINT32_t *)(INTC_Base+0x0004))	/* FIQ Status Register */
#define	VICRAWINTR				*((volatile UINT32_t *)(INTC_Base+0x0008))	/* Raw Interrupt Status Register */
#define	VICINTSELECT			*((volatile UINT32_t *)(INTC_Base+0x000C))	/* Interrupt Select Register  */
#define	VICINTENABLE			*((volatile UINT32_t *)(INTC_Base+0x0010))	/* Interrupt Enable Register */
#define	VICINTENCLEAR			*((volatile UINT32_t *)(INTC_Base+0x0014))	/* Interrupt Enable Clear Register */
#define	VICSOFTINT				*((volatile UINT32_t *)(INTC_Base+0x0018))	/* Software Interrupt Register */
#define	VICSOFTINTCLEAR			*((volatile UINT32_t *)(INTC_Base+0x001C))	/* Software Interrupt Clear Register */
#define	VICPROTECTION			*((volatile UINT32_t *)(INTC_Base+0x0020))	/* Protection Enable Register */
#define	VICSWPRIORITYMASK		*((volatile UINT32_t *)(INTC_Base+0x0024))	/* Software Priority Mask Register */
/* #define Reserved				*((volatile UINT32_t *)(INTC_Base+0x0028)) */
/* ---- */
#define	VICVECTADDR0			*((volatile UINT32_t *)(INTC_Base+0x0100))	/* Vector Address 0 Register */
#define	VICVECTADDR1			*((volatile UINT32_t *)(INTC_Base+0x0104))	/* Vector Address 1 Register */
#define	VICVECTADDR2			*((volatile UINT32_t *)(INTC_Base+0x0108))	/* Vector Address 2 Register */
#define	VICVECTADDR3			*((volatile UINT32_t *)(INTC_Base+0x010C))	/* Vector Address 3 Register */
#define	VICVECTADDR4			*((volatile UINT32_t *)(INTC_Base+0x0110))	/* Vector Address 4 Register */
#define	VICVECTADDR5			*((volatile UINT32_t *)(INTC_Base+0x0114))	/* Vector Address 5 Register */
#define	VICVECTADDR6			*((volatile UINT32_t *)(INTC_Base+0x0118))	/* Vector Address 6 Register */
#define	VICVECTADDR7			*((volatile UINT32_t *)(INTC_Base+0x011C))	/* Vector Address 7 Register */
#define	VICVECTADDR8			*((volatile UINT32_t *)(INTC_Base+0x0120))	/* Vector Address 8 Register */
#define	VICVECTADDR9			*((volatile UINT32_t *)(INTC_Base+0x0124))	/* Vector Address 9 Register */
#define	VICVECTADDR10			*((volatile UINT32_t *)(INTC_Base+0x0128))	/* Vector Address 10 Register */
#define	VICVECTADDR11			*((volatile UINT32_t *)(INTC_Base+0x012C))	/* Vector Address 11 Register */
#define	VICVECTADDR12			*((volatile UINT32_t *)(INTC_Base+0x0130))	/* Vector Address 12 Register */
#define	VICVECTADDR13			*((volatile UINT32_t *)(INTC_Base+0x0134))	/* Vector Address 13 Register */
#define	VICVECTADDR14			*((volatile UINT32_t *)(INTC_Base+0x0138))	/* Vector Address 14 Register */
#define	VICVECTADDR15			*((volatile UINT32_t *)(INTC_Base+0x013C))	/* Vector Address 15 Register */
#define	VICVECTADDR16			*((volatile UINT32_t *)(INTC_Base+0x0140))	/* Vector Address 16 Register */
#define	VICVECTADDR17			*((volatile UINT32_t *)(INTC_Base+0x0144))	/* Vector Address 17 Register */
#define	VICVECTADDR18			*((volatile UINT32_t *)(INTC_Base+0x0148))	/* Vector Address 18 Register */
/* #define VICVECTADDR19			*((volatile UINT32_t *)(INTC_Base+0x014C))	Vector Address 19 Register */
#define	VICVECTADDR20			*((volatile UINT32_t *)(INTC_Base+0x0150))	/* Vector Address 20 Register */
#define	VICVECTADDR21			*((volatile UINT32_t *)(INTC_Base+0x0154))	/* Vector Address 21 Register */
#define	VICVECTADDR22			*((volatile UINT32_t *)(INTC_Base+0x0158))	/* Vector Address 22 Register */
#define	VICVECTADDR23			*((volatile UINT32_t *)(INTC_Base+0x015C))	/* Vector Address 23 Register */
/* #define	VICVECTADDR24			*((volatile UINT32_t *)(INTC_Base+0x0160))	Vector Address 24 Register */
/* #define	VICVECTADDR25			*((volatile UINT32_t *)(INTC_Base+0x0164))	Vector Address 25 Register */
#define	VICVECTADDR26			*((volatile UINT32_t *)(INTC_Base+0x0168))	/* Vector Address 26 Register */
#define	VICVECTADDR27			*((volatile UINT32_t *)(INTC_Base+0x016C))	/* Vector Address 27 Register */
#define	VICVECTADDR28			*((volatile UINT32_t *)(INTC_Base+0x0170))	/* Vector Address 28 Register */
#define	VICVECTADDR29			*((volatile UINT32_t *)(INTC_Base+0x0174))	/* Vector Address 29 Register */
#define	VICVECTADDR30			*((volatile UINT32_t *)(INTC_Base+0x0178))	/* Vector Address 30 Register */
#define	VICVECTADDR31			*((volatile UINT32_t *)(INTC_Base+0x017C))	/* Vector Address 31 Register */
/* ---- */
#define	VICVECTPRIORITY0		*((volatile UINT32_t *)(INTC_Base+0x0200))	/* Vector Priority 0 Register */
#define	VICVECTPRIORITY1		*((volatile UINT32_t *)(INTC_Base+0x0204))	/* Vector Priority 1 Register */
#define	VICVECTPRIORITY2		*((volatile UINT32_t *)(INTC_Base+0x0208))	/* Vector Priority 2 Register */
#define	VICVECTPRIORITY3		*((volatile UINT32_t *)(INTC_Base+0x020C))	/* Vector Priority 3 Register */
#define	VICVECTPRIORITY4		*((volatile UINT32_t *)(INTC_Base+0x0210))	/* Vector Priority 4 Register */
#define	VICVECTPRIORITY5		*((volatile UINT32_t *)(INTC_Base+0x0214))	/* Vector Priority 5 Register */
#define	VICVECTPRIORITY6		*((volatile UINT32_t *)(INTC_Base+0x0218))	/* Vector Priority 6 Register */
#define	VICVECTPRIORITY7		*((volatile UINT32_t *)(INTC_Base+0x021C))	/* Vector Priority 7 Register */
#define	VICVECTPRIORITY8		*((volatile UINT32_t *)(INTC_Base+0x0220))	/* Vector Priority 8 Register */
#define	VICVECTPRIORITY9		*((volatile UINT32_t *)(INTC_Base+0x0224))	/* Vector Priority 9 Register */
#define	VICVECTPRIORITY10		*((volatile UINT32_t *)(INTC_Base+0x0228))	/* Vector Priority 10 Register */
#define	VICVECTPRIORITY11		*((volatile UINT32_t *)(INTC_Base+0x022C))	/* Vector Priority 11 Register */
#define	VICVECTPRIORITY12		*((volatile UINT32_t *)(INTC_Base+0x0230))	/* Vector Priority 12 Register */
#define	VICVECTPRIORITY13		*((volatile UINT32_t *)(INTC_Base+0x0234))	/* Vector Priority 13 Register */
#define	VICVECTPRIORITY14		*((volatile UINT32_t *)(INTC_Base+0x0238))	/* Vector Priority 14 Register */
#define	VICVECTPRIORITY15		*((volatile UINT32_t *)(INTC_Base+0x023C))	/* Vector Priority 15 Register */
#define	VICVECTPRIORITY16		*((volatile UINT32_t *)(INTC_Base+0x0240))	/* Vector Priority 16 Register */
#define	VICVECTPRIORITY17		*((volatile UINT32_t *)(INTC_Base+0x0244))	/* Vector Priority 17 Register */
#define	VICVECTPRIORITY18		*((volatile UINT32_t *)(INTC_Base+0x0248))	/* Vector Priority 18 Register */
/* #define	VICVECTPRIORITY19		*((volatile UINT32_t *)(INTC_Base+0x024C))	Vector Priority 19 Register */
#define	VICVECTPRIORITY20		*((volatile UINT32_t *)(INTC_Base+0x0250))	/* Vector Priority 20 Register */
#define	VICVECTPRIORITY21		*((volatile UINT32_t *)(INTC_Base+0x0254))	/* Vector Priority 21 Register */
#define	VICVECTPRIORITY22		*((volatile UINT32_t *)(INTC_Base+0x0258))	/* Vector Priority 22 Register */
#define	VICVECTPRIORITY23		*((volatile UINT32_t *)(INTC_Base+0x025C))	/* Vector Priority 23 Register */
/* #define	VICVECTPRIORITY24		*((volatile UINT32_t *)(INTC_Base+0x0260))	Vector Priority 24 Register */
/* #define	VICVECTPRIORITY25		*((volatile UINT32_t *)(INTC_Base+0x0264))	Vector Priority 25 Register */
#define	VICVECTPRIORITY26		*((volatile UINT32_t *)(INTC_Base+0x0268))	/* Vector Priority 26 Register */
#define	VICVECTPRIORITY27		*((volatile UINT32_t *)(INTC_Base+0x026C))	/* Vector Priority 27 Register */
#define	VICVECTPRIORITY28		*((volatile UINT32_t *)(INTC_Base+0x0270))	/* Vector Priority 28 Register */
#define	VICVECTPRIORITY29		*((volatile UINT32_t *)(INTC_Base+0x0274))	/* Vector Priority 29 Register */
#define	VICVECTPRIORITY30		*((volatile UINT32_t *)(INTC_Base+0x0278))	/* Vector Priority 30 Register */
#define	VICVECTPRIORITY31		*((volatile UINT32_t *)(INTC_Base+0x027C))	/* Vector Priority 31 Register */
/* ---- */
#define	VICADDRESS				*((volatile UINT32_t *)(INTC_Base+0x0F00))	/* Vector Address Register */

/* ------------------------------------------------------------------------ */
/* DMAC						 : 0xF4100000	*/
#define	DMACIntStatus			*((volatile UINT32_t *)(DMAC_Base+0x0000))	/* DMAC Interrupt Status Register */
#define	DMACIntTCStatus			*((volatile UINT32_t *)(DMAC_Base+0x0004))	/* DMAC Interrupt Terminal Count Status Register */
#define	DMACIntTCClear			*((volatile UINT32_t *)(DMAC_Base+0x0008))	/* DMAC Interrupt Terminal Count Clear Register */
#define	DMACIntErrorStatus		*((volatile UINT32_t *)(DMAC_Base+0x000C))	/* DMAC Interrupt Error Status Register */
#define	DMACIntErrClr			*((volatile UINT32_t *)(DMAC_Base+0x0010))	/* DMAC Interrupt Error Clear Register */
#define	DMACRawIntTCStatus		*((volatile UINT32_t *)(DMAC_Base+0x0014))	/* DMAC Raw Interrupt Terminal Count Status Register */
#define	DMACRawIntErrorStatus	*((volatile UINT32_t *)(DMAC_Base+0x0018))	/* DMAC Raw Error Interrupt Status Register */
#define	DMACEnbldChns			*((volatile UINT32_t *)(DMAC_Base+0x001C))	/* DMAC Enabled Channel Register */
#define	DMACSoftBReq			*((volatile UINT32_t *)(DMAC_Base+0x0020))	/* DMAC Software Burst Request Register */
#define	DMACSoftSReq			*((volatile UINT32_t *)(DMAC_Base+0x0024))	/* DMAC Software Single Request Register */
/* #define Reserved				*((volatile UINT32_t *)(DMAC_Base+0x0028)) */
/* #define Reserved				*((volatile UINT32_t *)(DMAC_Base+0x002C)) */
#define	DMACConfiguration		*((volatile UINT32_t *)(DMAC_Base+0x0030))	/* DMAC Configuration Register */
/* #define Reserved				*((volatile UINT32_t *)(DMAC_Base+0x0034)) */
/* ---- */
#define	DMACC0SrcAddr			*((volatile UINT32_t *)(DMAC_Base+0x0100))	/* DMAC Channel0 Source Address Register */
#define	DMACC0DestAddr			*((volatile UINT32_t *)(DMAC_Base+0x0104))	/* DMAC Channel0 Destination Address Register */
#define	DMACC0LLI				*((volatile UINT32_t *)(DMAC_Base+0x0108))	/* DMAC Channel0 Linked List Item Register */
#define	DMACC0Control			*((volatile UINT32_t *)(DMAC_Base+0x010C))	/* DMAC Channel0 Control Register */
#define	DMACC0Configuration		*((volatile UINT32_t *)(DMAC_Base+0x0110))	/* DMAC Channel0 Configuration Register */
/* ---- */
#define	DMACC1SrcAddr			*((volatile UINT32_t *)(DMAC_Base+0x0120))	/* DMAC Channel1 Source Address Register */
#define	DMACC1DestAddr			*((volatile UINT32_t *)(DMAC_Base+0x0124))	/* DMAC Channel1 Destination Address Register */
#define	DMACC1LLI				*((volatile UINT32_t *)(DMAC_Base+0x0128))	/* DMAC Channel1 Linked List Item Register */
#define	DMACC1Control			*((volatile UINT32_t *)(DMAC_Base+0x012C))	/* DMAC Channel1 Control Register */
#define	DMACC1Configuration		*((volatile UINT32_t *)(DMAC_Base+0x0130))	/* DMAC Channel1 Configuration Register */
/* ---- */
#define	DMACC2SrcAddr			*((volatile UINT32_t *)(DMAC_Base+0x0140))	/* DMAC Channel2 Source Address Register */
#define	DMACC2DestAddr			*((volatile UINT32_t *)(DMAC_Base+0x0144))	/* DMAC Channel2 Destination Address Register */
#define	DMACC2LLI				*((volatile UINT32_t *)(DMAC_Base+0x0148))	/* DMAC Channel2 Linked List Item Register */
#define	DMACC2Control			*((volatile UINT32_t *)(DMAC_Base+0x014C))	/* DMAC Channel2 Control Register */
#define	DMACC2Configuration		*((volatile UINT32_t *)(DMAC_Base+0x0150))	/* DMAC Channel2 Configuration Register */
/* ---- */
#define	DMACC3SrcAddr			*((volatile UINT32_t *)(DMAC_Base+0x0160))	/* DMAC Channel3 Source Address Register */
#define	DMACC3DestAddr			*((volatile UINT32_t *)(DMAC_Base+0x0164))	/* DMAC Channel3 Destination Address Register */
#define	DMACC3LLI				*((volatile UINT32_t *)(DMAC_Base+0x0168))	/* DMAC Channel3 Linked List Item Register */
#define	DMACC3Control			*((volatile UINT32_t *)(DMAC_Base+0x016C))	/* DMAC Channel3 Control Register */
#define	DMACC3Configuration		*((volatile UINT32_t *)(DMAC_Base+0x0170))	/* DMAC Channel3 Configuration Register */
/* ---- */
#define	DMACC4SrcAddr			*((volatile UINT32_t *)(DMAC_Base+0x0180))	/* DMAC Channel4 Source Address Register */
#define	DMACC4DestAddr			*((volatile UINT32_t *)(DMAC_Base+0x0184))	/* DMAC Channel4 Destination Address Register */
#define	DMACC4LLI				*((volatile UINT32_t *)(DMAC_Base+0x0188))	/* DMAC Channel4 Linked List Item Register */
#define	DMACC4Control			*((volatile UINT32_t *)(DMAC_Base+0x018C))	/* DMAC Channel4 Control Register */
#define	DMACC4Configuration		*((volatile UINT32_t *)(DMAC_Base+0x0190))	/* DMAC Channel4 Configuration Register */
/* ---- */
#define	DMACC5SrcAddr			*((volatile UINT32_t *)(DMAC_Base+0x01A0))	/* DMAC Channel5 Source Address Register */
#define	DMACC5DestAddr			*((volatile UINT32_t *)(DMAC_Base+0x01A4))	/* DMAC Channel5 Destination Address Register */
#define	DMACC5LLI				*((volatile UINT32_t *)(DMAC_Base+0x01A8))	/* DMAC Channel5 Linked List Item Register */
#define	DMACC5Control			*((volatile UINT32_t *)(DMAC_Base+0x01AC))	/* DMAC Channel5 Control Register */
#define	DMACC5Configuration		*((volatile UINT32_t *)(DMAC_Base+0x01B0))	/* DMAC Channel5 Configuration Register */
/* ---- */
#define	DMACC6SrcAddr			*((volatile UINT32_t *)(DMAC_Base+0x01C0))	/* DMAC Channel6 Source Address Register */
#define	DMACC6DestAddr			*((volatile UINT32_t *)(DMAC_Base+0x01C4))	/* DMAC Channel6 Destination Address Register */
#define	DMACC6LLI				*((volatile UINT32_t *)(DMAC_Base+0x01C8))	/* DMAC Channel6 Linked List Item Register */
#define	DMACC6Control			*((volatile UINT32_t *)(DMAC_Base+0x01CC))	/* DMAC Channel6 Control Register */
#define	DMACC6Configuration		*((volatile UINT32_t *)(DMAC_Base+0x01D0))	/* DMAC Channel6 Configuration Register */
/* ---- */
#define	DMACC7SrcAddr			*((volatile UINT32_t *)(DMAC_Base+0x01E0))	/* DMAC Channel7 Source Address Register */
#define	DMACC7DestAddr			*((volatile UINT32_t *)(DMAC_Base+0x01E4))	/* DMAC Channel7 Destination Address Register */
#define	DMACC7LLI				*((volatile UINT32_t *)(DMAC_Base+0x01E8))	/* DMAC Channel7 Linked List Item Register */
#define	DMACC7Control			*((volatile UINT32_t *)(DMAC_Base+0x01EC))	/* DMAC Channel7 Control Register */
#define	DMACC7Configuration		*((volatile UINT32_t *)(DMAC_Base+0x01F0))	/* DMAC Channel7 Configuration Register */

/* ------------------------------------------------------------------------ */
/* GPIO A		 			 : 0xF0800000	*/
#define	GPIOADATA				*((volatile UINT32_t *)(GPIO_A_Base+0x03FC))	/* PortA Data Regsiter */
/* #define	GPIOADIR				*((volatile UINT32_t *)(GPIO_A_Base+0x0400))	 PortA Data Direction Register */
/* ---- */
/* #define	GPIOAFR1				*((volatile UINT32_t *)(GPIO_A_Base+0x0424))	 PortA Function Register1 */
/* #define	GPIOAFR2				*((volatile UINT32_t *)(GPIO_A_Base+0x0428))	 PortA Function Register2 */
/* ---- */
#define	GPIOAIS					*((volatile UINT32_t *)(GPIO_A_Base+0x0804))	/* PortA Interrupt Selection Register (Level and Edge) */
#define	GPIOAIBE				*((volatile UINT32_t *)(GPIO_A_Base+0x0808))	/* PortA Interrupt Selection Register (Fellow edge and Both edge) */
#define	GPIOAIEV				*((volatile UINT32_t *)(GPIO_A_Base+0x080C))	/* PortA Interrupt Selection Register (Fall down edge/Low level and Rising up edge/High level) */
#define	GPIOAIE					*((volatile UINT32_t *)(GPIO_A_Base+0x0810))	/* PortA Interrupt Enable Register */
#define	GPIOARIS				*((volatile UINT32_t *)(GPIO_A_Base+0x0814))	/* PortA Interrupt Status Register (Raw) */
#define	GPIOAMIS				*((volatile UINT32_t *)(GPIO_A_Base+0x0818))	/* PortA Interrupt Status Register (Masked) */
#define	GPIOAIC					*((volatile UINT32_t *)(GPIO_A_Base+0x081C))	/* PortA Interrupt Clear Register */
/* ---- */
/* #define	GPIOAODE				*((volatile UINT32_t *)(GPIO_AB_Base+0x0C00))	 PortA Open-drain Output Enable Register */

/* ------------------------------------------------------------------------ */
/* GPIO B					 : 0xF0801000	*/
#define	GPIOBDATA				*((volatile UINT32_t *)(GPIO_B_Base+0x03FC))	/* PortB Data Regsiter */
/* #define	GPIOBDIR				*((volatile UINT32_t *)(GPIO_B_Base+0x0400))	 PortB Data Direction Register */
/* ---- */
#define	GPIOBFR1				*((volatile UINT32_t *)(GPIO_B_Base+0x0424))	/* PortB Function Register1 */
#define	GPIOBFR2				*((volatile UINT32_t *)(GPIO_B_Base+0x0428))	/* PortB Function Register2 */
/* ---- */
/* #define	GPIOBIS					*((volatile UINT32_t *)(GPIO_B_Base+0x0804))	 PortB Interrupt Selection Register (Level and Edge) */
/* #define	GPIOBIBE				*((volatile UINT32_t *)(GPIO_B_Base+0x0808))	 PortB Interrupt Selection Register (Fellow edge and Both edge) */
/* #define	GPIOBIEV				*((volatile UINT32_t *)(GPIO_B_Base+0x080C))	 PortB Interrupt Selection Register (Fall down edge/Low level and Rising up edge/High level) */
/* #define	GPIOBIE					*((volatile UINT32_t *)(GPIO_B_Base+0x0810))	 PortB Interrupt Enable Register */
/* #define	GPIOBRIS				*((volatile UINT32_t *)(GPIO_B_Base+0x0814))	 PortB Interrupt Status Register (Raw) */
/* #define	GPIOBMIS				*((volatile UINT32_t *)(GPIO_B_Base+0x0818))	 PortB Interrupt Status Register (Masked) */
/* #define	GPIOBIC					*((volatile UINT32_t *)(GPIO_B_Base+0x081C))	 PortB Interrupt Clear Register */
/* ---- */
#define	GPIOBODE				*((volatile UINT32_t *)(GPIO_B_Base+0x0C00))	/* PortB Open-drain Output Enable Register */

/* ------------------------------------------------------------------------ */
/* GPIO C					 : 0xF0802000	*/
#define	GPIOCDATA				*((volatile UINT32_t *)(GPIO_C_Base+0x03FC))	/* PortC Data Regsiter */
#define	GPIOCDIR				*((volatile UINT32_t *)(GPIO_C_Base+0x0400))	/* PortC Data Direction Register */
/* ---- */
#define	GPIOCFR1				*((volatile UINT32_t *)(GPIO_C_Base+0x0424))	/* PortC Function Register1 */
#define	GPIOCFR2				*((volatile UINT32_t *)(GPIO_C_Base+0x0428))	/* PortC Function Register2 */
/* ---- */
#define	GPIOCIS					*((volatile UINT32_t *)(GPIO_C_Base+0x0804))	/* PortC Interrupt Selection Register (Level and Edge) */
#define	GPIOCIBE				*((volatile UINT32_t *)(GPIO_C_Base+0x0808))	/* PortC Interrupt Selection Register (Fellow edge and Both edge) */
#define	GPIOCIEV				*((volatile UINT32_t *)(GPIO_C_Base+0x080C))	/* PortC Interrupt Selection Register (Fall down edge/Low level and Rising up edge/High level) */
#define	GPIOCIE					*((volatile UINT32_t *)(GPIO_C_Base+0x0810))	/* PortC Interrupt Enable Register */
#define	GPIOCRIS				*((volatile UINT32_t *)(GPIO_C_Base+0x0814))	/* PortC Interrupt Status Register (Raw) */
#define	GPIOCMIS				*((volatile UINT32_t *)(GPIO_C_Base+0x0818))	/* PortC Interrupt Status Register (Masked) */
#define	GPIOCIC					*((volatile UINT32_t *)(GPIO_C_Base+0x081C))	/* PortC Interrupt Clear Register */
/* ---- */
#define	GPIOCODE				*((volatile UINT32_t *)(GPIO_C_Base+0x0C00))	/* PortC Open-drain Output Enable Register */

/* ------------------------------------------------------------------------ */
/* GPIO D					 : 0xF0803000	*/
#define	GPIODDATA				*((volatile UINT32_t *)(GPIO_D_Base+0x03FC))	/* PortD Data Regsiter */
/* #define	GPIODDIR				*((volatile UINT32_t *)(GPIO_D_Base+0x0400))	 PortD Data Direction Register */
/* ---- */
#define	GPIODFR1				*((volatile UINT32_t *)(GPIO_D_Base+0x0424))	/* PortD Function Register1 */
#define	GPIODFR2				*((volatile UINT32_t *)(GPIO_D_Base+0x0428))	/* PortD Function Register2 */
/* ---- */
#define	GPIODIS					*((volatile UINT32_t *)(GPIO_D_Base+0x0804))	/* PortD Interrupt Selection Register (Level and Edge) */
#define	GPIODIBE				*((volatile UINT32_t *)(GPIO_D_Base+0x0808))	/* PortD Interrupt Selection Register (Fellow edge and Both edge) */
#define	GPIODIEV				*((volatile UINT32_t *)(GPIO_D_Base+0x080C))	/* PortD Interrupt Selection Register (Fall down edge/Low level and Rising up edge/High level) */
#define	GPIODIE					*((volatile UINT32_t *)(GPIO_D_Base+0x0810))	/* PortD Interrupt Enable Register */
#define	GPIODRIS				*((volatile UINT32_t *)(GPIO_D_Base+0x0814))	/* PortD Interrupt Status Register (Raw) */
#define	GPIODMIS				*((volatile UINT32_t *)(GPIO_D_Base+0x0818))	/* PortD Interrupt Status Register (Masked) */
#define	GPIODIC					*((volatile UINT32_t *)(GPIO_D_Base+0x081C))	/* PortD Interrupt Clear Register */
/* ---- */
/* #define	GPIODODE				*((volatile UINT32_t *)(GPIO_AB_Base+0x0C00))	 PortD Open-drain Output Enable Register */

/* ------------------------------------------------------------------------ */
/* GPIO F					 : 0xF0805000	*/
#define	GPIOFDATA				*((volatile UINT32_t *)(GPIO_F_Base+0x03FC))	/* PortF Data Regsiter */
#define	GPIOFDIR				*((volatile UINT32_t *)(GPIO_F_Base+0x0400))	/* PortF Data Direction Register */
/* ---- */
#define	GPIOFFR1				*((volatile UINT32_t *)(GPIO_F_Base+0x0424))	/* PortF Function Register1 */
#define	GPIOFFR2				*((volatile UINT32_t *)(GPIO_F_Base+0x0428))	/* PortF Function Register2 */
/* ---- */
#define	GPIOFIS					*((volatile UINT32_t *)(GPIO_F_Base+0x0804))	/* PortF Interrupt Selection Register (Level and Edge) */
#define	GPIOFIBE				*((volatile UINT32_t *)(GPIO_F_Base+0x0808))	/* PortF Interrupt Selection Register (Fellow edge and Both edge) */
#define	GPIOFIEV				*((volatile UINT32_t *)(GPIO_F_Base+0x080C))	/* PortF Interrupt Selection Register (Fall down edge/Low level and Rising up edge/High level) */
#define	GPIOFIE					*((volatile UINT32_t *)(GPIO_F_Base+0x0810))	/* PortF Interrupt Enable Register */
#define	GPIOFRIS				*((volatile UINT32_t *)(GPIO_F_Base+0x0814))	/* PortF Interrupt Status Register (Raw) */
#define	GPIOFMIS				*((volatile UINT32_t *)(GPIO_F_Base+0x0818))	/* PortF Interrupt Status Register (Masked) */
#define	GPIOFIC					*((volatile UINT32_t *)(GPIO_F_Base+0x081C))	/* PortF Interrupt Clear Register */
/* ---- */
#define	GPIOFODE				*((volatile UINT32_t *)(GPIO_F_Base+0x0C00))	/* PortF Open-drain Output Enable Register */

/* ------------------------------------------------------------------------ */
/* GPIO G					 : 0xF0806000	*/
#define	GPIOGDATA				*((volatile UINT32_t *)(GPIO_G_Base+0x03FC))	/* PortG Data Regsiter */
#define	GPIOGDIR				*((volatile UINT32_t *)(GPIO_G_Base+0x0400))	/* PortG Data Direction Register */
/* ---- */
#define	GPIOGFR1				*((volatile UINT32_t *)(GPIO_G_Base+0x0424))	/* PortG Function Register1 */
/* #define	GPIOGFR2				*((volatile UINT32_t *)(GPIO_G_Base+0x0428))	 PortG Function Register2 */
/* ---- */
/* #define	GPIOGIS					*((volatile UINT32_t *)(GPIO_G_Base+0x0804))	 PortG Interrupt Selection Register (Level and Edge) */
/* #define	GPIOGIBE				*((volatile UINT32_t *)(GPIO_G_Base+0x0808))	 PortG Interrupt Selection Register (Fellow edge and Both edge) */
/* #define	GPIOGIEV				*((volatile UINT32_t *)(GPIO_G_Base+0x080C))	 PortG Interrupt Selection Register (Fall down edge/Low level and Rising up edge/High level) */
/* #define	GPIOGIE					*((volatile UINT32_t *)(GPIO_G_Base+0x0810))	 PortG Interrupt Enable Register */
/* #define	GPIOGRIS				*((volatile UINT32_t *)(GPIO_G_Base+0x0814))	 PortG Interrupt Status Register (Raw) */
/* #define	GPIOGMIS				*((volatile UINT32_t *)(GPIO_G_Base+0x0818))	 PortG Interrupt Status Register (Masked) */
/* #define	GPIOGIC					*((volatile UINT32_t *)(GPIO_G_Base+0x081C))	 PortG Interrupt Clear Register */
/* ---- */
/* #define	GPIOGODE				*((volatile UINT32_t *)(GPIO_G_Base+0x0C00))	 PortG Open-drain Output Enable Register */

/* ------------------------------------------------------------------------ */
/* GPIO J					 : 0xF0808000	*/
#define	GPIOJDATA				*((volatile UINT32_t *)(GPIO_J_Base+0x03FC))	/* PortJ Data Regsiter */
#define	GPIOJDIR				*((volatile UINT32_t *)(GPIO_J_Base+0x0400))	/* PortJ Data Direction Register */
/* ---- */
#define	GPIOJFR1				*((volatile UINT32_t *)(GPIO_J_Base+0x0424))	/* PortJ Function Register1 */
#define	GPIOJFR2				*((volatile UINT32_t *)(GPIO_J_Base+0x0428))	/* PortJ Function Register2 */
/* ---- */
/* #define	GPIOJIS					*((volatile UINT32_t *)(GPIO_J_Base+0x0804))	 PortJ Interrupt Selection Register (Level and Edge) */
/* #define	GPIOJIBE				*((volatile UINT32_t *)(GPIO_J_Base+0x0808))	 PortJ Interrupt Selection Register (Fellow edge and Both edge) */
/* #define	GPIOJIEV				*((volatile UINT32_t *)(GPIO_J_Base+0x080C))	 PortJ Interrupt Selection Register (Fall down edge/Low level and Rising up edge/High level) */
/* #define	GPIOJIE					*((volatile UINT32_t *)(GPIO_J_Base+0x0810))	 PortJ Interrupt Enable Register */
/* #define	GPIOJRIS				*((volatile UINT32_t *)(GPIO_J_Base+0x0814))	 PortJ Interrupt Status Register (Raw) */
/* #define	GPIOJMIS				*((volatile UINT32_t *)(GPIO_J_Base+0x0818))	 PortJ Interrupt Status Register (Masked) */
/* #define	GPIOJIC					*((volatile UINT32_t *)(GPIO_J_Base+0x081C))	 PortJ Interrupt Clear Register */
/* ---- */
/* #define	GPIOJODE				*((volatile UINT32_t *)(GPIO_J_Base+0x0C00))	 PortJ Open-drain Output Enable Register */

/* ------------------------------------------------------------------------ */
/* GPIO K					 : 0xF0809000	*/
#define	GPIOKDATA				*((volatile UINT32_t *)(GPIO_K_Base+0x03FC))	/* PortK Data Regsiter */
#define	GPIOKDIR				*((volatile UINT32_t *)(GPIO_K_Base+0x0400))	/* PortK Data Direction Register */
/* ---- */
#define	GPIOKFR1				*((volatile UINT32_t *)(GPIO_K_Base+0x0424))	/* PortK Function Register1 */
#define	GPIOKFR2				*((volatile UINT32_t *)(GPIO_K_Base+0x0428))	/* PortK Function Register2 */
/* ---- */
/* #define	GPIOKIS					*((volatile UINT32_t *)(GPIO_K_Base+0x0804))	 PortK Interrupt Selection Register (Level and Edge) */
/* #define	GPIOKIBE				*((volatile UINT32_t *)(GPIO_K_Base+0x0808))	 PortK Interrupt Selection Register (Fellow edge and Both edge) */
/* #define	GPIOKIEV				*((volatile UINT32_t *)(GPIO_K_Base+0x080C))	 PortK Interrupt Selection Register (Fall down edge/Low level and Rising up edge/High level) */
/* #define	GPIOKIE					*((volatile UINT32_t *)(GPIO_K_Base+0x0810))	 PortK Interrupt Enable Register */
/* #define	GPIOKRIS				*((volatile UINT32_t *)(GPIO_K_Base+0x0814))	 PortK Interrupt Status Register (Raw) */
/* #define	GPIOKMIS				*((volatile UINT32_t *)(GPIO_K_Base+0x0818))	 PortK Interrupt Status Register (Masked) */
/* #define	GPIOKIC					*((volatile UINT32_t *)(GPIO_K_Base+0x081C))	 PortK Interrupt Clear Register */
/* ---- */
/* #define	GPIOKODE				*((volatile UINT32_t *)(GPIO_K_Base+0x0C00))	 PortK Open-drain Output Enable Register */

/* ------------------------------------------------------------------------ */
/* GPIO L					 : 0xF080A000	*/
#define	GPIOLDATA				*((volatile UINT32_t *)(GPIO_L_Base+0x03FC))	/* PortL Data Regsiter */
/* ---- */
#define	GPIOLDIR				*((volatile UINT32_t *)(GPIO_L_Base+0x0400))	/* PortL Data Direction Register */
/* ---- */
#define	GPIOLFR1				*((volatile UINT32_t *)(GPIO_L_Base+0x0424))	/* PortL Function Register1 */
#define	GPIOLFR2				*((volatile UINT32_t *)(GPIO_L_Base+0x0428))	/* PortL Function Register2 */
/* ---- */
/* #define	GPIOLIS					*((volatile UINT32_t *)(GPIO_L_Base+0x0804))	 PortL Interrupt Selection Register (Level and Edge) */
/* #define	GPIOLIBE				*((volatile UINT32_t *)(GPIO_L_Base+0x0808))	 PortL Interrupt Selection Register (Fellow edge and Both edge) */
/* #define	GPIOLIEV				*((volatile UINT32_t *)(GPIO_L_Base+0x080C))	 PortL Interrupt Selection Register (Fall down edge/Low level and Rising up edge/High level) */
/* #define	GPIOLIE					*((volatile UINT32_t *)(GPIO_L_Base+0x0810))	 PortL Interrupt Enable Register */
/* #define	GPIOLRIS				*((volatile UINT32_t *)(GPIO_L_Base+0x0814))	 PortL Interrupt Status Register (Raw) */
/* #define	GPIOLMIS				*((volatile UINT32_t *)(GPIO_L_Base+0x0818))	 PortL Interrupt Status Register (Masked) */
/* #define	GPIOLIC					*((volatile UINT32_t *)(GPIO_L_Base+0x081C))	 PortL Interrupt Clear Register */
/* ---- */
/* #define	GPIOLODE				*((volatile UINT32_t *)(GPIO_L_Base+0x0C00))	 PortL Open-drain Output Enable Register */

/* ------------------------------------------------------------------------ */
/* GPIO M					 : 0xF080B000	*/
#define	GPIOMDATA				*((volatile UINT32_t *)(GPIO_M_Base+0x03FC))	/* PortM Data Regsiter */
#define	GPIOMDIR				*((volatile UINT32_t *)(GPIO_M_Base+0x0400))	/* PortM Data Direction Register */
/* ---- */
#define	GPIOMFR1				*((volatile UINT32_t *)(GPIO_M_Base+0x0424))	/* PortM Function Register1 */
#define	GPIOMFR2				*((volatile UINT32_t *)(GPIO_M_Base+0x0428))	/* PortM Function Register2 */
/* ---- */
/* #define	GPIOMIS					*((volatile UINT32_t *)(GPIO_M_Base+0x0804))	 PortM Interrupt Selection Register (Level and Edge) */
/* #define	GPIOMIBE				*((volatile UINT32_t *)(GPIO_M_Base+0x0808))	 PortM Interrupt Selection Register (Fellow edge and Both edge) */
/* #define	GPIOMIEV				*((volatile UINT32_t *)(GPIO_M_Base+0x080C))	 PortM Interrupt Selection Register (Fall down edge/Low level and Rising up edge/High level) */
/* #define	GPIOMIE					*((volatile UINT32_t *)(GPIO_M_Base+0x0810))	 PortM Interrupt Enable Register */
/* #define	GPIOMRIS				*((volatile UINT32_t *)(GPIO_M_Base+0x0814))	 PortM Interrupt Status Register (Raw) */
/* #define	GPIOMMIS				*((volatile UINT32_t *)(GPIO_M_Base+0x0818))	 PortM Interrupt Status Register (Masked) */
/* #define	GPIOMIC					*((volatile UINT32_t *)(GPIO_M_Base+0x081C))	 PortM Interrupt Clear Register */
/* ---- */
/* #define	GPIOMODE				*((volatile UINT32_t *)(GPIO_M_Base+0x0C00))	 PortM Open-drain Output Enable Register */

/* ------------------------------------------------------------------------ */
/* GPIO N					 : 0xF080C000	*/
#define	GPIONDATA				*((volatile UINT32_t *)(GPIO_N_Base+0x03FC))	/* PortN Data Regsiter */
#define	GPIONDIR				*((volatile UINT32_t *)(GPIO_N_Base+0x0400))	/* PortN Data Direction Register */
/* ---- */
#define	GPIONFR1				*((volatile UINT32_t *)(GPIO_N_Base+0x0424))	/* PortN Function Register1 */
#define	GPIONFR2				*((volatile UINT32_t *)(GPIO_N_Base+0x0428))	/* PortN Function Register2 */
/* ---- */
#define	GPIONIS					*((volatile UINT32_t *)(GPIO_N_Base+0x0804))	/* PortN Interrupt Selection Register (Level and Edge) */
#define	GPIONIBE				*((volatile UINT32_t *)(GPIO_N_Base+0x0808))	/* PortN Interrupt Selection Register (Fellow edge and Both edge) */
#define	GPIONIEV				*((volatile UINT32_t *)(GPIO_N_Base+0x080C))	/* PortN Interrupt Selection Register (Fall down edge/Low level and Rising up edge/High level) */
#define	GPIONIE					*((volatile UINT32_t *)(GPIO_N_Base+0x0810))	/* PortN Interrupt Enable Register */
#define	GPIONRIS				*((volatile UINT32_t *)(GPIO_N_Base+0x0814))	/* PortN Interrupt Status Register (Raw) */
#define	GPIONMIS				*((volatile UINT32_t *)(GPIO_N_Base+0x0818))	/* PortN Interrupt Status Register (Masked) */
#define	GPIONIC					*((volatile UINT32_t *)(GPIO_N_Base+0x081C))	/* PortN Interrupt Clear Register */
/* ---- */
/* #define	GPIONODE				*((volatile UINT32_t *)(GPIO_N_Base+0x0C00))	 PortN Open-drain Output Enable Register */

/* ------------------------------------------------------------------------ */
/* GPIO R		 			: 0xF080E000	*/
#define	GPIORDATA				*((volatile UINT32_t *)(GPIO_R_Base+0x03FC))	/* PortR Data Regsiter */
#define	GPIORDIR				*((volatile UINT32_t *)(GPIO_R_Base+0x0400))	/* PortR Data Direction Register */
/* ---- */
#define	GPIORFR1				*((volatile UINT32_t *)(GPIO_R_Base+0x0424))	/* PortR Function Register1 */
#define	GPIORFR2				*((volatile UINT32_t *)(GPIO_R_Base+0x0428))	/* PortR Function Register2 */
/* ---- */
#define	GPIORIS					*((volatile UINT32_t *)(GPIO_R_Base+0x0804))	/* PortR Interrupt Selection Register (Level and Edge) */
#define	GPIORIBE				*((volatile UINT32_t *)(GPIO_R_Base+0x0808))	/* PortR Interrupt Selection Register (Fellow edge and Both edge) */
#define	GPIORIEV				*((volatile UINT32_t *)(GPIO_R_Base+0x080C))	/* PortR Interrupt Selection Register (Fall down edge/Low level and Rising up edge/High level) */
#define	GPIORIE					*((volatile UINT32_t *)(GPIO_R_Base+0x0810))	/* PortR Interrupt Enable Register */
#define	GPIORRIS				*((volatile UINT32_t *)(GPIO_R_Base+0x0814))	/* PortR Interrupt Status Register (Raw) */
#define	GPIORMIS				*((volatile UINT32_t *)(GPIO_R_Base+0x0818))	/* PortR Interrupt Status Register (Masked) */
#define	GPIORIC					*((volatile UINT32_t *)(GPIO_R_Base+0x081C))	/* PortR Interrupt Clear Register */
/* ---- */
/* #define	GPIORODE				*((volatile UINT32_t *)(GPIO_R_Base+0x0C00))	 PortR Open-drain Output Enable Register */

/* ------------------------------------------------------------------------ */
/* GPIO T					 : 0xF080F000	*/
#define	GPIOTDATA				*((volatile UINT32_t *)(GPIO_T_Base+0x03FC))	/* PortT Data Regsiter */
#define	GPIOTDIR				*((volatile UINT32_t *)(GPIO_T_Base+0x0400))	/* PortT Data Direction Register */
/* ---- */
#define	GPIOTFR1				*((volatile UINT32_t *)(GPIO_T_Base+0x0424))	/* PortT Function Register1 */
#define	GPIOTFR2				*((volatile UINT32_t *)(GPIO_T_Base+0x0428))	
/* ---- */
/* #define	GPIOTIS					*((volatile UINT32_t *)(GPIO_T_Base+0x0804))	 PortT Interrupt Selection Register (Level and Edge) */
/* #define	GPIOTIBE				*((volatile UINT32_t *)(GPIO_T_Base+0x0808))	 PortT Interrupt Selection Register (Fellow edge and Both edge) */
/* #define	GPIOTIEV				*((volatile UINT32_t *)(GPIO_T_Base+0x080C))	 PortT Interrupt Selection Register (Fall down edge/Low level and Rising up edge/High level) */
/* #define	GPIOTIE					*((volatile UINT32_t *)(GPIO_T_Base+0x0810))	 PortT Interrupt Enable Register */
/* #define	GPIOTRIS				*((volatile UINT32_t *)(GPIO_T_Base+0x0814))	 PortT Interrupt Status Register (Raw) */
/* #define	GPIOTMIS				*((volatile UINT32_t *)(GPIO_T_Base+0x0818))	 PortT Interrupt Status Register (Masked) */
/* #define	GPIOTIC					*((volatile UINT32_t *)(GPIO_T_Base+0x081C))	 PortT Interrupt Clear Register */
/* ---- */
/* #define	GPIOTODE				*((volatile UINT32_t *)(GPIO_T_Base+0x0C00))	 PortT Open-drain Output Enable Register */

/* ------------------------------------------------------------------------ */
/* GPIO U					 : 0xF0804000	*/
#define	GPIOUDATA				*((volatile UINT32_t *)(GPIO_U_Base+0x03FC))	/* PortU Data Regsiter */
#define	GPIOUDIR				*((volatile UINT32_t *)(GPIO_U_Base+0x0400))	/* PortU Data Direction Register */
/* ---- */
#define	GPIOUFR1				*((volatile UINT32_t *)(GPIO_U_Base+0x0424))	/* PortU Function Register1 */
#define	GPIOUFR2				*((volatile UINT32_t *)(GPIO_U_Base+0x0428))	/* PortU Function Register2 */
/* ---- */
/* #define	GPIOUIS					*((volatile UINT32_t *)(GPIO_U_Base+0x0804))	 PortU Interrupt Selection Register (Level and Edge) */
/* #define	GPIOUIBE				*((volatile UINT32_t *)(GPIO_U_Base+0x0808))	 PortU Interrupt Selection Register (Fellow edge and Both edge) */
/* #define	GPIOUIEV				*((volatile UINT32_t *)(GPIO_U_Base+0x080C))	 PortU Interrupt Selection Register (Fall down edge/Low level and Rising up edge/High level) */
/* #define	GPIOUIE					*((volatile UINT32_t *)(GPIO_U_Base+0x0810))	 PortU Interrupt Enable Register */
/* #define	GPIOURIS				*((volatile UINT32_t *)(GPIO_U_Base+0x0814))	 PortU Interrupt Status Register (Raw) */
/* #define	GPIOUMIS				*((volatile UINT32_t *)(GPIO_U_Base+0x0818))	 PortU Interrupt Status Register (Masked) */
/* #define	GPIOUIC					*((volatile UINT32_t *)(GPIO_U_Base+0x081C))	 PortU Interrupt Clear Register */
/* ---- */
/* #define	GPIOUODE				*((volatile UINT32_t *)(GPIO_U_Base+0x0C00))	 PortU Open-drain Output Enable Register */

/* ------------------------------------------------------------------------ */
/* GPIO V					 : 0xF0807000	*/
#define	GPIOVDATA				*((volatile UINT32_t *)(GPIO_V_Base+0x03FC))	/* PortV Data Regsiter */
#define	GPIOVDIR				*((volatile UINT32_t *)(GPIO_V_Base+0x0400))	/* PortV Data Direction Register */
/* ---- */
#define	GPIOVFR1				*((volatile UINT32_t *)(GPIO_V_Base+0x0424))	/* PortV Function Register1 */
#define	GPIOVFR2				*((volatile UINT32_t *)(GPIO_V_Base+0x0428))	/* PortV Function Register2 */
/* ---- */
/* #define	GPIOVIS					*((volatile UINT32_t *)(GPIO_V_Base+0x0804))	 PortV Interrupt Selection Register (Level and Edge) */
/* #define	GPIOVIBE				*((volatile UINT32_t *)(GPIO_V_Base+0x0808))	 PortV Interrupt Selection Register (Fellow edge and Both edge) */
/* #define	GPIOVIEV				*((volatile UINT32_t *)(GPIO_V_Base+0x080C))	 PortV Interrupt Selection Register (Fall down edge/Low level and Rising up edge/High level) */
/* #define	GPIOVIE					*((volatile UINT32_t *)(GPIO_V_Base+0x0810))	 PortV Interrupt Enable Register */
/* #define	GPIOVRIS				*((volatile UINT32_t *)(GPIO_V_Base+0x0814))	 PortV Interrupt Status Register (Raw) */
/* #define	GPIOVMIS				*((volatile UINT32_t *)(GPIO_V_Base+0x0818))	 PortV Interrupt Status Register (Masked) */
/* #define	GPIOVIC					*((volatile UINT32_t *)(GPIO_V_Base+0x081C))	 PortV Interrupt Clear Register */
/* ---- */
/* #define	GPIOVODE				*((volatile UINT32_t *)(GPIO_V_Base+0x0C00))	 PortV Open-drain Output Enable Register */

/* ------------------------------------------------------------------------ */
/* EBI			 			 : 0xF00A0000	*/
#define	smc_timeout				*((volatile UINT32_t *)(EBI_Base+0x0050))	/* SMC Time Out Register  */

/* ------------------------------------------------------------------------ */
/* MPMC0_DMC				 : 0xF4300000	*/
#define	dmc_memc_status_3		*((volatile UINT32_t *)(MPMC0_DMC_Base+0x000))	/* DMC Memory Controller Status Register */
#define	dmc_memc_cmd_3			*((volatile UINT32_t *)(MPMC0_DMC_Base+0x004))	/* DMC Memory Controller Command Register */
#define	dmc_direct_cmd_3		*((volatile UINT32_t *)(MPMC0_DMC_Base+0x008))	/* DMC Direct Command Register */
#define	dmc_memory_cfg_3		*((volatile UINT32_t *)(MPMC0_DMC_Base+0x00C))	/* DMC Memory Configuration Register */
#define	dmc_refresh_prd_3		*((volatile UINT32_t *)(MPMC0_DMC_Base+0x010))	/* DMC Refresh Period Register */
#define	dmc_cas_latency_3		*((volatile UINT32_t *)(MPMC0_DMC_Base+0x014))	/* DMC CAS Latency Register */
#define	dmc_t_dqss_3			*((volatile UINT32_t *)(MPMC0_DMC_Base+0x018))	/* DMC t_dqss Register */
#define	dmc_t_mrd_3				*((volatile UINT32_t *)(MPMC0_DMC_Base+0x01C))	/* DMC t_mrd Register */
#define	dmc_t_ras_3				*((volatile UINT32_t *)(MPMC0_DMC_Base+0x020))	/* DMC t_ras Register */
#define	dmc_t_rc_3				*((volatile UINT32_t *)(MPMC0_DMC_Base+0x024))	/* DMC t_rc Register */
#define	dmc_t_rcd_3				*((volatile UINT32_t *)(MPMC0_DMC_Base+0x028))	/* DMC t_rcd Register */
#define	dmc_t_rfc_3				*((volatile UINT32_t *)(MPMC0_DMC_Base+0x02C))	/* DMC t_rfc Register */
#define	dmc_t_rp_3				*((volatile UINT32_t *)(MPMC0_DMC_Base+0x030))	/* DMC t_rp Register */
#define	dmc_t_rrd_3				*((volatile UINT32_t *)(MPMC0_DMC_Base+0x034))	/* DMC t_rrd Register */
#define	dmc_t_wr_3				*((volatile UINT32_t *)(MPMC0_DMC_Base+0x038))	/* DMC t_wr Register */
#define	dmc_t_wtr_3				*((volatile UINT32_t *)(MPMC0_DMC_Base+0x03C))	/* DMC t_wtr Register */
#define	dmc_t_xp_3				*((volatile UINT32_t *)(MPMC0_DMC_Base+0x040))	/* DMC t_xp Register */
#define	dmc_t_xsr_3				*((volatile UINT32_t *)(MPMC0_DMC_Base+0x044))	/* DMC t_xsr Register */
#define	dmc_t_esr_3				*((volatile UINT32_t *)(MPMC0_DMC_Base+0x048))	/* DMC t_esr Register */
/* ---- */
#define	dmc_id_0_cfg_3			*((volatile UINT32_t *)(MPMC0_DMC_Base+0x100))	/* DMC id_<0-3>_cfg Registers */
#define	dmc_id_1_cfg_3			*((volatile UINT32_t *)(MPMC0_DMC_Base+0x104))	/* DMC id_<0-3>_cfg Registers */
#define	dmc_id_2_cfg_3			*((volatile UINT32_t *)(MPMC0_DMC_Base+0x108))	/* DMC id_<0-3>_cfg Registers */
#define	dmc_id_3_cfg_3			*((volatile UINT32_t *)(MPMC0_DMC_Base+0x10C))	/* DMC id_<0-3>_cfg Registers */
/* ---- */
#define	dmc_chip_0_cfg_3		*((volatile UINT32_t *)(MPMC0_DMC_Base+0x200))	/* DMC chip_0_cfg Registers */
/* #define Reserved				*((volatile UINT32_t *)(MPMC0_DMC_Base+0x204)) */
/* #define Reserved				*((volatile UINT32_t *)(MPMC0_DMC_Base+0x208)) */
/* #define Reserved				*((volatile UINT32_t *)(MPMC0_DMC_Base+0x20C)) */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(MPMC0_DMC_Base+0x300)) */
#define	dmc_user_config_3		*((volatile UINT32_t *)(MPMC0_DMC_Base+0x304))	/* DMC user_config Register */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(MPMC0_DMC_Base+0xE00)) */
/* #define Reserved				*((volatile UINT32_t *)(MPMC0_DMC_Base+0xE04)) */
/* #define Reserved				*((volatile UINT32_t *)(MPMC0_DMC_Base+0xE08)) */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(MPMC0_DMC_Base+0xFE0)) */
/* ...                                                                             */
/* #define Reserved				*((volatile UINT32_t *)(MPMC0_DMC_Base+0xFEC)) */
/* #define Reserved				*((volatile UINT32_t *)(MPMC0_DMC_Base+0xFF0)) */
/* ...                                                                             */
/* #define Reserved				*((volatile UINT32_t *)(MPMC0_DMC_Base+0xFFC)) */

/* ------------------------------------------------------------------------ */
/* MPMC0_SMC				 : 0xF4301000	*/
/* #define Reserved				*((volatile UINT32_t *)(MPMC0_SMC_Base+0x000)) */
#define	smc_memif_cfg_3			*((volatile UINT32_t *)(MPMC0_SMC_Base+0x004))	/* SMC Memory Interface Configuration Register */
/* #define Reserved				*((volatile UINT32_t *)(MPMC0_SMC_Base+0x008)) */
/* #define Reserved				*((volatile UINT32_t *)(MPMC0_SMC_Base+0x00C)) */
#define	smc_direct_cmd_3		*((volatile UINT32_t *)(MPMC0_SMC_Base+0x010))	/* SMC Direct Command Register */
#define	smc_set_cycles_3		*((volatile UINT32_t *)(MPMC0_SMC_Base+0x014))	/* SMC Set Cycles Register */
#define	smc_set_opmode_3		*((volatile UINT32_t *)(MPMC0_SMC_Base+0x018))	/* SMC Set Opmode Register */
/* #define Reserved				*((volatile UINT32_t *)(MPMC0_SMC_Base+0x020)) */
/* ---- */
#define	smc_sram_cycles0_0_3	*((volatile UINT32_t *)(MPMC0_SMC_Base+0x100))	/* SMC SRAM Cycles Registers <0-3> */
#define	smc_opmode0_0_3			*((volatile UINT32_t *)(MPMC0_SMC_Base+0x104))	/* SMC Opmode Registers <0-3> */
/* ---- */
#define	smc_sram_cycles0_1_3	*((volatile UINT32_t *)(MPMC0_SMC_Base+0x120))	/* SMC SRAM Cycles Registers <0-3> */
#define	smc_opmode0_1_3			*((volatile UINT32_t *)(MPMC0_SMC_Base+0x124))	/* SMC Opmode Registers <0-3> */
/* ---- */
#define	smc_sram_cycles0_2_3	*((volatile UINT32_t *)(MPMC0_SMC_Base+0x140))	/* SMC SRAM Cycles Registers <0-3> */
#define	smc_opmode0_2_3			*((volatile UINT32_t *)(MPMC0_SMC_Base+0x144))	/* SMC Opmode Registers <0-3> */
/* ---- */
#define	smc_sram_cycles0_3_3	*((volatile UINT32_t *)(MPMC0_SMC_Base+0x160))	/* SMC SRAM Cycles Registers <0-3> */
#define	smc_opmode0_3_3			*((volatile UINT32_t *)(MPMC0_SMC_Base+0x164))	/* SMC Opmode Registers <0-3> */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(MPMC0_SMC_Base+0x200)) */
/* #define Reserved				*((volatile UINT32_t *)(MPMC0_SMC_Base+0x204)) */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(MPMC0_SMC_Base+0xE00)) */
/* #define Reserved				*((volatile UINT32_t *)(MPMC0_SMC_Base+0xE04)) */
/* #define Reserved				*((volatile UINT32_t *)(MPMC0_SMC_Base+0xE08)) */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(MPMC0_SMC_Base+0xFE0)) */
/* ...                                                                             */
/* #define Reserved				*((volatile UINT32_t *)(MPMC0_SMC_Base+0xFEC)) */
/* #define Reserved				*((volatile UINT32_t *)(MPMC0_SMC_Base+0xFF0)) */
/* ...                                                                             */
/* #define Reserved				*((volatile UINT32_t *)(MPMC0_SMC_Base+0xFFC)) */

/* ------------------------------------------------------------------------ */
/* MPMEC1_DMC				 : 0xF4310000	*/
#define	dmc_memc_status_5		*((volatile UINT32_t *)(MPMC1_DMC_Base+0x000))	/* DMC Memory Controller Status Register */
#define	dmc_memc_cmd_5			*((volatile UINT32_t *)(MPMC1_DMC_Base+0x004))	/* DMC Memory Controller Command Register */
#define	dmc_direct_cmd_5		*((volatile UINT32_t *)(MPMC1_DMC_Base+0x008))	/* DMC Direct Command Register */
#define	dmc_memory_cfg_5		*((volatile UINT32_t *)(MPMC1_DMC_Base+0x00C))	/* DMC Memory Configuration Register */
#define	dmc_refresh_prd_5		*((volatile UINT32_t *)(MPMC1_DMC_Base+0x010))	/* DMC Refresh Period Register */
#define	dmc_cas_latency_5		*((volatile UINT32_t *)(MPMC1_DMC_Base+0x014))	/* DMC CAS Latency Register */
#define	dmc_t_dqss_5			*((volatile UINT32_t *)(MPMC1_DMC_Base+0x018))	/* DMC t_dqss Register */
#define	dmc_t_mrd_5				*((volatile UINT32_t *)(MPMC1_DMC_Base+0x01C))	/* DMC t_mrd Register */
#define	dmc_t_ras_5				*((volatile UINT32_t *)(MPMC1_DMC_Base+0x020))	/* DMC t_ras Register */
#define	dmc_t_rc_5				*((volatile UINT32_t *)(MPMC1_DMC_Base+0x024))	/* DMC t_rc Register */
#define	dmc_t_rcd_5				*((volatile UINT32_t *)(MPMC1_DMC_Base+0x028))	/* DMC t_rcd Register */
#define	dmc_t_rfc_5				*((volatile UINT32_t *)(MPMC1_DMC_Base+0x02C))	/* DMC t_rfc Register */
#define	dmc_t_rp_5				*((volatile UINT32_t *)(MPMC1_DMC_Base+0x030))	/* DMC t_rp Register */
#define	dmc_t_rrd_5				*((volatile UINT32_t *)(MPMC1_DMC_Base+0x034))	/* DMC t_rrd Register */
#define	dmc_t_wr_5				*((volatile UINT32_t *)(MPMC1_DMC_Base+0x038))	/* DMC t_wr Register */
#define	dmc_t_wtr_5				*((volatile UINT32_t *)(MPMC1_DMC_Base+0x03C))	/* DMC t_wtr Register */
#define	dmc_t_xp_5				*((volatile UINT32_t *)(MPMC1_DMC_Base+0x040))	/* DMC t_xp Register */
#define	dmc_t_xsr_5				*((volatile UINT32_t *)(MPMC1_DMC_Base+0x044))	/* DMC t_xsr Register */
#define	dmc_t_esr_5				*((volatile UINT32_t *)(MPMC1_DMC_Base+0x048))	/* DMC t_esr Register */
/* ---- */
#define	dmc_id_0_cfg_5			*((volatile UINT32_t *)(MPMC1_DMC_Base+0x100))	/* DMC id_<0-5>_cfg Registers */
#define	dmc_id_1_cfg_5			*((volatile UINT32_t *)(MPMC1_DMC_Base+0x104))	/* DMC id_<0-5>_cfg Registers */
#define	dmc_id_2_cfg_5			*((volatile UINT32_t *)(MPMC1_DMC_Base+0x108))	/* DMC id_<0-5>_cfg Registers */
#define	dmc_id_3_cfg_5			*((volatile UINT32_t *)(MPMC1_DMC_Base+0x10C))	/* DMC id_<0-5>_cfg Registers */
#define	dmc_id_4_cfg_5			*((volatile UINT32_t *)(MPMC1_DMC_Base+0x110))	/* DMC id_<0-5>_cfg Registers */
#define	dmc_id_5_cfg_5			*((volatile UINT32_t *)(MPMC1_DMC_Base+0x114))	/* DMC id_<0-5>_cfg Registers */
/* ---- */
#define	dmc_chip_0_cfg_5		*((volatile UINT32_t *)(MPMC1_DMC_Base+0x200))	/* DMC chip_0_cfg Registers */
/* #define Reserved				*((volatile UINT32_t *)(MPMC1_DMC_Base+0x204)) */
/* #define Reserved				*((volatile UINT32_t *)(MPMC1_DMC_Base+0x208)) */
/* #define Reserved				*((volatile UINT32_t *)(MPMC1_DMC_Base+0x20C)) */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(MPMC1_DMC_Base+0x300)) */
#define	dmc_user_config_5		*((volatile UINT32_t *)(MPMC1_DMC_Base+0x304))	/* DMC user_config Register */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(MPMC1_DMC_Base+0xE00)) */
/* #define Reserved				*((volatile UINT32_t *)(MPMC1_DMC_Base+0xE04)) */
/* #define Reserved				*((volatile UINT32_t *)(MPMC1_DMC_Base+0xE08)) */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(MPMC1_DMC_Base+0xFE0)) */
/* ...                                                                             */
/* #define Reserved				*((volatile UINT32_t *)(MPMC1_DMC_Base+0xFEC)) */
/* #define Reserved				*((volatile UINT32_t *)(MPMC1_DMC_Base+0xFF0)) */
/* ...                                                                             */
/* #define Reserved				*((volatile UINT32_t *)(MPMC1_DMC_Base+0xFFC)) */

/* ------------------------------------------------------------------------ */
/* MPMC1_SMC				 : 0xF4311000	*/
#define	smc_memc_status_5		*((volatile UINT32_t *)(MPMC1_SMC_Base+0x000))	/* SMC Memory Controller Status Register */
#define	smc_memif_cfg_5			*((volatile UINT32_t *)(MPMC1_SMC_Base+0x004))	/* SMC Memory Interface Configuration Register */
/* #define Reserved				*((volatile UINT32_t *)(MPMC1_SMC_Base+0x008)) */
/* #define Reserved				*((volatile UINT32_t *)(MPMC1_SMC_Base+0x00C)) */
#define	smc_direct_cmd_5		*((volatile UINT32_t *)(MPMC1_SMC_Base+0x010))	/* SMC Direct Command Register */
#define	smc_set_cycles_5		*((volatile UINT32_t *)(MPMC1_SMC_Base+0x014))	/* SMC Set Cycles Register */
#define	smc_set_opmode_5		*((volatile UINT32_t *)(MPMC1_SMC_Base+0x018))	/* SMC Set Opmode Register */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(MPMC1_SMC_Base+0x020)) */
/* ---- */
#define	smc_sram_cycles0_0_5	*((volatile UINT32_t *)(MPMC1_SMC_Base+0x100))	/* SMC SRAM Cycles Registers <0-3> */
#define	smc_opmode0_0_5			*((volatile UINT32_t *)(MPMC1_SMC_Base+0x104))	/* SMC Opmode Registers <0-3> */
/* ---- */
#define	smc_sram_cycles0_1_5	*((volatile UINT32_t *)(MPMC1_SMC_Base+0x120))	/* SMC SRAM Cycles Registers <0-3> */
#define	smc_opmode0_1_5			*((volatile UINT32_t *)(MPMC1_SMC_Base+0x124))	/* SMC Opmode Registers <0-3> */
/* ---- */
#define	smc_sram_cycles0_2_5	*((volatile UINT32_t *)(MPMC1_SMC_Base+0x140))	/* SMC SRAM Cycles Registers <0-3> */
#define	smc_opmode0_2_5			*((volatile UINT32_t *)(MPMC1_SMC_Base+0x144))	/* SMC Opmode Registers <0-3> */
/* ---- */
#define	smc_sram_cycles0_3_5	*((volatile UINT32_t *)(MPMC1_SMC_Base+0x160))	/* SMC SRAM Cycles Registers <0-3> */
#define	smc_opmode0_3_5			*((volatile UINT32_t *)(MPMC1_SMC_Base+0x164))	/* SMC Opmode Registers <0-3> */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(MPMC1_SMC_Base+0x200)) */
/* #define Reserved				*((volatile UINT32_t *)(MPMC1_SMC_Base+0x204)) */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(MPMC1_SMC_Base+0xE00)) */
/* #define Reserved				*((volatile UINT32_t *)(MPMC1_SMC_Base+0xE04)) */
/* #define Reserved				*((volatile UINT32_t *)(MPMC1_SMC_Base+0xE08)) */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(MPMC1_SMC_Base+0xFE0)) */
/* ...                                                                             */
/* #define Reserved				*((volatile UINT32_t *)(MPMC1_SMC_Base+0xFEC)) */
/* #define Reserved				*((volatile UINT32_t *)(MPMC1_SMC_Base+0xFF0)) */
/* ...                                                                             */
/* #define Reserved				*((volatile UINT32_t *)(MPMC1_SMC_Base+0xFFC)) */

/* ------------------------------------------------------------------------ */
/* NDFC						 : 0xF2010000	*/
#define	NDFMCR0					*((volatile UINT32_t *)(NDFC_Base+0x0000))	/* NAND-Flash Control Register-0 */
#define	NDFMCR1					*((volatile UINT32_t *)(NDFC_Base+0x0004))	/* NAND-Flash Control Register-1 */
#define	NDFMCR2					*((volatile UINT32_t *)(NDFC_Base+0x0008))	/* NAND-Flash Control Register-2 */
#define	NDFINTC					*((volatile UINT32_t *)(NDFC_Base+0x000C))	/* NAND-Flash Interrupt Control Register */
#define	NDFDTR					*((volatile UINT32_t *)(NDFC_Base+0x0010))	/* NAND-Flash Data Register */
#define	NDECCRD0				*((volatile UINT32_t *)(NDFC_Base+0x0020))	/* NAND-Flash ECC-code Read Register-0 */
#define	NDECCRD1				*((volatile UINT32_t *)(NDFC_Base+0x0024))	/* NAND-Flash ECC-code Read Register-1 */
#define	NDECCRD2				*((volatile UINT32_t *)(NDFC_Base+0x0028))	/* NAND-Flash ECC-code Read Register-2 */
/* ---- */
#define	NDRSCA0					*((volatile UINT32_t *)(NDFC_Base+0x0030))	/* NAND-Flash Reed-Solomon Calculation result Address Register-0 */
#define	NDRSCD0					*((volatile UINT32_t *)(NDFC_Base+0x0034))	/* NAND-Flash Reed-Solomon Calculation result Data Register-0 */
#define	NDRSCA1					*((volatile UINT32_t *)(NDFC_Base+0x0038))	/* NAND-Flash Reed-Solomon Calculation result Address Register-1 */
#define	NDRSCD1					*((volatile UINT32_t *)(NDFC_Base+0x003C))	/* NAND-Flash Reed-Solomon Calculation result Data Register-1 */
#define	NDRSCA2					*((volatile UINT32_t *)(NDFC_Base+0x0040))	/* NAND-Flash Reed-Solomon Calculation result Address Register-2 */
#define	NDRSCD2					*((volatile UINT32_t *)(NDFC_Base+0x0044))	/* NAND-Flash Reed-Solomon Calculation result Data Register-2 */
#define	NDRSCA3					*((volatile UINT32_t *)(NDFC_Base+0x0048))	/* NAND-Flash Reed-Solomon Calculation result Address Register-3 */
#define	NDRSCD3					*((volatile UINT32_t *)(NDFC_Base+0x004C))	/* NAND-Flash Reed-Solomon Calculation result Data Register-3 */

/* ------------------------------------------------------------------------ */
/* TMRB_PWM					 : 0xF0040000	*/
#define	Timer0Load				*((volatile UINT32_t *)(TMR16_01_Base+0x000))	/* Timer0 Load value */
#define	Timer0Value				*((volatile UINT32_t *)(TMR16_01_Base+0x004))	/* The current value for Timer0 */
#define	Timer0Control			*((volatile UINT32_t *)(TMR16_01_Base+0x008))	/* Timer0 control register */
#define	Timer0IntClr			*((volatile UINT32_t *)(TMR16_01_Base+0x00C))	/* Timer0 interrupt clear */
#define	Timer0RIS				*((volatile UINT32_t *)(TMR16_01_Base+0x010))	/* Timer0 raw interrupt status */
#define	Timer0MIS				*((volatile UINT32_t *)(TMR16_01_Base+0x014))	/* Timer0 masked interrupt status */
#define	Timer0BGLoad			*((volatile UINT32_t *)(TMR16_01_Base+0x018))	/* Background load value for Timer0 */
#define	Timer0Mode				*((volatile UINT32_t *)(TMR16_01_Base+0x01C))	/* Timer0 mode register */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_01_Base+0x020)) */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_01_Base+0x040)) */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_01_Base+0x060)) */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_01_Base+0x064)) */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_01_Base+0x068)) */
/* ---- */
#define	Timer0Compare1			*((volatile UINT32_t *)(TMR16_01_Base+0x0A0))	/* Timer0 Compare value */
/* ---- */
#define	Timer0CmpIntClr1		*((volatile UINT32_t *)(TMR16_01_Base+0x0C0))	/* Timer0 Compare Interrupt clear */
/* ---- */
#define	Timer0CmpEn				*((volatile UINT32_t *)(TMR16_01_Base+0x0E0))	/* Timer0 Compare Enable */
#define	Timer0CmpRIS			*((volatile UINT32_t *)(TMR16_01_Base+0x0E4))	/* Timer0 Compare raw interrupt status */
#define	Timer0CmpMIS			*((volatile UINT32_t *)(TMR16_01_Base+0x0E8))	/* Timer0 Compare masked int status */
#define	Timer0BGCmp				*((volatile UINT32_t *)(TMR16_01_Base+0x0EC))	/* Background compare value for Timer0 */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_01_Base+0x0F0)) */
/* ---- */
#define	Timer1Load				*((volatile UINT32_t *)(TMR16_01_Base+0x100))	/* Timer1 Load value */
#define	Timer1Value				*((volatile UINT32_t *)(TMR16_01_Base+0x104))	/* The current value for Timer1 */
#define	Timer1Control			*((volatile UINT32_t *)(TMR16_01_Base+0x108))	/* Timer1 control register */
#define	Timer1IntClr			*((volatile UINT32_t *)(TMR16_01_Base+0x10C))	/* Timer1 interrupt clear */
#define	Timer1RIS				*((volatile UINT32_t *)(TMR16_01_Base+0x110))	/* Timer1 raw interrupt status */
#define	Timer1MIS				*((volatile UINT32_t *)(TMR16_01_Base+0x114))	/* Timer1 masked interrupt status */
#define	Timer1BGLoad			*((volatile UINT32_t *)(TMR16_01_Base+0x118))	/* Background load value for Timer1 */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_01_Base+0x120)) */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_01_Base+0x140)) */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_01_Base+0x160)) */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_01_Base+0x164)) */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_01_Base+0x168)) */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_01_Base+0x1A0)) */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_01_Base+0x1C0)) */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_01_Base+0x1E0)) */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_01_Base+0x1E4)) */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_01_Base+0x1E8)) */

/* ------------------------------------------------------------------------ */
/* TMR16_23					 : 0xF0041000	*/
#define	Timer2Load				*((volatile UINT32_t *)(TMR16_23_Base+0x000))	/* Timer2 Load value */
#define	Timer2Value				*((volatile UINT32_t *)(TMR16_23_Base+0x004))	/* The current value for Timer2 */
#define	Timer2Control			*((volatile UINT32_t *)(TMR16_23_Base+0x008))	/* Timer2 control register */
#define	Timer2IntClr			*((volatile UINT32_t *)(TMR16_23_Base+0x00C))	/* Timer2 interrupt clear */
#define	Timer2RIS				*((volatile UINT32_t *)(TMR16_23_Base+0x010))	/* Timer2 raw interrupt status */
#define	Timer2MIS				*((volatile UINT32_t *)(TMR16_23_Base+0x014))	/* Timer2 masked interrupt status */
#define	Timer2BGLoad			*((volatile UINT32_t *)(TMR16_23_Base+0x018))	/* Background load value for Timer2 */
#define	Timer2Mode				*((volatile UINT32_t *)(TMR16_23_Base+0x01C))	/* Timer2 mode register */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_23_Base+0x020)) */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_23_Base+0x040)) */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_23_Base+0x060)) */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_23_Base+0x064)) */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_23_Base+0x068)) */
/* ---- */
#define	Timer2Compare1			*((volatile UINT32_t *)(TMR16_23_Base+0x0A0))	/* Timer2 mode register */
/* ---- */
#define	Timer2CmpIntClr1		*((volatile UINT32_t *)(TMR16_23_Base+0x0C0))	/* Timer2 Compare Interrupt clear */
/* ---- */
#define	Timer2CmpEn				*((volatile UINT32_t *)(TMR16_23_Base+0x0E0))	/* Timer2 Compare Enable */
#define	Timer2CmpRIS			*((volatile UINT32_t *)(TMR16_23_Base+0x0E4))	/* Timer2 Compare raw interrupt status */
#define	Timer2CmpMIS			*((volatile UINT32_t *)(TMR16_23_Base+0x0E8))	/* Timer2 Compare masked int status */
#define	Timer2BGCmp				*((volatile UINT32_t *)(TMR16_23_Base+0x0EC))	/* Background compare value for Timer2 */
/* ---- */
#define	Timer3Load				*((volatile UINT32_t *)(TMR16_23_Base+0x100))	/* Timer3 Load value */
#define	Timer3Value				*((volatile UINT32_t *)(TMR16_23_Base+0x104))	/* The current value for Timer3 */
#define	Timer3Control			*((volatile UINT32_t *)(TMR16_23_Base+0x108))	/* Timer3 control register */
#define	Timer3IntClr			*((volatile UINT32_t *)(TMR16_23_Base+0x10C))	/* Timer3 interrupt clear */
#define	Timer3RIS				*((volatile UINT32_t *)(TMR16_23_Base+0x110))	/* Timer3 raw interrupt status */
#define	Timer3MIS				*((volatile UINT32_t *)(TMR16_23_Base+0x114))	/* Timer3 masked interrupt status */
#define	Timer3BGLoad			*((volatile UINT32_t *)(TMR16_23_Base+0x118))	/* Background load value for Timer3 */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_23_Base+0x120)) */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_23_Base+0x140)) */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_23_Base+0x160)) */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_23_Base+0x164)) */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_23_Base+0x168)) */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_23_Base+0x1A0)) */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_23_Base+0x1C0)) */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_23_Base+0x1E0)) */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_23_Base+0x1E4)) */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_23_Base+0x1E8)) */

/* ------------------------------------------------------------------------ */
/* TMR16_45					 : 0xF0042000	*/
#define	Timer4Load				*((volatile UINT32_t *)(TMR16_45_Base+0x000))	/* Timer4 Load value */
#define	Timer4Value				*((volatile UINT32_t *)(TMR16_45_Base+0x004))	/* The current value for Timer4 */
#define	Timer4Control			*((volatile UINT32_t *)(TMR16_45_Base+0x008))	/* Timer4 control register */
#define	Timer4IntClr			*((volatile UINT32_t *)(TMR16_45_Base+0x00C))	/* Timer4 interrupt clear */
#define	Timer4RIS				*((volatile UINT32_t *)(TMR16_45_Base+0x010))	/* Timer4 raw interrupt status */
#define	Timer4MIS				*((volatile UINT32_t *)(TMR16_45_Base+0x014))	/* Timer4 masked interrupt status */
#define	Timer4BGLoad			*((volatile UINT32_t *)(TMR16_45_Base+0x018))	/* Background load value for Timer4 */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_45_Base+0x01C)) */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_45_Base+0x020)) */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_45_Base+0x040)) */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_45_Base+0x060)) */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_45_Base+0x064)) */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_45_Base+0x068)) */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_45_Base+0x0A0)) */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_45_Base+0x0C0)) */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_45_Base+0x0E0)) */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_45_Base+0x0E4)) */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_45_Base+0x0E8)) */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_45_Base+0x0EC)) */
/* ---- */
#define	Timer5Load				*((volatile UINT32_t *)(TMR16_45_Base+0x100))	/* Timer5 Load value */
#define	Timer5Value				*((volatile UINT32_t *)(TMR16_45_Base+0x104))	/* The current value for Timer5 */
#define	Timer5Control			*((volatile UINT32_t *)(TMR16_45_Base+0x108))	/* Timer5 control register */
#define	Timer5IntClr			*((volatile UINT32_t *)(TMR16_45_Base+0x10C))	/* Timer5 interrupt clear  r */
#define	Timer5RIS				*((volatile UINT32_t *)(TMR16_45_Base+0x110))	/* Timer5 raw interrupt status */
#define	Timer5MIS				*((volatile UINT32_t *)(TMR16_45_Base+0x114))	/* Timer5 masked interrupt status */
#define	Timer5BGLoad			*((volatile UINT32_t *)(TMR16_45_Base+0x118))	/* Background load value for Timer5 */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_45_Base+0x120)) */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_45_Base+0x140)) */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_45_Base+0x160)) */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_45_Base+0x164)) */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_45_Base+0x168)) */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_45_Base+0x1A0)) */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_45_Base+0x1C0)) */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_45_Base+0x1E0)) */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_45_Base+0x1E4)) */
/* #define Reserved				*((volatile UINT32_t *)(TMR16_45_Base+0x1E8)) */

/* ------------------------------------------------------------------------ */
/* UART 0					 : 0xF2000000	*/
#define	UART0DR					*((volatile UINT32_t *)(UART0_Base+0x000))	/* UART 0 Data Register */
#define	UART0SR					*((volatile UINT32_t *)(UART0_Base+0x004))	/* UART 0 RX Status Register */
#define	UART0ECR				UART0SR										/* UART 0 Error Clear Register */
/* #define Reserved				*((volatile UINT32_t *)(UART0_Base+0x008)) */
/* #define Reserved				*((volatile UINT32_t *)(UART0_Base+0x00C)) */
/* #define Reserved				*((volatile UINT32_t *)(UART0_Base+0x010)) */
/* #define Reserved				*((volatile UINT32_t *)(UART0_Base+0x014)) */
#define	UART0FR					*((volatile UINT32_t *)(UART0_Base+0x018))	/* UART 0 Flag Register */
/* #define Reserved				*((volatile UINT32_t *)(UART0_Base+0x01C)) */
#define	UART0ILPR				*((volatile UINT32_t *)(UART0_Base+0x020))	/* UART 0 IrDA Low Power Counter Register */
#define	UART0IBRD				*((volatile UINT32_t *)(UART0_Base+0x024))	/* UART 0 Integer Baud Rate Divisor Register */
#define	UART0FBRD				*((volatile UINT32_t *)(UART0_Base+0x028))	/* UART 0 Fraction Baud Rate Divisor Register */
#define	UART0LCR_H				*((volatile UINT32_t *)(UART0_Base+0x02C))	/* UART 0 Data Format Control Register */
#define	UART0CR					*((volatile UINT32_t *)(UART0_Base+0x030))	/* UART 0 Control Register */
#define	UART0IFLS				*((volatile UINT32_t *)(UART0_Base+0x034))	/* UART 0 Interrupt FIF Level Select Register */
#define	UART0IMSC				*((volatile UINT32_t *)(UART0_Base+0x038))	/* UART 0 Interrupt Mask Set/Clear Register */
#define	UART0RIS				*((volatile UINT32_t *)(UART0_Base+0x03C))	/* UART 0 Raw Interrupt Status Register */
#define	UART0MIS				*((volatile UINT32_t *)(UART0_Base+0x040))	/* UART 0 Mask Interrupt Status Register */
#define	UART0ICR				*((volatile UINT32_t *)(UART0_Base+0x044))	/* UART 0 Interrupt Clear Register */
#define	UART0DMACR				*((volatile UINT32_t *)(UART0_Base+0x048))	/* UART 0 DMA Control Register */
/* #define Reserved				*((volatile UINT32_t *)(UART0_Base+0x04C)) */
/* ...                                                                          */
/* #define Reserved				*((volatile UINT32_t *)(UART0_Base+0x07C)) */
/* #define Reserved				*((volatile UINT32_t *)(UART0_Base+0x080)) */
/* ...                                                                          */
/* #define Reserved				*((volatile UINT32_t *)(UART0_Base+0x08C)) */
/* #define Reserved				*((volatile UINT32_t *)(UART0_Base+0x090)) */
/* ...                                                                          */
/* #define Reserved				*((volatile UINT32_t *)(UART0_Base+0xFCC)) */
/* #define Reserved				*((volatile UINT32_t *)(UART0_Base+0xFD0)) */
/* ...                                                                          */
/* #define Reserved				*((volatile UINT32_t *)(UART0_Base+0xFDC)) */
/* #define Reserved				*((volatile UINT32_t *)(UART0_Base+0xFE0)) */
/* #define Reserved				*((volatile UINT32_t *)(UART0_Base+0xFE4)) */
/* #define Reserved				*((volatile UINT32_t *)(UART0_Base+0xFE8)) */
/* #define Reserved				*((volatile UINT32_t *)(UART0_Base+0xFEC)) */
/* #define Reserved				*((volatile UINT32_t *)(UART0_Base+0xFF0)) */
/* #define Reserved				*((volatile UINT32_t *)(UART0_Base+0xFF4)) */
/* #define Reserved				*((volatile UINT32_t *)(UART0_Base+0xFF8)) */
/* #define Reserved				*((volatile UINT32_t *)(UART0_Base+0xFFC)) */

/* ------------------------------------------------------------------------ */
/* UART 1					 : 0xF2001000	*/
#define	UART1DR					*((volatile UINT32_t *)(UART1_Base+0x000))	/* UART 1 Data Register */
#define	UART1SR					*((volatile UINT32_t *)(UART1_Base+0x004))	/* UART 1 RX Status Register */
#define	UART1ECR				UART1SR										/* UART 1 Error Clear Register */
/* #define Reserved				*((volatile UINT32_t *)(UART1_Base+0x008)) */
/* #define Reserved				*((volatile UINT32_t *)(UART1_Base+0x00C)) */
/* #define Reserved				*((volatile UINT32_t *)(UART1_Base+0x010)) */
/* #define Reserved				*((volatile UINT32_t *)(UART1_Base+0x014)) */
#define	UART1FR					*((volatile UINT32_t *)(UART1_Base+0x018))	/* UART 1 Flag Register */
/* #define Reserved				*((volatile UINT32_t *)(UART1_Base+0x01C)) */
/* #define Reserved				*((volatile UINT32_t *)(UART1_Base+0x020)) */
#define	UART1IBRD				*((volatile UINT32_t *)(UART1_Base+0x024))	/* UART 1 Integer Baud Rate Divisor Register */
#define	UART1FBRD				*((volatile UINT32_t *)(UART1_Base+0x028))	/* UART 1 Fraction Baud Rate Divisor Register */
#define	UART1LCR_H				*((volatile UINT32_t *)(UART1_Base+0x02C))	/* UART 1 Data Format Control Register */
#define	UART1CR					*((volatile UINT32_t *)(UART1_Base+0x030))	/* UART 1 Control Register */
#define	UART1IFLS				*((volatile UINT32_t *)(UART1_Base+0x034))	/* UART 1 Interrupt FIF Level Select Register */
#define	UART1IMSC				*((volatile UINT32_t *)(UART1_Base+0x038))	/* UART 1 Interrupt Mask Set/Clear Register */
#define	UART1RIS				*((volatile UINT32_t *)(UART1_Base+0x03C))	/* UART 1 Raw Interrupt Status Register */
#define	UART1MIS				*((volatile UINT32_t *)(UART1_Base+0x040))	/* UART 1 Mask Interrupt Status Register */
#define	UART1ICR				*((volatile UINT32_t *)(UART1_Base+0x044))	/* UART 1 Interrupt Clear Register */
/* #define Reserved				*((volatile UINT32_t *)(UART1_Base+0x048)) */
/* #define Reserved				*((volatile UINT32_t *)(UART1_Base+0x04C)) */
/* ...                                                                          */
/* #define Reserved				*((volatile UINT32_t *)(UART1_Base+0x07C)) */
/* #define Reserved				*((volatile UINT32_t *)(UART1_Base+0x080)) */
/* ...                                                                          */
/* #define Reserved				*((volatile UINT32_t *)(UART1_Base+0x08C)) */
/* #define Reserved				*((volatile UINT32_t *)(UART1_Base+0x090)) */
/* ...                                                                          */
/* #define Reserved				*((volatile UINT32_t *)(UART1_Base+0xFCC)) */
/* #define Reserved				*((volatile UINT32_t *)(UART1_Base+0xFD0)) */
/* ...                                                                          */
/* #define Reserved				*((volatile UINT32_t *)(UART1_Base+0xFDC)) */
/* #define Reserved				*((volatile UINT32_t *)(UART1_Base+0xFE0)) */
/* #define Reserved				*((volatile UINT32_t *)(UART1_Base+0xFE4)) */
/* #define Reserved				*((volatile UINT32_t *)(UART1_Base+0xFE8)) */
/* #define Reserved				*((volatile UINT32_t *)(UART1_Base+0xFEC)) */
/* #define Reserved				*((volatile UINT32_t *)(UART1_Base+0xFF0)) */
/* #define Reserved				*((volatile UINT32_t *)(UART1_Base+0xFF4)) */
/* #define Reserved				*((volatile UINT32_t *)(UART1_Base+0xFF8)) */
/* #define Reserved				*((volatile UINT32_t *)(UART1_Base+0xFFC)) */

/* ------------------------------------------------------------------------ */
/* UART 2					 : 0xF2004000	*/
#define	UART2DR					*((volatile UINT32_t *)(UART2_Base+0x000))	/* UART 2 Data Register */
#define	UART2SR					*((volatile UINT32_t *)(UART2_Base+0x004))	/* UART 2 RX Status Register */
#define	UART2ECR				UART2SR										/* UART 2 Error Clear Register */
/* #define Reserved				*((volatile UINT32_t *)(UART2_Base+0x008)) */
/* #define Reserved				*((volatile UINT32_t *)(UART2_Base+0x00C)) */
/* #define Reserved				*((volatile UINT32_t *)(UART2_Base+0x010)) */
/* #define Reserved				*((volatile UINT32_t *)(UART2_Base+0x014)) */
#define	UART2FR					*((volatile UINT32_t *)(UART2_Base+0x018))	/* UART 2 Flag Register */
/* #define Reserved				*((volatile UINT32_t *)(UART2_Base+0x01C)) */
/* #define Reserved				*((volatile UINT32_t *)(UART2_Base+0x020)) */
#define	UART2IBRD				*((volatile UINT32_t *)(UART2_Base+0x024))	/* UART 2 Integer Baud Rate Divisor Register */
#define	UART2FBRD				*((volatile UINT32_t *)(UART2_Base+0x028))	/* UART 2 Fraction Baud Rate Divisor Register */
#define	UART2LCR_H				*((volatile UINT32_t *)(UART2_Base+0x02C))	/* UART 2 Data Format Control Register */
#define	UART2CR					*((volatile UINT32_t *)(UART2_Base+0x030))	/* UART 2 Control Register */
#define	UART2IFLS				*((volatile UINT32_t *)(UART2_Base+0x034))	/* UART 2 Interrupt FIF Level Select Register */
#define	UART2IMSC				*((volatile UINT32_t *)(UART2_Base+0x038))	/* UART 2 Interrupt Mask Set/Clear Register */
#define	UART2RIS				*((volatile UINT32_t *)(UART2_Base+0x03C))	/* UART 2 Raw Interrupt Status Register */
#define	UART2MIS				*((volatile UINT32_t *)(UART2_Base+0x040))	/* UART 2 Mask Interrupt Status Register */
#define	UART2ICR				*((volatile UINT32_t *)(UART2_Base+0x044))	/* UART 2 Interrupt Clear Register */
/* #define Reserved				*((volatile UINT32_t *)(UART2_Base+0x048)) */
/* #define Reserved				*((volatile UINT32_t *)(UART2_Base+0x04C)) */
/* ...                                                                          */
/* #define Reserved				*((volatile UINT32_t *)(UART2_Base+0x07C)) */
/* #define Reserved				*((volatile UINT32_t *)(UART2_Base+0x080)) */
/* ...                                                                          */
/* #define Reserved				*((volatile UINT32_t *)(UART2_Base+0x08C)) */
/* #define Reserved				*((volatile UINT32_t *)(UART2_Base+0x090)) */
/* ...                                                                          */
/* #define Reserved				*((volatile UINT32_t *)(UART2_Base+0xFCC)) */
/* #define Reserved				*((volatile UINT32_t *)(UART2_Base+0xFD0)) */
/* ...                                                                          */
/* #define Reserved				*((volatile UINT32_t *)(UART2_Base+0xFDC)) */
/* #define Reserved				*((volatile UINT32_t *)(UART2_Base+0xFE0)) */
/* #define Reserved				*((volatile UINT32_t *)(UART2_Base+0xFE4)) */
/* #define Reserved				*((volatile UINT32_t *)(UART2_Base+0xFE8)) */
/* #define Reserved				*((volatile UINT32_t *)(UART2_Base+0xFEC)) */
/* #define Reserved				*((volatile UINT32_t *)(UART2_Base+0xFF0)) */
/* #define Reserved				*((volatile UINT32_t *)(UART2_Base+0xFF4)) */
/* #define Reserved				*((volatile UINT32_t *)(UART2_Base+0xFF8)) */
/* #define Reserved				*((volatile UINT32_t *)(UART2_Base+0xFFC)) */

/* ------------------------------------------------------------------------ */
/* I2C_0					 : 0xF0070000	*/
#define	I2C0CR1					*((volatile UINT32_t *)(I2C0_Base+0x000))	/* I2C0 Control Register 1 */
#define	I2C0DBR					*((volatile UINT32_t *)(I2C0_Base+0x004))	/* I2C0 Data Buffer Register  */
#define	I2C0AR					*((volatile UINT32_t *)(I2C0_Base+0x008))	/* I2C0 (Slave) Address Register */
#define	I2C0CR2					*((volatile UINT32_t *)(I2C0_Base+0x00C))	/* I2C0 Control Register 2 */
#define	I2C0SR					I2C0CR2										/* I2C0 Status Register */
#define	I2C0PRS					*((volatile UINT32_t *)(I2C0_Base+0x010))	/* I2C0 Prescaler Clock Set Register */
#define	I2C0IE					*((volatile UINT32_t *)(I2C0_Base+0x014))	/* I2C0 Interrupt Enable Register */
#define	I2C0IR					*((volatile UINT32_t *)(I2C0_Base+0x018))	/* I2C0 Interrupt Register */

/* ------------------------------------------------------------------------ */
/* I2C_1					 : 0xF0071000	*/
#define	I2C1CR1					*((volatile UINT32_t *)(I2C1_Base+0x000))	/* I2C1 Control Register 1 */
#define	I2C1DBR					*((volatile UINT32_t *)(I2C1_Base+0x004))	/* I2C1 Data Buffer Register  */
#define	I2C1AR					*((volatile UINT32_t *)(I2C1_Base+0x008))	/* I2C1 (Slave) Address Register */
#define	I2C1CR2					*((volatile UINT32_t *)(I2C1_Base+0x00C))	/* I2C1 Control Register 2 */
#define	I2C1SR					I2C1CR2										/* I2C1 Status Register */
#define	I2C1PRS					*((volatile UINT32_t *)(I2C1_Base+0x010))	/* I2C1 Prescaler Clock Set Register */
#define	I2C1IE					*((volatile UINT32_t *)(I2C1_Base+0x014))	/* I2C1 Interrupt Enable Register */
#define	I2C1IR					*((volatile UINT32_t *)(I2C1_Base+0x018))	/* I2C1 Interrupt Register */

/* ------------------------------------------------------------------------ */
/* SSP0						 : 0xF2002000	*/
#define	SSP0CR0					*((volatile UINT32_t *)(SSP0_Base+0x000))	/* SSP 0 Control Register 0 */
#define	SSP0CR1					*((volatile UINT32_t *)(SSP0_Base+0x004))	/* SSP 0 Control Register 1 */
#define	SSP0DR					*((volatile UINT32_t *)(SSP0_Base+0x008))	/* SSP 0 Data Register */
#define	SSP0SR					*((volatile UINT32_t *)(SSP0_Base+0x00C))	/* SSP 0 Status Register */
#define	SSP0CPSR				*((volatile UINT32_t *)(SSP0_Base+0x010))	/* SSP 0 Clock Prescale Register */
#define	SSP0IMSC				*((volatile UINT32_t *)(SSP0_Base+0x014))	/* SSP 0 Interrupt Mask Set/Clear Control Register */
#define	SSP0RIS					*((volatile UINT32_t *)(SSP0_Base+0x018))	/* SSP 0 Raw Interrupt Status Register */
#define	SSP0MIS					*((volatile UINT32_t *)(SSP0_Base+0x01C))	/* SSP 0 Mask Interrupt Status Register */
#define	SSP0ICR					*((volatile UINT32_t *)(SSP0_Base+0x020))	/* SSP 0 Interrupt Clear Register */
#define	SSP0DMACR				*((volatile UINT32_t *)(SSP0_Base+0x024))	/* DMAC Control Register */
/* #define Reserved				*((volatile UINT32_t *)(SSP0_Base+0x028)) */
/* ...                                                                          */
/* #define Reserved				*((volatile UINT32_t *)(SSP0_Base+0xFFC)) */

/* ------------------------------------------------------------------------ */
/* SSP1						 : 0xF2003000	*/
#define	SSP1CR0					*((volatile UINT32_t *)(SSP1_Base+0x000))	/* SSP 1 Control Register 0 */
#define	SSP1CR1					*((volatile UINT32_t *)(SSP1_Base+0x004))	/* SSP 1 Control Register 1 */
#define	SSP1DR					*((volatile UINT32_t *)(SSP1_Base+0x008))	/* SSP 1 Data Register */
#define	SSP1SR					*((volatile UINT32_t *)(SSP1_Base+0x00C))	/* SSP 1 Status Register */
#define	SSP1CPSR				*((volatile UINT32_t *)(SSP1_Base+0x010))	/* SSP 1 Clock Prescale Register */
#define	SSP1IMSC				*((volatile UINT32_t *)(SSP1_Base+0x014))	/* SSP 1 Interrupt Mask Set/Clear Control Register */
#define	SSP1RIS					*((volatile UINT32_t *)(SSP1_Base+0x018))	/* SSP 1 Raw Interrupt Status Register */
#define	SSP1MIS					*((volatile UINT32_t *)(SSP1_Base+0x01C))	/* SSP 1 Mask Interrupt Status Register */
#define	SSP1ICR					*((volatile UINT32_t *)(SSP1_Base+0x020))	/* SSP 1 Interrupt Clear Register */
#define	SSP1DMACR				*((volatile UINT32_t *)(SSP1_Base+0x024))	/* SSP 1 DMA Control Register */
/* #define Reserved				*((volatile UINT32_t *)(SSP1_Base+0x028)) */
/* ...                                                                          */
/* #define Reserved				*((volatile UINT32_t *)(SSP1_Base+0xFFC)) */

/* ------------------------------------------------------------------------ */
/* UDC2						 : 0xF4400000	*/
#define	UDINTSTS				*((volatile UINT32_t *)(UDC2_Base+0x0000))	/* Interrupt Status Register */
#define	UDINTENB				*((volatile UINT32_t *)(UDC2_Base+0x0004))	/* Interrupt Enable Register */
#define	UDMWTOUT				*((volatile UINT32_t *)(UDC2_Base+0x0008))	/* Master Write Timeout Register */
#define	UDC2STSET				*((volatile UINT32_t *)(UDC2_Base+0x000C))	/* UDC2 Setting Register */
#define	UDMSTSET				*((volatile UINT32_t *)(UDC2_Base+0x0010))	/* DMAC Setting Register */
#define	DMACRDREQ				*((volatile UINT32_t *)(UDC2_Base+0x0014))	/* DMAC Read Requset Register */
#define	DMACRDVL				*((volatile UINT32_t *)(UDC2_Base+0x0018))	/* DMAC Read Value Register */
#define	UDC2RDREQ				*((volatile UINT32_t *)(UDC2_Base+0x001C))	/* UDC2 Read Request Register */
#define	UDC2RDVL				*((volatile UINT32_t *)(UDC2_Base+0x0020))	/* UDC2 Read Value Register */
/* #define Reserved				*((volatile UINT32_t *)(UDC2_Base+0x0024)) */
/* ...                                                                         */
/* #define Reserved				*((volatile UINT32_t *)(UDC2_Base+0x0038)) */
#define	ARBTSET					*((volatile UINT32_t *)(UDC2_Base+0x003C))	/* Arbiter Setting Register  */
#define	UDMWSADR				*((volatile UINT32_t *)(UDC2_Base+0x0040))	/* Master Write Start Address Register */
#define	UDMWEADR				*((volatile UINT32_t *)(UDC2_Base+0x0044))	/* Master Write End Address Register */
#define	UDMWCADR				*((volatile UINT32_t *)(UDC2_Base+0x0048))	/* Master Write Current Address Register */
#define	UDMWAHBADR				*((volatile UINT32_t *)(UDC2_Base+0x004C))	/* Master Write AHB Address Register */
#define	UDMRSADR				*((volatile UINT32_t *)(UDC2_Base+0x0050))	/* Master Read Start Address Register */
#define	UDMREADR				*((volatile UINT32_t *)(UDC2_Base+0x0054))	/* Master Read End Address Register */
#define	UDMRCADR				*((volatile UINT32_t *)(UDC2_Base+0x0058))	/* Master Read Current Address Register */
#define	UDMRAHBADR				*((volatile UINT32_t *)(UDC2_Base+0x005C))	/* Master Read AHB Address Register */
/* #define Reserved				*((volatile UINT32_t *)(UDC2_Base+0x0060)) */
/* ...                                                                         */
/* #define Reserved				*((volatile UINT32_t *)(UDC2_Base+0x007C)) */
#define	UDPWCTL					*((volatile UINT32_t *)(UDC2_Base+0x0080))	/* Power Detect Control Register */
#define	UDMSTSTS				*((volatile UINT32_t *)(UDC2_Base+0x0084))	/* Master Status Register */
#define	UDTOUTCNT				*((volatile UINT32_t *)(UDC2_Base+0x0088))	/* Timeout Count Register */
/* #define Reserved				*((volatile UINT32_t *)(UDC2_Base+0x008C)) */
/* ...                                                                         */
/* #define Reserved				*((volatile UINT32_t *)(UDC2_Base+0x01FC)) */
#if(MADDRESS_ACCESS == 0)
#define	UD2ADR					*((volatile UINT32_t *)(UDC2_Base+0x0200))	/* UDC2 Address-State Register */
#define	UD2FRM					*((volatile UINT32_t *)(UDC2_Base+0x0204))	/* UDC2 Frame Register */
#define	UD2TMD					*((volatile UINT32_t *)(UDC2_Base+0x0208))	/* UDC2 USB-Testmode Register */
#define	UD2CMD					*((volatile UINT32_t *)(UDC2_Base+0x020C))	/* UDC2 Command Register */
#define	UD2BRQ					*((volatile UINT32_t *)(UDC2_Base+0x0210))	/* UDC2 bRequest-bmRequestType Register */
#define	UD2WVL					*((volatile UINT32_t *)(UDC2_Base+0x0214))	/* UDC2 wValue Register */
#define	UD2WIDX					*((volatile UINT32_t *)(UDC2_Base+0x0218))	/* UDC2 wIndex Register */
#define	UD2WLGTH				*((volatile UINT32_t *)(UDC2_Base+0x021C))	/* UDC2 wLength Register */
#define	UD2INT					*((volatile UINT32_t *)(UDC2_Base+0x0220))	/* UDC2 INT Register */
#define	UD2INTEP				*((volatile UINT32_t *)(UDC2_Base+0x0224))	/* UDC2 INT_EP Register */
#define	UD2INTEPMASK			*((volatile UINT32_t *)(UDC2_Base+0x0228))	/* UDC2 INT_EP_MASK Register */
#define	UD2INTRX0				*((volatile UINT32_t *)(UDC2_Base+0x022C))	/* UDC2 INT_RX_DATA0 Register */
#define	UD2EP0MSZ				*((volatile UINT32_t *)(UDC2_Base+0x0230))	/* UDC2 EP0_MaxPacketSize Register */
#define	UD2EP0STS				*((volatile UINT32_t *)(UDC2_Base+0x0234))	/* UDC2 EP0_Status Register */
#define	UD2EP0DSZ				*((volatile UINT32_t *)(UDC2_Base+0x0238))	/* UDC2 EP0_Datasize Register */
#define	UD2EP0FIFO				*((volatile UINT32_t *)(UDC2_Base+0x023C))	/* UDC2 EP0_FIFO Register */
#define	UD2EP1MSZ				*((volatile UINT32_t *)(UDC2_Base+0x0240))	/* UDC2 EP1_MaxPacketSize Register */
#define	UD2EP1STS				*((volatile UINT32_t *)(UDC2_Base+0x0244))	/* UDC2 EP1_Status Register */
#define	UD2EP1DSZ				*((volatile UINT32_t *)(UDC2_Base+0x0248))	/* UDC2 EP1_Datasize Register */
#define	UD2EP1FIFO				*((volatile UINT32_t *)(UDC2_Base+0x024C))	/* UDC2 EP1_FIFO Register */
#define	UD2EP2MSZ				*((volatile UINT32_t *)(UDC2_Base+0x0250))	/* UDC2 EP2_MaxPacketSize Register */
#define	UD2EP2STS				*((volatile UINT32_t *)(UDC2_Base+0x0254))	/* UDC2 EP2_Status Register */
#define	UD2EP2DSZ				*((volatile UINT32_t *)(UDC2_Base+0x0258))	/* UDC2 EP2_Datasize Register */
#define	UD2EP2FIFO				*((volatile UINT32_t *)(UDC2_Base+0x025C))	/* UDC2 EP2_FIFO Register */
#define	UD2EP3MSZ				*((volatile UINT32_t *)(UDC2_Base+0x0260))	/* UDC2 EP3_MaxPacketSize Register */
#define	UD2EP3STS				*((volatile UINT32_t *)(UDC2_Base+0x0264))	/* UDC2 EP3_Status Register */
#define	UD2EP3DSZ				*((volatile UINT32_t *)(UDC2_Base+0x0268))	/* UDC2 EP3_Datasize Register */
#define	UD2EP3FIFO				*((volatile UINT32_t *)(UDC2_Base+0x026C))	/* UDC2 EP3_FIFO Register */
/* #define Reserved				*((volatile UINT32_t *)(UDC2_Base+0x0270) */
/* ...                                                                         */
/* #define Reserved				*((volatile UINT32_t *)(UDC2_Base+0x032C)) */
#define	UD2INTNAK				*((volatile UINT32_t *)(UDC2_Base+0x0330))	/* UDC2 INT_NAK Register */
#define	UD2INTNAKMSK			*((volatile UINT32_t *)(UDC2_Base+0x0334))	/* UDC2 INT_NAK_MASK Register */
/* #define Reserved				*((volatile UINT32_t *)(UDC2_Base+0x0338)) */
/* ...                                                                         */
/* #define Reserved				*((volatile UINT32_t *)(UDC2_Base+0x03FC)) */
#else /* #if(MADDRESS_ACCESS == 0)	*/
#define	UD2ADR					( UINT32_t )(UDC2_Base+0x0200)	/* UDC2 Address-State Register */
#define	UD2FRM					( UINT32_t )(UDC2_Base+0x0204)	/* UDC2 Frame Register */
#define	UD2TMD					( UINT32_t )(UDC2_Base+0x0208)	/* UDC2 USB-Testmode Register */
#define	UD2CMD					( UINT32_t )(UDC2_Base+0x020C)	/* UDC2 Command Register */
#define	UD2BRQ					( UINT32_t )(UDC2_Base+0x0210)	/* UDC2 bRequest-bmRequestType Register */
#define	UD2WVL					( UINT32_t )(UDC2_Base+0x0214)	/* UDC2 wValue Register */
#define	UD2WIDX					( UINT32_t )(UDC2_Base+0x0218)	/* UDC2 wIndex Register */
#define	UD2WLGTH				( UINT32_t )(UDC2_Base+0x021C)	/* UDC2 wLength Register */
#define	UD2INT					( UINT32_t )(UDC2_Base+0x0220)	/* UDC2 INT Register */
#define	UD2INTEP				( UINT32_t )(UDC2_Base+0x0224)	/* UDC2 INT_EP Register */
#define	UD2INTEPMASK			( UINT32_t )(UDC2_Base+0x0228)	/* UDC2 INT_EP_MASK Register */
#define	UD2INTRX0				( UINT32_t )(UDC2_Base+0x022C)	/* UDC2 INT_RX_DATA0 Register */
#define	UD2EP0MSZ				( UINT32_t )(UDC2_Base+0x0230)	/* UDC2 EP0_MaxPacketSize Register */
#define	UD2EP0STS				( UINT32_t )(UDC2_Base+0x0234)	/* UDC2 EP0_Status Register */
#define	UD2EP0DSZ				( UINT32_t )(UDC2_Base+0x0238)	/* UDC2 EP0_Datasize Register */
#define	UD2EP0FIFO				( UINT32_t )(UDC2_Base+0x023C)	/* UDC2 EP0_FIFO Register */
#define	UD2EP1MSZ				( UINT32_t )(UDC2_Base+0x0240)	/* UDC2 EP1_MaxPacketSize Register */
#define	UD2EP1STS				( UINT32_t )(UDC2_Base+0x0244)	/* UDC2 EP1_Status Register */
#define	UD2EP1DSZ				( UINT32_t )(UDC2_Base+0x0248)	/* UDC2 EP1_Datasize Register */
#define	UD2EP1FIFO				( UINT32_t )(UDC2_Base+0x024C)	/* UDC2 EP1_FIFO Register */
#define	UD2EP2MSZ				( UINT32_t )(UDC2_Base+0x0250)	/* UDC2 EP2_MaxPacketSize Register */
#define	UD2EP2STS				( UINT32_t )(UDC2_Base+0x0254)	/* UDC2 EP2_Status Register */
#define	UD2EP2DSZ				( UINT32_t )(UDC2_Base+0x0258)	/* UDC2 EP2_Datasize Register */
#define	UD2EP2FIFO				( UINT32_t )(UDC2_Base+0x025C)	/* UDC2 EP2_FIFO Register */
#define	UD2EP3MSZ				( UINT32_t )(UDC2_Base+0x0260)	/* UDC2 EP3_MaxPacketSize Register */
#define	UD2EP3STS				( UINT32_t )(UDC2_Base+0x0264)	/* UDC2 EP3_Status Register */
#define	UD2EP3DSZ				( UINT32_t )(UDC2_Base+0x0268)	/* UDC2 EP3_Datasize Register */
#define	UD2EP3FIFO				( UINT32_t )(UDC2_Base+0x026C)	/* UDC2 EP3_FIFO Register */
/* #define	Reserved				( UINT32_t )(UDC2_Base+0x0270) */
/* ...                                                                         */
/* #define	Reserved				( UINT32_t )(UDC2_Base+0x032C) */
#define	UD2INTNAK				( UINT32_t )(UDC2_Base+0x0330)	/* UDC2 INT_NAK Register */
#define	UD2INTNAKMSK			( UINT32_t )(UDC2_Base+0x0334)	/* UDC2 INT_NAK_MASK Register */
#define	UD2INTNAK				( UINT32_t )(UDC2_Base+0x0330)	/* UDC2 INT_NAK Register */
#define	UD2INTNAKMSK			( UINT32_t )(UDC2_Base+0x0334)	/* UDC2 INT_NAK_MASK Register */
/* #define	Reserved				( UINT32_t *)(UDC2_Base+0x0338) */
/* ...                                                                         */
/* #define	Reserved				( UINT32_t *)(UDC2_Base+0x03FC) */
#endif

/* ------------------------------------------------------------------------ */
/* I2S						 : 0xF2040000	*/
#define	I2STCON					*((volatile UINT32_t *)(I2S_Base+0x0000))	/* I2S TX Control Register */
#define	I2STSLVON				*((volatile UINT32_t *)(I2S_Base+0x0004))	/* I2S TX Slave WS/SCK Control Register */
#define	I2STFCLR				*((volatile UINT32_t *)(I2S_Base+0x0008))	/* I2S TX FIFO CLR ON/OFF Register */
#define	I2STMS					*((volatile UINT32_t *)(I2S_Base+0x000C))	/* I2S TX Master/Slave Register */
#define	I2STMCON				*((volatile UINT32_t *)(I2S_Base+0x0010))	/* I2S TX Master I2S1WS/I2S1SCLK Period Register */
#define	I2STMSTP				*((volatile UINT32_t *)(I2S_Base+0x0014))	/* I2S TX Master Stop Register */
#define	I2STDMA1				*((volatile UINT32_t *)(I2S_Base+0x0018))	/* I2S TX DMA Ready Register */
/* #define Reserved				*((volatile UINT32_t *)(I2S_Base+0x001C)) */
#define	I2SRCON					*((volatile UINT32_t *)(I2S_Base+0x0020))	/* I2S RX Control Register */
#define	I2SRSLVON				*((volatile UINT32_t *)(I2S_Base+0x0024))	/* I2S RX Slave WS/SCK Control Register */
#define	I2SFRFCLR				*((volatile UINT32_t *)(I2S_Base+0x0028))	/* I2S RX FIFO CLR ON/OFF Register */
#define	I2SRMS					*((volatile UINT32_t *)(I2S_Base+0x002C))	/* I2S RX Master/Slave Register */
#define	I2SRMCON				*((volatile UINT32_t *)(I2S_Base+0x0030))	/* I2S RX Master I2S1WS/I2S1SCLK Period Register*/
#define	I2SRMSTP				*((volatile UINT32_t *)(I2S_Base+0x0034))	/* I2S RX Master Stop Register */
#define	I2SRDMA1				*((volatile UINT32_t *)(I2S_Base+0x0038))	/* I2S RX DMA Ready Register */
/* #define Reserved				*((volatile UINT32_t *)(I2S_Base+0x003C)) */
/* ---- */
#define	I2SCOMMON				*((volatile UINT32_t *)(I2S_Base+0x0044))	/* I2S Common Control Register */
#define	I2STST					*((volatile UINT32_t *)(I2S_Base+0x0048))	/* I2S TX Status Register */
#define	I2SRST					*((volatile UINT32_t *)(I2S_Base+0x004C))	/* I2S RX Status Register */
#define	I2SINT					*((volatile UINT32_t *)(I2S_Base+0x0050))	/* I2S Interrupt Register */
#define	I2SINTMSK				*((volatile UINT32_t *)(I2S_Base+0x0054))	/* I2S Interrupt Mask Register */
/* ---- */
#define	I2STDAT					*((volatile UINT32_t *)(I2S_Base+0x1000))	/* I2S TX FIFO Window DMA Target -0x1FFF */
#define	I2SRDAT					*((volatile UINT32_t *)(I2S_Base+0x2000))	/* I2S RX FIFO Window DMA Target -0x2FFF */

/* ------------------------------------------------------------------------ */
/* LCDC						 : 0xF4200000	*/
#define	LCDTiming0				*((volatile UINT32_t *)(LCDC_Base+0x0000))	/* LCD Horizontal Control Register */
#define	LCDTiming1				*((volatile UINT32_t *)(LCDC_Base+0x0004))	/* LCD Vertical Control Register */
#define	LCDTiming2				*((volatile UINT32_t *)(LCDC_Base+0x0008))	/* LCD Clock/Signal Polarity Control Register */
#define	LCDTiming3				*((volatile UINT32_t *)(LCDC_Base+0x000C))	/* LCD Row Termination Control Register */
#define	LCDUPBASE				*((volatile UINT32_t *)(LCDC_Base+0x0010))	/* LCD Upper Panel Frame Base Address Register */
#define	LCDLPBASE				*((volatile UINT32_t *)(LCDC_Base+0x0014))	/* LCD Lower Panel Frame Base Address Register */
#define	LCDIMSC					*((volatile UINT32_t *)(LCDC_Base+0x0018))	/* LCD Interrupt Mask Set/Clear Register */
#define	LCDControl				*((volatile UINT32_t *)(LCDC_Base+0x001C))	/* LCD Control Register */
#define	LCDRIS					*((volatile UINT32_t *)(LCDC_Base+0x0020))	/* LCD Raw Interrupt Status Register */
#define	LCDMIS					*((volatile UINT32_t *)(LCDC_Base+0x0024))	/* LCD Mask Interrupt Status Register */
#define	LCDICR					*((volatile UINT32_t *)(LCDC_Base+0x0028))	/* LCD Interrupt Clear Register */
#define	LCDUPCURR				*((volatile UINT32_t *)(LCDC_Base+0x002C))	/* LCD Upper Panel Current Address Value Registers */
#define	LCDLPCURR				*((volatile UINT32_t *)(LCDC_Base+0x0030))	/* LCD Lower Panel Current Address Value Registers */
/* ---- */
#define	LCDPalette				*((volatile UINT32_t *)(LCDC_Base+0x0200))	/* LCD Color Palette Register -0x3FC */

/* ------------------------------------------------------------------------ */
/* LCDCOP					 : 0xF00B0000	*/
#define	STN64CR					*((volatile UINT32_t *)(LCDCOP_Base+0x0000))	/* STN64 Control register */

/* ------------------------------------------------------------------------ */
/* LCDDA					 : 0xF2050000	*/
#define	LDACR0					*((volatile UINT32_t *)(LCDDA_Base+0x0000))	/* LCDDA Control Register 0 */
#define	LDADRSRC1				*((volatile UINT32_t *)(LCDDA_Base+0x0004))	/* LCDDA Density Ratio of Source 1 Picture */
#define	LDADRSRC0				*((volatile UINT32_t *)(LCDDA_Base+0x0008))	/* LCDDA Density Ratio of Source 0 Picture */
#define	LDAFCPSRC1				*((volatile UINT32_t *)(LCDDA_Base+0x000C))	/* LCDDA Replaced Font Area Color pallet of Source1 */
#define	LDAEFCPSRC1				*((volatile UINT32_t *)(LCDDA_Base+0x0010))	/* LCDDA Replaced Except Font Area Color pallet of Source1 */
#define	LDADVSRC1				*((volatile UINT32_t *)(LCDDA_Base+0x0014))	/* LCDDA Replaced Except Font Area Color pallet of Source1 */
#define	LDACR2					*((volatile UINT32_t *)(LCDDA_Base+0x0018))	/* LCDDA Control Register 2 */
#define	LDADXDST				*((volatile UINT32_t *)(LCDDA_Base+0x001C))	/* LCDDA X-Delta Value (Write Step) address Register of Destination */
#define	LDADYDST				*((volatile UINT32_t *)(LCDDA_Base+0x0020))	/* LCDDA Y-Delta Value (Write Step) address Register of Destination */
#define	LDASSIZE				*((volatile UINT32_t *)(LCDDA_Base+0x0024))	/* LCDDA Source Picture Size */
#define	LDADSIZE				*((volatile UINT32_t *)(LCDDA_Base+0x0028))	/* LCDDA Destination Picture Size */
#define	LDAS0AD					*((volatile UINT32_t *)(LCDDA_Base+0x002C))	/* LCDDA Source 0 Start Address */
#define	LDADAD					*((volatile UINT32_t *)(LCDDA_Base+0x0030))	/* LCDDA Destination Start Address */
#define	LDACR1					*((volatile UINT32_t *)(LCDDA_Base+0x0034))	/* LCDDA Control Register1 */
#define	LDADVSRC0				*((volatile UINT32_t *)(LCDDA_Base+0x0038))	/* LCDDA Delta Value (Read Step) address Register of Source 0 */

/* ------------------------------------------------------------------------ */
/* TSI						 : 0xF0060000	*/
#define	TSICR0					*((volatile UINT32_t *)(TSI_Base+0x01F0))	/* TSI Control Register0 */
#define	TSICR1					*((volatile UINT32_t *)(TSI_Base+0x01F4))	/* TSI Control Register1 */

/* ------------------------------------------------------------------------ */
/* CMSI						 : 0xF2020000	*/
#define	CMSCR					*((volatile UINT32_t *)(CMSI_Base+0x0000))	/* CMOS Image Sensor Control Register */
#define	CMSCV					*((volatile UINT32_t *)(CMSI_Base+0x0004))	/* CMOS Image Sensor Color Space Conversion Register */
#define	CMSCVP0					*((volatile UINT32_t *)(CMSI_Base+0x0008))	/* CMOS Image Sensor Color Conversion Parameter Register0 */
#define	CMSCVP1					*((volatile UINT32_t *)(CMSI_Base+0x000C))	/* CMOS Image Sensor Color Conversion Parameter Register1 */
#define	CMSYD					*((volatile UINT32_t *)(CMSI_Base+0x0010))	/* CMOS Image Sensor Soft Conversrion Y-data Resister */
#define	CMSUD					*((volatile UINT32_t *)(CMSI_Base+0x0014))	/* CMOS Image Sensor Soft Conversrion U-data Resister */
#define	CMSVD					*((volatile UINT32_t *)(CMSI_Base+0x0018))	/* CMOS Image Sensor Soft Conversrion V-data Resister */
/* ---- */
#define	CMSFPT					*((volatile UINT32_t *)(CMSI_Base+0x0020))	/* CMOS Image Sensor FIFO Port Read Register */
#define	CMSSCTR					*((volatile UINT32_t *)(CMSI_Base+0x0024))	/* CMOS Image Sensor Scaling & Trimming Control Register */
/* ---- */
#define	CMSTS					*((volatile UINT32_t *)(CMSI_Base+0x0030))	/* CMOS Image Sensor Trimming Space Start Point Setting Register */
#define	CMSTE					*((volatile UINT32_t *)(CMSI_Base+0x0034))	/* CMOS Image Sensor Trimming Space End Point Setting Register */
/* ---- */
#define	CMSSCDMA				*((volatile UINT32_t *)(CMSI_Base+0x0040))	/* CMOS Image Sensor Soft Conversrion DMA YUV-Data */

/* ------------------------------------------------------------------------ */
/* RTC_MLD					 : 0xF0030000	*/
#define	RTCDATA					*((volatile UINT32_t *)(RTC_MLD_Base+0x0000))	/* RTC Data Register */
#define	RTCCOMP					*((volatile UINT32_t *)(RTC_MLD_Base+0x0004))	/* RTC Compare Register */
#define	RTCPRST					*((volatile UINT32_t *)(RTC_MLD_Base+0x0008))	/* RTC Preset Register */
/* ---- */
#define	MLDALMINV				*((volatile UINT32_t *)(RTC_MLD_Base+0x0100))	/* Melody Alarm Invert Register */
#define	MLDALMSEL				*((volatile UINT32_t *)(RTC_MLD_Base+0x0104))	/* Melody Alarm signal Select Register */
#define	ALMCNTCR				*((volatile UINT32_t *)(RTC_MLD_Base+0x0108))	/* Alarm Counter Control Register */
#define	ALMPATERN				*((volatile UINT32_t *)(RTC_MLD_Base+0x010C))	/* Alarm Pattern Register */
#define	MLDCNTCR				*((volatile UINT32_t *)(RTC_MLD_Base+0x0110))	/* Melody Counter Control Register */
#define	MLDFRQ					*((volatile UINT32_t *)(RTC_MLD_Base+0x0114))	/* Melody Frequency Register */
/* ---- */
#define	RTCALMINCTR				*((volatile UINT32_t *)(RTC_MLD_Base+0x0200))	/* RTC ALM Interrupt Control Register */
#define	RTCALMMIS				*((volatile UINT32_t *)(RTC_MLD_Base+0x0204))	/* RTC ALM Interrupt Status Register */

/* ------------------------------------------------------------------------ */
/* ADC						 : 0xF0080000	*/
#define	ADREG0L					*((volatile UINT32_t *)(ADC_Base+0x0000))	/* AD Conversion Result Register 0 Low */
#define	ADREG0H					*((volatile UINT32_t *)(ADC_Base+0x0004))	/* AD Conversion Result Register 0 High */
#define	ADREG1L					*((volatile UINT32_t *)(ADC_Base+0x0008))	/* AD Conversion Result Register 1 Low */
#define	ADREG1H					*((volatile UINT32_t *)(ADC_Base+0x000C))	/* AD Conversion Result Register 1 High */
#define	ADREG2L					*((volatile UINT32_t *)(ADC_Base+0x0010))	/* AD Conversion Result Register 2 Low */
#define	ADREG2H					*((volatile UINT32_t *)(ADC_Base+0x0014))	/* AD Conversion Result Register 2 High */
#define	ADREG3L					*((volatile UINT32_t *)(ADC_Base+0x0018))	/* AD Conversion Result Register 3 Low */
#define	ADREG3H					*((volatile UINT32_t *)(ADC_Base+0x001C))	/* AD Conversion Result Register 3 High */
#define	ADREG4L					*((volatile UINT32_t *)(ADC_Base+0x0020))	/* AD Conversion Result Register 4 Low */
#define	ADREG4H					*((volatile UINT32_t *)(ADC_Base+0x0024))	/* AD Conversion Result Register 4 High */
#define	ADREG5L					*((volatile UINT32_t *)(ADC_Base+0x0028))	/* AD Conversion Result Register 5 Low */
#define	ADREG5H					*((volatile UINT32_t *)(ADC_Base+0x002C))	/* AD Conversion Result Register 5 High */
#define	ADREG6L					*((volatile UINT32_t *)(ADC_Base+0x0030))	/* AD Conversion Result Register 6 Low */
#define	ADREG6H					*((volatile UINT32_t *)(ADC_Base+0x0034))	/* AD Conversion Result Register 6 High */
#define	ADREG7L					*((volatile UINT32_t *)(ADC_Base+0x0038))	/* AD Conversion Result Register 7 Low */
#define	ADREG7H					*((volatile UINT32_t *)(ADC_Base+0x003C))	/* AD Conversion Result Register 7 High */
#define	ADREGSPL				*((volatile UINT32_t *)(ADC_Base+0x0040))	/* High-priority AD Conversion Result Register SP Low */
#define	ADREGSPH				*((volatile UINT32_t *)(ADC_Base+0x0044))	/* High-priority AD Conversion Result Register SP High */
#define	ADCOMREGL				*((volatile UINT32_t *)(ADC_Base+0x0048))	/* AD Conversion Result Compare Criterion Register 0 Low */
#define	ADCOMREGH				*((volatile UINT32_t *)(ADC_Base+0x004C))	/* AD Conversion Result Compare Criterion Register 0 High */
#define	ADMOD0					*((volatile UINT32_t *)(ADC_Base+0x0050))	/* AD Mode Control Register 0 */
#define	ADMOD1					*((volatile UINT32_t *)(ADC_Base+0x0054))	/* AD Mode Control Register 1 */
#define	ADMOD2					*((volatile UINT32_t *)(ADC_Base+0x0058))	/* AD Mode Control Register 2 */
#define	ADMOD3					*((volatile UINT32_t *)(ADC_Base+0x005C))	/* AD Mode Control Register 3 */
#define	ADMOD4					*((volatile UINT32_t *)(ADC_Base+0x0060))	/* AD Mode Control Register 4 */
/* #define Reserved				*((volatile UINT32_t *)(ADC_Base+0x0064)) */
/* #define Reserved				*((volatile UINT32_t *)(ADC_Base+0x0068)) */
/* #define Reserved				*((volatile UINT32_t *)(ADC_Base+0x006C)) */
#define	ADCLK					*((volatile UINT32_t *)(ADC_Base+0x0070))	/* AD Conversion Clock Setting Register */
#define	ADIE					*((volatile UINT32_t *)(ADC_Base+0x0074))	/* AD Interrupt Enable Register */
#define	ADIS					*((volatile UINT32_t *)(ADC_Base+0x0078))	/* AD Interrupt Status Register */
#define	ADIC					*((volatile UINT32_t *)(ADC_Base+0x007C))	/* AD Interrupt Clear Register */
/* #define Reserved				*((volatile UINT32_t *)(ADC_Base+0x0080)) */
/* ...                                                                         */
/* #define Reserved				*((volatile UINT32_t *)(ADC_Base+0x0FFC) */

/* ------------------------------------------------------------------------ */
/* WDT						 : 0xF0010000	*/
#define	WdogLoad				*((volatile UINT32_t *)(WDT_Base+0x0000))	/* Watchdog load register */
#define	WdogValue				*((volatile UINT32_t *)(WDT_Base+0x0004))	/* The current value for the watchdog counter */
#define	WdogControl				*((volatile UINT32_t *)(WDT_Base+0x0008))	/* Watchdog control register */
#define	WdogIntClr				*((volatile UINT32_t *)(WDT_Base+0x000C))	/* Clears the watchdog interrupt */
#define	WdogRIS					*((volatile UINT32_t *)(WDT_Base+0x0010))	/* Watchdog raw interrupt status */
#define	WdogMIS					*((volatile UINT32_t *)(WDT_Base+0x0014))	/* Watchdog masked interrupt status */
/* ---- */
#define	WdogLock				*((volatile UINT32_t *)(WDT_Base+0x0C00))	/* Watchdog Lock register */

/* ------------------------------------------------------------------------ */
/* PMC						 : 0xF0020000	*/
#define	BPADATA					*((volatile UINT32_t *)(PMC_Base+0x0900))	/* PortA Data Set Register when Power Cut Mode */
#define	BPBDATA					*((volatile UINT32_t *)(PMC_Base+0x0904))	/* PortB Data Set Register when Power Cut Mode */
#define	BPCDATA					*((volatile UINT32_t *)(PMC_Base+0x0908))	/* PortC Data Set Register when Power Cut Mode */
#define	BPDDATA					*((volatile UINT32_t *)(PMC_Base+0x090C))	/* PortD Data Set Register when Power Cut Mode */
/* #define Reserved				*((volatile UINT32_t *)(PMC_Base+0x0910) */
#define	BPFDATA					*((volatile UINT32_t *)(PMC_Base+0x0914))	/* PortF Data Set Register when Power Cut Mode */
#define	BPGDATA					*((volatile UINT32_t *)(PMC_Base+0x0918))	/* PortG Data Set Register when Power Cut Mode */
/* #define Reserved				*((volatile UINT32_t *)(PMC_Base+0x091C) */
/* ---- */
#define	BPJDATA					*((volatile UINT32_t *)(PMC_Base+0x0924))	/* PortJ Data Set Register when Power Cut Mode */
#define	BPKDATA					*((volatile UINT32_t *)(PMC_Base+0x0928))	/* PortK Data Set Register when Power Cut Mode */
#define	BPLDATA					*((volatile UINT32_t *)(PMC_Base+0x092C))	/* PortL Data Set Register when Power Cut Mode */
#define	BPMDATA					*((volatile UINT32_t *)(PMC_Base+0x0930))	/* PortM Data Set Register when Power Cut Mode */
#define	BPNDATA					*((volatile UINT32_t *)(PMC_Base+0x0934))	/* PortN Data Set Register when Power Cut Mode */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(PMC_Base+0x093C) */
/* ---- */
#define	BPRDATA					*((volatile UINT32_t *)(PMC_Base+0x0944))	/* PortR Data Set Register when Power Cut Mode */
/* ---- */
#define	BPTDATA					*((volatile UINT32_t *)(PMC_Base+0x094C))	/* PortT Data Set Register when Power Cut Mode */
#define	BPUDATA					*((volatile UINT32_t *)(PMC_Base+0x0950))	/* PortU Data Set Register when Power Cut Mode */
#define	BPVDATA					*((volatile UINT32_t *)(PMC_Base+0x0954))	/* PortV Data Set Register when Power Cut Mode */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(PMC_Base+0x0B80)) */
#define	BPBOE					*((volatile UINT32_t *)(PMC_Base+0x0B84))	/* PortB Data Out Enable Control when Power Cut Mode */
#define	BPCOE					*((volatile UINT32_t *)(PMC_Base+0x0B88))	/* PortC Data Out Enable Control when Power Cut Mode */
#define	BPDOE					*((volatile UINT32_t *)(PMC_Base+0x0B8C))	/* PortD Data Out Enable Control when Power Cut Mode */
/* #define Reserved				*((volatile UINT32_t *)(PMC_Base+0x0B90)) */
#define	BPFOE					*((volatile UINT32_t *)(PMC_Base+0x0B94))	/* PortF Data Out Enable Control when Power Cut Mode */
#define	BPGOE					*((volatile UINT32_t *)(PMC_Base+0x0B98))	/* PortG Data Out Enable Control when Power Cut Mode */
/* #define Reserved				*((volatile UINT32_t *)(PMC_Base+0x0B9C)) */
/* ---- */
#define	BPJOE					*((volatile UINT32_t *)(PMC_Base+0x0BA4))	/* PortJ Data Out Enable Control when Power Cut Mode */
#define	BPKOE					*((volatile UINT32_t *)(PMC_Base+0x0BA8))	/* PortK Data Out Enable Control when Power Cut Mode */
#define	BPLOE					*((volatile UINT32_t *)(PMC_Base+0x0BAC))	/* PortL Data Out Enable Control when Power Cut Mode */
#define	BPMOE					*((volatile UINT32_t *)(PMC_Base+0x0BB0))	/* PortM Data Out Enable Control when Power Cut Mode */
#define	BPNOE					*((volatile UINT32_t *)(PMC_Base+0x0BB4))	/* PortN Data Out Enable Control when Power Cut Mode */
/* ---- */
/* #define Reserved				*((volatile UINT32_t *)(PMC_Base+0x0BBC)) */
/* ---- */
#define	BPROE					*((volatile UINT32_t *)(PMC_Base+0x0BC4))	/* PortR Data Out Enable Control when Power Cut Mode */
/* ---- */
#define	BPTOE					*((volatile UINT32_t *)(PMC_Base+0x0BCC))	/* PortT Data Out Enable Control when Power Cut Mode */
#define	BPUOE					*((volatile UINT32_t *)(PMC_Base+0x0BD0))	/* PortU Data Out Enable Control when Power Cut Mode */
#define	BPVOE					*((volatile UINT32_t *)(PMC_Base+0x0BD4))	/* PortV Data Out Enable Control when Power Cut Mode */
/* ---- */
#define	BSADATA					*((volatile UINT32_t *)(PMC_Base+0x0800))	/* SA Data Set Register when Power Cut Mode */
#define	BSBDATA					*((volatile UINT32_t *)(PMC_Base+0x0804))	/* SB Data Set Register when Power Cut Mode */
#define	BSCDATA					*((volatile UINT32_t *)(PMC_Base+0x0808))	/* SC Data Set Register when Power Cut Mode */
#define	BSDDATA					*((volatile UINT32_t *)(PMC_Base+0x080C))	/* SD Data Set Register when Power Cut Mode */
#define	BSEDATA					*((volatile UINT32_t *)(PMC_Base+0x0810))	/* SE Data Set Register when Power Cut Mode */
#define	BSFDATA					*((volatile UINT32_t *)(PMC_Base+0x0814))	/* SF Data Set Register when Power Cut Mode */
#define	BSGDATA					*((volatile UINT32_t *)(PMC_Base+0x0818))	/* SG Data Set Register when Power Cut Mode */
#define	BSHDATA					*((volatile UINT32_t *)(PMC_Base+0x081C))	/* SH Data Set Register when Power Cut Mode */
/* ---- */
#define	BSJDATA					*((volatile UINT32_t *)(PMC_Base+0x0824))	/* SJ Data Set Register when Power Cut Mode */
#define	BSKDATA					*((volatile UINT32_t *)(PMC_Base+0x0828))	/* SK Data Set Register when Power Cut Mode */
#define	BSLDATA					*((volatile UINT32_t *)(PMC_Base+0x082C))	/* SL Data Set Register when Power Cut Mode */
/* ---- */
#define	BSTDATA					*((volatile UINT32_t *)(PMC_Base+0x084C))	/* ST Data Set Register when Power Cut Mode */
#define	BSUDATA					*((volatile UINT32_t *)(PMC_Base+0x0850))	/* SU Data Set Register when Power Cut Mode */
/* ---- */
#define	BSAOE					*((volatile UINT32_t *)(PMC_Base+0x0A80))	/* SA Data Out Enable Control when Power Cut Mode */
#define	BSBOE					*((volatile UINT32_t *)(PMC_Base+0x0A84))	/* SB Data Out Enable Control when Power Cut Mode */
#define	BSCOE					*((volatile UINT32_t *)(PMC_Base+0x0A88))	/* SC Data Out Enable Control when Power Cut Mode */
#define	BSDOE					*((volatile UINT32_t *)(PMC_Base+0x0A8C))	/* SD Data Out Enable Control when Power Cut Mode */
#define	BSEOE					*((volatile UINT32_t *)(PMC_Base+0x0A90))	/* SE Data Out Enable Control when Power Cut Mode */
#define	BSFOE					*((volatile UINT32_t *)(PMC_Base+0x0A94))	/* SF Data Out Enable Control when Power Cut Mode */
#define	BSGOE					*((volatile UINT32_t *)(PMC_Base+0x0A98))	/* SG Data Out Enable Control when Power Cut Mode */
#define	BSHOE					*((volatile UINT32_t *)(PMC_Base+0x0A9C))	/* SH Data Out Enable Control when Power Cut Mode */
/* ---- */
#define	BSJOE					*((volatile UINT32_t *)(PMC_Base+0x0AA4))	/* SJ Data Out Enable Control when Power Cut Mode */
#define	BSKOE					*((volatile UINT32_t *)(PMC_Base+0x0AA8))	/* SK Data Out Enable Control when Power Cut Mode */
#define	BSLOE					*((volatile UINT32_t *)(PMC_Base+0x0AAC))	/* SL Data Out Enable Control when Power Cut Mode */
/* ---- */
#define	BSTOE					*((volatile UINT32_t *)(PMC_Base+0x0ACC))	/* ST Data Out Enable Control when Power Cut Mode */
#define	BSUOE					*((volatile UINT32_t *)(PMC_Base+0x0AD0))	/* SU Data Out Enable Control when Power Cut Mode */
/* #define Reserved				*((volatile UINT32_t *)(PMC_Base+0x0AD4)) */
/* #define Reserved				*((volatile UINT32_t *)(PMC_Base+0x0AD8)) */
/* ---- */
#define BPAIE					*((volatile UINT32_t *)(PMC_Base+0x0D80))	/* PortA WakeUp Input Enable */
/* ---- */
#define BPCIE					*((volatile UINT32_t *)(PMC_Base+0x0D88))	/* PortC WakeUp Input Enable */
#define BPDIE					*((volatile UINT32_t *)(PMC_Base+0x0D8C))	/* PortD WakeUp Input Enable */
/* ---- */
#define BPFIE					*((volatile UINT32_t *)(PMC_Base+0x0D94))	/* PortF WakeUp Input Enable */
/* ---- */
#define BPNIE					*((volatile UINT32_t *)(PMC_Base+0x0DB4))	/* PortN WakeUp Input Enable */
/* ---- */
#define BPRIE					*((volatile UINT32_t *)(PMC_Base+0x0DC4))	/* PortR WakeUp Input Enable */
/* ---- */
#define	BPARELE					*((volatile UINT32_t *)(PMC_Base+0x0200))	/* PortA Enable Write Register of Wake-up trigger from Power Cut Mode */
#define	BPDRELE					*((volatile UINT32_t *)(PMC_Base+0x0204))	/* PortD Enable Write Register of Wake-up trigger from Power Cut Mode */
#define	BRTRELE					*((volatile UINT32_t *)(PMC_Base+0x0208))	/* RTC Request Enable Register of Wake-up trigger from Power Cut Mode */
#define	BRXRELE					*((volatile UINT32_t *)(PMC_Base+0x020C))	/* Other Port Enable Register of Wake-up trigger from Power Cut Mode */
/* ---- */
#define	BPAEDGE	 				*((volatile UINT32_t *)(PMC_Base+0x0220))	/* PortA Selection Register of Wake-up trigger Edge from Power Cut Mode */
#define	BPDEDGE					*((volatile UINT32_t *)(PMC_Base+0x0224))	/* PortD Selection Register of Wake-up trigger Edge from Power Cut Mode */
/* ---- */
#define	BPXEDGE					*((volatile UINT32_t *)(PMC_Base+0x022C))	/* Other Port Selection Register of Wake-up trigger Edge from Power Cut Mode */
/* ---- */
#define	BADRINT					*((volatile UINT32_t *)(PMC_Base+0x0240))	/* PortA Wake-up Interrupt status Register */
#define	BPDRINT					*((volatile UINT32_t *)(PMC_Base+0x0244))	/* PortD Wake-up Interrupt status Register */
#define	BRTRINT					*((volatile UINT32_t *)(PMC_Base+0x0248))	/* RTC Wake-up Interrupt status Register */
#define	BPXRINT					*((volatile UINT32_t *)(PMC_Base+0x024C))	/* Other Port Wake-up Interrupt status Register */
/* ---- */
#define	PMCDRV					*((volatile UINT32_t *)(PMC_Base+0x0260))	/* External Port Driverbility control Register */
/* ---- */
#define	DMCCKECTL				*((volatile UINT32_t *)(PMC_Base+0x0280))	/* DMCCKE pin setting Register(PCM mode) */
/* ---- */
#define	PMCCTL					*((volatile UINT32_t *)(PMC_Base+0x0300))	/* Power Management Circuit Control Register */
/* ---- */
#define	PMCWV1					*((volatile UINT32_t *)(PMC_Base+0x400))	/* Control for PMC Register */
/* ---- */
#define	PMCWV2					*((volatile UINT32_t *)(PMC_Base+0x408))	/* Control for XT_CTL Control Register */
/* ---- */
#define PMCRES					*((volatile UINT32_t *)(PMC_Base+0x41C))	/* PCM_ON Flag Clear Register for PMCCTL_R<PMC_ON> */

/* ------------------------------------------------------------------------ */
/* USBH						 : 0xF4500000	*/
#define	HcRevision				*((volatile UINT32_t *)(USBH_Base+0x0000))	/*  */
#define	HcControl				*((volatile UINT32_t *)(USBH_Base+0x0004))	/*  */
#define	HcCommandStatus			*((volatile UINT32_t *)(USBH_Base+0x0008))	/*  */
#define	HcInterruptStatus		*((volatile UINT32_t *)(USBH_Base+0x000C))	/*  */
#define	HcInterruptEnable		*((volatile UINT32_t *)(USBH_Base+0x0010))	/*  */
#define	HcInterruptDisable		*((volatile UINT32_t *)(USBH_Base+0x0014))	/*  */
#define	HcHCCA					*((volatile UINT32_t *)(USBH_Base+0x0018))	/*  */
#define	HcPeriodCurrentED		*((volatile UINT32_t *)(USBH_Base+0x001C))	/*  */
#define	HcControlHeadED			*((volatile UINT32_t *)(USBH_Base+0x0020))	/*  */
#define	HcControlCurrentED		*((volatile UINT32_t *)(USBH_Base+0x0024))	/*  */
#define	HcBulkHeadED			*((volatile UINT32_t *)(USBH_Base+0x0028))	/*  */
#define	HcBulkCurrentED			*((volatile UINT32_t *)(USBH_Base+0x002C))	/*  */
#define	HcDoneHead				*((volatile UINT32_t *)(USBH_Base+0x0030))	/*  */
#define	HcFmInterval			*((volatile UINT32_t *)(USBH_Base+0x0034))	/*  */
#define	HcFmRemaining			*((volatile UINT32_t *)(USBH_Base+0x0038))	/*  */
#define	HcFmNumber				*((volatile UINT32_t *)(USBH_Base+0x003C))	/*  */
#define	HcPeriodStart			*((volatile UINT32_t *)(USBH_Base+0x0040))	/*  */
#define	HcLSThreshold			*((volatile UINT32_t *)(USBH_Base+0x0044))	/*  */
#define	HcRhDescriptorA			*((volatile UINT32_t *)(USBH_Base+0x0048))	/*  */
#define	HcRhDescripterB			*((volatile UINT32_t *)(USBH_Base+0x004C))	/*  */
#define	HcRhStatus				*((volatile UINT32_t *)(USBH_Base+0x0050))	/*  */
#define	HcRhPortStatus			*((volatile UINT32_t *)(USBH_Base+0x0054))	/*  */
/* #define Reserved				*((volatile UINT32_t *)(USBH_Base+0x0058))	  */
/* #define Reserved				*((volatile UINT32_t *)(USBH_Base+0x005C))	  */
/* ---- */
#define	HCBCR0					*((volatile UINT32_t *)(USBH_Base+0x0080))	/*  */

/* ------------------------------------------------------------------------ */
/* OFD						 : 0xF0090000	*/
#define	CLKSCR1					*((volatile UINT32_t *)(OFD_Base+0x0000))	/* Oscillation frequency detectiton control register1 */
#define	CLKSCR2					*((volatile UINT32_t *)(OFD_Base+0x0004))	/* Oscillation frequency detectiton control register2 */
#define	CLKSCR3					*((volatile UINT32_t *)(OFD_Base+0x0008))	/* Oscillation frequency detectiton control register3 */
/* ---- */
#define	CLKSMN					*((volatile UINT32_t *)(OFD_Base+0x0010))	/* Lower detection frequency setting register */
/* ---- */
#define	CLKSMX					*((volatile UINT32_t *)(OFD_Base+0x0020))	/* Higher detection frequency setting register */

/* ************************************************************************ */
#endif	/* _tmpa900cm_h_ */
