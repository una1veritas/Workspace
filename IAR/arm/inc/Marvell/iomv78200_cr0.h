/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Marvell MV78200 Core 0
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler.
 **
 **    (c) Copyright IAR Systems 2010
 **
 **    $Revision: 39950 $
 **
 **    Note: Only little endian addressing of 8 bit registers.
 ***************************************************************************/

#ifndef __IOMV78200_CR0_H
#define __IOMV78200_CR0_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    MV78200 SPECIAL FUNCTION REGISTERS
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

/* Window 1-5,8 Control Register */
typedef struct{
__REG32 win_en            : 1;
__REG32                   : 3;
__REG32 Target			      : 4;
__REG32 Attr				      : 8;
__REG32 Size			        :16;
} __amr_wcr_bits;

/* Window 6-7 Control Register */
typedef struct{
__REG32 win_en            : 1;
__REG32 WinWrProt         : 1;
__REG32                   : 2;
__REG32 Target			      : 4;
__REG32 Attr				      : 8;
__REG32 Size			        :16;
} __amr_w6cr_bits;

/* CPU Configuration Register */
typedef struct{
__REG32 			            	: 1;
__REG32 VecInitLoc        	: 1;
__REG32 AHBErrorProp      	: 1;
__REG32 EndianInit        	: 1;
__REG32 SelMRVL_ID        	: 1;
__REG32 MMU_Disabled      	: 1;
__REG32 					        	: 2;
__REG32 CPU2MbusLTickDrv		: 4;
__REG32 CPU2MbusLTickSample	: 4;
__REG32                   	: 6;
__REG32 ECCOn					      : 1;
__REG32 							      : 9;
} __csr_cr_bits;

/* CPU Control and Status Register */
typedef struct{
__REG32 PEX0En             	: 1;
__REG32 PEX1En	          	: 1;
__REG32 SelfInt		        	: 1;
__REG32 CPUReset	        	: 1;
__REG32 				          	:11;
__REG32 BigEndian		      	: 1;
__REG32 L2Exist							: 1;
__REG32 L2WT								: 1;
__REG32 										: 3;
__REG32 L2c_size						: 1;
__REG32 										: 2;
__REG32 L2CTagParEn					: 1;
__REG32                   	: 2;
__REG32 MPCpuId					    : 1;
__REG32 							      : 4;
} __csr_csr_bits;

/* Mbus-L to Mbus Bridge Interrupt Cause Register */
typedef struct{
__REG32 CPUSelfInt         	: 1;
__REG32 CPUTimer0IntReq    	: 1;
__REG32 CPUTimer1IntReq    	: 1;
__REG32 CPUWDTimerIntReq   	: 1;
__REG32 AccessErr		      	: 1;
__REG32 Bit64Err  					: 1;
__REG32 CPUTimer2IntReq			: 1;
__REG32 CPUTimer3IntReq			: 1;
__REG32 										:24;
} __csr_m2micr_bits;

/* Mbus-L to Mbus Bridge Interrupt Mask Register */
typedef struct{
__REG32 CPUSelfIntMask     	: 1;
__REG32 CPUTimer0IntMask  	: 1;
__REG32 CPUTimer1IntMask   	: 1;
__REG32 CPUWDTimerIntMask  	: 1;
__REG32 AccessErrMask      	: 1;
__REG32 Bit64ErrMask  			: 1;
__REG32 CPUTimer2IntMask		: 1;
__REG32 CPUTimer3IntMask		: 1;
__REG32 										:24;
} __csr_m2mimr_bits;

/* Power Management Control Register */
typedef struct{
__REG32                    	: 1;
__REG32 GE3_PU            	: 1;
__REG32 GE2_PU             	: 1;
__REG32 GE1_PU  	          : 1;
__REG32 GE0_PU  	          : 1;
__REG32 PEX00_PU           	: 1;
__REG32 PEX01_PU           	: 1;
__REG32 PEX02_PU           	: 1;
__REG32 PEX03_PU           	: 1;
__REG32 PEX10_PU           	: 1;
__REG32 PEX11_PU           	: 1;
__REG32 PEX12_PU           	: 1;
__REG32 PEX13_PU           	: 1;
__REG32 SATA0_PHY_PU       	: 1;
__REG32 SATA0_HC_PU       	: 1;
__REG32 SATA1_PHY_PU       	: 1;
__REG32 SATA1_HC_PU        	: 1;
__REG32 USB0_PU            	: 1;
__REG32 USB1_PU            	: 1;
__REG32 USB2_PU       			: 1;
__REG32 IDMA_PU         		: 1;
__REG32 XOR_PU          		: 1;
__REG32 Crypto_PU        		: 1;
__REG32 Device_PU        		: 1;
__REG32 TDM_PU          		: 1;
__REG32 										: 7;
} __csr_pmcr_bits;

/* CPU Timing Adjustment Register */
typedef struct{
__REG32                    	:21;
__REG32 WaitForWBEmptyNcb   : 1;
__REG32 										:10;
} __csr_tar_bits;

/* CPU Interface MbusTime Out Register */
typedef struct{
__REG32 Timeout             : 8;
__REG32 										: 8;
__REG32 TimeoutEn           : 1;
__REG32 										:15;
} __csr_mtor_bits;

/* CPU Memory Power Management Control Register */
typedef struct{
__REG32 cpu_mem_power_down  : 1;
__REG32 										:31;
} __csr_mpmcr_bits;

/* L2 RAM Power Management Control Register */
typedef struct{
__REG32 L2RAMConfiguration2 : 8;
__REG32 										:24;
} __csr_l2pmcr_bits;

/* Inbound Doorbell Register */
typedef struct{
__REG32 InboundIntCs0       : 1;
__REG32 InboundIntCs1       : 1;
__REG32 InboundIntCs2       : 1;
__REG32 InboundIntCs3       : 1;
__REG32 InboundIntCs4       : 1;
__REG32 InboundIntCs5       : 1;
__REG32 InboundIntCs6       : 1;
__REG32 InboundIntCs7       : 1;
__REG32 InboundIntCs8       : 1;
__REG32 InboundIntCs9       : 1;
__REG32 InboundIntCs10      : 1;
__REG32 InboundIntCs11      : 1;
__REG32 InboundIntCs12      : 1;
__REG32 InboundIntCs13      : 1;
__REG32 InboundIntCs14      : 1;
__REG32 InboundIntCs15      : 1;
__REG32 InboundIntCs16      : 1;
__REG32 InboundIntCs17      : 1;
__REG32 InboundIntCs18      : 1;
__REG32 InboundIntCs19      : 1;
__REG32 InboundIntCs20      : 1;
__REG32 InboundIntCs21      : 1;
__REG32 InboundIntCs22      : 1;
__REG32 InboundIntCs23      : 1;
__REG32 InboundIntCs24      : 1;
__REG32 InboundIntCs25      : 1;
__REG32 InboundIntCs26      : 1;
__REG32 InboundIntCs27      : 1;
__REG32 InboundIntCs28      : 1;
__REG32 InboundIntCs29      : 1;
__REG32 InboundIntCs30      : 1;
__REG32 InboundIntCs31      : 1;
} __db_id_bits;

/* Inbound Doorbell Mask Register */
typedef struct{
__REG32 InboundIntCsMask0       : 1;
__REG32 InboundIntCsMask1       : 1;
__REG32 InboundIntCsMask2       : 1;
__REG32 InboundIntCsMask3       : 1;
__REG32 InboundIntCsMask4       : 1;
__REG32 InboundIntCsMask5       : 1;
__REG32 InboundIntCsMask6       : 1;
__REG32 InboundIntCsMask7       : 1;
__REG32 InboundIntCsMask8       : 1;
__REG32 InboundIntCsMask9       : 1;
__REG32 InboundIntCsMask10      : 1;
__REG32 InboundIntCsMask11      : 1;
__REG32 InboundIntCsMask12      : 1;
__REG32 InboundIntCsMask13      : 1;
__REG32 InboundIntCsMask14      : 1;
__REG32 InboundIntCsMask15      : 1;
__REG32 InboundIntCsMask16      : 1;
__REG32 InboundIntCsMask17      : 1;
__REG32 InboundIntCsMask18      : 1;
__REG32 InboundIntCsMask19      : 1;
__REG32 InboundIntCsMask20      : 1;
__REG32 InboundIntCsMask21      : 1;
__REG32 InboundIntCsMask22      : 1;
__REG32 InboundIntCsMask23      : 1;
__REG32 InboundIntCsMask24      : 1;
__REG32 InboundIntCsMask25      : 1;
__REG32 InboundIntCsMask26      : 1;
__REG32 InboundIntCsMask27      : 1;
__REG32 InboundIntCsMask28      : 1;
__REG32 InboundIntCsMask29      : 1;
__REG32 InboundIntCsMask30      : 1;
__REG32 InboundIntCsMask31      : 1;
} __db_idm_bits;

/* Outbound Doorbell Register */
typedef struct{
__REG32 OutboundIntCs0       : 1;
__REG32 OutboundIntCs1       : 1;
__REG32 OutboundIntCs2       : 1;
__REG32 OutboundIntCs3       : 1;
__REG32 OutboundIntCs4       : 1;
__REG32 OutboundIntCs5       : 1;
__REG32 OutboundIntCs6       : 1;
__REG32 OutboundIntCs7       : 1;
__REG32 OutboundIntCs8       : 1;
__REG32 OutboundIntCs9       : 1;
__REG32 OutboundIntCs10      : 1;
__REG32 OutboundIntCs11      : 1;
__REG32 OutboundIntCs12      : 1;
__REG32 OutboundIntCs13      : 1;
__REG32 OutboundIntCs14      : 1;
__REG32 OutboundIntCs15      : 1;
__REG32 OutboundIntCs16      : 1;
__REG32 OutboundIntCs17      : 1;
__REG32 OutboundIntCs18      : 1;
__REG32 OutboundIntCs19      : 1;
__REG32 OutboundIntCs20      : 1;
__REG32 OutboundIntCs21      : 1;
__REG32 OutboundIntCs22      : 1;
__REG32 OutboundIntCs23      : 1;
__REG32 OutboundIntCs24      : 1;
__REG32 OutboundIntCs25      : 1;
__REG32 OutboundIntCs26      : 1;
__REG32 OutboundIntCs27      : 1;
__REG32 OutboundIntCs28      : 1;
__REG32 OutboundIntCs29      : 1;
__REG32 OutboundIntCs30      : 1;
__REG32 OutboundIntCs31      : 1;
} __db_od_bits;

/* Outbound Doorbell Mask Register */
typedef struct{
__REG32 OutboundIntCsMask0       : 1;
__REG32 OutboundIntCsMask1       : 1;
__REG32 OutboundIntCsMask2       : 1;
__REG32 OutboundIntCsMask3       : 1;
__REG32 OutboundIntCsMask4       : 1;
__REG32 OutboundIntCsMask5       : 1;
__REG32 OutboundIntCsMask6       : 1;
__REG32 OutboundIntCsMask7       : 1;
__REG32 OutboundIntCsMask8       : 1;
__REG32 OutboundIntCsMask9       : 1;
__REG32 OutboundIntCsMask10      : 1;
__REG32 OutboundIntCsMask11      : 1;
__REG32 OutboundIntCsMask12      : 1;
__REG32 OutboundIntCsMask13      : 1;
__REG32 OutboundIntCsMask14      : 1;
__REG32 OutboundIntCsMask15      : 1;
__REG32 OutboundIntCsMask16      : 1;
__REG32 OutboundIntCsMask17      : 1;
__REG32 OutboundIntCsMask18      : 1;
__REG32 OutboundIntCsMask19      : 1;
__REG32 OutboundIntCsMask20      : 1;
__REG32 OutboundIntCsMask21      : 1;
__REG32 OutboundIntCsMask22      : 1;
__REG32 OutboundIntCsMask23      : 1;
__REG32 OutboundIntCsMask24      : 1;
__REG32 OutboundIntCsMask25      : 1;
__REG32 OutboundIntCsMask26      : 1;
__REG32 OutboundIntCsMask27      : 1;
__REG32 OutboundIntCsMask28      : 1;
__REG32 OutboundIntCsMask29      : 1;
__REG32 OutboundIntCsMask30      : 1;
__REG32 OutboundIntCsMask31      : 1;
} __db_odm_bits;

/* Semaphore0 Register */
typedef struct{
__REG32 Semaphore0        : 3;
__REG32                   : 5;
__REG32 Semaphore1        : 3;
__REG32                   : 5;
__REG32 Semaphore2        : 3;
__REG32                   : 5;
__REG32 Semaphore3        : 3;
__REG32                   : 5;
} __semaph_bits;

/* Window X Base Address Register */
typedef struct{
__REG32 EN                : 1;
__REG32                   :15;
__REG32 Base              :16;
} __l2ncam_wbar_bits;

/* Window X Size Address Register */
typedef struct{
__REG32                   :16;
__REG32 Size              :16;
} __l2ncam_wsar_bits;

/* CS Window X Size Register */
typedef struct{
__REG32 En                : 1;
__REG32 WrProtect         : 1;
__REG32 Win0_CS           : 2;
__REG32                   :20;
__REG32 Size              : 8;
} __ddr_cswsar_bits;

/* SDRAM Configuration Register */
typedef struct{
__REG32 Refresh           :14;
__REG32                   : 1;
__REG32 DDR64_32          : 1;
__REG32 P2DWr             : 1;
__REG32 RegDIMM           : 1;
__REG32 ECC               : 1;
__REG32 IErr              : 1;
__REG32                   : 4;
__REG32 SRMode            : 1;
__REG32                   : 7;
} __ddr_sdram_cr_bits;

/* DDR Controller Control (Low) Register */
typedef struct{
__REG32                   : 4;
__REG32 _2T               : 1;
__REG32 SRClk             : 1;
__REG32 CtrlPos           : 1;
__REG32                   : 5;
__REG32 Clk1Drv           : 1;
__REG32 Clk2Drv           : 1;
__REG32                   : 4;
__REG32 LockEn            : 1;
__REG32                   : 1;
__REG32 SBOutDel          : 4;
__REG32 SBInDel           : 4;
__REG32                   : 4;
} __ddr_cclr_bits;

/* SDRAM Timing (Low) Register */
typedef struct{
__REG32 tRAS              : 4;
__REG32 tRCD              : 4;
__REG32 tRP               : 4;
__REG32 tWR               : 4;
__REG32 tWTR              : 4;
__REG32 EtRAS             : 1;
__REG32                   : 3;
__REG32 tRRD              : 4;
__REG32 tRTP              : 4;
} __ddr_sdram_tlr_bits;

/* SDRAM Timing (High) Register */
typedef struct{
__REG32 tRFC              : 7;
__REG32 tR2R              : 2;
__REG32 tR2W_W2R          : 2;
__REG32 tW2W              : 2;
__REG32                   :19;
} __ddr_sdram_thr_bits;

/* SDRAM Address Control Register */
typedef struct{
__REG32 CS0Width          : 2;
__REG32 CS0Size           : 2;
__REG32 CS1Width          : 2;
__REG32 CS1Size           : 2;
__REG32 CS2Width          : 2;
__REG32 CS2Size           : 2;
__REG32 CS3Width          : 2;
__REG32 CS3Size           : 2;
__REG32 CS0AddrSel        : 1;
__REG32 CS1AddrSel        : 1;
__REG32 CS2AddrSel        : 1;
__REG32 CS3AddrSel        : 1;
__REG32                   :12;
} __ddr_sdram_acr_bits;

/* SDRAM Open Pages Control Register */
typedef struct{
__REG32 OPEn              : 1;
__REG32                   :31;
} __ddr_sdram_opcr_bits;

/* SDRAM Operation Register */
typedef struct{
__REG32 Cmd               : 4;
__REG32                   :28;
} __ddr_sdram_or_bits;

/* SDRAM Mode Register */
typedef struct{
__REG32 BL                : 3;
__REG32 BT                : 1;
__REG32 CL                : 3;
__REG32 TM                : 1;
__REG32 DLLRst            : 1;
__REG32 WR                : 3;
__REG32 PD                : 1;
__REG32                   :19;
} __ddr_sdram_mr_bits;

/* Extended DRAM Mode Register */
typedef struct{
__REG32 DLLDis            : 1;
__REG32 DS                : 1;
__REG32 Rtt0              : 1;
__REG32 AL                : 3;
__REG32 Rtt1              : 1;
__REG32 OCD               : 3;
__REG32 DQS               : 1;
__REG32 RDQS              : 1;
__REG32 Qoff              : 1;
__REG32                   :19;
} __ddr_dram_emr_bits;

/* DDR Controller Control (High) Register */
typedef struct{
__REG32                   : 3;
__REG32 MbusBurstChop     : 1;
__REG32                   : 3;
__REG32 D2PLat            : 1;
__REG32 P2DLat            : 1;
__REG32 AddHalfcc2DO      : 1;
__REG32 PupZeroSkewSel    : 1;
__REG32 WrMeshDelay       : 1;
__REG32                   :20;
} __ddr_cchr_bits;

/* DDR2 SDRAM Timing (Low) Register */
typedef struct{
__REG32                   : 4;
__REG32 tODT_ON_RD        : 4;
__REG32 tODT_OFF_RD       : 4;
__REG32 tODT_ON_CTL_RD    : 4;
__REG32 tODT_OFF_CTL_RD   : 4;
__REG32                   :12;
} __ddr_ddr2_tlr_bits;

/* SDRAM Operation Control Register */
typedef struct{
__REG32 CS                : 2;
__REG32                   :30;
} __ddr_sdram_ocr_bits;

/* SDRAM Interface Mbus Control0 (Low) Register */
typedef struct{
__REG32 Arb0              : 4;
__REG32 Arb1              : 4;
__REG32 Arb2              : 4;
__REG32 Arb3              : 4;
__REG32 Arb4              : 4;
__REG32 Arb5              : 4;
__REG32 Arb6              : 4;
__REG32 Arb7              : 4;
} __ddr_sdram_imc0lr_bits;

/* SDRAM Interface Mbus Control0 (High) Register */
typedef struct{
__REG32 Arb8              : 4;
__REG32 Arb9              : 4;
__REG32 Arb10             : 4;
__REG32 Arb11             : 4;
__REG32 Arb12             : 4;
__REG32 Arb13             : 4;
__REG32 Arb14             : 4;
__REG32 Arb15             : 4;
} __ddr_sdram_imc0hr_bits;

/* SDRAM Interface Mbus Timeout Register */
typedef struct{
__REG32 Timeout           : 8;
__REG32                   : 8;
__REG32 TimeoutDis        : 1;
__REG32 UnitPowerSave     : 1;
__REG32                   :14;
} __ddr_sdram_imtor_bits;

/* DDR2 SDRAM Timing (High) Register */
typedef struct{
__REG32 tODT_ON_WR        : 4;
__REG32 tODT_OFF_WR       : 4;
__REG32 tODT_ON_CTL_WR    : 4;
__REG32 tODT_OFF_CTL_WR   : 4;
__REG32                   :16;
} __ddr_ddr2_thr_bits;

/* SDRAM Initialization Control Register */
typedef struct{
__REG32 InitEn            : 1;
__REG32                   :31;
} __ddr_sdram_icr_bits;

/* SDRAM FTDLL Left Configuration Register */
/* SDRAM FTDLL Right Configuration Register */
/* SDRAM FTDLL Up Configuration Register */
typedef struct{
__REG32 OVERRIDE_PHD      : 1;
__REG32 FTDLL_INC_A       : 1;
__REG32 FTDLL_DEC_A       : 1;
__REG32 FTDLL_INC_B       : 1;
__REG32 FTDLL_DEC_B       : 1;
__REG32 FTDLL_INC_C       : 1;
__REG32 FTDLL_DEC_C       : 1;
__REG32                   : 4;
__REG32 FTDLL_OFFSET_CTL_A: 2;
__REG32 FTDLL_OFFSET_VAL_A: 5;
__REG32 FTDLL_OFFSET_CTL_B: 2;
__REG32 FTDLL_OFFSET_VAL_B: 5;
__REG32 FTDLL_OFFSET_CTL_C: 2;
__REG32 FTDLL_OFFSET_VAL_C: 5;
} __ddr_sdram_ftdll_lcr_bits;

/* Extended DRAM Mode 2 Register */
typedef struct{
__REG32 EMRS2             :15;
__REG32                   :17;
} __ddr_dram_em2r_bits;

/* Extended DRAM Mode 3 Register */
typedef struct{
__REG32 EMRS3             :15;
__REG32                   :17;
} __ddr_dram_em3r_bits;

/* SDRAM ODT Control (Low) Register */
typedef struct{
__REG32 ODT0Rd            : 4;
__REG32 ODT1Rd            : 4;
__REG32 ODT2Rd            : 4;
__REG32 ODT3Rd            : 4;
__REG32 ODT0Wr            : 4;
__REG32 ODT1Wr            : 4;
__REG32 ODT2Wr            : 4;
__REG32 ODT3Wr            : 4;
} __ddr_sdram_odt_clr_bits;

/* SDRAM ODT Control (High) Register */
typedef struct{
__REG32 ODT0En            : 2;
__REG32 ODT1En            : 2;
__REG32 ODT2En            : 2;
__REG32 ODT3En            : 2;
__REG32                   :24;
} __ddr_sdram_odt_chr_bits;

/* DDR Controller ODT Control Register */
typedef struct{
__REG32 ODTRd             : 4;
__REG32 ODTWr             : 4;
__REG32 ODTEn             : 2;
__REG32 DQ_ODTSel         : 2;
__REG32 STARTBURST_ODTSel : 2;
__REG32 STARTBURST_ODTEn  : 1;
__REG32 ODT_Unit          : 1;
__REG32 ODT_DrvN          : 5;
__REG32 ODT_DrvP          : 5;
__REG32                   : 6;
} __ddr_sdram_odt_cr_bits;

/* Read Buffer Select Register */
typedef struct{
__REG32 RdBuff0           : 1;
__REG32 RdBuff1           : 1;
__REG32 RdBuff2           : 1;
__REG32 RdBuff3           : 1;
__REG32 RdBuff4           : 1;
__REG32 RdBuff5           : 1;
__REG32 RdBuff6           : 1;
__REG32 RdBuff7           : 1;
__REG32 RdBuff8           : 1;
__REG32 RdBuff9           : 1;
__REG32 RdBuff10          : 1;
__REG32 RdBuff11          : 1;
__REG32 RdBuff12          : 1;
__REG32 RdBuff13          : 1;
__REG32 RdBuff14          : 1;
__REG32 RdBuff15          : 1;
__REG32                   :16;
} __ddr_rbsr_bits;

/* DDR SDRAM Address/Control Pads Calibration Register */
/* DDR SDRAM DQ Pads Calibration Register */
typedef struct{
__REG32 DrvN              : 6;
__REG32 DrvP              : 6;
__REG32                   : 4;
__REG32 TuneEn            : 1;
__REG32 LockN             : 6;
__REG32 LockP             : 6;
__REG32                   : 2;
__REG32 WrEn              : 1;
} __ddr_sdram_acpcr_bits;

/* SDRAM Interface Mbus Control1 (Low) Register */
typedef struct{
__REG32 Arb16             : 4;
__REG32 Arb17             : 4;
__REG32 Arb18             : 4;
__REG32 Arb19             : 4;
__REG32 Arb20             : 4;
__REG32 Arb21             : 4;
__REG32 Arb22             : 4;
__REG32 Arb23             : 4;
} __ddr_sdram_mc1lr_bits;

/* SDRAM Interface Mbus Control1 (Low) Register */
typedef struct{
__REG32 Arb24             : 4;
__REG32 Arb25             : 4;
__REG32 Arb26             : 4;
__REG32 Arb27             : 4;
__REG32 Arb28             : 4;
__REG32 Arb29             : 4;
__REG32 Arb30             : 4;
__REG32 Arb31             : 4;
} __ddr_sdram_mc1hr_bits;

/* FC FTDLL Configuration Register */
typedef struct{
__REG32 OVERRIDE_PHD      : 1;
__REG32 FTDLL_INC_A       : 1;
__REG32                   : 2;
__REG32 FTDLL_DEC_B       : 1;
__REG32                   : 6;
__REG32 FTDLL_OFFSET_CTL_A: 2;
__REG32 FTDLL_OFFSET_VAL_A: 5;
__REG32                   :14;
} __ddr_fc_ftdll_cr_bits;

/* Dual CPU Arbiter Register */
typedef struct{
__REG32 ArbMode           : 2;
__REG32                   :30;
} __ddr_dcpuar_bits;

/* SDRAM Received ECC Register */
typedef struct{
__REG32 ECCReg            : 8;
__REG32                   :24;
} __ddr_sdarm_reccr_bits;

/* SDRAM Calculated ECC Register */
typedef struct{
__REG32 ECCCalc           : 8;
__REG32                   :24;
} __ddr_sdarm_ceccr_bits;

/* SDRAM Error Address Register */
typedef struct{
__REG32 ErrType           : 1;
__REG32 CS                : 2;
__REG32 ECCAddr           :29;
} __ddr_sdarm_errar_bits;

/* SDRAM ECC Control Register */
typedef struct{
__REG32 ForceECC          : 8;
__REG32 ForceEn           : 1;
__REG32 PerrProp          : 1;
__REG32                   : 1;
__REG32 DPPar             : 1;
__REG32                   : 4;
__REG32 ThrEcc            : 8;
__REG32                   : 8;
} __ddr_sdarm_ecccr_bits;

/* DDR Controller Interrupt Cause Register */
typedef struct{
__REG32 SBit              : 1;
__REG32 DBit              : 1;
__REG32 DPErr             : 1;
__REG32                   :29;
} __ddr_cicr_bits;

/* DDR Controller Interrupt Mask Register */
typedef struct{
__REG32 SBit_Int_En       : 1;
__REG32 DBit_Int_En       : 1;
__REG32 DPErr_Int_En      : 1;
__REG32                   :29;
} __ddr_cimr_bits;

/* DEV_BOOTCSn Read Parameters Register */
typedef struct{
__REG32 TurnOff           : 6;
__REG32 Acc2First         : 6;
__REG32 RdSetup           : 5;
__REG32 Acc2Next          : 6;
__REG32 RdHold            : 5;
__REG32 BadrSkew          : 2;
__REG32 DevWidth          : 2;
} __dev_rdbootcsn_bits;

/* DEV_BOOTCSn Write Parameters Register */
typedef struct{
__REG32 ALE2Wr            : 6;
__REG32                   : 2;
__REG32 WrLow             : 6;
__REG32                   : 2;
__REG32 WrHigh            : 6;
__REG32                   :10;
} __dev_wrbootcsn_bits;

/* DEV_CSn[n] Read Parameters Register (n=0-3) */
typedef struct{
__REG32 TurnOff           : 6;
__REG32 Acc2First         : 6;
__REG32 RdSetup           : 5;
__REG32 Acc2Next          : 6;
__REG32 RdHold            : 5;
__REG32 BadrSkew          : 2;
__REG32 DevWidth          : 2;
} __dev_rdcsn_bits;

/* DEV_CSn[n] WriteParameters Register (n=0-3) */
typedef struct{
__REG32 ALE2Wr            : 6;
__REG32                   : 2;
__REG32 WrLow             : 6;
__REG32                   : 2;
__REG32 WrHigh            : 6;
__REG32                   :10;
} __dev_wrcsn_bits;

/* NAND Flash Control Register */
typedef struct{
__REG32 NFBoot            : 1;
__REG32 NFActCEnBoot      : 1;
__REG32 NF0               : 1;
__REG32 NFActCEn0         : 1;
__REG32 NF1               : 1;
__REG32 NFActCEn1         : 1;
__REG32 NF2               : 1;
__REG32 NFActCEn2         : 1;
__REG32 NFISD             : 1;
__REG32 NFOEnW            : 5;
__REG32 NFTr              : 5;
__REG32 NFOEnDel          : 1;
__REG32 NF3               : 1;
__REG32 NFActCEn3         : 1;
__REG32 NF_NumAddrPhase   : 2;
__REG32 NF_Boot_Type      : 1;
__REG32                   : 7;
} __dev_nandcr_bits;

/* Device Bus Interface Control Register */
typedef struct{
__REG32 Timeout           :16;
__REG32 OEWE_shared       : 1;
__REG32 ALE_Timing        : 2;
__REG32 ForceParityErr    : 1;
__REG32                   :12;
} __dev_ifcr_bits;

/* Device Bus Sync Control Register */
typedef struct{
__REG32 TclkDivideValue     : 4;
__REG32 BootCSReadyIgnore   : 1;
__REG32 BootCSReadyPolarity : 1;
__REG32 BootCSReadyDelay    : 3;
__REG32 CS0ReadyIgnore      : 1;
__REG32 CS0ReadyPolarity    : 1;
__REG32 CS0ReadyDelay       : 3;
__REG32 CS1ReadyIgnore      : 1;
__REG32 CS1ReadyPolarity    : 1;
__REG32 CS1ReadyDelay       : 3;
__REG32 CS2ReadyIgnore      : 1;
__REG32 CS2ReadyPolarity    : 1;
__REG32 CS2ReadyDelay       : 3;
__REG32 CS3ReadyIgnore      : 1;
__REG32 CS3ReadyPolarity    : 1;
__REG32 CS3ReadyDelay       : 3;
__REG32                     : 2;
__REG32 TclkDivLoadActive   : 1;
} __dev_scr_bits;

/* Device Bus Interrupt Cause Register */
typedef struct{
__REG32 RuDPErr             : 1;
__REG32 DRdyErr             : 1;
__REG32                     :30;
} __dev_icr_bits;

/* Device Bus Interrupt Mask Register */
typedef struct{
__REG32 MaskParityErr       : 1;
__REG32 MaskReadyErr        : 1;
__REG32                     :30;
} __dev_imcr_bits;

/* PCI Express Window X Control Register */
typedef struct{
__REG32 WinEn               : 1;
__REG32 BarMap              : 1;
__REG32 SlvWrSpltCnt        : 1;
__REG32                     : 1;
__REG32 Target              : 4;
__REG32 Attr                : 8;
__REG32 Size                :16;
} __pex_wcr_bits;

/* PCI Express Window X Remap Register */
typedef struct{
__REG32 RemapEn             : 1;
__REG32                     :15;
__REG32 Remap               :16;
} __pex_wrr_bits;

/* PCI Express Default Window Control Register */
typedef struct{
__REG32                     : 2;
__REG32 SlvWrSpltCnt        : 1;
__REG32                     : 1;
__REG32 Target              : 4;
__REG32 Attr                : 8;
__REG32 Size                :16;
} __pex_dwcr_bits;

/* PCI Express Expansion ROM Window Control Register */
typedef struct{
__REG32                     : 4;
__REG32 Target              : 4;
__REG32 Attr                : 8;
__REG32 Size                :16;
} __pex_eromwcr_bits;

/* PCI Express Expansion ROM Window Remap Register */
typedef struct{
__REG32 RemapEn             : 1;
__REG32                     :15;
__REG32 Attr                :16;
} __pex_eromwrr_bits;

/* PCI Express BAR x Control Register */
typedef struct{
__REG32 BarEn               : 1;
__REG32                     :15;
__REG32 BarSize             :16;
} __pex_barcr_bits;

/* PCI Express Expansion ROM BAR Control Register */
typedef struct{
__REG32 ExpROMEn            : 1;
__REG32                     :18;
__REG32 ExpROMSz            : 3;
__REG32                     :10;
} __pex_erombarcr_bits;

/* PCI Express Configuration Address Register */
typedef struct{
__REG32                     : 2;
__REG32 RegNum              : 6;
__REG32 FunctNum            : 3;
__REG32 DevNum              : 5;
__REG32 BusNum              : 8;
__REG32 ExtRegNum           : 4;
__REG32                     : 3;
__REG32 ConfigEn            : 1;
} __pex_car_bits;

/* PCI Express Device and Vendor ID Register */
typedef struct{
__REG32 VenID               :16;
__REG32 DevID               :16;
} __pex_dvidr_bits;

/* PCI Express Command and Status Register */
typedef struct{
__REG32 IOEn                : 1;
__REG32 MemEn               : 1;
__REG32 MasEn               : 1;
__REG32                     : 3;
__REG32 PErrEn              : 1;
__REG32                     : 1;
__REG32 SErrEn              : 1;
__REG32                     : 1;
__REG32 IntDis              : 1;
__REG32                     : 8;
__REG32 IntStat             : 1;
__REG32 CapList             : 1;
__REG32                     : 3;
__REG32 MasDataPerr         : 1;
__REG32                     : 2;
__REG32 STarAbort           : 1;
__REG32 RTAbort             : 1;
__REG32 RMAbort             : 1;
__REG32 SSysErr             : 1;
__REG32 DetParErr           : 1;
} __pex_csr_bits;

/* PCI Express Class Code and Revision ID Register */
typedef struct{
__REG32 RevID               : 8;
__REG32 ProgIF              : 8;
__REG32 SubClass            : 8;
__REG32 BaseClass           : 8;
} __pex_ccridr_bits;

/* PCI Express BIST Header Type and Cache Line Size Register */
typedef struct{
__REG32 CacheLine           : 8;
__REG32                     : 8;
__REG32 HeadType            : 8;
__REG32 BISTComp            : 4;
__REG32                     : 2;
__REG32 BISTAct             : 1;
__REG32 BISTCap             : 1;
} __pex_bisthtclsr_bits;

/* PCI Express BAR0 Internal Register */
typedef struct{
__REG32 Space               : 1;
__REG32 Type                : 2;
__REG32 Prefetch            : 1;
__REG32                     :16;
__REG32 Base                :12;
} __pex_barir_bits;

/* PCI Express Subsystem Device and Vendor ID Register */
typedef struct{
__REG32 SSVenID             :16;
__REG32 SSDevID             :16;
} __pex_ssdvidr_bits;

/* PCI Express Expansion ROM BAR Register */
typedef struct{
__REG32 RomEn               : 1;
__REG32                     :18;
__REG32 RomBase             :13;
} __pex_erombarr_bits;

/* PCI Express Capability List Pointer Register */
typedef struct{
__REG32 CapPtr              : 8;
__REG32                     :24;
} __pex_clpr_bits;

/* PCI Express Interrupt Pin and Line Register */
typedef struct{
__REG32 IntLine             : 8;
__REG32 IntPin              : 8;
__REG32                     :16;
} __pex_iplr_bits;

/* PCI Express Interrupt Pin and Line Register */
typedef struct{
__REG32 CapID               : 8;
__REG32 NextPtr             : 8;
__REG32 PMCVer              : 3;
__REG32                     : 2;
__REG32 DSI                 : 1;
__REG32 AuxCur              : 3;
__REG32 D1Sup               : 1;
__REG32 D2Sup               : 1;
__REG32 PMESup              : 5;
} __pex_pmchr_bits;

/* PCI Express Power Management Control and Status Register */
typedef struct{
__REG32 PMState             : 2;
__REG32                     : 1;
__REG32 No_Soft_Reset       : 1;
__REG32                     : 4;
__REG32 PME_en              : 1;
__REG32 PMDataSel           : 4;
__REG32 PMDataScale         : 2;
__REG32 PMEStat             : 1;
__REG32                     : 8;
__REG32 PMData              : 8;
} __pex_pmcsr_bits;

/* PCI Express MSI Message Control Register */
typedef struct{
__REG32 CapID               : 8;
__REG32 NextPtr             : 8;
__REG32 MSIEn               : 1;
__REG32 MultiCap            : 3;
__REG32 MultiEn             : 3;
__REG32 Addr64              : 1;
__REG32                     : 8;
} __pex_msimcr_bits;

/* PCI Express MSI Message Data Register */
typedef struct{
__REG32 MSIData             :16;
__REG32                     :16;
} __pex_msimdr_bits;

/* PCI Express Capability Register */
typedef struct{
__REG32 CapID               : 8;
__REG32 NextPtr             : 8;
__REG32 CapVer              : 4;
__REG32 DevType             : 4;
__REG32 SlotImp             : 1;
__REG32 IntMsgNum           : 5;
__REG32                     : 2;
} __pex_cr_bits;

/* PCI Express Device Capabilities Register */
typedef struct{
__REG32 MaxPldSizeSup       : 3;
__REG32                     : 3;
__REG32 EPL0sAccLat         : 3;
__REG32 EPL1AccLat          : 3;
__REG32                     : 3;
__REG32 RuleBaseErrorRep    : 1;
__REG32                     : 2;
__REG32 CapSPLVal           : 8;
__REG32 CapSPLScl           : 2;
__REG32                     : 4;
} __pex_dcr_bits;

/* PCI Express Device Control Status Register */
typedef struct{
__REG32 CorErrRepEn         : 1;
__REG32 NFErrRepEn          : 1;
__REG32 FErrRepEn           : 1;
__REG32 URRepEn             : 1;
__REG32 EnRO                : 1;
__REG32 MaxPldSz            : 3;
__REG32                     : 3;
__REG32 EnNS                : 1;
__REG32 MaxRdRqSz           : 3;
__REG32                     : 1;
__REG32 CorErrDet           : 1;
__REG32 NFErrDet            : 1;
__REG32 FErrDet             : 1;
__REG32 URDet               : 1;
__REG32                     : 1;
__REG32 TransPend           : 1;
__REG32                     :10;
} __pex_dcsr_bits;

/* PCI Express Link Capabilities Register */
typedef struct{
__REG32 MaxLinkSpd          : 4;
__REG32 MaxLnkWdth          : 6;
__REG32 AspmSup             : 2;
__REG32 L0sExtLat           : 3;
__REG32 L1ExtLat            : 3;
__REG32 ClockPowerMng       : 1;
__REG32                     : 5;
__REG32 PortNum             : 8;
} __pex_lcr_bits;

/* PCI Express Link Control Status Register */
typedef struct{
__REG32 AspmCnt             : 2;
__REG32                     : 1;
__REG32 RCB                 : 1;
__REG32 LnkDis              : 1;
__REG32 RetrnLnk            : 1;
__REG32 CmnClkCfg           : 1;
__REG32 ExtdSnc             : 1;
__REG32 EnClkPwrMng         : 1;
__REG32                     : 7;
__REG32 LnkSpd              : 4;
__REG32 NegLnkWdth          : 6;
__REG32                     : 1;
__REG32 LnkTrn              : 1;
__REG32 SltClkCfg           : 1;
__REG32                     : 3;
} __pex_lcsr_bits;

/* PCI Express Advanced Error Report Header Register */
typedef struct{
__REG32                     : 4;
__REG32 DLPrtErr            : 1;
__REG32                     : 7;
__REG32 RPsnTlpErr          : 1;
__REG32                     : 1;
__REG32 CmpTOErr            : 1;
__REG32 CAErr               : 1;
__REG32 UnexpCmpErr         : 1;
__REG32                     : 1;
__REG32 MalfTlpErr          : 1;
__REG32                     : 1;
__REG32 URErr               : 1;
__REG32                     :11;
} __pex_aerhr_bits;

/* PCI Express Uncorrectable Error Mask Register */
typedef struct{
__REG32                     : 4;
__REG32 DLPrtErrMsk         : 1;
__REG32                     : 7;
__REG32 RPsnTlpErrMsk       : 1;
__REG32                     : 1;
__REG32 CmpTOErrMsk         : 1;
__REG32 CAErrMsk            : 1;
__REG32 UnexpCmpErrMsk      : 1;
__REG32                     : 1;
__REG32 MalfTlpErrMsk       : 1;
__REG32                     : 1;
__REG32 URErrMsk            : 1;
__REG32                     :11;
} __pex_uestr_bits;

/* PCI Express Uncorrectable Error Severity Register */
typedef struct{
__REG32                     : 4;
__REG32 DLPrtErrSev         : 1;
__REG32                     : 7;
__REG32 RPsnTlpErrSev       : 1;
__REG32                     : 1;
__REG32 CmpTOErrSev         : 1;
__REG32 CAErrSev            : 1;
__REG32 UnexpCmpErrSev      : 1;
__REG32                     : 1;
__REG32 MalfTlpErrSev       : 1;
__REG32                     : 1;
__REG32 URErrSev            : 1;
__REG32                     :11;
} __pex_uemr_bits;

/* PCI Express Uncorrectable Error Severity Register */
typedef struct{
__REG32                     : 4;
__REG32 DLPrtErrSev         : 1;
__REG32                     : 7;
__REG32 RPsnTlpErrSev       : 1;
__REG32                     : 1;
__REG32 CmpTOErrSev         : 1;
__REG32 CAErrSev            : 1;
__REG32 UnexpCmpErrSev      : 1;
__REG32                     : 1;
__REG32 MalfTlpErrSev       : 1;
__REG32                     : 1;
__REG32 URErrSev            : 1;
__REG32                     :11;
} __pex_uesr_bits;

/* PCI Express Correctable Error Status Register */
typedef struct{
__REG32 RcvErr              : 1;
__REG32                     : 5;
__REG32 BadTlpErr           : 1;
__REG32 BadDllpErr          : 1;
__REG32 RplyRllovrErr       : 1;
__REG32                     : 3;
__REG32 RplyTOErr           : 1;
__REG32 AdvNonFatalErr      : 1;
__REG32                     :18;
} __pex_cestr_bits;

/* PCI Express Correctable Error Mask Register */
typedef struct{
__REG32 RcvMsk              : 1;
__REG32                     : 5;
__REG32 BadTlpMsk           : 1;
__REG32 BadDllpErrMsk       : 1;
__REG32 RplyRllovrMsk       : 1;
__REG32                     : 3;
__REG32 RplyTOMsk           : 1;
__REG32 AdvNonFatalMsk      : 1;
__REG32                     :18;
} __pex_cemr_bits;

/* PCI Express Advanced Error Capability and Control Register */
typedef struct{
__REG32 FrstErrPtr          : 5;
__REG32                     :27;
} __pex_aeccr_bits;

/* PCI Express Control Register */
typedef struct{
__REG32 ConfLinkX1            : 1;
__REG32 ConfRootComplex       : 1;
__REG32 CfgMapToMemEn         : 1;
__REG32                       :21;
__REG32 ConfMstrHotReset      : 1;
__REG32                       : 1;
__REG32 ConfMstrLb            : 1;
__REG32 ConfMstrDisScrmb      : 1;
__REG32                       : 2;
__REG32 Conf_Training_Disable : 1;
__REG32 Crs_Enable            : 1;
} __pex_ctrlr_bits;

/* PCI Express Status Register */
typedef struct{
__REG32 DLDown                : 1;
__REG32                       : 7;
__REG32 PexBusNum             : 8;
__REG32 PexDevNum             : 5;
__REG32                       : 3;
__REG32 PexSlvHotReset        : 1;
__REG32 PexSlvDisLink         : 1;
__REG32 PexSlvLb              : 1;
__REG32 PexSlvDisScrmb        : 1;
__REG32                       : 4;
} __pex_str_bits;

/* PCI Express RC SSPL Register */
typedef struct{
__REG32 SlotPowerLimitValue   : 8;
__REG32 SlotPowerLimitScale   : 2;
__REG32                       : 6;
__REG32 SsplMsgEnable         : 1;
__REG32                       :15;
} __pex_rcssplr_bits;

/* PCI Express Completion Timeout Register */
typedef struct{
__REG32 ConfCmpToThrshld      :16;
__REG32                       :16;
} __pex_ctr_bits;

/* PCI Express RC PME Register */
typedef struct{
__REG32 PMERequesterID        :16;
__REG32 PMEStatus             : 1;
__REG32 PMEPending            : 1;
__REG32                       :14;
} __pex_rcpmer_bits;

/* PCI Express PM Register */
typedef struct{
__REG32 L1_aspm_en            : 1;
__REG32 l1_aspm_ack           : 1;
__REG32                       : 2;
__REG32 SendTurnOffAckMsg     : 1;
__REG32 SendTurnOffMsg        : 1;
__REG32                       :10;
__REG32 Ref_clk_off_en        : 1;
__REG32                       :15;
} __pex_pmr_bits;

/* PCI Express Flow Control Register */
typedef struct{
__REG32 ConfPhInitFc          : 8;
__REG32 ConfNphInitFc         : 8;
__REG32 ConfChInitFc          : 8;
__REG32 ConfFcUpdateTo        : 8;
} __pex_fcr_bits;

/* PCI Express Acknowledge Timers (4X) Register */
typedef struct{
__REG32 AckLatTOX4            :16;
__REG32 AckRplyTOX4           :16;
} __pex_at4r_bits;

/* PCI Express Acknowledge Timers (1X) Register */
typedef struct{
__REG32 AckLatTOX1            :16;
__REG32 AckRplyTOX1           :16;
} __pex_at1r_bits;

/* PCI Express RAM Parity Protection Control Register */
typedef struct{
__REG32 RxPrtyGen             : 1;
__REG32 RxPrtyChkEn           : 1;
__REG32 TxPrtyGen             : 1;
__REG32 TxPrtyChkEn           : 1;
__REG32                       :28;
} __pex_ramppcr_bits;

/* PCI Express Debug Control Register */
typedef struct{
__REG32 DirectScrmbleDis      : 1;
__REG32 DisComliance          : 1;
__REG32                       : 3;
__REG32 dbg_power_up          : 1;
__REG32 dis_pclock_cg         : 1;
__REG32 dis_core_clk_cg       : 1;
__REG32 dis_ram_pd            : 1;
__REG32                       : 6;
__REG32 MaskLinkDis           : 1;
__REG32 ConfMskLinkFail       : 1;
__REG32 ConfMskHotReset       : 1;
__REG32                       : 2;
__REG32 SoftReset             : 1;
__REG32                       :11;
} __pex_dbgcr_bits;

/* PCI Express TL Control Register */
typedef struct{
__REG32 RxCmplPushDis         : 1;
__REG32 RxNpPushDis           : 1;
__REG32 TxCmplPushDis         : 1;
__REG32 TxNpPushDis           : 1;
__REG32                       :28;
} __pex_tlcr_bits;

/* PCI Express PHY Indirect Access Register */
typedef struct{
__REG32 PhyData               :16;
__REG32 PhyAddr               :14;
__REG32                       : 1;
__REG32 PhyAccssMd            : 1;
} __pex_phyiar_bits;

/* PCI Express Interrupt Cause Register */
typedef struct{
__REG32 TxReqInDldownErr      : 1;
__REG32 MDis                  : 1;
__REG32                       : 1;
__REG32 ErrWrToReg            : 1;
__REG32 HitDfltWinErr         : 1;
__REG32                       : 1;
__REG32 RxRamParErr           : 1;
__REG32 TxRamParErr           : 1;
__REG32 CorErrDet             : 1;
__REG32 NFErrDet              : 1;
__REG32 FErrDet               : 1;
__REG32 DstateChange          : 1;
__REG32 BIST                  : 1;
__REG32                       : 1;
__REG32 FlowCtrlProtocol      : 1;
__REG32 RcvUrCaErr            : 1;
__REG32 RcvErrFatal           : 1;
__REG32 RcvErrNonFatal        : 1;
__REG32 RcvErrCor             : 1;
__REG32 RcvCRS                : 1;
__REG32 PexSlvHotReset        : 1;
__REG32 PexSlvDisLink         : 1;
__REG32 PexSlvLb              : 1;
__REG32 PexLinkFail           : 1;
__REG32 RcvIntA               : 1;
__REG32 RcvIntB               : 1;
__REG32 RcvIntC               : 1;
__REG32 RcvIntD               : 1;
__REG32 RcvPmPme              : 1;
__REG32 RcvTurnOff            : 1;
__REG32                       : 1;
__REG32 RcvMsi                : 1;
} __pex_icr_bits;

/* PCI Express Mbus Adapter Control Register */
typedef struct{
__REG32                       :20;
__REG32 RxDPPropEn            : 1;
__REG32 TxDPPropEn            : 1;
__REG32                       :10;
} __pex_macr_bits;

/* PCI Express Mbus Arbiter Control Register (Low) */
typedef struct{
__REG32 Arb0                  : 4;
__REG32 Arb1                  : 4;
__REG32 Arb2                  : 4;
__REG32 Arb3                  : 4;
__REG32 Arb4                  : 4;
__REG32 Arb5                  : 4;
__REG32 Arb6                  : 4;
__REG32 Arb7                  : 4;
} __pex_maclr_bits;

/* PCI Express Mbus Arbiter Control Register (High) */
typedef struct{
__REG32 Arb8                  : 4;
__REG32 Arb9                  : 4;
__REG32 Arb10                 : 4;
__REG32 Arb11                 : 4;
__REG32 Arb12                 : 4;
__REG32 Arb13                 : 4;
__REG32 Arb14                 : 4;
__REG32 Arb15                 : 4;
} __pex_machr_bits;

/* PCI Express Mbus Arbiter Control Register (High) */
typedef struct{
__REG32 Timeout               : 8;
__REG32                       : 8;
__REG32 TimeoutEn             : 1;
__REG32                       :15;
} __pex_matr_bits;

/* PHY Address Register */
typedef struct{
__REG32 PhyAd                 : 5;
__REG32                       :27;
} __gbe_phyar_bits;

/* SMI Register */
typedef struct{
__REG32 Data                  :16;
__REG32 PhyAd                 : 5;
__REG32 RegAd                 : 5;
__REG32 Opcode                : 1;
__REG32 ReadValid             : 1;
__REG32 Busy                  : 1;
__REG32                       : 3;
} __gbe_simr_bits;

/* Ethernet Unit Default ID (EUDID) Register */
typedef struct{
__REG32 DIDR                  : 4;
__REG32 DATTR                 : 8;
__REG32                       :20;
} __gbe_eudid_bits;

/* Ethernet Unit Interrupt Cause (EUIC) Register */
typedef struct{
__REG32 EtherIntSum           : 1;
__REG32 Parity                : 1;
__REG32 AddressViolation      : 1;
__REG32 AddressNoMatch        : 1;
__REG32 SMIdone               : 1;
__REG32 Count_wa              : 1;
__REG32                       : 1;
__REG32 InternalAddrError     : 1;
__REG32                       :24;
} __gbe_euic_bits;

/* Ethernet Unit Interrupt Mask (EUIM) Register */
typedef struct{
__REG32 Various               :14;
__REG32                       :18;
} __gbe_euim_bits;

/* Ethernet Unit Internal Address Error (EUIAE) Register */
typedef struct{
__REG32 InternalAddress       :14;
__REG32                       :18;
} __gbe_euiae_bits;

/* Ethernet Unit Control (EUC) Register */
typedef struct{
__REG32 Port_DPPar            : 1;
__REG32 Polling               : 1;
__REG32                       :19;
__REG32 P2P_Loopback          : 1;
__REG32                       :10;
} __gbe_euc_bits;

/* Base Address Register (n=0 5) */
typedef struct{
__REG32 Target                : 4;
__REG32                       : 4;
__REG32 Attr                  : 8;
__REG32 Base                  :16;
} __gbe_ba_bits;

/* Size (S) Register (n=0 5) */
typedef struct{
__REG32                       :16;
__REG32 Size                  :16;
} __gbe_sr_bits;

/* Base Address Enable (BARE) Register */
typedef struct{
__REG32 En                    : 6;
__REG32                       :26;
} __gbe_bare_bits;

/* Ethernet Port Access Protect (EPAP) Register */
typedef struct{
__REG32 Win0                  : 2;
__REG32 Win1                  : 2;
__REG32 Win2                  : 2;
__REG32 Win3                  : 2;
__REG32 Win4                  : 2;
__REG32 Win5                  : 2;
__REG32                       :20;
} __gbe_epap_bits;

/* Port Configuration (PxC) Register */
typedef struct{
__REG32 UPM                   : 1;
__REG32 RXQ                   : 3;
__REG32 RXQArp                : 3;
__REG32 RB                    : 1;
__REG32 RBIP                  : 1;
__REG32 RBArp                 : 1;
__REG32                       : 2;
__REG32 AMNoTxES              : 1;
__REG32                       : 1;
__REG32 TCP_CapEn             : 1;
__REG32 UDP_CapEn             : 1;
__REG32 TCPQ                  : 3;
__REG32 UDPQ                  : 3;
__REG32 BPDUQ                 : 3;
__REG32 RxCS                  : 1;
__REG32                       : 6;
} __gbe_pxc_bits;

/* Port Configuration Extend (PxCX) Register */
typedef struct{
__REG32                       : 1;
__REG32 Span                  : 1;
__REG32                       : 1;
__REG32 TxCRCDis              : 1;
__REG32                       :28;
} __gbe_pxcx_bits;

/* MII Serial Parameters Register */
typedef struct{
__REG32                       :13;
__REG32 IPG_DATA              : 4;
__REG32                       :15;
} __gbe_miispr_bits;

/* VLAN EtherType (EVLANE) Register */
typedef struct{
__REG32 VL_EtherType          :16;
__REG32                       :16;
} __gbe_evlane_bits;

/* MAC Address Low (MACAL) Register */
typedef struct{
__REG32 MAC                   :16;
__REG32                       :16;
} __gbe_macal_bits;

/* SDMA Configuration (SDC) Register */
typedef struct{
__REG32 RIFB                  : 1;
__REG32 RxBSZ                 : 3;
__REG32 BLMR                  : 1;
__REG32 BLMT                  : 1;
__REG32 SwapMode              : 1;
__REG32 IPGIntRx              :15;
__REG32 TxBSZ                 : 3;
__REG32 IPGIntRx15            : 1;
__REG32                       : 6;
} __gbe_sdc_bits;

/* IP Differentiated Services CodePoint x to Priority (DSCP0-5) Register */
typedef struct{
__REG32 TOS_Q                 :30;
__REG32                       : 2;
} __gbe_dscp_bits;

/* IP Differentiated Services CodePoint 6 to Priority (DSCP6) Register */
typedef struct{
__REG32 TOS_Q                 :12;
__REG32                       :20;
} __gbe_dscp6_bits;

/* Port Serial Control0 (PSC0) Register */
typedef struct{
__REG32 PortEn                : 1;
__REG32 ForceLinkPass         : 1;
__REG32 AN_Duplex             : 1;
__REG32 AN_FC                 : 1;
__REG32 Pause_Adv             : 1;
__REG32 ForceFCMode           : 2;
__REG32 ForceBPMode           : 2;
__REG32                       : 1;
__REG32 ForceLinkFail         : 1;
__REG32                       : 2;
__REG32 ANSpeed               : 1;
__REG32 DTEAdvert             : 1;
__REG32 MiiPhy                : 1;
__REG32 MiiSS                 : 1;
__REG32 MRU                   : 3;
__REG32                       : 1;
__REG32 Set_FullDx            : 1;
__REG32 SetFCEn               : 1;
__REG32 SetGMIISpeed          : 1;
__REG32 SetMIISpeed           : 1;
__REG32                       : 7;
} __gbe_psc0_bits;

/* Port Serial Control0 (PSC0) Register */
typedef struct{
__REG32 Priority              :24;
__REG32                       : 8;
} __gbe_vpt2p_bits;

/* Ethernet Port Status 0 (PS0) Register */
typedef struct{
__REG32                       : 1;
__REG32 LinkUp                : 1;
__REG32 FullDx                : 1;
__REG32 EnFC                  : 1;
__REG32 GMIISpeed             : 1;
__REG32 MIISpeed              : 1;
__REG32                       : 1;
__REG32 TxInProg              : 1;
__REG32                       : 2;
__REG32 TxFIFOEmp             : 1;
__REG32 RxFIFO1Emp            : 1;
__REG32 RxFIFO2Emp            : 1;
__REG32                       :19;
} __gbe_ps0_bits;

/* Transmit Queue Command (TQC) Register */
typedef struct{
__REG32 ENQ                   : 8;
__REG32 DISQ                  : 8;
__REG32                       :16;
} __gbe_tqc_bits;

/* Port Serial Control1 (PSC1) Register */
typedef struct{
__REG32                       : 3;
__REG32 RGMIIEn               : 1;
__REG32 port_reset            : 1;
__REG32                       : 6;
__REG32 PortType              : 1;
__REG32                       : 3;
__REG32 en_col_on_bp          : 1;
__REG32 col_domain_limit      : 6;
__REG32 en_mii_odd_pre        : 1;
__REG32                       : 9;
} __gbe_psc1_bits;

/* Ethernet Port Status1 (PS1) Register */
typedef struct{
__REG32 PortRxPause           : 1;
__REG32 PortTxPause           : 1;
__REG32 PortDoingPressure     : 1;
__REG32 SyncFail10ms          : 1;
__REG32 AnDone                : 1;
__REG32                       :27;
} __gbe_ps1_bits;

/* Marvell Header Register */
typedef struct{
__REG32 MHEn                  : 1;
__REG32 DAPrefix              : 2;
__REG32                       : 1;
__REG32 SPID                  : 4;
__REG32 MHMask                : 2;
__REG32 DSAEn                 : 2;
__REG32 SPID45                : 2;
__REG32 SDIDEn                : 1;
__REG32                       : 1;
__REG32 SDevID                : 5;
__REG32                       :11;
} __gbe_mhr_bits;

/* Port Interrupt Cause (IC) Register */
/* Port Interrupt Mask (PIM) Register */
typedef struct{
__REG32 RxBuffer              : 1;
__REG32 Extend                : 1;
__REG32 RxBufferQueue0        : 1;
__REG32 RxBufferQueue1        : 1;
__REG32 RxBufferQueue2        : 1;
__REG32 RxBufferQueue3        : 1;
__REG32 RxBufferQueue4        : 1;
__REG32 RxBufferQueue5        : 1;
__REG32 RxBufferQueue6        : 1;
__REG32 RxBufferQueue7        : 1;
__REG32 RxError               : 1;
__REG32 RxErrorQueue0         : 1;
__REG32 RxErrorQueue1         : 1;
__REG32 RxErrorQueue2         : 1;
__REG32 RxErrorQueue3         : 1;
__REG32 RxErrorQueue4         : 1;
__REG32 RxErrorQueue5         : 1;
__REG32 RxErrorQueue6         : 1;
__REG32 RxErrorQueue7         : 1;
__REG32 TxEnd0                : 1;
__REG32 TxEnd1                : 1;
__REG32 TxEnd2                : 1;
__REG32 TxEnd3                : 1;
__REG32 TxEnd4                : 1;
__REG32 TxEnd5                : 1;
__REG32 TxEnd6                : 1;
__REG32 TxEnd7                : 1;
__REG32                       : 4;
__REG32 EtherIntSum           : 1;
} __gbe_ic_bits;

/* Port Interrupt Cause Extend (ICE) Register */
/* Port Extend Interrupt Mask (PEIM) Register */
typedef struct{
__REG32 TxBuffer0             : 1;
__REG32 TxBuffer1             : 1;
__REG32 TxBuffer2             : 1;
__REG32 TxBuffer3             : 1;
__REG32 TxBuffer4             : 1;
__REG32 TxBuffer5             : 1;
__REG32 TxBuffer6             : 1;
__REG32 TxBuffer7             : 1;
__REG32 TxError0              : 1;
__REG32 TxError1              : 1;
__REG32 TxError2              : 1;
__REG32 TxError3              : 1;
__REG32 TxError4              : 1;
__REG32 TxError5              : 1;
__REG32 TxError6              : 1;
__REG32 TxError7              : 1;
__REG32 PhySTC                : 1;
__REG32                       : 1;
__REG32 RxOVR                 : 1;
__REG32 TxUdr                 : 1;
__REG32 LinkChange            : 1;
__REG32                       : 2;
__REG32 InternalAddrError     : 1;
__REG32                       : 7;
__REG32 EtherIntSum           : 1;
} __gbe_ice_bits;

/* Port Tx FIFO Urgent Threshold (PxTFUT) Register */
typedef struct{
__REG32                       : 4;
__REG32 IPGIntTx              :16;
__REG32                       :12;
} __gbe_pxtfut_bits;

/* Port Rx Minimal Frame Size (PxMFS) Register */
typedef struct{
__REG32 RxMFS                 : 7;
__REG32                       :25;
} __gbe_pxmfs_bits;

/* Port Internal Address Error (PIAE) Register */
typedef struct{
__REG32 InternalAddress       : 9;
__REG32                       :23;
} __gbe_piae_bits;

/* Ethernet Type Priority Register */
typedef struct{
__REG32 EtherTypePriEn        : 1;
__REG32 EtherTypePriFrstEn    : 1;
__REG32 EtherTypePriQ         : 3;
__REG32 EtherTyprPriVal       :16;
__REG32 ForceUnicstHit        : 1;
__REG32                       :10;
} __gbe_etpr_bits;

/* Transmit Queue Fixed Priority Configuration (TQFPC) Register */
typedef struct{
__REG32 FIXPR                 : 8;
__REG32                       :24;
} __gbe_tqfpc_bits;

/* Port Transmit Token-Bucket Rate Configuration (PTTBRC) Register */
typedef struct{
__REG32 PTKNRT                :10;
__REG32                       :22;
} __gbe_pttbrc_bits;

/* Maximum Transmit Unit (MTU) Register */
typedef struct{
__REG32 MTU                   : 6;
__REG32                       :26;
} __gbe_mtu_bits;

/* Port Maximum Token Bucket Size (PMTBS) Register */
typedef struct{
__REG32 PMTBS                 :16;
__REG32                       :16;
} __gbe_pmtbs_bits;

/* Receive Queue Command (RQC) Register */
typedef struct{
__REG32 ENQ                   : 8;
__REG32 DISQ                  : 8;
__REG32                       :16;
} __gbe_rqc_bits;

/* Transmit Queue Token-Bucket Counter (TQxTBCNT) Register (n=0 7) */
typedef struct{
__REG32 TKNBKT                :30;
__REG32                       : 2;
} __gbe_tqxtbcnt_bits;

/* Transmit Queue Token Bucket Configuration (TQxTBC) Register (n=0 7) */
typedef struct{
__REG32 TKNRT                 :10;
__REG32 MTBS                  :16;
__REG32                       : 6;
} __gbe_tqxtbc_bits;

/* Transmit Queue Arbiter Configuration (TQxAC) Register (n=0 7) */
typedef struct{
__REG32 WRRWGT                : 8;
__REG32 WRR_BC                :16;
__REG32                       : 8;
} __gbe_tqxac_bits;

/* Port Transmit Token-Bucket Counter (PTTBC) Register */
typedef struct{
__REG32 PTKNBKT               :30;
__REG32                       : 2;
} __gbe_pttbc_bits;

/* Destination Address Filter Special Multicast Table (DFSMT) Register (n=0 63) */
/* Destination Address Filter Other Multicast Table (DFOMT) Register (n=0 63) */
/* Address Filter Unicast Table (DFUT) Register (n=0 3) */
typedef struct{
__REG32 Pass0                 : 1;
__REG32 Queue0                : 3;
__REG32                       : 4;
__REG32 Pass1                 : 1;
__REG32 Queue1                : 3;
__REG32                       : 4;
__REG32 Pass2                 : 1;
__REG32 Queue2                : 3;
__REG32                       : 4;
__REG32 Pass3                 : 1;
__REG32 Queue3                : 3;
__REG32                       : 4;
} __gbe_dfsmt_bits;

/* USB 2.0 WindowX Control Register */
typedef struct{
__REG32 win_en                : 1;
__REG32 WrBL                  : 1;
__REG32                       : 2;
__REG32 Target                : 4;
__REG32 Attr                  : 8;
__REG32 Size                  :16;
} __usb2_wcr_bits;

/* USB 2.0 WindowX Base Register */
typedef struct{
__REG32                       :16;
__REG32 Base                  :16;
} __usb2_wbr_bits;

/* USB 2.0 Bridge Control Register */
typedef struct{
__REG32                       : 4;
__REG32 BS                    : 1;
__REG32                       :27;
} __usb2_bcr_bits;

/* USB 2.0 Bridge Interrupt Cause Register */
typedef struct{
__REG32 AddrDecErr            : 1;
__REG32 HOF                   : 1;
__REG32 DOF                   : 1;
__REG32 DUF                   : 1;
__REG32                       :28;
} __usb2_bicr_bits;

/* USB 2.0 PHY Configuration0 Register */
typedef struct{
__REG32 StartIPG              : 7;
__REG32                       : 1;
__REG32 NonStartIPG           : 7;
__REG32                       :17;
} __usb2_phyc0r_bits;

/* USB 2.0 Power Control Register */
typedef struct{
__REG32 Pu                    : 1;
__REG32 PuPll                 : 1;
__REG32 SUSPENDM              : 1;
__REG32 VBUS_PWR_FAULT        : 1;
__REG32 PWRCTL_WAKEUP         : 1;
__REG32                       : 3;
__REG32 REG_ARC_DPDM_MODE     : 1;
__REG32 REG_DP_PULLDOWN       : 1;
__REG32 REG_DM_PULLDOWN       : 1;
__REG32                       :12;
__REG32 utmi_sessend          : 1;
__REG32 utmi_vbus_valid       : 1;
__REG32 utmi_avalid           : 1;
__REG32 utmi_bvalid           : 1;
__REG32 TX_BIT_STUFF          : 1;
__REG32                       : 4;
} __usb2_pcr_bits;

/* USB 2.0 PHY PLL Control Register */
typedef struct{
__REG32                       :21;
__REG32 VCOCAL_START          : 1;
__REG32                       :10;
} __usb2_phypllcr_bits;

/* USB 2.0 PHY Tx Control Register */
typedef struct{
__REG32                       :12;
__REG32 REG_RCAL_START        : 1;
__REG32                       :19;
} __usb2_phytxcr_bits;

/* USB 2.0 PHY Rx Control Register */
typedef struct{
__REG32                       : 4;
__REG32 SQ_THRESH             : 4;
__REG32                       :24;
} __usb2_phyrxcr_bits;

/* USB 2.0 PHY IVREF Control Register */
typedef struct{
__REG32                       : 8;
__REG32 TXVDD12               : 2;
__REG32                       :22;
} __usb2_phyivrefcr_bits;

/* AES Decryption Command Register */
typedef struct{
__REG32 AesDecKeyMode         : 2;
__REG32 AesDecMakeKey         : 1;
__REG32                       : 1;
__REG32 DataByteSwap          : 1;
__REG32                       : 3;
__REG32 OutByteSwap           : 1;
__REG32                       :21;
__REG32 AesDecKeyReady        : 1;
__REG32 Termination           : 1;
} __aesdcr_bits;

/* AES Encryption Command Register */
typedef struct{
__REG32 AesEncKeyMode         : 2;
__REG32                       : 2;
__REG32 DataByteSwap          : 1;
__REG32                       : 3;
__REG32 OutByteSwap           : 1;
__REG32                       :22;
__REG32 Termination           : 1;
} __aesecr_bits;

/* DES Command Register */
typedef struct{
__REG32 Direction             : 1;
__REG32 Algorithm             : 1;
__REG32 TripleDESMode         : 1;
__REG32 DESMode               : 1;
__REG32 DataByteSwap          : 1;
__REG32                       : 1;
__REG32 IVByteSwap            : 1;
__REG32                       : 1;
__REG32 OutByteSwap           : 1;
__REG32                       :20;
__REG32 WriteAllow            : 1;
__REG32 AllTermination        : 1;
__REG32 Termination           : 1;
} __deccr_bits;

/* Cryptographic Engine/Security Accelerator/TDMA Interrupt Cause Register */
typedef struct{
__REG32 ZInt0                 : 1;
__REG32 ZInt1                 : 1;
__REG32 ZInt2                 : 1;
__REG32 ZInt3                 : 1;
__REG32 ZInt4                 : 1;
__REG32 AccInt0               : 1;
__REG32                       : 1;
__REG32 AccAndTDMAInt         : 1;
__REG32                       : 1;
__REG32 TDMACompIinterrupt    : 1;
__REG32 TDMAOwnIinterrupt     : 1;
__REG32 AccAndTDMAInt_CM      : 1;
__REG32 TPErr                 : 1;
__REG32                       :19;
} __cesaicr_bits;

/* Security Accelerator Command Register */
typedef struct{
__REG32 EnSecurityAccl        : 1;
__REG32                       : 1;
__REG32 DsSecurityAccl        : 1;
__REG32                       :29;
} __sacr_bits;

/* Security Accelerator Descriptor Pointer Register */
typedef struct{
__REG32 SecurityAcclDescPtr0  :16;
__REG32                       :16;
} __sadpr_bits;

/* Security Accelerator Configuration Register */
typedef struct{
__REG32 StopOnDecodeDigestErr : 1;
__REG32                       : 6;
__REG32 WaitForTDMA           : 1;
__REG32                       : 1;
__REG32 ActivateTDMA          : 1;
__REG32                       : 1;
__REG32 MultiPacketChainMode  : 1;
__REG32 TPPar                 : 1;
__REG32                       :19;
} __sacnfr_bits;

/* Security Accelerator Status Register */
typedef struct{
__REG32 AccActive             : 1;
__REG32                       : 7;
__REG32 DecodeDigestErr       : 1;
__REG32                       : 4;
__REG32 AcclState             :19;
} __saasr_bits;

/* SHA-1/MD5 Authentication Command Register */
typedef struct{
__REG32 Algorithm             : 1;
__REG32 Mode                  : 1;
__REG32 DataByteSwap          : 1;
__REG32                       : 1;
__REG32 IVByteSwap            : 1;
__REG32                       :26;
__REG32 Termination           : 1;
} __sha1md5acr_bits;

/* Window Control Register (n=0 3) */
typedef struct{
__REG32 Enable                : 1;
__REG32                       : 3;
__REG32 TargetID              : 4;
__REG32 Attr                  : 8;
__REG32 Size                  :16;
} __dtmawcr_bits;

/* TDMA Control Registers */
typedef struct{
__REG32 DstBurstLimit         : 3;
__REG32                       : 1;
__REG32 OutstandingRdEn       : 1;
__REG32                       : 1;
__REG32 SrcBurstLimit         : 3;
__REG32 ChainMode             : 1;
__REG32                       : 1;
__REG32 BS                    : 1;
__REG32 TDMAEn                : 1;
__REG32 FetchND               : 1;
__REG32 TDMAAct               : 1;
__REG32                       :17;
} __dtmacr_bits;

/* TDMA Byte Count Register */
typedef struct{
__REG32 ByteCnt               :16;
__REG32                       :15;
__REG32 Own                   : 1;
} __dtmbcr_bits;

/* TDMA Error Cause Register */
typedef struct{
__REG32 Miss                  : 1;
__REG32 DoubleHit             : 1;
__REG32 BothHit               : 1;
__REG32 DataError             : 1;
__REG32                       :28;
} __dtmdecr_bits;

/* CSU System Clock Prescaler Register */
typedef struct{
__REG32 SclockDivLow          : 8;
__REG32 SclockDivHigh         : 8;
__REG32                       :16;
} __tdm_csu_scpr_bits;

/* CSU Global Control Register */
typedef struct{
__REG32 CODECEnable           : 1;
__REG32                       :31;
} __tdm_csu_gcr_bits;

/* SPI Control Register */
typedef struct{
__REG32                       :10;
__REG32 SPIStat               : 1;
__REG32                       :21;
} __tdm_spi_cr_bits;

/* CODEC Access Command Low Register */
typedef struct{
__REG32 BYTE0                 : 8;
__REG32 BYTE1                 : 8;
__REG32                       :16;
} __tdm_codec_aclr_bits;

/* CODEC Access Command High Register */
typedef struct{
__REG32 BYTE2                 : 8;
__REG32 BYTE3                 : 8;
__REG32                       :16;
} __tdm_codec_achr_bits;

/* CODEC Registers Access Control */
typedef struct{
__REG32 BYTES_TO_XFER         : 2;
__REG32 LSB_MSB               : 1;
__REG32 RD_WR                 : 1;
__REG32 BYTE_TO_READ          : 1;
__REG32 LO_SPEED_CLK          : 1;
__REG32 CS_HIGH_CNT_VAL_READ  :10;
__REG32                       :16;
} __tdm_codec_rac_bits;

/* CODEC Registers Access Control */
typedef struct{
__REG32 CODEC_READ_DATA_LO_XFER : 8;
__REG32 CODEC_READ_DATA_HI      : 8;
__REG32                         :16;
} __tdm_codec_rdr_bits;

/* CODEC Registers Access Control1 */
typedef struct{
__REG32 CS_HIGH_CNT_VAL_WRITE   :10;
__REG32                         :22;
} __tdm_codec_rac1_bits;

/* PCM Control Register */
typedef struct{
__REG32 MstrPclkn               : 1;
__REG32 MasterFsn               : 1;
__REG32 DataPol                 : 1;
__REG32 FsPol                   : 1;
__REG32 InvFs                   : 1;
__REG32 LongFs                  : 1;
__REG32 PcmSampleSize           : 1;
__REG32                         : 1;
__REG32 CH0DlyEn                : 1;
__REG32 CH1DlyEn                : 1;
__REG32 CH0QualEn               : 1;
__REG32 CH1QualEn               : 1;
__REG32 QualPol                 : 1;
__REG32 CH0QualTyp              : 1;
__REG32 CH1QualTyp              : 1;
__REG32 DAA_CS_CTRL             : 1;
__REG32 CH0WBand                : 1;
__REG32 CH1WBand                : 1;
__REG32                         :13;
__REG32 PerfBit                 : 1;
} __tdm_pcm_cr_bits;

/* Channel Time Slot Control Register */
typedef struct{
__REG32 CH0RxTSlot              : 8;
__REG32 CH0TxTSlot              : 8;
__REG32 CH1RxTSlot              : 8;
__REG32 CH1TxTSlot              : 8;
} __tdm_ctscr_bits;

/* Channel 0/1 Delay Control Register */
typedef struct{
__REG32 CHRxDly                 :10;
__REG32                         : 6;
__REG32 CHnTxDly                :10;
__REG32                         : 6;
} __tdm_cdcr_bits;

/* Channel 0/1 Enable and Disable Register (n=0 1) */
typedef struct{
__REG32 CHnRxEn                 : 1;
__REG32                         : 7;
__REG32 CHnTxEn                 : 1;
__REG32                         :23;
} __tdm_cedr_bits;

/* Channel 0/1 Buffer Ownership Register (n=0 1) */
typedef struct{
__REG32 RX_DMA_ST_ADDR_OWN_CHx  : 1;
__REG32                         : 7;
__REG32 TX_DMA_ST_ADDR_OWN_CHx  : 1;
__REG32                         :23;
} __tdm_cbor_bits;

/* Channel 0/1 Total Sample Count Register (n=0 1) */
typedef struct{
__REG32 CH_TOTAL_SMPL_CNT       : 8;
__REG32 CH_INT_SMPL_CNT         : 8;
__REG32                         :16;
} __tdm_ctotscr_bits;

/* Number of Time Slots Register */
typedef struct{
__REG32 NO_OF_TS                : 8;
__REG32                         :24;
} __tdm_ntsr_bits;

/* TDM PCM Clock Rate Divider Register */
typedef struct{
__REG32 PCM_CLK_DIV             : 8;
__REG32                         :24;
} __tdm_pcm_crdr_bits;

/* Interrupt Status Register */
typedef struct{
__REG32 OFLOW_CH0_INT           : 1;
__REG32 UFLOW_CH0_INT           : 1;
__REG32 OFLOW_CH1_INT           : 1;
__REG32 UFLOW_CH1_INT           : 1;
__REG32 SCOCH0_RX_INT           : 1;
__REG32 SCOCH0_TX_INT           : 1;
__REG32 SCOCH1_RX_INT           : 1;
__REG32 SCOCH1_TX_INT           : 1;
__REG32 CH0_RX_IDLE             : 1;
__REG32 CH0_TX_IDLE             : 1;
__REG32 CH1_RX_IDLE             : 1;
__REG32 CH1_TX_IDLE             : 1;
__REG32 RXFIFO0_FULL_INT        : 1;
__REG32 TXFIFO0_EMPTY_INT       : 1;
__REG32 RXFIFO1_FULL_INT        : 1;
__REG32 TXFIFO1_EMPTY_INT       : 1;
__REG32 DMA_ABORTED_INT         : 1;
__REG32 CODEC_INT               : 1;
__REG32                         :14;
} __tdm_isr_bits;

/* Miscellaneous Control Register */
typedef struct{
__REG32 CODEC_RST               : 1;
__REG32                         :31;
} __tdm_mcr_bits;

/* Current Time Slot Register */
typedef struct{
__REG32 CUR_TS                  : 8;
__REG32                         :24;
} __tdm_ctsr_bits;

/* TDM Revision Register */
typedef struct{
__REG32 TDM_IP_REV              :16;
__REG32                         :16;
} __tdm_rr_bits;

/* TDM Channel 0/1 Debug Register (n=0 1) */
typedef struct{
__REG32 RX_WR_CTRL_STATE_CH_RX  : 2;
__REG32 PCM_RX_IF_STATE_CH_RX   : 3;
__REG32 GNT_CH_RX               : 1;
__REG32 REQ_CH_RX               : 1;
__REG32 OVERFLOW_PENDING_CH_RX  : 1;
__REG32 CUR_SAMPLE_CNT_CH_RX    : 8;
__REG32 TX_RD_CTRL_STATE_CH_TX  : 3;
__REG32 PCM_TX_IF_STATE_CH_TX   : 3;
__REG32 GNT_CH_TX               : 1;
__REG32 REQ_CH_TX               : 1;
__REG32 CUR_SAMPLE_CNT_CH_TX    : 8;
} __tdm_cdr_bits;

/* TDM DMA Abort Register 2 */
typedef struct{
__REG32 DMA_TRANSFER_LENGTH_ABORTED   :16;
__REG32 DMA_ID_ABORTED                :12;
__REG32 DMA_CYCLE_TYPE_ABORTED        : 1;
__REG32                               : 3;
} __tdm_dmaar2_bits;

/* TDM Channel 0/1 Wideband Delay Control Register */
typedef struct{
__REG32 CH_WBAND_RX_DLY :10;
__REG32                 : 6;
__REG32 CH_WBAND_TX_DLY :10;
__REG32                 : 6;
} __tdm_cwdcr_bits;

/* TDM/SPI Interface Pin Multiplexing Register */
typedef struct{
__REG32 SPIOutEnn       : 1;
__REG32                 :31;
} __tdm_ipmr_bits;

/* TDM-MBUS Configuration Register */
typedef struct{
__REG32 Timeout         : 8;
__REG32                 : 8;
__REG32 TimeoutEn       : 1;
__REG32                 :15;
} __tdm_mbuscr_bits;

/* Window0/1/2/3 Control Register */
typedef struct{
__REG32 WinEn           : 1;
__REG32                 : 3;
__REG32 Target          : 4;
__REG32 Attr            : 8;
__REG32 Size            :16;
} __tdm_wcr_bits;

/* TDM-Mbus Configuration 1 Register */
typedef struct{
__REG32 PCMReset        : 1;
__REG32                 :31;
} __tdm_mbusc1r_bits;

/* Basic DMA Command Register */
typedef struct{
__REG32 Start               : 1;
__REG32                     : 2;
__REG32 Read                : 1;
__REG32                     : 4;
__REG32 DRegionValid        : 1;
__REG32 DataRegionLast      : 1;
__REG32 ContFromPrev        : 1;
__REG32                     : 5;
__REG32 DataRegionByteCount :16;
} __satahc_bdmacr_bits;

/* Basic DMA Status Register */
typedef struct{
__REG32 BasicDMAActive      : 1;
__REG32 BasicDMAError       : 1;
__REG32 BasicDMAPaused      : 1;
__REG32 BasicDMALast        : 1;
__REG32                     :28;
} __satahc_bdmasr_bits;

/* EDMA Configuration Register */
typedef struct{
__REG32                     : 5;
__REG32 eSATANatvCmdQue     : 1;
__REG32                     : 2;
__REG32 eRdBSz              : 1;
__REG32 eQue                : 1;
__REG32                     : 1;
__REG32 eRdBSzExt           : 1;
__REG32                     : 1;
__REG32 eWrBufferLen        : 1;
__REG32                     : 2;
__REG32 eEDMAFBS            : 1;
__REG32 eCutThroughEn       : 1;
__REG32 eEarlyCompletionEn  : 1;
__REG32                     : 3;
__REG32 eHostQueueCacheEn   : 1;
__REG32 eMaskRxPM           : 1;
__REG32 ResumeDis           : 1;
__REG32                     : 1;
__REG32 eDMAFBS             : 1;
__REG32                     : 5;
} __satahc_edmacfgr_bits;

/* EDMA Interrupt Error Cause Register */
typedef struct{
__REG32                     : 2;
__REG32 eDevErr             : 1;
__REG32 eDevDis             : 1;
__REG32 eDevCon             : 1;
__REG32 SerrInt             : 1;
__REG32                     : 1;
__REG32 eSelfDis            : 1;
__REG32 eTransInt           : 1;
__REG32                     : 3;
__REG32 eIORdyErr           : 1;
__REG32 LinkCtlRxErr        : 4;
__REG32 LinkDataRxErr       : 4;
__REG32 LinkCtlTxErr        : 5;
__REG32 LinkDataTxErr       : 5;
__REG32 TransProtErr        : 1;
} __satahc_edmaiecr_bits;

/* EDMA Request Queue In-Pointer Register */
typedef struct{
__REG32                     : 5;
__REG32 eRqQIP              : 5;
__REG32 eRqQBA              :22;
} __satahc_edmarqir_bits;

/* EDMA Request Queue Out-Pointer Register */
typedef struct{
__REG32                     : 5;
__REG32 eRqQOP              : 5;
__REG32                     :22;
} __satahc_edmarqor_bits;

/* EDMA Response Queue In-Pointer Register */
typedef struct{
__REG32                     : 3;
__REG32 eRpQIP              : 5;
__REG32                     :24;
} __satahc_edmarsqir_bits;

/* EDMA Response Queue Out-Pointer Register */
typedef struct{
__REG32                     : 3;
__REG32 eRPQOP              : 5;
__REG32 eRPQBA              :24;
} __satahc_edmarsqor_bits;

/* EDMA Command Register */
typedef struct{
__REG32 eEnEDMA             : 1;
__REG32 eDsEDMA             : 1;
__REG32 eAtaRst             : 1;
__REG32                     : 1;
__REG32 eEDMAFrz            : 1;
__REG32                     :27;
} __satahc_edmacr_bits;

/* EDMA Command Register */
typedef struct{
__REG32 eDevQueTAG          : 5;
__REG32 eDevDir             : 1;
__REG32 eCacheEmpty         : 1;
__REG32 EDMAIdle            : 1;
__REG32 eSTATE              : 8;
__REG32 eIOId               : 6;
__REG32                     :10;
} __satahc_edmasr_bits;

/* EDMA IORdy Timeout Register */
typedef struct{
__REG32 eIORdyTimeout       :16;
__REG32                     :16;
} __satahc_edmaiortr_bits;

/* EDMA Command Delay Threshold Register */
typedef struct{
__REG32 CmdDelayThrshd      :16;
__REG32                     :15;
__REG32 CMDDataoutDelayEn   : 1;
} __satahc_edmaicdtr_bits;

/* SATAHCA Configuration Register */
typedef struct{
__REG32 Timeout             : 8;
__REG32 DmaBS               : 1;
__REG32 EDmaBS              : 1;
__REG32 PrdpBS              : 1;
__REG32                     : 5;
__REG32 TimeoutEn           : 1;
__REG32                     : 7;
__REG32 CoalDis             : 1;
__REG32                     : 7;
} __satahca_cnfr_bits;

/* SATAHCA Request Queue Out-Pointer Register */
typedef struct{
__REG32 eRQQOP0             : 7;
__REG32                     :25;
} __satahca_rqor_bits;

/* SATAHCA Response Queue In-Pointer Register */
typedef struct{
__REG32 eRPQIP0             : 7;
__REG32                     :25;
} __satahca_rsqir_bits;

/* SATAHCA Interrupt Coalescing Threshold Register */
typedef struct{
__REG32 SAICOALT            : 8;
__REG32                     :24;
} __satahca_ictr_bits;

/* SATAHCA Interrupt Time Threshold Register */
typedef struct{
__REG32 SAITMTH             :24;
__REG32                     : 8;
} __satahca_ittr_bits;

/* SATAHCA Interrupt Cause Register */
typedef struct{
__REG32 SaCrpb0DoneDMA0Done : 1;
__REG32 SaCrpb1DoneDMA1Done : 1;
__REG32                     : 2;
__REG32 SaIntCoal           : 1;
__REG32                     : 3;
__REG32 SaDevInterrupt0     : 1;
__REG32 SaDevInterrupt1     : 1;
__REG32                     :22;
} __satahca_icr_bits;

/* SATAHCA Main Interrupt Cause Register */
/* SATAHCA Main Interrupt Mask Register */
typedef struct{
__REG32 Sata0Err            : 1;
__REG32 Sata0Done           : 1;
__REG32 Sata1Err            : 1;
__REG32 Sata1Done           : 1;
__REG32 Sata0DmaDone        : 1;
__REG32 Sata1DmaDone        : 1;
__REG32                     : 2;
__REG32 SataCoalDone        : 1;
__REG32                     :23;
} __satahca_micr_bits;

/* SATAHCA Main Interrupt Mask Register */
typedef struct{
__REG32 act_led_blink       : 1;
__REG32                     : 1;
__REG32 act_presence        : 1;
__REG32 led_polarity        : 1;
__REG32                     :28;
} __satahca_ledcr_bits;

/* SATAHCA WindowX Control Register */
typedef struct{
__REG32 WinEn               : 1;
__REG32 WrBL                : 1;
__REG32                     : 2;
__REG32 Target              : 4;
__REG32 Attr                : 8;
__REG32 Size                :16;
} __satahca_wcr_bits;

/* Serial-ATA Interface Configuration Register */
typedef struct{
__REG32 RefClkCnf           : 2;
__REG32                     : 4;
__REG32 PhySSCEn            : 1;
__REG32 Gen2En              : 1;
__REG32 CommEn              : 1;
__REG32 PhyShutdown         : 1;
__REG32 TargetMode          : 1;
__REG32 ComChannel          : 1;
__REG32                     : 1;
__REG32 EMPH_LVLADJ_EN      : 1;
__REG32 TX_EMPH_EN          : 1;
__REG32 EMPH_TYPE_PRE       : 1;
__REG32                     : 3;
__REG32 CLK_DET_EN          : 1;
__REG32                     : 4;
__REG32 IgnoreBsy           : 1;
__REG32 LinkRstEn           : 1;
__REG32 CmdRetxDs           : 1;
__REG32                     : 5;
} __satahc_icfgr_bits;

/* Serial-ATA PLL Configuration Register */
typedef struct{
__REG32 Gen1FBDIV           : 8;
__REG32 Gen1REFDIV          : 2;
__REG32 Gen1INTPI           : 3;
__REG32 Gen2FBDIV           : 8;
__REG32 Gen2REFDIV          : 2;
__REG32 Gen2INTPI           : 3;
__REG32                     : 6;
} __satahc_pllcnfr_bits;

/* SStatus Register */
typedef struct{
__REG32 DET                 : 4;
__REG32 SPD                 : 4;
__REG32 IPM                 : 4;
__REG32                     :20;
} __satahc_ssr_bits;

/* SError Register */
typedef struct{
__REG32                     : 1;
__REG32 M                   : 1;
__REG32                     :14;
__REG32 N                   : 1;
__REG32                     : 1;
__REG32 W                   : 1;
__REG32 B                   : 1;
__REG32 C                   : 1;
__REG32 D                   : 1;
__REG32 H                   : 1;
__REG32 S                   : 1;
__REG32 T                   : 1;
__REG32                     : 1;
__REG32 X                   : 1;
__REG32                     : 5;
} __satahc_ser_bits;

/* SControl Register */
typedef struct{
__REG32 DET                 : 4;
__REG32 SPD                 : 4;
__REG32 IPM                 : 4;
__REG32 SPM                 : 4;
__REG32                     :16;
} __satahc_scr_bits;

/* LTMode Register */
typedef struct{
__REG32 RcvWaterMark        : 6;
__REG32                     : 1;
__REG32 NearEndLBEn         : 1;
__REG32                     :24;
} __satahc_ltmr_bits;

/* PHY Mode 3 Register */
typedef struct{
__REG32 TX_OFFSET_READY     : 1;
__REG32                     : 1;
__REG32 AVG_WINDOW          : 2;
__REG32 INIT_TXFOFFS        :10;
__REG32 FRC_TXFOFFS         : 1;
__REG32 AUTO_TX_OFFSET      : 1;
__REG32 SEL_DSPREAD         : 3;
__REG32 SSC_DSPREAD         : 1;
__REG32 SSC_EN_MASK_XOR     : 1;
__REG32 SEL_MUCNT_LEN       : 2;
__REG32 SELMUFF             : 2;
__REG32 SELMUFI             : 2;
__REG32 SELMUPF             : 2;
__REG32 SELMUPI             : 2;
__REG32 MUCNT_EN            : 1;
} __satahc_phym3r_bits;

/* PHY Mode 4 Register */
typedef struct{
__REG32 SATU_OD8                : 1;
__REG32 RXSAT_DIS               : 1;
__REG32 FLOOP_EN                : 1;
__REG32 INIT_RXOFFS             : 8;
__REG32 RFRC_RXOFFS             : 1;
__REG32                         : 6;
__REG32 HOTPLUG_TIMER           : 3;
__REG32 RXCLK_SEL               : 1;
__REG32 TXCLK_SEL               : 1;
__REG32 CLK_MONITOR_EN          : 1;
__REG32 SQUELCH_FLOOP_ON        : 1;
__REG32 PortSelector            : 1;
__REG32 DISABLE_MISMATCH        : 1;
__REG32 SPEED_CHANGE_SEND_IDLE  : 1;
__REG32 FREEZE_AFTER_LOCK       : 1;
__REG32 DISSwap                 : 1;
__REG32 PARTIAL_TRAINING        : 1;
__REG32 OOB_Bypass              : 1;
} __satahc_phym4r_bits;

/* PHY Mode 1 Register */
typedef struct{
__REG32                         : 1;
__REG32 VTHVCOCAL               : 2;
__REG32 EXTKVCO                 : 3;
__REG32 EXTKVCO_EN              : 1;
__REG32 ICP                     : 4;
__REG32                         : 5;
__REG32 RxVCom                  : 2;
__REG32 RxVDD                   : 2;
__REG32 TXVDD                   : 2;
__REG32                         :10;
} __satahc_phym1r_bits;

/* PHY Mode 2 Register */
typedef struct{
__REG32 FORCE_PU_TX             : 1;
__REG32 FORCE_PU_RX             : 1;
__REG32 PU_PLL                  : 1;
__REG32 PU_IVREF                : 1;
__REG32 PD_TX_AUTO              : 1;
__REG32                         : 1;
__REG32 TM_CLK_STAT             : 1;
__REG32                         : 4;
__REG32 LOOPBACK                : 1;
__REG32                         : 8;
__REG32 TXIMP                   : 4;
__REG32 EXTIMP_EN               : 1;
__REG32 ND                      : 1;
__REG32 EXTRXIMP                : 4;
__REG32 PLL_CAL_EN              : 1;
__REG32 IMP_CAL_EN              : 1;
} __satahc_phym2r_bits;

/* BIST Control Register */
typedef struct{
__REG32 BISTPattern             : 8;
__REG32 BISTMode                : 1;
__REG32 BISTEn                  : 1;
__REG32 BISTResult              : 1;
__REG32                         :21;
} __satahc_bistcr_bits;

/* Serial-ATA Interface Control Register */
typedef struct{
__REG32 PMportTx                : 4;
__REG32                         : 4;
__REG32 VendorUqMd              : 1;
__REG32 VendorUqSend            : 1;
__REG32                         : 6;
__REG32 eDMAActivate            : 1;
__REG32                         : 7;
__REG32 ClearStatus             : 1;
__REG32 SendSftRst              : 1;
__REG32                         : 6;
} __satahc_icr_bits;

/* Serial-ATA Interface Test Control Register */
typedef struct{
__REG32 MBistEn                 : 1;
__REG32 TransFrmSizExt          : 1;
__REG32                         : 6;
__REG32 LBEnable                : 1;
__REG32 LBPattern               : 4;
__REG32 LBStartRd               : 1;
__REG32 TransFrmSiz             : 2;
__REG32 PortNumDevErr           :16;
} __satahc_itcr_bits;

/* Serial-ATA Interface Test Control Register */
typedef struct{
__REG32 FISTypeRx               : 8;
__REG32 PMportRx                : 4;
__REG32 VendorUqDn              : 1;
__REG32 VendorUqErr             : 1;
__REG32 MBistRdy                : 1;
__REG32 MBistFail               : 1;
__REG32 AbortCommand            : 1;
__REG32 LBPass                  : 1;
__REG32 DMAAct                  : 1;
__REG32 PIOAct                  : 1;
__REG32 RxHdAct                 : 1;
__REG32 TxHdAct                 : 1;
__REG32 PlugIn                  : 1;
__REG32 LinkDown                : 1;
__REG32 TransFsmSts             : 5;
__REG32                         : 1;
__REG32 RxBIST                  : 1;
__REG32 N                       : 1;
} __satahc_isr_bits;

/* FIS Configuration Register */
typedef struct{
__REG32 FISWait4RdyEn           : 8;
__REG32 FISWait4HostRdyEn       : 8;
__REG32 FISDMAActiveSyncResp    : 1;
__REG32 FISUnrecTypeCont        : 1;
__REG32                         :14;
} __satahc_fiscr_bits;

/* FIS Interrupt Cause Register */
typedef struct{
__REG32 FISWait4Rdy             : 8;
__REG32 FISWait4HostRdy         : 8;
__REG32                         : 8;
__REG32 FISTxDone               : 1;
__REG32 FISTxErr                : 1;
__REG32                         : 6;
} __satahc_fisicr_bits;

/* PHYMODE9_GEN1/2 Register */
typedef struct{
__REG32 TXAMP                   : 4;
__REG32 TX_PRE_EMPH             : 4;
__REG32                         : 6;
__REG32 TXAMP4                  : 1;
__REG32                         :17;
} __satahc_phym9genr_bits;

/* PHYCFG Register */
typedef struct{
__REG32 SQ_THRESHOLD            : 4;
__REG32 VTH_DISCON              : 5;
__REG32                         :23;
} __satahc_phycfgr_bits;

/* PHYTCTL Register */
typedef struct{
__REG32 TEST_ANA                : 6;
__REG32 ND                      : 6;
__REG32 LINK_TESTSEL            : 4;
__REG32                         :16;
} __satahc_phytctlr_bits;

/* PHYMODE10 Register */
typedef struct{
__REG32 AVG                     :10;
__REG32                         : 1;
__REG32 AVG_READY               : 1;
__REG32 AVG_READ_EN             : 1;
__REG32                         : 3;
__REG32 NXT_AVG                 :10;
__REG32                         : 1;
__REG32 PEAK_READY              : 1;
__REG32                         : 4;
} __satahc_phymode10r_bits;

/* Base Address Register (n=0 7) */
typedef struct{
__REG32 Target                  : 4;
__REG32                         : 4;
__REG32 Attr                    : 8;
__REG32 Base                    :16;
} __idma_bar_bits;

/* Size Register (n=0 7) */
typedef struct{
__REG32                         :16;
__REG32 Size                    :16;
} __idma_sr_bits;

/* Channel Access Protect Register (n=0 3) */
typedef struct{
__REG32 Win0                    : 2;
__REG32 Win1                    : 2;
__REG32 Win2                    : 2;
__REG32 Win3                    : 2;
__REG32 Win4                    : 2;
__REG32 Win5                    : 2;
__REG32 Win6                    : 2;
__REG32 Win7                    : 2;
__REG32                         :16;
} __idma_capr_bits;

/* Base Address Enable Register */
typedef struct{
__REG32 En0                     : 1;
__REG32 En1                     : 1;
__REG32 En2                     : 1;
__REG32 En3                     : 1;
__REG32 En4                     : 1;
__REG32 En5                     : 1;
__REG32 En6                     : 1;
__REG32 En7                     : 1;
__REG32                         :24;
} __idma_baer_bits;

/* Channel Control (Low) Register (n=0 3) */
typedef struct{
__REG32 DstBurstLimit           : 3;
__REG32 SrcHold                 : 1;
__REG32                         : 1;
__REG32 DestHold                : 1;
__REG32 SrcBurstLimit           : 3;
__REG32 ChainMode               : 1;
__REG32 IntMode                 : 1;
__REG32 DemandMode              : 1;
__REG32 ChanEn                  : 1;
__REG32 FetchND                 : 1;
__REG32 ChanAct                 : 1;
__REG32 DMAReqDir               : 1;
__REG32                         : 1;
__REG32 CDEn                    : 1;
__REG32                         : 2;
__REG32 Abr                     : 1;
__REG32 SAddrOvr                : 2;
__REG32 DAddrOvr                : 2;
__REG32 NAddrOvr                : 2;
__REG32                         : 4;
__REG32 DescMode                : 1;
} __idma_cclr_bits;

/* Channel Control (High) Register (n=0 3) */
typedef struct{
__REG32                         : 1;
__REG32 DescBS                  : 1;
__REG32                         : 5;
__REG32 DPPar                   : 1;
__REG32                         :24;
} __idma_cchr_bits;

/* Mbus Timeout Register */
typedef struct{
__REG32 Timeout                 : 8;
__REG32                         : 8;
__REG32 TimeoutEn               : 1;
__REG32                         :15;
} __idma_mbustr_bits;

/* Channel IDMA Byte Count Register */
typedef struct{
__REG32 ByteCnt                 :24;
__REG32                         : 6;
__REG32 BCLeft                  : 1;
__REG32 Own                     : 1;
} __idma_cbcr_bits;

/* Interrupt Cause Register */
typedef struct{
__REG32 Comp                    : 1;
__REG32 AddrMiss                : 1;
__REG32 AccProt                 : 1;
__REG32 WrProt                  : 1;
__REG32 Own                     : 1;
__REG32                         : 2;
__REG32 DPErr                   : 1;
__REG32 Various1                : 5;
__REG32                         : 3;
__REG32 Various2                : 5;
__REG32                         : 3;
__REG32 Various3                : 5;
__REG32                         : 3;
} __idma_icr_bits;

/* Error Select Register */
typedef struct{
__REG32 Sel                     : 5;
__REG32                         :27;
} __idma_esr_bits;

/* XOR Engine [0..1] Window Control (XExWCR) Register (n=0 1) */
typedef struct{
__REG32 Win0en                  : 1;
__REG32 Win1en                  : 1;
__REG32 Win2en                  : 1;
__REG32 Win3en                  : 1;
__REG32 Win4en                  : 1;
__REG32 Win5en                  : 1;
__REG32 Win6en                  : 1;
__REG32 Win7en                  : 1;
__REG32                         : 8;
__REG32 Win0acc                 : 2;
__REG32 Win1acc                 : 2;
__REG32 Win2acc                 : 2;
__REG32 Win3acc                 : 2;
__REG32 Win4acc                 : 2;
__REG32 Win5acc                 : 2;
__REG32 Win6acc                 : 2;
__REG32 Win7acc                 : 2;
} __xexwcr_bits;

/* XOR Engine Base Address (XEBARx) Register (n=0 7) */
typedef struct{
__REG32 Target                  : 4;
__REG32                         : 4;
__REG32 Attr                    : 8;
__REG32 Base                    :16;
} __xebarx_bits;

/* XOR Engine Size Mask (XESMRx) Register (n=0 7) */
typedef struct{
__REG32                         :16;
__REG32 SizeMask                :16;
} __xesmrx_bits;

/* XOR Engine [0..1] Address Override Control (XEAOCR) Register (n=0 1) */
typedef struct{
__REG32 SA0OvrEn                : 1;
__REG32 SA0OvrPtr               : 2;
__REG32 SA1OvrEn                : 1;
__REG32 SA1OvrPtr               : 2;
__REG32 SA2OvrEn                : 1;
__REG32 SA2OvrPtr               : 2;
__REG32 SA3OvrEn                : 1;
__REG32 SA3OvrPtr               : 2;
__REG32 SA4OvrEn                : 1;
__REG32 SA4OvrPtr               : 2;
__REG32 SA5OvrEn                : 1;
__REG32 SA5OvrPtr               : 2;
__REG32 SA6OvrEn                : 1;
__REG32 SA6OvrPtr               : 2;
__REG32 SA7OvrEn                : 1;
__REG32 SA7OvrPtr               : 2;
__REG32 DAOvrEn                 : 1;
__REG32 DAOvrPtr                : 2;
__REG32 NDAOvrEn                : 1;
__REG32 NDAOvrPtr               : 2;
__REG32                         : 2;
} __xeaocrx_bits;

/* XOR Engine Channel Arbiter (XECHAR) Register */
typedef struct{
__REG32 Slice0                  : 1;
__REG32 Slice1                  : 1;
__REG32 Slice2                  : 1;
__REG32 Slice3                  : 1;
__REG32 Slice4                  : 1;
__REG32 Slice5                  : 1;
__REG32 Slice6                  : 1;
__REG32 Slice7                  : 1;
__REG32                         :24;
} __xechar_bits;

/* XOR Engine [0..1] Configuration (XExCR) Register (n=0 1) */
typedef struct{
__REG32 OperationMode           : 3;
__REG32                         : 1;
__REG32 SrcBurstLimit           : 3;
__REG32                         : 1;
__REG32 DstBurstLimit           : 3;
__REG32                         : 1;
__REG32 DrdResSwp               : 1;
__REG32 DwrReqSwp               : 1;
__REG32 DesSwp                  : 1;
__REG32 RegAccProtect           : 1;
__REG32                         :16;
} __xexcr_bits;

/* XOR Engine [0..1] Activation (XExACTR) Register (n=0 1) */
typedef struct{
__REG32 XEStart                 : 1;
__REG32 XEstop                  : 1;
__REG32 XEpause                 : 1;
__REG32 XErestart               : 1;
__REG32 XEstatus                : 2;
__REG32                         :26;
} __xexactr_bits;

/* XOR Engine Timer Mode Control (XETMCR) Register */
typedef struct{
__REG32 TimerEn                 : 1;
__REG32                         : 7;
__REG32 SectionSizeCtrl         : 5;
__REG32                         :19;
} __xetmcr_bits;

/* XOR Engine Interrupt Cause (XEICR1) Register */
typedef struct{
__REG32 EOD0                    : 1;
__REG32 EOC0                    : 1;
__REG32 Stopped0                : 1;
__REG32 Paused0                 : 1;
__REG32 AddrDecode0             : 1;
__REG32 AccProt0                : 1;
__REG32 WrProt0                 : 1;
__REG32 OwnErr0                 : 1;
__REG32 IntParityErr0           : 1;
__REG32 XbarErr0                : 1;
__REG32                         : 6;
__REG32 EOD1                    : 1;
__REG32 EOC1                    : 1;
__REG32 Stopped1                : 1;
__REG32 Paused1                 : 1;
__REG32 AddrDecode1             : 1;
__REG32 AccProt1                : 1;
__REG32 WrProt1                 : 1;
__REG32 OwnErr1                 : 1;
__REG32 IntParityErr1           : 1;
__REG32 XbarErr1                : 1;
__REG32                         : 6;
} __xeicr1_bits;

/* XOR Engine Interrupt Mask (XEIMR) Register */
typedef struct{
__REG32 EODMask0                : 1;
__REG32 EOCMask0                : 1;
__REG32 StoppedMask0            : 1;
__REG32 PauseMask0              : 1;
__REG32 AddrDecodeMask0         : 1;
__REG32 AccProtMask0            : 1;
__REG32 WrProtMask0             : 1;
__REG32 OwnMask0                : 1;
__REG32 IntParityMask0          : 1;
__REG32 XbarMask0               : 1;
__REG32                         : 6;
__REG32 EODMask1                : 1;
__REG32 EOCMask1                : 1;
__REG32 StoppedMask1            : 1;
__REG32 PauseMask1              : 1;
__REG32 AddrDecodeMask1         : 1;
__REG32 AccProtMask1            : 1;
__REG32 WrProtMask1             : 1;
__REG32 OwnMask1                : 1;
__REG32 IntParityMask1          : 1;
__REG32 XbarMask1               : 1;
__REG32                         : 6;
} __xeimr_bits;

/* XOR Engine Error Cause (XEECR) Register */
typedef struct{
__REG32 ErrorType               : 5;
__REG32                         :27;
} __xeecr_bits;

/* UART External Control Register */
typedef struct{
__REG32 External_Writes_Enable  : 1;
__REG32                         : 7;
__REG32 DMAMode                 : 8;
__REG32                         :16;
} __uart_ecr_bits;

/* Interrupt Enable (IER) Register */
typedef struct{
__REG8  RxDataIntEn             : 1;
__REG8  TxHoldIntEn             : 1;
__REG8  RxLineStatIntEn         : 1;
__REG8  ModStatIntEn            : 1;
__REG8                          : 4;
} __uart_ier_bits;

/* FIFO Control (FCR) Register */
typedef union {
/* UARTx_FCR */
struct{
__REG8  FIFOEn                  : 1;
__REG8  RxFIFOReset             : 1;
__REG8  TxFIFOReset             : 1;
__REG8                          : 3;
__REG8  RxTrigger               : 2;
};
/* UARTx_IIR */
struct{
__REG8  InterruptID             : 4;
__REG8                          : 2;
__REG8  _FIFOEn                 : 2;
};
} __uart_fcr_bits;

/* Line Control (LCR) Register */
typedef struct{
__REG8  WLS                     : 2;
__REG8  Stop                    : 1;
__REG8  PEN                     : 1;
__REG8  EPS                     : 1;
__REG8                          : 1;
__REG8  Break                   : 1;
__REG8  DivLatchRdWrt           : 1;
} __uart_lcr_bits;

/* Modem Control (MCR) Register */
typedef struct{
__REG8                          : 1;
__REG8  RTS                     : 1;
__REG8                          : 2;
__REG8  Loopback                : 1;
__REG8  AFCE                    : 1;
__REG8                          : 2;
} __uart_mcr_bits;

/* Line Status (LSR) Register */
typedef struct{
__REG8  DataRxStat              : 1;
__REG8  OverRunErr              : 1;
__REG8  ParErr                  : 1;
__REG8  FrameErr                : 1;
__REG8  BI                      : 1;
__REG8  THRE                    : 1;
__REG8  TxEmpty                 : 1;
__REG8  RxFIFOErr               : 1;
} __uart_lsr_bits;

/* Modem Status (MSR) Register */
typedef struct{
__REG8  DCTS                    : 1;
__REG8                          : 3;
__REG8  CTS                     : 1;
__REG8                          : 3;
} __uart_msr_bits;

/* TWSI Slave Address Register */
typedef struct{
__REG8  GCE                     : 1;
__REG8  SAddr                   : 7;
} __twsi_sar_bits;

/* TWSI Control Register */
typedef struct{
__REG8                          : 2;
__REG8  ACK                     : 1;
__REG8  IFlg                    : 1;
__REG8  Stop                    : 1;
__REG8  Start                   : 1;
__REG8  TWSIEn                  : 1;
__REG8  IntEn                   : 1;
} __twsi_cr_bits;

/* TWSI Baud Rate Register */
typedef struct{
__REG8  N                       : 3;
__REG8  M                       : 4;
__REG8                          : 1;
} __twsi_brr_bits;

/* Serial Memory Interface Control Register */
typedef struct{
__REG32 FCSEn                   : 1;
__REG32 SMemRdy                 : 1;
__REG32                         :30;
} __spi_cr_bits;

/* Serial Memory Interface Configuration Register */
typedef struct{
__REG32 SpiClkPrescale          : 5;
__REG32 BYTE_LEN                : 1;
__REG32                         : 2;
__REG32 AddrBurstLen            : 2;
__REG32 BurstReadCommand        : 1;
__REG32                         :21;
} __spi_cfgr_bits;

/* Serial Memory Interface Interrupt Mask Register */
typedef struct{
__REG32 SMemRdiIntMask          : 1;
__REG32                         :31;
} __spi_icr_bits;

/* CPU Timers Control Register */
typedef struct{
__REG32 CPUTimer0En             : 1;
__REG32 CPUTimer0Auto           : 1;
__REG32 CPUTimer1En             : 1;
__REG32 CPUTimer1Auto           : 1;
__REG32 CPUWDTimerEn            : 1;
__REG32 CPUWDTimerAuto          : 1;
__REG32 CPUTimer2En             : 1;
__REG32 CPUTimer2Auto           : 1;
__REG32 CPUTimer3En             : 1;
__REG32 CPUTimer3Auto           : 1;
__REG32                         :22;
} __tmr_cr_bits;

/* GPIO Data Out Register */
typedef struct{
__REG32 GPIODOut0               : 1;
__REG32 GPIODOut1               : 1;
__REG32 GPIODOut2               : 1;
__REG32 GPIODOut3               : 1;
__REG32 GPIODOut4               : 1;
__REG32 GPIODOut5               : 1;
__REG32 GPIODOut6               : 1;
__REG32 GPIODOut7               : 1;
__REG32 GPIODOut8               : 1;
__REG32 GPIODOut9               : 1;
__REG32 GPIODOut10              : 1;
__REG32 GPIODOut11              : 1;
__REG32 GPIODOut12              : 1;
__REG32 GPIODOut13              : 1;
__REG32 GPIODOut14              : 1;
__REG32 GPIODOut15              : 1;
__REG32 GPIODOut16              : 1;
__REG32 GPIODOut17              : 1;
__REG32 GPIODOut18              : 1;
__REG32 GPIODOut19              : 1;
__REG32 GPIODOut20              : 1;
__REG32 GPIODOut21              : 1;
__REG32 GPIODOut22              : 1;
__REG32 GPIODOut23              : 1;
__REG32 GPIODOut24              : 1;
__REG32 GPIODOut25              : 1;
__REG32 GPIODOut26              : 1;
__REG32 GPIODOut27              : 1;
__REG32 GPIODOut28              : 1;
__REG32 GPIODOut29              : 1;
__REG32 GPIODOut30              : 1;
__REG32 GPIODOut31              : 1;
} __gpio_dor_bits;

/* GPIO Control Register */
typedef struct{
__REG32 GPIODOutEn0               : 1;
__REG32 GPIODOutEn1               : 1;
__REG32 GPIODOutEn2               : 1;
__REG32 GPIODOutEn3               : 1;
__REG32 GPIODOutEn4               : 1;
__REG32 GPIODOutEn5               : 1;
__REG32 GPIODOutEn6               : 1;
__REG32 GPIODOutEn7               : 1;
__REG32 GPIODOutEn8               : 1;
__REG32 GPIODOutEn9               : 1;
__REG32 GPIODOutEn10              : 1;
__REG32 GPIODOutEn11              : 1;
__REG32 GPIODOutEn12              : 1;
__REG32 GPIODOutEn13              : 1;
__REG32 GPIODOutEn14              : 1;
__REG32 GPIODOutEn15              : 1;
__REG32 GPIODOutEn16              : 1;
__REG32 GPIODOutEn17              : 1;
__REG32 GPIODOutEn18              : 1;
__REG32 GPIODOutEn19              : 1;
__REG32 GPIODOutEn20              : 1;
__REG32 GPIODOutEn21              : 1;
__REG32 GPIODOutEn22              : 1;
__REG32 GPIODOutEn23              : 1;
__REG32 GPIODOutEn24              : 1;
__REG32 GPIODOutEn25              : 1;
__REG32 GPIODOutEn26              : 1;
__REG32 GPIODOutEn27              : 1;
__REG32 GPIODOutEn28              : 1;
__REG32 GPIODOutEn29              : 1;
__REG32 GPIODOutEn30              : 1;
__REG32 GPIODOutEn31              : 1;
} __gpio_cr_bits;

/* GPIO Blink Enable Register */
typedef struct{
__REG32 GPIODBlink0               : 1;
__REG32 GPIODBlink1               : 1;
__REG32 GPIODBlink2               : 1;
__REG32 GPIODBlink3               : 1;
__REG32 GPIODBlink4               : 1;
__REG32 GPIODBlink5               : 1;
__REG32 GPIODBlink6               : 1;
__REG32 GPIODBlink7               : 1;
__REG32 GPIODBlink8               : 1;
__REG32 GPIODBlink9               : 1;
__REG32 GPIODBlink10              : 1;
__REG32 GPIODBlink11              : 1;
__REG32 GPIODBlink12              : 1;
__REG32 GPIODBlink13              : 1;
__REG32 GPIODBlink14              : 1;
__REG32 GPIODBlink15              : 1;
__REG32 GPIODBlink16              : 1;
__REG32 GPIODBlink17              : 1;
__REG32 GPIODBlink18              : 1;
__REG32 GPIODBlink19              : 1;
__REG32 GPIODBlink20              : 1;
__REG32 GPIODBlink21              : 1;
__REG32 GPIODBlink22              : 1;
__REG32 GPIODBlink23              : 1;
__REG32 GPIODBlink24              : 1;
__REG32 GPIODBlink25              : 1;
__REG32 GPIODBlink26              : 1;
__REG32 GPIODBlink27              : 1;
__REG32 GPIODBlink28              : 1;
__REG32 GPIODBlink29              : 1;
__REG32 GPIODBlink30              : 1;
__REG32 GPIODBlink31              : 1;
} __gpio_ber_bits;

/* GPIO Data In Polarity Register */
typedef struct{
__REG32 GPIODInActLow0               : 1;
__REG32 GPIODInActLow1               : 1;
__REG32 GPIODInActLow2               : 1;
__REG32 GPIODInActLow3               : 1;
__REG32 GPIODInActLow4               : 1;
__REG32 GPIODInActLow5               : 1;
__REG32 GPIODInActLow6               : 1;
__REG32 GPIODInActLow7               : 1;
__REG32 GPIODInActLow8               : 1;
__REG32 GPIODInActLow9               : 1;
__REG32 GPIODInActLow10              : 1;
__REG32 GPIODInActLow11              : 1;
__REG32 GPIODInActLow12              : 1;
__REG32 GPIODInActLow13              : 1;
__REG32 GPIODInActLow14              : 1;
__REG32 GPIODInActLow15              : 1;
__REG32 GPIODInActLow16              : 1;
__REG32 GPIODInActLow17              : 1;
__REG32 GPIODInActLow18              : 1;
__REG32 GPIODInActLow19              : 1;
__REG32 GPIODInActLow20              : 1;
__REG32 GPIODInActLow21              : 1;
__REG32 GPIODInActLow22              : 1;
__REG32 GPIODInActLow23              : 1;
__REG32 GPIODInActLow24              : 1;
__REG32 GPIODInActLow25              : 1;
__REG32 GPIODInActLow26              : 1;
__REG32 GPIODInActLow27              : 1;
__REG32 GPIODInActLow28              : 1;
__REG32 GPIODInActLow29              : 1;
__REG32 GPIODInActLow30              : 1;
__REG32 GPIODInActLow31              : 1;
} __gpio_dipr_bits;

/* GPIO Data In Register */
typedef struct{
__REG32 GPIODIn0               : 1;
__REG32 GPIODIn1               : 1;
__REG32 GPIODIn2               : 1;
__REG32 GPIODIn3               : 1;
__REG32 GPIODIn4               : 1;
__REG32 GPIODIn5               : 1;
__REG32 GPIODIn6               : 1;
__REG32 GPIODIn7               : 1;
__REG32 GPIODIn8               : 1;
__REG32 GPIODIn9               : 1;
__REG32 GPIODIn10              : 1;
__REG32 GPIODIn11              : 1;
__REG32 GPIODIn12              : 1;
__REG32 GPIODIn13              : 1;
__REG32 GPIODIn14              : 1;
__REG32 GPIODIn15              : 1;
__REG32 GPIODIn16              : 1;
__REG32 GPIODIn17              : 1;
__REG32 GPIODIn18              : 1;
__REG32 GPIODIn19              : 1;
__REG32 GPIODIn20              : 1;
__REG32 GPIODIn21              : 1;
__REG32 GPIODIn22              : 1;
__REG32 GPIODIn23              : 1;
__REG32 GPIODIn24              : 1;
__REG32 GPIODIn25              : 1;
__REG32 GPIODIn26              : 1;
__REG32 GPIODIn27              : 1;
__REG32 GPIODIn28              : 1;
__REG32 GPIODIn29              : 1;
__REG32 GPIODIn30              : 1;
__REG32 GPIODIn31              : 1;
} __gpio_dir_bits;

/* GPIO Edge Sensitive Interrupt Cause Register */
typedef struct{
__REG32 GPIOInt0               : 1;
__REG32 GPIOInt1               : 1;
__REG32 GPIOInt2               : 1;
__REG32 GPIOInt3               : 1;
__REG32 GPIOInt4               : 1;
__REG32 GPIOInt5               : 1;
__REG32 GPIOInt6               : 1;
__REG32 GPIOInt7               : 1;
__REG32 GPIOInt8               : 1;
__REG32 GPIOInt9               : 1;
__REG32 GPIOInt10              : 1;
__REG32 GPIOInt11              : 1;
__REG32 GPIOInt12              : 1;
__REG32 GPIOInt13              : 1;
__REG32 GPIOInt14              : 1;
__REG32 GPIOInt15              : 1;
__REG32 GPIOInt16              : 1;
__REG32 GPIOInt17              : 1;
__REG32 GPIOInt18              : 1;
__REG32 GPIOInt19              : 1;
__REG32 GPIOInt20              : 1;
__REG32 GPIOInt21              : 1;
__REG32 GPIOInt22              : 1;
__REG32 GPIOInt23              : 1;
__REG32 GPIOInt24              : 1;
__REG32 GPIOInt25              : 1;
__REG32 GPIOInt26              : 1;
__REG32 GPIOInt27              : 1;
__REG32 GPIOInt28              : 1;
__REG32 GPIOInt29              : 1;
__REG32 GPIOInt30              : 1;
__REG32 GPIOInt31              : 1;
} __gpio_esicr_bits;

/* CPU0 GPIO Edge Sensitive Interrupt Mask Register */
typedef struct{
__REG32 GPIOIntEdgeMask0               : 1;
__REG32 GPIOIntEdgeMask1               : 1;
__REG32 GPIOIntEdgeMask2               : 1;
__REG32 GPIOIntEdgeMask3               : 1;
__REG32 GPIOIntEdgeMask4               : 1;
__REG32 GPIOIntEdgeMask5               : 1;
__REG32 GPIOIntEdgeMask6               : 1;
__REG32 GPIOIntEdgeMask7               : 1;
__REG32 GPIOIntEdgeMask8               : 1;
__REG32 GPIOIntEdgeMask9               : 1;
__REG32 GPIOIntEdgeMask10              : 1;
__REG32 GPIOIntEdgeMask11              : 1;
__REG32 GPIOIntEdgeMask12              : 1;
__REG32 GPIOIntEdgeMask13              : 1;
__REG32 GPIOIntEdgeMask14              : 1;
__REG32 GPIOIntEdgeMask15              : 1;
__REG32 GPIOIntEdgeMask16              : 1;
__REG32 GPIOIntEdgeMask17              : 1;
__REG32 GPIOIntEdgeMask18              : 1;
__REG32 GPIOIntEdgeMask19              : 1;
__REG32 GPIOIntEdgeMask20              : 1;
__REG32 GPIOIntEdgeMask21              : 1;
__REG32 GPIOIntEdgeMask22              : 1;
__REG32 GPIOIntEdgeMask23              : 1;
__REG32 GPIOIntEdgeMask24              : 1;
__REG32 GPIOIntEdgeMask25              : 1;
__REG32 GPIOIntEdgeMask26              : 1;
__REG32 GPIOIntEdgeMask27              : 1;
__REG32 GPIOIntEdgeMask28              : 1;
__REG32 GPIOIntEdgeMask29              : 1;
__REG32 GPIOIntEdgeMask30              : 1;
__REG32 GPIOIntEdgeMask31              : 1;
} __gpio_esimr_bits;

/* CPU0 GPIO Level Sensitive Interrupt Mask Register */
typedef struct{
__REG32 GPIOIntLevelMask0               : 1;
__REG32 GPIOIntLevelMask1               : 1;
__REG32 GPIOIntLevelMask2               : 1;
__REG32 GPIOIntLevelMask3               : 1;
__REG32 GPIOIntLevelMask4               : 1;
__REG32 GPIOIntLevelMask5               : 1;
__REG32 GPIOIntLevelMask6               : 1;
__REG32 GPIOIntLevelMask7               : 1;
__REG32 GPIOIntLevelMask8               : 1;
__REG32 GPIOIntLevelMask9               : 1;
__REG32 GPIOIntLevelMask10              : 1;
__REG32 GPIOIntLevelMask11              : 1;
__REG32 GPIOIntLevelMask12              : 1;
__REG32 GPIOIntLevelMask13              : 1;
__REG32 GPIOIntLevelMask14              : 1;
__REG32 GPIOIntLevelMask15              : 1;
__REG32 GPIOIntLevelMask16              : 1;
__REG32 GPIOIntLevelMask17              : 1;
__REG32 GPIOIntLevelMask18              : 1;
__REG32 GPIOIntLevelMask19              : 1;
__REG32 GPIOIntLevelMask20              : 1;
__REG32 GPIOIntLevelMask21              : 1;
__REG32 GPIOIntLevelMask22              : 1;
__REG32 GPIOIntLevelMask23              : 1;
__REG32 GPIOIntLevelMask24              : 1;
__REG32 GPIOIntLevelMask25              : 1;
__REG32 GPIOIntLevelMask26              : 1;
__REG32 GPIOIntLevelMask27              : 1;
__REG32 GPIOIntLevelMask28              : 1;
__REG32 GPIOIntLevelMask29              : 1;
__REG32 GPIOIntLevelMask30              : 1;
__REG32 GPIOIntLevelMask31              : 1;
} __gpio_lsimr_bits;

/* GPIO Data Out Set Register */
typedef struct{
__REG32 GPIODOutSet0               : 1;
__REG32 GPIODOutSet1               : 1;
__REG32 GPIODOutSet2               : 1;
__REG32 GPIODOutSet3               : 1;
__REG32 GPIODOutSet4               : 1;
__REG32 GPIODOutSet5               : 1;
__REG32 GPIODOutSet6               : 1;
__REG32 GPIODOutSet7               : 1;
__REG32 GPIODOutSet8               : 1;
__REG32 GPIODOutSet9               : 1;
__REG32 GPIODOutSet10              : 1;
__REG32 GPIODOutSet11              : 1;
__REG32 GPIODOutSet12              : 1;
__REG32 GPIODOutSet13              : 1;
__REG32 GPIODOutSet14              : 1;
__REG32 GPIODOutSet15              : 1;
__REG32 GPIODOutSet16              : 1;
__REG32 GPIODOutSet17              : 1;
__REG32 GPIODOutSet18              : 1;
__REG32 GPIODOutSet19              : 1;
__REG32 GPIODOutSet20              : 1;
__REG32 GPIODOutSet21              : 1;
__REG32 GPIODOutSet22              : 1;
__REG32 GPIODOutSet23              : 1;
__REG32 GPIODOutSet24              : 1;
__REG32 GPIODOutSet25              : 1;
__REG32 GPIODOutSet26              : 1;
__REG32 GPIODOutSet27              : 1;
__REG32 GPIODOutSet28              : 1;
__REG32 GPIODOutSet29              : 1;
__REG32 GPIODOutSet30              : 1;
__REG32 GPIODOutSet31              : 1;
} __gpio_dosr_bits;

/* GPIO Data Out Clear Register */
typedef struct{
__REG32 GPIODOutClear0               : 1;
__REG32 GPIODOutClear1               : 1;
__REG32 GPIODOutClear2               : 1;
__REG32 GPIODOutClear3               : 1;
__REG32 GPIODOutClear4               : 1;
__REG32 GPIODOutClear5               : 1;
__REG32 GPIODOutClear6               : 1;
__REG32 GPIODOutClear7               : 1;
__REG32 GPIODOutClear8               : 1;
__REG32 GPIODOutClear9               : 1;
__REG32 GPIODOutClear10              : 1;
__REG32 GPIODOutClear11              : 1;
__REG32 GPIODOutClear12              : 1;
__REG32 GPIODOutClear13              : 1;
__REG32 GPIODOutClear14              : 1;
__REG32 GPIODOutClear15              : 1;
__REG32 GPIODOutClear16              : 1;
__REG32 GPIODOutClear17              : 1;
__REG32 GPIODOutClear18              : 1;
__REG32 GPIODOutClear19              : 1;
__REG32 GPIODOutClear20              : 1;
__REG32 GPIODOutClear21              : 1;
__REG32 GPIODOutClear22              : 1;
__REG32 GPIODOutClear23              : 1;
__REG32 GPIODOutClear24              : 1;
__REG32 GPIODOutClear25              : 1;
__REG32 GPIODOutClear26              : 1;
__REG32 GPIODOutClear27              : 1;
__REG32 GPIODOutClear28              : 1;
__REG32 GPIODOutClear29              : 1;
__REG32 GPIODOutClear30              : 1;
__REG32 GPIODOutClear31              : 1;
} __gpio_docr_bits;

/* GPIO Control Set Register */
typedef struct{
__REG32 GPIODOutEnSet0               : 1;
__REG32 GPIODOutEnSet1               : 1;
__REG32 GPIODOutEnSet2               : 1;
__REG32 GPIODOutEnSet3               : 1;
__REG32 GPIODOutEnSet4               : 1;
__REG32 GPIODOutEnSet5               : 1;
__REG32 GPIODOutEnSet6               : 1;
__REG32 GPIODOutEnSet7               : 1;
__REG32 GPIODOutEnSet8               : 1;
__REG32 GPIODOutEnSet9               : 1;
__REG32 GPIODOutEnSet10              : 1;
__REG32 GPIODOutEnSet11              : 1;
__REG32 GPIODOutEnSet12              : 1;
__REG32 GPIODOutEnSet13              : 1;
__REG32 GPIODOutEnSet14              : 1;
__REG32 GPIODOutEnSet15              : 1;
__REG32 GPIODOutEnSet16              : 1;
__REG32 GPIODOutEnSet17              : 1;
__REG32 GPIODOutEnSet18              : 1;
__REG32 GPIODOutEnSet19              : 1;
__REG32 GPIODOutEnSet20              : 1;
__REG32 GPIODOutEnSet21              : 1;
__REG32 GPIODOutEnSet22              : 1;
__REG32 GPIODOutEnSet23              : 1;
__REG32 GPIODOutEnSet24              : 1;
__REG32 GPIODOutEnSet25              : 1;
__REG32 GPIODOutEnSet26              : 1;
__REG32 GPIODOutEnSet27              : 1;
__REG32 GPIODOutEnSet28              : 1;
__REG32 GPIODOutEnSet29              : 1;
__REG32 GPIODOutEnSet30              : 1;
__REG32 GPIODOutEnSet31              : 1;
} __gpio_csr_bits;

/* GPIO Control Clear Register */
typedef struct{
__REG32 GPIODOutEnClear0               : 1;
__REG32 GPIODOutEnClear1               : 1;
__REG32 GPIODOutEnClear2               : 1;
__REG32 GPIODOutEnClear3               : 1;
__REG32 GPIODOutEnClear4               : 1;
__REG32 GPIODOutEnClear5               : 1;
__REG32 GPIODOutEnClear6               : 1;
__REG32 GPIODOutEnClear7               : 1;
__REG32 GPIODOutEnClear8               : 1;
__REG32 GPIODOutEnClear9               : 1;
__REG32 GPIODOutEnClear10              : 1;
__REG32 GPIODOutEnClear11              : 1;
__REG32 GPIODOutEnClear12              : 1;
__REG32 GPIODOutEnClear13              : 1;
__REG32 GPIODOutEnClear14              : 1;
__REG32 GPIODOutEnClear15              : 1;
__REG32 GPIODOutEnClear16              : 1;
__REG32 GPIODOutEnClear17              : 1;
__REG32 GPIODOutEnClear18              : 1;
__REG32 GPIODOutEnClear19              : 1;
__REG32 GPIODOutEnClear20              : 1;
__REG32 GPIODOutEnClear21              : 1;
__REG32 GPIODOutEnClear22              : 1;
__REG32 GPIODOutEnClear23              : 1;
__REG32 GPIODOutEnClear24              : 1;
__REG32 GPIODOutEnClear25              : 1;
__REG32 GPIODOutEnClear26              : 1;
__REG32 GPIODOutEnClear27              : 1;
__REG32 GPIODOutEnClear28              : 1;
__REG32 GPIODOutEnClear29              : 1;
__REG32 GPIODOutEnClear30              : 1;
__REG32 GPIODOutEnClear31              : 1;
} __gpio_ccr_bits;

/* Main Interrupt Error Cause Register */
typedef struct{
__REG32 CryptErr            : 1;
__REG32 DevErr              : 1;
__REG32 DMAErr              : 1;
__REG32 CPUErr              : 1;
__REG32 PEX0Err             : 1;
__REG32 PEX1Err             : 1;
__REG32 GbEErr              : 1;
__REG32                     : 1;
__REG32 USBErr              : 1;
__REG32 DRAMErr             : 1;
__REG32 XORErr              : 1;
__REG32 L2cCorrectableErr   : 1;
__REG32 L2cUnCorrectableErr : 1;
__REG32 L1CDataParErr       : 1;
__REG32 L2CTagParErr        : 1;
__REG32 WD                  : 1;
__REG32 L2CPerfCntOverflow  : 1;
__REG32                     :15;
} __ic_miecr_bits;

/* Main Interrupt Cause ( Low ) Register */
typedef struct{
__REG32 ErrSum              : 1;
__REG32 SPI                 : 1;
__REG32 TWSI0               : 1;
__REG32 TWSI1               : 1;
__REG32 IDMA0               : 1;
__REG32 IDMA1               : 1;
__REG32 IDMA2               : 1;
__REG32 IDMA3               : 1;
__REG32 Timer0              : 1;
__REG32 Timer1              : 1;
__REG32 Timer2              : 1;
__REG32 Timer3              : 1;
__REG32 UART0               : 1;
__REG32 UART1               : 1;
__REG32 UART2               : 1;
__REG32 UART3               : 1;
__REG32 USB0                : 1;
__REG32 USB1                : 1;
__REG32 USB2                : 1;
__REG32 Crypto              : 1;
__REG32                     : 2;
__REG32 XOR0                : 1;
__REG32 XOR1                : 1;
__REG32                     : 2;
__REG32 SATA                : 1;
__REG32 TDMI_INT            : 1;
__REG32                     : 4;
} __ic_miclr_bits;

/* Main Interrupt Cause ( High ) Register */
typedef struct{
__REG32 PEX00INTA           : 1;
__REG32 PEX01INTA           : 1;
__REG32 PEX02INTA           : 1;
__REG32 PEX03INTA           : 1;
__REG32 PEX10INTA           : 1;
__REG32 PEX11INTA           : 1;
__REG32 PEX12INTA           : 1;
__REG32 PEX13INTA           : 1;
__REG32 GE00Sum             : 1;
__REG32 GE00Rx              : 1;
__REG32 GE00Tx              : 1;
__REG32 GE00Misc            : 1;
__REG32 GE01Sum             : 1;
__REG32 GE01Rx              : 1;
__REG32 GE01Tx              : 1;
__REG32 GE01Misc            : 1;
__REG32 GE10Sum             : 1;
__REG32 GE10Rx              : 1;
__REG32 GE10Tx              : 1;
__REG32 GE10Misc            : 1;
__REG32 GE11Sum             : 1;
__REG32 GE11Rx              : 1;
__REG32 GE11Tx              : 1;
__REG32 GE11Misc            : 1;
__REG32 GPIO0_7             : 1;
__REG32 GPIO8_15            : 1;
__REG32 GPIO16_23           : 1;
__REG32 GPIO24_31           : 1;
__REG32 DB_IN               : 1;
__REG32 DB_OUT              : 1;
__REG32                     : 2;
} __ic_michr_bits;

/* IRQ Select Cause Register */
typedef struct{
__REG32 PEX00INTA           : 1;
__REG32 PEX01INTA           : 1;
__REG32 PEX02INTA           : 1;
__REG32 PEX03INTA           : 1;
__REG32 PEX10INTA           : 1;
__REG32 PEX11INTA           : 1;
__REG32 PEX12INTA           : 1;
__REG32 PEX13INTA           : 1;
__REG32 GE00Sum             : 1;
__REG32 GE00Rx              : 1;
__REG32 GE00Tx              : 1;
__REG32 GE00Misc            : 1;
__REG32 GE01Sum             : 1;
__REG32 GE01Rx              : 1;
__REG32 GE01Tx              : 1;
__REG32 GE01Misc            : 1;
__REG32 GE10Sum             : 1;
__REG32 GE10Rx              : 1;
__REG32 GE10Tx              : 1;
__REG32 GE10Misc            : 1;
__REG32 GE11Sum             : 1;
__REG32 GE11Rx              : 1;
__REG32 GE11Tx              : 1;
__REG32 GE11Misc            : 1;
__REG32 GPIO0_7             : 1;
__REG32 GPIO8_15            : 1;
__REG32 GPIO16_23           : 1;
__REG32 GPIO24_31           : 1;
__REG32 DB_IN               : 1;
__REG32 DB_OUT              : 1;
__REG32 Sel                 : 1;
__REG32 Stat                : 1;
} __ic_irqscr_bits;

/* MPP Control 0 Register */
typedef struct{
__REG32 MPPSel0             : 4;
__REG32 MPPSel1             : 4;
__REG32 MPPSel2             : 4;
__REG32 MPPSel3             : 4;
__REG32 MPPSel4             : 4;
__REG32 MPPSel5             : 4;
__REG32 MPPSel6             : 4;
__REG32 MPPSel7             : 4;
} __mpp_c0r_bits;

/* MPP Control 1 Register */
typedef struct{
__REG32 MPPSel8             : 4;
__REG32 MPPSel9             : 4;
__REG32 MPPSel10            : 4;
__REG32 MPPSel11            : 4;
__REG32 MPPSel12            : 4;
__REG32 MPPSel13            : 4;
__REG32 MPPSel14            : 4;
__REG32 MPPSel15            : 4;
} __mpp_c1r_bits;

/* MPP Control 2 Register */
typedef struct{
__REG32 MPPSel16            : 4;
__REG32 MPPSel17            : 4;
__REG32 MPPSel18            : 4;
__REG32 MPPSel19            : 4;
__REG32 MPPSel20            : 4;
__REG32 MPPSel21            : 4;
__REG32 MPPSel22            : 4;
__REG32 MPPSel23            : 4;
} __mpp_c2r_bits;

/* MPP Control 3 Register */
typedef struct{
__REG32 MPPSel24            : 4;
__REG32 MPPSel25            : 4;
__REG32 MPPSel26            : 4;
__REG32 MPPSel27            : 4;
__REG32 MPPSel28            : 4;
__REG32 MPPSel29            : 4;
__REG32 MPPSel30            : 4;
__REG32 MPPSel31            : 4;
} __mpp_c3r_bits;

/* MPP Control 4 Register */
typedef struct{
__REG32 MPPSel32            : 4;
__REG32 MPPSel33            : 4;
__REG32 MPPSel34            : 4;
__REG32 MPPSel35            : 4;
__REG32 MPPSel36            : 4;
__REG32 MPPSel37            : 4;
__REG32 MPPSel38            : 4;
__REG32 MPPSel39            : 4;
} __mpp_c4r_bits;

/* MPP Control 5 Register */
typedef struct{
__REG32 MPPSel40            : 4;
__REG32 MPPSel41            : 4;
__REG32 MPPSel42            : 4;
__REG32 MPPSel43            : 4;
__REG32 MPPSel44            : 4;
__REG32 MPPSel45            : 4;
__REG32 MPPSel46            : 4;
__REG32 MPPSel47            : 4;
} __mpp_c5r_bits;

/* MPP Control 6 Register */
typedef struct{
__REG32 MPPSel48            : 4;
__REG32 MPPSel49            : 4;
__REG32                     :24;
} __mpp_c6r_bits;

/* Segment<n> Pads Calibration Register (n=0 6) */
typedef struct{
__REG32                     :12;
__REG32 DrvP                : 4;
__REG32 DrvN                : 4;
__REG32                     :12;
} __spcr_bits;

/* Sample at Reset ( high ) Register */
typedef struct{
__REG32 DEV_ALE             : 2;
__REG32 DEV_WEn             : 4;
__REG32 DEV_A               : 3;
__REG32 GE0_TXD             : 4;
__REG32 GE0_TXCTL           : 1;
__REG32                     :18;
} __srhr_bits;

/* RSTOUTn Mask Register */
typedef struct{
__REG32 PexRstOutEn         : 1;
__REG32 WDRstOutEn          : 1;
__REG32 SoftRstOutEn        : 1;
__REG32                     :29;
} __rstoutmr_bits;

/* System Soft Reset Register */
typedef struct{
__REG32 SystemSoftRst       : 1;
__REG32                     :31;
} __srlr_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */

/* Declarations common to compiler and assembler **************************/

/***************************************************************************
 **
 ** CPU AMR
 **
 ***************************************************************************/
__IO_REG32_BIT(CPU_AMR_W0CR,          0xD0020000,__READ_WRITE ,__amr_wcr_bits);
__IO_REG32(		 CPU_AMR_W0BR,          0xD0020004,__READ_WRITE );
__IO_REG32(		 CPU_AMR_W0RLR,         0xD0020008,__READ_WRITE );
__IO_REG32(		 CPU_AMR_W0RHR,         0xD002000C,__READ_WRITE );
__IO_REG32_BIT(CPU_AMR_W1CR,          0xD0020010,__READ_WRITE ,__amr_wcr_bits);
__IO_REG32(		 CPU_AMR_W1BR,          0xD0020014,__READ_WRITE );
__IO_REG32(		 CPU_AMR_W1RLR,         0xD0020018,__READ_WRITE );
__IO_REG32(		 CPU_AMR_W1RHR,         0xD002001C,__READ_WRITE );
__IO_REG32_BIT(CPU_AMR_W2CR,          0xD0020020,__READ_WRITE ,__amr_wcr_bits);
__IO_REG32(		 CPU_AMR_W2BR,          0xD0020024,__READ_WRITE );
__IO_REG32(		 CPU_AMR_W2RLR,         0xD0020028,__READ_WRITE );
__IO_REG32(		 CPU_AMR_W2RHR,         0xD002002C,__READ_WRITE );
__IO_REG32_BIT(CPU_AMR_W3CR,          0xD0020030,__READ_WRITE ,__amr_wcr_bits);
__IO_REG32(		 CPU_AMR_W3BR,          0xD0020034,__READ_WRITE );
__IO_REG32(		 CPU_AMR_W3RLR,         0xD0020038,__READ_WRITE );
__IO_REG32(		 CPU_AMR_W3RHR,         0xD002003C,__READ_WRITE );
__IO_REG32_BIT(CPU_AMR_W4CR,          0xD0020040,__READ_WRITE ,__amr_wcr_bits);
__IO_REG32(		 CPU_AMR_W4BR,          0xD0020044,__READ_WRITE );
__IO_REG32(		 CPU_AMR_W4RLR,         0xD0020048,__READ_WRITE );
__IO_REG32(		 CPU_AMR_W4RHR,         0xD002004C,__READ_WRITE );
__IO_REG32_BIT(CPU_AMR_W5CR,          0xD0020050,__READ_WRITE ,__amr_wcr_bits);
__IO_REG32(		 CPU_AMR_W5BR,          0xD0020054,__READ_WRITE );
__IO_REG32(		 CPU_AMR_W5RLR,         0xD0020058,__READ_WRITE );
__IO_REG32(		 CPU_AMR_W5RHR,         0xD002005C,__READ_WRITE );
__IO_REG32_BIT(CPU_AMR_W6CR,          0xD0020060,__READ_WRITE ,__amr_w6cr_bits);
__IO_REG32(		 CPU_AMR_W6BR,          0xD0020064,__READ_WRITE );
__IO_REG32(		 CPU_AMR_W6RLR,         0xD0020068,__READ_WRITE );
__IO_REG32(		 CPU_AMR_W6RHR,         0xD002006C,__READ_WRITE );
__IO_REG32_BIT(CPU_AMR_W7CR,          0xD0020070,__READ_WRITE ,__amr_w6cr_bits);
__IO_REG32(		 CPU_AMR_W7BR,          0xD0020074,__READ_WRITE );
__IO_REG32(		 CPU_AMR_W7RLR,         0xD0020078,__READ_WRITE );
__IO_REG32(		 CPU_AMR_W7RHR,         0xD002007C,__READ_WRITE );
__IO_REG32(		 CPU_AMR_IRBA,          0xD0020080,__READ_WRITE );
__IO_REG32_BIT(CPU_AMR_W8CR,          0xD0020900,__READ_WRITE ,__amr_wcr_bits);
__IO_REG32(		 CPU_AMR_W8BR,          0xD0020904,__READ_WRITE );
__IO_REG32_BIT(CPU_AMR_W9CR,          0xD0020910,__READ_WRITE ,__amr_wcr_bits);
__IO_REG32(		 CPU_AMR_W9BR,          0xD0020914,__READ_WRITE );
__IO_REG32_BIT(CPU_AMR_W10CR,         0xD0020920,__READ_WRITE ,__amr_wcr_bits);
__IO_REG32(		 CPU_AMR_W10BR,         0xD0020924,__READ_WRITE );
__IO_REG32_BIT(CPU_AMR_W11CR,         0xD0020930,__READ_WRITE ,__amr_wcr_bits);
__IO_REG32(		 CPU_AMR_W11BR,         0xD0020934,__READ_WRITE );
__IO_REG32_BIT(CPU_AMR_W12CR,         0xD0020940,__READ_WRITE ,__amr_wcr_bits);
__IO_REG32(		 CPU_AMR_W12BR,         0xD0020944,__READ_WRITE );
__IO_REG32_BIT(CPU_AMR_W13CR,         0xD0020950,__READ_WRITE ,__amr_wcr_bits);
__IO_REG32(		 CPU_AMR_W13BR,         0xD0020954,__READ_WRITE );

/***************************************************************************
 **
 ** CPU CSR
 **
 ***************************************************************************/
__IO_REG32_BIT(CPU_CSR_CR,            0xD0020100,__READ_WRITE ,__csr_cr_bits);
__IO_REG32_BIT(CPU_CSR_CSR,           0xD0020104,__READ_WRITE ,__csr_csr_bits);
__IO_REG32_BIT(CPU_CSR_M2MICR,        0xD0020110,__READ_WRITE ,__csr_m2micr_bits);
__IO_REG32_BIT(CPU_CSR_M2MIMR,        0xD0020114,__READ_WRITE ,__csr_m2mimr_bits);
__IO_REG32_BIT(CPU_CSR_PMCR,          0xD002011C,__READ_WRITE ,__csr_pmcr_bits);
__IO_REG32_BIT(CPU_CSR_TAR,           0xD0020120,__READ_WRITE ,__csr_tar_bits);
__IO_REG32(    CPU_CSR_L1RT0R,        0xD0020128,__READ_WRITE );
__IO_REG32(    CPU_CSR_L1RT1R,        0xD002012C,__READ_WRITE );
__IO_REG32_BIT(CPU_CSR_MTOR,          0xD0020130,__READ_WRITE ,__csr_mtor_bits);
__IO_REG32(    CPU_CSR_L2RT0R,        0xD0020134,__READ_WRITE );
__IO_REG32(    CPU_CSR_L2RT1R,        0xD0020138,__READ_WRITE );
__IO_REG32_BIT(CPU_CSR_MPMCR,         0xD0020140,__READ_WRITE ,__csr_mpmcr_bits);
__IO_REG32_BIT(CPU_CSR_L2PMCR,        0xD0020144,__READ_WRITE ,__csr_l2pmcr_bits);

/***************************************************************************
 **
 ** CPU Doorbell
 **
 ***************************************************************************/
__IO_REG32_BIT(CPU_DB_ID,             0xD0020400,__READ_WRITE ,__db_id_bits);
__IO_REG32_BIT(CPU_DB_IDM,            0xD0020404,__READ_WRITE ,__db_idm_bits);
__IO_REG32_BIT(CPU_DB_OD,             0xD0020408,__READ_WRITE ,__db_od_bits);
__IO_REG32_BIT(CPU_DB_ODM,            0xD002040C,__READ_WRITE ,__db_odm_bits);

/***************************************************************************
 **
 ** CPU Semaphores
 **
 ***************************************************************************/
__IO_REG32_BIT(CPU_SEMAPH0,           0xD0020500,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH1,           0xD0020504,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH2,           0xD0020508,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH3,           0xD002050C,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH4,           0xD0020510,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH5,           0xD0020514,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH6,           0xD0020518,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH7,           0xD002051C,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH8,           0xD0020520,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH9,           0xD0020524,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH10,          0xD0020528,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH11,          0xD002052C,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH12,          0xD0020530,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH13,          0xD0020534,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH14,          0xD0020538,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH15,          0xD002053C,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH16,          0xD0020540,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH17,          0xD0020544,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH18,          0xD0020548,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH19,          0xD002054C,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH20,          0xD0020550,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH21,          0xD0020554,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH22,          0xD0020558,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH23,          0xD002055C,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH24,          0xD0020560,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH25,          0xD0020564,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH26,          0xD0020568,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH27,          0xD002056C,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH28,          0xD0020570,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH29,          0xD0020574,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH30,          0xD0020578,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH31,          0xD002057C,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH32,          0xD0020580,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH33,          0xD0020584,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH34,          0xD0020588,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH35,          0xD002058C,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH36,          0xD0020590,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH37,          0xD0020594,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH38,          0xD0020598,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH39,          0xD002059C,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH40,          0xD00205A0,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH41,          0xD00205A4,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH42,          0xD00205A8,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH43,          0xD00205AC,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH44,          0xD00205B0,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH45,          0xD00205B4,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH46,          0xD00205B8,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH47,          0xD00205BC,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH48,          0xD00205C0,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH49,          0xD00205C4,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH50,          0xD00205C8,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH51,          0xD00205CC,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH52,          0xD00205D0,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH53,          0xD00205D4,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH54,          0xD00205D8,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH55,          0xD00205DC,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH56,          0xD00205E0,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH57,          0xD00205E4,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH58,          0xD00205E8,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH59,          0xD00205EC,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH60,          0xD00205F0,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH61,          0xD00205F4,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH62,          0xD00205F8,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH63,          0xD00205FC,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH64,          0xD0020600,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH65,          0xD0020604,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH66,          0xD0020608,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH67,          0xD002060C,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH68,          0xD0020610,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH69,          0xD0020614,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH70,          0xD0020618,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH71,          0xD002061C,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH72,          0xD0020620,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH73,          0xD0020624,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH74,          0xD0020628,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH75,          0xD002062C,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH76,          0xD0020630,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH77,          0xD0020634,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH78,          0xD0020638,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH79,          0xD002063C,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH80,          0xD0020640,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH81,          0xD0020644,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH82,          0xD0020648,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH83,          0xD002064C,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH84,          0xD0020650,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH85,          0xD0020654,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH86,          0xD0020658,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH87,          0xD002065C,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH88,          0xD0020660,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH89,          0xD0020664,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH90,          0xD0020668,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH91,          0xD002066C,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH92,          0xD0020670,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH93,          0xD0020674,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH94,          0xD0020678,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH95,          0xD002067C,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH96,          0xD0020680,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH97,          0xD0020684,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH98,          0xD0020688,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH99,          0xD002068C,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH100,         0xD0020690,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH101,         0xD0020694,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH102,         0xD0020698,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH103,         0xD002069C,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH104,         0xD00206A0,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH105,         0xD00206A4,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH106,         0xD00206A8,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH107,         0xD00206AC,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH108,         0xD00206B0,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH109,         0xD00206B4,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH110,         0xD00206B8,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH111,         0xD00206BC,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH112,         0xD00206C0,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH113,         0xD00206C4,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH114,         0xD00206C8,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH115,         0xD00206CC,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH116,         0xD00206D0,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH117,         0xD00206D4,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH118,         0xD00206D8,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH119,         0xD00206DC,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH120,         0xD00206E0,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH121,         0xD00206E4,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH122,         0xD00206E8,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH123,         0xD00206EC,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH124,         0xD00206F0,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH125,         0xD00206F4,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH126,         0xD00206F8,__READ       ,__semaph_bits);
__IO_REG32_BIT(CPU_SEMAPH127,         0xD00206FC,__READ       ,__semaph_bits);

/***************************************************************************
 **
 ** CPU L2 Non-Cache Address Map
 **
 ***************************************************************************/
__IO_REG32_BIT(CPU_L2NCAM_W0BAR,      0xD0020A00,__READ_WRITE ,__l2ncam_wbar_bits);
__IO_REG32_BIT(CPU_L2NCAM_W0SAR,      0xD0020A04,__READ_WRITE ,__l2ncam_wsar_bits);
__IO_REG32_BIT(CPU_L2NCAM_W1BAR,      0xD0020A08,__READ_WRITE ,__l2ncam_wbar_bits);
__IO_REG32_BIT(CPU_L2NCAM_W1SAR,      0xD0020A0C,__READ_WRITE ,__l2ncam_wsar_bits);
__IO_REG32_BIT(CPU_L2NCAM_W2BAR,      0xD0020A10,__READ_WRITE ,__l2ncam_wbar_bits);
__IO_REG32_BIT(CPU_L2NCAM_W2SAR,      0xD0020A14,__READ_WRITE ,__l2ncam_wsar_bits);
__IO_REG32_BIT(CPU_L2NCAM_W3BAR,      0xD0020A18,__READ_WRITE ,__l2ncam_wbar_bits);
__IO_REG32_BIT(CPU_L2NCAM_W3SAR,      0xD0020A1C,__READ_WRITE ,__l2ncam_wsar_bits);

/***************************************************************************
 **
 ** DDR SDRMA
 **
 ***************************************************************************/
__IO_REG32(    DDR_CSW0BAR,           0xD0001500,__READ_WRITE );
__IO_REG32_BIT(DDR_CSW0SAR,           0xD0001504,__READ_WRITE ,__ddr_cswsar_bits);
__IO_REG32(    DDR_CSW1BAR,           0xD0001508,__READ_WRITE );
__IO_REG32_BIT(DDR_CSW1SAR,           0xD000150C,__READ_WRITE ,__ddr_cswsar_bits);
__IO_REG32(    DDR_CSW2BAR,           0xD0001510,__READ_WRITE );
__IO_REG32_BIT(DDR_CSW2SAR,           0xD0001514,__READ_WRITE ,__ddr_cswsar_bits);
__IO_REG32(    DDR_CSW3BAR,           0xD0001518,__READ_WRITE );
__IO_REG32_BIT(DDR_CSW3SAR,           0xD000151C,__READ_WRITE ,__ddr_cswsar_bits);
__IO_REG32_BIT(DDR_SDRAM_CR,          0xD0001400,__READ_WRITE ,__ddr_sdram_cr_bits);
__IO_REG32_BIT(DDR_CCLR,              0xD0001404,__READ_WRITE ,__ddr_cclr_bits);
__IO_REG32_BIT(DDR_SDRAM_TLR,         0xD0001408,__READ_WRITE ,__ddr_sdram_tlr_bits);
__IO_REG32_BIT(DDR_SDRAM_THR,         0xD000140C,__READ_WRITE ,__ddr_sdram_thr_bits);
__IO_REG32_BIT(DDR_SDRAM_ACR,         0xD0001410,__READ_WRITE ,__ddr_sdram_acr_bits);
__IO_REG32_BIT(DDR_SDRAM_OPCR,        0xD0001414,__READ_WRITE ,__ddr_sdram_opcr_bits);
__IO_REG32_BIT(DDR_SDRAM_OR,          0xD0001418,__READ_WRITE ,__ddr_sdram_or_bits);
__IO_REG32_BIT(DDR_SDRAM_MR,          0xD000141C,__READ_WRITE ,__ddr_sdram_mr_bits);
__IO_REG32_BIT(DDR_DRAM_EMR,          0xD0001420,__READ_WRITE ,__ddr_dram_emr_bits);
__IO_REG32_BIT(DDR_CCHR,              0xD0001424,__READ_WRITE ,__ddr_cchr_bits);
__IO_REG32_BIT(DDR_DDR2_TLR,          0xD0001428,__READ_WRITE ,__ddr_ddr2_tlr_bits);
__IO_REG32_BIT(DDR_SDRAM_OCR,         0xD000142C,__READ_WRITE ,__ddr_sdram_ocr_bits);
__IO_REG32_BIT(DDR_SDRAM_IMC0LR,      0xD0001430,__READ_WRITE ,__ddr_sdram_imc0lr_bits);
__IO_REG32_BIT(DDR_SDRAM_IMC0HR,      0xD0001434,__READ_WRITE ,__ddr_sdram_imc0hr_bits);
__IO_REG32_BIT(DDR_SDRAM_IMTOR,       0xD0001438,__READ_WRITE ,__ddr_sdram_imtor_bits);
__IO_REG32_BIT(DDR_DDR2_THR,          0xD000147C,__READ_WRITE ,__ddr_ddr2_thr_bits);
__IO_REG32_BIT(DDR_SDRAM_ICR,         0xD0001480,__READ_WRITE ,__ddr_sdram_icr_bits);
__IO_REG32_BIT(DDR_SDRAM_FTDLL_LCR,   0xD0001484,__READ_WRITE ,__ddr_sdram_ftdll_lcr_bits);
__IO_REG32_BIT(DDR_DRAM_EM2R,         0xD000148C,__READ_WRITE ,__ddr_dram_em2r_bits);
__IO_REG32_BIT(DDR_DRAM_EM3R,         0xD0001490,__READ_WRITE ,__ddr_dram_em3r_bits);
__IO_REG32_BIT(DDR_SDRAM_ODT_CLR,     0xD0001494,__READ_WRITE ,__ddr_sdram_odt_clr_bits);
__IO_REG32_BIT(DDR_SDRAM_ODT_CHR,     0xD0001498,__READ_WRITE ,__ddr_sdram_odt_chr_bits);
__IO_REG32_BIT(DDR_SDRAM_ODT_CR,      0xD000149C,__READ_WRITE ,__ddr_sdram_odt_cr_bits);
__IO_REG32_BIT(DDR_RBSR,              0xD00014A4,__READ_WRITE ,__ddr_rbsr_bits);
__IO_REG32_BIT(DDR_SDRAM_ACPCR,       0xD00014C0,__READ_WRITE ,__ddr_sdram_acpcr_bits);
__IO_REG32_BIT(DDR_SDRAM_DQPCR,       0xD00014C4,__READ_WRITE ,__ddr_sdram_acpcr_bits);
__IO_REG32_BIT(DDR_SDRAM_MC1LR,       0xD00014F4,__READ_WRITE ,__ddr_sdram_mc1lr_bits);
__IO_REG32_BIT(DDR_SDRAM_MC1HR,       0xD00014F8,__READ_WRITE ,__ddr_sdram_mc1hr_bits);
__IO_REG32_BIT(DDR_SDRAM_FTDLL_RCR,   0xD000161C,__READ_WRITE ,__ddr_sdram_ftdll_lcr_bits);
__IO_REG32_BIT(DDR_SDRAM_FTDLL_UCR,   0xD0001620,__READ_WRITE ,__ddr_sdram_ftdll_lcr_bits);
__IO_REG32_BIT(DDR_FC_FTDLL_CR,       0xD0001624,__READ_WRITE ,__ddr_fc_ftdll_cr_bits);
__IO_REG32_BIT(DDR_DCPUAR,            0xD0001628,__READ_WRITE ,__ddr_dcpuar_bits);
__IO_REG32(    DDR_SDARM_ERRDHR,      0xD0001440,__READ_WRITE );
__IO_REG32(    DDR_SDARM_ERRDLR,      0xD0001444,__READ_WRITE );
__IO_REG32_BIT(DDR_SDARM_RECCR,       0xD0001448,__READ_WRITE ,__ddr_sdarm_reccr_bits);
__IO_REG32_BIT(DDR_SDARM_CECCR,       0xD000144C,__READ_WRITE ,__ddr_sdarm_ceccr_bits);
__IO_REG32_BIT(DDR_SDARM_ERRAR,       0xD0001450,__READ_WRITE ,__ddr_sdarm_errar_bits);
__IO_REG32_BIT(DDR_SDARM_ECCCR,       0xD0001454,__READ_WRITE ,__ddr_sdarm_ecccr_bits);
__IO_REG32(    DDR_SBERRCR,           0xD0001458,__READ_WRITE );
__IO_REG32(    DDR_DBERRCR,           0xD000145C,__READ_WRITE );
__IO_REG32_BIT(DDR_CICR,              0xD00014D0,__READ_WRITE ,__ddr_cicr_bits);
__IO_REG32_BIT(DDR_CIMR,              0xD00014D4,__READ_WRITE ,__ddr_cimr_bits);

/***************************************************************************
 **
 ** Device Bus Controller
 **
 ***************************************************************************/
__IO_REG32_BIT(DEV_RDBOOTCSn,         0xD0010400,__READ_WRITE ,__dev_rdbootcsn_bits);
__IO_REG32_BIT(DEV_WRBOOTCSn,         0xD0010404,__READ_WRITE ,__dev_wrbootcsn_bits);
__IO_REG32_BIT(DEV_RDCSn0,            0xD0010408,__READ_WRITE ,__dev_rdcsn_bits);
__IO_REG32_BIT(DEV_RDCSn1,            0xD0010410,__READ_WRITE ,__dev_rdcsn_bits);
__IO_REG32_BIT(DEV_RDCSn2,            0xD0010418,__READ_WRITE ,__dev_rdcsn_bits);
__IO_REG32_BIT(DEV_RDCSn3,            0xD0010420,__READ_WRITE ,__dev_rdcsn_bits);
__IO_REG32_BIT(DEV_WRCSn0,            0xD001040C,__READ_WRITE ,__dev_wrcsn_bits);
__IO_REG32_BIT(DEV_WRCSn1,            0xD0010414,__READ_WRITE ,__dev_wrcsn_bits);
__IO_REG32_BIT(DEV_WRCSn2,            0xD001041C,__READ_WRITE ,__dev_wrcsn_bits);
__IO_REG32_BIT(DEV_WRCSn3,            0xD0010424,__READ_WRITE ,__dev_wrcsn_bits);
__IO_REG32_BIT(DEV_NANDCR,            0xD0010470,__READ_WRITE ,__dev_nandcr_bits);
__IO_REG32_BIT(DEV_IFCR,              0xD00104C0,__READ_WRITE ,__dev_ifcr_bits);
__IO_REG32(    DEV_ERRAR,             0xD00104C4,__READ_WRITE );
__IO_REG32_BIT(DEV_SCR,               0xD00104C8,__READ_WRITE ,__dev_scr_bits);
__IO_REG32_BIT(DEV_ICR,               0xD00104D0,__READ_WRITE ,__dev_icr_bits);
__IO_REG32_BIT(DEV_IMCR,              0xD00104D4,__READ_WRITE ,__dev_imcr_bits);

/***************************************************************************
 **
 ** PCIE 0
 **
 ***************************************************************************/
__IO_REG32_BIT(PEX00_W0CR,            0xD0041820,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX00_W0BR,            0xD0041824,__READ_WRITE );
__IO_REG32_BIT(PEX00_W0RR,            0xD004182C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32_BIT(PEX00_W1CR,            0xD0041830,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX00_W1BR,            0xD0041834,__READ_WRITE );
__IO_REG32_BIT(PEX00_W1RR,            0xD004183C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32_BIT(PEX00_W2CR,            0xD0041840,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX00_W2BR,            0xD0041844,__READ_WRITE );
__IO_REG32_BIT(PEX00_W2RR,            0xD004184C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32_BIT(PEX00_W3CR,            0xD0041850,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX00_W3BR,            0xD0041854,__READ_WRITE );
__IO_REG32_BIT(PEX00_W3RR,            0xD004185C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32_BIT(PEX00_W4CR,            0xD0041860,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX00_W4BR,            0xD0041864,__READ_WRITE );
__IO_REG32_BIT(PEX00_W4RR,            0xD004186C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32(    PEX00_W4RHR,           0xD0041870,__READ_WRITE );
__IO_REG32_BIT(PEX00_W5CR,            0xD0041880,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX00_W5BR,            0xD0041884,__READ_WRITE );
__IO_REG32_BIT(PEX00_W5RR,            0xD004188C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32(    PEX00_W5RHR,           0xD0041890,__READ_WRITE );
__IO_REG32_BIT(PEX00_DWCR,            0xD00418B0,__READ_WRITE ,__pex_dwcr_bits);
__IO_REG32_BIT(PEX00_EROMWCR,         0xD00418C0,__READ_WRITE ,__pex_eromwcr_bits);
__IO_REG32_BIT(PEX00_EROMWRR,         0xD00418C4,__READ_WRITE ,__pex_eromwrr_bits);
__IO_REG32_BIT(PEX00_BAR1CR,          0xD0041804,__READ_WRITE ,__pex_barcr_bits);
__IO_REG32_BIT(PEX00_BAR2CR,          0xD0041808,__READ_WRITE ,__pex_barcr_bits);
__IO_REG32_BIT(PEX00_EROMBARCR,       0xD004180C,__READ_WRITE ,__pex_erombarcr_bits);
__IO_REG32_BIT(PEX00_CAR,             0xD00418F8,__READ_WRITE ,__pex_car_bits);
__IO_REG32(    PEX00_CDR,             0xD00418FC,__READ_WRITE );
__IO_REG32_BIT(PEX00_DVIDR,           0xD0040000,__READ       ,__pex_dvidr_bits);
__IO_REG32_BIT(PEX00_CSR,             0xD0040004,__READ_WRITE ,__pex_csr_bits);
__IO_REG32_BIT(PEX00_CCRIDR,          0xD0040008,__READ       ,__pex_ccridr_bits);
__IO_REG32_BIT(PEX00_BISTHTCLSR,      0xD004000C,__READ_WRITE ,__pex_bisthtclsr_bits);
__IO_REG32_BIT(PEX00_BAR0IR,          0xD0040010,__READ_WRITE ,__pex_barir_bits);
__IO_REG32(    PEX00_BAR0IHR,         0xD0040014,__READ_WRITE );
__IO_REG32_BIT(PEX00_BAR1IR,          0xD0040018,__READ_WRITE ,__pex_barir_bits);
__IO_REG32(    PEX00_BAR1IHR,         0xD004001C,__READ_WRITE );
__IO_REG32_BIT(PEX00_BAR2IR,          0xD0040020,__READ_WRITE ,__pex_barir_bits);
__IO_REG32(    PEX00_BAR2IHR,         0xD0040024,__READ_WRITE );
__IO_REG32_BIT(PEX00_SSDVIDR,         0xD004002C,__READ       ,__pex_ssdvidr_bits);
__IO_REG32_BIT(PEX00_EROMBARR,        0xD0040030,__READ_WRITE ,__pex_erombarr_bits);
__IO_REG32_BIT(PEX00_CLPR,            0xD0040034,__READ       ,__pex_clpr_bits);
__IO_REG32_BIT(PEX00_IPLR,            0xD004003C,__READ_WRITE ,__pex_iplr_bits);
__IO_REG32_BIT(PEX00_PMCHR,           0xD0040040,__READ       ,__pex_pmchr_bits);
__IO_REG32_BIT(PEX00_PMCSR,           0xD0040044,__READ_WRITE ,__pex_pmcsr_bits);
__IO_REG32_BIT(PEX00_MSIMCR,          0xD0040050,__READ_WRITE ,__pex_msimcr_bits);
__IO_REG32(    PEX00_MSIMAR,          0xD0040054,__READ_WRITE );
__IO_REG32(    PEX00_MSIMAHR,         0xD0040058,__READ_WRITE );
__IO_REG32_BIT(PEX00_MSIMDR,          0xD004005C,__READ_WRITE ,__pex_msimdr_bits);
__IO_REG32_BIT(PEX00_CR,              0xD0040060,__READ       ,__pex_cr_bits);
__IO_REG32_BIT(PEX00_DCR,             0xD0040064,__READ       ,__pex_dcr_bits);
__IO_REG32_BIT(PEX00_DCSR,            0xD0040068,__READ_WRITE ,__pex_dcsr_bits);
__IO_REG32_BIT(PEX00_LCR,             0xD004006C,__READ       ,__pex_lcr_bits);
__IO_REG32_BIT(PEX00_LCSR,            0xD0040070,__READ_WRITE ,__pex_lcsr_bits);
__IO_REG32_BIT(PEX00_AERHR,           0xD0040100,__READ_WRITE ,__pex_aerhr_bits);
__IO_REG32_BIT(PEX00_UESTR,           0xD0040104,__READ_WRITE ,__pex_uestr_bits);
__IO_REG32_BIT(PEX00_UEMR,            0xD0040108,__READ_WRITE ,__pex_uemr_bits);
__IO_REG32_BIT(PEX00_UESR,            0xD004010C,__READ_WRITE ,__pex_uesr_bits);
__IO_REG32_BIT(PEX00_CESTR,           0xD0040110,__READ_WRITE ,__pex_cestr_bits);
__IO_REG32_BIT(PEX00_CEMR,            0xD0040114,__READ_WRITE ,__pex_cemr_bits);
__IO_REG32_BIT(PEX00_AECCR,           0xD0040118,__READ       ,__pex_aeccr_bits);
__IO_REG32(    PEX00_HL1R,            0xD004011C,__READ       );
__IO_REG32(    PEX00_HL2R,            0xD0040120,__READ       );
__IO_REG32(    PEX00_HL3R,            0xD0040124,__READ       );
__IO_REG32(    PEX00_HL4R,            0xD0040128,__READ       );
__IO_REG32_BIT(PEX00_CTRLR,           0xD0041A00,__READ_WRITE ,__pex_ctrlr_bits);
__IO_REG32_BIT(PEX00_STR,             0xD0041A04,__READ_WRITE ,__pex_str_bits);
__IO_REG32_BIT(PEX00_RCSSPLR,         0xD0041A0C,__READ_WRITE ,__pex_rcssplr_bits);
__IO_REG32_BIT(PEX00_CTR,             0xD0041A10,__READ_WRITE ,__pex_ctr_bits);
__IO_REG32_BIT(PEX00_RCPMER,          0xD0041A14,__READ       ,__pex_rcpmer_bits);
__IO_REG32_BIT(PEX00_PMR,             0xD0041A18,__READ_WRITE ,__pex_pmr_bits);
__IO_REG32_BIT(PEX00_FCR,             0xD0041A20,__READ_WRITE ,__pex_fcr_bits);
__IO_REG32_BIT(PEX00_AT4R,            0xD0041A30,__READ_WRITE ,__pex_at4r_bits);
__IO_REG32_BIT(PEX00_AT1R,            0xD0041A40,__READ_WRITE ,__pex_at1r_bits);
__IO_REG32_BIT(PEX00_RAMPPCR,         0xD0041A50,__READ_WRITE ,__pex_ramppcr_bits);
__IO_REG32_BIT(PEX00_DBGCR,           0xD0041A60,__READ_WRITE ,__pex_dbgcr_bits);
__IO_REG32_BIT(PEX00_TLCR,            0xD0041AB0,__READ_WRITE ,__pex_tlcr_bits);
__IO_REG32_BIT(PEX00_PHYIAR,          0xD0041B00,__READ_WRITE ,__pex_phyiar_bits);
__IO_REG32_BIT(PEX00_ICR,             0xD0041900,__READ       ,__pex_icr_bits);
__IO_REG32_BIT(PEX00_IMR,             0xD0041910,__READ_WRITE ,__pex_icr_bits);
__IO_REG32_BIT(PEX00_MACR,            0xD00418D0,__READ_WRITE ,__pex_macr_bits);
__IO_REG32_BIT(PEX00_MACLR,           0xD00418E0,__READ_WRITE ,__pex_maclr_bits);
__IO_REG32_BIT(PEX00_MACHR,           0xD00418E4,__READ_WRITE ,__pex_machr_bits);
__IO_REG32_BIT(PEX00_MATR,            0xD00418E8,__READ_WRITE ,__pex_matr_bits);
__IO_REG32_BIT(PEX01_W0CR,            0xD0045820,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX01_W0BR,            0xD0045824,__READ_WRITE );
__IO_REG32_BIT(PEX01_W0RR,            0xD004582C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32_BIT(PEX01_W1CR,            0xD0045830,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX01_W1BR,            0xD0045834,__READ_WRITE );
__IO_REG32_BIT(PEX01_W1RR,            0xD004583C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32_BIT(PEX01_W2CR,            0xD0045840,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX01_W2BR,            0xD0045844,__READ_WRITE );
__IO_REG32_BIT(PEX01_W2RR,            0xD004584C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32_BIT(PEX01_W3CR,            0xD0045850,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX01_W3BR,            0xD0045854,__READ_WRITE );
__IO_REG32_BIT(PEX01_W3RR,            0xD004585C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32_BIT(PEX01_W4CR,            0xD0045860,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX01_W4BR,            0xD0045864,__READ_WRITE );
__IO_REG32_BIT(PEX01_W4RR,            0xD004586C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32(    PEX01_W4RHR,           0xD0045870,__READ_WRITE );
__IO_REG32_BIT(PEX01_W5CR,            0xD0045880,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX01_W5BR,            0xD0045884,__READ_WRITE );
__IO_REG32_BIT(PEX01_W5RR,            0xD004588C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32(    PEX01_W5RHR,           0xD0045890,__READ_WRITE );
__IO_REG32_BIT(PEX01_DWCR,            0xD00458B0,__READ_WRITE ,__pex_dwcr_bits);
__IO_REG32_BIT(PEX01_EROMWCR,         0xD00458C0,__READ_WRITE ,__pex_eromwcr_bits);
__IO_REG32_BIT(PEX01_EROMWRR,         0xD00458C4,__READ_WRITE ,__pex_eromwrr_bits);
__IO_REG32_BIT(PEX01_BAR1CR,          0xD0045804,__READ_WRITE ,__pex_barcr_bits);
__IO_REG32_BIT(PEX01_BAR2CR,          0xD0045808,__READ_WRITE ,__pex_barcr_bits);
__IO_REG32_BIT(PEX01_EROMBARCR,       0xD004580C,__READ_WRITE ,__pex_erombarcr_bits);
__IO_REG32_BIT(PEX01_CAR,             0xD00458F8,__READ_WRITE ,__pex_car_bits);
__IO_REG32(    PEX01_CDR,             0xD00458FC,__READ_WRITE );
__IO_REG32_BIT(PEX01_DVIDR,           0xD0044000,__READ       ,__pex_dvidr_bits);
__IO_REG32_BIT(PEX01_CSR,             0xD0044004,__READ_WRITE ,__pex_csr_bits);
__IO_REG32_BIT(PEX01_CCRIDR,          0xD0044008,__READ       ,__pex_ccridr_bits);
__IO_REG32_BIT(PEX01_BISTHTCLSR,      0xD004400C,__READ_WRITE ,__pex_bisthtclsr_bits);
__IO_REG32_BIT(PEX01_BAR0IR,          0xD0044010,__READ_WRITE ,__pex_barir_bits);
__IO_REG32(    PEX01_BAR0IHR,         0xD0044014,__READ_WRITE );
__IO_REG32_BIT(PEX01_BAR1IR,          0xD0044018,__READ_WRITE ,__pex_barir_bits);
__IO_REG32(    PEX01_BAR1IHR,         0xD004401C,__READ_WRITE );
__IO_REG32_BIT(PEX01_BAR2IR,          0xD0044020,__READ_WRITE ,__pex_barir_bits);
__IO_REG32(    PEX01_BAR2IHR,         0xD0044024,__READ_WRITE );
__IO_REG32_BIT(PEX01_SSDVIDR,         0xD004402C,__READ       ,__pex_ssdvidr_bits);
__IO_REG32_BIT(PEX01_EROMBARR,        0xD0044030,__READ_WRITE ,__pex_erombarr_bits);
__IO_REG32_BIT(PEX01_CLPR,            0xD0044034,__READ       ,__pex_clpr_bits);
__IO_REG32_BIT(PEX01_IPLR,            0xD004403C,__READ_WRITE ,__pex_iplr_bits);
__IO_REG32_BIT(PEX01_PMCHR,           0xD0044040,__READ       ,__pex_pmchr_bits);
__IO_REG32_BIT(PEX01_PMCSR,           0xD0044044,__READ_WRITE ,__pex_pmcsr_bits);
__IO_REG32_BIT(PEX01_MSIMCR,          0xD0044050,__READ_WRITE ,__pex_msimcr_bits);
__IO_REG32(    PEX01_MSIMAR,          0xD0044054,__READ_WRITE );
__IO_REG32(    PEX01_MSIMAHR,         0xD0044058,__READ_WRITE );
__IO_REG32_BIT(PEX01_MSIMDR,          0xD004405C,__READ_WRITE ,__pex_msimdr_bits);
__IO_REG32_BIT(PEX01_CR,              0xD0044060,__READ       ,__pex_cr_bits);
__IO_REG32_BIT(PEX01_DCR,             0xD0044064,__READ       ,__pex_dcr_bits);
__IO_REG32_BIT(PEX01_DCSR,            0xD0044068,__READ_WRITE ,__pex_dcsr_bits);
__IO_REG32_BIT(PEX01_LCR,             0xD004406C,__READ       ,__pex_lcr_bits);
__IO_REG32_BIT(PEX01_LCSR,            0xD0044070,__READ_WRITE ,__pex_lcsr_bits);
__IO_REG32_BIT(PEX01_AERHR,           0xD0044100,__READ_WRITE ,__pex_aerhr_bits);
__IO_REG32_BIT(PEX01_UESTR,           0xD0044104,__READ_WRITE ,__pex_uestr_bits);
__IO_REG32_BIT(PEX01_UEMR,            0xD0044108,__READ_WRITE ,__pex_uemr_bits);
__IO_REG32_BIT(PEX01_UESR,            0xD004410C,__READ_WRITE ,__pex_uesr_bits);
__IO_REG32_BIT(PEX01_CESTR,           0xD0044110,__READ_WRITE ,__pex_cestr_bits);
__IO_REG32_BIT(PEX01_CEMR,            0xD0044114,__READ_WRITE ,__pex_cemr_bits);
__IO_REG32_BIT(PEX01_AECCR,           0xD0044118,__READ       ,__pex_aeccr_bits);
__IO_REG32(    PEX01_HL1R,            0xD004411C,__READ       );
__IO_REG32(    PEX01_HL2R,            0xD0044120,__READ       );
__IO_REG32(    PEX01_HL3R,            0xD0044124,__READ       );
__IO_REG32(    PEX01_HL4R,            0xD0044128,__READ       );
__IO_REG32_BIT(PEX01_CTRLR,           0xD0045A00,__READ_WRITE ,__pex_ctrlr_bits);
__IO_REG32_BIT(PEX01_STR,             0xD0045A04,__READ_WRITE ,__pex_str_bits);
__IO_REG32_BIT(PEX01_RCSSPLR,         0xD0045A0C,__READ_WRITE ,__pex_rcssplr_bits);
__IO_REG32_BIT(PEX01_CTR,             0xD0045A10,__READ_WRITE ,__pex_ctr_bits);
__IO_REG32_BIT(PEX01_RCPMER,          0xD0045A14,__READ       ,__pex_rcpmer_bits);
__IO_REG32_BIT(PEX01_PMR,             0xD0045A18,__READ_WRITE ,__pex_pmr_bits);
__IO_REG32_BIT(PEX01_FCR,             0xD0045A20,__READ_WRITE ,__pex_fcr_bits);
__IO_REG32_BIT(PEX01_AT4R,            0xD0045A30,__READ_WRITE ,__pex_at4r_bits);
__IO_REG32_BIT(PEX01_AT1R,            0xD0045A40,__READ_WRITE ,__pex_at1r_bits);
__IO_REG32_BIT(PEX01_RAMPPCR,         0xD0045A50,__READ_WRITE ,__pex_ramppcr_bits);
__IO_REG32_BIT(PEX01_DBGCR,           0xD0045A60,__READ_WRITE ,__pex_dbgcr_bits);
__IO_REG32_BIT(PEX01_TLCR,            0xD0045AB0,__READ_WRITE ,__pex_tlcr_bits);
__IO_REG32_BIT(PEX01_PHYIAR,          0xD0045B00,__READ_WRITE ,__pex_phyiar_bits);
__IO_REG32_BIT(PEX01_ICR,             0xD0045900,__READ       ,__pex_icr_bits);
__IO_REG32_BIT(PEX01_IMR,             0xD0045910,__READ_WRITE ,__pex_icr_bits);
__IO_REG32_BIT(PEX01_MACR,            0xD00458D0,__READ_WRITE ,__pex_macr_bits);
__IO_REG32_BIT(PEX01_MACLR,           0xD00458E0,__READ_WRITE ,__pex_maclr_bits);
__IO_REG32_BIT(PEX01_MACHR,           0xD00458E4,__READ_WRITE ,__pex_machr_bits);
__IO_REG32_BIT(PEX01_MATR,            0xD00458E8,__READ_WRITE ,__pex_matr_bits);
__IO_REG32_BIT(PEX02_W0CR,            0xD0049820,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX02_W0BR,            0xD0049824,__READ_WRITE );
__IO_REG32_BIT(PEX02_W0RR,            0xD004982C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32_BIT(PEX02_W1CR,            0xD0049830,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX02_W1BR,            0xD0049834,__READ_WRITE );
__IO_REG32_BIT(PEX02_W1RR,            0xD004983C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32_BIT(PEX02_W2CR,            0xD0049840,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX02_W2BR,            0xD0049844,__READ_WRITE );
__IO_REG32_BIT(PEX02_W2RR,            0xD004984C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32_BIT(PEX02_W3CR,            0xD0049850,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX02_W3BR,            0xD0049854,__READ_WRITE );
__IO_REG32_BIT(PEX02_W3RR,            0xD004985C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32_BIT(PEX02_W4CR,            0xD0049860,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX02_W4BR,            0xD0049864,__READ_WRITE );
__IO_REG32_BIT(PEX02_W4RR,            0xD004986C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32(    PEX02_W4RHR,           0xD0049870,__READ_WRITE );
__IO_REG32_BIT(PEX02_W5CR,            0xD0049880,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX02_W5BR,            0xD0049884,__READ_WRITE );
__IO_REG32_BIT(PEX02_W5RR,            0xD004988C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32(    PEX02_W5RHR,           0xD0049890,__READ_WRITE );
__IO_REG32_BIT(PEX02_DWCR,            0xD00498B0,__READ_WRITE ,__pex_dwcr_bits);
__IO_REG32_BIT(PEX02_EROMWCR,         0xD00498C0,__READ_WRITE ,__pex_eromwcr_bits);
__IO_REG32_BIT(PEX02_EROMWRR,         0xD00498C4,__READ_WRITE ,__pex_eromwrr_bits);
__IO_REG32_BIT(PEX02_BAR1CR,          0xD0049804,__READ_WRITE ,__pex_barcr_bits);
__IO_REG32_BIT(PEX02_BAR2CR,          0xD0049808,__READ_WRITE ,__pex_barcr_bits);
__IO_REG32_BIT(PEX02_EROMBARCR,       0xD004980C,__READ_WRITE ,__pex_erombarcr_bits);
__IO_REG32_BIT(PEX02_CAR,             0xD00498F8,__READ_WRITE ,__pex_car_bits);
__IO_REG32(    PEX02_CDR,             0xD00498FC,__READ_WRITE );
__IO_REG32_BIT(PEX02_DVIDR,           0xD0048000,__READ       ,__pex_dvidr_bits);
__IO_REG32_BIT(PEX02_CSR,             0xD0048004,__READ_WRITE ,__pex_csr_bits);
__IO_REG32_BIT(PEX02_CCRIDR,          0xD0048008,__READ       ,__pex_ccridr_bits);
__IO_REG32_BIT(PEX02_BISTHTCLSR,      0xD004800C,__READ_WRITE ,__pex_bisthtclsr_bits);
__IO_REG32_BIT(PEX02_BAR0IR,          0xD0048010,__READ_WRITE ,__pex_barir_bits);
__IO_REG32(    PEX02_BAR0IHR,         0xD0048014,__READ_WRITE );
__IO_REG32_BIT(PEX02_BAR1IR,          0xD0048018,__READ_WRITE ,__pex_barir_bits);
__IO_REG32(    PEX02_BAR1IHR,         0xD004801C,__READ_WRITE );
__IO_REG32_BIT(PEX02_BAR2IR,          0xD0048020,__READ_WRITE ,__pex_barir_bits);
__IO_REG32(    PEX02_BAR2IHR,         0xD0048024,__READ_WRITE );
__IO_REG32_BIT(PEX02_SSDVIDR,         0xD004802C,__READ       ,__pex_ssdvidr_bits);
__IO_REG32_BIT(PEX02_EROMBARR,        0xD0048030,__READ_WRITE ,__pex_erombarr_bits);
__IO_REG32_BIT(PEX02_CLPR,            0xD0048034,__READ       ,__pex_clpr_bits);
__IO_REG32_BIT(PEX02_IPLR,            0xD004803C,__READ_WRITE ,__pex_iplr_bits);
__IO_REG32_BIT(PEX02_PMCHR,           0xD0048040,__READ       ,__pex_pmchr_bits);
__IO_REG32_BIT(PEX02_PMCSR,           0xD0048044,__READ_WRITE ,__pex_pmcsr_bits);
__IO_REG32_BIT(PEX02_MSIMCR,          0xD0048050,__READ_WRITE ,__pex_msimcr_bits);
__IO_REG32(    PEX02_MSIMAR,          0xD0048054,__READ_WRITE );
__IO_REG32(    PEX02_MSIMAHR,         0xD0048058,__READ_WRITE );
__IO_REG32_BIT(PEX02_MSIMDR,          0xD004805C,__READ_WRITE ,__pex_msimdr_bits);
__IO_REG32_BIT(PEX02_CR,              0xD0048060,__READ       ,__pex_cr_bits);
__IO_REG32_BIT(PEX02_DCR,             0xD0048064,__READ       ,__pex_dcr_bits);
__IO_REG32_BIT(PEX02_DCSR,            0xD0048068,__READ_WRITE ,__pex_dcsr_bits);
__IO_REG32_BIT(PEX02_LCR,             0xD004806C,__READ       ,__pex_lcr_bits);
__IO_REG32_BIT(PEX02_LCSR,            0xD0048070,__READ_WRITE ,__pex_lcsr_bits);
__IO_REG32_BIT(PEX02_AERHR,           0xD0048100,__READ_WRITE ,__pex_aerhr_bits);
__IO_REG32_BIT(PEX02_UESTR,           0xD0048104,__READ_WRITE ,__pex_uestr_bits);
__IO_REG32_BIT(PEX02_UEMR,            0xD0048108,__READ_WRITE ,__pex_uemr_bits);
__IO_REG32_BIT(PEX02_UESR,            0xD004810C,__READ_WRITE ,__pex_uesr_bits);
__IO_REG32_BIT(PEX02_CESTR,           0xD0048110,__READ_WRITE ,__pex_cestr_bits);
__IO_REG32_BIT(PEX02_CEMR,            0xD0048114,__READ_WRITE ,__pex_cemr_bits);
__IO_REG32_BIT(PEX02_AECCR,           0xD0048118,__READ       ,__pex_aeccr_bits);
__IO_REG32(    PEX02_HL1R,            0xD004811C,__READ       );
__IO_REG32(    PEX02_HL2R,            0xD0048120,__READ       );
__IO_REG32(    PEX02_HL3R,            0xD0048124,__READ       );
__IO_REG32(    PEX02_HL4R,            0xD0048128,__READ       );
__IO_REG32_BIT(PEX02_CTRLR,           0xD0049A00,__READ_WRITE ,__pex_ctrlr_bits);
__IO_REG32_BIT(PEX02_STR,             0xD0049A04,__READ_WRITE ,__pex_str_bits);
__IO_REG32_BIT(PEX02_RCSSPLR,         0xD0049A0C,__READ_WRITE ,__pex_rcssplr_bits);
__IO_REG32_BIT(PEX02_CTR,             0xD0049A10,__READ_WRITE ,__pex_ctr_bits);
__IO_REG32_BIT(PEX02_RCPMER,          0xD0049A14,__READ       ,__pex_rcpmer_bits);
__IO_REG32_BIT(PEX02_PMR,             0xD0049A18,__READ_WRITE ,__pex_pmr_bits);
__IO_REG32_BIT(PEX02_FCR,             0xD0049A20,__READ_WRITE ,__pex_fcr_bits);
__IO_REG32_BIT(PEX02_AT4R,            0xD0049A30,__READ_WRITE ,__pex_at4r_bits);
__IO_REG32_BIT(PEX02_AT1R,            0xD0049A40,__READ_WRITE ,__pex_at1r_bits);
__IO_REG32_BIT(PEX02_RAMPPCR,         0xD0049A50,__READ_WRITE ,__pex_ramppcr_bits);
__IO_REG32_BIT(PEX02_DBGCR,           0xD0049A60,__READ_WRITE ,__pex_dbgcr_bits);
__IO_REG32_BIT(PEX02_TLCR,            0xD0049AB0,__READ_WRITE ,__pex_tlcr_bits);
__IO_REG32_BIT(PEX02_PHYIAR,          0xD0049B00,__READ_WRITE ,__pex_phyiar_bits);
__IO_REG32_BIT(PEX02_ICR,             0xD0049900,__READ       ,__pex_icr_bits);
__IO_REG32_BIT(PEX02_IMR,             0xD0049910,__READ_WRITE ,__pex_icr_bits);
__IO_REG32_BIT(PEX02_MACR,            0xD00498D0,__READ_WRITE ,__pex_macr_bits);
__IO_REG32_BIT(PEX02_MACLR,           0xD00498E0,__READ_WRITE ,__pex_maclr_bits);
__IO_REG32_BIT(PEX02_MACHR,           0xD00498E4,__READ_WRITE ,__pex_machr_bits);
__IO_REG32_BIT(PEX02_MATR,            0xD00498E8,__READ_WRITE ,__pex_matr_bits);
__IO_REG32_BIT(PEX03_W0CR,            0xD004D820,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX03_W0BR,            0xD004D824,__READ_WRITE );
__IO_REG32_BIT(PEX03_W0RR,            0xD004D82C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32_BIT(PEX03_W1CR,            0xD004D830,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX03_W1BR,            0xD004D834,__READ_WRITE );
__IO_REG32_BIT(PEX03_W1RR,            0xD004D83C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32_BIT(PEX03_W2CR,            0xD004D840,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX03_W2BR,            0xD004D844,__READ_WRITE );
__IO_REG32_BIT(PEX03_W2RR,            0xD004D84C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32_BIT(PEX03_W3CR,            0xD004D850,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX03_W3BR,            0xD004D854,__READ_WRITE );
__IO_REG32_BIT(PEX03_W3RR,            0xD004D85C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32_BIT(PEX03_W4CR,            0xD004D860,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX03_W4BR,            0xD004D864,__READ_WRITE );
__IO_REG32_BIT(PEX03_W4RR,            0xD004D86C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32(    PEX03_W4RHR,           0xD004D870,__READ_WRITE );
__IO_REG32_BIT(PEX03_W5CR,            0xD004D880,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX03_W5BR,            0xD004D884,__READ_WRITE );
__IO_REG32_BIT(PEX03_W5RR,            0xD004D88C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32(    PEX03_W5RHR,           0xD004D890,__READ_WRITE );
__IO_REG32_BIT(PEX03_DWCR,            0xD004D8B0,__READ_WRITE ,__pex_dwcr_bits);
__IO_REG32_BIT(PEX03_EROMWCR,         0xD004D8C0,__READ_WRITE ,__pex_eromwcr_bits);
__IO_REG32_BIT(PEX03_EROMWRR,         0xD004D8C4,__READ_WRITE ,__pex_eromwrr_bits);
__IO_REG32_BIT(PEX03_BAR1CR,          0xD004D804,__READ_WRITE ,__pex_barcr_bits);
__IO_REG32_BIT(PEX03_BAR2CR,          0xD004D808,__READ_WRITE ,__pex_barcr_bits);
__IO_REG32_BIT(PEX03_EROMBARCR,       0xD004D80C,__READ_WRITE ,__pex_erombarcr_bits);
__IO_REG32_BIT(PEX03_CAR,             0xD004D8F8,__READ_WRITE ,__pex_car_bits);
__IO_REG32(    PEX03_CDR,             0xD004D8FC,__READ_WRITE );
__IO_REG32_BIT(PEX03_DVIDR,           0xD004C000,__READ       ,__pex_dvidr_bits);
__IO_REG32_BIT(PEX03_CSR,             0xD004C004,__READ_WRITE ,__pex_csr_bits);
__IO_REG32_BIT(PEX03_CCRIDR,          0xD004C008,__READ       ,__pex_ccridr_bits);
__IO_REG32_BIT(PEX03_BISTHTCLSR,      0xD004C00C,__READ_WRITE ,__pex_bisthtclsr_bits);
__IO_REG32_BIT(PEX03_BAR0IR,          0xD004C010,__READ_WRITE ,__pex_barir_bits);
__IO_REG32(    PEX03_BAR0IHR,         0xD004C014,__READ_WRITE );
__IO_REG32_BIT(PEX03_BAR1IR,          0xD004C018,__READ_WRITE ,__pex_barir_bits);
__IO_REG32(    PEX03_BAR1IHR,         0xD004C01C,__READ_WRITE );
__IO_REG32_BIT(PEX03_BAR2IR,          0xD004C020,__READ_WRITE ,__pex_barir_bits);
__IO_REG32(    PEX03_BAR2IHR,         0xD004C024,__READ_WRITE );
__IO_REG32_BIT(PEX03_SSDVIDR,         0xD004C02C,__READ       ,__pex_ssdvidr_bits);
__IO_REG32_BIT(PEX03_EROMBARR,        0xD004C030,__READ_WRITE ,__pex_erombarr_bits);
__IO_REG32_BIT(PEX03_CLPR,            0xD004C034,__READ       ,__pex_clpr_bits);
__IO_REG32_BIT(PEX03_IPLR,            0xD004C03C,__READ_WRITE ,__pex_iplr_bits);
__IO_REG32_BIT(PEX03_PMCHR,           0xD004C040,__READ       ,__pex_pmchr_bits);
__IO_REG32_BIT(PEX03_PMCSR,           0xD004C044,__READ_WRITE ,__pex_pmcsr_bits);
__IO_REG32_BIT(PEX03_MSIMCR,          0xD004C050,__READ_WRITE ,__pex_msimcr_bits);
__IO_REG32(    PEX03_MSIMAR,          0xD004C054,__READ_WRITE );
__IO_REG32(    PEX03_MSIMAHR,         0xD004C058,__READ_WRITE );
__IO_REG32_BIT(PEX03_MSIMDR,          0xD004C05C,__READ_WRITE ,__pex_msimdr_bits);
__IO_REG32_BIT(PEX03_CR,              0xD004C060,__READ       ,__pex_cr_bits);
__IO_REG32_BIT(PEX03_DCR,             0xD004C064,__READ       ,__pex_dcr_bits);
__IO_REG32_BIT(PEX03_DCSR,            0xD004C068,__READ_WRITE ,__pex_dcsr_bits);
__IO_REG32_BIT(PEX03_LCR,             0xD004C06C,__READ       ,__pex_lcr_bits);
__IO_REG32_BIT(PEX03_LCSR,            0xD004C070,__READ_WRITE ,__pex_lcsr_bits);
__IO_REG32_BIT(PEX03_AERHR,           0xD004C100,__READ_WRITE ,__pex_aerhr_bits);
__IO_REG32_BIT(PEX03_UESTR,           0xD004C104,__READ_WRITE ,__pex_uestr_bits);
__IO_REG32_BIT(PEX03_UEMR,            0xD004C108,__READ_WRITE ,__pex_uemr_bits);
__IO_REG32_BIT(PEX03_UESR,            0xD004C10C,__READ_WRITE ,__pex_uesr_bits);
__IO_REG32_BIT(PEX03_CESTR,           0xD004C110,__READ_WRITE ,__pex_cestr_bits);
__IO_REG32_BIT(PEX03_CEMR,            0xD004C114,__READ_WRITE ,__pex_cemr_bits);
__IO_REG32_BIT(PEX03_AECCR,           0xD004C118,__READ       ,__pex_aeccr_bits);
__IO_REG32(    PEX03_HL1R,            0xD004C11C,__READ       );
__IO_REG32(    PEX03_HL2R,            0xD004C120,__READ       );
__IO_REG32(    PEX03_HL3R,            0xD004C124,__READ       );
__IO_REG32(    PEX03_HL4R,            0xD004C128,__READ       );
__IO_REG32_BIT(PEX03_CTRLR,           0xD004DA00,__READ_WRITE ,__pex_ctrlr_bits);
__IO_REG32_BIT(PEX03_STR,             0xD004DA04,__READ_WRITE ,__pex_str_bits);
__IO_REG32_BIT(PEX03_RCSSPLR,         0xD004DA0C,__READ_WRITE ,__pex_rcssplr_bits);
__IO_REG32_BIT(PEX03_CTR,             0xD004DA10,__READ_WRITE ,__pex_ctr_bits);
__IO_REG32_BIT(PEX03_RCPMER,          0xD004DA14,__READ       ,__pex_rcpmer_bits);
__IO_REG32_BIT(PEX03_PMR,             0xD004DA18,__READ_WRITE ,__pex_pmr_bits);
__IO_REG32_BIT(PEX03_FCR,             0xD004DA20,__READ_WRITE ,__pex_fcr_bits);
__IO_REG32_BIT(PEX03_AT4R,            0xD004DA30,__READ_WRITE ,__pex_at4r_bits);
__IO_REG32_BIT(PEX03_AT1R,            0xD004DA40,__READ_WRITE ,__pex_at1r_bits);
__IO_REG32_BIT(PEX03_RAMPPCR,         0xD004DA50,__READ_WRITE ,__pex_ramppcr_bits);
__IO_REG32_BIT(PEX03_DBGCR,           0xD004DA60,__READ_WRITE ,__pex_dbgcr_bits);
__IO_REG32_BIT(PEX03_TLCR,            0xD004DAB0,__READ_WRITE ,__pex_tlcr_bits);
__IO_REG32_BIT(PEX03_PHYIAR,          0xD004DB00,__READ_WRITE ,__pex_phyiar_bits);
__IO_REG32_BIT(PEX03_ICR,             0xD004D900,__READ       ,__pex_icr_bits);
__IO_REG32_BIT(PEX03_IMR,             0xD004D910,__READ_WRITE ,__pex_icr_bits);
__IO_REG32_BIT(PEX03_MACR,            0xD004D8D0,__READ_WRITE ,__pex_macr_bits);
__IO_REG32_BIT(PEX03_MACLR,           0xD004D8E0,__READ_WRITE ,__pex_maclr_bits);
__IO_REG32_BIT(PEX03_MACHR,           0xD004D8E4,__READ_WRITE ,__pex_machr_bits);
__IO_REG32_BIT(PEX03_MATR,            0xD004D8E8,__READ_WRITE ,__pex_matr_bits);

/***************************************************************************
 **
 ** PCIE 1
 **
 ***************************************************************************/
__IO_REG32_BIT(PEX10_W0CR,            0xD0081820,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX10_W0BR,            0xD0081824,__READ_WRITE );
__IO_REG32_BIT(PEX10_W0RR,            0xD008182C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32_BIT(PEX10_W1CR,            0xD0081830,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX10_W1BR,            0xD0081834,__READ_WRITE );
__IO_REG32_BIT(PEX10_W1RR,            0xD008183C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32_BIT(PEX10_W2CR,            0xD0081840,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX10_W2BR,            0xD0081844,__READ_WRITE );
__IO_REG32_BIT(PEX10_W2RR,            0xD008184C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32_BIT(PEX10_W3CR,            0xD0081850,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX10_W3BR,            0xD0081854,__READ_WRITE );
__IO_REG32_BIT(PEX10_W3RR,            0xD008185C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32_BIT(PEX10_W4CR,            0xD0081860,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX10_W4BR,            0xD0081864,__READ_WRITE );
__IO_REG32_BIT(PEX10_W4RR,            0xD008186C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32(    PEX10_W4RHR,           0xD0081870,__READ_WRITE );
__IO_REG32_BIT(PEX10_W5CR,            0xD0081880,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX10_W5BR,            0xD0081884,__READ_WRITE );
__IO_REG32_BIT(PEX10_W5RR,            0xD008188C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32(    PEX10_W5RHR,           0xD0081890,__READ_WRITE );
__IO_REG32_BIT(PEX10_DWCR,            0xD00818B0,__READ_WRITE ,__pex_dwcr_bits);
__IO_REG32_BIT(PEX10_EROMWCR,         0xD00818C0,__READ_WRITE ,__pex_eromwcr_bits);
__IO_REG32_BIT(PEX10_EROMWRR,         0xD00818C4,__READ_WRITE ,__pex_eromwrr_bits);
__IO_REG32_BIT(PEX10_BAR1CR,          0xD0081804,__READ_WRITE ,__pex_barcr_bits);
__IO_REG32_BIT(PEX10_BAR2CR,          0xD0081808,__READ_WRITE ,__pex_barcr_bits);
__IO_REG32_BIT(PEX10_EROMBARCR,       0xD008180C,__READ_WRITE ,__pex_erombarcr_bits);
__IO_REG32_BIT(PEX10_CAR,             0xD00818F8,__READ_WRITE ,__pex_car_bits);
__IO_REG32(    PEX10_CDR,             0xD00818FC,__READ_WRITE );
__IO_REG32_BIT(PEX10_DVIDR,           0xD0080000,__READ       ,__pex_dvidr_bits);
__IO_REG32_BIT(PEX10_CSR,             0xD0080004,__READ_WRITE ,__pex_csr_bits);
__IO_REG32_BIT(PEX10_CCRIDR,          0xD0080008,__READ       ,__pex_ccridr_bits);
__IO_REG32_BIT(PEX10_BISTHTCLSR,      0xD008000C,__READ_WRITE ,__pex_bisthtclsr_bits);
__IO_REG32_BIT(PEX10_BAR0IR,          0xD0080010,__READ_WRITE ,__pex_barir_bits);
__IO_REG32(    PEX10_BAR0IHR,         0xD0080014,__READ_WRITE );
__IO_REG32_BIT(PEX10_BAR1IR,          0xD0080018,__READ_WRITE ,__pex_barir_bits);
__IO_REG32(    PEX10_BAR1IHR,         0xD008001C,__READ_WRITE );
__IO_REG32_BIT(PEX10_BAR2IR,          0xD0080020,__READ_WRITE ,__pex_barir_bits);
__IO_REG32(    PEX10_BAR2IHR,         0xD0080024,__READ_WRITE );
__IO_REG32_BIT(PEX10_SSDVIDR,         0xD008002C,__READ       ,__pex_ssdvidr_bits);
__IO_REG32_BIT(PEX10_EROMBARR,        0xD0080030,__READ_WRITE ,__pex_erombarr_bits);
__IO_REG32_BIT(PEX10_CLPR,            0xD0080034,__READ       ,__pex_clpr_bits);
__IO_REG32_BIT(PEX10_IPLR,            0xD008003C,__READ_WRITE ,__pex_iplr_bits);
__IO_REG32_BIT(PEX10_PMCHR,           0xD0080040,__READ       ,__pex_pmchr_bits);
__IO_REG32_BIT(PEX10_PMCSR,           0xD0080044,__READ_WRITE ,__pex_pmcsr_bits);
__IO_REG32_BIT(PEX10_MSIMCR,          0xD0080050,__READ_WRITE ,__pex_msimcr_bits);
__IO_REG32(    PEX10_MSIMAR,          0xD0080054,__READ_WRITE );
__IO_REG32(    PEX10_MSIMAHR,         0xD0080058,__READ_WRITE );
__IO_REG32_BIT(PEX10_MSIMDR,          0xD008005C,__READ_WRITE ,__pex_msimdr_bits);
__IO_REG32_BIT(PEX10_CR,              0xD0080060,__READ       ,__pex_cr_bits);
__IO_REG32_BIT(PEX10_DCR,             0xD0080064,__READ       ,__pex_dcr_bits);
__IO_REG32_BIT(PEX10_DCSR,            0xD0080068,__READ_WRITE ,__pex_dcsr_bits);
__IO_REG32_BIT(PEX10_LCR,             0xD008006C,__READ       ,__pex_lcr_bits);
__IO_REG32_BIT(PEX10_LCSR,            0xD0080070,__READ_WRITE ,__pex_lcsr_bits);
__IO_REG32_BIT(PEX10_AERHR,           0xD0080100,__READ_WRITE ,__pex_aerhr_bits);
__IO_REG32_BIT(PEX10_UESTR,           0xD0080104,__READ_WRITE ,__pex_uestr_bits);
__IO_REG32_BIT(PEX10_UEMR,            0xD0080108,__READ_WRITE ,__pex_uemr_bits);
__IO_REG32_BIT(PEX10_UESR,            0xD008010C,__READ_WRITE ,__pex_uesr_bits);
__IO_REG32_BIT(PEX10_CESTR,           0xD0080110,__READ_WRITE ,__pex_cestr_bits);
__IO_REG32_BIT(PEX10_CEMR,            0xD0080114,__READ_WRITE ,__pex_cemr_bits);
__IO_REG32_BIT(PEX10_AECCR,           0xD0080118,__READ       ,__pex_aeccr_bits);
__IO_REG32(    PEX10_HL1R,            0xD008011C,__READ       );
__IO_REG32(    PEX10_HL2R,            0xD0080120,__READ       );
__IO_REG32(    PEX10_HL3R,            0xD0080124,__READ       );
__IO_REG32(    PEX10_HL4R,            0xD0080128,__READ       );
__IO_REG32_BIT(PEX10_CTRLR,           0xD0081A00,__READ_WRITE ,__pex_ctrlr_bits);
__IO_REG32_BIT(PEX10_STR,             0xD0081A04,__READ_WRITE ,__pex_str_bits);
__IO_REG32_BIT(PEX10_RCSSPLR,         0xD0081A0C,__READ_WRITE ,__pex_rcssplr_bits);
__IO_REG32_BIT(PEX10_CTR,             0xD0081A10,__READ_WRITE ,__pex_ctr_bits);
__IO_REG32_BIT(PEX10_RCPMER,          0xD0081A14,__READ       ,__pex_rcpmer_bits);
__IO_REG32_BIT(PEX10_PMR,             0xD0081A18,__READ_WRITE ,__pex_pmr_bits);
__IO_REG32_BIT(PEX10_FCR,             0xD0081A20,__READ_WRITE ,__pex_fcr_bits);
__IO_REG32_BIT(PEX10_AT4R,            0xD0081A30,__READ_WRITE ,__pex_at4r_bits);
__IO_REG32_BIT(PEX10_AT1R,            0xD0081A40,__READ_WRITE ,__pex_at1r_bits);
__IO_REG32_BIT(PEX10_RAMPPCR,         0xD0081A50,__READ_WRITE ,__pex_ramppcr_bits);
__IO_REG32_BIT(PEX10_DBGCR,           0xD0081A60,__READ_WRITE ,__pex_dbgcr_bits);
__IO_REG32_BIT(PEX10_TLCR,            0xD0081AB0,__READ_WRITE ,__pex_tlcr_bits);
__IO_REG32_BIT(PEX10_PHYIAR,          0xD0081B00,__READ_WRITE ,__pex_phyiar_bits);
__IO_REG32_BIT(PEX10_ICR,             0xD0081900,__READ       ,__pex_icr_bits);
__IO_REG32_BIT(PEX10_IMR,             0xD0081910,__READ_WRITE ,__pex_icr_bits);
__IO_REG32_BIT(PEX10_MACR,            0xD00818D0,__READ_WRITE ,__pex_macr_bits);
__IO_REG32_BIT(PEX10_MACLR,           0xD00818E0,__READ_WRITE ,__pex_maclr_bits);
__IO_REG32_BIT(PEX10_MACHR,           0xD00818E4,__READ_WRITE ,__pex_machr_bits);
__IO_REG32_BIT(PEX10_MATR,            0xD00818E8,__READ_WRITE ,__pex_matr_bits);
__IO_REG32_BIT(PEX11_W0CR,            0xD0085820,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX11_W0BR,            0xD0085824,__READ_WRITE );
__IO_REG32_BIT(PEX11_W0RR,            0xD008582C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32_BIT(PEX11_W1CR,            0xD0085830,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX11_W1BR,            0xD0085834,__READ_WRITE );
__IO_REG32_BIT(PEX11_W1RR,            0xD008583C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32_BIT(PEX11_W2CR,            0xD0085840,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX11_W2BR,            0xD0085844,__READ_WRITE );
__IO_REG32_BIT(PEX11_W2RR,            0xD008584C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32_BIT(PEX11_W3CR,            0xD0085850,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX11_W3BR,            0xD0085854,__READ_WRITE );
__IO_REG32_BIT(PEX11_W3RR,            0xD008585C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32_BIT(PEX11_W4CR,            0xD0085860,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX11_W4BR,            0xD0085864,__READ_WRITE );
__IO_REG32_BIT(PEX11_W4RR,            0xD008586C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32(    PEX11_W4RHR,           0xD0085870,__READ_WRITE );
__IO_REG32_BIT(PEX11_W5CR,            0xD0085880,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX11_W5BR,            0xD0085884,__READ_WRITE );
__IO_REG32_BIT(PEX11_W5RR,            0xD008588C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32(    PEX11_W5RHR,           0xD0085890,__READ_WRITE );
__IO_REG32_BIT(PEX11_DWCR,            0xD00858B0,__READ_WRITE ,__pex_dwcr_bits);
__IO_REG32_BIT(PEX11_EROMWCR,         0xD00858C0,__READ_WRITE ,__pex_eromwcr_bits);
__IO_REG32_BIT(PEX11_EROMWRR,         0xD00858C4,__READ_WRITE ,__pex_eromwrr_bits);
__IO_REG32_BIT(PEX11_BAR1CR,          0xD0085804,__READ_WRITE ,__pex_barcr_bits);
__IO_REG32_BIT(PEX11_BAR2CR,          0xD0085808,__READ_WRITE ,__pex_barcr_bits);
__IO_REG32_BIT(PEX11_EROMBARCR,       0xD008580C,__READ_WRITE ,__pex_erombarcr_bits);
__IO_REG32_BIT(PEX11_CAR,             0xD00858F8,__READ_WRITE ,__pex_car_bits);
__IO_REG32(    PEX11_CDR,             0xD00858FC,__READ_WRITE );
__IO_REG32_BIT(PEX11_DVIDR,           0xD0084000,__READ       ,__pex_dvidr_bits);
__IO_REG32_BIT(PEX11_CSR,             0xD0084004,__READ_WRITE ,__pex_csr_bits);
__IO_REG32_BIT(PEX11_CCRIDR,          0xD0084008,__READ       ,__pex_ccridr_bits);
__IO_REG32_BIT(PEX11_BISTHTCLSR,      0xD008400C,__READ_WRITE ,__pex_bisthtclsr_bits);
__IO_REG32_BIT(PEX11_BAR0IR,          0xD0084010,__READ_WRITE ,__pex_barir_bits);
__IO_REG32(    PEX11_BAR0IHR,         0xD0084014,__READ_WRITE );
__IO_REG32_BIT(PEX11_BAR1IR,          0xD0084018,__READ_WRITE ,__pex_barir_bits);
__IO_REG32(    PEX11_BAR1IHR,         0xD008401C,__READ_WRITE );
__IO_REG32_BIT(PEX11_BAR2IR,          0xD0084020,__READ_WRITE ,__pex_barir_bits);
__IO_REG32(    PEX11_BAR2IHR,         0xD0084024,__READ_WRITE );
__IO_REG32_BIT(PEX11_SSDVIDR,         0xD008402C,__READ       ,__pex_ssdvidr_bits);
__IO_REG32_BIT(PEX11_EROMBARR,        0xD0084030,__READ_WRITE ,__pex_erombarr_bits);
__IO_REG32_BIT(PEX11_CLPR,            0xD0084034,__READ       ,__pex_clpr_bits);
__IO_REG32_BIT(PEX11_IPLR,            0xD008403C,__READ_WRITE ,__pex_iplr_bits);
__IO_REG32_BIT(PEX11_PMCHR,           0xD0084040,__READ       ,__pex_pmchr_bits);
__IO_REG32_BIT(PEX11_PMCSR,           0xD0084044,__READ_WRITE ,__pex_pmcsr_bits);
__IO_REG32_BIT(PEX11_MSIMCR,          0xD0084050,__READ_WRITE ,__pex_msimcr_bits);
__IO_REG32(    PEX11_MSIMAR,          0xD0084054,__READ_WRITE );
__IO_REG32(    PEX11_MSIMAHR,         0xD0084058,__READ_WRITE );
__IO_REG32_BIT(PEX11_MSIMDR,          0xD008405C,__READ_WRITE ,__pex_msimdr_bits);
__IO_REG32_BIT(PEX11_CR,              0xD0084060,__READ       ,__pex_cr_bits);
__IO_REG32_BIT(PEX11_DCR,             0xD0084064,__READ       ,__pex_dcr_bits);
__IO_REG32_BIT(PEX11_DCSR,            0xD0084068,__READ_WRITE ,__pex_dcsr_bits);
__IO_REG32_BIT(PEX11_LCR,             0xD008406C,__READ       ,__pex_lcr_bits);
__IO_REG32_BIT(PEX11_LCSR,            0xD0084070,__READ_WRITE ,__pex_lcsr_bits);
__IO_REG32_BIT(PEX11_AERHR,           0xD0084100,__READ_WRITE ,__pex_aerhr_bits);
__IO_REG32_BIT(PEX11_UESTR,           0xD0084104,__READ_WRITE ,__pex_uestr_bits);
__IO_REG32_BIT(PEX11_UEMR,            0xD0084108,__READ_WRITE ,__pex_uemr_bits);
__IO_REG32_BIT(PEX11_UESR,            0xD008410C,__READ_WRITE ,__pex_uesr_bits);
__IO_REG32_BIT(PEX11_CESTR,           0xD0084110,__READ_WRITE ,__pex_cestr_bits);
__IO_REG32_BIT(PEX11_CEMR,            0xD0084114,__READ_WRITE ,__pex_cemr_bits);
__IO_REG32_BIT(PEX11_AECCR,           0xD0084118,__READ       ,__pex_aeccr_bits);
__IO_REG32(    PEX11_HL1R,            0xD008411C,__READ       );
__IO_REG32(    PEX11_HL2R,            0xD0084120,__READ       );
__IO_REG32(    PEX11_HL3R,            0xD0084124,__READ       );
__IO_REG32(    PEX11_HL4R,            0xD0084128,__READ       );
__IO_REG32_BIT(PEX11_CTRLR,           0xD0085A00,__READ_WRITE ,__pex_ctrlr_bits);
__IO_REG32_BIT(PEX11_STR,             0xD0085A04,__READ_WRITE ,__pex_str_bits);
__IO_REG32_BIT(PEX11_RCSSPLR,         0xD0085A0C,__READ_WRITE ,__pex_rcssplr_bits);
__IO_REG32_BIT(PEX11_CTR,             0xD0085A10,__READ_WRITE ,__pex_ctr_bits);
__IO_REG32_BIT(PEX11_RCPMER,          0xD0085A14,__READ       ,__pex_rcpmer_bits);
__IO_REG32_BIT(PEX11_PMR,             0xD0085A18,__READ_WRITE ,__pex_pmr_bits);
__IO_REG32_BIT(PEX11_FCR,             0xD0085A20,__READ_WRITE ,__pex_fcr_bits);
__IO_REG32_BIT(PEX11_AT4R,            0xD0085A30,__READ_WRITE ,__pex_at4r_bits);
__IO_REG32_BIT(PEX11_AT1R,            0xD0085A40,__READ_WRITE ,__pex_at1r_bits);
__IO_REG32_BIT(PEX11_RAMPPCR,         0xD0085A50,__READ_WRITE ,__pex_ramppcr_bits);
__IO_REG32_BIT(PEX11_DBGCR,           0xD0085A60,__READ_WRITE ,__pex_dbgcr_bits);
__IO_REG32_BIT(PEX11_TLCR,            0xD0085AB0,__READ_WRITE ,__pex_tlcr_bits);
__IO_REG32_BIT(PEX11_PHYIAR,          0xD0085B00,__READ_WRITE ,__pex_phyiar_bits);
__IO_REG32_BIT(PEX11_ICR,             0xD0085900,__READ       ,__pex_icr_bits);
__IO_REG32_BIT(PEX11_IMR,             0xD0085910,__READ_WRITE ,__pex_icr_bits);
__IO_REG32_BIT(PEX11_MACR,            0xD00858D0,__READ_WRITE ,__pex_macr_bits);
__IO_REG32_BIT(PEX11_MACLR,           0xD00858E0,__READ_WRITE ,__pex_maclr_bits);
__IO_REG32_BIT(PEX11_MACHR,           0xD00858E4,__READ_WRITE ,__pex_machr_bits);
__IO_REG32_BIT(PEX11_MATR,            0xD00858E8,__READ_WRITE ,__pex_matr_bits);
__IO_REG32_BIT(PEX12_W0CR,            0xD0089820,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX12_W0BR,            0xD0089824,__READ_WRITE );
__IO_REG32_BIT(PEX12_W0RR,            0xD008982C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32_BIT(PEX12_W1CR,            0xD0089830,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX12_W1BR,            0xD0089834,__READ_WRITE );
__IO_REG32_BIT(PEX12_W1RR,            0xD008983C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32_BIT(PEX12_W2CR,            0xD0089840,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX12_W2BR,            0xD0089844,__READ_WRITE );
__IO_REG32_BIT(PEX12_W2RR,            0xD008984C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32_BIT(PEX12_W3CR,            0xD0089850,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX12_W3BR,            0xD0089854,__READ_WRITE );
__IO_REG32_BIT(PEX12_W3RR,            0xD008985C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32_BIT(PEX12_W4CR,            0xD0089860,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX12_W4BR,            0xD0089864,__READ_WRITE );
__IO_REG32_BIT(PEX12_W4RR,            0xD008986C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32(    PEX12_W4RHR,           0xD0089870,__READ_WRITE );
__IO_REG32_BIT(PEX12_W5CR,            0xD0089880,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX12_W5BR,            0xD0089884,__READ_WRITE );
__IO_REG32_BIT(PEX12_W5RR,            0xD008988C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32(    PEX12_W5RHR,           0xD0089890,__READ_WRITE );
__IO_REG32_BIT(PEX12_DWCR,            0xD00898B0,__READ_WRITE ,__pex_dwcr_bits);
__IO_REG32_BIT(PEX12_EROMWCR,         0xD00898C0,__READ_WRITE ,__pex_eromwcr_bits);
__IO_REG32_BIT(PEX12_EROMWRR,         0xD00898C4,__READ_WRITE ,__pex_eromwrr_bits);
__IO_REG32_BIT(PEX12_BAR1CR,          0xD0089804,__READ_WRITE ,__pex_barcr_bits);
__IO_REG32_BIT(PEX12_BAR2CR,          0xD0089808,__READ_WRITE ,__pex_barcr_bits);
__IO_REG32_BIT(PEX12_EROMBARCR,       0xD008980C,__READ_WRITE ,__pex_erombarcr_bits);
__IO_REG32_BIT(PEX12_CAR,             0xD00898F8,__READ_WRITE ,__pex_car_bits);
__IO_REG32(    PEX12_CDR,             0xD00898FC,__READ_WRITE );
__IO_REG32_BIT(PEX12_DVIDR,           0xD0088000,__READ       ,__pex_dvidr_bits);
__IO_REG32_BIT(PEX12_CSR,             0xD0088004,__READ_WRITE ,__pex_csr_bits);
__IO_REG32_BIT(PEX12_CCRIDR,          0xD0088008,__READ       ,__pex_ccridr_bits);
__IO_REG32_BIT(PEX12_BISTHTCLSR,      0xD008800C,__READ_WRITE ,__pex_bisthtclsr_bits);
__IO_REG32_BIT(PEX12_BAR0IR,          0xD0088010,__READ_WRITE ,__pex_barir_bits);
__IO_REG32(    PEX12_BAR0IHR,         0xD0088014,__READ_WRITE );
__IO_REG32_BIT(PEX12_BAR1IR,          0xD0088018,__READ_WRITE ,__pex_barir_bits);
__IO_REG32(    PEX12_BAR1IHR,         0xD008801C,__READ_WRITE );
__IO_REG32_BIT(PEX12_BAR2IR,          0xD0088020,__READ_WRITE ,__pex_barir_bits);
__IO_REG32(    PEX12_BAR2IHR,         0xD0088024,__READ_WRITE );
__IO_REG32_BIT(PEX12_SSDVIDR,         0xD008802C,__READ       ,__pex_ssdvidr_bits);
__IO_REG32_BIT(PEX12_EROMBARR,        0xD0088030,__READ_WRITE ,__pex_erombarr_bits);
__IO_REG32_BIT(PEX12_CLPR,            0xD0088034,__READ       ,__pex_clpr_bits);
__IO_REG32_BIT(PEX12_IPLR,            0xD008803C,__READ_WRITE ,__pex_iplr_bits);
__IO_REG32_BIT(PEX12_PMCHR,           0xD0088040,__READ       ,__pex_pmchr_bits);
__IO_REG32_BIT(PEX12_PMCSR,           0xD0088044,__READ_WRITE ,__pex_pmcsr_bits);
__IO_REG32_BIT(PEX12_MSIMCR,          0xD0088050,__READ_WRITE ,__pex_msimcr_bits);
__IO_REG32(    PEX12_MSIMAR,          0xD0088054,__READ_WRITE );
__IO_REG32(    PEX12_MSIMAHR,         0xD0088058,__READ_WRITE );
__IO_REG32_BIT(PEX12_MSIMDR,          0xD008805C,__READ_WRITE ,__pex_msimdr_bits);
__IO_REG32_BIT(PEX12_CR,              0xD0088060,__READ       ,__pex_cr_bits);
__IO_REG32_BIT(PEX12_DCR,             0xD0088064,__READ       ,__pex_dcr_bits);
__IO_REG32_BIT(PEX12_DCSR,            0xD0088068,__READ_WRITE ,__pex_dcsr_bits);
__IO_REG32_BIT(PEX12_LCR,             0xD008806C,__READ       ,__pex_lcr_bits);
__IO_REG32_BIT(PEX12_LCSR,            0xD0088070,__READ_WRITE ,__pex_lcsr_bits);
__IO_REG32_BIT(PEX12_AERHR,           0xD0088100,__READ_WRITE ,__pex_aerhr_bits);
__IO_REG32_BIT(PEX12_UESTR,           0xD0088104,__READ_WRITE ,__pex_uestr_bits);
__IO_REG32_BIT(PEX12_UEMR,            0xD0088108,__READ_WRITE ,__pex_uemr_bits);
__IO_REG32_BIT(PEX12_UESR,            0xD008810C,__READ_WRITE ,__pex_uesr_bits);
__IO_REG32_BIT(PEX12_CESTR,           0xD0088110,__READ_WRITE ,__pex_cestr_bits);
__IO_REG32_BIT(PEX12_CEMR,            0xD0088114,__READ_WRITE ,__pex_cemr_bits);
__IO_REG32_BIT(PEX12_AECCR,           0xD0088118,__READ       ,__pex_aeccr_bits);
__IO_REG32(    PEX12_HL1R,            0xD008811C,__READ       );
__IO_REG32(    PEX12_HL2R,            0xD0088120,__READ       );
__IO_REG32(    PEX12_HL3R,            0xD0088124,__READ       );
__IO_REG32(    PEX12_HL4R,            0xD0088128,__READ       );
__IO_REG32_BIT(PEX12_CTRLR,           0xD0089A00,__READ_WRITE ,__pex_ctrlr_bits);
__IO_REG32_BIT(PEX12_STR,             0xD0089A04,__READ_WRITE ,__pex_str_bits);
__IO_REG32_BIT(PEX12_RCSSPLR,         0xD0089A0C,__READ_WRITE ,__pex_rcssplr_bits);
__IO_REG32_BIT(PEX12_CTR,             0xD0089A10,__READ_WRITE ,__pex_ctr_bits);
__IO_REG32_BIT(PEX12_RCPMER,          0xD0089A14,__READ       ,__pex_rcpmer_bits);
__IO_REG32_BIT(PEX12_PMR,             0xD0089A18,__READ_WRITE ,__pex_pmr_bits);
__IO_REG32_BIT(PEX12_FCR,             0xD0089A20,__READ_WRITE ,__pex_fcr_bits);
__IO_REG32_BIT(PEX12_AT4R,            0xD0089A30,__READ_WRITE ,__pex_at4r_bits);
__IO_REG32_BIT(PEX12_AT1R,            0xD0089A40,__READ_WRITE ,__pex_at1r_bits);
__IO_REG32_BIT(PEX12_RAMPPCR,         0xD0089A50,__READ_WRITE ,__pex_ramppcr_bits);
__IO_REG32_BIT(PEX12_DBGCR,           0xD0089A60,__READ_WRITE ,__pex_dbgcr_bits);
__IO_REG32_BIT(PEX12_TLCR,            0xD0089AB0,__READ_WRITE ,__pex_tlcr_bits);
__IO_REG32_BIT(PEX12_PHYIAR,          0xD0089B00,__READ_WRITE ,__pex_phyiar_bits);
__IO_REG32_BIT(PEX12_ICR,             0xD0089900,__READ       ,__pex_icr_bits);
__IO_REG32_BIT(PEX12_IMR,             0xD0089910,__READ_WRITE ,__pex_icr_bits);
__IO_REG32_BIT(PEX12_MACR,            0xD00898D0,__READ_WRITE ,__pex_macr_bits);
__IO_REG32_BIT(PEX12_MACLR,           0xD00898E0,__READ_WRITE ,__pex_maclr_bits);
__IO_REG32_BIT(PEX12_MACHR,           0xD00898E4,__READ_WRITE ,__pex_machr_bits);
__IO_REG32_BIT(PEX12_MATR,            0xD00898E8,__READ_WRITE ,__pex_matr_bits);
__IO_REG32_BIT(PEX13_W0CR,            0xD008D820,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX13_W0BR,            0xD008D824,__READ_WRITE );
__IO_REG32_BIT(PEX13_W0RR,            0xD008D82C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32_BIT(PEX13_W1CR,            0xD008D830,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX13_W1BR,            0xD008D834,__READ_WRITE );
__IO_REG32_BIT(PEX13_W1RR,            0xD008D83C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32_BIT(PEX13_W2CR,            0xD008D840,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX13_W2BR,            0xD008D844,__READ_WRITE );
__IO_REG32_BIT(PEX13_W2RR,            0xD008D84C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32_BIT(PEX13_W3CR,            0xD008D850,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX13_W3BR,            0xD008D854,__READ_WRITE );
__IO_REG32_BIT(PEX13_W3RR,            0xD008D85C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32_BIT(PEX13_W4CR,            0xD008D860,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX13_W4BR,            0xD008D864,__READ_WRITE );
__IO_REG32_BIT(PEX13_W4RR,            0xD008D86C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32(    PEX13_W4RHR,           0xD008D870,__READ_WRITE );
__IO_REG32_BIT(PEX13_W5CR,            0xD008D880,__READ_WRITE ,__pex_wcr_bits);
__IO_REG32(    PEX13_W5BR,            0xD008D884,__READ_WRITE );
__IO_REG32_BIT(PEX13_W5RR,            0xD008D88C,__READ_WRITE ,__pex_wrr_bits);
__IO_REG32(    PEX13_W5RHR,           0xD008D890,__READ_WRITE );
__IO_REG32_BIT(PEX13_DWCR,            0xD008D8B0,__READ_WRITE ,__pex_dwcr_bits);
__IO_REG32_BIT(PEX13_EROMWCR,         0xD008D8C0,__READ_WRITE ,__pex_eromwcr_bits);
__IO_REG32_BIT(PEX13_EROMWRR,         0xD008D8C4,__READ_WRITE ,__pex_eromwrr_bits);
__IO_REG32_BIT(PEX13_BAR1CR,          0xD008D804,__READ_WRITE ,__pex_barcr_bits);
__IO_REG32_BIT(PEX13_BAR2CR,          0xD008D808,__READ_WRITE ,__pex_barcr_bits);
__IO_REG32_BIT(PEX13_EROMBARCR,       0xD008D80C,__READ_WRITE ,__pex_erombarcr_bits);
__IO_REG32_BIT(PEX13_CAR,             0xD008D8F8,__READ_WRITE ,__pex_car_bits);
__IO_REG32(    PEX13_CDR,             0xD008D8FC,__READ_WRITE );
__IO_REG32_BIT(PEX13_DVIDR,           0xD008C000,__READ       ,__pex_dvidr_bits);
__IO_REG32_BIT(PEX13_CSR,             0xD008C004,__READ_WRITE ,__pex_csr_bits);
__IO_REG32_BIT(PEX13_CCRIDR,          0xD008C008,__READ       ,__pex_ccridr_bits);
__IO_REG32_BIT(PEX13_BISTHTCLSR,      0xD008C00C,__READ_WRITE ,__pex_bisthtclsr_bits);
__IO_REG32_BIT(PEX13_BAR0IR,          0xD008C010,__READ_WRITE ,__pex_barir_bits);
__IO_REG32(    PEX13_BAR0IHR,         0xD008C014,__READ_WRITE );
__IO_REG32_BIT(PEX13_BAR1IR,          0xD008C018,__READ_WRITE ,__pex_barir_bits);
__IO_REG32(    PEX13_BAR1IHR,         0xD008C01C,__READ_WRITE );
__IO_REG32_BIT(PEX13_BAR2IR,          0xD008C020,__READ_WRITE ,__pex_barir_bits);
__IO_REG32(    PEX13_BAR2IHR,         0xD008C024,__READ_WRITE );
__IO_REG32_BIT(PEX13_SSDVIDR,         0xD008C02C,__READ       ,__pex_ssdvidr_bits);
__IO_REG32_BIT(PEX13_EROMBARR,        0xD008C030,__READ_WRITE ,__pex_erombarr_bits);
__IO_REG32_BIT(PEX13_CLPR,            0xD008C034,__READ       ,__pex_clpr_bits);
__IO_REG32_BIT(PEX13_IPLR,            0xD008C03C,__READ_WRITE ,__pex_iplr_bits);
__IO_REG32_BIT(PEX13_PMCHR,           0xD008C040,__READ       ,__pex_pmchr_bits);
__IO_REG32_BIT(PEX13_PMCSR,           0xD008C044,__READ_WRITE ,__pex_pmcsr_bits);
__IO_REG32_BIT(PEX13_MSIMCR,          0xD008C050,__READ_WRITE ,__pex_msimcr_bits);
__IO_REG32(    PEX13_MSIMAR,          0xD008C054,__READ_WRITE );
__IO_REG32(    PEX13_MSIMAHR,         0xD008C058,__READ_WRITE );
__IO_REG32_BIT(PEX13_MSIMDR,          0xD008C05C,__READ_WRITE ,__pex_msimdr_bits);
__IO_REG32_BIT(PEX13_CR,              0xD008C060,__READ       ,__pex_cr_bits);
__IO_REG32_BIT(PEX13_DCR,             0xD008C064,__READ       ,__pex_dcr_bits);
__IO_REG32_BIT(PEX13_DCSR,            0xD008C068,__READ_WRITE ,__pex_dcsr_bits);
__IO_REG32_BIT(PEX13_LCR,             0xD008C06C,__READ       ,__pex_lcr_bits);
__IO_REG32_BIT(PEX13_LCSR,            0xD008C070,__READ_WRITE ,__pex_lcsr_bits);
__IO_REG32_BIT(PEX13_AERHR,           0xD008C100,__READ_WRITE ,__pex_aerhr_bits);
__IO_REG32_BIT(PEX13_UESTR,           0xD008C104,__READ_WRITE ,__pex_uestr_bits);
__IO_REG32_BIT(PEX13_UEMR,            0xD008C108,__READ_WRITE ,__pex_uemr_bits);
__IO_REG32_BIT(PEX13_UESR,            0xD008C10C,__READ_WRITE ,__pex_uesr_bits);
__IO_REG32_BIT(PEX13_CESTR,           0xD008C110,__READ_WRITE ,__pex_cestr_bits);
__IO_REG32_BIT(PEX13_CEMR,            0xD008C114,__READ_WRITE ,__pex_cemr_bits);
__IO_REG32_BIT(PEX13_AECCR,           0xD008C118,__READ       ,__pex_aeccr_bits);
__IO_REG32(    PEX13_HL1R,            0xD008C11C,__READ       );
__IO_REG32(    PEX13_HL2R,            0xD008C120,__READ       );
__IO_REG32(    PEX13_HL3R,            0xD008C124,__READ       );
__IO_REG32(    PEX13_HL4R,            0xD008C128,__READ       );
__IO_REG32_BIT(PEX13_CTRLR,           0xD008DA00,__READ_WRITE ,__pex_ctrlr_bits);
__IO_REG32_BIT(PEX13_STR,             0xD008DA04,__READ_WRITE ,__pex_str_bits);
__IO_REG32_BIT(PEX13_RCSSPLR,         0xD008DA0C,__READ_WRITE ,__pex_rcssplr_bits);
__IO_REG32_BIT(PEX13_CTR,             0xD008DA10,__READ_WRITE ,__pex_ctr_bits);
__IO_REG32_BIT(PEX13_RCPMER,          0xD008DA14,__READ       ,__pex_rcpmer_bits);
__IO_REG32_BIT(PEX13_PMR,             0xD008DA18,__READ_WRITE ,__pex_pmr_bits);
__IO_REG32_BIT(PEX13_FCR,             0xD008DA20,__READ_WRITE ,__pex_fcr_bits);
__IO_REG32_BIT(PEX13_AT4R,            0xD008DA30,__READ_WRITE ,__pex_at4r_bits);
__IO_REG32_BIT(PEX13_AT1R,            0xD008DA40,__READ_WRITE ,__pex_at1r_bits);
__IO_REG32_BIT(PEX13_RAMPPCR,         0xD008DA50,__READ_WRITE ,__pex_ramppcr_bits);
__IO_REG32_BIT(PEX13_DBGCR,           0xD008DA60,__READ_WRITE ,__pex_dbgcr_bits);
__IO_REG32_BIT(PEX13_TLCR,            0xD008DAB0,__READ_WRITE ,__pex_tlcr_bits);
__IO_REG32_BIT(PEX13_PHYIAR,          0xD008DB00,__READ_WRITE ,__pex_phyiar_bits);
__IO_REG32_BIT(PEX13_ICR,             0xD008D900,__READ       ,__pex_icr_bits);
__IO_REG32_BIT(PEX13_IMR,             0xD008D910,__READ_WRITE ,__pex_icr_bits);
__IO_REG32_BIT(PEX13_MACR,            0xD008D8D0,__READ_WRITE ,__pex_macr_bits);
__IO_REG32_BIT(PEX13_MACLR,           0xD008D8E0,__READ_WRITE ,__pex_maclr_bits);
__IO_REG32_BIT(PEX13_MACHR,           0xD008D8E4,__READ_WRITE ,__pex_machr_bits);
__IO_REG32_BIT(PEX13_MATR,            0xD008D8E8,__READ_WRITE ,__pex_matr_bits);

/***************************************************************************
 **
 ** GbE2
 **
 ***************************************************************************/
__IO_REG32_BIT(GbE2_PHYAR,                    0xD0032000,__READ_WRITE ,__gbe_phyar_bits);
__IO_REG32_BIT(GbE2_SIMR,                     0xD0032004,__READ_WRITE ,__gbe_simr_bits);
__IO_REG32(    GbE2_EUDA,                     0xD0032008,__READ_WRITE );
__IO_REG32_BIT(GbE2_EUDID,                    0xD003200C,__READ_WRITE ,__gbe_eudid_bits);
__IO_REG32_BIT(GbE2_EUIC,                     0xD0032080,__READ       ,__gbe_euic_bits);
__IO_REG32_BIT(GbE2_EUIM,                     0xD0032084,__READ_WRITE ,__gbe_euim_bits);
__IO_REG32(    GbE2_EUEA,                     0xD0032094,__READ       );
__IO_REG32_BIT(GbE2_EUIAE,                    0xD0032098,__READ       ,__gbe_euiae_bits);
__IO_REG32_BIT(GbE2_EUC,                      0xD00320B0,__READ_WRITE ,__gbe_euc_bits);
__IO_REG32_BIT(GbE2_BA0,                      0xD0032200,__READ_WRITE ,__gbe_ba_bits);
__IO_REG32_BIT(GbE2_BA1,                      0xD0032208,__READ_WRITE ,__gbe_ba_bits);
__IO_REG32_BIT(GbE2_BA2,                      0xD0032210,__READ_WRITE ,__gbe_ba_bits);
__IO_REG32_BIT(GbE2_BA3,                      0xD0032218,__READ_WRITE ,__gbe_ba_bits);
__IO_REG32_BIT(GbE2_BA4,                      0xD0032220,__READ_WRITE ,__gbe_ba_bits);
__IO_REG32_BIT(GbE2_BA5,                      0xD0032228,__READ_WRITE ,__gbe_ba_bits);
__IO_REG32_BIT(GbE2_SR0,                      0xD0032204,__READ_WRITE ,__gbe_sr_bits);
__IO_REG32_BIT(GbE2_SR1,                      0xD003220C,__READ_WRITE ,__gbe_sr_bits);
__IO_REG32_BIT(GbE2_SR2,                      0xD0032214,__READ_WRITE ,__gbe_sr_bits);
__IO_REG32_BIT(GbE2_SR3,                      0xD003221C,__READ_WRITE ,__gbe_sr_bits);
__IO_REG32_BIT(GbE2_SR4,                      0xD0032224,__READ_WRITE ,__gbe_sr_bits);
__IO_REG32_BIT(GbE2_SR5,                      0xD003222C,__READ_WRITE ,__gbe_sr_bits);
__IO_REG32(    GbE2_HARR0,                    0xD0032280,__READ_WRITE );
__IO_REG32(    GbE2_HARR1,                    0xD0032284,__READ_WRITE );
__IO_REG32(    GbE2_HARR2,                    0xD0032288,__READ_WRITE );
__IO_REG32(    GbE2_HARR3,                    0xD003228C,__READ_WRITE );
__IO_REG32_BIT(GbE2_BARE,                     0xD0032290,__READ_WRITE ,__gbe_bare_bits);
__IO_REG32_BIT(GbE2_EPAP,                     0xD0032294,__READ_WRITE ,__gbe_epap_bits);
__IO_REG32_BIT(GbE2_PxC,                      0xD0032400,__READ_WRITE ,__gbe_pxc_bits);
__IO_REG32_BIT(GbE2_PxCX,                     0xD0032404,__READ_WRITE ,__gbe_pxcx_bits);
__IO_REG32_BIT(GbE2_MIISPR,                   0xD0032408,__READ_WRITE ,__gbe_miispr_bits);
__IO_REG32_BIT(GbE2_EVLANE,                   0xD0032410,__READ_WRITE ,__gbe_evlane_bits);
__IO_REG32_BIT(GbE2_MACAL,                    0xD0032414,__READ_WRITE ,__gbe_macal_bits);
__IO_REG32(    GbE2_MACAH,                    0xD0032418,__READ_WRITE );
__IO_REG32_BIT(GbE2_SDC,                      0xD003241C,__READ_WRITE ,__gbe_sdc_bits);
__IO_REG32_BIT(GbE2_DSCP0,                    0xD0032420,__READ_WRITE ,__gbe_dscp_bits);
__IO_REG32_BIT(GbE2_DSCP1,                    0xD0032424,__READ_WRITE ,__gbe_dscp_bits);
__IO_REG32_BIT(GbE2_DSCP2,                    0xD0032428,__READ_WRITE ,__gbe_dscp_bits);
__IO_REG32_BIT(GbE2_DSCP3,                    0xD003242C,__READ_WRITE ,__gbe_dscp_bits);
__IO_REG32_BIT(GbE2_DSCP4,                    0xD0032430,__READ_WRITE ,__gbe_dscp_bits);
__IO_REG32_BIT(GbE2_DSCP5,                    0xD0032434,__READ_WRITE ,__gbe_dscp_bits);
__IO_REG32_BIT(GbE2_DSCP6,                    0xD0032438,__READ_WRITE ,__gbe_dscp_bits);
__IO_REG32_BIT(GbE2_PSC0,                     0xD003243C,__READ_WRITE ,__gbe_psc0_bits);
__IO_REG32_BIT(GbE2_VPT2P,                    0xD0032440,__READ_WRITE ,__gbe_vpt2p_bits);
__IO_REG32_BIT(GbE2_PS0,                      0xD0032444,__READ_WRITE ,__gbe_ps0_bits);
__IO_REG32_BIT(GbE2_TQC,                      0xD0032448,__READ_WRITE ,__gbe_tqc_bits);
__IO_REG32_BIT(GbE2_PSC1,                     0xD003244C,__READ_WRITE ,__gbe_psc1_bits);
__IO_REG32_BIT(GbE2_PS1,                      0xD0032450,__READ_WRITE ,__gbe_ps1_bits);
__IO_REG32_BIT(GbE2_MHR,                      0xD0032454,__READ_WRITE ,__gbe_mhr_bits);
__IO_REG32_BIT(GbE2_IC,                       0xD0032460,__READ_WRITE ,__gbe_ic_bits);
__IO_REG32_BIT(GbE2_ICE,                      0xD0032464,__READ       ,__gbe_ice_bits);
__IO_REG32_BIT(GbE2_PIM,                      0xD0032468,__READ_WRITE ,__gbe_ic_bits);
__IO_REG32_BIT(GbE2_PEIM,                     0xD003246C,__READ_WRITE ,__gbe_ice_bits);
__IO_REG32_BIT(GbE2_PxTFUT,                   0xD0032474,__READ_WRITE ,__gbe_pxtfut_bits);
__IO_REG32_BIT(GbE2_PxMFS,                    0xD003247C,__READ_WRITE ,__gbe_pxmfs_bits);
__IO_REG32(    GbE2_PxDFC,                    0xD0032484,__READ       );
__IO_REG32(    GbE2_PxOFC,                    0xD0032488,__READ       );
__IO_REG32_BIT(GbE2_PIAE,                     0xD0032494,__READ       ,__gbe_piae_bits);
__IO_REG32_BIT(GbE2_ETPR,                     0xD00324BC,__READ_WRITE ,__gbe_etpr_bits);
__IO_REG32_BIT(GbE2_TQFPC,                    0xD00324DC,__READ_WRITE ,__gbe_tqfpc_bits);
__IO_REG32_BIT(GbE2_PTTBRC,                   0xD00324E0,__READ_WRITE ,__gbe_pttbrc_bits);
__IO_REG32_BIT(GbE2_MTU,                      0xD00324E8,__READ_WRITE ,__gbe_mtu_bits);
__IO_REG32_BIT(GbE2_PMTBS,                    0xD00324EC,__READ_WRITE ,__gbe_pmtbs_bits);
__IO_REG32(    GbE2_CRDP0,                    0xD003260C,__READ_WRITE );
__IO_REG32(    GbE2_CRDP1,                    0xD003261C,__READ_WRITE );
__IO_REG32(    GbE2_CRDP2,                    0xD003262C,__READ_WRITE );
__IO_REG32(    GbE2_CRDP3,                    0xD003263C,__READ_WRITE );
__IO_REG32(    GbE2_CRDP4,                    0xD003264C,__READ_WRITE );
__IO_REG32(    GbE2_CRDP5,                    0xD003265C,__READ_WRITE );
__IO_REG32(    GbE2_CRDP6,                    0xD003266C,__READ_WRITE );
__IO_REG32(    GbE2_CRDP7,                    0xD003267C,__READ_WRITE );
__IO_REG32_BIT(GbE2_RQC,                      0xD0032680,__READ_WRITE ,__gbe_rqc_bits);
__IO_REG32(    GbE2_TCSDPR,                   0xD0032684,__READ       );
__IO_REG32(    GbE2_TCQDP0,                   0xD00326C0,__READ_WRITE );
__IO_REG32(    GbE2_TCQDP1,                   0xD00326C4,__READ_WRITE );
__IO_REG32(    GbE2_TCQDP2,                   0xD00326C8,__READ_WRITE );
__IO_REG32(    GbE2_TCQDP3,                   0xD00326CC,__READ_WRITE );
__IO_REG32(    GbE2_TCQDP4,                   0xD00326D0,__READ_WRITE );
__IO_REG32(    GbE2_TCQDP5,                   0xD00326D4,__READ_WRITE );
__IO_REG32(    GbE2_TCQDP6,                   0xD00326D8,__READ_WRITE );
__IO_REG32(    GbE2_TCQDP7,                   0xD00326DC,__READ_WRITE );
__IO_REG32_BIT(GbE2_TQxTBCNT0,                0xD0032700,__READ_WRITE ,__gbe_tqxtbcnt_bits);
__IO_REG32_BIT(GbE2_TQxTBCNT1,                0xD0032710,__READ_WRITE ,__gbe_tqxtbcnt_bits);
__IO_REG32_BIT(GbE2_TQxTBCNT2,                0xD0032720,__READ_WRITE ,__gbe_tqxtbcnt_bits);
__IO_REG32_BIT(GbE2_TQxTBCNT3,                0xD0032730,__READ_WRITE ,__gbe_tqxtbcnt_bits);
__IO_REG32_BIT(GbE2_TQxTBCNT4,                0xD0032740,__READ_WRITE ,__gbe_tqxtbcnt_bits);
__IO_REG32_BIT(GbE2_TQxTBCNT5,                0xD0032750,__READ_WRITE ,__gbe_tqxtbcnt_bits);
__IO_REG32_BIT(GbE2_TQxTBCNT6,                0xD0032760,__READ_WRITE ,__gbe_tqxtbcnt_bits);
__IO_REG32_BIT(GbE2_TQxTBCNT7,                0xD0032770,__READ_WRITE ,__gbe_tqxtbcnt_bits);
__IO_REG32_BIT(GbE2_TQxTBC0,                  0xD0032704,__READ_WRITE ,__gbe_tqxtbc_bits);
__IO_REG32_BIT(GbE2_TQxTBC1,                  0xD0032714,__READ_WRITE ,__gbe_tqxtbc_bits);
__IO_REG32_BIT(GbE2_TQxTBC2,                  0xD0032724,__READ_WRITE ,__gbe_tqxtbc_bits);
__IO_REG32_BIT(GbE2_TQxTBC3,                  0xD0032734,__READ_WRITE ,__gbe_tqxtbc_bits);
__IO_REG32_BIT(GbE2_TQxTBC4,                  0xD0032744,__READ_WRITE ,__gbe_tqxtbc_bits);
__IO_REG32_BIT(GbE2_TQxTBC5,                  0xD0032754,__READ_WRITE ,__gbe_tqxtbc_bits);
__IO_REG32_BIT(GbE2_TQxTBC6,                  0xD0032764,__READ_WRITE ,__gbe_tqxtbc_bits);
__IO_REG32_BIT(GbE2_TQxTBC7,                  0xD0032774,__READ_WRITE ,__gbe_tqxtbc_bits);
__IO_REG32_BIT(GbE2_TQxAC0,                   0xD0032708,__READ_WRITE ,__gbe_tqxac_bits);
__IO_REG32_BIT(GbE2_TQxAC1,                   0xD0032718,__READ_WRITE ,__gbe_tqxac_bits);
__IO_REG32_BIT(GbE2_TQxAC2,                   0xD0032728,__READ_WRITE ,__gbe_tqxac_bits);
__IO_REG32_BIT(GbE2_TQxAC3,                   0xD0032738,__READ_WRITE ,__gbe_tqxac_bits);
__IO_REG32_BIT(GbE2_TQxAC4,                   0xD0032748,__READ_WRITE ,__gbe_tqxac_bits);
__IO_REG32_BIT(GbE2_TQxAC5,                   0xD0032758,__READ_WRITE ,__gbe_tqxac_bits);
__IO_REG32_BIT(GbE2_TQxAC6,                   0xD0032768,__READ_WRITE ,__gbe_tqxac_bits);
__IO_REG32_BIT(GbE2_TQxAC7,                   0xD0032778,__READ_WRITE ,__gbe_tqxac_bits);
__IO_REG32_BIT(GbE2_PTTBC,                    0xD0032780,__READ_WRITE ,__gbe_pttbc_bits);
__IO_REG32_BIT(GbE2_DFSMT0,                   0xD0033400,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT1,                   0xD0033404,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT2,                   0xD0033408,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT3,                   0xD003340C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT4,                   0xD0033410,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT5,                   0xD0033414,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT6,                   0xD0033418,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT7,                   0xD003341C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT8,                   0xD0033420,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT9,                   0xD0033424,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT10,                  0xD0033428,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT11,                  0xD003342C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT12,                  0xD0033430,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT13,                  0xD0033434,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT14,                  0xD0033438,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT15,                  0xD003343C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT16,                  0xD0033440,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT17,                  0xD0033444,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT18,                  0xD0033448,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT19,                  0xD003344C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT20,                  0xD0033450,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT21,                  0xD0033454,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT22,                  0xD0033458,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT23,                  0xD003345C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT24,                  0xD0033460,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT25,                  0xD0033464,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT26,                  0xD0033468,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT27,                  0xD003346C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT28,                  0xD0033470,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT29,                  0xD0033474,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT30,                  0xD0033478,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT31,                  0xD003347C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT32,                  0xD0033480,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT33,                  0xD0033484,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT34,                  0xD0033488,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT35,                  0xD003348C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT36,                  0xD0033490,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT37,                  0xD0033494,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT38,                  0xD0033498,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT39,                  0xD003349C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT40,                  0xD00334A0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT41,                  0xD00334A4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT42,                  0xD00334A8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT43,                  0xD00334AC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT44,                  0xD00334B0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT45,                  0xD00334B4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT46,                  0xD00334B8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT47,                  0xD00334BC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT48,                  0xD00334C0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT49,                  0xD00334C4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT50,                  0xD00334C8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT51,                  0xD00334CC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT52,                  0xD00334D0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT53,                  0xD00334D4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT54,                  0xD00334D8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT55,                  0xD00334DC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT56,                  0xD00334E0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT57,                  0xD00334E4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT58,                  0xD00334E8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT59,                  0xD00334EC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT60,                  0xD00334F0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT61,                  0xD00334F4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT62,                  0xD00334F8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFSMT63,                  0xD00334FC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT0,                   0xD0033500,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT1,                   0xD0033504,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT2,                   0xD0033508,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT3,                   0xD003350C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT4,                   0xD0033510,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT5,                   0xD0033514,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT6,                   0xD0033518,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT7,                   0xD003351C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT8,                   0xD0033520,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT9,                   0xD0033524,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT10,                  0xD0033528,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT11,                  0xD003352C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT12,                  0xD0033530,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT13,                  0xD0033534,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT14,                  0xD0033538,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT15,                  0xD003353C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT16,                  0xD0033540,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT17,                  0xD0033544,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT18,                  0xD0033548,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT19,                  0xD003354C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT20,                  0xD0033550,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT21,                  0xD0033554,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT22,                  0xD0033558,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT23,                  0xD003355C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT24,                  0xD0033560,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT25,                  0xD0033564,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT26,                  0xD0033568,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT27,                  0xD003356C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT28,                  0xD0033570,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT29,                  0xD0033574,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT30,                  0xD0033578,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT31,                  0xD003357C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT32,                  0xD0033580,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT33,                  0xD0033584,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT34,                  0xD0033588,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT35,                  0xD003358C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT36,                  0xD0033590,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT37,                  0xD0033594,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT38,                  0xD0033598,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT39,                  0xD003359C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT40,                  0xD00335A0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT41,                  0xD00335A4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT42,                  0xD00335A8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT43,                  0xD00335AC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT44,                  0xD00335B0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT45,                  0xD00335B4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT46,                  0xD00335B8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT47,                  0xD00335BC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT48,                  0xD00335C0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT49,                  0xD00335C4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT50,                  0xD00335C8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT51,                  0xD00335CC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT52,                  0xD00335D0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT53,                  0xD00335D4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT54,                  0xD00335D8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT55,                  0xD00335DC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT56,                  0xD00335E0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT57,                  0xD00335E4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT58,                  0xD00335E8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT59,                  0xD00335EC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT60,                  0xD00335F0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT61,                  0xD00335F4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT62,                  0xD00335F8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFOMT63,                  0xD00335FC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFUT0,                    0xD0033600,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFUT1,                    0xD0033604,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFUT2,                    0xD0033608,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE2_DFUT3,                    0xD003360C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32(    GbE2_GoodOctetsReceivedL,      0xD0033000,__READ_WRITE );
__IO_REG32(    GbE2_GoodOctetsReceivedH,      0xD0033004,__READ_WRITE );
__IO_REG32(    GbE2_BadOctetsReceived,        0xD0033008,__READ_WRITE );
__IO_REG32(    GbE2_MACTransError,            0xD003300C,__READ_WRITE );
__IO_REG32(    GbE2_GoodFramesReceived,       0xD0033010,__READ_WRITE );
__IO_REG32(    GbE2_BadFramesReceived,        0xD0033014,__READ_WRITE );
__IO_REG32(    GbE2_BroadcastFramesReceived,  0xD0033018,__READ_WRITE );
__IO_REG32(    GbE2_MulticastFramesReceived,  0xD003301C,__READ_WRITE );
__IO_REG32(    GbE2_Frames64Octets,           0xD0033020,__READ_WRITE );
__IO_REG32(    GbE2_Frames65to127Octets,      0xD0033024,__READ_WRITE );
__IO_REG32(    GbE2_Frames128to255Octets,     0xD0033028,__READ_WRITE );
__IO_REG32(    GbE2_Frames256to511Octets,     0xD003302C,__READ_WRITE );
__IO_REG32(    GbE2_Frames512to1023Octets,    0xD0033030,__READ_WRITE );
__IO_REG32(    GbE2_Frames1024toMaxOctets,    0xD0033034,__READ_WRITE );
__IO_REG32(    GbE2_GoodOctetsSentL,          0xD0033038,__READ_WRITE );
__IO_REG32(    GbE2_GoodOctetsSentH,          0xD003303C,__READ_WRITE );
__IO_REG32(    GbE2_GoodFramesSent,           0xD0033040,__READ_WRITE );
__IO_REG32(    GbE2_ExcessiveCollision,       0xD0033044,__READ_WRITE );
__IO_REG32(    GbE2_MulticastFramesSent,      0xD0033048,__READ_WRITE );
__IO_REG32(    GbE2_BroadcastFramesSent,      0xD003304C,__READ_WRITE );
__IO_REG32(    GbE2_UnrecogMACControl,        0xD0033050,__READ_WRITE );
__IO_REG32(    GbE2_FCSent,                   0xD0033054,__READ_WRITE );
__IO_REG32(    GbE2_GoodFCReceived,           0xD0033058,__READ_WRITE );
__IO_REG32(    GbE2_BadFCReceived,            0xD003305C,__READ_WRITE );
__IO_REG32(    GbE2_Undersize,                0xD0033060,__READ_WRITE );
__IO_REG32(    GbE2_Fragments,                0xD0033064,__READ_WRITE );
__IO_REG32(    GbE2_Oversize,                 0xD0033068,__READ_WRITE );
__IO_REG32(    GbE2_Jabber,                   0xD003306C,__READ_WRITE );
__IO_REG32(    GbE2_MACRcvError,              0xD0033070,__READ_WRITE );
__IO_REG32(    GbE2_BadCRC,                   0xD0033074,__READ_WRITE );
__IO_REG32(    GbE2_Collisions,               0xD0033078,__READ_WRITE );
__IO_REG32(    GbE2_LateCollision,            0xD003307C,__READ_WRITE );

/***************************************************************************
 **
 ** GbE3
 **
 ***************************************************************************/
__IO_REG32_BIT(GbE3_PHYAR,                    0xD0036000,__READ_WRITE ,__gbe_phyar_bits);
__IO_REG32_BIT(GbE3_SIMR,                     0xD0036004,__READ_WRITE ,__gbe_simr_bits);
__IO_REG32(    GbE3_EUDA,                     0xD0036008,__READ_WRITE );
__IO_REG32_BIT(GbE3_EUDID,                    0xD003600C,__READ_WRITE ,__gbe_eudid_bits);
__IO_REG32_BIT(GbE3_EUIC,                     0xD0036080,__READ       ,__gbe_euic_bits);
__IO_REG32_BIT(GbE3_EUIM,                     0xD0036084,__READ_WRITE ,__gbe_euim_bits);
__IO_REG32(    GbE3_EUEA,                     0xD0036094,__READ       );
__IO_REG32_BIT(GbE3_EUIAE,                    0xD0036098,__READ       ,__gbe_euiae_bits);
__IO_REG32_BIT(GbE3_EUC,                      0xD00360B0,__READ_WRITE ,__gbe_euc_bits);
__IO_REG32_BIT(GbE3_BA0,                      0xD0036200,__READ_WRITE ,__gbe_ba_bits);
__IO_REG32_BIT(GbE3_BA1,                      0xD0036208,__READ_WRITE ,__gbe_ba_bits);
__IO_REG32_BIT(GbE3_BA2,                      0xD0036210,__READ_WRITE ,__gbe_ba_bits);
__IO_REG32_BIT(GbE3_BA3,                      0xD0036218,__READ_WRITE ,__gbe_ba_bits);
__IO_REG32_BIT(GbE3_BA4,                      0xD0036220,__READ_WRITE ,__gbe_ba_bits);
__IO_REG32_BIT(GbE3_BA5,                      0xD0036228,__READ_WRITE ,__gbe_ba_bits);
__IO_REG32_BIT(GbE3_SR0,                      0xD0036204,__READ_WRITE ,__gbe_sr_bits);
__IO_REG32_BIT(GbE3_SR1,                      0xD003620C,__READ_WRITE ,__gbe_sr_bits);
__IO_REG32_BIT(GbE3_SR2,                      0xD0036214,__READ_WRITE ,__gbe_sr_bits);
__IO_REG32_BIT(GbE3_SR3,                      0xD003621C,__READ_WRITE ,__gbe_sr_bits);
__IO_REG32_BIT(GbE3_SR4,                      0xD0036224,__READ_WRITE ,__gbe_sr_bits);
__IO_REG32_BIT(GbE3_SR5,                      0xD003622C,__READ_WRITE ,__gbe_sr_bits);
__IO_REG32(    GbE3_HARR0,                    0xD0036280,__READ_WRITE );
__IO_REG32(    GbE3_HARR1,                    0xD0036284,__READ_WRITE );
__IO_REG32(    GbE3_HARR2,                    0xD0036288,__READ_WRITE );
__IO_REG32(    GbE3_HARR3,                    0xD003628C,__READ_WRITE );
__IO_REG32_BIT(GbE3_BARE,                     0xD0036290,__READ_WRITE ,__gbe_bare_bits);
__IO_REG32_BIT(GbE3_EPAP,                     0xD0036294,__READ_WRITE ,__gbe_epap_bits);
__IO_REG32_BIT(GbE3_PxC,                      0xD0036400,__READ_WRITE ,__gbe_pxc_bits);
__IO_REG32_BIT(GbE3_PxCX,                     0xD0036404,__READ_WRITE ,__gbe_pxcx_bits);
__IO_REG32_BIT(GbE3_MIISPR,                   0xD0036408,__READ_WRITE ,__gbe_miispr_bits);
__IO_REG32_BIT(GbE3_EVLANE,                   0xD0036410,__READ_WRITE ,__gbe_evlane_bits);
__IO_REG32_BIT(GbE3_MACAL,                    0xD0036414,__READ_WRITE ,__gbe_macal_bits);
__IO_REG32(    GbE3_MACAH,                    0xD0036418,__READ_WRITE );
__IO_REG32_BIT(GbE3_SDC,                      0xD003641C,__READ_WRITE ,__gbe_sdc_bits);
__IO_REG32_BIT(GbE3_DSCP0,                    0xD0036420,__READ_WRITE ,__gbe_dscp_bits);
__IO_REG32_BIT(GbE3_DSCP1,                    0xD0036424,__READ_WRITE ,__gbe_dscp_bits);
__IO_REG32_BIT(GbE3_DSCP2,                    0xD0036428,__READ_WRITE ,__gbe_dscp_bits);
__IO_REG32_BIT(GbE3_DSCP3,                    0xD003642C,__READ_WRITE ,__gbe_dscp_bits);
__IO_REG32_BIT(GbE3_DSCP4,                    0xD0036430,__READ_WRITE ,__gbe_dscp_bits);
__IO_REG32_BIT(GbE3_DSCP5,                    0xD0036434,__READ_WRITE ,__gbe_dscp_bits);
__IO_REG32_BIT(GbE3_DSCP6,                    0xD0036438,__READ_WRITE ,__gbe_dscp_bits);
__IO_REG32_BIT(GbE3_PSC0,                     0xD003643C,__READ_WRITE ,__gbe_psc0_bits);
__IO_REG32_BIT(GbE3_VPT2P,                    0xD0036440,__READ_WRITE ,__gbe_vpt2p_bits);
__IO_REG32_BIT(GbE3_PS0,                      0xD0036444,__READ_WRITE ,__gbe_ps0_bits);
__IO_REG32_BIT(GbE3_TQC,                      0xD0036448,__READ_WRITE ,__gbe_tqc_bits);
__IO_REG32_BIT(GbE3_PSC1,                     0xD003644C,__READ_WRITE ,__gbe_psc1_bits);
__IO_REG32_BIT(GbE3_PS1,                      0xD0036450,__READ_WRITE ,__gbe_ps1_bits);
__IO_REG32_BIT(GbE3_MHR,                      0xD0036454,__READ_WRITE ,__gbe_mhr_bits);
__IO_REG32_BIT(GbE3_IC,                       0xD0036460,__READ_WRITE ,__gbe_ic_bits);
__IO_REG32_BIT(GbE3_ICE,                      0xD0036464,__READ       ,__gbe_ice_bits);
__IO_REG32_BIT(GbE3_PIM,                      0xD0036468,__READ_WRITE ,__gbe_ic_bits);
__IO_REG32_BIT(GbE3_PEIM,                     0xD003646C,__READ_WRITE ,__gbe_ice_bits);
__IO_REG32_BIT(GbE3_PxTFUT,                   0xD0036474,__READ_WRITE ,__gbe_pxtfut_bits);
__IO_REG32_BIT(GbE3_PxMFS,                    0xD003647C,__READ_WRITE ,__gbe_pxmfs_bits);
__IO_REG32(    GbE3_PxDFC,                    0xD0036484,__READ       );
__IO_REG32(    GbE3_PxOFC,                    0xD0036488,__READ       );
__IO_REG32_BIT(GbE3_PIAE,                     0xD0036494,__READ       ,__gbe_piae_bits);
__IO_REG32_BIT(GbE3_ETPR,                     0xD00364BC,__READ_WRITE ,__gbe_etpr_bits);
__IO_REG32_BIT(GbE3_TQFPC,                    0xD00364DC,__READ_WRITE ,__gbe_tqfpc_bits);
__IO_REG32_BIT(GbE3_PTTBRC,                   0xD00364E0,__READ_WRITE ,__gbe_pttbrc_bits);
__IO_REG32_BIT(GbE3_MTU,                      0xD00364E8,__READ_WRITE ,__gbe_mtu_bits);
__IO_REG32_BIT(GbE3_PMTBS,                    0xD00364EC,__READ_WRITE ,__gbe_pmtbs_bits);
__IO_REG32(    GbE3_CRDP0,                    0xD003660C,__READ_WRITE );
__IO_REG32(    GbE3_CRDP1,                    0xD003661C,__READ_WRITE );
__IO_REG32(    GbE3_CRDP2,                    0xD003662C,__READ_WRITE );
__IO_REG32(    GbE3_CRDP3,                    0xD003663C,__READ_WRITE );
__IO_REG32(    GbE3_CRDP4,                    0xD003664C,__READ_WRITE );
__IO_REG32(    GbE3_CRDP5,                    0xD003665C,__READ_WRITE );
__IO_REG32(    GbE3_CRDP6,                    0xD003666C,__READ_WRITE );
__IO_REG32(    GbE3_CRDP7,                    0xD003667C,__READ_WRITE );
__IO_REG32_BIT(GbE3_RQC,                      0xD0036680,__READ_WRITE ,__gbe_rqc_bits);
__IO_REG32(    GbE3_TCSDPR,                   0xD0036684,__READ       );
__IO_REG32(    GbE3_TCQDP0,                   0xD00366C0,__READ_WRITE );
__IO_REG32(    GbE3_TCQDP1,                   0xD00366C4,__READ_WRITE );
__IO_REG32(    GbE3_TCQDP2,                   0xD00366C8,__READ_WRITE );
__IO_REG32(    GbE3_TCQDP3,                   0xD00366CC,__READ_WRITE );
__IO_REG32(    GbE3_TCQDP4,                   0xD00366D0,__READ_WRITE );
__IO_REG32(    GbE3_TCQDP5,                   0xD00366D4,__READ_WRITE );
__IO_REG32(    GbE3_TCQDP6,                   0xD00366D8,__READ_WRITE );
__IO_REG32(    GbE3_TCQDP7,                   0xD00366DC,__READ_WRITE );
__IO_REG32_BIT(GbE3_TQxTBCNT0,                0xD0036700,__READ_WRITE ,__gbe_tqxtbcnt_bits);
__IO_REG32_BIT(GbE3_TQxTBCNT1,                0xD0036710,__READ_WRITE ,__gbe_tqxtbcnt_bits);
__IO_REG32_BIT(GbE3_TQxTBCNT2,                0xD0036720,__READ_WRITE ,__gbe_tqxtbcnt_bits);
__IO_REG32_BIT(GbE3_TQxTBCNT3,                0xD0036730,__READ_WRITE ,__gbe_tqxtbcnt_bits);
__IO_REG32_BIT(GbE3_TQxTBCNT4,                0xD0036740,__READ_WRITE ,__gbe_tqxtbcnt_bits);
__IO_REG32_BIT(GbE3_TQxTBCNT5,                0xD0036750,__READ_WRITE ,__gbe_tqxtbcnt_bits);
__IO_REG32_BIT(GbE3_TQxTBCNT6,                0xD0036760,__READ_WRITE ,__gbe_tqxtbcnt_bits);
__IO_REG32_BIT(GbE3_TQxTBCNT7,                0xD0036770,__READ_WRITE ,__gbe_tqxtbcnt_bits);
__IO_REG32_BIT(GbE3_TQxTBC0,                  0xD0036704,__READ_WRITE ,__gbe_tqxtbc_bits);
__IO_REG32_BIT(GbE3_TQxTBC1,                  0xD0036714,__READ_WRITE ,__gbe_tqxtbc_bits);
__IO_REG32_BIT(GbE3_TQxTBC2,                  0xD0036724,__READ_WRITE ,__gbe_tqxtbc_bits);
__IO_REG32_BIT(GbE3_TQxTBC3,                  0xD0036734,__READ_WRITE ,__gbe_tqxtbc_bits);
__IO_REG32_BIT(GbE3_TQxTBC4,                  0xD0036744,__READ_WRITE ,__gbe_tqxtbc_bits);
__IO_REG32_BIT(GbE3_TQxTBC5,                  0xD0036754,__READ_WRITE ,__gbe_tqxtbc_bits);
__IO_REG32_BIT(GbE3_TQxTBC6,                  0xD0036764,__READ_WRITE ,__gbe_tqxtbc_bits);
__IO_REG32_BIT(GbE3_TQxTBC7,                  0xD0036774,__READ_WRITE ,__gbe_tqxtbc_bits);
__IO_REG32_BIT(GbE3_TQxAC0,                   0xD0036708,__READ_WRITE ,__gbe_tqxac_bits);
__IO_REG32_BIT(GbE3_TQxAC1,                   0xD0036718,__READ_WRITE ,__gbe_tqxac_bits);
__IO_REG32_BIT(GbE3_TQxAC2,                   0xD0036728,__READ_WRITE ,__gbe_tqxac_bits);
__IO_REG32_BIT(GbE3_TQxAC3,                   0xD0036738,__READ_WRITE ,__gbe_tqxac_bits);
__IO_REG32_BIT(GbE3_TQxAC4,                   0xD0036748,__READ_WRITE ,__gbe_tqxac_bits);
__IO_REG32_BIT(GbE3_TQxAC5,                   0xD0036758,__READ_WRITE ,__gbe_tqxac_bits);
__IO_REG32_BIT(GbE3_TQxAC6,                   0xD0036768,__READ_WRITE ,__gbe_tqxac_bits);
__IO_REG32_BIT(GbE3_TQxAC7,                   0xD0036778,__READ_WRITE ,__gbe_tqxac_bits);
__IO_REG32_BIT(GbE3_PTTBC,                    0xD0036780,__READ_WRITE ,__gbe_pttbc_bits);
__IO_REG32_BIT(GbE3_DFSMT0,                   0xD0037400,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT1,                   0xD0037404,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT2,                   0xD0037408,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT3,                   0xD003740C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT4,                   0xD0037410,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT5,                   0xD0037414,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT6,                   0xD0037418,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT7,                   0xD003741C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT8,                   0xD0037420,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT9,                   0xD0037424,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT10,                  0xD0037428,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT11,                  0xD003742C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT12,                  0xD0037430,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT13,                  0xD0037434,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT14,                  0xD0037438,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT15,                  0xD003743C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT16,                  0xD0037440,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT17,                  0xD0037444,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT18,                  0xD0037448,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT19,                  0xD003744C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT20,                  0xD0037450,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT21,                  0xD0037454,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT22,                  0xD0037458,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT23,                  0xD003745C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT24,                  0xD0037460,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT25,                  0xD0037464,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT26,                  0xD0037468,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT27,                  0xD003746C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT28,                  0xD0037470,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT29,                  0xD0037474,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT30,                  0xD0037478,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT31,                  0xD003747C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT32,                  0xD0037480,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT33,                  0xD0037484,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT34,                  0xD0037488,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT35,                  0xD003748C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT36,                  0xD0037490,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT37,                  0xD0037494,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT38,                  0xD0037498,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT39,                  0xD003749C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT40,                  0xD00374A0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT41,                  0xD00374A4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT42,                  0xD00374A8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT43,                  0xD00374AC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT44,                  0xD00374B0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT45,                  0xD00374B4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT46,                  0xD00374B8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT47,                  0xD00374BC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT48,                  0xD00374C0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT49,                  0xD00374C4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT50,                  0xD00374C8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT51,                  0xD00374CC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT52,                  0xD00374D0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT53,                  0xD00374D4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT54,                  0xD00374D8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT55,                  0xD00374DC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT56,                  0xD00374E0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT57,                  0xD00374E4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT58,                  0xD00374E8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT59,                  0xD00374EC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT60,                  0xD00374F0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT61,                  0xD00374F4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT62,                  0xD00374F8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFSMT63,                  0xD00374FC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT0,                   0xD0037500,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT1,                   0xD0037504,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT2,                   0xD0037508,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT3,                   0xD003750C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT4,                   0xD0037510,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT5,                   0xD0037514,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT6,                   0xD0037518,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT7,                   0xD003751C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT8,                   0xD0037520,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT9,                   0xD0037524,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT10,                  0xD0037528,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT11,                  0xD003752C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT12,                  0xD0037530,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT13,                  0xD0037534,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT14,                  0xD0037538,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT15,                  0xD003753C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT16,                  0xD0037540,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT17,                  0xD0037544,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT18,                  0xD0037548,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT19,                  0xD003754C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT20,                  0xD0037550,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT21,                  0xD0037554,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT22,                  0xD0037558,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT23,                  0xD003755C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT24,                  0xD0037560,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT25,                  0xD0037564,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT26,                  0xD0037568,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT27,                  0xD003756C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT28,                  0xD0037570,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT29,                  0xD0037574,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT30,                  0xD0037578,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT31,                  0xD003757C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT32,                  0xD0037580,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT33,                  0xD0037584,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT34,                  0xD0037588,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT35,                  0xD003758C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT36,                  0xD0037590,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT37,                  0xD0037594,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT38,                  0xD0037598,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT39,                  0xD003759C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT40,                  0xD00375A0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT41,                  0xD00375A4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT42,                  0xD00375A8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT43,                  0xD00375AC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT44,                  0xD00375B0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT45,                  0xD00375B4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT46,                  0xD00375B8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT47,                  0xD00375BC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT48,                  0xD00375C0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT49,                  0xD00375C4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT50,                  0xD00375C8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT51,                  0xD00375CC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT52,                  0xD00375D0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT53,                  0xD00375D4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT54,                  0xD00375D8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT55,                  0xD00375DC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT56,                  0xD00375E0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT57,                  0xD00375E4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT58,                  0xD00375E8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT59,                  0xD00375EC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT60,                  0xD00375F0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT61,                  0xD00375F4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT62,                  0xD00375F8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFOMT63,                  0xD00375FC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFUT0,                    0xD0037600,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFUT1,                    0xD0037604,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFUT2,                    0xD0037608,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE3_DFUT3,                    0xD003760C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32(    GbE3_GoodOctetsReceivedL,      0xD0037000,__READ_WRITE );
__IO_REG32(    GbE3_GoodOctetsReceivedH,      0xD0037004,__READ_WRITE );
__IO_REG32(    GbE3_BadOctetsReceived,        0xD0037008,__READ_WRITE );
__IO_REG32(    GbE3_MACTransError,            0xD003700C,__READ_WRITE );
__IO_REG32(    GbE3_GoodFramesReceived,       0xD0037010,__READ_WRITE );
__IO_REG32(    GbE3_BadFramesReceived,        0xD0037014,__READ_WRITE );
__IO_REG32(    GbE3_BroadcastFramesReceived,  0xD0037018,__READ_WRITE );
__IO_REG32(    GbE3_MulticastFramesReceived,  0xD003701C,__READ_WRITE );
__IO_REG32(    GbE3_Frames64Octets,           0xD0037020,__READ_WRITE );
__IO_REG32(    GbE3_Frames65to127Octets,      0xD0037024,__READ_WRITE );
__IO_REG32(    GbE3_Frames128to255Octets,     0xD0037028,__READ_WRITE );
__IO_REG32(    GbE3_Frames256to511Octets,     0xD003702C,__READ_WRITE );
__IO_REG32(    GbE3_Frames512to1023Octets,    0xD0037030,__READ_WRITE );
__IO_REG32(    GbE3_Frames1024toMaxOctets,    0xD0037034,__READ_WRITE );
__IO_REG32(    GbE3_GoodOctetsSentL,          0xD0037038,__READ_WRITE );
__IO_REG32(    GbE3_GoodOctetsSentH,          0xD003703C,__READ_WRITE );
__IO_REG32(    GbE3_GoodFramesSent,           0xD0037040,__READ_WRITE );
__IO_REG32(    GbE3_ExcessiveCollision,       0xD0037044,__READ_WRITE );
__IO_REG32(    GbE3_MulticastFramesSent,      0xD0037048,__READ_WRITE );
__IO_REG32(    GbE3_BroadcastFramesSent,      0xD003704C,__READ_WRITE );
__IO_REG32(    GbE3_UnrecogMACControl,        0xD0037050,__READ_WRITE );
__IO_REG32(    GbE3_FCSent,                   0xD0037054,__READ_WRITE );
__IO_REG32(    GbE3_GoodFCReceived,           0xD0037058,__READ_WRITE );
__IO_REG32(    GbE3_BadFCReceived,            0xD003705C,__READ_WRITE );
__IO_REG32(    GbE3_Undersize,                0xD0037060,__READ_WRITE );
__IO_REG32(    GbE3_Fragments,                0xD0037064,__READ_WRITE );
__IO_REG32(    GbE3_Oversize,                 0xD0037068,__READ_WRITE );
__IO_REG32(    GbE3_Jabber,                   0xD003706C,__READ_WRITE );
__IO_REG32(    GbE3_MACRcvError,              0xD0037070,__READ_WRITE );
__IO_REG32(    GbE3_BadCRC,                   0xD0037074,__READ_WRITE );
__IO_REG32(    GbE3_Collisions,               0xD0037078,__READ_WRITE );
__IO_REG32(    GbE3_LateCollision,            0xD003707C,__READ_WRITE );

/***************************************************************************
 **
 ** GbE0
 **
 ***************************************************************************/
__IO_REG32_BIT(GbE0_PHYAR,                    0xD0072000,__READ_WRITE ,__gbe_phyar_bits);
__IO_REG32_BIT(GbE0_SIMR,                     0xD0072004,__READ_WRITE ,__gbe_simr_bits);
__IO_REG32(    GbE0_EUDA,                     0xD0072008,__READ_WRITE );
__IO_REG32_BIT(GbE0_EUDID,                    0xD007200C,__READ_WRITE ,__gbe_eudid_bits);
__IO_REG32_BIT(GbE0_EUIC,                     0xD0072080,__READ       ,__gbe_euic_bits);
__IO_REG32_BIT(GbE0_EUIM,                     0xD0072084,__READ_WRITE ,__gbe_euim_bits);
__IO_REG32(    GbE0_EUEA,                     0xD0072094,__READ       );
__IO_REG32_BIT(GbE0_EUIAE,                    0xD0072098,__READ       ,__gbe_euiae_bits);
__IO_REG32_BIT(GbE0_EUC,                      0xD00720B0,__READ_WRITE ,__gbe_euc_bits);
__IO_REG32_BIT(GbE0_BA0,                      0xD0072200,__READ_WRITE ,__gbe_ba_bits);
__IO_REG32_BIT(GbE0_BA1,                      0xD0072208,__READ_WRITE ,__gbe_ba_bits);
__IO_REG32_BIT(GbE0_BA2,                      0xD0072210,__READ_WRITE ,__gbe_ba_bits);
__IO_REG32_BIT(GbE0_BA3,                      0xD0072218,__READ_WRITE ,__gbe_ba_bits);
__IO_REG32_BIT(GbE0_BA4,                      0xD0072220,__READ_WRITE ,__gbe_ba_bits);
__IO_REG32_BIT(GbE0_BA5,                      0xD0072228,__READ_WRITE ,__gbe_ba_bits);
__IO_REG32_BIT(GbE0_SR0,                      0xD0072204,__READ_WRITE ,__gbe_sr_bits);
__IO_REG32_BIT(GbE0_SR1,                      0xD007220C,__READ_WRITE ,__gbe_sr_bits);
__IO_REG32_BIT(GbE0_SR2,                      0xD0072214,__READ_WRITE ,__gbe_sr_bits);
__IO_REG32_BIT(GbE0_SR3,                      0xD007221C,__READ_WRITE ,__gbe_sr_bits);
__IO_REG32_BIT(GbE0_SR4,                      0xD0072224,__READ_WRITE ,__gbe_sr_bits);
__IO_REG32_BIT(GbE0_SR5,                      0xD007222C,__READ_WRITE ,__gbe_sr_bits);
__IO_REG32(    GbE0_HARR0,                    0xD0072280,__READ_WRITE );
__IO_REG32(    GbE0_HARR1,                    0xD0072284,__READ_WRITE );
__IO_REG32(    GbE0_HARR2,                    0xD0072288,__READ_WRITE );
__IO_REG32(    GbE0_HARR3,                    0xD007228C,__READ_WRITE );
__IO_REG32_BIT(GbE0_BARE,                     0xD0072290,__READ_WRITE ,__gbe_bare_bits);
__IO_REG32_BIT(GbE0_EPAP,                     0xD0072294,__READ_WRITE ,__gbe_epap_bits);
__IO_REG32_BIT(GbE0_PxC,                      0xD0072400,__READ_WRITE ,__gbe_pxc_bits);
__IO_REG32_BIT(GbE0_PxCX,                     0xD0072404,__READ_WRITE ,__gbe_pxcx_bits);
__IO_REG32_BIT(GbE0_MIISPR,                   0xD0072408,__READ_WRITE ,__gbe_miispr_bits);
__IO_REG32_BIT(GbE0_EVLANE,                   0xD0072410,__READ_WRITE ,__gbe_evlane_bits);
__IO_REG32_BIT(GbE0_MACAL,                    0xD0072414,__READ_WRITE ,__gbe_macal_bits);
__IO_REG32(    GbE0_MACAH,                    0xD0072418,__READ_WRITE );
__IO_REG32_BIT(GbE0_SDC,                      0xD007241C,__READ_WRITE ,__gbe_sdc_bits);
__IO_REG32_BIT(GbE0_DSCP0,                    0xD0072420,__READ_WRITE ,__gbe_dscp_bits);
__IO_REG32_BIT(GbE0_DSCP1,                    0xD0072424,__READ_WRITE ,__gbe_dscp_bits);
__IO_REG32_BIT(GbE0_DSCP2,                    0xD0072428,__READ_WRITE ,__gbe_dscp_bits);
__IO_REG32_BIT(GbE0_DSCP3,                    0xD007242C,__READ_WRITE ,__gbe_dscp_bits);
__IO_REG32_BIT(GbE0_DSCP4,                    0xD0072430,__READ_WRITE ,__gbe_dscp_bits);
__IO_REG32_BIT(GbE0_DSCP5,                    0xD0072434,__READ_WRITE ,__gbe_dscp_bits);
__IO_REG32_BIT(GbE0_DSCP6,                    0xD0072438,__READ_WRITE ,__gbe_dscp_bits);
__IO_REG32_BIT(GbE0_PSC0,                     0xD007243C,__READ_WRITE ,__gbe_psc0_bits);
__IO_REG32_BIT(GbE0_VPT2P,                    0xD0072440,__READ_WRITE ,__gbe_vpt2p_bits);
__IO_REG32_BIT(GbE0_PS0,                      0xD0072444,__READ_WRITE ,__gbe_ps0_bits);
__IO_REG32_BIT(GbE0_TQC,                      0xD0072448,__READ_WRITE ,__gbe_tqc_bits);
__IO_REG32_BIT(GbE0_PSC1,                     0xD007244C,__READ_WRITE ,__gbe_psc1_bits);
__IO_REG32_BIT(GbE0_PS1,                      0xD0072450,__READ_WRITE ,__gbe_ps1_bits);
__IO_REG32_BIT(GbE0_MHR,                      0xD0072454,__READ_WRITE ,__gbe_mhr_bits);
__IO_REG32_BIT(GbE0_IC,                       0xD0072460,__READ_WRITE ,__gbe_ic_bits);
__IO_REG32_BIT(GbE0_ICE,                      0xD0072464,__READ       ,__gbe_ice_bits);
__IO_REG32_BIT(GbE0_PIM,                      0xD0072468,__READ_WRITE ,__gbe_ic_bits);
__IO_REG32_BIT(GbE0_PEIM,                     0xD007246C,__READ_WRITE ,__gbe_ice_bits);
__IO_REG32_BIT(GbE0_PxTFUT,                   0xD0072474,__READ_WRITE ,__gbe_pxtfut_bits);
__IO_REG32_BIT(GbE0_PxMFS,                    0xD007247C,__READ_WRITE ,__gbe_pxmfs_bits);
__IO_REG32(    GbE0_PxDFC,                    0xD0072484,__READ       );
__IO_REG32(    GbE0_PxOFC,                    0xD0072488,__READ       );
__IO_REG32_BIT(GbE0_PIAE,                     0xD0072494,__READ       ,__gbe_piae_bits);
__IO_REG32_BIT(GbE0_ETPR,                     0xD00724BC,__READ_WRITE ,__gbe_etpr_bits);
__IO_REG32_BIT(GbE0_TQFPC,                    0xD00724DC,__READ_WRITE ,__gbe_tqfpc_bits);
__IO_REG32_BIT(GbE0_PTTBRC,                   0xD00724E0,__READ_WRITE ,__gbe_pttbrc_bits);
__IO_REG32_BIT(GbE0_MTU,                      0xD00724E8,__READ_WRITE ,__gbe_mtu_bits);
__IO_REG32_BIT(GbE0_PMTBS,                    0xD00724EC,__READ_WRITE ,__gbe_pmtbs_bits);
__IO_REG32(    GbE0_CRDP0,                    0xD007260C,__READ_WRITE );
__IO_REG32(    GbE0_CRDP1,                    0xD007261C,__READ_WRITE );
__IO_REG32(    GbE0_CRDP2,                    0xD007262C,__READ_WRITE );
__IO_REG32(    GbE0_CRDP3,                    0xD007263C,__READ_WRITE );
__IO_REG32(    GbE0_CRDP4,                    0xD007264C,__READ_WRITE );
__IO_REG32(    GbE0_CRDP5,                    0xD007265C,__READ_WRITE );
__IO_REG32(    GbE0_CRDP6,                    0xD007266C,__READ_WRITE );
__IO_REG32(    GbE0_CRDP7,                    0xD007267C,__READ_WRITE );
__IO_REG32_BIT(GbE0_RQC,                      0xD0072680,__READ_WRITE ,__gbe_rqc_bits);
__IO_REG32(    GbE0_TCSDPR,                   0xD0072684,__READ       );
__IO_REG32(    GbE0_TCQDP0,                   0xD00726C0,__READ_WRITE );
__IO_REG32(    GbE0_TCQDP1,                   0xD00726C4,__READ_WRITE );
__IO_REG32(    GbE0_TCQDP2,                   0xD00726C8,__READ_WRITE );
__IO_REG32(    GbE0_TCQDP3,                   0xD00726CC,__READ_WRITE );
__IO_REG32(    GbE0_TCQDP4,                   0xD00726D0,__READ_WRITE );
__IO_REG32(    GbE0_TCQDP5,                   0xD00726D4,__READ_WRITE );
__IO_REG32(    GbE0_TCQDP6,                   0xD00726D8,__READ_WRITE );
__IO_REG32(    GbE0_TCQDP7,                   0xD00726DC,__READ_WRITE );
__IO_REG32_BIT(GbE0_TQxTBCNT0,                0xD0072700,__READ_WRITE ,__gbe_tqxtbcnt_bits);
__IO_REG32_BIT(GbE0_TQxTBCNT1,                0xD0072710,__READ_WRITE ,__gbe_tqxtbcnt_bits);
__IO_REG32_BIT(GbE0_TQxTBCNT2,                0xD0072720,__READ_WRITE ,__gbe_tqxtbcnt_bits);
__IO_REG32_BIT(GbE0_TQxTBCNT3,                0xD0072730,__READ_WRITE ,__gbe_tqxtbcnt_bits);
__IO_REG32_BIT(GbE0_TQxTBCNT4,                0xD0072740,__READ_WRITE ,__gbe_tqxtbcnt_bits);
__IO_REG32_BIT(GbE0_TQxTBCNT5,                0xD0072750,__READ_WRITE ,__gbe_tqxtbcnt_bits);
__IO_REG32_BIT(GbE0_TQxTBCNT6,                0xD0072760,__READ_WRITE ,__gbe_tqxtbcnt_bits);
__IO_REG32_BIT(GbE0_TQxTBCNT7,                0xD0072770,__READ_WRITE ,__gbe_tqxtbcnt_bits);
__IO_REG32_BIT(GbE0_TQxTBC0,                  0xD0072704,__READ_WRITE ,__gbe_tqxtbc_bits);
__IO_REG32_BIT(GbE0_TQxTBC1,                  0xD0072714,__READ_WRITE ,__gbe_tqxtbc_bits);
__IO_REG32_BIT(GbE0_TQxTBC2,                  0xD0072724,__READ_WRITE ,__gbe_tqxtbc_bits);
__IO_REG32_BIT(GbE0_TQxTBC3,                  0xD0072734,__READ_WRITE ,__gbe_tqxtbc_bits);
__IO_REG32_BIT(GbE0_TQxTBC4,                  0xD0072744,__READ_WRITE ,__gbe_tqxtbc_bits);
__IO_REG32_BIT(GbE0_TQxTBC5,                  0xD0072754,__READ_WRITE ,__gbe_tqxtbc_bits);
__IO_REG32_BIT(GbE0_TQxTBC6,                  0xD0072764,__READ_WRITE ,__gbe_tqxtbc_bits);
__IO_REG32_BIT(GbE0_TQxTBC7,                  0xD0072774,__READ_WRITE ,__gbe_tqxtbc_bits);
__IO_REG32_BIT(GbE0_TQxAC0,                   0xD0072708,__READ_WRITE ,__gbe_tqxac_bits);
__IO_REG32_BIT(GbE0_TQxAC1,                   0xD0072718,__READ_WRITE ,__gbe_tqxac_bits);
__IO_REG32_BIT(GbE0_TQxAC2,                   0xD0072728,__READ_WRITE ,__gbe_tqxac_bits);
__IO_REG32_BIT(GbE0_TQxAC3,                   0xD0072738,__READ_WRITE ,__gbe_tqxac_bits);
__IO_REG32_BIT(GbE0_TQxAC4,                   0xD0072748,__READ_WRITE ,__gbe_tqxac_bits);
__IO_REG32_BIT(GbE0_TQxAC5,                   0xD0072758,__READ_WRITE ,__gbe_tqxac_bits);
__IO_REG32_BIT(GbE0_TQxAC6,                   0xD0072768,__READ_WRITE ,__gbe_tqxac_bits);
__IO_REG32_BIT(GbE0_TQxAC7,                   0xD0072778,__READ_WRITE ,__gbe_tqxac_bits);
__IO_REG32_BIT(GbE0_PTTBC,                    0xD0072780,__READ_WRITE ,__gbe_pttbc_bits);
__IO_REG32_BIT(GbE0_DFSMT0,                   0xD0073400,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT1,                   0xD0073404,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT2,                   0xD0073408,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT3,                   0xD007340C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT4,                   0xD0073410,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT5,                   0xD0073414,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT6,                   0xD0073418,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT7,                   0xD007341C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT8,                   0xD0073420,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT9,                   0xD0073424,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT10,                  0xD0073428,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT11,                  0xD007342C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT12,                  0xD0073430,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT13,                  0xD0073434,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT14,                  0xD0073438,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT15,                  0xD007343C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT16,                  0xD0073440,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT17,                  0xD0073444,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT18,                  0xD0073448,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT19,                  0xD007344C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT20,                  0xD0073450,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT21,                  0xD0073454,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT22,                  0xD0073458,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT23,                  0xD007345C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT24,                  0xD0073460,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT25,                  0xD0073464,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT26,                  0xD0073468,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT27,                  0xD007346C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT28,                  0xD0073470,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT29,                  0xD0073474,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT30,                  0xD0073478,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT31,                  0xD007347C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT32,                  0xD0073480,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT33,                  0xD0073484,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT34,                  0xD0073488,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT35,                  0xD007348C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT36,                  0xD0073490,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT37,                  0xD0073494,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT38,                  0xD0073498,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT39,                  0xD007349C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT40,                  0xD00734A0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT41,                  0xD00734A4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT42,                  0xD00734A8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT43,                  0xD00734AC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT44,                  0xD00734B0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT45,                  0xD00734B4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT46,                  0xD00734B8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT47,                  0xD00734BC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT48,                  0xD00734C0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT49,                  0xD00734C4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT50,                  0xD00734C8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT51,                  0xD00734CC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT52,                  0xD00734D0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT53,                  0xD00734D4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT54,                  0xD00734D8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT55,                  0xD00734DC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT56,                  0xD00734E0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT57,                  0xD00734E4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT58,                  0xD00734E8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT59,                  0xD00734EC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT60,                  0xD00734F0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT61,                  0xD00734F4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT62,                  0xD00734F8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFSMT63,                  0xD00734FC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT0,                   0xD0073500,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT1,                   0xD0073504,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT2,                   0xD0073508,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT3,                   0xD007350C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT4,                   0xD0073510,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT5,                   0xD0073514,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT6,                   0xD0073518,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT7,                   0xD007351C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT8,                   0xD0073520,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT9,                   0xD0073524,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT10,                  0xD0073528,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT11,                  0xD007352C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT12,                  0xD0073530,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT13,                  0xD0073534,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT14,                  0xD0073538,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT15,                  0xD007353C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT16,                  0xD0073540,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT17,                  0xD0073544,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT18,                  0xD0073548,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT19,                  0xD007354C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT20,                  0xD0073550,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT21,                  0xD0073554,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT22,                  0xD0073558,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT23,                  0xD007355C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT24,                  0xD0073560,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT25,                  0xD0073564,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT26,                  0xD0073568,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT27,                  0xD007356C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT28,                  0xD0073570,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT29,                  0xD0073574,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT30,                  0xD0073578,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT31,                  0xD007357C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT32,                  0xD0073580,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT33,                  0xD0073584,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT34,                  0xD0073588,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT35,                  0xD007358C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT36,                  0xD0073590,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT37,                  0xD0073594,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT38,                  0xD0073598,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT39,                  0xD007359C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT40,                  0xD00735A0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT41,                  0xD00735A4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT42,                  0xD00735A8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT43,                  0xD00735AC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT44,                  0xD00735B0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT45,                  0xD00735B4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT46,                  0xD00735B8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT47,                  0xD00735BC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT48,                  0xD00735C0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT49,                  0xD00735C4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT50,                  0xD00735C8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT51,                  0xD00735CC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT52,                  0xD00735D0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT53,                  0xD00735D4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT54,                  0xD00735D8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT55,                  0xD00735DC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT56,                  0xD00735E0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT57,                  0xD00735E4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT58,                  0xD00735E8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT59,                  0xD00735EC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT60,                  0xD00735F0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT61,                  0xD00735F4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT62,                  0xD00735F8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFOMT63,                  0xD00735FC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFUT0,                    0xD0073600,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFUT1,                    0xD0073604,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFUT2,                    0xD0073608,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE0_DFUT3,                    0xD007360C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32(    GbE0_GoodOctetsReceivedL,      0xD0073000,__READ_WRITE );
__IO_REG32(    GbE0_GoodOctetsReceivedH,      0xD0073004,__READ_WRITE );
__IO_REG32(    GbE0_BadOctetsReceived,        0xD0073008,__READ_WRITE );
__IO_REG32(    GbE0_MACTransError,            0xD007300C,__READ_WRITE );
__IO_REG32(    GbE0_GoodFramesReceived,       0xD0073010,__READ_WRITE );
__IO_REG32(    GbE0_BadFramesReceived,        0xD0073014,__READ_WRITE );
__IO_REG32(    GbE0_BroadcastFramesReceived,  0xD0073018,__READ_WRITE );
__IO_REG32(    GbE0_MulticastFramesReceived,  0xD007301C,__READ_WRITE );
__IO_REG32(    GbE0_Frames64Octets,           0xD0073020,__READ_WRITE );
__IO_REG32(    GbE0_Frames65to127Octets,      0xD0073024,__READ_WRITE );
__IO_REG32(    GbE0_Frames128to255Octets,     0xD0073028,__READ_WRITE );
__IO_REG32(    GbE0_Frames256to511Octets,     0xD007302C,__READ_WRITE );
__IO_REG32(    GbE0_Frames512to1023Octets,    0xD0073030,__READ_WRITE );
__IO_REG32(    GbE0_Frames1024toMaxOctets,    0xD0073034,__READ_WRITE );
__IO_REG32(    GbE0_GoodOctetsSentL,          0xD0073038,__READ_WRITE );
__IO_REG32(    GbE0_GoodOctetsSentH,          0xD007303C,__READ_WRITE );
__IO_REG32(    GbE0_GoodFramesSent,           0xD0073040,__READ_WRITE );
__IO_REG32(    GbE0_ExcessiveCollision,       0xD0073044,__READ_WRITE );
__IO_REG32(    GbE0_MulticastFramesSent,      0xD0073048,__READ_WRITE );
__IO_REG32(    GbE0_BroadcastFramesSent,      0xD007304C,__READ_WRITE );
__IO_REG32(    GbE0_UnrecogMACControl,        0xD0073050,__READ_WRITE );
__IO_REG32(    GbE0_FCSent,                   0xD0073054,__READ_WRITE );
__IO_REG32(    GbE0_GoodFCReceived,           0xD0073058,__READ_WRITE );
__IO_REG32(    GbE0_BadFCReceived,            0xD007305C,__READ_WRITE );
__IO_REG32(    GbE0_Undersize,                0xD0073060,__READ_WRITE );
__IO_REG32(    GbE0_Fragments,                0xD0073064,__READ_WRITE );
__IO_REG32(    GbE0_Oversize,                 0xD0073068,__READ_WRITE );
__IO_REG32(    GbE0_Jabber,                   0xD007306C,__READ_WRITE );
__IO_REG32(    GbE0_MACRcvError,              0xD0073070,__READ_WRITE );
__IO_REG32(    GbE0_BadCRC,                   0xD0073074,__READ_WRITE );
__IO_REG32(    GbE0_Collisions,               0xD0073078,__READ_WRITE );
__IO_REG32(    GbE0_LateCollision,            0xD007307C,__READ_WRITE );

/***************************************************************************
 **
 ** GbE1
 **
 ***************************************************************************/
__IO_REG32_BIT(GbE1_PHYAR,                    0xD0076000,__READ_WRITE ,__gbe_phyar_bits);
__IO_REG32_BIT(GbE1_SIMR,                     0xD0076004,__READ_WRITE ,__gbe_simr_bits);
__IO_REG32(    GbE1_EUDA,                     0xD0076008,__READ_WRITE );
__IO_REG32_BIT(GbE1_EUDID,                    0xD007600C,__READ_WRITE ,__gbe_eudid_bits);
__IO_REG32_BIT(GbE1_EUIC,                     0xD0076080,__READ       ,__gbe_euic_bits);
__IO_REG32_BIT(GbE1_EUIM,                     0xD0076084,__READ_WRITE ,__gbe_euim_bits);
__IO_REG32(    GbE1_EUEA,                     0xD0076094,__READ       );
__IO_REG32_BIT(GbE1_EUIAE,                    0xD0076098,__READ       ,__gbe_euiae_bits);
__IO_REG32_BIT(GbE1_EUC,                      0xD00760B0,__READ_WRITE ,__gbe_euc_bits);
__IO_REG32_BIT(GbE1_BA0,                      0xD0076200,__READ_WRITE ,__gbe_ba_bits);
__IO_REG32_BIT(GbE1_BA1,                      0xD0076208,__READ_WRITE ,__gbe_ba_bits);
__IO_REG32_BIT(GbE1_BA2,                      0xD0076210,__READ_WRITE ,__gbe_ba_bits);
__IO_REG32_BIT(GbE1_BA3,                      0xD0076218,__READ_WRITE ,__gbe_ba_bits);
__IO_REG32_BIT(GbE1_BA4,                      0xD0076220,__READ_WRITE ,__gbe_ba_bits);
__IO_REG32_BIT(GbE1_BA5,                      0xD0076228,__READ_WRITE ,__gbe_ba_bits);
__IO_REG32_BIT(GbE1_SR0,                      0xD0076204,__READ_WRITE ,__gbe_sr_bits);
__IO_REG32_BIT(GbE1_SR1,                      0xD007620C,__READ_WRITE ,__gbe_sr_bits);
__IO_REG32_BIT(GbE1_SR2,                      0xD0076214,__READ_WRITE ,__gbe_sr_bits);
__IO_REG32_BIT(GbE1_SR3,                      0xD007621C,__READ_WRITE ,__gbe_sr_bits);
__IO_REG32_BIT(GbE1_SR4,                      0xD0076224,__READ_WRITE ,__gbe_sr_bits);
__IO_REG32_BIT(GbE1_SR5,                      0xD007622C,__READ_WRITE ,__gbe_sr_bits);
__IO_REG32(    GbE1_HARR0,                    0xD0076280,__READ_WRITE );
__IO_REG32(    GbE1_HARR1,                    0xD0076284,__READ_WRITE );
__IO_REG32(    GbE1_HARR2,                    0xD0076288,__READ_WRITE );
__IO_REG32(    GbE1_HARR3,                    0xD007628C,__READ_WRITE );
__IO_REG32_BIT(GbE1_BARE,                     0xD0076290,__READ_WRITE ,__gbe_bare_bits);
__IO_REG32_BIT(GbE1_EPAP,                     0xD0076294,__READ_WRITE ,__gbe_epap_bits);
__IO_REG32_BIT(GbE1_PxC,                      0xD0076400,__READ_WRITE ,__gbe_pxc_bits);
__IO_REG32_BIT(GbE1_PxCX,                     0xD0076404,__READ_WRITE ,__gbe_pxcx_bits);
__IO_REG32_BIT(GbE1_MIISPR,                   0xD0076408,__READ_WRITE ,__gbe_miispr_bits);
__IO_REG32_BIT(GbE1_EVLANE,                   0xD0076410,__READ_WRITE ,__gbe_evlane_bits);
__IO_REG32_BIT(GbE1_MACAL,                    0xD0076414,__READ_WRITE ,__gbe_macal_bits);
__IO_REG32(    GbE1_MACAH,                    0xD0076418,__READ_WRITE );
__IO_REG32_BIT(GbE1_SDC,                      0xD007641C,__READ_WRITE ,__gbe_sdc_bits);
__IO_REG32_BIT(GbE1_DSCP0,                    0xD0076420,__READ_WRITE ,__gbe_dscp_bits);
__IO_REG32_BIT(GbE1_DSCP1,                    0xD0076424,__READ_WRITE ,__gbe_dscp_bits);
__IO_REG32_BIT(GbE1_DSCP2,                    0xD0076428,__READ_WRITE ,__gbe_dscp_bits);
__IO_REG32_BIT(GbE1_DSCP3,                    0xD007642C,__READ_WRITE ,__gbe_dscp_bits);
__IO_REG32_BIT(GbE1_DSCP4,                    0xD0076430,__READ_WRITE ,__gbe_dscp_bits);
__IO_REG32_BIT(GbE1_DSCP5,                    0xD0076434,__READ_WRITE ,__gbe_dscp_bits);
__IO_REG32_BIT(GbE1_DSCP6,                    0xD0076438,__READ_WRITE ,__gbe_dscp_bits);
__IO_REG32_BIT(GbE1_PSC0,                     0xD007643C,__READ_WRITE ,__gbe_psc0_bits);
__IO_REG32_BIT(GbE1_VPT2P,                    0xD0076440,__READ_WRITE ,__gbe_vpt2p_bits);
__IO_REG32_BIT(GbE1_PS0,                      0xD0076444,__READ_WRITE ,__gbe_ps0_bits);
__IO_REG32_BIT(GbE1_TQC,                      0xD0076448,__READ_WRITE ,__gbe_tqc_bits);
__IO_REG32_BIT(GbE1_PSC1,                     0xD007644C,__READ_WRITE ,__gbe_psc1_bits);
__IO_REG32_BIT(GbE1_PS1,                      0xD0076450,__READ_WRITE ,__gbe_ps1_bits);
__IO_REG32_BIT(GbE1_MHR,                      0xD0076454,__READ_WRITE ,__gbe_mhr_bits);
__IO_REG32_BIT(GbE1_IC,                       0xD0076460,__READ_WRITE ,__gbe_ic_bits);
__IO_REG32_BIT(GbE1_ICE,                      0xD0076464,__READ       ,__gbe_ice_bits);
__IO_REG32_BIT(GbE1_PIM,                      0xD0076468,__READ_WRITE ,__gbe_ic_bits);
__IO_REG32_BIT(GbE1_PEIM,                     0xD007646C,__READ_WRITE ,__gbe_ice_bits);
__IO_REG32_BIT(GbE1_PxTFUT,                   0xD0076474,__READ_WRITE ,__gbe_pxtfut_bits);
__IO_REG32_BIT(GbE1_PxMFS,                    0xD007647C,__READ_WRITE ,__gbe_pxmfs_bits);
__IO_REG32(    GbE1_PxDFC,                    0xD0076484,__READ       );
__IO_REG32(    GbE1_PxOFC,                    0xD0076488,__READ       );
__IO_REG32_BIT(GbE1_PIAE,                     0xD0076494,__READ       ,__gbe_piae_bits);
__IO_REG32_BIT(GbE1_ETPR,                     0xD00764BC,__READ_WRITE ,__gbe_etpr_bits);
__IO_REG32_BIT(GbE1_TQFPC,                    0xD00764DC,__READ_WRITE ,__gbe_tqfpc_bits);
__IO_REG32_BIT(GbE1_PTTBRC,                   0xD00764E0,__READ_WRITE ,__gbe_pttbrc_bits);
__IO_REG32_BIT(GbE1_MTU,                      0xD00764E8,__READ_WRITE ,__gbe_mtu_bits);
__IO_REG32_BIT(GbE1_PMTBS,                    0xD00764EC,__READ_WRITE ,__gbe_pmtbs_bits);
__IO_REG32(    GbE1_CRDP0,                    0xD007660C,__READ_WRITE );
__IO_REG32(    GbE1_CRDP1,                    0xD007661C,__READ_WRITE );
__IO_REG32(    GbE1_CRDP2,                    0xD007662C,__READ_WRITE );
__IO_REG32(    GbE1_CRDP3,                    0xD007663C,__READ_WRITE );
__IO_REG32(    GbE1_CRDP4,                    0xD007664C,__READ_WRITE );
__IO_REG32(    GbE1_CRDP5,                    0xD007665C,__READ_WRITE );
__IO_REG32(    GbE1_CRDP6,                    0xD007666C,__READ_WRITE );
__IO_REG32(    GbE1_CRDP7,                    0xD007667C,__READ_WRITE );
__IO_REG32_BIT(GbE1_RQC,                      0xD0076680,__READ_WRITE ,__gbe_rqc_bits);
__IO_REG32(    GbE1_TCSDPR,                   0xD0076684,__READ       );
__IO_REG32(    GbE1_TCQDP0,                   0xD00766C0,__READ_WRITE );
__IO_REG32(    GbE1_TCQDP1,                   0xD00766C4,__READ_WRITE );
__IO_REG32(    GbE1_TCQDP2,                   0xD00766C8,__READ_WRITE );
__IO_REG32(    GbE1_TCQDP3,                   0xD00766CC,__READ_WRITE );
__IO_REG32(    GbE1_TCQDP4,                   0xD00766D0,__READ_WRITE );
__IO_REG32(    GbE1_TCQDP5,                   0xD00766D4,__READ_WRITE );
__IO_REG32(    GbE1_TCQDP6,                   0xD00766D8,__READ_WRITE );
__IO_REG32(    GbE1_TCQDP7,                   0xD00766DC,__READ_WRITE );
__IO_REG32_BIT(GbE1_TQxTBCNT0,                0xD0076700,__READ_WRITE ,__gbe_tqxtbcnt_bits);
__IO_REG32_BIT(GbE1_TQxTBCNT1,                0xD0076710,__READ_WRITE ,__gbe_tqxtbcnt_bits);
__IO_REG32_BIT(GbE1_TQxTBCNT2,                0xD0076720,__READ_WRITE ,__gbe_tqxtbcnt_bits);
__IO_REG32_BIT(GbE1_TQxTBCNT3,                0xD0076730,__READ_WRITE ,__gbe_tqxtbcnt_bits);
__IO_REG32_BIT(GbE1_TQxTBCNT4,                0xD0076740,__READ_WRITE ,__gbe_tqxtbcnt_bits);
__IO_REG32_BIT(GbE1_TQxTBCNT5,                0xD0076750,__READ_WRITE ,__gbe_tqxtbcnt_bits);
__IO_REG32_BIT(GbE1_TQxTBCNT6,                0xD0076760,__READ_WRITE ,__gbe_tqxtbcnt_bits);
__IO_REG32_BIT(GbE1_TQxTBCNT7,                0xD0076770,__READ_WRITE ,__gbe_tqxtbcnt_bits);
__IO_REG32_BIT(GbE1_TQxTBC0,                  0xD0076704,__READ_WRITE ,__gbe_tqxtbc_bits);
__IO_REG32_BIT(GbE1_TQxTBC1,                  0xD0076714,__READ_WRITE ,__gbe_tqxtbc_bits);
__IO_REG32_BIT(GbE1_TQxTBC2,                  0xD0076724,__READ_WRITE ,__gbe_tqxtbc_bits);
__IO_REG32_BIT(GbE1_TQxTBC3,                  0xD0076734,__READ_WRITE ,__gbe_tqxtbc_bits);
__IO_REG32_BIT(GbE1_TQxTBC4,                  0xD0076744,__READ_WRITE ,__gbe_tqxtbc_bits);
__IO_REG32_BIT(GbE1_TQxTBC5,                  0xD0076754,__READ_WRITE ,__gbe_tqxtbc_bits);
__IO_REG32_BIT(GbE1_TQxTBC6,                  0xD0076764,__READ_WRITE ,__gbe_tqxtbc_bits);
__IO_REG32_BIT(GbE1_TQxTBC7,                  0xD0076774,__READ_WRITE ,__gbe_tqxtbc_bits);
__IO_REG32_BIT(GbE1_TQxAC0,                   0xD0076708,__READ_WRITE ,__gbe_tqxac_bits);
__IO_REG32_BIT(GbE1_TQxAC1,                   0xD0076718,__READ_WRITE ,__gbe_tqxac_bits);
__IO_REG32_BIT(GbE1_TQxAC2,                   0xD0076728,__READ_WRITE ,__gbe_tqxac_bits);
__IO_REG32_BIT(GbE1_TQxAC3,                   0xD0076738,__READ_WRITE ,__gbe_tqxac_bits);
__IO_REG32_BIT(GbE1_TQxAC4,                   0xD0076748,__READ_WRITE ,__gbe_tqxac_bits);
__IO_REG32_BIT(GbE1_TQxAC5,                   0xD0076758,__READ_WRITE ,__gbe_tqxac_bits);
__IO_REG32_BIT(GbE1_TQxAC6,                   0xD0076768,__READ_WRITE ,__gbe_tqxac_bits);
__IO_REG32_BIT(GbE1_TQxAC7,                   0xD0076778,__READ_WRITE ,__gbe_tqxac_bits);
__IO_REG32_BIT(GbE1_PTTBC,                    0xD0076780,__READ_WRITE ,__gbe_pttbc_bits);
__IO_REG32_BIT(GbE1_DFSMT0,                   0xD0077400,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT1,                   0xD0077404,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT2,                   0xD0077408,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT3,                   0xD007740C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT4,                   0xD0077410,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT5,                   0xD0077414,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT6,                   0xD0077418,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT7,                   0xD007741C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT8,                   0xD0077420,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT9,                   0xD0077424,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT10,                  0xD0077428,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT11,                  0xD007742C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT12,                  0xD0077430,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT13,                  0xD0077434,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT14,                  0xD0077438,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT15,                  0xD007743C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT16,                  0xD0077440,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT17,                  0xD0077444,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT18,                  0xD0077448,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT19,                  0xD007744C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT20,                  0xD0077450,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT21,                  0xD0077454,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT22,                  0xD0077458,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT23,                  0xD007745C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT24,                  0xD0077460,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT25,                  0xD0077464,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT26,                  0xD0077468,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT27,                  0xD007746C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT28,                  0xD0077470,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT29,                  0xD0077474,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT30,                  0xD0077478,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT31,                  0xD007747C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT32,                  0xD0077480,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT33,                  0xD0077484,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT34,                  0xD0077488,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT35,                  0xD007748C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT36,                  0xD0077490,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT37,                  0xD0077494,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT38,                  0xD0077498,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT39,                  0xD007749C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT40,                  0xD00774A0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT41,                  0xD00774A4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT42,                  0xD00774A8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT43,                  0xD00774AC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT44,                  0xD00774B0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT45,                  0xD00774B4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT46,                  0xD00774B8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT47,                  0xD00774BC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT48,                  0xD00774C0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT49,                  0xD00774C4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT50,                  0xD00774C8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT51,                  0xD00774CC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT52,                  0xD00774D0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT53,                  0xD00774D4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT54,                  0xD00774D8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT55,                  0xD00774DC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT56,                  0xD00774E0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT57,                  0xD00774E4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT58,                  0xD00774E8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT59,                  0xD00774EC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT60,                  0xD00774F0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT61,                  0xD00774F4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT62,                  0xD00774F8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFSMT63,                  0xD00774FC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT0,                   0xD0077500,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT1,                   0xD0077504,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT2,                   0xD0077508,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT3,                   0xD007750C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT4,                   0xD0077510,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT5,                   0xD0077514,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT6,                   0xD0077518,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT7,                   0xD007751C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT8,                   0xD0077520,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT9,                   0xD0077524,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT10,                  0xD0077528,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT11,                  0xD007752C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT12,                  0xD0077530,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT13,                  0xD0077534,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT14,                  0xD0077538,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT15,                  0xD007753C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT16,                  0xD0077540,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT17,                  0xD0077544,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT18,                  0xD0077548,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT19,                  0xD007754C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT20,                  0xD0077550,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT21,                  0xD0077554,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT22,                  0xD0077558,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT23,                  0xD007755C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT24,                  0xD0077560,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT25,                  0xD0077564,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT26,                  0xD0077568,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT27,                  0xD007756C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT28,                  0xD0077570,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT29,                  0xD0077574,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT30,                  0xD0077578,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT31,                  0xD007757C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT32,                  0xD0077580,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT33,                  0xD0077584,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT34,                  0xD0077588,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT35,                  0xD007758C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT36,                  0xD0077590,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT37,                  0xD0077594,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT38,                  0xD0077598,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT39,                  0xD007759C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT40,                  0xD00775A0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT41,                  0xD00775A4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT42,                  0xD00775A8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT43,                  0xD00775AC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT44,                  0xD00775B0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT45,                  0xD00775B4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT46,                  0xD00775B8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT47,                  0xD00775BC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT48,                  0xD00775C0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT49,                  0xD00775C4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT50,                  0xD00775C8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT51,                  0xD00775CC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT52,                  0xD00775D0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT53,                  0xD00775D4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT54,                  0xD00775D8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT55,                  0xD00775DC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT56,                  0xD00775E0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT57,                  0xD00775E4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT58,                  0xD00775E8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT59,                  0xD00775EC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT60,                  0xD00775F0,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT61,                  0xD00775F4,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT62,                  0xD00775F8,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFOMT63,                  0xD00775FC,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFUT0,                    0xD0077600,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFUT1,                    0xD0077604,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFUT2,                    0xD0077608,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32_BIT(GbE1_DFUT3,                    0xD007760C,__READ_WRITE ,__gbe_dfsmt_bits);
__IO_REG32(    GbE1_GoodOctetsReceivedL,      0xD0077000,__READ_WRITE );
__IO_REG32(    GbE1_GoodOctetsReceivedH,      0xD0077004,__READ_WRITE );
__IO_REG32(    GbE1_BadOctetsReceived,        0xD0077008,__READ_WRITE );
__IO_REG32(    GbE1_MACTransError,            0xD007700C,__READ_WRITE );
__IO_REG32(    GbE1_GoodFramesReceived,       0xD0077010,__READ_WRITE );
__IO_REG32(    GbE1_BadFramesReceived,        0xD0077014,__READ_WRITE );
__IO_REG32(    GbE1_BroadcastFramesReceived,  0xD0077018,__READ_WRITE );
__IO_REG32(    GbE1_MulticastFramesReceived,  0xD007701C,__READ_WRITE );
__IO_REG32(    GbE1_Frames64Octets,           0xD0077020,__READ_WRITE );
__IO_REG32(    GbE1_Frames65to127Octets,      0xD0077024,__READ_WRITE );
__IO_REG32(    GbE1_Frames128to255Octets,     0xD0077028,__READ_WRITE );
__IO_REG32(    GbE1_Frames256to511Octets,     0xD007702C,__READ_WRITE );
__IO_REG32(    GbE1_Frames512to1023Octets,    0xD0077030,__READ_WRITE );
__IO_REG32(    GbE1_Frames1024toMaxOctets,    0xD0077034,__READ_WRITE );
__IO_REG32(    GbE1_GoodOctetsSentL,          0xD0077038,__READ_WRITE );
__IO_REG32(    GbE1_GoodOctetsSentH,          0xD007703C,__READ_WRITE );
__IO_REG32(    GbE1_GoodFramesSent,           0xD0077040,__READ_WRITE );
__IO_REG32(    GbE1_ExcessiveCollision,       0xD0077044,__READ_WRITE );
__IO_REG32(    GbE1_MulticastFramesSent,      0xD0077048,__READ_WRITE );
__IO_REG32(    GbE1_BroadcastFramesSent,      0xD007704C,__READ_WRITE );
__IO_REG32(    GbE1_UnrecogMACControl,        0xD0077050,__READ_WRITE );
__IO_REG32(    GbE1_FCSent,                   0xD0077054,__READ_WRITE );
__IO_REG32(    GbE1_GoodFCReceived,           0xD0077058,__READ_WRITE );
__IO_REG32(    GbE1_BadFCReceived,            0xD007705C,__READ_WRITE );
__IO_REG32(    GbE1_Undersize,                0xD0077060,__READ_WRITE );
__IO_REG32(    GbE1_Fragments,                0xD0077064,__READ_WRITE );
__IO_REG32(    GbE1_Oversize,                 0xD0077068,__READ_WRITE );
__IO_REG32(    GbE1_Jabber,                   0xD007706C,__READ_WRITE );
__IO_REG32(    GbE1_MACRcvError,              0xD0077070,__READ_WRITE );
__IO_REG32(    GbE1_BadCRC,                   0xD0077074,__READ_WRITE );
__IO_REG32(    GbE1_Collisions,               0xD0077078,__READ_WRITE );
__IO_REG32(    GbE1_LateCollision,            0xD007707C,__READ_WRITE );

/***************************************************************************
 **
 ** USB2 0
 **
 ***************************************************************************/
__IO_REG32_BIT(USB20_BCSR,            0xD0050300,__READ_WRITE ,__usb2_bcr_bits);
__IO_REG32_BIT(USB20_BICR,            0xD0050310,__READ       ,__usb2_bicr_bits);
__IO_REG32_BIT(USB20_BIMR,            0xD0050314,__READ_WRITE ,__usb2_bicr_bits);
__IO_REG32(    USB20_BEAR,            0xD005031C,__READ       );
__IO_REG32_BIT(USB20_PHYC0R,          0xD0050360,__READ_WRITE ,__usb2_phyc0r_bits);
__IO_REG32_BIT(USB20_PCR,             0xD0050400,__READ_WRITE ,__usb2_pcr_bits);
__IO_REG32_BIT(USB20_PHYPLLCR,        0xD0050410,__READ_WRITE ,__usb2_phypllcr_bits);
__IO_REG32_BIT(USB20_PHYTXCR,         0xD0050420,__READ_WRITE ,__usb2_phytxcr_bits);
__IO_REG32_BIT(USB20_PHYRXCR,         0xD0050430,__READ_WRITE ,__usb2_phyrxcr_bits);
__IO_REG32_BIT(USB20_PHYIVREFCR,      0xD0050440,__READ_WRITE ,__usb2_phyivrefcr_bits);
__IO_REG32_BIT(USB20_WCR0,            0xD0050320,__READ_WRITE ,__usb2_wcr_bits);
__IO_REG32_BIT(USB20_WBR0,            0xD0050324,__READ_WRITE ,__usb2_wbr_bits);
__IO_REG32_BIT(USB20_WCR1,            0xD0050330,__READ_WRITE ,__usb2_wcr_bits);
__IO_REG32_BIT(USB20_WBR1,            0xD0050334,__READ_WRITE ,__usb2_wbr_bits);
__IO_REG32_BIT(USB20_WCR2,            0xD0050340,__READ_WRITE ,__usb2_wcr_bits);
__IO_REG32_BIT(USB20_WBR2,            0xD0050344,__READ_WRITE ,__usb2_wbr_bits);
__IO_REG32_BIT(USB20_WCR3,            0xD0050350,__READ_WRITE ,__usb2_wcr_bits);
__IO_REG32_BIT(USB20_WBR3,            0xD0050354,__READ_WRITE ,__usb2_wbr_bits);
#if 0
__IO_REG32_BIT(USB20_ID,              0xD0050000,__READ_WRITE ,__usb2_id_bits);
__IO_REG32_BIT(USB20_HWGENERAL,       0xD0050004,__READ_WRITE ,__usb2_hwgeneral_bits);
__IO_REG32_BIT(USB20_HWHOST,          0xD0050008,__READ_WRITE ,__usb2_hwhost_bits);
__IO_REG32_BIT(USB20_HWDEVICE,        0xD005000C,__READ_WRITE ,__usb2_hwdevice_bits);
__IO_REG32_BIT(USB20_HWTXBUF,         0xD0050010,__READ_WRITE ,__usb2_hwtxbuf_bits);
__IO_REG32_BIT(USB20_HWRXBUF,         0xD0050014,__READ_WRITE ,__usb2_hwrxbuf_bits);
__IO_REG32_BIT(USB20_HWTTTXBUF,       0xD0050018,__READ_WRITE ,__usb2_hwtttxbuf_bits);
__IO_REG32_BIT(USB20_HWTTRXBUF,       0xD005001C,__READ_WRITE ,__usb2_hwttrxbuf_bits);
__IO_REG8_BIT( USB20_CAPLENGTH,       0xD0050100,__READ_WRITE ,__usb2_caplength_bits);
__IO_REG16_BIT(USB20_HCIVERSION,      0xD0050102,__READ_WRITE ,__usb2_hciversion_bits);
__IO_REG32_BIT(USB20_HCSPARAMS,       0xD0050104,__READ_WRITE ,__usb2_hcsparams_bits);
__IO_REG32_BIT(USB20_HCCPARAMS,       0xD0050108,__READ_WRITE ,__usb2_hccparams_bits);
__IO_REG16_BIT(USB20_DCIVERSION,      0xD0050120,__READ_WRITE ,__usb2_dciversion_bits);
__IO_REG32_BIT(USB20_DCCPARAMS,       0xD0050124,__READ_WRITE ,__usb2_dccparams_bits);
__IO_REG32_BIT(USB20_USBCMD,          0xD0050140,__READ_WRITE ,__usb2_usbcmd_bits);
__IO_REG32_BIT(USB20_USBSTS,          0xD0050144,__READ_WRITE ,__usb2_usbsts_bits);
__IO_REG32_BIT(USB20_USBINTR,         0xD0050148,__READ_WRITE ,__usb2_usbintr_bits);
__IO_REG32_BIT(USB20_FRINDEX,         0xD005014C,__READ_WRITE ,__usb2_frindex_bits);
__IO_REG32_BIT(USB20_PERIODICLISTBASE,0xD0050154,__READ_WRITE ,__usb2_periodiclistbase_bits);
__IO_REG32_BIT(USB20_ASYNCLISTADDR,   0xD0050158,__READ_WRITE ,__usb2_asynclistaddr_bits);
__IO_REG32_BIT(USB20_TTCTRL,          0xD005015C,__READ_WRITE ,__usb2_ttctrl_bits);
__IO_REG32_BIT(USB20_BURSTSIZE,       0xD0050160,__READ_WRITE ,__usb2_burstsize_bits);
__IO_REG32_BIT(USB20_TXFILLTUNING,    0xD0050164,__READ_WRITE ,__usb2_txfilltuning_bits);
__IO_REG32_BIT(USB20_TXTTFILLTUNING,  0xD0050168,__READ_WRITE ,__usb2_txttfilltuning_bits);
__IO_REG32_BIT(USB20_CONFIGFLAG,      0xD0050180,__READ_WRITE ,__usb2_configflag_bits);
__IO_REG32_BIT(USB20_PORTSC1,         0xD0050184,__READ_WRITE ,__usb2_portsc1_bits);
__IO_REG32_BIT(USB20_OTGSC,           0xD00501A4,__READ_WRITE ,__usb2_otgsc_bits);
__IO_REG32_BIT(USB20_USBMODE,         0xD00501A8,__READ_WRITE ,__usb2_usbmode_bits);
__IO_REG32_BIT(USB20_ENPDTSETUPSTAT,  0xD00501AC,__READ_WRITE ,__usb2_enpdtsetupstat_bits);
__IO_REG32_BIT(USB20_ENDPTPRIME,      0xD00501B0,__READ_WRITE ,__usb2_endptprime_bits);
__IO_REG32_BIT(USB20_ENDPTFLUSH,      0xD00501B4,__READ_WRITE ,__usb2_endptflush_bits);
__IO_REG32_BIT(USB20_ENDPTSTATUS,     0xD00501B8,__READ_WRITE ,__usb2_endptstatus_bits);
__IO_REG32_BIT(USB20_ENDPTCOMPLETE,   0xD00501BC,__READ_WRITE ,__usb2_endptcomplete_bits);
__IO_REG32_BIT(USB20_ENDPTCTRL0,      0xD00501C0,__READ_WRITE ,__usb2_endptctrl0_bits);
__IO_REG32_BIT(USB20_ENDPTCTRL1,      0xD00501C4,__READ_WRITE ,__usb2_endptctrl1_bits);
__IO_REG32_BIT(USB20_ENDPTCTRL2,      0xD00501C8,__READ_WRITE ,__usb2_endptctrl2_bits);
__IO_REG32_BIT(USB20_ENDPTCTRL3,      0xD00501CC,__READ_WRITE ,__usb2_endptctrl3_bits);
#endif

/***************************************************************************
 **
 ** USB2 1
 **
 ***************************************************************************/
__IO_REG32_BIT(USB21_BCSR,            0xD0051300,__READ_WRITE ,__usb2_bcr_bits);
__IO_REG32_BIT(USB21_BICR,            0xD0051310,__READ       ,__usb2_bicr_bits);
__IO_REG32_BIT(USB21_BIMR,            0xD0051314,__READ_WRITE ,__usb2_bicr_bits);
__IO_REG32(    USB21_BEAR,            0xD005131C,__READ       );
__IO_REG32_BIT(USB21_PHYC0R,          0xD0051360,__READ_WRITE ,__usb2_phyc0r_bits);
__IO_REG32_BIT(USB21_PCR,             0xD0051400,__READ_WRITE ,__usb2_pcr_bits);
__IO_REG32_BIT(USB21_PHYPLLCR,        0xD0051410,__READ_WRITE ,__usb2_phypllcr_bits);
__IO_REG32_BIT(USB21_PHYTXCR,         0xD0051420,__READ_WRITE ,__usb2_phytxcr_bits);
__IO_REG32_BIT(USB21_PHYRXCR,         0xD0051430,__READ_WRITE ,__usb2_phyrxcr_bits);
__IO_REG32_BIT(USB21_PHYIVREFCR,      0xD0051440,__READ_WRITE ,__usb2_phyivrefcr_bits);
__IO_REG32_BIT(USB21_WCR0,            0xD0051320,__READ_WRITE ,__usb2_wcr_bits);
__IO_REG32_BIT(USB21_WBR0,            0xD0051324,__READ_WRITE ,__usb2_wbr_bits);
__IO_REG32_BIT(USB21_WCR1,            0xD0051330,__READ_WRITE ,__usb2_wcr_bits);
__IO_REG32_BIT(USB21_WBR1,            0xD0051334,__READ_WRITE ,__usb2_wbr_bits);
__IO_REG32_BIT(USB21_WCR2,            0xD0051340,__READ_WRITE ,__usb2_wcr_bits);
__IO_REG32_BIT(USB21_WBR2,            0xD0051344,__READ_WRITE ,__usb2_wbr_bits);
__IO_REG32_BIT(USB21_WCR3,            0xD0051350,__READ_WRITE ,__usb2_wcr_bits);
__IO_REG32_BIT(USB21_WBR3,            0xD0051354,__READ_WRITE ,__usb2_wbr_bits);
#if 0
__IO_REG32_BIT(USB21_ID,              0xD0051000,__READ_WRITE ,__usb2_id_bits);
__IO_REG32_BIT(USB21_HWGENERAL,       0xD0051004,__READ_WRITE ,__usb2_hwgeneral_bits);
__IO_REG32_BIT(USB21_HWHOST,          0xD0051008,__READ_WRITE ,__usb2_hwhost_bits);
__IO_REG32_BIT(USB21_HWDEVICE,        0xD005100C,__READ_WRITE ,__usb2_hwdevice_bits);
__IO_REG32_BIT(USB21_HWTXBUF,         0xD0051010,__READ_WRITE ,__usb2_hwtxbuf_bits);
__IO_REG32_BIT(USB21_HWRXBUF,         0xD0051014,__READ_WRITE ,__usb2_hwrxbuf_bits);
__IO_REG32_BIT(USB21_HWTTTXBUF,       0xD0051018,__READ_WRITE ,__usb2_hwtttxbuf_bits);
__IO_REG32_BIT(USB21_HWTTRXBUF,       0xD005101C,__READ_WRITE ,__usb2_hwttrxbuf_bits);
__IO_REG8_BIT( USB21_CAPLENGTH,       0xD0051100,__READ_WRITE ,__usb2_caplength_bits);
__IO_REG16_BIT(USB21_HCIVERSION,      0xD0051102,__READ_WRITE ,__usb2_hciversion_bits);
__IO_REG32_BIT(USB21_HCSPARAMS,       0xD0051104,__READ_WRITE ,__usb2_hcsparams_bits);
__IO_REG32_BIT(USB21_HCCPARAMS,       0xD0051108,__READ_WRITE ,__usb2_hccparams_bits);
__IO_REG16_BIT(USB21_DCIVERSION,      0xD0051120,__READ_WRITE ,__usb2_dciversion_bits);
__IO_REG32_BIT(USB21_DCCPARAMS,       0xD0051124,__READ_WRITE ,__usb2_dccparams_bits);
__IO_REG32_BIT(USB21_USBCMD,          0xD0051140,__READ_WRITE ,__usb2_usbcmd_bits);
__IO_REG32_BIT(USB21_USBSTS,          0xD0051144,__READ_WRITE ,__usb2_usbsts_bits);
__IO_REG32_BIT(USB21_USBINTR,         0xD0051148,__READ_WRITE ,__usb2_usbintr_bits);
__IO_REG32_BIT(USB21_FRINDEX,         0xD005114C,__READ_WRITE ,__usb2_frindex_bits);
__IO_REG32_BIT(USB21_PERIODICLISTBASE,0xD0051154,__READ_WRITE ,__usb2_periodiclistbase_bits);
__IO_REG32_BIT(USB21_ASYNCLISTADDR,   0xD0051158,__READ_WRITE ,__usb2_asynclistaddr_bits);
__IO_REG32_BIT(USB21_TTCTRL,          0xD005115C,__READ_WRITE ,__usb2_ttctrl_bits);
__IO_REG32_BIT(USB21_BURSTSIZE,       0xD0051160,__READ_WRITE ,__usb2_burstsize_bits);
__IO_REG32_BIT(USB21_TXFILLTUNING,    0xD0051164,__READ_WRITE ,__usb2_txfilltuning_bits);
__IO_REG32_BIT(USB21_TXTTFILLTUNING,  0xD0051168,__READ_WRITE ,__usb2_txttfilltuning_bits);
__IO_REG32_BIT(USB21_CONFIGFLAG,      0xD0051180,__READ_WRITE ,__usb2_configflag_bits);
__IO_REG32_BIT(USB21_PORTSC1,         0xD0051184,__READ_WRITE ,__usb2_portsc1_bits);
__IO_REG32_BIT(USB21_OTGSC,           0xD00511A4,__READ_WRITE ,__usb2_otgsc_bits);
__IO_REG32_BIT(USB21_USBMODE,         0xD00511A8,__READ_WRITE ,__usb2_usbmode_bits);
__IO_REG32_BIT(USB21_ENPDTSETUPSTAT,  0xD00511AC,__READ_WRITE ,__usb2_enpdtsetupstat_bits);
__IO_REG32_BIT(USB21_ENDPTPRIME,      0xD00511B0,__READ_WRITE ,__usb2_endptprime_bits);
__IO_REG32_BIT(USB21_ENDPTFLUSH,      0xD00511B4,__READ_WRITE ,__usb2_endptflush_bits);
__IO_REG32_BIT(USB21_ENDPTSTATUS,     0xD00511B8,__READ_WRITE ,__usb2_endptstatus_bits);
__IO_REG32_BIT(USB21_ENDPTCOMPLETE,   0xD00511BC,__READ_WRITE ,__usb2_endptcomplete_bits);
__IO_REG32_BIT(USB21_ENDPTCTRL0,      0xD00511C0,__READ_WRITE ,__usb2_endptctrl0_bits);
__IO_REG32_BIT(USB21_ENDPTCTRL1,      0xD00511C4,__READ_WRITE ,__usb2_endptctrl1_bits);
__IO_REG32_BIT(USB21_ENDPTCTRL2,      0xD00511C8,__READ_WRITE ,__usb2_endptctrl2_bits);
__IO_REG32_BIT(USB21_ENDPTCTRL3,      0xD00511CC,__READ_WRITE ,__usb2_endptctrl3_bits);
#endif

/***************************************************************************
 **
 ** USB2 2
 **
 ***************************************************************************/
__IO_REG32_BIT(USB22_BCSR,            0xD0052300,__READ_WRITE ,__usb2_bcr_bits);
__IO_REG32_BIT(USB22_BICR,            0xD0052310,__READ       ,__usb2_bicr_bits);
__IO_REG32_BIT(USB22_BIMR,            0xD0052314,__READ_WRITE ,__usb2_bicr_bits);
__IO_REG32(    USB22_BEAR,            0xD005231C,__READ       );
__IO_REG32_BIT(USB22_PHYC0R,          0xD0052360,__READ_WRITE ,__usb2_phyc0r_bits);
__IO_REG32_BIT(USB22_PCR,             0xD0052400,__READ_WRITE ,__usb2_pcr_bits);
__IO_REG32_BIT(USB22_PHYPLLCR,        0xD0052410,__READ_WRITE ,__usb2_phypllcr_bits);
__IO_REG32_BIT(USB22_PHYTXCR,         0xD0052420,__READ_WRITE ,__usb2_phytxcr_bits);
__IO_REG32_BIT(USB22_PHYRXCR,         0xD0052430,__READ_WRITE ,__usb2_phyrxcr_bits);
__IO_REG32_BIT(USB22_PHYIVREFCR,      0xD0052440,__READ_WRITE ,__usb2_phyivrefcr_bits);
__IO_REG32_BIT(USB22_WCR0,            0xD0052320,__READ_WRITE ,__usb2_wcr_bits);
__IO_REG32_BIT(USB22_WBR0,            0xD0052324,__READ_WRITE ,__usb2_wbr_bits);
__IO_REG32_BIT(USB22_WCR1,            0xD0052330,__READ_WRITE ,__usb2_wcr_bits);
__IO_REG32_BIT(USB22_WBR1,            0xD0052334,__READ_WRITE ,__usb2_wbr_bits);
__IO_REG32_BIT(USB22_WCR2,            0xD0052340,__READ_WRITE ,__usb2_wcr_bits);
__IO_REG32_BIT(USB22_WBR2,            0xD0052344,__READ_WRITE ,__usb2_wbr_bits);
__IO_REG32_BIT(USB22_WCR3,            0xD0052350,__READ_WRITE ,__usb2_wcr_bits);
__IO_REG32_BIT(USB22_WBR3,            0xD0052354,__READ_WRITE ,__usb2_wbr_bits);
#if 0
__IO_REG32_BIT(USB22_ID,              0xD0052000,__READ_WRITE ,__usb2_id_bits);
__IO_REG32_BIT(USB22_HWGENERAL,       0xD0052004,__READ_WRITE ,__usb2_hwgeneral_bits);
__IO_REG32_BIT(USB22_HWHOST,          0xD0052008,__READ_WRITE ,__usb2_hwhost_bits);
__IO_REG32_BIT(USB22_HWDEVICE,        0xD005200C,__READ_WRITE ,__usb2_hwdevice_bits);
__IO_REG32_BIT(USB22_HWTXBUF,         0xD0052010,__READ_WRITE ,__usb2_hwtxbuf_bits);
__IO_REG32_BIT(USB22_HWRXBUF,         0xD0052014,__READ_WRITE ,__usb2_hwrxbuf_bits);
__IO_REG32_BIT(USB22_HWTTTXBUF,       0xD0052018,__READ_WRITE ,__usb2_hwtttxbuf_bits);
__IO_REG32_BIT(USB22_HWTTRXBUF,       0xD005201C,__READ_WRITE ,__usb2_hwttrxbuf_bits);
__IO_REG8_BIT( USB22_CAPLENGTH,       0xD0052100,__READ_WRITE ,__usb2_caplength_bits);
__IO_REG16_BIT(USB22_HCIVERSION,      0xD0052102,__READ_WRITE ,__usb2_hciversion_bits);
__IO_REG32_BIT(USB22_HCSPARAMS,       0xD0052104,__READ_WRITE ,__usb2_hcsparams_bits);
__IO_REG32_BIT(USB22_HCCPARAMS,       0xD0052108,__READ_WRITE ,__usb2_hccparams_bits);
__IO_REG16_BIT(USB22_DCIVERSION,      0xD0052120,__READ_WRITE ,__usb2_dciversion_bits);
__IO_REG32_BIT(USB22_DCCPARAMS,       0xD0052124,__READ_WRITE ,__usb2_dccparams_bits);
__IO_REG32_BIT(USB22_USBCMD,          0xD0052140,__READ_WRITE ,__usb2_usbcmd_bits);
__IO_REG32_BIT(USB22_USBSTS,          0xD0052144,__READ_WRITE ,__usb2_usbsts_bits);
__IO_REG32_BIT(USB22_USBINTR,         0xD0052148,__READ_WRITE ,__usb2_usbintr_bits);
__IO_REG32_BIT(USB22_FRINDEX,         0xD005214C,__READ_WRITE ,__usb2_frindex_bits);
__IO_REG32_BIT(USB22_PERIODICLISTBASE,0xD0052154,__READ_WRITE ,__usb2_periodiclistbase_bits);
__IO_REG32_BIT(USB22_ASYNCLISTADDR,   0xD0052158,__READ_WRITE ,__usb2_asynclistaddr_bits);
__IO_REG32_BIT(USB22_TTCTRL,          0xD005215C,__READ_WRITE ,__usb2_ttctrl_bits);
__IO_REG32_BIT(USB22_BURSTSIZE,       0xD0052160,__READ_WRITE ,__usb2_burstsize_bits);
__IO_REG32_BIT(USB22_TXFILLTUNING,    0xD0052164,__READ_WRITE ,__usb2_txfilltuning_bits);
__IO_REG32_BIT(USB22_TXTTFILLTUNING,  0xD0052168,__READ_WRITE ,__usb2_txttfilltuning_bits);
__IO_REG32_BIT(USB22_CONFIGFLAG,      0xD0052180,__READ_WRITE ,__usb2_configflag_bits);
__IO_REG32_BIT(USB22_PORTSC1,         0xD0052184,__READ_WRITE ,__usb2_portsc1_bits);
__IO_REG32_BIT(USB22_OTGSC,           0xD00521A4,__READ_WRITE ,__usb2_otgsc_bits);
__IO_REG32_BIT(USB22_USBMODE,         0xD00521A8,__READ_WRITE ,__usb2_usbmode_bits);
__IO_REG32_BIT(USB22_ENPDTSETUPSTAT,  0xD00521AC,__READ_WRITE ,__usb2_enpdtsetupstat_bits);
__IO_REG32_BIT(USB22_ENDPTPRIME,      0xD00521B0,__READ_WRITE ,__usb2_endptprime_bits);
__IO_REG32_BIT(USB22_ENDPTFLUSH,      0xD00521B4,__READ_WRITE ,__usb2_endptflush_bits);
__IO_REG32_BIT(USB22_ENDPTSTATUS,     0xD00521B8,__READ_WRITE ,__usb2_endptstatus_bits);
__IO_REG32_BIT(USB22_ENDPTCOMPLETE,   0xD00521BC,__READ_WRITE ,__usb2_endptcomplete_bits);
__IO_REG32_BIT(USB22_ENDPTCTRL0,      0xD00521C0,__READ_WRITE ,__usb2_endptctrl0_bits);
__IO_REG32_BIT(USB22_ENDPTCTRL1,      0xD00521C4,__READ_WRITE ,__usb2_endptctrl1_bits);
__IO_REG32_BIT(USB22_ENDPTCTRL2,      0xD00521C8,__READ_WRITE ,__usb2_endptctrl2_bits);
__IO_REG32_BIT(USB22_ENDPTCTRL3,      0xD00521CC,__READ_WRITE ,__usb2_endptctrl3_bits);
#endif

/***************************************************************************
 **
 ** AES
 **
 ***************************************************************************/
__IO_REG32(    AESDKC7R,              0xD009DDC0,__READ_WRITE );
__IO_REG32(    AESDKC6R,              0xD009DDC4,__READ_WRITE );
__IO_REG32(    AESDKC5R,              0xD009DDC8,__READ_WRITE );
__IO_REG32(    AESDKC4R,              0xD009DDCC,__READ_WRITE );
__IO_REG32(    AESDKC3R,              0xD009DDD0,__READ_WRITE );
__IO_REG32(    AESDKC2R,              0xD009DDD4,__READ_WRITE );
__IO_REG32(    AESDKC1R,              0xD009DDD8,__READ_WRITE );
__IO_REG32(    AESDKC0R,              0xD009DDDC,__READ_WRITE );
__IO_REG32(    AESDDIOC3R,            0xD009DDE0,__READ_WRITE );
__IO_REG32(    AESDDIOC2R,            0xD009DDE4,__READ_WRITE );
__IO_REG32(    AESDDIOC1R,            0xD009DDE8,__READ_WRITE );
__IO_REG32(    AESDDIOC0R,            0xD009DDEC,__READ_WRITE );
__IO_REG32_BIT(AESDCR,                0xD009DDF0,__READ_WRITE ,__aesdcr_bits);
__IO_REG32(    AESEKC7R,              0xD009DD80,__READ_WRITE );
__IO_REG32(    AESEKC6R,              0xD009DD84,__READ_WRITE );
__IO_REG32(    AESEKC5R,              0xD009DD88,__READ_WRITE );
__IO_REG32(    AESEKC4R,              0xD009DD8C,__READ_WRITE );
__IO_REG32(    AESEKC3R,              0xD009DD90,__READ_WRITE );
__IO_REG32(    AESEKC2R,              0xD009DD94,__READ_WRITE );
__IO_REG32(    AESEKC1R,              0xD009DD98,__READ_WRITE );
__IO_REG32(    AESEKC0R,              0xD009DD9C,__READ_WRITE );
__IO_REG32(    AESEDIOC3R,            0xD009DDA0,__READ_WRITE );
__IO_REG32(    AESEDIOC2R,            0xD009DDA4,__READ_WRITE );
__IO_REG32(    AESEDIOC1R,            0xD009DDA8,__READ_WRITE );
__IO_REG32(    AESEDIOC0R,            0xD009DDAC,__READ_WRITE );
__IO_REG32_BIT(AESECR,                0xD009DDB0,__READ_WRITE ,__aesecr_bits);

/***************************************************************************
 **
 ** DEC
 **
 ***************************************************************************/
__IO_REG32(    DECIVLR,               0xD009DD40,__READ_WRITE );
__IO_REG32(    DECIVHR,               0xD009DD44,__READ_WRITE );
__IO_REG32(    DECK0LR,               0xD009DD48,__READ_WRITE );
__IO_REG32(    DECK0HR,               0xD009DD4C,__READ_WRITE );
__IO_REG32(    DECK1LR,               0xD009DD50,__READ_WRITE );
__IO_REG32(    DECK1HR,               0xD009DD54,__READ_WRITE );
__IO_REG32_BIT(DECCR,                 0xD009DD58,__READ_WRITE ,__deccr_bits);
__IO_REG32(    DECK2LR,               0xD009DD60,__READ_WRITE );
__IO_REG32(    DECK2HR,               0xD009DD64,__READ_WRITE );
__IO_REG32(    DECDBLR,               0xD009DD70,__WRITE      );
__IO_REG32(    DECDBHR,               0xD009DD74,__WRITE      );
__IO_REG32(    DECDOLR,               0xD009DD78,__READ       );
__IO_REG32(    DECDOHR,               0xD009DD7C,__READ       );

/***************************************************************************
 **
 ** CESA (Cryptographic Engine and Security Accelerator)
 **
 ***************************************************************************/
__IO_REG32_BIT(CESAICR,               0xD009DE20,__READ       ,__cesaicr_bits);
__IO_REG32_BIT(CESAIMR,               0xD009DE24,__READ_WRITE ,__cesaicr_bits);

/***************************************************************************
 **
 ** Security Accelerator
 **
 ***************************************************************************/
__IO_REG32_BIT(SACR,                  0xD009DE00,__READ_WRITE ,__sacr_bits);
__IO_REG32_BIT(SADPR,                 0xD009DE04,__READ_WRITE ,__sadpr_bits);
__IO_REG32_BIT(SACNFR,                0xD009DE08,__READ_WRITE ,__sacnfr_bits);
__IO_REG32_BIT(SAASR,                 0xD009DE0C,__READ_WRITE ,__saasr_bits);

/***************************************************************************
 **
 ** SHA-1 and MD5 Interface
 **
 ***************************************************************************/
__IO_REG32(    SHA1MD5IVAR,           0xD009DD00,__READ_WRITE );
__IO_REG32(    SHA1MD5IVBR,           0xD009DD04,__READ_WRITE );
__IO_REG32(    SHA1MD5IVCR,           0xD009DD08,__READ_WRITE );
__IO_REG32(    SHA1MD5IVDR,           0xD009DD0C,__READ_WRITE );
__IO_REG32(    SHA1MD5IVER,           0xD009DD10,__READ_WRITE );
__IO_REG32(    SHA1MD5ACR,            0xD009DD18,__READ       );
__IO_REG32(    SHA1MD5BCLR,           0xD009DD20,__READ       );
__IO_REG32(    SHA1MD5BCHR,           0xD009DD24,__READ       );
__IO_REG32(    SHA1MD5DIR,            0xD009DD38,__READ       );

/***************************************************************************
 **
 ** TDMA
 **
 ***************************************************************************/
__IO_REG32(    DTMABAR0,              0xD0090A00,__READ_WRITE );
__IO_REG32(    DTMABAR1,              0xD0090A08,__READ_WRITE );
__IO_REG32(    DTMABAR2,              0xD0090A10,__READ_WRITE );
__IO_REG32(    DTMABAR3,              0xD0090A18,__READ_WRITE );
__IO_REG32_BIT(DTMAWCR0,              0xD0090A04,__READ_WRITE ,__dtmawcr_bits);
__IO_REG32_BIT(DTMAWCR1,              0xD0090A0C,__READ_WRITE ,__dtmawcr_bits);
__IO_REG32_BIT(DTMAWCR2,              0xD0090A14,__READ_WRITE ,__dtmawcr_bits);
__IO_REG32_BIT(DTMAWCR3,              0xD0090A1C,__READ_WRITE ,__dtmawcr_bits);
__IO_REG32_BIT(DTMACR,                0xD0090840,__READ_WRITE ,__dtmacr_bits);
__IO_REG32_BIT(DTMBCR,                0xD0090800,__READ_WRITE ,__dtmbcr_bits);
__IO_REG32(    DTMSAR,                0xD0090810,__READ_WRITE );
__IO_REG32(    DTMDAR,                0xD0090820,__READ_WRITE );
__IO_REG32(    DTMDNDPR,              0xD0090830,__READ_WRITE );
__IO_REG32(    DTMDCDPR,              0xD0090870,__READ       );
__IO_REG32_BIT(DTMDECR,               0xD00908C8,__READ_WRITE ,__dtmdecr_bits);
__IO_REG32_BIT(DTMDEMR,               0xD00908CC,__READ_WRITE ,__dtmdecr_bits);

/***************************************************************************
 **
 ** TDM
 **
 ***************************************************************************/
__IO_REG32_BIT(TDM_CSU_SCPR,          0xD00B3100,__READ_WRITE ,__tdm_csu_scpr_bits);
__IO_REG32_BIT(TDM_CSU_GCR,           0xD00B3104,__READ_WRITE ,__tdm_csu_gcr_bits);
__IO_REG32_BIT(TDM_SPI_CR,            0xD00B3108,__READ_WRITE ,__tdm_spi_cr_bits);
__IO_REG32_BIT(TDM_CODEC_ACLR,        0xD00B3130,__READ_WRITE ,__tdm_codec_aclr_bits);
__IO_REG32_BIT(TDM_CODEC_ACHR,        0xD00B3134,__READ_WRITE ,__tdm_codec_achr_bits);
__IO_REG32_BIT(TDM_CODEC_RAC,         0xD00B3138,__READ_WRITE ,__tdm_codec_rac_bits);
__IO_REG32_BIT(TDM_CODEC_RDR,         0xD00B313C,__READ       ,__tdm_codec_rdr_bits);
__IO_REG32_BIT(TDM_CODEC_RAC1,        0xD00B3140,__READ_WRITE ,__tdm_codec_rac1_bits);
__IO_REG32_BIT(TDM_PCM_CR,            0xD00B0000,__READ_WRITE ,__tdm_pcm_cr_bits);
__IO_REG32_BIT(TDM_CTSCR,             0xD00B0004,__READ_WRITE ,__tdm_ctscr_bits);
__IO_REG32_BIT(TDM_C0DCR,             0xD00B0008,__READ_WRITE ,__tdm_cdcr_bits);
__IO_REG32_BIT(TDM_C1DCR,             0xD00B000C,__READ_WRITE ,__tdm_cdcr_bits);
__IO_REG32_BIT(TDM_C0EDR,             0xD00B0010,__READ_WRITE ,__tdm_cedr_bits);
__IO_REG32_BIT(TDM_C1EDR,             0xD00B0020,__READ_WRITE ,__tdm_cedr_bits);
__IO_REG32_BIT(TDM_C0BOR,             0xD00B0014,__READ_WRITE ,__tdm_cbor_bits);
__IO_REG32_BIT(TDM_C1BOR,             0xD00B0024,__READ_WRITE ,__tdm_cbor_bits);
__IO_REG32(    TDM_C0TDSAR,           0xD00B0018,__READ_WRITE );
__IO_REG32(    TDM_C0RDSAR,           0xD00B001C,__READ_WRITE );
__IO_REG32(    TDM_C1TDSAR,           0xD00B0028,__READ_WRITE );
__IO_REG32(    TDM_C1RDSAR,           0xD00B002C,__READ_WRITE );
__IO_REG32_BIT(TDM_C0TSCR,            0xD00B0030,__READ_WRITE ,__tdm_ctotscr_bits);
__IO_REG32_BIT(TDM_C1TSCR,            0xD00B0034,__READ_WRITE ,__tdm_ctotscr_bits);
__IO_REG32_BIT(TDM_NTSR,              0xD00B0038,__READ_WRITE ,__tdm_ntsr_bits);
__IO_REG32_BIT(TDM_PCM_CRDR,          0xD00B003C,__READ_WRITE ,__tdm_pcm_crdr_bits);
__IO_REG32_BIT(TDM_IEMR,              0xD00B0040,__READ_WRITE ,__tdm_isr_bits);
__IO_REG32_BIT(TDM_ISMR,              0xD00B0048,__READ_WRITE ,__tdm_isr_bits);
__IO_REG32_BIT(TDM_IRSR,              0xD00B004C,__READ_WRITE ,__tdm_isr_bits);
__IO_REG32_BIT(TDM_ISR,               0xD00B0050,__READ_WRITE ,__tdm_isr_bits);
__IO_REG32(    TDM_DD,                0xD00B0054,__READ_WRITE );
__IO_REG32_BIT(TDM_MCR,               0xD00B0058,__READ_WRITE ,__tdm_mcr_bits);
__IO_REG32(    TDM_C0TDCAR,           0xD00B0060,__READ       );
__IO_REG32(    TDM_C1TDCAR,           0xD00B0068,__READ       );
__IO_REG32(    TDM_C0RDCAR,           0xD00B0064,__READ       );
__IO_REG32(    TDM_C1RDCAR,           0xD00B006C,__READ       );
__IO_REG32_BIT(TDM_CTSR,              0xD00B0070,__READ       ,__tdm_ctsr_bits);
__IO_REG32_BIT(TDM_RR,                0xD00B0074,__READ       ,__tdm_rr_bits);
__IO_REG32_BIT(TDM_C0DR,              0xD00B0078,__READ       ,__tdm_cdr_bits);
__IO_REG32_BIT(TDM_C1DR,              0xD00B007C,__READ       ,__tdm_cdr_bits);
__IO_REG32(    TDM_DMAAR1,            0xD00B0080,__READ       );
__IO_REG32_BIT(TDM_DMAAR2,            0xD00B0084,__READ       ,__tdm_dmaar2_bits);
__IO_REG32_BIT(TDM_C0WDCR,            0xD00B0088,__READ_WRITE ,__tdm_cwdcr_bits);
__IO_REG32_BIT(TDM_C1WDCR,            0xD00B008C,__READ_WRITE ,__tdm_cwdcr_bits);
__IO_REG32_BIT(TDM_IPMR,              0xD00B4000,__READ_WRITE ,__tdm_ipmr_bits);
__IO_REG32_BIT(TDM_MBUSCR,            0xD00B4010,__READ_WRITE ,__tdm_mbuscr_bits);
__IO_REG32_BIT(TDM_W0CR,              0xD00B4030,__READ_WRITE ,__tdm_wcr_bits);
__IO_REG32(    TDM_W0BR,              0xD00B4034,__READ_WRITE );
__IO_REG32_BIT(TDM_W1CR,              0xD00B4040,__READ_WRITE ,__tdm_wcr_bits);
__IO_REG32(    TDM_W1BR,              0xD00B4044,__READ_WRITE );
__IO_REG32_BIT(TDM_W2CR,              0xD00B4050,__READ_WRITE ,__tdm_wcr_bits);
__IO_REG32(    TDM_W2BR,              0xD00B4054,__READ_WRITE );
__IO_REG32_BIT(TDM_W3CR,              0xD00B4060,__READ_WRITE ,__tdm_wcr_bits);
__IO_REG32(    TDM_W3BR,              0xD00B4064,__READ_WRITE );
__IO_REG32_BIT(TDM_MBUSC1R,           0xD00B4070,__READ_WRITE ,__tdm_mbusc1r_bits);

/***************************************************************************
 **
 ** SATAHC0
 **
 ***************************************************************************/
__IO_REG32_BIT(SATAHC0_BDMACR,        0xD00A2224,__READ_WRITE ,__satahc_bdmacr_bits);
__IO_REG32_BIT(SATAHC0_BDMASR,        0xD00A2228,__READ_WRITE ,__satahc_bdmasr_bits);
__IO_REG32(    SATAHC0_DTLBAR,        0xD00A222C,__READ_WRITE );
__IO_REG32(    SATAHC0_DTHBAR,        0xD00A2230,__READ_WRITE );
__IO_REG32(    SATAHC0_DRLAR,         0xD00A2234,__READ_WRITE );
__IO_REG32(    SATAHC0_DRHAR,         0xD00A2238,__READ_WRITE );
__IO_REG32_BIT(SATAHC0_EDMACFGR,      0xD00A2000,__READ_WRITE ,__satahc_edmacfgr_bits);
__IO_REG32_BIT(SATAHC0_EDMAIECR,      0xD00A2008,__READ       ,__satahc_edmaiecr_bits);
__IO_REG32_BIT(SATAHC0_EDMAIEMR,      0xD00A200C,__READ_WRITE ,__satahc_edmaiecr_bits);
__IO_REG32(    SATAHC0_EDMARQBAHR,    0xD00A2010,__READ_WRITE );
__IO_REG32_BIT(SATAHC0_EDMARQIR,      0xD00A2014,__READ_WRITE ,__satahc_edmarqir_bits);
__IO_REG32_BIT(SATAHC0_EDMARQOR,      0xD00A2018,__READ_WRITE ,__satahc_edmarqor_bits);
__IO_REG32(    SATAHC0_EDMARSQBAHR,   0xD00A201C,__READ_WRITE );
__IO_REG32_BIT(SATAHC0_EDMARSQIR,     0xD00A2020,__READ_WRITE ,__satahc_edmarsqir_bits);
__IO_REG32_BIT(SATAHC0_EDMARSQOR,     0xD00A2024,__READ_WRITE ,__satahc_edmarsqor_bits);
__IO_REG32_BIT(SATAHC0_EDMACR,        0xD00A2028,__READ_WRITE ,__satahc_edmacr_bits);
__IO_REG32_BIT(SATAHC0_EDMASR,        0xD00A2030,__READ_WRITE ,__satahc_edmasr_bits);
__IO_REG32_BIT(SATAHC0_EDMAIORTR,     0xD00A2034,__READ_WRITE ,__satahc_edmaiortr_bits);
__IO_REG32_BIT(SATAHC0_EDMAICDTR,     0xD00A2040,__READ_WRITE ,__satahc_edmaicdtr_bits);
__IO_REG32_BIT(SATAHC0_EDMAHCR,       0xD00A2060,__READ_WRITE ,__satahc_edmaiecr_bits);
__IO_REG32(    SATAHC0_EDMANCQTCQSR,  0xD00A2094,__READ_WRITE );
__IO_REG32_BIT(SATAHC0_ICFGR,         0xD00A2050,__READ_WRITE ,__satahc_icfgr_bits);
__IO_REG32_BIT(SATAHC0_PLLCNFR,       0xD00A2054,__READ_WRITE ,__satahc_pllcnfr_bits);
__IO_REG32_BIT(SATAHC0_SSR,           0xD00A2300,__READ       ,__satahc_ssr_bits);
__IO_REG32_BIT(SATAHC0_SER,           0xD00A2304,__READ_WRITE ,__satahc_ser_bits);
__IO_REG32_BIT(SATAHC0_SCR,           0xD00A2308,__READ_WRITE ,__satahc_scr_bits);
__IO_REG32_BIT(SATAHC0_LTMR,          0xD00A230C,__READ_WRITE ,__satahc_ltmr_bits);
__IO_REG32_BIT(SATAHC0_PHYM3R,        0xD00A2310,__READ_WRITE ,__satahc_phym3r_bits);
__IO_REG32_BIT(SATAHC0_PHYM4R,        0xD00A2314,__READ_WRITE ,__satahc_phym4r_bits);
__IO_REG32_BIT(SATAHC0_PHYM1R,        0xD00A232C,__READ_WRITE ,__satahc_phym1r_bits);
__IO_REG32_BIT(SATAHC0_PHYM2R,        0xD00A2330,__READ_WRITE ,__satahc_phym2r_bits);
__IO_REG32_BIT(SATAHC0_BISTCR,        0xD00A2334,__READ_WRITE ,__satahc_bistcr_bits);
__IO_REG32(    SATAHC0_BISTDW1R,      0xD00A2338,__READ_WRITE );
__IO_REG32(    SATAHC0_BISTDW2R,      0xD00A233C,__READ_WRITE );
__IO_REG32(    SATAHC0_SEIMR,         0xD00A2340,__READ_WRITE );
__IO_REG32_BIT(SATAHC0_ICR,           0xD00A2344,__READ_WRITE ,__satahc_icr_bits);
__IO_REG32_BIT(SATAHC0_ITCR,          0xD00A2348,__READ_WRITE ,__satahc_itcr_bits);
__IO_REG32_BIT(SATAHC0_ISR,           0xD00A234C,__READ       ,__satahc_isr_bits);
__IO_REG32(    SATAHC0_VUR,           0xD00A235C,__READ_WRITE );
__IO_REG32_BIT(SATAHC0_FISCR,         0xD00A2360,__READ_WRITE ,__satahc_fiscr_bits);
__IO_REG32_BIT(SATAHC0_FISICR,        0xD00A2364,__READ       ,__satahc_fisicr_bits);
__IO_REG32_BIT(SATAHC0_FISIMR,        0xD00A2368,__READ_WRITE ,__satahc_fisicr_bits);
__IO_REG32(    SATAHC0_FISWD0,        0xD00A2370,__READ       );
__IO_REG32(    SATAHC0_FISWD1,        0xD00A2374,__READ       );
__IO_REG32(    SATAHC0_FISWD2,        0xD00A2378,__READ       );
__IO_REG32(    SATAHC0_FISWD3,        0xD00A237C,__READ       );
__IO_REG32(    SATAHC0_FISWD4,        0xD00A2380,__READ       );
__IO_REG32(    SATAHC0_FISWD5,        0xD00A2384,__READ       );
__IO_REG32(    SATAHC0_FISWD6,        0xD00A2388,__READ       );
//__IO_REG32_BIT(SATAHC0_PHYM8GEN2R,    0xD00A2390,__READ_WRITE ,__satahc_phym8gen2r_bits);
//__IO_REG32_BIT(SATAHC0_PHYM8GEN1R,    0xD00A2394,__READ_WRITE ,__satahc_phym8gen1r_bits);
__IO_REG32_BIT(SATAHC0_PHYM9GEN2R,    0xD00A2398,__READ_WRITE ,__satahc_phym9genr_bits);
__IO_REG32_BIT(SATAHC0_PHYM9GEN1R,    0xD00A239C,__READ_WRITE ,__satahc_phym9genr_bits);
__IO_REG32_BIT(SATAHC0_PHYCFGR,       0xD00A23A0,__READ_WRITE ,__satahc_phycfgr_bits);
__IO_REG32_BIT(SATAHC0_PHYTCTLR,      0xD00A23A4,__READ_WRITE ,__satahc_phytctlr_bits);
__IO_REG32_BIT(SATAHC0_PHYMODE10R,    0xD00A23A8,__READ_WRITE ,__satahc_phymode10r_bits);
//__IO_REG32_BIT(SATAHC0_PHYMODE12R,    0xD00A23AC,__READ_WRITE ,__satahc_phymode12r_bits);

/***************************************************************************
 **
 ** SATAHC1
 **
 ***************************************************************************/
__IO_REG32_BIT(SATAHC1_BDMACR,        0xD00A4224,__READ_WRITE ,__satahc_bdmacr_bits);
__IO_REG32_BIT(SATAHC1_BDMASR,        0xD00A4228,__READ_WRITE ,__satahc_bdmasr_bits);
__IO_REG32(    SATAHC1_DTLBAR,        0xD00A422C,__READ_WRITE );
__IO_REG32(    SATAHC1_DTHBAR,        0xD00A4230,__READ_WRITE );
__IO_REG32(    SATAHC1_DRLAR,         0xD00A4234,__READ_WRITE );
__IO_REG32(    SATAHC1_DRHAR,         0xD00A4238,__READ_WRITE );
__IO_REG32_BIT(SATAHC1_EDMACFGR,      0xD00A4000,__READ_WRITE ,__satahc_edmacfgr_bits);
__IO_REG32_BIT(SATAHC1_EDMAIECR,      0xD00A4008,__READ       ,__satahc_edmaiecr_bits);
__IO_REG32_BIT(SATAHC1_EDMAIEMR,      0xD00A400C,__READ_WRITE ,__satahc_edmaiecr_bits);
__IO_REG32(    SATAHC1_EDMARQBAHR,    0xD00A4010,__READ_WRITE );
__IO_REG32_BIT(SATAHC1_EDMARQIR,      0xD00A4014,__READ_WRITE ,__satahc_edmarqir_bits);
__IO_REG32_BIT(SATAHC1_EDMARQOR,      0xD00A4018,__READ_WRITE ,__satahc_edmarqor_bits);
__IO_REG32(    SATAHC1_EDMARSQBAHR,   0xD00A401C,__READ_WRITE );
__IO_REG32_BIT(SATAHC1_EDMARSQIR,     0xD00A4020,__READ_WRITE ,__satahc_edmarsqir_bits);
__IO_REG32_BIT(SATAHC1_EDMARSQOR,     0xD00A4024,__READ_WRITE ,__satahc_edmarsqor_bits);
__IO_REG32_BIT(SATAHC1_EDMACR,        0xD00A4028,__READ_WRITE ,__satahc_edmacr_bits);
__IO_REG32_BIT(SATAHC1_EDMASR,        0xD00A4030,__READ_WRITE ,__satahc_edmasr_bits);
__IO_REG32_BIT(SATAHC1_EDMAIORTR,     0xD00A4034,__READ_WRITE ,__satahc_edmaiortr_bits);
__IO_REG32_BIT(SATAHC1_EDMAICDTR,     0xD00A4040,__READ_WRITE ,__satahc_edmaicdtr_bits);
__IO_REG32_BIT(SATAHC1_EDMAHCR,       0xD00A4060,__READ_WRITE ,__satahc_edmaiecr_bits);
__IO_REG32(    SATAHC1_EDMANCQTCQSR,  0xD00A4094,__READ_WRITE );
__IO_REG32_BIT(SATAHC1_ICFGR,         0xD00A4050,__READ_WRITE ,__satahc_icfgr_bits);
__IO_REG32_BIT(SATAHC1_PLLCNFR,       0xD00A4054,__READ_WRITE ,__satahc_pllcnfr_bits);
__IO_REG32_BIT(SATAHC1_SSR,           0xD00A4300,__READ       ,__satahc_ssr_bits);
__IO_REG32_BIT(SATAHC1_SER,           0xD00A4304,__READ_WRITE ,__satahc_ser_bits);
__IO_REG32_BIT(SATAHC1_SCR,           0xD00A4308,__READ_WRITE ,__satahc_scr_bits);
__IO_REG32_BIT(SATAHC1_LTMR,          0xD00A430C,__READ_WRITE ,__satahc_ltmr_bits);
__IO_REG32_BIT(SATAHC1_PHYM3R,        0xD00A4310,__READ_WRITE ,__satahc_phym3r_bits);
__IO_REG32_BIT(SATAHC1_PHYM4R,        0xD00A4314,__READ_WRITE ,__satahc_phym4r_bits);
__IO_REG32_BIT(SATAHC1_PHYM1R,        0xD00A432C,__READ_WRITE ,__satahc_phym1r_bits);
__IO_REG32_BIT(SATAHC1_PHYM2R,        0xD00A4330,__READ_WRITE ,__satahc_phym2r_bits);
__IO_REG32_BIT(SATAHC1_BISTCR,        0xD00A4334,__READ_WRITE ,__satahc_bistcr_bits);
__IO_REG32(    SATAHC1_BISTDW1R,      0xD00A4338,__READ_WRITE );
__IO_REG32(    SATAHC1_BISTDW2R,      0xD00A433C,__READ_WRITE );
__IO_REG32(    SATAHC1_SEIMR,         0xD00A4340,__READ_WRITE );
__IO_REG32_BIT(SATAHC1_ICR,           0xD00A4344,__READ_WRITE ,__satahc_icr_bits);
__IO_REG32_BIT(SATAHC1_ITCR,          0xD00A4348,__READ_WRITE ,__satahc_itcr_bits);
__IO_REG32_BIT(SATAHC1_ISR,           0xD00A434C,__READ       ,__satahc_isr_bits);
__IO_REG32(    SATAHC1_VUR,           0xD00A435C,__READ_WRITE );
__IO_REG32_BIT(SATAHC1_FISCR,         0xD00A4360,__READ_WRITE ,__satahc_fiscr_bits);
__IO_REG32_BIT(SATAHC1_FISICR,        0xD00A4364,__READ       ,__satahc_fisicr_bits);
__IO_REG32_BIT(SATAHC1_FISIMR,        0xD00A4368,__READ_WRITE ,__satahc_fisicr_bits);
__IO_REG32(    SATAHC1_FISWD0,        0xD00A4370,__READ       );
__IO_REG32(    SATAHC1_FISWD1,        0xD00A4374,__READ       );
__IO_REG32(    SATAHC1_FISWD2,        0xD00A4378,__READ       );
__IO_REG32(    SATAHC1_FISWD3,        0xD00A437C,__READ       );
__IO_REG32(    SATAHC1_FISWD4,        0xD00A4380,__READ       );
__IO_REG32(    SATAHC1_FISWD5,        0xD00A4384,__READ       );
__IO_REG32(    SATAHC1_FISWD6,        0xD00A4388,__READ       );
//__IO_REG32_BIT(SATAHC1_PHYM8GEN2R,    0xD00A4390,__READ_WRITE ,__satahc_phym8gen2r_bits);
//__IO_REG32_BIT(SATAHC1_PHYM8GEN1R,    0xD00A4394,__READ_WRITE ,__satahc_phym8gen1r_bits);
__IO_REG32_BIT(SATAHC1_PHYM9GEN2R,    0xD00A4398,__READ_WRITE ,__satahc_phym9genr_bits);
__IO_REG32_BIT(SATAHC1_PHYM9GEN1R,    0xD00A439C,__READ_WRITE ,__satahc_phym9genr_bits);
__IO_REG32_BIT(SATAHC1_PHYCFGR,       0xD00A43A0,__READ_WRITE ,__satahc_phycfgr_bits);
__IO_REG32_BIT(SATAHC1_PHYTCTLR,      0xD00A43A4,__READ_WRITE ,__satahc_phytctlr_bits);
__IO_REG32_BIT(SATAHC1_PHYMODE10R,    0xD00A43A8,__READ_WRITE ,__satahc_phymode10r_bits);
//__IO_REG32_BIT(SATAHC1_PHYMODE12R,    0xD00A23AC,__READ_WRITE ,__satahc_phymode12r_bits);

/***************************************************************************
 **
 ** SATAHC Arbiter
 **
 ***************************************************************************/
__IO_REG32_BIT(SATAHCA_CNFR,          0xD00A0000,__READ_WRITE ,__satahca_cnfr_bits);
__IO_REG32_BIT(SATAHCA_RQOR,          0xD00A0004,__READ_WRITE ,__satahca_rqor_bits);
__IO_REG32_BIT(SATAHCA_RSQIR,         0xD00A0008,__READ_WRITE ,__satahca_rsqir_bits);
__IO_REG32_BIT(SATAHCA_ICTR,          0xD00A000C,__READ_WRITE ,__satahca_ictr_bits);
__IO_REG32_BIT(SATAHCA_ITTR,          0xD00A0010,__READ_WRITE ,__satahca_ittr_bits);
__IO_REG32_BIT(SATAHCA_ICR,           0xD00A0014,__READ       ,__satahca_icr_bits);
__IO_REG32_BIT(SATAHCA_MICR,          0xD00A0020,__READ       ,__satahca_micr_bits);
__IO_REG32_BIT(SATAHCA_MIMR,          0xD00A0024,__READ_WRITE ,__satahca_micr_bits);
__IO_REG32_BIT(SATAHCA_LEDCR,         0xD00A002C,__READ_WRITE ,__satahca_ledcr_bits);
__IO_REG32_BIT(SATAHCA_W0CR,          0xD00A0030,__READ_WRITE ,__satahca_wcr_bits);
__IO_REG32(    SATAHCA_W0BR,          0xD00A0034,__READ_WRITE );
__IO_REG32_BIT(SATAHCA_W1CR,          0xD00A0040,__READ_WRITE ,__satahca_wcr_bits);
__IO_REG32(    SATAHCA_W1BR,          0xD00A0044,__READ_WRITE );
__IO_REG32_BIT(SATAHCA_W2CR,          0xD00A0050,__READ_WRITE ,__satahca_wcr_bits);
__IO_REG32(    SATAHCA_W2BR,          0xD00A0054,__READ_WRITE );
__IO_REG32_BIT(SATAHCA_W3CR,          0xD00A0060,__READ_WRITE ,__satahca_wcr_bits);
__IO_REG32(    SATAHCA_W3BR,          0xD00A0064,__READ_WRITE );

/***************************************************************************
 **
 ** IDMA
 **
 ***************************************************************************/
__IO_REG32_BIT(IDMA_BAR0,             0xD0060A00,__READ_WRITE ,__idma_bar_bits);
__IO_REG32_BIT(IDMA_BAR1,             0xD0060A08,__READ_WRITE ,__idma_bar_bits);
__IO_REG32_BIT(IDMA_BAR2,             0xD0060A10,__READ_WRITE ,__idma_bar_bits);
__IO_REG32_BIT(IDMA_BAR3,             0xD0060A18,__READ_WRITE ,__idma_bar_bits);
__IO_REG32_BIT(IDMA_BAR4,             0xD0060A20,__READ_WRITE ,__idma_bar_bits);
__IO_REG32_BIT(IDMA_BAR5,             0xD0060A28,__READ_WRITE ,__idma_bar_bits);
__IO_REG32_BIT(IDMA_BAR6,             0xD0060A30,__READ_WRITE ,__idma_bar_bits);
__IO_REG32_BIT(IDMA_BAR7,             0xD0060A38,__READ_WRITE ,__idma_bar_bits);
__IO_REG32_BIT(IDMA_SR0,              0xD0060A04,__READ_WRITE ,__idma_sr_bits);
__IO_REG32_BIT(IDMA_SR1,              0xD0060A0C,__READ_WRITE ,__idma_sr_bits);
__IO_REG32_BIT(IDMA_SR2,              0xD0060A14,__READ_WRITE ,__idma_sr_bits);
__IO_REG32_BIT(IDMA_SR3,              0xD0060A1C,__READ_WRITE ,__idma_sr_bits);
__IO_REG32_BIT(IDMA_SR4,              0xD0060A24,__READ_WRITE ,__idma_sr_bits);
__IO_REG32_BIT(IDMA_SR5,              0xD0060A2C,__READ_WRITE ,__idma_sr_bits);
__IO_REG32_BIT(IDMA_SR6,              0xD0060A34,__READ_WRITE ,__idma_sr_bits);
__IO_REG32_BIT(IDMA_SR7,              0xD0060A3C,__READ_WRITE ,__idma_sr_bits);
__IO_REG32(    IDMA_HARR0,            0xD0060A60,__READ_WRITE );
__IO_REG32(    IDMA_HARR1,            0xD0060A64,__READ_WRITE );
__IO_REG32(    IDMA_HARR2,            0xD0060A68,__READ_WRITE );
__IO_REG32(    IDMA_HARR3,            0xD0060A6C,__READ_WRITE );
__IO_REG32_BIT(IDMA_CAPR0,            0xD0060A70,__READ_WRITE ,__idma_capr_bits);
__IO_REG32_BIT(IDMA_CAPR1,            0xD0060A74,__READ_WRITE ,__idma_capr_bits);
__IO_REG32_BIT(IDMA_CAPR2,            0xD0060A78,__READ_WRITE ,__idma_capr_bits);
__IO_REG32_BIT(IDMA_CAPR3,            0xD0060A7C,__READ_WRITE ,__idma_capr_bits);
__IO_REG32_BIT(IDMA_BAER,             0xD0060A80,__READ_WRITE ,__idma_baer_bits);
__IO_REG32_BIT(IDMA_CCLR0,            0xD0060840,__READ_WRITE ,__idma_cclr_bits);
__IO_REG32_BIT(IDMA_CCLR1,            0xD0060844,__READ_WRITE ,__idma_cclr_bits);
__IO_REG32_BIT(IDMA_CCLR2,            0xD0060848,__READ_WRITE ,__idma_cclr_bits);
__IO_REG32_BIT(IDMA_CCLR3,            0xD006084C,__READ_WRITE ,__idma_cclr_bits);
__IO_REG32_BIT(IDMA_CCHR0,            0xD0060880,__READ_WRITE ,__idma_cchr_bits);
__IO_REG32_BIT(IDMA_CCHR1,            0xD0060884,__READ_WRITE ,__idma_cchr_bits);
__IO_REG32_BIT(IDMA_CCHR2,            0xD0060888,__READ_WRITE ,__idma_cchr_bits);
__IO_REG32_BIT(IDMA_CCHR3,            0xD006088C,__READ_WRITE ,__idma_cchr_bits);
__IO_REG32_BIT(IDMA_MBUSTR,           0xD00608D0,__READ_WRITE ,__idma_mbustr_bits);
__IO_REG32_BIT(IDMA_CBCR0,            0xD0060800,__READ_WRITE ,__idma_cbcr_bits);
__IO_REG32_BIT(IDMA_CBCR1,            0xD0060804,__READ_WRITE ,__idma_cbcr_bits);
__IO_REG32_BIT(IDMA_CBCR2,            0xD0060808,__READ_WRITE ,__idma_cbcr_bits);
__IO_REG32_BIT(IDMA_CBCR3,            0xD006080C,__READ_WRITE ,__idma_cbcr_bits);
__IO_REG32(    IDMA_CSAR0,            0xD0060810,__READ_WRITE );
__IO_REG32(    IDMA_CSAR1,            0xD0060814,__READ_WRITE );
__IO_REG32(    IDMA_CSAR2,            0xD0060818,__READ_WRITE );
__IO_REG32(    IDMA_CSAR3,            0xD006081C,__READ_WRITE );
__IO_REG32(    IDMA_CDAR0,            0xD0060820,__READ_WRITE );
__IO_REG32(    IDMA_CDAR1,            0xD0060824,__READ_WRITE );
__IO_REG32(    IDMA_CDAR2,            0xD0060828,__READ_WRITE );
__IO_REG32(    IDMA_CDAR3,            0xD006082C,__READ_WRITE );
__IO_REG32(    IDMA_CNDPR0,           0xD0060830,__READ_WRITE );
__IO_REG32(    IDMA_CNDPR1,           0xD0060834,__READ_WRITE );
__IO_REG32(    IDMA_CNDPR2,           0xD0060838,__READ_WRITE );
__IO_REG32(    IDMA_CNDPR3,           0xD006083C,__READ_WRITE );
__IO_REG32(    IDMA_CCDPR0,           0xD0060870,__READ_WRITE );
__IO_REG32(    IDMA_CCDPR1,           0xD0060874,__READ_WRITE );
__IO_REG32(    IDMA_CCDPR2,           0xD0060878,__READ_WRITE );
__IO_REG32(    IDMA_CCDPR3,           0xD006087C,__READ_WRITE );
__IO_REG32_BIT(IDMA_ICR,              0xD00608C0,__READ_WRITE ,__idma_icr_bits);
__IO_REG32_BIT(IDMA_IMR,              0xD00608C4,__READ_WRITE ,__idma_icr_bits);
__IO_REG32(    IDMA_EAR,              0xD00608C8,__READ_WRITE );
__IO_REG32_BIT(IDMA_ESR,              0xD00608CC,__READ_WRITE ,__idma_esr_bits);

/***************************************************************************
 **
 ** XOR
 **
 ***************************************************************************/
__IO_REG32_BIT(XE0WCR,                0xD0060B40,__READ_WRITE ,__xexwcr_bits);
__IO_REG32_BIT(XE1WCR,                0xD0060B44,__READ_WRITE ,__xexwcr_bits);
__IO_REG32_BIT(XEBAR0,                0xD0060B50,__READ_WRITE ,__xebarx_bits);
__IO_REG32_BIT(XEBAR1,                0xD0060B54,__READ_WRITE ,__xebarx_bits);
__IO_REG32_BIT(XEBAR2,                0xD0060B58,__READ_WRITE ,__xebarx_bits);
__IO_REG32_BIT(XEBAR3,                0xD0060B5C,__READ_WRITE ,__xebarx_bits);
__IO_REG32_BIT(XEBAR4,                0xD0060B60,__READ_WRITE ,__xebarx_bits);
__IO_REG32_BIT(XEBAR5,                0xD0060B64,__READ_WRITE ,__xebarx_bits);
__IO_REG32_BIT(XEBAR6,                0xD0060B68,__READ_WRITE ,__xebarx_bits);
__IO_REG32_BIT(XEBAR7,                0xD0060B6C,__READ_WRITE ,__xebarx_bits);
__IO_REG32_BIT(XESMR0,                0xD0060B70,__READ_WRITE ,__xesmrx_bits);
__IO_REG32_BIT(XESMR1,                0xD0060B74,__READ_WRITE ,__xesmrx_bits);
__IO_REG32_BIT(XESMR2,                0xD0060B78,__READ_WRITE ,__xesmrx_bits);
__IO_REG32_BIT(XESMR3,                0xD0060B7C,__READ_WRITE ,__xesmrx_bits);
__IO_REG32_BIT(XESMR4,                0xD0060B80,__READ_WRITE ,__xesmrx_bits);
__IO_REG32_BIT(XESMR5,                0xD0060B84,__READ_WRITE ,__xesmrx_bits);
__IO_REG32_BIT(XESMR6,                0xD0060B88,__READ_WRITE ,__xesmrx_bits);
__IO_REG32_BIT(XESMR7,                0xD0060B8C,__READ_WRITE ,__xesmrx_bits);
__IO_REG32(    XEHARR0,               0xD0060B90,__READ_WRITE );
__IO_REG32(    XEHARR1,               0xD0060B94,__READ_WRITE );
__IO_REG32(    XEHARR2,               0xD0060B98,__READ_WRITE );
__IO_REG32(    XEHARR3,               0xD0060B9C,__READ_WRITE );
__IO_REG32_BIT(XEAOCR0,               0xD0060BA0,__READ_WRITE ,__xeaocrx_bits);
__IO_REG32_BIT(XEAOCR1,               0xD0060BA4,__READ_WRITE ,__xeaocrx_bits);
__IO_REG32_BIT(XECHAR,                0xD0060900,__READ_WRITE ,__xechar_bits);
__IO_REG32_BIT(XE0CR,                 0xD0060910,__READ_WRITE ,__xexcr_bits);
__IO_REG32_BIT(XE1CR,                 0xD0060914,__READ_WRITE ,__xexcr_bits);
__IO_REG32_BIT(XE0ACTR,               0xD0060920,__READ_WRITE ,__xexactr_bits);
__IO_REG32_BIT(XE1ACTR,               0xD0060924,__READ_WRITE ,__xexactr_bits);
__IO_REG32(    XE0NDPR,               0xD0060B00,__READ_WRITE );
__IO_REG32(    XE1NDPR,               0xD0060B04,__READ_WRITE );
__IO_REG32(    XE0CDPR,               0xD0060B10,__READ       );
__IO_REG32(    XE1CDPR,               0xD0060B14,__READ       );
__IO_REG32(    XE0BCR,                0xD0060B20,__READ       );
__IO_REG32(    XE1BCR,                0xD0060B24,__READ       );
__IO_REG32(    XE0DPR0,               0xD0060BB0,__READ_WRITE );
__IO_REG32(    XE1DPR0,               0xD0060BB4,__READ_WRITE );
__IO_REG32(    XE0BSR,                0xD0060BC0,__READ_WRITE );
__IO_REG32(    XE1BSR,                0xD0060BC4,__READ_WRITE );
__IO_REG32_BIT(XETMCR,                0xD0060BD0,__READ_WRITE ,__xetmcr_bits);
__IO_REG32(    XETMIVR,               0xD0060BD4,__READ_WRITE );
__IO_REG32(    XETMCVR,               0xD0060BD8,__READ       );
__IO_REG32(    XEIVRL,                0xD0060BE0,__READ_WRITE );
__IO_REG32(    XEIVRH,                0xD0060BE4,__READ_WRITE );
__IO_REG32_BIT(XEICR1,                0xD0060930,__READ_WRITE ,__xeicr1_bits);
__IO_REG32_BIT(XEIMR,                 0xD0060940,__READ_WRITE ,__xeimr_bits);
__IO_REG32_BIT(XEECR,                 0xD0060950,__READ       ,__xeecr_bits);
__IO_REG32(    XEEAR,                 0xD0060960,__READ       );

/***************************************************************************
 **
 ** UART
 **
 ***************************************************************************/
__IO_REG32_BIT(UART_ECR,              0xD0010700,__READ_WRITE ,__uart_ecr_bits);

/***************************************************************************
 **
 ** UART0
 **
 ***************************************************************************/
__IO_REG8(     UART0_RBR,             0xD0012000,__READ_WRITE );
#define        UART0_THR      UART0_RBR
#define        UART0_DLL      UART0_DLL
__IO_REG8_BIT( UART0_IER,             0xD0012004,__READ_WRITE ,__uart_ier_bits);
#define        UART0_DLH      UART0_IER
__IO_REG8_BIT( UART0_FCR,             0xD0012008,__READ_WRITE ,__uart_fcr_bits);
#define        UART0_IIR      UART0_FCR
#define        UART0_IIR_bit  UART0_FCR_bit
__IO_REG8_BIT( UART0_LCR,             0xD001200C,__READ_WRITE ,__uart_lcr_bits);
__IO_REG8_BIT( UART0_MCR,             0xD0012010,__READ_WRITE ,__uart_mcr_bits);
__IO_REG8_BIT( UART0_LSR,             0xD0012014,__READ       ,__uart_lsr_bits);
__IO_REG8_BIT( UART0_MSR,             0xD0012018,__READ_WRITE ,__uart_msr_bits);
__IO_REG8(     UART0_SCR,             0xD001201C,__READ_WRITE );

/***************************************************************************
 **
 ** UART1
 **
 ***************************************************************************/
__IO_REG8(     UART1_RBR,             0xD0012100,__READ_WRITE );
#define        UART1_THR      UART1_RBR
#define        UART1_DLL      UART1_DLL
__IO_REG8_BIT( UART1_IER,             0xD0012104,__READ_WRITE ,__uart_ier_bits);
#define        UART1_DLH      UART1_IER
__IO_REG8_BIT( UART1_FCR,             0xD0012108,__READ_WRITE ,__uart_fcr_bits);
#define        UART1_IIR      UART1_FCR
#define        UART1_IIR_bit  UART1_FCR_bit
__IO_REG8_BIT( UART1_LCR,             0xD001210C,__READ_WRITE ,__uart_lcr_bits);
__IO_REG8_BIT( UART1_MCR,             0xD0012110,__READ_WRITE ,__uart_mcr_bits);
__IO_REG8_BIT( UART1_LSR,             0xD0012114,__READ       ,__uart_lsr_bits);
__IO_REG8_BIT( UART1_MSR,             0xD0012118,__READ_WRITE ,__uart_msr_bits);
__IO_REG8(     UART1_SCR,             0xD001211C,__READ_WRITE );

/***************************************************************************
 **
 ** UART2
 **
 ***************************************************************************/
__IO_REG8(     UART2_RBR,             0xD0012200,__READ_WRITE );
#define        UART2_THR      UART2_RBR
#define        UART2_DLL      UART2_DLL
__IO_REG8_BIT( UART2_IER,             0xD0012204,__READ_WRITE ,__uart_ier_bits);
#define        UART2_DLH      UART2_IER
__IO_REG8_BIT( UART2_FCR,             0xD0012208,__READ_WRITE ,__uart_fcr_bits);
#define        UART2_IIR      UART2_FCR
#define        UART2_IIR_bit  UART2_FCR_bit
__IO_REG8_BIT( UART2_LCR,             0xD001220C,__READ_WRITE ,__uart_lcr_bits);
__IO_REG8_BIT( UART2_MCR,             0xD0012210,__READ_WRITE ,__uart_mcr_bits);
__IO_REG8_BIT( UART2_LSR,             0xD0012214,__READ       ,__uart_lsr_bits);
__IO_REG8_BIT( UART2_MSR,             0xD0012218,__READ_WRITE ,__uart_msr_bits);
__IO_REG8(     UART2_SCR,             0xD001221C,__READ_WRITE );

/***************************************************************************
 **
 ** UART3
 **
 ***************************************************************************/
__IO_REG8(     UART3_RBR,             0xD0012300,__READ_WRITE );
#define        UART3_THR      UART3_RBR
#define        UART3_DLL      UART3_DLL
__IO_REG8_BIT( UART3_IER,             0xD0012304,__READ_WRITE ,__uart_ier_bits);
#define        UART3_DLH      UART3_IER
__IO_REG8_BIT( UART3_FCR,             0xD0012308,__READ_WRITE ,__uart_fcr_bits);
#define        UART3_IIR      UART3_FCR
#define        UART3_IIR_bit  UART3_FCR_bit
__IO_REG8_BIT( UART3_LCR,             0xD001230C,__READ_WRITE ,__uart_lcr_bits);
__IO_REG8_BIT( UART3_MCR,             0xD0012310,__READ_WRITE ,__uart_mcr_bits);
__IO_REG8_BIT( UART3_LSR,             0xD0012314,__READ       ,__uart_lsr_bits);
__IO_REG8_BIT( UART3_MSR,             0xD0012318,__READ_WRITE ,__uart_msr_bits);
__IO_REG8(     UART3_SCR,             0xD001231C,__READ_WRITE );

/***************************************************************************
 **
 ** TWSI0
 **
 ***************************************************************************/
__IO_REG8_BIT( TWSI0_SAR,             0xD0011000,__READ_WRITE ,__twsi_sar_bits);
__IO_REG8(     TWSI0_DR,              0xD0011004,__READ_WRITE );
__IO_REG8_BIT( TWSI0_CR,              0xD0011008,__READ_WRITE ,__twsi_cr_bits);
__IO_REG8_BIT( TWSI0_BRR,             0xD001100C,__READ_WRITE ,__twsi_brr_bits);
#define        TWSI0_SR         TWSI0_BRR
__IO_REG8(     TWSI0_ESAR,            0xD0011010,__READ_WRITE );
__IO_REG32(    TWSI0_SRR,             0xD001101C,__WRITE      );

/***************************************************************************
 **
 ** TWSI1
 **
 ***************************************************************************/
__IO_REG8_BIT( TWSI1_SAR,             0xD0011100,__READ_WRITE ,__twsi_sar_bits);
__IO_REG8(     TWSI1_DR,              0xD0011104,__READ_WRITE );
__IO_REG8_BIT( TWSI1_CR,              0xD0011108,__READ_WRITE ,__twsi_cr_bits);
__IO_REG8_BIT( TWSI1_BRR,             0xD001110C,__READ_WRITE ,__twsi_brr_bits);
#define        TWSI1_SR         TWSI1_BRR
__IO_REG8(     TWSI1_ESAR,            0xD0011110,__READ_WRITE );
__IO_REG32(    TWSI1_SRR,             0xD001111C,__WRITE      );

/***************************************************************************
 **
 ** SPI
 **
 ***************************************************************************/
__IO_REG32_BIT(SPI_CR,                0xD0010600,__READ_WRITE ,__spi_cr_bits);
__IO_REG32_BIT(SPI_CFGR,              0xD0010604,__READ_WRITE ,__spi_cfgr_bits);
__IO_REG16(    SPI_ODR,               0xD0010608,__READ_WRITE );
__IO_REG16(    SPI_IDR,               0xD001060C,__READ_WRITE );
__IO_REG32_BIT(SPI_ICR,               0xD0010610,__READ_WRITE ,__spi_icr_bits);
__IO_REG32_BIT(SPI_IMR,               0xD0010614,__READ_WRITE ,__spi_icr_bits);

/***************************************************************************
 **
 ** TMR
 **
 ***************************************************************************/
__IO_REG32_BIT(TMR_CR,                0xD0020300,__READ_WRITE ,__tmr_cr_bits);
__IO_REG32(    TMR0_RR,               0xD0020310,__READ_WRITE );
__IO_REG32(    TMR0_R,                0xD0020314,__READ_WRITE );
__IO_REG32(    TMR1_RR,               0xD0020318,__READ_WRITE );
__IO_REG32(    TMR1_R,                0xD002031C,__READ_WRITE );
__IO_REG32(    TMR_WDRR,              0xD0020320,__READ_WRITE );
__IO_REG32(    TMR_WDR,               0xD0020324,__READ_WRITE );
__IO_REG32(    TMR2_RR,               0xD0020330,__READ_WRITE );
__IO_REG32(    TMR2_R,                0xD0020334,__READ_WRITE );
__IO_REG32(    TMR3_RR,               0xD0020338,__READ_WRITE );
__IO_REG32(    TMR3_R,                0xD002033C,__READ_WRITE );

/***************************************************************************
 **
 ** GPIO
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO_DOR,              0xD0010100,__READ_WRITE ,__gpio_dor_bits);
__IO_REG32_BIT(GPIO_CR,               0xD0010104,__READ_WRITE ,__gpio_cr_bits);
__IO_REG32_BIT(GPIO_BER,              0xD0010108,__READ_WRITE ,__gpio_ber_bits);
__IO_REG32_BIT(GPIO_DIPR,             0xD001010C,__READ_WRITE ,__gpio_dipr_bits);
__IO_REG32_BIT(GPIO_DIR,              0xD0010110,__READ       ,__gpio_dir_bits);
__IO_REG32_BIT(GPIO_ESICR,            0xD0010114,__READ_WRITE ,__gpio_esicr_bits);
__IO_REG32_BIT(GPIO_ESIMR,            0xD0010118,__READ_WRITE ,__gpio_esimr_bits);
__IO_REG32_BIT(GPIO_LSIMR,            0xD001011C,__READ_WRITE ,__gpio_lsimr_bits);
__IO_REG32_BIT(GPIO_DOSR,             0xD0010120,__WRITE      ,__gpio_dosr_bits);
__IO_REG32_BIT(GPIO_DOCR,             0xD0010124,__WRITE      ,__gpio_docr_bits);
__IO_REG32_BIT(GPIO_CSR,              0xD0010128,__WRITE      ,__gpio_csr_bits);
__IO_REG32_BIT(GPIO_CCR,              0xD001012C,__WRITE      ,__gpio_ccr_bits);

/***************************************************************************
 **
 ** IC
 **
 ***************************************************************************/
__IO_REG32_BIT(IC_MIECR,              0xD0020200,__READ       ,__ic_miecr_bits);
__IO_REG32_BIT(IC_MICLR,              0xD0020204,__READ       ,__ic_miclr_bits);
__IO_REG32_BIT(IC_MICHR,              0xD0020208,__READ       ,__ic_michr_bits);
__IO_REG32_BIT(IC_IRQIMER,            0xD002020C,__READ_WRITE ,__ic_miecr_bits);
__IO_REG32_BIT(IC_IRQIMLR,            0xD0020210,__READ_WRITE ,__ic_miclr_bits);
__IO_REG32_BIT(IC_IRQIMHR,            0xD0020214,__READ_WRITE ,__ic_michr_bits);
__IO_REG32_BIT(IC_IRQSCR,             0xD0020218,__READ       ,__ic_irqscr_bits);
__IO_REG32_BIT(IC_FIQIMER,            0xD002021C,__READ_WRITE ,__ic_miecr_bits);
__IO_REG32_BIT(IC_FIQIMLR,            0xD0020220,__READ_WRITE ,__ic_miclr_bits);
__IO_REG32_BIT(IC_FIQIMHR,            0xD0020224,__READ_WRITE ,__ic_michr_bits);
__IO_REG32_BIT(IC_FIQSCR,             0xD0020228,__READ       ,__ic_miecr_bits);
__IO_REG32_BIT(IC_EIMER,              0xD002022C,__READ_WRITE ,__ic_miecr_bits);
__IO_REG32_BIT(IC_EIMLR,              0xD0020230,__READ_WRITE ,__ic_miclr_bits);
__IO_REG32_BIT(IC_EIMHR,              0xD0020234,__READ_WRITE ,__ic_michr_bits);
__IO_REG32_BIT(IC_ESCR,               0xD0020238,__READ       ,__ic_miecr_bits);

/***************************************************************************
 **
 ** MPP
 **
 ***************************************************************************/
__IO_REG32_BIT(MPP_C0R,               0xD0010000,__READ_WRITE ,__mpp_c0r_bits);
__IO_REG32_BIT(MPP_C1R,               0xD0010004,__READ_WRITE ,__mpp_c1r_bits);
__IO_REG32_BIT(MPP_C2R,               0xD0010008,__READ_WRITE ,__mpp_c2r_bits);
__IO_REG32_BIT(MPP_C3R,               0xD001000C,__READ_WRITE ,__mpp_c3r_bits);
__IO_REG32_BIT(MPP_C4R,               0xD0010010,__READ_WRITE ,__mpp_c4r_bits);
__IO_REG32_BIT(MPP_C5R,               0xD0010014,__READ_WRITE ,__mpp_c5r_bits);
__IO_REG32_BIT(MPP_C6R,               0xD0010018,__READ_WRITE ,__mpp_c6r_bits);

/***************************************************************************
 **
 ** Misc
 **
 ***************************************************************************/
__IO_REG32(    GU0R,                  0xD00100E0,__READ_WRITE );
__IO_REG32(    GU1R,                  0xD00100E4,__READ_WRITE );
__IO_REG32(    GU2R,                  0xD00100E8,__READ_WRITE );
__IO_REG32(    GU3R,                  0xD00100EC,__READ_WRITE );
__IO_REG32_BIT(SPCR0,                 0xD0013000,__READ_WRITE ,__spcr_bits);
__IO_REG32_BIT(SPCR1,                 0xD0013004,__READ_WRITE ,__spcr_bits);
__IO_REG32_BIT(SPCR2,                 0xD0013008,__READ_WRITE ,__spcr_bits);
__IO_REG32_BIT(SPCR3,                 0xD001300C,__READ_WRITE ,__spcr_bits);
__IO_REG32_BIT(SPCR4,                 0xD0013010,__READ_WRITE ,__spcr_bits);
__IO_REG32_BIT(SPCR5,                 0xD0013014,__READ_WRITE ,__spcr_bits);
__IO_REG32_BIT(SPCR6,                 0xD0013018,__READ_WRITE ,__spcr_bits);

/***************************************************************************
 **
 ** Reset
 **
 ***************************************************************************/
__IO_REG32(    SRLR,                  0xD0010030,__READ_WRITE );
__IO_REG32_BIT(SRHR,                  0xD0010034,__READ_WRITE ,__srhr_bits);
__IO_REG32_BIT(RSTOUTMR,              0xD0020108,__READ_WRITE ,__rstoutmr_bits);
__IO_REG32_BIT(SSRR,                  0xD002010C,__READ_WRITE ,__srlr_bits);

/***************************************************************************
 **  Assembler-specific declarations
 ***************************************************************************/
#ifdef __IAR_SYSTEMS_ASM__
#endif    /* __IAR_SYSTEMS_ASM__ */

#endif    /* __IOMV78200_CR0_H */
