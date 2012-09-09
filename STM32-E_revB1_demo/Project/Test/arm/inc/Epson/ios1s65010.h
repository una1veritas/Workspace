/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Epson S1S65010
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler.
 **
 **    (c) Copyright IAR Systems 2007
 **
 **    $Revision: 30246 $
 **
 **    Note: Only little endian addressing of 8 bit registers.
 ***************************************************************************/

#ifndef __IOS1S65010_H
#define __IOS1S65010_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    S1S65010 SPECIAL FUNCTION REGISTERS
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

/* DMA Channel x Transfer Count Register (TCRx) */
typedef struct{
__REG32 TC          :24;
__REG32             : 8;
} __dmac1_tcr_bits;

/* DMA Channel x Control Register (CTLx) */
typedef struct{
__REG32 DE          : 1;
__REG32 TE          : 1;
__REG32 IE          : 1;
__REG32 TS          : 2;
__REG32 TM          : 1;
__REG32 RIM         : 1;
__REG32             : 1;
__REG32 RS          : 4;
__REG32 SAM         : 2;
__REG32 DAM         : 2;
__REG32 AL          : 1;
__REG32 AM          : 1;
__REG32             : 1;
__REG32 IDLE        : 1;
__REG32             :12;
} __dmac1_ctl_bits;

/* DMA Channel Operating Select Register (OPSR) */
typedef struct{
__REG32 DGE         : 1;
__REG32             : 7;
__REG32 DPM         : 2;
__REG32             :22;
} __dmac1_opsr_bits;

/* Camera Clock Frequency Setting Register */
typedef struct{
__REG16 CLK         : 5;
__REG16             :11;
} __cam_ccfsr_bits;

/* Camera Signal Setting Register */
typedef struct{
__REG16 VICE        : 1;
__REG16 VREFAS      : 1;
__REG16 HREFAS      : 1;
__REG16 YUVDFS      : 2;
__REG16 CMS         : 1;
__REG16             :10;
} __cam_cssr_bits;

/* Camera Mode Setting Register */
typedef struct{
__REG16 CME           : 1;
__REG16               : 2;
__REG16 CLKDIS        : 1;
__REG16               : 3;
__REG16 ITU_BT656_EN  : 1;
__REG16               : 2;
__REG16 FSM           : 1;
__REG16               : 1;
__REG16 PULL_DONW_DIS : 1;
__REG16               : 3;
} __cam_cmsr_bits;

/* Camera Frame Control Register */
typedef struct{
__REG16 INTE          : 1;
__REG16 INTP          : 1;
__REG16 FSC           : 3;
__REG16 SSD           : 1;
__REG16 SFCE          : 1;
__REG16 FCIC          : 1;
__REG16 JPEGRDCM      : 1;
__REG16               : 7;
} __cam_cfcr_bits;

/* Camera Control Register */
typedef struct{
__REG16 SR                  : 1;
__REG16 INTFCLR             : 1;
__REG16 FCST                : 1;
__REG16 FCSP                : 1;
__REG16                     : 4;
__REG16 ITU_RBT656_ERR0_CLR : 1;
__REG16 ITU_RBT656_ERR1_CLR : 1;
__REG16                     : 6;
} __cam_ccr_bits;

/* Camera Status Register */
typedef struct{
__REG16                 : 1;
__REG16 FCIS            : 1;
__REG16 FCSF            : 1;
__REG16 FCBS            : 1;
__REG16 EFS             : 1;
__REG16                 : 1;
__REG16 CVSYNC          : 1;
__REG16                 : 1;
__REG16 ITU_RBT656_ERR0 : 1;
__REG16 ITU_RBT656_ERR1 : 1;
__REG16                 : 6;
} __cam_csr_bits;

/* Global Resizer Control Register */
typedef struct{
__REG16                 : 8;
__REG16 ACTAGAIN        : 1;
__REG16                 : 7;
} __jrsz_grcr_bits;

/* Capture Control State Register */
typedef struct{
__REG16 STATE           : 4;
__REG16                 :12;
} __jrsz_ccsr_bits;

/* Capture Data Setting Register */
typedef struct{
__REG16 DFS             : 1;
__REG16                 :15;
} __jrsz_cdsr_bits;

/* Capture Resizer Control Register */
typedef struct{
__REG16 EN              : 1;
__REG16                 : 6;
__REG16 SR              : 1;
__REG16                 : 8;
} __jrsz_crcr_bits;

/* Capture Resizer Start X Position Register */
typedef struct{
__REG16 X               :11;
__REG16                 : 5;
} __jrsz_crsxpr_bits;

/* Capture Resizer Start Y Position Register */
typedef struct{
__REG16 Y               :11;
__REG16                 : 5;
} __jrsz_crsypr_bits;

/* Capture Resizer End X Position Register */
typedef struct{
__REG16 X               :11;
__REG16                 : 5;
} __jrsz_crexpr_bits;

/* Capture Resizer End Y Position Register */
typedef struct{
__REG16 Y               :11;
__REG16                 : 5;
} __jrsz_creypr_bits;

/* Capture Resizer Scaling Rate Register */
typedef struct{
__REG16 SCALLING_RATE   : 4;
__REG16                 :12;
} __jrsz_crsrr_bits;

/* Capture Resizer Scaling Mode Register */
typedef struct{
__REG16 SCALLING_MODE   : 2;
__REG16                 :14;
} __jrsz_crsmr_bits;

/* JPEG Control Register */
typedef struct{
__REG16 MOD_EN          : 1;
__REG16 MODE            : 3;
__REG16 UVDT_DIS        : 1;
__REG16                 : 2;
__REG16 SR              : 1;
__REG16 ROT180_EN       : 1;
__REG16                 : 5;
__REG16 JMFOM           : 1;
__REG16 JEFM            : 1;
} __jctl_cr_bits;

/* JPEG Status Flag Register 
   JPEG Raw Status Flag Register */
typedef struct{
__REG16 JLBIF             : 1;
__REG16 JCIF              : 1;
__REG16 JLBOIF            : 1;
__REG16                   : 1;
__REG16 JDMDIF            : 1;
__REG16                   : 3;
__REG16 JFIFO_EMPTY_IF    : 1;
__REG16 JFIFO_FULL_IF     : 1;
__REG16 JFIFO_THR_TRG_IF  : 1;
__REG16 ESLVIF            : 1;
__REG16 JFIFO_THR_TRG_STA : 2;
__REG16 JCFOS             : 1;
__REG16                   : 1;
} __jctl_sr_bits;

/* JPEG Interrupt Control Register */
typedef struct{
__REG16 JLBIE             : 1;
__REG16 JCIE              : 1;
__REG16 JLBOIE            : 1;
__REG16                   : 1;
__REG16 JDMDIE            : 1;
__REG16                   : 3;
__REG16 JFIFO_EMPTY_IE    : 1;
__REG16 JFIFO_FULL_IE     : 1;
__REG16 JFIFO_THR_TRG_IE  : 1;
__REG16 ESLVIE            : 1;
__REG16                   : 4;
} __jctl_icr_bits;

/* JPEG Codec Start/Stop Control Register */
typedef struct{
__REG16 START             : 1;
__REG16                   :15;
} __jctl_csscr_bits;

/* JPEG Huffman Table Automatic Setting Register */
typedef struct{
__REG16 HTAS              : 1;
__REG16 HTASNSM           : 1;
__REG16                   :14;
} __jctl_htasr_bits;

/* JPEG FIFO Control Register */
typedef struct{
__REG16                   : 1;
__REG16 FIFO_DIR          : 1;
__REG16 FIFO_CLR          : 1;
__REG16                   : 1;
__REG16 FIFO_TRG_THR      : 2;
__REG16                   :10;
} __jfifo_cr_bits;

/* JPEG FIFO Status Register */
typedef struct{
__REG16 EMPTY_STA         : 1;
__REG16 FULL_STA          : 1;
__REG16 THR_STA           : 2;
__REG16                   :12;
} __jfifo_sr_bits;

/* JPEG FIFO Size Register */
typedef struct{
__REG16 SIZE              :15;
__REG16                   : 1;
} __jfifo_size_bits;

/* JPEG Encode Size Limit Register 1 */
typedef struct{
__REG16 LIMIT_HIGH        : 8;
__REG16                   : 8;
} __jfifo_eslr1_bits;

/* JPEG Encode Size Result Register 1 */
typedef struct{
__REG16 RESULT_HIGH       : 8;
__REG16                   : 8;
} __jfifo_esrr1_bits;

/* JPEG Line Buffer Status Flag Register 
   JPEG Line Buffer Raw Status Flag Register
   JPEG Line Buffer Current Status Flag Register */
typedef struct{
__REG16                   : 1;
__REG16 HALF_IF           : 1;
__REG16 FULL_IF           : 1;
__REG16 EMPTY_IF          : 1;
__REG16                   :12;
} __jlb_sr_bits;

/* JPEG Line Buffer Interrupt Control Register */
typedef struct{
__REG16                   : 1;
__REG16 HALF_IE           : 1;
__REG16 FULL_IE           : 1;
__REG16 EMPTY_IE          : 1;
__REG16                   :12;
} __jlb_icr_bits;

/* JPEG Line Buffer Horizontal Pixel Support Size Register */
typedef struct{
__REG16 H_SIZE_SET        : 3;
__REG16                   : 1;
__REG16 H_SIZE            :12;
} __jlb_hpssr_bits;

/* JPEG Line Buffer Memory Address Offset Register */
typedef struct{
__REG16 OFFSET            : 7;
__REG16                   : 9;
} __jlb_maor_bits;

/* JPEG Operation Mode Settings Register */
typedef struct{
__REG16 FORM_SEL          : 2;
__REG16 OP_MODE           : 1;
__REG16 INST_MRK          : 1;
__REG16                   :12;
} __jcodec_omsr_bits;

/* JPEG Command Setting Register */
typedef struct{
__REG16 START             : 1;
__REG16                   : 6;
__REG16 SR                : 1;
__REG16                   : 8;
} __jcodec_csr_bits;

/* JPEG Operation Status Register */
typedef struct{
__REG16 OP_START          : 1;
__REG16                   :15;
} __jcodec_osr_bits;

/* JPEG Quantization Table Number Register */
typedef struct{
__REG16 Y                 : 1;
__REG16 U                 : 1;
__REG16 V                 : 1;
__REG16                   :13;
} __jcodec_qtnr_bits;

/* JPEG Huffman Table Number Register */
typedef struct{
__REG16 Y_DC              : 1;
__REG16 Y_AC              : 1;
__REG16 U_DC              : 1;
__REG16 U_AC              : 1;
__REG16 V_DC              : 1;
__REG16 V_AC              : 1;
__REG16                   :10;
} __jcodec_htnr_bits;

/* JPEG DRI Setting Register 0 */
typedef struct{
__REG16 DRI_VAL_HIGH      : 8;
__REG16                   : 8;
} __jcodec_drisr0_bits;

/* JPEG DRI Setting Register 1 */
typedef struct{
__REG16 DRI_VAL_LOW       : 8;
__REG16                   : 8;
} __jcodec_drisr1_bits;

/* JPEG Vertical Pixel Size Register 0 */
typedef struct{
__REG16 Y_VAL_HIGH        : 8;
__REG16                   : 8;
} __jcodec_vpsr0_bits;

/* JPEG Vertical Pixel Size Register 1 */
typedef struct{
__REG16 Y_VAL_LOW         : 8;
__REG16                   : 8;
} __jcodec_vpsr1_bits;

/* JPEG Horizontal Pixel Size Register 0 */
typedef struct{
__REG16 X_VAL_HIGH        : 8;
__REG16                   : 8;
} __jcodec_hpsr0_bits;

/* JPEG Horizontal Pixel Size Register 1 */
typedef struct{
__REG16 X_VAL_LOW         : 8;
__REG16                   : 8;
} __jcodec_hpsr1_bits;

/* JPEG RST Marker Operation Setting Register */
typedef struct{
__REG16 RST_MRK_OP        : 2;
__REG16                   :14;
} __jcodec_rstmosr_bits;

/* JPEG RST Marker Operation Status Register */
typedef struct{
__REG16                   : 3;
__REG16 ERROR             : 4;
__REG16                   : 9;
} __jcodec_rstmostr_bits;

/* JPEG Insert Marker Data Registers */
typedef struct{
__REG16 MARKER            : 8;
__REG16                   : 8;
} __jcodec_imdr_bits;

/* JPEG Quantization Table No. 0 Register */
typedef struct{
__REG16 DATA              : 8;
__REG16                   : 8;
} __jcodec_qt0r_bits;

/* JPEG Quantization Table No. 1 Register */
typedef struct{
__REG16 DATA              : 8;
__REG16                   : 8;
} __jcodec_qt1r_bits;

/* JPEG DC Huffman Table No. 0 Register 0 */
typedef struct{
__REG16 DATA              : 8;
__REG16                   : 8;
} __jcodec_dcht0r0_bits;

/* JPEG DC Huffman Table No. 0 Register 1 */
typedef struct{
__REG16 DATA              : 4;
__REG16                   :12;
} __jcodec_dcht0r1_bits;

/* JPEG AC Huffman Table No. 0 Register 0 */
typedef struct{
__REG16 DATA              : 8;
__REG16                   : 8;
} __jcodec_acht0r0_bits;

/* JPEG AC Huffman Table No. 0 Register 1 */
typedef struct{
__REG16 DATA              : 8;
__REG16                   : 8;
} __jcodec_acht0r1_bits;

/* JPEG DC Huffman Table No. 1 Register 0 */
typedef struct{
__REG16 DATA              : 8;
__REG16                   : 8;
} __jcodec_dcht1r0_bits;

/* JPEG DC Huffman Table No. 1 Register 1 */
typedef struct{
__REG16 DATA              : 4;
__REG16                   :12;
} __jcodec_dcht1r1_bits;

/* JPEG AC Huffman Table No. 1 Register 0 */
typedef struct{
__REG16 DATA              : 8;
__REG16                   : 8;
} __jcodec_acht1r0_bits;

/* JPEG AC Huffman Table No. 1 Register 1 */
typedef struct{
__REG16 DATA              : 8;
__REG16                   : 8;
} __jcodec_acht1r1_bits;

/* JPEG DMA Transfer Count Register (JTCR) */
typedef struct{
__REG32 TC                :24;
__REG32                   : 8;
} __jtcr_bits;

/* JPEG DMA Control Register (JCTL) */
typedef struct{
__REG32 DE                : 1;
__REG32 JTE               : 1;
__REG32 IE                : 1;
__REG32 TS                : 2;
__REG32 TM                : 1;
__REG32                   : 2;
__REG32 RS                : 4;
__REG32 SAM               : 2;
__REG32 DAM               : 2;
__REG32 AL                : 1;
__REG32 AM                : 1;
__REG32                   : 2;
__REG32 JCS               : 1;
__REG32 JIE               : 1;
__REG32 JS                : 1;
__REG32                   : 9;
} __jctl_bits;

/* JPEG DMA Block Count Register (JBCR) */
typedef struct{
__REG32 BC                :24;
__REG32                   : 8;
} __jbcr_bits;

/* JPEG DMA Destination Offset Address Register (JOFR) */
typedef struct{
__REG32 OFFSET            :24;
__REG32                   : 8;
} __jofr_bits;

/* JPEG DMA Block End Count Register (JBER) */
typedef struct{
__REG32 BEC               :24;
__REG32                   : 8;
} __jber_bits;

/* JPEG DMA Expansion Register (JHID) */
typedef struct{
__REG32                   :15;
__REG32 SR                : 1;
__REG32                   :16;
} __jhid_bits;

/* JPEG DMA FIFO Data Select Mode Register (JFSM) */
typedef struct{
__REG32                   : 3;
__REG32 FM                : 1;
__REG32                   :28;
} __jfsm_bits;

/* DMA Channel x Transfer Count Register (TCR) */
typedef struct{
__REG32 TC                :24;
__REG32                   : 8;
} __dmac2_tcr_bits;

/* DMA Channel x Control Register (CTL) */
typedef struct{
__REG32 DE                : 1;
__REG32 TE                : 1;
__REG32 IE                : 1;
__REG32 TS                : 2;
__REG32 TM                : 1;
__REG32 RIM               : 1;
__REG32                   : 1;
__REG32 RS                : 4;
__REG32 SAM               : 2;
__REG32 DAM               : 2;
__REG32 AL                : 1;
__REG32 AM                : 1;
__REG32 IB4               : 1;
__REG32                   :13;
} __dmac2_ctl_bits;

/* DMA Channel Operating Select Register (OPSR) */
typedef struct{
__REG32 DGE               : 1;
__REG32                   : 7;
__REG32 DPM               : 1;
__REG32 DPE               : 1;
__REG32                   :22;
} __dmac2_opsr_bits;

/* DMA Channel MISC Register (MISC) */
typedef struct{
__REG32 DPL               : 2;
__REG32                   :13;
__REG32 SR                : 1;
__REG32                   :16;
} __dmac2_misc_bits;

/* DMA Channel Transfer Complete Control Register (TECL) */
typedef struct{
__REG32                   :12;
__REG32 ENTE              : 1;
__REG32 STTE              : 1;
__REG32                   :18;
} __dmac2_tecl_bits;

/* ETH Interrupt Status Register */
typedef struct{
__REG32                   :10;
__REG32 PFT               : 1;
__REG32 MIIMAC            : 1;
__REG32 LNK_UP            : 1;
__REG32                   : 9;
__REG32 TX_FIFO_UND       : 1;
__REG32 RX_FIFO_OVR       : 1;
__REG32                   : 1;
__REG32 TX_ACCS_ERR       : 1;
__REG32 TX_DSCR_END       : 1;
__REG32 TX_CMPL           : 1;
__REG32                   : 1;
__REG32 RX_ACCS_ERR       : 1;
__REG32 RX_DSCR_ERR       : 1;
__REG32 RX_CMPL           : 1;
} __eth_isr_bits;

/* ETH Interrupt Enable Register */
typedef struct{
__REG32                   :10;
__REG32 PFTIE             : 1;
__REG32 MIIMACIE          : 1;
__REG32 LNK_UP_IE         : 1;
__REG32                   : 9;
__REG32 TX_FIFO_UND_EI    : 1;
__REG32 RX_FIFO_OVR_EI    : 1;
__REG32                   : 1;
__REG32 TX_ACCS_ERR_EI    : 1;
__REG32 TX_DSCR_END_EI    : 1;
__REG32 TX_CMPL_EI        : 1;
__REG32                   : 1;
__REG32 RX_ACCS_ERR_EI    : 1;
__REG32 RX_DSCR_ERR_EI    : 1;
__REG32 RX_CMPL_EI        : 1;
} __eth_ier_bits;

/* ETH Reset Register */
typedef struct{
__REG32                   :13;
__REG32 PHY_RST           : 1;
__REG32 RX_RST            : 1;
__REG32 TX_RST            : 1;
__REG32                   :15;
__REG32 ALL_RST           : 1;
} __eth_rr_bits;

/* ETH PHY Status Register */
typedef struct{
__REG32 DUPLEX            : 1;
__REG32 SPEED             : 1;
__REG32 LINK              : 1;
__REG32                   :29;
} __eth_physr_bits;

/* ETH DMA Command Register */
typedef struct{
__REG32                       :15;
__REG32 TX_DMA_STA            : 1;
__REG32                       :14;
__REG32 RX_FIFO_OVR_AUTO_REC  : 1;
__REG32 RX_DMA_EN             : 1;
} __eth_dmacr_bits;

/* ETH Mode Register */
typedef struct{
__REG32                       :24;
__REG32 BURST_LENGHT          : 3;
__REG32                       : 2;
__REG32 DUPLEX_MODE           : 1;
__REG32 AUTO_MODE             : 1;
__REG32 BIG_ENDIAN            : 1;
} __eth_mr_bits;

/* ETH TX Mode Register */
typedef struct{
__REG32                       : 8;
__REG32 TX_FIFO_EMPTY_THR     : 3;
__REG32                       : 1;
__REG32 TX_FIFO_FULL_THR      : 2;
__REG32                       : 2;
__REG32 TRAN_START_THR        : 2;
__REG32                       : 1;
__REG32 STR_FRW               : 1;
__REG32                       : 8;
__REG32 LCR                   : 1;
__REG32 NR                    : 1;
__REG32 SPE                   : 1;
__REG32 LPE                   : 1;
} __eth_txmr_bits;

/* ETH RX Mode Register */
typedef struct{
__REG32                       : 8;
__REG32 RX_FIFO_EMPTY_THR     : 2;
__REG32                       : 2;
__REG32 RX_FIFO_FULL_THR      : 2;
__REG32                       : 2;
__REG32 READ_TRG_THR          : 3;
__REG32                       :11;
__REG32 MCFE                  : 1;
__REG32 AFE                   : 1;
} __eth_rxmr_bits;

/* ETH RX Mode Register */
typedef struct{
__REG32 DATA                  :16;
__REG32 REG_ADR               : 5;
__REG32 PHY_ADR               : 5;
__REG32 RW                    : 1;
__REG32                       : 5;
} __eth_miimr_bits;

/* ETH MAC Address Registers 1 to 8: Upper 16 bits */
typedef struct{
__REG32 ADDR_UPPER            :16;
__REG32                       :16;
} __eth_macadru_bits;

/* ETH Flow Control Register */
typedef struct{
__REG32                       :31;
__REG32 FCE                   : 1;
} __eth_fcr_bits;

/* ETH Pause Request Register */
typedef struct{
__REG32                       :31;
__REG32 PFR                   : 1;
} __eth_prr_bits;

/* ETH Buffer Management Enable Register */
typedef struct{
__REG32 BME                   : 1;
__REG32                       :31;
} __eth_bmer_bits;

/* ETH Buffer Free Register */
typedef struct{
__REG32 BF                    : 1;
__REG32                       :31;
} __eth_bfr_bits;

/* ETH Buffer Information Register */
typedef struct{
__REG32 CAPACITY              :10;
__REG32                       : 6;
__REG32 ABILITY               :10;
__REG32                       : 6;
} __eth_bir_bits;

/* ETH Pause Information Register */
typedef struct{
__REG32 PAUSE_TRAN_THR        :10;
__REG32                       : 6;
__REG32 PAUSE_TIME            :16;
} __eth_pir_bits;

/* ETH TX FIFO Status Register */
typedef struct{
__REG32                       :24;
__REG32 FC                    : 3;
__REG32 TX_FIFO_STA           : 3;
__REG32 ALM_EMPTY             : 1;
__REG32 ALM_FULL              : 1;
} __eth_txfifosr_bits;

/* ETH RX FIFO Status Register */
typedef struct{
__REG32                       :16;
__REG32 STR_WORDS             :12;
__REG32                       : 1;
__REG32 READ_TRG              : 1;
__REG32 ALM_EMPTY             : 1;
__REG32 ALM_FULL              : 1;
} __eth_rxfifosr_bits;

/* APB WAIT0 Register (APBWAIT0) */
typedef struct{
__REG32 PW00CNF               : 2;
__REG32 PW01CNF               : 2;
__REG32 PW02CNF               : 2;
__REG32 PW03CNF               : 2;
__REG32 PW04CNF               : 2;
__REG32 PW05CNF               : 2;
__REG32 PW06CNF               : 2;
__REG32 PW07CNF               : 2;
__REG32 PW08CNF               : 2;
__REG32 PW09CNF               : 2;
__REG32 PW0ACNF               : 2;
__REG32 PW0BCNF               : 2;
__REG32 PW0CCNF               : 2;
__REG32 PW0DCNF               : 2;
__REG32 PW0ECNF               : 2;
__REG32 PW0FCNF               : 2;
} __apbwait0_bits;

/* APB WAIT1 Register (APBWAIT1) */
typedef struct{
__REG32 PW10CNF               : 2;
__REG32 PW11CNF               : 2;
__REG32 PW12CNF               : 2;
__REG32 PW13CNF               : 2;
__REG32 PW14CNF               : 2;
__REG32 PW15CNF               : 2;
__REG32 PW16CNF               : 2;
__REG32 PW17CNF               : 2;
__REG32 PW18CNF               : 2;
__REG32 PW19CNF               : 2;
__REG32 PW1ACNF               : 2;
__REG32 PW1BCNF               : 2;
__REG32 PW1CCNF               : 2;
__REG32 PW1DCNF               : 2;
__REG32 PW1ECNF               : 2;
__REG32 PW1FCNF               : 2;
} __apbwait1_bits;

/* SYS Chip ID Register (CHIPID) */
typedef struct{
__REG32 REV                   : 3;
__REG32                       : 5;
__REG32 ID                    :24;
} __chipid_bits;

/* SYS PLL Setting Register 1 (PLLSET1) */
typedef struct{
__REG32 L_CNTR                :10;
__REG32 W_DIV                 : 2;
__REG32 N_CNTR                : 4;
__REG32 VC                    : 4;
__REG32 RS                    : 4;
__REG32 CP                    : 5;
__REG32 CS                    : 2;
__REG32                       : 1;
} __pllset1_bits;

/* SYS PLL Setting Register 2 (PLLSET2) */
typedef struct{
__REG32 PLLEN                 : 1;
__REG32                       :31;
} __pllset2_bits;

/* SYS HALT Mode Clock Control Register (HALTMODE) */
typedef struct{
__REG32 HALT_MDCLK            : 4;
__REG32                       : 2;
__REG32 CPUCKSEL              : 2;
__REG32                       :24;
} __haltmode_bits;

/* SYS I/O Clock Control Register (IOCLKCTL) */
typedef struct{
__REG32 ETH_CLKEN             : 1;
__REG32 CF_CLKEN              : 1;
__REG32 TIMER_CLKEN           : 1;
__REG32 I2C_CLKEN             : 1;
__REG32 SPI_CLKEN             : 1;
__REG32 DMAC2_CLKEN           : 1;
__REG32 UART_CLKEN            : 1;
__REG32                       : 1;
__REG32 I2S_CLKEN             : 1;
__REG32                       :23;
} __ioclkctl_bits;

/* SYS Clock Select Register (CLK32SEL) */
typedef struct{
__REG32 CLKSEL                : 1;
__REG32                       :31;
} __clk32sel_bits;

/* SYS Memory Remap Register (REMAP) */
typedef struct{
__REG32 REMAP1                : 1;
__REG32 REMAP2                : 1;
__REG32                       :30;
} __remap_bits;

/* SYS UART Clock Divider Register (UARTDIV) */
typedef struct{
__REG32 UARTCLKDIV            : 8;
__REG32                       :24;
} __uartdiv_bits;

/* SYS MD Bus Pull-down Control Register (MDPLDCTL) */
typedef struct{
__REG32 MDPLDNDIS0            : 1;
__REG32 MDPLDNDIS1            : 1;
__REG32 MDPLDNDIS2            : 1;
__REG32 MDPLDNDIS3            : 1;
__REG32 MDPLDNDIS4            : 1;
__REG32 MDPLDNDIS5            : 1;
__REG32 MDPLDNDIS6            : 1;
__REG32 MDPLDNDIS7            : 1;
__REG32 MDPLDNDIS8            : 1;
__REG32 MDPLDNDIS9            : 1;
__REG32 MDPLDNDIS10           : 1;
__REG32 MDPLDNDIS11           : 1;
__REG32 MDPLDNDIS12           : 1;
__REG32 MDPLDNDIS13           : 1;
__REG32 MDPLDNDIS14           : 1;
__REG32 MDPLDNDIS15           : 1;
__REG32                       :16;
} __mdpldctl_bits;

/* SYS GPIOC Resistor Control Register (PORTCRCTL) */
typedef struct{
__REG32 PORTCPDDIS0           : 1;
__REG32 PORTCPDDIS1           : 1;
__REG32 PORTCPDDIS2           : 1;
__REG32 PORTCPDDIS3           : 1;
__REG32 PORTCPDDIS4           : 1;
__REG32 PORTCPDDIS5           : 1;
__REG32 PORTCPDDIS6           : 1;
__REG32 PORTCPDDIS7           : 1;
__REG32                       :24;
} __portcrctl_bits;

/* SYS GPIOD Resistor Control Register (PORTDRCTL) */
typedef struct{
__REG32                       : 2;
__REG32 PORTDPDDIS2           : 1;
__REG32 PORTDPDDIS3           : 1;
__REG32 PORTDPDDIS4           : 1;
__REG32 PORTDPDDIS5           : 1;
__REG32 PORTDPDDIS6           : 1;
__REG32 PORTDPDDIS7           : 1;
__REG32                       :24;
} __portdrctl_bits;

/* SYS GPIOE Resistor Control Register (PORTERCTL) */
typedef struct{
__REG32 PORTEPDDIS0           : 1;
__REG32 PORTEPDDIS1           : 1;
__REG32 PORTEPDDIS2           : 1;
__REG32 PORTEPDDIS3           : 1;
__REG32 PORTEPDDIS4           : 1;
__REG32 PORTEPDDIS5           : 1;
__REG32 PORTEPDDIS6           : 1;
__REG32 PORTEPDDIS7           : 1;
__REG32                       :24;
} __porterctl_bits;

/* SYS Embedded Memory Control Register (EMBMEMCTL) */
typedef struct{
__REG32 EMBWAITEN             : 2;
__REG32                       : 2;
__REG32 EMBRAMSEL             : 2;
__REG32                       :26;
} __embmemctl_bits;

/* MEMC Configuration Register for Device x (CFGx) */
typedef struct{
__REG32 MTYPE                 : 4;
__REG32                       : 2;
__REG32 XBW                   : 2;
__REG32                       : 8;
__REG32 STAD                  : 7;
__REG32                       : 1;
__REG32 EDAD                  : 7;
__REG32                       : 1;
} __memc_cfg_bits;

/* MEMC Timing Register for Device [3*:0] (RAMTMG[3:0]) */
typedef struct{
__REG32                       : 4;
__REG32 WAITRD                : 5;
__REG32                       : 1;
__REG32 WAITWR                : 5;
__REG32                       : 5;
__REG32 WAITOE                : 5;
__REG32                       : 1;
__REG32 WAITWE                : 5;
__REG32                       : 1;
} __ramtmg_bits;

/* MEMC Control Register for Device [3*:0] (RAMCNTL[3:0]) */
typedef struct{
__REG32 RBLE                  : 1;
__REG32 MWAITPOL              : 2;
__REG32 WPROTECT              : 1;
__REG32                       :28;
} __ramcntl_bits;

/* MEMC Mode Register for SDRAM (SDMR) */
typedef struct{
__REG32 BL                    : 3;
__REG32 BT                    : 1;
__REG32 CL                    : 3;
__REG32 OP_MODE               : 2;
__REG32 WBM                   : 1;
__REG32                       :22;
} __sdmr_bits;

/* MEMC Configuration Register for SDRAM (SDCNFG) */
typedef struct{
__REG32                       : 2;
__REG32 BNUM                  : 2;
__REG32                       : 2;
__REG32 REF                   : 2;
__REG32 APCG                  : 1;
__REG32 TRCD                  : 2;
__REG32                       : 9;
__REG32 COLW                  : 4;
__REG32                       : 1;
__REG32 CKE_CTRL              : 1;
__REG32 CLK_CTRL              : 1;
__REG32                       : 5;
} __sdcnfg_bits;

/* MEMC Advanced Configuration Register for SDRAM (SDADVCNFG) */
typedef struct{
__REG32                       : 2;
__REG32 SELF                  : 2;
__REG32                       : 2;
__REG32 RESELF                : 1;
__REG32 CLK_FORCE             : 1;
__REG32 AREFWAIT              : 4;
__REG32                       : 4;
__REG32 SREFCNT               : 4;
__REG32                       :12;
} __sdadvcnfg_bits;

/* MEMC Initialization Control Register (SDINIT) */
typedef struct{
__REG32                       : 2;
__REG32 DEVSEL                : 2;
__REG32                       : 1;
__REG32 PCG_ALL               : 1;
__REG32 AREF                  : 1;
__REG32 LMR                   : 1;
__REG32                       : 7;
__REG32 INIT_SD               : 1;
__REG32                       :16;
} __sdinit_bits;

/* MEMC Refresh Timer Register for SDRAM (SDREF) */
typedef struct{
__REG32 REFTIME               :12;
__REG32                       :20;
} __sdref_bits;

/* MEMC Status Register for SDRAM (SDSTAT) */
/*
typedef struct{
__REG32 DEVST20               : 1;
__REG32 DEVST21               : 1;
__REG32 DEVST22               : 1;
__REG32 DEVST23               : 1;
__REG32                       : 4;
__REG32 DEVST30               : 1;
__REG32 DEVST31               : 1;
__REG32 DEVST32               : 1;
__REG32 DEVST33               : 1;
__REG32                       :20;
} __sdstat_bits;
*/
typedef struct{
__REG32 DEVST20               : 1;
__REG32 DEVST21               : 1;
__REG32 DEVST22               : 1;
__REG32 DEVST23               : 1;
__REG32                       :28;
} __sdstat_bits;

/* INTRC IRQ Status Register
   IRQ Raw Status Register
   IRQ Enable Register
   IRQ Enable Clear Register */
typedef struct{
__REG32 WDT                   : 1;
__REG32 SI                    : 1;
__REG32 DBGRX                 : 1;
__REG32 DBGTX                 : 1;
__REG32 TIM_CH0               : 1;
__REG32 TIM_CH1               : 1;
__REG32 TIM_CH2               : 1;
__REG32 ETH                   : 1;
__REG32 JPEG                  : 1;
__REG32 DMA1                  : 1;
__REG32 JDMA                  : 1;
__REG32 CAM                   : 1;
__REG32                       : 1;
__REG32 DMA2                  : 1;
__REG32 GPIOA_B               : 1;
__REG32 SPI                   : 1;
__REG32 I2C                   : 1;
__REG32 UART                  : 1;
__REG32 RTC                   : 1;
__REG32 CF                    : 1;
__REG32 INT0                  : 1;
__REG32 INT1                  : 1;
__REG32 INT2                  : 1;
__REG32 UARTL                 : 1;
__REG32 INT3                  : 1;
__REG32 INT4                  : 1;
__REG32 INT5                  : 1;
__REG32 INT6                  : 1;
__REG32 INT7                  : 1;
__REG32 INT8                  : 1;
__REG32 I2S0                  : 1;
__REG32 I2S1                  : 1;
} __irq_sr_bits;

/* INTRC Software IRQ Register */
typedef struct{
__REG32                       : 1;
__REG32 SI                    : 1;
__REG32                       :30;
} __irq_sir_bits;

/* INTRC IRQ Level Register */
typedef struct{
__REG32 INT_LEV0              : 1;
__REG32 INT_LEV1              : 1;
__REG32 INT_LEV2              : 1;
__REG32 INT_LEV3              : 1;
__REG32 INT_LEV4              : 1;
__REG32 INT_LEV5              : 1;
__REG32 INT_LEV6              : 1;
__REG32 INT_LEV7              : 1;
__REG32 INT_LEV8              : 1;
__REG32 INT_LEV9              : 1;
__REG32 INT_LEV10             : 1;
__REG32 INT_LEV11             : 1;
__REG32 INT_LEV12             : 1;
__REG32 INT_LEV13             : 1;
__REG32 INT_LEV14             : 1;
__REG32 INT_LEV15             : 1;
__REG32 INT_LEV16             : 1;
__REG32 INT_LEV17             : 1;
__REG32 INT_LEV18             : 1;
__REG32 INT_LEV19             : 1;
__REG32 INT_LEV20             : 1;
__REG32 INT_LEV21             : 1;
__REG32 INT_LEV22             : 1;
__REG32 INT_LEV23             : 1;
__REG32 INT_LEV24             : 1;
__REG32 INT_LEV25             : 1;
__REG32 INT_LEV26             : 1;
__REG32 INT_LEV27             : 1;
__REG32 INT_LEV28             : 1;
__REG32 INT_LEV29             : 1;
__REG32 INT_LEV30             : 1;
__REG32 INT_LEV31             : 1;
} __irq_lr_bits;

/* INTRC IRQ Polarity Register */
typedef struct{
__REG32 INT_POL0              : 1;
__REG32 INT_POL1              : 1;
__REG32 INT_POL2              : 1;
__REG32 INT_POL3              : 1;
__REG32 INT_POL4              : 1;
__REG32 INT_POL5              : 1;
__REG32 INT_POL6              : 1;
__REG32 INT_POL7              : 1;
__REG32 INT_POL8              : 1;
__REG32 INT_POL9              : 1;
__REG32 INT_POL10             : 1;
__REG32 INT_POL11             : 1;
__REG32 INT_POL12             : 1;
__REG32 INT_POL13             : 1;
__REG32 INT_POL14             : 1;
__REG32 INT_POL15             : 1;
__REG32 INT_POL16             : 1;
__REG32 INT_POL17             : 1;
__REG32 INT_POL18             : 1;
__REG32 INT_POL19             : 1;
__REG32 INT_POL20             : 1;
__REG32 INT_POL21             : 1;
__REG32 INT_POL22             : 1;
__REG32 INT_POL23             : 1;
__REG32 INT_POL24             : 1;
__REG32 INT_POL25             : 1;
__REG32 INT_POL26             : 1;
__REG32 INT_POL27             : 1;
__REG32 INT_POL28             : 1;
__REG32 INT_POL29             : 1;
__REG32 INT_POL30             : 1;
__REG32 INT_POL31             : 1;
} __irq_pr_bits;

/* INTRC IRQ Trigger Reset Register */
typedef struct{
__REG32 INT_TRG_RST0          : 1;
__REG32 INT_TRG_RST1          : 1;
__REG32 INT_TRG_RST2          : 1;
__REG32 INT_TRG_RST3          : 1;
__REG32 INT_TRG_RST4          : 1;
__REG32 INT_TRG_RST5          : 1;
__REG32 INT_TRG_RST6          : 1;
__REG32 INT_TRG_RST7          : 1;
__REG32 INT_TRG_RST8          : 1;
__REG32 INT_TRG_RST9          : 1;
__REG32 INT_TRG_RST10         : 1;
__REG32 INT_TRG_RST11         : 1;
__REG32 INT_TRG_RST12         : 1;
__REG32 INT_TRG_RST13         : 1;
__REG32 INT_TRG_RST14         : 1;
__REG32 INT_TRG_RST15         : 1;
__REG32 INT_TRG_RST16         : 1;
__REG32 INT_TRG_RST17         : 1;
__REG32 INT_TRG_RST18         : 1;
__REG32 INT_TRG_RST19         : 1;
__REG32 INT_TRG_RST20         : 1;
__REG32 INT_TRG_RST21         : 1;
__REG32 INT_TRG_RST22         : 1;
__REG32 INT_TRG_RST23         : 1;
__REG32 INT_TRG_RST24         : 1;
__REG32 INT_TRG_RST25         : 1;
__REG32 INT_TRG_RST26         : 1;
__REG32 INT_TRG_RST27         : 1;
__REG32 INT_TRG_RST28         : 1;
__REG32 INT_TRG_RST29         : 1;
__REG32 INT_TRG_RST30         : 1;
__REG32 INT_TRG_RST31         : 1;
} __irq_trp_bits;

/* INTRC FIQ Status Register
   FIQ Raw Status Register
   FIQ Enable Register
   FIQ Enable Clear Register */
typedef struct{
__REG32 WDT                   : 1;
__REG32 GPIOB0                : 1;
__REG32                       :30;
} __fiq_sr_bits;

/* INTRC FIQ Level Register */
typedef struct{
__REG32 INT_LEV0              : 1;
__REG32 INT_LEV1              : 1;
__REG32                       :30;
} __fiq_lr_bits;

/* INTRC FIQ Polarity Register */
typedef struct{
__REG32 INT_POL0              : 1;
__REG32 INT_POL1              : 1;
__REG32                       :30;
} __fiq_pr_bits;

/* INTRC FIQ Trigger Reset Register */
typedef struct{
__REG32 INT_TRG_RST0          : 1;
__REG32 INT_TRG_RST1          : 1;
__REG32                       :30;
} __fiq_trr_bits;

/* UART Interrupt Enable Register (IER) */
typedef struct{
__REG8  ERBFI                 : 1;
__REG8  ETBEI                 : 1;
__REG8  ELSI                  : 1;
__REG8  EDSSI                 : 1;
__REG8                        : 3;
__REG8  EPTBEI                : 1;
} __uart_ier_bits;

/* UART Interrupt Identify Register (IIR)
   FIFO Control Register (FCR) */
typedef union{
/* UART_IIR*/
struct {
__REG8  IID                   : 4;
__REG8                        : 2;
__REG8  FFEN                  : 2;
};
/* UART_FCR*/
struct {
__REG8  EFIFO                 : 1;
__REG8  RCVRFR                : 1;
__REG8  XMITFR                : 1;
__REG8  DMAMS                 : 1;
__REG8  XMITT                 : 2;
__REG8  RCVRT                 : 2;
};
} __uart_iir_bits;

/* UART Line Control Register (LCR) */
typedef struct{
__REG8  WLS                   : 2;
__REG8  STB                   : 1;
__REG8  PEN                   : 1;
__REG8  EPS                   : 1;
__REG8                        : 1;
__REG8  SBRK                  : 1;
__REG8  DLAB                  : 1;
} __uart_lcr_bits;

/* UART Modem Control Register (MCR) */
typedef struct{
__REG8  DTR                   : 1;
__REG8  RTS                   : 1;
__REG8  OUT1                  : 1;
__REG8  OUT2                  : 1;
__REG8  LOOP                  : 1;
__REG8  AFCE                  : 1;
__REG8                        : 2;
} __uart_mcr_bits;

/* UART Line Status Register (LSR) */
typedef struct{
__REG8  DR                    : 1;
__REG8  OE                    : 1;
__REG8  PE                    : 1;
__REG8  FE                    : 1;
__REG8  BI                    : 1;
__REG8  THRE                  : 1;
__REG8  TEMT                  : 1;
__REG8  RCVRE                 : 1;
} __uart_lsr_bits;

/* UART Modem Status Register (MSR) */
typedef struct{
__REG8  DCTS                  : 1;
__REG8  DDSR                  : 1;
__REG8  TERI                  : 1;
__REG8  DDCD                  : 1;
__REG8  CTS                   : 1;
__REG8  DSR                   : 1;
__REG8  RI                    : 1;
__REG8  DCD                   : 1;
} __uart_msr_bits;

/* UART Test 0 Register */
typedef struct{
__REG8  TM                    : 1;
__REG8                        : 7;
} __uart_t0_bits;

/* UART Test 1 Register */
typedef struct{
__REG8  CTS_TST               : 1;
__REG8  DSR_TST               : 1;
__REG8  RI_TST                : 1;
__REG8  DCD_TST               : 1;
__REG8                        : 4;
} __uart_t1_bits;

/* UART Test Status 0 Register (TS0) */
typedef struct{
__REG8  CTS_RAW_STA           : 1;
__REG8  DSR_RAW_STA           : 1;
__REG8  RI_RAW_STA            : 1;
__REG8  DCD_RAW_STA           : 1;
__REG8                        : 4;
} __uart_ts0_bits;

/* UART Test Status 1 Register (TS1) */
typedef struct{
__REG8  CTS_STA               : 1;
__REG8  DSR_STA               : 1;
__REG8  RI_STA                : 1;
__REG8  DCD_STA               : 1;
__REG8                        : 4;
} __uart_ts1_bits;

/* UART Test Status 2 Register (TS2) */
typedef struct{
__REG8  DTR_STA               : 1;
__REG8  RTS_STA               : 1;
__REG8  OUT1_STA              : 1;
__REG8  OUT2_STA              : 1;
__REG8  BAUDOUT_STA           : 1;
__REG8                        : 3;
} __uart_ts2_bits;

/* UART Test Status 3 Register (TS3) */
typedef struct{
__REG8  INTR_STA              : 1;
__REG8  RXRDY_STA             : 1;
__REG8  TXRDY_STA             : 1;
__REG8                        : 5;
} __uart_ts3_bits;

/* UARTL Interrupt Enable Register (IER) */
typedef struct{
__REG8  ERBFI                 : 1;
__REG8  ETBEI                 : 1;
__REG8  ELSI                  : 1;
__REG8                        : 5;
} __uartl_ier_bits;

/* UARTL Interrupt Identifier Register (IIR) */
typedef struct{
__REG8  IID                   : 4;
__REG8                        : 4;
} __uartl_iir_bits;

/* UARTL Line Control Register (LCR) */
typedef struct{
__REG8  WLS                   : 2;
__REG8  STB                   : 1;
__REG8  PEN                   : 1;
__REG8                        : 2;
__REG8  SBRK                  : 1;
__REG8  DLAB                  : 1;
} __uartl_lcr_bits;

/* UARTL Line Status Register (LSR) */
typedef struct{
__REG8  DR                    : 1;
__REG8  OE                    : 1;
__REG8                        : 1;
__REG8  FE                    : 1;
__REG8  BI                    : 1;
__REG8  THRE                  : 1;
__REG8  TEMT                  : 1;
__REG8                        : 1;
} __uartl_lsr_bits;

/* UARTL Test Status 2 Register (TS2) */
typedef struct{
__REG8                        : 4;
__REG8  BAUDOUT_STA           : 1;
__REG8                        : 3;
} __uartl_ts2_bits;

/* UARTL Test Status 3 Register (TS3) */
typedef struct{
__REG8  INTR_STA              : 1;
__REG8                        : 7;
} __uartl_ts3_bits;

/* I2C Control Register */
typedef struct{
__REG8  TRNS                  : 3;
__REG8  TACK                  : 1;
__REG8  CLKW                  : 1;
__REG8  SR                    : 1;
__REG8                        : 2;
} __i2c_cr_bits;

/* I2C Bus Status Register */
typedef struct{
__REG8  FINISH                : 1;
__REG8  ERROR                 : 1;
__REG8  BUSY                  : 1;
__REG8  USING                 : 1;
__REG8  SCL                   : 1;
__REG8  SDA                   : 1;
__REG8                        : 1;
__REG8  RUN                   : 1;
} __i2c_bsr_bits;

/* I2C Error Status Register */
typedef struct{
__REG8  START_CON             : 1;
__REG8  STOP_CON              : 1;
__REG8  SDA_MISM_ERR          : 1;
__REG8  SCL_MISM_ERR          : 1;
__REG8  NACK                  : 1;
__REG8                        : 3;
} __i2c_esr_bits;

/* I2C Interrupt Control/Status Register */
typedef struct{
__REG8  CCIE                  : 1;
__REG8  EIE                   : 1;
__REG8  CCIF                  : 1;
__REG8  EIF                   : 1;
__REG8                        : 4;
} __i2c_icsr_bits;

/* I2C-Bus Sample Clock Frequency Divisor Register */
typedef struct{
__REG8  DIV                   : 4;
__REG8                        : 4;
} __i2c_scfdr_bits;

/* I2C SCL Clock Frequency Divisor Register */
typedef struct{
__REG8  DIV                   : 3;
__REG8                        : 5;
} __i2c_cfdr_bits;

/* I2C I/O Control Register */
typedef struct{
__REG8  SCL_SAMP_EN           : 1;
__REG8  SCL_HIGH_DRV_EN       : 1;
__REG8                        : 2;
__REG8  SDA_SAMP_EN           : 1;
__REG8  SDA_HIGH_DRV_EN       : 1;
__REG8                        : 2;
} __i2c_iocr_bits;

/* I2C DMA Mode Register */
typedef struct{
__REG8  DMA_MODE              : 2;
__REG8                        : 6;
} __i2c_dmam_bits;

/* I2C DMA Status Register */
typedef struct{
__REG8  WDREQ_MON             : 1;
__REG8  RDREQ_MON             : 1;
__REG8  RBUF_UPDATE           : 1;
__REG8  TBUF_EMPTY            : 1;
__REG8                        : 4;
} __i2c_dmasr_bits;

/* I2S Control Register */
typedef struct{
__REG16 I2SEN                 : 1;
__REG16 MST                   : 1;
__REG16 TX                    : 1;
__REG16 DMAEN                 : 1;
__REG16 MONO                  : 1;
__REG16 DATAWIDTH             : 2;
__REG16 SFTRST                : 1;
__REG16 CLKSEL                : 1;
__REG16 CLKOUTEN              : 1;
__REG16 FRAMECYC              : 2;
__REG16 CNVM2S                : 1;
__REG16                       : 3;
} __i2s_cr_bits;

/* I2S[1:0] Clock Frequency Divisors Register */
typedef struct{
__REG16 CLKDIV                : 8;
__REG16                       : 8;
} __i2s_cfdr_bits;

/* I2S[1:0] Interrupt Status Registers */
typedef struct{
__REG16 EMPTYFLG              : 1;
__REG16 FULLFLG               : 1;
__REG16 NOTEMPTYFLG           : 1;
__REG16 NOTFULLFLG            : 1;
__REG16 UNDERFLOWFLG          : 1;
__REG16 OVERFLOWFLG           : 1;
__REG16                       :10;
} __i2s_isr_bits;

/* I2S[1:0] Interrupt Raw Status Registers */
typedef struct{
__REG16 RAWEMPTYFLG           : 1;
__REG16 RAWFULLFLG            : 1;
__REG16 RAWNOTEMPTYFLG        : 1;
__REG16 RAWNOTFULLFLG         : 1;
__REG16 RAWUNDERFLOWFLG       : 1;
__REG16 RAWOVERFLOWFLG        : 1;
__REG16                       :10;
} __i2s_risr_bits;

/* I2S[1:0] Interrupt Enable Registers */
typedef struct{
__REG16 EMPTYIRQEN            : 1;
__REG16 FULLIRQEN             : 1;
__REG16 NOTEMPTYIRQEN         : 1;
__REG16 NOTFULLIRQEN          : 1;
__REG16 UNDERFLOWIRQEN        : 1;
__REG16 OVERFLOWIRQEN         : 1;
__REG16                       :10;
} __i2s_ier_bits;

/* I2S[1:0] Current Status Registers */
typedef struct{
__REG16 EMPTYSTS              : 1;
__REG16 FULLSTS               : 1;
__REG16 NOTEMPTYSTS           : 1;
__REG16 NOTFULLSTS            : 1;
__REG16                       : 3;
__REG16 DMASTS                : 1;
__REG16 FIFORPNTR             : 4;
__REG16 FIFOWPNTR             : 4;
} __i2s_cisr_bits;

/* SPI Control Register 1 */
typedef struct{
__REG32 ENA                   : 1;
__REG32 MODE                  : 1;
__REG32 RXDATA_RAW            : 1;
__REG32 CLKS                  : 1;
__REG32 MCBR                  : 3;
__REG32                       : 1;
__REG32 CPOL                  : 1;
__REG32 CPHA                  : 1;
__REG32 BPT                   : 5;
__REG32                       :17;
} __spi_cr1_bits;

/* SPI Control Register 2 */
typedef struct{
__REG32                       : 8;
__REG32 SSC                   : 1;
__REG32 SSP                   : 1;
__REG32 SS                    : 1;
__REG32 SSA                   : 1;
__REG32                       :20;
} __spi_cr2_bits;

/* SPI Wait Register */
typedef struct{
__REG32 WAIT                  :16;
__REG32                       :16;
} __spi_wr_bits;

/* SPI Status Register */
typedef struct{
__REG32                       : 2;
__REG32 RDFF                  : 1;
__REG32 RDOF                  : 1;
__REG32 TDEF                  : 1;
__REG32 MFEF                  : 1;
__REG32 BSYF                  : 1;
__REG32                       :25;
} __spi_sr_bits;

/* SPI Interrupt Control Register */
typedef struct{
__REG32 IRQE                  : 1;
__REG32 MIRQ                  : 1;
__REG32 RFIE                  : 1;
__REG32 ROIE                  : 1;
__REG32 TEIE                  : 1;
__REG32 MFIE                  : 1;
__REG32                       :26;
} __spi_icr_bits;

/* CF Card Interface Control Register (CFCTL) */
typedef struct{
__REG16 PCKMD                 : 2;
__REG16 CFCARDEN              : 1;
__REG16 CFRST                 : 1;
__REG16 PROG_IDLE_EN          : 1;
__REG16 IOIS8_MEM             : 1;
__REG16 IOIS8_IO              : 1;
__REG16                       : 1;
__REG16 PROG_CYC              : 4;
__REG16 PROG_IDLE             : 3;
__REG16 PROG_CYCEN            : 1;
} __cfctl_bits;

/* CF Card Pin Status Register (CFPINSTS) */
typedef struct{
__REG16 CD1                   : 1;
__REG16 CD2                   : 1;
__REG16 VS1                   : 1;
__REG16 VS2                   : 1;
__REG16 BVD1_STSCHG           : 1;
__REG16 BVD2                  : 1;
__REG16 IREQ1                 : 1;
__REG16 WP                    : 1;
__REG16 IREQ2                 : 1;
__REG16                       : 7;
} __cfpinsts_bits;

/* CF Card IRQ Source & Clear Register (CFINTRSTS) 
   CF Card IRQ Status Register (CFINTSTS) */
typedef struct{
__REG16 CD1                   : 1;
__REG16 CD2                   : 1;
__REG16                       : 2;
__REG16 BVD1_STSCHG           : 1;
__REG16                       : 1;
__REG16 IREQ1                 : 1;
__REG16                       : 1;
__REG16 IREQ2                 : 1;
__REG16                       : 7;
} __cfintrsts_bits;

/* CF Card IRQ Enable Register (CFINTMSTS) */
typedef struct{
__REG16 CD1EN                 : 1;
__REG16 CD2EN                 : 1;
__REG16                       : 2;
__REG16 BVD1EN_STSCHGEN       : 1;
__REG16                       : 1;
__REG16 IREQEN1               : 1;
__REG16                       : 1;
__REG16 IREQEN2               : 1;
__REG16                       : 7;
} __cfintmsts_bits;

/* CF Card MISC Register (CFMISC) */
typedef struct{
__REG16 CSRDEN                : 1;
__REG16                       :15;
} __cfmisc_bits;

/* Timer x Control Register (TMxCTRL) */
typedef struct{
__REG16 IE                    : 1;
__REG16 ILR                   : 1;
__REG16 DIV                   : 3;
__REG16 MODE                  : 1;
__REG16                       : 1;
__REG16 ENA                   : 1;
__REG16 PRESC                 : 2;
__REG16                       : 6;
} __tmctrl_bits;

/* Timer x Port Output Control Register (TMxPOUT) */
typedef struct{
__REG16 LEVEL                 : 1;
__REG16 ENA                   : 1;
__REG16 MODE                  : 2;
__REG16                       :12;
} __tmpout_bits;

/* Prescaler x Control Register (PSxCTRL) */
typedef struct{
__REG16 PRESC                 : 8;
__REG16                       : 5;
__REG16 DIV                   : 3;
} __psctrl_bits;

/* Timer IRQ Status Register (TMIRQSTS) */
typedef struct{
__REG16 TIMER0_IRQ            : 1;
__REG16 TIMER1_IRQ            : 1;
__REG16 TIMER2_IRQ            : 1;
__REG16                       :13;
} __tmirqsts_bits;

/* RTC Run/Stop Control Register */
typedef struct{
__REG8  TCRUN                 : 1;
__REG8  TCRST                 : 1;
__REG8                        : 3;
__REG8  BUSYWIDTH             : 2;
__REG8  BUSY                  : 1;
} __rtc_cr_bits;

/* RTC Interrupt Register */
typedef struct{
__REG8  TCAF                  : 1;
__REG8  TCIF                  : 1;
__REG8  TCASE                 : 3;
__REG8  TCISE                 : 3;
} __rtc_ir_bits;

/* RTC Timer Divider Register */
typedef struct{
__REG8  TCD0                  : 1;
__REG8  TCD1                  : 1;
__REG8  TCD2                  : 1;
__REG8  TCD3                  : 1;
__REG8  TCD4                  : 1;
__REG8  TCD5                  : 1;
__REG8  TCD6                  : 1;
__REG8  TCD7                  : 1;
} __rtc_tdr_bits;

/* RTC Seconds Counter Register */
typedef struct{
__REG8  TCSD                  : 6;
__REG8                        : 2;
} __rtc_scr_bits;

/* RTC Minutes Counter Register */
typedef struct{
__REG8  TCMD                  : 6;
__REG8                        : 2;
} __rtc_mcr_bits;

/* RTC Hours Counter Register */
typedef struct{
__REG8  TCHD                  : 5;
__REG8                        : 3;
} __rtc_hcr_bits;

/*RTC Alarm Minutes Compare Register */
typedef struct{
__REG8  TCCM                  : 6;
__REG8                        : 2;
} __rtc_amcr_bits;

/* RTC Alarm Hours Compare Register */
typedef struct{
__REG8  TCCH                  : 5;
__REG8                        : 3;
} __rtc_ahcr_bits;

/* RTC Alarm Days Compare Register */
typedef struct{
__REG16 TCCD                  : 9;
__REG16                       : 7;
} __rtc_adcr_bits;

/* RTC Test Register */
typedef struct{
__REG8  RTST                  : 5;
__REG8                        : 3;
} __rtc_tr_bits;

/* RTC Prescaler Register */
typedef struct{
__REG8  TCP                   : 7;
__REG8                        : 1;
} __rtc_pr_bits;

/* Watchdog Timer Control Register */
typedef struct{
__REG16 DIV                   : 3;
__REG16                       : 1;
__REG16 WDTO                  : 1;
__REG16 WDTEN                 : 1;
__REG16 WDTSTA                : 1;
__REG16                       : 9;
} __wdt_tcr_bits;

/* GPIOA Data Register (GPIOA_DATA) */
typedef struct{
__REG8  GPIOA0                : 1;
__REG8  GPIOA1                : 1;
__REG8  GPIOA2                : 1;
__REG8  GPIOA3                : 1;
__REG8  GPIOA4                : 1;
__REG8  GPIOA5                : 1;
__REG8  GPIOA6                : 1;
__REG8  GPIOA7                : 1;
} __gpioa_data_bits;

/* GPIOA Pin Function Register (GPIOA_FNC) */
typedef struct{
__REG16 GPA0MD                : 2;
__REG16 GPA1MD                : 2;
__REG16 GPA2MD                : 2;
__REG16 GPA3MD                : 2;
__REG16 GPA4MD                : 2;
__REG16 GPA5MD                : 2;
__REG16 GPA6MD                : 2;
__REG16 GPA7MD                : 2;
} __gpioa_fnc_bits;

/* GPIOB Data Register (GPIOB_DATA) */
typedef struct{
__REG8  GPIOB0                : 1;
__REG8  GPIOB1                : 1;
__REG8  GPIOB2                : 1;
__REG8  GPIOB3                : 1;
__REG8  GPIOB4                : 1;
__REG8  GPIOB5                : 1;
__REG8  GPIOB6                : 1;
__REG8  GPIOB7                : 1;
} __gpiob_data_bits;

/* GPIOB Pin Function Register (GPIOB_FNC) */
typedef struct{
__REG16 GPB0MD                : 2;
__REG16 GPB1MD                : 2;
__REG16 GPB2MD                : 2;
__REG16 GPB3MD                : 2;
__REG16 GPB4MD                : 2;
__REG16 GPB5MD                : 2;
__REG16 GPB6MD                : 2;
__REG16 GPB7MD                : 2;
} __gpiob_fnc_bits;

/* GPIOC Data Register (GPIOC_DATA) */
typedef struct{
__REG8  GPIOC0                : 1;
__REG8  GPIOC1                : 1;
__REG8  GPIOC2                : 1;
__REG8  GPIOC3                : 1;
__REG8  GPIOC4                : 1;
__REG8  GPIOC5                : 1;
__REG8  GPIOC6                : 1;
__REG8  GPIOC7                : 1;
} __gpioc_data_bits;

/* GPIOC Pin Function Register (GPIOC_FNC) */
typedef struct{
__REG16 GPC0MD                : 2;
__REG16 GPC1MD                : 2;
__REG16 GPC2MD                : 2;
__REG16 GPC3MD                : 2;
__REG16 GPC4MD                : 2;
__REG16 GPC5MD                : 2;
__REG16 GPC6MD                : 2;
__REG16 GPC7MD                : 2;
} __gpioc_fnc_bits;

/* GPIOD Data Register (GPIOD_DATA) */
typedef struct{
__REG8  GPIOD0                : 1;
__REG8  GPIOD1                : 1;
__REG8  GPIOD2                : 1;
__REG8  GPIOD3                : 1;
__REG8  GPIOD4                : 1;
__REG8  GPIOD5                : 1;
__REG8  GPIOD6                : 1;
__REG8  GPIOD7                : 1;
} __gpiod_data_bits;

/* GPIOD Pin Function Register (GPIOD_FNC) */
typedef struct{
__REG16 GPD0MD                : 2;
__REG16 GPD1MD                : 2;
__REG16 GPD2MD                : 2;
__REG16 GPD3MD                : 2;
__REG16 GPD4MD                : 2;
__REG16 GPD5MD                : 2;
__REG16 GPD6MD                : 2;
__REG16 GPD7MD                : 2;
} __gpiod_fnc_bits;

/* GPIOE Data Register (GPIOE_DATA) */
typedef struct{
__REG8  GPIOE0                : 1;
__REG8  GPIOE1                : 1;
__REG8  GPIOE2                : 1;
__REG8  GPIOE3                : 1;
__REG8  GPIOE4                : 1;
__REG8  GPIOE5                : 1;
__REG8  GPIOE6                : 1;
__REG8  GPIOE7                : 1;
} __gpioe_data_bits;

/* GPIOE Pin Function Register (GPIOE_FNC) */
typedef struct{
__REG16 GPE0MD                : 2;
__REG16 GPE1MD                : 2;
__REG16 GPE2MD                : 2;
__REG16 GPE3MD                : 2;
__REG16 GPE4MD                : 2;
__REG16 GPE5MD                : 2;
__REG16 GPE6MD                : 2;
__REG16 GPE7MD                : 2;
} __gpioe_fnc_bits;

/* GPIOF Data Register (GPIOF_DATA) */
typedef struct{
__REG8  GPIOF0                : 1;
__REG8  GPIOF1                : 1;
__REG8  GPIOF2                : 1;
__REG8  GPIOF3                : 1;
__REG8  GPIOF4                : 1;
__REG8  GPIOF5                : 1;
__REG8  GPIOF6                : 1;
__REG8  GPIOF7                : 1;
} __gpiof_data_bits;

/* GPIOF Pin Function Register (GPIOF_FNC) */
typedef struct{
__REG16 GPF0MD                : 2;
__REG16 GPF1MD                : 2;
__REG16 GPF2MD                : 2;
__REG16 GPF3MD                : 2;
__REG16 GPF4MD                : 2;
__REG16 GPF5MD                : 2;
__REG16 GPF6MD                : 2;
__REG16 GPF7MD                : 2;
} __gpiof_fnc_bits;

/* GPIOG Data Register (GPIOG_DATA) */
typedef struct{
__REG8  GPIOG0                : 1;
__REG8  GPIOG1                : 1;
__REG8  GPIOG2                : 1;
__REG8  GPIOG3                : 1;
__REG8  GPIOG4                : 1;
__REG8  GPIOG5                : 1;
__REG8  GPIOG6                : 1;
__REG8  GPIOG7                : 1;
} __gpiog_data_bits;

/* GPIOG Pin Function Register (GPIOG_FNC) */
typedef struct{
__REG16 GPG0MD                : 2;
__REG16 GPG1MD                : 2;
__REG16 GPG2MD                : 2;
__REG16 GPG3MD                : 2;
__REG16 GPG4MD                : 2;
__REG16 GPG5MD                : 2;
__REG16 GPG6MD                : 2;
__REG16 GPG7MD                : 2;
} __gpiog_fnc_bits;

/* GPIOH Data Register (GPIOH_DATA) */
typedef struct{
__REG8  GPIOH0                : 1;
__REG8                        : 7;
} __gpioh_data_bits;

/* GPIOH Pin Function Register (GPIOH_FNC) */
typedef struct{
__REG16 GPH0MD                : 2;
__REG16                       :14;
} __gpioh_fnc_bits;

/* GPIOA&B IRQ Type Register (GPIOAB_ITYP) */
typedef struct{
__REG16 PORTA_IRQ_TYPE0       : 1;
__REG16 PORTA_IRQ_TYPE1       : 1;
__REG16 PORTA_IRQ_TYPE2       : 1;
__REG16 PORTA_IRQ_TYPE3       : 1;
__REG16 PORTA_IRQ_TYPE4       : 1;
__REG16 PORTA_IRQ_TYPE5       : 1;
__REG16 PORTA_IRQ_TYPE6       : 1;
__REG16 PORTA_IRQ_TYPE7       : 1;
__REG16 PORTB_IRQ_TYPE0       : 1;
__REG16 PORTB_IRQ_TYPE1       : 1;
__REG16 PORTB_IRQ_TYPE2       : 1;
__REG16 PORTB_IRQ_TYPE3       : 1;
__REG16 PORTB_IRQ_TYPE4       : 1;
__REG16 PORTB_IRQ_TYPE5       : 1;
__REG16 PORTB_IRQ_TYPE6       : 1;
__REG16 PORTB_IRQ_TYPE7       : 1;
} __gpioab_ityp_bits;

/* GPIOA&B IRQ Polarity Register (GPIOAB_IPOL) */
typedef struct{
__REG16 PORTA_IRQ_POL0        : 1;
__REG16 PORTA_IRQ_POL1        : 1;
__REG16 PORTA_IRQ_POL2        : 1;
__REG16 PORTA_IRQ_POL3        : 1;
__REG16 PORTA_IRQ_POL4        : 1;
__REG16 PORTA_IRQ_POL5        : 1;
__REG16 PORTA_IRQ_POL6        : 1;
__REG16 PORTA_IRQ_POL7        : 1;
__REG16 PORTB_IRQ_POL0        : 1;
__REG16 PORTB_IRQ_POL1        : 1;
__REG16 PORTB_IRQ_POL2        : 1;
__REG16 PORTB_IRQ_POL3        : 1;
__REG16 PORTB_IRQ_POL4        : 1;
__REG16 PORTB_IRQ_POL5        : 1;
__REG16 PORTB_IRQ_POL6        : 1;
__REG16 PORTB_IRQ_POL7        : 1;
} __gpioab_ipol_bits;

/* GPIOA&B IRQ Enable Register (GPIOAB_IEN) */
typedef struct{
__REG16 PORTA_IRQ_IEN0        : 1;
__REG16 PORTA_IRQ_IEN1        : 1;
__REG16 PORTA_IRQ_IEN2        : 1;
__REG16 PORTA_IRQ_IEN3        : 1;
__REG16 PORTA_IRQ_IEN4        : 1;
__REG16 PORTA_IRQ_IEN5        : 1;
__REG16 PORTA_IRQ_IEN6        : 1;
__REG16 PORTA_IRQ_IEN7        : 1;
__REG16 PORTB_IRQ_IEN0        : 1;
__REG16 PORTB_IRQ_IEN1        : 1;
__REG16 PORTB_IRQ_IEN2        : 1;
__REG16 PORTB_IRQ_IEN3        : 1;
__REG16 PORTB_IRQ_IEN4        : 1;
__REG16 PORTB_IRQ_IEN5        : 1;
__REG16 PORTB_IRQ_IEN6        : 1;
__REG16 PORTB_IRQ_IEN7        : 1;
} __gpioab_ien_bits;

/* GPIOA&B IRQ Status & Clear Register (GPIOAB_ISTS) */
typedef struct{
__REG16 PORTA_IRQ0            : 1;
__REG16 PORTA_IRQ1            : 1;
__REG16 PORTA_IRQ2            : 1;
__REG16 PORTA_IRQ3            : 1;
__REG16 PORTA_IRQ4            : 1;
__REG16 PORTA_IRQ5            : 1;
__REG16 PORTA_IRQ6            : 1;
__REG16 PORTA_IRQ7            : 1;
__REG16 PORTB_IRQ0            : 1;
__REG16 PORTB_IRQ1            : 1;
__REG16 PORTB_IRQ2            : 1;
__REG16 PORTB_IRQ3            : 1;
__REG16 PORTB_IRQ4            : 1;
__REG16 PORTB_IRQ5            : 1;
__REG16 PORTB_IRQ6            : 1;
__REG16 PORTB_IRQ7            : 1;
} __gpioab_ists_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */

/* Declarations common to compiler and assembler **************************/

/***************************************************************************
 **
 ** DMAC1
 **
 ***************************************************************************/
__IO_REG32(    DMAC1_SAR0,            0xFFFE3000,__READ_WRITE );
__IO_REG32(    DMAC1_DAR0,            0xFFFE3004,__READ_WRITE );
__IO_REG32_BIT(DMAC1_TCR0,            0xFFFE3008,__READ_WRITE ,__dmac1_tcr_bits);
__IO_REG32_BIT(DMAC1_CTL0,            0xFFFE300C,__READ_WRITE ,__dmac1_ctl_bits);
__IO_REG32(    DMAC1_SAR1,            0xFFFE3010,__READ_WRITE );
__IO_REG32(    DMAC1_DAR1,            0xFFFE3014,__READ_WRITE );
__IO_REG32_BIT(DMAC1_TCR1,            0xFFFE3018,__READ_WRITE ,__dmac1_tcr_bits);
__IO_REG32_BIT(DMAC1_CTL1,            0xFFFE301C,__READ_WRITE ,__dmac1_ctl_bits);
__IO_REG32(    DMAC1_SAR2,            0xFFFE3020,__READ_WRITE );
__IO_REG32(    DMAC1_DAR2,            0xFFFE3024,__READ_WRITE );
__IO_REG32_BIT(DMAC1_TCR2,            0xFFFE3028,__READ_WRITE ,__dmac1_tcr_bits);
__IO_REG32_BIT(DMAC1_CTL2,            0xFFFE302C,__READ_WRITE ,__dmac1_ctl_bits);
__IO_REG32(    DMAC1_SAR3,            0xFFFE3030,__READ_WRITE );
__IO_REG32(    DMAC1_DAR3,            0xFFFE3034,__READ_WRITE );
__IO_REG32_BIT(DMAC1_TCR3,            0xFFFE3038,__READ_WRITE ,__dmac1_tcr_bits);
__IO_REG32_BIT(DMAC1_CTL3,            0xFFFE303C,__READ_WRITE ,__dmac1_ctl_bits);
__IO_REG32_BIT(DMAC1_OPSR,            0xFFFE3060,__READ_WRITE ,__dmac1_opsr_bits);

/***************************************************************************
 **
 ** CAM
 **
 ***************************************************************************/
__IO_REG16_BIT(CAM_CCFSR,             0xFFFE8000,__READ_WRITE ,__cam_ccfsr_bits);
__IO_REG16_BIT(CAM_CSSR,              0xFFFE8004,__READ_WRITE ,__cam_cssr_bits);
__IO_REG16_BIT(CAM_CMSR,              0xFFFE8020,__READ_WRITE ,__cam_cmsr_bits);
__IO_REG16_BIT(CAM_CFCR,              0xFFFE8024,__READ_WRITE ,__cam_cfcr_bits);
__IO_REG16_BIT(CAM_CCR,               0xFFFE8028,__WRITE      ,__cam_ccr_bits);
__IO_REG16_BIT(CAM_CSR,               0xFFFE802C,__READ       ,__cam_csr_bits);

/***************************************************************************
 **
 ** JPEG
 **
 ***************************************************************************/
__IO_REG16_BIT(JRSZ_GRCR,             0xFFFE9060,__WRITE      ,__jrsz_grcr_bits);
__IO_REG16_BIT(JRSZ_CCSR,             0xFFFE9064,__READ       ,__jrsz_ccsr_bits);
__IO_REG16_BIT(JRSZ_CDSR,             0xFFFE9068,__READ_WRITE ,__jrsz_cdsr_bits);
__IO_REG16_BIT(JRSZ_CRCR,             0xFFFE90C0,__READ_WRITE ,__jrsz_crcr_bits);
__IO_REG16_BIT(JRSZ_CRSXPR,           0xFFFE90C8,__READ_WRITE ,__jrsz_crsxpr_bits);
__IO_REG16_BIT(JRSZ_CRSYPR,           0xFFFE90CC,__READ_WRITE ,__jrsz_crsypr_bits);
__IO_REG16_BIT(JRSZ_CREXPR,           0xFFFE90D0,__READ_WRITE ,__jrsz_crexpr_bits);
__IO_REG16_BIT(JRSZ_CREYPR,           0xFFFE90D4,__READ_WRITE ,__jrsz_creypr_bits);
__IO_REG16_BIT(JRSZ_CRSRR,            0xFFFE90D8,__READ_WRITE ,__jrsz_crsrr_bits);
__IO_REG16_BIT(JRSZ_CRSMR,            0xFFFE90DC,__READ_WRITE ,__jrsz_crsmr_bits);
__IO_REG16_BIT(JCTL_CR,               0xFFFEA000,__READ_WRITE ,__jctl_cr_bits);
__IO_REG16_BIT(JCTL_SR,               0xFFFEA004,__READ_WRITE ,__jctl_sr_bits);
__IO_REG16_BIT(JCTL_RSR,              0xFFFEA008,__READ       ,__jctl_sr_bits);
__IO_REG16_BIT(JCTL_ICR,              0xFFFEA00C,__READ_WRITE ,__jctl_icr_bits);
__IO_REG16_BIT(JCTL_CSSCR,            0xFFFEA014,__WRITE      ,__jctl_csscr_bits);
__IO_REG16_BIT(JCTL_HTASR,            0xFFFEA020,__READ_WRITE ,__jctl_htasr_bits);
__IO_REG16_BIT(JFIFO_CR,              0xFFFEA040,__READ_WRITE ,__jfifo_cr_bits);
__IO_REG16_BIT(JFIFO_SR,              0xFFFEA044,__READ       ,__jfifo_sr_bits);
__IO_REG16_BIT(JFIFO_SIZE,            0xFFFEA048,__READ_WRITE ,__jfifo_size_bits);
__IO_REG32(    JFIFO_RWPR,            0xFFFEA04C,__READ_WRITE );
__IO_REG16(    JFIFO_ESLR0,           0xFFFEA060,__READ_WRITE );
__IO_REG16_BIT(JFIFO_ESLR1,           0xFFFEA064,__READ_WRITE ,__jfifo_eslr1_bits);
__IO_REG16(    JFIFO_ESRR0,           0xFFFEA068,__READ       );
__IO_REG16_BIT(JFIFO_ESRR1,           0xFFFEA06C,__READ       ,__jfifo_esrr1_bits);
__IO_REG16_BIT(JLB_SR,                0xFFFEA080,__READ_WRITE ,__jlb_sr_bits);
__IO_REG16_BIT(JLB_RSR,               0xFFFEA084,__READ       ,__jlb_sr_bits);
__IO_REG16_BIT(JLB_CSR,               0xFFFEA088,__READ       ,__jlb_sr_bits);
__IO_REG16_BIT(JLB_ICR,               0xFFFEA08C,__READ_WRITE ,__jlb_icr_bits);
__IO_REG16_BIT(JLB_HPSSR,             0xFFFEA0A0,__READ_WRITE ,__jlb_hpssr_bits);
__IO_REG16_BIT(JLB_MAOR,              0xFFFEA0A4,__READ_WRITE ,__jlb_maor_bits);
__IO_REG32(    JLB_RWPR,              0xFFFEA0C0,__READ_WRITE );
__IO_REG16_BIT(JCODEC_OMSR,           0xFFFEB000,__READ_WRITE ,__jcodec_omsr_bits);
__IO_REG16_BIT(JCODEC_CSR,            0xFFFEB004,__WRITE      ,__jcodec_csr_bits);
__IO_REG16_BIT(JCODEC_OSR,            0xFFFEB008,__READ       ,__jcodec_osr_bits);
__IO_REG16_BIT(JCODEC_QTNR,           0xFFFEB00C,__READ_WRITE ,__jcodec_qtnr_bits);
__IO_REG16_BIT(JCODEC_HTNR,           0xFFFEB010,__READ_WRITE ,__jcodec_htnr_bits);
__IO_REG16_BIT(JCODEC_DRISR0,         0xFFFEB014,__READ_WRITE ,__jcodec_drisr0_bits);
__IO_REG16_BIT(JCODEC_DRISR1,         0xFFFEB018,__READ_WRITE ,__jcodec_drisr1_bits);
__IO_REG16_BIT(JCODEC_VPSR0,          0xFFFEB01C,__READ_WRITE ,__jcodec_vpsr0_bits);
__IO_REG16_BIT(JCODEC_VPSR1,          0xFFFEB020,__READ_WRITE ,__jcodec_vpsr1_bits);
__IO_REG16_BIT(JCODEC_HPSR0,          0xFFFEB024,__READ_WRITE ,__jcodec_hpsr0_bits);
__IO_REG16_BIT(JCODEC_HPSR1,          0xFFFEB028,__READ_WRITE ,__jcodec_hpsr1_bits);
__IO_REG16_BIT(JCODEC_RSTMOSR,        0xFFFEB038,__READ_WRITE ,__jcodec_rstmosr_bits);
__IO_REG16_BIT(JCODEC_RSTMOSTR,       0xFFFEB03C,__READ       ,__jcodec_rstmostr_bits);
__IO_REG16_BIT(JCODEC_IMDR0,          0xFFFEB040,__READ_WRITE ,__jcodec_imdr_bits);
__IO_REG16_BIT(JCODEC_IMDR1,          0xFFFEB044,__READ_WRITE ,__jcodec_imdr_bits);
__IO_REG16_BIT(JCODEC_IMDR2,          0xFFFEB048,__READ_WRITE ,__jcodec_imdr_bits);
__IO_REG16_BIT(JCODEC_IMDR3,          0xFFFEB04C,__READ_WRITE ,__jcodec_imdr_bits);
__IO_REG16_BIT(JCODEC_IMDR4,          0xFFFEB050,__READ_WRITE ,__jcodec_imdr_bits);
__IO_REG16_BIT(JCODEC_IMDR5,          0xFFFEB054,__READ_WRITE ,__jcodec_imdr_bits);
__IO_REG16_BIT(JCODEC_IMDR6,          0xFFFEB058,__READ_WRITE ,__jcodec_imdr_bits);
__IO_REG16_BIT(JCODEC_IMDR7,          0xFFFEB05C,__READ_WRITE ,__jcodec_imdr_bits);
__IO_REG16_BIT(JCODEC_IMDR8,          0xFFFEB060,__READ_WRITE ,__jcodec_imdr_bits);
__IO_REG16_BIT(JCODEC_IMDR9,          0xFFFEB064,__READ_WRITE ,__jcodec_imdr_bits);
__IO_REG16_BIT(JCODEC_IMDR10,         0xFFFEB068,__READ_WRITE ,__jcodec_imdr_bits);
__IO_REG16_BIT(JCODEC_IMDR11,         0xFFFEB06C,__READ_WRITE ,__jcodec_imdr_bits);
__IO_REG16_BIT(JCODEC_IMDR12,         0xFFFEB070,__READ_WRITE ,__jcodec_imdr_bits);
__IO_REG16_BIT(JCODEC_IMDR13,         0xFFFEB074,__READ_WRITE ,__jcodec_imdr_bits);
__IO_REG16_BIT(JCODEC_IMDR14,         0xFFFEB078,__READ_WRITE ,__jcodec_imdr_bits);
__IO_REG16_BIT(JCODEC_IMDR15,         0xFFFEB07C,__READ_WRITE ,__jcodec_imdr_bits);
__IO_REG16_BIT(JCODEC_IMDR16,         0xFFFEB080,__READ_WRITE ,__jcodec_imdr_bits);
__IO_REG16_BIT(JCODEC_IMDR17,         0xFFFEB084,__READ_WRITE ,__jcodec_imdr_bits);
__IO_REG16_BIT(JCODEC_IMDR18,         0xFFFEB088,__READ_WRITE ,__jcodec_imdr_bits);
__IO_REG16_BIT(JCODEC_IMDR19,         0xFFFEB08C,__READ_WRITE ,__jcodec_imdr_bits);
__IO_REG16_BIT(JCODEC_IMDR20,         0xFFFEB090,__READ_WRITE ,__jcodec_imdr_bits);
__IO_REG16_BIT(JCODEC_IMDR21,         0xFFFEB094,__READ_WRITE ,__jcodec_imdr_bits);
__IO_REG16_BIT(JCODEC_IMDR22,         0xFFFEB098,__READ_WRITE ,__jcodec_imdr_bits);
__IO_REG16_BIT(JCODEC_IMDR23,         0xFFFEB09C,__READ_WRITE ,__jcodec_imdr_bits);
__IO_REG16_BIT(JCODEC_IMDR24,         0xFFFEB0A0,__READ_WRITE ,__jcodec_imdr_bits);
__IO_REG16_BIT(JCODEC_IMDR25,         0xFFFEB0A4,__READ_WRITE ,__jcodec_imdr_bits);
__IO_REG16_BIT(JCODEC_IMDR26,         0xFFFEB0A8,__READ_WRITE ,__jcodec_imdr_bits);
__IO_REG16_BIT(JCODEC_IMDR27,         0xFFFEB0AC,__READ_WRITE ,__jcodec_imdr_bits);
__IO_REG16_BIT(JCODEC_IMDR28,         0xFFFEB0B0,__READ_WRITE ,__jcodec_imdr_bits);
__IO_REG16_BIT(JCODEC_IMDR29,         0xFFFEB0B4,__READ_WRITE ,__jcodec_imdr_bits);
__IO_REG16_BIT(JCODEC_IMDR30,         0xFFFEB0B8,__READ_WRITE ,__jcodec_imdr_bits);
__IO_REG16_BIT(JCODEC_IMDR31,         0xFFFEB0BC,__READ_WRITE ,__jcodec_imdr_bits);
__IO_REG16_BIT(JCODEC_IMDR32,         0xFFFEB0C0,__READ_WRITE ,__jcodec_imdr_bits);
__IO_REG16_BIT(JCODEC_IMDR33,         0xFFFEB0C4,__READ_WRITE ,__jcodec_imdr_bits);
__IO_REG16_BIT(JCODEC_IMDR34,         0xFFFEB0C8,__READ_WRITE ,__jcodec_imdr_bits);
__IO_REG16_BIT(JCODEC_IMDR35,         0xFFFEB0CC,__READ_WRITE ,__jcodec_imdr_bits);
__IO_REG16_BIT(JCODEC_QT0R0,          0xFFFEB400,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R1,          0xFFFEB404,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R2,          0xFFFEB408,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R3,          0xFFFEB40C,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R4,          0xFFFEB410,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R5,          0xFFFEB414,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R6,          0xFFFEB418,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R7,          0xFFFEB41C,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R8,          0xFFFEB420,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R9,          0xFFFEB424,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R10,         0xFFFEB428,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R11,         0xFFFEB42C,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R12,         0xFFFEB430,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R13,         0xFFFEB434,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R14,         0xFFFEB438,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R15,         0xFFFEB43C,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R16,         0xFFFEB440,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R17,         0xFFFEB444,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R18,         0xFFFEB448,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R19,         0xFFFEB44C,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R20,         0xFFFEB450,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R21,         0xFFFEB454,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R22,         0xFFFEB458,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R23,         0xFFFEB45C,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R24,         0xFFFEB460,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R25,         0xFFFEB464,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R26,         0xFFFEB468,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R27,         0xFFFEB46C,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R28,         0xFFFEB470,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R29,         0xFFFEB474,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R30,         0xFFFEB478,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R31,         0xFFFEB47C,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R32,         0xFFFEB480,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R33,         0xFFFEB484,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R34,         0xFFFEB488,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R35,         0xFFFEB48C,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R36,         0xFFFEB490,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R37,         0xFFFEB494,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R38,         0xFFFEB498,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R39,         0xFFFEB49C,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R40,         0xFFFEB4A0,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R41,         0xFFFEB4A4,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R42,         0xFFFEB4A8,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R43,         0xFFFEB4AC,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R44,         0xFFFEB4B0,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R45,         0xFFFEB4B4,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R46,         0xFFFEB4B8,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R47,         0xFFFEB4BC,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R48,         0xFFFEB4C0,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R49,         0xFFFEB4C4,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R50,         0xFFFEB4C8,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R51,         0xFFFEB4CC,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R52,         0xFFFEB4D0,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R53,         0xFFFEB4D4,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R54,         0xFFFEB4D8,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R55,         0xFFFEB4DC,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R56,         0xFFFEB4E0,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R57,         0xFFFEB4E4,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R58,         0xFFFEB4E8,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R59,         0xFFFEB4EC,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R60,         0xFFFEB4F0,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R61,         0xFFFEB4F4,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R62,         0xFFFEB4F8,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT0R63,         0xFFFEB4FC,__READ_WRITE ,__jcodec_qt0r_bits);
__IO_REG16_BIT(JCODEC_QT1R0,          0xFFFEB500,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R1,          0xFFFEB504,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R2,          0xFFFEB508,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R3,          0xFFFEB50C,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R4,          0xFFFEB510,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R5,          0xFFFEB514,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R6,          0xFFFEB518,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R7,          0xFFFEB51C,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R8,          0xFFFEB520,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R9,          0xFFFEB524,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R10,         0xFFFEB528,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R11,         0xFFFEB52C,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R12,         0xFFFEB530,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R13,         0xFFFEB534,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R14,         0xFFFEB538,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R15,         0xFFFEB53C,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R16,         0xFFFEB540,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R17,         0xFFFEB544,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R18,         0xFFFEB548,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R19,         0xFFFEB54C,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R20,         0xFFFEB550,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R21,         0xFFFEB554,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R22,         0xFFFEB558,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R23,         0xFFFEB55C,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R24,         0xFFFEB560,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R25,         0xFFFEB564,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R26,         0xFFFEB568,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R27,         0xFFFEB56C,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R28,         0xFFFEB570,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R29,         0xFFFEB574,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R30,         0xFFFEB578,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R31,         0xFFFEB57C,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R32,         0xFFFEB580,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R33,         0xFFFEB584,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R34,         0xFFFEB588,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R35,         0xFFFEB58C,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R36,         0xFFFEB590,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R37,         0xFFFEB594,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R38,         0xFFFEB598,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R39,         0xFFFEB59C,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R40,         0xFFFEB5A0,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R41,         0xFFFEB5A4,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R42,         0xFFFEB5A8,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R43,         0xFFFEB5AC,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R44,         0xFFFEB5B0,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R45,         0xFFFEB5B4,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R46,         0xFFFEB5B8,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R47,         0xFFFEB5BC,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R48,         0xFFFEB5C0,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R49,         0xFFFEB5C4,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R50,         0xFFFEB5C8,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R51,         0xFFFEB5CC,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R52,         0xFFFEB5D0,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R53,         0xFFFEB5D4,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R54,         0xFFFEB5D8,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R55,         0xFFFEB5DC,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R56,         0xFFFEB5E0,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R57,         0xFFFEB5E4,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R58,         0xFFFEB5E8,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R59,         0xFFFEB5EC,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R60,         0xFFFEB5F0,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R61,         0xFFFEB5F4,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R62,         0xFFFEB5F8,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_QT1R63,         0xFFFEB5FC,__READ_WRITE ,__jcodec_qt1r_bits);
__IO_REG16_BIT(JCODEC_DCHT0R0_0,      0xFFFEB800,__WRITE      ,__jcodec_dcht0r0_bits);
__IO_REG16_BIT(JCODEC_DCHT0R0_1,      0xFFFEB804,__WRITE      ,__jcodec_dcht0r0_bits);
__IO_REG16_BIT(JCODEC_DCHT0R0_2,      0xFFFEB808,__WRITE      ,__jcodec_dcht0r0_bits);
__IO_REG16_BIT(JCODEC_DCHT0R0_3,      0xFFFEB80C,__WRITE      ,__jcodec_dcht0r0_bits);
__IO_REG16_BIT(JCODEC_DCHT0R0_4,      0xFFFEB810,__WRITE      ,__jcodec_dcht0r0_bits);
__IO_REG16_BIT(JCODEC_DCHT0R0_5,      0xFFFEB814,__WRITE      ,__jcodec_dcht0r0_bits);
__IO_REG16_BIT(JCODEC_DCHT0R0_6,      0xFFFEB818,__WRITE      ,__jcodec_dcht0r0_bits);
__IO_REG16_BIT(JCODEC_DCHT0R0_7,      0xFFFEB81C,__WRITE      ,__jcodec_dcht0r0_bits);
__IO_REG16_BIT(JCODEC_DCHT0R0_8,      0xFFFEB820,__WRITE      ,__jcodec_dcht0r0_bits);
__IO_REG16_BIT(JCODEC_DCHT0R0_9,      0xFFFEB824,__WRITE      ,__jcodec_dcht0r0_bits);
__IO_REG16_BIT(JCODEC_DCHT0R0_10,     0xFFFEB828,__WRITE      ,__jcodec_dcht0r0_bits);
__IO_REG16_BIT(JCODEC_DCHT0R0_11,     0xFFFEB82C,__WRITE      ,__jcodec_dcht0r0_bits);
__IO_REG16_BIT(JCODEC_DCHT0R0_12,     0xFFFEB830,__WRITE      ,__jcodec_dcht0r0_bits);
__IO_REG16_BIT(JCODEC_DCHT0R0_13,     0xFFFEB834,__WRITE      ,__jcodec_dcht0r0_bits);
__IO_REG16_BIT(JCODEC_DCHT0R0_14,     0xFFFEB838,__WRITE      ,__jcodec_dcht0r0_bits);
__IO_REG16_BIT(JCODEC_DCHT0R0_15,     0xFFFEB83C,__WRITE      ,__jcodec_dcht0r0_bits);
__IO_REG16_BIT(JCODEC_DCHT0R1_0,      0xFFFEB840,__WRITE      ,__jcodec_dcht0r1_bits);
__IO_REG16_BIT(JCODEC_DCHT0R1_1,      0xFFFEB844,__WRITE      ,__jcodec_dcht0r1_bits);
__IO_REG16_BIT(JCODEC_DCHT0R1_2,      0xFFFEB848,__WRITE      ,__jcodec_dcht0r1_bits);
__IO_REG16_BIT(JCODEC_DCHT0R1_3,      0xFFFEB84C,__WRITE      ,__jcodec_dcht0r1_bits);
__IO_REG16_BIT(JCODEC_DCHT0R1_4,      0xFFFEB850,__WRITE      ,__jcodec_dcht0r1_bits);
__IO_REG16_BIT(JCODEC_DCHT0R1_5,      0xFFFEB854,__WRITE      ,__jcodec_dcht0r1_bits);
__IO_REG16_BIT(JCODEC_DCHT0R1_6,      0xFFFEB858,__WRITE      ,__jcodec_dcht0r1_bits);
__IO_REG16_BIT(JCODEC_DCHT0R1_7,      0xFFFEB85C,__WRITE      ,__jcodec_dcht0r1_bits);
__IO_REG16_BIT(JCODEC_DCHT0R1_8,      0xFFFEB860,__WRITE      ,__jcodec_dcht0r1_bits);
__IO_REG16_BIT(JCODEC_DCHT0R1_9,      0xFFFEB864,__WRITE      ,__jcodec_dcht0r1_bits);
__IO_REG16_BIT(JCODEC_DCHT0R1_10,     0xFFFEB868,__WRITE      ,__jcodec_dcht0r1_bits);
__IO_REG16_BIT(JCODEC_DCHT0R1_11,     0xFFFEB86C,__WRITE      ,__jcodec_dcht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R0_0,      0xFFFEB880,__WRITE      ,__jcodec_acht0r0_bits);
__IO_REG16_BIT(JCODEC_ACHT0R0_1,      0xFFFEB884,__WRITE      ,__jcodec_acht0r0_bits);
__IO_REG16_BIT(JCODEC_ACHT0R0_2,      0xFFFEB888,__WRITE      ,__jcodec_acht0r0_bits);
__IO_REG16_BIT(JCODEC_ACHT0R0_3,      0xFFFEB88C,__WRITE      ,__jcodec_acht0r0_bits);
__IO_REG16_BIT(JCODEC_ACHT0R0_4,      0xFFFEB890,__WRITE      ,__jcodec_acht0r0_bits);
__IO_REG16_BIT(JCODEC_ACHT0R0_5,      0xFFFEB894,__WRITE      ,__jcodec_acht0r0_bits);
__IO_REG16_BIT(JCODEC_ACHT0R0_6,      0xFFFEB898,__WRITE      ,__jcodec_acht0r0_bits);
__IO_REG16_BIT(JCODEC_ACHT0R0_7,      0xFFFEB89C,__WRITE      ,__jcodec_acht0r0_bits);
__IO_REG16_BIT(JCODEC_ACHT0R0_8,      0xFFFEB8A0,__WRITE      ,__jcodec_acht0r0_bits);
__IO_REG16_BIT(JCODEC_ACHT0R0_9,      0xFFFEB8A4,__WRITE      ,__jcodec_acht0r0_bits);
__IO_REG16_BIT(JCODEC_ACHT0R0_10,     0xFFFEB8A8,__WRITE      ,__jcodec_acht0r0_bits);
__IO_REG16_BIT(JCODEC_ACHT0R0_11,     0xFFFEB8AC,__WRITE      ,__jcodec_acht0r0_bits);
__IO_REG16_BIT(JCODEC_ACHT0R0_12,     0xFFFEB8B0,__WRITE      ,__jcodec_acht0r0_bits);
__IO_REG16_BIT(JCODEC_ACHT0R0_13,     0xFFFEB8B4,__WRITE      ,__jcodec_acht0r0_bits);
__IO_REG16_BIT(JCODEC_ACHT0R0_14,     0xFFFEB8B8,__WRITE      ,__jcodec_acht0r0_bits);
__IO_REG16_BIT(JCODEC_ACHT0R0_15,     0xFFFEB8BC,__WRITE      ,__jcodec_acht0r0_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_0,      0xFFFEB8C0,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_1,      0xFFFEB8C4,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_2,      0xFFFEB8C8,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_3,      0xFFFEB8CC,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_4,      0xFFFEB8D0,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_5,      0xFFFEB8D4,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_6,      0xFFFEB8D8,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_7,      0xFFFEB8DC,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_8,      0xFFFEB8E0,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_9,      0xFFFEB8E4,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_10,     0xFFFEB8E8,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_11,     0xFFFEB8EC,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_12,     0xFFFEB8F0,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_13,     0xFFFEB8F4,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_14,     0xFFFEB8F8,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_15,     0xFFFEB8FC,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_16,     0xFFFEB900,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_17,     0xFFFEB904,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_18,     0xFFFEB908,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_19,     0xFFFEB90C,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_20,     0xFFFEB910,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_21,     0xFFFEB914,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_22,     0xFFFEB918,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_23,     0xFFFEB91C,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_24,     0xFFFEB920,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_25,     0xFFFEB924,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_26,     0xFFFEB928,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_27,     0xFFFEB92C,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_28,     0xFFFEB930,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_29,     0xFFFEB934,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_30,     0xFFFEB938,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_31,     0xFFFEB93C,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_32,     0xFFFEB940,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_33,     0xFFFEB944,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_34,     0xFFFEB948,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_35,     0xFFFEB94C,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_36,     0xFFFEB950,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_37,     0xFFFEB954,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_38,     0xFFFEB958,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_39,     0xFFFEB95C,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_40,     0xFFFEB960,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_41,     0xFFFEB964,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_42,     0xFFFEB968,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_43,     0xFFFEB96C,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_44,     0xFFFEB970,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_45,     0xFFFEB974,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_46,     0xFFFEB978,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_47,     0xFFFEB97C,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_48,     0xFFFEB980,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_49,     0xFFFEB984,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_50,     0xFFFEB988,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_51,     0xFFFEB98C,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_52,     0xFFFEB990,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_53,     0xFFFEB994,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_54,     0xFFFEB998,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_55,     0xFFFEB99C,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_56,     0xFFFEB9A0,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_57,     0xFFFEB9A4,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_58,     0xFFFEB9A8,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_59,     0xFFFEB9AC,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_60,     0xFFFEB9B0,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_61,     0xFFFEB9B4,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_62,     0xFFFEB9B8,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_63,     0xFFFEB9BC,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_64,     0xFFFEB9C0,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_65,     0xFFFEB9C4,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_66,     0xFFFEB9C8,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_67,     0xFFFEB9CC,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_68,     0xFFFEB9D0,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_69,     0xFFFEB9D4,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_70,     0xFFFEB9D8,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_71,     0xFFFEB9DC,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_72,     0xFFFEB9E0,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_73,     0xFFFEB9E4,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_74,     0xFFFEB9E8,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_75,     0xFFFEB9EC,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_76,     0xFFFEB9F0,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_77,     0xFFFEB9F4,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_78,     0xFFFEB9F8,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_79,     0xFFFEB9FC,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_80,     0xFFFEBA00,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_81,     0xFFFEBA04,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_82,     0xFFFEBA08,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_83,     0xFFFEBA0C,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_84,     0xFFFEBA10,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_85,     0xFFFEBA14,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_86,     0xFFFEBA18,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_87,     0xFFFEBA1C,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_88,     0xFFFEBA20,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_89,     0xFFFEBA24,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_90,     0xFFFEBA28,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_91,     0xFFFEBA2C,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_92,     0xFFFEBA30,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_93,     0xFFFEBA34,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_94,     0xFFFEBA38,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_95,     0xFFFEBA3C,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_96,     0xFFFEBA40,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_97,     0xFFFEBA44,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_98,     0xFFFEBA48,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_99,     0xFFFEBA4C,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_100,    0xFFFEBA50,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_101,    0xFFFEBA54,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_102,    0xFFFEBA58,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_103,    0xFFFEBA5C,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_104,    0xFFFEBA60,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_105,    0xFFFEBA64,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_106,    0xFFFEBA68,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_107,    0xFFFEBA6C,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_108,    0xFFFEBA70,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_109,    0xFFFEBA74,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_110,    0xFFFEBA78,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_111,    0xFFFEBA7C,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_112,    0xFFFEBA80,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_113,    0xFFFEBA84,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_114,    0xFFFEBA88,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_115,    0xFFFEBA8C,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_116,    0xFFFEBA90,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_117,    0xFFFEBA94,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_118,    0xFFFEBA98,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_119,    0xFFFEBA9C,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_120,    0xFFFEBAA0,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_121,    0xFFFEBAA4,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_122,    0xFFFEBAA8,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_123,    0xFFFEBAAC,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_124,    0xFFFEBAB0,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_125,    0xFFFEBAB4,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_126,    0xFFFEBAB8,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_127,    0xFFFEBABC,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_128,    0xFFFEBAC0,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_129,    0xFFFEBAC4,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_130,    0xFFFEBAC8,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_131,    0xFFFEBACC,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_132,    0xFFFEBAD0,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_133,    0xFFFEBAD4,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_134,    0xFFFEBAD8,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_135,    0xFFFEBADC,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_136,    0xFFFEBAE0,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_137,    0xFFFEBAE4,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_138,    0xFFFEBAE8,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_139,    0xFFFEBAEC,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_140,    0xFFFEBAF0,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_141,    0xFFFEBAF4,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_142,    0xFFFEBAF8,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_143,    0xFFFEBAFC,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_144,    0xFFFEBB00,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_145,    0xFFFEBB04,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_146,    0xFFFEBB08,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_147,    0xFFFEBB0C,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_148,    0xFFFEBB10,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_149,    0xFFFEBB14,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_150,    0xFFFEBB18,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_151,    0xFFFEBB1C,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_152,    0xFFFEBB20,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_153,    0xFFFEBB24,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_154,    0xFFFEBB28,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_155,    0xFFFEBB2C,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_156,    0xFFFEBB30,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_157,    0xFFFEBB34,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_158,    0xFFFEBB38,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_159,    0xFFFEBB3C,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_160,    0xFFFEBB40,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_ACHT0R1_161,    0xFFFEBB44,__WRITE      ,__jcodec_acht0r1_bits);
__IO_REG16_BIT(JCODEC_DCHT1R0_0,      0xFFFEBC00,__WRITE      ,__jcodec_dcht1r0_bits);
__IO_REG16_BIT(JCODEC_DCHT1R0_1,      0xFFFEBC04,__WRITE      ,__jcodec_dcht1r0_bits);
__IO_REG16_BIT(JCODEC_DCHT1R0_2,      0xFFFEBC08,__WRITE      ,__jcodec_dcht1r0_bits);
__IO_REG16_BIT(JCODEC_DCHT1R0_3,      0xFFFEBC0C,__WRITE      ,__jcodec_dcht1r0_bits);
__IO_REG16_BIT(JCODEC_DCHT1R0_4,      0xFFFEBC10,__WRITE      ,__jcodec_dcht1r0_bits);
__IO_REG16_BIT(JCODEC_DCHT1R0_5,      0xFFFEBC14,__WRITE      ,__jcodec_dcht1r0_bits);
__IO_REG16_BIT(JCODEC_DCHT1R0_6,      0xFFFEBC18,__WRITE      ,__jcodec_dcht1r0_bits);
__IO_REG16_BIT(JCODEC_DCHT1R0_7,      0xFFFEBC1C,__WRITE      ,__jcodec_dcht1r0_bits);
__IO_REG16_BIT(JCODEC_DCHT1R0_8,      0xFFFEBC20,__WRITE      ,__jcodec_dcht1r0_bits);
__IO_REG16_BIT(JCODEC_DCHT1R0_9,      0xFFFEBC24,__WRITE      ,__jcodec_dcht1r0_bits);
__IO_REG16_BIT(JCODEC_DCHT1R0_10,     0xFFFEBC28,__WRITE      ,__jcodec_dcht1r0_bits);
__IO_REG16_BIT(JCODEC_DCHT1R0_11,     0xFFFEBC2C,__WRITE      ,__jcodec_dcht1r0_bits);
__IO_REG16_BIT(JCODEC_DCHT1R0_12,     0xFFFEBC30,__WRITE      ,__jcodec_dcht1r0_bits);
__IO_REG16_BIT(JCODEC_DCHT1R0_13,     0xFFFEBC34,__WRITE      ,__jcodec_dcht1r0_bits);
__IO_REG16_BIT(JCODEC_DCHT1R0_14,     0xFFFEBC38,__WRITE      ,__jcodec_dcht1r0_bits);
__IO_REG16_BIT(JCODEC_DCHT1R0_15,     0xFFFEBC3C,__WRITE      ,__jcodec_dcht1r0_bits);
__IO_REG16_BIT(JCODEC_DCHT1R1_0,      0xFFFEBC40,__WRITE      ,__jcodec_dcht1r1_bits);
__IO_REG16_BIT(JCODEC_DCHT1R1_1,      0xFFFEBC44,__WRITE      ,__jcodec_dcht1r1_bits);
__IO_REG16_BIT(JCODEC_DCHT1R1_2,      0xFFFEBC48,__WRITE      ,__jcodec_dcht1r1_bits);
__IO_REG16_BIT(JCODEC_DCHT1R1_3,      0xFFFEBC4C,__WRITE      ,__jcodec_dcht1r1_bits);
__IO_REG16_BIT(JCODEC_DCHT1R1_4,      0xFFFEBC50,__WRITE      ,__jcodec_dcht1r1_bits);
__IO_REG16_BIT(JCODEC_DCHT1R1_5,      0xFFFEBC54,__WRITE      ,__jcodec_dcht1r1_bits);
__IO_REG16_BIT(JCODEC_DCHT1R1_6,      0xFFFEBC58,__WRITE      ,__jcodec_dcht1r1_bits);
__IO_REG16_BIT(JCODEC_DCHT1R1_7,      0xFFFEBC5C,__WRITE      ,__jcodec_dcht1r1_bits);
__IO_REG16_BIT(JCODEC_DCHT1R1_8,      0xFFFEBC60,__WRITE      ,__jcodec_dcht1r1_bits);
__IO_REG16_BIT(JCODEC_DCHT1R1_9,      0xFFFEBC64,__WRITE      ,__jcodec_dcht1r1_bits);
__IO_REG16_BIT(JCODEC_DCHT1R1_10,     0xFFFEBC68,__WRITE      ,__jcodec_dcht1r1_bits);
__IO_REG16_BIT(JCODEC_DCHT1R1_11,     0xFFFEBC6C,__WRITE      ,__jcodec_dcht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R0_0,      0xFFFEBC80,__WRITE      ,__jcodec_acht1r0_bits);
__IO_REG16_BIT(JCODEC_ACHT1R0_1,      0xFFFEBC84,__WRITE      ,__jcodec_acht1r0_bits);
__IO_REG16_BIT(JCODEC_ACHT1R0_2,      0xFFFEBC88,__WRITE      ,__jcodec_acht1r0_bits);
__IO_REG16_BIT(JCODEC_ACHT1R0_3,      0xFFFEBC8C,__WRITE      ,__jcodec_acht1r0_bits);
__IO_REG16_BIT(JCODEC_ACHT1R0_4,      0xFFFEBC90,__WRITE      ,__jcodec_acht1r0_bits);
__IO_REG16_BIT(JCODEC_ACHT1R0_5,      0xFFFEBC94,__WRITE      ,__jcodec_acht1r0_bits);
__IO_REG16_BIT(JCODEC_ACHT1R0_6,      0xFFFEBC98,__WRITE      ,__jcodec_acht1r0_bits);
__IO_REG16_BIT(JCODEC_ACHT1R0_7,      0xFFFEBC9C,__WRITE      ,__jcodec_acht1r0_bits);
__IO_REG16_BIT(JCODEC_ACHT1R0_8,      0xFFFEBCA0,__WRITE      ,__jcodec_acht1r0_bits);
__IO_REG16_BIT(JCODEC_ACHT1R0_9,      0xFFFEBCA4,__WRITE      ,__jcodec_acht1r0_bits);
__IO_REG16_BIT(JCODEC_ACHT1R0_10,     0xFFFEBCA8,__WRITE      ,__jcodec_acht1r0_bits);
__IO_REG16_BIT(JCODEC_ACHT1R0_11,     0xFFFEBCAC,__WRITE      ,__jcodec_acht1r0_bits);
__IO_REG16_BIT(JCODEC_ACHT1R0_12,     0xFFFEBCB0,__WRITE      ,__jcodec_acht1r0_bits);
__IO_REG16_BIT(JCODEC_ACHT1R0_13,     0xFFFEBCB4,__WRITE      ,__jcodec_acht1r0_bits);
__IO_REG16_BIT(JCODEC_ACHT1R0_14,     0xFFFEBCB8,__WRITE      ,__jcodec_acht1r0_bits);
__IO_REG16_BIT(JCODEC_ACHT1R0_15,     0xFFFEBCBC,__WRITE      ,__jcodec_acht1r0_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_0,      0xFFFEBCC0,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_1,      0xFFFEBCC4,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_2,      0xFFFEBCC8,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_3,      0xFFFEBCCC,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_4,      0xFFFEBCD0,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_5,      0xFFFEBCD4,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_6,      0xFFFEBCD8,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_7,      0xFFFEBCDC,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_8,      0xFFFEBCE0,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_9,      0xFFFEBCE4,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_10,     0xFFFEBCE8,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_11,     0xFFFEBCEC,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_12,     0xFFFEBCF0,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_13,     0xFFFEBCF4,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_14,     0xFFFEBCF8,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_15,     0xFFFEBCFC,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_16,     0xFFFEBD00,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_17,     0xFFFEBD04,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_18,     0xFFFEBD08,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_19,     0xFFFEBD0C,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_20,     0xFFFEBD10,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_21,     0xFFFEBD14,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_22,     0xFFFEBD18,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_23,     0xFFFEBD1C,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_24,     0xFFFEBD20,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_25,     0xFFFEBD24,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_26,     0xFFFEBD28,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_27,     0xFFFEBD2C,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_28,     0xFFFEBD30,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_29,     0xFFFEBD34,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_30,     0xFFFEBD38,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_31,     0xFFFEBD3C,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_32,     0xFFFEBD40,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_33,     0xFFFEBD44,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_34,     0xFFFEBD48,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_35,     0xFFFEBD4C,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_36,     0xFFFEBD50,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_37,     0xFFFEBD54,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_38,     0xFFFEBD58,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_39,     0xFFFEBD5C,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_40,     0xFFFEBD60,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_41,     0xFFFEBD64,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_42,     0xFFFEBD68,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_43,     0xFFFEBD6C,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_44,     0xFFFEBD70,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_45,     0xFFFEBD74,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_46,     0xFFFEBD78,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_47,     0xFFFEBD7C,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_48,     0xFFFEBD80,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_49,     0xFFFEBD84,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_50,     0xFFFEBD88,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_51,     0xFFFEBD8C,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_52,     0xFFFEBD90,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_53,     0xFFFEBD94,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_54,     0xFFFEBD98,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_55,     0xFFFEBD9C,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_56,     0xFFFEBDA0,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_57,     0xFFFEBDA4,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_58,     0xFFFEBDA8,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_59,     0xFFFEBDAC,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_60,     0xFFFEBDB0,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_61,     0xFFFEBDB4,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_62,     0xFFFEBDB8,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_63,     0xFFFEBDBC,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_64,     0xFFFEBDC0,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_65,     0xFFFEBDC4,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_66,     0xFFFEBDC8,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_67,     0xFFFEBDCC,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_68,     0xFFFEBDD0,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_69,     0xFFFEBDD4,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_70,     0xFFFEBDD8,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_71,     0xFFFEBDDC,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_72,     0xFFFEBDE0,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_73,     0xFFFEBDE4,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_74,     0xFFFEBDE8,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_75,     0xFFFEBDEC,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_76,     0xFFFEBDF0,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_77,     0xFFFEBDF4,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_78,     0xFFFEBDF8,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_79,     0xFFFEBDFC,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_80,     0xFFFEBE00,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_81,     0xFFFEBE04,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_82,     0xFFFEBE08,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_83,     0xFFFEBE0C,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_84,     0xFFFEBE10,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_85,     0xFFFEBE14,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_86,     0xFFFEBE18,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_87,     0xFFFEBE1C,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_88,     0xFFFEBE20,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_89,     0xFFFEBE24,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_90,     0xFFFEBE28,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_91,     0xFFFEBE2C,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_92,     0xFFFEBE30,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_93,     0xFFFEBE34,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_94,     0xFFFEBE38,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_95,     0xFFFEBE3C,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_96,     0xFFFEBE40,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_97,     0xFFFEBE44,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_98,     0xFFFEBE48,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_99,     0xFFFEBE4C,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_100,    0xFFFEBE50,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_101,    0xFFFEBE54,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_102,    0xFFFEBE58,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_103,    0xFFFEBE5C,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_104,    0xFFFEBE60,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_105,    0xFFFEBE64,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_106,    0xFFFEBE68,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_107,    0xFFFEBE6C,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_108,    0xFFFEBE70,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_109,    0xFFFEBE74,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_110,    0xFFFEBE78,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_111,    0xFFFEBE7C,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_112,    0xFFFEBE80,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_113,    0xFFFEBE84,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_114,    0xFFFEBE88,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_115,    0xFFFEBE8C,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_116,    0xFFFEBE90,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_117,    0xFFFEBE94,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_118,    0xFFFEBE98,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_119,    0xFFFEBE9C,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_120,    0xFFFEBEA0,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_121,    0xFFFEBEA4,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_122,    0xFFFEBEA8,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_123,    0xFFFEBEAC,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_124,    0xFFFEBEB0,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_125,    0xFFFEBEB4,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_126,    0xFFFEBEB8,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_127,    0xFFFEBEBC,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_128,    0xFFFEBEC0,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_129,    0xFFFEBEC4,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_130,    0xFFFEBEC8,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_131,    0xFFFEBECC,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_132,    0xFFFEBED0,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_133,    0xFFFEBED4,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_134,    0xFFFEBED8,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_135,    0xFFFEBEDC,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_136,    0xFFFEBEE0,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_137,    0xFFFEBEE4,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_138,    0xFFFEBEE8,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_139,    0xFFFEBEEC,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_140,    0xFFFEBEF0,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_141,    0xFFFEBEF4,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_142,    0xFFFEBEF8,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_143,    0xFFFEBEFC,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_144,    0xFFFEBF00,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_145,    0xFFFEBF04,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_146,    0xFFFEBF08,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_147,    0xFFFEBF0C,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_148,    0xFFFEBF10,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_149,    0xFFFEBF14,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_150,    0xFFFEBF18,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_151,    0xFFFEBF1C,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_152,    0xFFFEBF20,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_153,    0xFFFEBF24,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_154,    0xFFFEBF28,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_155,    0xFFFEBF2C,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_156,    0xFFFEBF30,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_157,    0xFFFEBF34,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_158,    0xFFFEBF38,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_159,    0xFFFEBF3C,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_160,    0xFFFEBF40,__WRITE      ,__jcodec_acht1r1_bits);
__IO_REG16_BIT(JCODEC_ACHT1R1_161,    0xFFFEBF44,__WRITE      ,__jcodec_acht1r1_bits);

/***************************************************************************
 **
 ** JDMA
 **
 ***************************************************************************/
__IO_REG32(    JSAR,                  0xFFFEC000,__READ_WRITE );
__IO_REG32(    JDAR,                  0xFFFEC004,__READ_WRITE );
__IO_REG32_BIT(JTCR,                  0xFFFEC008,__READ_WRITE ,__jtcr_bits);
__IO_REG32_BIT(JCTL,                  0xFFFEC00C,__READ_WRITE ,__jctl_bits);
__IO_REG32_BIT(JBCR,                  0xFFFEC010,__READ_WRITE ,__jbcr_bits);
__IO_REG32_BIT(JOFR,                  0xFFFEC014,__READ_WRITE ,__jofr_bits);
__IO_REG32_BIT(JBER,                  0xFFFEC018,__READ_WRITE ,__jber_bits);
__IO_REG32_BIT(JHID,                  0xFFFEC020,__READ_WRITE ,__jhid_bits);
__IO_REG32_BIT(JFSM,                  0xFFFEC040,__READ_WRITE ,__jfsm_bits);

/***************************************************************************
 **
 ** DMAC2
 **
 ***************************************************************************/
__IO_REG32(    DMAC2_SAR0,            0xFFFF9000,__READ_WRITE );
__IO_REG32(    DMAC2_DAR0,            0xFFFF9004,__READ_WRITE );
__IO_REG32_BIT(DMAC2_TCR0,            0xFFFF9008,__READ_WRITE ,__dmac2_tcr_bits);
__IO_REG32_BIT(DMAC2_CTL0,            0xFFFF900C,__READ_WRITE ,__dmac2_ctl_bits);
__IO_REG32(    DMAC2_SAR1,            0xFFFF9010,__READ_WRITE );
__IO_REG32(    DMAC2_DAR1,            0xFFFF9014,__READ_WRITE );
__IO_REG32_BIT(DMAC2_TCR1,            0xFFFF9018,__READ_WRITE ,__dmac2_tcr_bits);
__IO_REG32_BIT(DMAC2_CTL1,            0xFFFF901C,__READ_WRITE ,__dmac2_ctl_bits);
__IO_REG32_BIT(DMAC2_OPSR,            0xFFFF9060,__READ_WRITE ,__dmac2_opsr_bits);
__IO_REG32_BIT(DMAC2_MISC,            0xFFFF9064,__READ_WRITE ,__dmac2_misc_bits);
__IO_REG32_BIT(DMAC2_TECL,            0xFFFF9070,__READ_WRITE ,__dmac2_tecl_bits);

/***************************************************************************
 **
 ** ETH
 **
 ***************************************************************************/
__IO_REG32_BIT(ETH_ISR,               0xFFFE2000,__READ       ,__eth_isr_bits);
__IO_REG32_BIT(ETH_IER,               0xFFFE2004,__READ_WRITE ,__eth_ier_bits);
__IO_REG32_BIT(ETH_RR,                0xFFFE2008,__READ_WRITE ,__eth_rr_bits);
__IO_REG32_BIT(ETH_PHYSR,             0xFFFE200C,__READ       ,__eth_physr_bits);
__IO_REG32_BIT(ETH_DMACR,             0xFFFE2010,__READ_WRITE ,__eth_dmacr_bits);
__IO_REG32(    ETH_TXDMAPR,           0xFFFE2018,__READ_WRITE );
__IO_REG32(    ETH_RXDMAPR,           0xFFFE201C,__READ_WRITE );
__IO_REG32_BIT(ETH_MR,                0xFFFE2020,__READ_WRITE ,__eth_mr_bits);
__IO_REG32_BIT(ETH_TXMR,              0xFFFE2024,__READ_WRITE ,__eth_txmr_bits);
__IO_REG32_BIT(ETH_RXMR,              0xFFFE2028,__READ_WRITE ,__eth_rxmr_bits);
__IO_REG32_BIT(ETH_MIIMR,             0xFFFE202C,__READ_WRITE ,__eth_miimr_bits);
__IO_REG32(    ETH_MACADR1L,          0xFFFE2030,__READ_WRITE );
__IO_REG32_BIT(ETH_MACADR1U,          0xFFFE2034,__READ_WRITE ,__eth_macadru_bits);
__IO_REG32(    ETH_MACADR2L,          0xFFFE2038,__READ_WRITE );
__IO_REG32_BIT(ETH_MACADR2U,          0xFFFE203C,__READ_WRITE ,__eth_macadru_bits);
__IO_REG32(    ETH_MACADR3L,          0xFFFE2040,__READ_WRITE );
__IO_REG32_BIT(ETH_MACADR3U,          0xFFFE2044,__READ_WRITE ,__eth_macadru_bits);
__IO_REG32(    ETH_MACADR4L,          0xFFFE2048,__READ_WRITE );
__IO_REG32_BIT(ETH_MACADR4U,          0xFFFE204C,__READ_WRITE ,__eth_macadru_bits);
__IO_REG32(    ETH_MACADR5L,          0xFFFE2050,__READ_WRITE );
__IO_REG32_BIT(ETH_MACADR5U,          0xFFFE2054,__READ_WRITE ,__eth_macadru_bits);
__IO_REG32(    ETH_MACADR6L,          0xFFFE2058,__READ_WRITE );
__IO_REG32_BIT(ETH_MACADR6U,          0xFFFE205C,__READ_WRITE ,__eth_macadru_bits);
__IO_REG32(    ETH_MACADR7L,          0xFFFE2060,__READ_WRITE );
__IO_REG32_BIT(ETH_MACADR7U,          0xFFFE2064,__READ_WRITE ,__eth_macadru_bits);
__IO_REG32(    ETH_MACADR8L,          0xFFFE2068,__READ_WRITE );
__IO_REG32_BIT(ETH_MACADR8U,          0xFFFE206C,__READ_WRITE ,__eth_macadru_bits);
__IO_REG32_BIT(ETH_FCR,               0xFFFE2070,__READ_WRITE ,__eth_fcr_bits);
__IO_REG32_BIT(ETH_PRR,               0xFFFE2074,__READ_WRITE ,__eth_prr_bits);
__IO_REG32(    ETH_PFDR1,             0xFFFE2078,__READ_WRITE );
__IO_REG32(    ETH_PFDR2,             0xFFFE207C,__READ_WRITE );
__IO_REG32(    ETH_PFDR3,             0xFFFE2080,__READ_WRITE );
__IO_REG32(    ETH_PFDR4,             0xFFFE2084,__READ_WRITE );
__IO_REG32(    ETH_PFDR5,             0xFFFE2088,__READ_WRITE );
__IO_REG32_BIT(ETH_BMER,              0xFFFE2090,__READ_WRITE ,__eth_bmer_bits);
__IO_REG32_BIT(ETH_BFR,               0xFFFE2094,__READ_WRITE ,__eth_bfr_bits);
__IO_REG32_BIT(ETH_BIR,               0xFFFE2098,__READ_WRITE ,__eth_bir_bits);
__IO_REG32_BIT(ETH_PIR,               0xFFFE209C,__READ_WRITE ,__eth_pir_bits);
__IO_REG32_BIT(ETH_TXFIFOSR,          0xFFFE20F0,__READ       ,__eth_txfifosr_bits);
__IO_REG32_BIT(ETH_RXFIFOSR,          0xFFFE20F4,__READ       ,__eth_rxfifosr_bits);

/***************************************************************************
 **
 ** APB
 **
 ***************************************************************************/
__IO_REG32_BIT(APBWAIT0,              0xFFFE0000,__READ_WRITE ,__apbwait0_bits);
__IO_REG32_BIT(APBWAIT1,              0xFFFE0004,__READ_WRITE ,__apbwait1_bits);

/***************************************************************************
 **
 ** SYS
 **
 ***************************************************************************/
__IO_REG32_BIT(CHIPID,                0xFFFFD000,__READ       ,__chipid_bits);
__IO_REG16(    CHIPCFG,               0xFFFFD004,__READ       );
__IO_REG32_BIT(PLLSET1,               0xFFFFD008,__READ_WRITE ,__pllset1_bits);
__IO_REG32_BIT(PLLSET2,               0xFFFFD00C,__READ_WRITE ,__pllset2_bits);
__IO_REG32_BIT(HALTMODE,              0xFFFFD010,__READ_WRITE ,__haltmode_bits);
__IO_REG32_BIT(IOCLKCTL,              0xFFFFD014,__READ_WRITE ,__ioclkctl_bits);
__IO_REG32_BIT(CLK32SEL,              0xFFFFD018,__READ_WRITE ,__clk32sel_bits);
__IO_REG32(    HALTCTL,               0xFFFFD01C,__WRITE      );
__IO_REG32_BIT(REMAP,                 0xFFFFD020,__READ_WRITE ,__remap_bits);
__IO_REG32(    SOFTRST,               0xFFFFD024,__WRITE      );
__IO_REG32_BIT(UARTDIV,               0xFFFFD028,__READ_WRITE ,__uartdiv_bits);
__IO_REG32_BIT(MDPLDCTL,              0xFFFFD02C,__READ_WRITE ,__mdpldctl_bits);
__IO_REG32_BIT(PORTCRCTL,             0xFFFFD030,__READ_WRITE ,__portcrctl_bits);
__IO_REG32_BIT(PORTDRCTL,             0xFFFFD034,__READ_WRITE ,__portdrctl_bits);
__IO_REG32_BIT(PORTERCTL,             0xFFFFD038,__READ_WRITE ,__porterctl_bits);
__IO_REG32(    ITESTM,                0xFFFFD03C,__READ_WRITE );
__IO_REG32_BIT(EMBMEMCTL,             0xFFFFD040,__READ_WRITE ,__embmemctl_bits);

/***************************************************************************
 **
 ** MEMC
 **
 ***************************************************************************/
__IO_REG32_BIT(MEMC_CFG0,             0xFFFFA000,__READ_WRITE ,__memc_cfg_bits);
__IO_REG32_BIT(MEMC_CFG1,             0xFFFFA004,__READ_WRITE ,__memc_cfg_bits);
__IO_REG32_BIT(MEMC_CFG2,             0xFFFFA008,__READ_WRITE ,__memc_cfg_bits);
/*__IO_REG32_BIT(MEMC_CFG3,             0xFFFFA00C,__READ_WRITE ,__memc_cfg_bits);*/
__IO_REG32_BIT(RAMTMG0,               0xFFFFA020,__READ_WRITE ,__ramtmg_bits);
__IO_REG32_BIT(RAMCNTL0,              0xFFFFA024,__READ_WRITE ,__ramcntl_bits);
__IO_REG32_BIT(RAMTMG1,               0xFFFFA030,__READ_WRITE ,__ramtmg_bits);
__IO_REG32_BIT(RAMCNTL1,              0xFFFFA034,__READ_WRITE ,__ramcntl_bits);
__IO_REG32_BIT(RAMTMG2,               0xFFFFA040,__READ_WRITE ,__ramtmg_bits);
__IO_REG32_BIT(RAMCNTL2,              0xFFFFA044,__READ_WRITE ,__ramcntl_bits);
/*__IO_REG32_BIT(RAMTMG3,               0xFFFFA050,__READ_WRITE ,__ramtmg_bits);*/
/*__IO_REG32_BIT(RAMCNTL3,              0xFFFFA054,__READ_WRITE ,__ramcntl_bits);*/
__IO_REG32_BIT(SDMR,                  0xFFFFA060,__READ_WRITE ,__sdmr_bits);
__IO_REG32_BIT(SDCNFG,                0xFFFFA070,__READ_WRITE ,__sdcnfg_bits);
__IO_REG32_BIT(SDADVCNFG,             0xFFFFA074,__READ_WRITE ,__sdadvcnfg_bits);
__IO_REG32_BIT(SDINIT,                0xFFFFA080,__READ_WRITE ,__sdinit_bits);
__IO_REG32_BIT(SDREF,                 0xFFFFA090,__READ_WRITE ,__sdref_bits);
__IO_REG32_BIT(SDSTAT,                0xFFFFA0A0,__READ       ,__sdstat_bits);

/***************************************************************************
 **
 ** INTC
 **
 ***************************************************************************/
__IO_REG32_BIT(IRQ_SR,                0xFFFFF000,__READ       ,__irq_sr_bits);
__IO_REG32_BIT(IRQ_RSR,               0xFFFFF004,__READ       ,__irq_sr_bits);
__IO_REG32_BIT(IRQ_ER,                0xFFFFF008,__READ_WRITE ,__irq_sr_bits);
__IO_REG32_BIT(IRQ_ECR,               0xFFFFF00C,__WRITE      ,__irq_sr_bits);
__IO_REG32_BIT(IRQ_SIR,               0xFFFFF010,__WRITE      ,__irq_sir_bits);
__IO_REG32_BIT(IRQ_LR,                0xFFFFF080,__READ_WRITE ,__irq_lr_bits);
__IO_REG32_BIT(IRQ_PR,                0xFFFFF084,__READ_WRITE ,__irq_pr_bits);
__IO_REG32_BIT(IRQ_TRR,               0xFFFFF088,__WRITE      ,__irq_trp_bits);
__IO_REG32_BIT(FIQ_SR,                0xFFFFF100,__READ       ,__fiq_sr_bits);
__IO_REG32_BIT(FIQ_RSR,               0xFFFFF104,__READ       ,__fiq_sr_bits);
__IO_REG32_BIT(FIQ_ER,                0xFFFFF108,__READ_WRITE ,__fiq_sr_bits);
__IO_REG32_BIT(FIQ_ECR,               0xFFFFF10C,__WRITE      ,__fiq_sr_bits);
__IO_REG32_BIT(FIQ_LR,                0xFFFFF180,__READ_WRITE ,__fiq_lr_bits);
__IO_REG32_BIT(FIQ_PR,                0xFFFFF184,__READ_WRITE ,__fiq_pr_bits);
__IO_REG32_BIT(FIQ_TRR,               0xFFFFF188,__WRITE      ,__fiq_trr_bits);

/***************************************************************************
 **
 ** UART
 **
 ***************************************************************************/
__IO_REG8(     UART_RBR,              0xFFFF5000,__READ_WRITE );
#define UART_THR      UART_RBR
#define UART_DLL      UART_RBR
__IO_REG8_BIT( UART_IER,              0xFFFF5004,__READ_WRITE ,__uart_ier_bits);
#define UART_DLM      UART_IER
__IO_REG8_BIT( UART_IIR,              0xFFFF5008,__READ_WRITE ,__uart_iir_bits);
#define UART_FCR      UART_IIR
#define UART_FCR_bit  UART_IIR_bit
__IO_REG8_BIT( UART_LCR,              0xFFFF500C,__READ_WRITE ,__uart_lcr_bits);
__IO_REG8_BIT( UART_MCR,              0xFFFF5010,__READ_WRITE ,__uart_mcr_bits);
__IO_REG8_BIT( UART_LSR,              0xFFFF5014,__READ       ,__uart_lsr_bits);
__IO_REG8_BIT( UART_MSR,              0xFFFF5018,__READ       ,__uart_msr_bits);
__IO_REG8(     UART_SCR,              0xFFFF501C,__READ_WRITE );
__IO_REG8_BIT( UART_T0,               0xFFFF5020,__READ_WRITE ,__uart_t0_bits);
__IO_REG8_BIT( UART_T1,               0xFFFF5024,__READ_WRITE ,__uart_t1_bits);
__IO_REG8_BIT( UART_TS0,              0xFFFF5028,__READ       ,__uart_ts0_bits);
__IO_REG8_BIT( UART_TS1,              0xFFFF502C,__READ       ,__uart_ts1_bits);
__IO_REG8_BIT( UART_TS2,              0xFFFF5030,__READ       ,__uart_ts2_bits);
__IO_REG8_BIT( UART_TS3,              0xFFFF503C,__READ       ,__uart_ts3_bits);

/***************************************************************************
 **
 ** UARTL
 **
 ***************************************************************************/
__IO_REG8(     UARTL_RBR,             0xFFFF6000,__READ_WRITE );
#define UARTL_THR     UARTL_RBR
#define UARTL_DLL     UARTL_RBR
__IO_REG8_BIT( UARTL_IER,             0xFFFF6004,__READ_WRITE ,__uartl_ier_bits);
#define UARTL_DLM     UARTL_IER
__IO_REG8_BIT( UARTL_IIR,             0xFFFF6008,__READ       ,__uartl_iir_bits);
__IO_REG8_BIT( UARTL_LCR,             0xFFFF600C,__READ_WRITE ,__uartl_lcr_bits);
__IO_REG8_BIT( UARTL_LSR,             0xFFFF6014,__READ       ,__uartl_lsr_bits);
__IO_REG8_BIT( UARTL_TS2,             0xFFFF6030,__READ       ,__uartl_ts2_bits);
__IO_REG8_BIT( UARTL_TS3,             0xFFFF603C,__READ       ,__uartl_ts3_bits);

/***************************************************************************
 **
 ** I2C
 **
 ***************************************************************************/
__IO_REG8(     I2C_TDR,               0xFFFED000,__READ_WRITE );
__IO_REG8(     I2C_RTR,               0xFFFED004,__READ       );
__IO_REG8_BIT( I2C_CR,                0xFFFED008,__READ_WRITE ,__i2c_cr_bits);
__IO_REG8_BIT( I2C_BSR,               0xFFFED00C,__READ       ,__i2c_bsr_bits);
__IO_REG8_BIT( I2C_ESR,               0xFFFED010,__READ       ,__i2c_esr_bits);
__IO_REG8_BIT( I2C_ICSR,              0xFFFED014,__READ_WRITE ,__i2c_icsr_bits);
__IO_REG8_BIT( I2C_SCFDR,             0xFFFED018,__READ_WRITE ,__i2c_scfdr_bits);
__IO_REG8_BIT( I2C_CFDR,              0xFFFED01C,__READ_WRITE ,__i2c_cfdr_bits);
__IO_REG8_BIT( I2C_IOCR,              0xFFFED020,__READ_WRITE ,__i2c_iocr_bits);
__IO_REG8_BIT( I2C_DMAM,              0xFFFED024,__READ_WRITE ,__i2c_dmam_bits);
__IO_REG8(     I2C_DMACLR,            0xFFFED028,__READ_WRITE );
__IO_REG8(     I2C_DMACHR,            0xFFFED02C,__READ_WRITE );
__IO_REG8_BIT( I2C_DMASR,             0xFFFED030,__READ       ,__i2c_dmasr_bits);

/***************************************************************************
 **
 ** I2S0
 **
 ***************************************************************************/
__IO_REG16_BIT(I2S0_CR,               0xFFFEE000,__READ_WRITE ,__i2s_cr_bits);
__IO_REG16_BIT(I2S0_CFDR,             0xFFFEE004,__READ_WRITE ,__i2s_cfdr_bits);
__IO_REG32(    I2S0_TPR,              0xFFFEE008,__READ_WRITE );
__IO_REG16_BIT(I2S0_ISR,              0xFFFEE010,__READ_WRITE ,__i2s_isr_bits);
__IO_REG16_BIT(I2S0_RISR,             0xFFFEE014,__READ       ,__i2s_risr_bits);
__IO_REG16_BIT(I2S0_IER,              0xFFFEE018,__READ_WRITE ,__i2s_ier_bits);
__IO_REG16_BIT(I2S0_CISR,             0xFFFEE01C,__READ       ,__i2s_cisr_bits);

/***************************************************************************
 **
 ** I2S1
 **
 ***************************************************************************/
__IO_REG16_BIT(I2S1_CR,               0xFFFEE040,__READ_WRITE ,__i2s_cr_bits);
__IO_REG16_BIT(I2S1_CFDR,             0xFFFEE044,__READ_WRITE ,__i2s_cfdr_bits);
__IO_REG32(    I2S1_TPR,              0xFFFEE048,__READ_WRITE );
__IO_REG16_BIT(I2S1_ISR,              0xFFFEE050,__READ_WRITE ,__i2s_isr_bits);
__IO_REG16_BIT(I2S1_RISR,             0xFFFEE054,__READ       ,__i2s_risr_bits);
__IO_REG16_BIT(I2S1_IER,              0xFFFEE058,__READ_WRITE ,__i2s_ier_bits);
__IO_REG16_BIT(I2S1_CISR,             0xFFFEE05C,__READ       ,__i2s_cisr_bits);

/***************************************************************************
 **
 ** SPI
 **
 ***************************************************************************/
__IO_REG32(    SPI_RDR,               0xFFFF2000,__READ       );
__IO_REG32(    SPI_TDR,               0xFFFF2004,__READ_WRITE );
__IO_REG32_BIT(SPI_CR1,               0xFFFF2008,__READ_WRITE ,__spi_cr1_bits);
__IO_REG32_BIT(SPI_CR2,               0xFFFF200C,__READ_WRITE ,__spi_cr2_bits);
__IO_REG32_BIT(SPI_WR,                0xFFFF2010,__READ_WRITE ,__spi_wr_bits);
__IO_REG32_BIT(SPI_SR,                0xFFFF2014,__READ       ,__spi_sr_bits);
__IO_REG32_BIT(SPI_ICR,               0xFFFF2018,__READ_WRITE ,__spi_icr_bits);

/***************************************************************************
 **
 ** CF
 **
 ***************************************************************************/
__IO_REG16_BIT(CFCTL,                 0xFFFE6000,__READ_WRITE ,__cfctl_bits);
__IO_REG16_BIT(CFPINSTS,              0xFFFE6004,__READ       ,__cfpinsts_bits);
__IO_REG16_BIT(CFINTRSTS,             0xFFFE6008,__READ_WRITE ,__cfintrsts_bits);
__IO_REG16_BIT(CFINTMSTS,             0xFFFE600C,__READ_WRITE ,__cfintmsts_bits);
__IO_REG16_BIT(CFINTSTS,              0xFFFE6010,__READ       ,__cfintrsts_bits);
__IO_REG16_BIT(CFMISC,                0xFFFE6014,__READ_WRITE ,__cfmisc_bits);

/***************************************************************************
 **
 ** TIMERs
 **
 ***************************************************************************/
__IO_REG16(    TM0LD,                 0xFFFFB000,__READ_WRITE );
__IO_REG16(    TM0CNT,                0xFFFFB004,__READ       );
__IO_REG16_BIT(TM0CTRL,               0xFFFFB008,__READ_WRITE ,__tmctrl_bits);
__IO_REG16(    TM0IRQ,                0xFFFFB00C,__WRITE      );
__IO_REG16_BIT(TM0POUT,               0xFFFFB010,__READ_WRITE ,__tmpout_bits);
__IO_REG16(    TM1LD,                 0xFFFFB020,__READ_WRITE );
__IO_REG16(    TM1CNT,                0xFFFFB024,__READ       );
__IO_REG16_BIT(TM1CTRL,               0xFFFFB028,__READ_WRITE ,__tmctrl_bits);
__IO_REG16(    TM1IRQ,                0xFFFFB02C,__WRITE      );
__IO_REG16_BIT(TM1POUT,               0xFFFFB030,__READ_WRITE ,__tmpout_bits);
__IO_REG16(    TM2LD,                 0xFFFFB040,__READ_WRITE );
__IO_REG16(    TM2CNT,                0xFFFFB044,__READ       );
__IO_REG16_BIT(TM2CTRL,               0xFFFFB048,__READ_WRITE ,__tmctrl_bits);
__IO_REG16(    TM2IRQ,                0xFFFFB04C,__WRITE      );
__IO_REG16_BIT(TM2POUT,               0xFFFFB050,__READ_WRITE ,__tmpout_bits);
__IO_REG16_BIT(PS0CTRL,               0xFFFFB0A0,__READ_WRITE ,__psctrl_bits);
__IO_REG16_BIT(PS1CTRL,               0xFFFFB0A4,__READ_WRITE ,__psctrl_bits);
__IO_REG16_BIT(TMIRQSTS,              0xFFFFB0B0,__READ       ,__tmirqsts_bits);

/***************************************************************************
 **
 ** RTC
 **
 ***************************************************************************/
__IO_REG8_BIT( RTC_CR,                0xFFFF8000,__READ_WRITE ,__rtc_cr_bits);
__IO_REG8_BIT( RTC_IR,                0xFFFF8004,__READ_WRITE ,__rtc_ir_bits);
__IO_REG8_BIT( RTC_TDR,               0xFFFF8008,__READ_WRITE ,__rtc_tdr_bits);
__IO_REG8_BIT( RTC_SCR,               0xFFFF800C,__READ_WRITE ,__rtc_scr_bits);
__IO_REG8_BIT( RTC_MCR,               0xFFFF8010,__READ_WRITE ,__rtc_mcr_bits);
__IO_REG8_BIT( RTC_HCR,               0xFFFF8014,__READ_WRITE ,__rtc_hcr_bits);
__IO_REG16(    RTC_DCR,               0xFFFF8018,__READ_WRITE );
__IO_REG8_BIT( RTC_AMCR,              0xFFFF8020,__READ_WRITE ,__rtc_amcr_bits);
__IO_REG8_BIT( RTC_AHCR,              0xFFFF8024,__READ_WRITE ,__rtc_ahcr_bits);
__IO_REG16_BIT(RTC_ADCR,              0xFFFF8028,__READ_WRITE ,__rtc_adcr_bits);
__IO_REG8_BIT( RTC_TR,                0xFFFF802C,__READ_WRITE ,__rtc_tr_bits);
__IO_REG8_BIT( RTC_PR,                0xFFFF8030,__READ_WRITE ,__rtc_pr_bits);
__IO_REG8(     RTC_TCR,               0xFFFF8034,__WRITE      );

/***************************************************************************
 **
 ** WDT
 **
 ***************************************************************************/
__IO_REG16(    WDT_TLR,               0xFFFFC000,__READ_WRITE );
__IO_REG16(    WDT_TCNTR,             0xFFFFC004,__READ       );
__IO_REG16_BIT(WDT_TCR,               0xFFFFC008,__READ_WRITE ,__wdt_tcr_bits);

/***************************************************************************
 **
 ** GPIO
 **
 ***************************************************************************/
__IO_REG8_BIT( GPIOA_DATA,            0xFFFF1000,__READ_WRITE ,__gpioa_data_bits);
__IO_REG16_BIT(GPIOA_FNC,             0xFFFF1004,__READ_WRITE ,__gpioa_fnc_bits);
__IO_REG8_BIT( GPIOB_DATA,            0xFFFF1008,__READ_WRITE ,__gpiob_data_bits);
__IO_REG16_BIT(GPIOB_FNC,             0xFFFF100C,__READ_WRITE ,__gpiob_fnc_bits);
__IO_REG8_BIT( GPIOC_DATA,            0xFFFF1010,__READ_WRITE ,__gpioc_data_bits);
__IO_REG16_BIT(GPIOC_FNC,             0xFFFF1014,__READ_WRITE ,__gpioc_fnc_bits);
__IO_REG8_BIT( GPIOD_DATA,            0xFFFF1018,__READ_WRITE ,__gpiod_data_bits);
__IO_REG16_BIT(GPIOD_FNC,             0xFFFF101C,__READ_WRITE ,__gpiod_fnc_bits);
__IO_REG8_BIT( GPIOE_DATA,            0xFFFF1020,__READ_WRITE ,__gpioe_data_bits);
__IO_REG16_BIT(GPIOE_FNC,             0xFFFF1024,__READ_WRITE ,__gpioe_fnc_bits);
__IO_REG8_BIT( GPIOF_DATA,            0xFFFF1028,__READ_WRITE ,__gpiof_data_bits);
__IO_REG16_BIT(GPIOF_FNC,             0xFFFF102C,__READ_WRITE ,__gpiof_fnc_bits);
__IO_REG8_BIT( GPIOG_DATA,            0xFFFF1030,__READ_WRITE ,__gpiog_data_bits);
__IO_REG16_BIT(GPIOG_FNC,             0xFFFF1034,__READ_WRITE ,__gpiog_fnc_bits);
__IO_REG8_BIT( GPIOH_DATA,            0xFFFF1038,__READ_WRITE ,__gpioh_data_bits);
__IO_REG16_BIT(GPIOH_FNC,             0xFFFF103C,__READ_WRITE ,__gpioh_fnc_bits);
__IO_REG16_BIT(GPIOAB_ITYP,           0xFFFF1040,__READ_WRITE ,__gpioab_ityp_bits);
__IO_REG16_BIT(GPIOAB_IPOL,           0xFFFF1044,__READ_WRITE ,__gpioab_ipol_bits);
__IO_REG16_BIT(GPIOAB_IEN,            0xFFFF1048,__READ_WRITE ,__gpioab_ien_bits);
__IO_REG16_BIT(GPIOAB_ISTS,           0xFFFF104C,__READ_WRITE ,__gpioab_ists_bits);

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
 **  VIC Interrupt channels
 **
 ***************************************************************************/
#define INT_FIQ_WDT      0  /* Watchdog                           */
#define INT_FIQ_GPIOB0   1  /* GPIOB0 pin                         */

#define INT_IRQ_WDT      0  /* Watchdog                           */
#define INT_IRQ_SW       1  /* Software request from register     */
#define INT_IRQ_DEBUGRX  2  /* Embedded ICE, DbgCommRx            */
#define INT_IRQ_DEBUGTX  3  /* Embedded ICE, DbgCommTx            */
#define INT_IRQ_TIMER0   4  /* Timer 16 bit channel 0             */
#define INT_IRQ_TIMER1   5  /* Timer 16 bit channel 1             */
#define INT_IRQ_TIMER2   6  /* Timer 16 bit channel 2             */
#define INT_IRQ_ETH      7  /* Ethernet Mac & E-DMA               */
#define INT_IRQ_JPEG     8  /* JPEG control                       */
#define INT_IRQ_DMAC1    9  /* DMAC on AHB1 bus                   */
#define INT_IRQ_JDMA    10  /* JPEG DMAC                          */
#define INT_IRQ_CAM     11  /* Camera interface                   */
/*#define INT_IRQ_        12     Reserved                             */
#define INT_IRQ_DMAC2   13  /* DMAC on AHB2 bus                   */
#define INT_IRQ_GPIOA_B 14  /* External interrupt GPIOA/B[7:0]    */
#define INT_IRQ_SPI     15  /* SPI TXRDY/RXRDY                    */
#define INT_IRQ_I2C     16  /* Transfer Complete                  */
#define INT_IRQ_UART    17  /* UART TXRDY/RXRDY                   */
#define INT_IRQ_RTC     18  /* Alarm or Timer tick                */
#define INT_IRQ_CF      19  /* CF card interface                  */
#define INT_IRQ_INT0    20  /* GPIOB0 direct input                */
#define INT_IRQ_INT1    21  /* GPIOB1 direct input                */
#define INT_IRQ_INT2    22  /* GPIOB2 direct input                */
#define INT_IRQ_UARTL   23  /* UART Lite                          */
#define INT_IRQ_INT3    24  /* GPIOB3 direct input                */
#define INT_IRQ_INT4    25  /* GPIOB4 direct input                */
#define INT_IRQ_INT5    26  /* GPIOB5 direct input                */
#define INT_IRQ_INT6    27  /* GPIOB6 direct input                */
#define INT_IRQ_INT7    28  /* GPIOB7 direct input                */
#define INT_IRQ_INT8    29  /* GPIOD0 direct input                */
#define INT_IRQ_I2S0    30  /* I2S CH0                            */
#define INT_IRQ_I2S1    31  /* I2S CH1                            */

#endif    /* __IOS1S65010_H */
