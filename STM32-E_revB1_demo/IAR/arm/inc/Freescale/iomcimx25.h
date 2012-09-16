/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Freescale MCIX25
 **
 **    Used with ICCARM and AARM.
 **
 **    (c) Copyright IAR Systems 2003
 **
 **    $Revision: 50408 $
 **
 ***************************************************************************/

#ifndef __MCIX25_H
#define __MCIX25_H


#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ICCARM/AARM
#endif


#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    MCIX25 SPECIAL FUNCTION REGISTERS
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

/* SW_MUX_CTL Register Definition*/
typedef struct {
__REG32 MUX_MODE       : 3;
__REG32                : 1;
__REG32 SION           : 1;
__REG32                :27;
} __iomuxc_sw_mux_ctl_pad_bits;

/* SW_SELECT_INPUT Register Definition*/
typedef struct {
__REG32 Daisy          : 3;
__REG32                :29;
} __iomuxc_select_input_bits;

/* SW_PAD_CTL Register Definition*/
typedef struct {
__REG32 SRE            : 1;
__REG32 DSE            : 2;
__REG32 ODE            : 1;
__REG32 PUS            : 2;
__REG32 PUE            : 1;
__REG32 PKE            : 1;
__REG32 HYS            : 1;
__REG32                : 4;
__REG32 DVS            : 1;
__REG32                :18;
} __iomuxc_sw_pad_ctl_pad_bits;

/* SW_PAD_CTL Register Definition*/
typedef struct {
__REG32                : 7;
__REG32 PKE            : 1;
__REG32                :24;
} __iomuxc_sw_pad_ctl_pad_ddr_bits;

/* SW_PAD_CTL_GRP_DDRTYPE Register*/
typedef struct {
__REG32                :11;
__REG32 DDR_TYPE       : 2;
__REG32                :19;
} __iomuxc_sw_pad_ctl_grp_ddrtype_bits;

/* SW_PAD_CTL_GRP_DVS Register*/
typedef struct {
__REG32                :13;
__REG32 DVS            : 1;
__REG32                :18;
} __iomuxc_sw_pad_ctl_grp_dvs_bits;

/* SW_PAD_CTL_GRP_DSE Register*/
typedef struct {
__REG32                : 1;
__REG32 DSE            : 2;
__REG32                :29;
} __iomuxc_sw_pad_ctl_grp_dse_bits;

/* General Purpose Register*/
typedef struct {
__REG32 SDCTL_CSD0_SEL_B  : 1;
__REG32 SDCTL_CSD1_SEL_B  : 1;
__REG32                   :30;
} __iomuxc_gpr1_bits;

/* General Purpose Register*/
typedef struct {
__REG32 OBSRV             : 7;
__REG32                   :25;
} __iomuxc_observe_int_mux_bits;


/* -------------------------------------------------------------------------*/
/*      One Wire                                                            */
/* -------------------------------------------------------------------------*/
/* O-Wire Control Register */
typedef struct{
__REG16       : 3;
__REG16 RDST  : 1;
__REG16 WR1   : 1;
__REG16 WR0   : 1;
__REG16 PST   : 1;
__REG16 RPP   : 1;
__REG16       : 8;
} __ow_control_bits;

/* O-Wire Time Divider Register */
typedef struct{
__REG16 DVDR  : 8;
__REG16       : 8;
} __ow_time_divider_bits;


/* O-Wire Reset Register */
typedef struct{
__REG16 RST  : 1;
__REG16      :15;
} __ow_reset_bits;

/* O-Wire Command Register */
typedef struct{
__REG16      : 1;
__REG16 SRA  : 1;
__REG16      :14;
} __ow_command_bits;

/* O-Wire Transmit/Receive Register */
typedef struct{
__REG16 DATA : 8;
__REG16      : 8;
} __ow_rx_tx_bits;

/* O-Wire Interrupt Register */
typedef struct{
__REG16 PD   : 1;
__REG16 PDR  : 1;
__REG16 TBE  : 1;
__REG16 TSRE : 1;
__REG16 RBF  : 1;
__REG16 RSRF : 1;
__REG16      :10;
} __ow_interrupt_bits;

/* O-Wire Interrupt Enable Register */
typedef struct{
__REG16 EPD  : 1;
__REG16 IAS  : 1;
__REG16 ETBE : 1;
__REG16 ETSE : 1;
__REG16 ERBF : 1;
__REG16 ERSF : 1;
__REG16      :10;
} __ow_interrupt_en_bits;

/* -------------------------------------------------------------------------*/
/*               AITC registers                                             */
/* -------------------------------------------------------------------------*/
/* Interrupt Control Register */
typedef struct {
__REG32          :21;
__REG32 FIDIS    : 1;
__REG32 NIDIS    : 1;
__REG32 BYP_EN   : 1;
__REG32          : 8;
} __intcntl_bits;

/* Normal Interrupt Mask Register */
typedef struct {
__REG32 NIMASK  : 5;
__REG32         :27;
} __nimask_bits;

/* Interrupt Enable Number Register*/
typedef struct {
__REG32 ENNUM  : 6;
__REG32        :26;
} __intennum_bits;

/* Interrupt Disable Number Register */
typedef struct {
__REG32 DISNUM  : 6;
__REG32         :26;
} __intdisnum_bits;

/* Interrupt Enable Register High */
typedef struct {
__REG32 INTENABLE32  : 1;
__REG32 INTENABLE33  : 1;
__REG32 INTENABLE34  : 1;
__REG32 INTENABLE35  : 1;
__REG32 INTENABLE36  : 1;
__REG32 INTENABLE37  : 1;
__REG32 INTENABLE38  : 1;
__REG32 INTENABLE39  : 1;
__REG32 INTENABLE40  : 1;
__REG32 INTENABLE41  : 1;
__REG32 INTENABLE42  : 1;
__REG32 INTENABLE43  : 1;
__REG32 INTENABLE44  : 1;
__REG32 INTENABLE45  : 1;
__REG32 INTENABLE46  : 1;
__REG32 INTENABLE47  : 1;
__REG32 INTENABLE48  : 1;
__REG32 INTENABLE49  : 1;
__REG32 INTENABLE50  : 1;
__REG32 INTENABLE51  : 1;
__REG32 INTENABLE52  : 1;
__REG32 INTENABLE53  : 1;
__REG32 INTENABLE54  : 1;
__REG32 INTENABLE55  : 1;
__REG32 INTENABLE56  : 1;
__REG32 INTENABLE57  : 1;
__REG32 INTENABLE58  : 1;
__REG32 INTENABLE59  : 1;
__REG32 INTENABLE60  : 1;
__REG32 INTENABLE61  : 1;
__REG32 INTENABLE62  : 1;
__REG32 INTENABLE63  : 1;
} __intenableh_bits;

/* Interrupt Enable Register Low */
typedef struct {
__REG32 INTENABLE0   : 1;
__REG32 INTENABLE1   : 1;
__REG32 INTENABLE2   : 1;
__REG32 INTENABLE3   : 1;
__REG32 INTENABLE4   : 1;
__REG32 INTENABLE5   : 1;
__REG32 INTENABLE6   : 1;
__REG32 INTENABLE7   : 1;
__REG32 INTENABLE8   : 1;
__REG32 INTENABLE9   : 1;
__REG32 INTENABLE10  : 1;
__REG32 INTENABLE11  : 1;
__REG32 INTENABLE12  : 1;
__REG32 INTENABLE13  : 1;
__REG32 INTENABLE14  : 1;
__REG32 INTENABLE15  : 1;
__REG32 INTENABLE16  : 1;
__REG32 INTENABLE17  : 1;
__REG32 INTENABLE18  : 1;
__REG32 INTENABLE19  : 1;
__REG32 INTENABLE20  : 1;
__REG32 INTENABLE21  : 1;
__REG32 INTENABLE22  : 1;
__REG32 INTENABLE23  : 1;
__REG32 INTENABLE24  : 1;
__REG32 INTENABLE25  : 1;
__REG32 INTENABLE26  : 1;
__REG32 INTENABLE27  : 1;
__REG32 INTENABLE28  : 1;
__REG32 INTENABLE29  : 1;
__REG32 INTENABLE30  : 1;
__REG32 INTENABLE31  : 1;
} __intenablel_bits;

/* Interrupt Type Register High */
typedef struct {
__REG32 INTTYPE32  : 1;
__REG32 INTTYPE33  : 1;
__REG32 INTTYPE34  : 1;
__REG32 INTTYPE35  : 1;
__REG32 INTTYPE36  : 1;
__REG32 INTTYPE37  : 1;
__REG32 INTTYPE38  : 1;
__REG32 INTTYPE39  : 1;
__REG32 INTTYPE40  : 1;
__REG32 INTTYPE41  : 1;
__REG32 INTTYPE42  : 1;
__REG32 INTTYPE43  : 1;
__REG32 INTTYPE44  : 1;
__REG32 INTTYPE45  : 1;
__REG32 INTTYPE46  : 1;
__REG32 INTTYPE47  : 1;
__REG32 INTTYPE48  : 1;
__REG32 INTTYPE49  : 1;
__REG32 INTTYPE50  : 1;
__REG32 INTTYPE51  : 1;
__REG32 INTTYPE52  : 1;
__REG32 INTTYPE53  : 1;
__REG32 INTTYPE54  : 1;
__REG32 INTTYPE55  : 1;
__REG32 INTTYPE56  : 1;
__REG32 INTTYPE57  : 1;
__REG32 INTTYPE58  : 1;
__REG32 INTTYPE59  : 1;
__REG32 INTTYPE60  : 1;
__REG32 INTTYPE61  : 1;
__REG32 INTTYPE62  : 1;
__REG32 INTTYPE63  : 1;
} __inttypeh_bits;

/* Interrupt Enable Register Low */
typedef struct {
__REG32 INTTYPE0   : 1;
__REG32 INTTYPE1   : 1;
__REG32 INTTYPE2   : 1;
__REG32 INTTYPE3   : 1;
__REG32 INTTYPE4   : 1;
__REG32 INTTYPE5   : 1;
__REG32 INTTYPE6   : 1;
__REG32 INTTYPE7   : 1;
__REG32 INTTYPE8   : 1;
__REG32 INTTYPE9   : 1;
__REG32 INTTYPE10  : 1;
__REG32 INTTYPE11  : 1;
__REG32 INTTYPE12  : 1;
__REG32 INTTYPE13  : 1;
__REG32 INTTYPE14  : 1;
__REG32 INTTYPE15  : 1;
__REG32 INTTYPE16  : 1;
__REG32 INTTYPE17  : 1;
__REG32 INTTYPE18  : 1;
__REG32 INTTYPE19  : 1;
__REG32 INTTYPE20  : 1;
__REG32 INTTYPE21  : 1;
__REG32 INTTYPE22  : 1;
__REG32 INTTYPE23  : 1;
__REG32 INTTYPE24  : 1;
__REG32 INTTYPE25  : 1;
__REG32 INTTYPE26  : 1;
__REG32 INTTYPE27  : 1;
__REG32 INTTYPE28  : 1;
__REG32 INTTYPE29  : 1;
__REG32 INTTYPE30  : 1;
__REG32 INTTYPE31  : 1;
} __inttypel_bits;

/* Normal Interrupt Priority Level Register 7 */
typedef struct {
__REG32 NIPR56  : 4;
__REG32 NIPR57  : 4;
__REG32 NIPR58  : 4;
__REG32 NIPR59  : 4;
__REG32 NIPR60  : 4;
__REG32 NIPR61  : 4;
__REG32 NIPR62  : 4;
__REG32 NIPR63  : 4;
} __nipriority7_bits;

/* Normal Interrupt Priority Level Register 6 */
typedef struct {
__REG32 NIPR48  : 4;
__REG32 NIPR49  : 4;
__REG32 NIPR50  : 4;
__REG32 NIPR51  : 4;
__REG32 NIPR52  : 4;
__REG32 NIPR53  : 4;
__REG32 NIPR54  : 4;
__REG32 NIPR55  : 4;
} __nipriority6_bits;

/* Normal Interrupt Priority Level Register 5 */
typedef struct {
__REG32 NIPR40  : 4;
__REG32 NIPR41  : 4;
__REG32 NIPR42  : 4;
__REG32 NIPR43  : 4;
__REG32 NIPR44  : 4;
__REG32 NIPR45  : 4;
__REG32 NIPR46  : 4;
__REG32 NIPR47  : 4;
} __nipriority5_bits;

/* Normal Interrupt Priority Level Register 4 */
typedef struct {
__REG32 NIPR32  : 4;
__REG32 NIPR33  : 4;
__REG32 NIPR34  : 4;
__REG32 NIPR35  : 4;
__REG32 NIPR36  : 4;
__REG32 NIPR37  : 4;
__REG32 NIPR38  : 4;
__REG32 NIPR39  : 4;
} __nipriority4_bits;

/* Normal Interrupt Priority Level Register 3 */
typedef struct {
__REG32 NIPR24  : 4;
__REG32 NIPR25  : 4;
__REG32 NIPR26  : 4;
__REG32 NIPR27  : 4;
__REG32 NIPR28  : 4;
__REG32 NIPR29  : 4;
__REG32 NIPR30  : 4;
__REG32 NIPR31  : 4;
} __nipriority3_bits;

/* Normal Interrupt Priority Level Register 2 */
typedef struct {
__REG32 NIPR16  : 4;
__REG32 NIPR17  : 4;
__REG32 NIPR18  : 4;
__REG32 NIPR19  : 4;
__REG32 NIPR20  : 4;
__REG32 NIPR21  : 4;
__REG32 NIPR22  : 4;
__REG32 NIPR23  : 4;
} __nipriority2_bits;

/* Normal Interrupt Priority Level Register 1 */
typedef struct {
__REG32 NIPR8   : 4;
__REG32 NIPR9   : 4;
__REG32 NIPR10  : 4;
__REG32 NIPR11  : 4;
__REG32 NIPR12  : 4;
__REG32 NIPR13  : 4;
__REG32 NIPR14  : 4;
__REG32 NIPR15  : 4;
} __nipriority1_bits;

/* Normal Interrupt Priority Level Register 0 */
typedef struct {
__REG32 NIPR0  : 4;
__REG32 NIPR1  : 4;
__REG32 NIPR2  : 4;
__REG32 NIPR3  : 4;
__REG32 NIPR4  : 4;
__REG32 NIPR5  : 4;
__REG32 NIPR6  : 4;
__REG32 NIPR7  : 4;
} __nipriority0_bits;

/* Normal Interrupt Vector and Status Register*/
typedef struct {
__REG32 NIPRILVL  :16;
__REG32 NIVECTOR  :16;
} __nivecsr_bits;

/* Interrupt Source Register High */
typedef struct {
__REG32 INTIN32  : 1;
__REG32 INTIN33  : 1;
__REG32 INTIN34  : 1;
__REG32 INTIN35  : 1;
__REG32 INTIN36  : 1;
__REG32 INTIN37  : 1;
__REG32 INTIN38  : 1;
__REG32 INTIN39  : 1;
__REG32 INTIN40  : 1;
__REG32 INTIN41  : 1;
__REG32 INTIN42  : 1;
__REG32 INTIN43  : 1;
__REG32 INTIN44  : 1;
__REG32 INTIN45  : 1;
__REG32 INTIN46  : 1;
__REG32 INTIN47  : 1;
__REG32 INTIN48  : 1;
__REG32 INTIN49  : 1;
__REG32 INTIN50  : 1;
__REG32 INTIN51  : 1;
__REG32 INTIN52  : 1;
__REG32 INTIN53  : 1;
__REG32 INTIN54  : 1;
__REG32 INTIN55  : 1;
__REG32 INTIN56  : 1;
__REG32 INTIN57  : 1;
__REG32 INTIN58  : 1;
__REG32 INTIN59  : 1;
__REG32 INTIN60  : 1;
__REG32 INTIN61  : 1;
__REG32 INTIN62  : 1;
__REG32 INTIN63  : 1;
} __intsrch_bits;

/* Interrupt Source Register Low */
typedef struct {
__REG32 INTIN0   : 1;
__REG32 INTIN1   : 1;
__REG32 INTIN2   : 1;
__REG32 INTIN3   : 1;
__REG32 INTIN4   : 1;
__REG32 INTIN5   : 1;
__REG32 INTIN6   : 1;
__REG32 INTIN7   : 1;
__REG32 INTIN8   : 1;
__REG32 INTIN9   : 1;
__REG32 INTIN10  : 1;
__REG32 INTIN11  : 1;
__REG32 INTIN12  : 1;
__REG32 INTIN13  : 1;
__REG32 INTIN14  : 1;
__REG32 INTIN15  : 1;
__REG32 INTIN16  : 1;
__REG32 INTIN17  : 1;
__REG32 INTIN18  : 1;
__REG32 INTIN19  : 1;
__REG32 INTIN20  : 1;
__REG32 INTIN21  : 1;
__REG32 INTIN22  : 1;
__REG32 INTIN23  : 1;
__REG32 INTIN24  : 1;
__REG32 INTIN25  : 1;
__REG32 INTIN26  : 1;
__REG32 INTIN27  : 1;
__REG32 INTIN28  : 1;
__REG32 INTIN29  : 1;
__REG32 INTIN30  : 1;
__REG32 INTIN31  : 1;
} __intsrcl_bits;

/* Interrupt Force Register High */
typedef struct {
__REG32 FORCE32  : 1;
__REG32 FORCE33  : 1;
__REG32 FORCE34  : 1;
__REG32 FORCE35  : 1;
__REG32 FORCE36  : 1;
__REG32 FORCE37  : 1;
__REG32 FORCE38  : 1;
__REG32 FORCE39  : 1;
__REG32 FORCE40  : 1;
__REG32 FORCE41  : 1;
__REG32 FORCE42  : 1;
__REG32 FORCE43  : 1;
__REG32 FORCE44  : 1;
__REG32 FORCE45  : 1;
__REG32 FORCE46  : 1;
__REG32 FORCE47  : 1;
__REG32 FORCE48  : 1;
__REG32 FORCE49  : 1;
__REG32 FORCE50  : 1;
__REG32 FORCE51  : 1;
__REG32 FORCE52  : 1;
__REG32 FORCE53  : 1;
__REG32 FORCE54  : 1;
__REG32 FORCE55  : 1;
__REG32 FORCE56  : 1;
__REG32 FORCE57  : 1;
__REG32 FORCE58  : 1;
__REG32 FORCE59  : 1;
__REG32 FORCE60  : 1;
__REG32 FORCE61  : 1;
__REG32 FORCE62  : 1;
__REG32 FORCE63  : 1;
} __intfrch_bits;

/* Interrupt Force Register Low */
typedef struct {
__REG32 FORCE0   : 1;
__REG32 FORCE1   : 1;
__REG32 FORCE2   : 1;
__REG32 FORCE3   : 1;
__REG32 FORCE4   : 1;
__REG32 FORCE5   : 1;
__REG32 FORCE6   : 1;
__REG32 FORCE7   : 1;
__REG32 FORCE8   : 1;
__REG32 FORCE9   : 1;
__REG32 FORCE10  : 1;
__REG32 FORCE11  : 1;
__REG32 FORCE12  : 1;
__REG32 FORCE13  : 1;
__REG32 FORCE14  : 1;
__REG32 FORCE15  : 1;
__REG32 FORCE16  : 1;
__REG32 FORCE17  : 1;
__REG32 FORCE18  : 1;
__REG32 FORCE19  : 1;
__REG32 FORCE20  : 1;
__REG32 FORCE21  : 1;
__REG32 FORCE22  : 1;
__REG32 FORCE23  : 1;
__REG32 FORCE24  : 1;
__REG32 FORCE25  : 1;
__REG32 FORCE26  : 1;
__REG32 FORCE27  : 1;
__REG32 FORCE28  : 1;
__REG32 FORCE29  : 1;
__REG32 FORCE30  : 1;
__REG32 FORCE31  : 1;
} __intfrcl_bits;

/* Normal Interrupt Pending Register High */
typedef struct {
__REG32 NIPEND32  : 1;
__REG32 NIPEND33  : 1;
__REG32 NIPEND34  : 1;
__REG32 NIPEND35  : 1;
__REG32 NIPEND36  : 1;
__REG32 NIPEND37  : 1;
__REG32 NIPEND38  : 1;
__REG32 NIPEND39  : 1;
__REG32 NIPEND40  : 1;
__REG32 NIPEND41  : 1;
__REG32 NIPEND42  : 1;
__REG32 NIPEND43  : 1;
__REG32 NIPEND44  : 1;
__REG32 NIPEND45  : 1;
__REG32 NIPEND46  : 1;
__REG32 NIPEND47  : 1;
__REG32 NIPEND48  : 1;
__REG32 NIPEND49  : 1;
__REG32 NIPEND50  : 1;
__REG32 NIPEND51  : 1;
__REG32 NIPEND52  : 1;
__REG32 NIPEND53  : 1;
__REG32 NIPEND54  : 1;
__REG32 NIPEND55  : 1;
__REG32 NIPEND56  : 1;
__REG32 NIPEND57  : 1;
__REG32 NIPEND58  : 1;
__REG32 NIPEND59  : 1;
__REG32 NIPEND60  : 1;
__REG32 NIPEND61  : 1;
__REG32 NIPEND62  : 1;
__REG32 NIPEND63  : 1;
} __nipndh_bits;

/* Normal Interrupt Pending Register Low */
typedef struct {
__REG32 NIPEND0   : 1;
__REG32 NIPEND1   : 1;
__REG32 NIPEND2   : 1;
__REG32 NIPEND3   : 1;
__REG32 NIPEND4   : 1;
__REG32 NIPEND5   : 1;
__REG32 NIPEND6   : 1;
__REG32 NIPEND7   : 1;
__REG32 NIPEND8   : 1;
__REG32 NIPEND9   : 1;
__REG32 NIPEND10  : 1;
__REG32 NIPEND11  : 1;
__REG32 NIPEND12  : 1;
__REG32 NIPEND13  : 1;
__REG32 NIPEND14  : 1;
__REG32 NIPEND15  : 1;
__REG32 NIPEND16  : 1;
__REG32 NIPEND17  : 1;
__REG32 NIPEND18  : 1;
__REG32 NIPEND19  : 1;
__REG32 NIPEND20  : 1;
__REG32 NIPEND21  : 1;
__REG32 NIPEND22  : 1;
__REG32 NIPEND23  : 1;
__REG32 NIPEND24  : 1;
__REG32 NIPEND25  : 1;
__REG32 NIPEND26  : 1;
__REG32 NIPEND27  : 1;
__REG32 NIPEND28  : 1;
__REG32 NIPEND29  : 1;
__REG32 NIPEND30  : 1;
__REG32 NIPEND31  : 1;
} __nipndl_bits;

/* Fast Interrupt Pending Register High */
typedef struct {
__REG32 FIPEND32  : 1;
__REG32 FIPEND33  : 1;
__REG32 FIPEND34  : 1;
__REG32 FIPEND35  : 1;
__REG32 FIPEND36  : 1;
__REG32 FIPEND37  : 1;
__REG32 FIPEND38  : 1;
__REG32 FIPEND39  : 1;
__REG32 FIPEND40  : 1;
__REG32 FIPEND41  : 1;
__REG32 FIPEND42  : 1;
__REG32 FIPEND43  : 1;
__REG32 FIPEND44  : 1;
__REG32 FIPEND45  : 1;
__REG32 FIPEND46  : 1;
__REG32 FIPEND47  : 1;
__REG32 FIPEND48  : 1;
__REG32 FIPEND49  : 1;
__REG32 FIPEND50  : 1;
__REG32 FIPEND51  : 1;
__REG32 FIPEND52  : 1;
__REG32 FIPEND53  : 1;
__REG32 FIPEND54  : 1;
__REG32 FIPEND55  : 1;
__REG32 FIPEND56  : 1;
__REG32 FIPEND57  : 1;
__REG32 FIPEND58  : 1;
__REG32 FIPEND59  : 1;
__REG32 FIPEND60  : 1;
__REG32 FIPEND61  : 1;
__REG32 FIPEND62  : 1;
__REG32 FIPEND63  : 1;
} __fipndh_bits;

/* Fast Interrupt Pending Register Low */
typedef struct {
__REG32 FIPEND0   : 1;
__REG32 FIPEND1   : 1;
__REG32 FIPEND2   : 1;
__REG32 FIPEND3   : 1;
__REG32 FIPEND4   : 1;
__REG32 FIPEND5   : 1;
__REG32 FIPEND6   : 1;
__REG32 FIPEND7   : 1;
__REG32 FIPEND8   : 1;
__REG32 FIPEND9   : 1;
__REG32 FIPEND10  : 1;
__REG32 FIPEND11  : 1;
__REG32 FIPEND12  : 1;
__REG32 FIPEND13  : 1;
__REG32 FIPEND14  : 1;
__REG32 FIPEND15  : 1;
__REG32 FIPEND16  : 1;
__REG32 FIPEND17  : 1;
__REG32 FIPEND18  : 1;
__REG32 FIPEND19  : 1;
__REG32 FIPEND20  : 1;
__REG32 FIPEND21  : 1;
__REG32 FIPEND22  : 1;
__REG32 FIPEND23  : 1;
__REG32 FIPEND24  : 1;
__REG32 FIPEND25  : 1;
__REG32 FIPEND26  : 1;
__REG32 FIPEND27  : 1;
__REG32 FIPEND28  : 1;
__REG32 FIPEND29  : 1;
__REG32 FIPEND30  : 1;
__REG32 FIPEND31  : 1;
} __fipndl_bits;

/* -------------------------------------------------------------------------*/
/*               ATA registers                                             */
/* -------------------------------------------------------------------------*/
/* ATA Control Register */
typedef struct {
__REG16 iordy_en            : 1;
__REG16 dma_write           : 1;
__REG16 dma_ultra_selected  : 1;
__REG16 dma_pending         : 1;
__REG16 fifo_rcv_en         : 1;
__REG16 fifo_tx_en          : 1;
__REG16 ata_rst_b           : 1;
__REG16 fifo_rst_b          : 1;
__REG16 dma_enable          : 1;
__REG16 dma_start_stop      : 1;
__REG16 dma_select          : 2;
__REG16 dma_srst            : 1;
__REG16                     : 3;
} __ata_control_bits;

/* Interrupt Pending Register */
typedef struct {
__REG8                      : 1;
__REG8  dma_trans_over      : 1;
__REG8  dma_err             : 1;
__REG8  ata_intrq2          : 1;
__REG8  controller_idle     : 1;
__REG8  fifo_overflow       : 1;
__REG8  fifo_underflow      : 1;
__REG8  ata_intrq1          : 1;
} __ata_interrupt_pending_bits;

/* Interrupt Clear Register */
typedef struct {
__REG8                      : 1;
__REG8  dma_trans_over      : 1;
__REG8  dma_err             : 1;
__REG8                      : 2;
__REG8  fifo_overflow       : 1;
__REG8  fifo_underflow      : 1;
__REG8                      : 1;
} __ata_interrupt_clear_bits;

/* ADMA_ERR_STATUS Register */
typedef struct {
__REG8  adma_err_state      : 2;
__REG8  adma_len_mismatch   : 1;
__REG8                      : 5;
} __ata_adma_err_status_bits;

/* BURST_LENGTH Register */
typedef struct {
__REG8  burst_length        : 6;
__REG8                      : 2;
} __ata_burst_length_bits;

/* -------------------------------------------------------------------------*/
/*      Digital Audio Mux (AUDMUX)                                          */
/* -------------------------------------------------------------------------*/
/* AUDMUX Port Timing Control Register */
typedef struct{
__REG32          :11;
__REG32 SYN      : 1;
__REG32 RCSEL    : 4;
__REG32 RCLKDIR  : 1;
__REG32 RFSEL    : 4;
__REG32 RFSDIR   : 1;
__REG32 TCSEL    : 4;
__REG32 TCLKDIR  : 1;
__REG32 TFSEL    : 4;
__REG32 TFSDIR   : 1;
} __ptcr_bits;

/* AUDMUX Port Data Control Register */
typedef struct{
__REG32 INMMASK1 : 8;
__REG32 MODE1    : 2;
__REG32          : 2;
__REG32 TXRXEN1  : 1;
__REG32 RXDSEL1  : 3;
__REG32          :16;
} __pdcr_bits;

/* AUDMUX CE Bus Network Mode Control Register */
typedef struct{
__REG32 CNTLOW   : 8;
__REG32 CNTHI    : 8;
__REG32 CLKPOL   : 1;
__REG32 FSPOL    : 1;
__REG32 CEN      : 1;
__REG32          :13;
} __cnmcr_bits;

/* -------------------------------------------------------------------------*/
/*     Clock Controller And Reset Module                                    */
/* -------------------------------------------------------------------------*/
/* MCU PLL Control Register - MPCTL
   USB PLL Control Register - UPCTL */
typedef struct {
__REG32 MFN   :10;
__REG32 MFI   : 4;
__REG32       : 1;
__REG32 LOCK  : 1;
__REG32 MFD   :10;
__REG32 PD    : 4;
__REG32       : 1;
__REG32 BRMO  : 1;
} __mpctl_bits;

/* Clock Control Register - CCTL */
typedef struct {
__REG32             :14;
__REG32 ARMSRC      : 1;
__REG32 CGCTL       : 1;
__REG32 USBDIV      : 6;
__REG32 MPLLBYPASS  : 1;
__REG32 UPLLDIS     : 1;
__REG32 LPCTL       : 2;
__REG32 UPLLRST     : 1;
__REG32 MPLLRST     : 1;
__REG32 AHBCLKDIV   : 2;
__REG32 ARMCLKDIV   : 2;
} __cctl_bits;

/* Clock Gating Control Register 0 - CGCR0 */
typedef struct {
__REG32 ipg_per_csi     : 1;
__REG32 ipg_per_epit    : 1;
__REG32 ipg_per_esai    : 1;
__REG32 ipg_per_esdhc1  : 1;
__REG32 ipg_per_esdhc2  : 1;
__REG32 ipg_per_gpt     : 1;
__REG32 ipg_per_i2c     : 1;
__REG32 ipg_per_lcdc    : 1;
__REG32 ipg_per_nfc     : 1;
__REG32 ipg_per_owire   : 1;
__REG32 ipg_per_pwm     : 1;
__REG32 ipg_per_sim1    : 1;
__REG32 ipg_per_sim2    : 1;
__REG32 ipg_per_ssi1    : 1;
__REG32 ipg_per_ssi2    : 1;
__REG32 ipg_per_uart    : 1;
__REG32 hclk_ata        : 1;
__REG32 hclk_brom       : 1;
__REG32 hclk_csi        : 1;
__REG32 hclk_emi        : 1;
__REG32 hclk_esai       : 1;
__REG32 hclk_esdhc1     : 1;
__REG32 hclk_esdhc2     : 1;
__REG32 hclk_fec        : 1;
__REG32 hclk_lcdc       : 1;
__REG32 hclk_rtic       : 1;
__REG32 hclk_sdma       : 1;
__REG32 hclk_slcdc      : 1;
__REG32 hclk_usbotg     : 1;
__REG32                 : 3;
} __cgcr0_bits;

/* Clock Gating Control Register 1 - CGCR1 */
typedef struct {
__REG32 ipg_clk_audmux  : 1;
__REG32 ipg_clk_ata     : 1;
__REG32 ipg_clk_can1    : 1;
__REG32 ipg_clk_can2    : 1;
__REG32 ipg_clk_csi     : 1;
__REG32 ipg_clk_cspi1   : 1;
__REG32 ipg_clk_cspi2   : 1;
__REG32 ipg_clk_cspi3   : 1;
__REG32 ipg_clk_dryice  : 1;
__REG32 ipg_clk_ect     : 1;
__REG32 ipg_clk_epit1   : 1;
__REG32 ipg_clk_epit2   : 1;
__REG32 ipg_clk_esai    : 1;
__REG32 ipg_clk_esdhc1  : 1;
__REG32 ipg_clk_esdhc2  : 1;
__REG32 ipg_clk_fec     : 1;
__REG32 ipg_clk_gpio1   : 1;
__REG32 ipg_clk_gpio2   : 1;
__REG32 ipg_clk_gpio3   : 1;
__REG32 ipg_clk_gpt1    : 1;
__REG32 ipg_clk_gpt2    : 1;
__REG32 ipg_clk_gpt3    : 1;
__REG32 ipg_clk_gpt4    : 1;
__REG32 ipg_clk_i2c1    : 1;
__REG32 ipg_clk_i2c2    : 1;
__REG32 ipg_clk_i2c3    : 1;
__REG32 ipg_clk_iim     : 1;
__REG32 ipg_clk_iomuxc  : 1;
__REG32 ipg_clk_kpp     : 1;
__REG32 ipg_clk_lcdc    : 1;
__REG32 ipg_clk_owire   : 1;
__REG32 ipg_clk_pwm1    : 1;
} __cgcr1_bits;

/* Clock Gating Control Register 2 - CGCR2 */
typedef struct {
__REG32 ipg_clk_pwm2    : 1;
__REG32 ipg_clk_pwm3    : 1;
__REG32 ipg_clk_pwm4    : 1;
__REG32 ipg_clk_rngb    : 1;
__REG32 ipg_clk_rtic    : 1;
__REG32 ipg_clk_scc     : 1;
__REG32 ipg_clk_sdma    : 1;
__REG32 ipg_clk_sim1    : 1;
__REG32 ipg_clk_sim2    : 1;
__REG32 ipg_clk_slcdc   : 1;
__REG32 ipg_clk_spba    : 1;
__REG32 ipg_clk_ssi1    : 1;
__REG32 ipg_clk_ssi2    : 1;
__REG32 ipg_clk_tchscrn : 1;
__REG32 ipg_clk_uart1   : 1;
__REG32 ipg_clk_uart2   : 1;
__REG32 ipg_clk_uart3   : 1;
__REG32 ipg_clk_uart4   : 1;
__REG32 ipg_clk_uart5   : 1;
__REG32 ipg_clk_wdog    : 1;
__REG32                 :12;
} __cgcr2_bits;

/* Per Clock Divider Register 0 - PCDR0 */
typedef struct {
__REG32 PERDIV0    : 6;
__REG32            : 2;
__REG32 PERDIV1    : 6;
__REG32            : 2;
__REG32 PERDIV2    : 6;
__REG32            : 2;
__REG32 PERDIV3    : 6;
__REG32            : 2;
} __pcdr0_bits;

/* Per Clock Divider Register 1 - PCDR1 */
typedef struct {
__REG32 PERDIV4    : 6;
__REG32            : 2;
__REG32 PERDIV5    : 6;
__REG32            : 2;
__REG32 PERDIV6    : 6;
__REG32            : 2;
__REG32 PERDIV7    : 6;
__REG32            : 2;
} __pcdr1_bits;

/* Per Clock Divider Register 2 - PCDR2 */
typedef struct {
__REG32 PERDIV8    : 6;
__REG32            : 2;
__REG32 PERDIV9    : 6;
__REG32            : 2;
__REG32 PERDIV10   : 6;
__REG32            : 2;
__REG32 PERDIV11   : 6;
__REG32            : 2;
} __pcdr2_bits;

/* Per Clock Divider Register 3 - PCDR3 */
typedef struct {
__REG32 PERDIV12   : 6;
__REG32            : 2;
__REG32 PERDIV13   : 6;
__REG32            : 2;
__REG32 PERDIV14   : 6;
__REG32            : 2;
__REG32 PERDIV15   : 6;
__REG32            : 2;
} __pcdr3_bits;

/* CRM Status Register - RCSR */
typedef struct {
__REG32 RESTS           : 4;
__REG32 MLC_SEL         : 1;
__REG32 EEPROM_CFG      : 1;
__REG32 BOOT_INT        : 1;
__REG32 SPARE_SIZE      : 1;
__REG32 NFC_FMS         : 1;
__REG32 NFC_4K          : 1;
__REG32 BOOT_REG        : 2;
__REG32 CLK_SEL         : 1;
__REG32                 : 1;
__REG32 NFC_16bit_SEL   : 1;
__REG32 SOFT_RESET      : 1;
__REG32 BT_RES          : 4;
__REG32 BT_SRC          : 2;
__REG32 USB_SRC         : 2;
__REG32 BUS_WIDTH       : 2;
__REG32 PAGE_SIZE       : 2;
__REG32 MEM_TYPE        : 2;
__REG32 MEM_CTRL        : 2;
} __rcsr_bits;

/* CRM Debug Register - CRDR */
typedef struct {
__REG32             :26;
__REG32 BT_LPB_FREQ : 3;
__REG32 BT_UART_SRC : 3;
} __crdr_bits;

/* DPTC Comparator Value Registers - DCVR0-DCVR3 */
typedef struct {
__REG32             : 2;
__REG32 ELV         :10;
__REG32 LLV         :10;
__REG32 ULV         :10;
} __dcvr_bits;

/* Load Tracking Register 0- LTR0 */
typedef struct {
__REG32 SIGD0       : 1;
__REG32 SIGD1       : 1;
__REG32 SIGD2       : 1;
__REG32 SIGD3       : 1;
__REG32 SIGD4       : 1;
__REG32 SIGD5       : 1;
__REG32 SIGD6       : 1;
__REG32 SIGD7       : 1;
__REG32 SIGD8       : 1;
__REG32 SIGD9       : 1;
__REG32 SIGD10      : 1;
__REG32 SIGD11      : 1;
__REG32 SIGD12      : 1;
__REG32 SIGD13      : 1;
__REG32 SIGD14      : 1;
__REG32 SIGD15      : 1;
__REG32 DNTHR       : 6;
__REG32 UPTHR       : 6;
__REG32 DIV3CK      : 2;
__REG32             : 2;
} __ltr0_bits;

/* Load Tracking Register 1- LTR1 */
typedef struct {
__REG32 PNCTHR      : 6;
__REG32 UPCNT       : 8;
__REG32 DNCNT       : 8;
__REG32 LTBRSR      : 1;
__REG32 LTBRSH      : 1;
__REG32             : 8;
} __ltr1_bits;

/* Load Tracking Register 2 - LTR2 */
typedef struct {
__REG32 EMAC        : 9;
__REG32             : 2;
__REG32 WSW9        : 3;
__REG32 WSW10       : 3;
__REG32 WSW11       : 3;
__REG32 WSW12       : 3;
__REG32 WSW13       : 3;
__REG32 WSW14       : 3;
__REG32 WSW15       : 3;
} __ltr2_bits;

/* Load Tracking Register 3 - LTR3 */
typedef struct {
__REG32             : 5;
__REG32 WSW0        : 3;
__REG32 WSW1        : 3;
__REG32 WSW2        : 3;
__REG32 WSW3        : 3;
__REG32 WSW4        : 3;
__REG32 WSW5        : 3;
__REG32 WSW6        : 3;
__REG32 WSW7        : 3;
__REG32 WSW8        : 3;
} __ltr3_bits;

/* Load Tracking Buffer Register 0 - LTBR0 */
typedef struct {
__REG32 LTS0        : 4;
__REG32 LTS1        : 4;
__REG32 LTS2        : 4;
__REG32 LTS3        : 4;
__REG32 LTS4        : 4;
__REG32 LTS5        : 4;
__REG32 LTS6        : 4;
__REG32 LTS7        : 4;
} __ltbr0_bits;

/* Load Tracking Buffer Register 1 - LTBR1 */
typedef struct {
__REG32 LTS8        : 4;
__REG32 LTS9        : 4;
__REG32 LTS10       : 4;
__REG32 LTS11       : 4;
__REG32 LTS12       : 4;
__REG32 LTS13       : 4;
__REG32 LTS14       : 4;
__REG32 LTS15       : 4;
} __ltbr1_bits;

/* Power Management Control Register 0 - PMCR0 */
typedef struct {
__REG32 DPTEN           : 1;
__REG32 PTVAI           : 2;
__REG32 PTVAIM          : 1;
__REG32 DVFEN           : 1;
__REG32 SCR             : 1;
__REG32 DRCE0           : 1;
__REG32 DRCE1           : 1;
__REG32 DRCE2           : 1;
__REG32 DRCE3           : 1;
__REG32 WFIM            : 1;
__REG32 DPVV            : 1;
__REG32 DPVCR           : 1;
__REG32 FSVAI           : 2;
__REG32 FSVAIM          : 1;
__REG32 DVFS_START      : 1;
__REG32 PTVIS           : 1;
__REG32 LBCF            : 2;
__REG32 LBFL            : 1;
__REG32 LBMI            : 1;
__REG32 DVFIS           : 1;
__REG32 DVFEV           : 1;
__REG32 DVFS_UPD_FINISH : 1;
__REG32                 : 3;
__REG32 DVSUP           : 2;
__REG32                 : 2;
} __pmcr0_bits;

/* Power Management Control Register 1 - PMCR1 */
typedef struct {
__REG32 DVGP            : 4;
__REG32                 : 4;
__REG32 CPSPA           : 4;
__REG32 CPFA            : 1;
__REG32 CPEN            : 1;
__REG32                 : 2;
__REG32 WBCN            : 8;
__REG32 CPSPA_EMI       : 4;
__REG32 CPFA_EMI        : 1;
__REG32 CPEN_EMI        : 1;
__REG32                 : 2;
} __pmcr1_bits;

/* Power Management Control Register 2 - PMCR2 */
typedef struct {
__REG32 DVFS_ACK        : 1;
__REG32 DVFS_REQ        : 1;
__REG32                 : 2;
__REG32 ARM_CLKON_CNT   : 4;
__REG32 ARM_MEMON_CNT   : 8;
__REG32 OSC24M_DOWN     : 1;
__REG32 VSTBY           : 1;
__REG32 ARMMEMDWN       : 1;
__REG32                 :13;
} __pmcr2_bits;

/* Misc Control Register - MCR */
typedef struct {
__REG32 PER_CLK_MUX     :16;
__REG32 USB_CLK_MUX     : 1;
__REG32 SSI1_CLK_MUX    : 1;
__REG32 SSI2_CLK_MUX    : 1;
__REG32 ESAI_CLK_MUX    : 1;
__REG32 CLKO_SEL        : 4;
__REG32 CLKO_DIV        : 6;
__REG32 CLKO_EN         : 1;
__REG32 USB_XTAL_MUX    : 1;
} __mcr_bits;

/* Low Power Interrupt Mask Registers - LPIMR0 */
typedef struct {
__REG32 LPIM0           : 1;
__REG32 LPIM1           : 1;
__REG32 LPIM2           : 1;
__REG32 LPIM3           : 1;
__REG32 LPIM4           : 1;
__REG32 LPIM5           : 1;
__REG32 LPIM6           : 1;
__REG32 LPIM7           : 1;
__REG32 LPIM8           : 1;
__REG32 LPIM9           : 1;
__REG32 LPIM10          : 1;
__REG32 LPIM11          : 1;
__REG32 LPIM12          : 1;
__REG32 LPIM13          : 1;
__REG32 LPIM14          : 1;
__REG32 LPIM15          : 1;
__REG32 LPIM16          : 1;
__REG32 LPIM17          : 1;
__REG32 LPIM18          : 1;
__REG32 LPIM19          : 1;
__REG32 LPIM20          : 1;
__REG32 LPIM21          : 1;
__REG32 LPIM22          : 1;
__REG32 LPIM23          : 1;
__REG32 LPIM24          : 1;
__REG32 LPIM25          : 1;
__REG32 LPIM26          : 1;
__REG32 LPIM27          : 1;
__REG32 LPIM28          : 1;
__REG32 LPIM29          : 1;
__REG32 LPIM30          : 1;
__REG32 LPIM31          : 1;
} __lpimr0_bits;

/* Low Power Interrupt Mask Registers - LPIMR1 */
typedef struct {
__REG32 LPIM32          : 1;
__REG32 LPIM33          : 1;
__REG32 LPIM34          : 1;
__REG32 LPIM35          : 1;
__REG32 LPIM36          : 1;
__REG32 LPIM37          : 1;
__REG32 LPIM38          : 1;
__REG32 LPIM39          : 1;
__REG32 LPIM40          : 1;
__REG32 LPIM41          : 1;
__REG32 LPIM42          : 1;
__REG32 LPIM43          : 1;
__REG32 LPIM44          : 1;
__REG32 LPIM45          : 1;
__REG32 LPIM46          : 1;
__REG32 LPIM47          : 1;
__REG32 LPIM48          : 1;
__REG32 LPIM49          : 1;
__REG32 LPIM50          : 1;
__REG32 LPIM51          : 1;
__REG32 LPIM52          : 1;
__REG32 LPIM53          : 1;
__REG32 LPIM54          : 1;
__REG32 LPIM55          : 1;
__REG32 LPIM56          : 1;
__REG32 LPIM57          : 1;
__REG32 LPIM58          : 1;
__REG32 LPIM59          : 1;
__REG32 LPIM60          : 1;
__REG32 LPIM61          : 1;
__REG32 LPIM62          : 1;
__REG32 LPIM63          : 1;
} __lpimr1_bits;

/* -------------------------------------------------------------------------*/
/*               CSI  registers                                             */
/* -------------------------------------------------------------------------*/
/* CSI Control Register 1 */
typedef struct{
__REG32 PIXEL_BIT           : 1;
__REG32 REDGE               : 1;
__REG32 INV_PCLK            : 1;
__REG32 INV_DATA            : 1;
__REG32 GCLK_MODE           : 1;
__REG32 CLR_RXFIFO          : 1;
__REG32 CLR_STATFIFO        : 1;
__REG32 PACK_DIR            : 1;
__REG32 FCC                 : 1;
__REG32 MCLKEN              : 1;
__REG32 CCIR_EN             : 1;
__REG32 HSYNC_POL           : 1;
__REG32 MCLKDIV             : 4;
__REG32 SOF_INTEN           : 1;
__REG32 SOF_POL             : 1;
__REG32 RXFF_INTEN          : 1;
__REG32 FB1_DMA_DONE_INTEN  : 1;
__REG32 FB2_DMA_DONE_INTEN  : 1;
__REG32 STATFF_INTEN        : 1;
__REG32 SFF_DMA_DONE_INTEN  : 1;
__REG32                     : 1;
__REG32 RF_OR_INTEN         : 1;
__REG32 SF_OR_INTEN         : 1;
__REG32 COF_INT_E           : 1;
__REG32 CCIR_MODE           : 1;
__REG32 PRP_IF_EN           : 1;
__REG32 EOF_INT_EN          : 1;
__REG32 EXT_VSYNC           : 1;
__REG32 SWAP16_EN           : 1;
} __csicr1_bits;

/* CSI Control Register 2 */
typedef struct{
__REG32 HSC                 : 8;
__REG32 VSC                 : 8;
__REG32 LVRM                : 3;
__REG32 BTS                 : 2;
__REG32                     : 2;
__REG32 SCE                 : 1;
__REG32 AFS                 : 2;
__REG32 DRM                 : 1;
__REG32                     : 1;
__REG32 DMA_BURST_TYPE_SFF  : 2;
__REG32 DMA_BURST_TYPE_RFF  : 2;
} __csicr2_bits;

/* CSI Control Register 3 */
typedef struct{
__REG32 ECC_AUTO_EN       : 1;
__REG32 ECC_INT_EN        : 1;
__REG32 ZERO_PACK_EN      : 1;
__REG32 TWO_8BIT_SENSOR   : 1;
__REG32 RxFF_LEVEL        : 3;
__REG32 HRESP_ERR_EN      : 1;
__REG32 STATFF_LEVEL      : 3;
__REG32 DMA_REQ_EN_SFF    : 1;
__REG32 DMA_REQ_EN_RFF    : 1;
__REG32 DMA_REFLASH_SFF   : 1;
__REG32 DMA_REFLASH_RFF   : 1;
__REG32 FRMCNT_RST        : 1;
__REG32 FRMCNT            :16;
} __csicr3_bits;

/* CSI Status Register */
typedef struct{
__REG32 DRDY              : 1;
__REG32 ECC_INT           : 1;
__REG32                   : 5;
__REG32 HRESP_ERR_INT     : 1;
__REG32                   : 5;
__REG32 COF_INT           : 1;
__REG32 F1_INT            : 1;
__REG32 F2_INT            : 1;
__REG32 SOF_INT           : 1;
__REG32 EOF_INT           : 1;
__REG32 RXFF_INT          : 1;
__REG32 DMA_TSF_DONE_FB1  : 1;
__REG32 DMA_TSF_DONE_FB2  : 1;
__REG32 STATFF_INT        : 1;
__REG32 DMA_TSF_DONE_SFF  : 1;
__REG32                   : 1;
__REG32 RFF_OR_INT        : 1;
__REG32 SFF_OR_INT        : 1;
__REG32                   : 6;
} __csisr_bits;

/* CSI RX Count Register */
typedef struct{
__REG32 RXCNT  :22;
__REG32        :10;
} __csirxcnt_bits;

/* CSI Frame Buffer Parameter Register (CSIFBUF_PARA) */
typedef struct{
__REG32 FBUF_STRIDE  :16;
__REG32              :16;
} __csifbuf_para_bits;

/* CSI Frame Buffer Parameter Register (CSIFBUF_PARA) */
typedef struct{
__REG32 IMAGE_HEIGHT :16;
__REG32 IMAGE_WIDTH  :16;
} __csiimag_para_bits;

/* -------------------------------------------------------------------------*/
/*      CSPI                                                                */
/* -------------------------------------------------------------------------*/
/* Control Registers */
typedef struct{
__REG32 EN            : 1;
__REG32 MODE          : 1;
__REG32 XCH           : 1;
__REG32 SMC           : 1;
__REG32 POL           : 1;
__REG32 PHA           : 1;
__REG32 SSCTL         : 1;
__REG32 SSPOL         : 1;
__REG32 DRCTL         : 2;
__REG32               : 2;
__REG32 CS            : 2;
__REG32               : 2;
__REG32 DATA_RATE     : 3;
__REG32               : 1;
__REG32 BURST_LENGTH  :12;
} __cspi_controlreg_bits;

/* Interrupt Control Register (INTREG) */
typedef struct{
__REG32 TEEN        : 1;
__REG32 THEN        : 1;
__REG32 TFEN        : 1;
__REG32 RREN        : 1;
__REG32 RHEN        : 1;
__REG32 RFEN        : 1;
__REG32 ROEN        : 1;
__REG32 TCEN        : 1;
__REG32             :24;
} __cspi_intreg_bits;

/* DMA Control Register (DMAREG) */
typedef struct{
__REG32 TEDEN       : 1;
__REG32 THDEN       : 1;
__REG32             : 2;
__REG32 RHDEN       : 1;
__REG32 RFDEN       : 1;
__REG32             :26;
} __cspi_dma_bits;

/* Status Register (STATREG) */
typedef struct{
__REG32 TE          : 1;
__REG32 TH          : 1;
__REG32 TF          : 1;
__REG32 RR          : 1;
__REG32 RH          : 1;
__REG32 RF          : 1;
__REG32 RO          : 1;
__REG32 TC          : 1;
__REG32             :24;
} __cspi_statreg_bits;

/* Sample Period Control Register */
typedef struct{
__REG32 SAMPLE_PERIOD :15;
__REG32 CSRC          : 1;
__REG32               :16;
} __cspi_period_bits;

/* Test Register */
typedef struct{
__REG32 TXCNT       : 4;
__REG32 RXCNT       : 4;
__REG32             : 6;
__REG32 LBC         : 1;
__REG32 SWAP        : 1;
__REG32             :16;
} __cspi_test_bits;

/* -------------------------------------------------------------------------*/
/*               Embedded Cross Trigger (ECT)                               */
/* -------------------------------------------------------------------------*/
/* CTI Control Register */
typedef struct{
__REG32 GLBEN       : 1;
__REG32             :31;
} __cticontrol_bits;

/* CTI Status Register */
typedef struct{
__REG32 LOCKED      : 1;
__REG32 DGBEN       : 1;
__REG32             :30;
} __ctistatus_bits;

/* CTI Protection Enable Register */
typedef struct{
__REG32 PROT        : 1;
__REG32             :31;
} __ctiprotection_bits;

/* CTI Interrupt Acknowledge Register */
typedef struct{
__REG32 INTACK0     : 1;
__REG32 INTACK1     : 1;
__REG32 INTACK2     : 1;
__REG32 INTACK3     : 1;
__REG32 INTACK4     : 1;
__REG32 INTACK5     : 1;
__REG32 INTACK6     : 1;
__REG32 INTACK7     : 1;
__REG32             :24;
} __ctiintack_bits;

/* CTI Application Trigger Set Register */
typedef struct{
__REG32 APPSET0     : 1;
__REG32 APPSET1     : 1;
__REG32 APPSET2     : 1;
__REG32 APPSET3     : 1;
__REG32             :28;
} __ctiappset_bits;

/* CTI Application Trigger Clear Register */
typedef struct{
__REG32 APPCLEAR0   : 1;
__REG32 APPCLEAR1   : 1;
__REG32 APPCLEAR2   : 1;
__REG32 APPCLEAR3   : 1;
__REG32             :28;
} __ctiappclear_bits;

/* CTI Application Pulse Register */
typedef struct{
__REG32 APPPULSE0   : 1;
__REG32 APPPULSE1   : 1;
__REG32 APPPULSE2   : 1;
__REG32 APPPULSE3   : 1;
__REG32             :28;
} __ctiapppulse_bits;

/* CTI Trigger to Channel Enable Register */
typedef struct{
__REG32 TRIGINEN0   : 1;
__REG32 TRIGINEN1   : 1;
__REG32 TRIGINEN2   : 1;
__REG32 TRIGINEN3   : 1;
__REG32             :28;
} __ctiinen_bits;

/* CTI Channel to Trigger Enable Register */
typedef struct{
__REG32 TRIGOUTEN0  : 1;
__REG32 TRIGOUTEN1  : 1;
__REG32 TRIGOUTEN2  : 1;
__REG32 TRIGOUTEN3  : 1;
__REG32             :28;
} __ctiouten_bits;

/* CTI Trigger In Status Register */
typedef struct{
__REG32 TRIGINSTATUS0 : 1;
__REG32 TRIGINSTATUS1 : 1;
__REG32 TRIGINSTATUS2 : 1;
__REG32 TRIGINSTATUS3 : 1;
__REG32 TRIGINSTATUS4 : 1;
__REG32 TRIGINSTATUS5 : 1;
__REG32 TRIGINSTATUS6 : 1;
__REG32 TRIGINSTATUS7 : 1;
__REG32               :24;
} __ctitriginstatus_bits;

/* CTI Trigger Out Status Register */
typedef struct{
__REG32 TRIGOUTSTATUS0: 1;
__REG32 TRIGOUTSTATUS1: 1;
__REG32 TRIGOUTSTATUS2: 1;
__REG32 TRIGOUTSTATUS3: 1;
__REG32 TRIGOUTSTATUS4: 1;
__REG32 TRIGOUTSTATUS5: 1;
__REG32 TRIGOUTSTATUS6: 1;
__REG32 TRIGOUTSTATUS7: 1;
__REG32               :24;
} __ctitrigoutstatus_bits;

/* CTI Channel In Status Register */
typedef struct{
__REG32 CTICHINSTATUS0: 1;
__REG32 CTICHINSTATUS1: 1;
__REG32 CTICHINSTATUS2: 1;
__REG32 CTICHINSTATUS3: 1;
__REG32               :28;
} __ctichinstatus_bits;

/* CTI Channel Out Status Register */
typedef struct{
__REG32 CTICHOUTSTATUS0 : 1;
__REG32 CTICHOUTSTATUS1 : 1;
__REG32 CTICHOUTSTATUS2 : 1;
__REG32 CTICHOUTSTATUS3 : 1;
__REG32                 :28;
} __ctichoutstatus_bits;

/* CTI Test Control Register */
typedef struct{
__REG32 CTITCR0         : 1;
__REG32 CTITCR1         : 1;
__REG32                 :30;
} __ctitcr_bits;

/* CTI Input Test Register 0 */
typedef struct{
__REG32 CTITIP00        : 1;
__REG32 CTITIP01        : 1;
__REG32 CTITIP02        : 1;
__REG32 CTITIP03        : 1;
__REG32 CTITIP04        : 1;
__REG32 CTITIP05        : 1;
__REG32 CTITIP06        : 1;
__REG32 CTITIP07        : 1;
__REG32                 :24;
} __ctiitip0_bits;

/* CTI Input Test Register 1 */
typedef struct{
__REG32 CTITIP10        : 1;
__REG32 CTITIP11        : 1;
__REG32 CTITIP12        : 1;
__REG32 CTITIP13        : 1;
__REG32                 :28;
} __ctiitip1_bits;

/* CTI Input Test Register 2 */
typedef struct{
__REG32 CTITIP20        : 1;
__REG32 CTITIP21        : 1;
__REG32 CTITIP22        : 1;
__REG32 CTITIP23        : 1;
__REG32 CTITIP24        : 1;
__REG32 CTITIP25        : 1;
__REG32 CTITIP26        : 1;
__REG32 CTITIP27        : 1;
__REG32                 :24;
} __ctiitip2_bits;

/* CTI Input Test Register 3 */
typedef struct{
__REG32 CTITIP30        : 1;
__REG32 CTITIP31        : 1;
__REG32 CTITIP32        : 1;
__REG32 CTITIP33        : 1;
__REG32                 :28;
} __ctiitip3_bits;

/* CTI Output Test Register 0 */
typedef struct{
__REG32 CTIITOP00       : 1;
__REG32 CTIITOP01       : 1;
__REG32 CTIITOP02       : 1;
__REG32 CTIITOP03       : 1;
__REG32 CTIITOP04       : 1;
__REG32 CTIITOP05       : 1;
__REG32 CTIITOP06       : 1;
__REG32 CTIITOP07       : 1;
__REG32                 :24;
} __ctiitop0_bits;

/* CTI Output Test Register 1 */
typedef struct{
__REG32 CTIITOP10       : 1;
__REG32 CTIITOP11       : 1;
__REG32 CTIITOP12       : 1;
__REG32 CTIITOP13       : 1;
__REG32                 :28;
} __ctiitop1_bits;

/* CTI Output Test Register 2 */
typedef struct{
__REG32 CTIITOP20       : 1;
__REG32 CTIITOP21       : 1;
__REG32 CTIITOP22       : 1;
__REG32 CTIITOP23       : 1;
__REG32 CTIITOP24       : 1;
__REG32 CTIITOP25       : 1;
__REG32 CTIITOP26       : 1;
__REG32 CTIITOP27       : 1;
__REG32                 :24;
} __ctiitop2_bits;

/* CTI Output Test Register 3 */
typedef struct{
__REG32 CTIITOP30       : 1;
__REG32 CTIITOP31       : 1;
__REG32 CTIITOP32       : 1;
__REG32 CTIITOP33       : 1;
__REG32                 :28;
} __ctiitop3_bits;

/* CTI Peripheral Identification Register 0 */
typedef struct{
__REG32 Partnumber0     : 8;
__REG32                 :24;
} __ctiperiphid0_bits;

/* CTI Peripheral Identification Register 1 */
typedef struct{
__REG32 Partnumber0     : 4;
__REG32 Designer0       : 4;
__REG32                 :24;
} __ctiperiphid1_bits;

/* CTI Peripheral Identification Register 2 */
typedef struct{
__REG32 Designer1       : 4;
__REG32 Revision        : 4;
__REG32                 :24;
} __ctiperiphid2_bits;

/* CTI Peripheral Identification Register 3 */
typedef struct{
__REG32 Configuration   : 8;
__REG32                 :24;
} __ctiperiphid3_bits;

/* CTI Identification Register 0 */
typedef struct{
__REG32 CTIPCELLID00    : 1;
__REG32 CTIPCELLID01    : 1;
__REG32 CTIPCELLID02    : 1;
__REG32 CTIPCELLID03    : 1;
__REG32 CTIPCELLID04    : 1;
__REG32 CTIPCELLID05    : 1;
__REG32 CTIPCELLID06    : 1;
__REG32 CTIPCELLID07    : 1;
__REG32                 :24;
} __ctipcellid0_bits;

/* CTI Identification Register 1 */
typedef struct{
__REG32 CTIPCELLID10    : 1;
__REG32 CTIPCELLID11    : 1;
__REG32 CTIPCELLID12    : 1;
__REG32 CTIPCELLID13    : 1;
__REG32 CTIPCELLID14    : 1;
__REG32 CTIPCELLID15    : 1;
__REG32 CTIPCELLID16    : 1;
__REG32 CTIPCELLID17    : 1;
__REG32                 :24;
} __ctipcellid1_bits;

/* CTI Identification Register 2 */
typedef struct{
__REG32 CTIPCELLID20    : 1;
__REG32 CTIPCELLID21    : 1;
__REG32 CTIPCELLID22    : 1;
__REG32 CTIPCELLID23    : 1;
__REG32 CTIPCELLID24    : 1;
__REG32 CTIPCELLID25    : 1;
__REG32 CTIPCELLID26    : 1;
__REG32 CTIPCELLID27    : 1;
__REG32                 :24;
} __ctipcellid2_bits;

/* CTI Identification Register 3 */
typedef struct{
__REG32       : 1;
__REG32 CTIPCELLID31    : 1;
__REG32 CTIPCELLID32    : 1;
__REG32 CTIPCELLID33    : 1;
__REG32 CTIPCELLID34    : 1;
__REG32 CTIPCELLID35    : 1;
__REG32 CTIPCELLID36    : 1;
__REG32 CTIPCELLID37    : 1;
__REG32                 :24;
} __ctipcellid3_bits;

/* -------------------------------------------------------------------------*/
/*               DryIce                                                     */
/* -------------------------------------------------------------------------*/

/* DryIce Time Counter LSB Register */
typedef struct{
__REG32                 :17;
__REG32 DTC             :15;
} __dtclr_bits;

/* DryIce Clock Alarm Register */
typedef struct{
__REG32                 :17;
__REG32 DCA             :15;
} __dcalr_bits;

/* DryIce Control Register (DCR) */
typedef struct{
__REG32 SWR             : 1;
__REG32                 : 1;
__REG32 MCE             : 1;
__REG32 TCE             : 1;
__REG32 APE             : 1;
__REG32                 : 9;
__REG32 OSCB            : 1;
__REG32 NSA             : 1;
__REG32 FSHL            : 1;
__REG32 TCSL            : 1;
__REG32 TCHL            : 1;
__REG32 MCSL            : 1;
__REG32 MCHL            : 1;
__REG32 PKWSL           : 1;
__REG32 PKWHL           : 1;
__REG32 PKRSL           : 1;
__REG32 PKRHL           : 1;
__REG32 RKSL            : 1;
__REG32 RKHL            : 1;
__REG32 KSSL            : 1;
__REG32 KSHL            : 1;
__REG32 TDCSL           : 1;
__REG32 TDCHL           : 1;
__REG32                 : 1;
} __dcr_bits;

/* DryIce Status Register (DSR) */
typedef struct{
__REG32 SVF             : 1;
__REG32 NVF             : 1;
__REG32 TCO             : 1;
__REG32 MCO             : 1;
__REG32 CAF             : 1;
__REG32 RKV             : 1;
__REG32 RKE             : 1;
__REG32 WEF             : 1;
__REG32 WCF             : 1;
__REG32 WNF             : 1;
__REG32 WBF             : 1;
__REG32 KCB             : 1;
__REG32                 : 4;
__REG32 VTD             : 1;
__REG32 CTD             : 1;
__REG32 TTD             : 1;
__REG32 SAD             : 1;
__REG32 EBD             : 1;
__REG32 ETAD            : 1;
__REG32 ETBD            : 1;
__REG32 WTD             : 1;
__REG32                 : 8;
} __dsr_bits;

/* DryIce Interrupt Enable Register (DIER) */
typedef struct{
__REG32 SVIE            : 1;
__REG32                 : 1;
__REG32 TOIE            : 1;
__REG32 MOIE            : 1;
__REG32 CAIE            : 1;
__REG32 RKIE            : 1;
__REG32                 : 1;
__REG32 WEIE            : 1;
__REG32 WCIE            : 1;
__REG32 WNIE            : 1;
__REG32                 :22;
} __dier_bits;

/* DryIce Key Select Register (DKSR) */
typedef struct{
__REG32 SKS             : 3;
__REG32                 :29;
} __dksr_bits;

/* DryIce Key Control Register (DKCR) */
typedef struct{
__REG32 LRK             : 1;
__REG32                 :31;
} __dkcr_bits;

/* DryIce Tamper Configuration Register (DTCR) */
typedef struct{
__REG32 VTE             : 1;
__REG32 CTE             : 1;
__REG32 TTE             : 1;
__REG32 SAIE            : 1;
__REG32 EBE             : 1;
__REG32 ETAE            : 1;
__REG32 ETBE            : 1;
__REG32 WTE             : 1;
__REG32 TOE             : 1;
__REG32 MOE             : 1;
__REG32                 : 5;
__REG32 SAOE            : 1;
__REG32 WGFE            : 1;
__REG32 WTGF            : 5;
__REG32 ETGFA           : 5;
__REG32 ETGFB           : 5;
} __dtcr_bits;

/* DryIce Analog Configuration Register (DACR) */
typedef struct{
__REG32 LTDC            : 3;
__REG32 HTDC            : 3;
__REG32 VRC             : 3;
__REG32                 :23;
} __dacr_bits;

/* -------------------------------------------------------------------------*/
/*      M3IF                                                                */
/* -------------------------------------------------------------------------*/
/* M3IF Control Register (M3IFCTL) */
typedef struct {
__REG32 MRRP     : 8;
__REG32 MLSD     : 3;
__REG32 MLSD_EN  : 1;
__REG32          :19;
__REG32 SDA      : 1;
} __m3ifctl_bits;

/* M3IF WaterMark Configuration Registers (M3IFWCFG0–M3IFWCFG7) */
typedef struct {
__REG32          :10;
__REG32 WBA      :22;
} __m3ifwcfg_bits;

/* M3IF WaterMark Control and Status Register (M3IFWCSR) */
typedef struct {
__REG32 WS0      : 1;
__REG32 WS1      : 1;
__REG32 WS2      : 1;
__REG32 WS3      : 1;
__REG32 WS4      : 1;
__REG32 WS5      : 1;
__REG32 WS6      : 1;
__REG32 WS7      : 1;
__REG32          :23;
__REG32 WIE      : 1;
} __m3ifwcsr_bits;

/* M3IF Snooping Configuration Register 0 (M3IFSCFG0) */
typedef struct {
__REG32 SE       : 1;
__REG32 SWSZ     : 4;
__REG32          : 6;
__REG32 SWBA     :21;
} __m3ifscfg0_bits;

/* M3IF Snooping Configuration Register 1 (M3IFSCFG1) */
typedef struct {
__REG32 SSE0_0   : 1;
__REG32 SSE0_1   : 1;
__REG32 SSE0_2   : 1;
__REG32 SSE0_3   : 1;
__REG32 SSE0_4   : 1;
__REG32 SSE0_5   : 1;
__REG32 SSE0_6   : 1;
__REG32 SSE0_7   : 1;
__REG32 SSE0_8   : 1;
__REG32 SSE0_9   : 1;
__REG32 SSE0_10  : 1;
__REG32 SSE0_11  : 1;
__REG32 SSE0_12  : 1;
__REG32 SSE0_13  : 1;
__REG32 SSE0_14  : 1;
__REG32 SSE0_15  : 1;
__REG32 SSE0_16  : 1;
__REG32 SSE0_17  : 1;
__REG32 SSE0_18  : 1;
__REG32 SSE0_19  : 1;
__REG32 SSE0_20  : 1;
__REG32 SSE0_21  : 1;
__REG32 SSE0_22  : 1;
__REG32 SSE0_23  : 1;
__REG32 SSE0_24  : 1;
__REG32 SSE0_25  : 1;
__REG32 SSE0_26  : 1;
__REG32 SSE0_27  : 1;
__REG32 SSE0_28  : 1;
__REG32 SSE0_29  : 1;
__REG32 SSE0_30  : 1;
__REG32 SSE0_31  : 1;
} __m3ifscfg1_bits;

/* M3IF Snooping Configuration Register 2 (M3IFSCFG2) */
typedef struct {
__REG32 SSE1_0   : 1;
__REG32 SSE1_1   : 1;
__REG32 SSE1_2   : 1;
__REG32 SSE1_3   : 1;
__REG32 SSE1_4   : 1;
__REG32 SSE1_5   : 1;
__REG32 SSE1_6   : 1;
__REG32 SSE1_7   : 1;
__REG32 SSE1_8   : 1;
__REG32 SSE1_9   : 1;
__REG32 SSE1_10  : 1;
__REG32 SSE1_11  : 1;
__REG32 SSE1_12  : 1;
__REG32 SSE1_13  : 1;
__REG32 SSE1_14  : 1;
__REG32 SSE1_15  : 1;
__REG32 SSE1_16  : 1;
__REG32 SSE1_17  : 1;
__REG32 SSE1_18  : 1;
__REG32 SSE1_19  : 1;
__REG32 SSE1_20  : 1;
__REG32 SSE1_21  : 1;
__REG32 SSE1_22  : 1;
__REG32 SSE1_23  : 1;
__REG32 SSE1_24  : 1;
__REG32 SSE1_25  : 1;
__REG32 SSE1_26  : 1;
__REG32 SSE1_27  : 1;
__REG32 SSE1_28  : 1;
__REG32 SSE1_29  : 1;
__REG32 SSE1_30  : 1;
__REG32 SSE1_31  : 1;
} __m3ifscfg2_bits;

/* M3IF Snooping Status Register 0 (M3IFSSR0) */
typedef struct {
__REG32 SSS0_0   : 1;
__REG32 SSS0_1   : 1;
__REG32 SSS0_2   : 1;
__REG32 SSS0_3   : 1;
__REG32 SSS0_4   : 1;
__REG32 SSS0_5   : 1;
__REG32 SSS0_6   : 1;
__REG32 SSS0_7   : 1;
__REG32 SSS0_8   : 1;
__REG32 SSS0_9   : 1;
__REG32 SSS0_10  : 1;
__REG32 SSS0_11  : 1;
__REG32 SSS0_12  : 1;
__REG32 SSS0_13  : 1;
__REG32 SSS0_14  : 1;
__REG32 SSS0_15  : 1;
__REG32 SSS0_16  : 1;
__REG32 SSS0_17  : 1;
__REG32 SSS0_18  : 1;
__REG32 SSS0_19  : 1;
__REG32 SSS0_20  : 1;
__REG32 SSS0_21  : 1;
__REG32 SSS0_22  : 1;
__REG32 SSS0_23  : 1;
__REG32 SSS0_24  : 1;
__REG32 SSS0_25  : 1;
__REG32 SSS0_26  : 1;
__REG32 SSS0_27  : 1;
__REG32 SSS0_28  : 1;
__REG32 SSS0_29  : 1;
__REG32 SSS0_30  : 1;
__REG32 SSS0_31  : 1;
} __m3ifssr0_bits;

/* M3IF Snooping Status Register 1 (M3IFSSR1) */
typedef struct {
__REG32 SSS1_0   : 1;
__REG32 SSS1_1   : 1;
__REG32 SSS1_2   : 1;
__REG32 SSS1_3   : 1;
__REG32 SSS1_4   : 1;
__REG32 SSS1_5   : 1;
__REG32 SSS1_6   : 1;
__REG32 SSS1_7   : 1;
__REG32 SSS1_8   : 1;
__REG32 SSS1_9   : 1;
__REG32 SSS1_10  : 1;
__REG32 SSS1_11  : 1;
__REG32 SSS1_12  : 1;
__REG32 SSS1_13  : 1;
__REG32 SSS1_14  : 1;
__REG32 SSS1_15  : 1;
__REG32 SSS1_16  : 1;
__REG32 SSS1_17  : 1;
__REG32 SSS1_18  : 1;
__REG32 SSS1_19  : 1;
__REG32 SSS1_20  : 1;
__REG32 SSS1_21  : 1;
__REG32 SSS1_22  : 1;
__REG32 SSS1_23  : 1;
__REG32 SSS1_24  : 1;
__REG32 SSS1_25  : 1;
__REG32 SSS1_26  : 1;
__REG32 SSS1_27  : 1;
__REG32 SSS1_28  : 1;
__REG32 SSS1_29  : 1;
__REG32 SSS1_30  : 1;
__REG32 SSS1_31  : 1;
} __m3ifssr1_bits;

/* M3IF Master Lock WEIM CSx Register (M3IFMLWEx) */
typedef struct {
__REG32 MLGE     : 3;
__REG32 MLGE_EN  : 1;
__REG32          :27;
__REG32 WEMA     : 1;
} __m3ifmlwe_bits;

/* -------------------------------------------------------------------------*/
/*               NAND Flash Controller (NFC)                                */
/* -------------------------------------------------------------------------*/
/* Buffer Number for Page Data Transfer To/From Flash Memory */
typedef struct{
__REG16 RBA         : 3;
__REG16             : 1;
__REG16 active_cs   : 2;
__REG16             :10;
} __nfc_rba_bits;

/* NFC Internal Buffer Lock Control */
typedef struct{
__REG16 BLS  : 2;
__REG16      :14;
} __nfc_iblc_bits;

/* NFC Controller Status/Result of Flash Operation */
typedef struct{
__REG16 NOSER1  : 4;
__REG16 NOSER2  : 4;
__REG16 NOSER3  : 4;
__REG16 NOSER4  : 4;
} __ecc_srr_bits;

/* NFC Controller Status/Result of Flash Operation */
typedef struct{
__REG16 NOSER5  : 4;
__REG16 NOSER6  : 4;
__REG16 NOSER7  : 4;
__REG16 NOSER8  : 4;
} __ecc_srr2_bits;

/* NFC SPare Area Size (SPAS) */
typedef struct{
__REG16 SPAS    : 8;
__REG16         : 8;
} __nfc_spas_bits;

/* NFC Nand Flash Write Protection */
typedef struct{
__REG16 WPC  : 3;
__REG16      :13;
} __nf_wr_prot_bits;

/* NFC NAND Flash Write Protection Status */
typedef struct{
__REG16 LTS0 : 1;
__REG16 LS0  : 1;
__REG16 US0  : 1;
__REG16 LTS1 : 1;
__REG16 LS1  : 1;
__REG16 US1  : 1;
__REG16 LTS2 : 1;
__REG16 LS2  : 1;
__REG16 US2  : 1;
__REG16 LTS3 : 1;
__REG16 LS3  : 1;
__REG16 US3  : 1;
__REG16      : 4;
} __nf_wr_prot_sta_bits;

/* NFC NAND Flash Operation Configuration 1 */
typedef struct{
__REG16 ECC_MODE  : 1;
__REG16 DMA_MODE  : 1;
__REG16 SP_EN     : 1;
__REG16 ECC_EN    : 1;
__REG16 INT_MASK  : 1;
__REG16 NF_BIG    : 1;
__REG16 NFC_RST   : 1;
__REG16 NF_CE     : 1;
__REG16 SYM       : 1;
__REG16 PPB       : 2;
__REG16 FP_INT    : 1;
__REG16           : 4;
} __nand_fc1_bits;

/* NFC NAND Flash Operation Configuration 2 */
typedef struct{
__REG16 FCMD  : 1;
__REG16 FADD  : 1;
__REG16 FDI   : 1;
__REG16 FDO   : 3;
__REG16       : 9;
__REG16 INT   : 1;
} __nand_fc2_bits;

/* -------------------------------------------------------------------------*/
/*      Enhanced SDRAM Controller(ESDRAMC)                                  */
/* -------------------------------------------------------------------------*/
/* ESDCTL0 and ESDCTL1 Control Registers */
typedef struct {
__REG32 PRCT     : 6;
__REG32          : 1;
__REG32 BL       : 1;
__REG32 FP       : 1;
__REG32          : 1;
__REG32 PWDT     : 2;
__REG32          : 1;
__REG32 SREFR    : 3;
__REG32 DSIZ     : 2;
__REG32          : 2;
__REG32 COL      : 2;
__REG32          : 2;
__REG32 ROW      : 3;
__REG32 SP       : 1;
__REG32 SMODE    : 3;
__REG32 SDE      : 1;
} __esdctl_bits;

/* ESDRAMC Configuration Registers (ESDCFG0 /ESDCFG1) */
typedef struct {
__REG32 tRC      : 4;
__REG32 tRCD     : 3;
__REG32          : 1;
__REG32 tCAS     : 2;
__REG32 tRRD     : 2;
__REG32 tRAS     : 3;
__REG32 tWR      : 1;
__REG32 tMRD     : 2;
__REG32 tRP      : 2;
__REG32 tWTR     : 1;
__REG32 tXP      : 2;
__REG32          : 9;
} __esdcfg_bits;

/* ESDMISC Miscellaneous Register (ESDMISC) */
typedef struct {
__REG32             : 1;
__REG32 RST         : 1;
__REG32 MDDR_EN     : 1;
__REG32 MDDR_DL_RST : 1;
__REG32 MDDR_MDIS   : 1;
__REG32 LHD         : 1;
__REG32 MA10_SHARE  : 1;
__REG32 FRC_MSR     : 1;
__REG32 DDR_EN      : 1;
__REG32 DDR2_EN     : 1;
__REG32             :21;
__REG32 SDRAMRDY    : 1;
} __esdmisc_bits;

/* MDDR Delay Line 1/2/5 Configuration Debug Register */
typedef struct {
__REG32 DLY_REG           : 8;
__REG32 DLY_ABS_OFFSET    : 8;
__REG32 DLY_OFFSET        : 8;
__REG32                   : 7;
__REG32 SEL_DLY_REG       : 1;
} __esdcdly_bits;

/* MDDR Delay Line Cycle Length Debug Register */
typedef struct {
__REG32 QTR_CYCLE_LENGTH  : 8;
__REG32                   :24;
} __esdcdlyl_bits;

/* MDDR All Delay Lines Configuration Debug Register */
typedef struct {
__REG32 MTM               : 4;
__REG32                   :27;
__REG32 SN                : 1;
} __esdcdly6_bits;

/* -------------------------------------------------------------------------*/
/*      WEIM                                                                */
/* -------------------------------------------------------------------------*/
/* Chip Select x Upper Control Register (CSCRxU) */
typedef struct {
__REG32 EDC      : 4;
__REG32 WWS      : 3;
__REG32 EW       : 1;
__REG32 WSC      : 6;
__REG32 CNC      : 2;
__REG32 DOL      : 4;
__REG32 SYNC     : 1;
__REG32 PME      : 1;
__REG32 PSZ      : 2;
__REG32 BCS      : 4;
__REG32 BCD      : 2;
__REG32 WP       : 1;
__REG32 SP       : 1;
} __cscru_bits;

/* Chip Select x Lower Control Register (CSCRxL) */
typedef struct {
__REG32 CSEN     : 1;
__REG32 WRAP     : 1;
__REG32 CRE      : 1;
__REG32 PSR      : 1;
__REG32 CSN      : 4;
__REG32 DSZ      : 3;
__REG32 EBC      : 1;
__REG32 CSA      : 4;
__REG32 EBWN     : 4;
__REG32 EBWA     : 4;
__REG32 OEN      : 4;
__REG32 OEA      : 4;
} __cscrl_bits;

/* Chip Select x Additional Control Register (CSCRxA) */
typedef struct {
__REG32 FCE      : 1;
__REG32 CNC2     : 1;
__REG32 AGE      : 1;
__REG32 WWU      : 1;
__REG32 DCT      : 2;
__REG32 DWW      : 2;
__REG32 LBA      : 2;
__REG32 LBN      : 3;
__REG32 LAH      : 2;
__REG32 MUM      : 1;
__REG32 RWN      : 4;
__REG32 RWA      : 4;
__REG32 EBRN     : 4;
__REG32 EBRA     : 4;
} __cscra_bits;

/* WEIM Configuration Register (WCR) */
typedef struct {
__REG32 MAS      : 1;
__REG32          : 1;
__REG32 BCM      : 1;
__REG32          :29;
} __weim_wcr_bits;

/* -------------------------------------------------------------------------*/
/*         Enhanced Periodic Interrupt Timer (EPIT)                         */
/* -------------------------------------------------------------------------*/
/* EPIT Control Register */
typedef struct {
__REG32 EN              : 1;
__REG32 ENMOD           : 1;
__REG32 OCIEN           : 1;
__REG32 RLD             : 1;
__REG32 PRESCALAR       :12;
__REG32 SWR             : 1;
__REG32 IOVW            : 1;
__REG32 DBGEN           : 1;
__REG32 WAITEN          : 1;
__REG32 RES             : 1;
__REG32 STOPEN          : 1;
__REG32 OM              : 2;
__REG32 CLKSRC          : 2;
__REG32                 : 6;
} __epitcr_bits;

/* EPIT Status Register */
typedef struct {
__REG32 OCIF            : 1;
__REG32                 :31;
} __epitsr_bits;

/* -------------------------------------------------------------------------*/
/*            Enhanced Serial Audio Interface (ESAI)                        */
/* -------------------------------------------------------------------------*/
/* ESAI Control Register (ECR) */
typedef struct {
__REG32 ESAIEN          : 1;
__REG32 ERST            : 1;
__REG32                 :14;
__REG32 ERO             : 1;
__REG32 ERI             : 1;
__REG32 ETO             : 1;
__REG32 ETI             : 1;
__REG32                 :12;
} __esai_ecr_bits;

/* ESAI Status Register (ESR) */
typedef struct {
__REG32 RD              : 1;
__REG32 RED             : 1;
__REG32 RDE             : 1;
__REG32 RLS             : 1;
__REG32 TD              : 1;
__REG32 TED             : 1;
__REG32 TDE             : 1;
__REG32 TLS             : 1;
__REG32 TFE             : 1;
__REG32 RFF             : 1;
__REG32 TINIT           : 1;
__REG32                 :21;
} __esai_esr_bits;

/* Transmit FIFO Configuration Register (TFCR) */
typedef struct {
__REG32 TFEN            : 1;
__REG32 TFR             : 1;
__REG32 TE0             : 1;
__REG32 TE1             : 1;
__REG32 TE2             : 1;
__REG32 TE3             : 1;
__REG32 TE4             : 1;
__REG32 TE5             : 1;
__REG32 TFWM            : 8;
__REG32 TWA             : 3;
__REG32 TIEN            : 1;
__REG32                 :12;
} __esai_tfcr_bits;

/* Transmit FIFO Status Register (TFSR) */
typedef struct {
__REG32 TFCNT           : 8;
__REG32 NTFI            : 3;
__REG32                 : 1;
__REG32 NTFO            : 3;
__REG32                 :17;
} __esai_tfsr_bits;

/* Receive FIFO Configuration Register (RFCR) */
typedef struct {
__REG32 RFEN            : 1;
__REG32 RFR             : 1;
__REG32 RE0             : 1;
__REG32 RE1             : 1;
__REG32 RE2             : 1;
__REG32 RE3             : 1;
__REG32                 : 2;
__REG32 RFWM            : 8;
__REG32 RWA             : 3;
__REG32 REXT            : 1;
__REG32                 :12;
} __esai_rfcr_bits;

/* Receive FIFO Status Register (RFSR) */
typedef struct {
__REG32 RFCNT           : 8;
__REG32 NRFI            : 2;
__REG32                 : 2;
__REG32 NRFO            : 2;
__REG32                 :18;
} __esai_rfsr_bits;

/* ESAI Transmit Data Registers (TX5, TX4, TX3, TX2,TX1,TX0) */
typedef struct {
__REG32 TX              :24;
__REG32                 : 8;
} __esai_tx_bits;

/* ESAI Transmit Slot Register (TSR) */
typedef struct {
__REG32 TSR             :24;
__REG32                 : 8;
} __esai_tsr_bits;

/* ESAI Receive Data Registers (RX3, RX2, RX1, RX0) */
typedef struct {
__REG32 RX              :24;
__REG32                 : 8;
} __esai_rx_bits;

/* ESAI Status Register (SAISR) */
typedef struct {
__REG32 IF0             : 1;
__REG32 IF1             : 1;
__REG32 IF2             : 1;
__REG32                 : 3;
__REG32 RFS             : 1;
__REG32 ROE             : 1;
__REG32 RDF             : 1;
__REG32 REDF            : 1;
__REG32 RODF            : 1;
__REG32                 : 2;
__REG32 TFS             : 1;
__REG32 TUE             : 1;
__REG32 TDE             : 1;
__REG32 TEDE            : 1;
__REG32 TODFE           : 1;
__REG32                 :14;
} __esai_saisr_bits;

/* ESAI Common Control Register (SAICR) */
typedef struct {
__REG32 OF0             : 1;
__REG32 OF1             : 1;
__REG32 OF2             : 1;
__REG32                 : 3;
__REG32 SYN             : 1;
__REG32 TEBE            : 1;
__REG32 ALC             : 1;
__REG32                 :23;
} __esai_saicr_bits;

/* ESAI Transmit Control Register (TCR) */
typedef struct {
__REG32 TE0             : 1;
__REG32 TE1             : 1;
__REG32 TE2             : 1;
__REG32 TE3             : 1;
__REG32 TE4             : 1;
__REG32 TE5             : 1;
__REG32 TSHFD           : 1;
__REG32 TWA             : 1;
__REG32 TMOD            : 2;
__REG32 TSWS            : 5;
__REG32 TFSL            : 1;
__REG32 TFSR            : 1;
__REG32 PADC            : 1;
__REG32                 : 1;
__REG32 TPR             : 1;
__REG32 TEIE            : 1;
__REG32 TDEIE           : 1;
__REG32 TIE             : 1;
__REG32 TLIE            : 1;
__REG32                 : 8;
} __esai_tcr_bits;

/* ESAI Transmitter Clock Control Register (TCCR) */
typedef struct {
__REG32 TPM             : 8;
__REG32 TPSR            : 1;
__REG32 TDC             : 5;
__REG32 TFP             : 4;
__REG32 TCKP            : 1;
__REG32 TFSP            : 1;
__REG32 THCKP           : 1;
__REG32 TCKD            : 1;
__REG32 TFSD            : 1;
__REG32 THCKD           : 1;
__REG32                 : 8;
} __esai_tccr_bits;

/* ESAI Receive Control Register (RCR) */
typedef struct {
__REG32 RPM             : 8;
__REG32 RPSR            : 1;
__REG32 RDC             : 5;
__REG32 RFP             : 3;
__REG32                 : 2;
__REG32 RPR             : 1;
__REG32 REIE            : 1;
__REG32 RDEIE           : 1;
__REG32 RIE             : 1;
__REG32 RLIE            : 1;
__REG32                 : 8;
} __esai_rcr_bits;

/* ESAI Receiver Clock Control Register (RCCR) */
typedef struct {
__REG32 RPM             : 8;
__REG32 RPSR            : 1;
__REG32 RDC             : 5;
__REG32 RFP             : 4;
__REG32 RCKP            : 1;
__REG32 RFSP            : 1;
__REG32 RHCKP           : 1;
__REG32 RCKD            : 1;
__REG32 RFSD            : 1;
__REG32 RHCKD           : 1;
__REG32                 : 8;
} __esai_rccr_bits;

/* ESAI Transmit Slot Mask Register A/B */
typedef struct {
__REG32 TS              :16;
__REG32                 :16;
} __esai_tsm_bits;

/* ESAI Receive Slot Mask Register A/B */
typedef struct {
__REG32 RS              :16;
__REG32                 :16;
} __esai_rsm_bits;

/* Port C Direction Register (PRRC) */
typedef struct {
__REG32 PDC             :12;
__REG32                 :20;
} __esai_prrc_bits;

/* Port C Control Register (PCRC) */
typedef struct {
__REG32 PC              :12;
__REG32                 :20;
} __esai_pcrc_bits;

/* -------------------------------------------------------------------------*/
/* Enhanced Secured Digital Host Controller version 2 (eSDHCv2)             */
/* -------------------------------------------------------------------------*/
/* Block Attributes Register */
typedef struct {
__REG32 BLKSZE          :13;
__REG32                 : 3;
__REG32 BLKCNT          :16;
} __esdhc_blkattr_bits;

/* Transfer Type Register */
typedef struct {
__REG32 DMAEN           : 1;
__REG32 BCEN            : 1;
__REG32 AC12EN          : 1;
__REG32                 : 1;
__REG32 DTDSEL          : 1;
__REG32 MSBSEL          : 1;
__REG32                 :10;
__REG32 RSPTYP          : 2;
__REG32                 : 1;
__REG32 CCCEN           : 1;
__REG32 CICEN           : 1;
__REG32 DPSEL           : 1;
__REG32 CMDTYP          : 2;
__REG32 CMDINX          : 6;
__REG32                 : 2;
} __esdhc_xfertyp_bits;

/* Present State Register */
typedef struct {
__REG32 CIHB            : 1;
__REG32 CDIHB           : 1;
__REG32 DLA             : 1;
__REG32                 : 1;
__REG32 IPGOFF          : 1;
__REG32 HCKOFF          : 1;
__REG32 PEROFF          : 1;
__REG32 SDOFF           : 1;
__REG32 WTA             : 1;
__REG32 RTA             : 1;
__REG32 BWEN            : 1;
__REG32 BREN            : 1;
__REG32                 : 4;
__REG32 CINS            : 1;
__REG32                 : 1;
__REG32 CDPL            : 1;
__REG32 WPSPL           : 1;
__REG32                 : 3;
__REG32 CLSL            : 1;
__REG32 DLSL            : 8;
} __esdhc_prsstat_bits;

/* Protocol Control Register */
typedef struct {
__REG32 LCTL            : 1;
__REG32 DTW             : 2;
__REG32 D3CD            : 1;
__REG32 EMODE           : 2;
__REG32 CDTL            : 1;
__REG32 CDSS            : 1;
__REG32 DMAS            : 2;
__REG32                 : 6;
__REG32 SABGREQ         : 1;
__REG32 CREQ            : 1;
__REG32 RWCTL           : 1;
__REG32 IABG            : 1;
__REG32                 : 4;
__REG32 WECINT          : 1;
__REG32 WECINS          : 1;
__REG32 WECRM           : 1;
__REG32                 : 5;
} __esdhc_proctl_bits;

/* System Control Register */
typedef struct {
__REG32 IPGEN           : 1;
__REG32 HCKEN           : 1;
__REG32 PEREN           : 1;
__REG32 SDCLKEN         : 1;
__REG32 DVS             : 4;
__REG32 SDCLKFS         : 8;
__REG32 DTOCV           : 4;
__REG32                 : 4;
__REG32 RSTA            : 1;
__REG32 RSTC            : 1;
__REG32 RSTD            : 1;
__REG32 INITA           : 1;
__REG32                 : 4;
} __esdhc_sysctl_bits;

/* Interrupt Status Register */
typedef struct {
__REG32 CC              : 1;
__REG32 TC              : 1;
__REG32 BGE             : 1;
__REG32 DINT            : 1;
__REG32 BWR             : 1;
__REG32 BRR             : 1;
__REG32 CINS            : 1;
__REG32 CRM             : 1;
__REG32 CINT            : 1;
__REG32                 : 7;
__REG32 CTOE            : 1;
__REG32 CCE             : 1;
__REG32 CEBE            : 1;
__REG32 CIE             : 1;
__REG32 DTOE            : 1;
__REG32 DCE             : 1;
__REG32 DEBE            : 1;
__REG32                 : 1;
__REG32 AC12E           : 1;
__REG32                 : 3;
__REG32 DMAE            : 1;
__REG32                 : 3;
} __esdhc_irqstat_bits;

/* Interrupt Status Enable Register */
typedef struct {
__REG32 CCSEN           : 1;
__REG32 TCSEN           : 1;
__REG32 BGESEN          : 1;
__REG32 DINTSEN         : 1;
__REG32 BWRSEN          : 1;
__REG32 BRRSEN          : 1;
__REG32 CINSSEN         : 1;
__REG32 CRMSEN          : 1;
__REG32 CINTSEN         : 1;
__REG32                 : 7;
__REG32 CTOESEN         : 1;
__REG32 CCESEN          : 1;
__REG32 CEBESEN         : 1;
__REG32 CIESEN          : 1;
__REG32 DTOESEN         : 1;
__REG32 DCESEN          : 1;
__REG32 DEBESEN         : 1;
__REG32                 : 1;
__REG32 AC12ESEN        : 1;
__REG32                 : 3;
__REG32 DMAESEN         : 1;
__REG32                 : 3;
} __esdhc_irqstaten_bits;

/* Interrupt Signal Enable Register */
typedef struct {
__REG32 CCIEN           : 1;
__REG32 TCIEN           : 1;
__REG32 BGEIEN          : 1;
__REG32 DINTIEN         : 1;
__REG32 BWRIEN          : 1;
__REG32 BRRIEN          : 1;
__REG32 CINSIEN         : 1;
__REG32 CRMIEN          : 1;
__REG32 CINTIEN         : 1;
__REG32                 : 7;
__REG32 CTOEIEN         : 1;
__REG32 CCEIEN          : 1;
__REG32 CEBEIEN         : 1;
__REG32 CIEIEN          : 1;
__REG32 DTOEIEN         : 1;
__REG32 DCEIEN          : 1;
__REG32 DEBEIEN         : 1;
__REG32                 : 1;
__REG32 AC12EIEN        : 1;
__REG32                 : 3;
__REG32 DMAEIEN         : 1;
__REG32                 : 3;
} __esdhc_irqsigen_bits;

/* Auto CMD12 Error Status Register */
typedef struct {
__REG32 AC12NE          : 1;
__REG32 AC12TOE         : 1;
__REG32 AC12EBE         : 1;
__REG32 AC12CE          : 1;
__REG32 AC12IE          : 1;
__REG32                 : 2;
__REG32 CNIBAC12E       : 1;
__REG32                 :24;
} __esdhc_autoc12err_bits;

/* Host Controller Capabilities */
typedef struct {
__REG32                 :16;
__REG32 MBL             : 3;
__REG32                 : 1;
__REG32 ADMAS           : 1;
__REG32 HSS             : 1;
__REG32 DMAS            : 1;
__REG32 SRS             : 1;
__REG32 VS33            : 1;
__REG32 VS30            : 1;
__REG32 VS18            : 1;
__REG32                 : 5;
} __esdhc_hostcapblt_bits;

/* Host Controller Capabilities */
typedef struct {
__REG32 RD_WML          : 8;
__REG32 RD_BRST_LEN     : 5;
__REG32                 : 3;
__REG32 WR_WML          : 8;
__REG32 WR_BRST_LEN     : 5;
__REG32                 : 3;
} __esdhc_wml_bits;

/* Force Event Register */
typedef struct {
__REG32 FEVTAC12NE      : 1;
__REG32 FEVTAC12TOE     : 1;
__REG32 FEVTAC12CE      : 1;
__REG32 FEVTAC12EBE     : 1;
__REG32 FEVTAC12IE      : 1;
__REG32                 : 2;
__REG32 FEVTCNIBAC12E   : 1;
__REG32                 : 8;
__REG32 FEVTCTOE        : 1;
__REG32 FEVTCCE         : 1;
__REG32 FEVTCEBE        : 1;
__REG32 FEVTCIE         : 1;
__REG32 FEVTDTOE        : 1;
__REG32 FEVTDCE         : 1;
__REG32 FEVTDEBE        : 1;
__REG32                 : 1;
__REG32 FEVTAC12E       : 1;
__REG32                 : 3;
__REG32 FEVTDMAE        : 1;
__REG32                 : 2;
__REG32 FEVTCINT        : 1;
} __esdhc_fevt_bits;

/* ADMA Error Status Register */
typedef struct {
__REG32 ADMAES          : 2;
__REG32 ADMALME         : 1;
__REG32                 :29;
} __esdhc_admaes_bits;

/* Host Controller Version */
typedef struct {
__REG32 SVN             : 8;
__REG32 VVN             : 8;
__REG32                 :16;
} __esdhc_hostver_bits;

/* -------------------------------------------------------------------------*/
/*      FEC                                                                 */
/* -------------------------------------------------------------------------*/
/* Ethernet Interrupt Event Register (EIR)
   Interrupt Mask Register (EIMR) */
typedef struct {
__REG32          :19;
__REG32 UN       : 1;
__REG32 RL       : 1;
__REG32 LC       : 1;
__REG32 EBERR    : 1;
__REG32 MII      : 1;
__REG32 RXB      : 1;
__REG32 RXF      : 1;
__REG32 TXB      : 1;
__REG32 TXF      : 1;
__REG32 GRA      : 1;
__REG32 BABT     : 1;
__REG32 BABR     : 1;
__REG32 HBERR    : 1;
} __fec_eir_bits;

/* Receive Descriptor Active Register (RDAR) */
typedef struct {
__REG32               :24;
__REG32 RDAR          : 1;
__REG32               : 7;
} __fec_rdar_bits;

/* Transmit Descriptor Active Register (TDAR) */
typedef struct {
__REG32               :24;
__REG32 TDAR          : 1;
__REG32               : 7;
} __fec_tdar_bits;

/* Ethernet Control Register (ECR) */
typedef struct {
__REG32 RESET         : 1;
__REG32 ETHER_EN      : 1;
__REG32               :30;
} __fec_ecr_bits;

/* MII Management Frame Register (MMFR) */
typedef struct {
__REG32 DATA          :16;
__REG32 TA            : 2;
__REG32 RA            : 5;
__REG32 PA            : 5;
__REG32 OP            : 2;
__REG32 ST            : 2;
} __fec_mmfr_bits;

/* MII Speed Control Register (MSCR) */
typedef struct {
__REG32               : 1;
__REG32 MII_SPEED     : 6;
__REG32 DIS_PREAMBLE  : 1;
__REG32               :24;
} __fec_mscr_bits;

/* MIB Control Register (MIBC) */
typedef struct {
__REG32               :30;
__REG32 MIB_IDLE      : 1;
__REG32 MIB_DIS       : 1;
} __fec_mibc_bits;

/* Receive Control Register (RCR) */
typedef struct {
__REG32 LOOP          : 1;
__REG32 DRT           : 1;
__REG32 MII_MODE      : 1;
__REG32 PROM          : 1;
__REG32 BC_REJ        : 1;
__REG32 FCE           : 1;
__REG32               :10;
__REG32 MAX_FL        :11;
__REG32               : 5;
} __fec_rcr_bits;

/* Transmit Control Register (TCR) */
typedef struct {
__REG32 GTS           : 1;
__REG32 HBC           : 1;
__REG32 FDEN          : 1;
__REG32 TFC_PAUSE     : 1;
__REG32 RFC_PAUSE     : 1;
__REG32               :27;
} __fec_tcr_bits;

/* Physical Address High Register (PAUR) */
typedef struct {
__REG32 TYPE          :16;
__REG32 PADDR2        :16;
} __fec_paur_bits;

/* Opcode/Pause Duration Register (OPD) */
typedef struct {
__REG32 PAUSE_DUR     :16;
__REG32 OPCODE        :16;
} __fec_opd_bits;

/* FIFO Transmit FIFO Watermark Register (TFWR) */
typedef struct {
__REG32 WMRK          : 2;
__REG32               :30;
} __fec_tfwr_bits;

/* FIFO Receive Bound Register (FRBR) */
typedef struct {
__REG32               : 2;
__REG32 R_BOUND       : 8;
__REG32               :22;
} __fec_frbr_bits;

/* FIFO Receive Start Register (FRSR) */
typedef struct {
__REG32               : 2;
__REG32 R_FSTART      : 8;
__REG32               :22;
} __fec_frsr_bits;

/* Receive Buffer Size Register (EMRBR) */
typedef struct {
__REG32               : 4;
__REG32 R_BUF_SIZE    : 7;
__REG32               :21;
} __fec_emrbr_bits;

/* MIIGSK Configuration Register (MIIGSK_CFGR) */
typedef struct {
__REG32 IF_MODE       : 2;
__REG32               : 2;
__REG32 LBMODE        : 1;
__REG32               : 1;
__REG32 FRCONT        : 1;
__REG32               :25;
} __fec_miigsk_cfgr_bits;

/* MIIGSK Enable Register (MIIGSK_ENR) */
typedef struct {
__REG32               : 1;
__REG32 EN            : 1;
__REG32 READY         : 1;
__REG32               :29;
} __fec_miigsk_enr_bits;

/* -------------------------------------------------------------------------*/
/*               Controller Area Network (FlexCAN)                          */
/* -------------------------------------------------------------------------*/
/* Module Configuration Register (MCR) */
typedef struct {
__REG32 MAXMB           : 6;
__REG32                 : 2;
__REG32 IDAM            : 2;
__REG32                 : 2;
__REG32 AEN             : 1;
__REG32 LPRO_EN         : 1;
__REG32                 : 2;
__REG32 BCC             : 1;
__REG32 SRX_DIS         : 1;
__REG32 DOZE            : 1;
__REG32 WAK_SRC         : 1;
__REG32 LPM_ACK         : 1;
__REG32 WRN_EN          : 1;
__REG32 SLF_WAK         : 1;
__REG32 SUPV            : 1;
__REG32 FRZ_ACK         : 1;
__REG32 SOFT_RST        : 1;
__REG32 WAK_MSK         : 1;
__REG32 NOT_RDY         : 1;
__REG32 HALT            : 1;
__REG32 FEN             : 1;
__REG32 FRZ             : 1;
__REG32 MDIS            : 1;
} __can_mcr_bits;

/* Control Register (CTRL) */
typedef struct {
__REG32 PROPSEG         : 3;
__REG32 LOM             : 1;
__REG32 LBUF            : 1;
__REG32 TSYN            : 1;
__REG32 BOFF_REC        : 1;
__REG32 SMP             : 1;
__REG32                 : 2;
__REG32 RWRN_MSK        : 1;
__REG32 TWRN_MSK        : 1;
__REG32 LPB             : 1;
__REG32 CLK_SRC         : 1;
__REG32 ERR_MSK         : 1;
__REG32 BOFF_MSK        : 1;
__REG32 PSEG2           : 3;
__REG32 PSEG1           : 3;
__REG32 RJW             : 2;
__REG32 PRESDIV         : 8;
} __can_ctrl_bits;

/* Free Running Timer (TIMER) */
typedef struct {
__REG32 TIMER           :16;
__REG32                 :16;
} __can_timer_bits;

/* Rx Global Mask (RXGMASK) */
typedef struct {
__REG32 MI0             : 1;
__REG32 MI1             : 1;
__REG32 MI2             : 1;
__REG32 MI3             : 1;
__REG32 MI4             : 1;
__REG32 MI5             : 1;
__REG32 MI6             : 1;
__REG32 MI7             : 1;
__REG32 MI8             : 1;
__REG32 MI9             : 1;
__REG32 MI10            : 1;
__REG32 MI11            : 1;
__REG32 MI12            : 1;
__REG32 MI13            : 1;
__REG32 MI14            : 1;
__REG32 MI15            : 1;
__REG32 MI16            : 1;
__REG32 MI17            : 1;
__REG32 MI18            : 1;
__REG32 MI19            : 1;
__REG32 MI20            : 1;
__REG32 MI21            : 1;
__REG32 MI22            : 1;
__REG32 MI23            : 1;
__REG32 MI24            : 1;
__REG32 MI25            : 1;
__REG32 MI26            : 1;
__REG32 MI27            : 1;
__REG32 MI28            : 1;
__REG32 MI29            : 1;
__REG32 MI30            : 1;
__REG32 MI31            : 1;
} __can_rxgmask_bits;

/* Error Counter Register (ECR) */
typedef struct {
__REG32 Tx_Err_Counter  : 8;
__REG32 Rx_Err_Counter  : 8;
__REG32                 :16;
} __can_ecr_bits;

/* Error and Status Register (ESR) */
typedef struct {
__REG32 WAK_INT         : 1;
__REG32 ERR_INT         : 1;
__REG32 BOFF_INT        : 1;
__REG32                 : 1;
__REG32 FLT_CONF        : 2;
__REG32 TXRX            : 1;
__REG32 IDLE            : 1;
__REG32 RX_WRN          : 1;
__REG32 TX_WRN          : 1;
__REG32 STF_ERR         : 1;
__REG32 FRM_ERR         : 1;
__REG32 CRC_ERR         : 1;
__REG32 ACK_ERR         : 1;
__REG32 BIT0_ERR        : 1;
__REG32 BIT1_ERR        : 1;
__REG32 RWRN_INT        : 1;
__REG32 TWRN_INT        : 1;
__REG32                 :14;
} __can_esr_bits;

/* Interrupt Masks 2 Register (IMASK2) */
typedef struct {
__REG32 BUF32M          : 1;
__REG32 BUF33M          : 1;
__REG32 BUF34M          : 1;
__REG32 BUF35M          : 1;
__REG32 BUF36M          : 1;
__REG32 BUF37M          : 1;
__REG32 BUF38M          : 1;
__REG32 BUF39M          : 1;
__REG32 BUF40M          : 1;
__REG32 BUF41M          : 1;
__REG32 BUF42M          : 1;
__REG32 BUF43M          : 1;
__REG32 BUF44M          : 1;
__REG32 BUF45M          : 1;
__REG32 BUF46M          : 1;
__REG32 BUF47M          : 1;
__REG32 BUF48M          : 1;
__REG32 BUF49M          : 1;
__REG32 BUF50M          : 1;
__REG32 BUF51M          : 1;
__REG32 BUF52M          : 1;
__REG32 BUF53M          : 1;
__REG32 BUF54M          : 1;
__REG32 BUF55M          : 1;
__REG32 BUF56M          : 1;
__REG32 BUF57M          : 1;
__REG32 BUF58M          : 1;
__REG32 BUF59M          : 1;
__REG32 BUF60M          : 1;
__REG32 BUF61M          : 1;
__REG32 BUF62M          : 1;
__REG32 BUF63M          : 1;
} __can_imask2_bits;

/* Interrupt Masks 1 Register (IMASK1) */
typedef struct {
__REG32 BUF0M           : 1;
__REG32 BUF1M           : 1;
__REG32 BUF2M           : 1;
__REG32 BUF3M           : 1;
__REG32 BUF4M           : 1;
__REG32 BUF5M           : 1;
__REG32 BUF6M           : 1;
__REG32 BUF7M           : 1;
__REG32 BUF8M           : 1;
__REG32 BUF9M           : 1;
__REG32 BUF10M          : 1;
__REG32 BUF11M          : 1;
__REG32 BUF12M          : 1;
__REG32 BUF13M          : 1;
__REG32 BUF14M          : 1;
__REG32 BUF15M          : 1;
__REG32 BUF16M          : 1;
__REG32 BUF17M          : 1;
__REG32 BUF18M          : 1;
__REG32 BUF19M          : 1;
__REG32 BUF20M          : 1;
__REG32 BUF21M          : 1;
__REG32 BUF22M          : 1;
__REG32 BUF23M          : 1;
__REG32 BUF24M          : 1;
__REG32 BUF25M          : 1;
__REG32 BUF26M          : 1;
__REG32 BUF27M          : 1;
__REG32 BUF28M          : 1;
__REG32 BUF29M          : 1;
__REG32 BUF30M          : 1;
__REG32 BUF31M          : 1;
} __can_imask1_bits;

/* Interrupt Flags 2 Register (IFLAG2) */
typedef struct {
__REG32 BUF32I          : 1;
__REG32 BUF33I          : 1;
__REG32 BUF34I          : 1;
__REG32 BUF35I          : 1;
__REG32 BUF36I          : 1;
__REG32 BUF37I          : 1;
__REG32 BUF38I          : 1;
__REG32 BUF39I          : 1;
__REG32 BUF40I          : 1;
__REG32 BUF41I          : 1;
__REG32 BUF42I          : 1;
__REG32 BUF43I          : 1;
__REG32 BUF44I          : 1;
__REG32 BUF45I          : 1;
__REG32 BUF46I          : 1;
__REG32 BUF47I          : 1;
__REG32 BUF48I          : 1;
__REG32 BUF49I          : 1;
__REG32 BUF50I          : 1;
__REG32 BUF51I          : 1;
__REG32 BUF52I          : 1;
__REG32 BUF53I          : 1;
__REG32 BUF54I          : 1;
__REG32 BUF55I          : 1;
__REG32 BUF56I          : 1;
__REG32 BUF57I          : 1;
__REG32 BUF58I          : 1;
__REG32 BUF59I          : 1;
__REG32 BUF60I          : 1;
__REG32 BUF61I          : 1;
__REG32 BUF62I          : 1;
__REG32 BUF63I          : 1;
} __can_iflag2_bits;

/* Interrupt Flags 2 Register (IFLAG1) */
typedef struct {
__REG32 BUF0I           : 1;
__REG32 BUF1I           : 1;
__REG32 BUF2I           : 1;
__REG32 BUF3I           : 1;
__REG32 BUF4I           : 1;
__REG32 BUF5I           : 1;
__REG32 BUF6I           : 1;
__REG32 BUF7I           : 1;
__REG32 BUF8I           : 1;
__REG32 BUF9I           : 1;
__REG32 BUF10I          : 1;
__REG32 BUF11I          : 1;
__REG32 BUF12I          : 1;
__REG32 BUF13I          : 1;
__REG32 BUF14I          : 1;
__REG32 BUF15I          : 1;
__REG32 BUF16I          : 1;
__REG32 BUF17I          : 1;
__REG32 BUF18I          : 1;
__REG32 BUF19I          : 1;
__REG32 BUF20I          : 1;
__REG32 BUF21I          : 1;
__REG32 BUF22I          : 1;
__REG32 BUF23I          : 1;
__REG32 BUF24I          : 1;
__REG32 BUF25I          : 1;
__REG32 BUF26I          : 1;
__REG32 BUF27I          : 1;
__REG32 BUF28I          : 1;
__REG32 BUF29I          : 1;
__REG32 BUF30I          : 1;
__REG32 BUF31I          : 1;
} __can_iflag1_bits;

/* -------------------------------------------------------------------------*/
/*              General Purpose Input/Output (GPIO)                         */
/* -------------------------------------------------------------------------*/
/* GPIO Data Register (DR) */
typedef struct {
__REG32 DR0             : 1;
__REG32 DR1             : 1;
__REG32 DR2             : 1;
__REG32 DR3             : 1;
__REG32 DR4             : 1;
__REG32 DR5             : 1;
__REG32 DR6             : 1;
__REG32 DR7             : 1;
__REG32 DR8             : 1;
__REG32 DR9             : 1;
__REG32 DR10            : 1;
__REG32 DR11            : 1;
__REG32 DR12            : 1;
__REG32 DR13            : 1;
__REG32 DR14            : 1;
__REG32 DR15            : 1;
__REG32 DR16            : 1;
__REG32 DR17            : 1;
__REG32 DR18            : 1;
__REG32 DR19            : 1;
__REG32 DR20            : 1;
__REG32 DR21            : 1;
__REG32 DR22            : 1;
__REG32 DR23            : 1;
__REG32 DR24            : 1;
__REG32 DR25            : 1;
__REG32 DR26            : 1;
__REG32 DR27            : 1;
__REG32 DR28            : 1;
__REG32 DR29            : 1;
__REG32 DR30            : 1;
__REG32 DR31            : 1;
} __gpio_dr_bits;

/* GPIO Direction Register (GDIR) */
typedef struct {
__REG32 GDIR0           : 1;
__REG32 GDIR1           : 1;
__REG32 GDIR2           : 1;
__REG32 GDIR3           : 1;
__REG32 GDIR4           : 1;
__REG32 GDIR5           : 1;
__REG32 GDIR6           : 1;
__REG32 GDIR7           : 1;
__REG32 GDIR8           : 1;
__REG32 GDIR9           : 1;
__REG32 GDIR10          : 1;
__REG32 GDIR11          : 1;
__REG32 GDIR12          : 1;
__REG32 GDIR13          : 1;
__REG32 GDIR14          : 1;
__REG32 GDIR15          : 1;
__REG32 GDIR16          : 1;
__REG32 GDIR17          : 1;
__REG32 GDIR18          : 1;
__REG32 GDIR19          : 1;
__REG32 GDIR20          : 1;
__REG32 GDIR21          : 1;
__REG32 GDIR22          : 1;
__REG32 GDIR23          : 1;
__REG32 GDIR24          : 1;
__REG32 GDIR25          : 1;
__REG32 GDIR26          : 1;
__REG32 GDIR27          : 1;
__REG32 GDIR28          : 1;
__REG32 GDIR29          : 1;
__REG32 GDIR30          : 1;
__REG32 GDIR31          : 1;
} __gpio_gdir_bits;

/* GPIO Pad Status Register (PSR) */
typedef struct {
__REG32 PSR0            : 1;
__REG32 PSR1            : 1;
__REG32 PSR2            : 1;
__REG32 PSR3            : 1;
__REG32 PSR4            : 1;
__REG32 PSR5            : 1;
__REG32 PSR6            : 1;
__REG32 PSR7            : 1;
__REG32 PSR8            : 1;
__REG32 PSR9            : 1;
__REG32 PSR10           : 1;
__REG32 PSR11           : 1;
__REG32 PSR12           : 1;
__REG32 PSR13           : 1;
__REG32 PSR14           : 1;
__REG32 PSR15           : 1;
__REG32 PSR16           : 1;
__REG32 PSR17           : 1;
__REG32 PSR18           : 1;
__REG32 PSR19           : 1;
__REG32 PSR20           : 1;
__REG32 PSR21           : 1;
__REG32 PSR22           : 1;
__REG32 PSR23           : 1;
__REG32 PSR24           : 1;
__REG32 PSR25           : 1;
__REG32 PSR26           : 1;
__REG32 PSR27           : 1;
__REG32 PSR28           : 1;
__REG32 PSR29           : 1;
__REG32 PSR30           : 1;
__REG32 PSR31           : 1;
} __gpio_psr_bits;

/* GPIO Interrupt Configuration Register1 (ICR1) */
typedef struct {
__REG32 ICR0            : 2;
__REG32 ICR1            : 2;
__REG32 ICR2            : 2;
__REG32 ICR3            : 2;
__REG32 ICR4            : 2;
__REG32 ICR5            : 2;
__REG32 ICR6            : 2;
__REG32 ICR7            : 2;
__REG32 ICR8            : 2;
__REG32 ICR9            : 2;
__REG32 ICR10           : 2;
__REG32 ICR11           : 2;
__REG32 ICR12           : 2;
__REG32 ICR13           : 2;
__REG32 ICR14           : 2;
__REG32 ICR15           : 2;
} __gpio_icr1_bits;

/* GPIO Interrupt Configuration Register2 (ICR2) */
typedef struct {
__REG32 ICR16           : 2;
__REG32 ICR17           : 2;
__REG32 ICR18           : 2;
__REG32 ICR19           : 2;
__REG32 ICR20           : 2;
__REG32 ICR21           : 2;
__REG32 ICR22           : 2;
__REG32 ICR23           : 2;
__REG32 ICR24           : 2;
__REG32 ICR25           : 2;
__REG32 ICR26           : 2;
__REG32 ICR27           : 2;
__REG32 ICR28           : 2;
__REG32 ICR29           : 2;
__REG32 ICR30           : 2;
__REG32 ICR31           : 2;
} __gpio_icr2_bits;

/* GPIO Interrupt Mask Register (IMR) */
typedef struct {
__REG32 IMR0            : 1;
__REG32 IMR1            : 1;
__REG32 IMR2            : 1;
__REG32 IMR3            : 1;
__REG32 IMR4            : 1;
__REG32 IMR5            : 1;
__REG32 IMR6            : 1;
__REG32 IMR7            : 1;
__REG32 IMR8            : 1;
__REG32 IMR9            : 1;
__REG32 IMR10           : 1;
__REG32 IMR11           : 1;
__REG32 IMR12           : 1;
__REG32 IMR13           : 1;
__REG32 IMR14           : 1;
__REG32 IMR15           : 1;
__REG32 IMR16           : 1;
__REG32 IMR17           : 1;
__REG32 IMR18           : 1;
__REG32 IMR19           : 1;
__REG32 IMR20           : 1;
__REG32 IMR21           : 1;
__REG32 IMR22           : 1;
__REG32 IMR23           : 1;
__REG32 IMR24           : 1;
__REG32 IMR25           : 1;
__REG32 IMR26           : 1;
__REG32 IMR27           : 1;
__REG32 IMR28           : 1;
__REG32 IMR29           : 1;
__REG32 IMR30           : 1;
__REG32 IMR31           : 1;
} __gpio_imr_bits;

/* GPIO Interrupt Status Register (ISR) */
typedef struct {
__REG32 ISR0            : 1;
__REG32 ISR1            : 1;
__REG32 ISR2            : 1;
__REG32 ISR3            : 1;
__REG32 ISR4            : 1;
__REG32 ISR5            : 1;
__REG32 ISR6            : 1;
__REG32 ISR7            : 1;
__REG32 ISR8            : 1;
__REG32 ISR9            : 1;
__REG32 ISR10           : 1;
__REG32 ISR11           : 1;
__REG32 ISR12           : 1;
__REG32 ISR13           : 1;
__REG32 ISR14           : 1;
__REG32 ISR15           : 1;
__REG32 ISR16           : 1;
__REG32 ISR17           : 1;
__REG32 ISR18           : 1;
__REG32 ISR19           : 1;
__REG32 ISR20           : 1;
__REG32 ISR21           : 1;
__REG32 ISR22           : 1;
__REG32 ISR23           : 1;
__REG32 ISR24           : 1;
__REG32 ISR25           : 1;
__REG32 ISR26           : 1;
__REG32 ISR27           : 1;
__REG32 ISR28           : 1;
__REG32 ISR29           : 1;
__REG32 ISR30           : 1;
__REG32 ISR31           : 1;
} __gpio_isr_bits;

/* GPIO Edge Select Register (EDGE_SEL) */
typedef struct {
__REG32 EDGE_SEL0            : 1;
__REG32 EDGE_SEL1            : 1;
__REG32 EDGE_SEL2            : 1;
__REG32 EDGE_SEL3            : 1;
__REG32 EDGE_SEL4            : 1;
__REG32 EDGE_SEL5            : 1;
__REG32 EDGE_SEL6            : 1;
__REG32 EDGE_SEL7            : 1;
__REG32 EDGE_SEL8            : 1;
__REG32 EDGE_SEL9            : 1;
__REG32 EDGE_SEL10           : 1;
__REG32 EDGE_SEL11           : 1;
__REG32 EDGE_SEL12           : 1;
__REG32 EDGE_SEL13           : 1;
__REG32 EDGE_SEL14           : 1;
__REG32 EDGE_SEL15           : 1;
__REG32 EDGE_SEL16           : 1;
__REG32 EDGE_SEL17           : 1;
__REG32 EDGE_SEL18           : 1;
__REG32 EDGE_SEL19           : 1;
__REG32 EDGE_SEL20           : 1;
__REG32 EDGE_SEL21           : 1;
__REG32 EDGE_SEL22           : 1;
__REG32 EDGE_SEL23           : 1;
__REG32 EDGE_SEL24           : 1;
__REG32 EDGE_SEL25           : 1;
__REG32 EDGE_SEL26           : 1;
__REG32 EDGE_SEL27           : 1;
__REG32 EDGE_SEL28           : 1;
__REG32 EDGE_SEL29           : 1;
__REG32 EDGE_SEL30           : 1;
__REG32 EDGE_SEL31           : 1;
} __gpio_edge_sel_bits;

/* -------------------------------------------------------------------------*/
/*               System control registers                                   */
/* -------------------------------------------------------------------------*/
/* GPT Control Register (GPTCR) */
typedef struct {
__REG32 EN              : 1;
__REG32 ENMOD           : 1;
__REG32 DBGEN           : 1;
__REG32 WAITEN          : 1;
__REG32                 : 1;
__REG32 STOPEN          : 1;
__REG32 CLKSRC          : 3;
__REG32 FRR             : 1;
__REG32                 : 5;
__REG32 SWR             : 1;
__REG32 IM1             : 2;
__REG32 IM2             : 2;
__REG32 OM1             : 3;
__REG32 OM2             : 3;
__REG32 OM3             : 3;
__REG32 FO1             : 1;
__REG32 FO2             : 1;
__REG32 FO3             : 1;
} __gptcr_bits;

/* GPT Prescaler Register (GPTPR) */
typedef struct {
__REG32 PRESCALER       :12;
__REG32                 :20;
} __gptpr_bits;

/* GPT Status Register (GPTSR) */
typedef struct {
__REG32 OF1             : 1;
__REG32 OF2             : 1;
__REG32 OF3             : 1;
__REG32 IF1             : 1;
__REG32 IF2             : 1;
__REG32 ROV             : 1;
__REG32                 :26;
} __gptsr_bits;

/* GPT Interrupt Register (GPTIR) */
typedef struct {
__REG32 OF1IE           : 1;
__REG32 OF2IE           : 1;
__REG32 OF3IE           : 1;
__REG32 IF1IE           : 1;
__REG32 IF2IE           : 1;
__REG32 ROVIE           : 1;
__REG32                 :26;
} __gptir_bits;

/* -------------------------------------------------------------------------*/
/*               I2C registers                                              */
/* -------------------------------------------------------------------------*/
typedef struct {        /* I2C Address Register  */
__REG16      : 1;      /* Bit  0       - reserved*/
__REG16 ADR  : 7;      /* Bits 1  - 7  - Slave Address - Contains the specific slave address to be used by the I2C module.*/
__REG16      : 8;      /* Bits 31 - 8  - Reserved*/
} __iadr_bits;

typedef struct {        /* I2C Frequency Divider Register (IFDR) */
__REG16 IC  : 6;       /* Bits 0  - 5   - I2C Clock Rate Divider - Prescales the clock for bit-rate selection.*/
__REG16     :10;       /* Bits 6  - 31  - Reserved*/
} __ifdr_bits;

typedef struct {        /* I2C Control Register (I2CR) */
__REG16       : 2;     /* Bits 0  - 1  - Reserved*/
__REG16 RSTA  : 1;     /* Bit  2       - Repeated START - Generates a repeated START condition*/
__REG16 TXAK  : 1;     /* Bit  3       - Transmit Acknowledge Enable (0 = Send ACK, 1 = Dont send ACK)*/
__REG16 MTX   : 1;     /* Bit  4       - Transmit/Receive Mode Select (0 = Rx, 1 = Tx)*/
__REG16 MSTA  : 1;     /* Bit  5       - Master/Slave Mode Select (0 = Slave, 1 = Master)*/
__REG16 IIEN  : 1;     /* Bit  6       - I2C Interrupt Enable*/
__REG16 IEN   : 1;     /* Bit  7       - I2C Enable*/
__REG16       : 8;     /* Bits 8 - 31  - Reserved*/
} __i2cr_bits;

typedef struct {        /* I2C Status Register (I2SR) */
__REG16 RXAK  : 1;     /* Bit  0       - Received Acknowledge (0 = ACK received, 1 = No ACK received)*/
__REG16 IIF   : 1;     /* Bit  1       - I2C interrupt - (0 = No Int. pending, 1 = Interrupt pending )*/
__REG16 SRW   : 1;     /* Bit  2       - Slave Read/Write - Indicates the value of the R/W command bit*/
__REG16       : 1;     /* Bit  3       - Reserved*/
__REG16 IAL   : 1;     /* Bit  4       - Arbitration Lost*/
__REG16 IBB   : 1;     /* Bit  5       - I2C Bus Busy*/
__REG16 IAAS  : 1;     /* Bit  6       - I2C Addressed As a Slave*/
__REG16 ICF   : 1;     /* Bit  7       - Data Transfer (0=In Progress, 1 = Complete)*/
__REG16       : 8;     /* Bits 8  - 31 - Reserved*/
} __i2sr_bits;

typedef struct {        /* I2C Data I/O Register (I2DR) */
__REG16 DATA  : 8;     /* Bits 0  - 7  - I2C Data to be transmitted / last byte received*/
__REG16       : 8;     /* Bits 8 - 31  - Reserved*/
} __i2dr_bits;

/* -------------------------------------------------------------------------*/
/*      IC Identification (IIM)                                                           */
/* -------------------------------------------------------------------------*/
/* Status Register (STAT) */
typedef struct {
__REG8  SNSD  : 1;
__REG8  PRGD  : 1;
__REG8        : 5;
__REG8  BUSY  : 1;
} __iim_stat_bits;

/* Status IRQ Mask (STATM) */
typedef struct {
__REG8  SNSD_M  : 1;
__REG8  PRGD_M  : 1;
__REG8          : 6;
} __iim_statm_bits;

/* Module Errors Register (ERR) */
typedef struct {
__REG8            : 1;
__REG8  PARITYE   : 1;
__REG8  SNSE      : 1;
__REG8  WLRE      : 1;
__REG8  RPE       : 1;
__REG8  OPE       : 1;
__REG8  WPE       : 1;
__REG8  PRGE      : 1;
} __iim_err_bits;

/* Error IRQ Mask Register (EMASK) */
typedef struct {
__REG8            : 1;
__REG8  PARITYE_M : 1;
__REG8  SNSE_M    : 1;
__REG8  WLRE_M    : 1;
__REG8  RPE_M     : 1;
__REG8  OPE_M     : 1;
__REG8  WPE_M     : 1;
__REG8  PRGE_M    : 1;
} __iim_emask_bits;

/* Fuse Control Register (FCTL) */
typedef struct {
__REG8  PRG         : 1;
__REG8  ESNS_1      : 1;
__REG8  ESNS_0      : 1;
__REG8  ESNS_N      : 1;
__REG8  PRG_LENGTH  : 3;
__REG8  DPC         : 1;
} __iim_fctl_bits;

/* Upper Address (UA) */
typedef struct {
__REG8  A           : 6;
__REG8              : 2;
} __iim_ua_bits;

/* Product Revision (PREV) */
typedef struct {
__REG8  PROD_VT     : 3;
__REG8  PROD_REV    : 5;
} __iim_prev_bits;

/* Software-Controllable Signals Register 0 (SCS0) */
typedef struct {
__REG8  SCS         : 6;
__REG8  HAB_JDE     : 1;
__REG8  LOCK        : 1;
} __iim_scs0_bits;

/* Software-Controllable Signals Registers 1 (SCS1) */
typedef struct {
__REG8  SCS         : 7;
__REG8  LOCK        : 1;
} __iim_scs1_bits;

/* Software-Controllable Signals Registers 2/3 (SCS2/3) */
typedef struct {
__REG8  fbrl0       : 1;
__REG8  fbrl1       : 1;
__REG8  fbrl2       : 1;
__REG8  SCS         : 4;
__REG8  LOCK        : 1;
} __iim_scs2_bits;

/* Fuse Bank 0 Access Protection Register — (FBAC0) */
typedef struct {
__REG8  SAHARA_LOCK : 1;
__REG8              : 2;
__REG8  FBESP       : 1;
__REG8  FBSP        : 1;
__REG8  FBRP        : 1;
__REG8  FBOP        : 1;
__REG8  FBWP        : 1;
} __iim_fbac0_bits;

/* Word1 of Fusebank0 */
typedef struct {
__REG8  HAB_CUS       : 5;
__REG8  SCC_EN        : 1;
__REG8  JTAG_DISABLE  : 1;
__REG8  BOOT_INT      : 1;
} __iim_fb0_word1_bits;

/* Word2 of Fusebank0 */
typedef struct {
__REG8  HAB_TYPE      : 4;
__REG8  HAB_SRS       : 3;
__REG8  SHW_EN        : 1;
} __iim_fb0_word2_bits;

/* Word3 of Fusebank0 */
typedef struct {
__REG8  MSHC_DIS      : 1;
__REG8  CPFA          : 1;
__REG8  CPSPA         : 4;
__REG8  SAHARA_EN     : 1;
__REG8  C_M_DISABLE   : 1;
} __iim_fb0_word3_bits;

/* Word4 of Fusebank0 */
typedef struct {
__REG8  CLK_A926_CTRL : 1;
__REG8  MEM_A926_CTRL : 1;
__REG8  CLK_SCC_CTRL  : 1;
__REG8  MEM_SCC_CTRL  : 1;
__REG8  VER_ID        : 4;
} __iim_fb0_word4_bits;

/* Fuse Bank 1Access Protection Register — (FBAC1) */
typedef struct {
__REG8  MAC_ADDR_LOCK : 1;
__REG8                : 2;
__REG8  FBESP         : 1;
__REG8  FBSP          : 1;
__REG8  FBRP          : 1;
__REG8  FBOP          : 1;
__REG8  FBWP          : 1;
} __iim_fbac1_bits;

/* -------------------------------------------------------------------------*/
/*      Keypad Port (KPP)                                                   */
/* -------------------------------------------------------------------------*/
/* Keypad Control Register */
typedef struct{
__REG16 KRE  : 8;
__REG16 KCO  : 8;
} __kpcr_bits;

/* Keypad Status Register */
typedef struct{
__REG16 KPKD    : 1;
__REG16 KPKR    : 1;
__REG16 KDSC    : 1;
__REG16 KRSS    : 1;
__REG16         : 4;
__REG16 KDIE    : 1;
__REG16 KRIE    : 1;
__REG16 KPP_EN  : 1;
__REG16         : 5;
} __kpsr_bits;

/* Keypad Data Direction Register */
typedef struct{
__REG16 KRDD  : 8;
__REG16 KCDD  : 8;
} __kddr_bits;

/* Keypad Data Direction Register */
typedef struct{
__REG16 KRD  : 8;
__REG16 KCD  : 8;
} __kpdr_bits;

/* -------------------------------------------------------------------------*/
/*      LCDC Registers                                                       */
/* -------------------------------------------------------------------------*/
/* LCDC Size Register */
typedef struct{
__REG32 YMAX  :10;
__REG32       :10;
__REG32 XMAX  : 6;
__REG32       : 2;
__REG32 GWLPM : 1;
__REG32       : 3;
} __lsr_bits;

/* LCDC Virtual Page Width Register */
typedef struct{
__REG32 VPW  :10;
__REG32      :22;
} __lvpwr_bits;

/* LCDC Panel Configuration Register */
typedef struct{
__REG32 PCD        : 6;
__REG32 SHARP      : 1;
__REG32 SCLKSEL    : 1;
__REG32 ACD        : 7;
__REG32 ACD_SEL    : 1;
__REG32 REV_VS     : 1;
__REG32 SWAP_SEL   : 1;
__REG32 END_SEL    : 1;
__REG32 SCLKIDLE   : 1;
__REG32 OEPOL      : 1;
__REG32 CLKPOL     : 1;
__REG32 LPPOL      : 1;
__REG32 FLMPOL     : 1;
__REG32 PIXPOL     : 1;
__REG32 BPIX       : 3;
__REG32 PBSIZ      : 2;
__REG32 COLOR      : 1;
__REG32 TFT        : 1;
} __lpcr_bits;

/* LCDC Horizontal Configuration Register */
typedef struct{
__REG32 H_WAIT_2  : 8;
__REG32 H_WAIT_1  : 8;
__REG32           :10;
__REG32 H_WIDTH   : 6;
} __lhcr_bits;

/* LCDC Vertical Configuration Register */
typedef struct{
__REG32 V_WAIT_2  : 8;
__REG32 V_WAIT_1  : 8;
__REG32           :10;
__REG32 V_WIDTH   : 6;
} __lvcr_bits;

/* LCDC Panning Offset Register */
typedef struct{
__REG32 POS  : 5;
__REG32      :27;
} __lpor_bits;

/* LCDC Cursor Position Register */
typedef struct{
__REG32 CYP  :10;
__REG32      : 6;
__REG32 CXP  :10;
__REG32      : 2;
__REG32 OP   : 1;
__REG32      : 1;
__REG32 CC   : 2;
} __lcpr_bits;

/* LCDC Cursor Width Height and Blink Register */
typedef struct{
__REG32 BD     : 8;
__REG32        : 8;
__REG32 CH     : 5;
__REG32        : 3;
__REG32 CW     : 5;
__REG32        : 2;
__REG32 BK_EN  : 1;
} __lcwhb_bits;

/* LCDC Color Cursor Mapping Register */
typedef struct{
__REG32 CUR_COL_B  : 6;
__REG32 CUR_COL_G  : 6;
__REG32 CUR_COL_R  : 6;
__REG32            :14;
} __lccmr_bits;

/* LCDC Sharp Configuration Register */
typedef struct{
__REG32 GRAY1             : 4;
__REG32 GRAY2             : 4;
__REG32 REV_TOGGLE_DELAY  : 4;
__REG32                   : 4;
__REG32 CLS_RISE_DELAY    : 8;
__REG32                   : 2;
__REG32 PS_RISE_DELAY     : 6;
} __lscr_bits;

/* LCDC PWM Contrast Control Register */
typedef struct{
__REG32 PW            : 8;
__REG32 CC_EN         : 1;
__REG32 SCR           : 2;
__REG32               : 4;
__REG32 LDMSK         : 1;
__REG32 CLS_HI_WIDTH  : 9;
__REG32               : 7;
} __lpccr_bits;

/* LCDC Refresh Mode Control Register */
typedef struct{
__REG32 SELF_REF  : 1;
__REG32           :31;
} __lrmcr_bits;

/* LCDC Graphic Window DMA Control Register */
typedef struct{
__REG32 TM     : 7;
__REG32        : 9;
__REG32 HM     : 7;
__REG32        : 8;
__REG32 BURST  : 1;
} __ldcr_bits;

/* LCDC Interrupt Configuration Register */
typedef struct{
__REG32 INTCON      : 1;
__REG32             : 1;
__REG32 INTSYN      : 1;
__REG32             : 1;
__REG32 GW_INT_CON  : 1;
__REG32             :27;
} __licr_bits;

/* LCDC Interrupt Enable Register */
typedef struct{
__REG32 BOF_EN         : 1;
__REG32 EOF_EN         : 1;
__REG32 ERR_RES_EN     : 1;
__REG32 UDR_ERR_EN     : 1;
__REG32 GW_BOF_EN      : 1;
__REG32 GW_EOF_EN      : 1;
__REG32 GW_ERR_RES_EN  : 1;
__REG32 GW_UDR_ERR_EN  : 1;
__REG32                :24;
} __lier_bits;

/* LCDC Interrupt Status Register */
typedef struct{
__REG32 BOF         : 1;
__REG32 EOFR        : 1;
__REG32 ERR_RES     : 1;
__REG32 UDR_ERR     : 1;
__REG32 GW_BOF      : 1;
__REG32 GW_EOF      : 1;
__REG32 GW_ERR_RES  : 1;
__REG32 GW_UDR_ERR  : 1;
__REG32             :24;
} __lisr_bits;

/* LCDC Graphic Window Size Register */
typedef struct{
__REG32 GWH  :10;
__REG32      :10;
__REG32 GWW  : 6;
__REG32      : 6;
} __lgwsr_bits;

/* LCDC Graphic Window Virtual Page Width Register */
typedef struct{
__REG32 GWVPW  :10;
__REG32        :22;
} __lgwvpwr_bits;

/* LCDC Graphic Window Panning Offset Register */
typedef struct{
__REG32 GWPO  : 5;
__REG32       :27;
} __lgwpor_bits;

/* LCDC Graphic Window Position Register */
typedef struct{
__REG32 GWYP  :10;
__REG32       : 6;
__REG32 GWXP  :10;
__REG32       : 6;
} __lgwpr_bits;

/* LCDC Graphic Window Control Register */
typedef struct{
__REG32 GWCKB   : 6;
__REG32 GWCKG   : 6;
__REG32 GWCKR   : 6;
__REG32         : 3;
__REG32 GW_RVS  : 1;
__REG32 GWE     : 1;
__REG32 GWCKE   : 1;
__REG32 GWAV    : 8;
} __lgwcr_bits;

/* LCDC Graphic Window Graphic Window DMA Control Register */
typedef struct{
__REG32 GWTM  : 7;
__REG32       : 9;
__REG32 GWHM  : 7;
__REG32       : 8;
__REG32 GWBT  : 1;
} __lgwdcr_bits;

/* LCDC AUS Mode Control Register */
typedef struct{
__REG32 AGWCKB    : 8;
__REG32 AGWCKG    : 8;
__REG32 AGWCKR    : 8;
__REG32           : 7;
__REG32 AUS_MODE  : 1;
} __lauscr_bits;

/* LCDC AUS Mode Cursor Control Register */
typedef struct{
__REG32 ACUR_COL_B  : 8;
__REG32 ACUR_COL_G  : 8;
__REG32 ACUR_COL_R  : 8;
__REG32             : 8;
} __lausccr_bits;

/***************************************************************************
 **
 **  Multi-layer AHB Crossbar Switch (MAX)
 **
 ***************************************************************************/
typedef struct { /*           Master Priority Register for Slave Port*/
                 /* Alternate Master Priority Register for Slave Port*/
__REG32 MSTR_0  : 3;     /* Bits 0  - 2  - Master 0 Priority*/
__REG32         : 1;     /* Bit  3       - Reserved*/
__REG32 MSTR_1  : 3;     /* Bits 4  - 6  - Master 1 Priority*/
__REG32         : 1;     /* Bit  7       - Reserved*/
__REG32 MSTR_2  : 3;     /* Bits 8  - 10 - Master 2 Priority*/
__REG32         : 1;     /* Bit  11      - Reserved*/
__REG32 MSTR_3  : 3;     /* Bits 12 - 14 - Master 3 Priority*/
__REG32         : 1;     /* Bit  15      - Reserved*/
__REG32 MSTR_4  : 3;     /* Bits 16 - 18 - Master 4 Priority*/
__REG32         :13;     /* Bits 19 - 31 - Reserved*/
} __mpr_bits;

typedef struct { /* Slave General Purpose Control Register for Slave Port */
__REG32 PARK  : 3;     /* Bits 0  - 2  - PARK*/
__REG32       : 1;     /* Bit  3       - Reserved*/
__REG32 PCTL  : 2;     /* Bits 4  - 5  - Parking Control*/
__REG32       : 2;     /* Bits 6  - 7  - Reserved*/
__REG32 ARB   : 2;     /* Bits 8  - 9  - Arbitration Mode*/
__REG32       :20;     /* Bits 10 - 29 - Reserved*/
__REG32 HLP   : 1;     /* Bit  30      - Halt Low Priority*/
__REG32 RO    : 1;     /* Bit  31      - Read Only*/
} __sgpcr_bits;

typedef struct { /* Master General Purpose Control Register for Master Port*/
__REG32 AULB  : 3;     /* Bits 0  - 2  - Arbitrate on Undefined Length Bursts*/
__REG32       :29;     /* Bits 3  - 31 - Reserved*/
} __mgpcr_bits;

/* -------------------------------------------------------------------------*/
/*      PWM Registers                                                       */
/* -------------------------------------------------------------------------*/
/* PWM control register */
typedef struct{
__REG32 EN         : 1;
__REG32 REPEAT     : 2;
__REG32 SWR        : 1;
__REG32 PRESCALER  :12;
__REG32 CLKSRC     : 2;
__REG32 POUTC      : 2;
__REG32 HCTR       : 1;
__REG32 BCTR       : 1;
__REG32 DBGEN      : 1;
__REG32 WAITEN     : 1;
__REG32 DOZEN      : 1;
__REG32 STOPEN     : 1;
__REG32 FWM        : 2;
__REG32            : 4;
} __pwmcr_bits;

/* PWM Status Register */
typedef struct{
__REG32 FIFOAV     : 3;
__REG32 FE         : 1;
__REG32 ROV        : 1;
__REG32 CMP        : 1;
__REG32 FWE        : 1;
__REG32            :25;
} __pwmsr_bits;

/* PWM Interrupt Register */
typedef struct{
__REG32 FIE     : 1;
__REG32 RIE     : 1;
__REG32 CIE     : 1;
__REG32         :29;
} __pwmir_bits;

/* PWM Sample Register */
typedef struct{
__REG32 SAMPLE  :16;
__REG32         :16;
} __pwmsar_bits;

/* PWM Period Register */
typedef struct{
__REG32 PERIOD  :16;
__REG32         :16;
} __pwmpr_bits;

/* PWM Counter Register */
typedef struct{
__REG32 COUNT   :16;
__REG32         :16;
} __pwmcnr_bits;

/* -------------------------------------------------------------------------*/
/*               Smart Direct Memory Access (SDMA) Controller               */
/* -------------------------------------------------------------------------*/
/* Channel Interrupts (INTR) */
typedef struct {
__REG32 HI0       : 1;
__REG32 HI1       : 1;
__REG32 HI2       : 1;
__REG32 HI3       : 1;
__REG32 HI4       : 1;
__REG32 HI5       : 1;
__REG32 HI6       : 1;
__REG32 HI7       : 1;
__REG32 HI8       : 1;
__REG32 HI9       : 1;
__REG32 HI10      : 1;
__REG32 HI11      : 1;
__REG32 HI12      : 1;
__REG32 HI13      : 1;
__REG32 HI14      : 1;
__REG32 HI15      : 1;
__REG32 HI16      : 1;
__REG32 HI17      : 1;
__REG32 HI18      : 1;
__REG32 HI19      : 1;
__REG32 HI20      : 1;
__REG32 HI21      : 1;
__REG32 HI22      : 1;
__REG32 HI23      : 1;
__REG32 HI24      : 1;
__REG32 HI25      : 1;
__REG32 HI26      : 1;
__REG32 HI27      : 1;
__REG32 HI28      : 1;
__REG32 HI29      : 1;
__REG32 HI30      : 1;
__REG32 HI31      : 1;
} __sdma_intr_bits;

/* Channel Stop/Channel Status (STOP_STAT) */
typedef struct {
__REG32 HE0       : 1;
__REG32 HE1       : 1;
__REG32 HE2       : 1;
__REG32 HE3       : 1;
__REG32 HE4       : 1;
__REG32 HE5       : 1;
__REG32 HE6       : 1;
__REG32 HE7       : 1;
__REG32 HE8       : 1;
__REG32 HE9       : 1;
__REG32 HE10      : 1;
__REG32 HE11      : 1;
__REG32 HE12      : 1;
__REG32 HE13      : 1;
__REG32 HE14      : 1;
__REG32 HE15      : 1;
__REG32 HE16      : 1;
__REG32 HE17      : 1;
__REG32 HE18      : 1;
__REG32 HE19      : 1;
__REG32 HE20      : 1;
__REG32 HE21      : 1;
__REG32 HE22      : 1;
__REG32 HE23      : 1;
__REG32 HE24      : 1;
__REG32 HE25      : 1;
__REG32 HE26      : 1;
__REG32 HE27      : 1;
__REG32 HE28      : 1;
__REG32 HE29      : 1;
__REG32 HE30      : 1;
__REG32 HE31      : 1;
} __sdma_stop_stat_bits;

/* Channel Start (HSTART) */
typedef struct {
__REG32 HSTART0       : 1;
__REG32 HSTART1       : 1;
__REG32 HSTART2       : 1;
__REG32 HSTART3       : 1;
__REG32 HSTART4       : 1;
__REG32 HSTART5       : 1;
__REG32 HSTART6       : 1;
__REG32 HSTART7       : 1;
__REG32 HSTART8       : 1;
__REG32 HSTART9       : 1;
__REG32 HSTART10      : 1;
__REG32 HSTART11      : 1;
__REG32 HSTART12      : 1;
__REG32 HSTART13      : 1;
__REG32 HSTART14      : 1;
__REG32 HSTART15      : 1;
__REG32 HSTART16      : 1;
__REG32 HSTART17      : 1;
__REG32 HSTART18      : 1;
__REG32 HSTART19      : 1;
__REG32 HSTART20      : 1;
__REG32 HSTART21      : 1;
__REG32 HSTART22      : 1;
__REG32 HSTART23      : 1;
__REG32 HSTART24      : 1;
__REG32 HSTART25      : 1;
__REG32 HSTART26      : 1;
__REG32 HSTART27      : 1;
__REG32 HSTART28      : 1;
__REG32 HSTART29      : 1;
__REG32 HSTART30      : 1;
__REG32 HSTART31      : 1;
} __sdma_hstart_bits;

/* Channel Event Override (EVTOVR) */
typedef struct {
__REG32 EO0       : 1;
__REG32 EO1       : 1;
__REG32 EO2       : 1;
__REG32 EO3       : 1;
__REG32 EO4       : 1;
__REG32 EO5       : 1;
__REG32 EO6       : 1;
__REG32 EO7       : 1;
__REG32 EO8       : 1;
__REG32 EO9       : 1;
__REG32 EO10      : 1;
__REG32 EO11      : 1;
__REG32 EO12      : 1;
__REG32 EO13      : 1;
__REG32 EO14      : 1;
__REG32 EO15      : 1;
__REG32 EO16      : 1;
__REG32 EO17      : 1;
__REG32 EO18      : 1;
__REG32 EO19      : 1;
__REG32 EO20      : 1;
__REG32 EO21      : 1;
__REG32 EO22      : 1;
__REG32 EO23      : 1;
__REG32 EO24      : 1;
__REG32 EO25      : 1;
__REG32 EO26      : 1;
__REG32 EO27      : 1;
__REG32 EO28      : 1;
__REG32 EO29      : 1;
__REG32 EO30      : 1;
__REG32 EO31      : 1;
} __sdma_evtovr_bits;

/* Channel BP Override (DSPOVR) */
typedef struct {
__REG32 DO0       : 1;
__REG32 DO1       : 1;
__REG32 DO2       : 1;
__REG32 DO3       : 1;
__REG32 DO4       : 1;
__REG32 DO5       : 1;
__REG32 DO6       : 1;
__REG32 DO7       : 1;
__REG32 DO8       : 1;
__REG32 DO9       : 1;
__REG32 DO10      : 1;
__REG32 DO11      : 1;
__REG32 DO12      : 1;
__REG32 DO13      : 1;
__REG32 DO14      : 1;
__REG32 DO15      : 1;
__REG32 DO16      : 1;
__REG32 DO17      : 1;
__REG32 DO18      : 1;
__REG32 DO19      : 1;
__REG32 DO20      : 1;
__REG32 DO21      : 1;
__REG32 DO22      : 1;
__REG32 DO23      : 1;
__REG32 DO24      : 1;
__REG32 DO25      : 1;
__REG32 DO26      : 1;
__REG32 DO27      : 1;
__REG32 DO28      : 1;
__REG32 DO29      : 1;
__REG32 DO30      : 1;
__REG32 DO31      : 1;
} __sdma_dspovr_bits;

/* Channel AP Override (HOSTOVR) */
typedef struct {
__REG32 HO0       : 1;
__REG32 HO1       : 1;
__REG32 HO2       : 1;
__REG32 HO3       : 1;
__REG32 HO4       : 1;
__REG32 HO5       : 1;
__REG32 HO6       : 1;
__REG32 HO7       : 1;
__REG32 HO8       : 1;
__REG32 HO9       : 1;
__REG32 HO10      : 1;
__REG32 HO11      : 1;
__REG32 HO12      : 1;
__REG32 HO13      : 1;
__REG32 HO14      : 1;
__REG32 HO15      : 1;
__REG32 HO16      : 1;
__REG32 HO17      : 1;
__REG32 HO18      : 1;
__REG32 HO19      : 1;
__REG32 HO20      : 1;
__REG32 HO21      : 1;
__REG32 HO22      : 1;
__REG32 HO23      : 1;
__REG32 HO24      : 1;
__REG32 HO25      : 1;
__REG32 HO26      : 1;
__REG32 HO27      : 1;
__REG32 HO28      : 1;
__REG32 HO29      : 1;
__REG32 HO30      : 1;
__REG32 HO31      : 1;
} __sdma_hostovr_bits;

/* Channel Event Pending (EVTPEND) */
typedef struct {
__REG32 EP0       : 1;
__REG32 EP1       : 1;
__REG32 EP2       : 1;
__REG32 EP3       : 1;
__REG32 EP4       : 1;
__REG32 EP5       : 1;
__REG32 EP6       : 1;
__REG32 EP7       : 1;
__REG32 EP8       : 1;
__REG32 EP9       : 1;
__REG32 EP10      : 1;
__REG32 EP11      : 1;
__REG32 EP12      : 1;
__REG32 EP13      : 1;
__REG32 EP14      : 1;
__REG32 EP15      : 1;
__REG32 EP16      : 1;
__REG32 EP17      : 1;
__REG32 EP18      : 1;
__REG32 EP19      : 1;
__REG32 EP20      : 1;
__REG32 EP21      : 1;
__REG32 EP22      : 1;
__REG32 EP23      : 1;
__REG32 EP24      : 1;
__REG32 EP25      : 1;
__REG32 EP26      : 1;
__REG32 EP27      : 1;
__REG32 EP28      : 1;
__REG32 EP29      : 1;
__REG32 EP30      : 1;
__REG32 EP31      : 1;
} __sdma_evtpend_bits;

/* Reset Register (RESET) */
typedef struct {
__REG32 RESET     : 1;
__REG32 RESCHED   : 1;
__REG32           :30;
} __sdma_reset_bits;

/* DMA Request Error Register (EVTERR) */
typedef struct {
__REG32 CHNERR0       : 1;
__REG32 CHNERR1       : 1;
__REG32 CHNERR2       : 1;
__REG32 CHNERR3       : 1;
__REG32 CHNERR4       : 1;
__REG32 CHNERR5       : 1;
__REG32 CHNERR6       : 1;
__REG32 CHNERR7       : 1;
__REG32 CHNERR8       : 1;
__REG32 CHNERR9       : 1;
__REG32 CHNERR10      : 1;
__REG32 CHNERR11      : 1;
__REG32 CHNERR12      : 1;
__REG32 CHNERR13      : 1;
__REG32 CHNERR14      : 1;
__REG32 CHNERR15      : 1;
__REG32 CHNERR16      : 1;
__REG32 CHNERR17      : 1;
__REG32 CHNERR18      : 1;
__REG32 CHNERR19      : 1;
__REG32 CHNERR20      : 1;
__REG32 CHNERR21      : 1;
__REG32 CHNERR22      : 1;
__REG32 CHNERR23      : 1;
__REG32 CHNERR24      : 1;
__REG32 CHNERR25      : 1;
__REG32 CHNERR26      : 1;
__REG32 CHNERR27      : 1;
__REG32 CHNERR28      : 1;
__REG32 CHNERR29      : 1;
__REG32 CHNERR30      : 1;
__REG32 CHNERR31      : 1;
} __sdma_evterr_bits;

/* Channel AP Interrupt Mask Flags (INTRMASK) */
typedef struct {
__REG32 HIMASK0       : 1;
__REG32 HIMASK1       : 1;
__REG32 HIMASK2       : 1;
__REG32 HIMASK3       : 1;
__REG32 HIMASK4       : 1;
__REG32 HIMASK5       : 1;
__REG32 HIMASK6       : 1;
__REG32 HIMASK7       : 1;
__REG32 HIMASK8       : 1;
__REG32 HIMASK9       : 1;
__REG32 HIMASK10      : 1;
__REG32 HIMASK11      : 1;
__REG32 HIMASK12      : 1;
__REG32 HIMASK13      : 1;
__REG32 HIMASK14      : 1;
__REG32 HIMASK15      : 1;
__REG32 HIMASK16      : 1;
__REG32 HIMASK17      : 1;
__REG32 HIMASK18      : 1;
__REG32 HIMASK19      : 1;
__REG32 HIMASK20      : 1;
__REG32 HIMASK21      : 1;
__REG32 HIMASK22      : 1;
__REG32 HIMASK23      : 1;
__REG32 HIMASK24      : 1;
__REG32 HIMASK25      : 1;
__REG32 HIMASK26      : 1;
__REG32 HIMASK27      : 1;
__REG32 HIMASK28      : 1;
__REG32 HIMASK29      : 1;
__REG32 HIMASK30      : 1;
__REG32 HIMASK31      : 1;
} __sdma_intrmask_bits;

/* Schedule Status (PSW) */
typedef struct {
__REG32 CCR           : 5;
__REG32 CCP           : 3;
__REG32 NCR           : 5;
__REG32 NCP           : 3;
__REG32               :16;
} __sdma_psw_bits;

/* DMA Request Error Register for Debug (EVTERRDBG) */
typedef struct {
__REG32 CHNERR0       : 1;
__REG32 CHNERR1       : 1;
__REG32 CHNERR2       : 1;
__REG32 CHNERR3       : 1;
__REG32 CHNERR4       : 1;
__REG32 CHNERR5       : 1;
__REG32 CHNERR6       : 1;
__REG32 CHNERR7       : 1;
__REG32 CHNERR8       : 1;
__REG32 CHNERR9       : 1;
__REG32 CHNERR10      : 1;
__REG32 CHNERR11      : 1;
__REG32 CHNERR12      : 1;
__REG32 CHNERR13      : 1;
__REG32 CHNERR14      : 1;
__REG32 CHNERR15      : 1;
__REG32 CHNERR16      : 1;
__REG32 CHNERR17      : 1;
__REG32 CHNERR18      : 1;
__REG32 CHNERR19      : 1;
__REG32 CHNERR20      : 1;
__REG32 CHNERR21      : 1;
__REG32 CHNERR22      : 1;
__REG32 CHNERR23      : 1;
__REG32 CHNERR24      : 1;
__REG32 CHNERR25      : 1;
__REG32 CHNERR26      : 1;
__REG32 CHNERR27      : 1;
__REG32 CHNERR28      : 1;
__REG32 CHNERR29      : 1;
__REG32 CHNERR30      : 1;
__REG32 CHNERR31      : 1;
} __sdma_evterrdbg_bits;

/* Configuration Register (CONFIG) */
typedef struct {
__REG32 CSM           : 2;
__REG32               : 2;
__REG32 ACR           : 1;
__REG32               : 6;
__REG32 RTDOBS        : 1;
__REG32 DSPDMA        : 1;
__REG32               :19;
} __sdma_config_bits;

/* SDMA Lock Register (SDMA_LOCK) */
typedef struct {
__REG32 LOCK              : 1;
__REG32 SRESET_LOCK_CLR   : 1;
__REG32                   :30;
} __sdma_lock_bits;

/* OnCE Enable (ONCE_ENB) */
typedef struct {
__REG32 ENB               : 1;
__REG32                   :31;
} __sdma_once_enb_bits;

/* OnCE Instruction Register (ONCE_INSTR) */
typedef struct {
__REG32 INSTR             :16;
__REG32                   :16;
} __sdma_once_instr_bits;

/* OnCE Status Register (ONCE_STAT) */
typedef struct {
__REG32 ECDR              : 3;
__REG32                   : 4;
__REG32 MST               : 1;
__REG32 SWB               : 1;
__REG32 ODR               : 1;
__REG32 EDR               : 1;
__REG32 RCV               : 1;
__REG32 PST               : 4;
__REG32                   :16;
} __sdma_once_stat_bits;

/* OnCE Command Register (ONCE_CMD) */
typedef struct {
__REG32 CMD               : 4;
__REG32                   :28;
} __sdma_once_cmd_bits;

/* Illegal Instruction Trap Address (ILLINSTADDR) */
typedef struct {
__REG32 ILLINSTADDR       :14;
__REG32                   :18;
} __sdma_illinstaddr_bits;

/* Channel 0 Boot Address (CHN0ADDR) */
typedef struct {
__REG32 CHN0ADDR          :14;
__REG32 SMSZ              : 1;
__REG32                   :17;
} __sdma_chn0addr_bits;

/* DMA Requests (EVT_MIRROR) */
typedef struct {
__REG32 EVENTS0       : 1;
__REG32 EVENTS1       : 1;
__REG32 EVENTS2       : 1;
__REG32 EVENTS3       : 1;
__REG32 EVENTS4       : 1;
__REG32 EVENTS5       : 1;
__REG32 EVENTS6       : 1;
__REG32 EVENTS7       : 1;
__REG32 EVENTS8       : 1;
__REG32 EVENTS9       : 1;
__REG32 EVENTS10      : 1;
__REG32 EVENTS11      : 1;
__REG32 EVENTS12      : 1;
__REG32 EVENTS13      : 1;
__REG32 EVENTS14      : 1;
__REG32 EVENTS15      : 1;
__REG32 EVENTS16      : 1;
__REG32 EVENTS17      : 1;
__REG32 EVENTS18      : 1;
__REG32 EVENTS19      : 1;
__REG32 EVENTS20      : 1;
__REG32 EVENTS21      : 1;
__REG32 EVENTS22      : 1;
__REG32 EVENTS23      : 1;
__REG32 EVENTS24      : 1;
__REG32 EVENTS25      : 1;
__REG32 EVENTS26      : 1;
__REG32 EVENTS27      : 1;
__REG32 EVENTS28      : 1;
__REG32 EVENTS29      : 1;
__REG32 EVENTS30      : 1;
__REG32 EVENTS31      : 1;
} __sdma_evt_mirror_bits;

/* DMA Requests 2 (EVT_MIRROR2) */
typedef struct {
__REG32 EVENTS32      : 1;
__REG32 EVENTS33      : 1;
__REG32 EVENTS34      : 1;
__REG32 EVENTS35      : 1;
__REG32 EVENTS36      : 1;
__REG32 EVENTS37      : 1;
__REG32 EVENTS38      : 1;
__REG32 EVENTS39      : 1;
__REG32 EVENTS40      : 1;
__REG32 EVENTS41      : 1;
__REG32 EVENTS42      : 1;
__REG32 EVENTS43      : 1;
__REG32 EVENTS44      : 1;
__REG32 EVENTS45      : 1;
__REG32 EVENTS46      : 1;
__REG32 EVENTS47      : 1;
__REG32               :16;
} __sdma_evt_mirror2_bits;

/* Cross-Trigger Events Configuration Register (XTRIG_CONF1) */
typedef struct {
__REG32 NUM0          : 6;
__REG32 CNF0          : 1;
__REG32               : 1;
__REG32 NUM1          : 6;
__REG32 CNF1          : 1;
__REG32               : 1;
__REG32 NUM2          : 6;
__REG32 CNF2          : 1;
__REG32               : 1;
__REG32 NUM3          : 6;
__REG32 CNF3          : 1;
__REG32               : 1;
} __sdma_xtrig_conf1_bits;

/* Cross-Trigger Events Configuration Register (XTRIG_CONF2) */
typedef struct {
__REG32 NUM4          : 6;
__REG32 CNF4          : 1;
__REG32               : 1;
__REG32 NUM5          : 6;
__REG32 CNF5          : 1;
__REG32               : 1;
__REG32 NUM6          : 6;
__REG32 CNF6          : 1;
__REG32               : 1;
__REG32 NUM7          : 6;
__REG32 CNF7          : 1;
__REG32               : 1;
} __sdma_xtrig_conf2_bits;

/* Channel Priority Registers (CHNPRIn) */
typedef struct {
__REG32 CHNPRI        : 3;
__REG32               :29;
} __sdma_chnpri_bits;

/* Channel Enable RAM (CHNENBLn) */
typedef struct {
__REG32 ENBL0       : 1;
__REG32 ENBL1       : 1;
__REG32 ENBL2       : 1;
__REG32 ENBL3       : 1;
__REG32 ENBL4       : 1;
__REG32 ENBL5       : 1;
__REG32 ENBL6       : 1;
__REG32 ENBL7       : 1;
__REG32 ENBL8       : 1;
__REG32 ENBL9       : 1;
__REG32 ENBL10      : 1;
__REG32 ENBL11      : 1;
__REG32 ENBL12      : 1;
__REG32 ENBL13      : 1;
__REG32 ENBL14      : 1;
__REG32 ENBL15      : 1;
__REG32 ENBL16      : 1;
__REG32 ENBL17      : 1;
__REG32 ENBL18      : 1;
__REG32 ENBL19      : 1;
__REG32 ENBL20      : 1;
__REG32 ENBL21      : 1;
__REG32 ENBL22      : 1;
__REG32 ENBL23      : 1;
__REG32 ENBL24      : 1;
__REG32 ENBL25      : 1;
__REG32 ENBL26      : 1;
__REG32 ENBL27      : 1;
__REG32 ENBL28      : 1;
__REG32 ENBL29      : 1;
__REG32 ENBL30      : 1;
__REG32 ENBL31      : 1;
} __sdma_chnenbl_bits;

/* -------------------------------------------------------------------------*/
/*               SIM Registers                                              */
/* -------------------------------------------------------------------------*/

/* SIM Port0/1 Control Register (PORT0/1_CNTL) */
typedef struct {
  __REG32 SAPD             : 1;
  __REG32 SVEN             : 1;
  __REG32 STEN             : 1;
  __REG32 SRST             : 1;
  __REG32 SCEN             : 1;
  __REG32 SCSP             : 1;
  __REG32 _3VOLT           : 1;
  __REG32 SFPD             : 1;
  __REG32                  :24;
} __sim_port_cntl_bits;

/* SIM Setup Register (SETUP) */
typedef struct {
  __REG32 AMODE            : 1;
  __REG32 SPS              : 1;
  __REG32                  :30;
} __sim_setup_bits;

/* SIM Port0/1 Detect Register (PORT0/1_DETECT) */
typedef struct {
  __REG32 SDIM             : 1;
  __REG32 SDI              : 1;
  __REG32 SPDP             : 1;
  __REG32 SPDS             : 1;
  __REG32                  :28;
} __sim_port_detect_bits;

/* SIM Port0/1 Transmit Buffer Register (PORT0/1_XMT_BUF) */
typedef struct {
  __REG32 PORT_XMT         : 8;
  __REG32                  :24;
} __sim_port_xmt_buf_bits;

/* SIM Port0/1 Receive Buffer Register (PORT0/1_RCV_BUF) */
typedef struct {
  __REG32 PORT_RCV         : 8;
  __REG32 PE               : 1;
  __REG32 FE               : 1;
  __REG32 CWT              : 1;
  __REG32                  :21;
} __sim_port_rcv_buf_bits;

/* SIM Control Register (CNTL) */
typedef struct {
  __REG32                  : 1;
  __REG32 ICM              : 1;
  __REG32 ANACK            : 1;
  __REG32 ONACK            : 1;
  __REG32 SAMPLE12         : 1;
  __REG32                  : 1;
  __REG32 BAUD_SEL         : 3;
  __REG32 GPCNT_CLK_SEL    : 2;
  __REG32 CWTEN            : 1;
  __REG32 LRCEN            : 1;
  __REG32 CRCEN            : 1;
  __REG32 XMT_CRC_LRC      : 1;
  __REG32 BWTEN            : 1;
  __REG32                  :16;
} __sim_cntl_bits;

/* SIM Clock Prescaler Register (CLK_PRESCALER) */
typedef struct {
  __REG32 CLK_PRESCALER    : 8;
  __REG32                  :24;
} __sim_clk_prescaler_bits;

/* SIM Receive Threshold Register (RCV_THRESHOLD) */
typedef struct {
  __REG32 RDT              : 9;
  __REG32 RTH              : 4;
  __REG32                  :19;
} __sim_rcv_threshold_bits;

/* SIM Enable Register (ENABLE) */
typedef struct {
  __REG32 RCVEN            : 1;
  __REG32 XMTEN            : 1;
  __REG32                  :30;
} __sim_enable_bits;

/* SIM Transmit Status Register (XMT_STATUS) */
typedef struct {
  __REG32 XTE              : 1;
  __REG32                  : 2;
  __REG32 TFE              : 1;
  __REG32 ETC              : 1;
  __REG32 TC               : 1;
  __REG32 TFO              : 1;
  __REG32 TDTF             : 1;
  __REG32 GPCNT            : 1;
  __REG32                  :23;
} __sim_xmt_status_bits;

/* SIM Receive Status Register (RCV_STATUS) */
typedef struct {
  __REG32 OEF              : 1;
  __REG32                  : 3;
  __REG32 RFD              : 1;
  __REG32 RDRF             : 1;
  __REG32 LRCOK            : 1;
  __REG32 CRCOK            : 1;
  __REG32 CWT              : 1;
  __REG32 RTE              : 1;
  __REG32 BWT              : 1;
  __REG32 BGT              : 1;
  __REG32                  :20;
} __sim_rcv_status_bits;

/* SIM Interrupt Mask Register (INT_MASK) */
typedef struct {
  __REG32 RIM              : 1;
  __REG32 TCIM             : 1;
  __REG32 OIM              : 1;
  __REG32 ETCIM            : 1;
  __REG32 TFEIM            : 1;
  __REG32 XTM              : 1;
  __REG32 TFOM             : 1;
  __REG32 TDTFM            : 1;
  __REG32 GPCNTM           : 1;
  __REG32 CWTM             : 1;
  __REG32 RTM              : 1;
  __REG32 BWTM             : 1;
  __REG32 BGTM             : 1;
  __REG32                  :19;
} __sim_int_mask_bits;

/* SIM Data Format Register (DATA_FORMAT) */
typedef struct {
  __REG32 IC               : 1;
  __REG32                  :31;
} __sim_data_format_bits;

/* SIM Transmit Threshold Register (XMT_THRESHOLD) */
typedef struct {
  __REG32 TDT              : 4;
  __REG32 XTH              : 4;
  __REG32                  :24;
} __sim_xmt_threshold_bits;

/* SIM Transmit Guard Control Register (GUARD_CNTL) */
typedef struct {
  __REG32 GETU             : 8;
  __REG32 RCVR11           : 1;
  __REG32                  :23;
} __sim_guard_cntl_bits;

/* SIM Open Drain Configuration Control Register (OD_CONFIG) */
typedef struct {
  __REG32 OD_P0            : 1;
  __REG32 OD_P             : 1;
  __REG32                  :30;
} __sim_od_config_bits;

/* SIM Reset Control Register (RESET_CNTL) */
typedef struct {
  __REG32 FLUSH_RCV        : 1;
  __REG32 FLUSH_XMT        : 1;
  __REG32 SOFT_RST         : 1;
  __REG32 KILL_CLOCK       : 1;
  __REG32 DOZE             : 1;
  __REG32 STOP             : 1;
  __REG32 DBUG             : 1;
  __REG32                  :25;
} __sim_reset_cntl_bits;

/* SIM Character Wait Time Register (CHAR_WAIT) */
typedef struct {
  __REG32 CWT              :16;
  __REG32                  :16;
} __sim_char_wait_bits;

/* SIM General Purpose Counter Register (GPCNT) */
typedef struct {
  __REG32 GPCNT            :16;
  __REG32                  :16;
} __sim_gpcnt_bits;

/* SIM Divisor Register (DIVISOR) */
typedef struct {
  __REG32 DIVISOR          : 8;
  __REG32                  :24;
} __sim_divisor_bits;

/* SIM Block Wait Time Register (BWT) */
typedef struct {
  __REG32 BWT              :16;
  __REG32                  :16;
} __sim_bwt_bits;

/* SIM Block Guard Time Register (BGT) */
typedef struct {
  __REG32 BGT              :16;
  __REG32                  :16;
} __sim_bgt_bits;

/* SIM Block Wait Time Register HIGH (BWT_H) */
typedef struct {
  __REG32 BGT              :16;
  __REG32                  :16;
} __sim_bwt_h_bits;

/* SIM Transmit FIFO Status Register (XMT_FIFO_STAT) */
typedef struct {
  __REG32 XMT_RPTR         : 4;
  __REG32 XMT_WPTR         : 4;
  __REG32 XMT_CNT          : 4;
  __REG32                  :20;
} __sim_xmt_fifo_stat_bits;

/* SIM Receive FIFO Counter Register (RCV_FIFO_CNT) */
typedef struct {
  __REG32 RCV_CNT          : 9;
  __REG32                  :23;
} __sim_rcv_fifo_cnt_bits;

/* SIM Receive FIFO Write Pointer Register (RCV_FIFO_WPTR) */
typedef struct {
  __REG32 RCV_WPTR         : 9;
  __REG32                  :23;
} __sim_rcv_fifo_wptr_bits;

/* SIM Receive FIFO Read Pointer Register (RCV_FIFO_RPTR) */
typedef struct {
  __REG32 RCV_RPTR         : 9;
  __REG32                  :23;
} __sim_rcv_fifo_rptr_bits;

/* -------------------------------------------------------------------------*/
/*      Smart Liquid Crystal Display Controller (SLCDC)                     */
/* -------------------------------------------------------------------------*/
/* SLCDC Data Buffer Size Register */
typedef struct{
__REG32 DATABUFSIZE  :17;
__REG32              :15;
} __data_buff_size_bits;

/* SLCDC Command Buffer Size Register */
typedef struct{
__REG32 COMBUFSIZE  :17;
__REG32             :15;
} __cmd_buff_size_bits;

/* SLCDC Command String Size Register */
typedef struct{
__REG32 COMSTRINGSIZ  : 8;
__REG32               :24;
} __string_size_bits;

/* SLCDC FIFO Configuration Register */
typedef struct{
__REG32 BURST  : 3;
__REG32        :29;
} __fifo_config_bits;

/* SLCDC LCD Controller Configuration Register */
typedef struct{
__REG32 WORDPPAGE  :13;
__REG32            :19;
} __lcd_config_bits;

/* SLCDC LCD Transfer Configuration Register */
typedef struct{
__REG32 SKCPOL        : 1;
__REG32 CSPOL         : 1;
__REG32 XFRMODE       : 1;
__REG32 WORDDEFCOM    : 1;
__REG32 WORDDEFDAT    : 1;
__REG32 WORDDEFWRITE  : 1;
__REG32               :10;
__REG32 IMGEND        : 2;
__REG32               :14;
} __lcdtransconfig_bits;

/* SLCDC Control/Status Register */
typedef struct{
__REG32 GO        : 1;
__REG32 ABORT     : 1;
__REG32 BUSY      : 1;
__REG32           : 1;
__REG32 TEA       : 1;
__REG32 UNDRFLOW  : 1;
__REG32 IRQ       : 1;
__REG32 IRQEN     : 1;
__REG32 PROT1     : 1;
__REG32           : 2;
__REG32 AUTOMODE  : 2;
__REG32           :19;
} __dma_ctrl_stat_bits;

/* SLCDC LCD Clock Configuration Register */
typedef struct{
__REG32 DIVIDE  : 6;
__REG32         :26;
} __lcd_clk_config_bits;

/* SLCDC LCD Write Data Register */
typedef struct{
__REG32 LCDDAT  :16;
__REG32 RS      : 1;
__REG32         :15;
} __lcd_write_data_bits;

/* -------------------------------------------------------------------------*/
/*               SSI registers                                              */
/* -------------------------------------------------------------------------*/
/* SSI Control/Status Register */
typedef struct{
__REG32 SSIEN       : 1;
__REG32 TE          : 1;
__REG32 RE          : 1;
__REG32 NET         : 1;
__REG32 SYN         : 1;
__REG32 I2S_MODE    : 2;
__REG32 SYS_CLK_EN  : 1;
__REG32 TCH_EN      : 1;
__REG32 CLK_IST     : 1;
__REG32 TFR_CLK_DIS : 1;
__REG32 RFR_CLK_DIS : 1;
__REG32             :20;
} __scsr_bits;

/* SSI Interrupt Status Register */
typedef struct{
__REG32 TFE0   : 1;
__REG32 TFE1   : 1;
__REG32 RFF0   : 1;
__REG32 RFF1   : 1;
__REG32 RLS    : 1;
__REG32 TLS    : 1;
__REG32 RFS    : 1;
__REG32 TFS    : 1;
__REG32 TUE0   : 1;
__REG32 TUE1   : 1;
__REG32 ROE0   : 1;
__REG32 ROE1   : 1;
__REG32 TDE0   : 1;
__REG32 TDE1   : 1;
__REG32 RDR0   : 1;
__REG32 RDR1   : 1;
__REG32 RXT    : 1;
__REG32 CMDDU  : 1;
__REG32 CMDAU  : 1;
__REG32        :13;
} __sisr_bits;

/* SSI Interrupt Enable Register */
typedef struct{
__REG32 TFE0_EN   : 1;
__REG32 TFE1_EN   : 1;
__REG32 RFF0_EN   : 1;
__REG32 RFF1_EN   : 1;
__REG32 RLS_EN    : 1;
__REG32 TLS_EN    : 1;
__REG32 RFS_EN    : 1;
__REG32 TFS_EN    : 1;
__REG32 TUE0_EN   : 1;
__REG32 TUE1_EN   : 1;
__REG32 ROE0_EN   : 1;
__REG32 ROE1_EN   : 1;
__REG32 TDE0_EN   : 1;
__REG32 TDE1_EN   : 1;
__REG32 RDR0_EN   : 1;
__REG32 RDR1_EN   : 1;
__REG32 RXT_EN    : 1;
__REG32 CMDDU_EN  : 1;
__REG32 CMDAU_EN  : 1;
__REG32 TIE       : 1;
__REG32 TDMAE     : 1;
__REG32 RIE       : 1;
__REG32 RDMAE     : 1;
__REG32 TFRC_EN   : 1;
__REG32 RFRC_EN   : 1;
__REG32           : 7;
} __sier_bits;

/* SSI Transmit Configuration Register */
typedef struct{
__REG32 TEFS    : 1;
__REG32 TFSL    : 1;
__REG32 TFSI    : 1;
__REG32 TSCKP   : 1;
__REG32 TSHFD   : 1;
__REG32 TXDIR   : 1;
__REG32 TFDIR   : 1;
__REG32 TFEN0   : 1;
__REG32 TFEN1   : 1;
__REG32 TXBIT0  : 1;
__REG32         :22;
} __stcr_bits;

/* SSI Receive Configuration Register */
typedef struct{
__REG32 REFS    : 1;
__REG32 RFSL    : 1;
__REG32 RFSI    : 1;
__REG32 RSCKP   : 1;
__REG32 RSHFD   : 1;
__REG32 RXDIR   : 1;
__REG32 RFDIR   : 1;
__REG32 RFEN0   : 1;
__REG32 RFEN1   : 1;
__REG32 RXBIT0  : 1;
__REG32 RXEXT   : 1;
__REG32         :21;
} __srcr_bits;

/* SSI Clock Control Register */
typedef struct{
__REG32 PM    : 8;
__REG32 DC    : 5;
__REG32 WL    : 4;
__REG32 PSR   : 1;
__REG32 DIV2  : 1;
__REG32       :13;
} __ssi_ccr_bits;

/* SSI FIFO Control/Status Register */
typedef struct{
__REG32 TFWM0   : 4;
__REG32 RFWM0   : 4;
__REG32 TFCNT0  : 4;
__REG32 RFCNT0  : 4;
__REG32 TFWM1   : 4;
__REG32 RFWM1   : 4;
__REG32 TFCNT1  : 4;
__REG32 RFCNT1  : 4;
} __ssi_sfcsr_bits;

/* SSI Test Register */
typedef struct{
__REG32 TXSTATE  : 5;
__REG32 TFS2RFS  : 1;
__REG32 TCK2RCK  : 1;
__REG32 TXD2RXD  : 1;
__REG32 RXSTATE  : 5;
__REG32 RFS2TFS  : 1;
__REG32 RCK2TCK  : 1;
__REG32 TEST     : 1;
__REG32          :16;
} __ssi_str_bits;

/* SSI Option Register */
typedef struct{
__REG32 SYNRST  : 1;
__REG32 WAIT    : 2;
__REG32 INIT    : 1;
__REG32 TX_CLR  : 1;
__REG32 RX_CLR  : 1;
__REG32 CLKOFF  : 1;
__REG32         :25;
} __ssi_sor_bits;

/* SSI AC97 Control Register */
typedef struct{
__REG32 A97EN  : 1;
__REG32 FV     : 1;
__REG32 TIF    : 1;
__REG32 RD     : 1;
__REG32 WR     : 1;
__REG32 FRDIV  : 6;
__REG32        :21;
} __ssi_sacnt_bits;

/* SSI AC97 Command Address Register */
typedef struct{
__REG32 SACADD  :19;
__REG32         :13;
} __ssi_sacadd_bits;

/* SSI AC97 Command Data Register */
typedef struct{
__REG32 SACDAT  :19;
__REG32         :13;
} __ssi_sacdat_bits;

/* SSI AC97 Tag Register */
typedef struct{
__REG32 SATAG  :16;
__REG32        :16;
} __ssi_satag_bits;

/* SSI AC97 Channel Status Register (SACCST) */
typedef struct{
__REG32 SACCST :10;
__REG32        :22;
} __ssi_saccst_bits;

/* SSI AC97 Channel Enable Register (SACCEN)*/
typedef struct{
__REG32 SACCEN :10;
__REG32        :22;
} __ssi_saccen_bits;

/* SSI AC97 Channel Disable Register (SACCDIS) */
typedef struct{
__REG32 SACCDIS :10;
__REG32         :22;
} __ssi_saccdis_bits;

/* -------------------------------------------------------------------------*/
/*               Touch Screen Controller (TSC)                              */
/* -------------------------------------------------------------------------*/
/* TSC General Config Register (TGCR) */
typedef struct {
__REG32 IPGCLKEN        : 1;
__REG32 TSCRST          : 1;
__REG32 FUNCRST         : 1;
__REG32                 : 1;
__REG32 SLPC            : 1;
__REG32 STLC            : 1;
__REG32 HSYNCEN         : 1;
__REG32 HSYNCPOL        : 1;
__REG32 POWERMODE       : 2;
__REG32 INTREFEN        : 1;
__REG32                 : 5;
__REG32 ADCCLKCFG       : 5;
__REG32                 : 2;
__REG32 PDEN            : 1;
__REG32 PDBEN           : 1;
__REG32 PDBTIME         : 7;
} __tgcr_bits;

/* TSC General Status Register (TGSR) */
typedef struct {
__REG32 TCQINT          : 1;
__REG32 GCQINT          : 1;
__REG32 SLPINT          : 1;
__REG32                 :13;
__REG32 TCQDMA          : 1;
__REG32 GCQDMA          : 1;
__REG32                 :14;
} __tgsr_bits;

/* Queue FIFO(TCQFIFO/GCQFIFO) */
typedef struct {
__REG32 ITEM_ID         : 4;
__REG32 ADCDATAOUT      :12;
__REG32                 :16;
} __tcqfifo_bits;

/* Queue Control Register(TCQCR/GCQCR) */
typedef struct {
__REG32 QSM             : 2;
__REG32 FQS             : 1;
__REG32 RPT             : 1;
__REG32 LAST_ITEM_ID    : 4;
__REG32 FIFOWATERMARK   : 4;
__REG32 REPEATWAIT      : 4;
__REG32 QRST            : 1;
__REG32 FRST            : 1;
__REG32 PDMSK           : 1;
__REG32 PDCFG           : 1;
__REG32                 :12;
} __tcqcr_bits;

/* Queue Status Register(TCQSR/GCQSR) */
typedef struct {
__REG32 PD              : 1;
__REG32 EOQ             : 1;
__REG32                 : 2;
__REG32 _FOR            : 1;
__REG32 FUR             : 1;
__REG32 FER             : 1;
__REG32                 : 1;
__REG32 FDN             : 5;
__REG32 EMPT            : 1;
__REG32 FULL            : 1;
__REG32 FDRY            : 1;
__REG32 FRPTR           : 5;
__REG32 FWPTR           : 5;
__REG32                 : 6;
} __tcqsr_bits;

/* Queue Mask Register(TCQMR/GCQMR) */
typedef struct {
__REG32 PDIRQMSK        : 1;
__REG32 EOQIRQMSK       : 1;
__REG32                 : 2;
__REG32 FORIRQMSK       : 1;
__REG32 FURIRQMSK       : 1;
__REG32 FERIRQMSK       : 1;
__REG32                 : 8;
__REG32 FDRYIRQMSK      : 1;
__REG32 PDDMAMSK        : 1;
__REG32 EOQDMAMSK       : 1;
__REG32                 : 2;
__REG32 FORDMAMSK       : 1;
__REG32 FURDMAMSK       : 1;
__REG32 FERDMAMSK       : 1;
__REG32                 : 8;
__REG32 FDRYDMAMSK      : 1;
} __tcqmr_bits;

/* TCQ ITEM Register(TCQITEM_7_0) */
typedef struct {
__REG32 ITEM0           : 4;
__REG32 ITEM1           : 4;
__REG32 ITEM2           : 4;
__REG32 ITEM3           : 4;
__REG32 ITEM4           : 4;
__REG32 ITEM5           : 4;
__REG32 ITEM6           : 4;
__REG32 ITEM7           : 4;
} __tcq_item_7_0_bits;

/* TCQ ITEM Register(TCQITEM_15_8) */
typedef struct {
__REG32 ITEM8           : 4;
__REG32 ITEM9           : 4;
__REG32 ITEM10          : 4;
__REG32 ITEM11          : 4;
__REG32 ITEM12          : 4;
__REG32 ITEM13          : 4;
__REG32 ITEM14          : 4;
__REG32 ITEM15          : 4;
} __tcq_item_15_8_bits;

/* Convert Config (TICR/TCC0~TCC7/GCC0~GCC7) */
typedef struct {
__REG32                 : 1;
__REG32 PENIACK         : 1;
__REG32 SELREFN         : 2;
__REG32 SELIN           : 3;
__REG32 SELREFP         : 2;
__REG32 XPULSW          : 1;
__REG32 XNURSW          : 2;
__REG32 YPLLSW          : 2;
__REG32 YNLRSW          : 1;
__REG32 WIPERSW         : 1;
__REG32 NOS             : 4;
__REG32 IGS             : 1;
__REG32                 : 3;
__REG32 SETTLING_TIME   : 8;
} __ticr_bits;

/* -------------------------------------------------------------------------*/
/*      UARTs                                                               */
/* -------------------------------------------------------------------------*/
/* UARTs Receiver Register */
typedef struct{
__REG32 RX_DATA  : 8;     /* Bits 0-7             - Recieve Data*/
__REG32          : 2;     /* Bits 8-9             - Reserved*/
__REG32 PRERR    : 1;     /* Bit  10              - Receive Parity Error 1=error*/
__REG32 BRK      : 1;     /* Bit  11              - Receive break Caracter detected 1 = detected*/
__REG32 FRMERR   : 1;     /* Bit  12              - Receive Framing Error 1=error*/
__REG32 OVRRUN   : 1;     /* Bit  13              - Receive Over run Error 1=error*/
__REG32 ERR      : 1;     /* Bit  14              - Receive Error Detect (OVR,FRM,BRK,PR 0=error*/
__REG32 CHARRDY  : 1;     /* Bit  15              - Character Ready*/
__REG32          :16;
} __urxd_bits;

/* UARTs Transmitter Register */
typedef struct{
__REG32 TX_DATA  : 8;     /* Bits 7-0             - Transmit Data*/
__REG32          :24;
} __utxd_bits;

/* UARTs Control Register 1 */
typedef struct{
__REG32 UARTEN    : 1;     /* Bit  0       - UART Enable 1 = Enable the UART*/
__REG32 DOZE      : 1;     /* Bit  1       - DOZE 1 = The UART is disabled when in DOZE state*/
__REG32 ATDMAEN   : 1;     /* Bit  2*/
__REG32 TDMAEN    : 1;     /* Bit  3       - Transmitter Ready DMA Enable 1 = enable*/
__REG32 SNDBRK    : 1;     /* Bit  4       - Send BREAK 1 = send break char continuous*/
__REG32 RTSDEN    : 1;     /* Bit  5       - RTS Delta Interrupt Enable 1 = enable*/
__REG32 TXMPTYEN  : 1;     /* Bit  6       - Transmitter Empty Interrupt Enable 1 = enable*/
__REG32 IREN      : 1;     /* Bit  7       - Infrared Interface Enable 1 = enable*/
__REG32 RXDMAEN   : 1;     /* Bit  8       - Receive Ready DMA Enable 1 = enable*/
__REG32 RRDYEN    : 1;     /* Bit  9       - Receiver Ready Interrupt Enable 1 = Enable*/
__REG32 ICD       : 2;     /* Bit  10-11   - Idle Condition Detect*/
                           /*              - 00 = Idle for more than 4 frames*/
                           /*              - 01 = Idle for more than 8 frames*/
                           /*              - 10 = Idle for more than 16 frames*/
                           /*              - 11 = Idle for more than 32 frames*/
__REG32 IDEN      : 1;     /* Bit  12      - Idle Condition Detected Interrupt en 1=en*/
__REG32 TRDYEN    : 1;     /* Bit  13      - Transmitter Ready Interrupt Enable 1=en*/
__REG32 ADBR      : 1;     /* Bit  14      - AutoBaud Rate Detection enable 1=en*/
__REG32 ADEN      : 1;     /* Bit  15      - AutoBaud Rate Detection Interrupt en 1=en*/
__REG32           :16;
} __ucr1_bits;

/* UARTs Control Register 2 */
typedef struct{
__REG32 SRST   : 1;     /* Bit  0       -Software Reset 0 = Reset the tx and rx state machines*/
__REG32 RXEN   : 1;     /* Bit  1       -Receiver Enable 1 = Enable*/
__REG32 TXEN   : 1;     /* Bit  2       -Transmitter Enable 1= enable*/
__REG32 ATEN   : 1;     /* Bit  3       -Aging Timer Enable—This bit is used to mask the aging timer interrupt (triggered with AGTIM)*/
__REG32 RTSEN  : 1;     /* Bit  4       -Request to Send Interrupt Enable 1=enable*/
__REG32 WS     : 1;     /* Bit  5       -Word Size 0 = 7bit, 1= 8 bit*/
__REG32 STPB   : 1;     /* Bit  6       -Stop 0= 1 stop bits, 1= 2 stop bits*/
__REG32 PROE   : 1;     /* Bit  7       -Parity Odd/Even 1=Odd*/
__REG32 PREN   : 1;     /* Bit  8       -Parity Enable 1=enable parity generator*/
__REG32 RTEC   : 2;     /* Bits 9-10    -Request to Send Edge Control*/
                        /*              - 00 = Trigger interrupt on a rising edge*/
                        /*              - 01 = Trigger interrupt on a falling edge*/
                        /*              - 1X = Trigger interrupt on any edge*/
__REG32 ESCEN  : 1;     /* Bit  11      -Escape Enable 1 = Enable escape sequence detection*/
__REG32 CTS    : 1;     /* Bit  12      -Clear to Send 1 = The UARTx_CTS pin is low (active)*/
__REG32 CTSC   : 1;     /* Bit  13      -UARTx_CTS Pin controlled by 1= receiver 0= CTS bit*/
__REG32 IRTS   : 1;     /* Bit  14      -Ignore UARTx_RTS Pin 1=ignore*/
__REG32 ESCI   : 1;     /* Bit  15      -Escape Sequence Interrupt En 1=enable*/
__REG32        :16;
} __ucr2_bits;

/* UARTs Control Register 3 */
typedef struct{
__REG32 ACIEN      : 1;
__REG32 INVT       : 1;
__REG32 RXDMUXSEL  : 1;
__REG32 DTRDEN     : 1;
__REG32 AWAKEN     : 1;
__REG32 AIRINTEN   : 1;
__REG32 RXDSEN     : 1;
__REG32 ADNIMP     : 1;
__REG32 RI         : 1;
__REG32 DCD        : 1;
__REG32 DSR        : 1;
__REG32 FRAERREN   : 1;
__REG32 PARERREN   : 1;
__REG32 DTREN      : 1;
__REG32 DPEC       : 2;
__REG32            :16;
} __ucr3_bits;

/* UARTs Control Register 4 */
typedef struct{
__REG32 DREN   : 1;     /* Bit  0       -Receive Data Ready Interrupt Enable 1= enable*/
__REG32 OREN   : 1;     /* Bit  1       -Receiver Overrun Interrupt Enable 1= enable*/
__REG32 BKEN   : 1;     /* Bit  2       -BREAK Condition Detected Interrupt en 1= enable*/
__REG32 TCEN   : 1;     /* Bit  3       -Transmit Complete Interrupt Enable1 = Enable*/
__REG32 LPBYP  : 1;     /* Bit  4       -Low Power Bypass—Allows to bypass the low power new features in UART for . To use during debug phase.*/
__REG32 IRSC   : 1;     /* Bit  5       -IR Special Case vote logic uses 1= uart ref clk*/
__REG32 IDDMAEN: 1;     /* Bit  6       -*/
__REG32 WKEN   : 1;     /* Bit  7       -WAKE Interrupt Enable 1= enable*/
__REG32 ENIRI  : 1;     /* Bit  8       -Serial Infrared Interrupt Enable 1= enable*/
__REG32 INVR   : 1;     /* Bit  9       -Inverted Infrared Reception 1= active high*/
__REG32 CTSTL  : 6;     /* Bits 10-15   -CTS Trigger Level*/
                        /*              000000 = 0 characters received*/
                        /*              000001 = 1 characters in the RxFIFO*/
                        /*              ...*/
                        /*              100000 = 32 characters in the RxFIFO (maximum)*/
                        /*              All Other Settings Reserved*/
__REG32        :16;
} __ucr4_bits;

/* UARTs FIFO Control Register */
typedef struct{
__REG32 RXTL    : 6;     /* Bits 0-5     -Receiver Trigger Level*/
                         /*              000000 = 0 characters received*/
                         /*              000001 = RxFIFO has 1 character*/
                         /*              ...*/
                         /*              011111 = RxFIFO has 31 characters*/
                         /*              100000 = RxFIFO has 32 characters (maximum)*/
                         /*              All Other Settings Reserved*/
__REG32 DCEDTE  : 1;     /* Bit  6       */
__REG32 RFDIV   : 3;     /* Bits 7-9     -Reference Frequency Divider*/
                         /*              000 = Divide input clock by 6*/
                         /*              001 = Divide input clock by 5*/
                         /*              010 = Divide input clock by 4*/
                         /*              011 = Divide input clock by 3*/
                         /*              100 = Divide input clock by 2*/
                         /*              101 = Divide input clock by 1*/
                         /*              110 = Divide input clock by 7*/
__REG32 TXTL    : 6;     /* Bits 10-15   -Transmitter Trigger Level*/
                         /*              000000 = Reserved*/
                         /*              000001 = Reserved*/
                         /*              000010 = TxFIFO has 2 or fewer characters*/
                         /*              ...*/
                         /*              011111 = TxFIFO has 31 or fewer characters*/
                         /*              100000 = TxFIFO has 32 characters (maximum)*/
                         /*              All Other Settings Reserved*/
__REG32        :16;
} __ufcr_bits;

/* UARTs Status Register 1 */
typedef struct{
__REG32            : 4;
__REG32 AWAKE      : 1;
__REG32 AIRINT     : 1;
__REG32 RXDS       : 1;
__REG32 DTRD       : 1;
__REG32 AGTIM      : 1;
__REG32 RRDY       : 1;
__REG32 FRAMERR    : 1;
__REG32 ESCF       : 1;
__REG32 RTSD       : 1;
__REG32 TRDY       : 1;
__REG32 RTSS       : 1;
__REG32 PARITYERR  : 1;
__REG32            :16;
} __usr1_bits;

/* UARTs Status Register 2 */
typedef struct{
__REG32 RDR      : 1;
__REG32 ORE      : 1;
__REG32 BRCD     : 1;
__REG32 TXDC     : 1;
__REG32 RTSF     : 1;
__REG32 DCDIN    : 1;
__REG32 DCDDELT  : 1;
__REG32 WAKE     : 1;
__REG32 IRINT    : 1;
__REG32 RIIN     : 1;
__REG32 RIDELT   : 1;
__REG32 ACST     : 1;
__REG32 IDLE     : 1;
__REG32 DTRF     : 1;
__REG32 TXFE     : 1;
__REG32 ADET     : 1;
__REG32          :16;
} __usr2_bits;

/* UARTs Escape Character Register */
typedef struct{
__REG32 ESC_CHAR  : 8;     /* Bits 0-7     -UART Escape Character*/
__REG32           :24;
} __uesc_bits;

/* UARTs Escape Timer Register */
typedef struct{
__REG32 TIM  :12;     /* Bits 0-11    -UART Escape Timer*/
__REG32      :20;
} __utim_bits;

/* UARTS Test Register 1 */
typedef struct{
__REG32 SOFTRST  : 1;
__REG32          : 2;
__REG32 RXFULL   : 1;
__REG32 TXFULL   : 1;
__REG32 RXEMPTY  : 1;
__REG32 TXEMPTY  : 1;
__REG32          : 2;
__REG32 RXDBG    : 1;
__REG32 LOOPIR   : 1;
__REG32 DBGEN    : 1;
__REG32 LOOP     : 1;
__REG32 FRCPERR  : 1;
__REG32          :18;
} __uts_bits;

/* -------------------------------------------------------------------------*/
/*      USB OTG/HOST Registers                                              */
/* -------------------------------------------------------------------------*/
/* USB Identification Register */
typedef struct{
__REG32 ID        : 6;
__REG32           : 2;
__REG32 NID       : 6;
__REG32           : 2;
__REG32 REVISION  : 8;
__REG32           : 8;
} __usb_id_bits;

/* USB General Hardware Parameters */
typedef struct{
__REG32 RT        : 1;
__REG32 CLKC      : 2;
__REG32 BWT       : 1;
__REG32 PHYW      : 2;
__REG32 PHYM      : 3;
__REG32 SM        : 1;
__REG32           :22;
} __usb_hwgeneral_bits;

/* USB Host Hardware Parameters */
typedef struct{
__REG32 HC        : 1;
__REG32 NPORT     : 3;
__REG32           :12;
__REG32 TTASY     : 8;
__REG32 TTPER     : 8;
} __usb_hwhost_bits;

/* USB Device Hardware Parameters */
typedef struct{
__REG32 DC        : 1;
__REG32 DEVEP     : 5;
__REG32           :26;
} __usb_hwdevice_bits;

/* USB TX Buffer Hardware Parameters */
typedef struct{
__REG32 TCBURST   : 8;
__REG32 TXADD     : 8;
__REG32 TXCHANADD : 8;
__REG32           : 7;
__REG32 TXLCR     : 1;
} __usb_hwtxbuf_bits;

/* USB RX Buffer Hardware Parameters */
typedef struct{
__REG32 RXBURST   : 8;
__REG32 RXADD     : 8;
__REG32           :16;
} __usb_hwrxbuf_bits;

/* USB Host Control Structural Parameters */
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
} __usb_hcsparams_bits;

/* USB Host Control Capability Parameters */
typedef struct{
__REG32 ADC       : 1;
__REG32 PFL       : 1;
__REG32 ASP       : 1;
__REG32           : 1;
__REG32 IST       : 4;
__REG32 EECP      : 8;
__REG32           :16;
} __usb_hccparams_bits;

/* USB Device Interface Version Number */
typedef struct{
__REG32 DCIVERSION  :16;
__REG32             :16;
} __usb_dciversion_bits;

/* USB Device Interface Version Number */
typedef struct{
__REG32 DEN         : 5;
__REG32             : 2;
__REG32 DC          : 1;
__REG32 HC          : 1;
__REG32             :23;
} __usb_dccparams_bits;

/* USB General Purpose Timer #0 Load Register */
typedef struct{
__REG32 GPTLD       :24;
__REG32             : 8;
} __usb_gptimer0ld_bits;

/* USB General Purpose Timer #0 Controller */
typedef struct{
__REG32 GPTCNT      :24;
__REG32 GPTMOD      : 1;
__REG32             : 5;
__REG32 GPTRST      : 1;
__REG32 GPTRUN      : 1;
} __usb_gptimer0ctrl_bits;

/* SBUSFG - control for the system bus interface */
typedef struct{
__REG32 AHBBRST     : 3;
__REG32             :29;
} __uog_sbuscfg_bits;

/* USB Command Register */
typedef struct{
__REG32 RS          : 1;
__REG32 RST         : 1;
__REG32 FS0         : 1;
__REG32 FS1         : 1;
__REG32 PSE         : 1;
__REG32 ASE         : 1;
__REG32 IAA         : 1;
__REG32 LR          : 1;
__REG32 ASP0        : 1;
__REG32 ASP1        : 1;
__REG32             : 1;
__REG32 ASPE        : 1;
__REG32             : 1;
__REG32 SUTW        : 1;
__REG32 ATDTW       : 1;
__REG32 FS2         : 1;
__REG32 ITC         : 8;
__REG32             : 8;
} __usb_usbcmd_bits;

/* USB Status */
typedef struct{
__REG32 UI          : 1;
__REG32 UEI         : 1;
__REG32 PCI         : 1;
__REG32 FRI         : 1;
__REG32 SEI         : 1;
__REG32 AAI         : 1;
__REG32 URI         : 1;
__REG32 SRI         : 1;
__REG32 SLI         : 1;
__REG32             : 1;
__REG32 ULPII       : 1;
__REG32             : 1;
__REG32 HCH         : 1;
__REG32 RCL         : 1;
__REG32 PS          : 1;
__REG32 AS          : 1;
__REG32             : 8;
__REG32 TI0         : 1;
__REG32 TI1         : 1;
__REG32             : 6;
} __usb_usbsts_bits;

/* USB Interrupt Enable */
typedef struct{
__REG32 UE          : 1;
__REG32 UEE         : 1;
__REG32 PCE         : 1;
__REG32 FRE         : 1;
__REG32 SEE         : 1;
__REG32 AAE         : 1;
__REG32 URE         : 1;
__REG32 SRE         : 1;
__REG32 SLE         : 1;
__REG32             : 1;
__REG32 ULPIE       : 1;
__REG32             :13;
__REG32 TIE0        : 1;
__REG32 TIE1        : 1;
__REG32             : 6;
} __usb_usbintr_bits;

/* USB Frame Index */
typedef struct{
__REG32 FRINDEX     :14;
__REG32             :18;
} __usb_frindex_bits;

/* USB OTG Host Controller Frame List Base Address
   Device Controller USB Device Address */
typedef union {
  /* UOG_PERIODICLISTBASE*/
  struct {
   __REG32 PERBASE     :32;
  };
  /* UOG_DEVICEADDR*/
  struct {
  __REG32             :25;
  __REG32 USBADR      : 7;
  };
} __usb_periodiclistbase_bits;

typedef union {
  /* UHx_PERIODICLISTBASE*/
  struct {
   __REG32 PERBASE     :32;
  };
  /* UHx_DEVICEADDR*/
  struct {
  __REG32             :25;
  __REG32 USBADR      : 7;
  };
} __uh_periodiclistbase_bits;

/* USB Host Controller Embedded TT Async. Buffer Status */
typedef struct{
__REG32 RXPBURST    : 8;
__REG32 TXPBURST    : 9;
__REG32             :15;
} __usb_burstsize_bits;

/* USB TXFILLTUNING */
typedef struct{
__REG32 TXSCHOH     : 8;
__REG32 TXSCHEALTH  : 5;
__REG32             : 3;
__REG32 TXFIFOTHRES : 6;
__REG32             :10;
} __usb_txfilltuning_bits;

/* USB ULPI VIEWPORT */
typedef struct{
__REG32 ULPIDATWR   : 8;
__REG32 ULPIDATRD   : 8;
__REG32 ULPIADDR    : 8;
__REG32 ULPIPORT    : 3;
__REG32 ULPISS      : 1;
__REG32             : 1;
__REG32 ULPIRW      : 1;
__REG32 ULPIRUN     : 1;
__REG32 ULPIWU      : 1;
} __usb_ulpiview_bits;

/* USB Port Status Control[1:8] */
typedef struct{
__REG32 CCS         : 1;
__REG32 CSC         : 1;
__REG32 PE          : 1;
__REG32 PEC         : 1;
__REG32 OCA         : 1;
__REG32 OCC         : 1;
__REG32 FPR         : 1;
__REG32 SUSP        : 1;
__REG32 PR          : 1;
__REG32 HSP         : 1;
__REG32 LS          : 2;
__REG32 PP          : 1;
__REG32 PO          : 1;
__REG32 PIC         : 2;
__REG32 PTC         : 4;
__REG32 WKCN        : 1;
__REG32 WKDS        : 1;
__REG32 WKOC        : 1;
__REG32 PHCD        : 1;
__REG32 PFSC        : 1;
__REG32             : 1;
__REG32 PSPD        : 2;
__REG32 PTW         : 1;
__REG32 STS         : 1;
__REG32 PTS         : 2;
} __usb_portsc_bits;

/* USB Status Control */
typedef struct{
__REG32 VD          : 1;
__REG32 VC          : 1;
__REG32             : 1;
__REG32 OT          : 1;
__REG32 DP          : 1;
__REG32 IDPU        : 1;
__REG32             : 2;
__REG32 ID          : 1;
__REG32 AVV         : 1;
__REG32 ASV         : 1;
__REG32 BSV         : 1;
__REG32 BSE         : 1;
__REG32 _1MST       : 1;
__REG32 DPS         : 1;
__REG32             : 1;
__REG32 IDIS        : 1;
__REG32 AVVIS       : 1;
__REG32 ASVIS       : 1;
__REG32 BSVIS       : 1;
__REG32 BSEIS       : 1;
__REG32 _1MSS       : 1;
__REG32 DPIS        : 1;
__REG32             : 1;
__REG32 IDIE        : 1;
__REG32 AVVIE       : 1;
__REG32 ASVIE       : 1;
__REG32 BSVIE       : 1;
__REG32 BSEIE       : 1;
__REG32 _1MSE       : 1;
__REG32 DPIE        : 1;
__REG32             : 1;
} __usb_otgsc_bits;

/* USB Device Mode */
typedef struct{
__REG32 CM          : 2;
__REG32 ES          : 1;
__REG32 SLOM        : 1;
__REG32 SDIS        : 1;
__REG32             :27;
} __usb_usbmode_bits;

/* USB Endpoint Setup Status */
typedef struct{
__REG32 ENDPTSETUPSTAT0   : 1;
__REG32 ENDPTSETUPSTAT1   : 1;
__REG32 ENDPTSETUPSTAT2   : 1;
__REG32 ENDPTSETUPSTAT3   : 1;
__REG32 ENDPTSETUPSTAT4   : 1;
__REG32 ENDPTSETUPSTAT5   : 1;
__REG32 ENDPTSETUPSTAT6   : 1;
__REG32 ENDPTSETUPSTAT7   : 1;
__REG32 ENDPTSETUPSTAT8   : 1;
__REG32 ENDPTSETUPSTAT9   : 1;
__REG32 ENDPTSETUPSTAT10  : 1;
__REG32 ENDPTSETUPSTAT11  : 1;
__REG32 ENDPTSETUPSTAT12  : 1;
__REG32 ENDPTSETUPSTAT13  : 1;
__REG32 ENDPTSETUPSTAT14  : 1;
__REG32 ENDPTSETUPSTAT15  : 1;
__REG32                   :16;
} __usb_endptsetupstat_bits;

/* USB Endpoint Initialization */
typedef struct{
__REG32 PERB0       : 1;
__REG32 PERB1       : 1;
__REG32 PERB2       : 1;
__REG32 PERB3       : 1;
__REG32 PERB4       : 1;
__REG32 PERB5       : 1;
__REG32 PERB6       : 1;
__REG32 PERB7       : 1;
__REG32 PERB8       : 1;
__REG32 PERB9       : 1;
__REG32 PERB10      : 1;
__REG32 PERB11      : 1;
__REG32 PERB12      : 1;
__REG32 PERB13      : 1;
__REG32 PERB14      : 1;
__REG32 PERB15      : 1;
__REG32 PETB0       : 1;
__REG32 PETB1       : 1;
__REG32 PETB2       : 1;
__REG32 PETB3       : 1;
__REG32 PETB4       : 1;
__REG32 PETB5       : 1;
__REG32 PETB6       : 1;
__REG32 PETB7       : 1;
__REG32 PETB8       : 1;
__REG32 PETB9       : 1;
__REG32 PETB10      : 1;
__REG32 PETB11      : 1;
__REG32 PETB12      : 1;
__REG32 PETB13      : 1;
__REG32 PETB14      : 1;
__REG32 PETB15      : 1;
} __usb_endptprime_bits;

/* USB Endpoint De-Initialize */
typedef struct{
__REG32 FERB0       : 1;
__REG32 FERB1       : 1;
__REG32 FERB2       : 1;
__REG32 FERB3       : 1;
__REG32 FERB4       : 1;
__REG32 FERB5       : 1;
__REG32 FERB6       : 1;
__REG32 FERB7       : 1;
__REG32 FERB8       : 1;
__REG32 FERB9       : 1;
__REG32 FERB10      : 1;
__REG32 FERB11      : 1;
__REG32 FERB12      : 1;
__REG32 FERB13      : 1;
__REG32 FERB14      : 1;
__REG32 FERB15      : 1;
__REG32 FETB0       : 1;
__REG32 FETB1       : 1;
__REG32 FETB2       : 1;
__REG32 FETB3       : 1;
__REG32 FETB4       : 1;
__REG32 FETB5       : 1;
__REG32 FETB6       : 1;
__REG32 FETB7       : 1;
__REG32 FETB8       : 1;
__REG32 FETB9       : 1;
__REG32 FETB10      : 1;
__REG32 FETB11      : 1;
__REG32 FETB12      : 1;
__REG32 FETB13      : 1;
__REG32 FETB14      : 1;
__REG32 FETB15      : 1;
} __usb_endptflush_bits;

/* USB Endpoint Status */
typedef struct{
__REG32 ERBR0       : 1;
__REG32 ERBR1       : 1;
__REG32 ERBR2       : 1;
__REG32 ERBR3       : 1;
__REG32 ERBR4       : 1;
__REG32 ERBR5       : 1;
__REG32 ERBR6       : 1;
__REG32 ERBR7       : 1;
__REG32 ERBR8       : 1;
__REG32 ERBR9       : 1;
__REG32 ERBR10      : 1;
__REG32 ERBR11      : 1;
__REG32 ERBR12      : 1;
__REG32 ERBR13      : 1;
__REG32 ERBR14      : 1;
__REG32 ERBR15      : 1;
__REG32 ETBR0       : 1;
__REG32 ETBR1       : 1;
__REG32 ETBR2       : 1;
__REG32 ETBR3       : 1;
__REG32 ETBR4       : 1;
__REG32 ETBR5       : 1;
__REG32 ETBR6       : 1;
__REG32 ETBR7       : 1;
__REG32 ETBR8       : 1;
__REG32 ETBR9       : 1;
__REG32 ETBR10      : 1;
__REG32 ETBR11      : 1;
__REG32 ETBR12      : 1;
__REG32 ETBR13      : 1;
__REG32 ETBR14      : 1;
__REG32 ETBR15      : 1;
} __usb_endptstat_bits;

/* USB Endpoint Compete */
typedef struct{
__REG32 ERCE0       : 1;
__REG32 ERCE1       : 1;
__REG32 ERCE2       : 1;
__REG32 ERCE3       : 1;
__REG32 ERCE4       : 1;
__REG32 ERCE5       : 1;
__REG32 ERCE6       : 1;
__REG32 ERCE7       : 1;
__REG32 ERCE8       : 1;
__REG32 ERCE9       : 1;
__REG32 ERCE10      : 1;
__REG32 ERCE11      : 1;
__REG32 ERCE12      : 1;
__REG32 ERCE13      : 1;
__REG32 ERCE14      : 1;
__REG32 ERCE15      : 1;
__REG32 ETCE0       : 1;
__REG32 ETCE1       : 1;
__REG32 ETCE2       : 1;
__REG32 ETCE3       : 1;
__REG32 ETCE4       : 1;
__REG32 ETCE5       : 1;
__REG32 ETCE6       : 1;
__REG32 ETCE7       : 1;
__REG32 ETCE8       : 1;
__REG32 ETCE9       : 1;
__REG32 ETCE10      : 1;
__REG32 ETCE11      : 1;
__REG32 ETCE12      : 1;
__REG32 ETCE13      : 1;
__REG32 ETCE14      : 1;
__REG32 ETCE15      : 1;
} __usb_endptcomplete_bits;

/* USB Endpoint Control 0 */
typedef struct{
__REG32 RXS         : 1;
__REG32             : 1;
__REG32 RXT         : 2;
__REG32             : 3;
__REG32 RXE         : 1;
__REG32             : 8;
__REG32 TXS         : 1;
__REG32             : 1;
__REG32 TXT         : 2;
__REG32             : 3;
__REG32 TXE         : 1;
__REG32             : 8;
} __usb_endptctrl0_bits;

/* USB Endpoint Control 1-15 */
typedef struct{
__REG32 RXS         : 1;
__REG32 RXD         : 1;
__REG32 RXT         : 2;
__REG32             : 1;
__REG32 RXI         : 1;
__REG32 RXR         : 1;
__REG32 RXE         : 1;
__REG32             : 8;
__REG32 TXS         : 1;
__REG32 TXD         : 1;
__REG32 TXT         : 2;
__REG32             : 1;
__REG32 TXI         : 1;
__REG32 TXR         : 1;
__REG32 TXE         : 1;
__REG32             : 8;
} __usb_endptctrl_bits;

/* -------------------------------------------------------------------------*/
/*      USB Registers                                                       */
/* -------------------------------------------------------------------------*/
/* USB Control Register */
typedef struct{
__REG32 OOCS      : 1;
__REG32 H2OCS     : 1;
__REG32 OCE       : 1;
__REG32 OCPOL     : 1;
__REG32 USBTE     : 1;
__REG32 H2DT      : 1;
__REG32 IPPUEDWN  : 1;
__REG32 IPPUEUP   : 1;
__REG32 IPPUIDP   : 1;
__REG32 XCSH2     : 1;
__REG32 XCSO      : 1;
__REG32 PP        : 1;
__REG32           : 4;
__REG32 H2PM      : 1;
__REG32           : 2;
__REG32 H2WIE     : 1;
__REG32 H2UIE     : 1;
__REG32 H2SIC     : 2;
__REG32 H2WIR     : 1;
__REG32 OPM       : 1;
__REG32 OEXTE     : 1;
__REG32 HEXTEN    : 1;
__REG32 OWIE      : 1;
__REG32 OUIE      : 1;
__REG32 OSIC      : 2;
__REG32 OWIR      : 1;
} __usb_ctrl_bits;

/* OTG Mirror Register (OTGMIRROR) */
typedef struct{
__REG32 IDDIG       : 1;
__REG32 ASESVLD     : 1;
__REG32 BSESVLD     : 1;
__REG32 VBUSVAL     : 1;
__REG32 SESEND      : 1;
__REG32             : 1;
__REG32 HOSTULPICLK : 1;
__REG32 OTGULPICLK  : 1;
__REG32 OTGUTMICLK  : 1;
__REG32             :23;
} __usb_otg_mirror_bits;

/* USB_PHY_CTRL_FUNC — OTG UTMI PHY Function Control reg */
typedef struct{
__REG32 utmi_VStatus        : 8;
__REG32 utmi_VControlData   : 8;
__REG32 utmi_Control        : 4;
__REG32 utmi_VControl_LoadM : 1;
__REG32 utmi_HostPort       : 1;
__REG32 utmi_LSFE           : 1;
__REG32 utmi_EVdo           : 1;
__REG32 utmi_USBEN          : 1;
__REG32 utmi_reset          : 1;
__REG32 utmi_suspendM       : 1;
__REG32 utmi_ClkValid       : 1;
__REG32                     : 4;
} __usb_phy_ctrl_func_bits;

/* USB_PHY_CTRL_TEST — OTG UTMI PHY Test Control register */
typedef struct{
__REG32 utmi_ft             : 1;
__REG32 utmi_nFC            : 1;
__REG32 utmi_LB             : 1;
__REG32 utmi_TM             : 2;
__REG32                     :27;
} __usb_phy_ctrl_test_bits;

/* -------------------------------------------------------------------------*/
/*      Watchdog Registers                                                  */
/* -------------------------------------------------------------------------*/
typedef struct {        /* Watchdog Control Register (0x10002000) Reset  (0x00000000)*/
__REG16 WDZST  : 1;     /* Bit  0       - Watchdog Low Power*/
__REG16 WDBG   : 1;     /* Bits 1       - Watchdog DEBUG Enable*/
__REG16 WDE    : 1;     /* Bit  2       - Watchdog Enable*/
__REG16 WRE    : 1;     /* Bit  3       - ~WDOG or ~WDOG_RESET Enable*/
__REG16 SRS    : 1;     /* Bit  4       - ~Software Reset Signal*/
__REG16 WDA    : 1;     /* Bit  5       - ~Watchdog Assertion*/
__REG16 WOE    : 1;     /* Bits 6*/
__REG16 WDW    : 1;     /* Bits 7*/
__REG16 WT     : 8;     /* Bits 8 - 15  - Watchdog Time-Out Field*/
} __wcr_bits;

typedef struct {        /* Watchdog Reset Status Register (0x10002004) Reset (*)*/
__REG16 SFTW  : 1;     /* Bit  0       - Software Reset*/
__REG16 TOUT  : 1;     /* Bit  1       - Time-out*/
__REG16       :14;     /* Bits 5  - 15 - Reserved*/
} __wrsr_bits;

/* -------------------------------------------------------------------------*/
/*      Shared Peripheral Bus Arbiter (SPBA)                                */
/* -------------------------------------------------------------------------*/
/* SPBA Register Definition */
typedef struct{
__REG32 RAR       : 3;
__REG32           :13;
__REG32 ROIn      : 2;
__REG32           :12;
__REG32 RMOn      : 2;
} __spba_prr_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */

/* Common declarations  ****************************************************/
/***************************************************************************
 **
 **  IOMUX
 **
 ***************************************************************************/
__IO_REG32_BIT(IOMUXC_GPR1,                       0x43FAC000,__READ_WRITE,__iomuxc_gpr1_bits);
__IO_REG32_BIT(IOMUXC_OBSERVE_INT_MUX,            0x43FAC004,__READ_WRITE,__iomuxc_observe_int_mux_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A10,         0x43FAC008,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A13,         0x43FAC00C,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A14,         0x43FAC010,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A15,         0x43FAC014,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A16,         0x43FAC018,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A17,         0x43FAC01C,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A18,         0x43FAC020,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A19,         0x43FAC024,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A20,         0x43FAC028,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A21,         0x43FAC02C,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A22,         0x43FAC030,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A23,         0x43FAC034,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A24,         0x43FAC038,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A25,         0x43FAC03C,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EB0,         0x43FAC040,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EB1,         0x43FAC044,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_OE,          0x43FAC048,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CS0,         0x43FAC04C,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CS1,         0x43FAC050,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CS4,         0x43FAC054,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CS5,         0x43FAC058,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NF_CE0,      0x43FAC05C,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_ECB,         0x43FAC060,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LBA,         0x43FAC064,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_BCLK,        0x43FAC068,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_RW,          0x43FAC06C,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NFWE_B,      0x43FAC070,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NFRE_B,      0x43FAC074,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NFALE,       0x43FAC078,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NFCLE,       0x43FAC07C,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NFWP_B,      0x43FAC080,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NFRB,        0x43FAC084,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_D15,         0x43FAC088,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_D14,         0x43FAC08C,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_D13,         0x43FAC090,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_D12,         0x43FAC094,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_D11,         0x43FAC098,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_D10,         0x43FAC09C,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_D9,          0x43FAC0A0,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_D8,          0x43FAC0A4,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_D7,          0x43FAC0A8,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_D6,          0x43FAC0AC,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_D5,          0x43FAC0B0,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_D4,          0x43FAC0B4,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_D3,          0x43FAC0B8,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_D2,          0x43FAC0BC,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_D1,          0x43FAC0C0,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_D0,          0x43FAC0C4,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD0,         0x43FAC0C8,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD1,         0x43FAC0CC,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD2,         0x43FAC0D0,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD3,         0x43FAC0D4,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD4,         0x43FAC0D8,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD5,         0x43FAC0DC,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD6,         0x43FAC0E0,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD7,         0x43FAC0E4,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD8,         0x43FAC0E8,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD9,         0x43FAC0EC,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD10,        0x43FAC0F0,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD11,        0x43FAC0F4,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD12,        0x43FAC0F8,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD13,        0x43FAC0FC,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD14,        0x43FAC100,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD15,        0x43FAC104,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_HSYNC,       0x43FAC108,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_VSYNC,       0x43FAC10C,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LSCLK,       0x43FAC110,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_OE_ACD,      0x43FAC114,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CONTRAST,    0x43FAC118,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_PWM,         0x43FAC11C,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_CSI_D2,          0x43FAC120,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_CSI_D3,          0x43FAC124,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_CSI_D4,          0x43FAC128,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_CSI_D5,          0x43FAC12C,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_CSI_D6,          0x43FAC130,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_CSI_D7,          0x43FAC134,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_CSI_D8,          0x43FAC138,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_CSI_D9,          0x43FAC13C,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_CSI_MCLK,        0x43FAC140,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_CSI_VSYNC,       0x43FAC144,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_CSI_HSYNC,       0x43FAC148,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_CSI_PIXCLK,      0x43FAC14C,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_I2C1_CLK,    0x43FAC150,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_I2C1_DAT,    0x43FAC154,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSPI1_MOSI,  0x43FAC158,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSPI1_MISO,  0x43FAC15C,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSPI1_SS0,   0x43FAC160,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSPI1_SS1,   0x43FAC164,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSPI1_SCLK,  0x43FAC168,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSPI1_RDY,   0x43FAC16C,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_UART1_RXD,   0x43FAC170,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_UART1_TXD,   0x43FAC174,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_UART1_RTS,   0x43FAC178,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_UART1_CTS,   0x43FAC17C,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_UART2_RXD,   0x43FAC180,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_UART2_TXD,   0x43FAC184,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_UART2_RTS,   0x43FAC188,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_UART2_CTS,   0x43FAC18C,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD1_CMD,     0x43FAC190,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD1_CLK,     0x43FAC194,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD1_DATA0,   0x43FAC198,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD1_DATA1,   0x43FAC19C,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD1_DATA2,   0x43FAC1A0,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD1_DATA3,   0x43FAC1A4,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_KPP_ROW0,    0x43FAC1A8,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_KPP_ROW1,    0x43FAC1AC,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_KPP_ROW2,    0x43FAC1B0,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_KPP_ROW3,    0x43FAC1B4,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_KPP_COL0,    0x43FAC1B8,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_KPP_COL1,    0x43FAC1BC,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_KPP_COL2,    0x43FAC1C0,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_KPP_COL3,    0x43FAC1C4,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_FEC_MDC,     0x43FAC1C8,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_FEC_MDIO,    0x43FAC1CC,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_FEC_TDATA0,  0x43FAC1D0,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_FEC_TDATA1,  0x43FAC1D4,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_FEC_TX_EN,   0x43FAC1D8,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_FEC_RDATA0,  0x43FAC1DC,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_FEC_RDATA1,  0x43FAC1E0,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_FEC_RX_DV,   0x43FAC1E4,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_FEC_TX_CLK,  0x43FAC1E8,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_RTCK,        0x43FAC1EC,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DE_B,        0x43FAC1F0,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO_A,      0x43FAC1F4,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO_B,      0x43FAC1F8,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO_C,      0x43FAC1FC,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO_D,      0x43FAC200,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO_E,      0x43FAC204,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO_F,      0x43FAC208,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EXT_ARMCLK,  0x43FAC20C,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_UPLL_BYPCLK, 0x43FAC210,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_VSTBY_REQ,   0x43FAC214,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_VSTBY_ACK,   0x43FAC218,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_POWER_FAIL,  0x43FAC21C,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CLKO,        0x43FAC220,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_BOOT_MODE0,  0x43FAC224,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_BOOT_MODE1,  0x43FAC228,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_A13,         0x43FAC22C,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_A14,         0x43FAC230,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_A15,         0x43FAC234,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_A17,         0x43FAC238,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_A18,         0x43FAC23C,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_A19,         0x43FAC240,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_A20,         0x43FAC244,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_A21,         0x43FAC248,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_A23,         0x43FAC24C,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_A24,         0x43FAC250,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_A25,         0x43FAC254,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EB0,         0x43FAC258,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EB1,         0x43FAC25C,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_OE,          0x43FAC260,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CS4,         0x43FAC264,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CS5,         0x43FAC268,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NF_CE0,      0x43FAC26C,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_ECB,         0x43FAC270,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LBA,         0x43FAC274,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_RW,          0x43FAC278,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NFRB,        0x43FAC27C,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_D15,         0x43FAC280,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_D14,         0x43FAC284,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_D13,         0x43FAC288,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_D12,         0x43FAC28C,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_D11,         0x43FAC290,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_D10,         0x43FAC294,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_D9,          0x43FAC298,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_D8,          0x43FAC29C,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_D7,          0x43FAC2A0,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_D6,          0x43FAC2A4,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_D5,          0x43FAC2A8,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_D4,          0x43FAC2AC,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_D3,          0x43FAC2B0,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_D2,          0x43FAC2B4,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_D1,          0x43FAC2B8,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_D0,          0x43FAC2BC,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD0,         0x43FAC2C0,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD1,         0x43FAC2C4,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD2,         0x43FAC2C8,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD3,         0x43FAC2CC,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD4,         0x43FAC2D0,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD5,         0x43FAC2D4,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD6,         0x43FAC2D8,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD7,         0x43FAC2DC,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD8,         0x43FAC2E0,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD9,         0x43FAC2E4,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD10,        0x43FAC2E8,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD11,        0x43FAC2EC,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD12,        0x43FAC2F0,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD13,        0x43FAC2F4,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD14,        0x43FAC2F8,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD15,        0x43FAC2FC,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_HSYNC,       0x43FAC300,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_VSYNC,       0x43FAC304,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LSCLK,       0x43FAC308,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_OE_ACD,      0x43FAC30C,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CONTRAST,    0x43FAC310,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_PWM,         0x43FAC314,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_CSI_D2,          0x43FAC318,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_CSI_D3,          0x43FAC31C,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_CSI_D4,          0x43FAC320,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_CSI_D5,          0x43FAC324,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_CSI_D6,          0x43FAC328,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_CSI_D7,          0x43FAC32C,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_CSI_D8,          0x43FAC330,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_CSI_D9,          0x43FAC334,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_CSI_MCLK,        0x43FAC338,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_CSI_VSYNC,       0x43FAC33C,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_CSI_HSYNC,       0x43FAC340,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_CSI_PIXCLK,      0x43FAC344,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_I2C1_CLK,    0x43FAC348,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_I2C1_DAT,    0x43FAC34C,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSPI1_MOSI,  0x43FAC350,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSPI1_MISO,  0x43FAC354,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSPI1_SS0,   0x43FAC358,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSPI1_SS1,   0x43FAC35C,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSPI1_SCLK,  0x43FAC360,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSPI1_RDY,   0x43FAC364,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_UART1_RXD,   0x43FAC368,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_UART1_TXD,   0x43FAC36C,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_UART1_RTS,   0x43FAC370,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_UART1_CTS,   0x43FAC374,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_UART2_RXD,   0x43FAC378,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_UART2_TXD,   0x43FAC37C,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_UART2_RTS,   0x43FAC380,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_UART2_CTS,   0x43FAC384,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD1_CMD,     0x43FAC388,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD1_CLK,     0x43FAC38C,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD1_DATA0,   0x43FAC390,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD1_DATA1,   0x43FAC394,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD1_DATA2,   0x43FAC398,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD1_DATA3,   0x43FAC39C,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_KPP_ROW0,    0x43FAC3A0,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_KPP_ROW1,    0x43FAC3A4,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_KPP_ROW2,    0x43FAC3A8,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_KPP_ROW3,    0x43FAC3AC,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_KPP_COL0,    0x43FAC3B0,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_KPP_COL1,    0x43FAC3B4,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_KPP_COL2,    0x43FAC3B8,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_KPP_COL3,    0x43FAC3BC,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_FEC_MDC,     0x43FAC3C0,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_FEC_MDIO,    0x43FAC3C4,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_FEC_TDATA0,  0x43FAC3C8,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_FEC_TDATA1,  0x43FAC3CC,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_FEC_TX_EN,   0x43FAC3D0,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_FEC_RDATA0,  0x43FAC3D4,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_FEC_RDATA1,  0x43FAC3D8,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_FEC_RX_DV,   0x43FAC3DC,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_FEC_TX_CLK,  0x43FAC3E0,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_RTCK,        0x43FAC3E4,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_TDO,         0x43FAC3E8,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DE_B,        0x43FAC3EC,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO_A,      0x43FAC3F0,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO_B,      0x43FAC3F4,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO_C,      0x43FAC3F8,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO_D,      0x43FAC3FC,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO_E,      0x43FAC400,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO_F,      0x43FAC404,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_VSTBY_REQ,   0x43FAC408,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_VSTBY_ACK,   0x43FAC40C,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_POWER_FAIL,  0x43FAC410,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CLKO,        0x43FAC414,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DVS_MISC,    0x43FAC418,__READ_WRITE,__iomuxc_sw_pad_ctl_grp_dvs_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DSE_FEC,     0x43FAC41C,__READ_WRITE,__iomuxc_sw_pad_ctl_grp_dse_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DVS_JTAG,    0x43FAC420,__READ_WRITE,__iomuxc_sw_pad_ctl_grp_dvs_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DSE_NFC,     0x43FAC424,__READ_WRITE,__iomuxc_sw_pad_ctl_grp_dse_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DSE_CSI,     0x43FAC428,__READ_WRITE,__iomuxc_sw_pad_ctl_grp_dse_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DSE_WEIM,    0x43FAC42C,__READ_WRITE,__iomuxc_sw_pad_ctl_grp_dse_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DSE_DDR,     0x43FAC430,__READ_WRITE,__iomuxc_sw_pad_ctl_grp_dse_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DVS_CRM,     0x43FAC434,__READ_WRITE,__iomuxc_sw_pad_ctl_grp_dvs_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DSE_KPP,     0x43FAC438,__READ_WRITE,__iomuxc_sw_pad_ctl_grp_dse_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DSE_SDHC1,   0x43FAC43C,__READ_WRITE,__iomuxc_sw_pad_ctl_grp_dse_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DSE_LCD,     0x43FAC440,__READ_WRITE,__iomuxc_sw_pad_ctl_grp_dse_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DSE_UART,    0x43FAC444,__READ_WRITE,__iomuxc_sw_pad_ctl_grp_dse_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DVS_NFC,     0x43FAC448,__READ_WRITE,__iomuxc_sw_pad_ctl_grp_dvs_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DVS_CSI,     0x43FAC44C,__READ_WRITE,__iomuxc_sw_pad_ctl_grp_dvs_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DSE_CSPI1,   0x43FAC450,__READ_WRITE,__iomuxc_sw_pad_ctl_grp_dse_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DDRTYPE,     0x43FAC454,__READ_WRITE,__iomuxc_sw_pad_ctl_grp_ddrtype_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DVS_SDHC1,   0x43FAC458,__READ_WRITE,__iomuxc_sw_pad_ctl_grp_dvs_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DVS_LCD,     0x43FAC45C,__READ_WRITE,__iomuxc_sw_pad_ctl_grp_dvs_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P4_INPUT_DA_AMX_SELECT_INPUT,     0x43FAC460,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P4_INPUT_DB_AMX_SELECT_INPUT,     0x43FAC464,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P4_INPUT_RXCLK_AMX_SELECT_INPUT,  0x43FAC468,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P4_INPUT_RXFS_AMX_SELECT_INPUT,   0x43FAC46C,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P4_INPUT_TXCLK_AMX_SELECT_INPUT,  0x43FAC470,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P4_INPUT_TXFS_AMX_SELECT_INPUT,   0x43FAC474,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P7_INPUT_DA_AMX_SELECT_INPUT,     0x43FAC478,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P7_INPUT_TXFS_AMX_SELECT_INPUT,   0x43FAC47C,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_CAN1_IPP_IND_CANRX_SELECT_INPUT,         0x43FAC480,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_CAN2_IPP_IND_CANRX_SELECT_INPUT,         0x43FAC484,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_CSI_IPP_CSI_D_0_SELECT_INPUT,            0x43FAC488,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_CSI_IPP_CSI_D_1_SELECT_INPUT,            0x43FAC48C,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_CSPI1_IPP_IND_SS3_B_SELECT_INPUT,        0x43FAC490,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_CSPI2_IPP_CSPI_CLK_IN_SELECT_INPUT,      0x43FAC494,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_CSPI2_IPP_IND_DATAREADY_B_SELECT_INPUT,  0x43FAC498,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_CSPI2_IPP_IND_MISO_SELECT_INPUT,         0x43FAC49C,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_CSPI2_IPP_IND_MOSI_SELECT_INPUT,         0x43FAC4A0,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_CSPI2_IPP_IND_SS0_B_SELECT_INPUT,        0x43FAC4A4,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_CSPI2_IPP_IND_SS1_B_SELECT_INPUT,        0x43FAC4A8,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_CSPI3_IPP_CSPI_CLK_IN_SELECT_INPUT,      0x43FAC4AC,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_CSPI3_IPP_IND_DATAREADY_B_SELECT_INPUT,  0x43FAC4B0,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_CSPI3_IPP_IND_MISO_SELECT_INPUT,         0x43FAC4B4,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_CSPI3_IPP_IND_MOSI_SELECT_INPUT,         0x43FAC4B8,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_CSPI3_IPP_IND_SS0_B_SELECT_INPUT,        0x43FAC4BC,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_CSPI3_IPP_IND_SS1_B_SELECT_INPUT,        0x43FAC4C0,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_CSPI3_IPP_IND_SS2_B_SELECT_INPUT,        0x43FAC4C4,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_CSPI3_IPP_IND_SS3_B_SELECT_INPUT,        0x43FAC4C8,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_ESDHC1_IPP_DAT4_IN_SELECT_INPUT,         0x43FAC4CC,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_ESDHC1_IPP_DAT5_IN_SELECT_INPUT,         0x43FAC4D0,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_ESDHC1_IPP_DAT6_IN_SELECT_INPUT,         0x43FAC4D4,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_ESDHC1_IPP_DAT7_IN_SELECT_INPUT,         0x43FAC4D8,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_ESDHC2_IPP_CARD_CLK_IN_SELECT_INPUT,     0x43FAC4DC,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_ESDHC2_IPP_CMD_IN_SELECT_INPUT,          0x43FAC4E0,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_ESDHC2_IPP_DAT0_IN_SELECT_INPUT,         0x43FAC4E4,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_ESDHC2_IPP_DAT1_IN_SELECT_INPUT,         0x43FAC4E8,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_ESDHC2_IPP_DAT2_IN_SELECT_INPUT,         0x43FAC4EC,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_ESDHC2_IPP_DAT3_IN_SELECT_INPUT,         0x43FAC4F0,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_ESDHC2_IPP_DAT4_IN_SELECT_INPUT,         0x43FAC4F4,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_ESDHC2_IPP_DAT5_IN_SELECT_INPUT,         0x43FAC4F8,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_ESDHC2_IPP_DAT6_IN_SELECT_INPUT,         0x43FAC4FC,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_ESDHC2_IPP_DAT7_IN_SELECT_INPUT,         0x43FAC500,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_FEC_FEC_COL_SELECT_INPUT,                0x43FAC504,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_FEC_FEC_CRS_SELECT_INPUT,                0x43FAC508,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_FEC_FEC_RDATA_2_SELECT_INPUT,            0x43FAC50C,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_FEC_FEC_RDATA_3_SELECT_INPUT,            0x43FAC510,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_FEC_FEC_RX_CLK_SELECT_INPUT,             0x43FAC514,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_FEC_FEC_RX_ER_SELECT_INPUT,              0x43FAC518,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_I2C2_IPP_SCL_IN_SELECT_INPUT,            0x43FAC51C,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_I2C2_IPP_SDA_IN_SELECT_INPUT,            0x43FAC520,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_I2C3_IPP_SCL_IN_SELECT_INPUT,            0x43FAC524,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_I2C3_IPP_SDA_IN_SELECT_INPUT,            0x43FAC528,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_KPP_IPP_IND_COL_4_SELECT_INPUT,          0x43FAC52C,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_KPP_IPP_IND_COL_5_SELECT_INPUT,          0x43FAC530,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_KPP_IPP_IND_COL_6_SELECT_INPUT,          0x43FAC534,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_KPP_IPP_IND_COL_7_SELECT_INPUT,          0x43FAC538,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_KPP_IPP_IND_ROW_4_SELECT_INPUT,          0x43FAC53C,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_KPP_IPP_IND_ROW_5_SELECT_INPUT,          0x43FAC540,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_KPP_IPP_IND_ROW_6_SELECT_INPUT,          0x43FAC544,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_KPP_IPP_IND_ROW_7_SELECT_INPUT,          0x43FAC548,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_SIM1_PIN_SIM_RCVD1_IN_SELECT_INPUT,      0x43FAC54C,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_SIM1_PIN_SIM_SIMPD1_SELECT_INPUT,        0x43FAC550,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_SIM1_SIM_RCVD1_IO_SELECT_INPUT,          0x43FAC554,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_SIM2_PIN_SIM_RCVD1_IN_SELECT_INPUT,      0x43FAC558,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_SIM2_PIN_SIM_SIMPD1_SELECT_INPUT,        0x43FAC55C,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_SIM2_SIM_RCVD1_IO_SELECT_INPUT,          0x43FAC560,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_UART3_IPP_UART_RTS_B_SELECT_INPUT,       0x43FAC564,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_UART3_IPP_UART_RXD_MUX_SELECT_INPUT,     0x43FAC568,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_UART4_IPP_UART_RTS_B_SELECT_INPUT,       0x43FAC56C,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_UART4_IPP_UART_RXD_MUX_SELECT_INPUT,     0x43FAC570,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_UART5_IPP_UART_RTS_B_SELECT_INPUT,       0x43FAC574,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_UART5_IPP_UART_RXD_MUX_SELECT_INPUT,     0x43FAC578,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_USB_TOP_IPP_IND_OTG_USB_OC_SELECT_INPUT, 0x43FAC57C,__READ_WRITE,__iomuxc_select_input_bits);
__IO_REG32_BIT(IOMUXC_USB_TOP_IPP_IND_UH2_USB_OC_SELECT_INPUT, 0x43FAC580,__READ_WRITE,__iomuxc_select_input_bits);

/***************************************************************************
 **
 **  1-Wire
 **
 ***************************************************************************/
__IO_REG16_BIT(OW_CONTROL,                0x43F9C000,__READ_WRITE ,__ow_control_bits);
__IO_REG16_BIT(OW_TIME_DIVIDER,           0x43F9C002,__READ_WRITE ,__ow_time_divider_bits);
__IO_REG16_BIT(OW_RESET,                  0x43F9C004,__READ_WRITE ,__ow_reset_bits);
__IO_REG16_BIT(OW_COMMAND,                0x43F9C006,__READ_WRITE ,__ow_command_bits);
__IO_REG16_BIT(OW_RX_TX,                  0x43F9C008,__READ_WRITE ,__ow_rx_tx_bits);
__IO_REG16_BIT(OW_INTERRUPT,              0x43F9C00A,__READ       ,__ow_interrupt_bits);
__IO_REG16_BIT(OW_INTERRUPT_EN,           0x43F9C00C,__READ_WRITE ,__ow_interrupt_en_bits);

/***************************************************************************
 **
 **  ASIC
 **
 ***************************************************************************/
__IO_REG32_BIT(INTCNTL,                   0x68000000,__READ_WRITE,__intcntl_bits);
__IO_REG32_BIT(NIMASK,                    0x68000004,__READ_WRITE,__nimask_bits);
__IO_REG32_BIT(INTENNUM,                  0x68000008,__READ_WRITE,__intennum_bits);
__IO_REG32_BIT(INTDISNUM,                 0x6000000C,__READ_WRITE,__intdisnum_bits);
__IO_REG32_BIT(INTENABLEH,                0x68000010,__READ_WRITE,__intenableh_bits);
__IO_REG32_BIT(INTENABLEL,                0x68000014,__READ_WRITE,__intenablel_bits);
__IO_REG32_BIT(INTTYPEH,                  0x68000018,__READ_WRITE,__inttypeh_bits);
__IO_REG32_BIT(INTTYPEL,                  0x6800001C,__READ_WRITE,__inttypel_bits);
__IO_REG32_BIT(NIPRIORITY7,               0x68000020,__READ_WRITE,__nipriority7_bits);
__IO_REG32_BIT(NIPRIORITY6,               0x68000024,__READ_WRITE,__nipriority6_bits);
__IO_REG32_BIT(NIPRIORITY5,               0x68000028,__READ_WRITE,__nipriority5_bits);
__IO_REG32_BIT(NIPRIORITY4,               0x6800002C,__READ_WRITE,__nipriority4_bits);
__IO_REG32_BIT(NIPRIORITY3,               0x68000030,__READ_WRITE,__nipriority3_bits);
__IO_REG32_BIT(NIPRIORITY2,               0x68000034,__READ_WRITE,__nipriority2_bits);
__IO_REG32_BIT(NIPRIORITY1,               0x68000038,__READ_WRITE,__nipriority1_bits);
__IO_REG32_BIT(NIPRIORITY0,               0x6800003C,__READ_WRITE,__nipriority0_bits);
__IO_REG32_BIT(NIVECSR,                   0x68000040,__READ      ,__nivecsr_bits);
__IO_REG32(    FIVECSR,                   0x68000044,__READ      );
__IO_REG32_BIT(INTSRCH,                   0x68000048,__READ      ,__intsrch_bits);
__IO_REG32_BIT(INTSRCL,                   0x6800004C,__READ      ,__intsrcl_bits);
__IO_REG32_BIT(INTFRCH,                   0x68000050,__READ_WRITE,__intfrch_bits);
__IO_REG32_BIT(INTFRCL,                   0x68000054,__READ_WRITE,__intfrcl_bits);
__IO_REG32_BIT(NIPNDH,                    0x68000058,__READ      ,__nipndh_bits);
__IO_REG32_BIT(NIPNDL,                    0x6800005C,__READ      ,__nipndl_bits);
__IO_REG32_BIT(FIPNDH,                    0x68000060,__READ      ,__fipndh_bits);
__IO_REG32_BIT(FIPNDL,                    0x68000064,__READ      ,__fipndl_bits);

/***************************************************************************
 **
 **  ATA
 **
 ***************************************************************************/
__IO_REG8(     ATA_TIME_OFF,              0x50020000,__READ_WRITE);
__IO_REG8(     ATA_TIME_ON,               0x50020001,__READ_WRITE);
__IO_REG8(     ATA_TIME_1,                0x50020002,__READ_WRITE);
__IO_REG8(     ATA_TIME_2W,               0x50020003,__READ_WRITE);
__IO_REG8(     ATA_TIME_2R,               0x50020004,__READ_WRITE);
__IO_REG8(     ATA_TIME_AX,               0x50020005,__READ_WRITE);
__IO_REG8(     ATA_TIME_RDX,              0x50020006,__READ_WRITE);
__IO_REG8(     ATA_TIME_4,                0x50020007,__READ_WRITE);
__IO_REG8(     ATA_TIME_9,                0x50020008,__READ_WRITE);
__IO_REG8(     ATA_TIME_M,                0x50020009,__READ_WRITE);
__IO_REG8(     ATA_TIME_JN,               0x5002000A,__READ_WRITE);
__IO_REG8(     ATA_TIME_D,                0x5002000B,__READ_WRITE);
__IO_REG8(     ATA_TIME_K,                0x5002000C,__READ_WRITE);
__IO_REG8(     ATA_TIME_ACK,              0x5002000D,__READ_WRITE);
__IO_REG8(     ATA_TIME_ENV,              0x5002000E,__READ_WRITE);
__IO_REG8(     ATA_TIME_RPX,              0x5002000F,__READ_WRITE);
__IO_REG8(     ATA_TIME_ZAH,              0x50020010,__READ_WRITE);
__IO_REG8(     ATA_TIME_MLIX,             0x50020011,__READ_WRITE);
__IO_REG8(     ATA_TIME_DVH,              0x50020012,__READ_WRITE);
__IO_REG8(     ATA_TIME_DZFS,             0x50020013,__READ_WRITE);
__IO_REG8(     ATA_TIME_DVS,              0x50020014,__READ_WRITE);
__IO_REG8(     ATA_TIME_CVH,              0x50020015,__READ_WRITE);
__IO_REG8(     ATA_TIME_SS,               0x50020016,__READ_WRITE);
__IO_REG8(     ATA_TIME_CYC,              0x50020017,__READ_WRITE);
__IO_REG32(    ATA_FIFO_DATA_32,          0x50020018,__READ_WRITE);
__IO_REG16(    ATA_FIFO_DATA_16,          0x5002001C,__READ_WRITE);
__IO_REG8(     ATA_FIFO_FILL,             0x50020020,__READ_WRITE);
__IO_REG16_BIT(ATA_CONTROL,               0x50020024,__READ_WRITE, __ata_control_bits);
__IO_REG8_BIT( ATA_INTERRUPT_PENDING,     0x50020028,__READ_WRITE, __ata_interrupt_pending_bits);
__IO_REG8_BIT( ATA_INTERRUPT_ENABLE,      0x5002002C,__READ_WRITE, __ata_interrupt_pending_bits);
__IO_REG8_BIT( ATA_INTERRUPT_CLEAR,       0x50020030,__WRITE     , __ata_interrupt_clear_bits);
__IO_REG8(     ATA_FIFO_ALARM,            0x50020034,__READ_WRITE);
__IO_REG8_BIT( ATA_ADMA_ERR_STATUS,       0x50020038,__WRITE     , __ata_adma_err_status_bits);
__IO_REG32(    ATA_SYS_DMA_BADDR,         0x5002003C,__READ_WRITE);
__IO_REG32(    ATA_SYS_ADMA_SYS_ADDR,     0x50020040,__READ_WRITE);
__IO_REG16(    ATA_BLOCK_CNT,             0x50020048,__READ_WRITE);
__IO_REG8_BIT( ATA_BURST_LENGTH,          0x5002004C,__READ_WRITE, __ata_burst_length_bits);
__IO_REG16(    ATA_SECTOR_SIZE,           0x50020050,__READ_WRITE);

/***************************************************************************
 **
 **  AUDMUX
 **
 ***************************************************************************/
__IO_REG32_BIT(PTCR1,                     0x43FB0000,__READ_WRITE,__ptcr_bits);
__IO_REG32_BIT(PDCR1,                     0x43FB0004,__READ_WRITE,__pdcr_bits);
__IO_REG32_BIT(PTCR2,                     0x43FB0008,__READ_WRITE,__ptcr_bits);
__IO_REG32_BIT(PDCR2,                     0x43FB000C,__READ_WRITE,__pdcr_bits);
__IO_REG32_BIT(PTCR3,                     0x43FB0010,__READ_WRITE,__ptcr_bits);
__IO_REG32_BIT(PDCR3,                     0x43FB0014,__READ_WRITE,__pdcr_bits);
__IO_REG32_BIT(PTCR4,                     0x43FB0018,__READ_WRITE,__ptcr_bits);
__IO_REG32_BIT(PDCR4,                     0x43FB001C,__READ_WRITE,__pdcr_bits);
__IO_REG32_BIT(PTCR5,                     0x43FB0020,__READ_WRITE,__ptcr_bits);
__IO_REG32_BIT(PDCR5,                     0x43FB0024,__READ_WRITE,__pdcr_bits);
__IO_REG32_BIT(PTCR6,                     0x43FB0028,__READ_WRITE,__ptcr_bits);
__IO_REG32_BIT(PDCR6,                     0x43FB002C,__READ_WRITE,__pdcr_bits);
__IO_REG32_BIT(PTCR7,                     0x43FB0030,__READ_WRITE,__ptcr_bits);
__IO_REG32_BIT(PDCR7,                     0x43FB0034,__READ_WRITE,__pdcr_bits);
__IO_REG32_BIT(CNMCR,                     0x43FB0038,__READ_WRITE,__cnmcr_bits);

/***************************************************************************
 **
 **  CRM
 **
 ***************************************************************************/
__IO_REG32_BIT(MPCTL,                     0x53F80000,__READ_WRITE,__mpctl_bits);
__IO_REG32_BIT(UPCTL,                     0x53F80004,__READ_WRITE,__mpctl_bits);
__IO_REG32_BIT(CCTL,                      0x53F80008,__READ_WRITE,__cctl_bits);
__IO_REG32_BIT(CGCR0,                     0x53F8000C,__READ_WRITE,__cgcr0_bits);
__IO_REG32_BIT(CGCR1,                     0x53F80010,__READ_WRITE,__cgcr1_bits);
__IO_REG32_BIT(CGCR2,                     0x53F80014,__READ_WRITE,__cgcr2_bits);
__IO_REG32_BIT(PCDR0,                     0x53F80018,__READ_WRITE,__pcdr0_bits);
__IO_REG32_BIT(PCDR1,                     0x53F8001C,__READ_WRITE,__pcdr1_bits);
__IO_REG32_BIT(PCDR2,                     0x53F80020,__READ_WRITE,__pcdr2_bits);
__IO_REG32_BIT(PCDR3,                     0x53F80024,__READ_WRITE,__pcdr3_bits);
__IO_REG32_BIT(RCSR,                      0x53F80028,__READ_WRITE,__rcsr_bits);
__IO_REG32_BIT(CRDR,                      0x53F8002C,__READ_WRITE,__crdr_bits);
__IO_REG32_BIT(DCVR0,                     0x53F80030,__READ_WRITE,__dcvr_bits);
__IO_REG32_BIT(DCVR1,                     0x53F80034,__READ_WRITE,__dcvr_bits);
__IO_REG32_BIT(DCVR2,                     0x53F80038,__READ_WRITE,__dcvr_bits);
__IO_REG32_BIT(DCVR3,                     0x53F8003C,__READ_WRITE,__dcvr_bits);
__IO_REG32_BIT(LTR0,                      0x53F80040,__READ_WRITE,__ltr0_bits);
__IO_REG32_BIT(LTR1,                      0x53F80044,__READ_WRITE,__ltr1_bits);
__IO_REG32_BIT(LTR2,                      0x53F80048,__READ_WRITE,__ltr2_bits);
__IO_REG32_BIT(LTR3,                      0x53F8004C,__READ_WRITE,__ltr3_bits);
__IO_REG32_BIT(LTBR0,                     0x53F80050,__READ_WRITE,__ltbr0_bits);
__IO_REG32_BIT(LTBR1,                     0x53F80054,__READ_WRITE,__ltbr1_bits);
__IO_REG32_BIT(PMCR0,                     0x53F80058,__READ_WRITE,__pmcr0_bits);
__IO_REG32_BIT(PMCR1,                     0x53F8005C,__READ_WRITE,__pmcr1_bits);
__IO_REG32_BIT(PMCR2,                     0x53F80060,__READ_WRITE,__pmcr2_bits);
__IO_REG32_BIT(MCR,                       0x53F80064,__READ_WRITE,__mcr_bits);
__IO_REG32_BIT(LPIMR0,                    0x53F80068,__READ_WRITE,__lpimr0_bits);
__IO_REG32_BIT(LPIMR1,                    0x53F8006C,__READ_WRITE,__lpimr1_bits);

/***************************************************************************
 **
 **  CSI
 **
 ***************************************************************************/
__IO_REG32_BIT(CSICR1,                    0x53FF8000,__READ_WRITE,__csicr1_bits);
__IO_REG32_BIT(CSICR2,                    0x53FF8004,__READ_WRITE,__csicr2_bits);
__IO_REG32_BIT(CSICR3,                    0x53FF8008,__READ_WRITE,__csicr3_bits);
__IO_REG32(    CSISTATFIFO,               0x53FF800C,__READ      );
__IO_REG32(    CSIRFIFO,                  0x53FF8010,__READ      );
__IO_REG32_BIT(CSIRXCNT,                  0x53FF8014,__READ_WRITE,__csirxcnt_bits);
__IO_REG32_BIT(CSISR,                     0x53FF8018,__READ_WRITE,__csisr_bits);
__IO_REG32(    CSIDMASA_STATFIFO,         0x53FF8020,__READ_WRITE);
__IO_REG32(    CSIDMATS_STATFIFO,         0x53FF8024,__READ_WRITE);
__IO_REG32(    CSIDMASA_FB1,              0x53FF8028,__READ_WRITE);
__IO_REG32(    CSIDMASA_FB2,              0x53FF802C,__READ_WRITE);
__IO_REG32_BIT(CSIFBUF_PARA,              0x53FF8030,__READ_WRITE,__csifbuf_para_bits);
__IO_REG32_BIT(CSIIMAG_PARA,              0x53FF8034,__READ_WRITE,__csiimag_para_bits);

/***************************************************************************
 **
 **  CSPI1
 **
 ***************************************************************************/
__IO_REG32(    CSPI1_RXDATA,              0x43FA4000,__READ      );
__IO_REG32(    CSPI1_TXDATA,              0x43FA4004,__WRITE     );
__IO_REG32_BIT(CSPI1_CONREG,              0x43FA4008,__READ_WRITE,__cspi_controlreg_bits);
__IO_REG32_BIT(CSPI1_INTREG,              0x43FA400C,__READ_WRITE,__cspi_intreg_bits);
__IO_REG32_BIT(CSPI1_DMAREG,              0x43FA4010,__READ_WRITE,__cspi_dma_bits);
__IO_REG32_BIT(CSPI1_STATREG,             0x43FA4014,__READ_WRITE,__cspi_statreg_bits);
__IO_REG32_BIT(CSPI1_PERIODREG,           0x43FA4018,__READ_WRITE,__cspi_period_bits);
__IO_REG32_BIT(CSPI1_TESTREG,             0x43FA401C,__READ_WRITE,__cspi_test_bits);

/***************************************************************************
 **
 **  CSPI2
 **
 ***************************************************************************/
__IO_REG32(    CSPI2_RXDATA,              0x50010000,__READ      );
__IO_REG32(    CSPI2_TXDATA,              0x50010004,__WRITE     );
__IO_REG32_BIT(CSPI2_CONREG,              0x50010008,__READ_WRITE,__cspi_controlreg_bits);
__IO_REG32_BIT(CSPI2_INTREG,              0x5001000C,__READ_WRITE,__cspi_intreg_bits);
__IO_REG32_BIT(CSPI2_DMAREG,              0x50010010,__READ_WRITE,__cspi_dma_bits);
__IO_REG32_BIT(CSPI2_STATREG,             0x50010014,__READ_WRITE,__cspi_statreg_bits);
__IO_REG32_BIT(CSPI2_PERIODREG,           0x50010018,__READ_WRITE,__cspi_period_bits);
__IO_REG32_BIT(CSPI2_TESTREG,             0x5001001C,__READ_WRITE,__cspi_test_bits);

/***************************************************************************
 **
 **  CSPI3
 **
 ***************************************************************************/
__IO_REG32(    CSPI3_RXDATA,              0x50004000,__READ      );
__IO_REG32(    CSPI3_TXDATA,              0x50004004,__WRITE     );
__IO_REG32_BIT(CSPI3_CONREG,              0x50004008,__READ_WRITE,__cspi_controlreg_bits);
__IO_REG32_BIT(CSPI3_INTREG,              0x5000400C,__READ_WRITE,__cspi_intreg_bits);
__IO_REG32_BIT(CSPI3_DMAREG,              0x50004010,__READ_WRITE,__cspi_dma_bits);
__IO_REG32_BIT(CSPI3_STATREG,             0x50004014,__READ_WRITE,__cspi_statreg_bits);
__IO_REG32_BIT(CSPI3_PERIODREG,           0x50004018,__READ_WRITE,__cspi_period_bits);
__IO_REG32_BIT(CSPI3_TESTREG,             0x5000401C,__READ_WRITE,__cspi_test_bits);

/***************************************************************************
 **
 **  ECTA
 **
 ***************************************************************************/
__IO_REG32_BIT(CTIACONTROL,               0x43FB8000,__READ_WRITE,__cticontrol_bits);
__IO_REG32_BIT(CTIASTATUS,                0x43FB8004,__READ      ,__ctistatus_bits);
__IO_REG32(    CTIALOCK,                  0x43FB8008,__WRITE);
__IO_REG32_BIT(CTIAPROTECTION,            0x43FB800C,__READ_WRITE,__ctiprotection_bits);
__IO_REG32_BIT(CTIAINTACK,                0x43FB8010,__WRITE     ,__ctiintack_bits);
__IO_REG32_BIT(CTIAAPPSET,                0x43FB8014,__READ_WRITE,__ctiappset_bits);
__IO_REG32_BIT(CTIAAPPCLEAR,              0x43FB8018,__WRITE     ,__ctiappclear_bits);
__IO_REG32_BIT(CTIAAPPPULSE,              0x43FB801C,__WRITE     ,__ctiapppulse_bits);
__IO_REG32_BIT(CTIAINEN0,                 0x43FB8020,__READ_WRITE,__ctiinen_bits);
__IO_REG32_BIT(CTIAINEN1,                 0x43FB8024,__READ_WRITE,__ctiinen_bits);
__IO_REG32_BIT(CTIAINEN2,                 0x43FB8028,__READ_WRITE,__ctiinen_bits);
__IO_REG32_BIT(CTIAINEN3,                 0x43FB802C,__READ_WRITE,__ctiinen_bits);
__IO_REG32_BIT(CTIAINEN4,                 0x43FB8030,__READ_WRITE,__ctiinen_bits);
__IO_REG32_BIT(CTIAINEN5,                 0x43FB8034,__READ_WRITE,__ctiinen_bits);
__IO_REG32_BIT(CTIAINEN6,                 0x43FB8038,__READ_WRITE,__ctiinen_bits);
__IO_REG32_BIT(CTIAINEN7,                 0x43FB803C,__READ_WRITE,__ctiinen_bits);
__IO_REG32_BIT(CTIAOUTEN0,                0x43FB80A0,__READ_WRITE,__ctiouten_bits);
__IO_REG32_BIT(CTIAOUTEN1,                0x43FB80A4,__READ_WRITE,__ctiouten_bits);
__IO_REG32_BIT(CTIAOUTEN2,                0x43FB80A8,__READ_WRITE,__ctiouten_bits);
__IO_REG32_BIT(CTIAOUTEN3,                0x43FB80AC,__READ_WRITE,__ctiouten_bits);
__IO_REG32_BIT(CTIAOUTEN4,                0x43FB80B0,__READ_WRITE,__ctiouten_bits);
__IO_REG32_BIT(CTIAOUTEN5,                0x43FB80B4,__READ_WRITE,__ctiouten_bits);
__IO_REG32_BIT(CTIAOUTEN6,                0x43FB80B8,__READ_WRITE,__ctiouten_bits);
__IO_REG32_BIT(CTIAOUTEN7,                0x43FB80BC,__READ_WRITE,__ctiouten_bits);
__IO_REG32_BIT(CTIATRIGINSTATUS,          0x43FB8130,__READ      ,__ctitriginstatus_bits);
__IO_REG32_BIT(CTIATRIGOUTSTATUS,         0x43FB8134,__READ      ,__ctitrigoutstatus_bits);
__IO_REG32_BIT(CTIACHINSTATUS,            0x43FB8138,__READ      ,__ctichinstatus_bits);
__IO_REG32_BIT(CTIACHOUTSTATUS,           0x43FB813C,__READ      ,__ctichoutstatus_bits);
__IO_REG32_BIT(CTIATCR,                   0x43FB8200,__READ_WRITE,__ctitcr_bits);
__IO_REG32_BIT(CTIAITIP0,                 0x43FB8204,__READ_WRITE,__ctiitip0_bits);
__IO_REG32_BIT(CTIAITIP1,                 0x43FB8208,__READ_WRITE,__ctiitip1_bits);
__IO_REG32_BIT(CTIAITIP2,                 0x43FB820C,__READ_WRITE,__ctiitip2_bits);
__IO_REG32_BIT(CTIAITIP3,                 0x43FB8210,__READ_WRITE,__ctiitip3_bits);
__IO_REG32_BIT(CTIAITOP0,                 0x43FB8214,__READ_WRITE,__ctiitop0_bits);
__IO_REG32_BIT(CTIAITOP1,                 0x43FB8218,__READ_WRITE,__ctiitop1_bits);
__IO_REG32_BIT(CTIAITOP2,                 0x43FB821C,__READ_WRITE,__ctiitop2_bits);
__IO_REG32_BIT(CTIAITOP3,                 0x43FB8220,__READ_WRITE,__ctiitop3_bits);
__IO_REG32_BIT(CTIAPERIPHID0,             0x43FB8FE0,__READ      ,__ctiperiphid0_bits);
__IO_REG32_BIT(CTIAPERIPHID1,             0x43FB8FE4,__READ      ,__ctiperiphid1_bits);
__IO_REG32_BIT(CTIAPERIPHID2,             0x43FB8FE8,__READ      ,__ctiperiphid2_bits);
__IO_REG32_BIT(CTIAPERIPHID3,             0x43FB8FEC,__READ      ,__ctiperiphid3_bits);
__IO_REG32_BIT(CTIAPCELLID0,              0x43FB8FF0,__READ      ,__ctipcellid0_bits);
__IO_REG32_BIT(CTIAPCELLID1,              0x43FB8FF4,__READ      ,__ctipcellid1_bits);
__IO_REG32_BIT(CTIAPCELLID2,              0x43FB8FF8,__READ      ,__ctipcellid2_bits);
__IO_REG32_BIT(CTIAPCELLID3,              0x43FB8FFC,__READ      ,__ctipcellid3_bits);

/***************************************************************************
 **
 **  ECTB
 **
 ***************************************************************************/
__IO_REG32_BIT(CTIBCONTROL,               0x43FBC000,__READ_WRITE,__cticontrol_bits);
__IO_REG32_BIT(CTIBSTATUS,                0x43FBC004,__READ      ,__ctistatus_bits);
__IO_REG32(    CTIBLOCK,                  0x43FBC008,__WRITE);
__IO_REG32_BIT(CTIBPROTECTION,            0x43FBC00C,__READ_WRITE,__ctiprotection_bits);
__IO_REG32_BIT(CTIBINTACK,                0x43FBC010,__WRITE     ,__ctiintack_bits);
__IO_REG32_BIT(CTIBAPPSET,                0x43FBC014,__READ_WRITE,__ctiappset_bits);
__IO_REG32_BIT(CTIBAPPCLEAR,              0x43FBC018,__WRITE     ,__ctiappclear_bits);
__IO_REG32_BIT(CTIBAPPPULSE,              0x43FBC01C,__WRITE     ,__ctiapppulse_bits);
__IO_REG32_BIT(CTIBINEN0,                 0x43FBC020,__READ_WRITE,__ctiinen_bits);
__IO_REG32_BIT(CTIBINEN1,                 0x43FBC024,__READ_WRITE,__ctiinen_bits);
__IO_REG32_BIT(CTIBINEN2,                 0x43FBC028,__READ_WRITE,__ctiinen_bits);
__IO_REG32_BIT(CTIBINEN3,                 0x43FBC02C,__READ_WRITE,__ctiinen_bits);
__IO_REG32_BIT(CTIBINEN4,                 0x43FBC030,__READ_WRITE,__ctiinen_bits);
__IO_REG32_BIT(CTIBINEN5,                 0x43FBC034,__READ_WRITE,__ctiinen_bits);
__IO_REG32_BIT(CTIBINEN6,                 0x43FBC038,__READ_WRITE,__ctiinen_bits);
__IO_REG32_BIT(CTIBINEN7,                 0x43FBC03C,__READ_WRITE,__ctiinen_bits);
__IO_REG32_BIT(CTIBOUTEN0,                0x43FBC0A0,__READ_WRITE,__ctiouten_bits);
__IO_REG32_BIT(CTIBOUTEN1,                0x43FBC0A4,__READ_WRITE,__ctiouten_bits);
__IO_REG32_BIT(CTIBOUTEN2,                0x43FBC0A8,__READ_WRITE,__ctiouten_bits);
__IO_REG32_BIT(CTIBOUTEN3,                0x43FBC0AC,__READ_WRITE,__ctiouten_bits);
__IO_REG32_BIT(CTIBOUTEN4,                0x43FBC0B0,__READ_WRITE,__ctiouten_bits);
__IO_REG32_BIT(CTIBOUTEN5,                0x43FBC0B4,__READ_WRITE,__ctiouten_bits);
__IO_REG32_BIT(CTIBOUTEN6,                0x43FBC0B8,__READ_WRITE,__ctiouten_bits);
__IO_REG32_BIT(CTIBOUTEN7,                0x43FBC0BC,__READ_WRITE,__ctiouten_bits);
__IO_REG32_BIT(CTIBTRIGINSTATUS,          0x43FBC130,__READ      ,__ctitriginstatus_bits);
__IO_REG32_BIT(CTIBTRIGOUTSTATUS,         0x43FBC134,__READ      ,__ctitrigoutstatus_bits);
__IO_REG32_BIT(CTIBCHINSTATUS,            0x43FBC138,__READ      ,__ctichinstatus_bits);
__IO_REG32_BIT(CTIBCHOUTSTATUS,           0x43FBC13C,__READ      ,__ctichoutstatus_bits);
__IO_REG32_BIT(CTIBTCR,                   0x43FBC200,__READ_WRITE,__ctitcr_bits);
__IO_REG32_BIT(CTIBITIP0,                 0x43FBC204,__READ_WRITE,__ctiitip0_bits);
__IO_REG32_BIT(CTIBITIP1,                 0x43FBC208,__READ_WRITE,__ctiitip1_bits);
__IO_REG32_BIT(CTIBITIP2,                 0x43FBC20C,__READ_WRITE,__ctiitip2_bits);
__IO_REG32_BIT(CTIBITIP3,                 0x43FBC210,__READ_WRITE,__ctiitip3_bits);
__IO_REG32_BIT(CTIBITOP0,                 0x43FBC214,__READ_WRITE,__ctiitop0_bits);
__IO_REG32_BIT(CTIBITOP1,                 0x43FBC218,__READ_WRITE,__ctiitop1_bits);
__IO_REG32_BIT(CTIBITOP2,                 0x43FBC21C,__READ_WRITE,__ctiitop2_bits);
__IO_REG32_BIT(CTIBITOP3,                 0x43FBC220,__READ_WRITE,__ctiitop3_bits);
__IO_REG32_BIT(CTIBPERIPHID0,             0x43FBCFE0,__READ      ,__ctiperiphid0_bits);
__IO_REG32_BIT(CTIBPERIPHID1,             0x43FBCFE4,__READ      ,__ctiperiphid1_bits);
__IO_REG32_BIT(CTIBPERIPHID2,             0x43FBCFE8,__READ      ,__ctiperiphid2_bits);
__IO_REG32_BIT(CTIBPERIPHID3,             0x43FBCFEC,__READ      ,__ctiperiphid3_bits);
__IO_REG32_BIT(CTIBPCELLID0,              0x43FBCFF0,__READ      ,__ctipcellid0_bits);
__IO_REG32_BIT(CTIBPCELLID1,              0x43FBCFF4,__READ      ,__ctipcellid1_bits);
__IO_REG32_BIT(CTIBPCELLID2,              0x43FBCFF8,__READ      ,__ctipcellid2_bits);
__IO_REG32_BIT(CTIBPCELLID3,              0x43FBCFFC,__READ      ,__ctipcellid3_bits);

/***************************************************************************
 **
 **  DRYICE
 **
 ***************************************************************************/
__IO_REG32(    DTCMR,                     0x53FFC000,__READ_WRITE);
__IO_REG32_BIT(DTCLR,                     0x53FFC004,__READ_WRITE,__dtclr_bits);
__IO_REG32(    DCAMR,                     0x53FFC008,__READ_WRITE);
__IO_REG32_BIT(DCALR,                     0x53FFC00C,__READ_WRITE,__dcalr_bits);
__IO_REG32_BIT(DCR,                       0x53FFC010,__READ_WRITE,__dcr_bits);
__IO_REG32_BIT(DSR,                       0x53FFC014,__READ_WRITE,__dsr_bits);
__IO_REG32_BIT(DIER,                      0x53FFC018,__READ_WRITE,__dier_bits);
__IO_REG32(    DMCR,                      0x53FFC01C,__READ_WRITE);
__IO_REG32_BIT(DKSR,                      0x53FFC020,__READ_WRITE,__dksr_bits);
__IO_REG32_BIT(DKCR,                      0x53FFC024,__READ_WRITE,__dkcr_bits);
__IO_REG32_BIT(DTCR,                      0x53FFC028,__READ_WRITE,__dtcr_bits);
__IO_REG32_BIT(DACR,                      0x53FFC02C,__READ_WRITE,__dacr_bits);
__IO_REG32(    DGPR,                      0x53FFC03C,__READ_WRITE);
__IO_REG32(    DPKR0,                     0x53FFC040,__READ);
__IO_REG32(    DPKR1,                     0x53FFC044,__READ);
__IO_REG32(    DPKR2,                     0x53FFC048,__READ);
__IO_REG32(    DPKR3,                     0x53FFC04C,__READ);
__IO_REG32(    DPKR4,                     0x53FFC050,__READ);
__IO_REG32(    DPKR5,                     0x53FFC054,__READ);
__IO_REG32(    DPKR6,                     0x53FFC058,__READ);
__IO_REG32(    DPKR7,                     0x53FFC05C,__READ);
__IO_REG32(    DRKR0,                     0x53FFC060,__READ);
__IO_REG32(    DRKR1,                     0x53FFC064,__READ);
__IO_REG32(    DRKR2,                     0x53FFC068,__READ);
__IO_REG32(    DRKR3,                     0x53FFC06C,__READ);
__IO_REG32(    DRKR4,                     0x53FFC070,__READ);
__IO_REG32(    DRKR5,                     0x53FFC074,__READ);
__IO_REG32(    DRKR6,                     0x53FFC078,__READ);
__IO_REG32(    DRKR7,                     0x53FFC07C,__READ);

/***************************************************************************
 **
 **  M3IF
 **
 ***************************************************************************/
__IO_REG32_BIT(M3IFCTL,                   0xB8003000,__READ_WRITE,__m3ifctl_bits);
__IO_REG32_BIT(M3IFWCFG0,                 0xB8003004,__READ_WRITE,__m3ifwcfg_bits);
__IO_REG32_BIT(M3IFWCFG1,                 0xB8003008,__READ_WRITE,__m3ifwcfg_bits);
__IO_REG32_BIT(M3IFWCFG2,                 0xB800300C,__READ_WRITE,__m3ifwcfg_bits);
__IO_REG32_BIT(M3IFWCFG3,                 0xB8003010,__READ_WRITE,__m3ifwcfg_bits);
__IO_REG32_BIT(M3IFWCFG4,                 0xB8003014,__READ_WRITE,__m3ifwcfg_bits);
__IO_REG32_BIT(M3IFWCFG5,                 0xB8003018,__READ_WRITE,__m3ifwcfg_bits);
__IO_REG32_BIT(M3IFWCFG6,                 0xB800301C,__READ_WRITE,__m3ifwcfg_bits);
__IO_REG32_BIT(M3IFWCFG7,                 0xB8003020,__READ_WRITE,__m3ifwcfg_bits);
__IO_REG32_BIT(M3IFWCSR,                  0xB8003024,__READ_WRITE,__m3ifwcsr_bits);
__IO_REG32_BIT(M3IFSCFG0,                 0xB8003028,__READ_WRITE,__m3ifscfg0_bits);
__IO_REG32_BIT(M3IFSCFG1,                 0xB800302C,__READ_WRITE,__m3ifscfg1_bits);
__IO_REG32_BIT(M3IFSCFG2,                 0xB8003030,__READ_WRITE,__m3ifscfg2_bits);
__IO_REG32_BIT(M3IFSSR0,                  0xB8003034,__READ_WRITE,__m3ifssr0_bits);
__IO_REG32_BIT(M3IFSSR1,                  0xB8003038,__READ_WRITE,__m3ifssr1_bits);
__IO_REG32_BIT(M3IFMLWE0,                 0xB8003040,__READ_WRITE,__m3ifmlwe_bits);
__IO_REG32_BIT(M3IFMLWE1,                 0xB8003044,__READ_WRITE,__m3ifmlwe_bits);
__IO_REG32_BIT(M3IFMLWE2,                 0xB8003048,__READ_WRITE,__m3ifmlwe_bits);
__IO_REG32_BIT(M3IFMLWE3,                 0xB800304C,__READ_WRITE,__m3ifmlwe_bits);
__IO_REG32_BIT(M3IFMLWE4,                 0xB8003050,__READ_WRITE,__m3ifmlwe_bits);
__IO_REG32_BIT(M3IFMLWE5,                 0xB8003054,__READ_WRITE,__m3ifmlwe_bits);

/***************************************************************************
 **
 **  NFC
 **
 ***************************************************************************/
__IO_REG16_BIT(RAM_BUFFER_ADDRESS,        0xBB001E04,__READ_WRITE,__nfc_rba_bits);
__IO_REG16(    NAND_FLASH_ADD,            0xBB001E06,__READ_WRITE);
__IO_REG16(    NAND_FLASH_CMD,            0xBB001E08,__READ_WRITE);
__IO_REG16_BIT(NFC_CONFIGURATION,         0xBB001E0A,__READ_WRITE,__nfc_iblc_bits);
__IO_REG16_BIT(ECC_STATUS_RESULT1,        0xBB001E0C,__READ      ,__ecc_srr_bits);
__IO_REG16_BIT(ECC_STATUS_RESULT2,        0xBB001E0E,__READ      ,__ecc_srr2_bits);
__IO_REG16_BIT(NFC_SPAS,                  0xBB001E10,__READ_WRITE,__nfc_spas_bits);
__IO_REG16_BIT(NF_WR_PROT,                0xBB001E12,__READ_WRITE,__nf_wr_prot_bits);
__IO_REG16_BIT(NAND_FLASH_WR_PR_ST,       0xBB001E18,__READ_WRITE,__nf_wr_prot_sta_bits);
__IO_REG16_BIT(NAND_FLASH_CONFIG1,        0xBB001E1A,__READ_WRITE,__nand_fc1_bits);
__IO_REG16_BIT(NAND_FLASH_CONFIG2,        0xBB001E1C,__READ_WRITE,__nand_fc2_bits);
__IO_REG16(    UNLOCK_START_BLK_ADD,      0xBB001E20,__READ_WRITE);
__IO_REG16(    UNLOCK_END_BLK_ADD,        0xBB001E22,__READ_WRITE);
__IO_REG16(    UNLOCK_START_BLK_ADD1,     0xBB001E24,__READ_WRITE);
__IO_REG16(    UNLOCK_END_BLK_ADD1,       0xBB001E26,__READ_WRITE);
__IO_REG16(    UNLOCK_START_BLK_ADD2,     0xBB001E28,__READ_WRITE);
__IO_REG16(    UNLOCK_END_BLK_ADD2,       0xBB001E2A,__READ_WRITE);
__IO_REG16(    UNLOCK_START_BLK_ADD3,     0xBB001E2C,__READ_WRITE);
__IO_REG16(    UNLOCK_END_BLK_ADD3,       0xBB001E2E,__READ_WRITE);

/***************************************************************************
 **
 **  ESDRAMC
 **
 ***************************************************************************/
__IO_REG32_BIT(ESDCTL0,                   0xB8001000,__READ_WRITE,__esdctl_bits);
__IO_REG32_BIT(ESDCFG0,                   0xB8001004,__READ_WRITE,__esdcfg_bits);
__IO_REG32_BIT(ESDCTL1,                   0xB8001008,__READ_WRITE,__esdctl_bits);
__IO_REG32_BIT(ESDCFG1,                   0xB800100C,__READ_WRITE,__esdcfg_bits);
__IO_REG32_BIT(ESDMISC,                   0xB8001010,__READ_WRITE,__esdmisc_bits);
__IO_REG32_BIT(ESDCDLY1,                  0xB8001020,__READ_WRITE,__esdcdly_bits);
__IO_REG32_BIT(ESDCDLY2,                  0xB8001024,__READ_WRITE,__esdcdly_bits);
__IO_REG32_BIT(ESDCDLY5,                  0xB8001030,__READ_WRITE,__esdcdly_bits);
__IO_REG32_BIT(ESDCDLYL,                  0xB8001034,__READ_WRITE,__esdcdlyl_bits);
__IO_REG32_BIT(ESDCDLY6,                  0xB8001038,__READ_WRITE,__esdcdly6_bits);

/***************************************************************************
 **
 **  WEIM
 **
 ***************************************************************************/
__IO_REG32_BIT(CSCR0U,                    0xB8002000,__READ_WRITE,__cscru_bits);
__IO_REG32_BIT(CSCR0L,                    0xB8002004,__READ_WRITE,__cscrl_bits);
__IO_REG32_BIT(CSCR0A,                    0xB8002008,__READ_WRITE,__cscra_bits);
__IO_REG32_BIT(CSCR1U,                    0xB8002010,__READ_WRITE,__cscru_bits);
__IO_REG32_BIT(CSCR1L,                    0xB8002014,__READ_WRITE,__cscrl_bits);
__IO_REG32_BIT(CSCR1A,                    0xB8002018,__READ_WRITE,__cscra_bits);
__IO_REG32_BIT(CSCR2U,                    0xB8002020,__READ_WRITE,__cscru_bits);
__IO_REG32_BIT(CSCR2L,                    0xB8002024,__READ_WRITE,__cscrl_bits);
__IO_REG32_BIT(CSCR2A,                    0xB8002028,__READ_WRITE,__cscra_bits);
__IO_REG32_BIT(CSCR3U,                    0xB8002030,__READ_WRITE,__cscru_bits);
__IO_REG32_BIT(CSCR3L,                    0xB8002034,__READ_WRITE,__cscrl_bits);
__IO_REG32_BIT(CSCR3A,                    0xB8002038,__READ_WRITE,__cscra_bits);
__IO_REG32_BIT(CSCR4U,                    0xB8002040,__READ_WRITE,__cscru_bits);
__IO_REG32_BIT(CSCR4L,                    0xB8002044,__READ_WRITE,__cscrl_bits);
__IO_REG32_BIT(CSCR4A,                    0xB8002048,__READ_WRITE,__cscra_bits);
__IO_REG32_BIT(CSCR5U,                    0xB8002050,__READ_WRITE,__cscru_bits);
__IO_REG32_BIT(CSCR5L,                    0xB8002054,__READ_WRITE,__cscrl_bits);
__IO_REG32_BIT(CSCR5A,                    0xB8002058,__READ_WRITE,__cscra_bits);
__IO_REG32_BIT(WEIM_WCR,                  0xB8002060,__READ_WRITE,__weim_wcr_bits);

/***************************************************************************
 **
 **  EPIT1
 **
 ***************************************************************************/
__IO_REG32_BIT(EPIT1CR,                   0x53F94000,__READ_WRITE,__epitcr_bits);
__IO_REG32_BIT(EPIT1SR,                   0x53F94004,__READ_WRITE,__epitsr_bits);
__IO_REG32(    EPIT1LR,                   0x53F94008,__READ_WRITE);
__IO_REG32(    EPIT1CMPR,                 0x53F9400C,__READ_WRITE);
__IO_REG32(    EPIT1CNR,                  0x53F94010,__READ      );

/***************************************************************************
 **
 **  EPIT2
 **
 ***************************************************************************/
__IO_REG32_BIT(EPIT2CR,                   0x53F98000,__READ_WRITE,__epitcr_bits);
__IO_REG32_BIT(EPIT2SR,                   0x53F98004,__READ_WRITE,__epitsr_bits);
__IO_REG32(    EPIT2LR,                   0x53F98008,__READ_WRITE);
__IO_REG32(    EPIT2CMPR,                 0x53F9800C,__READ_WRITE);
__IO_REG32(    EPIT2CNR,                  0x53F98010,__READ      );

/***************************************************************************
 **
 **  ESAI
 **
 ***************************************************************************/
__IO_REG32(    ESAI_ETDR,                 0x50018000,__WRITE     );
__IO_REG32(    ESAI_ERDR,                 0x50018004,__READ      );
__IO_REG32_BIT(ESAI_ECR,                  0x50018008,__READ_WRITE,__esai_ecr_bits);
__IO_REG32_BIT(ESAI_ESR,                  0x5001800C,__READ      ,__esai_esr_bits);
__IO_REG32_BIT(ESAI_TFCR,                 0x50018010,__READ_WRITE,__esai_tfcr_bits);
__IO_REG32_BIT(ESAI_TFSR,                 0x50018014,__READ      ,__esai_tfsr_bits);
__IO_REG32_BIT(ESAI_RFCR,                 0x50018018,__READ_WRITE,__esai_rfcr_bits);
__IO_REG32_BIT(ESAI_RFSR,                 0x5001801C,__READ      ,__esai_rfsr_bits);
__IO_REG32_BIT(ESAI_TX0,                  0x50018080,__WRITE     ,__esai_tx_bits);
__IO_REG32_BIT(ESAI_TX1,                  0x50018084,__WRITE     ,__esai_tx_bits);
__IO_REG32_BIT(ESAI_TX2,                  0x50018088,__WRITE     ,__esai_tx_bits);
__IO_REG32_BIT(ESAI_TX3,                  0x5001808C,__WRITE     ,__esai_tx_bits);
__IO_REG32_BIT(ESAI_TX4,                  0x50018090,__WRITE     ,__esai_tx_bits);
__IO_REG32_BIT(ESAI_TX5,                  0x50018094,__WRITE     ,__esai_tx_bits);
__IO_REG32_BIT(ESAI_TSR,                  0x50018098,__WRITE     ,__esai_tsr_bits);
__IO_REG32_BIT(ESAI_RX0,                  0x500180A0,__READ      ,__esai_rx_bits);
__IO_REG32_BIT(ESAI_RX1,                  0x500180A4,__READ      ,__esai_rx_bits);
__IO_REG32_BIT(ESAI_RX2,                  0x500180A8,__READ      ,__esai_rx_bits);
__IO_REG32_BIT(ESAI_RX3,                  0x500180AC,__READ      ,__esai_rx_bits);
__IO_REG32_BIT(ESAI_SAISR,                0x500180CC,__READ      ,__esai_saisr_bits);
__IO_REG32_BIT(ESAI_SAICR,                0x500180D0,__READ_WRITE,__esai_saicr_bits);
__IO_REG32_BIT(ESAI_TCR,                  0x500180D4,__READ_WRITE,__esai_tcr_bits);
__IO_REG32_BIT(ESAI_TCCR,                 0x500180D8,__READ_WRITE,__esai_tccr_bits);
__IO_REG32_BIT(ESAI_RCR,                  0x500180DC,__READ_WRITE,__esai_rcr_bits);
__IO_REG32_BIT(ESAI_RCCR,                 0x500180E0,__READ_WRITE,__esai_rccr_bits);
__IO_REG32_BIT(ESAI_TSMA,                 0x500180E4,__READ_WRITE,__esai_tsm_bits);
__IO_REG32_BIT(ESAI_TSMB,                 0x500180E8,__READ_WRITE,__esai_tsm_bits);
__IO_REG32_BIT(ESAI_RSMA,                 0x500180EC,__READ_WRITE,__esai_rsm_bits);
__IO_REG32_BIT(ESAI_RSMB,                 0x500180F0,__READ_WRITE,__esai_rsm_bits);
__IO_REG32_BIT(ESAI_PRRC,                 0x500180F8,__READ_WRITE,__esai_prrc_bits);
__IO_REG32_BIT(ESAI_PCRC,                 0x500180FC,__READ_WRITE,__esai_pcrc_bits);

/***************************************************************************
 **
 **  eSDHC1
 **
 ***************************************************************************/
__IO_REG32(    ESDHC1_DSADDR,             0x53FB4000,__READ_WRITE);
__IO_REG32_BIT(ESDHC1_BLKATTR,            0x53FB4004,__READ_WRITE,__esdhc_blkattr_bits);
__IO_REG32(    ESDHC1_CMDARG,             0x53FB4008,__READ_WRITE);
__IO_REG32_BIT(ESDHC1_XFERTYP,            0x53FB400C,__READ_WRITE,__esdhc_xfertyp_bits);
__IO_REG32(    ESDHC1_CMDRSP0,            0x53FB4010,__READ      );
__IO_REG32(    ESDHC1_CMDRSP1,            0x53FB4014,__READ      );
__IO_REG32(    ESDHC1_CMDRSP2,            0x53FB4018,__READ      );
__IO_REG32(    ESDHC1_CMDRSP3,            0x53FB401C,__READ      );
__IO_REG32(    ESDHC1_DATPORT,            0x53FB4020,__READ_WRITE);
__IO_REG32_BIT(ESDHC1_PRSSTAT,            0x53FB4024,__READ      ,__esdhc_prsstat_bits);
__IO_REG32_BIT(ESDHC1_PROCTL,             0x53FB4028,__READ_WRITE,__esdhc_proctl_bits);
__IO_REG32_BIT(ESDHC1_SYSCTL,             0x53FB402C,__READ_WRITE,__esdhc_sysctl_bits);
__IO_REG32_BIT(ESDHC1_IRQSTAT,            0x53FB4030,__READ      ,__esdhc_irqstat_bits);
__IO_REG32_BIT(ESDHC1_IRQSTATEN,          0x53FB4034,__READ_WRITE,__esdhc_irqstaten_bits);
__IO_REG32_BIT(ESDHC1_IRQSIGEN,           0x53FB4038,__READ      ,__esdhc_irqsigen_bits);
__IO_REG32_BIT(ESDHC1_AUTOC12ERR,         0x53FB403C,__READ      ,__esdhc_autoc12err_bits);
__IO_REG32_BIT(ESDHC1_HOSTCAPBLT,         0x53FB4040,__READ      ,__esdhc_hostcapblt_bits);
__IO_REG32_BIT(ESDHC1_WML,                0x53FB4044,__READ_WRITE,__esdhc_wml_bits);
__IO_REG32_BIT(ESDHC1_FEVT,               0x53FB4050,__WRITE     ,__esdhc_fevt_bits);
__IO_REG32_BIT(ESDHC1_HOSTVER,            0x53FB40FC,__READ      ,__esdhc_hostver_bits);

/***************************************************************************
 **
 **  eSDHC2
 **
 ***************************************************************************/
__IO_REG32(    ESDHC2_DSADDR,             0x53FB8000,__READ_WRITE);
__IO_REG32_BIT(ESDHC2_BLKATTR,            0x53FB8004,__READ_WRITE,__esdhc_blkattr_bits);
__IO_REG32(    ESDHC2_CMDARG,             0x53FB8008,__READ_WRITE);
__IO_REG32_BIT(ESDHC2_XFERTYP,            0x53FB800C,__READ_WRITE,__esdhc_xfertyp_bits);
__IO_REG32(    ESDHC2_CMDRSP0,            0x53FB8010,__READ      );
__IO_REG32(    ESDHC2_CMDRSP1,            0x53FB8014,__READ      );
__IO_REG32(    ESDHC2_CMDRSP2,            0x53FB8018,__READ      );
__IO_REG32(    ESDHC2_CMDRSP3,            0x53FB801C,__READ      );
__IO_REG32(    ESDHC2_DATPORT,            0x53FB8020,__READ_WRITE);
__IO_REG32_BIT(ESDHC2_PRSSTAT,            0x53FB8024,__READ      ,__esdhc_prsstat_bits);
__IO_REG32_BIT(ESDHC2_PROCTL,             0x53FB8028,__READ_WRITE,__esdhc_proctl_bits);
__IO_REG32_BIT(ESDHC2_SYSCTL,             0x53FB802C,__READ_WRITE,__esdhc_sysctl_bits);
__IO_REG32_BIT(ESDHC2_IRQSTAT,            0x53FB8030,__READ      ,__esdhc_irqstat_bits);
__IO_REG32_BIT(ESDHC2_IRQSTATEN,          0x53FB8034,__READ_WRITE,__esdhc_irqstaten_bits);
__IO_REG32_BIT(ESDHC2_IRQSIGEN,           0x53FB8038,__READ      ,__esdhc_irqsigen_bits);
__IO_REG32_BIT(ESDHC2_AUTOC12ERR,         0x53FB803C,__READ      ,__esdhc_autoc12err_bits);
__IO_REG32_BIT(ESDHC2_HOSTCAPBLT,         0x53FB8040,__READ      ,__esdhc_hostcapblt_bits);
__IO_REG32_BIT(ESDHC2_WML,                0x53FB8044,__READ_WRITE,__esdhc_wml_bits);
__IO_REG32_BIT(ESDHC2_FEVT,               0x53FB8050,__WRITE     ,__esdhc_fevt_bits);
__IO_REG32_BIT(ESDHC2_HOSTVER,            0x53FB80FC,__READ      ,__esdhc_hostver_bits);

 /***************************************************************************
 **
 **  FEC
 **
 ***************************************************************************/
__IO_REG32_BIT(FEC_EIR,                   0x50038004,__READ_WRITE,__fec_eir_bits);
__IO_REG32_BIT(FEC_EIMR,                  0x50038008,__READ_WRITE,__fec_eir_bits);
__IO_REG32_BIT(FEC_RDAR,                  0x50038010,__READ_WRITE,__fec_rdar_bits);
__IO_REG32_BIT(FEC_TDAR,                  0x50038014,__READ_WRITE,__fec_tdar_bits);
__IO_REG32_BIT(FEC_ECR,                   0x50038024,__READ_WRITE,__fec_ecr_bits);
__IO_REG32_BIT(FEC_MMFR,                  0x50038040,__READ_WRITE,__fec_mmfr_bits);
__IO_REG32_BIT(FEC_MSCR,                  0x50038044,__READ_WRITE,__fec_mscr_bits);
__IO_REG32_BIT(FEC_MIBC,                  0x50038064,__READ_WRITE,__fec_mibc_bits);
__IO_REG32_BIT(FEC_RCR,                   0x50038084,__READ_WRITE,__fec_rcr_bits);
__IO_REG32_BIT(FEC_TCR,                   0x500380C4,__READ_WRITE,__fec_tcr_bits);
__IO_REG32(    FEC_PALR,                  0x500380E4,__READ_WRITE);
__IO_REG32_BIT(FEC_PAUR,                  0x500380E8,__READ_WRITE,__fec_paur_bits);
__IO_REG32_BIT(FEC_OPD,                   0x500380EC,__READ_WRITE,__fec_opd_bits);
__IO_REG32(    FEC_IAUR,                  0x50038118,__READ_WRITE);
__IO_REG32(    FEC_IALR,                  0x5003811C,__READ_WRITE);
__IO_REG32(    FEC_GAUR,                  0x50038120,__READ_WRITE);
__IO_REG32(    FEC_GALR,                  0x50038124,__READ_WRITE);
__IO_REG32_BIT(FEC_TFWR,                  0x50038144,__READ_WRITE,__fec_tfwr_bits);
__IO_REG32_BIT(FEC_FRBR,                  0x5003814C,__READ      ,__fec_frbr_bits);
__IO_REG32_BIT(FEC_FRSR,                  0x50038150,__READ_WRITE,__fec_frsr_bits);
__IO_REG32(    FEC_ERDSR,                 0x50038180,__READ_WRITE);
__IO_REG32(    FEC_ETDSR,                 0x50038184,__READ_WRITE);
__IO_REG32_BIT(FEC_EMRBR,                 0x50038188,__READ_WRITE,__fec_emrbr_bits);
__IO_REG32(    FEC_RMON_T_DROP,           0x50038200,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_PACKETS,        0x50038204,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_BC_PKT,         0x50038208,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_MC_PKT,         0x5003820C,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_CRC_ALIGN,      0x50038210,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_UNDERSIZE,      0x50038214,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_OVERSIZE,       0x50038218,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_FRAG,           0x5003821C,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_JAB,            0x50038220,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_COL,            0x50038224,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_P64,            0x50038228,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_P65TO127,       0x5003822C,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_P128TO255,      0x50038230,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_P256TO511,      0x50038234,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_P512TO1023,     0x50038238,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_P1024TO2047,    0x5003823C,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_P_GTE2048,      0x50038240,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_OCTETS,         0x50038244,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_DROP,           0x50038248,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_FRAME_OK,       0x5003824C,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_1COL,           0x50038250,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_MCOL,           0x50038254,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_DEF,            0x50038258,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_LCOL,           0x5003825C,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_EXCOL,          0x50038260,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_MACERR,         0x50038264,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_CSERR,          0x50038268,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_SQE,            0x5003826C,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_FDXFC,          0x50038270,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_OCTETS_OK,      0x50038274,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_PACKETS,        0x50038284,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_BC_PKT,         0x50038288,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_MC_PKT,         0x5003828C,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_CRC_ALIGN,      0x50038290,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_UNDERSIZE,      0x50038294,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_OVERSIZE,       0x50038298,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_FRAG,           0x5003829C,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_JAB,            0x500382A0,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_RESVD_0,        0x500382A4,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_P64,            0x500382A8,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_P65TO127,       0x500382AC,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_P128TO255,      0x500382B0,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_P256TO511,      0x500382B4,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_P512TO1023,     0x500382B8,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_P1024TO2047,    0x500382BC,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_P_GTE2048,      0x500382C0,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_OCTETS,         0x500382C4,__READ_WRITE);
__IO_REG32(    FEC_IEEE_R_DROP,           0x500382C8,__READ_WRITE);
__IO_REG32(    FEC_IEEE_R_FRAME_OK,       0x500382CC,__READ_WRITE);
__IO_REG32(    FEC_IEEE_R_CRC,            0x500382D0,__READ_WRITE);
__IO_REG32(    FEC_IEEE_R_ALIGN,          0x500382D4,__READ_WRITE);
__IO_REG32(    FEC_IEEE_R_MACERR,         0x500382D8,__READ_WRITE);
__IO_REG32(    FEC_IEEE_R_FDXFC,          0x500382DC,__READ_WRITE);
__IO_REG32(    FEC_IEEE_R_OCTETS_OK,      0x500382E0,__READ_WRITE);
__IO_REG32_BIT(FEC_MIIGSK_CFGR,           0x50038300,__READ_WRITE,__fec_miigsk_cfgr_bits);
__IO_REG32_BIT(FEC_MIIGSK_ENR,            0x50038308,__READ_WRITE,__fec_miigsk_enr_bits);

/***************************************************************************
 **
 **  FlexCAN1
 **
 ***************************************************************************/
__IO_REG32_BIT(CAN1_MCR,                  0x43F88000,__READ_WRITE,__can_mcr_bits);
__IO_REG32_BIT(CAN1_CTRL,                 0x43F88004,__READ_WRITE,__can_ctrl_bits);
__IO_REG32_BIT(CAN1_TIMER,                0x43F88008,__READ_WRITE,__can_timer_bits);
__IO_REG32_BIT(CAN1_RXGMASK,              0x43F88010,__READ_WRITE,__can_rxgmask_bits);
__IO_REG32(    CAN1_RX14MASK,             0x43F88014,__READ_WRITE);
__IO_REG32(    CAN1_RX15MASK,             0x43F88018,__READ_WRITE);
__IO_REG32_BIT(CAN1_ECR,                  0x43F8801C,__READ_WRITE,__can_ecr_bits);
__IO_REG32_BIT(CAN1_ESR,                  0x43F88020,__READ_WRITE,__can_esr_bits);
__IO_REG32_BIT(CAN1_IMASK2,               0x43F88024,__READ_WRITE,__can_imask2_bits);
__IO_REG32_BIT(CAN1_IMASK1,               0x43F88028,__READ_WRITE,__can_imask1_bits);
__IO_REG32_BIT(CAN1_IFLAG2,               0x43F8802C,__READ_WRITE,__can_iflag2_bits);
__IO_REG32_BIT(CAN1_IFLAG1,               0x43F88030,__READ_WRITE,__can_iflag1_bits);
__IO_REG32(    CAN1_MB0_15_BASE_ADDR,     0x43F88080,__READ_WRITE);
__IO_REG32(    CAN1_MB16_31_BASE_ADDR,    0x43F88180,__READ_WRITE);
__IO_REG32(    CAN1_MB32_63_BASE_ADDR,    0x43F88280,__READ_WRITE);
__IO_REG32(    CAN1_RXIMR0_15_BASE_ADDR,  0x43F88880,__READ_WRITE);
__IO_REG32(    CAN1_RXIMR16_31_BASE_ADDR, 0x43F888C0,__READ_WRITE);
__IO_REG32(    CAN1_RXIMR32_63_BASE_ADDR, 0x43F88900,__READ_WRITE);

/***************************************************************************
 **
 **  FlexCAN2
 **
 ***************************************************************************/
__IO_REG32_BIT(CAN2_MCR,                  0x43F8C000,__READ_WRITE,__can_mcr_bits);
__IO_REG32_BIT(CAN2_CTRL,                 0x43F8C004,__READ_WRITE,__can_ctrl_bits);
__IO_REG32_BIT(CAN2_TIMER,                0x43F8C008,__READ_WRITE,__can_timer_bits);
__IO_REG32_BIT(CAN2_RXGMASK,              0x43F8C010,__READ_WRITE,__can_rxgmask_bits);
__IO_REG32(    CAN2_RX14MASK,             0x43F8C014,__READ_WRITE);
__IO_REG32(    CAN2_RX15MASK,             0x43F8C018,__READ_WRITE);
__IO_REG32_BIT(CAN2_ECR,                  0x43F8C01C,__READ_WRITE,__can_ecr_bits);
__IO_REG32_BIT(CAN2_ESR,                  0x43F8C020,__READ_WRITE,__can_esr_bits);
__IO_REG32_BIT(CAN2_IMASK2,               0x43F8C024,__READ_WRITE,__can_imask2_bits);
__IO_REG32_BIT(CAN2_IMASK1,               0x43F8C028,__READ_WRITE,__can_imask1_bits);
__IO_REG32_BIT(CAN2_IFLAG2,               0x43F8C02C,__READ_WRITE,__can_iflag2_bits);
__IO_REG32_BIT(CAN2_IFLAG1,               0x43F8C030,__READ_WRITE,__can_iflag1_bits);
__IO_REG32(    CAN2_MB0_15_BASE_ADDR,     0x43F8C080,__READ_WRITE);
__IO_REG32(    CAN2_MB16_31_BASE_ADDR,    0x43F8C180,__READ_WRITE);
__IO_REG32(    CAN2_MB32_63_BASE_ADDR,    0x43F8C280,__READ_WRITE);
__IO_REG32(    CAN2_RXIMR0_15_BASE_ADDR,  0x43F8C880,__READ_WRITE);
__IO_REG32(    CAN2_RXIMR16_31_BASE_ADDR, 0x43F8C8C0,__READ_WRITE);
__IO_REG32(    CAN2_RXIMR32_63_BASE_ADDR, 0x43F8C900,__READ_WRITE);

/***************************************************************************
 **
 **  GPIO1
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO1_DR,                  0x53FCC000,__READ_WRITE,__gpio_dr_bits);
__IO_REG32_BIT(GPIO1_GDIR,                0x53FCC004,__READ_WRITE,__gpio_gdir_bits);
__IO_REG32_BIT(GPIO1_PSR,                 0x53FCC008,__READ      ,__gpio_psr_bits);
__IO_REG32_BIT(GPIO1_ICR1,                0x53FCC00C,__READ_WRITE,__gpio_icr1_bits);
__IO_REG32_BIT(GPIO1_ICR2,                0x53FCC010,__READ_WRITE,__gpio_icr2_bits);
__IO_REG32_BIT(GPIO1_IMR,                 0x53FCC014,__READ_WRITE,__gpio_imr_bits);
__IO_REG32_BIT(GPIO1_ISR,                 0x53FCC018,__READ_WRITE,__gpio_isr_bits);
__IO_REG32_BIT(GPIO1_EDGE_SEL,            0x53FCC01C,__READ_WRITE,__gpio_edge_sel_bits);

/***************************************************************************
 **
 **  GPIO2
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO2_DR,                  0x53FD0000,__READ_WRITE,__gpio_dr_bits);
__IO_REG32_BIT(GPIO2_GDIR,                0x53FD0004,__READ_WRITE,__gpio_gdir_bits);
__IO_REG32_BIT(GPIO2_PSR,                 0x53FD0008,__READ      ,__gpio_psr_bits);
__IO_REG32_BIT(GPIO2_ICR1,                0x53FD000C,__READ_WRITE,__gpio_icr1_bits);
__IO_REG32_BIT(GPIO2_ICR2,                0x53FD0010,__READ_WRITE,__gpio_icr2_bits);
__IO_REG32_BIT(GPIO2_IMR,                 0x53FD0014,__READ_WRITE,__gpio_imr_bits);
__IO_REG32_BIT(GPIO2_ISR,                 0x53FD0018,__READ_WRITE,__gpio_isr_bits);
__IO_REG32_BIT(GPIO2_EDGE_SEL,            0x53FD001C,__READ_WRITE,__gpio_edge_sel_bits);

/***************************************************************************
 **
 **  GPIO3
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO3_DR,                  0x53FA4000,__READ_WRITE,__gpio_dr_bits);
__IO_REG32_BIT(GPIO3_GDIR,                0x53FA4004,__READ_WRITE,__gpio_gdir_bits);
__IO_REG32_BIT(GPIO3_PSR,                 0x53FA4008,__READ      ,__gpio_psr_bits);
__IO_REG32_BIT(GPIO3_ICR1,                0x53FA400C,__READ_WRITE,__gpio_icr1_bits);
__IO_REG32_BIT(GPIO3_ICR2,                0x53FA4010,__READ_WRITE,__gpio_icr2_bits);
__IO_REG32_BIT(GPIO3_IMR,                 0x53FA4014,__READ_WRITE,__gpio_imr_bits);
__IO_REG32_BIT(GPIO3_ISR,                 0x53FA4018,__READ_WRITE,__gpio_isr_bits);
__IO_REG32_BIT(GPIO3_EDGE_SEL,            0x53FA401C,__READ_WRITE,__gpio_edge_sel_bits);

/***************************************************************************
 **
 **  GPIO4
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO4_DR,                  0x53F9C000,__READ_WRITE,__gpio_dr_bits);
__IO_REG32_BIT(GPIO4_GDIR,                0x53F9C004,__READ_WRITE,__gpio_gdir_bits);
__IO_REG32_BIT(GPIO4_PSR,                 0x53F9C008,__READ      ,__gpio_psr_bits);
__IO_REG32_BIT(GPIO4_ICR1,                0x53F9C00C,__READ_WRITE,__gpio_icr1_bits);
__IO_REG32_BIT(GPIO4_ICR2,                0x53F9C010,__READ_WRITE,__gpio_icr2_bits);
__IO_REG32_BIT(GPIO4_IMR,                 0x53F9C014,__READ_WRITE,__gpio_imr_bits);
__IO_REG32_BIT(GPIO4_ISR,                 0x53F9C018,__READ_WRITE,__gpio_isr_bits);
__IO_REG32_BIT(GPIO4_EDGE_SEL,            0x53F9C01C,__READ_WRITE,__gpio_edge_sel_bits);

/***************************************************************************
 **
 **  GPT1
 **
 ***************************************************************************/
__IO_REG32_BIT(GPT1CR,                    0x53F90000,__READ_WRITE,__gptcr_bits);
__IO_REG32_BIT(GPT1PR,                    0x53F90004,__READ_WRITE,__gptpr_bits);
__IO_REG32_BIT(GPT1SR,                    0x53F90008,__READ_WRITE,__gptsr_bits);
__IO_REG32_BIT(GPT1IR,                    0x53F9000C,__READ_WRITE,__gptir_bits);
__IO_REG32(    GPT1OCR1,                  0x53F90010,__READ_WRITE);
__IO_REG32(    GPT1OCR2,                  0x53F90014,__READ_WRITE);
__IO_REG32(    GPT1OCR3,                  0x53F90018,__READ_WRITE);
__IO_REG32(    GPT1ICR1,                  0x53F9001C,__READ      );
__IO_REG32(    GPT1ICR2,                  0x53F90020,__READ      );
__IO_REG32(    GPT1CNT,                   0x53F90024,__READ      );

/***************************************************************************
 **
 **  GPT2
 **
 ***************************************************************************/
__IO_REG32_BIT(GPT2CR,                    0x53F8C000,__READ_WRITE,__gptcr_bits);
__IO_REG32_BIT(GPT2PR,                    0x53F8C004,__READ_WRITE,__gptpr_bits);
__IO_REG32_BIT(GPT2SR,                    0x53F8C008,__READ_WRITE,__gptsr_bits);
__IO_REG32_BIT(GPT2IR,                    0x53F8C00C,__READ_WRITE,__gptir_bits);
__IO_REG32(    GPT2OCR1,                  0x53F8C010,__READ_WRITE);
__IO_REG32(    GPT2OCR2,                  0x53F8C014,__READ_WRITE);
__IO_REG32(    GPT2OCR3,                  0x53F8C018,__READ_WRITE);
__IO_REG32(    GPT2ICR1,                  0x53F8C01C,__READ      );
__IO_REG32(    GPT2ICR2,                  0x53F8C020,__READ      );
__IO_REG32(    GPT2CNT,                   0x53F8C024,__READ      );

/***************************************************************************
 **
 **  GPT3
 **
 ***************************************************************************/
__IO_REG32_BIT(GPT3CR,                    0x53F88000,__READ_WRITE,__gptcr_bits);
__IO_REG32_BIT(GPT3PR,                    0x53F88004,__READ_WRITE,__gptpr_bits);
__IO_REG32_BIT(GPT3SR,                    0x53F88008,__READ_WRITE,__gptsr_bits);
__IO_REG32_BIT(GPT3IR,                    0x53F8800C,__READ_WRITE,__gptir_bits);
__IO_REG32(    GPT3OCR1,                  0x53F88010,__READ_WRITE);
__IO_REG32(    GPT3OCR2,                  0x53F88014,__READ_WRITE);
__IO_REG32(    GPT3OCR3,                  0x53F88018,__READ_WRITE);
__IO_REG32(    GPT3ICR1,                  0x53F8801C,__READ      );
__IO_REG32(    GPT3ICR2,                  0x53F88020,__READ      );
__IO_REG32(    GPT3CNT,                   0x53F88024,__READ      );

/***************************************************************************
 **
 **  GPT4
 **
 ***************************************************************************/
__IO_REG32_BIT(GPT4CR,                    0x53F84000,__READ_WRITE,__gptcr_bits);
__IO_REG32_BIT(GPT4PR,                    0x53F84004,__READ_WRITE,__gptpr_bits);
__IO_REG32_BIT(GPT4SR,                    0x53F84008,__READ_WRITE,__gptsr_bits);
__IO_REG32_BIT(GPT4IR,                    0x53F8400C,__READ_WRITE,__gptir_bits);
__IO_REG32(    GPT4OCR1,                  0x53F84010,__READ_WRITE);
__IO_REG32(    GPT4OCR2,                  0x53F84014,__READ_WRITE);
__IO_REG32(    GPT4OCR3,                  0x53F84018,__READ_WRITE);
__IO_REG32(    GPT4ICR1,                  0x53F8401C,__READ      );
__IO_REG32(    GPT4ICR2,                  0x53F84020,__READ      );
__IO_REG32(    GPT4CNT,                   0x53F84024,__READ      );

/***************************************************************************
 **
 **  I2C1
 **
 ***************************************************************************/
__IO_REG16_BIT(IADR1,                     0x43F80000,__READ_WRITE,__iadr_bits);
__IO_REG16_BIT(IFDR1,                     0x43F80004,__READ_WRITE,__ifdr_bits);
__IO_REG16_BIT(I2CR1,                     0x43F80008,__READ_WRITE,__i2cr_bits);
__IO_REG16_BIT(I2SR1,                     0x43F8000C,__READ_WRITE,__i2sr_bits);
__IO_REG16_BIT(I2DR1,                     0x43F80010,__READ_WRITE,__i2dr_bits);

/***************************************************************************
 **
 **  I2C2
 **
 ***************************************************************************/
__IO_REG16_BIT(IADR2,                     0x43F98000,__READ_WRITE,__iadr_bits);
__IO_REG16_BIT(IFDR2,                     0x43F98004,__READ_WRITE,__ifdr_bits);
__IO_REG16_BIT(I2CR2,                     0x43F98008,__READ_WRITE,__i2cr_bits);
__IO_REG16_BIT(I2SR2,                     0x43F9800C,__READ_WRITE,__i2sr_bits);
__IO_REG16_BIT(I2DR2,                     0x43F98010,__READ_WRITE,__i2dr_bits);

/***************************************************************************
 **
 **  I2C3
 **
 ***************************************************************************/
__IO_REG16_BIT(IADR3,                     0x43F84000,__READ_WRITE,__iadr_bits);
__IO_REG16_BIT(IFDR3,                     0x43F84004,__READ_WRITE,__ifdr_bits);
__IO_REG16_BIT(I2CR3,                     0x43F84008,__READ_WRITE,__i2cr_bits);
__IO_REG16_BIT(I2SR3,                     0x43F8400C,__READ_WRITE,__i2sr_bits);
__IO_REG16_BIT(I2DR3,                     0x43F84010,__READ_WRITE,__i2dr_bits);

/***************************************************************************
 **
 **  IIM
 **
 ***************************************************************************/
__IO_REG8_BIT( IIM_STAT,                  0x53FF0000,__READ_WRITE,__iim_stat_bits);
__IO_REG8_BIT( IIM_STATM,                 0x53FF0004,__READ_WRITE,__iim_statm_bits);
__IO_REG8_BIT( IIM_ERR,                   0x53FF0008,__READ_WRITE,__iim_err_bits);
__IO_REG8_BIT( IIM_EMASK,                 0x53FF000C,__READ_WRITE,__iim_emask_bits);
__IO_REG8_BIT( IIM_FCTL,                  0x53FF0010,__READ_WRITE,__iim_fctl_bits);
__IO_REG8_BIT( IIM_UA,                    0x53FF0014,__READ_WRITE,__iim_ua_bits);
__IO_REG8(     IIM_LA,                    0x53FF0018,__READ_WRITE);
__IO_REG8(     IIM_SDAT,                  0x53FF001C,__READ      );
__IO_REG8_BIT( IIM_PREV,                  0x53FF0020,__READ      ,__iim_prev_bits);
__IO_REG8(     IIM_SREV,                  0x53FF0024,__READ      );
__IO_REG8(     IIM_PREG_P,                0x53FF0028,__READ_WRITE);
__IO_REG8_BIT( IIM_SCS0,                  0x53FF002C,__READ_WRITE,__iim_scs0_bits);
__IO_REG8_BIT( IIM_SCS1,                  0x53FF0030,__READ_WRITE,__iim_scs1_bits);
__IO_REG8_BIT( IIM_SCS2,                  0x53FF0034,__READ_WRITE,__iim_scs2_bits);
__IO_REG8_BIT( IIM_SCS3,                  0x53FF0038,__READ_WRITE,__iim_scs2_bits);
__IO_REG8_BIT( IIM_FBAC0,                 0x53FF0800,__READ_WRITE,__iim_fbac0_bits);
__IO_REG8_BIT( IIM_FB0_WORD1,             0x53FF0804,__READ_WRITE,__iim_fb0_word1_bits);
__IO_REG8_BIT( IIM_FB0_WORD2,             0x53FF0808,__READ_WRITE,__iim_fb0_word2_bits);
__IO_REG8_BIT( IIM_FB0_WORD3,             0x53FF080C,__READ_WRITE,__iim_fb0_word3_bits);
__IO_REG8_BIT( IIM_FB0_WORD4,             0x53FF0810,__READ_WRITE,__iim_fb0_word4_bits);
__IO_REG8(     IIM_SI_ID0,                0x53FF0814,__READ_WRITE);
__IO_REG8(     IIM_SI_ID1,                0x53FF0818,__READ_WRITE);
__IO_REG8(     IIM_SI_ID2,                0x53FF081C,__READ_WRITE);
__IO_REG8(     IIM_SI_ID3,                0x53FF0820,__READ_WRITE);
__IO_REG8(     IIM_SI_ID4,                0x53FF0824,__READ_WRITE);
__IO_REG8(     IIM_SI_ID5,                0x53FF0828,__READ_WRITE);
__IO_REG8(     IIM_SCC_KEY0,              0x53FF082C,__WRITE     );
__IO_REG8(     IIM_SCC_KEY1,              0x53FF0830,__WRITE     );
__IO_REG8(     IIM_SCC_KEY2,              0x53FF0834,__WRITE     );
__IO_REG8(     IIM_SCC_KEY3,              0x53FF0838,__WRITE     );
__IO_REG8(     IIM_SCC_KEY4,              0x53FF083C,__WRITE     );
__IO_REG8(     IIM_SCC_KEY5,              0x53FF0840,__WRITE     );
__IO_REG8(     IIM_SCC_KEY6,              0x53FF0844,__WRITE     );
__IO_REG8(     IIM_SCC_KEY7,              0x53FF0848,__WRITE     );
__IO_REG8(     IIM_SCC_KEY8,              0x53FF084C,__WRITE     );
__IO_REG8(     IIM_SCC_KEY9,              0x53FF0850,__WRITE     );
__IO_REG8(     IIM_SCC_KEY10,             0x53FF0854,__WRITE     );
__IO_REG8(     IIM_SCC_KEY11,             0x53FF0858,__WRITE     );
__IO_REG8(     IIM_SCC_KEY12,             0x53FF085C,__WRITE     );
__IO_REG8(     IIM_SCC_KEY13,             0x53FF0860,__WRITE     );
__IO_REG8(     IIM_SCC_KEY14,             0x53FF0864,__WRITE     );
__IO_REG8(     IIM_SCC_KEY15,             0x53FF0868,__WRITE     );
__IO_REG8(     IIM_SCC_KEY16,             0x53FF086C,__WRITE     );
__IO_REG8(     IIM_SCC_KEY17,             0x53FF0870,__WRITE     );
__IO_REG8(     IIM_SCC_KEY18,             0x53FF0874,__WRITE     );
__IO_REG8(     IIM_SCC_KEY19,             0x53FF0878,__WRITE     );
__IO_REG8(     IIM_SCC_KEY20,             0x53FF087C,__WRITE     );
__IO_REG8_BIT( IIM_FBAC1,                 0x53FF0C00,__READ_WRITE,__iim_fbac1_bits);
__IO_REG8(     IIM_MAC_ADDR0,             0x53FF0C04,__READ_WRITE);
__IO_REG8(     IIM_MAC_ADDR1,             0x53FF0C08,__READ_WRITE);
__IO_REG8(     IIM_MAC_ADDR2,             0x53FF0C0C,__READ_WRITE);
__IO_REG8(     IIM_MAC_ADDR3,             0x53FF0C10,__READ_WRITE);
__IO_REG8(     IIM_MAC_ADDR4,             0x53FF0C14,__READ_WRITE);
__IO_REG8(     IIM_MAC_ADDR5,             0x53FF0C18,__READ_WRITE);

/***************************************************************************
 **
 **  KPP
 **
 ***************************************************************************/
__IO_REG16_BIT(KPCR,                      0x43FA8000,__READ_WRITE,__kpcr_bits);
__IO_REG16_BIT(KPSR,                      0x43FA8002,__READ_WRITE,__kpsr_bits);
__IO_REG16_BIT(KDDR,                      0x43FA8004,__READ_WRITE,__kddr_bits);
__IO_REG16_BIT(KPDR,                      0x43FA8006,__READ_WRITE,__kpdr_bits);

/***************************************************************************
 **
 **  LCDC
 **
 ***************************************************************************/
__IO_REG32(    LSSAR,                     0x53FBC000,__READ_WRITE);
__IO_REG32_BIT(_LSR,                      0x53FBC004,__READ_WRITE,__lsr_bits);
__IO_REG32_BIT(LVPWR,                     0x53FBC008,__READ_WRITE,__lvpwr_bits);
__IO_REG32_BIT(LCPR,                      0x53FBC00C,__READ_WRITE,__lcpr_bits);
__IO_REG32_BIT(LCWHBR,                    0x53FBC010,__READ_WRITE,__lcwhb_bits);
__IO_REG32_BIT(LCCMR,                     0x53FBC014,__READ_WRITE,__lccmr_bits);
__IO_REG32_BIT(LPCR,                      0x53FBC018,__READ_WRITE,__lpcr_bits);
__IO_REG32_BIT(LHCR,                      0x53FBC01C,__READ_WRITE,__lhcr_bits);
__IO_REG32_BIT(LVCR,                      0x53FBC020,__READ_WRITE,__lvcr_bits);
__IO_REG32_BIT(LPOR,                      0x53FBC024,__READ_WRITE,__lpor_bits);
__IO_REG32_BIT(LSCR,                      0x53FBC028,__READ_WRITE,__lscr_bits);
__IO_REG32_BIT(LPCCR,                     0x53FBC02C,__READ_WRITE,__lpccr_bits);
__IO_REG32_BIT(LDCR,                      0x53FBC030,__READ_WRITE,__ldcr_bits);
__IO_REG32_BIT(LRMCR,                     0x53FBC034,__READ_WRITE,__lrmcr_bits);
__IO_REG32_BIT(LICR,                      0x53FBC038,__READ_WRITE,__licr_bits);
__IO_REG32_BIT(LIER,                      0x53FBC03C,__READ_WRITE,__lier_bits);
__IO_REG32_BIT(LISR,                      0x53FBC040,__READ      ,__lisr_bits);
__IO_REG32(    LGWSAR,                    0x53FBC050,__READ_WRITE);
__IO_REG32_BIT(LGWSR,                     0x53FBC054,__READ_WRITE,__lgwsr_bits);
__IO_REG32_BIT(LGWVPWR,                   0x53FBC058,__READ_WRITE,__lgwvpwr_bits);
__IO_REG32_BIT(LGWPOR,                    0x53FBC05C,__READ_WRITE,__lgwpor_bits);
__IO_REG32_BIT(LGWPR,                     0x53FBC060,__READ_WRITE,__lgwpr_bits);
__IO_REG32_BIT(LGWCR,                     0x53FBC064,__READ_WRITE,__lgwcr_bits);
__IO_REG32_BIT(LGWDCR,                    0x53FBC068,__READ_WRITE,__lgwdcr_bits);
__IO_REG32_BIT(LAUSCR,                    0x53FBC080,__READ_WRITE,__lauscr_bits);
__IO_REG32_BIT(LAUSCCR,                   0x53FBC084,__READ_WRITE,__lausccr_bits);
__IO_REG32(    LBGLUT_BASE,               0x53FBC800,__READ_WRITE);
__IO_REG32(    LGWLUT_BASE,               0x53FBCC00,__READ_WRITE);

/***************************************************************************
 **
 **  MAX
 **
 ***************************************************************************/
__IO_REG32_BIT(MPR0,                      0x43F04000,__READ_WRITE,__mpr_bits);
__IO_REG32_BIT(SGPCR0,                    0x43F04010,__READ_WRITE,__sgpcr_bits);
__IO_REG32_BIT(MPR1,                      0x43F04100,__READ_WRITE,__mpr_bits);
__IO_REG32_BIT(SGPCR1,                    0x43F04110,__READ_WRITE,__sgpcr_bits);
__IO_REG32_BIT(MPR2,                      0x43F04200,__READ_WRITE,__mpr_bits);
__IO_REG32_BIT(SGPCR2,                    0x43F04210,__READ_WRITE,__sgpcr_bits);
__IO_REG32_BIT(MPR3,                      0x43F04300,__READ_WRITE,__mpr_bits);
__IO_REG32_BIT(SGPCR3,                    0x43F04310,__READ_WRITE,__sgpcr_bits);
__IO_REG32_BIT(MPR4,                      0x43F04400,__READ_WRITE,__mpr_bits);
__IO_REG32_BIT(SGPCR4,                    0x43F04410,__READ_WRITE,__sgpcr_bits);
__IO_REG32_BIT(MGPCR0,                    0x43F04800,__READ_WRITE,__mgpcr_bits);
__IO_REG32_BIT(MGPCR1,                    0x43F04900,__READ_WRITE,__mgpcr_bits);
__IO_REG32_BIT(MGPCR2,                    0x43F04A00,__READ_WRITE,__mgpcr_bits);
__IO_REG32_BIT(MGPCR3,                    0x43F04B00,__READ_WRITE,__mgpcr_bits);
__IO_REG32_BIT(MGPCR4,                    0x43F04C00,__READ_WRITE,__mgpcr_bits);

/***************************************************************************
 **
 **  PWM1
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM1CR,                    0x53FE0000,__READ_WRITE,__pwmcr_bits);
__IO_REG32_BIT(PWM1SR,                    0x53FE0004,__READ_WRITE,__pwmsr_bits);
__IO_REG32_BIT(PWM1IR,                    0x53FE0008,__READ_WRITE,__pwmir_bits);
__IO_REG32_BIT(PWM1SAR,                   0x53FE000C,__READ_WRITE,__pwmsar_bits);
__IO_REG32_BIT(PWM1PR,                    0x53FE0010,__READ_WRITE,__pwmpr_bits);
__IO_REG32_BIT(PWM1CNR,                   0x53FE0014,__READ      ,__pwmcnr_bits);

/***************************************************************************
 **
 **  PWM2
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM2CR,                    0x53FA0000,__READ_WRITE,__pwmcr_bits);
__IO_REG32_BIT(PWM2SR,                    0x53FA0004,__READ_WRITE,__pwmsr_bits);
__IO_REG32_BIT(PWM2IR,                    0x53FA0008,__READ_WRITE,__pwmir_bits);
__IO_REG32_BIT(PWM2SAR,                   0x53FA000C,__READ_WRITE,__pwmsar_bits);
__IO_REG32_BIT(PWM2PR,                    0x53FA0010,__READ_WRITE,__pwmpr_bits);
__IO_REG32_BIT(PWM2CNR,                   0x53FA0014,__READ      ,__pwmcnr_bits);

/***************************************************************************
 **
 **  PWM3
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM3CR,                    0x53FA8000,__READ_WRITE,__pwmcr_bits);
__IO_REG32_BIT(PWM3SR,                    0x53FA8004,__READ_WRITE,__pwmsr_bits);
__IO_REG32_BIT(PWM3IR,                    0x53FA8008,__READ_WRITE,__pwmir_bits);
__IO_REG32_BIT(PWM3SAR,                   0x53FA800C,__READ_WRITE,__pwmsar_bits);
__IO_REG32_BIT(PWM3PR,                    0x53FA8010,__READ_WRITE,__pwmpr_bits);
__IO_REG32_BIT(PWM3CNR,                   0x53FA8014,__READ      ,__pwmcnr_bits);

/***************************************************************************
 **
 **  PWM4
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM4CR,                    0x53FC8000,__READ_WRITE,__pwmcr_bits);
__IO_REG32_BIT(PWM4SR,                    0x53FC8004,__READ_WRITE,__pwmsr_bits);
__IO_REG32_BIT(PWM4IR,                    0x53FC8008,__READ_WRITE,__pwmir_bits);
__IO_REG32_BIT(PWM4SAR,                   0x53FC800C,__READ_WRITE,__pwmsar_bits);
__IO_REG32_BIT(PWM4PR,                    0x53FC8010,__READ_WRITE,__pwmpr_bits);
__IO_REG32_BIT(PWM4CNR,                   0x53FC8014,__READ      ,__pwmcnr_bits);

/***************************************************************************
 **
 **  SDMA
 **
 ***************************************************************************/
__IO_REG32(    SDMA_MC0PTR,               0x53FD4000,__READ_WRITE);
__IO_REG32_BIT(SDMA_INTR,                 0x53FD4004,__READ_WRITE,__sdma_intr_bits);
__IO_REG32_BIT(SDMA_STOP_STAT,            0x53FD4008,__READ      ,__sdma_stop_stat_bits);
__IO_REG32_BIT(SDMA_HSTART,               0x53FD400C,__READ_WRITE,__sdma_hstart_bits);
__IO_REG32_BIT(SDMA_EVTOVR,               0x53FD4010,__READ_WRITE,__sdma_evtovr_bits);
__IO_REG32_BIT(SDMA_DSPOVR,               0x53FD4014,__READ_WRITE,__sdma_dspovr_bits);
__IO_REG32_BIT(SDMA_HOSTOVR,              0x53FD4018,__READ_WRITE,__sdma_hostovr_bits);
__IO_REG32_BIT(SDMA_EVTPEND,              0x53FD401C,__READ      ,__sdma_evtpend_bits);
__IO_REG32_BIT(SDMA_RESET,                0x53FD4024,__READ      ,__sdma_reset_bits);
__IO_REG32_BIT(SDMA_EVTERR,               0x53FD4028,__READ      ,__sdma_evterr_bits);
__IO_REG32_BIT(SDMA_INTRMASK,             0x53FD402C,__READ_WRITE,__sdma_intrmask_bits);
__IO_REG32_BIT(SDMA_PSW,                  0x53FD4030,__READ      ,__sdma_psw_bits);
__IO_REG32_BIT(SDMA_EVTERRDBG,            0x53FD4034,__READ      ,__sdma_evterrdbg_bits);
__IO_REG32_BIT(SDMA_CONFIG,               0x53FD4038,__READ_WRITE,__sdma_config_bits);
__IO_REG32_BIT(SDMA_LOCK,                 0x53FD403C,__READ_WRITE,__sdma_lock_bits);
__IO_REG32_BIT(SDMA_ONCE_ENB,             0x53FD4040,__READ_WRITE,__sdma_once_enb_bits);
__IO_REG32(    SDMA_ONCE_DATA,            0x53FD4044,__READ_WRITE);
__IO_REG32_BIT(SDMA_ONCE_INSTR,           0x53FD4048,__READ_WRITE,__sdma_once_instr_bits);
__IO_REG32_BIT(SDMA_ONCE_STAT,            0x53FD404C,__READ      ,__sdma_once_stat_bits);
__IO_REG32_BIT(SDMA_ONCE_CMD,             0x53FD4050,__READ_WRITE,__sdma_once_cmd_bits);
__IO_REG32_BIT(SDMA_ILLINSTADDR,          0x53FD4058,__READ_WRITE,__sdma_illinstaddr_bits);
__IO_REG32_BIT(SDMA_CHN0ADDR,             0x53FD405C,__READ_WRITE,__sdma_chn0addr_bits);
__IO_REG32_BIT(SDMA_EVT_MIRROR,           0x53FD4060,__READ      ,__sdma_evt_mirror_bits);
__IO_REG32_BIT(SDMA_EVT_MIRROR2,          0x53FD4064,__READ      ,__sdma_evt_mirror2_bits);
__IO_REG32_BIT(SDMA_XTRIG_CONF1,          0x53FD4070,__READ_WRITE,__sdma_xtrig_conf1_bits);
__IO_REG32_BIT(SDMA_XTRIG_CONF2,          0x53FD4074,__READ_WRITE,__sdma_xtrig_conf2_bits);
__IO_REG32_BIT(SDMA_CHNPRI0,              0x53FD4100,__READ_WRITE,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI1,              0x53FD4104,__READ_WRITE,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI2,              0x53FD4108,__READ_WRITE,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI3,              0x53FD410C,__READ_WRITE,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI4,              0x53FD4110,__READ_WRITE,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI5,              0x53FD4114,__READ_WRITE,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI6,              0x53FD4118,__READ_WRITE,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI7,              0x53FD411C,__READ_WRITE,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI8,              0x53FD4120,__READ_WRITE,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI9,              0x53FD4124,__READ_WRITE,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI10,             0x53FD4128,__READ_WRITE,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI11,             0x53FD412C,__READ_WRITE,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI12,             0x53FD4130,__READ_WRITE,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI13,             0x53FD4134,__READ_WRITE,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI14,             0x53FD4138,__READ_WRITE,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI15,             0x53FD413C,__READ_WRITE,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI16,             0x53FD4140,__READ_WRITE,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI17,             0x53FD4144,__READ_WRITE,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI18,             0x53FD4148,__READ_WRITE,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI19,             0x53FD414C,__READ_WRITE,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI20,             0x53FD4150,__READ_WRITE,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI21,             0x53FD4154,__READ_WRITE,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI22,             0x53FD4158,__READ_WRITE,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI23,             0x53FD415C,__READ_WRITE,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI24,             0x53FD4160,__READ_WRITE,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI25,             0x53FD4164,__READ_WRITE,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI26,             0x53FD4168,__READ_WRITE,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI27,             0x53FD416C,__READ_WRITE,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI28,             0x53FD4170,__READ_WRITE,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI29,             0x53FD4174,__READ_WRITE,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI30,             0x53FD4178,__READ_WRITE,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI31,             0x53FD417C,__READ_WRITE,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNENBL0,             0x53FD4200,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL1,             0x53FD4204,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL2,             0x53FD4208,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL3,             0x53FD420C,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL4,             0x53FD4210,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL5,             0x53FD4214,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL6,             0x53FD4218,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL7,             0x53FD421C,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL8,             0x53FD4220,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL9,             0x53FD4224,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL10,            0x53FD4228,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL11,            0x53FD422C,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL12,            0x53FD4230,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL13,            0x53FD4234,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL14,            0x53FD4238,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL15,            0x53FD423C,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL16,            0x53FD4240,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL17,            0x53FD4244,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL18,            0x53FD4248,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL19,            0x53FD424C,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL20,            0x53FD4250,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL21,            0x53FD4254,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL22,            0x53FD4258,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL23,            0x53FD425C,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL24,            0x53FD4260,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL25,            0x53FD4264,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL26,            0x53FD4268,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL27,            0x53FD426C,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL28,            0x53FD4270,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL29,            0x53FD4274,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL30,            0x53FD4278,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL31,            0x53FD427C,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL32,            0x53FD4280,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL33,            0x53FD4284,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL34,            0x53FD4288,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL35,            0x53FD428C,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL36,            0x53FD4290,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL37,            0x53FD4294,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL38,            0x53FD4298,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL39,            0x53FD429C,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL40,            0x53FD42A0,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL41,            0x53FD42A4,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL42,            0x53FD42A8,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL43,            0x53FD42AC,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL44,            0x53FD42B0,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL45,            0x53FD42B4,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL46,            0x53FD42B8,__READ_WRITE,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL47,            0x53FD42BC,__READ_WRITE,__sdma_chnenbl_bits);

/***************************************************************************
 **
 **  SIM1
 **
 ***************************************************************************/
__IO_REG32_BIT(SIM1_PORT1_CNTL,           0x50024000,__READ_WRITE,__sim_port_cntl_bits);
__IO_REG32_BIT(SIM1_SETUP,                0x50024004,__READ_WRITE,__sim_setup_bits);
__IO_REG32_BIT(SIM1_PORT1_DETECT,         0x50024008,__READ_WRITE,__sim_port_detect_bits);
__IO_REG32_BIT(SIM1_PORT1_XMT_BUF,        0x5002400C,__READ_WRITE,__sim_port_xmt_buf_bits);
__IO_REG32_BIT(SIM1_PORT1_RCV_BUF,        0x50024010,__READ      ,__sim_port_rcv_buf_bits);
__IO_REG32_BIT(SIM1_PORT0_CNTL,           0x50024014,__READ_WRITE,__sim_port_cntl_bits);
__IO_REG32_BIT(SIM1_CNTL,                 0x50024018,__READ_WRITE,__sim_cntl_bits);
__IO_REG32_BIT(SIM1_CLK_PRESCALER,        0x5002401C,__READ_WRITE,__sim_clk_prescaler_bits);
__IO_REG32_BIT(SIM1_RCV_THRESHOLD,        0x50024020,__READ_WRITE,__sim_rcv_threshold_bits);
__IO_REG32_BIT(SIM1_ENABLE,               0x50024024,__READ_WRITE,__sim_enable_bits);
__IO_REG32_BIT(SIM1_XMT_STATUS,           0x50024028,__READ_WRITE,__sim_xmt_status_bits);
__IO_REG32_BIT(SIM1_RCV_STATUS,           0x5002402C,__READ_WRITE,__sim_rcv_status_bits);
__IO_REG32_BIT(SIM1_INT_MASK,             0x50024030,__READ_WRITE,__sim_int_mask_bits);
__IO_REG32_BIT(SIM1_PORT0_XMT_BUF,        0x50024034,__READ_WRITE,__sim_port_xmt_buf_bits);
__IO_REG32_BIT(SIM1_PORT0_RCV_BUF,        0x50024038,__READ      ,__sim_port_rcv_buf_bits);
__IO_REG32_BIT(SIM1_PORT0_DETECT,         0x5002403C,__READ_WRITE,__sim_port_detect_bits);
__IO_REG32_BIT(SIM1_DATA_FORMAT,          0x50024040,__READ_WRITE,__sim_data_format_bits);
__IO_REG32_BIT(SIM1_XMT_THRESHOLD,        0x50024044,__READ_WRITE,__sim_xmt_threshold_bits);
__IO_REG32_BIT(SIM1_GUARD_CNTL,           0x50024048,__READ_WRITE,__sim_guard_cntl_bits);
__IO_REG32_BIT(SIM1_OD_CONFIG,            0x5002404C,__READ_WRITE,__sim_od_config_bits);
__IO_REG32_BIT(SIM1_RESET_CNTL,           0x50024050,__READ_WRITE,__sim_reset_cntl_bits);
__IO_REG32_BIT(SIM1_CHAR_WAIT,            0x50024054,__READ_WRITE,__sim_char_wait_bits);
__IO_REG32_BIT(SIM1_GPCNT,                0x50024058,__READ_WRITE,__sim_gpcnt_bits);
__IO_REG32_BIT(SIM1_DIVISOR,              0x5002405C,__READ_WRITE,__sim_divisor_bits);
__IO_REG32_BIT(SIM1_BWT,                  0x50024060,__READ_WRITE,__sim_bwt_bits);
__IO_REG32_BIT(SIM1_BGT,                  0x50024064,__READ_WRITE,__sim_bgt_bits);
__IO_REG32_BIT(SIM1_BWT_H,                0x50024068,__READ_WRITE,__sim_bwt_h_bits);
__IO_REG32_BIT(SIM1_XMT_FIFO_STAT,        0x5002406C,__READ      ,__sim_xmt_fifo_stat_bits);
__IO_REG32_BIT(SIM1_RCV_FIFO_CNT,         0x50024070,__READ      ,__sim_rcv_fifo_cnt_bits);
__IO_REG32_BIT(SIM1_RCV_FIFO_WPTR,        0x50024074,__READ      ,__sim_rcv_fifo_wptr_bits);
__IO_REG32_BIT(SIM1_RCV_FIFO_RPTR,        0x50024078,__READ      ,__sim_rcv_fifo_rptr_bits);

/***************************************************************************
 **
 **  SIM2
 **
 ***************************************************************************/
__IO_REG32_BIT(SIM2_PORT1_CNTL,           0x50028000,__READ_WRITE,__sim_port_cntl_bits);
__IO_REG32_BIT(SIM2_SETUP,                0x50028004,__READ_WRITE,__sim_setup_bits);
__IO_REG32_BIT(SIM2_PORT1_DETECT,         0x50028008,__READ_WRITE,__sim_port_detect_bits);
__IO_REG32_BIT(SIM2_PORT1_XMT_BUF,        0x5002800C,__READ_WRITE,__sim_port_xmt_buf_bits);
__IO_REG32_BIT(SIM2_PORT1_RCV_BUF,        0x50028010,__READ      ,__sim_port_rcv_buf_bits);
__IO_REG32_BIT(SIM2_PORT0_CNTL,           0x50028014,__READ_WRITE,__sim_port_cntl_bits);
__IO_REG32_BIT(SIM2_CNTL,                 0x50028018,__READ_WRITE,__sim_cntl_bits);
__IO_REG32_BIT(SIM2_CLK_PRESCALER,        0x5002801C,__READ_WRITE,__sim_clk_prescaler_bits);
__IO_REG32_BIT(SIM2_RCV_THRESHOLD,        0x50028020,__READ_WRITE,__sim_rcv_threshold_bits);
__IO_REG32_BIT(SIM2_ENABLE,               0x50028024,__READ_WRITE,__sim_enable_bits);
__IO_REG32_BIT(SIM2_XMT_STATUS,           0x50028028,__READ_WRITE,__sim_xmt_status_bits);
__IO_REG32_BIT(SIM2_RCV_STATUS,           0x5002802C,__READ_WRITE,__sim_rcv_status_bits);
__IO_REG32_BIT(SIM2_INT_MASK,             0x50028030,__READ_WRITE,__sim_int_mask_bits);
__IO_REG32_BIT(SIM2_PORT0_XMT_BUF,        0x50028034,__READ_WRITE,__sim_port_xmt_buf_bits);
__IO_REG32_BIT(SIM2_PORT0_RCV_BUF,        0x50028038,__READ      ,__sim_port_rcv_buf_bits);
__IO_REG32_BIT(SIM2_PORT0_DETECT,         0x5002803C,__READ_WRITE,__sim_port_detect_bits);
__IO_REG32_BIT(SIM2_DATA_FORMAT,          0x50028040,__READ_WRITE,__sim_data_format_bits);
__IO_REG32_BIT(SIM2_XMT_THRESHOLD,        0x50028044,__READ_WRITE,__sim_xmt_threshold_bits);
__IO_REG32_BIT(SIM2_GUARD_CNTL,           0x50028048,__READ_WRITE,__sim_guard_cntl_bits);
__IO_REG32_BIT(SIM2_OD_CONFIG,            0x5002804C,__READ_WRITE,__sim_od_config_bits);
__IO_REG32_BIT(SIM2_RESET_CNTL,           0x50028050,__READ_WRITE,__sim_reset_cntl_bits);
__IO_REG32_BIT(SIM2_CHAR_WAIT,            0x50028054,__READ_WRITE,__sim_char_wait_bits);
__IO_REG32_BIT(SIM2_GPCNT,                0x50028058,__READ_WRITE,__sim_gpcnt_bits);
__IO_REG32_BIT(SIM2_DIVISOR,              0x5002805C,__READ_WRITE,__sim_divisor_bits);
__IO_REG32_BIT(SIM2_BWT,                  0x50028060,__READ_WRITE,__sim_bwt_bits);
__IO_REG32_BIT(SIM2_BGT,                  0x50028064,__READ_WRITE,__sim_bgt_bits);
__IO_REG32_BIT(SIM2_BWT_H,                0x50028068,__READ_WRITE,__sim_bwt_h_bits);
__IO_REG32_BIT(SIM2_XMT_FIFO_STAT,        0x5002806C,__READ      ,__sim_xmt_fifo_stat_bits);
__IO_REG32_BIT(SIM2_RCV_FIFO_CNT,         0x50028070,__READ      ,__sim_rcv_fifo_cnt_bits);
__IO_REG32_BIT(SIM2_RCV_FIFO_WPTR,        0x50028074,__READ      ,__sim_rcv_fifo_wptr_bits);
__IO_REG32_BIT(SIM2_RCV_FIFO_RPTR,        0x50028078,__READ      ,__sim_rcv_fifo_rptr_bits);

/***************************************************************************
 **
 **  Smart Liquid Crystal Display Controller (SLCDC)
 **
 ***************************************************************************/
__IO_REG32(    DATA_BASE_ADDR,            0x53FC0000,__READ_WRITE);
#define DATABASEADR         DATA_BASE_ADDR
__IO_REG32_BIT(ATA_BUFF_SIZE,             0x53FC0004,__READ_WRITE,__data_buff_size_bits);
#define LCDDATABUFSIZE      ATA_BUFF_SIZE
#define LCDDATABUFSIZE_bit  ATA_BUFF_SIZE_bit
__IO_REG32(    CMD_BASE_ADDR,             0x53FC0008,__READ_WRITE);
#define COMBASEADR          CMD_BASE_ADDR
__IO_REG32_BIT(CMD_BUFF_SIZE,             0x53FC000C,__READ_WRITE,__cmd_buff_size_bits);
#define COMBUFSIZ           CMD_BUFF_SIZE
#define COMBUFSIZ_bit       CMD_BUFF_SIZE_bit
__IO_REG32_BIT(STRING_SIZE,               0x53FC0010,__READ_WRITE,__string_size_bits);
#define LCDCOMSTRINGSIZ     STRING_SIZE
#define LCDCOMSTRINGSIZ_bit STRING_SIZE_bit
__IO_REG32_BIT(FIFO_CONFIG,               0x53FC0014,__READ_WRITE,__fifo_config_bits);
#define FIFOCONFIG          FIFO_CONFIG
#define FIFOCONFIG_bit      FIFO_CONFIG_bit
__IO_REG32_BIT(LCD_CONFIG,                0x53FC0018,__READ_WRITE,__lcd_config_bits);
#define LCDCONFIG           LCD_CONFIG
#define LCDCONFIG_bit       LCD_CONFIG_bit
__IO_REG32_BIT(LCDTRANSCONFIG,            0x53FC001C,__READ_WRITE,__lcdtransconfig_bits);
__IO_REG32_BIT(DMA_CTRL_STAT,             0x53FC0020,__READ_WRITE,__dma_ctrl_stat_bits);
#define SLCDCCONTROL        DMA_CTRL_STAT
#define SLCDCCONTROL_bit    DMA_CTRL_STAT_bit
__IO_REG32_BIT(LCD_CLK_CONFIG,            0x53FC0024,__READ_WRITE,__lcd_clk_config_bits);
#define LCDCLOCKCONFIG      LCD_CLK_CONFIG
#define LCDCLOCKCONFIG_bit  LCD_CLK_CONFIG_bit
__IO_REG32_BIT(LCD_WRITE_DATA,            0x53FC0028,__READ_WRITE,__lcd_write_data_bits);

/***************************************************************************
 **
 **  SSI1
 **
 ***************************************************************************/
__IO_REG32(    SSI1_STX0,                 0x50034000,__READ_WRITE);
__IO_REG32(    SSI1_STX1,                 0x50034004,__READ_WRITE);
__IO_REG32(    SSI1_SRX0,                 0x50034008,__READ      );
__IO_REG32(    SSI1_SRX1,                 0x5003400C,__READ      );
__IO_REG32_BIT(SSI1_SCR,                  0x50034010,__READ_WRITE,__scsr_bits);
__IO_REG32_BIT(SSI1_SISR,                 0x50034014,__READ      ,__sisr_bits);
__IO_REG32_BIT(SSI1_SIER,                 0x50034018,__READ_WRITE,__sier_bits);
__IO_REG32_BIT(SSI1_STCR,                 0x5003401C,__READ_WRITE,__stcr_bits);
__IO_REG32_BIT(SSI1_SRCR,                 0x50034020,__READ_WRITE,__srcr_bits);
__IO_REG32_BIT(SSI1_STCCR,                0x50034024,__READ_WRITE,__ssi_ccr_bits);
__IO_REG32_BIT(SSI1_SRCCR,                0x50034028,__READ_WRITE,__ssi_ccr_bits);
__IO_REG32_BIT(SSI1_SFCSR,                0x5003402C,__READ_WRITE,__ssi_sfcsr_bits);
__IO_REG32_BIT(SSI1_STR,                  0x50034030,__READ_WRITE,__ssi_str_bits);
__IO_REG32_BIT(SSI1_SOR,                  0x50034034,__READ_WRITE,__ssi_sor_bits);
__IO_REG32_BIT(SSI1_SACNT,                0x50034038,__READ_WRITE,__ssi_sacnt_bits);
__IO_REG32_BIT(SSI1_SACADD,               0x5003403C,__READ_WRITE,__ssi_sacadd_bits);
__IO_REG32_BIT(SSI1_SACDAT,               0x50034040,__READ_WRITE,__ssi_sacdat_bits);
__IO_REG32_BIT(SSI1_SATAG,                0x50034044,__READ_WRITE,__ssi_satag_bits);
__IO_REG32(    SSI1_STMSK,                0x50034048,__READ_WRITE);
__IO_REG32(    SSI1_SRMSK,                0x5003404C,__READ_WRITE);
__IO_REG32_BIT(SSI1_SACCST,               0x50034050,__READ      ,__ssi_saccst_bits);
__IO_REG32_BIT(SSI1_SACCEN,               0x50034054,__WRITE     ,__ssi_saccen_bits);
__IO_REG32_BIT(SSI1_SACCDIS,              0x50034058,__WRITE     ,__ssi_saccdis_bits);

/***************************************************************************
 **
 **  SSI2
 **
 ***************************************************************************/
__IO_REG32(    SSI2_STX0,                 0x50014000,__READ_WRITE);
__IO_REG32(    SSI2_STX1,                 0x50014004,__READ_WRITE);
__IO_REG32(    SSI2_SRX0,                 0x50014008,__READ      );
__IO_REG32(    SSI2_SRX1,                 0x5001400C,__READ      );
__IO_REG32_BIT(SSI2_SCR,                  0x50014010,__READ_WRITE,__scsr_bits);
__IO_REG32_BIT(SSI2_SISR,                 0x50014014,__READ      ,__sisr_bits);
__IO_REG32_BIT(SSI2_SIER,                 0x50014018,__READ_WRITE,__sier_bits);
__IO_REG32_BIT(SSI2_STCR,                 0x5001401C,__READ_WRITE,__stcr_bits);
__IO_REG32_BIT(SSI2_SRCR,                 0x50014020,__READ_WRITE,__srcr_bits);
__IO_REG32_BIT(SSI2_STCCR,                0x50014024,__READ_WRITE,__ssi_ccr_bits);
__IO_REG32_BIT(SSI2_SRCCR,                0x50014028,__READ_WRITE,__ssi_ccr_bits);
__IO_REG32_BIT(SSI2_SFCSR,                0x5001402C,__READ_WRITE,__ssi_sfcsr_bits);
__IO_REG32_BIT(SSI2_STR,                  0x50014030,__READ_WRITE,__ssi_str_bits);
__IO_REG32_BIT(SSI2_SOR,                  0x50014034,__READ_WRITE,__ssi_sor_bits);
__IO_REG32_BIT(SSI2_SACNT,                0x50014038,__READ_WRITE,__ssi_sacnt_bits);
__IO_REG32_BIT(SSI2_SACADD,               0x5001403C,__READ_WRITE,__ssi_sacadd_bits);
__IO_REG32_BIT(SSI2_SACDAT,               0x50014040,__READ_WRITE,__ssi_sacdat_bits);
__IO_REG32_BIT(SSI2_SATAG,                0x50014044,__READ_WRITE,__ssi_satag_bits);
__IO_REG32(    SSI2_STMSK,                0x50014048,__READ_WRITE);
__IO_REG32(    SSI2_SRMSK,                0x5001404C,__READ_WRITE);
__IO_REG32_BIT(SSI2_SACCST,               0x50014050,__READ      ,__ssi_saccst_bits);
__IO_REG32_BIT(SSI2_SACCEN,               0x50014054,__WRITE     ,__ssi_saccen_bits);
__IO_REG32_BIT(SSI2_SACCDIS,              0x50014058,__WRITE     ,__ssi_saccdis_bits);

/***************************************************************************
 **
 **  TSC
 **
 ***************************************************************************/
__IO_REG32_BIT(TGCR,                      0x50030000,__READ_WRITE,__tgcr_bits);
__IO_REG32_BIT(TGSR,                      0x50030004,__READ      ,__tgsr_bits);
__IO_REG32_BIT(TICR,                      0x50030008,__READ_WRITE,__ticr_bits);
__IO_REG32_BIT(TCQFIFO,                   0x50030400,__READ      ,__tcqfifo_bits);
__IO_REG32_BIT(TCQCR,                     0x50030404,__READ_WRITE,__tcqcr_bits);
__IO_REG32_BIT(TCQSR,                     0x50030408,__READ_WRITE,__tcqsr_bits);
__IO_REG32_BIT(TCQMR,                     0x5003040C,__READ_WRITE,__tcqmr_bits);
__IO_REG32_BIT(TCQ_ITEM_7_0,              0x50030420,__READ_WRITE,__tcq_item_7_0_bits);
__IO_REG32_BIT(TCQ_ITEM_15_8,             0x50030424,__READ_WRITE,__tcq_item_15_8_bits);
__IO_REG32_BIT(TCC0,                      0x50030440,__READ_WRITE,__ticr_bits);
__IO_REG32_BIT(TCC1,                      0x50030444,__READ_WRITE,__ticr_bits);
__IO_REG32_BIT(TCC2,                      0x50030448,__READ_WRITE,__ticr_bits);
__IO_REG32_BIT(TCC3,                      0x5003044C,__READ_WRITE,__ticr_bits);
__IO_REG32_BIT(TCC4,                      0x50030450,__READ_WRITE,__ticr_bits);
__IO_REG32_BIT(TCC5,                      0x50030454,__READ_WRITE,__ticr_bits);
__IO_REG32_BIT(TCC6,                      0x50030458,__READ_WRITE,__ticr_bits);
__IO_REG32_BIT(TCC7,                      0x5003045C,__READ_WRITE,__ticr_bits);
__IO_REG32_BIT(GCQFIFO,                   0x50030800,__READ      ,__tcqfifo_bits);
__IO_REG32_BIT(GCQCR,                     0x50030804,__READ_WRITE,__tcqcr_bits);
__IO_REG32_BIT(GCQSR,                     0x50030808,__READ_WRITE,__tcqsr_bits);
__IO_REG32_BIT(GCQMR,                     0x5003080C,__READ_WRITE,__tcqmr_bits);
__IO_REG32_BIT(GCQ_ITEM_7_0,              0x50030820,__READ_WRITE,__tcq_item_7_0_bits);
__IO_REG32_BIT(GCQ_ITEM_15_8,             0x50030824,__READ_WRITE,__tcq_item_15_8_bits);
__IO_REG32_BIT(GCC0,                      0x50030840,__READ_WRITE,__ticr_bits);
__IO_REG32_BIT(GCC1,                      0x50030844,__READ_WRITE,__ticr_bits);
__IO_REG32_BIT(GCC2,                      0x50030848,__READ_WRITE,__ticr_bits);
__IO_REG32_BIT(GCC3,                      0x5003084C,__READ_WRITE,__ticr_bits);
__IO_REG32_BIT(GCC4,                      0x50030850,__READ_WRITE,__ticr_bits);
__IO_REG32_BIT(GCC5,                      0x50030854,__READ_WRITE,__ticr_bits);
__IO_REG32_BIT(GCC6,                      0x50030858,__READ_WRITE,__ticr_bits);
__IO_REG32_BIT(GCC7,                      0x5003085C,__READ_WRITE,__ticr_bits);

/***************************************************************************
 **
 **  UART1
 **
 ***************************************************************************/
__IO_REG32_BIT(URXD_1,                    0x43F90000,__READ      ,__urxd_bits);
__IO_REG32_BIT(UTXD_1,                    0x43F90040,__WRITE     ,__utxd_bits);
__IO_REG32_BIT(UCR1_1,                    0x43F90080,__READ_WRITE,__ucr1_bits);
__IO_REG32_BIT(UCR2_1,                    0x43F90084,__READ_WRITE,__ucr2_bits);
__IO_REG32_BIT(UCR3_1,                    0x43F90088,__READ_WRITE,__ucr3_bits);
__IO_REG32_BIT(UCR4_1,                    0x43F9008C,__READ_WRITE,__ucr4_bits);
__IO_REG32_BIT(UFCR_1,                    0x43F90090,__READ_WRITE,__ufcr_bits);
__IO_REG32_BIT(USR1_1,                    0x43F90094,__READ_WRITE,__usr1_bits);
__IO_REG32_BIT(USR2_1,                    0x43F90098,__READ_WRITE,__usr2_bits);
__IO_REG32_BIT(UESC_1,                    0x43F9009C,__READ_WRITE,__uesc_bits);
__IO_REG32_BIT(UTIM_1,                    0x43F900A0,__READ_WRITE,__utim_bits);
__IO_REG32(    UBIR_1,                    0x43F900A4,__READ_WRITE);
__IO_REG32(    UBMR_1,                    0x43F900A8,__READ_WRITE);
__IO_REG32(    UBRC_1,                    0x43F900AC,__READ_WRITE);
__IO_REG32(    ONEMS_1,                   0x43F900B0,__READ_WRITE);
__IO_REG32_BIT(UTS_1,                     0x43F900B4,__READ_WRITE,__uts_bits);

/***************************************************************************
 **
 **  UART2
 **
 ***************************************************************************/
__IO_REG32_BIT(URXD_2,                    0x43F94000,__READ      ,__urxd_bits);
__IO_REG32_BIT(UTXD_2,                    0x43F94040,__WRITE     ,__utxd_bits);
__IO_REG32_BIT(UCR1_2,                    0x43F94080,__READ_WRITE,__ucr1_bits);
__IO_REG32_BIT(UCR2_2,                    0x43F94084,__READ_WRITE,__ucr2_bits);
__IO_REG32_BIT(UCR3_2,                    0x43F94088,__READ_WRITE,__ucr3_bits);
__IO_REG32_BIT(UCR4_2,                    0x43F9408C,__READ_WRITE,__ucr4_bits);
__IO_REG32_BIT(UFCR_2,                    0x43F94090,__READ_WRITE,__ufcr_bits);
__IO_REG32_BIT(USR1_2,                    0x43F94094,__READ_WRITE,__usr1_bits);
__IO_REG32_BIT(USR2_2,                    0x43F94098,__READ_WRITE,__usr2_bits);
__IO_REG32_BIT(UESC_2,                    0x43F9409C,__READ_WRITE,__uesc_bits);
__IO_REG32_BIT(UTIM_2,                    0x43F940A0,__READ_WRITE,__utim_bits);
__IO_REG32(    UBIR_2,                    0x43F940A4,__READ_WRITE);
__IO_REG32(    UBMR_2,                    0x43F940A8,__READ_WRITE);
__IO_REG32(    UBRC_2,                    0x43F940AC,__READ_WRITE);
__IO_REG32(    ONEMS_2,                   0x43F940B0,__READ_WRITE);
__IO_REG32_BIT(UTS_2,                     0x43F940B4,__READ_WRITE,__uts_bits);

/***************************************************************************
 **
 **  UART3
 **
 ***************************************************************************/
__IO_REG32_BIT(URXD_3,                    0x5000C000,__READ      ,__urxd_bits);
__IO_REG32_BIT(UTXD_3,                    0x5000C040,__WRITE     ,__utxd_bits);
__IO_REG32_BIT(UCR1_3,                    0x5000C080,__READ_WRITE,__ucr1_bits);
__IO_REG32_BIT(UCR2_3,                    0x5000C084,__READ_WRITE,__ucr2_bits);
__IO_REG32_BIT(UCR3_3,                    0x5000C088,__READ_WRITE,__ucr3_bits);
__IO_REG32_BIT(UCR4_3,                    0x5000C08C,__READ_WRITE,__ucr4_bits);
__IO_REG32_BIT(UFCR_3,                    0x5000C090,__READ_WRITE,__ufcr_bits);
__IO_REG32_BIT(USR1_3,                    0x5000C094,__READ_WRITE,__usr1_bits);
__IO_REG32_BIT(USR2_3,                    0x5000C098,__READ_WRITE,__usr2_bits);
__IO_REG32_BIT(UESC_3,                    0x5000C09C,__READ_WRITE,__uesc_bits);
__IO_REG32_BIT(UTIM_3,                    0x5000C0A0,__READ_WRITE,__utim_bits);
__IO_REG32(    UBIR_3,                    0x5000C0A4,__READ_WRITE);
__IO_REG32(    UBMR_3,                    0x5000C0A8,__READ_WRITE);
__IO_REG32(    UBRC_3,                    0x5000C0AC,__READ_WRITE);
__IO_REG32(    ONEMS_3,                   0x5000C0B0,__READ_WRITE);
__IO_REG32_BIT(UTS_3,                     0x5000C0B4,__READ_WRITE,__uts_bits);

/***************************************************************************
 **
 **  UART4
 **
 ***************************************************************************/
__IO_REG32_BIT(URXD_4,                    0x50008000,__READ      ,__urxd_bits);
__IO_REG32_BIT(UTXD_4,                    0x50008040,__WRITE     ,__utxd_bits);
__IO_REG32_BIT(UCR1_4,                    0x50008080,__READ_WRITE,__ucr1_bits);
__IO_REG32_BIT(UCR2_4,                    0x50008084,__READ_WRITE,__ucr2_bits);
__IO_REG32_BIT(UCR3_4,                    0x50008088,__READ_WRITE,__ucr3_bits);
__IO_REG32_BIT(UCR4_4,                    0x5000808C,__READ_WRITE,__ucr4_bits);
__IO_REG32_BIT(UFCR_4,                    0x50008090,__READ_WRITE,__ufcr_bits);
__IO_REG32_BIT(USR1_4,                    0x50008094,__READ_WRITE,__usr1_bits);
__IO_REG32_BIT(USR2_4,                    0x50008098,__READ_WRITE,__usr2_bits);
__IO_REG32_BIT(UESC_4,                    0x5000809C,__READ_WRITE,__uesc_bits);
__IO_REG32_BIT(UTIM_4,                    0x500080A0,__READ_WRITE,__utim_bits);
__IO_REG32(    UBIR_4,                    0x500080A4,__READ_WRITE);
__IO_REG32(    UBMR_4,                    0x500080A8,__READ_WRITE);
__IO_REG32(    UBRC_4,                    0x500080AC,__READ_WRITE);
__IO_REG32(    ONEMS_4,                   0x500080B0,__READ_WRITE);
__IO_REG32_BIT(UTS_4,                     0x500080B4,__READ_WRITE,__uts_bits);

/***************************************************************************
 **
 **  UART5
 **
 ***************************************************************************/
__IO_REG32_BIT(URXD_5,                    0x5002C000,__READ      ,__urxd_bits);
__IO_REG32_BIT(UTXD_5,                    0x5002C040,__WRITE     ,__utxd_bits);
__IO_REG32_BIT(UCR1_5,                    0x5002C080,__READ_WRITE,__ucr1_bits);
__IO_REG32_BIT(UCR2_5,                    0x5002C084,__READ_WRITE,__ucr2_bits);
__IO_REG32_BIT(UCR3_5,                    0x5002C088,__READ_WRITE,__ucr3_bits);
__IO_REG32_BIT(UCR4_5,                    0x5002C08C,__READ_WRITE,__ucr4_bits);
__IO_REG32_BIT(UFCR_5,                    0x5002C090,__READ_WRITE,__ufcr_bits);
__IO_REG32_BIT(USR1_5,                    0x5002C094,__READ_WRITE,__usr1_bits);
__IO_REG32_BIT(USR2_5,                    0x5002C098,__READ_WRITE,__usr2_bits);
__IO_REG32_BIT(UESC_5,                    0x5002C09C,__READ_WRITE,__uesc_bits);
__IO_REG32_BIT(UTIM_5,                    0x5002C0A0,__READ_WRITE,__utim_bits);
__IO_REG32(    UBIR_5,                    0x5002C0A4,__READ_WRITE);
__IO_REG32(    UBMR_5,                    0x5002C0A8,__READ_WRITE);
__IO_REG32(    UBRC_5,                    0x5002C0AC,__READ_WRITE);
__IO_REG32(    ONEMS_5,                   0x5002C0B0,__READ_WRITE);
__IO_REG32_BIT(UTS_5,                     0x5002C0B4,__READ_WRITE,__uts_bits);

/***************************************************************************
 **
 **  USB OTG
 **
 ***************************************************************************/
__IO_REG32_BIT(UOG_ID,                    0x53FF4000,__READ      ,__usb_id_bits);
__IO_REG32_BIT(UOG_HWGENERAL,             0x53FF4004,__READ      ,__usb_hwgeneral_bits);
__IO_REG32_BIT(UOG_HWHOST,                0x53FF4008,__READ      ,__usb_hwhost_bits);
__IO_REG32_BIT(UOG_HWDEVICE,              0x53FF400C,__READ      ,__usb_hwdevice_bits);
__IO_REG32_BIT(UOG_HWTXBUF,               0x53FF4010,__READ      ,__usb_hwtxbuf_bits);
__IO_REG32_BIT(UOG_HWRXBUF,               0x53FF4014,__READ      ,__usb_hwrxbuf_bits);
__IO_REG32_BIT(UOG_GPTIMER0LD,            0x53FF4080,__READ_WRITE,__usb_gptimer0ld_bits);
__IO_REG32_BIT(UOG_GPTIMER0CTRL,          0x53FF4084,__READ_WRITE,__usb_gptimer0ctrl_bits);
__IO_REG32(    UOG_GPTIMER1LD,            0x53FF4088,__READ_WRITE);
__IO_REG32(    UOG_GPTIMER1CTRL,          0x53FF408C,__READ_WRITE);
__IO_REG32_BIT(UOG_SBUSCFG,               0x53FF4090,__READ_WRITE,__uog_sbuscfg_bits);
__IO_REG8(     UOG_CAPLENGTH,             0x53FF4100,__READ      );
__IO_REG16(    UOG_HCIVERSION,            0x53FF4102,__READ      );
__IO_REG32_BIT(UOG_HCSPARAMS,             0x53FF4104,__READ      ,__usb_hcsparams_bits);
__IO_REG32_BIT(UOG_HCCPARAMS,             0x53FF4108,__READ      ,__usb_hccparams_bits);
__IO_REG32_BIT(UOG_DCIVERSION,            0x53FF4120,__READ      ,__usb_dciversion_bits);
__IO_REG32_BIT(UOG_DCCPARAMS,             0x53FF4124,__READ      ,__usb_dccparams_bits);
__IO_REG32_BIT(UOG_USBCMD,                0x53FF4140,__READ_WRITE,__usb_usbcmd_bits);
__IO_REG32_BIT(UOG_USBSTS,                0x53FF4144,__READ_WRITE,__usb_usbsts_bits);
__IO_REG32_BIT(UOG_USBINTR,               0x53FF4148,__READ_WRITE,__usb_usbintr_bits);
__IO_REG32_BIT(UOG_FRINDEX,               0x53FF414C,__READ_WRITE,__usb_frindex_bits);
__IO_REG32_BIT(UOG_PERIODICLISTBASE,      0x53FF4154,__READ_WRITE,__usb_periodiclistbase_bits);
#define UOG_DEVICEADDR      UOG_PERIODICLISTBASE
#define UOG_DEVICEADDR_bit  UOG_PERIODICLISTBASE_bit
__IO_REG32(    UOG_ASYNCLISTADDR,         0x53FF4158,__READ_WRITE);
#define UOG_ENDPOINTLISTADDR  UOG_ASYNCLISTADDR
__IO_REG32_BIT(UOG_BURSTSIZE,             0x53FF4160,__READ_WRITE,__usb_burstsize_bits);
__IO_REG32_BIT(UOG_TXFILLTUNING,          0x53FF4164,__READ_WRITE,__usb_txfilltuning_bits);
__IO_REG32_BIT(UOG_ULPIVIEW,              0x53FF4170,__READ_WRITE,__usb_ulpiview_bits);
__IO_REG32(    UOG_CFGFLAG,               0x53FF4180,__READ      );
__IO_REG32_BIT(UOG_PORTSC1,               0x53FF4184,__READ_WRITE,__usb_portsc_bits);
__IO_REG32_BIT(UOG_OTGSC,                 0x53FF41A4,__READ_WRITE,__usb_otgsc_bits);
__IO_REG32_BIT(UOG_USBMODE,               0x53FF41A8,__READ_WRITE,__usb_usbmode_bits);
__IO_REG32_BIT(UOG_ENDPTSETUPSTAT,        0x53FF41AC,__READ_WRITE,__usb_endptsetupstat_bits);
__IO_REG32_BIT(UOG_ENDPTPRIME,            0x53FF41B0,__READ_WRITE,__usb_endptprime_bits);
__IO_REG32_BIT(UOG_ENDPTFLUSH,            0x53FF41B4,__READ_WRITE,__usb_endptflush_bits);
__IO_REG32_BIT(UOG_ENDPTSTAT,             0x53FF41B8,__READ      ,__usb_endptstat_bits);
__IO_REG32_BIT(UOG_ENDPTCOMPLETE,         0x53FF41BC,__READ_WRITE,__usb_endptcomplete_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL0,            0x53FF41C0,__READ_WRITE,__usb_endptctrl0_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL1,            0x53FF41C4,__READ_WRITE,__usb_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL2,            0x53FF41C8,__READ_WRITE,__usb_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL3,            0x53FF41CC,__READ_WRITE,__usb_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL4,            0x53FF41D0,__READ_WRITE,__usb_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL5,            0x53FF41D4,__READ_WRITE,__usb_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL6,            0x53FF41D8,__READ_WRITE,__usb_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL7,            0x53FF41DC,__READ_WRITE,__usb_endptctrl_bits);

/***************************************************************************
 **
 **  USB HOST2
 **
 ***************************************************************************/
__IO_REG32_BIT(UH2_ID,                    0x53FF4400,__READ      ,__usb_id_bits);
__IO_REG32_BIT(UH2_HWGENERAL,             0x53FF4404,__READ      ,__usb_hwgeneral_bits);
__IO_REG32_BIT(UH2_HWHOST,                0x53FF4408,__READ      ,__usb_hwhost_bits);
__IO_REG32_BIT(UH2_HWTXBUF,               0x53FF4410,__READ      ,__usb_hwtxbuf_bits);
__IO_REG32_BIT(UH2_HWRXBUF,               0x53FF4414,__READ      ,__usb_hwrxbuf_bits);
__IO_REG32_BIT(UH2_GPTIMER0LD,            0x53FF4480,__READ_WRITE,__usb_gptimer0ld_bits);
__IO_REG32_BIT(UH2_GPTIMER0CTRL,          0x53FF4484,__READ_WRITE,__usb_gptimer0ctrl_bits);
__IO_REG32(    UH2_GPTIMER1LD,            0x53FF4488,__READ_WRITE);
__IO_REG32(    UH2_GPTIMER1CTRL,          0x53FF448C,__READ_WRITE);
__IO_REG32_BIT(UH2_SBUSCFG,               0x53FF4490,__READ_WRITE,__uog_sbuscfg_bits);
__IO_REG16(    UH2_CAPLENGTH,             0x53FF4500,__READ      );
__IO_REG16(    UH2_HCIVERSION,            0x53FF4502,__READ      );
__IO_REG32_BIT(UH2_HCSPARAMS,             0x53FF4504,__READ      ,__usb_hcsparams_bits);
__IO_REG32_BIT(UH2_HCCPARAMS,             0x53FF4508,__READ      ,__usb_hccparams_bits);
__IO_REG32_BIT(UH2_USBCMD,                0x53FF4540,__READ_WRITE,__usb_usbcmd_bits);
__IO_REG32_BIT(UH2_USBSTS,                0x53FF4544,__READ_WRITE,__usb_usbsts_bits);
__IO_REG32_BIT(UH2_USBINTR,               0x53FF4548,__READ_WRITE,__usb_usbintr_bits);
__IO_REG32_BIT(UH2_FRINDEX,               0x53FF454C,__READ_WRITE,__usb_frindex_bits);
__IO_REG32_BIT(UH2_PERIODICLISTBASE,      0x53FF4554,__READ_WRITE,__uh_periodiclistbase_bits);
#define UH2_DEVICEADDR      UH2_PERIODICLISTBASE
#define UH2_DEVICEADDR_bit  UH2_PERIODICLISTBASE_bit
__IO_REG32(    UH2_ASYNCLISTADDR,         0x53FF4558,__READ_WRITE);
__IO_REG32_BIT(UH2_BURSTSIZE,             0x53FF4560,__READ_WRITE,__usb_burstsize_bits);
__IO_REG32_BIT(UH2_TXFILLTUNING,          0x53FF4564,__READ_WRITE,__usb_txfilltuning_bits);
__IO_REG32_BIT(UH2_ULPIVIEW,              0x53FF4570,__READ_WRITE,__usb_ulpiview_bits);
__IO_REG32_BIT(UH2_PORTSC1,               0x53FF4584,__READ_WRITE,__usb_portsc_bits);
__IO_REG32_BIT(UH2_USBMODE,               0x53FF45A8,__READ_WRITE,__usb_usbmode_bits);

/***************************************************************************
 **
 **  USB
 **
 ***************************************************************************/
__IO_REG32_BIT(USB_CTRL,                  0x53FF4600,__READ_WRITE,__usb_ctrl_bits);
__IO_REG32_BIT(USB_OTG_MIRROR,            0x53FF4604,__READ_WRITE,__usb_otg_mirror_bits);
__IO_REG32_BIT(USB_PHY_CTRL_FUNC,         0x53FF4608,__READ_WRITE,__usb_phy_ctrl_func_bits);
__IO_REG32_BIT(USB_PHY_CTRL_TEST,         0x53FF460C,__READ_WRITE,__usb_phy_ctrl_test_bits);

/***************************************************************************
 **
 **  WDOG
 **
 ***************************************************************************/
__IO_REG16_BIT(WCR,                       0x53FDC000,__READ_WRITE,__wcr_bits);
__IO_REG16(    WSR,                       0x53FDC002,__READ_WRITE);
__IO_REG16_BIT(WRSR,                      0x53FDC004,__READ      ,__wrsr_bits);

/***************************************************************************
 **
 **  SPBA
 **
 ***************************************************************************/
__IO_REG32_BIT(SPBA_PRR0,                 0x50000000,__READ_WRITE,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR1,                 0x50000004,__READ_WRITE,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR2,                 0x50000008,__READ_WRITE,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR3,                 0x5000000C,__READ_WRITE,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR4,                 0x50000010,__READ_WRITE,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR5,                 0x50000014,__READ_WRITE,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR6,                 0x50000018,__READ_WRITE,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR7,                 0x5000001C,__READ_WRITE,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR8,                 0x50000020,__READ_WRITE,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR9,                 0x50000024,__READ_WRITE,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR10,                0x50000028,__READ_WRITE,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR11,                0x5000002C,__READ_WRITE,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR12,                0x50000030,__READ_WRITE,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR13,                0x50000034,__READ_WRITE,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR14,                0x50000038,__READ_WRITE,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR15,                0x5000003C,__READ_WRITE,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR16,                0x50000040,__READ_WRITE,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR17,                0x50000044,__READ_WRITE,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR18,                0x50000048,__READ_WRITE,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR19,                0x5000004C,__READ_WRITE,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR20,                0x50000050,__READ_WRITE,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR21,                0x50000054,__READ_WRITE,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR22,                0x50000058,__READ_WRITE,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR23,                0x5000005C,__READ_WRITE,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR24,                0x50000060,__READ_WRITE,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR25,                0x50000064,__READ_WRITE,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR26,                0x50000068,__READ_WRITE,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR27,                0x5000006C,__READ_WRITE,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR28,                0x50000070,__READ_WRITE,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR29,                0x50000074,__READ_WRITE,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR30,                0x50000078,__READ_WRITE,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR31,                0x5000007C,__READ_WRITE,__spba_prr_bits);

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
 **   MCIX25 DMA channels
 **
 ***************************************************************************/
#define DMA_EXT_DMA_0           0
#define DMA_CRM                 1
#define DMA_ATA_TXFER_END       2
#define DMA_ATA_TX_FIFO         3
#define DMA_ATA_RX_FIFO         4
#define DMA_CSPI2_RX            6
#define DMA_CSPI2_TX            7
#define DMA_CSPI1_RX            8
#define DMA_CSPI1_TX            9
#define DMA_UART3_RX           10
#define DMA_UART3_TX           11
#define DMA_UART4_RX           12
#define DMA_UART4_TX           13
#define DMA_EXT_DMA_1          14
#define DMA_EXT_DMA_2          15
#define DMA_UART2_RX           16
#define DMA_UART2_TX           17
#define DMA_UART1_RX           18
#define DMA_UART1_TX           19
#define DMA_SSI2_RX1           22
#define DMA_SSI2_TX1           23
#define DMA_SSI2_RX0           24
#define DMA_SSI2_TX0           25
#define DMA_SSI1_RX1           26
#define DMA_SSI1_TX1           27
#define DMA_SSI1_RX0           28
#define DMA_SSI1_TX0           29
#define DMA_NANDFC             30
#define DMA_ECT                31
#define DMA_ESAI_RX            32
#define DMA_ESAI_TX            33
#define DMA_CSPI3_RX           34
#define DMA_CSPI3_TX           35
#define DMA_SIM2_RX            36
#define DMA_SIM2_TX            37
#define DMA_SIM1_RX            38
#define DMA_SIM1_TX            39
#define DMA_TSC_GCQ            44
#define DMA_TSC_TCQ            45
#define DMA_UART5_RX           46
#define DMA_UART5_TX           47

/***************************************************************************
 **
 **   MCIX25 interrupt sources
 **
 ***************************************************************************/
#define INT_CSPI3              0              /* Configurable SPI (CSPI3)*/
#define INT_GPT4               1              /* General Purpose Timer (GPT4)*/
#define INT_OWIRE              2              /* One wire*/
#define INT_I2C1               3              /* I2C Bus Controller (I2C1)*/
#define INT_I2C2               4              /* I2C Bus Controller (I2C2)*/
#define INT_UART4              5              /* UART4*/
#define INT_RTIC               6              /* Real Time Integrity Checker (RTIC)*/
#define INT_ESAI               7              /*   */
#define INT_ESDHC2             8              /* Secured Digital Host Controller (SDHC2)*/
#define INT_ESDHC1             9              /* Secured Digital Host Controller (SDHC1)*/
#define INT_I2C3               10             /* I2C Bus Controller (I2C3)*/
#define INT_SS2                11             /* Synchronous Serial Interface (SSI2)*/
#define INT_SS1                12             /* Synchronous Serial Interface (SSI1)*/
#define INT_CSPI2              13             /* Configurable SPI (CSPI2)*/
#define INT_CSPI1              14             /* Configurable SPI (CSPI1)*/
#define INT_ATA                15             /* Advvanced Technology Attachment (ATA)*/
#define INT_GPIO3              16             /* General Purpose Input/Output (GPIO3)*/
#define INT_CSI                17             /* CMOS Sensor Interface (CSI)*/
#define INT_UART3              18             /* UART3*/
#define INT_IIM                19             /* IC Identify Module (IIM)*/
#define INT_SIM1               20             /*   */
#define INT_SIM2               21             /*   */
#define INT_RNGB               22             /*   */
#define INT_GPIO4              23             /* General Purpose Input/Output (GPIO4)*/
#define INT_KPP                24             /* Key Pad Port (KPP)*/
#define INT_RTC                25             /* Real-Time Clock (RTC)*/
#define INT_PWM                26             /* Pulse Width Modulator (PWM)*/
#define INT_EPIT2              27             /*   */
#define INT_EPIT1              28             /*   */
#define INT_GPT3               29             /* General Purpose Timer (GPT3)*/
#define INT_POWER_FAIL         30             /*   */
#define INT_CRM                31             /*   */
#define INT_UART2              32             /* UART2*/
#define INT_NANDFC             33             /* Nand Flash Controller (NFC)*/
#define INT_SDMA               34             /*   */
#define INT_USBHTG             35             /*   */
#define INT_PWM2               36             /* Pulse Width Modulator (PWM2)*/
#define INT_USBOTG             37             /* USB OTG*/
#define INT_SLCDC              38             /* Smart LCD Controller (SLCDC)*/
#define INT_LCDC               39             /* LCD Controller (LCDC)*/
#define INT_UART5              40             /* UART5*/
#define INT_PWM3               41             /* Pulse Width Modulator (PWM3)*/
#define INT_PWM4               42             /* Pulse Width Modulator (PWM4)*/
#define INT_CAN1               43             /*   */
#define INT_CAN2               44             /*   */
#define INT_UART1              45             /* UART1*/
#define INT_TSC                46             /* Touchscreen controller */
#define INT_ECT                48             /*   */
#define INT_SMN                49             /* SCC SMN*/
#define INT_SCM                50             /* SCC SCM*/
#define INT_GPIO2              51             /* General Purpose Input/Output (GPIO2)*/
#define INT_GPIO1              52             /* General Purpose Input/Output (GPIO1)*/
#define INT_GPT2               53             /* General Purpose Timer (GPT2)*/
#define INT_GPT1               54             /* General Purpose Timer (GPT1)*/
#define INT_WDOG               55             /* Watchdog (WDOG)*/
#define INT_DRY_ICE            56             /*   */
#define INT_FEC                57             /*   */
#define INT_EXT_INT5           58             /*   */
#define INT_EXT_INT4           59             /*   */
#define INT_EXT_INT3           60             /*   */
#define INT_EXT_INT2           61             /*   */
#define INT_EXT_INT1           62             /*   */
#define INT_EXT_INT0           63             /*   */

/***************************************************************************
 **
 **   IOMUX mux mode constants
 **
 ***************************************************************************/

#define  ALT0_MUX_MODE     0
#define  ALT1_MUX_MODE     1
#define  ALT2_MUX_MODE     2
#define  ALT3_MUX_MODE     3
#define  ALT4_MUX_MODE     4
#define  ALT5_MUX_MODE     5
#define  ALT6_MUX_MODE     6
#define  ALT7_MUX_MODE     7

#endif    /* __MCIX25_H */
