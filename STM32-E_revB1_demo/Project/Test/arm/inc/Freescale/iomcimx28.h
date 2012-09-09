/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Freescale MCIX28
 **
 **    Used with ICCARM and AARM.
 **
 **    (c) Copyright IAR Systems 2011
 **
 **    $Revision: 50408 $
 **
 ***************************************************************************/

#ifndef __MCIX28_H
#define __MCIX28_H


#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ICCARM/AARM
#endif


#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    MCIX28 SPECIAL FUNCTION REGISTERS
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

/* Interrupt Collector Level Acknowledge Register (HW_ICOLL_LEVELACK) */
typedef struct {
__REG32 IRQLEVELACK    : 4;
__REG32                :28;
} __hw_icoll_levelack_bits;

/* Interrupt Collector Control Register (HW_ICOLL_CTRL) */
typedef struct {
__REG32                   :16;
__REG32 IRQ_FINAL_ENABLE  : 1;
__REG32 FIQ_FINAL_ENABLE  : 1;
__REG32 ARM_RSE_MODE      : 1;
__REG32 NO_NESTING        : 1;
__REG32 BYPASS_FSM        : 1;
__REG32 VECTOR_PITCH      : 3;
__REG32                   : 6;
__REG32 CLKGATE           : 1;
__REG32 SFTRST            : 1;
} __hw_icoll_ctrl_bits;

/* Interrupt Collector Status Register (HW_ICOLL_STAT) */
typedef struct {
__REG32 VECTOR_NUMBER     : 7;
__REG32                   :25;
} __hw_icoll_stat_bits;

/* Interrupt Collector Raw Interrupt Input Register 0 (HW_ICOLL_RAW0) */
typedef struct {
__REG32 RAW_IRQ0          : 1;
__REG32 RAW_IRQ1          : 1;
__REG32 RAW_IRQ2          : 1;
__REG32 RAW_IRQ3          : 1;
__REG32 RAW_IRQ4          : 1;
__REG32 RAW_IRQ5          : 1;
__REG32 RAW_IRQ6          : 1;
__REG32 RAW_IRQ7          : 1;
__REG32 RAW_IRQ8          : 1;
__REG32 RAW_IRQ9          : 1;
__REG32 RAW_IRQ10         : 1;
__REG32 RAW_IRQ11         : 1;
__REG32 RAW_IRQ12         : 1;
__REG32 RAW_IRQ13         : 1;
__REG32 RAW_IRQ14         : 1;
__REG32 RAW_IRQ15         : 1;
__REG32 RAW_IRQ16         : 1;
__REG32 RAW_IRQ17         : 1;
__REG32 RAW_IRQ18         : 1;
__REG32 RAW_IRQ19         : 1;
__REG32 RAW_IRQ20         : 1;
__REG32 RAW_IRQ21         : 1;
__REG32 RAW_IRQ22         : 1;
__REG32 RAW_IRQ23         : 1;
__REG32 RAW_IRQ24         : 1;
__REG32 RAW_IRQ25         : 1;
__REG32 RAW_IRQ26         : 1;
__REG32 RAW_IRQ27         : 1;
__REG32 RAW_IRQ28         : 1;
__REG32 RAW_IRQ29         : 1;
__REG32 RAW_IRQ30         : 1;
__REG32 RAW_IRQ31         : 1;
} __hw_icoll_raw0_bits;

/* Interrupt Collector Raw Interrupt Input Register 1 (HW_ICOLL_RAW1) */
typedef struct {
__REG32 RAW_IRQ32         : 1;
__REG32 RAW_IRQ33         : 1;
__REG32 RAW_IRQ34         : 1;
__REG32 RAW_IRQ35         : 1;
__REG32 RAW_IRQ36         : 1;
__REG32 RAW_IRQ37         : 1;
__REG32 RAW_IRQ38         : 1;
__REG32 RAW_IRQ39         : 1;
__REG32 RAW_IRQ40         : 1;
__REG32 RAW_IRQ41         : 1;
__REG32 RAW_IRQ42         : 1;
__REG32 RAW_IRQ43         : 1;
__REG32 RAW_IRQ44         : 1;
__REG32 RAW_IRQ45         : 1;
__REG32 RAW_IRQ46         : 1;
__REG32 RAW_IRQ47         : 1;
__REG32 RAW_IRQ48         : 1;
__REG32 RAW_IRQ49         : 1;
__REG32 RAW_IRQ50         : 1;
__REG32 RAW_IRQ51         : 1;
__REG32 RAW_IRQ52         : 1;
__REG32 RAW_IRQ53         : 1;
__REG32 RAW_IRQ54         : 1;
__REG32 RAW_IRQ55         : 1;
__REG32 RAW_IRQ56         : 1;
__REG32 RAW_IRQ57         : 1;
__REG32 RAW_IRQ58         : 1;
__REG32 RAW_IRQ59         : 1;
__REG32 RAW_IRQ60         : 1;
__REG32 RAW_IRQ61         : 1;
__REG32 RAW_IRQ62         : 1;
__REG32 RAW_IRQ63         : 1;
} __hw_icoll_raw1_bits;

/* Interrupt Collector Raw Interrupt Input Register 2 (HW_ICOLL_RAW2) */
typedef struct {
__REG32 RAW_IRQ64         : 1;
__REG32 RAW_IRQ65         : 1;
__REG32 RAW_IRQ66         : 1;
__REG32 RAW_IRQ67         : 1;
__REG32 RAW_IRQ68         : 1;
__REG32 RAW_IRQ69         : 1;
__REG32 RAW_IRQ70         : 1;
__REG32 RAW_IRQ71         : 1;
__REG32 RAW_IRQ72         : 1;
__REG32 RAW_IRQ73         : 1;
__REG32 RAW_IRQ74         : 1;
__REG32 RAW_IRQ75         : 1;
__REG32 RAW_IRQ76         : 1;
__REG32 RAW_IRQ77         : 1;
__REG32 RAW_IRQ78         : 1;
__REG32 RAW_IRQ79         : 1;
__REG32 RAW_IRQ80         : 1;
__REG32 RAW_IRQ81         : 1;
__REG32 RAW_IRQ82         : 1;
__REG32 RAW_IRQ83         : 1;
__REG32 RAW_IRQ84         : 1;
__REG32 RAW_IRQ85         : 1;
__REG32 RAW_IRQ86         : 1;
__REG32 RAW_IRQ87         : 1;
__REG32 RAW_IRQ88         : 1;
__REG32 RAW_IRQ89         : 1;
__REG32 RAW_IRQ90         : 1;
__REG32 RAW_IRQ91         : 1;
__REG32 RAW_IRQ92         : 1;
__REG32 RAW_IRQ93         : 1;
__REG32 RAW_IRQ94         : 1;
__REG32 RAW_IRQ95         : 1;
} __hw_icoll_raw2_bits;

/* Interrupt Collector Raw Interrupt Input Register 3 (HW_ICOLL_RAW3) */
typedef struct {
__REG32 RAW_IRQ96         : 1;
__REG32 RAW_IRQ97         : 1;
__REG32 RAW_IRQ98         : 1;
__REG32 RAW_IRQ99         : 1;
__REG32 RAW_IRQ100        : 1;
__REG32 RAW_IRQ101        : 1;
__REG32 RAW_IRQ102        : 1;
__REG32 RAW_IRQ103        : 1;
__REG32 RAW_IRQ104        : 1;
__REG32 RAW_IRQ105        : 1;
__REG32 RAW_IRQ106        : 1;
__REG32 RAW_IRQ107        : 1;
__REG32 RAW_IRQ108        : 1;
__REG32 RAW_IRQ109        : 1;
__REG32 RAW_IRQ110        : 1;
__REG32 RAW_IRQ111        : 1;
__REG32 RAW_IRQ112        : 1;
__REG32 RAW_IRQ113        : 1;
__REG32 RAW_IRQ114        : 1;
__REG32 RAW_IRQ115        : 1;
__REG32 RAW_IRQ116        : 1;
__REG32 RAW_IRQ117        : 1;
__REG32 RAW_IRQ118        : 1;
__REG32 RAW_IRQ119        : 1;
__REG32 RAW_IRQ120        : 1;
__REG32 RAW_IRQ121        : 1;
__REG32 RAW_IRQ122        : 1;
__REG32 RAW_IRQ123        : 1;
__REG32 RAW_IRQ124        : 1;
__REG32 RAW_IRQ125        : 1;
__REG32 RAW_IRQ126        : 1;
__REG32 RAW_IRQ127        : 1;
} __hw_icoll_raw3_bits;

/* Interrupt Collector Interrupt Register 0-127 (HW_ICOLL_INTERRUPT0-127) */
typedef struct {
__REG32 PRIORITY          : 2;
__REG32 ENABLE            : 1;
__REG32 SOFTIRQ           : 1;
__REG32 ENFIQ             : 1;
__REG32                   :27;
} __hw_icoll_interrupt_bits;

/* Interrupt Collector Debug Register 0 (HW_ICOLL_DEBUG) */
typedef struct {
__REG32 VECTOR_FSM        :10;
__REG32                   : 6;
__REG32 IRQ               : 1;
__REG32 FIQ               : 1;
__REG32                   : 2;
__REG32 REQUESTS_BY_LEVEL : 4;
__REG32 LEVEL_REQUESTS    : 4;
__REG32 INSERVICE         : 4;
} __hw_icoll_debug_bits;

/* Interrupt Collector Debug Flag Register (HW_ICOLL_DBGFLAG) */
typedef struct {
__REG32 FLAG              :16;
__REG32                   :16;
} __hw_icoll_dbgflag_bits;

/* Interrupt Collector Debug Read Request Register 0 (HW_ICOLL_DBGREQUEST0) */
typedef struct {
__REG32 BITS0           : 1;
__REG32 BITS1           : 1;
__REG32 BITS2           : 1;
__REG32 BITS3           : 1;
__REG32 BITS4           : 1;
__REG32 BITS5           : 1;
__REG32 BITS6           : 1;
__REG32 BITS7           : 1;
__REG32 BITS8           : 1;
__REG32 BITS9           : 1;
__REG32 BITS10          : 1;
__REG32 BITS11          : 1;
__REG32 BITS12          : 1;
__REG32 BITS13          : 1;
__REG32 BITS14          : 1;
__REG32 BITS15          : 1;
__REG32 BITS16          : 1;
__REG32 BITS17          : 1;
__REG32 BITS18          : 1;
__REG32 BITS19          : 1;
__REG32 BITS20          : 1;
__REG32 BITS21          : 1;
__REG32 BITS22          : 1;
__REG32 BITS23          : 1;
__REG32 BITS24          : 1;
__REG32 BITS25          : 1;
__REG32 BITS26          : 1;
__REG32 BITS27          : 1;
__REG32 BITS28          : 1;
__REG32 BITS29          : 1;
__REG32 BITS30          : 1;
__REG32 BITS31          : 1;
} __hw_icoll_dbgrequest1_bits;

/* Interrupt Collector Debug Read Request Register 1 (HW_ICOLL_DBGREQUEST1) */
typedef struct {
__REG32 BITS32          : 1;
__REG32 BITS33          : 1;
__REG32 BITS34          : 1;
__REG32 BITS35          : 1;
__REG32 BITS36          : 1;
__REG32 BITS37          : 1;
__REG32 BITS38          : 1;
__REG32 BITS39          : 1;
__REG32 BITS40          : 1;
__REG32 BITS41          : 1;
__REG32 BITS42          : 1;
__REG32 BITS43          : 1;
__REG32 BITS44          : 1;
__REG32 BITS45          : 1;
__REG32 BITS46          : 1;
__REG32 BITS47          : 1;
__REG32 BITS48          : 1;
__REG32 BITS49          : 1;
__REG32 BITS50          : 1;
__REG32 BITS51          : 1;
__REG32 BITS52          : 1;
__REG32 BITS53          : 1;
__REG32 BITS54          : 1;
__REG32 BITS55          : 1;
__REG32 BITS56          : 1;
__REG32 BITS57          : 1;
__REG32 BITS58          : 1;
__REG32 BITS59          : 1;
__REG32 BITS60          : 1;
__REG32 BITS61          : 1;
__REG32 BITS62          : 1;
__REG32 BITS63          : 1;
} __hw_icoll_dbgrequest0_bits;

/* Interrupt Collector Debug Read Request Register 2 (HW_ICOLL_DBGREQUEST2) */

typedef struct {
__REG32 BITS64          : 1;
__REG32 BITS65          : 1;
__REG32 BITS66          : 1;
__REG32 BITS67          : 1;
__REG32 BITS68          : 1;
__REG32 BITS69          : 1;
__REG32 BITS70          : 1;
__REG32 BITS71          : 1;
__REG32 BITS72          : 1;
__REG32 BITS73          : 1;
__REG32 BITS74          : 1;
__REG32 BITS75          : 1;
__REG32 BITS76          : 1;
__REG32 BITS77          : 1;
__REG32 BITS78          : 1;
__REG32 BITS79          : 1;
__REG32 BITS80          : 1;
__REG32 BITS81          : 1;
__REG32 BITS82          : 1;
__REG32 BITS83          : 1;
__REG32 BITS84          : 1;
__REG32 BITS85          : 1;
__REG32 BITS86          : 1;
__REG32 BITS87          : 1;
__REG32 BITS88          : 1;
__REG32 BITS89          : 1;
__REG32 BITS90          : 1;
__REG32 BITS91          : 1;
__REG32 BITS92          : 1;
__REG32 BITS93          : 1;
__REG32 BITS94          : 1;
__REG32 BITS95          : 1;
} __hw_icoll_dbgrequest2_bits;

/* Interrupt Collector Debug Read Request Register 3 (HW_ICOLL_DBGREQUEST3) */
typedef struct {
__REG32 BITS96          : 1;
__REG32 BITS97          : 1;
__REG32 BITS98          : 1;
__REG32 BITS99          : 1;
__REG32 BITS100         : 1;
__REG32 BITS101         : 1;
__REG32 BITS102         : 1;
__REG32 BITS103         : 1;
__REG32 BITS104         : 1;
__REG32 BITS105         : 1;
__REG32 BITS106         : 1;
__REG32 BITS107         : 1;
__REG32 BITS108         : 1;
__REG32 BITS109         : 1;
__REG32 BITS110         : 1;
__REG32 BITS111         : 1;
__REG32 BITS112         : 1;
__REG32 BITS113         : 1;
__REG32 BITS114         : 1;
__REG32 BITS115         : 1;
__REG32 BITS116         : 1;
__REG32 BITS117         : 1;
__REG32 BITS118         : 1;
__REG32 BITS119         : 1;
__REG32 BITS120         : 1;
__REG32 BITS121         : 1;
__REG32 BITS122         : 1;
__REG32 BITS123         : 1;
__REG32 BITS124         : 1;
__REG32 BITS125         : 1;
__REG32 BITS126         : 1;
__REG32 BITS127         : 1;
} __hw_icoll_dbgrequest3_bits;

/* Interrupt Collector Version Register (HW_ICOLL_VERSION) */
typedef struct {
__REG32 STEP            :16;
__REG32 MINOR           : 8;
__REG32 MAJOR           : 8;
} __hw_icoll_version_bits;

/* AHB to APBH Bridge Control and Status Register 0 (HW_APBH_CTRL0) */
typedef struct {
__REG32 CLKGATE_CHANNEL :16;
__REG32                 :12;
__REG32 APB_BURST_EN    : 1;
__REG32 AHB_BURST8_EN   : 1;
__REG32 CLKGATE         : 1;
__REG32 SFTRST          : 1;
} __hw_apbh_ctrl0_bits;

/* AHB to APBH Bridge Control and Status Register 1 (HW_APBH_CTRL1) */
typedef struct {
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
} __hw_apbh_ctrl1_bits;

/* AHB to APBH Bridge Control and Status Register 2 (HW_APBH_CTRL2) */
typedef struct {
__REG32 CH0_ERROR_IRQ         : 1;
__REG32 CH1_ERROR_IRQ         : 1;
__REG32 CH2_ERROR_IRQ         : 1;
__REG32 CH3_ERROR_IRQ         : 1;
__REG32 CH4_ERROR_IRQ         : 1;
__REG32 CH5_ERROR_IRQ         : 1;
__REG32 CH6_ERROR_IRQ         : 1;
__REG32 CH7_ERROR_IRQ         : 1;
__REG32 CH8_ERROR_IRQ         : 1;
__REG32 CH9_ERROR_IRQ         : 1;
__REG32 CH10_ERROR_IRQ        : 1;
__REG32 CH11_ERROR_IRQ        : 1;
__REG32 CH12_ERROR_IRQ        : 1;
__REG32 CH13_ERROR_IRQ        : 1;
__REG32 CH14_ERROR_IRQ        : 1;
__REG32 CH15_ERROR_IRQ        : 1;
__REG32 CH0_ERROR_STATUS      : 1;
__REG32 CH1_ERROR_STATUS      : 1;
__REG32 CH2_ERROR_STATUS      : 1;
__REG32 CH3_ERROR_STATUS      : 1;
__REG32 CH4_ERROR_STATUS      : 1;
__REG32 CH5_ERROR_STATUS      : 1;
__REG32 CH6_ERROR_STATUS      : 1;
__REG32 CH7_ERROR_STATUS      : 1;
__REG32 CH8_ERROR_STATUS      : 1;
__REG32 CH9_ERROR_STATUS      : 1;
__REG32 CH10_ERROR_STATUS     : 1;
__REG32 CH11_ERROR_STATUS     : 1;
__REG32 CH12_ERROR_STATUS     : 1;
__REG32 CH13_ERROR_STATUS     : 1;
__REG32 CH14_ERROR_STATUS     : 1;
__REG32 CH15_ERROR_STATUS     : 1;
} __hw_apbh_ctrl2_bits;

/* AHB to APBH Bridge Channel Register (HW_APBH_CHANNEL_CTRL) */
typedef struct {
__REG32 CH0_FREEZE_CHANNEL    : 1;
__REG32 CH1_FREEZE_CHANNEL    : 1;
__REG32 CH2_FREEZE_CHANNEL    : 1;
__REG32 CH3_FREEZE_CHANNEL    : 1;
__REG32 CH4_FREEZE_CHANNEL    : 1;
__REG32 CH5_FREEZE_CHANNEL    : 1;
__REG32 CH6_FREEZE_CHANNEL    : 1;
__REG32 CH7_FREEZE_CHANNEL    : 1;
__REG32 CH8_FREEZE_CHANNEL    : 1;
__REG32 CH9_FREEZE_CHANNEL    : 1;
__REG32 CH10_FREEZE_CHANNEL   : 1;
__REG32 CH11_FREEZE_CHANNEL   : 1;
__REG32 CH12_FREEZE_CHANNEL   : 1;
__REG32 CH13_FREEZE_CHANNEL   : 1;
__REG32 CH14_FREEZE_CHANNEL   : 1;
__REG32 CH15_FREEZE_CHANNEL   : 1;
__REG32 CH0_RESET_CHANNEL     : 1;
__REG32 CH1_RESET_CHANNEL     : 1;
__REG32 CH2_RESET_CHANNEL     : 1;
__REG32 CH3_RESET_CHANNEL     : 1;
__REG32 CH4_RESET_CHANNEL     : 1;
__REG32 CH5_RESET_CHANNEL     : 1;
__REG32 CH6_RESET_CHANNEL     : 1;
__REG32 CH7_RESET_CHANNEL     : 1;
__REG32 CH8_RESET_CHANNEL     : 1;
__REG32 CH9_RESET_CHANNEL     : 1;
__REG32 CH10_RESET_CHANNEL    : 1;
__REG32 CH11_RESET_CHANNEL    : 1;
__REG32 CH12_RESET_CHANNEL    : 1;
__REG32 CH13_RESET_CHANNEL    : 1;
__REG32 CH14_RESET_CHANNEL    : 1;
__REG32 CH15_RESET_CHANNEL    : 1;
} __hw_apbh_channel_ctrl_bits;

/* AHB to APBH DMA Device Assignment Register (HW_APBH_DEVSEL) */
typedef struct {
__REG32 CH0                   : 2;
__REG32 CH1                   : 2;
__REG32 CH2                   : 2;
__REG32 CH3                   : 2;
__REG32 CH4                   : 2;
__REG32 CH5                   : 2;
__REG32 CH6                   : 2;
__REG32 CH7                   : 2;
__REG32 CH8                   : 2;
__REG32 CH9                   : 2;
__REG32 CH10                  : 2;
__REG32 CH11                  : 2;
__REG32 CH12                  : 2;
__REG32 CH13                  : 2;
__REG32 CH14                  : 2;
__REG32 CH15                  : 2;
} __hw_apbh_devsel_bits;

/* AHB to APBH DMA burst size (HW_APBH_DMA_BURST_SIZE) */
typedef struct {
__REG32 CH0                   : 2;
__REG32 CH1                   : 2;
__REG32 CH2                   : 2;
__REG32 CH3                   : 2;
__REG32 CH4                   : 2;
__REG32 CH5                   : 2;
__REG32 CH6                   : 2;
__REG32 CH7                   : 2;
__REG32 CH8                   : 2;
__REG32 CH9                   : 2;
__REG32 CH10                  : 2;
__REG32 CH11                  : 2;
__REG32 CH12                  : 2;
__REG32 CH13                  : 2;
__REG32 CH14                  : 2;
__REG32 CH15                  : 2;
} __hw_apbh_dma_burst_size_bits;

/* AHB to APBH DMA Debug Register (HW_APBH_DEBUG) */
typedef struct {
__REG32 GPMI_ONE_FIFO         : 1;
__REG32                       :31;
} __hw_apbh_debug_bits;

/* APBH DMA Channel 0 Command Register (HW_APBH_CH0_CMD) */
typedef struct {
__REG32 COMMAND               : 2;
__REG32 CHAIN                 : 1;
__REG32 IRQONCMPLT            : 1;
__REG32 NANDLOCK              : 1;
__REG32 NANDWAIT4READY        : 1;
__REG32 SEMAPHORE             : 1;
__REG32 WAIT4ENDCMD           : 1;
__REG32 HALTONTERMINATE       : 1;
__REG32                       : 3;
__REG32 CMDWORDS              : 4;
__REG32 XFER_COUNT            :16;
} __hw_apbh_ch_cmd_bits;

/* APBH DMA Channel 0 Semaphore Register (HW_APBH_CH0_SEMA) */
typedef struct {
__REG32 INCREMENT_SEMA        : 8;
__REG32                       : 8;
__REG32 PHORE                 : 8;
__REG32                       : 8;
} __hw_apbh_ch_sema_bits;

/* AHB to APBH DMA Channel 0 Debug Information (HW_APBH_CH0_DEBUG1) */
typedef struct {
__REG32 STATEMACHINE          : 5;
__REG32                       :15;
__REG32 WR_FIFO_FULL          : 1;
__REG32 WR_FIFO_EMPTY         : 1;
__REG32 RD_FIFO_FULL          : 1;
__REG32 RD_FIFO_EMPTY         : 1;
__REG32 NEXTCMDADDRVALID      : 1;
__REG32 LOCK                  : 1;
__REG32 READY                 : 1;
__REG32 SENSE                 : 1;
__REG32 END                   : 1;
__REG32 KICK                  : 1;
__REG32 BURST                 : 1;
__REG32 REQ                   : 1;
} __hw_apbh_ch_debug1_bits;

/* APBH Bridge Version Register (HW_APBH_VERSION) */
typedef struct {
__REG32 STEP                  :16;
__REG32 MINOR                 : 8;
__REG32 MAJOR                 : 8;
} __hw_apbh_version_bits;

/* AHB to APBX Bridge Control Register 0 (HW_APBX_CTRL0) */
typedef struct {
__REG32                       :30;
__REG32 CLKGATE               : 1;
__REG32 SFTRST                : 1;
} __hw_apbx_ctrl0_bits;

/* AHB to APBX Bridge Control Register 1 (HW_APBX_CTRL1) */
typedef struct {
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
} __hw_apbx_ctrl1_bits;

/* AHB to APBX Bridge Control Register 2 (HW_APBX_CTRL2) */
typedef struct {
__REG32 CH0_ERROR_IRQ         : 1;
__REG32 CH1_ERROR_IRQ         : 1;
__REG32 CH2_ERROR_IRQ         : 1;
__REG32 CH3_ERROR_IRQ         : 1;
__REG32 CH4_ERROR_IRQ         : 1;
__REG32 CH5_ERROR_IRQ         : 1;
__REG32 CH6_ERROR_IRQ         : 1;
__REG32 CH7_ERROR_IRQ         : 1;
__REG32 CH8_ERROR_IRQ         : 1;
__REG32 CH9_ERROR_IRQ         : 1;
__REG32 CH10_ERROR_IRQ        : 1;
__REG32 CH11_ERROR_IRQ        : 1;
__REG32 CH12_ERROR_IRQ        : 1;
__REG32 CH13_ERROR_IRQ        : 1;
__REG32 CH14_ERROR_IRQ        : 1;
__REG32 CH15_ERROR_IRQ        : 1;
__REG32 CH0_ERROR_STATUS      : 1;
__REG32 CH1_ERROR_STATUS      : 1;
__REG32 CH2_ERROR_STATUS      : 1;
__REG32 CH3_ERROR_STATUS      : 1;
__REG32 CH4_ERROR_STATUS      : 1;
__REG32 CH5_ERROR_STATUS      : 1;
__REG32 CH6_ERROR_STATUS      : 1;
__REG32 CH7_ERROR_STATUS      : 1;
__REG32 CH8_ERROR_STATUS      : 1;
__REG32 CH9_ERROR_STATUS      : 1;
__REG32 CH10_ERROR_STATUS     : 1;
__REG32 CH11_ERROR_STATUS     : 1;
__REG32 CH12_ERROR_STATUS     : 1;
__REG32 CH13_ERROR_STATUS     : 1;
__REG32 CH14_ERROR_STATUS     : 1;
__REG32 CH15_ERROR_STATUS     : 1;
} __hw_apbx_ctrl2_bits;

/* AHB to APBX Bridge Channel Register (HW_APBX_CHANNEL_CTRL) */
typedef struct {
__REG32 CH0_FREEZE_CHANNEL    : 1;
__REG32 CH1_FREEZE_CHANNEL    : 1;
__REG32 CH2_FREEZE_CHANNEL    : 1;
__REG32 CH3_FREEZE_CHANNEL    : 1;
__REG32 CH4_FREEZE_CHANNEL    : 1;
__REG32 CH5_FREEZE_CHANNEL    : 1;
__REG32 CH6_FREEZE_CHANNEL    : 1;
__REG32 CH7_FREEZE_CHANNEL    : 1;
__REG32 CH8_FREEZE_CHANNEL    : 1;
__REG32 CH9_FREEZE_CHANNEL    : 1;
__REG32 CH10_FREEZE_CHANNEL   : 1;
__REG32 CH11_FREEZE_CHANNEL   : 1;
__REG32 CH12_FREEZE_CHANNEL   : 1;
__REG32 CH13_FREEZE_CHANNEL   : 1;
__REG32 CH14_FREEZE_CHANNEL   : 1;
__REG32 CH15_FREEZE_CHANNEL   : 1;
__REG32 CH0_RESET_CHANNEL     : 1;
__REG32 CH1_RESET_CHANNEL     : 1;
__REG32 CH2_RESET_CHANNEL     : 1;
__REG32 CH3_RESET_CHANNEL     : 1;
__REG32 CH4_RESET_CHANNEL     : 1;
__REG32 CH5_RESET_CHANNEL     : 1;
__REG32 CH6_RESET_CHANNEL     : 1;
__REG32 CH7_RESET_CHANNEL     : 1;
__REG32 CH8_RESET_CHANNEL     : 1;
__REG32 CH9_RESET_CHANNEL     : 1;
__REG32 CH10_RESET_CHANNEL    : 1;
__REG32 CH11_RESET_CHANNEL    : 1;
__REG32 CH12_RESET_CHANNEL    : 1;
__REG32 CH13_RESET_CHANNEL    : 1;
__REG32 CH14_RESET_CHANNEL    : 1;
__REG32 CH15_RESET_CHANNEL    : 1;
} __hw_apbx_channel_ctrl_bits;

/* AHB to APBX Bridge Channel Register (HW_APBX_CHANNEL_CTRL) */
typedef struct {
__REG32 CH0                   : 2;
__REG32 CH1                   : 2;
__REG32 CH2                   : 2;
__REG32 CH3                   : 2;
__REG32 CH4                   : 2;
__REG32 CH5                   : 2;
__REG32 CH6                   : 2;
__REG32 CH7                   : 2;
__REG32 CH8                   : 2;
__REG32 CH9                   : 2;
__REG32 CH10                  : 2;
__REG32 CH11                  : 2;
__REG32 CH12                  : 2;
__REG32 CH13                  : 2;
__REG32 CH14                  : 2;
__REG32 CH15                  : 2;
} __hw_apbx_devsel_bits;

/* APBX DMA Channel n Command Register (HW_APBX_CHn_CMD) */
typedef struct {
__REG32 COMMAND               : 2;
__REG32 CHAIN                 : 1;
__REG32 IRQONCMPLT            : 1;
__REG32                       : 2;
__REG32 SEMAPHORE             : 1;
__REG32 WAIT4ENDCMD           : 1;
__REG32                       : 4;
__REG32 CMDWORDS              : 4;
__REG32 XFER_COUNT            :16;
} __hw_apbx_ch_cmd_bits;

/* APBX DMA Channel n Semaphore Register (HW_APBX_CHn_SEMA) */
typedef struct {
__REG32 INCREMENT_SEMA        : 8;
__REG32                       : 8;
__REG32 PHORE                 : 8;
__REG32                       : 8;
} __hw_apbx_ch_sema_bits;

/* AHB to APBX DMA Channel n Debug Information (HW_APBX_CHn_DEBUG1) */
typedef struct {
__REG32 STATEMACHINE          : 5;
__REG32                       :15;
__REG32 WR_FIFO_FULL          : 1;
__REG32 WR_FIFO_EMPTY         : 1;
__REG32 RD_FIFO_FULL          : 1;
__REG32 RD_FIFO_EMPTY         : 1;
__REG32 NEXTCMDADDRVALID      : 1;
__REG32                       : 3;
__REG32 END                   : 1;
__REG32 KICK                  : 1;
__REG32 BURST                 : 1;
__REG32 REQ                   : 1;
} __hw_apbx_ch_debug1_bits;

/* APBX Bridge Version Register (HW_APBX_VERSION) */
typedef struct {
__REG32 STEP                  :16;
__REG32 MINOR                 : 8;
__REG32 MAJOR                 : 8;
} __hw_apbx_version_bits;

/* PINCTRL Block Control Register (HW_PINCTRL_CTRL) */
typedef struct {
__REG32 IRQOUT0               : 1;
__REG32 IRQOUT1               : 1;
__REG32 IRQOUT2               : 1;
__REG32 IRQOUT3               : 1;
__REG32 IRQOUT4               : 1;
__REG32                       :15;
__REG32 PRESENT0              : 1;
__REG32 PRESENT1              : 1;
__REG32 PRESENT2              : 1;
__REG32 PRESENT3              : 1;
__REG32 PRESENT4              : 1;
__REG32                       : 5;
__REG32 CLKGATE               : 1;
__REG32 SFTRST                : 1;
} __hw_pinctrl_ctrl_bits;

/* PINCTRL Pin Mux Select Register 0 (HW_PINCTRL_MUXSEL0) */
typedef struct {
__REG32 BANK0_PIN00           : 2;
__REG32 BANK0_PIN01           : 2;
__REG32 BANK0_PIN02           : 2;
__REG32 BANK0_PIN03           : 2;
__REG32 BANK0_PIN04           : 2;
__REG32 BANK0_PIN05           : 2;
__REG32 BANK0_PIN06           : 2;
__REG32 BANK0_PIN07           : 2;
__REG32                       :16;
} __hw_pinctrl_muxsel0_bits;

/* PINCTRL Pin Mux Select Register 1 (HW_PINCTRL_MUXSEL1) */
typedef struct {
__REG32 BANK0_PIN16           : 2;
__REG32 BANK0_PIN17           : 2;
__REG32 BANK0_PIN18           : 2;
__REG32 BANK0_PIN19           : 2;
__REG32 BANK0_PIN20           : 2;
__REG32 BANK0_PIN21           : 2;
__REG32 BANK0_PIN22           : 2;
__REG32 BANK0_PIN23           : 2;
__REG32 BANK0_PIN24           : 2;
__REG32 BANK0_PIN25           : 2;
__REG32 BANK0_PIN26           : 2;
__REG32 BANK0_PIN27           : 2;
__REG32 BANK0_PIN28           : 2;
__REG32                       : 6;
} __hw_pinctrl_muxsel1_bits;

/* PINCTRL Pin Mux Select Register 2 (HW_PINCTRL_MUXSEL2) */
typedef struct {
__REG32 BANK1_PIN00           : 2;
__REG32 BANK1_PIN01           : 2;
__REG32 BANK1_PIN02           : 2;
__REG32 BANK1_PIN03           : 2;
__REG32 BANK1_PIN04           : 2;
__REG32 BANK1_PIN05           : 2;
__REG32 BANK1_PIN06           : 2;
__REG32 BANK1_PIN07           : 2;
__REG32 BANK1_PIN08           : 2;
__REG32 BANK1_PIN09           : 2;
__REG32 BANK1_PIN10           : 2;
__REG32 BANK1_PIN11           : 2;
__REG32 BANK1_PIN12           : 2;
__REG32 BANK1_PIN13           : 2;
__REG32 BANK1_PIN14           : 2;
__REG32 BANK1_PIN15           : 2;
} __hw_pinctrl_muxsel2_bits;

/* PINCTRL Pin Mux Select Register 3 (HW_PINCTRL_MUXSEL3) */
typedef struct {
__REG32 BANK1_PIN16           : 2;
__REG32 BANK1_PIN17           : 2;
__REG32 BANK1_PIN18           : 2;
__REG32 BANK1_PIN19           : 2;
__REG32 BANK1_PIN20           : 2;
__REG32 BANK1_PIN21           : 2;
__REG32 BANK1_PIN22           : 2;
__REG32 BANK1_PIN23           : 2;
__REG32 BANK1_PIN24           : 2;
__REG32 BANK1_PIN25           : 2;
__REG32 BANK1_PIN26           : 2;
__REG32 BANK1_PIN27           : 2;
__REG32 BANK1_PIN28           : 2;
__REG32 BANK1_PIN29           : 2;
__REG32 BANK1_PIN30           : 2;
__REG32 BANK1_PIN31           : 2;
} __hw_pinctrl_muxsel3_bits;

/* PINCTRL Pin Mux Select Register 4 (HW_PINCTRL_MUXSEL4) */
typedef struct {
__REG32 BANK2_PIN00           : 2;
__REG32 BANK2_PIN01           : 2;
__REG32 BANK2_PIN02           : 2;
__REG32 BANK2_PIN03           : 2;
__REG32 BANK2_PIN04           : 2;
__REG32 BANK2_PIN05           : 2;
__REG32 BANK2_PIN06           : 2;
__REG32 BANK2_PIN07           : 2;
__REG32 BANK2_PIN08           : 2;
__REG32 BANK2_PIN09           : 2;
__REG32 BANK2_PIN10           : 2;
__REG32                       : 2;
__REG32 BANK2_PIN12           : 2;
__REG32 BANK2_PIN13           : 2;
__REG32 BANK2_PIN14           : 2;
__REG32 BANK2_PIN15           : 2;
} __hw_pinctrl_muxsel4_bits;

/* PINCTRL Pin Mux Select Register 5 (HW_PINCTRL_MUXSEL5) */
typedef struct {
__REG32 BANK2_PIN16           : 2;
__REG32 BANK2_PIN17           : 2;
__REG32 BANK2_PIN18           : 2;
__REG32 BANK2_PIN19           : 2;
__REG32 BANK2_PIN20           : 2;
__REG32 BANK2_PIN21           : 2;
__REG32                       : 4;
__REG32 BANK2_PIN24           : 2;
__REG32 BANK2_PIN25           : 2;
__REG32 BANK2_PIN26           : 2;
__REG32 BANK2_PIN27           : 2;
__REG32                       : 8;
} __hw_pinctrl_muxsel5_bits;

/* PINCTRL Pin Mux Select Register 6 (HW_PINCTRL_MUXSEL6) */
typedef struct {
__REG32 BANK3_PIN00           : 2;
__REG32 BANK3_PIN01           : 2;
__REG32 BANK3_PIN02           : 2;
__REG32 BANK3_PIN03           : 2;
__REG32 BANK3_PIN04           : 2;
__REG32 BANK3_PIN05           : 2;
__REG32 BANK3_PIN06           : 2;
__REG32 BANK3_PIN07           : 2;
__REG32 BANK3_PIN08           : 2;
__REG32 BANK3_PIN09           : 2;
__REG32 BANK3_PIN10           : 2;
__REG32 BANK3_PIN11           : 2;
__REG32 BANK3_PIN12           : 2;
__REG32 BANK3_PIN13           : 2;
__REG32 BANK3_PIN14           : 2;
__REG32 BANK3_PIN15           : 2;
} __hw_pinctrl_muxsel6_bits;

/* PINCTRL Pin Mux Select Register 7 (HW_PINCTRL_MUXSEL7) */
typedef struct {
__REG32 BANK3_PIN16           : 2;
__REG32 BANK3_PIN17           : 2;
__REG32 BANK3_PIN18           : 2;
__REG32                       : 2;
__REG32 BANK3_PIN20           : 2;
__REG32 BANK3_PIN21           : 2;
__REG32 BANK3_PIN22           : 2;
__REG32 BANK3_PIN23           : 2;
__REG32 BANK3_PIN24           : 2;
__REG32 BANK3_PIN25           : 2;
__REG32 BANK3_PIN26           : 2;
__REG32 BANK3_PIN27           : 2;
__REG32 BANK3_PIN28           : 2;
__REG32 BANK3_PIN29           : 2;
__REG32 BANK3_PIN30           : 2;
__REG32                       : 2;
} __hw_pinctrl_muxsel7_bits;

/* PINCTRL Pin Mux Select Register 8 (HW_PINCTRL_MUXSEL8) */
typedef struct {
__REG32 BANK4_PIN00           : 2;
__REG32 BANK4_PIN01           : 2;
__REG32 BANK4_PIN02           : 2;
__REG32 BANK4_PIN03           : 2;
__REG32 BANK4_PIN04           : 2;
__REG32 BANK4_PIN05           : 2;
__REG32 BANK4_PIN06           : 2;
__REG32 BANK4_PIN07           : 2;
__REG32 BANK4_PIN08           : 2;
__REG32 BANK4_PIN09           : 2;
__REG32 BANK4_PIN10           : 2;
__REG32 BANK4_PIN11           : 2;
__REG32 BANK4_PIN12           : 2;
__REG32 BANK4_PIN13           : 2;
__REG32 BANK4_PIN14           : 2;
__REG32 BANK4_PIN15           : 2;
} __hw_pinctrl_muxsel8_bits;

/* PINCTRL Pin Mux Select Register 9 (HW_PINCTRL_MUXSEL9) */
typedef struct {
__REG32 BANK4_PIN16           : 2;
__REG32                       : 6;
__REG32 BANK4_PIN20           : 2;
__REG32                       :22;
} __hw_pinctrl_muxsel9_bits;

/* PINCTRL Pin Mux Select Register 10 (HW_PINCTRL_MUXSEL10) */
typedef struct {
__REG32 BANK5_PIN00           : 2;
__REG32 BANK5_PIN01           : 2;
__REG32 BANK5_PIN02           : 2;
__REG32 BANK5_PIN03           : 2;
__REG32 BANK5_PIN04           : 2;
__REG32 BANK5_PIN05           : 2;
__REG32 BANK5_PIN06           : 2;
__REG32 BANK5_PIN07           : 2;
__REG32 BANK5_PIN08           : 2;
__REG32 BANK5_PIN09           : 2;
__REG32 BANK5_PIN10           : 2;
__REG32 BANK5_PIN11           : 2;
__REG32 BANK5_PIN12           : 2;
__REG32 BANK5_PIN13           : 2;
__REG32 BANK5_PIN14           : 2;
__REG32 BANK5_PIN15           : 2;
} __hw_pinctrl_muxsel10_bits;

/* PINCTRL Pin Mux Select Register 11 (HW_PINCTRL_MUXSEL11) */
typedef struct {
__REG32 BANK5_PIN16           : 2;
__REG32 BANK5_PIN17           : 2;
__REG32 BANK5_PIN18           : 2;
__REG32 BANK5_PIN19           : 2;
__REG32 BANK5_PIN20           : 2;
__REG32 BANK5_PIN21           : 2;
__REG32 BANK5_PIN22           : 2;
__REG32 BANK5_PIN23           : 2;
__REG32                       : 4;
__REG32 BANK5_PIN26           : 2;
__REG32                       :10;
} __hw_pinctrl_muxsel11_bits;

/* PINCTRL Pin Mux Select Register 12 (HW_PINCTRL_MUXSEL12) */
typedef struct {
__REG32 BANK6_PIN00           : 2;
__REG32 BANK6_PIN01           : 2;
__REG32 BANK6_PIN02           : 2;
__REG32 BANK6_PIN03           : 2;
__REG32 BANK6_PIN04           : 2;
__REG32 BANK6_PIN05           : 2;
__REG32 BANK6_PIN06           : 2;
__REG32 BANK6_PIN07           : 2;
__REG32 BANK6_PIN08           : 2;
__REG32 BANK6_PIN09           : 2;
__REG32 BANK6_PIN10           : 2;
__REG32 BANK6_PIN11           : 2;
__REG32 BANK6_PIN12           : 2;
__REG32 BANK6_PIN13           : 2;
__REG32 BANK6_PIN14           : 2;
__REG32                       : 2;
} __hw_pinctrl_muxsel12_bits;

/* PINCTRL Pin Mux Select Register 13 (HW_PINCTRL_MUXSEL13) */
typedef struct {
__REG32 BANK6_PIN16           : 2;
__REG32 BANK6_PIN17           : 2;
__REG32 BANK6_PIN18           : 2;
__REG32 BANK6_PIN19           : 2;
__REG32 BANK6_PIN20           : 2;
__REG32 BANK6_PIN21           : 2;
__REG32 BANK6_PIN22           : 2;
__REG32 BANK6_PIN23           : 2;
__REG32 BANK6_PIN24           : 2;
__REG32                       :14;
} __hw_pinctrl_muxsel13_bits;

/* PINCTRL Drive Strength and Voltage Register 0 (HW_PINCTRL_DRIVE0) */
typedef struct {
__REG32 BANK0_PIN00_MA        : 2;
__REG32 BANK0_PIN00_V         : 1;
__REG32                       : 1;
__REG32 BANK0_PIN01_MA        : 2;
__REG32 BANK0_PIN01_V         : 1;
__REG32                       : 1;
__REG32 BANK0_PIN02_MA        : 2;
__REG32 BANK0_PIN02_V         : 1;
__REG32                       : 1;
__REG32 BANK0_PIN03_MA        : 2;
__REG32 BANK0_PIN03_V         : 1;
__REG32                       : 1;
__REG32 BANK0_PIN04_MA        : 2;
__REG32 BANK0_PIN04_V         : 1;
__REG32                       : 1;
__REG32 BANK0_PIN05_MA        : 2;
__REG32 BANK0_PIN05_V         : 1;
__REG32                       : 1;
__REG32 BANK0_PIN06_MA        : 2;
__REG32 BANK0_PIN06_V         : 1;
__REG32                       : 1;
__REG32 BANK0_PIN07_MA        : 2;
__REG32 BANK0_PIN07_V         : 1;
__REG32                       : 1;
} __hw_pinctrl_drive0_bits;

/* PINCTRL Drive Strength and Voltage Register 2 (HW_PINCTRL_DRIVE2) */
typedef struct {
__REG32 BANK0_PIN16_MA        : 2;
__REG32 BANK0_PIN16_V         : 1;
__REG32                       : 1;
__REG32 BANK0_PIN17_MA        : 2;
__REG32 BANK0_PIN17_V         : 1;
__REG32                       : 1;
__REG32 BANK0_PIN18_MA        : 2;
__REG32 BANK0_PIN18_V         : 1;
__REG32                       : 1;
__REG32 BANK0_PIN19_MA        : 2;
__REG32 BANK0_PIN19_V         : 1;
__REG32                       : 1;
__REG32 BANK0_PIN20_MA        : 2;
__REG32 BANK0_PIN20_V         : 1;
__REG32                       : 1;
__REG32 BANK0_PIN21_MA        : 2;
__REG32 BANK0_PIN21_V         : 1;
__REG32                       : 1;
__REG32 BANK0_PIN22_MA        : 2;
__REG32 BANK0_PIN22_V         : 1;
__REG32                       : 1;
__REG32 BANK0_PIN23_MA        : 2;
__REG32 BANK0_PIN23_V         : 1;
__REG32                       : 1;
} __hw_pinctrl_drive2_bits;

/* PINCTRL Drive Strength and Voltage Register 3 (HW_PINCTRL_DRIVE3) */
typedef struct {
__REG32 BANK0_PIN24_MA        : 2;
__REG32 BANK0_PIN24_V         : 1;
__REG32                       : 1;
__REG32 BANK0_PIN25_MA        : 2;
__REG32 BANK0_PIN25_V         : 1;
__REG32                       : 1;
__REG32 BANK0_PIN26_MA        : 2;
__REG32 BANK0_PIN26_V         : 1;
__REG32                       : 1;
__REG32 BANK0_PIN27_MA        : 2;
__REG32 BANK0_PIN27_V         : 1;
__REG32                       : 1;
__REG32 BANK0_PIN28_MA        : 2;
__REG32 BANK0_PIN28_V         : 1;
__REG32                       :13;
} __hw_pinctrl_drive3_bits;

/* PINCTRL Drive Strength and Voltage Register 4 (HW_PINCTRL_DRIVE4) */
typedef struct {
__REG32 BANK1_PIN00_MA        : 2;
__REG32 BANK1_PIN00_V         : 1;
__REG32                       : 1;
__REG32 BANK1_PIN01_MA        : 2;
__REG32 BANK1_PIN01_V         : 1;
__REG32                       : 1;
__REG32 BANK1_PIN02_MA        : 2;
__REG32 BANK1_PIN02_V         : 1;
__REG32                       : 1;
__REG32 BANK1_PIN03_MA        : 2;
__REG32 BANK1_PIN03_V         : 1;
__REG32                       : 1;
__REG32 BANK1_PIN04_MA        : 2;
__REG32 BANK1_PIN04_V         : 1;
__REG32                       : 1;
__REG32 BANK1_PIN05_MA        : 2;
__REG32 BANK1_PIN05_V         : 1;
__REG32                       : 1;
__REG32 BANK1_PIN06_MA        : 2;
__REG32 BANK1_PIN06_V         : 1;
__REG32                       : 1;
__REG32 BANK1_PIN07_MA        : 2;
__REG32 BANK1_PIN07_V         : 1;
__REG32                       : 1;
} __hw_pinctrl_drive4_bits;

/* PINCTRL Drive Strength and Voltage Register 5 (HW_PINCTRL_DRIVE5) */
typedef struct {
__REG32 BANK1_PIN08_MA        : 2;
__REG32 BANK1_PIN08_V         : 1;
__REG32                       : 1;
__REG32 BANK1_PIN09_MA        : 2;
__REG32 BANK1_PIN09_V         : 1;
__REG32                       : 1;
__REG32 BANK1_PIN10_MA        : 2;
__REG32 BANK1_PIN10_V         : 1;
__REG32                       : 1;
__REG32 BANK1_PIN11_MA        : 2;
__REG32 BANK1_PIN11_V         : 1;
__REG32                       : 1;
__REG32 BANK1_PIN12_MA        : 2;
__REG32 BANK1_PIN12_V         : 1;
__REG32                       : 1;
__REG32 BANK1_PIN13_MA        : 2;
__REG32 BANK1_PIN13_V         : 1;
__REG32                       : 1;
__REG32 BANK1_PIN14_MA        : 2;
__REG32 BANK1_PIN14_V         : 1;
__REG32                       : 1;
__REG32 BANK1_PIN15_MA        : 2;
__REG32 BANK1_PIN15_V         : 1;
__REG32                       : 1;
} __hw_pinctrl_drive5_bits;

/* PINCTRL Drive Strength and Voltage Register 6 (HW_PINCTRL_DRIVE6) */
typedef struct {
__REG32 BANK1_PIN16_MA        : 2;
__REG32 BANK1_PIN16_V         : 1;
__REG32                       : 1;
__REG32 BANK1_PIN17_MA        : 2;
__REG32 BANK1_PIN17_V         : 1;
__REG32                       : 1;
__REG32 BANK1_PIN18_MA        : 2;
__REG32 BANK1_PIN18_V         : 1;
__REG32                       : 1;
__REG32 BANK1_PIN19_MA        : 2;
__REG32 BANK1_PIN19_V         : 1;
__REG32                       : 1;
__REG32 BANK1_PIN20_MA        : 2;
__REG32 BANK1_PIN20_V         : 1;
__REG32                       : 1;
__REG32 BANK1_PIN21_MA        : 2;
__REG32 BANK1_PIN21_V         : 1;
__REG32                       : 1;
__REG32 BANK1_PIN22_MA        : 2;
__REG32 BANK1_PIN22_V         : 1;
__REG32                       : 1;
__REG32 BANK1_PIN23_MA        : 2;
__REG32 BANK1_PIN23_V         : 1;
__REG32                       : 1;
} __hw_pinctrl_drive6_bits;

/* PINCTRL Drive Strength and Voltage Register 7 (HW_PINCTRL_DRIVE7) */
typedef struct {
__REG32 BANK1_PIN24_MA        : 2;
__REG32 BANK1_PIN24_V         : 1;
__REG32                       : 1;
__REG32 BANK1_PIN25_MA        : 2;
__REG32 BANK1_PIN25_V         : 1;
__REG32                       : 1;
__REG32 BANK1_PIN26_MA        : 2;
__REG32 BANK1_PIN26_V         : 1;
__REG32                       : 1;
__REG32 BANK1_PIN27_MA        : 2;
__REG32 BANK1_PIN27_V         : 1;
__REG32                       : 1;
__REG32 BANK1_PIN28_MA        : 2;
__REG32 BANK1_PIN28_V         : 1;
__REG32                       : 1;
__REG32 BANK1_PIN29_MA        : 2;
__REG32 BANK1_PIN29_V         : 1;
__REG32                       : 1;
__REG32 BANK1_PIN30_MA        : 2;
__REG32 BANK1_PIN30_V         : 1;
__REG32                       : 1;
__REG32 BANK1_PIN31_MA        : 2;
__REG32 BANK1_PIN31_V         : 1;
__REG32                       : 1;
} __hw_pinctrl_drive7_bits;

/* PINCTRL Drive Strength and Voltage Register 8 (HW_PINCTRL_DRIVE8) */
typedef struct {
__REG32 BANK2_PIN00_MA        : 2;
__REG32 BANK2_PIN00_V         : 1;
__REG32                       : 1;
__REG32 BANK2_PIN01_MA        : 2;
__REG32 BANK2_PIN01_V         : 1;
__REG32                       : 1;
__REG32 BANK2_PIN02_MA        : 2;
__REG32 BANK2_PIN02_V         : 1;
__REG32                       : 1;
__REG32 BANK2_PIN03_MA        : 2;
__REG32 BANK2_PIN03_V         : 1;
__REG32                       : 1;
__REG32 BANK2_PIN04_MA        : 2;
__REG32 BANK2_PIN04_V         : 1;
__REG32                       : 1;
__REG32 BANK2_PIN05_MA        : 2;
__REG32 BANK2_PIN05_V         : 1;
__REG32                       : 1;
__REG32 BANK2_PIN06_MA        : 2;
__REG32 BANK2_PIN06_V         : 1;
__REG32                       : 1;
__REG32 BANK2_PIN07_MA        : 2;
__REG32 BANK2_PIN07_V         : 1;
__REG32                       : 1;
} __hw_pinctrl_drive8_bits;

/* PINCTRL Drive Strength and Voltage Register 9 (HW_PINCTRL_DRIVE9) */
typedef struct {
__REG32 BANK2_PIN08_MA        : 2;
__REG32 BANK2_PIN08_V         : 1;
__REG32                       : 1;
__REG32 BANK2_PIN09_MA        : 2;
__REG32 BANK2_PIN09_V         : 1;
__REG32                       : 1;
__REG32 BANK2_PIN10_MA        : 2;
__REG32 BANK2_PIN10_V         : 1;
__REG32                       : 5;
__REG32 BANK2_PIN12_MA        : 2;
__REG32 BANK2_PIN12_V         : 1;
__REG32                       : 1;
__REG32 BANK2_PIN13_MA        : 2;
__REG32 BANK2_PIN13_V         : 1;
__REG32                       : 1;
__REG32 BANK2_PIN14_MA        : 2;
__REG32 BANK2_PIN14_V         : 1;
__REG32                       : 1;
__REG32 BANK2_PIN15_MA        : 2;
__REG32 BANK2_PIN15_V         : 1;
__REG32                       : 1;
} __hw_pinctrl_drive9_bits;

/* PINCTRL Drive Strength and Voltage Register 10 (HW_PINCTRL_DRIVE10) */
typedef struct {
__REG32 BANK2_PIN16_MA        : 2;
__REG32 BANK2_PIN16_V         : 1;
__REG32                       : 1;
__REG32 BANK2_PIN17_MA        : 2;
__REG32 BANK2_PIN17_V         : 1;
__REG32                       : 1;
__REG32 BANK2_PIN18_MA        : 2;
__REG32 BANK2_PIN18_V         : 1;
__REG32                       : 1;
__REG32 BANK2_PIN19_MA        : 2;
__REG32 BANK2_PIN19_V         : 1;
__REG32                       : 1;
__REG32 BANK2_PIN20_MA        : 2;
__REG32 BANK2_PIN20_V         : 1;
__REG32                       : 1;
__REG32 BANK2_PIN21_MA        : 2;
__REG32 BANK2_PIN21_V         : 1;
__REG32                       : 9;
} __hw_pinctrl_drive10_bits;

/* PINCTRL Drive Strength and Voltage Register 11 (HW_PINCTRL_DRIVE11) */
typedef struct {
__REG32 BANK2_PIN24_MA        : 2;
__REG32 BANK2_PIN24_V         : 1;
__REG32                       : 1;
__REG32 BANK2_PIN25_MA        : 2;
__REG32 BANK2_PIN25_V         : 1;
__REG32                       : 1;
__REG32 BANK2_PIN26_MA        : 2;
__REG32 BANK2_PIN26_V         : 1;
__REG32                       : 1;
__REG32 BANK2_PIN27_MA        : 2;
__REG32 BANK2_PIN27_V         : 1;
__REG32                       :17;
} __hw_pinctrl_drive11_bits;

/* PINCTRL Drive Strength and Voltage Register 12 (HW_PINCTRL_DRIVE12) */
typedef struct {
__REG32 BANK3_PIN00_MA        : 2;
__REG32 BANK3_PIN00_V         : 1;
__REG32                       : 1;
__REG32 BANK3_PIN01_MA        : 2;
__REG32 BANK3_PIN01_V         : 1;
__REG32                       : 1;
__REG32 BANK3_PIN02_MA        : 2;
__REG32 BANK3_PIN02_V         : 1;
__REG32                       : 1;
__REG32 BANK3_PIN03_MA        : 2;
__REG32 BANK3_PIN03_V         : 1;
__REG32                       : 1;
__REG32 BANK3_PIN04_MA        : 2;
__REG32 BANK3_PIN04_V         : 1;
__REG32                       : 1;
__REG32 BANK3_PIN05_MA        : 2;
__REG32 BANK3_PIN05_V         : 1;
__REG32                       : 1;
__REG32 BANK3_PIN06_MA        : 2;
__REG32 BANK3_PIN06_V         : 1;
__REG32                       : 1;
__REG32 BANK3_PIN07_MA        : 2;
__REG32 BANK3_PIN07_V         : 1;
__REG32                       : 1;
} __hw_pinctrl_drive12_bits;

/* PINCTRL Drive Strength and Voltage Register 13 (HW_PINCTRL_DRIVE13) */
typedef struct {
__REG32 BANK3_PIN08_MA        : 2;
__REG32 BANK3_PIN08_V         : 1;
__REG32                       : 1;
__REG32 BANK3_PIN09_MA        : 2;
__REG32 BANK3_PIN09_V         : 1;
__REG32                       : 1;
__REG32 BANK3_PIN10_MA        : 2;
__REG32 BANK3_PIN10_V         : 1;
__REG32                       : 1;
__REG32 BANK3_PIN11_MA        : 2;
__REG32 BANK3_PIN11_V         : 1;
__REG32                       : 1;
__REG32 BANK3_PIN12_MA        : 2;
__REG32 BANK3_PIN12_V         : 1;
__REG32                       : 1;
__REG32 BANK3_PIN13_MA        : 2;
__REG32 BANK3_PIN13_V         : 1;
__REG32                       : 1;
__REG32 BANK3_PIN14_MA        : 2;
__REG32 BANK3_PIN14_V         : 1;
__REG32                       : 1;
__REG32 BANK3_PIN15_MA        : 2;
__REG32 BANK3_PIN15_V         : 1;
__REG32                       : 1;
} __hw_pinctrl_drive13_bits;

/* PINCTRL Drive Strength and Voltage Register 14 (HW_PINCTRL_DRIVE14) */
typedef struct {
__REG32 BANK3_PIN16_MA        : 2;
__REG32 BANK3_PIN16_V         : 1;
__REG32                       : 1;
__REG32 BANK3_PIN17_MA        : 2;
__REG32 BANK3_PIN17_V         : 1;
__REG32                       : 1;
__REG32 BANK3_PIN18_MA        : 2;
__REG32 BANK3_PIN18_V         : 1;
__REG32                       : 5;
__REG32 BANK3_PIN20_MA        : 2;
__REG32 BANK3_PIN20_V         : 1;
__REG32                       : 1;
__REG32 BANK3_PIN21_MA        : 2;
__REG32 BANK3_PIN21_V         : 1;
__REG32                       : 1;
__REG32 BANK3_PIN22_MA        : 2;
__REG32 BANK3_PIN22_V         : 1;
__REG32                       : 1;
__REG32 BANK3_PIN23_MA        : 2;
__REG32 BANK3_PIN23_V         : 1;
__REG32                       : 1;
} __hw_pinctrl_drive14_bits;

/* PINCTRL Drive Strength and Voltage Register 15 (HW_PINCTRL_DRIVE15) */
typedef struct {
__REG32 BANK3_PIN24_MA        : 2;
__REG32 BANK3_PIN24_V         : 1;
__REG32                       : 1;
__REG32 BANK3_PIN25_MA        : 2;
__REG32 BANK3_PIN25_V         : 1;
__REG32                       : 1;
__REG32 BANK3_PIN26_MA        : 2;
__REG32 BANK3_PIN26_V         : 1;
__REG32                       : 1;
__REG32 BANK3_PIN27_MA        : 2;
__REG32 BANK3_PIN27_V         : 1;
__REG32                       : 1;
__REG32 BANK3_PIN28_MA        : 2;
__REG32 BANK3_PIN28_V         : 1;
__REG32                       : 1;
__REG32 BANK3_PIN29_MA        : 2;
__REG32 BANK3_PIN29_V         : 1;
__REG32                       : 1;
__REG32 BANK3_PIN30_MA        : 2;
__REG32 BANK3_PIN30_V         : 1;
__REG32                       : 5;
} __hw_pinctrl_drive15_bits;

/* PINCTRL Drive Strength and Voltage Register 16 (HW_PINCTRL_DRIVE16) */
typedef struct {
__REG32 BANK4_PIN00_MA        : 2;
__REG32 BANK4_PIN00_V         : 1;
__REG32                       : 1;
__REG32 BANK4_PIN01_MA        : 2;
__REG32 BANK4_PIN01_V         : 1;
__REG32                       : 1;
__REG32 BANK4_PIN02_MA        : 2;
__REG32 BANK4_PIN02_V         : 1;
__REG32                       : 1;
__REG32 BANK4_PIN03_MA        : 2;
__REG32 BANK4_PIN03_V         : 1;
__REG32                       : 1;
__REG32 BANK4_PIN04_MA        : 2;
__REG32 BANK4_PIN04_V         : 1;
__REG32                       : 1;
__REG32 BANK4_PIN05_MA        : 2;
__REG32 BANK4_PIN05_V         : 1;
__REG32                       : 1;
__REG32 BANK4_PIN06_MA        : 2;
__REG32 BANK4_PIN06_V         : 1;
__REG32                       : 1;
__REG32 BANK4_PIN07_MA        : 2;
__REG32 BANK4_PIN07_V         : 1;
__REG32                       : 1;
} __hw_pinctrl_drive16_bits;

/* PINCTRL Drive Strength and Voltage Register 17 (HW_PINCTRL_DRIVE17) */
typedef struct {
__REG32 BANK4_PIN08_MA        : 2;
__REG32 BANK4_PIN08_V         : 1;
__REG32                       : 1;
__REG32 BANK4_PIN09_MA        : 2;
__REG32 BANK4_PIN09_V         : 1;
__REG32                       : 1;
__REG32 BANK4_PIN10_MA        : 2;
__REG32 BANK4_PIN10_V         : 1;
__REG32                       : 1;
__REG32 BANK4_PIN11_MA        : 2;
__REG32 BANK4_PIN11_V         : 1;
__REG32                       : 1;
__REG32 BANK4_PIN12_MA        : 2;
__REG32 BANK4_PIN12_V         : 1;
__REG32                       : 1;
__REG32 BANK4_PIN13_MA        : 2;
__REG32 BANK4_PIN13_V         : 1;
__REG32                       : 1;
__REG32 BANK4_PIN14_MA        : 2;
__REG32 BANK4_PIN14_V         : 1;
__REG32                       : 1;
__REG32 BANK4_PIN15_MA        : 2;
__REG32 BANK4_PIN15_V         : 1;
__REG32                       : 1;
} __hw_pinctrl_drive17_bits;

/* PINCTRL Drive Strength and Voltage Register 18 (HW_PINCTRL_DRIVE18) */
typedef struct {
__REG32 BANK4_PIN16_MA        : 2;
__REG32 BANK4_PIN16_V         : 1;
__REG32                       :13;
__REG32 BANK4_PIN20_MA        : 2;
__REG32 BANK4_PIN20_V         : 1;
__REG32                       :13;
} __hw_pinctrl_drive18_bits;

/* PINCTRL Bank 0 Pull Up Resistor Enable Register (HW_PINCTRL_PULL0) */
typedef struct {
__REG32 BANK0_PIN00           : 1;
__REG32 BANK0_PIN01           : 1;
__REG32 BANK0_PIN02           : 1;
__REG32 BANK0_PIN03           : 1;
__REG32 BANK0_PIN04           : 1;
__REG32 BANK0_PIN05           : 1;
__REG32 BANK0_PIN06           : 1;
__REG32 BANK0_PIN07           : 1;
__REG32                       : 8;
__REG32 BANK0_PIN16           : 1;
__REG32 BANK0_PIN17           : 1;
__REG32 BANK0_PIN18           : 1;
__REG32 BANK0_PIN19           : 1;
__REG32 BANK0_PIN20           : 1;
__REG32 BANK0_PIN21           : 1;
__REG32 BANK0_PIN22           : 1;
__REG32 BANK0_PIN23           : 1;
__REG32 BANK0_PIN24           : 1;
__REG32 BANK0_PIN25           : 1;
__REG32 BANK0_PIN26           : 1;
__REG32 BANK0_PIN27           : 1;
__REG32 BANK0_PIN28           : 1;
__REG32                       : 3;
} __hw_pinctrl_pull0_bits;

/* PINCTRL Bank 1 Pull Up Resistor Enable Register (HW_PINCTRL_PULL1) */
typedef struct {
__REG32 BANK1_PIN00           : 1;
__REG32 BANK1_PIN01           : 1;
__REG32 BANK1_PIN02           : 1;
__REG32 BANK1_PIN03           : 1;
__REG32 BANK1_PIN04           : 1;
__REG32 BANK1_PIN05           : 1;
__REG32 BANK1_PIN06           : 1;
__REG32 BANK1_PIN07           : 1;
__REG32 BANK1_PIN08           : 1;
__REG32 BANK1_PIN09           : 1;
__REG32 BANK1_PIN10           : 1;
__REG32 BANK1_PIN11           : 1;
__REG32 BANK1_PIN12           : 1;
__REG32 BANK1_PIN13           : 1;
__REG32 BANK1_PIN14           : 1;
__REG32 BANK1_PIN15           : 1;
__REG32 BANK1_PIN16           : 1;
__REG32 BANK1_PIN17           : 1;
__REG32 BANK1_PIN18           : 1;
__REG32 BANK1_PIN19           : 1;
__REG32 BANK1_PIN20           : 1;
__REG32 BANK1_PIN21           : 1;
__REG32 BANK1_PIN22           : 1;
__REG32 BANK1_PIN23           : 1;
__REG32 BANK1_PIN24           : 1;
__REG32 BANK1_PIN25           : 1;
__REG32 BANK1_PIN26           : 1;
__REG32 BANK1_PIN27           : 1;
__REG32 BANK1_PIN28           : 1;
__REG32 BANK1_PIN29           : 1;
__REG32 BANK1_PIN30           : 1;
__REG32 BANK1_PIN31           : 1;
} __hw_pinctrl_pull1_bits;

/* PINCTRL Bank 2 Pull Up Resistor Enable Register (HW_PINCTRL_PULL2) */
typedef struct {
__REG32 BANK2_PIN00           : 1;
__REG32 BANK2_PIN01           : 1;
__REG32 BANK2_PIN02           : 1;
__REG32 BANK2_PIN03           : 1;
__REG32 BANK2_PIN04           : 1;
__REG32 BANK2_PIN05           : 1;
__REG32 BANK2_PIN06           : 1;
__REG32 BANK2_PIN07           : 1;
__REG32 BANK2_PIN08           : 1;
__REG32 BANK2_PIN09           : 1;
__REG32 BANK2_PIN10           : 1;
__REG32                       : 1;
__REG32 BANK2_PIN12           : 1;
__REG32 BANK2_PIN13           : 1;
__REG32 BANK2_PIN14           : 1;
__REG32 BANK2_PIN15           : 1;
__REG32 BANK2_PIN16           : 1;
__REG32 BANK2_PIN17           : 1;
__REG32 BANK2_PIN18           : 1;
__REG32 BANK2_PIN19           : 1;
__REG32 BANK2_PIN20           : 1;
__REG32 BANK2_PIN21           : 1;
__REG32                       : 2;
__REG32 BANK2_PIN24           : 1;
__REG32 BANK2_PIN25           : 1;
__REG32 BANK2_PIN26           : 1;
__REG32 BANK2_PIN27           : 1;
__REG32                       : 4;
} __hw_pinctrl_pull2_bits;

/* PINCTRL Bank 3 Pull Up Resistor Enable Register (HW_PINCTRL_PULL3) */
typedef struct {
__REG32 BANK3_PIN00           : 1;
__REG32 BANK3_PIN01           : 1;
__REG32 BANK3_PIN02           : 1;
__REG32 BANK3_PIN03           : 1;
__REG32 BANK3_PIN04           : 1;
__REG32 BANK3_PIN05           : 1;
__REG32 BANK3_PIN06           : 1;
__REG32 BANK3_PIN07           : 1;
__REG32 BANK3_PIN08           : 1;
__REG32 BANK3_PIN09           : 1;
__REG32 BANK3_PIN10           : 1;
__REG32 BANK3_PIN11           : 1;
__REG32 BANK3_PIN12           : 1;
__REG32 BANK3_PIN13           : 1;
__REG32 BANK3_PIN14           : 1;
__REG32 BANK3_PIN15           : 1;
__REG32 BANK3_PIN16           : 1;
__REG32 BANK3_PIN17           : 1;
__REG32 BANK3_PIN18           : 1;
__REG32                       : 1;
__REG32 BANK3_PIN20           : 1;
__REG32 BANK3_PIN21           : 1;
__REG32 BANK3_PIN22           : 1;
__REG32 BANK3_PIN23           : 1;
__REG32 BANK3_PIN24           : 1;
__REG32 BANK3_PIN25           : 1;
__REG32 BANK3_PIN26           : 1;
__REG32 BANK3_PIN27           : 1;
__REG32 BANK3_PIN28           : 1;
__REG32 BANK3_PIN29           : 1;
__REG32 BANK3_PIN30           : 1;
__REG32                       : 1;
} __hw_pinctrl_pull3_bits;

/* PINCTRL Bank 4 Pull Up Resistor Enable Register (HW_PINCTRL_PULL4) */
typedef struct {
__REG32 BANK4_PIN00           : 1;
__REG32 BANK4_PIN01           : 1;
__REG32 BANK4_PIN02           : 1;
__REG32 BANK4_PIN03           : 1;
__REG32 BANK4_PIN04           : 1;
__REG32 BANK4_PIN05           : 1;
__REG32 BANK4_PIN06           : 1;
__REG32 BANK4_PIN07           : 1;
__REG32 BANK4_PIN08           : 1;
__REG32 BANK4_PIN09           : 1;
__REG32 BANK4_PIN10           : 1;
__REG32 BANK4_PIN11           : 1;
__REG32 BANK4_PIN12           : 1;
__REG32 BANK4_PIN13           : 1;
__REG32 BANK4_PIN14           : 1;
__REG32 BANK4_PIN15           : 1;
__REG32 BANK4_PIN16           : 1;
__REG32                       : 3;
__REG32 BANK4_PIN20           : 1;
__REG32                       :11;
} __hw_pinctrl_pull4_bits;

/* PINCTRL Bank 5 Pull Up Resistor Enable Register (HW_PINCTRL_PULL5) */
typedef struct {
__REG32 BANK5_PIN00           : 1;
__REG32 BANK5_PIN01           : 1;
__REG32 BANK5_PIN02           : 1;
__REG32 BANK5_PIN03           : 1;
__REG32 BANK5_PIN04           : 1;
__REG32 BANK5_PIN05           : 1;
__REG32 BANK5_PIN06           : 1;
__REG32 BANK5_PIN07           : 1;
__REG32 BANK5_PIN08           : 1;
__REG32 BANK5_PIN09           : 1;
__REG32 BANK5_PIN10           : 1;
__REG32 BANK5_PIN11           : 1;
__REG32 BANK5_PIN12           : 1;
__REG32 BANK5_PIN13           : 1;
__REG32 BANK5_PIN14           : 1;
__REG32 BANK5_PIN15           : 1;
__REG32 BANK5_PIN16           : 1;
__REG32 BANK5_PIN17           : 1;
__REG32 BANK5_PIN18           : 1;
__REG32 BANK5_PIN19           : 1;
__REG32 BANK5_PIN20           : 1;
__REG32 BANK5_PIN21           : 1;
__REG32 BANK5_PIN22           : 1;
__REG32 BANK5_PIN23           : 1;
__REG32                       : 2;
__REG32 BANK5_PIN26           : 1;
__REG32                       : 5;
} __hw_pinctrl_pull5_bits;

/* PINCTRL Bank 6 Pull Up Resistor Enable Register (HW_PINCTRL_PULL6) */
typedef struct {
__REG32 BANK6_PIN00           : 1;
__REG32 BANK6_PIN01           : 1;
__REG32 BANK6_PIN02           : 1;
__REG32 BANK6_PIN03           : 1;
__REG32 BANK6_PIN04           : 1;
__REG32 BANK6_PIN05           : 1;
__REG32 BANK6_PIN06           : 1;
__REG32 BANK6_PIN07           : 1;
__REG32 BANK6_PIN08           : 1;
__REG32 BANK6_PIN09           : 1;
__REG32 BANK6_PIN10           : 1;
__REG32 BANK6_PIN11           : 1;
__REG32 BANK6_PIN12           : 1;
__REG32 BANK6_PIN13           : 1;
__REG32 BANK6_PIN14           : 1;
__REG32 BANK6_PIN15           : 1;
__REG32 BANK6_PIN16           : 1;
__REG32 BANK6_PIN17           : 1;
__REG32 BANK6_PIN18           : 1;
__REG32 BANK6_PIN19           : 1;
__REG32 BANK6_PIN20           : 1;
__REG32 BANK6_PIN21           : 1;
__REG32 BANK6_PIN22           : 1;
__REG32 BANK6_PIN23           : 1;
__REG32 BANK6_PIN24           : 1;
__REG32                       : 7;
} __hw_pinctrl_pull6_bits;

/* PINCTRL Bank 0 Data Output Register (HW_PINCTRL_DOUT0) */
typedef struct {
__REG32 DOUT00            : 1;
__REG32 DOUT01            : 1;
__REG32 DOUT02            : 1;
__REG32 DOUT03            : 1;
__REG32 DOUT04            : 1;
__REG32 DOUT05            : 1;
__REG32 DOUT06            : 1;
__REG32 DOUT07            : 1;
__REG32 DOUT08            : 1;
__REG32 DOUT09            : 1;
__REG32 DOUT10            : 1;
__REG32 DOUT11            : 1;
__REG32 DOUT12            : 1;
__REG32 DOUT13            : 1;
__REG32 DOUT14            : 1;
__REG32 DOUT15            : 1;
__REG32 DOUT16            : 1;
__REG32 DOUT17            : 1;
__REG32 DOUT18            : 1;
__REG32 DOUT19            : 1;
__REG32 DOUT20            : 1;
__REG32 DOUT21            : 1;
__REG32 DOUT22            : 1;
__REG32 DOUT23            : 1;
__REG32 DOUT24            : 1;
__REG32 DOUT25            : 1;
__REG32 DOUT26            : 1;
__REG32 DOUT27            : 1;
__REG32 DOUT28            : 1;
__REG32                   : 3;
} __hw_pinctrl_dout0_bits;

/* PINCTRL Bank 1 Data Output Register (HW_PINCTRL_DOUT1) */
typedef struct {
__REG32 DOUT00            : 1;
__REG32 DOUT01            : 1;
__REG32 DOUT02            : 1;
__REG32 DOUT03            : 1;
__REG32 DOUT04            : 1;
__REG32 DOUT05            : 1;
__REG32 DOUT06            : 1;
__REG32 DOUT07            : 1;
__REG32 DOUT08            : 1;
__REG32 DOUT09            : 1;
__REG32 DOUT10            : 1;
__REG32 DOUT11            : 1;
__REG32 DOUT12            : 1;
__REG32 DOUT13            : 1;
__REG32 DOUT14            : 1;
__REG32 DOUT15            : 1;
__REG32 DOUT16            : 1;
__REG32 DOUT17            : 1;
__REG32 DOUT18            : 1;
__REG32 DOUT19            : 1;
__REG32 DOUT20            : 1;
__REG32 DOUT21            : 1;
__REG32 DOUT22            : 1;
__REG32 DOUT23            : 1;
__REG32 DOUT24            : 1;
__REG32 DOUT25            : 1;
__REG32 DOUT26            : 1;
__REG32 DOUT27            : 1;
__REG32 DOUT28            : 1;
__REG32 DOUT29            : 1;
__REG32 DOUT30            : 1;
__REG32 DOUT31            : 1;
} __hw_pinctrl_dout1_bits;

/* PINCTRL Bank 2 Data Output Register (HW_PINCTRL_DOUT2) */
typedef struct {
__REG32 DOUT00            : 1;
__REG32 DOUT01            : 1;
__REG32 DOUT02            : 1;
__REG32 DOUT03            : 1;
__REG32 DOUT04            : 1;
__REG32 DOUT05            : 1;
__REG32 DOUT06            : 1;
__REG32 DOUT07            : 1;
__REG32 DOUT08            : 1;
__REG32 DOUT09            : 1;
__REG32 DOUT10            : 1;
__REG32 DOUT11            : 1;
__REG32 DOUT12            : 1;
__REG32 DOUT13            : 1;
__REG32 DOUT14            : 1;
__REG32 DOUT15            : 1;
__REG32 DOUT16            : 1;
__REG32 DOUT17            : 1;
__REG32 DOUT18            : 1;
__REG32 DOUT19            : 1;
__REG32 DOUT20            : 1;
__REG32 DOUT21            : 1;
__REG32 DOUT22            : 1;
__REG32 DOUT23            : 1;
__REG32 DOUT24            : 1;
__REG32 DOUT25            : 1;
__REG32 DOUT26            : 1;
__REG32 DOUT27            : 1;
__REG32                   : 4;
} __hw_pinctrl_dout2_bits;

/* PINCTRL Bank 3 Data Output Register (HW_PINCTRL_DOUT3) */
typedef struct {
__REG32 DOUT00            : 1;
__REG32 DOUT01            : 1;
__REG32 DOUT02            : 1;
__REG32 DOUT03            : 1;
__REG32 DOUT04            : 1;
__REG32 DOUT05            : 1;
__REG32 DOUT06            : 1;
__REG32 DOUT07            : 1;
__REG32 DOUT08            : 1;
__REG32 DOUT09            : 1;
__REG32 DOUT10            : 1;
__REG32 DOUT11            : 1;
__REG32 DOUT12            : 1;
__REG32 DOUT13            : 1;
__REG32 DOUT14            : 1;
__REG32 DOUT15            : 1;
__REG32 DOUT16            : 1;
__REG32 DOUT17            : 1;
__REG32 DOUT18            : 1;
__REG32 DOUT19            : 1;
__REG32 DOUT20            : 1;
__REG32 DOUT21            : 1;
__REG32 DOUT22            : 1;
__REG32 DOUT23            : 1;
__REG32 DOUT24            : 1;
__REG32 DOUT25            : 1;
__REG32 DOUT26            : 1;
__REG32 DOUT27            : 1;
__REG32 DOUT28            : 1;
__REG32 DOUT29            : 1;
__REG32 DOUT30            : 1;
__REG32                   : 1;
} __hw_pinctrl_dout3_bits;

/* PINCTRL Bank 4 Data Output Register (HW_PINCTRL_DOUT4) */
typedef struct {
__REG32 DOUT00            : 1;
__REG32 DOUT01            : 1;
__REG32 DOUT02            : 1;
__REG32 DOUT03            : 1;
__REG32 DOUT04            : 1;
__REG32 DOUT05            : 1;
__REG32 DOUT06            : 1;
__REG32 DOUT07            : 1;
__REG32 DOUT08            : 1;
__REG32 DOUT09            : 1;
__REG32 DOUT10            : 1;
__REG32 DOUT11            : 1;
__REG32 DOUT12            : 1;
__REG32 DOUT13            : 1;
__REG32 DOUT14            : 1;
__REG32 DOUT15            : 1;
__REG32 DOUT16            : 1;
__REG32 DOUT17            : 1;
__REG32 DOUT18            : 1;
__REG32 DOUT19            : 1;
__REG32 DOUT20            : 1;
__REG32                   :11;
} __hw_pinctrl_dout4_bits;

/* PINCTRL Bank 0 Data Input Register (HW_PINCTRL_DIN0) */
typedef struct {
__REG32 DIN00           : 1;
__REG32 DIN01           : 1;
__REG32 DIN02           : 1;
__REG32 DIN03           : 1;
__REG32 DIN04           : 1;
__REG32 DIN05           : 1;
__REG32 DIN06           : 1;
__REG32 DIN07           : 1;
__REG32 DIN08           : 1;
__REG32 DIN09           : 1;
__REG32 DIN10           : 1;
__REG32 DIN11           : 1;
__REG32 DIN12           : 1;
__REG32 DIN13           : 1;
__REG32 DIN14           : 1;
__REG32 DIN15           : 1;
__REG32 DIN16           : 1;
__REG32 DIN17           : 1;
__REG32 DIN18           : 1;
__REG32 DIN19           : 1;
__REG32 DIN20           : 1;
__REG32 DIN21           : 1;
__REG32 DIN22           : 1;
__REG32 DIN23           : 1;
__REG32 DIN24           : 1;
__REG32 DIN25           : 1;
__REG32 DIN26           : 1;
__REG32 DIN27           : 1;
__REG32 DIN28           : 1;
__REG32                 : 3;
} __hw_pinctrl_din0_bits;

/* PINCTRL Bank 1 Data Input Register (HW_PINCTRL_DIN1) */
typedef struct {
__REG32 DIN00           : 1;
__REG32 DIN01           : 1;
__REG32 DIN02           : 1;
__REG32 DIN03           : 1;
__REG32 DIN04           : 1;
__REG32 DIN05           : 1;
__REG32 DIN06           : 1;
__REG32 DIN07           : 1;
__REG32 DIN08           : 1;
__REG32 DIN09           : 1;
__REG32 DIN10           : 1;
__REG32 DIN11           : 1;
__REG32 DIN12           : 1;
__REG32 DIN13           : 1;
__REG32 DIN14           : 1;
__REG32 DIN15           : 1;
__REG32 DIN16           : 1;
__REG32 DIN17           : 1;
__REG32 DIN18           : 1;
__REG32 DIN19           : 1;
__REG32 DIN20           : 1;
__REG32 DIN21           : 1;
__REG32 DIN22           : 1;
__REG32 DIN23           : 1;
__REG32 DIN24           : 1;
__REG32 DIN25           : 1;
__REG32 DIN26           : 1;
__REG32 DIN27           : 1;
__REG32 DIN28           : 1;
__REG32 DIN29           : 1;
__REG32 DIN30           : 1;
__REG32 DIN31           : 1;
} __hw_pinctrl_din1_bits;

/* PINCTRL Bank 2 Data Input Register (HW_PINCTRL_DIN2) */
typedef struct {
__REG32 DIN00           : 1;
__REG32 DIN01           : 1;
__REG32 DIN02           : 1;
__REG32 DIN03           : 1;
__REG32 DIN04           : 1;
__REG32 DIN05           : 1;
__REG32 DIN06           : 1;
__REG32 DIN07           : 1;
__REG32 DIN08           : 1;
__REG32 DIN09           : 1;
__REG32 DIN10           : 1;
__REG32 DIN11           : 1;
__REG32 DIN12           : 1;
__REG32 DIN13           : 1;
__REG32 DIN14           : 1;
__REG32 DIN15           : 1;
__REG32 DIN16           : 1;
__REG32 DIN17           : 1;
__REG32 DIN18           : 1;
__REG32 DIN19           : 1;
__REG32 DIN20           : 1;
__REG32 DIN21           : 1;
__REG32 DIN22           : 1;
__REG32 DIN23           : 1;
__REG32 DIN24           : 1;
__REG32 DIN25           : 1;
__REG32 DIN26           : 1;
__REG32 DIN27           : 1;
__REG32                 : 4;
} __hw_pinctrl_din2_bits;

/* PINCTRL Bank 3 Data Input Register (HW_PINCTRL_DIN3) */
typedef struct {
__REG32 DIN00           : 1;
__REG32 DIN01           : 1;
__REG32 DIN02           : 1;
__REG32 DIN03           : 1;
__REG32 DIN04           : 1;
__REG32 DIN05           : 1;
__REG32 DIN06           : 1;
__REG32 DIN07           : 1;
__REG32 DIN08           : 1;
__REG32 DIN09           : 1;
__REG32 DIN10           : 1;
__REG32 DIN11           : 1;
__REG32 DIN12           : 1;
__REG32 DIN13           : 1;
__REG32 DIN14           : 1;
__REG32 DIN15           : 1;
__REG32 DIN16           : 1;
__REG32 DIN17           : 1;
__REG32 DIN18           : 1;
__REG32 DIN19           : 1;
__REG32 DIN20           : 1;
__REG32 DIN21           : 1;
__REG32 DIN22           : 1;
__REG32 DIN23           : 1;
__REG32 DIN24           : 1;
__REG32 DIN25           : 1;
__REG32 DIN26           : 1;
__REG32 DIN27           : 1;
__REG32 DIN28           : 1;
__REG32 DIN29           : 1;
__REG32 DIN30           : 1;
__REG32                 : 1;
} __hw_pinctrl_din3_bits;

/* PINCTRL Bank 4 Data Input Register (HW_PINCTRL_DIN4) */
typedef struct {
__REG32 DIN00           : 1;
__REG32 DIN01           : 1;
__REG32 DIN02           : 1;
__REG32 DIN03           : 1;
__REG32 DIN04           : 1;
__REG32 DIN05           : 1;
__REG32 DIN06           : 1;
__REG32 DIN07           : 1;
__REG32 DIN08           : 1;
__REG32 DIN09           : 1;
__REG32 DIN10           : 1;
__REG32 DIN11           : 1;
__REG32 DIN12           : 1;
__REG32 DIN13           : 1;
__REG32 DIN14           : 1;
__REG32 DIN15           : 1;
__REG32 DIN16           : 1;
__REG32 DIN17           : 1;
__REG32 DIN18           : 1;
__REG32 DIN19           : 1;
__REG32 DIN20           : 1;
__REG32                 :11;
} __hw_pinctrl_din4_bits;

/* PINCTRL Bank 0 Data Output Enable Register (HW_PINCTRL_DOE0) */
typedef struct {
__REG32 DOE00           : 1;
__REG32 DOE01           : 1;
__REG32 DOE02           : 1;
__REG32 DOE03           : 1;
__REG32 DOE04           : 1;
__REG32 DOE05           : 1;
__REG32 DOE06           : 1;
__REG32 DOE07           : 1;
__REG32 DOE08           : 1;
__REG32 DOE09           : 1;
__REG32 DOE10           : 1;
__REG32 DOE11           : 1;
__REG32 DOE12           : 1;
__REG32 DOE13           : 1;
__REG32 DOE14           : 1;
__REG32 DOE15           : 1;
__REG32 DOE16           : 1;
__REG32 DOE17           : 1;
__REG32 DOE18           : 1;
__REG32 DOE19           : 1;
__REG32 DOE20           : 1;
__REG32 DOE21           : 1;
__REG32 DOE22           : 1;
__REG32 DOE23           : 1;
__REG32 DOE24           : 1;
__REG32 DOE25           : 1;
__REG32 DOE26           : 1;
__REG32 DOE27           : 1;
__REG32 DOE28           : 1;
__REG32                 : 3;
} __hw_pinctrl_doe0_bits;

/* PINCTRL Bank 1 Data Output Enable Register (HW_PINCTRL_DOE1) */
typedef struct {
__REG32 DOE00           : 1;
__REG32 DOE01           : 1;
__REG32 DOE02           : 1;
__REG32 DOE03           : 1;
__REG32 DOE04           : 1;
__REG32 DOE05           : 1;
__REG32 DOE06           : 1;
__REG32 DOE07           : 1;
__REG32 DOE08           : 1;
__REG32 DOE09           : 1;
__REG32 DOE10           : 1;
__REG32 DOE11           : 1;
__REG32 DOE12           : 1;
__REG32 DOE13           : 1;
__REG32 DOE14           : 1;
__REG32 DOE15           : 1;
__REG32 DOE16           : 1;
__REG32 DOE17           : 1;
__REG32 DOE18           : 1;
__REG32 DOE19           : 1;
__REG32 DOE20           : 1;
__REG32 DOE21           : 1;
__REG32 DOE22           : 1;
__REG32 DOE23           : 1;
__REG32 DOE24           : 1;
__REG32 DOE25           : 1;
__REG32 DOE26           : 1;
__REG32 DOE27           : 1;
__REG32 DOE28           : 1;
__REG32 DOE29           : 1;
__REG32 DOE30           : 1;
__REG32 DOE31           : 1;
} __hw_pinctrl_doe1_bits;

/* PINCTRL Bank 2 Data Output Enable Register (HW_PINCTRL_DOE2) */
typedef struct {
__REG32 DOE00           : 1;
__REG32 DOE01           : 1;
__REG32 DOE02           : 1;
__REG32 DOE03           : 1;
__REG32 DOE04           : 1;
__REG32 DOE05           : 1;
__REG32 DOE06           : 1;
__REG32 DOE07           : 1;
__REG32 DOE08           : 1;
__REG32 DOE09           : 1;
__REG32 DOE10           : 1;
__REG32 DOE11           : 1;
__REG32 DOE12           : 1;
__REG32 DOE13           : 1;
__REG32 DOE14           : 1;
__REG32 DOE15           : 1;
__REG32 DOE16           : 1;
__REG32 DOE17           : 1;
__REG32 DOE18           : 1;
__REG32 DOE19           : 1;
__REG32 DOE20           : 1;
__REG32 DOE21           : 1;
__REG32 DOE22           : 1;
__REG32 DOE23           : 1;
__REG32 DOE24           : 1;
__REG32 DOE25           : 1;
__REG32 DOE26           : 1;
__REG32 DOE27           : 1;
__REG32                 : 4;
} __hw_pinctrl_doe2_bits;

/* PINCTRL Bank 3 Data Output Enable Register (HW_PINCTRL_DOE3) */
typedef struct {
__REG32 DOE00           : 1;
__REG32 DOE01           : 1;
__REG32 DOE02           : 1;
__REG32 DOE03           : 1;
__REG32 DOE04           : 1;
__REG32 DOE05           : 1;
__REG32 DOE06           : 1;
__REG32 DOE07           : 1;
__REG32 DOE08           : 1;
__REG32 DOE09           : 1;
__REG32 DOE10           : 1;
__REG32 DOE11           : 1;
__REG32 DOE12           : 1;
__REG32 DOE13           : 1;
__REG32 DOE14           : 1;
__REG32 DOE15           : 1;
__REG32 DOE16           : 1;
__REG32 DOE17           : 1;
__REG32 DOE18           : 1;
__REG32 DOE19           : 1;
__REG32 DOE20           : 1;
__REG32 DOE21           : 1;
__REG32 DOE22           : 1;
__REG32 DOE23           : 1;
__REG32 DOE24           : 1;
__REG32 DOE25           : 1;
__REG32 DOE26           : 1;
__REG32 DOE27           : 1;
__REG32 DOE28           : 1;
__REG32 DOE29           : 1;
__REG32 DOE30           : 1;
__REG32                 : 1;
} __hw_pinctrl_doe3_bits;

/* PINCTRL Bank 4 Data Output Enable Register (HW_PINCTRL_DOE4) */
typedef struct {
__REG32 DOE00           : 1;
__REG32 DOE01           : 1;
__REG32 DOE02           : 1;
__REG32 DOE03           : 1;
__REG32 DOE04           : 1;
__REG32 DOE05           : 1;
__REG32 DOE06           : 1;
__REG32 DOE07           : 1;
__REG32 DOE08           : 1;
__REG32 DOE09           : 1;
__REG32 DOE10           : 1;
__REG32 DOE11           : 1;
__REG32 DOE12           : 1;
__REG32 DOE13           : 1;
__REG32 DOE14           : 1;
__REG32 DOE15           : 1;
__REG32 DOE16           : 1;
__REG32 DOE17           : 1;
__REG32 DOE18           : 1;
__REG32 DOE19           : 1;
__REG32 DOE20           : 1;
__REG32                 :11;
} __hw_pinctrl_doe4_bits;

/* PINCTRL Bank 0 Interrupt Select Register (HW_PINCTRL_PIN2IRQ0) */
typedef struct {
__REG32 PIN2IRQ00           : 1;
__REG32 PIN2IRQ01           : 1;
__REG32 PIN2IRQ02           : 1;
__REG32 PIN2IRQ03           : 1;
__REG32 PIN2IRQ04           : 1;
__REG32 PIN2IRQ05           : 1;
__REG32 PIN2IRQ06           : 1;
__REG32 PIN2IRQ07           : 1;
__REG32 PIN2IRQ08           : 1;
__REG32 PIN2IRQ09           : 1;
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
__REG32                     : 3;
} __hw_pinctrl_pin2irq0_bits;

/* PINCTRL Bank 1 Interrupt Select Register (HW_PINCTRL_PIN2IRQ1) */
typedef struct {
__REG32 PIN2IRQ00           : 1;
__REG32 PIN2IRQ01           : 1;
__REG32 PIN2IRQ02           : 1;
__REG32 PIN2IRQ03           : 1;
__REG32 PIN2IRQ04           : 1;
__REG32 PIN2IRQ05           : 1;
__REG32 PIN2IRQ06           : 1;
__REG32 PIN2IRQ07           : 1;
__REG32 PIN2IRQ08           : 1;
__REG32 PIN2IRQ09           : 1;
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
} __hw_pinctrl_pin2irq1_bits;

/* PINCTRL Bank 2 Interrupt Select Register (HW_PINCTRL_PIN2IRQ2) */
typedef struct {
__REG32 PIN2IRQ00           : 1;
__REG32 PIN2IRQ01           : 1;
__REG32 PIN2IRQ02           : 1;
__REG32 PIN2IRQ03           : 1;
__REG32 PIN2IRQ04           : 1;
__REG32 PIN2IRQ05           : 1;
__REG32 PIN2IRQ06           : 1;
__REG32 PIN2IRQ07           : 1;
__REG32 PIN2IRQ08           : 1;
__REG32 PIN2IRQ09           : 1;
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
__REG32                     : 4;
} __hw_pinctrl_pin2irq2_bits;

/* PINCTRL Bank 3 Interrupt Select Register (HW_PINCTRL_PIN2IRQ3) */
typedef struct {
__REG32 PIN2IRQ00           : 1;
__REG32 PIN2IRQ01           : 1;
__REG32 PIN2IRQ02           : 1;
__REG32 PIN2IRQ03           : 1;
__REG32 PIN2IRQ04           : 1;
__REG32 PIN2IRQ05           : 1;
__REG32 PIN2IRQ06           : 1;
__REG32 PIN2IRQ07           : 1;
__REG32 PIN2IRQ08           : 1;
__REG32 PIN2IRQ09           : 1;
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
} __hw_pinctrl_pin2irq3_bits;

/* PINCTRL Bank 4 Interrupt Select Register (HW_PINCTRL_PIN2IRQ4) */
typedef struct {
__REG32 PIN2IRQ00           : 1;
__REG32 PIN2IRQ01           : 1;
__REG32 PIN2IRQ02           : 1;
__REG32 PIN2IRQ03           : 1;
__REG32 PIN2IRQ04           : 1;
__REG32 PIN2IRQ05           : 1;
__REG32 PIN2IRQ06           : 1;
__REG32 PIN2IRQ07           : 1;
__REG32 PIN2IRQ08           : 1;
__REG32 PIN2IRQ09           : 1;
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
__REG32                     :11;
} __hw_pinctrl_pin2irq4_bits;

/* PINCTRL Bank 0 Interrupt Mask Register (HW_PINCTRL_IRQEN0) */
typedef struct {
__REG32 IRQEN00           : 1;
__REG32 IRQEN01           : 1;
__REG32 IRQEN02           : 1;
__REG32 IRQEN03           : 1;
__REG32 IRQEN04           : 1;
__REG32 IRQEN05           : 1;
__REG32 IRQEN06           : 1;
__REG32 IRQEN07           : 1;
__REG32 IRQEN08           : 1;
__REG32 IRQEN09           : 1;
__REG32 IRQEN10           : 1;
__REG32 IRQEN11           : 1;
__REG32 IRQEN12           : 1;
__REG32 IRQEN13           : 1;
__REG32 IRQEN14           : 1;
__REG32 IRQEN15           : 1;
__REG32 IRQEN16           : 1;
__REG32 IRQEN17           : 1;
__REG32 IRQEN18           : 1;
__REG32 IRQEN19           : 1;
__REG32 IRQEN20           : 1;
__REG32 IRQEN21           : 1;
__REG32 IRQEN22           : 1;
__REG32 IRQEN23           : 1;
__REG32 IRQEN24           : 1;
__REG32 IRQEN25           : 1;
__REG32 IRQEN26           : 1;
__REG32 IRQEN27           : 1;
__REG32 IRQEN28           : 1;
__REG32                   : 3;
} __hw_pinctrl_irqen0_bits;

/* PINCTRL Bank 1 Interrupt Mask Register (HW_PINCTRL_IRQEN1) */
typedef struct {
__REG32 IRQEN00           : 1;
__REG32 IRQEN01           : 1;
__REG32 IRQEN02           : 1;
__REG32 IRQEN03           : 1;
__REG32 IRQEN04           : 1;
__REG32 IRQEN05           : 1;
__REG32 IRQEN06           : 1;
__REG32 IRQEN07           : 1;
__REG32 IRQEN08           : 1;
__REG32 IRQEN09           : 1;
__REG32 IRQEN10           : 1;
__REG32 IRQEN11           : 1;
__REG32 IRQEN12           : 1;
__REG32 IRQEN13           : 1;
__REG32 IRQEN14           : 1;
__REG32 IRQEN15           : 1;
__REG32 IRQEN16           : 1;
__REG32 IRQEN17           : 1;
__REG32 IRQEN18           : 1;
__REG32 IRQEN19           : 1;
__REG32 IRQEN20           : 1;
__REG32 IRQEN21           : 1;
__REG32 IRQEN22           : 1;
__REG32 IRQEN23           : 1;
__REG32 IRQEN24           : 1;
__REG32 IRQEN25           : 1;
__REG32 IRQEN26           : 1;
__REG32 IRQEN27           : 1;
__REG32 IRQEN28           : 1;
__REG32 IRQEN29           : 1;
__REG32 IRQEN30           : 1;
__REG32 IRQEN31           : 1;
} __hw_pinctrl_irqen1_bits;

/* PINCTRL Bank 2 Interrupt Mask Register (HW_PINCTRL_IRQEN2) */
typedef struct {
__REG32 IRQEN00           : 1;
__REG32 IRQEN01           : 1;
__REG32 IRQEN02           : 1;
__REG32 IRQEN03           : 1;
__REG32 IRQEN04           : 1;
__REG32 IRQEN05           : 1;
__REG32 IRQEN06           : 1;
__REG32 IRQEN07           : 1;
__REG32 IRQEN08           : 1;
__REG32 IRQEN09           : 1;
__REG32 IRQEN10           : 1;
__REG32 IRQEN11           : 1;
__REG32 IRQEN12           : 1;
__REG32 IRQEN13           : 1;
__REG32 IRQEN14           : 1;
__REG32 IRQEN15           : 1;
__REG32 IRQEN16           : 1;
__REG32 IRQEN17           : 1;
__REG32 IRQEN18           : 1;
__REG32 IRQEN19           : 1;
__REG32 IRQEN20           : 1;
__REG32 IRQEN21           : 1;
__REG32 IRQEN22           : 1;
__REG32 IRQEN23           : 1;
__REG32 IRQEN24           : 1;
__REG32 IRQEN25           : 1;
__REG32 IRQEN26           : 1;
__REG32 IRQEN27           : 1;
__REG32                   : 4;
} __hw_pinctrl_irqen2_bits;

/* PINCTRL Bank 3 Interrupt Mask Register (HW_PINCTRL_IRQEN3) */
typedef struct {
__REG32 IRQEN00           : 1;
__REG32 IRQEN01           : 1;
__REG32 IRQEN02           : 1;
__REG32 IRQEN03           : 1;
__REG32 IRQEN04           : 1;
__REG32 IRQEN05           : 1;
__REG32 IRQEN06           : 1;
__REG32 IRQEN07           : 1;
__REG32 IRQEN08           : 1;
__REG32 IRQEN09           : 1;
__REG32 IRQEN10           : 1;
__REG32 IRQEN11           : 1;
__REG32 IRQEN12           : 1;
__REG32 IRQEN13           : 1;
__REG32 IRQEN14           : 1;
__REG32 IRQEN15           : 1;
__REG32 IRQEN16           : 1;
__REG32 IRQEN17           : 1;
__REG32 IRQEN18           : 1;
__REG32 IRQEN19           : 1;
__REG32 IRQEN20           : 1;
__REG32 IRQEN21           : 1;
__REG32 IRQEN22           : 1;
__REG32 IRQEN23           : 1;
__REG32 IRQEN24           : 1;
__REG32 IRQEN25           : 1;
__REG32 IRQEN26           : 1;
__REG32 IRQEN27           : 1;
__REG32 IRQEN28           : 1;
__REG32 IRQEN29           : 1;
__REG32 IRQEN30           : 1;
__REG32                   : 1;
} __hw_pinctrl_irqen3_bits;

/* PINCTRL Bank 4 Interrupt Mask Register (HW_PINCTRL_IRQEN4) */
typedef struct {
__REG32 IRQEN00           : 1;
__REG32 IRQEN01           : 1;
__REG32 IRQEN02           : 1;
__REG32 IRQEN03           : 1;
__REG32 IRQEN04           : 1;
__REG32 IRQEN05           : 1;
__REG32 IRQEN06           : 1;
__REG32 IRQEN07           : 1;
__REG32 IRQEN08           : 1;
__REG32 IRQEN09           : 1;
__REG32 IRQEN10           : 1;
__REG32 IRQEN11           : 1;
__REG32 IRQEN12           : 1;
__REG32 IRQEN13           : 1;
__REG32 IRQEN14           : 1;
__REG32 IRQEN15           : 1;
__REG32 IRQEN16           : 1;
__REG32 IRQEN17           : 1;
__REG32 IRQEN18           : 1;
__REG32 IRQEN19           : 1;
__REG32 IRQEN20           : 1;
__REG32                   :11;
} __hw_pinctrl_irqen4_bits;

/* PINCTRL Bank 0 Interrupt Level/Edge Register (HW_PINCTRL_IRQLEVEL0) */
typedef struct {
__REG32 IRQLEVEL00            : 1;
__REG32 IRQLEVEL01            : 1;
__REG32 IRQLEVEL02            : 1;
__REG32 IRQLEVEL03            : 1;
__REG32 IRQLEVEL04            : 1;
__REG32 IRQLEVEL05            : 1;
__REG32 IRQLEVEL06            : 1;
__REG32 IRQLEVEL07            : 1;
__REG32 IRQLEVEL08            : 1;
__REG32 IRQLEVEL09            : 1;
__REG32 IRQLEVEL10            : 1;
__REG32 IRQLEVEL11            : 1;
__REG32 IRQLEVEL12            : 1;
__REG32 IRQLEVEL13            : 1;
__REG32 IRQLEVEL14            : 1;
__REG32 IRQLEVEL15            : 1;
__REG32 IRQLEVEL16            : 1;
__REG32 IRQLEVEL17            : 1;
__REG32 IRQLEVEL18            : 1;
__REG32 IRQLEVEL19            : 1;
__REG32 IRQLEVEL20            : 1;
__REG32 IRQLEVEL21            : 1;
__REG32 IRQLEVEL22            : 1;
__REG32 IRQLEVEL23            : 1;
__REG32 IRQLEVEL24            : 1;
__REG32 IRQLEVEL25            : 1;
__REG32 IRQLEVEL26            : 1;
__REG32 IRQLEVEL27            : 1;
__REG32 IRQLEVEL28            : 1;
__REG32 IRQLEVEL29            : 1;
__REG32 IRQLEVEL30            : 1;
__REG32 IRQLEVEL31            : 1;
} __hw_pinctrl_irqlevel0_bits;

/* PINCTRL Bank 1 Interrupt Level/Edge Register (HW_PINCTRL_IRQLEVEL1) */
typedef struct {
__REG32 IRQLEVEL00            : 1;
__REG32 IRQLEVEL01            : 1;
__REG32 IRQLEVEL02            : 1;
__REG32 IRQLEVEL03            : 1;
__REG32 IRQLEVEL04            : 1;
__REG32 IRQLEVEL05            : 1;
__REG32 IRQLEVEL06            : 1;
__REG32 IRQLEVEL07            : 1;
__REG32 IRQLEVEL08            : 1;
__REG32 IRQLEVEL09            : 1;
__REG32 IRQLEVEL10            : 1;
__REG32 IRQLEVEL11            : 1;
__REG32 IRQLEVEL12            : 1;
__REG32 IRQLEVEL13            : 1;
__REG32 IRQLEVEL14            : 1;
__REG32 IRQLEVEL15            : 1;
__REG32 IRQLEVEL16            : 1;
__REG32 IRQLEVEL17            : 1;
__REG32 IRQLEVEL18            : 1;
__REG32 IRQLEVEL19            : 1;
__REG32 IRQLEVEL20            : 1;
__REG32 IRQLEVEL21            : 1;
__REG32 IRQLEVEL22            : 1;
__REG32 IRQLEVEL23            : 1;
__REG32 IRQLEVEL24            : 1;
__REG32 IRQLEVEL25            : 1;
__REG32 IRQLEVEL26            : 1;
__REG32 IRQLEVEL27            : 1;
__REG32 IRQLEVEL28            : 1;
__REG32 IRQLEVEL29            : 1;
__REG32 IRQLEVEL30            : 1;
__REG32 IRQLEVEL31            : 1;
} __hw_pinctrl_irqlevel1_bits;

/* PINCTRL Bank 2 Interrupt Level/Edge Register (HW_PINCTRL_IRQLEVEL2) */
typedef struct {
__REG32 IRQLEVEL00            : 1;
__REG32 IRQLEVEL01            : 1;
__REG32 IRQLEVEL02            : 1;
__REG32 IRQLEVEL03            : 1;
__REG32 IRQLEVEL04            : 1;
__REG32 IRQLEVEL05            : 1;
__REG32 IRQLEVEL06            : 1;
__REG32 IRQLEVEL07            : 1;
__REG32 IRQLEVEL08            : 1;
__REG32 IRQLEVEL09            : 1;
__REG32 IRQLEVEL10            : 1;
__REG32 IRQLEVEL11            : 1;
__REG32 IRQLEVEL12            : 1;
__REG32 IRQLEVEL13            : 1;
__REG32 IRQLEVEL14            : 1;
__REG32 IRQLEVEL15            : 1;
__REG32 IRQLEVEL16            : 1;
__REG32 IRQLEVEL17            : 1;
__REG32 IRQLEVEL18            : 1;
__REG32 IRQLEVEL19            : 1;
__REG32 IRQLEVEL20            : 1;
__REG32 IRQLEVEL21            : 1;
__REG32 IRQLEVEL22            : 1;
__REG32 IRQLEVEL23            : 1;
__REG32 IRQLEVEL24            : 1;
__REG32 IRQLEVEL25            : 1;
__REG32 IRQLEVEL26            : 1;
__REG32 IRQLEVEL27            : 1;
__REG32                       : 4;
} __hw_pinctrl_irqlevel2_bits;

/* PINCTRL Bank 3 Interrupt Level/Edge Register (HW_PINCTRL_IRQLEVEL3) */
typedef struct {
__REG32 IRQLEVEL00            : 1;
__REG32 IRQLEVEL01            : 1;
__REG32 IRQLEVEL02            : 1;
__REG32 IRQLEVEL03            : 1;
__REG32 IRQLEVEL04            : 1;
__REG32 IRQLEVEL05            : 1;
__REG32 IRQLEVEL06            : 1;
__REG32 IRQLEVEL07            : 1;
__REG32 IRQLEVEL08            : 1;
__REG32 IRQLEVEL09            : 1;
__REG32 IRQLEVEL10            : 1;
__REG32 IRQLEVEL11            : 1;
__REG32 IRQLEVEL12            : 1;
__REG32 IRQLEVEL13            : 1;
__REG32 IRQLEVEL14            : 1;
__REG32 IRQLEVEL15            : 1;
__REG32 IRQLEVEL16            : 1;
__REG32 IRQLEVEL17            : 1;
__REG32 IRQLEVEL18            : 1;
__REG32 IRQLEVEL19            : 1;
__REG32 IRQLEVEL20            : 1;
__REG32 IRQLEVEL21            : 1;
__REG32 IRQLEVEL22            : 1;
__REG32 IRQLEVEL23            : 1;
__REG32 IRQLEVEL24            : 1;
__REG32 IRQLEVEL25            : 1;
__REG32 IRQLEVEL26            : 1;
__REG32 IRQLEVEL27            : 1;
__REG32 IRQLEVEL28            : 1;
__REG32 IRQLEVEL29            : 1;
__REG32 IRQLEVEL30            : 1;
__REG32                       : 1;
} __hw_pinctrl_irqlevel3_bits;

/* PINCTRL Bank 4 Interrupt Level/Edge Register (HW_PINCTRL_IRQLEVEL4) */
typedef struct {
__REG32 IRQLEVEL00            : 1;
__REG32 IRQLEVEL01            : 1;
__REG32 IRQLEVEL02            : 1;
__REG32 IRQLEVEL03            : 1;
__REG32 IRQLEVEL04            : 1;
__REG32 IRQLEVEL05            : 1;
__REG32 IRQLEVEL06            : 1;
__REG32 IRQLEVEL07            : 1;
__REG32 IRQLEVEL08            : 1;
__REG32 IRQLEVEL09            : 1;
__REG32 IRQLEVEL10            : 1;
__REG32 IRQLEVEL11            : 1;
__REG32 IRQLEVEL12            : 1;
__REG32 IRQLEVEL13            : 1;
__REG32 IRQLEVEL14            : 1;
__REG32 IRQLEVEL15            : 1;
__REG32 IRQLEVEL16            : 1;
__REG32 IRQLEVEL17            : 1;
__REG32 IRQLEVEL18            : 1;
__REG32 IRQLEVEL19            : 1;
__REG32 IRQLEVEL20            : 1;
__REG32                       :11;
} __hw_pinctrl_irqlevel4_bits;

/* PINCTRL Bank 0 Interrupt Polarity Register (HW_PINCTRL_IRQPOL0) */
typedef struct {
__REG32 IRQPOL00            : 1;
__REG32 IRQPOL01            : 1;
__REG32 IRQPOL02            : 1;
__REG32 IRQPOL03            : 1;
__REG32 IRQPOL04            : 1;
__REG32 IRQPOL05            : 1;
__REG32 IRQPOL06            : 1;
__REG32 IRQPOL07            : 1;
__REG32 IRQPOL08            : 1;
__REG32 IRQPOL09            : 1;
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
__REG32                     : 3;
} __hw_pinctrl_irqpol0_bits;

/* PINCTRL Bank 1 Interrupt Polarity Register (HW_PINCTRL_IRQPOL1) */
typedef struct {
__REG32 IRQPOL00            : 1;
__REG32 IRQPOL01            : 1;
__REG32 IRQPOL02            : 1;
__REG32 IRQPOL03            : 1;
__REG32 IRQPOL04            : 1;
__REG32 IRQPOL05            : 1;
__REG32 IRQPOL06            : 1;
__REG32 IRQPOL07            : 1;
__REG32 IRQPOL08            : 1;
__REG32 IRQPOL09            : 1;
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
} __hw_pinctrl_irqpol1_bits;

/* PINCTRL Bank 2 Interrupt Polarity Register (HW_PINCTRL_IRQPOL2) */
typedef struct {
__REG32 IRQPOL00            : 1;
__REG32 IRQPOL01            : 1;
__REG32 IRQPOL02            : 1;
__REG32 IRQPOL03            : 1;
__REG32 IRQPOL04            : 1;
__REG32 IRQPOL05            : 1;
__REG32 IRQPOL06            : 1;
__REG32 IRQPOL07            : 1;
__REG32 IRQPOL08            : 1;
__REG32 IRQPOL09            : 1;
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
__REG32                     : 4;
} __hw_pinctrl_irqpol2_bits;

/* PINCTRL Bank 3 Interrupt Polarity Register (HW_PINCTRL_IRQPOL3) */
typedef struct {
__REG32 IRQPOL00            : 1;
__REG32 IRQPOL01            : 1;
__REG32 IRQPOL02            : 1;
__REG32 IRQPOL03            : 1;
__REG32 IRQPOL04            : 1;
__REG32 IRQPOL05            : 1;
__REG32 IRQPOL06            : 1;
__REG32 IRQPOL07            : 1;
__REG32 IRQPOL08            : 1;
__REG32 IRQPOL09            : 1;
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
} __hw_pinctrl_irqpol3_bits;

/* PINCTRL Bank 4 Interrupt Polarity Register (HW_PINCTRL_IRQPOL4) */
typedef struct {
__REG32 IRQPOL00            : 1;
__REG32 IRQPOL01            : 1;
__REG32 IRQPOL02            : 1;
__REG32 IRQPOL03            : 1;
__REG32 IRQPOL04            : 1;
__REG32 IRQPOL05            : 1;
__REG32 IRQPOL06            : 1;
__REG32 IRQPOL07            : 1;
__REG32 IRQPOL08            : 1;
__REG32 IRQPOL09            : 1;
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
__REG32                     :11;
} __hw_pinctrl_irqpol4_bits;

/* PINCTRL Bank 0 Interrupt Status Register (HW_PINCTRL_IRQSTAT0) */
typedef struct {
__REG32 IRQSTAT00           : 1;
__REG32 IRQSTAT01           : 1;
__REG32 IRQSTAT02           : 1;
__REG32 IRQSTAT03           : 1;
__REG32 IRQSTAT04           : 1;
__REG32 IRQSTAT05           : 1;
__REG32 IRQSTAT06           : 1;
__REG32 IRQSTAT07           : 1;
__REG32 IRQSTAT08           : 1;
__REG32 IRQSTAT09           : 1;
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
__REG32                     : 3;
} __hw_pinctrl_irqstat0_bits;

/* PINCTRL Bank 1 Interrupt Status Register (HW_PINCTRL_IRQSTAT1) */
typedef struct {
__REG32 IRQSTAT00           : 1;
__REG32 IRQSTAT01           : 1;
__REG32 IRQSTAT02           : 1;
__REG32 IRQSTAT03           : 1;
__REG32 IRQSTAT04           : 1;
__REG32 IRQSTAT05           : 1;
__REG32 IRQSTAT06           : 1;
__REG32 IRQSTAT07           : 1;
__REG32 IRQSTAT08           : 1;
__REG32 IRQSTAT09           : 1;
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
} __hw_pinctrl_irqstat1_bits;

/* PINCTRL Bank 2 Interrupt Status Register (HW_PINCTRL_IRQSTAT2) */
typedef struct {
__REG32 IRQSTAT00           : 1;
__REG32 IRQSTAT01           : 1;
__REG32 IRQSTAT02           : 1;
__REG32 IRQSTAT03           : 1;
__REG32 IRQSTAT04           : 1;
__REG32 IRQSTAT05           : 1;
__REG32 IRQSTAT06           : 1;
__REG32 IRQSTAT07           : 1;
__REG32 IRQSTAT08           : 1;
__REG32 IRQSTAT09           : 1;
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
__REG32                     : 4;
} __hw_pinctrl_irqstat2_bits;

/* PINCTRL Bank 3 Interrupt Status Register (HW_PINCTRL_IRQSTAT3) */
typedef struct {
__REG32 IRQSTAT00           : 1;
__REG32 IRQSTAT01           : 1;
__REG32 IRQSTAT02           : 1;
__REG32 IRQSTAT03           : 1;
__REG32 IRQSTAT04           : 1;
__REG32 IRQSTAT05           : 1;
__REG32 IRQSTAT06           : 1;
__REG32 IRQSTAT07           : 1;
__REG32 IRQSTAT08           : 1;
__REG32 IRQSTAT09           : 1;
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
} __hw_pinctrl_irqstat3_bits;

/* PINCTRL Bank 4 Interrupt Status Register (HW_PINCTRL_IRQSTAT4) */
typedef struct {
__REG32 IRQSTAT00           : 1;
__REG32 IRQSTAT01           : 1;
__REG32 IRQSTAT02           : 1;
__REG32 IRQSTAT03           : 1;
__REG32 IRQSTAT04           : 1;
__REG32 IRQSTAT05           : 1;
__REG32 IRQSTAT06           : 1;
__REG32 IRQSTAT07           : 1;
__REG32 IRQSTAT08           : 1;
__REG32 IRQSTAT09           : 1;
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
__REG32                     :11;
} __hw_pinctrl_irqstat4_bits;

/* PINCTRL EMI Slice ODT Control (HW_PINCTRL_EMI_ODT_CTRL) */
typedef struct {
__REG32 SLICE0_TLOAD        : 2;
__REG32 SLICE0_CALIB        : 2;
__REG32 SLICE1_TLOAD        : 2;
__REG32 SLICE1_CALIB        : 2;
__REG32 SLICE2_TLOAD        : 2;
__REG32 SLICE2_CALIB        : 2;
__REG32 SLICE3_TLOAD        : 2;
__REG32 SLICE3_CALIB        : 2;
__REG32 DUALPAD_TLOAD       : 2;
__REG32 DUALPAD_CALIB       : 2;
__REG32 CONTROL_TLOAD       : 2;
__REG32 CONTROL_CALIB       : 2;
__REG32 ADDRESS_TLOAD       : 2;
__REG32 ADDRESS_CALIB       : 2;
__REG32                     : 4;
} __hw_pinctrl_emi_odt_ctrl_bits;

/* PINCTRL EMI Slice DS Control (HW_PINCTRL_EMI_DS_CTRL) */
typedef struct {
__REG32 SLICE0_MA           : 2;
__REG32 SLICE1_MA           : 2;
__REG32 SLICE2_MA           : 2;
__REG32 SLICE3_MA           : 2;
__REG32 DUALPAD_MA          : 2;
__REG32 CONTROL_MA          : 2;
__REG32 ADDRESS_MA          : 2;
__REG32                     : 2;
__REG32 DDR_MODE            : 2;
__REG32                     :14;
} __hw_pinctrl_emi_ds_ctrl_bits;

/* System PLL0, System/USB0 PLL Control Register 0 (HW_CLKCTRL_PLL0CTRL0) */
typedef struct {
__REG32                     :17;
__REG32 POWER               : 1;
__REG32 EN_USB_CLKS         : 1;
__REG32                     : 1;
__REG32 DIV_SEL             : 2;
__REG32                     : 2;
__REG32 CP_SEL              : 2;
__REG32                     : 2;
__REG32 LFR_SEL             : 2;
__REG32                     : 2;
} __hw_clkctrl_pll0_ctrl0_bits;

/* System PLL0, System/USB0 PLL Control Register 1 (HW_CLKCTRL_PLL0CTRL1) */
typedef struct {
__REG32 LOCK_COUNT          :16;
__REG32                     :14;
__REG32 FORCE_LOCK          : 1;
__REG32 LOCK                : 1;
} __hw_clkctrl_pll0_ctrl1_bits;

/* System PLL1, USB1 PLL Control Register 0 (HW_CLKCTRL_PLL1CTRL0) */
typedef struct {
__REG32                     :17;
__REG32 POWER               : 1;
__REG32 EN_USB_CLKS         : 1;
__REG32                     : 1;
__REG32 DIV_SEL             : 2;
__REG32                     : 2;
__REG32 CP_SEL              : 2;
__REG32                     : 2;
__REG32 LFR_SEL             : 2;
__REG32                     : 1;
__REG32 CLKGATEEMI          : 1;
} __hw_clkctrl_pll1_ctrl0_bits;

/* System PLL1, USB1 PLL Control Register 1 (HW_CLKCTRL_PLL1CTRL1) */
typedef struct {
__REG32 LOCK_COUNT          :16;
__REG32                     :14;
__REG32 FORCE_LOCK          : 1;
__REG32 LOCK                : 1;
} __hw_clkctrl_pll1_ctrl1_bits;

/* System PLL2, USB1 PLL Control Register 0 (HW_CLKCTRL_PLL2CTRL0) */
typedef struct {
__REG32                     :23;
__REG32 POWER               : 1;
__REG32 CP_SEL              : 2;
__REG32 HOLD_RING_OFF_B     : 1;
__REG32                     : 1;
__REG32 LFR_SEL             : 2;
__REG32                     : 1;
__REG32 CLKGATE             : 1;
} __hw_clkctrl_pll2_ctrl0_bits;

/* System PLL2, USB1 PLL Control Register 0 (HW_CLKCTRL_PLL2CTRL0) */
typedef struct {
__REG32 DIV_CPU             : 6;
__REG32                     : 4;
__REG32 DIV_CPU_FRAC_EN     : 1;
__REG32                     : 1;
__REG32 INTERRUPT_WAIT      : 1;
__REG32                     : 3;
__REG32 DIV_XTAL            :10;
__REG32 DIV_XTAL_FRAC_EN    : 1;
__REG32                     : 1;
__REG32 BUSY_REF_CPU        : 1;
__REG32 BUSY_REF_XTAL       : 1;
__REG32                     : 2;
} __hw_clkctrl_cpu_bits;

/* AHB, APBH Bus Clock Control Register (HW_CLKCTRL_HBUS) */
typedef struct {
__REG32 DIV                     : 5;
__REG32 DIV_FRAC_EN             : 1;
__REG32                         :10;
__REG32 SLOW_DIV                : 3;
__REG32 AUTO_CLEAR_DIV_ENABLE   : 1;
__REG32 ASM_ENABLE              : 1;
__REG32 CPU_INSTR_AS_ENABLE     : 1;
__REG32 CPU_DATA_AS_ENABLE      : 1;
__REG32 TRAFFIC_AS_ENABLE       : 1;
__REG32 TRAFFIC_JAM_AS_ENABLE   : 1;
__REG32 APBXDMA_AS_ENABLE       : 1;
__REG32 APBHDMA_AS_ENABLE       : 1;
__REG32 ASM_EMIPORT_AS_ENABLE   : 1;
__REG32                         : 1;
__REG32 PXP_AS_ENABLE           : 1;
__REG32 DCP_AS_ENABLE           : 1;
__REG32 ASM_BUSY                : 1;
} __hw_clkctrl_hbus_bits;

/* APBX Clock Control Register (HW_CLKCTRL_XBUS) */
typedef struct {
__REG32 DIV                     :10;
__REG32 DIV_FRAC_EN             : 1;
__REG32 AUTO_CLEAR_DIV_ENABLE   : 1;
__REG32                         :19;
__REG32 BUSY                    : 1;
} __hw_clkctrl_xbus_bits;

/* XTAL Clock Control Register (HW_CLKCTRL_XTAL) */
typedef struct {
__REG32 DIV_UART                : 2;
__REG32                         :24;
__REG32 TIMROT_CLK32K_GATE      : 1;
__REG32                         : 2;
__REG32 PWM_CLK24M_GATE         : 1;
__REG32                         : 1;
__REG32 UART_CLK_GATE           : 1;
} __hw_clkctrl_xtal_bits;

/* Synchronous Serial Portn Clock Control Register (HW_CLKCTRL_SSPn) */
typedef struct {
__REG32 DIV                     : 9;
__REG32 DIV_FRAC_EN             : 1;
__REG32                         :19;
__REG32 BUSY                    : 1;
__REG32                         : 1;
__REG32 CLKGATE                 : 1;
} __hw_clkctrl_ssp_bits;

/* General-Purpose Media Interface Clock Control Register (HW_CLKCTRL_GPMI) */
typedef struct {
__REG32 DIV                     :10;
__REG32 DIV_FRAC_EN             : 1;
__REG32                         :18;
__REG32 BUSY                    : 1;
__REG32                         : 1;
__REG32 CLKGATE                 : 1;
} __hw_clkctrl_gpmi_bits;

/* SPDIF Clock Control Register (HW_CLKCTRL_SPDIF) */
typedef struct {
__REG32                         :31;
__REG32 CLKGATE                 : 1;
} __hw_clkctrl_spdif_bits;

/* EMI Clock Control Register (HW_CLKCTRL_EMI) */
typedef struct {
__REG32 DIV_EMI                 : 6;
__REG32                         : 2;
__REG32 DIV_XTAL                : 4;
__REG32                         : 4;
__REG32 DCC_RESYNC_ENABLE       : 1;
__REG32 BUSY_DCC_RESYNC         : 1;
__REG32                         : 8;
__REG32 BUSY_SYNC_MODE          : 1;
__REG32 BUSY_REF_CPU            : 1;
__REG32 BUSY_REF_EMI            : 1;
__REG32 BUSY_REF_XTAL           : 1;
__REG32 SYNC_MODE_EN            : 1;
__REG32 CLKGATE                 : 1;
} __hw_clkctrl_emi_bits;

/* SAIFn Clock Control Register (HW_CLKCTRL_SAIFn) */
typedef struct {
__REG32 DIV                     :16;
__REG32 DIV_FRAC_EN             : 1;
__REG32                         :12;
__REG32 BUSY                    : 1;
__REG32                         : 1;
__REG32 CLKGATE                 : 1;
} __hw_clkctrl_saif_bits;

/* CLK_DIS_LCDIF Clock Control Register (HW_CLKCTRL_DIS_LCDIF) */
typedef struct {
__REG32 DIV                     :13;
__REG32 DIV_FRAC_EN             : 1;
__REG32                         :15;
__REG32 BUSY                    : 1;
__REG32                         : 1;
__REG32 CLKGATE                 : 1;
} __hw_clkctrl_dis_lcdif_bits;

/* ETM Clock Control Register (HW_CLKCTRL_ETM) */
typedef struct {
__REG32 DIV                     : 7;
__REG32 DIV_FRAC_EN             : 1;
__REG32                         :21;
__REG32 BUSY                    : 1;
__REG32                         : 1;
__REG32 CLKGATE                 : 1;
} __hw_clkctrl_etm_bits;

/* ENET Clock Control Register (HW_CLKCTRL_ENET) */
typedef struct {
__REG32                         :16;
__REG32 RESET_BY_SW             : 1;
__REG32 RESET_BY_SW_CHIP        : 1;
__REG32 CLK_OUT_EN              : 1;
__REG32 TIME_SEL                : 2;
__REG32 DIV_TIME                : 6;
__REG32 BUSY_TIME               : 1;
__REG32                         : 1;
__REG32 STATUS                  : 1;
__REG32 DISABLE                 : 1;
__REG32 SLEEP                   : 1;
} __hw_clkctrl_enet_bits;

/* HSADC Clock Control Register (HW_CLKCTRL_HSADC) */
typedef struct {
__REG32                         :28;
__REG32 FREQDIV                 : 2;
__REG32 RESETB                  : 1;
__REG32                         : 1;
} __hw_clkctrl_hsadc_bits;

/* FLEXCAN Clock Control Register (HW_CLKCTRL_FLEXCAN) */
typedef struct {
__REG32                         :27;
__REG32 CAN1_STATUS             : 1;
__REG32 STOP_CAN1               : 1;
__REG32 CAN0_STATUS             : 1;
__REG32 STOP_CAN0               : 1;
__REG32                         : 1;
} __hw_clkctrl_flexcan_bits;

/* Fractional Clock Control Register 0 (HW_CLKCTRL_FRAC0) */
typedef struct {
__REG32 CPUFRAC                 : 6;
__REG32 CPU_STABLE              : 1;
__REG32 CLKGATECPU              : 1;
__REG32 EMIFRAC                 : 6;
__REG32 EMI_STABLE              : 1;
__REG32 CLKGATEEMI              : 1;
__REG32 IO1FRAC                 : 6;
__REG32 IO1_STABLE              : 1;
__REG32 CLKGATEIO1              : 1;
__REG32 IO0FRAC                 : 6;
__REG32 IO0_STABLE              : 1;
__REG32 CLKGATEIO0              : 1;
} __hw_clkctrl_frac0_bits;

/* Fractional Clock Control Register 1 (HW_CLKCTRL_FRAC1) */
typedef struct {
__REG32 PIXFRAC                 : 6;
__REG32 PIX_STABLE              : 1;
__REG32 CLKGATEPIX              : 1;
__REG32 HSADCFRAC               : 6;
__REG32 HSADC_STABLE            : 1;
__REG32 CLKGATEHSADC            : 1;
__REG32 GPMIFRAC                : 6;
__REG32 GPMI_STABLE             : 1;
__REG32 CLKGATEGPMI             : 1;
__REG32                         : 8;
} __hw_clkctrl_frac1_bits;

/* Clock Frequency Sequence Control Register (HW_CLKCTRL_CLKSEQ) */
typedef struct {
__REG32 BYPASS_SAIF0            : 1;
__REG32 BYPASS_SAIF1            : 1;
__REG32 BYPASS_GPMI             : 1;
__REG32 BYPASS_SSP0             : 1;
__REG32 BYPASS_SSP1             : 1;
__REG32 BYPASS_SSP2             : 1;
__REG32 BYPASS_SSP3             : 1;
__REG32 BYPASS_EMI              : 1;
__REG32 BYPASS_ETM              : 1;
__REG32                         : 5;
__REG32 BYPASS_DIS_LCDIF        : 1;
__REG32                         : 3;
__REG32 BYPASS_CPU              : 1;
__REG32                         :13;
} __hw_clkctrl_clkseq_bits;

/* System Reset Control Register (HW_CLKCTRL_RESET) */
typedef struct {
__REG32 DIG                     : 1;
__REG32 CHIP                    : 1;
__REG32 THERMAL_RESET_DEFAULT   : 1;
__REG32 THERMAL_RESET_ENABLE    : 1;
__REG32 EXTERNAL_RESET_ENABLE   : 1;
__REG32 WDOG_POR_DISABLE        : 1;
__REG32                         :26;
} __hw_clkctrl_reset_bits;

/* ClkCtrl Status (HW_CLKCTRL_STATUS) */
typedef struct {
__REG32                         :30;
__REG32 CPU_LIMIT               : 2;
} __hw_clkctrl_status_bits;

/* ClkCtrl Version (HW_CLKCTRL_VERSION) */
typedef struct {
__REG32 STEP                    :16;
__REG32 MINOR                   : 8;
__REG32 MAJOR                   : 8;
} __hw_clkctrl_version_bits;

/* Power Control Register (HW_POWER_CTRL) */
typedef struct {
__REG32 ENIRQ_VDD5V_GT_VDDIO    : 1;
__REG32 VDD5V_GT_VDDIO_IRQ      : 1;
__REG32 POLARITY_VDD5V_GT_VDDIO : 1;
__REG32 ENIRQ_VBUS_VALID        : 1;
__REG32 VBUSVALID_IRQ           : 1;
__REG32 POLARITY_VBUSVALID      : 1;
__REG32 ENIRQ_VDDD_BO           : 1;
__REG32 VDDD_BO_IRQ             : 1;
__REG32 ENIRQ_VDDA_BO           : 1;
__REG32 VDDA_BO_IRQ             : 1;
__REG32 ENIRQ_VDDIO_BO          : 1;
__REG32 VDDIO_BO_IRQ            : 1;
__REG32 ENIRQBATT_BO            : 1;
__REG32 BATT_BO_IRQ             : 1;
__REG32 ENIRQ_DC_OK             : 1;
__REG32 DC_OK_IRQ               : 1;
__REG32 POLARITY_DC_OK          : 1;
__REG32 ENIRQ_PSWITCH           : 1;
__REG32 POLARITY_PSWITCH        : 1;
__REG32 PSWITCH_IRQ_SRC         : 1;
__REG32 PSWITCH_IRQ             : 1;
__REG32 ENIRQ_VDD5V_DROOP       : 1;
__REG32 VDD5V_DROOP_IRQ         : 1;
__REG32 ENIRQ_DCDC4P2_BO        : 1;
__REG32 DCDC4P2_BO_IRQ          : 1;
__REG32                         : 2;
__REG32 PSWITCH_MID_TRAN        : 1;
__REG32                         : 4;
} __hw_power_ctrl_bits;

/* DC-DC 5V Control Register (HW_POWER_5VCTRL) */
typedef struct {
__REG32 ENABLE_DCDC             : 1;
__REG32 PWRUP_VBUS_CMPS         : 1;
__REG32 ILIMIT_EQ_ZERO          : 1;
__REG32 VBUSVALID_TO_B          : 1;
__REG32 VBUSVALID_5VDETECT      : 1;
__REG32 DCDC_XFER               : 1;
__REG32 ENABLE_LINREG_ILIMIT    : 1;
__REG32 PWDN_5VBRNOUT           : 1;
__REG32 VBUSVALID_TRSH          : 3;
__REG32                         : 1;
__REG32 CHARGE_4P2_ILIMIT       : 6;
__REG32                         : 2;
__REG32 PWD_CHARGE_4P2          : 2;
__REG32                         : 2;
__REG32 HEADROOM_ADJ            : 3;
__REG32                         : 1;
__REG32 VBUSDROOP_TRSH          : 2;
__REG32                         : 2;
} __hw_power_5vctrl_bits;

/* DC-DC Minimum Power and Miscellaneous Control Register (HW_POWER_MINPWR) */
typedef struct {
__REG32 DC_HALFCLK              : 1;
__REG32 EN_DC_PFM               : 1;
__REG32 DC_STOPCLK              : 1;
__REG32 PWD_XTAL24              : 1;
__REG32 LESSANA_I               : 1;
__REG32 HALF_FETS               : 1;
__REG32 DOUBLE_FETS             : 1;
__REG32 VBG_OFF                 : 1;
__REG32 SELECT_OSC              : 1;
__REG32 ENABLE_OSC              : 1;
__REG32 PWD_ANA_CMPS            : 1;
__REG32 USE_VDDXTAL_VBG         : 1;
__REG32 PWD_BO                  : 1;
__REG32                         : 1;
__REG32 LOWPWR_4P2              : 1;
__REG32                         :17;
} __hw_power_minpwr_bits;

/* Battery Charge Control Register (HW_POWER_CHARGE) */
typedef struct {
__REG32 BATTCHRG_I              : 6;
__REG32                         : 2;
__REG32 STOP_ILIMIT             : 4;
__REG32 ENABLE_CHARGER_USB0     : 1;
__REG32 ENABLE_CHARGER_USB1     : 1;
__REG32                         : 2;
__REG32 PWD_BATTCHRG            : 1;
__REG32                         : 1;
__REG32 LIION_4P1               : 1;
__REG32 CHRG_STS_OFF            : 1;
__REG32 ENABLE_FAULT_DETECT     : 1;
__REG32                         : 1;
__REG32 ENABLE_LOAD             : 1;
__REG32                         : 1;
__REG32 ADJ_VOLT                : 3;
__REG32                         : 5;
} __hw_power_charge_bits;

/* VDDD Supply Targets and Brownouts Control Register (HW_POWER_VDDDCTRL) */
typedef struct {
__REG32 TRG                     : 5;
__REG32                         : 3;
__REG32 BO_OFFSET               : 3;
__REG32                         : 5;
__REG32 LINREG_OFFSET           : 2;
__REG32                         : 2;
__REG32 DISABLE_FET             : 1;
__REG32 ENABLE_LINREG           : 1;
__REG32 DISABLE_STEPPING        : 1;
__REG32 PWDN_BRNOUT             : 1;
__REG32                         : 4;
__REG32 ADJTN                   : 4;
} __hw_power_vdddctrl_bits;

/* VDDA Supply Targets and Brownouts Control Register (HW_POWER_VDDACTRL) */
typedef struct {
__REG32 TRG                     : 5;
__REG32                         : 3;
__REG32 BO_OFFSET               : 3;
__REG32                         : 1;
__REG32 LINREG_OFFSET           : 2;
__REG32                         : 2;
__REG32 DISABLE_FET             : 1;
__REG32 ENABLE_LINREG           : 1;
__REG32 DISABLE_STEPPING        : 1;
__REG32 PWDN_BRNOUT             : 1;
__REG32                         :12;
} __hw_power_vddactrl_bits;

/* VDDIO Supply Targets and Brownouts Control Register (HW_POWER_VDDIOCTRL) */
typedef struct {
__REG32 TRG                     : 5;
__REG32                         : 3;
__REG32 BO_OFFSET               : 3;
__REG32                         : 1;
__REG32 LINREG_OFFSET           : 2;
__REG32                         : 2;
__REG32 DISABLE_FET             : 1;
__REG32 ENABLE_LINREG           : 1;
__REG32 DISABLE_STEPPING        : 1;
__REG32                         : 1;
__REG32 ADJTN                   : 4;
__REG32                         : 8;
} __hw_power_vddioctrl_bits;

/* VDDMEM Supply Targets Control Register (HW_POWER_VDDMEMCTRL) */
typedef struct {
__REG32 TRG                     : 5;
__REG32 BO_OFFSET               : 3;
__REG32 ENABLE_LINREG           : 1;
__REG32 ENABLE_ILIMIT           : 1;
__REG32 PULLDOWN_ACTIVE         : 1;
__REG32                         :21;
} __hw_power_vddmemctrl_bits;

/* DC-DC Converter 4.2V Control Register (HW_POWER_DCDC4P2) */
typedef struct {
__REG32 CMPTRIP                 : 5;
__REG32                         : 3;
__REG32 BO                      : 5;
__REG32                         : 3;
__REG32 TRG                     : 3;
__REG32                         : 1;
__REG32 HYST_THRESH             : 1;
__REG32 HYST_DIR                : 1;
__REG32 ENABLE_DCDC             : 1;
__REG32 ENABLE_4P2              : 1;
__REG32 ISTEAL_THRESH           : 2;
__REG32                         : 2;
__REG32 DROPOUT_CTRL            : 4;
} __hw_power_dcdc4p2_bits;

/* DC-DC Miscellaneous Register (HW_POWER_MISC) */
typedef struct {
__REG32 SEL_PLLCLK              : 1;
__REG32 TEST                    : 1;
__REG32 DELAY_TIMING            : 1;
__REG32 DISABLEFET_BO_LOGIC     : 1;
__REG32 FREQSEL                 : 3;
__REG32                         :25;
} __hw_power_misc_bits;

/* DC-DC Duty Cycle Limits Control Register (HW_POWER_DCLIMITS) */
typedef struct {
__REG32 NEGLIMIT                : 7;
__REG32                         : 1;
__REG32 POSLIMIT_BUCK           : 7;
__REG32                         :17;
} __hw_power_dclimits_bits;

/* Converter Loop Behavior Control Register (HW_POWER_LOOPCTRL) */
typedef struct {
__REG32 DC_C                    : 2;
__REG32                         : 2;
__REG32 DC_R                    : 4;
__REG32 DC_FF                   : 3;
__REG32                         : 1;
__REG32 EN_RCSCALE              : 2;
__REG32 RCSCALE_THRESH          : 1;
__REG32 DF_HYST_THRESH          : 1;
__REG32 CM_HYST_THRESH          : 1;
__REG32 EN_DF_HYST              : 1;
__REG32 EN_CM_HYST              : 1;
__REG32 HYST_SIGN               : 1;
__REG32 TOGGLE_DIF              : 1;
__REG32                         :11;
} __hw_power_loopctrl_bits;

/* Power Subsystem Status Register (HW_POWER_STS) */
typedef struct {
__REG32 SESSEND0                : 1;
__REG32 VBUSVALID0              : 1;
__REG32 BVALID0                 : 1;
__REG32 AVALID0                 : 1;
__REG32 VDD5V_DROOP             : 1;
__REG32 VDD5V_GT_VDDIO          : 1;
__REG32 VDDD_BO                 : 1;
__REG32 VDDA_BO                 : 1;
__REG32 VDDIO_BO                : 1;
__REG32 DC_OK                   : 1;
__REG32 DCDC_4P2_BO             : 1;
__REG32 CHRGSTS                 : 1;
__REG32 VDD5V_FAULT             : 1;
__REG32 BATT_BO                 : 1;
__REG32 SESSEND0_STATUS         : 1;
__REG32 VBUSVALID0_STATUS       : 1;
__REG32 BVALID0_STATUS          : 1;
__REG32 AVALID0_STATUS          : 1;
__REG32 VDDMEM_BO               : 1;
__REG32 THERMAL_WARNING         : 1;
__REG32 PSWITCH                 : 2;
__REG32                         : 2;
__REG32 PWRUP_SOURCE            : 6;
__REG32                         : 2;
} __hw_power_sts_bits;

/* Transistor Speed Control and Status Register (HW_POWER_SPEED) */
typedef struct {
__REG32 CTRL                    : 2;
__REG32                         : 4;
__REG32 STATUS_SEL              : 2;
__REG32 STATUS                  :16;
__REG32                         : 8;
} __hw_power_speed_bits;

/* Battery Level Monitor Register (HW_POWER_BATTMONITOR) */
typedef struct {
__REG32 BRWNOUT_LVL                     : 5;
__REG32                                 : 3;
__REG32 BRWNOUT_PWD                     : 1;
__REG32 PWDN_BATTBRNOUT                 : 1;
__REG32 EN_BATADJ                       : 1;
__REG32 PWDN_BATTBRNOUT_5VDETECT_ENABLE : 1;
__REG32                                 : 4;
__REG32 BATT_VAL                        :10;
__REG32                                 : 6;
} __hw_power_battmonitor_bits;

/* Power Module Reset Register (HW_POWER_RESET) */
typedef struct {
__REG32 PWD                             : 1;
	
__REG32 PWD_OFF                         : 1;
__REG32 FASTFALLPSWITCH_OFF             : 1;
__REG32                                 :13;
__REG32 UNLOCK                          :16;
} __hw_power_reset_bits;

/* Power Module Debug Register (HW_POWER_DEBUG) */
typedef struct {
__REG32 SESSENDPIOLOCK                  : 1;
__REG32 BVALIDPIOLOCK                   : 1;
__REG32 AVALIDPIOLOCK                   : 1;
__REG32 VBUSVALIDPIOLOCK                : 1;
__REG32                                 :28;
} __hw_power_debug_bits;

/* Power Module Thermal Reset Register (HW_POWER_THERMAL) */
typedef struct {
__REG32 TEMP_THRESHOLD                  : 3;
__REG32 OFFSET_ADJ_ENABLE               : 1;
__REG32 OFFSET_ADJ                      : 2;
__REG32 LOW_POWER                       : 1;
__REG32 PWD                             : 1;
__REG32 TEST                            : 1;
__REG32                                 :23;
} __hw_power_thermal_bits;

/* Power Module USB1 Manual Controls Register (HW_POWER_USB1CTRL) */
typedef struct {
__REG32 SESSEND1                        : 1;
__REG32 VBUSVALID1                      : 1;
__REG32 BVALID1                         : 1;
__REG32 AVALID1                         : 1;
__REG32                                 :28;
} __hw_power_usb1ctrl_bits;

/* Power Module Version Register (HW_POWER_VERSION) */
typedef struct {
__REG32 STEP                            :16;
__REG32 MINOR                           : 8;
__REG32 MAJOR                           : 8;
} __hw_power_version_bits;

/* Analog Clock Control Register (HW_POWER_ANACLKCTRL) */
typedef struct {
__REG32 INDIV                           : 3;
__REG32                                 : 1;
__REG32 INCLK_SHIFT                     : 2;
__REG32                                 : 2;
__REG32 INVERT_INCLK                    : 1;
__REG32 SLOW_DITHER                     : 1;
__REG32 DITHER_OFF                      : 1;
__REG32                                 :15;
__REG32 CKGATE_I                        : 1;
__REG32 INVERT_OUTCLK                   : 1;
__REG32 OUTDIV                          : 3;
__REG32 CKGATE_O                        : 1;
} __hw_power_anaclkctrl_bits;

/* POWER Reference Control Register (HW_POWER_REFCTRL) */
typedef struct {
__REG32                                 : 4;
__REG32 VAG_VAL                         : 4;
__REG32 ANA_REFVAL                      : 4;
__REG32 ADJ_VAG                         : 1;
__REG32 ADJ_ANA                         : 1;
__REG32 VDDXTAL_TO_VDDD                 : 1;
__REG32                                 : 1;
__REG32 BIAS_CTRL                       : 2;
__REG32                                 : 1;
__REG32 LOW_PWR                         : 1;
__REG32 VBG_ADJ                         : 3;
__REG32                                 : 1;
__REG32 XTAL_BGR_BIAS                   : 1;
__REG32 RAISE_REF                       : 1;
__REG32 FASTSETTLING                    : 1;
__REG32                                 : 5;
} __hw_power_refctrl_bits;

/* DCP Control Register 0 (HW_DCP_CTRL) */
typedef struct {
__REG32 CHANNEL_INTERRUPT_ENABLE        : 8;
__REG32                                 :13;
__REG32 ENABLE_CONTEXT_SWITCHING        : 1;
__REG32 ENABLE_CONTEXT_CACHING          : 1;
__REG32 GATHER_RESIDUAL_WRITES          : 1;
__REG32                                 : 4;
__REG32 PRESENT_SHA                     : 1;
__REG32 PRESENT_CRYPTO                  : 1;
__REG32 CLKGATE                         : 1;
__REG32 SFTRST                          : 1;
} __hw_dcp_ctrl_bits;

/* DCP Status Register (HW_DCP_STAT) */
typedef struct {
__REG32 IRQ                             : 4;
__REG32                                 :12;
__REG32 READY_CHANNELS                  : 8;
__REG32 CUR_CHANNEL                     : 4;
__REG32 OTP_KEY_READY                   : 1;
__REG32                                 : 3;
} __hw_dcp_stat_bits;

/* DCP Channel Control Register (HW_DCP_CHANNELCTRL) */
typedef struct {
__REG32 ENABLE_CHANNEL                  : 8;
__REG32 HIGH_PRIORITY_CHANNEL           : 8;
__REG32 CH0_IRQ_MERGED                  : 1;
__REG32                                 :15;
} __hw_dcp_channelctrl_bits;

/* DCP Capability 0 Register (HW_DCP_CAPABILITY0) */
typedef struct {
__REG32 NUM_KEYS                        : 8;
__REG32 NUM_CHANNELS                    : 4;
__REG32                                 :17;
__REG32 DISABLE_UNIQUE_KEY              : 1;
__REG32 ENABLE_TZONE                    : 1;
__REG32 DISABLE_DECRYPT                 : 1;
} __hw_dcp_capability0_bits;

/* DCP Capability 1 Register (HW_DCP_CAPABILITY1) */
typedef struct {
__REG32 CIPHER_ALGORITHMS               :16;
__REG32 HASH_ALGORITHMS                 :16;
} __hw_dcp_capability1_bits;

/* DCP Key Index (HW_DCP_KEY) */
typedef struct {
__REG32 SUBWORD                         : 2;
__REG32                                 : 2;
__REG32 INDEX                           : 2;
__REG32                                 :26;
} __hw_dcp_key_bits;

/* DCP Work Packet 1 Status Register (HW_DCP_PACKET1) */
typedef struct {
__REG32 INTERRUPT                       : 1;
__REG32 DECR_SEMAPHORE                  : 1;
__REG32 CHAIN                           : 1;
__REG32 CHAIN_CONTIGUOUS                : 1;
__REG32 ENABLE_MEMCOPY                  : 1;
__REG32 ENABLE_CIPHER                   : 1;
__REG32 ENABLE_HASH                     : 1;
__REG32 ENABLE_BLIT                     : 1;
__REG32 CIPHER_ENCRYPT                  : 1;
__REG32 CIPHER_INIT                     : 1;
__REG32 OTP_KEY                         : 1;
__REG32 PAYLOAD_KEY                     : 1;
__REG32 HASH_INIT                       : 1;
__REG32 HASH_TERM                       : 1;
__REG32 CHECK_HASH                      : 1;
__REG32 HASH_OUTPUT                     : 1;
__REG32 CONSTANT_FILL                   : 1;
__REG32 TEST_SEMA_IRQ                   : 1;
__REG32 KEY_BYTESWAP                    : 1;
__REG32 KEY_WORDSWAP                    : 1;
__REG32 INPUT_BYTESWAP                  : 1;
__REG32 INPUT_WORDSWAP                  : 1;
__REG32 OUTPUT_BYTESWAP                 : 1;
__REG32 OUTPUT_WORDSWAP                 : 1;
__REG32                                 : 8;
} __hw_dcp_packet1_bits;

/* DCP Work Packet 2 Status Register (HW_DCP_PACKET2) */
typedef struct {
__REG32 CIPHER_SELECT                   : 4;
__REG32 CIPHER_MODE                     : 4;
__REG32 KEY_SELECT                      : 8;
__REG32 HASH_SELECT                     : 4;
__REG32                                 : 4;
__REG32 CIPHER_CFG                      : 8;
} __hw_dcp_packet2_bits;

/* DCP Work Packet 2 Status Register (HW_DCP_PACKET2) */
typedef struct {
__REG32 INCREMEN                        : 8;
__REG32                                 : 8;
__REG32 VALUE                           : 8;
__REG32                                 : 8;
} __hw_dcp_chsema_bits;

/* DCP Channel n Status Register (HW_DCP_CHnSTAT) */
typedef struct {
__REG32                                 : 1;
__REG32 HASH_MISMATCH                   : 1;
__REG32 ERROR_SETUP                     : 1;
__REG32 ERROR_PACKET                    : 1;
__REG32 ERROR_SRC                       : 1;
__REG32 ERROR_DST                       : 1;
__REG32 ERROR_PAGEFAULT                 : 1;
__REG32                                 : 9;
__REG32 ERROR_CODE                      : 8;
__REG32 TAG                             : 8;
} __hw_dcp_chstat_bits;

/* DCP Channel n Options Register (HW_DCP_CH0nPTS) */
typedef struct {
__REG32 RECOVERY_TIMER                  :16;
__REG32                                 :16;
} __hw_dcp_chopts_bits;

/* DCP Debug Select Register (HW_DCP_DBGSELECT) */
typedef struct {
__REG32 INDEX                           : 8;
__REG32                                 :24;
} __hw_dcp_dbgselect_bits;

/* DCP Page Table Register (HW_DCP_PAGETABLE) */
typedef struct {
__REG32 ENABLE                          : 1;
__REG32 FLUSH                           : 1;
__REG32 BASE                            :30;
} __hw_dcp_pagetable_bits;

/* DCP Version Register (HW_DCP_VERSION) */
typedef struct {
__REG32 STEP                            :16;
__REG32 MINOR                           : 8;
__REG32 MAJOR                           : 8;
} __hw_dcp_version_bits;

/* DRAM Control Register 00 (HW_DRAM_CTL00) */
typedef struct {
__REG32 BRESP_TIMING                    : 1;
__REG32 SREFRESH_ENTER                  : 1;
__REG32 CKE_SELECT                      : 1;
__REG32 USER_DEF_REG_0_1                :29;
} __hw_dram_ctl00_bits;

/* AXI Monitor Control (HW_DRAM_CTL01) */
typedef struct {
__REG32 MON_DISABLE                     : 4;
__REG32 SLVERR                          : 4;
__REG32 MON_DBG_STB                     : 1;
__REG32 USER_DEF_REG_1                  :23;
} __hw_dram_ctl01_bits;

/* DRAM Control Register 02 (HW_DRAM_CTL02) */
typedef struct {
__REG32 MON_DISABLE                     : 4;
__REG32 SLVERR                          : 4;
__REG32 MON_DBG_STB                     : 1;
__REG32 USER_DEF_REG_1                  :23;
} __hw_dram_ctl02_bits;

/* DRAM Control Register 08 (HW_DRAM_CTL08) */
typedef struct {
__REG32 COMMAND_ACCEPTED                : 4;
__REG32 CKE_STATUS                      : 1;
__REG32 SREFRESH_ACK                    : 1;
__REG32 Q_ALMOST_ULL                    : 1;
__REG32 REFRESH_IN_PROCESS              : 1;
__REG32 CONTROLLER_BUSY                 : 1;
__REG32 USER_DEF_REG_RO_0               :23;
} __hw_dram_ctl08_bits;

/* AXI0 Debug n (HW_DRAM_CTL10) */
typedef struct {
__REG32 WDATA_CNT                       : 8;
__REG32 WRESP_CNT                       : 8;
__REG32 READ_CNT                        : 8;
__REG32                                 : 8;
} __hw_dram_ctl10_bits;

/* AXI0 Debug n (HW_DRAM_CTL11) */
typedef struct {
__REG32 WLEN                            : 8;
__REG32 RLEN                            : 8;
__REG32 RSTATE                          : 8;
__REG32 WSTATE                          : 8;
} __hw_dram_ctl11_bits;

/* DRAM Control Register 16 (HW_DRAM_CTL16) */
typedef struct {
__REG32 START                           : 1;
__REG32                                 :15;
__REG32 POWER_DOWN                      : 1;
__REG32                                 : 7;
__REG32 WRITE_MODEREG                   : 1;
__REG32                                 : 7;
} __hw_dram_ctl16_bits;

/* DRAM Control Register 17 (HW_DRAM_CTL17) */
typedef struct {
__REG32 SREFRESH                        : 1;
__REG32                                 : 7;
__REG32 ENABLE_QUICK_SREFRESH           : 1;
__REG32                                 : 7;
__REG32 AREFRESH                        : 1;
__REG32                                 : 7;
__REG32 AUTO_REFRESH_MODE               : 1;
__REG32                                 : 7;
} __hw_dram_ctl17_bits;

/* DRAM Control Register 21 (HW_DRAM_CTL21) */
typedef struct {
__REG32 DLL_BYPASS_MODE                 : 1;
__REG32                                 : 7;
__REG32 DLLLOCKREG                      : 1;
__REG32                                 : 7;
__REG32 DLL_LOCK                        : 8;
__REG32 CKE_DELAY                       : 3;
__REG32                                 : 5;
} __hw_dram_ctl21_bits;

/* DRAM Control Register 22 (HW_DRAM_CTL22) */
typedef struct {
__REG32 LOWPOWER_AUTO_ENABLE            : 5;
__REG32                                 : 3;
__REG32 LOWPOWER_CONTROL                : 5;
__REG32                                 : 3;
__REG32 LOWPOWER_RSVD2REFRESH_ENABLE    : 4;
__REG32                                 :12;
} __hw_dram_ctl22_bits;

/* DRAM Control Register 23 (HW_DRAM_CTL23) */
typedef struct {
__REG32 LOWPOWER_EXTERNAL_CNT           :16;
__REG32 LOWPOWER_INTERNAL_CNT           :16;
} __hw_dram_ctl23_bits;

/* DRAM Control Register 24 (HW_DRAM_CTL24) */
typedef struct {
__REG32 LOWPOWER_REFRESH_HOLD           :16;
__REG32 LOWPOWER_SELF_REFRESH_CNT       :16;
} __hw_dram_ctl24_bits;

/* DRAM Control Register 25 (HW_DRAM_CTL25) */
typedef struct {
__REG32 LOWPOWER_POWER_DOWN_CNT         :16;
__REG32                                 :16;
} __hw_dram_ctl25_bits;

/* DRAM Control Register 26 (HW_DRAM_CTL26) */
typedef struct {
__REG32 PLACEMENT_EN                    : 1;
__REG32                                 : 7;
__REG32 ADDR_CMP_EN                     : 1;
__REG32                                 : 7;
__REG32 PRIORITY_EN                     : 1;
__REG32                                 :15;
} __hw_dram_ctl26_bits;

/* DRAM Control Register 27 (HW_DRAM_CTL27) */
typedef struct {
__REG32 RW_SAME_EN                      : 1;
__REG32                                 : 7;
__REG32 BANK_SPLIT_EN                   : 1;
__REG32                                 : 7;
__REG32 SWAP_EN                         : 1;
__REG32                                 : 7;
__REG32 SWAP_PORT_RW_SAME_EN            : 1;
__REG32                                 : 7;
} __hw_dram_ctl27_bits;

/* DRAM Control Register 28 (HW_DRAM_CTL28) */
typedef struct {
__REG32 ACTIVE_AGING                    : 1;
__REG32                                 : 7;
__REG32 COMMAND_AGE_COUNT               : 4;
__REG32                                 : 4;
__REG32 AGE_COUNT                       : 4;
__REG32                                 : 4;
__REG32 Q_FULLNESS                      : 3;
__REG32                                 : 5;
} __hw_dram_ctl28_bits;

/* DRAM Control Register 29 (HW_DRAM_CTL29) */
typedef struct {
__REG32 APREBIT                         : 4;
__REG32                                 : 4;
__REG32 ADDR_PINS                       : 3;
__REG32                                 : 5;
__REG32 COLUMN_SIZE                     : 3;
__REG32                                 : 5;
__REG32 CS_MAP                          : 4;
__REG32                                 : 4;
} __hw_dram_ctl29_bits;

/* DRAM Control Register 30 (HW_DRAM_CTL30) */
typedef struct {
__REG32 MAX_COL_REG                     : 4;
__REG32                                 : 4;
__REG32 MAX_ROW_REG                     : 4;
__REG32                                 : 4;
__REG32 MAX_CS_REG                      : 3;
__REG32                                 :13;
} __hw_dram_ctl30_bits;

/* DRAM Control Register 31 (HW_DRAM_CTL31) */
typedef struct {
__REG32 DQS_N_EN                        : 1;
__REG32                                 : 7;
__REG32 DRIVE_DQ_DQS                    : 1;
__REG32                                 : 7;
__REG32 EIGHT_BANK_MODE                 : 1;
__REG32                                 :15;
} __hw_dram_ctl31_bits;

/* DRAM Control Register 32 (HW_DRAM_CTL32) */
typedef struct {
__REG32 REG_DIMM_ENABLE                 : 1;
__REG32                                 : 7;
__REG32 REDUC                           : 1;
__REG32                                 :23;
} __hw_dram_ctl32_bits;

/* DRAM Control Register 33 (HW_DRAM_CTL33) */
typedef struct {
__REG32 AP                              : 1;
__REG32                                 : 7;
__REG32 CONCURRENTAP                    : 1;
__REG32                                 :23;
} __hw_dram_ctl33_bits;

/* DRAM Control Register 34 (HW_DRAM_CTL34) */
typedef struct {
__REG32 INTRPTAPBURST                   : 1;
__REG32                                 : 7;
__REG32 INTRPTREADA                     : 1;
__REG32                                 : 7;
__REG32 INTRPTWRITEA                    : 1;
__REG32                                 : 7;
__REG32 WRITEINTERP                     : 1;
__REG32                                 : 7;
} __hw_dram_ctl34_bits;

/* DRAM Control Register 35 (HW_DRAM_CTL35) */
typedef struct {
__REG32 INITAREF                        : 4;
__REG32                                 : 4;
__REG32 NO_CMD_INIT                     : 1;
__REG32                                 : 7;
__REG32 PWRUP_SREFRESH_EXIT             : 1;
__REG32                                 :15;
} __hw_dram_ctl35_bits;

/* DRAM Control Register 36 (HW_DRAM_CTL36) */
typedef struct {
__REG32 FAST_WRITE                      : 1;
__REG32                                 :15;
__REG32 TRAS_LOCKOUT                    : 1;
__REG32                                 : 7;
__REG32 TREF_ENABLE                     : 1;
__REG32                                 : 7;
} __hw_dram_ctl36_bits;

/* DRAM Control Register 37 (HW_DRAM_CTL37) */
typedef struct {
__REG32 WRLAT                           : 4;
__REG32                                 : 4;
__REG32 CASLAT                          : 3;
__REG32                                 : 5;
__REG32 CASLAT_LIN                      : 4;
__REG32                                 : 4;
__REG32 CASLAT_LIN_GATE                 : 4;
__REG32                                 : 4;
} __hw_dram_ctl37_bits;

/* DRAM Control Register 38 (HW_DRAM_CTL38) */
typedef struct {
__REG32 TCKE                            : 3;
__REG32                                 : 5;
__REG32 TCPD                            :16;
__REG32 TDAL                            : 5;
__REG32                                 : 3;
} __hw_dram_ctl38_bits;

/* DRAM Control Register 39 (HW_DRAM_CTL39) */
typedef struct {
__REG32 TDLL                            :16;
__REG32                                 : 8;
__REG32 TFAW                            : 6;
__REG32                                 : 2;
} __hw_dram_ctl39_bits;

/* DRAM Control Register 40 (HW_DRAM_CTL40) */
typedef struct {
__REG32 TINIT                           :24;
__REG32 TMRD                            : 5;
__REG32                                 : 3;
} __hw_dram_ctl40_bits;

/* DRAM Control Register 41 (HW_DRAM_CTL41) */
typedef struct {
__REG32 TRC                             : 6;
__REG32                                 : 2;
__REG32 TRCD_INT                        : 8;
__REG32 TPDEX                           :16;
} __hw_dram_ctl41_bits;

/* DRAM Control Register 42 (HW_DRAM_CTL42) */
typedef struct {
__REG32 TRAS_MIN                        : 8;
__REG32 TRAS_MAX                        :16;
__REG32                                 : 8;
} __hw_dram_ctl42_bits;

/* DRAM Control Register 43 (HW_DRAM_CTL43) */
typedef struct {
__REG32 TREF                            :14;
__REG32                                 : 2;
__REG32 TRFC                            : 8;
__REG32 TRP                             : 4;
__REG32                                 : 4;
} __hw_dram_ctl43_bits;

/* DRAM Control Register 44 (HW_DRAM_CTL44) */
typedef struct {
__REG32 TRRD                            : 3;
__REG32                                 : 5;
__REG32 TRTP                            : 3;
__REG32                                 : 5;
__REG32 TWR_INT                         : 5;
__REG32                                 : 3;
__REG32 TWTR                            : 4;
__REG32                                 : 4;
} __hw_dram_ctl44_bits;

/* DRAM Control Register 45 (HW_DRAM_CTL45) */
typedef struct {
__REG32 TXSNR                           :16;
__REG32 TXSR                            :16;
} __hw_dram_ctl45_bits;

/* DRAM Control Register 48 (HW_DRAM_CTL48) */
typedef struct {
__REG32 AXI0_FIFO_TYPE_REG              : 2;
__REG32                                 : 6;
__REG32 AXI0_BDW                        : 7;
__REG32                                 : 1;
__REG32 AXI0_BDW_OVFLOW                 : 1;
__REG32                                 : 7;
__REG32 AXI0_CURRENT_BDW                : 7;
__REG32                                 : 1;
} __hw_dram_ctl48_bits;

/* DRAM Control Register 49 (HW_DRAM_CTL49) */
typedef struct {
__REG32 AXI0_R_PRIORITY                 : 3;
__REG32                                 : 5;
__REG32 AXI0_W_PRIORITY                 : 3;
__REG32                                 : 5;
__REG32 AXI0_EN_SIZE_LT_WIDTH_INSTR     :16;
} __hw_dram_ctl49_bits;

/* DRAM Control Register 50 (HW_DRAM_CTL50) */
typedef struct {
__REG32 AXI1_FIFO_TYPE_REG              : 2;
__REG32                                 : 6;
__REG32 AXI1_BDW                        : 7;
__REG32                                 : 1;
__REG32 AXI1_BDW_OVFLOW                 : 1;
__REG32                                 : 7;
__REG32 AXI1_CURRENT_BDW                : 7;
__REG32                                 : 1;
} __hw_dram_ctl50_bits;

/* DRAM Control Register 51 (HW_DRAM_CTL51) */
typedef struct {
__REG32 AXI1_R_PRIORITY                 : 3;
__REG32                                 : 5;
__REG32 AXI1_W_PRIORITY                 : 3;
__REG32                                 : 5;
__REG32 AXI1_EN_SIZE_LT_WIDTH_INSTR     :16;
} __hw_dram_ctl51_bits;

/* DRAM Control Register 52 (HW_DRAM_CTL52) */
typedef struct {
__REG32 AXI2_FIFO_TYPE_REG              : 2;
__REG32                                 : 6;
__REG32 AXI2_BDW                        : 7;
__REG32                                 : 1;
__REG32 AXI2_BDW_OVFLOW                 : 1;
__REG32                                 : 7;
__REG32 AXI2_CURRENT_BDW                : 7;
__REG32                                 : 1;
} __hw_dram_ctl52_bits;

/* DRAM Control Register 53 (HW_DRAM_CTL53) */
typedef struct {
__REG32 AXI2_R_PRIORITY                 : 3;
__REG32                                 : 5;
__REG32 AXI2_W_PRIORITY                 : 3;
__REG32                                 : 5;
__REG32 AXI2_EN_SIZE_LT_WIDTH_INSTR     :16;
} __hw_dram_ctl53_bits;

/* DRAM Control Register 54 (HW_DRAM_CTL54) */
typedef struct {
__REG32 AXI3_FIFO_TYPE_REG              : 2;
__REG32                                 : 6;
__REG32 AXI3_BDW                        : 7;
__REG32                                 : 1;
__REG32 AXI3_BDW_OVFLOW                 : 1;
__REG32                                 : 7;
__REG32 AXI3_CURRENT_BDW                : 7;
__REG32                                 : 1;
} __hw_dram_ctl54_bits;

/* DRAM Control Register 55 (HW_DRAM_CTL55) */
typedef struct {
__REG32 AXI3_R_PRIORITY                 : 3;
__REG32                                 : 5;
__REG32 AXI3_W_PRIORITY                 : 3;
__REG32                                 : 5;
__REG32 AXI3_EN_SIZE_LT_WIDTH_INSTR     :16;
} __hw_dram_ctl55_bits;

/* DRAM Control Register 56 (HW_DRAM_CTL56) */
typedef struct {
__REG32 ARB_CMD_Q_THRESHOLD             : 3;
__REG32                                 :29;
} __hw_dram_ctl56_bits;

/* DRAM Control Register 58 (HW_DRAM_CTL58) */
typedef struct {
__REG32 INT_MASK                        :11;
__REG32                                 : 5;
__REG32 INT_STATUS                      :11;
__REG32                                 : 5;
} __hw_dram_ctl58_bits;

/* DRAM Control Register 60 (HW_DRAM_CTL60) */
typedef struct {
__REG32 OUT_OF_RANGE_ADDR               : 2;
__REG32                                 :30;
} __hw_dram_ctl60_bits;

/* DRAM Control Register 61 (HW_DRAM_CTL61) */
typedef struct {
__REG32 OUT_OF_RANGE_SOURCE_ID          :13;
__REG32                                 : 3;
__REG32 OUT_OF_RANGE_LENGTH             : 7;
__REG32                                 : 1;
__REG32 OUT_OF_RANGE_TYPE               : 6;
__REG32                                 : 2;
} __hw_dram_ctl61_bits;

/* DRAM Control Register 63 (HW_DRAM_CTL63) */
typedef struct {
__REG32 PORT_CMD_ERROR_ADDR             : 2;
__REG32                                 :30;
} __hw_dram_ctl63_bits;

/* DRAM Control Register 64 (HW_DRAM_CTL64) */
typedef struct {
__REG32 PORT_CMD_ERROR_TYPE             : 4;
__REG32                                 : 4;
__REG32 PORT_CMD_ERROR_ID               :13;
__REG32                                 :11;
} __hw_dram_ctl64_bits;

/* DRAM Control Register 65 (HW_DRAM_CTL65) */
typedef struct {
__REG32 PORT_DATA_ERROR_TYPE            : 3;
__REG32                                 : 5;
__REG32 PORT_DATA_ERROR_ID              :13;
__REG32                                 :11;
} __hw_dram_ctl65_bits;

/* DRAM Control Register 66 (HW_DRAM_CTL66) */
typedef struct {
__REG32 TDFI_CTRLUPD_MAX                :14;
__REG32                                 : 2;
__REG32 TDFI_CTRLUPD_MIN                : 4;
__REG32                                 :12;
} __hw_dram_ctl66_bits;

/* DRAM Control Register 67 (HW_DRAM_CTL67) */
typedef struct {
__REG32 TDFI_CTRL_DELAY                 : 4;
__REG32                                 : 4;
__REG32 DRAM_CLK_ENABLE                 : 4;
__REG32                                 : 4;
__REG32 TDFI_DRAM_CLK_DISABLE           : 3;
__REG32                                 : 5;
__REG32 TDFI_DRAM_CLK_ENABLE            : 4;
__REG32                                 : 4;
} __hw_dram_ctl67_bits;

/* DRAM Control Register 68 (HW_DRAM_CTL68) */
typedef struct {
__REG32 TDFI_PHYUPD_RESP                :14;
__REG32                                 : 2;
__REG32 TDFI_PHYUPD_TYPE0               :14;
__REG32                                 : 2;
} __hw_dram_ctl68_bits;

/* DRAM Control Register 69 (HW_DRAM_CTL69) */
typedef struct {
__REG32 TDFI_PHY_WRLAT                  : 4;
__REG32                                 : 4;
__REG32 TDFI_PHY_WRLAT_BASE             : 4;
__REG32                                 :20;
} __hw_dram_ctl69_bits;

/* DRAM Control Register 70 (HW_DRAM_CTL70) */
typedef struct {
__REG32 TDFI_PHY_RDLAT                  : 4;
__REG32                                 : 4;
__REG32 TDFI_RDDATA_EN                  : 4;
__REG32                                 : 4;
__REG32 TDFI_RDDATA_EN_BASE             : 4;
__REG32                                 :12;
} __hw_dram_ctl70_bits;

/* DRAM Control Register 81 (HW_DRAM_CTL81) */
typedef struct {
__REG32 OCD_ADJUST_PDN_CS_0             : 5;
__REG32                                 : 3;
__REG32 OCD_ADJUST_PUP_CS_0             : 5;
__REG32                                 :19;
} __hw_dram_ctl81_bits;

/* DRAM Control Register 82 (HW_DRAM_CTL82) */
typedef struct {
__REG32                                 :24;
__REG32 ODT_ALT_EN                      : 1;
__REG32                                 : 7;
} __hw_dram_ctl82_bits;

/* DRAM Control Register 83 (HW_DRAM_CTL83) */
typedef struct {
__REG32 ODT_RD_MAP_CS0                  : 4;
__REG32                                 : 4;
__REG32 ODT_RD_MAP_CS1                  : 4;
__REG32                                 : 4;
__REG32 ODT_RD_MAP_CS2                  : 4;
__REG32                                 : 4;
__REG32 ODT_RD_MAP_CS3                  : 4;
__REG32                                 : 4;
} __hw_dram_ctl83_bits;

/* DRAM Control Register 84 (HW_DRAM_CTL84) */
typedef struct {
__REG32 ODT_WR_MAP_CS0                  : 4;
__REG32                                 : 4;
__REG32 ODT_WR_MAP_CS1                  : 4;
__REG32                                 : 4;
__REG32 ODT_WR_MAP_CS2                  : 4;
__REG32                                 : 4;
__REG32 ODT_WR_MAP_CS3                  : 4;
__REG32                                 : 4;
} __hw_dram_ctl84_bits;

/* DRAM Control Register 86 (HW_DRAM_CTL86) */
typedef struct {
__REG32 VERSION                         :16;
__REG32                                 :16;
} __hw_dram_ctl86_bits;

/* DRAM Control Register 107 (HW_DRAM_CTL107) */
typedef struct {
__REG32 DLL_OBS_REG_1_0                 : 8;
__REG32                                 :24;
} __hw_dram_ctl107_bits;

/* DRAM Control Register 112 (HW_DRAM_CTL112) */
typedef struct {
__REG32 DLL_OBS_REG_1_1                 : 8;
__REG32                                 :24;
} __hw_dram_ctl112_bits;

/* DRAM Control Register 117 (HW_DRAM_CTL117) */
typedef struct {
__REG32 DLL_OBS_REG_1_2                 : 8;
__REG32                                 :24;
} __hw_dram_ctl117_bits;

/* DRAM Control Register 122 (HW_DRAM_CTL122) */
typedef struct {
__REG32 DLL_OBS_REG_1_3                 : 8;
__REG32                                 :24;
} __hw_dram_ctl122_bits;

/* DRAM Control Register 127 (HW_DRAM_CTL127) */
typedef struct {
__REG32 DLL_OBS_REG_2_0                 : 8;
__REG32                                 :24;
} __hw_dram_ctl127_bits;

/* DRAM Control Register 132 (HW_DRAM_CTL132) */
typedef struct {
__REG32 DLL_OBS_REG_2_1                 : 8;
__REG32                                 :24;
} __hw_dram_ctl132_bits;

/* DRAM Control Register 137 (HW_DRAM_CTL137) */
typedef struct {
__REG32 DLL_OBS_REG_2_2                 : 8;
__REG32                                 :24;
} __hw_dram_ctl137_bits;

/* DRAM Control Register 142 (HW_DRAM_CTL142) */
typedef struct {
__REG32 DLL_OBS_REG_2_3                 : 8;
__REG32                                 :24;
} __hw_dram_ctl142_bits;

/* DRAM Control Register 147 (HW_DRAM_CTL147) */
typedef struct {
__REG32 DLL_OBS_REG_3_0                 : 8;
__REG32                                 :24;
} __hw_dram_ctl147_bits;

/* DRAM Control Register 152 (HW_DRAM_CTL152) */
typedef struct {
__REG32 DLL_OBS_REG_3_1                 : 8;
__REG32                                 :24;
} __hw_dram_ctl152_bits;

/* DRAM Control Register 157 (HW_DRAM_CTL157) */
typedef struct {
__REG32 DLL_OBS_REG_3_2                 : 8;
__REG32                                 :24;
} __hw_dram_ctl157_bits;

/* DRAM Control Register 162 (HW_DRAM_CTL162) */
typedef struct {
__REG32 DLL_OBS_REG_3_3                 : 8;
__REG32                                 : 8;
__REG32 W2R_DIFFCS_DLY                  : 3;
__REG32                                 : 5;
__REG32 W2R_SAMECS_DLY                  : 3;
__REG32                                 : 5;
} __hw_dram_ctl162_bits;

/* DRAM Control Register 163 (HW_DRAM_CTL163) */
typedef struct {
__REG32 DRAM_CLASS                      : 4;
__REG32                                 : 4;
__REG32 RDLAT_ADJ                       : 4;
__REG32                                 : 4;
__REG32 WRLAT_ADJ                       : 4;
__REG32                                 : 4;
__REG32 DLL_RST_ADJ_DLY                 : 8;
} __hw_dram_ctl163_bits;

/* DRAM Control Register 164 (HW_DRAM_CTL164) */
typedef struct {
__REG32 TMOD                            : 8;
__REG32 INT_ACK                         :10;
__REG32                                 :14;
} __hw_dram_ctl164_bits;

/* DRAM Control Register 171 (HW_DRAM_CTL171) */
typedef struct {
__REG32 DLL_RST_DELAY                   :16;
__REG32 AXI4_BDW_OVFLOW                 : 1;
__REG32                                 : 7;
__REG32 AXI5_BDW_OVFLOW                 : 1;
__REG32                                 : 7;
} __hw_dram_ctl171_bits;

/* DRAM Control Register 172 (HW_DRAM_CTL172) */
typedef struct {
__REG32 CKE_STATUS                      : 1;
__REG32                                 : 7;
__REG32 CONCURRENTAP_WR_ONLY            : 1;
__REG32                                 : 7;
__REG32 RESYNC_DLL                      : 1;
__REG32                                 : 7;
__REG32 RESYNC_DLL_PER_AREF_EN          : 1;
__REG32                                 : 7;
} __hw_dram_ctl172_bits;

/* DRAM Control Register 173 (HW_DRAM_CTL173) */
typedef struct {
__REG32 AXI4_FIFO_TYPE_REG              : 2;
__REG32                                 : 6;
__REG32 AXI5_FIFO_TYPE_REG              : 2;
__REG32                                 : 6;
__REG32 AXI4_R_PRIORITY                 : 3;
__REG32                                 : 5;
__REG32 AXI4_W_PRIORITY                 : 3;
__REG32                                 : 5;
} __hw_dram_ctl173_bits;

/* DRAM Control Register 174 (HW_DRAM_CTL174) */
typedef struct {
__REG32 AXI5_R_PRIORITY                 : 3;
__REG32                                 : 5;
__REG32 AXI5_W_PRIORITY                 : 3;
__REG32                                 : 5;
__REG32 R2R_DIFFCS_DLY                  : 3;
__REG32                                 : 5;
__REG32 R2R_SAMECS_DLY                  : 3;
__REG32                                 : 5;
} __hw_dram_ctl174_bits;

/* DRAM Control Register 175 (HW_DRAM_CTL175) */
typedef struct {
__REG32 R2W_DIFFCS_DLY                  : 3;
__REG32                                 : 5;
__REG32 R2W_SAMECS_DLY                  : 3;
__REG32                                 : 5;
__REG32 TBST_INT_INTERVAL               : 3;
__REG32                                 : 5;
__REG32 W2W_DIFFCS_DLY                  : 3;
__REG32                                 : 5;
} __hw_dram_ctl175_bits;

/* DRAM Control Register 176 (HW_DRAM_CTL176) */
typedef struct {
__REG32 W2W_SAMECS_DLY                  : 3;
__REG32                                 : 5;
__REG32 ADD_ODT_CLK_DIFFTYPE_DIFFCS     : 4;
__REG32                                 : 4;
__REG32 ADD_ODT_CLK_DIFFTYPE_SAMECS     : 4;
__REG32                                 : 4;
__REG32 ADD_ODT_CLK_SAMETYPE_DIFFCS     : 4;
__REG32                                 : 4;
} __hw_dram_ctl176_bits;

/* DRAM Control Register 177 (HW_DRAM_CTL177) */
typedef struct {
__REG32 CKSRE                           : 4;
__REG32                                 : 4;
__REG32 CKSRX                           : 4;
__REG32                                 : 4;
__REG32 TRP_AB                          : 4;
__REG32                                 : 4;
__REG32 TCCD                            : 4;
__REG32                                 : 4;
} __hw_dram_ctl177_bits;

/* DRAM Control Register 178 (HW_DRAM_CTL178) */
typedef struct {
__REG32 TCKESR                          : 5;
__REG32                                 : 3;
__REG32 AXI4_BDW                        : 7;
__REG32                                 : 1;
__REG32 AXI4_CURRENT_BDW                : 7;
__REG32                                 : 1;
__REG32 AXI5_BDW                        : 7;
__REG32                                 : 1;
} __hw_dram_ctl178_bits;

/* DRAM Control Register 179 (HW_DRAM_CTL179) */
typedef struct {
__REG32 AXI5_CURRENT_BDW                : 7;
__REG32                                 : 1;
__REG32 TDFI_PHYUPD_TYPE1               :14;
__REG32                                 :10;
} __hw_dram_ctl179_bits;

/* DRAM Control Register 180 (HW_DRAM_CTL180) */
typedef struct {
__REG32 TDFI_PHYUPD_TYPE2               :14;
__REG32                                 : 2;
__REG32 TDFI_PHYUPD_TYPE3               :14;
__REG32                                 : 2;
} __hw_dram_ctl180_bits;

/* DRAM Control Register 181 (HW_DRAM_CTL181) */
typedef struct {
__REG32 MR0_DATA_0                      :15;
__REG32                                 : 1;
__REG32 MR0_DATA_1                      :15;
__REG32                                 : 1;
} __hw_dram_ctl181_bits;

/* DRAM Control Register 182 (HW_DRAM_CTL182) */
typedef struct {
__REG32 MR0_DATA_2                      :15;
__REG32                                 : 1;
__REG32 MR0_DATA_3                      :15;
__REG32                                 : 1;
} __hw_dram_ctl182_bits;

/* DRAM Control Register 183 (HW_DRAM_CTL183) */
typedef struct {
__REG32 MR1_DATA_0                      :15;
__REG32                                 : 1;
__REG32 MR1_DATA_1                      :15;
__REG32                                 : 1;
} __hw_dram_ctl183_bits;

/* DRAM Control Register 184 (HW_DRAM_CTL184) */
typedef struct {
__REG32 MR1_DATA_2                      :15;
__REG32                                 : 1;
__REG32 MR1_DATA_3                      :15;
__REG32                                 : 1;
} __hw_dram_ctl184_bits;

/* DRAM Control Register 185 (HW_DRAM_CTL185) */
typedef struct {
__REG32 MR2_DATA_0                      :15;
__REG32                                 : 1;
__REG32 MR2_DATA_1                      :15;
__REG32                                 : 1;
} __hw_dram_ctl185_bits;

/* DRAM Control Register 186 (HW_DRAM_CTL186) */
typedef struct {
__REG32 MR2_DATA_2                      :15;
__REG32                                 : 1;
__REG32 MR2_DATA_3                      :15;
__REG32                                 : 1;
} __hw_dram_ctl186_bits;

/* DRAM Control Register 187 (HW_DRAM_CTL187) */
typedef struct {
__REG32 MR3_DATA_0                      :15;
__REG32                                 : 1;
__REG32 MR3_DATA_1                      :15;
__REG32                                 : 1;
} __hw_dram_ctl187_bits;

/* DRAM Control Register 188 (HW_DRAM_CTL188) */
typedef struct {
__REG32 MR3_DATA_2                      :15;
__REG32                                 : 1;
__REG32 MR3_DATA_3                      :15;
__REG32                                 : 1;
} __hw_dram_ctl188_bits;

/* DRAM Control Register 189 (HW_DRAM_CTL189) */
typedef struct {
__REG32 AXI4_EN_SIZE_LT_WIDTH_INSTR     :16;
__REG32 AXI5_EN_SIZE_LT_WIDTH_INSTR     :16;
} __hw_dram_ctl189_bits;

/* GPMI Control Register 0 */
typedef struct {
__REG32 XFER_COUNT          :16;
__REG32 ADDRESS_INCREMENT   : 1;
__REG32 ADDRESS             : 3;
__REG32 CS                  : 3;
__REG32 WORD_LENGTH         : 1;
__REG32 COMMAND_MODE        : 2;
__REG32                     : 1;
__REG32 LOCK_CS             : 1;
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
__REG32 ABORT_WAIT_FOR_READY  : 3;
__REG32 ABORT_WAIT_REQUEST    : 1;
__REG32 BURST_EN              : 1;
__REG32 TIMEOUT_IRQ           : 1;
__REG32                       : 1;
__REG32 DMA2ECC_MODE          : 1;
__REG32 RDN_DELAY             : 4;
__REG32 HALF_PERIOD           : 1;
__REG32 DLL_ENABLE            : 1;
__REG32 BCH_MODE              : 1;
__REG32 GANGED_RDYBUSY        : 1;
__REG32 TIMEOUT_IRQ_EN        : 1;
__REG32                       : 1;
__REG32 WRN_DLY_SEL           : 2;
__REG32 DECOUPLE_CS           : 1;
__REG32                       : 7;
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
__REG32 PRESENT               : 1;
__REG32 FIFO_FULL             : 1;
__REG32 FIFO_EMPTY            : 1;
__REG32 INVALID_BUFFER_MASK   : 1;
__REG32 GPMI_RDY1             : 1;
__REG32                       : 3;
__REG32 DEV0_ERROR            : 1;
__REG32 DEV1_ERROR            : 1;
__REG32 DEV2_ERROR            : 1;
__REG32 DEV3_ERROR            : 1;
__REG32 DEV4_ERROR            : 1;
__REG32 DEV5_ERROR            : 1;
__REG32 DEV6_ERROR            : 1;
__REG32 DEV7_ERROR            : 1;
__REG32 RDY_TIMEOUT           : 8;
__REG32 READY_BUSY            : 8;
} __hw_gpmi_stat_bits;

/* GPMI Debug Information Register */
typedef struct {
__REG32 CMD_END               : 8;
__REG32 DMAREQ                : 8;
__REG32 DMA_SENSE             : 8;
__REG32 WAIT_FOR_READY_END    : 8;
} __hw_gpmi_debug_bits;

/* GPMI Version Register */
typedef struct {
__REG32 STEP                  :16;
__REG32 MINOR                 : 8;
__REG32 MAJOR                 : 8;
} __hw_gpmi_version_bits;

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

/* SSP Control Register 0 (HW_SSP_CTRL0) */
typedef struct {
__REG32                       :16;
__REG32 ENABLE                : 1;
__REG32 GET_RESP              : 1;
__REG32 CHECK_RESP            : 1;
__REG32 LONG_RESP             : 1;
__REG32 WAIT_FOR_CMD          : 1;
__REG32 WAIT_FOR_IRQ          : 1;
__REG32 BUS_WIDTH             : 2;
__REG32 DATA_XFER             : 1;
__REG32 READ                  : 1;
__REG32 IGNORE_CRC            : 1;
__REG32 LOCK_CS               : 1;
__REG32 SDIO_IRQ_CHECK        : 1;
__REG32 RUN                   : 1;
__REG32 CLKGATE               : 1;
__REG32 SFTRST                : 1;
} __hw_ssp_ctrl0_bits;

/* SD/MMC Command Register 0 (HW_SSP_CMD0) */
typedef struct {
__REG32 CMD                   : 8;
__REG32                       :12;
__REG32 APPEND_8CYC           : 1;
__REG32 CONT_CLKING_EN        : 1;
__REG32 SLOW_CLKING_EN        : 1;
__REG32 BOOT_ACK_EN           : 1;
__REG32 PRIM_BOOT_OP_EN       : 1;
__REG32 DBL_DATA_RATE_EN      : 1;
__REG32 SOFT_TERMINATE        : 1;
__REG32                       : 5;
} __hw_ssp_cmd0_bits;

/* SD/MMC BLOCK SIZE and COUNT Register (HW_SSP_BLOCK_SIZE) */
typedef struct {
__REG32 BLOCK_SIZE            : 4;
__REG32 APPEND_8CYC           :24;
__REG32                       : 4;
} __hw_ssp_block_size_bits;

/* SSP Timing Register (HW_SSP_TIMING) */
typedef struct {
__REG32 CLOCK_RATE            : 8;
__REG32 CLOCK_DIVIDE          : 8;
__REG32 TIMEOUT               :16;
} __hw_ssp_timing_bits;

/* SSP Control Register 1 (HW_SSP_CTRL1) */
typedef struct {
__REG32 SSP_MODE              : 4;
__REG32 WORD_LENGTH           : 4;
__REG32 SLAVE_MODE            : 1;
__REG32 POLARITY              : 1;
__REG32 PHASE                 : 1;
__REG32 SLAVE_OUT_DISABLE     : 1;
__REG32                       : 1;
__REG32 DMA_ENABLE            : 1;
__REG32 FIFO_OVERRUN_IRQ_EN   : 1;
__REG32 FIFO_OVERRUN_IRQ      : 1;
__REG32 RECV_TIMEOUT_IRQ_EN   : 1;
__REG32 RECV_TIMEOUT_IRQ      : 1;
__REG32                       : 2;
__REG32 FIFO_UNDERRUN_EN      : 1;
__REG32 FIFO_UNDERRUN_IRQ     : 1;
__REG32 DATA_CRC_IRQ_EN       : 1;
__REG32 DATA_CRC_IRQ          : 1;
__REG32 DATA_TIMEOUT_IRQ_EN   : 1;
__REG32 DATA_TIMEOUT_IRQ      : 1;
__REG32 RESP_TIMEOUT_IRQ_EN   : 1;
__REG32 RESP_TIMEOUT_IRQ      : 1;
__REG32 RESP_ERR_IRQ_EN       : 1;
__REG32 RESP_ERR_IRQ          : 1;
__REG32 SDIO_IRQ_EN           : 1;
__REG32 SDIO_IRQ              : 1;
} __hw_ssp_ctrl1_bits;

/* SD/MMC Double Data Rate Control Register (HW_SSP_DDR_CTRL) */
typedef struct {
__REG32 TXCLK_DELAY_TYPE      : 1;
__REG32 NIBBLE_POS            : 1;
__REG32                       :28;
__REG32 DMA_BURST_TYPE        : 2;
} __hw_ssp_ddr_ctrl_bits;

/* SD/MMC DLL Control Register (HW_SSP_DLL_CTRL) */
typedef struct {
__REG32 ENABLE                : 1;
__REG32 RESET                 : 1;
__REG32 SLV_FORCE_UPD         : 1;
__REG32 SLV_DLY_TARGET        : 4;
__REG32 GATE_UPDATE           : 1;
__REG32                       : 1;
__REG32 SLV_OVERRIDE          : 1;
__REG32 SLV_OVERRIDE_VAL      : 6;
__REG32                       : 4;
__REG32 SLV_UPDATE_INT        : 8;
__REG32 REF_UPDATE_INT        : 4;
} __hw_ssp_dll_ctrl_bits;

/* SSP Status Register (HW_SSP_STATUS) */
typedef struct {
__REG32 BUSY                  : 1;
__REG32                       : 1;
__REG32 DATA_BUSY             : 1;
__REG32 CMD_BUSY              : 1;
__REG32 FIFO_UNDRFLW          : 1;
__REG32 FIFO_EMPTY            : 1;
__REG32                       : 2;
__REG32 FIFO_FULL             : 1;
__REG32 FIFO_OVRFLW           : 1;
__REG32                       : 1;
__REG32 RECV_TIMEOUT_STAT     : 1;
__REG32 TIMEOUT               : 1;
__REG32 DATA_CRC_ERR          : 1;
__REG32 RESP_TIMEOUT          : 1;
__REG32 RESP_ERR              : 1;
__REG32 RESP_CRC_ERR          : 1;
__REG32 SDIO_IRQ              : 1;
__REG32 DMAEND                : 1;
__REG32 DMAREQ                : 1;
__REG32 DMATERM               : 1;
__REG32 DMASENSE              : 1;
__REG32 DMABURST              : 1;
__REG32                       : 5;
__REG32 CARD_DETECT           : 1;
__REG32 SD_PRESENT            : 1;
__REG32                       : 1;
__REG32 PRESENT               : 1;
} __hw_ssp_status_bits;

/* SD/MMC DLL Status Register (HW_SSP_DLL_STS) */
typedef struct {
__REG32 SLV_LOCK              : 1;
__REG32 REF_LOCK              : 1;
__REG32 SLV_SEL               : 6;
__REG32 REF_SEL               : 6;
__REG32                       :18;
} __hw_ssp_dll_sts_bits;

/* SSP Debug Register (HW_SSP_DEBUG) */
typedef struct {
__REG32 SSP_RXD               : 8;
__REG32 SSP_RESP              : 1;
__REG32 SSP_CMD               : 1;
__REG32 CMD_SM                : 2;
__REG32 MMC_SM                : 4;
__REG32 DMA_SM                : 3;
__REG32 CMD_OE                : 1;
__REG32                       : 4;
__REG32 DAT_SM                : 3;
__REG32 DATA_STALL            : 1;
__REG32 DATACRC_ERR           : 4;
} __hw_ssp_debug_bits;

/* SSP Version Register (HW_SSP_VERSION) */
typedef struct {
__REG32 STEP                  :16;
__REG32 MINOR                 : 8;
__REG32 MAJOR                 : 8;
} __hw_ssp_version_bits;

/* DIGCTL Control Register */
typedef struct {
__REG32 LATCH_ENTROPY           : 1;
__REG32 JTAG_SHIELD             : 1;
__REG32 USB_CLKGATE             : 1;
__REG32 DEBUG_DISABLE           : 1;
__REG32                         : 6;
__REG32 SAIF_CLKMUX_SEL         : 2;
__REG32                         : 1;
__REG32 AUART01_LOOPBACK        : 1;
__REG32 DUART_LOOPBACK          : 1;
__REG32 SAIF_LOOPBACK           : 1;
__REG32 USB1_CLKGATE            : 1;
__REG32 DIGITAL_TESTMODE        : 1;
__REG32 ANALOG_TESTMODE         : 1;
__REG32 USB0_TESTMODE           : 1;
__REG32 USB1_TESTMODE           : 1;
__REG32 USB0_OVERCURRENT_POL    : 1;
__REG32 USB1_OVERCURRENT_POL    : 1;
__REG32 USB0_OVERCURRENT_ENABLE : 1;
__REG32 USB1_OVERCURRENT_ENABLE : 1;
__REG32                         : 5;
__REG32 XTAL24M_GATE            : 1;
__REG32                         : 1;
} __hw_digctl_ctrl_bits;

/* DIGCTL Status Register */
typedef struct {
__REG32 WRITTEN             : 1;
__REG32 PACKAGE_TYPE        : 3;
__REG32 JTAG_IN_USE         : 1;
__REG32                     :19;
__REG32 USB1_DEVICE_PRESENT : 1;
__REG32 USB1_HOST_PRESENT   : 1;
__REG32 USB1_OTG_PRESENT    : 1;
__REG32 USB1_HS_PRESENT     : 1;
__REG32 USB0_DEVICE_PRESENT : 1;
__REG32 USB0_HOST_PRESENT   : 1;
__REG32 USB0_OTG_PRESENT    : 1;
__REG32 USB0_HS_PRESENT     : 1;
} __hw_digctl_status_bits;

/* On-Chip RAM Control Register */
typedef struct {
__REG32                     : 8;
__REG32 DEBUG_CODE          : 5;
__REG32 DEBUG_ENABLE        : 1;
__REG32                     :18;
} __hw_digctl_ramctrl_bits;

/* On-Chip RAM Repair Address Register */
typedef struct {
__REG32 ADDR                :16;
__REG32                     :16;
} __hw_digctl_ramrepair_bits;

/* EMI Status Register (HW_DIGCTL_EMI_STATUS) */
typedef struct {
__REG32 POWER_MODE          : 5;
__REG32                     :27;
} __hw_digctl_emi_status_bits;

/* On-Chip Memories Read Margin Register (HW_DIGCTL_READ_MARGIN) */
typedef struct {
__REG32 ROM                 : 4;
__REG32                     :28;
} __hw_digctl_read_margin_bits;

/* BIST Control Register (HW_DIGCTL_BIST_CTL) */
typedef struct {
__REG32 CAN_BIST_START      : 1;
__REG32 CACHE_BIST_START    : 1;
__REG32 DMA0_BIST_START     : 1;
__REG32 DMA1_BIST_START     : 1;
__REG32 USB0_BIST_START     : 1;
__REG32 USB1_BIST_START     : 1;
__REG32 ENET_BIST_START     : 1;
__REG32 DCP_BIST_START      : 1;
__REG32 LCDIF_BIST_START    : 1;
__REG32 PXP_BIST_START      : 1;
__REG32 OCRAM_BIST_START    : 1;
__REG32 OCRAM_BIST_DONE     : 1;
__REG32 OCRAM_BIST_FAIL     : 1;
__REG32 OCRAM_BIST_PASS     : 1;
__REG32 OCRAM_BIST_RETENTION: 1;
__REG32                     :12;
__REG32 BIST_RESUME         : 1;
__REG32 BIST_CHECKB         : 1;
__REG32 BIST_DEBUGZ         : 1;
__REG32 BIST_RESETN         : 1;
__REG32 BIST_TESTMODE       : 1;
} __hw_digctl_bist_ctl_bits;

/* DIGCTL Status Register (HW_DIGCTL_BIST_STATUS) */
typedef struct {
__REG32 CAN_BIST_DONE       : 1;
__REG32 CACHE_BIST_DONE     : 1;
__REG32 DMA0_BIST_DONE      : 1;
__REG32 DMA1_BIST_DONE      : 1;
__REG32 USB0_BIST_DONE      : 1;
__REG32 USB1_BIST_DONE      : 1;
__REG32 ENET_BIST_DONE      : 1;
__REG32 DCP_BIST_DONE       : 1;
__REG32 LCDIF_BIST_DONE     : 1;
__REG32 PXP_BIST_DONE       : 1;
__REG32 CAN_BIST_FAIL       : 1;
__REG32 CACHE_BIST_FAIL     : 1;
__REG32 DMA0_BIST_FAIL      : 1;
__REG32 DMA1_BIST_FAIL      : 1;
__REG32 USB0_BIST_FAIL      : 1;
__REG32 USB1_BIST_FAIL      : 1;
__REG32 ENET_BIST_FAIL      : 1;
__REG32 DCP_BIST_FAIL       : 1;
__REG32 LCDIF_BIST_FAIL     : 1;
__REG32 PXP_BIST_FAIL       : 1;
__REG32 CAN_BIST_RETENTION  : 1;
__REG32 CACHE_BIST_RETENTION: 1;
__REG32 DMA0_BIST_RETENTION : 1;
__REG32 DMA1_BIST_RETENTION : 1;
__REG32 USB0_BIST_RETENTION : 1;
__REG32 USB1_BIST_RETENTION : 1;
__REG32 ENET_BIST_RETENTION : 1;
__REG32 DCP_BIST_RETENTION  : 1;
__REG32 LCDIF_BIST_RETENTION: 1;
__REG32 PXP_BIST_RETENTION  : 1;
__REG32                     : 2;
} __hw_digctl_bist_status_bits;

/* USB LOOP BACK (HW_DIGCTL_USB_LOOPBACK) */
typedef struct {
__REG32 UTMO0_DIG_TST0      : 1;
__REG32 UTMO0_DIG_TST1      : 1;
__REG32 UTMO1_DIG_TST0      : 1;
__REG32 UTMO1_DIG_TST1      : 1;
__REG32 UTMI0_DIG_TST0      : 1;
__REG32 UTMI0_DIG_TST1      : 1;
__REG32 TSTI0_TX_HIZ        : 1;
__REG32 TSTI0_TX_EN         : 1;
__REG32 TSTI0_TX_HS         : 1;
__REG32 TSTI0_TX_LS         : 1;
__REG32 USB0_TST_START      : 1;
__REG32 UTMI1_DIG_TST0      : 1;
__REG32 UTMI1_DIG_TST1      : 1;
__REG32 TSTI1_TX_HIZ        : 1;
__REG32 TSTI1_TX_EN         : 1;
__REG32 TSTI1_TX_HS         : 1;
__REG32 TSTI1_TX_LS         : 1;
__REG32 USB1_TST_START      : 1;
__REG32                     :14;
} __hw_digctl_usb_loopback_bits;

/* SRAM Status Register 8 */
typedef struct {
__REG32 FAILADDR00          :16;
__REG32 FAILADDR01          :16;
} __hw_digctl_ocram_status8_bits;

/* SRAM Status Register 9 */
typedef struct {
__REG32 FAILADDR10          :16;
__REG32 FAILADDR11          :16;
} __hw_digctl_ocram_status9_bits;

/* SRAM Status Register 10 */
typedef struct {
__REG32 FAILADDR20          :16;
__REG32 FAILADDR21          :16;
} __hw_digctl_ocram_status10_bits;

/* SRAM Status Register 11 */
typedef struct {
__REG32 FAILADDR30          :16;
__REG32 FAILADDR31          :16;
} __hw_digctl_ocram_status11_bits;

/* SRAM Status Register 12 */
typedef struct {
__REG32 FAILSTATE00         : 7;
__REG32                     : 1;
__REG32 FAILSTATE01         : 7;
__REG32                     : 1;
__REG32 FAILSTATE10         : 7;
__REG32                     : 1;
__REG32 FAILSTATE11         : 7;
__REG32                     : 1;
} __hw_digctl_ocram_status12_bits;

/* SRAM Status Register 13 */
typedef struct {
__REG32 FAILSTATE20         : 7;
__REG32                     : 1;
__REG32 FAILSTATE21         : 7;
__REG32                     : 1;
__REG32 FAILSTATE30         : 7;
__REG32                     : 1;
__REG32 FAILSTATE31         : 7;
__REG32                     : 1;
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

/* Debug Trap Control and Status for AHB Layer 0 and 3 (HW_DIGCTL_DEBUG_TRAP) */
typedef struct {
__REG32 TRAP_ENABLE         : 1;
__REG32 TRAP_IN_RANGE       : 1;
__REG32 TRAP_L0_IRQ         : 1;
__REG32 TRAP_L3_IRQ         : 1;
__REG32 TRAP_L3_MASTER_ID   : 3;
__REG32                     : 1;
__REG32 TRAP_L0_MASTER_ID   : 2;
__REG32                     :22;
} __hw_digctl_debug_trap_bits;

/* AHB Statistics Control Register (HW_DIGCTL_AHB_STATS_SELECT) */
typedef struct {
__REG32 L1_MASTER_SELECT    : 8;
__REG32 L2_MASTER_SELECT    : 8;
__REG32 L3_MASTER_SELECT    : 8;
__REG32                     : 8;
} __hw_digctl_ahb_stats_select_bits;

/* Default First Level Page Table Movable PTE Locator 0 (HW_DIGCTL_MPTE0_LOC) */
typedef struct {
__REG32 LOC                 :12;
__REG32                     :12;
__REG32 SPAN                : 3;
__REG32                     : 4;
__REG32 DIS               : 1;
} __hw_digctl_mpte0_loc_bits;

/* Default First Level Page Table Movable PTE Locator n (HW_DIGCTL_MPTEn_LOC) */
typedef struct {
__REG32 LOC                 :12;
__REG32                     :12;
__REG32 SPAN                : 3;
__REG32                     : 4;
__REG32 DIS               : 1;
} __hw_digctl_mpte_loc_bits;

/* OTP Controller Control Register */
typedef struct {
__REG32 ADDR                : 6;
__REG32                     : 2;
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
__REG32                         :29;
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
__REG32 SRK_SHADOW              : 1;
__REG32 UNALLOCATED             : 3;
__REG32 SRK                     : 1;
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
__REG32 USE_ALT_DEBUG_UART_PINS     : 2;
__REG32 DISABLE_RECOVERY_MODE       : 1;
__REG32                             : 1;
__REG32 ENABLE_UNENCRYPTED_BOOT     : 1;
__REG32 ENABLE_USB_BOOT_SERIAL_NUM  : 1;
__REG32 DISABLE_SPI_NOR_FAST_READ   : 1;
__REG32 EMMC_USE_DDR                : 1;
__REG32 SSP_SCK_INDEX               : 4;
__REG32 SD_BUS_WIDTH                : 2;
__REG32 SD_POWER_UP_DELAY           : 6;
__REG32 SD_POWER_GATE_GPIO          : 2;
__REG32 SD_MMC_MODE                 : 2;
__REG32 BOOT_MODE                   : 8;
} __hw_ocotp_rom0_bits;

/* Shadow Register for OTP Bank3 Word1 (ROM Use 1) */
typedef struct {
__REG32 NUMBER_OF_NANDS             : 3;
__REG32                             : 1;
__REG32 BOOT_SEARCH_STRIDE          : 4;
__REG32 BOOT_SEARCH_COUNT           : 4;
__REG32                             : 1;
__REG32 SD_INIT_SEQ_1_DISABLE       : 1;
__REG32 SD_CMD0_DISABLE             : 1;
__REG32 SD_INIT_SEQ_2_ENABLE        : 1;
__REG32 SD_INCREASE_INIT_SEQ_TIME   : 1;
__REG32 SSP0_EXT_PULLUP             : 1;
__REG32 SSP1_EXT_PULLUP             : 1;
__REG32 UNTOUCH_INTERNAL_SSP_PULLUP : 1;
__REG32 ENABLE_NAND0_CE_RDY_PULLUP  : 1;
__REG32 ENABLE_NAND1_CE_RDY_PULLUP  : 1;
__REG32 ENABLE_NAND2_CE_RDY_PULLUP  : 1;
__REG32 ENABLE_NAND3_CE_RDY_PULLUP  : 1;
__REG32 ENABLE_NAND4_CE_RDY_PULLUP  : 1;
__REG32 ENABLE_NAND5_CE_RDY_PULLUP  : 1;
__REG32 ENABLE_NAND6_CE_RDY_PULLUP  : 1;
__REG32 ENABLE_NAND7_CE_RDY_PULLUP  : 1;
__REG32 SSP2_EXT_PULLUP             : 1;
__REG32 SSP3_EXT_PULLUP             : 1;
__REG32                             : 2;
} __hw_ocotp_rom1_bits;

/* Shadow Register for OTP Bank3 Word2 (ROM Use 2) */
typedef struct {
__REG32 USB_PID                     :16;
__REG32 USB_VID                     :16;
} __hw_ocotp_rom2_bits;

/* Shadow Register for OTP Bank3 Word3 (ROM Use 3) (HW_OCOTP_ROM3) */
typedef struct {
__REG32                             :10;
__REG32 ALT_FAST_BOOT               : 1;
__REG32 USB_VID                     : 1;
__REG32                             :20;
} __hw_ocotp_rom3_bits;

/* Shadow Register for OTP Bank3 Word4 (ROM Use 4) (HW_OCOTP_ROM4) */
typedef struct {
__REG32 NAND_ROW_ADDRESS_BYTES      : 4;
__REG32 NAND_COLUMN_ADDRESS_BYTES   : 4;
__REG32 NAND_READ_CMD_CODE1         : 8;
__REG32 NAND_READ_CMD_CODE2         : 8;
__REG32                             : 7;
__REG32 NAND_BMR                    : 1;
} __hw_ocotp_rom4_bits;

/* Shadow Register for OTP Bank3 Word7 (ROM Use 7) */
typedef struct {
__REG32 ENABLE_PIN_BOOT_CHECK       : 1;
__REG32 MMU_DISABLE                 : 1;
__REG32 ENABLE_ARM_ICACHE           : 1;
__REG32 I2C_USE_400KHZ              : 1;
__REG32                             : 4;
__REG32 ENABLE_SSP_12MA_DRIVE       : 1;
__REG32 RESET_USB_PHY_AT_STARTUP    : 1;
__REG32                             : 1;
__REG32 HAB_DISABLE                 : 1;
__REG32 RECOVERY_BOOT_MODE          : 8;
__REG32 HAB_CONFIG                  : 2;
__REG32 ARM_PLL_DISABLE             : 1;
__REG32 FORCE_RECOVERY_DISABLE      : 1;
__REG32                             : 8;
} __hw_ocotp_rom7_bits;

/* OTP Controller Version Register */
typedef struct {
__REG32 STEP                        :16;
__REG32 MINOR                       : 8;
__REG32 MAJOR                       : 8;
} __hw_ocotp_version_bits;

/* PerfMon Control Register (HW_PERFMON_CTRL) */
typedef struct {
__REG32 RUN                         : 1;
__REG32 SNAP                        : 1;
__REG32 CLR                         : 1;
__REG32 READ_EN                     : 1;
__REG32 TRAP_ENABLE                 : 1;
__REG32 TRAP_IN_RANGE               : 1;
__REG32 LATENCY_ENABLE              : 1;
__REG32 TRAP_IRQ_EN                 : 1;
__REG32 LATENCY_IRQ_EN              : 1;
__REG32 BUS_ERR_IRQ_EN              : 1;
__REG32 TRAP_IRQ                    : 1;
__REG32 LATENCY_IRQ                 : 1;
__REG32 BUS_ERR_IRQ                 : 1;
__REG32                             : 3;
__REG32 IRQ_MID                     : 8;
__REG32                             : 6;
__REG32 CLKGATE                     : 1;
__REG32 SFTRST                      : 1;
} __hw_perfmon_ctrl_bits;

/* PerfMon Master Enable Register (HW_PERFMON_MASTER_EN) */
typedef struct {
__REG32 MID0                        : 1;
__REG32 MID1                        : 1;
__REG32 MID2                        : 1;
__REG32 MID3                        : 1;
__REG32 MID4                        : 1;
__REG32 MID5                        : 1;
__REG32 MID6                        : 1;
__REG32 MID7                        : 1;
__REG32 MID8                        : 1;
__REG32 MID9                        : 1;
__REG32 MID10                       : 1;
__REG32 MID11                       : 1;
__REG32 MID12                       : 1;
__REG32 MID13                       : 1;
__REG32 MID14                       : 1;
__REG32 MID15                       : 1;
__REG32                             :16;
} __hw_perfmon_master_en_bits;

/* PerfMon Latency Threshold Register (HW_PERFMON_LAT_THRESHOLD) */
typedef struct {
__REG32 VALUE                       :12;
__REG32                             :20;
} __hw_perfmon_lat_threshold_bits;

/* PerfMon Maximum Latency Register (HW_PERFMON_MAX_LATENCY) */
typedef struct {
__REG32 COUNT                       :12;
__REG32                             : 3;
__REG32 TAGID                       : 8;
__REG32 ASIZE                       : 3;
__REG32 ALEN                        : 4;
__REG32 ABURST                      : 2;
} __hw_perfmon_max_latency_bits;

/* PerfMon Debug Register (HW_PERFMON_DEBUG) */
typedef struct {
__REG32 ERR_MID                     : 1;
__REG32 TOTAL_CYCLE_CLR_EN          : 1;
__REG32                             :30;
} __hw_perfmon_debug_bits;

/* PerfMon Version Register (HW_PERFMON_VERSION) */
typedef struct {
__REG32 STEP                        :16;
__REG32 MINOR                       : 8;
__REG32 MAJOR                       : 8;
} __hw_perfmon_version_bits;

/* Real-Time Clock Control Register (HW_RTC_CTRL) */
typedef struct {
__REG32 ALARM_IRQ_EN                : 1;
__REG32 ONEMSEC_IRQ_EN              : 1;
__REG32 ALARM_IRQ                   : 1;
__REG32 ONEMSEC_IRQ                 : 1;
__REG32 WATCHDOGEN                  : 1;
__REG32 FORCE_UPDATE                : 1;
__REG32 SUPPRESS_COPY2ANALOG        : 1;
__REG32                             :23;
__REG32 CLKGATE                     : 1;
__REG32 SFTRST                      : 1;
} __hw_rtc_ctrl_bits;

/* Real-Time Clock Status Register (HW_RTC_STAT)*/
typedef struct {
__REG32                             : 8;
__REG32 NEW_REGS                    : 8;
__REG32 STALE_REGS                  : 8;
__REG32                             : 3;
__REG32 XTAL32768_PRESENT           : 1;
__REG32 XTAL32000_PRESENT           : 1;
__REG32 WATCHDOG_PRESENT            : 1;
__REG32 ALARM_PRESENT               : 1;
__REG32 RTC_PRESENT                 : 1;
} __hw_rtc_stat_bits;

/* Persistent State Register 0 (HW_RTC_PERSISTENT0) */
typedef struct {
__REG32 CLOCKSOURCE                 : 1;
__REG32 ALARM_WAKE_EN               : 1;
__REG32 ALARM_EN                    : 1;
__REG32 LCK_SECS                    : 1;
__REG32 XTAL24MHZ_PWRUP             : 1;
__REG32 XTAL32KHZ_PWRUP             : 1;
__REG32 XTAL32_FREQ                 : 1;
__REG32 ALARM_WAKE                  : 1;
__REG32 MSEC_RES                    : 5;
__REG32 DISABLE_XTALOK              : 1;
__REG32 LOWERBIAS                   : 2;
__REG32 DISABLE_PSWITCH             : 1;
__REG32 AUTO_RESTART                : 1;
__REG32 ENABLE_LRADC_PWRUP          : 1;
__REG32                             : 1;
__REG32 THERMAL_RESET               : 1;
__REG32 EXTERNAL_RESET              : 1;
__REG32                             : 6;
__REG32 ADJ_POSLIMITBUCK            : 4;
} __hw_rtc_persistent0_bits;

/* Real-Time Clock Debug Register (HW_RTC_DEBUG) */
typedef struct {
__REG32 WATCHDOG_RESET              : 1;
__REG32 WATCHDOG_RESET_MASK         : 1;
__REG32                             :30;
} __hw_rtc_debug_bits;

/* Real-Time Clock Version Register (HW_RTC_VERSION) */
typedef struct {
__REG32 STEP                        :16;
__REG32 MINOR                       : 8;
__REG32 MAJOR                       : 8;
} __hw_rtc_version_bits;

/* Rotary Decoder Control Register (HW_TIMROT_ROTCTRL) */
typedef struct {
__REG32 SELECT_A                    : 4;
__REG32 SELECT_B                    : 4;
__REG32 POLARITY_A                  : 1;
__REG32 POLARITY_B                  : 1;
__REG32 OVERSAMPLE                  : 2;
__REG32 RELATIVE                    : 1;
__REG32                             : 3;
__REG32 DIVIDER                     : 6;
__REG32 STATE                       : 3;
__REG32 TIM0_PRESENT                : 1;
__REG32 TIM1_PRESENT                : 1;
__REG32 TIM2_PRESENT                : 1;
__REG32 TIM3_PRESENT                : 1;
__REG32 ROTARY_PRESENT              : 1;
__REG32 CLKGATE                     : 1;
__REG32 SFTRST                      : 1;
} __hw_timrot_rotctrl_bits;

/* Rotary Decoder Up/Down Counter Register (HW_TIMROT_ROTCOUNT) */
typedef struct {
__REG32 UPDOWN                      :16;
__REG32                             :16;
} __hw_timrot_rotcount_bits;

/* Timer 0 Control and Status Register (HW_TIMROT_TIMCTRL0) */
typedef struct {
__REG32 SELECT                      : 4;
__REG32 PRESCALE                    : 2;
__REG32 RELOAD                      : 1;
__REG32 UPDATE                      : 1;
__REG32 POLARITY                    : 1;
__REG32                             : 2;
__REG32 MATCH_MODE                  : 1;
__REG32                             : 2;
__REG32 IRQ_EN                      : 1;
__REG32 IRQ                         : 1;
__REG32                             :16;
} __hw_timrot_timctrl_bits;

/* TIMROT Version Register (HW_TIMROT_VERSION) */
typedef struct {
__REG32 STEP                        :16;
__REG32 MINOR                       : 8;
__REG32 MAJOR                       : 8;
} __hw_timrot_version_bits;

/* UART Data Register (HW_UARTDBG_DR) */
typedef struct {
__REG32 DATA                        : 8;
__REG32 FE                          : 1;
__REG32 PE                          : 1;
__REG32 BE                          : 1;
__REG32 OE                          : 1;
__REG32                             :20;
} __hw_uartdbg_dr_bits;

/* UART Receive Status Register (Read) / Error Clear Register (Write) (HW_UARTDBG_ECR) */
typedef struct {
__REG32 FE                          : 1;
__REG32 PE                          : 1;
__REG32 BE                          : 1;
__REG32 OE                          : 1;
__REG32 EC                          : 4;
__REG32                             :24;
} __hw_uartdbg_ecr_bits;

/* UART Flag Register (HW_UARTDBG_FR) */
typedef struct {
__REG32 CTS                         : 1;
__REG32 DSR                         : 1;
__REG32 DCD                         : 1;
__REG32 BUSY                        : 1;
__REG32 RXFE                        : 1;
__REG32 TXFF                        : 1;
__REG32 RXFF                        : 1;
__REG32 TXFE                        : 1;
__REG32 RI                          : 1;
__REG32                             :23;
} __hw_uartdbg_fr_bits;

/* UART IrDA Low-Power Counter Register (HW_UARTDBG_ILPR) */
typedef struct {
__REG32 ILPDVSR                     : 8;
__REG32                             :24;
} __hw_uartdbg_ilpr_bits;

/* UART Integer Baud Rate Divisor Register (HW_UARTDBG_IBRD) */
typedef struct {
__REG32 BAUD_DIVINT                 :16;
__REG32                             :16;
} __hw_uartdbg_ibrd_bits;

/* UART Fractional Baud Rate Divisor Register (HW_UARTDBG_FBRD) */
typedef struct {
__REG32 BAUD_DIVFRAC                : 6;
__REG32                             :26;
} __hw_uartdbg_fbrd_bits;

/* UART Line Control Register, HIGH Byte (HW_UARTDBG_H) */
typedef struct {
__REG32 BRK                         : 1;
__REG32 PEN                         : 1;
__REG32 EPS                         : 1;
__REG32 STP2                        : 1;
__REG32 FEN                         : 1;
__REG32 WLEN                        : 2;
__REG32 SPS                         : 1;
__REG32                             :24;
} __hw_uartdbg_h_bits;

/* UART Control Register (HW_UARTDBG_CR) */
typedef struct {
__REG32 UARTEN                      : 1;
__REG32 SIREN                       : 1;
__REG32 SIRLP                       : 1;
__REG32                             : 4;
__REG32 LBE                         : 1;
__REG32 TXE                         : 1;
__REG32 RXE                         : 1;
__REG32 DTR                         : 1;
__REG32 RTS                         : 1;
__REG32 OUT1                        : 1;
__REG32 OUT2                        : 1;
__REG32 RTSEN                       : 1;
__REG32 CTSEN                       : 1;
__REG32                             :16;
} __hw_uartdbg_cr_bits;

/* UART Interrupt FIFO Level Select Register (HW_UARTDBG_IFLS) */
typedef struct {
__REG32 TXIFLSEL                    : 3;
__REG32 RXIFLSEL                    : 3;
__REG32                             :26;
} __hw_uartdbg_ifls_bits;

/* UART Interrupt Mask Set/Clear Register (HW_UARTDBG_IMSC) */
typedef struct {
__REG32 RIMIM                       : 1;
__REG32 CTSMIM                      : 1;
__REG32 DCDMIM                      : 1;
__REG32 DSRMIM                      : 1;
__REG32 RXIM                        : 1;
__REG32 TXIM                        : 1;
__REG32 RTIM                        : 1;
__REG32 FEIM                        : 1;
__REG32 PEIM                        : 1;
__REG32 BEIM                        : 1;
__REG32 OEIM                        : 1;
__REG32                             :21;
} __hw_uartdbg_imsc_bits;

/* UART Raw Interrupt Status Register (HW_UARTDBG_RIS) */
typedef struct {
__REG32 RIRMIS                      : 1;
__REG32 CTSRMIS                     : 1;
__REG32 DCDRMIS                     : 1;
__REG32 DSRRMIS                     : 1;
__REG32 RXRIS                       : 1;
__REG32 TXRIS                       : 1;
__REG32 RTRIS                       : 1;
__REG32 FERIS                       : 1;
__REG32 PERIS                       : 1;
__REG32 BERIS                       : 1;
__REG32 OERIS                       : 1;
__REG32                             :21;
} __hw_uartdbg_ris_bits;

/* UART Masked Interrupt Status Register (HW_UARTDBG_MIS) */
typedef struct {
__REG32 RIMMIS                      : 1;
__REG32 CTSMMIS                     : 1;
__REG32 DCDMMIS                     : 1;
__REG32 DSRMMIS                     : 1;
__REG32 RXMIS                       : 1;
__REG32 TXMIS                       : 1;
__REG32 RTMIS                       : 1;
__REG32 FEMIS                       : 1;
__REG32 PEMIS                       : 1;
__REG32 BEMIS                       : 1;
__REG32 OEMIS                       : 1;
__REG32                             :21;
} __hw_uartdbg_mis_bits;

/* UART Interrupt Clear Register (HW_UARTDBG_ICR) */
typedef struct {
__REG32 RIMIC                       : 1;
__REG32 CTSMIC                      : 1;
__REG32 DCDMIC                      : 1;
__REG32 DSRMIC                      : 1;
__REG32 RXIC                        : 1;
__REG32 TXIC                        : 1;
__REG32 RTIC                        : 1;
__REG32 FEIC                        : 1;
__REG32 PEIC                        : 1;
__REG32 BEIC                        : 1;
__REG32 OEIC                        : 1;
__REG32                             :21;
} __hw_uartdbg_icr_bits;

/* UART DMA Control Register (HW_UARTDBG_DMACR) */
typedef struct {
__REG32 RXDMAE                      : 1;
__REG32 TXDMAE                      : 1;
__REG32 DMAONERR                    : 1;
__REG32                             :29;
} __hw_uartdbg_dmacr_bits;

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
__REG32 PROP_SEG        : 3;
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
__REG32 TX_ERR_COUNTER  : 8;
__REG32 RX_ERR_COUNTER  : 8;
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

/* Glitch Filter Width Register (CAN_GFWR) */
typedef struct {
__REG32 GFWR            : 8;
__REG32                 :24;
} __can_gfwr_bits;

/* ENET MAC Interrupt Event Register (HW_ENET_MAC_EIR) */
typedef struct {
__REG32                 :15;
__REG32 TS_TIMER        : 1;
__REG32 TS_AVAIL        : 1;
__REG32 WAKEUP          : 1;
__REG32 PLR             : 1;
__REG32 UN              : 1;
__REG32 RL              : 1;
__REG32 LC              : 1;
__REG32 EBERR           : 1;
__REG32 MII             : 1;
__REG32 RXB             : 1;
__REG32 RXF             : 1;
__REG32 TXB             : 1;
__REG32 TXF             : 1;
__REG32 GRA             : 1;
__REG32 BABT            : 1;
__REG32 BABR            : 1;
__REG32                 : 1;
} __hw_enet_mac_eir_bits;

/* ENET MAC Receive Descriptor Active Register (HW_ENET_MAC_RDAR) */
typedef struct {
__REG32                 :24;
__REG32 RDAR            : 1;
__REG32                 : 7;
} __hw_enet_mac_rdar_bits;

/* ENET MAC Transmit Descriptor Active Register (HW_ENET_MAC_TDAR) */
typedef struct {
__REG32                 :24;
__REG32 TDAR            : 1;
__REG32                 : 7;
} __hw_enet_mac_tdar_bits;

/* ENET MAC Control Register (HW_ENET_MAC_ECR) */
typedef struct {
__REG32 RESET           : 1;
__REG32 ETHER_EN        : 1;
__REG32 MAGIC_ENA       : 1;
__REG32 SLEEP           : 1;
__REG32 ENA_1588        : 1;
__REG32                 : 1;
__REG32 DBG_EN          : 1;
__REG32                 :25;
} __hw_enet_mac_ecr_bits;

/* ENET MAC MII Management Frame Register (HW_ENET_MAC_MMFR) */
typedef struct {
__REG32 DATA            :16;
__REG32 TA              : 2;
__REG32 RA              : 5;
__REG32 PA              : 5;
__REG32 OP              : 2;
__REG32 ST              : 2;
} __hw_enet_mac_mmfr_bits;

/* ENET MAC MII Speed Control Register (HW_ENET_MAC_MSCR) */
typedef struct {
__REG32                 : 1;
__REG32 MII_SPEED       : 6;
__REG32 DIS_PRE         : 1;
__REG32 HOLDTIME        : 3;
__REG32                 :21;
} __hw_enet_mac_mscr_bits;

/* ENET MAC MIB Control/Status Register (HW_ENET_MAC_MIBC) */
typedef struct {
__REG32                 :29;
__REG32 MIB_CLEAR       : 1;
__REG32 MIB_IDLE        : 1;
__REG32 MIB_DIS         : 1;
} __hw_enet_mac_mibc_bits;

/* ENET MAC Receive Control Register (HW_ENET_MAC_RCR) */
typedef struct {
__REG32 LOOP            : 1;
__REG32 DRT             : 1;
__REG32 MII_MODE        : 1;
__REG32 PROM            : 1;
__REG32 BC_REJ          : 1;
__REG32 FCE             : 1;
__REG32                 : 2;
__REG32 RMII_MODE       : 1;
__REG32 RMII_10T        : 1;
__REG32                 : 2;
__REG32 PAD_EN          : 1;
__REG32 PAUSE_FWD       : 1;
__REG32 CRC_FWD         : 1;
__REG32 CNTL_FRM_ENA    : 1;
__REG32 MAX_FL          :14;
__REG32 NO_LGTH_CHECK   : 1;
__REG32 GRS             : 1;
} __hw_enet_mac_rcr_bits;

/* ENET MAC Transmit Control Register (HW_ENET_MAC_TCR) */
typedef struct {
__REG32 GTS             : 1;
__REG32                 : 1;
__REG32 FEDN            : 1;
__REG32 TFC_PAUSE       : 1;
__REG32 RFC_PAUSE       : 1;
__REG32 TX_ADDR_SEL     : 3;
__REG32 TX_ADDR_INS     : 1;
__REG32 TX_CRC_FWD      : 1;
__REG32                 :22;
} __hw_enet_mac_tcr_bits;

/* ENET MAC Physical Address Upper Register (HW_ENET_MAC_PAUR) */
typedef struct {
__REG32 TYPE            :16;
__REG32 PADDR2          :16;
} __hw_enet_mac_paur_bits;

/* ENET MAC Opcode/Pause Duration Register (HW_ENET_MAC_OPD) */
typedef struct {
__REG32 PAUSE_DUR       :16;
__REG32                 :16;
} __hw_enet_mac_opd_bits;

/* ENET MAC Transmit FIFO Watermark and Store and Forward Control Register (HW_ENET_MAC_TFW_SFCR) */
typedef struct {
__REG32 TFWR            : 6;
__REG32                 : 2;
__REG32 STR_FWD         : 1;
__REG32                 :23;
} __hw_enet_mac_tfw_sfcr_bits;

/* ENET MAC FIFO Receive Bound Register (HW_ENET_MAC_FRBR) */
typedef struct {
__REG32                 : 2;
__REG32 R_BOUND         : 8;
__REG32                 :22;
} __hw_enet_mac_frbr_bits;

/* ENET MAC FIFO Receive FIFO Start Register (HW_ENET_MAC_FRSR) */
typedef struct {
__REG32                 : 2;
__REG32 R_FSTART        : 8;
__REG32                 :22;
} __hw_enet_mac_frsr_bits;

/* ENET MAC Maximum Receive Buffer Size Register (HW_ENET_MAC_EMRBR) */
typedef struct {
__REG32                 : 4;
__REG32 R_BUF_SIZE      : 7;
__REG32                 :21;
} __hw_enet_mac_emrbr_bits;

/* ENET MAC Receive FIFO Section Full Threshold Register (HW_ENET_MAC_RX_SECTION_FULL) */
typedef struct {
__REG32 RX_SECTION_FULL : 8;
__REG32                 :24;
} __hw_enet_mac_rx_section_full_bits;

/* ENET MAC Receive FIFO Section Empty Threshold Register (HW_ENET_MAC_RX_SECTION_EMPTY) */
typedef struct {
__REG32 RX_SECTION_EMPTY  : 8;
__REG32                   :24;
} __hw_enet_mac_rx_section_empty_bits;

/* ENET MAC Receive FIFO Almost Empty Threshold Register (HW_ENET_MAC_RX_ALMOST_EMPTY) */
typedef struct {
__REG32 RX_ALMOST_EMPTY   : 8;
__REG32                   :24;
} __hw_enet_mac_rx_almost_empty_bits;

/* ENET MAC Receive FIFO Almost Full Thresholdt Register (HW_ENET_MAC_RX_ALMOST_FULL) */
typedef struct {
__REG32 RX_ALMOST_FULL    : 8;
__REG32                   :24;
} __hw_enet_mac_rx_almost_full_bits;

/* ENET MAC Receive FIFO Almost Full Thresholdt Register (HW_ENET_MAC_RX_ALMOST_FULL) */
typedef struct {
__REG32 TX_SECTION_EMPTY  : 8;
__REG32                   :24;
} __hw_enet_mac_tx_section_empty_bits;

/* ENET MAC Transmit FIFO Almost Empty Threshold Register (HW_ENET_MAC_TX_ALMOST_EMPTY) */
typedef struct {
__REG32 TX_ALMOST_EMPTY   : 8;
__REG32                   :24;
} __hw_enet_mac_tx_almost_empty_bits;

/* ENET MAC Transmit FIFO Almost Full Threshold Register (HW_ENET_MAC_TX_ALMOST_FULL) */
typedef struct {
__REG32 TX_ALMOST_FULL    : 8;
__REG32                   :24;
} __hw_enet_mac_tx_almost_full_bits;

/* ENET MAC Transmit Inter-Packet Gap Register (HW_ENET_MAC_TX_IPG_LENGTH) */
typedef struct {
__REG32 TX_IPG_LENGTH     : 5;
__REG32                   :27;
} __hw_enet_mac_tx_ipg_length_bits;

/* ENET MAC Frame Truncation Length Register (HW_ENET_MAC_TRUNC_FL) */
typedef struct {
__REG32 TRUNC_FL          :14;
__REG32                   :18;
} __hw_enet_mac_trunc_fl_bits;

/* ENET MAC Accelerator Transmit Function Configuration Register (HW_ENET_MAC_IPACCTXCONF) */
typedef struct {
__REG32 SHIFT16           : 1;
__REG32                   : 2;
__REG32 TX_IPCHK_INS      : 1;
__REG32 TX_PROTCHK_INS    : 1;
__REG32                   :27;
} __hw_enet_mac_ipacctxconf_bits;

/* ENET MAC Accelerator Receive Function Configuration Register (HW_ENET_MAC_IPACCRXCONF) */
typedef struct {
__REG32 RX_IP_PAD_REMOVE    : 1;
__REG32 RX_IPERR_DISCARD    : 1;
__REG32 RX_PROTERR_DISCARD  : 1;
__REG32                     : 3;
__REG32 RX_LINEERR_DISC     : 1;
__REG32 SHIFT16             : 1;
__REG32                     :24;
} __hw_enet_mac_ipaccrxconf_bits;

/* ENET MAC IEEE1588 Timer Control Register (HW_ENET_MAC_ATIME_CTRL) */
typedef struct {
__REG32 ENABLE              : 1;
__REG32 ONE_SHOT            : 1;
__REG32 EVT_OFFSET_ENA      : 1;
__REG32 EVT_OFFSET_RST      : 1;
__REG32 EVT_PERIOD_ENA      : 1;
__REG32 EVT_PERIOD_RST      : 1;
__REG32                     : 1;
__REG32 PIN_PERIOD_ENA      : 1;
__REG32                     : 1;
__REG32 RESTART             : 1;
__REG32                     : 1;
__REG32 CAPTURE             : 1;
__REG32                     : 1;
__REG32 FRC_SLAVE           : 1;
__REG32                     :18;
} __hw_enet_mac_atime_ctrl_bits;

/* ENET MAC IEEE1588 Correction counter wrap around value Register (HW_ENET_MAC_ATIME_CORR) */
typedef struct {
__REG32 ATIME_CORR          :31;
__REG32                     : 1;
} __hw_enet_mac_atime_corr_bits;

/* ENET MAC IEEE1588 Clock period of the timestamping clock
	(ts_clk) in nanoseconds and correction increment Register
	(HW_ENET_MAC_ATIME_INC) */
typedef struct {
__REG32 ATIME_INC           : 7;
__REG32                     : 1;
__REG32 ATIME_INC_CORR      : 7;
__REG32                     :17;
} __hw_enet_mac_atime_inc_bits;

/* ENET MAC IEEE1588 Timestamp of the last Frame Register (HW_ENET_MAC_TS_TIMESTAMP) */
typedef struct {
__REG32 ATIME_INC           : 7;
__REG32                     : 1;
__REG32 ATIME_INC_CORR      : 7;
__REG32                     :17;
} __hw_enet_mac_ts_timestamp_bits;

/* ENET MAC IEEE1588 Interrupt register. (HW_ENET_MAC_CCB_INT) */
/* ENET MAC IEEE1588 Interrupt enable mask register (HW_ENET_MAC_CCB_INT_MASK) */
typedef struct {
__REG32 CAPTURE0            : 1;
__REG32 CAPTURE1            : 1;
__REG32 CAPTURE2            : 1;
__REG32 CAPTURE3            : 1;
__REG32                     :12;
__REG32 COMPARE0            : 1;
__REG32 COMPARE1            : 1;
__REG32 COMPARE2            : 1;
__REG32 COMPARE3            : 1;
__REG32                     :12;
} __hw_enet_mac_ccb_int_bits;

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
__REG32 MULTI_MASTER          : 1;
__REG32                       : 1;
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
__REG32 SLAVE_IRQ                 : 1;
__REG32 SLAVE_STOP_IRQ            : 1;
__REG32 MASTER_LOSS_IRQ           : 1;
__REG32 EARLY_TERM_IRQ            : 1;
__REG32 OVERSIZE_XFER_TERM_IRQ    : 1;
__REG32 NO_SLAVE_ACK_IRQ          : 1;
__REG32 DATA_ENGINE_CMPLT_IRQ     : 1;
__REG32 BUS_FREE_IRQ              : 1;
__REG32 SLAVE_IRQ_EN              : 1;
__REG32 SLAVE_STOP_IRQ_EN         : 1;
__REG32 MASTER_LOSS_IRQ_EN        : 1;
__REG32 EARLY_TERM_IRQ_EN         : 1;
__REG32 OVERSIZE_XFER_TERM_IRQ_EN : 1;
__REG32 NO_SLAVE_ACK_IRQ_EN       : 1;
__REG32 DATA_ENGINE_CMPLT_IRQ_EN  : 1;
__REG32 BUS_FREE_IRQ_EN           : 1;
__REG32 SLAVE_ADDRESS_BYTE        : 8;
__REG32 BCAST_SLAVE_EN            : 1;
__REG32 FORCE_CLK_IDLE            : 1;
__REG32 FORCE_DATA_IDLE           : 1;
__REG32 ACK_MODE                  : 1;
__REG32 CLR_GOT_A_NAK             : 1;
__REG32 WR_QUEUE_IRQ              : 1;
__REG32 RD_QUEUE_IRQ              : 1;
__REG32                           : 1;
} __hw_i2c_ctrl1_bits;

/* I2C Status Register */
typedef struct {
__REG32 SLAVE_IRQ_SMR               : 1;
__REG32 SLAVE_STOP_IRQ_SMR          : 1;
__REG32 MASTER_LOSS_IRQ_SMR         : 1;
__REG32 EARLY_TERM_IRQ_SMR          : 1;
__REG32 OVERSIZE_XFER_TERM_IRQ_SMR  : 1;
__REG32 NO_SLAVE_ACK_IRQ_SMR        : 1;
__REG32 DATA_ENGINE_CMPLT_IRQ_SMR   : 1;
__REG32 BUS_FREE_IRQ_SMR            : 1;
__REG32 SLAVE_BUSY                  : 1;
__REG32 DATA_ENGINE_BUSY            : 1;
__REG32 CLK_GEN_BUSY                : 1;
__REG32 BUS_BUSY                    : 1;
__REG32 DATA_ENGINE_DMA_WAIT        : 1;
__REG32 SLAVE_SEARCHING             : 1;
__REG32 SLAVE_FOUND                 : 1;
__REG32 SLAVE_ADDR_EQ_ZERO          : 1;
__REG32 RCVD_SLAVE_ADDR             : 8;
__REG32                             : 4;
__REG32 GOT_A_NAK                   : 1;
__REG32 ANY_ENABLED_IRQ             : 1;
__REG32 SLAVE_PRESENT               : 1;
__REG32 MASTER_PRESENT              : 1;
} __hw_i2c_stat_bits;

/* I2C Queue control reg. */
typedef struct {
__REG32 WR_QUEUE_IRQ_EN             : 1;
__REG32 RD_QUEUE_IRQ_EN             : 1;
__REG32 PIO_QUEUE_MODE              : 1;
__REG32 WR_CLEAR                    : 1;
__REG32 RD_CLEAR                    : 1;
__REG32 QUEUE_RUN                   : 1;
__REG32                             : 2;
__REG32 WR_THRESH                   : 5;
__REG32                             : 3;
__REG32 RD_THRESH                   : 5;
__REG32                             :11;
} __hw_i2c_queuectrl_bits;

/* I2C Queue Status Register */
typedef struct {
__REG32 WR_QUEUE_CNT                : 5;
__REG32 WR_QUEUE_EMPTY              : 1;
__REG32 WR_QUEUE_FULL               : 1;
__REG32                             : 1;
__REG32 RD_QUEUE_CNT                : 5;
__REG32 RD_QUEUE_EMPTY              : 1;
__REG32 RD_QUEUE_FULL               : 1;
__REG32                             :17;
} __hw_i2c_queuestat_bits;

/* I2C Queue command reg */
typedef struct {
__REG32 XFER_COUNT                  :16;
__REG32 DIRECTION                   : 1;
__REG32 MASTER_MODE                 : 1;
__REG32 SLAVE_ADDRESS_ENABLE        : 1;
__REG32 PRE_SEND_START              : 1;
__REG32 POST_SEND_STOP              : 1;
__REG32 RETAIN_CLOCK                : 1;
__REG32 CLOCK_HELD                  : 1;
__REG32 MULTI_MASTER                : 1;
__REG32                             : 1;
__REG32 SEND_NAK_ON_LAST            : 1;
__REG32 ACKNOWLEDGE                 : 1;
__REG32 PRE_ACK                     : 1;
__REG32                             : 4;
} __hw_i2c_queuecmd_bits;

/* I2C Device Debug Register 0 */
typedef struct {
__REG32 SLAVE_STATE           :10;
__REG32 SLAVE_HOLD_CLK        : 1;
__REG32 STATE_LATCH           : 1;
__REG32 CHANGE_TOGGLE         : 1;
__REG32 GRAB_TOGGLE           : 1;
__REG32 STOP_TOGGLE           : 1;
__REG32 START_TOGGLE          : 1;
__REG32 DMA_STATE             :10;
__REG32 STATE_VALUE           : 2;
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

/* PWM Control and Status Register Description */
typedef struct {
__REG32 PWM0_ENABLE           : 1;
__REG32 PWM1_ENABLE           : 1;
__REG32 PWM2_ENABLE           : 1;
__REG32 PWM3_ENABLE           : 1;
__REG32 PWM4_ENABLE           : 1;
__REG32 PWM5_ENABLE           : 1;
__REG32 PWM6_ENABLE           : 1;
__REG32 PWM7_ENABLE           : 1;
__REG32                       : 1;
__REG32 OUTPUT_CUTOFF_EN      : 1;
__REG32                       :12;
__REG32 PWM0_PRESENT          : 1;
__REG32 PWM1_PRESENT          : 1;
__REG32 PWM2_PRESENT          : 1;
__REG32 PWM3_PRESENT          : 1;
__REG32 PWM4_PRESENT          : 1;
__REG32 PWM5_PRESENT          : 1;
__REG32 PWM6_PRESENT          : 1;
__REG32 PWM7_PRESENT          : 1;
__REG32 CLKGATE               : 1;
__REG32 SFTRST                : 1;
} __hw_pwm_ctrl_bits;

/* PWM Channel 0 Active Register */
/* PWM Channel 1 Active Register */
/* PWM Channel 2 Active Register */
/* PWM Channel 3 Active Register */
/* PWM Channel 4 Active Register */
/* PWM Channel 5 Active Register */
/* PWM Channel 6 Active Register */
/* PWM Channel 7 Active Register */
typedef struct {
__REG32 ACTIVE          :16;
__REG32 INACTIVE        :16;
} __hw_pwm_activex_bits;

/* PWM Channel 0 Period Register */
/* PWM Channel 1 Period Register */
/* PWM Channel 2 Period Register */
/* PWM Channel 3 Period Register */
/* PWM Channel 4 Period Register */
/* PWM Channel 5 Period Register */
/* PWM Channel 6 Period Register */
/* PWM Channel 7 Period Register */
typedef struct {
__REG32 PERIOD          :16;
__REG32 ACTIVE_STATE    : 2;
__REG32 INACTIVE_STATE  : 2;
__REG32 CDIV            : 3;
__REG32 MATT            : 1;
__REG32 MATT_SEL        : 1;
__REG32 HSADC_CLK_SEL   : 1;
__REG32 HSADC_OUT       : 1;
__REG32                 : 5;
} __hw_pwm_periodx_bits;

/* PWM Version Register */
typedef struct {
__REG32 STEP            :16;
__REG32 MINOR           : 8;
__REG32 MAJOR           : 8;
} __hw_pwm_version_bits;

/* ENET SWI revision (HW_ENET_SWI_REVISION) */
typedef struct {
__REG32 CORE_REVISION     :16;
__REG32 CUSTOMER_REVISION :16;
} __hw_enet_swi_revision_bits;

/* ENET SWI Port Enable Bits. (HW_ENET_SWI_PORT_ENA) */
typedef struct {
__REG32 ENA_TRANSMIT_0    : 1;
__REG32 ENA_TRANSMIT_1    : 1;
__REG32 ENA_TRANSMIT_2    : 1;
__REG32                   :13;
__REG32 ENA_RECEIVE_0     : 1;
__REG32 ENA_RECEIVE_1     : 1;
__REG32 ENA_RECEIVE_2     : 1;
__REG32                   :13;
} __hw_enet_swi_port_ena_bits;

/* ENET SWI Verify VLAN domain. (HW_ENET_SWI_VLAN_VERIFY) */
typedef struct {
__REG32 VLAN_VERIFY_0     : 1;
__REG32 VLAN_VERIFY_1     : 1;
__REG32 VLAN_VERIFY_2     : 1;
__REG32                   :13;
__REG32 DISCARD_P0        : 1;
__REG32 DISCARD_P1        : 1;
__REG32 DISCARD_P2        : 1;
__REG32                   :13;
} __hw_enet_swi_vlan_verify_bits;

/* ENET SWI Default broadcast resolution. (HW_ENET_SWI_BCAST_DEFAULT_MASK) */
typedef struct {
__REG32 BCAST_DEFAULT_MASK_0  : 1;
__REG32 BCAST_DEFAULT_MASK_1  : 1;
__REG32 BCAST_DEFAULT_MASK_2  : 1;
__REG32                       :29;
} __hw_enet_swi_bcast_default_mask_bits;

/* ENET SWI Default multicast resolution. (HW_ENET_SWI_MCAST_DEFAULT_MASK) */
typedef struct {
__REG32 MCAST_DEFAULT_MASK_0  : 1;
__REG32 MCAST_DEFAULT_MASK_1  : 1;
__REG32 MCAST_DEFAULT_MASK_2  : 1;
__REG32                       :29;
} __hw_enet_swi_mcast_default_mask_bits;

/* ENET SWI Define port in blocking state
	 and enable or disable learning. (HW_ENET_SWI_INPUT_LEARN_BLOCK) */
typedef struct {
__REG32 BLOCKING_ENA_P0       : 1;
__REG32 BLOCKING_ENA_P1       : 1;
__REG32 BLOCKING_ENA_P2       : 1;
__REG32                       :13;
__REG32 LEARNING_DI_P0        : 1;
__REG32 LEARNING_DI_P1        : 1;
__REG32 LEARNING_DI_P2        : 1;
__REG32                       :13;
} __hw_enet_swi_input_learn_block_bits;

/* ENET SWI Bridge Management Port Configuration.
	(HW_ENET_SWI_MGMT_CONFIG) */
typedef struct {
__REG32 PORT                  : 4;
__REG32                       : 1;
__REG32 MESSAGE_TRANSMITTED   : 1;
__REG32 ENABLE                : 1;
__REG32 DISCARD               : 1;
__REG32                       : 5;
__REG32 PRIORITY              : 3;
__REG32 PORTMASK              : 3;
__REG32                       :13;
} __hw_enet_swi_mgmt_config_bits;

/* ENET SWI Defines several global configuration settings.
	(HW_ENET_SWI_MODE_CONFIG) */
typedef struct {
__REG32 PORT                  : 4;
__REG32                       : 1;
__REG32 MESSAGE_TRANSMITTED   : 1;
__REG32 ENABLE                : 1;
__REG32 DISCARD               : 1;
__REG32                       : 5;
__REG32 PRIORITY              : 3;
__REG32 PORTMASK              : 3;
__REG32                       :13;
} __hw_enet_swi_mode_config_bits;

/* ENET SWI Define behavior of VLAN input manipulation function
	(HW_ENET_SWI_VLAN_IN_MODE) */
typedef struct {
__REG32 VLAN_IN_MODE_0        : 2;
__REG32 VLAN_IN_MODE_1        : 2;
__REG32 VLAN_IN_MODE_2        : 2;
__REG32                       :26;
} __hw_enet_swi_vlan_in_mode_bits;

/* ENET SWI Define behavior of VLAN output manipulation
	function (HW_ENET_SWI_VLAN_OUT_MODE) */
typedef struct {
__REG32 VLAN_OUT_MODE_0       : 2;
__REG32 VLAN_OUT_MODE_1       : 2;
__REG32 VLAN_OUT_MODE_2       : 2;
__REG32                       :26;
} __hw_enet_swi_vlan_out_mode_bits;

/* ENET SWI Enable the input processing according to the
	VLAN_IN_MODE for a port (HW_ENET_SWI_VLAN_IN_MODE_ENA) */
typedef struct {
__REG32 VLAN_IN_MODE_ENA_0    : 1;
__REG32 VLAN_IN_MODE_ENA_1    : 1;
__REG32 VLAN_IN_MODE_ENA_2    : 1;
__REG32                       :29;
} __hw_enet_swi_vlan_in_mode_ena_bits;

/* ENET SWI The VLAN type field value to expect to identify a
	VLAN tagged frame. (HW_ENET_SWI_VLAN_TAG_ID) */
typedef struct {
__REG32 SWI_VLAN_TAG_ID       :16;
__REG32                       :16;
} __hw_enet_swi_vlan_tag_id_bits;

/* ENET SWI Port Mirroring configuration.
	(HW_ENET_SWI_MIRROR_CONTROL) */
typedef struct {
__REG32 PORTX                 : 4;
__REG32 MIRROR_ENABLE         : 1;
__REG32 ING_MAP_ENABLE        : 1;
__REG32 EG_MAP_ENABLE         : 1;
__REG32 ING_SA_MATCH          : 1;
__REG32 ING_DA_MATCH          : 1;
__REG32 EG_SA_MATCH           : 1;
__REG32 EG_DA_MATCH           : 1;
__REG32                       :21;
} __hw_enet_swi_mirror_control_bits;

/* ENET SWI Port Mirroring Egress port definitions.
	(HW_ENET_SWI_MIRROR_EG_MAP) */
typedef struct {
__REG32 MIRROR_EG_MAP_0       : 1;
__REG32 MIRROR_EG_MAP_1       : 1;
__REG32 MIRROR_EG_MAP_2       : 1;
__REG32                       :29;
} __hw_enet_swi_mirror_eg_map_bits;

/* ENET SWI Port Mirroring Ingress port definitions.
(HW_ENET_SWI_MIRROR_ING_MAP) */
typedef struct {
__REG32 MIRROR_ING_MAP_0      : 1;
__REG32 MIRROR_ING_MAP_1      : 1;
__REG32 MIRROR_ING_MAP_2      : 1;
__REG32                       :29;
} __hw_enet_swi_mirror_ing_map_bits;

/* ENET SWI Port Mirroring Ingress port definitions.
	(HW_ENET_SWI_MIRROR_ING_MAP) */
typedef struct {
__REG32 MIRROR_ISRC_1         :16;
__REG32                       :16;
} __hw_enet_swi_mirror_isrc_1_bits;

/* ENET SWI Ingress Destination MAC Address for Mirror filtering.
	(HW_ENET_SWI_MIRROR_IDST_1) */
typedef struct {
__REG32 MIRROR_IDST_1         :16;
__REG32                       :16;
} __hw_enet_swi_mirror_idst_1_bits;

/* ENET SWI Egress Source MAC Address for Mirror filtering.
	(HW_ENET_SWI_MIRROR_ESRC_1) */
typedef struct {
__REG32 MIRROR_ESRC_1         :16;
__REG32                       :16;
} __hw_enet_swi_mirror_esrc_1_bits;

/* ENET SWI Egress Destination MAC Address for Mirror filtering.
	(HW_ENET_SWI_MIRROR_EDST_1) */
typedef struct {
__REG32 MIRROR_ESRC_1         :16;
__REG32                       :16;
} __hw_enet_swi_mirror_edst_1_bits;

/* ENET SWI Count Value for Mirror filtering.
	(HW_ENET_SWI_MIRROR_CNT) */
typedef struct {
__REG32 MIRROR_CNT            : 8;
__REG32                       :24;
} __hw_enet_swi_mirror_cnt_bits;

/* ENET SWI Memory Manager Status.
	(HW_ENET_SWI_OQMGR_STATUS) */
typedef struct {
__REG32 BUSY_INITIALIZING     : 1;
__REG32 NO_CELL_LATCH         : 1;
__REG32 MEM_FULL              : 1;
__REG32 MEM_FULL_LATCH        : 1;
__REG32                       : 2;
__REG32 DEQUEUE_GRANT         : 1;
__REG32                       : 9;
__REG32 CELLS_AVAILABLE       : 8;
__REG32                       : 8;
} __hw_enet_swi_oqmgr_status_bits;

/* ENET SWI Low Memory threshold.
	(HW_ENET_SWI_QMGR_MINCELLS) */
typedef struct {
__REG32 QMGR_MINCELLS         : 8;
__REG32                       :24;
} __hw_enet_swi_qmgr_mincells_bits;

/* ENET SWI Port Congestion status (internal).
	(HW_ENET_SWI_QMGR_CONGEST_STAT) */
typedef struct {
__REG32 QMGR_CONGEST_STAT_0   : 1;
__REG32 QMGR_CONGEST_STAT_1   : 1;
__REG32 QMGR_CONGEST_STAT_2   : 1;
__REG32                       :29;
} __hw_enet_swi_qmgr_congest_stat_bits;

/* ENET SWI Switch input and output interface status (internal).
	(HW_ENET_SWI_QMGR_IFACE_STAT)*/
typedef struct {
__REG32 OUTPUT_0              : 1;
__REG32 OUTPUT_1              : 1;
__REG32 OUTPUT_2              : 1;
__REG32                       :13;
__REG32 INPUT_0               : 1;
__REG32 INPUT_1               : 1;
__REG32 INPUT_2               : 1;
__REG32                       :13;
} __hw_enet_swi_qmgr_iface_stat_bits;

/* ENET SWI Queue weights for each queue.
	(HW_ENET_SWI_QM_WEIGHTS) */
typedef struct {
__REG32 QUEUE_0               : 5;
__REG32                       : 3;
__REG32 QUEUE_1               : 5;
__REG32                       : 3;
__REG32 QUEUE_2               : 5;
__REG32                       : 3;
__REG32 QUEUE_3               : 5;
__REG32                       : 3;
} __hw_enet_swi_qm_weights_bits;

/* ENET SWI Define congestion threshold for Port0 backpressure.
	(HW_ENET_SWI_QMGR_MINCELLSP0) */
typedef struct {
__REG32 QMGR_MINCELLSP0       : 8;
__REG32                       :24;
} __hw_enet_swi_qmgr_mincellsp0_bits;

/* ENET SWI Enable forced forwarding for a frame processed from
	port 0 (HW_ENET_SWI_FORCE_FWD_P0) */
typedef struct {
__REG32 FORCE_ENABLE          : 1;
__REG32                       : 1;
__REG32 FORCE_DESTINATION     : 2;
__REG32                       :28;
} __hw_enet_swi_force_fwd_p0_bits;

/* ENET SWI Port Snooping function. Eight independent entries
	are available. (HW_ENET_SWI_PORTSNOOPx) */
typedef struct {
__REG32 ENABLE                : 1;
__REG32 MODE                  : 2;
__REG32 COMPARE_DEST          : 1;
__REG32 COMPARE_SOURCE        : 1;
__REG32                       :11;
__REG32 DESTINATION_PORT      :16;
} __hw_enet_swi_portsnoop_bits;

/* ENET SWI Port Snooping function. Eight independent entries
	are available. (HW_ENET_SWI_PORTSNOOPx) */
typedef struct {
__REG32 ENABLE                : 1;
__REG32 MODE                  : 2;
__REG32                       : 5;
__REG32 PROTOCOL              : 8;
__REG32                       :16;
} __hw_enet_swi_ipsnoop_bits;

/* ENET SWI Port X VLAN priority resolution map
	(HW_ENET_SWI_VLAN_PRIORITY0) */
typedef struct {
__REG32 P0                    : 3;
__REG32 P1                    : 3;
__REG32 P2                    : 3;
__REG32 P3                    : 3;
__REG32 P4                    : 3;
__REG32 P5                    : 3;
__REG32 P6                    : 3;
__REG32 P7                    : 3;
__REG32                       : 8;
} __hw_enet_swi_vlan_priority_bits;

/* ENET SWI IPv4 and IPv6 priority resolution table programming
	(HW_ENET_SWI_IP_PRIORITY) */
typedef struct {
__REG32 ADDRESS               : 8;
__REG32 IPV4_SELECT           : 1;
__REG32 PRIORITY_PORT0        : 2;
__REG32 PRIORITY_PORT1        : 2;
__REG32 PRIORITY_PORT2        : 2;
__REG32                       :16;
__REG32 READ                  : 1;
} __hw_enet_swi_ip_priority_bits;

/* ENET SWI Port 0 Priority resolution configuration
	(HW_ENET_SWI_PRIORITY_CFG0) */
typedef struct {
__REG32 VLAN_EN               : 1;
__REG32 IP_EN                 : 1;
__REG32 MAC_EN                : 1;
__REG32                       : 1;
__REG32 DEFAULT_PRIORITY      : 3;
__REG32                       :25;
} __hw_enet_swi_priority_cfg_bits;

/* ENET SWI Port 0 VLAN-ID field for VLAN input manipulation
	function (HW_ENET_SWI_SYSTEM_TAGINFOn) */
typedef struct {
__REG32 SYSTEM_TAGINFO0       :16;
__REG32                       :16;
} __hw_enet_swi_system_taginfo_bits;

/* ENET SWI Port 0 VLAN-ID field for VLAN input manipulation
	function (HW_ENET_SWI_SYSTEM_TAGINFOn) */
typedef struct {
__REG32 PORT_0                : 1;
__REG32 PORT_1                : 1;
__REG32 PORT_2                : 1;
__REG32 VLAN_ID_0             :12;
__REG32                       :17;
} __hw_enet_swi_vlan_res_table_bits;

/* ENET SWI Interrupt Event Register (HW_ENET_SWI_EIR) */
/* ENET SWI Interrupt Mask Register (HW_ENET_SWI_EIMR) */
typedef struct {
__REG32 EBERR                 : 1;
__REG32 RXB                   : 1;
__REG32 RXF                   : 1;
__REG32 TXB                   : 1;
__REG32 TXF                   : 1;
__REG32 QM                    : 1;
__REG32 OD0                   : 1;
__REG32 OD1                   : 1;
__REG32 OD2                   : 1;
__REG32 LRN                   : 1;
__REG32                       :22;
} __hw_enet_swi_eir_bits;

/* ENET SWI Maximum Receive Buffer Size (HW_ENET_SWI_EMRBR) */
typedef struct {
__REG32                       : 4;
__REG32 EMRBR                 :10;
__REG32                       :18;
} __hw_enet_swi_emrbr_bits;

/* ENET SWI Learning Record B(1) (HW_ENET_SWI_LRN_REC_1) */
typedef struct {
__REG32 MAC_ADDR1             :16;
__REG32 HASH                  : 8;
__REG32 SW_PORT               : 2;
__REG32                       : 6;
} __hw_enet_swi_lrn_rec_1_bits;

/* ENET SWI Learning data available status. (HW_ENET_SWI_LRN_STATUS) */
typedef struct {
__REG32 LRN_STATUS            : 1;
__REG32                       :31;
} __hw_enet_swi_lrn_status_bits;

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
__REG32 SIREN           : 1;
__REG32 SIRLP           : 1;
__REG32                 : 3;
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
__REG32 ABDIS           : 1;
__REG32                 : 4;
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
__REG32 ABDIEN          : 1;
__REG32                 : 4;
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

/* Identification Register (HW_USBCTRL_ID) */
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

/* General Hardware Parameters Register (HW_USBCTRL_HWGENERAL) */
typedef struct{
__REG32 RT        : 1;
__REG32 CLKC      : 2;
__REG32 BWT       : 1;
__REG32 PHYW      : 2;
__REG32 PHYM      : 3;
__REG32 SM        : 2;
__REG32           :21;
} __hw_usbctrl_hwgeneral_bits;

/* Host Hardware Parameters Register (HW_USBCTRL_HWHOST) */
typedef struct{
__REG32 HC        : 1;
__REG32 NPORT     : 3;
__REG32           :12;
__REG32 TTASY     : 8;
__REG32 TTPER     : 8;
} __hw_usbctrl_hwhost_bits;

/* Device Hardware Parameters Register (HW_USBCTRL_HWDEVICE) */
typedef struct{
__REG32 DC        : 1;
__REG32 DEVEP     : 5;
__REG32           :26;
} __hw_usbctrl_hwdevice_bits;

/* TX Buffer Hardware Parameters Register (HW_USBCTRL_HWTXBUF) */
typedef struct{
__REG32 TXBURST   : 8;
__REG32 TXADD     : 8;
__REG32 TXCHANADD : 8;
__REG32           : 7;
__REG32 TXLCR     : 1;
} __hw_usbctrl_hwtxbuf_bits;

/* RX Buffer Hardware Parameters Register (HW_USBCTRL_HWRXBUF) */
typedef struct{
__REG32 RXBURST   : 8;
__REG32 RXADD     : 8;
__REG32           :16;
} __hw_usbctrl_hwrxbuf_bits;

/* Host Control Structural Parameters (EHCI-Compliant with Extensions) Register (HW_USBCTRL_HCSPARAMS) */
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

/* Host Control Capability Parameters (EHCI-Compliant) Register (HW_USBCTRL_HCCPARAMS) */
typedef struct{
__REG32 ADC       : 1;
__REG32 PFL       : 1;
__REG32 ASP       : 1;
__REG32           : 1;
__REG32 IST       : 4;
__REG32 EECP      : 8;
__REG32           :16;
} __hw_usbctrl_hccparams_bits;

/* Device Interface Version Number (Non-EHCI-Compliant) Register (HW_USBCTRL_DCIVERSION) */
typedef struct{
__REG32 DCIVERSION  :16;
__REG32             :16;
} __hw_usbctrl_dciversion_bits;

/* Capability Length and HCI Version (EHCI-Compliant) Register (HW_USBCTRL_CAPLENGTH) */
typedef struct{
__REG32 CAPLENGTH : 8;
__REG32           : 8;
__REG32 HCIVERSION:16;
} __hw_usbctrl_caplength_bits;

/* Device Control Capability Parameters (Non-EHCI-Compliant) Register (HW_USBCTRL_DCCPARAMS) */
typedef struct{
__REG32 DEN       : 5;
__REG32           : 2;
__REG32 DC        : 1;
__REG32 HC        : 1;
__REG32           :23;
} __hw_usbctrl_dccparams_bits;

/* System Bus Configuration (Non-EHCI-Compliant) Register (HW_USBCTRL_SBUSCFG) */
typedef struct{
__REG32 AHBBRST     : 3;
__REG32             :29;
} __hw_usbctrl_sbuscfg_bits;

/* General-Purpose Timer 0 Load (Non-EHCI-Compliant) Register (HW_USBCTRL_GPTIMER0LD) */
typedef struct{
__REG32 GPTLD       :24;
__REG32             : 8;
} __hw_usbctrl_gptimerxld_bits;

/* General-Purpose Timer 0 Control (Non-EHCI-Compliant) Register (HW_USBCTRL_GPTIMER0CTRL) */
typedef struct{
__REG32 GPTCNT      :24;
__REG32 GPTMOD      : 1;
__REG32             : 5;
__REG32 GPTRST      : 1;
__REG32 GPTRUN      : 1;
} __hw_usbctrl_gptimerxctrl_bits;

/* USB Command Register (HW_USBCTRL_USBCMD) */
typedef struct{
__REG32 RS        : 1;
__REG32 RST       : 1;
__REG32 FS0       : 1;
__REG32 FS1       : 1;
__REG32 PSE       : 1;
__REG32 ASE       : 1;
__REG32 IAA       : 1;
__REG32 LR        : 1;
__REG32 ASP       : 2;
__REG32           : 1;
__REG32 ASPE      : 1;
__REG32           : 1;
__REG32 SUTW      : 1;
__REG32 ADTW      : 1;
__REG32 FS2       : 1;
__REG32 ITC       : 8;
__REG32           : 8;
} __hw_usbctrl_usbcmd_bits;

/* USB Status Register (HW_USBCTRL_USBSTS) */
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

/* USB Interrupt Enable Register (HW_USBCTRL_USBINTR) */
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

/* USB Frame Index Register (HW_USBCTRL_FRINDEX) */
typedef struct{
__REG32 UINDEX    : 3;
__REG32 FRINDEX   :11;
__REG32           :18;
} __hw_usbctrl_frindex_bits;

/* Frame List Base Address Register (Host Controller mode) (HW_USBCTRL_PERIODICLISTBASE) */
/* USB Device Address Register (Device Controller mode) (HW_USBCTRL_DEVICEADDR) */
typedef union {
	/* HW_USBCTRLx_PERIODICLISTBASE */
	struct{
		__REG32           :12;
		__REG32 PERBASE   :20;
	};
	/* HW_USBCTRLx_DEVICEADDR */
	struct{
		__REG32           :24;
		__REG32 USBADRA   : 1;
		__REG32 USBADR    : 7;
	};
} __hw_usbctrl_periodiclistbase_bits;

/* Next Asynchronous Address Register (Host Controller mode)(HW_USBCTRL_ASYNCLISTADDR) */
/* Endpoint List Address Register (Device Controller mode)(HW_USBCTRL_ENDPOINTLISTADDR) */
typedef union {
	/* HW_USBCTRLx_ASYNCLISTADDR */
	struct{
		__REG32           : 5;
		__REG32 ASYBASE   :27;
	};
	/* HW_USBCTRLx_ENDPOINTLISTADDR */
	struct{
		__REG32           :11;
		__REG32 EPBASE    :21;
	};
} __hw_usbctrl_asynclistaddr_bits;

/* Embedded TT Asynchronous Buffer Status and Control Register (Host Controller mode) (HW_USBCTRL_TTCTRL) */
typedef struct{
__REG32           :24;
__REG32 TTHA      : 7;
__REG32           : 1;
} __hw_usbctrl_ttctrl_bits;

/* Programmable Burst Size Register (HW_USBCTRL_BURSTSIZE) */
typedef struct{
__REG32 RXPBURST  : 8;
__REG32 TXPBURST  : 8;
__REG32           :16;
} __hw_usbctrl_burstsize_bits;

/* Host Transmit Pre-Buffer Packet Timing Register (HW_USBCTRL_TXFILLTUNING)r */
typedef struct{
__REG32 TXSCHOH     : 7;
__REG32             : 1;
__REG32 TXSCHHEALTH : 5;
__REG32             : 3;
__REG32 TXFIFOTHRES : 6;
__REG32             :10;
} __hw_usbctrl_txfilltuning_bits;

/* Inter-Chip Control Register (HW_USBCTRL_IC_USB) */
typedef struct{
__REG32 IC_VDD      : 3;
__REG32 IC_ENABLE   : 1;
__REG32             :28;
} __hw_usbctrl_ic_usb_bits;

/* ULPI Viewport Register (HW_USBCTRL_ULPI) */
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

/* Endpoint NAK Register (HW_USBCTRL_ENDPTNAK) */
typedef struct{
__REG32 EPRN0     : 1;
__REG32 EPRN1     : 1;
__REG32 EPRN2     : 1;
__REG32 EPRN3     : 1;
__REG32 EPRN4     : 1;
__REG32 EPRN5     : 1;
__REG32 EPRN6     : 1;
__REG32 EPRN7     : 1;
__REG32           : 8;
__REG32 EPTN0     : 1;
__REG32 EPTN1     : 1;
__REG32 EPTN2     : 1;
__REG32 EPTN3     : 1;
__REG32 EPTN4     : 1;
__REG32 EPTN5     : 1;
__REG32 EPTN6     : 1;
__REG32 EPTN7     : 1;
__REG32           : 8;
} __hw_usbctrl_endptnak_bits;

/* Endpoint NAK Enable Register (HW_USBCTRL_ENDPTNAKEN) */
typedef struct{
__REG32 EPRNE0      : 1;
__REG32 EPRNE1      : 1;
__REG32 EPRNE2      : 1;
__REG32 EPRNE3      : 1;
__REG32 EPRNE4      : 1;
__REG32 EPRNE5      : 1;
__REG32 EPRNE6      : 1;
__REG32 EPRNE7      : 1;
__REG32             : 8;
__REG32 EPTNE0      : 1;
__REG32 EPTNE1      : 1;
__REG32 EPTNE2      : 1;
__REG32 EPTNE3      : 1;
__REG32 EPTNE4      : 1;
__REG32 EPTNE5      : 1;
__REG32 EPTNE6      : 1;
__REG32 EPTNE7      : 1;
__REG32             : 8;
} __hw_usbctrl_endptnaken_bits;

/* Port Status and Control 1 Register (HW_USBCTRL_PORTSC1) */
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
__REG32 PTS2      : 1;
__REG32 PSPD      : 2;
__REG32 PTW       : 1;
__REG32 STS       : 1;
__REG32 PTS       : 2;
} __hw_usbctrl_portsc_bits;

/* OTG Status and Control Register (HW_USBCTRL_OTGSC) */
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

/* USB Device Mode Register (HW_USBCTRL_USBMODE) */
typedef struct{
__REG32 CM        : 2;
__REG32 ES        : 1;
__REG32 SLOM      : 1;
__REG32 SDIS      : 1;
__REG32 VBPS      : 1;
__REG32           : 9;
__REG32 SRT       : 1;
__REG32           :16;
} __hw_usbctrl_usbmode_bits;

/* Endpoint Setup Status Register (HW_USBCTRL_ENDPTSETUPSTAT) */
typedef struct{
__REG32 ENDPTSETUPSTAT0   : 1;
__REG32 ENDPTSETUPSTAT1   : 1;
__REG32 ENDPTSETUPSTAT2   : 1;
__REG32 ENDPTSETUPSTAT3   : 1;
__REG32 ENDPTSETUPSTAT4   : 1;
__REG32 ENDPTSETUPSTAT5   : 1;
__REG32 ENDPTSETUPSTAT6   : 1;
__REG32 ENDPTSETUPSTAT7   : 1;
__REG32                   :24;
} __hw_usbctrl_endptsetupstat_bits;

/* Endpoint Initialization Register (HW_USBCTRL_ENDPTPRIME) */
typedef struct{
__REG32 PERB0       : 1;
__REG32 PERB1       : 1;
__REG32 PERB2       : 1;
__REG32 PERB3       : 1;
__REG32 PERB4       : 1;
__REG32 PERB5       : 1;
__REG32 PERB6       : 1;
__REG32 PERB7       : 1;
__REG32             : 8;
__REG32 PETB0       : 1;
__REG32 PETB1       : 1;
__REG32 PETB2       : 1;
__REG32 PETB3       : 1;
__REG32 PETB4       : 1;
__REG32 PETB5       : 1;
__REG32 PETB6       : 1;
__REG32 PETB7       : 1;
__REG32             : 8;
} __hw_usbctrl_endptprime_bits;

/* Endpoint De-Initialize Register (HW_USBCTRL_ENDPTFLUSH) */
typedef struct{
__REG32 FERB0       : 1;
__REG32 FERB1       : 1;
__REG32 FERB2       : 1;
__REG32 FERB3       : 1;
__REG32 FERB4       : 1;
__REG32 FERB5       : 1;
__REG32 FERB6       : 1;
__REG32 FERB7       : 1;
__REG32             : 8;
__REG32 FETB0       : 1;
__REG32 FETB1       : 1;
__REG32 FETB2       : 1;
__REG32 FETB3       : 1;
__REG32 FETB4       : 1;
__REG32 FETB5       : 1;
__REG32 FETB6       : 1;
__REG32 FETB7       : 1;
__REG32             : 8;
} __hw_usbctrl_endptflush_bits;

/* Endpoint Status Register (HW_USBCTRL_ENDPTSTAT) */
typedef struct{
__REG32 ERBR0       : 1;
__REG32 ERBR1       : 1;
__REG32 ERBR2       : 1;
__REG32 ERBR3       : 1;
__REG32 ERBR4       : 1;
__REG32 ERBR5       : 1;
__REG32 ERBR6       : 1;
__REG32 ERBR7       : 1;
__REG32             : 8;
__REG32 ETBR0       : 1;
__REG32 ETBR1       : 1;
__REG32 ETBR2       : 1;
__REG32 ETBR3       : 1;
__REG32 ETBR4       : 1;
__REG32 ETBR5       : 1;
__REG32 ETBR6       : 1;
__REG32 ETBR7       : 1;
__REG32             : 8;
} __hw_usbctrl_endptstat_bits;

/* Endpoint Complete Register (HW_USBCTRL_ENDPTCOMPLETE) */
typedef struct{
__REG32 ERCE0       : 1;
__REG32 ERCE1       : 1;
__REG32 ERCE2       : 1;
__REG32 ERCE3       : 1;
__REG32 ERCE4       : 1;
__REG32 ERCE5       : 1;
__REG32 ERCE6       : 1;
__REG32 ERCE7       : 1;
__REG32             : 8;
__REG32 ETCE0       : 1;
__REG32 ETCE1       : 1;
__REG32 ETCE2       : 1;
__REG32 ETCE3       : 1;
__REG32 ETCE4       : 1;
__REG32 ETCE5       : 1;
__REG32 ETCE6       : 1;
__REG32 ETCE7       : 1;
__REG32             : 8;
} __hw_usbctrl_endptcomplete_bits;

/* Endpoint Control 0 Register (HW_USBCTRL_ENDPTCTRL0) */
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

/* Endpoint Control x Registers (ENDPTCTRLx, x = 1…7) */
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

/* USB PHY Power-Down Register (HW_USBPHY_PWD) */
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

/* USB PHY Transmitter Control Register (HW_USBPHY_TX) */
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

/* USB PHY Receiver Control Register (HW_USBPHY_RX) */
typedef struct{
__REG32 ENVADJ                : 3;
__REG32                       : 1;
__REG32 DISCONADJ             : 3;
__REG32                       :15;
__REG32 RXDBYPASS             : 1;
__REG32                       : 9;
} __hw_usbphy_rx_bits;

/* USB PHY General Control Register (HW_USBPHY_CTRL) */
typedef struct{
__REG32                       : 1;
__REG32 ENHOSTDISCONDETECT    : 1;
__REG32 ENIRQHOSTDISCON       : 1;
__REG32 HOSTDISCONDETECT_IRQ  : 1;
__REG32 ENDEVPLUGINDETECT     : 1;
__REG32 DEVPLUGIN_POLARITY    : 1;
__REG32                       : 1;
__REG32 ENOTGIDDETECT         : 1;
__REG32 RESUMEIRQSTICKY       : 1;
__REG32 ENIRQRESUMEDETECT     : 1;
__REG32 RESUME_IRQ            : 1;
__REG32 ENIRQDEVPLUGIN        : 1;
__REG32 DEVPLUGIN_IRQ         : 1;
__REG32 DATA_ON_LRADC         : 1;
__REG32 ENUTMILEVEL2          : 1;
__REG32 ENUTMILEVEL3          : 1;
__REG32 ENIRQWAKEUP           : 1;
__REG32 WAKEUP_IRQ            : 1;
__REG32 ENAUTO_PWRON_PLL      : 1;
__REG32 ENAUTOCLR_CLKGATE     : 1;
__REG32 ENAUTOCLR_PHY_PWD     : 1;
__REG32 ENDPDMCHG_WKUP        : 1;
__REG32 ENIDCHG_WKUP          : 1;
__REG32 ENVBUSCHG_WKUP        : 1;
__REG32 FSDLL_RST_EN          : 1;
__REG32 ENAUTOCLR_USBCLKGATE  : 1;
__REG32 ENAUTOSET_USBCLKS     : 1;
__REG32                       : 1;
__REG32 HOST_FORCE_LS_SE0     : 1;
__REG32 UTMI_SUSPENDM         : 1;
__REG32 CLKGATE               : 1;
__REG32 SFTRST                : 1;
} __hw_usbphy_ctrl_bits;

/* USB PHY Status Register (HW_USBPHY_STATUS) */
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

/* USB PHY Debug Register (HW_USBPHY_DEBUG) */
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

/* UTMI Debug Status Register 0 (HW_USBPHY_DEBUG0_STATUS) */
typedef struct{
__REG32 LOOP_BACK_FAIL_COUNT    :16;
__REG32 UTMI_RXERROR_FAIL_COUNT :10;
__REG32 SQUELCH_COUNT           : 6;
} __hw_usbphy_debug0_status_bits;

/* UTMI Debug Status Register 1 (HW_USBPHY_DEBUG1) */
typedef struct{
__REG32 DBG_ADDRESS             : 4;
__REG32                         : 8;
__REG32 ENTX2TX                 : 1;
__REG32 ENTAILADJVD             : 2;
__REG32                         :17;
} __hw_usbphy_debug1_bits;

/* UTMI RTL Version (HW_USBPHY_VERSION) */
typedef struct{
__REG32 STEP                    :16;
__REG32 MINOR                   : 8;
__REG32 MAJOR                   : 8;
} __hw_usbphy_version_bits;

/* USB PHY IP Block Register (HW_USBPHY_IP) */
typedef struct{
__REG32 PLL_POWER               : 1;
__REG32 PLL_LOCKED              : 1;
__REG32 EN_USB_CLKS             : 1;
__REG32                         :13;
__REG32 ANALOG_TESTMODE         : 1;
__REG32 TSTI_TX_DM              : 1;
__REG32 TSTI_TX_DP              : 1;
__REG32 CP_SEL                  : 2;
__REG32 LFR_SEL                 : 2;
__REG32 DIV_SEL                 : 2;
__REG32                         : 7;
} __hw_usbphy_ip_bits;

/* LCDIF General Control0 Register (HW_LCDIF_CTRL0) */
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
__REG32 READ_WRITEB         : 1;
__REG32 YCBCR422_INPUT      : 1;
__REG32 CLKGATE             : 1;
__REG32 SFTRST              : 1;
} __hw_lcdif_ctrl_bits;

/* LCDIF General Control1 Register (HW_LCDIF_CTRL1) */
typedef struct {
__REG32 RESET                 : 1;
__REG32 MODE86                : 1;
__REG32 BUSY_ENABLE           : 1;
__REG32                       : 5;
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
__REG32 COMBINE_MPU_WR_STRB   : 1;
__REG32                       : 4;
} __hw_lcdif_ctrl1_bits;

/* LCDIF General Control2 Register (HW_LCDIF_CTRL2) */
typedef struct {
__REG32                       : 1;
__REG32 INITIAL_DUMMY_READ    : 3;
__REG32 RD_MD_NUM_PKG_SUBWORDS: 3;
__REG32                       : 1;
__REG32 RD_MD_6_BIT_INPUT     : 1;
__REG32 RD_MD_OUTPUT_IN_RGB_FR: 1;
__REG32 RD_PACK_DIR           : 1;
__REG32                       : 1;
__REG32 EVEN_LINE_PATTERN     : 3;
__REG32                       : 1;
__REG32 ODD_LINE_PATTERN      : 3;
__REG32                       : 1;
__REG32 BURST_LEN_8           : 1;
__REG32 OUTSTANDING_REQS      : 3;
__REG32                       : 8;
} __hw_lcdif_ctrl2_bits;

/* LCDIF Horizontal and Vertical Valid Data Count Register (HW_LCDIF_TRANSFER_COUNT) */
typedef struct {
__REG32 H_COUNT       :16;
__REG32 V_COUNT       :16;
} __hw_lcdif_transfer_count_bits;

/* LCD Interface Timing Register (HW_LCDIF_TIMING) */
typedef struct {
__REG32 DATA_SETUP    : 8;
__REG32 DATA_HOLD     : 8;
__REG32 CMD_SETUP     : 8;
__REG32 CMD_HOLD      : 8;
} __hw_lcdif_timing_bits;

/* LCDIF VSYNC Mode and Dotclk Mode Control Register0 (HW_LCDIF_VDCTRL0) */
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

/* LCDIF VSYNC Mode and Dotclk Mode Control Register2 (HW_LCDIF_VDCTRL2)*/
typedef struct {
__REG32 HSYNC_PERIOD      :18;
__REG32 HSYNC_PULSE_WIDTH :14;
} __hw_lcdif_vdctrl2_bits;

/* LCDIF VSYNC Mode and Dotclk Mode Control Register3 (HW_LCDIF_VDCTRL3) */
typedef struct {
__REG32 VWAIT_CNT         :16;
__REG32 HWAIT_CNT         :12;
__REG32 VSYNC_ONLY        : 1;
__REG32 MUX_SYNC_SIGNALS  : 1;
__REG32                   : 2;
} __hw_lcdif_vdctrl3_bits;

/* LCDIF VSYNC Mode and Dotclk Mode Control Register4 (HW_LCDIF_VDCTRL4) */
typedef struct {
__REG32 DOTCLK_HVLD_D_CNT :18;
__REG32 SYNC_SIGNALS_ON   : 1;
__REG32                   :10;
__REG32 DOTCLK_DLY_SEL    : 3;
} __hw_lcdif_vdctrl4_bits;

/* Digital Video Interface Control0 Register (HW_LCDIF_DVICTRL0) */
typedef struct {
__REG32 H_BLANKING_CNT    :12;
__REG32                   : 4;
__REG32 H_ACTIVE_CNT      :12;
__REG32                   : 4;
} __hw_lcdif_dvictrl0_bits;

/* Digital Video Interface Control1 Register (HW_LCDIF_DVICTRL1) */
typedef struct {
__REG32 F2_START_LINE     :10;
__REG32 F1_END_LINE       :10;
__REG32 F1_START_LINE     :10;
__REG32                   : 2;
} __hw_lcdif_dvictrl1_bits;

/* Digital Video Interface Control2 Register (HW_LCDIF_DVICTRL2) */
typedef struct {
__REG32 V1_BLANK_END_LINE   :10;
__REG32 V1_BLANK_START_LINE :10;
__REG32 F2_END_LINE         :10;
__REG32                     : 2;
} __hw_lcdif_dvictrl2_bits;

/* Digital Video Interface Control3 Register (HW_LCDIF_DVICTRL3) */
typedef struct {
__REG32 V_LINES_CNT         :10;
__REG32 V2_BLANK_END_LINE   :10;
__REG32 V2_BLANK_START_LINE :10;
__REG32                     : 2;
} __hw_lcdif_dvictrl3_bits;

/* Digital Video Interface Control4 Register (HW_LCDIF_DVICTRL4) */
typedef struct {
__REG32 H_FILL_CNT        : 8;
__REG32 CR_FILL_VALUE     : 8;
__REG32 CB_FILL_VALUE     : 8;
__REG32 Y_FILL_VALUE      : 8;
} __hw_lcdif_dvictrl4_bits;

/* RGB to YCbCr 4:2:2 CSC Coefficient0 Register (HW_LCDIF_CSC_COEFF0) */
typedef struct {
__REG32 CSC_SUBSAMPLE_FILTER  : 2;
__REG32                       :14;
__REG32 C0                    :10;
__REG32                       : 6;
} __hw_lcdif_csc_coeff0_bits;

/* RGB to YCbCr 4:2:2 CSC Coefficient1 Register (HW_LCDIF_CSC_COEFF1) */
typedef struct {
__REG32 C1            :10;
__REG32               : 6;
__REG32 C2            :10;
__REG32               : 6;
} __hw_lcdif_csc_coeff1_bits;

/* RGB to YCbCr 4:2:2 CSC Coefficent2 Register (HW_LCDIF_CSC_COEFF2) */
typedef struct {
__REG32 C3            :10;
__REG32               : 6;
__REG32 C4            :10;
__REG32               : 6;
} __hw_lcdif_csc_coeff2_bits;

/* RGB to YCbCr 4:2:2 CSC Coefficient3 Register (HW_LCDIF_CSC_COEFF3) */
typedef struct {
__REG32 C5            :10;
__REG32               : 6;
__REG32 C6            :10;
__REG32               : 6;
} __hw_lcdif_csc_coeff3_bits;

/* RGB to YCbCr 4:2:2 CSC Coefficient4 Register (HW_LCDIF_CSC_COEFF4)  */
typedef struct {
__REG32 C7            :10;
__REG32               : 6;
__REG32 C8            :10;
__REG32               : 6;
} __hw_lcdif_csc_coeff4_bits;

/* RGB to YCbCr 4:2:2 CSC Offset Register (HW_LCDIF_CSC_OFFSET) */
typedef struct {
__REG32 Y_OFFSET      : 9;
__REG32               : 7;
__REG32 CBCR_OFFSET   : 9;
__REG32               : 7;
} __hw_lcdif_csc_offset_bits;

/* RGB to YCbCr 4:2:2 CSC Limit Register (HW_LCDIF_CSC_LIMIT) */
typedef struct {
__REG32 Y_MAX         : 8;
__REG32 Y_MIN         : 8;
__REG32 CBCR_MAX      : 8;
__REG32 CBCR_MIN      : 8;
} __hw_lcdif_csc_limit_bits;

/* LCD Interface Data Register (HW_LCDIF_DATA) */
typedef struct {
__REG32 DATA_ZERO     : 8;
__REG32 DATA_ONE      : 8;
__REG32 DATA_TWO      : 8;
__REG32 DATA_THREE    : 8;
} __hw_lcdif_data_bits;

/* LCD Interface Status Register (HW_LCDIF_STAT) */
typedef struct {
__REG32 LFIFO_COUNT       : 9;
__REG32                   :15;
__REG32 DVI_CURRENT_FIELD : 1;
__REG32 BUSY              : 1;
__REG32 TXFIFO_EMPTY      : 1;
__REG32 TXFIFO_FULL       : 1;
__REG32 LFIFO_EMPTY       : 1;
__REG32 LFIFO_FULL        : 1;
__REG32 DMA_REQ           : 1;
__REG32 PRESENT           : 1;
} __hw_lcdif_stat_bits;

/* LCD Interface Version Register (HW_LCDIF_VERSION) */
typedef struct {
__REG32 STEP              :16;
__REG32 MINOR             : 8;
__REG32 MAJOR             : 8;
} __hw_lcdif_version_bits;

/* LCD Interface Debug0 Register (HW_LCDIF_DEBUG0) */
typedef struct {
__REG32 MST_WORDS           : 4;
__REG32 MST_OUTSTANDING_REQS: 5;
__REG32 MST_AVALID          : 1;
__REG32 CUR_REQ_STATE       : 2;
__REG32                     : 4;
__REG32 CUR_STATE           : 7;
__REG32 EMPTY_WORD          : 1;
__REG32 CUR_FRAME_TX        : 1;
__REG32 VSYNC               : 1;
__REG32 HSYNC               : 1;
__REG32 ENABLE              : 1;
__REG32 DMACMDKICK          : 1;
__REG32 SYNC_SIGNALS_ON_REG : 1;
__REG32 W_FOR_VSYNC_EDG_OUT : 1;
__REG32 STRMNG_END_DTC      : 1;
} __hw_lcdif_debug0_bits;

/* LCD Interface Debug1 Register (HW_LCDIF_DEBUG1) */
typedef struct {
__REG32 V_DATA_COUNT        :16;
__REG32 H_DATA_COUNT        :16;
} __hw_lcdif_debug1_bits;

/* PXP Control Register 0 (HW_PXP_CTRL) */
typedef struct {
__REG32 ENABLE            : 1;
__REG32 IRQ_ENABLE        : 1;
__REG32 NEXT_IRQ_ENABLE   : 1;
__REG32                   : 1;
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
__REG32 BLOCK_SIZE        : 1;
__REG32 INTERLACED_INPUT  : 2;
__REG32 INTERLACED_OUTPUT : 2;
__REG32                   : 2;
__REG32 CLKGATE           : 1;
__REG32 SFTRST            : 1;
} __hw_pxp_ctrl_bits;

/* PXP Status Register (HW_PXP_STAT) */
typedef struct {
__REG32 IRQ               : 1;
__REG32 AXI_WRITE_ERROR   : 1;
__REG32 AXI_READ_ERROR    : 1;
__REG32 NEXT_IRQ          : 1;
__REG32 AXI_ERROR_ID      : 4;
__REG32                   : 8;
__REG32 BLOCKY            : 8;
__REG32 BLOCKX            : 8;
} __hw_pxp_stat_bits;

/* PXP Output Buffer Size (HW_PXP_OUTSIZE) */
typedef struct {
__REG32 HEIGHT            :12;
__REG32 WIDTH             :12;
__REG32 ALPHA             : 8;
} __hw_pxp_rgbsize_bits;

/* PXP Source 0 (video) Buffer Parameters (HW_PXP_S0PARAM) */
typedef struct {
__REG32 HEIGHT            : 8;
__REG32 WIDTH             : 8;
__REG32 YBASE             : 8;
__REG32 XBASE             : 8;
} __hw_pxp_s0param_bits;

/* Source 0 Cropping Register (HW_PXP_S0CROP) */
typedef struct {
__REG32 HEIGHT            : 8;
__REG32 WIDTH             : 8;
__REG32 YBASE             : 8;
__REG32 XBASE             : 8;
} __hw_pxp_s0crop_bits;

/* Source 0 Scale Factor Register (HW_PXP_S0SCALE) */
typedef struct {
__REG32 XSCALE            :15;
__REG32                   : 1;
__REG32 YSCALE            :15;
__REG32                   : 1;
} __hw_pxp_s0scale_bits;

/* Source 0 Scale Offset Register (HW_PXP_S0OFFSET) */
typedef struct {
__REG32 XOFFSET           :12;
__REG32                   : 4;
__REG32 YOFFSET           :12;
__REG32                   : 4;
} __hw_pxp_s0offset_bits;

/* Color Space Conversion Coefficient Register 0 (HW_PXP_CSCCOEFF0) */
typedef struct {
__REG32 Y_OFFSET          : 9;
__REG32 UV_OFFSET         : 9;
__REG32 C0                :11;
__REG32                   : 2;
__REG32 YCBCR_MODE        : 1;
} __hw_pxp_csccoeff0_bits;

/* Color Space Conversion Coefficient Register 1 (HW_PXP_CSCCOEFF1)*/
typedef struct {
__REG32 C4                :11;
__REG32                   : 5;
__REG32 C1                :11;
__REG32                   : 5;
} __hw_pxp_csccoeff1_bits;

/* Color Space Conversion Coefficient Register 2 (HW_PXP_CSCCOEFF2) */
typedef struct {
__REG32 C3                :11;
__REG32                   : 5;
__REG32 C2                :11;
__REG32                   : 5;
} __hw_pxp_csccoeff2_bits;

/* PXP Next Frame Pointer (HW_PXP_NEXT) */
typedef struct {
__REG32 ENABLED           : 1;
__REG32                   : 1;
__REG32 POINTER           :30;
} __hw_pxp_next_bits;

/* PXP S0 Color Key Low (HW_PXP_S0COLORKEYLOW) */
typedef struct {
__REG32 PIXEL             :24;
__REG32                   : 8;
} __hw_pxp_s0colorkeylow_bits;

/* PXP S0 Color Key High (HW_PXP_S0COLORKEYHIGH) */
typedef struct {
__REG32 PIXEL             :24;
__REG32                   : 8;
} __hw_pxp_s0colorkeyhigh_bits;

/* PXP Overlay Color Key Low (HW_PXP_OLCOLORKEYLOW) */
typedef struct {
__REG32 PIXEL             :24;
__REG32                   : 8;
} __hw_pxp_olcolorkeylow_bits;

/* PXP Overlay Color Key High (HW_PXP_OLCOLORKEYHIGH) */
typedef struct {
__REG32 PIXEL             :24;
__REG32                   : 8;
} __hw_pxp_olcolorkeyhigh_bits;

/* PXP Debug Control Register (HW_PXP_DEBUGCTRL) */
typedef struct {
__REG32 SELECT            : 8;
__REG32 RESET_TLB_STATS   : 1;
__REG32                   :23;
} __hw_pxp_debugctrl_bits;

/* PXP Version Register (HW_PXP_VERSION) */
typedef struct {
__REG32 STEP              :16;
__REG32 MINOR             : 8;
__REG32 MAJOR             : 8;
} __hw_pxp_version_bits;

/* PXP Overlay 0 Size (HW_PXP_OL0SIZE) */
typedef struct {
__REG32 HEIGHT            : 8;
__REG32 WIDTH             : 8;
__REG32 YBASE             : 8;
__REG32 XBASE             : 8;
} __hw_pxp_olxsize_bits;

/* PXP Overlay 0 Parameters (HW_PXP_OL0PARAM) */
typedef struct {
__REG32 ENABLE            : 1;
__REG32 ALPHA_CNTL        : 2;
__REG32 ENABLE_COLORKEY   : 1;
__REG32 FORMAT            : 4;
__REG32 ALPHA             : 8;
__REG32 ROP               : 4;
__REG32                   :12;
} __hw_pxp_olxparam_bits;

/* SAIF Control Register (HW_SAIF_CTRL) */
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
__REG32 LRCLK_PULSE         : 1;
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

/* SAIF Status Register (HW_SAIF_STAT) */
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

/* SAIF Data Register (HW_SAIF_DATA) */
typedef struct {
__REG32 PCM_LEFT            :16;
__REG32 PCM_RIGHT           :16;
} __hw_saif_data_bits;

/* SAIF Version Register (HW_SAIF_VERSION) */
typedef struct {
__REG32 STEP                :16;
__REG32 MINOR               : 8;
__REG32 MAJOR               : 8;
} __hw_saif_version_bits;

/* SPDIF Control Register (HW_SPDIF_CTRL) */
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

/* SPDIF Status Register (HW_SPDIF_STAT) */
typedef struct {
__REG32 END_XFER            : 1;
__REG32                     :30;
__REG32 PRESENT             : 1;
} __hw_spdif_stat_bits;

/* SPDIF Frame Control Register (HW_SPDIF_FRAMECTRL) */
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

/* SPDIF Sample Rate Register (HW_SPDIF_SRR) */
typedef struct {
__REG32 RATE                :20;
__REG32                     : 8;
__REG32 BASEMULT            : 3;
__REG32                     : 1;
} __hw_spdif_srr_bits;

/* SPDIF Debug Register (HW_SPDIF_DEBUG) */
typedef struct {
__REG32 FIFO_STATUS         : 1;
__REG32 DMA_PREQ            : 1;
__REG32                     :30;
} __hw_spdif_debug_bits;

/* SPDIF Write Data Register (HW_SPDIF_DATA) */
typedef struct {
__REG32 LOW                 :16;
__REG32 HIGH                :16;
} __hw_spdif_data_bits;

/* SPDIF Version Register (HW_SPDIF_VERSION) */
typedef struct {
__REG32 STEP                :16;
__REG32 MINOR               : 8;
__REG32 MAJOR               : 8;
} __hw_spdif_version_bits;

/* HSADC Control Register 0 (HW_HSADC_CTRL0) */
typedef struct {
__REG32 HSADC_RUN             : 1;
__REG32 TDC                   : 5;
__REG32 HSADC_PRESENT         : 1;
__REG32                       : 5;
__REG32 ADCSSB                : 3;
__REG32 ADCSHWS               : 1;
__REG32 ADCSE                 : 1;
__REG32 ADCSP                 : 2;
__REG32 DISCARD               : 2;
__REG32                       : 6;
__REG32 SW_TRIG               : 1;
__REG32 TRIG_SRC              : 2;
__REG32 CLKGATE               : 1;
__REG32 SFTRST                : 1;
} __hw_hsadc_ctrl0_bits;

/* HSADC Control Register 1 (HW_HSADC_CTRL1) */
typedef struct {
__REG32 INTR                  : 1;
__REG32 ITOS                  : 1;
__REG32 IFIFOOVS              : 1;
__REG32 IADCDS                : 1;
__REG32 IEOSS                 : 1;
__REG32 FIFORE                : 1;
__REG32                       :20;
__REG32 ISCLR                 : 1;
__REG32 ICLR                  : 1;
__REG32 ITOE                  : 1;
__REG32 IFIFOOVE              : 1;
__REG32 IADCDE                : 1;
__REG32 IEOSE                 : 1;
} __hw_hsadc_ctrl1_bits;

/* HSADC Control Register 2 (HW_HSADC_CTRL2) */
typedef struct {
__REG32 ADC_PRECHARGE         : 1;
__REG32 ADC_CHANNEL_SEL       : 3;
__REG32 ADC_SH_BYPASS         : 1;
__REG32 ONCHIP_GROUND         : 1;
__REG32 SAH_BIAS_ADJ          : 2;
__REG32 DAC_ADJCURRENT        : 1;
__REG32 DAC_ADJHEADROOM       : 1;
__REG32 SAH_GAIN_ADJ          : 3;
__REG32 POWER_DOWN            : 1;
__REG32                       :18;
} __hw_hsadc_ctrl2_bits;

/* HSADC Debug Information 0 Register (HW_HSADC_DBG_INFO0) */
typedef struct {
__REG32 HSADC_FSM_STATE       : 3;
__REG32 DMA_FSM_STATE         : 3;
__REG32                       :26;
} __hw_hsadc_dbg_info0_bits;

/* HSADC Version Register (HW_HSADC_VERSION) */
typedef struct {
__REG32 STEP                  :16;
__REG32 MINOR                 : 8;
__REG32 MAJOR                 : 8;
} __hw_hsadc_version_bits;

/* LRADC Control Register 0 (HW_LRADC_CTRL0) */
typedef struct {
__REG32 SCHEDULE_CH0          : 1;
__REG32 SCHEDULE_CH1          : 1;
__REG32 SCHEDULE_CH2          : 1;
__REG32 SCHEDULE_CH3          : 1;
__REG32 SCHEDULE_CH4          : 1;
__REG32 SCHEDULE_CH5          : 1;
__REG32 SCHEDULE_CH6          : 1;
__REG32 SCHEDULE_CH7          : 1;
__REG32                       : 8;
__REG32 XPULSW                : 1;
__REG32 XNURSW                : 2;
__REG32 YPLLSW                : 2;
__REG32 YNLRSW                : 1;
__REG32 TOUCH_SCREEN_TYPE     : 1;
__REG32 TOUCH_DETECT_ENABLE   : 1;
__REG32 BUTTON0_DETECT_ENABLE : 1;
__REG32 BUTTON1_DETECT_ENABLE : 1;
__REG32 ONCHIP_GROUNDREF      : 1;
__REG32                       : 3;
__REG32 CLKGATE               : 1;
__REG32 SFTRST                : 1;
} __hw_lradc_ctrl0_bits;

/* LRADC Control Register 1 (HW_LRADC_CTRL1) */
typedef struct {
__REG32 LRADC0_IRQ              : 1;
__REG32 LRADC1_IRQ              : 1;
__REG32 LRADC2_IRQ              : 1;
__REG32 LRADC3_IRQ              : 1;
__REG32 LRADC4_IRQ              : 1;
__REG32 LRADC5_IRQ              : 1;
__REG32 LRADC6_IRQ              : 1;
__REG32 LRADC7_IRQ              : 1;
__REG32 TOUCH_DETECT_IRQ        : 1;
__REG32 THRESHOLD0_DETECT_IRQ   : 1;
__REG32 THRESHOLD1_DETECT_IRQ   : 1;
__REG32 BUTTON0_DETECT_IRQ      : 1;
__REG32 BUTTON1_DETECT_IRQ      : 1;
__REG32                         : 3;
__REG32 LRADC0_IRQ_EN           : 1;
__REG32 LRADC1_IRQ_EN           : 1;
__REG32 LRADC2_IRQ_EN           : 1;
__REG32 LRADC3_IRQ_EN           : 1;
__REG32 LRADC4_IRQ_EN           : 1;
__REG32 LRADC5_IRQ_EN           : 1;
__REG32 LRADC6_IRQ_EN           : 1;
__REG32 LRADC7_IRQ_EN           : 1;
__REG32 TOUCH_DETECT_IRQ_EN     : 1;
__REG32 THRESHOLD0_DETECT_IRQ_EN: 1;
__REG32 THRESHOLD1_DETECT_IRQ_EN: 1;
__REG32 BUTTON0_DETECT_IRQ_EN   : 1;
__REG32 BUTTON1_DETECT_IRQ_EN   : 1;
__REG32                         : 3;
} __hw_lradc_ctrl1_bits;

/* LRADC Control Register 2 (HW_LRADC_CTRL2) */
typedef struct {
__REG32 TEMP_ISRC0            : 4;
__REG32 TEMP_ISRC1            : 4;
__REG32 TEMP_SENSOR_IENABLE0  : 1;
__REG32 TEMP_SENSOR_IENABLE1  : 1;
__REG32                       : 2;
__REG32 DISABLE_MUXAMP_BYPASS : 1;
__REG32 VTHSENSE              : 2;
__REG32 TEMPSENSE_PWD         : 1;
__REG32                       : 8;
__REG32 DIVIDE_BY_TWO         : 8;
} __hw_lradc_ctrl2_bits;

/* LRADC Control Register 3 (HW_LRADC_CTRL3) */
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

/* LRADC Status Register (HW_LRADC_STATUS) */
typedef struct {
__REG32 TOUCH_DETECT_RAW      : 1;
__REG32 BUTTON0_DETECT_RAW    : 1;
__REG32 BUTTON1_DETECT_RAW    : 1;
__REG32                       :13;
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
__REG32 BUTTON0_PRESENT       : 1;
__REG32 BUTTON1_PRESENT       : 1;
__REG32                       : 3;
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

/* LRADC Scheduling Delay 0 (HW_LRADC_DELAY0) */
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

/* LRADC Debug Register 0 (HW_LRADC_DEBUG0) */
typedef struct {
__REG32 STATE                 :12;
__REG32                       : 4;
__REG32 READONLY              :16;
} __hw_lradc_debug0_bits;

/* LRADC Debug Register 1 (HW_LRADC_DEBUG1) */
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

/* LRADC Battery Conversion Register (HW_LRADC_CONVERSION) */
typedef struct {
__REG32 SCALED_BATT_VOLTAGE   :10;
__REG32                       : 6;
__REG32 SCALE_FACTOR          : 2;
__REG32                       : 2;
__REG32 AUTOMATIC             : 1;
__REG32                       :11;
} __hw_lradc_conversion_bits;

/* LRADC Theshold0 Register (HW_LRADC_THRESHOLDx) */
typedef struct {
__REG32 VALUE                 :18;
__REG32 SETTING               : 2;
__REG32 CHANNEL_SEL           : 3;
__REG32 BATTCHRG_DISABLE      : 1;
__REG32 ENABLE                : 1;
__REG32                       : 7;
} __hw_lradc_threshold_bits;

/* LRADC Control Register 4 (HW_LRADC_CTRL4) */
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

#endif    /* __IAR_SYSTEMS_ICC__ */

/* Common declarations  ****************************************************/
/***************************************************************************
 **
 **  ICOLL
 **
 ***************************************************************************/
__IO_REG32(    HW_ICOLL_VECTOR,           0x80000000,__READ_WRITE );
__IO_REG32(    HW_ICOLL_VECTOR_SET,       0x80000004,__WRITE      );
__IO_REG32(    HW_ICOLL_VECTOR_CLR,       0x80000008,__WRITE      );
__IO_REG32(    HW_ICOLL_VECTOR_TOG,       0x8000000C,__WRITE      );
__IO_REG32_BIT(HW_ICOLL_LEVELACK,         0x80000010,__READ_WRITE ,__hw_icoll_levelack_bits);
__IO_REG32_BIT(HW_ICOLL_CTRL,             0x80000020,__READ_WRITE ,__hw_icoll_ctrl_bits);
__IO_REG32_BIT(HW_ICOLL_CTRL_SET,         0x80000024,__READ_WRITE ,__hw_icoll_ctrl_bits);
__IO_REG32_BIT(HW_ICOLL_CTRL_CLR,         0x80000028,__READ_WRITE ,__hw_icoll_ctrl_bits);
__IO_REG32_BIT(HW_ICOLL_CTRL_TOG,         0x8000002C,__READ_WRITE ,__hw_icoll_ctrl_bits);
__IO_REG32(    HW_ICOLL_VBASE,            0x80000040,__READ_WRITE );
__IO_REG32(    HW_ICOLL_VBASE_SET,        0x80000044,__WRITE      );
__IO_REG32(    HW_ICOLL_VBASE_CLR,        0x80000048,__WRITE      );
__IO_REG32(    HW_ICOLL_VBASE_TOG,        0x8000004C,__WRITE      );
__IO_REG32_BIT(HW_ICOLL_STAT,             0x80000070,__READ       ,__hw_icoll_stat_bits);
__IO_REG32_BIT(HW_ICOLL_RAW0,             0x800000A0,__READ       ,__hw_icoll_raw0_bits);
__IO_REG32_BIT(HW_ICOLL_RAW1,             0x800000B0,__READ       ,__hw_icoll_raw1_bits);
__IO_REG32_BIT(HW_ICOLL_RAW2,             0x800000C0,__READ       ,__hw_icoll_raw2_bits);
__IO_REG32_BIT(HW_ICOLL_RAW3,             0x800000D0,__READ       ,__hw_icoll_raw3_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT0,       0x80000120,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT0_SET,   0x80000124,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT0_CLR,   0x80000128,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT0_TOG,   0x8000012C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT1,       0x80000130,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT1_SET,   0x80000134,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT1_CLR,   0x80000138,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT1_TOG,   0x8000013C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT2,       0x80000140,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT2_SET,   0x80000144,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT2_CLR,   0x80000148,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT2_TOG,   0x8000014C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT3,       0x80000150,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT3_SET,   0x80000154,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT3_CLR,   0x80000158,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT3_TOG,   0x8000015C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT4,       0x80000160,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT4_SET,   0x80000164,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT4_CLR,   0x80000168,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT4_TOG,   0x8000016C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT5,       0x80000170,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT5_SET,   0x80000174,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT5_CLR,   0x80000178,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT5_TOG,   0x8000017C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT6,       0x80000180,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT6_SET,   0x80000184,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT6_CLR,   0x80000188,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT6_TOG,   0x8000018C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT7,       0x80000190,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT7_SET,   0x80000194,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT7_CLR,   0x80000198,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT7_TOG,   0x8000019C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT8,       0x800001A0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT8_SET,   0x800001A4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT8_CLR,   0x800001A8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT8_TOG,   0x800001AC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT9,       0x800001B0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT9_SET,   0x800001B4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT9_CLR,   0x800001B8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT9_TOG,   0x800001BC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT10,      0x800001C0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET10,  0x800001C4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR10,  0x800001C8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG10,  0x800001CC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT11,      0x800001D0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET11,  0x800001D4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR11,  0x800001D8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG11,  0x800001DC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT12,      0x800001E0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET12,  0x800001E4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR12,  0x800001E8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG12,  0x800001EC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT13,      0x800001F0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET13,  0x800001F4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR13,  0x800001F8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG13,  0x800001FC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT14,      0x80000200,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET14,  0x80000204,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR14,  0x80000208,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG14,  0x8000020C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT15,      0x80000210,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET15,  0x80000214,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR15,  0x80000218,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG15,  0x8000021C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT16,      0x80000220,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET16,  0x80000224,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR16,  0x80000228,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG16,  0x8000022C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT17,      0x80000230,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET17,  0x80000234,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR17,  0x80000238,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG17,  0x8000023C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT18,      0x80000240,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET18,  0x80000244,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR18,  0x80000248,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG18,  0x8000024C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT19,      0x80000250,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET19,  0x80000254,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR19,  0x80000258,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG19,  0x8000025C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT20,      0x80000260,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET20,  0x80000264,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR20,  0x80000268,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG20,  0x8000026C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT21,      0x80000270,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET21,  0x80000274,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR21,  0x80000278,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG21,  0x8000027C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT22,      0x80000280,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET22,  0x80000284,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR22,  0x80000288,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG22,  0x8000028C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT23,      0x80000290,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET23,  0x80000294,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR23,  0x80000298,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG23,  0x8000029C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT24,      0x800002A0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET24,  0x800002A4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR24,  0x800002A8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG24,  0x800002AC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT25,      0x800002B0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET25,  0x800002B4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR25,  0x800002B8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG25,  0x800002BC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT26,      0x800002C0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET26,  0x800002C4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR26,  0x800002C8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG26,  0x800002CC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT27,      0x800002D0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET27,  0x800002D4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR27,  0x800002D8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG27,  0x800002DC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT28,      0x800002E0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET28,  0x800002E4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR28,  0x800002E8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG28,  0x800002EC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT29,      0x800002F0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET29,  0x800002F4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR29,  0x800002F8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG29,  0x800002FC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT30,      0x80000300,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET30,  0x80000304,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR30,  0x80000308,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG30,  0x8000030C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT31,      0x80000310,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET31,  0x80000314,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR31,  0x80000318,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG31,  0x8000031C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT32,      0x80000320,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET32,  0x80000324,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR32,  0x80000328,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG32,  0x8000032C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT33,      0x80000330,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET33,  0x80000334,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR33,  0x80000338,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG33,  0x8000033C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT34,      0x80000340,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET34,  0x80000344,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR34,  0x80000348,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG34,  0x8000034C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT35,      0x80000350,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET35,  0x80000354,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR35,  0x80000358,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG35,  0x8000035C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT36,      0x80000360,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET36,  0x80000364,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR36,  0x80000368,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG36,  0x8000036C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT37,      0x80000370,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET37,  0x80000374,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR37,  0x80000378,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG37,  0x8000037C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT38,      0x80000380,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET38,  0x80000384,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR38,  0x80000388,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG38,  0x8000038C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT39,      0x80000390,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET39,  0x80000394,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR39,  0x80000398,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG39,  0x8000039C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT40,      0x800003A0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET40,  0x800003A4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR40,  0x800003A8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG40,  0x800003AC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT41,      0x800003B0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET41,  0x800003B4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR41,  0x800003B8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG41,  0x800003BC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT42,      0x800003C0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET42,  0x800003C4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR42,  0x800003C8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG42,  0x800003CC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT43,      0x800003D0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET43,  0x800003D4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR43,  0x800003D8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG43,  0x800003DC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT44,      0x800003E0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET44,  0x800003E4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR44,  0x800003E8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG44,  0x800003EC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT45,      0x800003F0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET45,  0x800003F4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR45,  0x800003F8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG45,  0x800003FC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT46,      0x80000400,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET46,  0x80000404,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR46,  0x80000408,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG46,  0x8000040C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT47,      0x80000410,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET47,  0x80000414,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR47,  0x80000418,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG47,  0x8000041C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT48,      0x80000420,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET48,  0x80000424,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR48,  0x80000428,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG48,  0x8000042C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT49,      0x80000430,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET49,  0x80000434,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR49,  0x80000438,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG49,  0x8000043C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT50,      0x80000440,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET50,  0x80000444,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR50,  0x80000448,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG50,  0x8000044C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT51,      0x80000450,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET51,  0x80000454,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR51,  0x80000458,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG51,  0x8000045C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT52,      0x80000460,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET52,  0x80000464,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR52,  0x80000468,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG52,  0x8000046C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT53,      0x80000470,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET53,  0x80000474,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR53,  0x80000478,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG53,  0x8000047C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT54,      0x80000480,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET54,  0x80000484,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR54,  0x80000488,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG54,  0x8000048C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT55,      0x80000490,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET55,  0x80000494,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR55,  0x80000498,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG55,  0x8000049C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT56,      0x800004A0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET56,  0x800004A4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR56,  0x800004A8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG56,  0x800004AC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT57,      0x800004B0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET57,  0x800004B4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR57,  0x800004B8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG57,  0x800004BC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT58,      0x800004C0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET58,  0x800004C4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR58,  0x800004C8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG58,  0x800004CC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT59,      0x800004D0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET59,  0x800004D4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR59,  0x800004D8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG59,  0x800004DC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT60,      0x800004E0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET60,  0x800004E4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR60,  0x800004E8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG60,  0x800004EC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT61,      0x800004F0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET61,  0x800004F4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR61,  0x800004F8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG61,  0x800004FC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT62,      0x80000500,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET62,  0x80000504,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR62,  0x80000508,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG62,  0x8000050C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT63,      0x80000510,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET63,  0x80000514,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR63,  0x80000518,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG63,  0x8000051C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT64,      0x80000520,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET64,  0x80000524,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR64,  0x80000528,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG64,  0x8000052C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT65,      0x80000530,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET65,  0x80000534,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR65,  0x80000538,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG65,  0x8000053C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT66,      0x80000540,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET66,  0x80000544,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR66,  0x80000548,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG66,  0x8000054C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT67,      0x80000550,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET67,  0x80000554,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR67,  0x80000558,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG67,  0x8000055C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT68,      0x80000560,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET68,  0x80000564,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR68,  0x80000568,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG68,  0x8000056C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT69,      0x80000570,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET69,  0x80000574,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR69,  0x80000578,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG69,  0x8000057C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT70,      0x80000580,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET70,  0x80000584,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR70,  0x80000588,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG70,  0x8000058C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT71,      0x80000590,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET71,  0x80000594,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR71,  0x80000598,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG71,  0x8000059C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT72,      0x800005A0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET72,  0x800005A4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR72,  0x800005A8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG72,  0x800005AC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT73,      0x800005B0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET73,  0x800005B4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR73,  0x800005B8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG73,  0x800005BC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT74,      0x800005C0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET74,  0x800005C4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR74,  0x800005C8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG74,  0x800005CC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT75,      0x800005D0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET75,  0x800005D4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR75,  0x800005D8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG75,  0x800005DC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT76,      0x800005E0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET76,  0x800005E4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR76,  0x800005E8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG76,  0x800005EC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT77,      0x800005F0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET77,  0x800005F4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR77,  0x800005F8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG77,  0x800005FC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT78,      0x80000600,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET78,  0x80000604,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR78,  0x80000608,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG78,  0x8000060C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT79,      0x80000610,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET79,  0x80000614,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR79,  0x80000618,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG79,  0x8000061C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT80,      0x80000620,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET80,  0x80000624,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR80,  0x80000628,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG80,  0x8000062C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT81,      0x80000630,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET81,  0x80000634,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR81,  0x80000638,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG81,  0x8000063C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT82,      0x80000640,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET82,  0x80000644,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR82,  0x80000648,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG82,  0x8000064C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT83,      0x80000650,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET83,  0x80000654,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR83,  0x80000658,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG83,  0x8000065C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT84,      0x80000660,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET84,  0x80000664,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR84,  0x80000668,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG84,  0x8000066C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT85,      0x80000670,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET85,  0x80000674,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR85,  0x80000678,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG85,  0x8000067C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT86,      0x80000680,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET86,  0x80000684,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR86,  0x80000688,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG86,  0x8000068C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT87,      0x80000690,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET87,  0x80000694,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR87,  0x80000698,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG87,  0x8000069C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT88,      0x800006A0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET88,  0x800006A4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR88,  0x800006A8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG88,  0x800006AC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT89,      0x800006B0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET89,  0x800006B4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR89,  0x800006B8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG89,  0x800006BC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT90,      0x800006C0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET90,  0x800006C4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR90,  0x800006C8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG90,  0x800006CC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT91,      0x800006D0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET91,  0x800006D4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR91,  0x800006D8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG91,  0x800006DC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT92,      0x800006E0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET92,  0x800006E4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR92,  0x800006E8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG92,  0x800006EC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT93,      0x800006F0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET93,  0x800006F4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR93,  0x800006F8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG93,  0x800006FC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT94,      0x80000700,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET94,  0x80000704,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR94,  0x80000708,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG94,  0x8000070C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT95,      0x80000710,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET95,  0x80000714,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR95,  0x80000718,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG95,  0x8000071C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT96,      0x80000720,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET96,  0x80000724,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR96,  0x80000728,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG96,  0x8000072C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT97,      0x80000730,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET97,  0x80000734,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR97,  0x80000738,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG97,  0x8000073C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT98,      0x80000740,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET98,  0x80000744,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR98,  0x80000748,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG98,  0x8000074C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT99,      0x80000750,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET99,  0x80000754,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR99,  0x80000758,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG99,  0x8000075C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT100,     0x80000760,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET100, 0x80000764,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR100, 0x80000768,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG100, 0x8000076C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT101,     0x80000770,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET101, 0x80000774,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR101, 0x80000778,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG101, 0x8000077C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT102,     0x80000780,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET102, 0x80000784,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR102, 0x80000788,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG102, 0x8000078C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT103,     0x80000790,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET103, 0x80000794,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR103, 0x80000798,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG103, 0x8000079C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT104,     0x800007A0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET104, 0x800007A4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR104, 0x800007A8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG104, 0x800007AC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT105,     0x800007B0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET105, 0x800007B4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR105, 0x800007B8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG105, 0x800007BC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT106,     0x800007C0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET106, 0x800007C4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR106, 0x800007C8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG106, 0x800007CC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT107,     0x800007D0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET107, 0x800007D4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR107, 0x800007D8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG107, 0x800007DC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT108,     0x800007E0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET108, 0x800007E4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR108, 0x800007E8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG108, 0x800007EC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT109,     0x800007F0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET109, 0x800007F4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR109, 0x800007F8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG109, 0x800007FC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT110,     0x80000800,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET110, 0x80000804,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR110, 0x80000808,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG110, 0x8000080C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT111,     0x80000810,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET111, 0x80000814,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR111, 0x80000818,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG111, 0x8000081C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT112,     0x80000820,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET112, 0x80000824,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR112, 0x80000828,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG112, 0x8000082C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT113,     0x80000830,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET113, 0x80000834,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR113, 0x80000838,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG113, 0x8000083C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT114,     0x80000840,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET114, 0x80000844,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR114, 0x80000848,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG114, 0x8000084C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT115,     0x80000850,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET115, 0x80000854,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR115, 0x80000858,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG115, 0x8000085C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT116,     0x80000860,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET116, 0x80000864,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR116, 0x80000868,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG116, 0x8000086C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT117,     0x80000870,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET117, 0x80000874,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR117, 0x80000878,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG117, 0x8000087C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT118,     0x80000880,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET118, 0x80000884,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR118, 0x80000888,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG118, 0x8000088C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT119,     0x80000890,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET119, 0x80000894,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR119, 0x80000898,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG119, 0x8000089C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT120,     0x800008A0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET120, 0x800008A4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR120, 0x800008A8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG120, 0x800008AC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT121,     0x800008B0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET121, 0x800008B4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR121, 0x800008B8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG121, 0x800008BC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT122,     0x800008C0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET122, 0x800008C4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR122, 0x800008C8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG122, 0x800008CC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT123,     0x800008D0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET123, 0x800008D4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR123, 0x800008D8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG123, 0x800008DC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT124,     0x800008E0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET124, 0x800008E4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR124, 0x800008E8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG124, 0x800008EC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT125,     0x800008F0,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET125, 0x800008F4,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR125, 0x800008F8,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG125, 0x800008FC,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT126,     0x80000900,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET126, 0x80000904,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR126, 0x80000908,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG126, 0x8000090C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT127,     0x80000910,__READ_WRITE ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_SET127, 0x80000914,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_CLR127, 0x80000918,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_INTERRUPT_TOG127, 0x8000091C,__WRITE      ,__hw_icoll_interrupt_bits);
__IO_REG32_BIT(HW_ICOLL_DEBUG,            0x80001120,__READ       ,__hw_icoll_debug_bits);
__IO_REG32(    HW_ICOLL_DBGREAD0,         0x80001130,__READ       );
__IO_REG32(    HW_ICOLL_DBGREAD1,         0x80001140,__READ       );
__IO_REG32_BIT(HW_ICOLL_DBGFLAG,          0x80001150,__READ       ,__hw_icoll_dbgflag_bits);
__IO_REG32_BIT(HW_ICOLL_DBGREQUEST0,      0x80001160,__READ       ,__hw_icoll_dbgrequest0_bits);
__IO_REG32_BIT(HW_ICOLL_DBGREQUEST1,      0x80001170,__READ       ,__hw_icoll_dbgrequest1_bits);
__IO_REG32_BIT(HW_ICOLL_DBGREQUEST2,      0x80001180,__READ       ,__hw_icoll_dbgrequest2_bits);
__IO_REG32_BIT(HW_ICOLL_DBGREQUEST3,      0x80001190,__READ       ,__hw_icoll_dbgrequest3_bits);
__IO_REG32_BIT(HW_ICOLL_VERSION,          0x800011E0,__READ       ,__hw_icoll_version_bits);

/***************************************************************************
 **
 **  APBH DMA
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_APBH_CTRL0,             0x80004000,__READ_WRITE ,__hw_apbh_ctrl0_bits);
__IO_REG32_BIT(HW_APBH_CTRL0_SET,         0x80004004,__WRITE      ,__hw_apbh_ctrl0_bits);
__IO_REG32_BIT(HW_APBH_CTRL0_CLR,         0x80004008,__WRITE      ,__hw_apbh_ctrl0_bits);
__IO_REG32_BIT(HW_APBH_CTRL0_TOG,         0x8000400C,__WRITE      ,__hw_apbh_ctrl0_bits);
__IO_REG32_BIT(HW_APBH_CTRL1,             0x80004010,__READ_WRITE ,__hw_apbh_ctrl1_bits);
__IO_REG32_BIT(HW_APBH_CTRL1_SET,         0x80004014,__WRITE      ,__hw_apbh_ctrl1_bits);
__IO_REG32_BIT(HW_APBH_CTRL1_CLR,         0x80004018,__WRITE      ,__hw_apbh_ctrl1_bits);
__IO_REG32_BIT(HW_APBH_CTRL1_TOG,         0x8000401C,__WRITE      ,__hw_apbh_ctrl1_bits);
__IO_REG32_BIT(HW_APBH_CTRL2,             0x80004020,__READ_WRITE ,__hw_apbh_ctrl2_bits);
__IO_REG32_BIT(HW_APBH_CTRL2_SET,         0x80004024,__WRITE      ,__hw_apbh_ctrl2_bits);
__IO_REG32_BIT(HW_APBH_CTRL2_CLR,         0x80004028,__WRITE      ,__hw_apbh_ctrl2_bits);
__IO_REG32_BIT(HW_APBH_CTRL2_TOG,         0x8000402C,__WRITE      ,__hw_apbh_ctrl2_bits);
__IO_REG32_BIT(HW_APBH_CHANNEL_CTRL,      0x80004030,__READ_WRITE ,__hw_apbh_channel_ctrl_bits);
__IO_REG32_BIT(HW_APBH_CHANNEL_CTRL_SET,  0x80004034,__WRITE      ,__hw_apbh_channel_ctrl_bits);
__IO_REG32_BIT(HW_APBH_CHANNEL_CTRL_CLR,  0x80004038,__WRITE      ,__hw_apbh_channel_ctrl_bits);
__IO_REG32_BIT(HW_APBH_CHANNEL_CTRL_TOG,  0x8000403C,__WRITE      ,__hw_apbh_channel_ctrl_bits);
__IO_REG32_BIT(HW_APBH_DEVSEL,            0x80004040,__READ       ,__hw_apbh_devsel_bits);
__IO_REG32_BIT(HW_APBH_DMA_BURST_SIZE,    0x80004050,__READ_WRITE ,__hw_apbh_dma_burst_size_bits);
__IO_REG32_BIT(HW_APBH_DEBUG,             0x80004100,__READ       ,__hw_apbh_debug_bits);
__IO_REG32(    HW_APBH_CH0_CURCMDAR,      0x80004110,__READ_WRITE );
__IO_REG32_BIT(HW_APBH_CH0_CMD,           0x80004120,__READ       ,__hw_apbh_ch_cmd_bits);
__IO_REG32(    HW_APBH_CH0_BAR,           0x80004130,__READ       );
__IO_REG32_BIT(HW_APBH_CH0_SEMA,          0x80004140,__READ_WRITE ,__hw_apbh_ch_sema_bits);
__IO_REG32_BIT(HW_APBH_CH0_DEBUG1,        0x80004150,__READ       ,__hw_apbh_ch_debug1_bits);
__IO_REG32(    HW_APBH_CH0_DEBUG2,        0x80004160,__READ       );
__IO_REG32(    HW_APBH_CH1_CURCMDAR,      0x80004170,__READ_WRITE );
__IO_REG32(    HW_APBH_CH1_NXTCMDAR,      0x80004180,__READ_WRITE );
__IO_REG32_BIT(HW_APBH_CH1_CMD,           0x80004190,__READ       ,__hw_apbh_ch_cmd_bits);
__IO_REG32(    HW_APBH_CH1_BAR,           0x800041A0,__READ       );
__IO_REG32_BIT(HW_APBH_CH1_SEMA,          0x800041B0,__READ_WRITE ,__hw_apbh_ch_sema_bits);
__IO_REG32_BIT(HW_APBH_CH1_DEBUG1,        0x800041C0,__READ       ,__hw_apbh_ch_debug1_bits);
__IO_REG32(    HW_APBH_CH1_DEBUG2,        0x800041D0,__READ       );
__IO_REG32(    HW_APBH_CH2_CURCMDAR,      0x800041E0,__READ       );
__IO_REG32(    HW_APBH_CH2_NXTCMDAR,      0x800041F0,__READ_WRITE );
__IO_REG32_BIT(HW_APBH_CH2_CMD,           0x80004200,__READ       ,__hw_apbh_ch_cmd_bits);
__IO_REG32(    HW_APBH_CH2_BAR,           0x80004210,__READ       );
__IO_REG32_BIT(HW_APBH_CH2_SEMA,          0x80004220,__READ_WRITE ,__hw_apbh_ch_sema_bits);
__IO_REG32_BIT(HW_APBH_CH2_DEBUG1,        0x80004230,__READ       ,__hw_apbh_ch_debug1_bits);
__IO_REG32(    HW_APBH_CH2_DEBUG2,        0x80004240,__READ       );
__IO_REG32(    HW_APBH_CH3_CURCMDAR,      0x80004250,__READ       );
__IO_REG32(    HW_APBH_CH3_NXTCMDAR,      0x80004260,__READ_WRITE );
__IO_REG32_BIT(HW_APBH_CH3_CMD,           0x80004270,__READ       ,__hw_apbh_ch_cmd_bits);
__IO_REG32(    HW_APBH_CH3_BAR,           0x80004280,__READ       );
__IO_REG32_BIT(HW_APBH_CH3_SEMA,          0x80004290,__READ_WRITE ,__hw_apbh_ch_sema_bits);
__IO_REG32_BIT(HW_APBH_CH3_DEBUG1,        0x800042A0,__READ       ,__hw_apbh_ch_debug1_bits);
__IO_REG32(    HW_APBH_CH3_DEBUG2,        0x800042B0,__READ       );
__IO_REG32(    HW_APBH_CH4_CURCMDAR,      0x800042C0,__READ       );
__IO_REG32(    HW_APBH_CH4_NXTCMDAR,      0x800042D0,__READ_WRITE );
__IO_REG32_BIT(HW_APBH_CH4_CMD,           0x800042E0,__READ       ,__hw_apbh_ch_cmd_bits);
__IO_REG32(    HW_APBH_CH4_BAR,           0x800042F0,__READ       );
__IO_REG32_BIT(HW_APBH_CH4_SEMA,          0x80004300,__READ_WRITE ,__hw_apbh_ch_sema_bits);
__IO_REG32_BIT(HW_APBH_CH4_DEBUG1,        0x80004310,__READ       ,__hw_apbh_ch_debug1_bits);
__IO_REG32(    HW_APBH_CH4_DEBUG2,        0x80004320,__READ       );
__IO_REG32(    HW_APBH_CH5_CURCMDAR,      0x80004330,__READ       );
__IO_REG32(    HW_APBH_CH5_NXTCMDAR,      0x80004340,__READ_WRITE );
__IO_REG32_BIT(HW_APBH_CH5_CMD,           0x80004350,__READ       ,__hw_apbh_ch_cmd_bits);
__IO_REG32(    HW_APBH_CH5_BAR,           0x80004360,__READ       );
__IO_REG32_BIT(HW_APBH_CH5_SEMA,          0x80004370,__READ_WRITE ,__hw_apbh_ch_sema_bits);
__IO_REG32_BIT(HW_APBH_CH5_DEBUG1,        0x80004380,__READ       ,__hw_apbh_ch_debug1_bits);
__IO_REG32(    HW_APBH_CH5_DEBUG2,        0x80004390,__READ       );
__IO_REG32(    HW_APBH_CH6_CURCMDAR,      0x800043A0,__READ       );
__IO_REG32(    HW_APBH_CH6_NXTCMDAR,      0x800043B0,__READ_WRITE );
__IO_REG32_BIT(HW_APBH_CH6_CMD,           0x800043C0,__READ       ,__hw_apbh_ch_cmd_bits);
__IO_REG32(    HW_APBH_CH6_BAR,           0x800043D0,__READ       );
__IO_REG32_BIT(HW_APBH_CH6_SEMA,          0x800043E0,__READ_WRITE ,__hw_apbh_ch_sema_bits);
__IO_REG32_BIT(HW_APBH_CH6_DEBUG1,        0x800043F0,__READ       ,__hw_apbh_ch_debug1_bits);
__IO_REG32(    HW_APBH_CH6_DEBUG2,        0x80004400,__READ       );
__IO_REG32(    HW_APBH_CH7_CURCMDAR,      0x80004410,__READ       );
__IO_REG32(    HW_APBH_CH7_NXTCMDAR,      0x80004420,__READ_WRITE );
__IO_REG32_BIT(HW_APBH_CH7_CMD,           0x80004430,__READ       ,__hw_apbh_ch_cmd_bits);
__IO_REG32(    HW_APBH_CH7_BAR,           0x80004440,__READ       );
__IO_REG32_BIT(HW_APBH_CH7_SEMA,          0x80004450,__READ_WRITE ,__hw_apbh_ch_sema_bits);
__IO_REG32_BIT(HW_APBH_CH7_DEBUG1,        0x80004460,__READ       ,__hw_apbh_ch_debug1_bits);
__IO_REG32(    HW_APBH_CH7_DEBUG2,        0x80004470,__READ       );
__IO_REG32(    HW_APBH_CH8_CURCMDAR,      0x80004480,__READ       );
__IO_REG32(    HW_APBH_CH8_NXTCMDAR,      0x80004490,__READ_WRITE );
__IO_REG32_BIT(HW_APBH_CH8_CMD,           0x800044A0,__READ       ,__hw_apbh_ch_cmd_bits);
__IO_REG32(    HW_APBH_CH8_BAR,           0x800044B0,__READ       );
__IO_REG32_BIT(HW_APBH_CH8_SEMA,          0x800044C0,__READ_WRITE ,__hw_apbh_ch_sema_bits);
__IO_REG32_BIT(HW_APBH_CH8_DEBUG1,        0x800044D0,__READ       ,__hw_apbh_ch_debug1_bits);
__IO_REG32(    HW_APBH_CH8_DEBUG2,        0x800044E0,__READ       );
__IO_REG32(    HW_APBH_CH9_CURCMDAR,      0x800044F0,__READ       );
__IO_REG32(    HW_APBH_CH9_NXTCMDAR,      0x80004500,__READ_WRITE );
__IO_REG32_BIT(HW_APBH_CH9_CMD,           0x80004510,__READ       ,__hw_apbh_ch_cmd_bits);
__IO_REG32(    HW_APBH_CH9_BAR,           0x80004520,__READ       );
__IO_REG32_BIT(HW_APBH_CH9_SEMA,          0x80004530,__READ_WRITE ,__hw_apbh_ch_sema_bits);
__IO_REG32_BIT(HW_APBH_CH9_DEBUG1,        0x80004540,__READ       ,__hw_apbh_ch_debug1_bits);
__IO_REG32(    HW_APBH_CH9_DEBUG2,        0x80004550,__READ       );
__IO_REG32(    HW_APBH_CH10_CURCMDAR,     0x80004560,__READ       );
__IO_REG32(    HW_APBH_CH10_NXTCMDAR,     0x80004570,__READ_WRITE );
__IO_REG32_BIT(HW_APBH_CH10_CMD,          0x80004580,__READ       ,__hw_apbh_ch_cmd_bits);
__IO_REG32(    HW_APBH_CH10_BAR,          0x80004590,__READ       );
__IO_REG32_BIT(HW_APBH_CH10_SEMA,         0x800045A0,__READ_WRITE ,__hw_apbh_ch_sema_bits);
__IO_REG32_BIT(HW_APBH_CH10_DEBUG1,       0x800045B0,__READ       ,__hw_apbh_ch_debug1_bits);
__IO_REG32(    HW_APBH_CH10_DEBUG2,       0x800045C0,__READ       );
__IO_REG32(    HW_APBH_CH11_CURCMDAR,     0x800045D0,__READ       );
__IO_REG32(    HW_APBH_CH11_NXTCMDAR,     0x800045E0,__READ_WRITE );
__IO_REG32_BIT(HW_APBH_CH11_CMD,          0x800045F0,__READ       ,__hw_apbh_ch_cmd_bits);
__IO_REG32(    HW_APBH_CH11_BAR,          0x80004600,__READ       );
__IO_REG32_BIT(HW_APBH_CH11_SEMA,         0x80004610,__READ_WRITE ,__hw_apbh_ch_sema_bits);
__IO_REG32_BIT(HW_APBH_CH11_DEBUG1,       0x80004620,__READ       ,__hw_apbh_ch_debug1_bits);
__IO_REG32(    HW_APBH_CH11_DEBUG2,       0x80004630,__READ       );
__IO_REG32(    HW_APBH_CH12_CURCMDAR,     0x80004640,__READ       );
__IO_REG32(    HW_APBH_CH12_NXTCMDAR,     0x80004650,__READ_WRITE );
__IO_REG32_BIT(HW_APBH_CH12_CMD,          0x80004660,__READ       ,__hw_apbh_ch_cmd_bits);
__IO_REG32(    HW_APBH_CH12_BAR,          0x80004670,__READ       );
__IO_REG32_BIT(HW_APBH_CH12_SEMA,         0x80004680,__READ_WRITE ,__hw_apbh_ch_sema_bits);
__IO_REG32_BIT(HW_APBH_CH12_DEBUG1,       0x80004690,__READ       ,__hw_apbh_ch_debug1_bits);
__IO_REG32(    HW_APBH_CH12_DEBUG2,       0x800046A0,__READ       );
__IO_REG32(    HW_APBH_CH13_CURCMDAR,     0x800046B0,__READ       );
__IO_REG32(    HW_APBH_CH13_NXTCMDAR,     0x800046C0,__READ_WRITE );
__IO_REG32_BIT(HW_APBH_CH13_CMD,          0x800046D0,__READ       ,__hw_apbh_ch_cmd_bits);
__IO_REG32(    HW_APBH_CH13_BAR,          0x800046E0,__READ       );
__IO_REG32_BIT(HW_APBH_CH13_SEMA,         0x800046F0,__READ_WRITE ,__hw_apbh_ch_sema_bits);
__IO_REG32_BIT(HW_APBH_CH13_DEBUG1,       0x80004700,__READ       ,__hw_apbh_ch_debug1_bits);
__IO_REG32(    HW_APBH_CH13_DEBUG2,       0x80004710,__READ       );
__IO_REG32(    HW_APBH_CH14_CURCMDAR,     0x80004720,__READ       );
__IO_REG32(    HW_APBH_CH14_NXTCMDAR,     0x80004730,__READ_WRITE );
__IO_REG32_BIT(HW_APBH_CH14_CMD,          0x80004740,__READ       ,__hw_apbh_ch_cmd_bits);
__IO_REG32(    HW_APBH_CH14_BAR,          0x80004750,__READ       );
__IO_REG32_BIT(HW_APBH_CH14_SEMA,         0x80004760,__READ_WRITE ,__hw_apbh_ch_sema_bits);
__IO_REG32_BIT(HW_APBH_CH14_DEBUG1,       0x80004770,__READ       ,__hw_apbh_ch_debug1_bits);
__IO_REG32(    HW_APBH_CH14_DEBUG2,       0x80004780,__READ       );
__IO_REG32(    HW_APBH_CH15_CURCMDAR,     0x80004790,__READ       );
__IO_REG32(    HW_APBH_CH15_NXTCMDAR,     0x800047A0,__READ_WRITE );
__IO_REG32_BIT(HW_APBH_CH15_CMD,          0x800047B0,__READ       ,__hw_apbh_ch_cmd_bits);
__IO_REG32(    HW_APBH_CH15_BAR,          0x800047C0,__READ       );
__IO_REG32_BIT(HW_APBH_CH15_SEMA,         0x800047D0,__READ_WRITE ,__hw_apbh_ch_sema_bits);
__IO_REG32_BIT(HW_APBH_CH15_DEBUG1,       0x800047E0,__READ       ,__hw_apbh_ch_debug1_bits);
__IO_REG32(    HW_APBH_CH15_DEBUG2,       0x800047F0,__READ       );
__IO_REG32_BIT(HW_APBH_VERSION,           0x80004800,__READ       ,__hw_apbh_version_bits);

/***************************************************************************
 **
 **  APBX DMA
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_APBX_CTRL0,             0x80024000,__READ_WRITE ,__hw_apbx_ctrl0_bits);
__IO_REG32_BIT(HW_APBX_CTRL0_SET,         0x80024004,__WRITE      ,__hw_apbx_ctrl0_bits);
__IO_REG32_BIT(HW_APBX_CTRL0_CLR,         0x80024008,__WRITE      ,__hw_apbx_ctrl0_bits);
__IO_REG32_BIT(HW_APBX_CTRL0_TOG,         0x8002400C,__WRITE      ,__hw_apbx_ctrl0_bits);
__IO_REG32_BIT(HW_APBX_CTRL1,             0x80024010,__READ_WRITE ,__hw_apbx_ctrl1_bits);
__IO_REG32_BIT(HW_APBX_CTRL1_SET,         0x80024014,__WRITE      ,__hw_apbx_ctrl1_bits);
__IO_REG32_BIT(HW_APBX_CTRL1_CLR,         0x80024018,__WRITE      ,__hw_apbx_ctrl1_bits);
__IO_REG32_BIT(HW_APBX_CTRL1_TOG,         0x8002401C,__WRITE      ,__hw_apbx_ctrl1_bits);
__IO_REG32_BIT(HW_APBX_CTRL2,             0x80024020,__READ_WRITE ,__hw_apbx_ctrl2_bits);
__IO_REG32_BIT(HW_APBX_CTRL2_SET,         0x80024024,__WRITE      ,__hw_apbx_ctrl2_bits);
__IO_REG32_BIT(HW_APBX_CTRL2_CLR,         0x80024028,__WRITE      ,__hw_apbx_ctrl2_bits);
__IO_REG32_BIT(HW_APBX_CTRL2_TOG,         0x8002402C,__WRITE      ,__hw_apbx_ctrl2_bits);
__IO_REG32_BIT(HW_APBX_CHANNEL_CTRL,      0x80024030,__READ_WRITE ,__hw_apbx_channel_ctrl_bits);
__IO_REG32_BIT(HW_APBX_CHANNEL_CTRL_SET,  0x80024034,__WRITE      ,__hw_apbx_channel_ctrl_bits);
__IO_REG32_BIT(HW_APBX_CHANNEL_CTRL_CLR,  0x80024038,__WRITE      ,__hw_apbx_channel_ctrl_bits);
__IO_REG32_BIT(HW_APBX_CHANNEL_CTRL_TOG,  0x8002403C,__WRITE      ,__hw_apbx_channel_ctrl_bits);
__IO_REG32_BIT(HW_APBX_DEVSEL,            0x80024040,__READ       ,__hw_apbx_devsel_bits);
__IO_REG32(    HW_APBX_CH0_CURCMDAR,      0x80024100,__READ       );
__IO_REG32(    HW_APBX_CH0_NXTCMDAR,      0x80024110,__READ_WRITE );
__IO_REG32_BIT(HW_APBX_CH0_CMD,           0x80024120,__READ       ,__hw_apbx_ch_cmd_bits);
__IO_REG32(    HW_APBX_CH0_BAR,           0x80024130,__READ       );
__IO_REG32_BIT(HW_APBX_CH0_SEMA,          0x80024140,__READ_WRITE ,__hw_apbx_ch_sema_bits);
__IO_REG32_BIT(HW_APBX_CH0_DEBUG1,        0x80024150,__READ       ,__hw_apbx_ch_debug1_bits);
__IO_REG32(    HW_APBX_CH0_DEBUG2,        0x80024160,__READ       );
__IO_REG32(    HW_APBX_CH1_CURCMDAR,      0x80024170,__READ       );
__IO_REG32(    HW_APBX_CH1_NXTCMDAR,      0x80024180,__READ_WRITE );
__IO_REG32_BIT(HW_APBX_CH1_CMD,           0x80024190,__READ       ,__hw_apbx_ch_cmd_bits);
__IO_REG32(    HW_APBX_CH1_BAR,           0x800241A0,__READ       );
__IO_REG32_BIT(HW_APBX_CH1_SEMA,          0x800241B0,__READ_WRITE ,__hw_apbx_ch_sema_bits);
__IO_REG32_BIT(HW_APBX_CH1_DEBUG1,        0x800241C0,__READ       ,__hw_apbx_ch_debug1_bits);
__IO_REG32(    HW_APBX_CH1_DEBUG2,        0x800241D0,__READ       );
__IO_REG32(    HW_APBX_CH2_CURCMDAR,      0x800241E0,__READ       );
__IO_REG32(    HW_APBX_CH2_NXTCMDAR,      0x800241F0,__READ_WRITE );
__IO_REG32_BIT(HW_APBX_CH2_CMD,           0x80024200,__READ       ,__hw_apbx_ch_cmd_bits);
__IO_REG32(    HW_APBX_CH2_BAR,           0x80024210,__READ       );
__IO_REG32_BIT(HW_APBX_CH2_SEMA,          0x80024220,__READ_WRITE ,__hw_apbx_ch_sema_bits);
__IO_REG32_BIT(HW_APBX_CH2_DEBUG1,        0x80024230,__READ       ,__hw_apbx_ch_debug1_bits);
__IO_REG32(    HW_APBX_CH2_DEBUG2,        0x80024240,__READ       );
__IO_REG32(    HW_APBX_CH3_CURCMDAR,      0x80024250,__READ       );
__IO_REG32(    HW_APBX_CH3_NXTCMDAR,      0x80024260,__READ_WRITE );
__IO_REG32_BIT(HW_APBX_CH3_CMD,           0x80024270,__READ       ,__hw_apbx_ch_cmd_bits);
__IO_REG32(    HW_APBX_CH3_BAR,           0x80024280,__READ       );
__IO_REG32_BIT(HW_APBX_CH3_SEMA,          0x80024290,__READ_WRITE ,__hw_apbx_ch_sema_bits);
__IO_REG32_BIT(HW_APBX_CH3_DEBUG1,        0x800242A0,__READ       ,__hw_apbx_ch_debug1_bits);
__IO_REG32(    HW_APBX_CH3_DEBUG2,        0x800242B0,__READ       );
__IO_REG32(    HW_APBX_CH4_CURCMDAR,      0x800242C0,__READ       );
__IO_REG32(    HW_APBX_CH4_NXTCMDAR,      0x800242D0,__READ_WRITE );
__IO_REG32_BIT(HW_APBX_CH4_CMD,           0x800242E0,__READ       ,__hw_apbx_ch_cmd_bits);
__IO_REG32(    HW_APBX_CH4_BAR,           0x800242F0,__READ       );
__IO_REG32_BIT(HW_APBX_CH4_SEMA,          0x80024300,__READ_WRITE ,__hw_apbx_ch_sema_bits);
__IO_REG32_BIT(HW_APBX_CH4_DEBUG1,        0x80024310,__READ       ,__hw_apbx_ch_debug1_bits);
__IO_REG32(    HW_APBX_CH4_DEBUG2,        0x80024320,__READ       );
__IO_REG32(    HW_APBX_CH5_CURCMDAR,      0x80024330,__READ       );
__IO_REG32(    HW_APBX_CH5_NXTCMDAR,      0x80024340,__READ_WRITE );
__IO_REG32_BIT(HW_APBX_CH5_CMD,           0x80024350,__READ       ,__hw_apbx_ch_cmd_bits);
__IO_REG32(    HW_APBX_CH5_BAR,           0x80024360,__READ       );
__IO_REG32_BIT(HW_APBX_CH5_SEMA,          0x80024370,__READ_WRITE ,__hw_apbx_ch_sema_bits);
__IO_REG32_BIT(HW_APBX_CH5_DEBUG1,        0x80024380,__READ       ,__hw_apbx_ch_debug1_bits);
__IO_REG32(    HW_APBX_CH5_DEBUG2,        0x80024390,__READ       );
__IO_REG32(    HW_APBX_CH6_CURCMDAR,      0x800243A0,__READ       );
__IO_REG32(    HW_APBX_CH6_NXTCMDAR,      0x800243B0,__READ_WRITE );
__IO_REG32_BIT(HW_APBX_CH6_CMD,           0x800243C0,__READ       ,__hw_apbx_ch_cmd_bits);
__IO_REG32(    HW_APBX_CH6_BAR,           0x800243D0,__READ       );
__IO_REG32_BIT(HW_APBX_CH6_SEMA,          0x800243E0,__READ_WRITE ,__hw_apbx_ch_sema_bits);
__IO_REG32_BIT(HW_APBX_CH6_DEBUG1,        0x800243F0,__READ       ,__hw_apbx_ch_debug1_bits);
__IO_REG32(    HW_APBX_CH6_DEBUG2,        0x80024400,__READ       );
__IO_REG32(    HW_APBX_CH7_CURCMDAR,      0x80024410,__READ       );
__IO_REG32(    HW_APBX_CH7_NXTCMDAR,      0x80024420,__READ_WRITE );
__IO_REG32_BIT(HW_APBX_CH7_CMD,           0x80024430,__READ       ,__hw_apbx_ch_cmd_bits);
__IO_REG32(    HW_APBX_CH7_BAR,           0x80024440,__READ       );
__IO_REG32_BIT(HW_APBX_CH7_SEMA,          0x80024450,__READ_WRITE ,__hw_apbx_ch_sema_bits);
__IO_REG32_BIT(HW_APBX_CH7_DEBUG1,        0x80024460,__READ       ,__hw_apbx_ch_debug1_bits);
__IO_REG32(    HW_APBX_CH7_DEBUG2,        0x80024470,__READ       );
__IO_REG32(    HW_APBX_CH8_CURCMDAR,      0x80024480,__READ       );
__IO_REG32(    HW_APBX_CH8_NXTCMDAR,      0x80024490,__READ_WRITE );
__IO_REG32_BIT(HW_APBX_CH8_CMD,           0x800244A0,__READ       ,__hw_apbx_ch_cmd_bits);
__IO_REG32(    HW_APBX_CH8_BAR,           0x800244B0,__READ       );
__IO_REG32_BIT(HW_APBX_CH8_SEMA,          0x800244C0,__READ_WRITE ,__hw_apbx_ch_sema_bits);
__IO_REG32_BIT(HW_APBX_CH8_DEBUG1,        0x800244D0,__READ       ,__hw_apbx_ch_debug1_bits);
__IO_REG32(    HW_APBX_CH8_DEBUG2,        0x800244E0,__READ       );
__IO_REG32(    HW_APBX_CH9_CURCMDAR,      0x800244F0,__READ       );
__IO_REG32(    HW_APBX_CH9_NXTCMDAR,      0x80024500,__READ_WRITE );
__IO_REG32_BIT(HW_APBX_CH9_CMD,           0x80024510,__READ       ,__hw_apbx_ch_cmd_bits);
__IO_REG32(    HW_APBX_CH9_BAR,           0x80024520,__READ       );
__IO_REG32_BIT(HW_APBX_CH9_SEMA,          0x80024530,__READ_WRITE ,__hw_apbx_ch_sema_bits);
__IO_REG32_BIT(HW_APBX_CH9_DEBUG1,        0x80024540,__READ       ,__hw_apbx_ch_debug1_bits);
__IO_REG32(    HW_APBX_CH9_DEBUG2,        0x80024550,__READ       );
__IO_REG32(    HW_APBX_CH10_CURCMDAR,     0x80024560,__READ       );
__IO_REG32(    HW_APBX_CH10_NXTCMDAR,     0x80024570,__READ_WRITE );
__IO_REG32_BIT(HW_APBX_CH10_CMD,          0x80024580,__READ       ,__hw_apbx_ch_cmd_bits);
__IO_REG32(    HW_APBX_CH10_BAR,          0x80024590,__READ       );
__IO_REG32_BIT(HW_APBX_CH10_SEMA,         0x800245A0,__READ_WRITE ,__hw_apbx_ch_sema_bits);
__IO_REG32_BIT(HW_APBX_CH10_DEBUG1,       0x800245B0,__READ       ,__hw_apbx_ch_debug1_bits);
__IO_REG32(    HW_APBX_CH10_DEBUG2,       0x800245C0,__READ       );
__IO_REG32(    HW_APBX_CH11_CURCMDAR,     0x800245D0,__READ       );
__IO_REG32(    HW_APBX_CH11_NXTCMDAR,     0x800245E0,__READ_WRITE );
__IO_REG32_BIT(HW_APBX_CH11_CMD,          0x800245F0,__READ       ,__hw_apbx_ch_cmd_bits);
__IO_REG32(    HW_APBX_CH11_BAR,          0x80024600,__READ       );
__IO_REG32_BIT(HW_APBX_CH11_SEMA,         0x80024610,__READ_WRITE ,__hw_apbx_ch_sema_bits);
__IO_REG32_BIT(HW_APBX_CH11_DEBUG1,       0x80024620,__READ       ,__hw_apbx_ch_debug1_bits);
__IO_REG32(    HW_APBX_CH11_DEBUG2,       0x80024630,__READ       );
__IO_REG32(    HW_APBX_CH12_CURCMDAR,     0x80024640,__READ       );
__IO_REG32(    HW_APBX_CH12_NXTCMDAR,     0x80024650,__READ_WRITE );
__IO_REG32_BIT(HW_APBX_CH12_CMD,          0x80024660,__READ       ,__hw_apbx_ch_cmd_bits);
__IO_REG32(    HW_APBX_CH12_BAR,          0x80024670,__READ       );
__IO_REG32_BIT(HW_APBX_CH12_SEMA,         0x80024680,__READ_WRITE ,__hw_apbx_ch_sema_bits);
__IO_REG32_BIT(HW_APBX_CH12_DEBUG1,       0x80024690,__READ       ,__hw_apbx_ch_debug1_bits);
__IO_REG32(    HW_APBX_CH12_DEBUG2,       0x800246A0,__READ       );
__IO_REG32(    HW_APBX_CH13_CURCMDAR,     0x800246B0,__READ       );
__IO_REG32(    HW_APBX_CH13_NXTCMDAR,     0x800246C0,__READ_WRITE );
__IO_REG32_BIT(HW_APBX_CH13_CMD,          0x800246D0,__READ       ,__hw_apbx_ch_cmd_bits);
__IO_REG32(    HW_APBX_CH13_BAR,          0x800246E0,__READ       );
__IO_REG32_BIT(HW_APBX_CH13_SEMA,         0x800246F0,__READ_WRITE ,__hw_apbx_ch_sema_bits);
__IO_REG32_BIT(HW_APBX_CH13_DEBUG1,       0x80024700,__READ       ,__hw_apbx_ch_debug1_bits);
__IO_REG32(    HW_APBX_CH13_DEBUG2,       0x80024710,__READ       );
__IO_REG32(    HW_APBX_CH14_CURCMDAR,     0x80024720,__READ       );
__IO_REG32(    HW_APBX_CH14_NXTCMDAR,     0x80024730,__READ_WRITE );
__IO_REG32_BIT(HW_APBX_CH14_CMD,          0x80024740,__READ       ,__hw_apbx_ch_cmd_bits);
__IO_REG32(    HW_APBX_CH14_BAR,          0x80024750,__READ       );
__IO_REG32_BIT(HW_APBX_CH14_SEMA,         0x80024760,__READ_WRITE ,__hw_apbx_ch_sema_bits);
__IO_REG32_BIT(HW_APBX_CH14_DEBUG1,       0x80024770,__READ       ,__hw_apbx_ch_debug1_bits);
__IO_REG32(    HW_APBX_CH14_DEBUG2,       0x80024780,__READ       );
__IO_REG32(    HW_APBX_CH15_CURCMDAR,     0x80024790,__READ       );
__IO_REG32(    HW_APBX_CH15_NXTCMDAR,     0x800247A0,__READ_WRITE );
__IO_REG32_BIT(HW_APBX_CH15_CMD,          0x800247B0,__READ       ,__hw_apbx_ch_cmd_bits);
__IO_REG32(    HW_APBX_CH15_BAR,          0x800247C0,__READ       );
__IO_REG32_BIT(HW_APBX_CH15_SEMA,         0x800247D0,__READ_WRITE ,__hw_apbx_ch_sema_bits);
__IO_REG32_BIT(HW_APBX_CH15_DEBUG1,       0x800247E0,__READ       ,__hw_apbx_ch_debug1_bits);
__IO_REG32(    HW_APBX_CH15_DEBUG2,       0x800247F0,__READ       );
__IO_REG32_BIT(HW_APBX_VERSION,           0x80024800,__READ       ,__hw_apbx_version_bits);

/***************************************************************************
 **
 **  PINCTRL
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_PINCTRL_CTRL,           0x80018000,__READ_WRITE ,__hw_pinctrl_ctrl_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL0,        0x80018100,__READ_WRITE ,__hw_pinctrl_muxsel0_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL0_SET,    0x80018104,__WRITE      ,__hw_pinctrl_muxsel0_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL0_CLR,    0x80018108,__WRITE      ,__hw_pinctrl_muxsel0_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL0_TOG,    0x8001810C,__WRITE      ,__hw_pinctrl_muxsel0_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL1,        0x80018110,__READ_WRITE ,__hw_pinctrl_muxsel1_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL1_SET,    0x80018114,__WRITE      ,__hw_pinctrl_muxsel1_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL1_CLR,    0x80018118,__WRITE      ,__hw_pinctrl_muxsel1_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL1_TOG,    0x8001811C,__WRITE      ,__hw_pinctrl_muxsel1_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL2,        0x80018120,__READ_WRITE ,__hw_pinctrl_muxsel2_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL2_SET,    0x80018124,__WRITE      ,__hw_pinctrl_muxsel2_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL2_CLR,    0x80018128,__WRITE      ,__hw_pinctrl_muxsel2_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL2_TOG,    0x8001812C,__WRITE      ,__hw_pinctrl_muxsel2_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL3,        0x80018130,__READ_WRITE ,__hw_pinctrl_muxsel3_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL3_SET,    0x80018134,__WRITE      ,__hw_pinctrl_muxsel3_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL3_CLR,    0x80018138,__WRITE      ,__hw_pinctrl_muxsel3_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL3_TOG,    0x8001813C,__WRITE      ,__hw_pinctrl_muxsel3_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL4,        0x80018140,__READ_WRITE ,__hw_pinctrl_muxsel4_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL4_SET,    0x80018144,__WRITE      ,__hw_pinctrl_muxsel4_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL4_CLR,    0x80018148,__WRITE      ,__hw_pinctrl_muxsel4_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL4_TOG,    0x8001814C,__WRITE      ,__hw_pinctrl_muxsel4_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL5,        0x80018150,__READ_WRITE ,__hw_pinctrl_muxsel5_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL5_SET,    0x80018154,__WRITE      ,__hw_pinctrl_muxsel5_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL5_CLR,    0x80018158,__WRITE      ,__hw_pinctrl_muxsel5_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL5_TOG,    0x8001815C,__WRITE      ,__hw_pinctrl_muxsel5_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL6,        0x80018160,__READ_WRITE ,__hw_pinctrl_muxsel6_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL6_SET,    0x80018164,__WRITE      ,__hw_pinctrl_muxsel6_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL6_CLR,    0x80018168,__WRITE      ,__hw_pinctrl_muxsel6_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL6_TOG,    0x8001816C,__WRITE      ,__hw_pinctrl_muxsel6_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL7,        0x80018170,__READ_WRITE ,__hw_pinctrl_muxsel7_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL7_SET,    0x80018174,__WRITE      ,__hw_pinctrl_muxsel7_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL7_CLR,    0x80018178,__WRITE      ,__hw_pinctrl_muxsel7_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL7_TOG,    0x8001817C,__WRITE      ,__hw_pinctrl_muxsel7_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL8,        0x80018180,__READ_WRITE ,__hw_pinctrl_muxsel8_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL8_SET,    0x80018184,__WRITE      ,__hw_pinctrl_muxsel8_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL8_CLR,    0x80018188,__WRITE      ,__hw_pinctrl_muxsel8_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL8_TOG,    0x8001818C,__WRITE      ,__hw_pinctrl_muxsel8_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL9,        0x80018190,__READ_WRITE ,__hw_pinctrl_muxsel9_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL9_SET,    0x80018194,__WRITE      ,__hw_pinctrl_muxsel9_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL9_CLR,    0x80018198,__WRITE      ,__hw_pinctrl_muxsel9_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL9_TOG,    0x8001819C,__WRITE      ,__hw_pinctrl_muxsel9_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL10,       0x800181A0,__READ_WRITE ,__hw_pinctrl_muxsel10_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL10_SET,   0x800181A4,__WRITE      ,__hw_pinctrl_muxsel10_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL10_CLR,   0x800181A8,__WRITE      ,__hw_pinctrl_muxsel10_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL10_TOG,   0x800181AC,__WRITE      ,__hw_pinctrl_muxsel10_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL11,       0x800181B0,__READ_WRITE ,__hw_pinctrl_muxsel11_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL11_SET,   0x800181B4,__WRITE      ,__hw_pinctrl_muxsel11_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL11_CLR,   0x800181B8,__WRITE      ,__hw_pinctrl_muxsel11_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL11_TOG,   0x800181BC,__WRITE      ,__hw_pinctrl_muxsel11_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL12,       0x800181C0,__READ_WRITE ,__hw_pinctrl_muxsel12_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL12_SET,   0x800181C4,__WRITE      ,__hw_pinctrl_muxsel12_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL12_CLR,   0x800181C8,__WRITE      ,__hw_pinctrl_muxsel12_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL12_TOG,   0x800181CC,__WRITE      ,__hw_pinctrl_muxsel12_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL13,       0x800181D0,__READ_WRITE ,__hw_pinctrl_muxsel13_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL13_SET,   0x800181D4,__WRITE      ,__hw_pinctrl_muxsel13_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL13_CLR,   0x800181D8,__WRITE      ,__hw_pinctrl_muxsel13_bits);
__IO_REG32_BIT(HW_PINCTRL_MUXSEL13_TOG,   0x800181DC,__WRITE      ,__hw_pinctrl_muxsel13_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE0,         0x80018300,__READ_WRITE ,__hw_pinctrl_drive0_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE0_SET,     0x80018304,__WRITE      ,__hw_pinctrl_drive0_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE0_CLR,     0x80018308,__WRITE      ,__hw_pinctrl_drive0_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE0_TOG,     0x8001830C,__WRITE      ,__hw_pinctrl_drive0_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE2,         0x80018320,__READ_WRITE ,__hw_pinctrl_drive2_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE2_SET,     0x80018324,__WRITE      ,__hw_pinctrl_drive2_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE2_CLR,     0x80018328,__WRITE      ,__hw_pinctrl_drive2_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE2_TOG,     0x8001832C,__WRITE      ,__hw_pinctrl_drive2_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE3,         0x80018330,__READ_WRITE ,__hw_pinctrl_drive3_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE3_SET,     0x80018334,__WRITE      ,__hw_pinctrl_drive3_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE3_CLR,     0x80018338,__WRITE      ,__hw_pinctrl_drive3_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE3_TOG,     0x8001833C,__WRITE      ,__hw_pinctrl_drive3_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE4,         0x80018340,__READ_WRITE ,__hw_pinctrl_drive4_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE4_SET,     0x80018344,__WRITE      ,__hw_pinctrl_drive4_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE4_CLR,     0x80018348,__WRITE      ,__hw_pinctrl_drive4_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE4_TOG,     0x8001834C,__WRITE      ,__hw_pinctrl_drive4_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE5,         0x80018350,__READ_WRITE ,__hw_pinctrl_drive5_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE5_SET,     0x80018354,__WRITE      ,__hw_pinctrl_drive5_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE5_CLR,     0x80018358,__WRITE      ,__hw_pinctrl_drive5_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE5_TOG,     0x8001835C,__WRITE      ,__hw_pinctrl_drive5_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE6,         0x80018360,__READ_WRITE ,__hw_pinctrl_drive6_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE6_SET,     0x80018364,__WRITE      ,__hw_pinctrl_drive6_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE6_CLR,     0x80018368,__WRITE      ,__hw_pinctrl_drive6_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE6_TOG,     0x8001836C,__WRITE      ,__hw_pinctrl_drive6_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE7,         0x80018370,__READ_WRITE ,__hw_pinctrl_drive7_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE7_SET,     0x80018374,__WRITE      ,__hw_pinctrl_drive7_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE7_CLR,     0x80018378,__WRITE      ,__hw_pinctrl_drive7_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE7_TOG,     0x8001837C,__WRITE      ,__hw_pinctrl_drive7_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE8,         0x80018380,__READ_WRITE ,__hw_pinctrl_drive8_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE8_SET,     0x80018384,__WRITE      ,__hw_pinctrl_drive8_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE8_CLR,     0x80018388,__WRITE      ,__hw_pinctrl_drive8_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE8_TOG,     0x8001838C,__WRITE      ,__hw_pinctrl_drive8_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE9,         0x80018390,__READ_WRITE ,__hw_pinctrl_drive9_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE9_SET,     0x80018394,__WRITE      ,__hw_pinctrl_drive9_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE9_CLR,     0x80018398,__WRITE      ,__hw_pinctrl_drive9_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE9_TOG,     0x8001839C,__WRITE      ,__hw_pinctrl_drive9_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE10,        0x800183A0,__READ_WRITE ,__hw_pinctrl_drive10_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE10_SET,    0x800183A4,__WRITE      ,__hw_pinctrl_drive10_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE10_CLR,    0x800183A8,__WRITE      ,__hw_pinctrl_drive10_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE10_TOG,    0x800183AC,__WRITE      ,__hw_pinctrl_drive10_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE11,        0x800183B0,__READ_WRITE ,__hw_pinctrl_drive11_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE11_SET,    0x800183B4,__WRITE      ,__hw_pinctrl_drive11_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE11_CLR,    0x800183B8,__WRITE      ,__hw_pinctrl_drive11_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE11_TOG,    0x800183BC,__WRITE      ,__hw_pinctrl_drive11_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE12,        0x800183C0,__READ_WRITE ,__hw_pinctrl_drive12_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE12_SET,    0x800183C4,__WRITE      ,__hw_pinctrl_drive12_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE12_CLR,    0x800183C8,__WRITE      ,__hw_pinctrl_drive12_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE12_TOG,    0x800183CC,__WRITE      ,__hw_pinctrl_drive12_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE13,        0x800183D0,__READ_WRITE ,__hw_pinctrl_drive13_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE13_SET,    0x800183D4,__WRITE      ,__hw_pinctrl_drive13_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE13_CLR,    0x800183D8,__WRITE      ,__hw_pinctrl_drive13_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE13_TOG,    0x800183DC,__WRITE      ,__hw_pinctrl_drive13_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE14,        0x800183E0,__READ_WRITE ,__hw_pinctrl_drive14_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE14_SET,    0x800183E4,__WRITE      ,__hw_pinctrl_drive14_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE14_CLR,    0x800183E8,__WRITE      ,__hw_pinctrl_drive14_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE14_TOG,    0x800183EC,__WRITE      ,__hw_pinctrl_drive14_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE15,        0x800183F0,__READ_WRITE ,__hw_pinctrl_drive15_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE15_SET,    0x800183F4,__WRITE      ,__hw_pinctrl_drive15_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE15_CLR,    0x800183F8,__WRITE      ,__hw_pinctrl_drive15_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE15_TOG,    0x800183FC,__WRITE      ,__hw_pinctrl_drive15_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE16,        0x80018400,__READ_WRITE ,__hw_pinctrl_drive16_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE16_SET,    0x80018404,__WRITE      ,__hw_pinctrl_drive16_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE16_CLR,    0x80018408,__WRITE      ,__hw_pinctrl_drive16_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE16_TOG,    0x8001840C,__WRITE      ,__hw_pinctrl_drive16_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE17,        0x80018410,__READ_WRITE ,__hw_pinctrl_drive17_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE17_SET,    0x80018414,__WRITE      ,__hw_pinctrl_drive17_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE17_CLR,    0x80018418,__WRITE      ,__hw_pinctrl_drive17_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE17_TOG,    0x8001841C,__WRITE      ,__hw_pinctrl_drive17_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE18,        0x80018420,__READ_WRITE ,__hw_pinctrl_drive18_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE18_SET,    0x80018424,__WRITE      ,__hw_pinctrl_drive18_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE18_CLR,    0x80018428,__WRITE      ,__hw_pinctrl_drive18_bits);
__IO_REG32_BIT(HW_PINCTRL_DRIVE18_TOG,    0x8001842C,__WRITE      ,__hw_pinctrl_drive18_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL0,          0x80018600,__READ_WRITE ,__hw_pinctrl_pull0_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL0_SET,      0x80018604,__WRITE      ,__hw_pinctrl_pull0_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL0_CLR,      0x80018608,__WRITE      ,__hw_pinctrl_pull0_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL0_TOG,      0x8001860C,__WRITE      ,__hw_pinctrl_pull0_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL1,          0x80018610,__READ_WRITE ,__hw_pinctrl_pull1_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL1_SET,      0x80018614,__WRITE      ,__hw_pinctrl_pull1_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL1_CLR,      0x80018618,__WRITE      ,__hw_pinctrl_pull1_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL1_TOG,      0x8001861C,__WRITE      ,__hw_pinctrl_pull1_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL2,          0x80018620,__READ_WRITE ,__hw_pinctrl_pull2_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL2_SET,      0x80018624,__WRITE      ,__hw_pinctrl_pull2_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL2_CLR,      0x80018628,__WRITE      ,__hw_pinctrl_pull2_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL2_TOG,      0x8001862C,__WRITE      ,__hw_pinctrl_pull2_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL3,          0x80018630,__READ_WRITE ,__hw_pinctrl_pull3_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL3_SET,      0x80018634,__WRITE      ,__hw_pinctrl_pull3_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL3_CLR,      0x80018638,__WRITE      ,__hw_pinctrl_pull3_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL3_TOG,      0x8001863C,__WRITE      ,__hw_pinctrl_pull3_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL4,          0x80018640,__READ_WRITE ,__hw_pinctrl_pull4_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL4_SET,      0x80018644,__WRITE      ,__hw_pinctrl_pull4_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL4_CLR,      0x80018648,__WRITE      ,__hw_pinctrl_pull4_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL4_TOG,      0x8001864C,__WRITE      ,__hw_pinctrl_pull4_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL5,          0x80018650,__READ_WRITE ,__hw_pinctrl_pull5_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL5_SET,      0x80018654,__WRITE      ,__hw_pinctrl_pull5_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL5_CLR,      0x80018658,__WRITE      ,__hw_pinctrl_pull5_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL5_TOG,      0x8001865C,__WRITE      ,__hw_pinctrl_pull5_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL6,          0x80018660,__READ_WRITE ,__hw_pinctrl_pull6_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL6_SET,      0x80018664,__WRITE      ,__hw_pinctrl_pull6_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL6_CLR,      0x80018668,__WRITE      ,__hw_pinctrl_pull6_bits);
__IO_REG32_BIT(HW_PINCTRL_PULL6_TOG,      0x8001866C,__WRITE      ,__hw_pinctrl_pull6_bits);
__IO_REG32_BIT(HW_PINCTRL_DOUT0,          0x80018700,__READ_WRITE ,__hw_pinctrl_dout0_bits);
__IO_REG32_BIT(HW_PINCTRL_DOUT0_SET,      0x80018704,__WRITE      ,__hw_pinctrl_dout0_bits);
__IO_REG32_BIT(HW_PINCTRL_DOUT0_CLR,      0x80018708,__WRITE      ,__hw_pinctrl_dout0_bits);
__IO_REG32_BIT(HW_PINCTRL_DOUT0_TOG,      0x8001870C,__WRITE      ,__hw_pinctrl_dout0_bits);
__IO_REG32_BIT(HW_PINCTRL_DOUT1,          0x80018710,__READ_WRITE ,__hw_pinctrl_dout1_bits);
__IO_REG32_BIT(HW_PINCTRL_DOUT1_SET,      0x80018714,__WRITE      ,__hw_pinctrl_dout1_bits);
__IO_REG32_BIT(HW_PINCTRL_DOUT1_CLR,      0x80018718,__WRITE      ,__hw_pinctrl_dout1_bits);
__IO_REG32_BIT(HW_PINCTRL_DOUT1_TOG,      0x8001871C,__WRITE      ,__hw_pinctrl_dout1_bits);
__IO_REG32_BIT(HW_PINCTRL_DOUT2,          0x80018720,__READ_WRITE ,__hw_pinctrl_dout2_bits);
__IO_REG32_BIT(HW_PINCTRL_DOUT2_SET,      0x80018724,__WRITE      ,__hw_pinctrl_dout2_bits);
__IO_REG32_BIT(HW_PINCTRL_DOUT2_CLR,      0x80018728,__WRITE      ,__hw_pinctrl_dout2_bits);
__IO_REG32_BIT(HW_PINCTRL_DOUT2_TOG,      0x8001872C,__WRITE      ,__hw_pinctrl_dout2_bits);
__IO_REG32_BIT(HW_PINCTRL_DOUT3,          0x80018730,__READ_WRITE ,__hw_pinctrl_dout3_bits);
__IO_REG32_BIT(HW_PINCTRL_DOUT3_SET,      0x80018734,__WRITE      ,__hw_pinctrl_dout3_bits);
__IO_REG32_BIT(HW_PINCTRL_DOUT3_CLR,      0x80018738,__WRITE      ,__hw_pinctrl_dout3_bits);
__IO_REG32_BIT(HW_PINCTRL_DOUT3_TOG,      0x8001873C,__WRITE      ,__hw_pinctrl_dout3_bits);
__IO_REG32_BIT(HW_PINCTRL_DOUT4,          0x80018740,__READ_WRITE ,__hw_pinctrl_dout4_bits);
__IO_REG32_BIT(HW_PINCTRL_DOUT4_SET,      0x80018744,__WRITE      ,__hw_pinctrl_dout4_bits);
__IO_REG32_BIT(HW_PINCTRL_DOUT4_CLR,      0x80018748,__WRITE      ,__hw_pinctrl_dout4_bits);
__IO_REG32_BIT(HW_PINCTRL_DOUT4_TOG,      0x8001874C,__WRITE      ,__hw_pinctrl_dout4_bits);
__IO_REG32_BIT(HW_PINCTRL_DIN0,           0x80018900,__READ       ,__hw_pinctrl_din0_bits);
__IO_REG32_BIT(HW_PINCTRL_DIN1,           0x80018910,__READ       ,__hw_pinctrl_din1_bits);
__IO_REG32_BIT(HW_PINCTRL_DIN2,           0x80018920,__READ       ,__hw_pinctrl_din2_bits);
__IO_REG32_BIT(HW_PINCTRL_DIN3,           0x80018930,__READ       ,__hw_pinctrl_din3_bits);
__IO_REG32_BIT(HW_PINCTRL_DIN4,           0x80018940,__READ       ,__hw_pinctrl_din4_bits);
__IO_REG32_BIT(HW_PINCTRL_DOE0,           0x80018B00,__READ_WRITE ,__hw_pinctrl_doe0_bits);
__IO_REG32_BIT(HW_PINCTRL_DOE0_SET,       0x80018B04,__WRITE      ,__hw_pinctrl_doe0_bits);
__IO_REG32_BIT(HW_PINCTRL_DOE0_CLR,       0x80018B08,__WRITE      ,__hw_pinctrl_doe0_bits);
__IO_REG32_BIT(HW_PINCTRL_DOE0_TOG,       0x80018B0C,__WRITE      ,__hw_pinctrl_doe0_bits);
__IO_REG32_BIT(HW_PINCTRL_DOE1,           0x80018B10,__READ_WRITE ,__hw_pinctrl_doe1_bits);
__IO_REG32_BIT(HW_PINCTRL_DOE1_SET,       0x80018B14,__WRITE      ,__hw_pinctrl_doe1_bits);
__IO_REG32_BIT(HW_PINCTRL_DOE1_CLR,       0x80018B18,__WRITE      ,__hw_pinctrl_doe1_bits);
__IO_REG32_BIT(HW_PINCTRL_DOE1_TOG,       0x80018B1C,__WRITE      ,__hw_pinctrl_doe1_bits);
__IO_REG32_BIT(HW_PINCTRL_DOE2,           0x80018B20,__READ_WRITE ,__hw_pinctrl_doe2_bits);
__IO_REG32_BIT(HW_PINCTRL_DOE2_SET,       0x80018B24,__WRITE      ,__hw_pinctrl_doe2_bits);
__IO_REG32_BIT(HW_PINCTRL_DOE2_CLR,       0x80018B28,__WRITE      ,__hw_pinctrl_doe2_bits);
__IO_REG32_BIT(HW_PINCTRL_DOE2_TOG,       0x80018B2C,__WRITE      ,__hw_pinctrl_doe2_bits);
__IO_REG32_BIT(HW_PINCTRL_DOE3,           0x80018B30,__READ_WRITE ,__hw_pinctrl_doe3_bits);
__IO_REG32_BIT(HW_PINCTRL_DOE3_SET,       0x80018B34,__WRITE      ,__hw_pinctrl_doe3_bits);
__IO_REG32_BIT(HW_PINCTRL_DOE3_CLR,       0x80018B38,__WRITE      ,__hw_pinctrl_doe3_bits);
__IO_REG32_BIT(HW_PINCTRL_DOE3_TOG,       0x80018B3C,__WRITE      ,__hw_pinctrl_doe3_bits);
__IO_REG32_BIT(HW_PINCTRL_DOE4,           0x80018B40,__READ_WRITE ,__hw_pinctrl_doe4_bits);
__IO_REG32_BIT(HW_PINCTRL_DOE4_SET,       0x80018B44,__WRITE      ,__hw_pinctrl_doe4_bits);
__IO_REG32_BIT(HW_PINCTRL_DOE4_CLR,       0x80018B48,__WRITE      ,__hw_pinctrl_doe4_bits);
__IO_REG32_BIT(HW_PINCTRL_DOE4_TOG,       0x80018B4C,__WRITE      ,__hw_pinctrl_doe4_bits);
__IO_REG32_BIT(HW_PINCTRL_PIN2IRQ0,       0x80019000,__READ_WRITE ,__hw_pinctrl_pin2irq0_bits);
__IO_REG32_BIT(HW_PINCTRL_PIN2IRQ0_SET,   0x80019004,__WRITE      ,__hw_pinctrl_pin2irq0_bits);
__IO_REG32_BIT(HW_PINCTRL_PIN2IRQ0_CLR,   0x80019008,__WRITE      ,__hw_pinctrl_pin2irq0_bits);
__IO_REG32_BIT(HW_PINCTRL_PIN2IRQ0_TOG,   0x8001900C,__WRITE      ,__hw_pinctrl_pin2irq0_bits);
__IO_REG32_BIT(HW_PINCTRL_PIN2IRQ1,       0x80019010,__READ_WRITE ,__hw_pinctrl_pin2irq1_bits);
__IO_REG32_BIT(HW_PINCTRL_PIN2IRQ1_SET,   0x80019014,__WRITE      ,__hw_pinctrl_pin2irq1_bits);
__IO_REG32_BIT(HW_PINCTRL_PIN2IRQ1_CLR,   0x80019018,__WRITE      ,__hw_pinctrl_pin2irq1_bits);
__IO_REG32_BIT(HW_PINCTRL_PIN2IRQ1_TOG,   0x8001901C,__WRITE      ,__hw_pinctrl_pin2irq1_bits);
__IO_REG32_BIT(HW_PINCTRL_PIN2IRQ2,       0x80019020,__READ_WRITE ,__hw_pinctrl_pin2irq2_bits);
__IO_REG32_BIT(HW_PINCTRL_PIN2IRQ2_SET,   0x80019024,__WRITE      ,__hw_pinctrl_pin2irq2_bits);
__IO_REG32_BIT(HW_PINCTRL_PIN2IRQ2_CLR,   0x80019028,__WRITE      ,__hw_pinctrl_pin2irq2_bits);
__IO_REG32_BIT(HW_PINCTRL_PIN2IRQ2_TOG,   0x8001902C,__WRITE      ,__hw_pinctrl_pin2irq2_bits);
__IO_REG32_BIT(HW_PINCTRL_PIN2IRQ3,       0x80019030,__READ_WRITE ,__hw_pinctrl_pin2irq3_bits);
__IO_REG32_BIT(HW_PINCTRL_PIN2IRQ3_SET,   0x80019034,__WRITE      ,__hw_pinctrl_pin2irq3_bits);
__IO_REG32_BIT(HW_PINCTRL_PIN2IRQ3_CLR,   0x80019038,__WRITE      ,__hw_pinctrl_pin2irq3_bits);
__IO_REG32_BIT(HW_PINCTRL_PIN2IRQ3_TOG,   0x8001903C,__WRITE      ,__hw_pinctrl_pin2irq3_bits);
__IO_REG32_BIT(HW_PINCTRL_PIN2IRQ4,       0x80019040,__READ_WRITE ,__hw_pinctrl_pin2irq4_bits);
__IO_REG32_BIT(HW_PINCTRL_PIN2IRQ4_SET,   0x80019044,__WRITE      ,__hw_pinctrl_pin2irq4_bits);
__IO_REG32_BIT(HW_PINCTRL_PIN2IRQ4_CLR,   0x80019048,__WRITE      ,__hw_pinctrl_pin2irq4_bits);
__IO_REG32_BIT(HW_PINCTRL_PIN2IRQ4_TOG,   0x8001904C,__WRITE      ,__hw_pinctrl_pin2irq4_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQEN0,         0x80019100,__READ_WRITE ,__hw_pinctrl_irqen0_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQEN0_SET,     0x80019104,__WRITE      ,__hw_pinctrl_irqen0_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQEN0_CLR,     0x80019108,__WRITE      ,__hw_pinctrl_irqen0_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQEN0_TOG,     0x8001910C,__WRITE      ,__hw_pinctrl_irqen0_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQEN1,         0x80019110,__READ_WRITE ,__hw_pinctrl_irqen1_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQEN1_SET,     0x80019114,__WRITE      ,__hw_pinctrl_irqen1_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQEN1_CLR,     0x80019118,__WRITE      ,__hw_pinctrl_irqen1_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQEN1_TOG,     0x8001911C,__WRITE      ,__hw_pinctrl_irqen1_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQEN2,         0x80019120,__READ_WRITE ,__hw_pinctrl_irqen2_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQEN2_SET,     0x80019124,__WRITE      ,__hw_pinctrl_irqen2_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQEN2_CLR,     0x80019128,__WRITE      ,__hw_pinctrl_irqen2_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQEN2_TOG,     0x8001912C,__WRITE      ,__hw_pinctrl_irqen2_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQEN3,         0x80019130,__READ_WRITE ,__hw_pinctrl_irqen3_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQEN3_SET,     0x80019134,__WRITE      ,__hw_pinctrl_irqen3_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQEN3_CLR,     0x80019138,__WRITE      ,__hw_pinctrl_irqen3_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQEN3_TOG,     0x8001913C,__WRITE      ,__hw_pinctrl_irqen3_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQEN4,         0x80019140,__READ_WRITE ,__hw_pinctrl_irqen4_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQEN4_SET,     0x80019144,__WRITE      ,__hw_pinctrl_irqen4_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQEN4_CLR,     0x80019148,__WRITE      ,__hw_pinctrl_irqen4_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQEN4_TOG,     0x8001914C,__WRITE      ,__hw_pinctrl_irqen4_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQLEVEL0,      0x80019200,__READ_WRITE ,__hw_pinctrl_irqlevel0_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQLEVEL0_SET,  0x80019204,__WRITE      ,__hw_pinctrl_irqlevel0_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQLEVEL0_CLR,  0x80019208,__WRITE      ,__hw_pinctrl_irqlevel0_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQLEVEL0_TOG,  0x8001920C,__WRITE      ,__hw_pinctrl_irqlevel0_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQLEVEL1,      0x80019210,__READ_WRITE ,__hw_pinctrl_irqlevel1_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQLEVEL1_SET,  0x80019214,__WRITE      ,__hw_pinctrl_irqlevel1_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQLEVEL1_CLR,  0x80019218,__WRITE      ,__hw_pinctrl_irqlevel1_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQLEVEL1_TOG,  0x8001921C,__WRITE      ,__hw_pinctrl_irqlevel1_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQLEVEL2,      0x80019220,__READ_WRITE ,__hw_pinctrl_irqlevel2_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQLEVEL2_SET,  0x80019224,__WRITE      ,__hw_pinctrl_irqlevel2_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQLEVEL2_CLR,  0x80019228,__WRITE      ,__hw_pinctrl_irqlevel2_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQLEVEL2_TOG,  0x8001922C,__WRITE      ,__hw_pinctrl_irqlevel2_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQLEVEL3,      0x80019230,__READ_WRITE ,__hw_pinctrl_irqlevel3_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQLEVEL3_SET,  0x80019234,__WRITE      ,__hw_pinctrl_irqlevel3_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQLEVEL3_CLR,  0x80019238,__WRITE      ,__hw_pinctrl_irqlevel3_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQLEVEL3_TOG,  0x8001923C,__WRITE      ,__hw_pinctrl_irqlevel3_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQLEVEL4,      0x80019240,__READ_WRITE ,__hw_pinctrl_irqlevel4_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQLEVEL4_SET,  0x80019244,__WRITE      ,__hw_pinctrl_irqlevel4_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQLEVEL4_CLR,  0x80019248,__WRITE      ,__hw_pinctrl_irqlevel4_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQLEVEL4_TOG,  0x8001924C,__WRITE      ,__hw_pinctrl_irqlevel4_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQPOL0,        0x80019300,__READ_WRITE ,__hw_pinctrl_irqpol0_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQPOL0_SET,    0x80019304,__WRITE      ,__hw_pinctrl_irqpol0_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQPOL0_CLR,    0x80019308,__WRITE      ,__hw_pinctrl_irqpol0_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQPOL0_TOG,    0x8001930C,__WRITE      ,__hw_pinctrl_irqpol0_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQPOL1,        0x80019310,__READ_WRITE ,__hw_pinctrl_irqpol1_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQPOL1_SET,    0x80019314,__WRITE      ,__hw_pinctrl_irqpol1_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQPOL1_CLR,    0x80019318,__WRITE      ,__hw_pinctrl_irqpol1_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQPOL1_TOG,    0x8001931C,__WRITE      ,__hw_pinctrl_irqpol1_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQPOL2,        0x80019320,__READ_WRITE ,__hw_pinctrl_irqpol2_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQPOL2_SET,    0x80019324,__WRITE      ,__hw_pinctrl_irqpol2_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQPOL2_CLR,    0x80019328,__WRITE      ,__hw_pinctrl_irqpol2_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQPOL2_TOG,    0x8001932C,__WRITE      ,__hw_pinctrl_irqpol2_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQPOL3,        0x80019330,__READ_WRITE ,__hw_pinctrl_irqpol3_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQPOL3_SET,    0x80019334,__WRITE      ,__hw_pinctrl_irqpol3_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQPOL3_CLR,    0x80019338,__WRITE      ,__hw_pinctrl_irqpol3_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQPOL3_TOG,    0x8001933C,__WRITE      ,__hw_pinctrl_irqpol3_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQPOL4,        0x80019340,__READ_WRITE ,__hw_pinctrl_irqpol4_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQPOL4_SET,    0x80019344,__WRITE      ,__hw_pinctrl_irqpol4_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQPOL4_CLR,    0x80019348,__WRITE      ,__hw_pinctrl_irqpol4_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQPOL4_TOG,    0x8001934C,__WRITE      ,__hw_pinctrl_irqpol4_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQSTAT0,       0x80019400,__READ_WRITE ,__hw_pinctrl_irqstat0_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQSTAT0_SET,   0x80019404,__WRITE      ,__hw_pinctrl_irqstat0_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQSTAT0_CLR,   0x80019408,__WRITE      ,__hw_pinctrl_irqstat0_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQSTAT0_TOG,   0x8001940C,__WRITE      ,__hw_pinctrl_irqstat0_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQSTAT1,       0x80019410,__READ_WRITE ,__hw_pinctrl_irqstat1_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQSTAT1_SET,   0x80019414,__WRITE      ,__hw_pinctrl_irqstat1_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQSTAT1_CLR,   0x80019418,__WRITE      ,__hw_pinctrl_irqstat1_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQSTAT1_TOG,   0x8001941C,__WRITE      ,__hw_pinctrl_irqstat1_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQSTAT2,       0x80019420,__READ_WRITE ,__hw_pinctrl_irqstat2_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQSTAT2_SET,   0x80019424,__WRITE      ,__hw_pinctrl_irqstat2_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQSTAT2_CLR,   0x80019428,__WRITE      ,__hw_pinctrl_irqstat2_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQSTAT2_TOG,   0x8001942C,__WRITE      ,__hw_pinctrl_irqstat2_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQSTAT3,       0x80019430,__READ_WRITE ,__hw_pinctrl_irqstat3_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQSTAT3_SET,   0x80019434,__WRITE      ,__hw_pinctrl_irqstat3_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQSTAT3_CLR,   0x80019438,__WRITE      ,__hw_pinctrl_irqstat3_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQSTAT3_TOG,   0x8001943C,__WRITE      ,__hw_pinctrl_irqstat3_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQSTAT4,       0x80019440,__READ_WRITE ,__hw_pinctrl_irqstat4_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQSTAT4_SET,   0x80019444,__WRITE      ,__hw_pinctrl_irqstat4_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQSTAT4_CLR,   0x80019448,__WRITE      ,__hw_pinctrl_irqstat4_bits);
__IO_REG32_BIT(HW_PINCTRL_IRQSTAT4_TOG,   0x8001944C,__WRITE      ,__hw_pinctrl_irqstat4_bits);
__IO_REG32_BIT(HW_PINCTRL_EMI_ODT_CTRL,   0x80019A40,__READ_WRITE ,__hw_pinctrl_emi_odt_ctrl_bits);
__IO_REG32_BIT(HW_PINCTRL_EMI_DS_CTRL,    0x80019B80,__READ_WRITE ,__hw_pinctrl_emi_ds_ctrl_bits);

/***************************************************************************
 **
 **  CLKCTRL
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_CLKCTRL_PLL0CTRL0,      0x80040000,__READ_WRITE,__hw_clkctrl_pll0_ctrl0_bits);
__IO_REG32_BIT(HW_CLKCTRL_PLL0CTRL0_SET,  0x80040004,__WRITE     ,__hw_clkctrl_pll0_ctrl0_bits);
__IO_REG32_BIT(HW_CLKCTRL_PLL0CTRL0_CLR,  0x80040008,__WRITE     ,__hw_clkctrl_pll0_ctrl0_bits);
__IO_REG32_BIT(HW_CLKCTRL_PLL0CTRL0_TOG,  0x8004000C,__WRITE     ,__hw_clkctrl_pll0_ctrl0_bits);
__IO_REG32_BIT(HW_CLKCTRL_PLL0CTRL1,      0x80040010,__READ_WRITE,__hw_clkctrl_pll0_ctrl1_bits);
__IO_REG32_BIT(HW_CLKCTRL_PLL1CTRL0,      0x80040020,__READ_WRITE,__hw_clkctrl_pll1_ctrl0_bits);
__IO_REG32_BIT(HW_CLKCTRL_PLL1CTRL0_SET,  0x80040024,__WRITE     ,__hw_clkctrl_pll1_ctrl0_bits);
__IO_REG32_BIT(HW_CLKCTRL_PLL1CTRL0_CLR,  0x80040028,__WRITE     ,__hw_clkctrl_pll1_ctrl0_bits);
__IO_REG32_BIT(HW_CLKCTRL_PLL1CTRL0_TOG,  0x8004002C,__WRITE     ,__hw_clkctrl_pll1_ctrl0_bits);
__IO_REG32_BIT(HW_CLKCTRL_PLL1CTRL1,      0x80040030,__READ_WRITE,__hw_clkctrl_pll1_ctrl1_bits);
__IO_REG32_BIT(HW_CLKCTRL_PLL2CTRL0,      0x80040040,__READ_WRITE,__hw_clkctrl_pll2_ctrl0_bits);
__IO_REG32_BIT(HW_CLKCTRL_CPU,            0x80040050,__READ_WRITE,__hw_clkctrl_cpu_bits);
__IO_REG32_BIT(HW_CLKCTRL_CPU_SET,        0x80040054,__WRITE     ,__hw_clkctrl_cpu_bits);
__IO_REG32_BIT(HW_CLKCTRL_CPU_CLR,        0x80040058,__WRITE     ,__hw_clkctrl_cpu_bits);
__IO_REG32_BIT(HW_CLKCTRL_CPU_TOG,        0x8004005C,__WRITE     ,__hw_clkctrl_cpu_bits);
__IO_REG32_BIT(HW_CLKCTRL_HBUS,           0x80040060,__READ_WRITE,__hw_clkctrl_hbus_bits);
__IO_REG32_BIT(HW_CLKCTRL_HBUS_SET,       0x80040064,__WRITE     ,__hw_clkctrl_hbus_bits);
__IO_REG32_BIT(HW_CLKCTRL_HBUS_CLR,       0x80040068,__WRITE     ,__hw_clkctrl_hbus_bits);
__IO_REG32_BIT(HW_CLKCTRL_HBUS_TOG,       0x8004006C,__WRITE     ,__hw_clkctrl_hbus_bits);
__IO_REG32_BIT(HW_CLKCTRL_XBUS,           0x80040070,__READ_WRITE,__hw_clkctrl_xbus_bits);
__IO_REG32_BIT(HW_CLKCTRL_XTAL,           0x80040080,__READ_WRITE,__hw_clkctrl_xtal_bits);
__IO_REG32_BIT(HW_CLKCTRL_XTAL_SET,       0x80040084,__WRITE     ,__hw_clkctrl_xtal_bits);
__IO_REG32_BIT(HW_CLKCTRL_XTAL_CLR,       0x80040088,__WRITE     ,__hw_clkctrl_xtal_bits);
__IO_REG32_BIT(HW_CLKCTRL_XTAL_TOG,       0x8004008C,__WRITE     ,__hw_clkctrl_xtal_bits);
__IO_REG32_BIT(HW_CLKCTRL_SSP0,           0x80040090,__READ_WRITE,__hw_clkctrl_ssp_bits);
__IO_REG32_BIT(HW_CLKCTRL_SSP1,           0x800400A0,__READ_WRITE,__hw_clkctrl_ssp_bits);
__IO_REG32_BIT(HW_CLKCTRL_SSP2,           0x800400B0,__READ_WRITE,__hw_clkctrl_ssp_bits);
__IO_REG32_BIT(HW_CLKCTRL_SSP3,           0x800400C0,__READ_WRITE,__hw_clkctrl_ssp_bits);
__IO_REG32_BIT(HW_CLKCTRL_GPMI,           0x800400D0,__READ_WRITE,__hw_clkctrl_gpmi_bits);
__IO_REG32_BIT(HW_CLKCTRL_SPDIF,          0x800400E0,__READ_WRITE,__hw_clkctrl_spdif_bits);
__IO_REG32_BIT(HW_CLKCTRL_EMI,            0x800400F0,__READ_WRITE,__hw_clkctrl_emi_bits);
__IO_REG32_BIT(HW_CLKCTRL_SAIF0,          0x80040100,__READ_WRITE,__hw_clkctrl_saif_bits);
__IO_REG32_BIT(HW_CLKCTRL_SAIF1,          0x80040110,__READ_WRITE,__hw_clkctrl_saif_bits);
__IO_REG32_BIT(HW_CLKCTRL_DIS_LCDIF,      0x80040120,__READ_WRITE,__hw_clkctrl_dis_lcdif_bits);
__IO_REG32_BIT(HW_CLKCTRL_ETM,            0x80040130,__READ_WRITE,__hw_clkctrl_etm_bits);
__IO_REG32_BIT(HW_CLKCTRL_ENET,           0x80040140,__READ_WRITE,__hw_clkctrl_enet_bits);
__IO_REG32_BIT(HW_CLKCTRL_HSADC,          0x80040150,__READ_WRITE,__hw_clkctrl_hsadc_bits);
__IO_REG32_BIT(HW_CLKCTRL_FLEXCAN,        0x80040160,__READ_WRITE,__hw_clkctrl_flexcan_bits);
__IO_REG32_BIT(HW_CLKCTRL_FRAC0,          0x800401B0,__READ_WRITE,__hw_clkctrl_frac0_bits);
__IO_REG32_BIT(HW_CLKCTRL_FRAC0_SET,      0x800401B4,__WRITE     ,__hw_clkctrl_frac0_bits);
__IO_REG32_BIT(HW_CLKCTRL_FRAC0_CLR,      0x800401B8,__WRITE     ,__hw_clkctrl_frac0_bits);
__IO_REG32_BIT(HW_CLKCTRL_FRAC0_TOG,      0x800401BC,__WRITE     ,__hw_clkctrl_frac0_bits);
__IO_REG32_BIT(HW_CLKCTRL_FRAC1,          0x800401C0,__READ_WRITE,__hw_clkctrl_frac1_bits);
__IO_REG32_BIT(HW_CLKCTRL_FRAC1_SET,      0x800401C4,__WRITE     ,__hw_clkctrl_frac1_bits);
__IO_REG32_BIT(HW_CLKCTRL_FRAC1_CLR,      0x800401C8,__WRITE     ,__hw_clkctrl_frac1_bits);
__IO_REG32_BIT(HW_CLKCTRL_FRAC1_TOG,      0x800401CC,__WRITE     ,__hw_clkctrl_frac1_bits);
__IO_REG32_BIT(HW_CLKCTRL_CLKSEQ,         0x800401D0,__READ_WRITE,__hw_clkctrl_clkseq_bits);
__IO_REG32_BIT(HW_CLKCTRL_CLKSEQ_SET,     0x800401D4,__WRITE     ,__hw_clkctrl_clkseq_bits);
__IO_REG32_BIT(HW_CLKCTRL_CLKSEQ_CLR,     0x800401D8,__WRITE     ,__hw_clkctrl_clkseq_bits);
__IO_REG32_BIT(HW_CLKCTRL_CLKSEQ_TOG,     0x800401DC,__WRITE     ,__hw_clkctrl_clkseq_bits);
__IO_REG32_BIT(HW_CLKCTRL_RESET,          0x800401E0,__READ_WRITE,__hw_clkctrl_reset_bits);
__IO_REG32_BIT(HW_CLKCTRL_STATUS,         0x800401F0,__READ      ,__hw_clkctrl_status_bits);
__IO_REG32_BIT(HW_CLKCTRL_VERSION,        0x80040200,__READ      ,__hw_clkctrl_version_bits);

/***************************************************************************
 **
 **  POWER
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_POWER_CTRL,             0x80044000,__READ_WRITE,__hw_power_ctrl_bits);
__IO_REG32_BIT(HW_POWER_CTRL_SET,         0x80044004,__WRITE     ,__hw_power_ctrl_bits);
__IO_REG32_BIT(HW_POWER_CTRL_CLR,         0x80044008,__WRITE     ,__hw_power_ctrl_bits);
__IO_REG32_BIT(HW_POWER_CTRL_TOG,         0x8004400C,__WRITE     ,__hw_power_ctrl_bits);
__IO_REG32_BIT(HW_POWER_5VCTRL,           0x80044010,__READ_WRITE,__hw_power_5vctrl_bits);
__IO_REG32_BIT(HW_POWER_5VCTRL_SET,       0x80044014,__WRITE     ,__hw_power_5vctrl_bits);
__IO_REG32_BIT(HW_POWER_5VCTRL_CLR,       0x80044018,__WRITE     ,__hw_power_5vctrl_bits);
__IO_REG32_BIT(HW_POWER_5VCTRL_TOG,       0x8004401C,__WRITE     ,__hw_power_5vctrl_bits);
__IO_REG32_BIT(HW_POWER_MINPWR,           0x80044020,__READ_WRITE,__hw_power_minpwr_bits);
__IO_REG32_BIT(HW_POWER_MINPWR_SET,       0x80044024,__WRITE     ,__hw_power_minpwr_bits);
__IO_REG32_BIT(HW_POWER_MINPWR_CLR,       0x80044028,__WRITE     ,__hw_power_minpwr_bits);
__IO_REG32_BIT(HW_POWER_MINPWR_TOG,       0x8004402C,__WRITE     ,__hw_power_minpwr_bits);
__IO_REG32_BIT(HW_POWER_CHARGE,           0x80044030,__READ_WRITE,__hw_power_charge_bits);
__IO_REG32_BIT(HW_POWER_CHARGE_SET,       0x80044034,__WRITE     ,__hw_power_charge_bits);
__IO_REG32_BIT(HW_POWER_CHARGE_CLR,       0x80044038,__WRITE     ,__hw_power_charge_bits);
__IO_REG32_BIT(HW_POWER_CHARGE_TOG,       0x8004403C,__WRITE     ,__hw_power_charge_bits);
__IO_REG32_BIT(HW_POWER_VDDDCTRL,         0x80044040,__READ_WRITE,__hw_power_vdddctrl_bits);
__IO_REG32_BIT(HW_POWER_VDDACTRL,         0x80044050,__READ_WRITE,__hw_power_vddactrl_bits);
__IO_REG32_BIT(HW_POWER_VDDIOCTRL,        0x80044060,__READ_WRITE,__hw_power_vddioctrl_bits);
__IO_REG32_BIT(HW_POWER_VDDMEMCTRL,       0x80044070,__READ_WRITE,__hw_power_vddmemctrl_bits);
__IO_REG32_BIT(HW_POWER_DCDC4P2,          0x80044080,__READ_WRITE,__hw_power_dcdc4p2_bits);
__IO_REG32_BIT(HW_POWER_MISC,             0x80044090,__READ_WRITE,__hw_power_misc_bits);
__IO_REG32_BIT(HW_POWER_DCLIMITS,         0x800440A0,__READ_WRITE,__hw_power_dclimits_bits);
__IO_REG32_BIT(HW_POWER_LOOPCTRL,         0x800440B0,__READ_WRITE,__hw_power_loopctrl_bits);
__IO_REG32_BIT(HW_POWER_LOOPCTRL_SET,     0x800440B4,__WRITE     ,__hw_power_loopctrl_bits);
__IO_REG32_BIT(HW_POWER_LOOPCTRL_CLR,     0x800440B8,__WRITE     ,__hw_power_loopctrl_bits);
__IO_REG32_BIT(HW_POWER_LOOPCTRL_TOG,     0x800440BC,__WRITE     ,__hw_power_loopctrl_bits);
__IO_REG32_BIT(HW_POWER_STS,              0x800440C0,__READ      ,__hw_power_sts_bits);
__IO_REG32_BIT(HW_POWER_SPEED,            0x800440D0,__READ_WRITE,__hw_power_speed_bits);
__IO_REG32_BIT(HW_POWER_SPEED_SET,        0x800440D4,__WRITE     ,__hw_power_speed_bits);
__IO_REG32_BIT(HW_POWER_SPEED_CLR,        0x800440D8,__WRITE     ,__hw_power_speed_bits);
__IO_REG32_BIT(HW_POWER_SPEED_TOG,        0x800440DC,__WRITE     ,__hw_power_speed_bits);
__IO_REG32_BIT(HW_POWER_BATTMONITOR,      0x800440E0,__READ_WRITE,__hw_power_battmonitor_bits);
__IO_REG32_BIT(HW_POWER_RESET,            0x80044100,__READ_WRITE,__hw_power_reset_bits);
__IO_REG32_BIT(HW_POWER_DEBUG,            0x80044110,__READ_WRITE,__hw_power_debug_bits);
__IO_REG32_BIT(HW_POWER_DEBUG_SET,        0x80044114,__WRITE     ,__hw_power_debug_bits);
__IO_REG32_BIT(HW_POWER_DEBUG_CLR,        0x80044118,__WRITE     ,__hw_power_debug_bits);
__IO_REG32_BIT(HW_POWER_DEBUG_TOG,        0x8004411C,__WRITE     ,__hw_power_debug_bits);
__IO_REG32_BIT(HW_POWER_THERMAL,          0x80044120,__READ_WRITE,__hw_power_thermal_bits);
__IO_REG32_BIT(HW_POWER_THERMAL_SET,      0x80044124,__WRITE     ,__hw_power_thermal_bits);
__IO_REG32_BIT(HW_POWER_THERMAL_CLR,      0x80044128,__WRITE     ,__hw_power_thermal_bits);
__IO_REG32_BIT(HW_POWER_THERMAL_TOG,      0x8004412C,__WRITE     ,__hw_power_thermal_bits);
__IO_REG32_BIT(HW_POWER_USB1CTRL,         0x80044130,__READ_WRITE,__hw_power_usb1ctrl_bits);
__IO_REG32_BIT(HW_POWER_USB1CTRL_SET,     0x80044134,__WRITE     ,__hw_power_usb1ctrl_bits);
__IO_REG32_BIT(HW_POWER_USB1CTRL_CLR,     0x80044138,__WRITE     ,__hw_power_usb1ctrl_bits);
__IO_REG32_BIT(HW_POWER_USB1CTRL_TOG,     0x8004413C,__WRITE     ,__hw_power_usb1ctrl_bits);
__IO_REG32(    HW_POWER_SPECIAL,          0x80044140,__READ_WRITE);
__IO_REG32(    HW_POWER_SPECIAL_SET,      0x80044144,__WRITE     );
__IO_REG32(    HW_POWER_SPECIAL_CLR,      0x80044148,__WRITE     );
__IO_REG32(    HW_POWER_SPECIAL_TOG,      0x8004414C,__WRITE     );
__IO_REG32_BIT(HW_POWER_VERSION,          0x80044150,__READ      ,__hw_power_version_bits);
__IO_REG32_BIT(HW_POWER_ANACLKCTRL,       0x80044160,__READ_WRITE,__hw_power_anaclkctrl_bits);
__IO_REG32_BIT(HW_POWER_ANACLKCTRL_SET,   0x80044164,__WRITE     ,__hw_power_anaclkctrl_bits);
__IO_REG32_BIT(HW_POWER_ANACLKCTRL_CLR,   0x80044168,__WRITE     ,__hw_power_anaclkctrl_bits);
__IO_REG32_BIT(HW_POWER_ANACLKCTRL_TOG,   0x8004416C,__WRITE     ,__hw_power_anaclkctrl_bits);
__IO_REG32_BIT(HW_POWER_REFCTRL,          0x80044170,__READ_WRITE,__hw_power_refctrl_bits);
__IO_REG32_BIT(HW_POWER_REFCTRL_SET,      0x80044174,__WRITE     ,__hw_power_refctrl_bits);
__IO_REG32_BIT(HW_POWER_REFCTRL_CLR,      0x80044178,__WRITE     ,__hw_power_refctrl_bits);
__IO_REG32_BIT(HW_POWER_REFCTRL_TOG,      0x8004417C,__WRITE     ,__hw_power_refctrl_bits);

/***************************************************************************
 **
 **  DCP
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_DCP_CTRL,               0x80028000,__READ_WRITE,__hw_dcp_ctrl_bits);
__IO_REG32_BIT(HW_DCP_CTRL_SET,           0x80028004,__WRITE     ,__hw_dcp_ctrl_bits);
__IO_REG32_BIT(HW_DCP_CTRL_CLR,           0x80028008,__WRITE     ,__hw_dcp_ctrl_bits);
__IO_REG32_BIT(HW_DCP_CTRL_TOG,           0x8002800C,__WRITE     ,__hw_dcp_ctrl_bits);
__IO_REG32_BIT(HW_DCP_STAT,               0x80028010,__READ_WRITE,__hw_dcp_stat_bits);
__IO_REG32_BIT(HW_DCP_STAT_SET,           0x80028014,__WRITE     ,__hw_dcp_stat_bits);
__IO_REG32_BIT(HW_DCP_STAT_CLR,           0x80028018,__WRITE     ,__hw_dcp_stat_bits);
__IO_REG32_BIT(HW_DCP_STAT_TOG,           0x8002801C,__WRITE     ,__hw_dcp_stat_bits);
__IO_REG32_BIT(HW_DCP_CHANNELCTRL,        0x80028020,__READ_WRITE,__hw_dcp_channelctrl_bits);
__IO_REG32_BIT(HW_DCP_CHANNELCTRL_SET,    0x80028024,__WRITE     ,__hw_dcp_channelctrl_bits);
__IO_REG32_BIT(HW_DCP_CHANNELCTRL_CLR,    0x80028028,__WRITE     ,__hw_dcp_channelctrl_bits);
__IO_REG32_BIT(HW_DCP_CHANNELCTRL_TOG,    0x8002802C,__WRITE     ,__hw_dcp_channelctrl_bits);
__IO_REG32_BIT(HW_DCP_CAPABILITY0,        0x80028030,__READ_WRITE,__hw_dcp_capability0_bits);
__IO_REG32_BIT(HW_DCP_CAPABILITY1,        0x80028040,__READ      ,__hw_dcp_capability1_bits);
__IO_REG32(    HW_DCP_CONTEXT,            0x80028050,__READ_WRITE);
__IO_REG32_BIT(HW_DCP_KEY,                0x80028060,__READ_WRITE,__hw_dcp_key_bits);
__IO_REG32(    HW_DCP_KEYDATA,            0x80028070,__READ_WRITE);
__IO_REG32(    HW_DCP_PACKET0,            0x80028080,__READ      );
__IO_REG32_BIT(HW_DCP_PACKET1,            0x80028090,__READ      ,__hw_dcp_packet1_bits);
__IO_REG32_BIT(HW_DCP_PACKET2,            0x800280A0,__READ      ,__hw_dcp_packet2_bits);
__IO_REG32(    HW_DCP_PACKET3,            0x800280B0,__READ      );
__IO_REG32(    HW_DCP_PACKET4,            0x800280C0,__READ      );
__IO_REG32(    HW_DCP_PACKET5,            0x800280D0,__READ      );
__IO_REG32(    HW_DCP_PACKET6,            0x800280E0,__READ      );
__IO_REG32(    HW_DCP_CH0CMDPTR,          0x80028100,__READ_WRITE);
__IO_REG32_BIT(HW_DCP_CH0SEMA,            0x80028110,__READ_WRITE,__hw_dcp_chsema_bits);
__IO_REG32_BIT(HW_DCP_CH0STAT,            0x80028120,__READ_WRITE,__hw_dcp_chstat_bits);
__IO_REG32_BIT(HW_DCP_CH0STAT_SET,        0x80028124,__WRITE     ,__hw_dcp_chstat_bits);
__IO_REG32_BIT(HW_DCP_CH0STAT_CLR,        0x80028128,__WRITE     ,__hw_dcp_chstat_bits);
__IO_REG32_BIT(HW_DCP_CH0STAT_TOG,        0x8002812C,__WRITE     ,__hw_dcp_chstat_bits);
__IO_REG32_BIT(HW_DCP_CH0OPTS,            0x80028130,__READ_WRITE,__hw_dcp_chopts_bits);
__IO_REG32_BIT(HW_DCP_CH0OPTS_SET,        0x80028134,__WRITE     ,__hw_dcp_chopts_bits);
__IO_REG32_BIT(HW_DCP_CH0OPTS_CLR,        0x80028138,__WRITE     ,__hw_dcp_chopts_bits);
__IO_REG32_BIT(HW_DCP_CH0OPTS_TOG,        0x8002813C,__WRITE     ,__hw_dcp_chopts_bits);
__IO_REG32(    HW_DCP_CH1CMDPTR,          0x80028140,__READ_WRITE);
__IO_REG32_BIT(HW_DCP_CH1SEMA,            0x80028150,__READ_WRITE,__hw_dcp_chsema_bits);
__IO_REG32_BIT(HW_DCP_CH1STAT,            0x80028160,__READ_WRITE,__hw_dcp_chstat_bits);
__IO_REG32_BIT(HW_DCP_CH1STAT_SET,        0x80028164,__WRITE     ,__hw_dcp_chstat_bits);
__IO_REG32_BIT(HW_DCP_CH1STAT_CLR,        0x80028168,__WRITE     ,__hw_dcp_chstat_bits);
__IO_REG32_BIT(HW_DCP_CH1STAT_TOG,        0x8002816C,__WRITE     ,__hw_dcp_chstat_bits);
__IO_REG32_BIT(HW_DCP_CH1OPTS,            0x80028170,__READ_WRITE,__hw_dcp_chopts_bits);
__IO_REG32_BIT(HW_DCP_CH1OPTS_SET,        0x80028174,__WRITE     ,__hw_dcp_chopts_bits);
__IO_REG32_BIT(HW_DCP_CH1OPTS_CLR,        0x80028178,__WRITE     ,__hw_dcp_chopts_bits);
__IO_REG32_BIT(HW_DCP_CH1OPTS_TOG,        0x8002817C,__WRITE     ,__hw_dcp_chopts_bits);
__IO_REG32(    HW_DCP_CH2CMDPTR,          0x80028180,__READ_WRITE);
__IO_REG32_BIT(HW_DCP_CH2SEMA,            0x80028190,__READ_WRITE,__hw_dcp_chsema_bits);
__IO_REG32_BIT(HW_DCP_CH2STAT,            0x800281A0,__READ_WRITE,__hw_dcp_chstat_bits);
__IO_REG32_BIT(HW_DCP_CH2STAT_SET,        0x800281A4,__WRITE     ,__hw_dcp_chstat_bits);
__IO_REG32_BIT(HW_DCP_CH2STAT_CLR,        0x800281A8,__WRITE     ,__hw_dcp_chstat_bits);
__IO_REG32_BIT(HW_DCP_CH2STAT_TOG,        0x800281AC,__WRITE     ,__hw_dcp_chstat_bits);
__IO_REG32_BIT(HW_DCP_CH2OPTS,            0x800281B0,__READ_WRITE,__hw_dcp_chopts_bits);
__IO_REG32_BIT(HW_DCP_CH2OPTS_SET,        0x800281B4,__WRITE     ,__hw_dcp_chopts_bits);
__IO_REG32_BIT(HW_DCP_CH2OPTS_CLR,        0x800281B8,__WRITE     ,__hw_dcp_chopts_bits);
__IO_REG32_BIT(HW_DCP_CH2OPTS_TOG,        0x800281BC,__WRITE     ,__hw_dcp_chopts_bits);
__IO_REG32(    HW_DCP_CH3CMDPTR,          0x800281C0,__READ_WRITE);
__IO_REG32_BIT(HW_DCP_CH3SEMA,            0x800281D0,__READ_WRITE,__hw_dcp_chsema_bits);
__IO_REG32_BIT(HW_DCP_CH3STAT,            0x800281E0,__READ_WRITE,__hw_dcp_chstat_bits);
__IO_REG32_BIT(HW_DCP_CH3STAT_SET,        0x800281E4,__WRITE     ,__hw_dcp_chstat_bits);
__IO_REG32_BIT(HW_DCP_CH3STAT_CLR,        0x800281E8,__WRITE     ,__hw_dcp_chstat_bits);
__IO_REG32_BIT(HW_DCP_CH3STAT_TOG,        0x800281EC,__WRITE     ,__hw_dcp_chstat_bits);
__IO_REG32_BIT(HW_DCP_CH3OPTS,            0x800281F0,__READ_WRITE,__hw_dcp_chopts_bits);
__IO_REG32_BIT(HW_DCP_CH3OPTS_SET,        0x800281F4,__WRITE     ,__hw_dcp_chopts_bits);
__IO_REG32_BIT(HW_DCP_CH3OPTS_CLR,        0x800281F8,__WRITE     ,__hw_dcp_chopts_bits);
__IO_REG32_BIT(HW_DCP_CH3OPTS_TOG,        0x800281FC,__WRITE     ,__hw_dcp_chopts_bits);
__IO_REG32_BIT(HW_DCP_DBGSELECT,          0x80028400,__READ_WRITE,__hw_dcp_dbgselect_bits);
__IO_REG32(    HW_DCP_DBGDATA,            0x80028410,__READ      );
__IO_REG32_BIT(HW_DCP_PAGETABLE,          0x80028420,__READ_WRITE,__hw_dcp_pagetable_bits);
__IO_REG32_BIT(HW_DCP_VERSION,            0x80028430,__READ      ,__hw_dcp_version_bits);

/***************************************************************************
 **
 **  DRAM
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_DRAM_CTL00,             0x800E0000,__READ_WRITE,__hw_dram_ctl00_bits);
__IO_REG32_BIT(HW_DRAM_CTL01,             0x800E0004,__READ_WRITE,__hw_dram_ctl01_bits);
__IO_REG32(    HW_DRAM_CTL02,             0x800E0008,__READ_WRITE);
__IO_REG32(    HW_DRAM_CTL03,             0x800E000C,__READ_WRITE);
__IO_REG32(    HW_DRAM_CTL04,             0x800E0010,__READ_WRITE);
__IO_REG32(    HW_DRAM_CTL05,             0x800E0014,__READ_WRITE);
__IO_REG32(    HW_DRAM_CTL06,             0x800E0018,__READ_WRITE);
__IO_REG32(    HW_DRAM_CTL07,             0x800E001C,__READ_WRITE);
__IO_REG32_BIT(HW_DRAM_CTL08,             0x800E0020,__READ_WRITE,__hw_dram_ctl08_bits);
__IO_REG32(    HW_DRAM_CTL09,             0x800E0024,__READ      );
__IO_REG32_BIT(HW_DRAM_CTL10,             0x800E0028,__READ      ,__hw_dram_ctl10_bits);
__IO_REG32_BIT(HW_DRAM_CTL11,             0x800E002C,__READ      ,__hw_dram_ctl11_bits);
__IO_REG32_BIT(HW_DRAM_CTL12,             0x800E0030,__READ      ,__hw_dram_ctl10_bits);
__IO_REG32_BIT(HW_DRAM_CTL13,             0x800E0034,__READ      ,__hw_dram_ctl11_bits);
__IO_REG32_BIT(HW_DRAM_CTL14,             0x800E0038,__READ      ,__hw_dram_ctl10_bits);
__IO_REG32_BIT(HW_DRAM_CTL15,             0x800E003C,__READ      ,__hw_dram_ctl11_bits);
__IO_REG32_BIT(HW_DRAM_CTL16,             0x800E0040,__READ_WRITE,__hw_dram_ctl16_bits);
__IO_REG32_BIT(HW_DRAM_CTL17,             0x800E0044,__READ_WRITE,__hw_dram_ctl17_bits);
__IO_REG32_BIT(HW_DRAM_CTL21,             0x800E0054,__READ_WRITE,__hw_dram_ctl21_bits);
__IO_REG32_BIT(HW_DRAM_CTL22,             0x800E0058,__READ_WRITE,__hw_dram_ctl22_bits);
__IO_REG32_BIT(HW_DRAM_CTL23,             0x800E005C,__READ_WRITE,__hw_dram_ctl23_bits);
__IO_REG32_BIT(HW_DRAM_CTL24,             0x800E0060,__READ_WRITE,__hw_dram_ctl24_bits);
__IO_REG32_BIT(HW_DRAM_CTL25,             0x800E0064,__READ_WRITE,__hw_dram_ctl25_bits);
__IO_REG32_BIT(HW_DRAM_CTL26,             0x800E0068,__READ_WRITE,__hw_dram_ctl26_bits);
__IO_REG32_BIT(HW_DRAM_CTL27,             0x800E006C,__READ_WRITE,__hw_dram_ctl27_bits);
__IO_REG32_BIT(HW_DRAM_CTL28,             0x800E0070,__READ_WRITE,__hw_dram_ctl28_bits);
__IO_REG32_BIT(HW_DRAM_CTL29,             0x800E0074,__READ_WRITE,__hw_dram_ctl29_bits);
__IO_REG32_BIT(HW_DRAM_CTL30,             0x800E0078,__READ      ,__hw_dram_ctl30_bits);
__IO_REG32_BIT(HW_DRAM_CTL31,             0x800E007C,__READ_WRITE,__hw_dram_ctl31_bits);
__IO_REG32_BIT(HW_DRAM_CTL32,             0x800E0080,__READ_WRITE,__hw_dram_ctl32_bits);
__IO_REG32_BIT(HW_DRAM_CTL33,             0x800E0084,__READ_WRITE,__hw_dram_ctl33_bits);
__IO_REG32_BIT(HW_DRAM_CTL34,             0x800E0088,__READ_WRITE,__hw_dram_ctl34_bits);
__IO_REG32_BIT(HW_DRAM_CTL35,             0x800E008C,__READ_WRITE,__hw_dram_ctl35_bits);
__IO_REG32_BIT(HW_DRAM_CTL36,             0x800E0090,__READ_WRITE,__hw_dram_ctl36_bits);
__IO_REG32_BIT(HW_DRAM_CTL37,             0x800E0094,__READ_WRITE,__hw_dram_ctl37_bits);
__IO_REG32_BIT(HW_DRAM_CTL38,             0x800E0098,__READ_WRITE,__hw_dram_ctl38_bits);
__IO_REG32_BIT(HW_DRAM_CTL39,             0x800E009C,__READ_WRITE,__hw_dram_ctl39_bits);
__IO_REG32_BIT(HW_DRAM_CTL40,             0x800E00A0,__READ_WRITE,__hw_dram_ctl40_bits);
__IO_REG32_BIT(HW_DRAM_CTL41,             0x800E00A4,__READ_WRITE,__hw_dram_ctl41_bits);
__IO_REG32_BIT(HW_DRAM_CTL42,             0x800E00A8,__READ_WRITE,__hw_dram_ctl42_bits);
__IO_REG32_BIT(HW_DRAM_CTL43,             0x800E00AC,__READ_WRITE,__hw_dram_ctl43_bits);
__IO_REG32_BIT(HW_DRAM_CTL44,             0x800E00B0,__READ_WRITE,__hw_dram_ctl44_bits);
__IO_REG32_BIT(HW_DRAM_CTL45,             0x800E00B4,__READ_WRITE,__hw_dram_ctl45_bits);
__IO_REG32_BIT(HW_DRAM_CTL48,             0x800E00C0,__READ_WRITE,__hw_dram_ctl48_bits);
__IO_REG32_BIT(HW_DRAM_CTL49,             0x800E00C4,__READ_WRITE,__hw_dram_ctl49_bits);
__IO_REG32_BIT(HW_DRAM_CTL50,             0x800E00C8,__READ_WRITE,__hw_dram_ctl50_bits);
__IO_REG32_BIT(HW_DRAM_CTL51,             0x800E00CC,__READ_WRITE,__hw_dram_ctl51_bits);
__IO_REG32_BIT(HW_DRAM_CTL52,             0x800E00D0,__READ_WRITE,__hw_dram_ctl52_bits);
__IO_REG32_BIT(HW_DRAM_CTL53,             0x800E00D4,__READ_WRITE,__hw_dram_ctl53_bits);
__IO_REG32_BIT(HW_DRAM_CTL54,             0x800E00D8,__READ_WRITE,__hw_dram_ctl54_bits);
__IO_REG32_BIT(HW_DRAM_CTL55,             0x800E00DC,__READ_WRITE,__hw_dram_ctl55_bits);
__IO_REG32_BIT(HW_DRAM_CTL56,             0x800E00E0,__READ_WRITE,__hw_dram_ctl56_bits);
__IO_REG32_BIT(HW_DRAM_CTL58,             0x800E00E8,__READ_WRITE,__hw_dram_ctl58_bits);
__IO_REG32(    HW_DRAM_CTL59,             0x800E00EC,__READ      );
__IO_REG32_BIT(HW_DRAM_CTL60,             0x800E00F0,__READ_WRITE,__hw_dram_ctl60_bits);
__IO_REG32_BIT(HW_DRAM_CTL61,             0x800E00F4,__READ_WRITE,__hw_dram_ctl61_bits);
__IO_REG32(    HW_DRAM_CTL62,             0x800E00F8,__READ      );
__IO_REG32_BIT(HW_DRAM_CTL63,             0x800E00FC,__READ      ,__hw_dram_ctl63_bits);
__IO_REG32_BIT(HW_DRAM_CTL64,             0x800E0100,__READ      ,__hw_dram_ctl64_bits);
__IO_REG32_BIT(HW_DRAM_CTL65,             0x800E0104,__READ      ,__hw_dram_ctl65_bits);
__IO_REG32_BIT(HW_DRAM_CTL66,             0x800E0108,__READ_WRITE,__hw_dram_ctl66_bits);
__IO_REG32_BIT(HW_DRAM_CTL67,             0x800E010C,__READ_WRITE,__hw_dram_ctl67_bits);
__IO_REG32_BIT(HW_DRAM_CTL68,             0x800E0110,__READ_WRITE,__hw_dram_ctl68_bits);
__IO_REG32_BIT(HW_DRAM_CTL69,             0x800E0114,__READ_WRITE,__hw_dram_ctl69_bits);
__IO_REG32_BIT(HW_DRAM_CTL70,             0x800E0118,__READ_WRITE,__hw_dram_ctl70_bits);
__IO_REG32(    HW_DRAM_CTL71,             0x800E011C,__READ_WRITE);
__IO_REG32(    HW_DRAM_CTL72,             0x800E0120,__READ_WRITE);
__IO_REG32(    HW_DRAM_CTL73,             0x800E0124,__READ_WRITE);
__IO_REG32(    HW_DRAM_CTL74,             0x800E0128,__READ_WRITE);
__IO_REG32(    HW_DRAM_CTL75,             0x800E012C,__READ_WRITE);
__IO_REG32(    HW_DRAM_CTL76,             0x800E0130,__READ_WRITE);
__IO_REG32(    HW_DRAM_CTL77,             0x800E0134,__READ_WRITE);
__IO_REG32(    HW_DRAM_CTL78,             0x800E0138,__READ_WRITE);
__IO_REG32(    HW_DRAM_CTL79,             0x800E013C,__READ_WRITE);
__IO_REG32(    HW_DRAM_CTL80,             0x800E0140,__READ_WRITE);
__IO_REG32_BIT(HW_DRAM_CTL81,             0x800E0144,__READ_WRITE,__hw_dram_ctl81_bits);
__IO_REG32_BIT(HW_DRAM_CTL82,             0x800E0148,__READ_WRITE,__hw_dram_ctl82_bits);
__IO_REG32_BIT(HW_DRAM_CTL83,             0x800E014C,__READ_WRITE,__hw_dram_ctl83_bits);
__IO_REG32_BIT(HW_DRAM_CTL84,             0x800E0150,__READ_WRITE,__hw_dram_ctl84_bits);
__IO_REG32(    HW_DRAM_CTL85,             0x800E0154,__READ_WRITE);
__IO_REG32_BIT(HW_DRAM_CTL86,             0x800E0158,__READ      ,__hw_dram_ctl86_bits);
__IO_REG32(    HW_DRAM_CTL87,             0x800E015C,__READ_WRITE);
__IO_REG32(    HW_DRAM_CTL88,             0x800E0160,__READ_WRITE);
__IO_REG32(    HW_DRAM_CTL89,             0x800E0164,__READ_WRITE);
__IO_REG32(    HW_DRAM_CTL90,             0x800E0168,__READ_WRITE);
__IO_REG32(    HW_DRAM_CTL91,             0x800E016C,__READ_WRITE);
__IO_REG32(    HW_DRAM_CTL92,             0x800E0170,__READ_WRITE);
__IO_REG32(    HW_DRAM_CTL93,             0x800E0174,__READ_WRITE);
__IO_REG32(    HW_DRAM_CTL94,             0x800E0178,__READ_WRITE);
__IO_REG32(    HW_DRAM_CTL95,             0x800E017C,__READ      );
__IO_REG32(    HW_DRAM_CTL96,             0x800E0180,__READ      );
__IO_REG32(    HW_DRAM_CTL97,             0x800E0184,__READ      );
__IO_REG32(    HW_DRAM_CTL98,             0x800E0188,__READ      );
__IO_REG32(    HW_DRAM_CTL99,             0x800E018C,__READ      );
__IO_REG32(    HW_DRAM_CTL100,            0x800E0190,__READ      );
__IO_REG32(    HW_DRAM_CTL101,            0x800E0194,__READ      );
__IO_REG32(    HW_DRAM_CTL102,            0x800E0198,__READ      );
__IO_REG32(    HW_DRAM_CTL103,            0x800E019C,__READ      );
__IO_REG32(    HW_DRAM_CTL104,            0x800E01A0,__READ      );
__IO_REG32(    HW_DRAM_CTL105,            0x800E01A4,__READ      );
__IO_REG32(    HW_DRAM_CTL106,            0x800E01A8,__READ      );
__IO_REG32_BIT(HW_DRAM_CTL107,            0x800E01AC,__READ      ,__hw_dram_ctl107_bits);
__IO_REG32(    HW_DRAM_CTL108,            0x800E01B0,__READ      );
__IO_REG32(    HW_DRAM_CTL109,            0x800E01B4,__READ      );
__IO_REG32(    HW_DRAM_CTL110,            0x800E01B8,__READ      );
__IO_REG32(    HW_DRAM_CTL111,            0x800E01BC,__READ      );
__IO_REG32_BIT(HW_DRAM_CTL112,            0x800E01C0,__READ      ,__hw_dram_ctl112_bits);
__IO_REG32(    HW_DRAM_CTL113,            0x800E01C4,__READ      );
__IO_REG32(    HW_DRAM_CTL114,            0x800E01C8,__READ      );
__IO_REG32(    HW_DRAM_CTL115,            0x800E01CC,__READ      );
__IO_REG32(    HW_DRAM_CTL116,            0x800E01D0,__READ      );
__IO_REG32_BIT(HW_DRAM_CTL117,            0x800E01D4,__READ      ,__hw_dram_ctl117_bits);
__IO_REG32(    HW_DRAM_CTL118,            0x800E01D8,__READ      );
__IO_REG32(    HW_DRAM_CTL119,            0x800E01DC,__READ      );
__IO_REG32(    HW_DRAM_CTL120,            0x800E01E0,__READ      );
__IO_REG32(    HW_DRAM_CTL121,            0x800E01E4,__READ      );
__IO_REG32_BIT(HW_DRAM_CTL122,            0x800E01E8,__READ      ,__hw_dram_ctl122_bits);
__IO_REG32(    HW_DRAM_CTL123,            0x800E01EC,__READ      );
__IO_REG32(    HW_DRAM_CTL124,            0x800E01F0,__READ      );
__IO_REG32(    HW_DRAM_CTL125,            0x800E01F4,__READ      );
__IO_REG32(    HW_DRAM_CTL126,            0x800E01F8,__READ      );
__IO_REG32_BIT(HW_DRAM_CTL127,            0x800E01FC,__READ      ,__hw_dram_ctl127_bits);
__IO_REG32(    HW_DRAM_CTL128,            0x800E0200,__READ      );
__IO_REG32(    HW_DRAM_CTL129,            0x800E0204,__READ      );
__IO_REG32(    HW_DRAM_CTL130,            0x800E0208,__READ      );
__IO_REG32(    HW_DRAM_CTL131,            0x800E020C,__READ      );
__IO_REG32_BIT(HW_DRAM_CTL132,            0x800E0210,__READ      ,__hw_dram_ctl132_bits);
__IO_REG32(    HW_DRAM_CTL133,            0x800E0214,__READ      );
__IO_REG32(    HW_DRAM_CTL134,            0x800E0218,__READ      );
__IO_REG32(    HW_DRAM_CTL135,            0x800E021C,__READ      );
__IO_REG32(    HW_DRAM_CTL136,            0x800E0220,__READ      );
__IO_REG32_BIT(HW_DRAM_CTL137,            0x800E0224,__READ      ,__hw_dram_ctl137_bits);
__IO_REG32(    HW_DRAM_CTL138,            0x800E0228,__READ      );
__IO_REG32(    HW_DRAM_CTL139,            0x800E022C,__READ      );
__IO_REG32(    HW_DRAM_CTL140,            0x800E0230,__READ      );
__IO_REG32(    HW_DRAM_CTL141,            0x800E0234,__READ      );
__IO_REG32_BIT(HW_DRAM_CTL142,            0x800E0238,__READ      ,__hw_dram_ctl142_bits);
__IO_REG32(    HW_DRAM_CTL143,            0x800E023C,__READ      );
__IO_REG32(    HW_DRAM_CTL144,            0x800E0240,__READ      );
__IO_REG32(    HW_DRAM_CTL145,            0x800E0244,__READ      );
__IO_REG32(    HW_DRAM_CTL146,            0x800E0248,__READ      );
__IO_REG32_BIT(HW_DRAM_CTL147,            0x800E024C,__READ      ,__hw_dram_ctl147_bits);
__IO_REG32(    HW_DRAM_CTL148,            0x800E0250,__READ      );
__IO_REG32(    HW_DRAM_CTL149,            0x800E0254,__READ      );
__IO_REG32(    HW_DRAM_CTL150,            0x800E0258,__READ      );
__IO_REG32(    HW_DRAM_CTL151,            0x800E025C,__READ      );
__IO_REG32_BIT(HW_DRAM_CTL152,            0x800E0260,__READ      ,__hw_dram_ctl152_bits);
__IO_REG32(    HW_DRAM_CTL153,            0x800E0264,__READ      );
__IO_REG32(    HW_DRAM_CTL154,            0x800E0268,__READ      );
__IO_REG32(    HW_DRAM_CTL155,            0x800E026C,__READ      );
__IO_REG32(    HW_DRAM_CTL156,            0x800E0270,__READ      );
__IO_REG32_BIT(HW_DRAM_CTL157,            0x800E0274,__READ      ,__hw_dram_ctl157_bits);
__IO_REG32(    HW_DRAM_CTL158,            0x800E0278,__READ      );
__IO_REG32(    HW_DRAM_CTL159,            0x800E027C,__READ      );
__IO_REG32(    HW_DRAM_CTL160,            0x800E0280,__READ      );
__IO_REG32(    HW_DRAM_CTL161,            0x800E0284,__READ      );
__IO_REG32_BIT(HW_DRAM_CTL162,            0x800E0288,__READ_WRITE,__hw_dram_ctl162_bits);
__IO_REG32_BIT(HW_DRAM_CTL163,            0x800E028C,__READ_WRITE,__hw_dram_ctl163_bits);
__IO_REG32_BIT(HW_DRAM_CTL164,            0x800E0290,__READ_WRITE,__hw_dram_ctl164_bits);
__IO_REG32_BIT(HW_DRAM_CTL171,            0x800E02AC,__READ_WRITE,__hw_dram_ctl171_bits);
__IO_REG32_BIT(HW_DRAM_CTL172,            0x800E02B0,__READ_WRITE,__hw_dram_ctl172_bits);
__IO_REG32_BIT(HW_DRAM_CTL173,            0x800E02B4,__READ_WRITE,__hw_dram_ctl173_bits);
__IO_REG32_BIT(HW_DRAM_CTL174,            0x800E02B8,__READ_WRITE,__hw_dram_ctl174_bits);
__IO_REG32_BIT(HW_DRAM_CTL175,            0x800E02BC,__READ_WRITE,__hw_dram_ctl175_bits);
__IO_REG32_BIT(HW_DRAM_CTL176,            0x800E02C0,__READ_WRITE,__hw_dram_ctl176_bits);
__IO_REG32_BIT(HW_DRAM_CTL177,            0x800E02C4,__READ_WRITE,__hw_dram_ctl177_bits);
__IO_REG32_BIT(HW_DRAM_CTL178,            0x800E02C8,__READ_WRITE,__hw_dram_ctl178_bits);
__IO_REG32_BIT(HW_DRAM_CTL179,            0x800E02CC,__READ_WRITE,__hw_dram_ctl179_bits);
__IO_REG32_BIT(HW_DRAM_CTL180,            0x800E02D0,__READ_WRITE,__hw_dram_ctl180_bits);
__IO_REG32_BIT(HW_DRAM_CTL181,            0x800E02D4,__READ_WRITE,__hw_dram_ctl181_bits);
__IO_REG32_BIT(HW_DRAM_CTL182,            0x800E02D8,__READ_WRITE,__hw_dram_ctl182_bits);
__IO_REG32_BIT(HW_DRAM_CTL183,            0x800E02DC,__READ_WRITE,__hw_dram_ctl183_bits);
__IO_REG32_BIT(HW_DRAM_CTL184,            0x800E02E0,__READ_WRITE,__hw_dram_ctl184_bits);
__IO_REG32_BIT(HW_DRAM_CTL185,            0x800E02E4,__READ_WRITE,__hw_dram_ctl185_bits);
__IO_REG32_BIT(HW_DRAM_CTL186,            0x800E02E8,__READ_WRITE,__hw_dram_ctl186_bits);
__IO_REG32_BIT(HW_DRAM_CTL187,            0x800E02EC,__READ_WRITE,__hw_dram_ctl187_bits);
__IO_REG32_BIT(HW_DRAM_CTL188,            0x800E02F0,__READ_WRITE,__hw_dram_ctl188_bits);
__IO_REG32_BIT(HW_DRAM_CTL189,            0x800E02F4,__READ_WRITE,__hw_dram_ctl189_bits);

/***************************************************************************
 **
 **  GPMI
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_GPMI_CTRL0,             0x8000C000,__READ_WRITE,__hw_gpmi_ctrl0_bits);
__IO_REG32_BIT(HW_GPMI_CTRL0_SET,         0x8000C004,__WRITE     ,__hw_gpmi_ctrl0_bits);
__IO_REG32_BIT(HW_GPMI_CTRL0_CLR,         0x8000C008,__WRITE     ,__hw_gpmi_ctrl0_bits);
__IO_REG32_BIT(HW_GPMI_CTRL0_TOG,         0x8000C00C,__WRITE     ,__hw_gpmi_ctrl0_bits);
__IO_REG32_BIT(HW_GPMI_COMPARE,           0x8000C010,__READ_WRITE,__hw_gpmi_compare_bits);
__IO_REG32_BIT(HW_GPMI_ECCCTRL,           0x8000C020,__READ_WRITE,__hw_gpmi_eccctrl_bits);
__IO_REG32_BIT(HW_GPMI_ECCCTRL_SET,       0x8000C024,__WRITE     ,__hw_gpmi_eccctrl_bits);
__IO_REG32_BIT(HW_GPMI_ECCCTRL_CLR,       0x8000C028,__WRITE     ,__hw_gpmi_eccctrl_bits);
__IO_REG32_BIT(HW_GPMI_ECCCTRL_TOG,       0x8000C02C,__WRITE     ,__hw_gpmi_eccctrl_bits);
__IO_REG32_BIT(HW_GPMI_ECCCOUNT,          0x8000C030,__READ_WRITE,__hw_gpmi_ecccount_bits);
__IO_REG32(    HW_GPMI_PAYLOAD,           0x8000C040,__READ_WRITE);
__IO_REG32(    HW_GPMI_AUXILIARY,         0x8000C050,__READ_WRITE);
__IO_REG32_BIT(HW_GPMI_CTRL1,             0x8000C060,__READ_WRITE,__hw_gpmi_ctrl1_bits);
__IO_REG32_BIT(HW_GPMI_CTRL1_SET,         0x8000C064,__WRITE     ,__hw_gpmi_ctrl1_bits);
__IO_REG32_BIT(HW_GPMI_CTRL1_CLR,         0x8000C068,__WRITE     ,__hw_gpmi_ctrl1_bits);
__IO_REG32_BIT(HW_GPMI_CTRL1_TOG,         0x8000C06C,__WRITE     ,__hw_gpmi_ctrl1_bits);
__IO_REG32_BIT(HW_GPMI_TIMING0,           0x8000C070,__READ_WRITE,__hw_gpmi_timing0_bits);
__IO_REG32_BIT(HW_GPMI_TIMING1,           0x8000C080,__READ_WRITE,__hw_gpmi_timing1_bits);
__IO_REG32(    HW_GPMI_DATA,              0x8000C0A0,__READ_WRITE);
__IO_REG32_BIT(HW_GPMI_STAT,              0x8000C0B0,__READ      ,__hw_gpmi_stat_bits);
__IO_REG32_BIT(HW_GPMI_DEBUG,             0x8000C0C0,__READ      ,__hw_gpmi_debug_bits);
__IO_REG32_BIT(HW_GPMI_VERSION,           0x8000C0D0,__READ      ,__hw_gpmi_version_bits);

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
 **  SSP0
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_SSP0_CTRL0,             0x80010000,__READ_WRITE,__hw_ssp_ctrl0_bits);
__IO_REG32_BIT(HW_SSP0_CTRL0_SET,         0x80010004,__WRITE     ,__hw_ssp_ctrl0_bits);
__IO_REG32_BIT(HW_SSP0_CTRL0_CLR,         0x80010008,__WRITE     ,__hw_ssp_ctrl0_bits);
__IO_REG32_BIT(HW_SSP0_CTRL0_TOG,         0x8001000C,__WRITE     ,__hw_ssp_ctrl0_bits);
__IO_REG32_BIT(HW_SSP0_CMD0,              0x80010010,__READ_WRITE,__hw_ssp_cmd0_bits);
__IO_REG32_BIT(HW_SSP0_CMD0_SET,          0x80010014,__WRITE     ,__hw_ssp_cmd0_bits);
__IO_REG32_BIT(HW_SSP0_CMD0_CLR,          0x80010018,__WRITE     ,__hw_ssp_cmd0_bits);
__IO_REG32_BIT(HW_SSP0_CMD0_TOG,          0x8001001C,__WRITE     ,__hw_ssp_cmd0_bits);
__IO_REG32(    HW_SSP0_CMD1,              0x80010020,__READ_WRITE);
__IO_REG32(    HW_SSP0_XFER_SIZE,         0x80010030,__READ_WRITE);
__IO_REG32_BIT(HW_SSP0_BLOCK_SIZE,        0x80010040,__READ_WRITE,__hw_ssp_block_size_bits);
__IO_REG32(    HW_SSP0_COMPREF,           0x80010050,__READ_WRITE);
__IO_REG32(    HW_SSP0_COMPMASK,          0x80010060,__READ_WRITE);
__IO_REG32_BIT(HW_SSP0_TIMING,            0x80010070,__READ_WRITE,__hw_ssp_timing_bits);
__IO_REG32_BIT(HW_SSP0_CTRL1,             0x80010080,__READ_WRITE,__hw_ssp_ctrl1_bits);
__IO_REG32_BIT(HW_SSP0_CTRL1_SET,         0x80010084,__WRITE     ,__hw_ssp_ctrl1_bits);
__IO_REG32_BIT(HW_SSP0_CTRL1_CLR,         0x80010088,__WRITE     ,__hw_ssp_ctrl1_bits);
__IO_REG32_BIT(HW_SSP0_CTRL1_TOG,         0x8001008C,__WRITE     ,__hw_ssp_ctrl1_bits);
__IO_REG32(    HW_SSP0_DATA,              0x80010090,__READ_WRITE);
__IO_REG32(    HW_SSP0_SDRESP0,           0x800100A0,__READ      );
__IO_REG32(    HW_SSP0_SDRESP1,           0x800100B0,__READ      );
__IO_REG32(    HW_SSP0_SDRESP2,           0x800100C0,__READ      );
__IO_REG32(    HW_SSP0_SDRESP3,           0x800100D0,__READ      );
__IO_REG32_BIT(HW_SSP0_DDR_CTRL,          0x800100E0,__READ_WRITE,__hw_ssp_ddr_ctrl_bits);
__IO_REG32_BIT(HW_SSP0_DLL_CTRL,          0x800100F0,__READ_WRITE,__hw_ssp_dll_ctrl_bits);
__IO_REG32_BIT(HW_SSP0_STATUS,            0x80010100,__READ      ,__hw_ssp_status_bits);
__IO_REG32_BIT(HW_SSP0_DLL_STS,           0x80010110,__READ      ,__hw_ssp_dll_sts_bits);
__IO_REG32_BIT(HW_SSP0_DEBUG,             0x80010120,__READ      ,__hw_ssp_debug_bits);
__IO_REG32_BIT(HW_SSP0_VERSION,           0x80010130,__READ      ,__hw_ssp_version_bits);

/***************************************************************************
 **
 **  SSP1
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_SSP1_CTRL0,             0x80012000,__READ_WRITE,__hw_ssp_ctrl0_bits);
__IO_REG32_BIT(HW_SSP1_CTRL0_SET,         0x80012004,__WRITE     ,__hw_ssp_ctrl0_bits);
__IO_REG32_BIT(HW_SSP1_CTRL0_CLR,         0x80012008,__WRITE     ,__hw_ssp_ctrl0_bits);
__IO_REG32_BIT(HW_SSP1_CTRL0_TOG,         0x8001200C,__WRITE     ,__hw_ssp_ctrl0_bits);
__IO_REG32_BIT(HW_SSP1_CMD0,              0x80012010,__READ_WRITE,__hw_ssp_cmd0_bits);
__IO_REG32_BIT(HW_SSP1_CMD0_SET,          0x80012014,__WRITE     ,__hw_ssp_cmd0_bits);
__IO_REG32_BIT(HW_SSP1_CMD0_CLR,          0x80012018,__WRITE     ,__hw_ssp_cmd0_bits);
__IO_REG32_BIT(HW_SSP1_CMD0_TOG,          0x8001201C,__WRITE     ,__hw_ssp_cmd0_bits);
__IO_REG32(    HW_SSP1_CMD1,              0x80012020,__READ_WRITE);
__IO_REG32(    HW_SSP1_XFER_SIZE,         0x80012030,__READ_WRITE);
__IO_REG32_BIT(HW_SSP1_BLOCK_SIZE,        0x80012040,__READ_WRITE,__hw_ssp_block_size_bits);
__IO_REG32(    HW_SSP1_COMPREF,           0x80012050,__READ_WRITE);
__IO_REG32(    HW_SSP1_COMPMASK,          0x80012060,__READ_WRITE);
__IO_REG32_BIT(HW_SSP1_TIMING,            0x80012070,__READ_WRITE,__hw_ssp_timing_bits);
__IO_REG32_BIT(HW_SSP1_CTRL1,             0x80012080,__READ_WRITE,__hw_ssp_ctrl1_bits);
__IO_REG32_BIT(HW_SSP1_CTRL1_SET,         0x80012084,__WRITE     ,__hw_ssp_ctrl1_bits);
__IO_REG32_BIT(HW_SSP1_CTRL1_CLR,         0x80012088,__WRITE     ,__hw_ssp_ctrl1_bits);
__IO_REG32_BIT(HW_SSP1_CTRL1_TOG,         0x8001208C,__WRITE     ,__hw_ssp_ctrl1_bits);
__IO_REG32(    HW_SSP1_DATA,              0x80012090,__READ_WRITE);
__IO_REG32(    HW_SSP1_SDRESP0,           0x800120A0,__READ      );
__IO_REG32(    HW_SSP1_SDRESP1,           0x800120B0,__READ      );
__IO_REG32(    HW_SSP1_SDRESP2,           0x800120C0,__READ      );
__IO_REG32(    HW_SSP1_SDRESP3,           0x800120D0,__READ      );
__IO_REG32_BIT(HW_SSP1_DDR_CTRL,          0x800120E0,__READ_WRITE,__hw_ssp_ddr_ctrl_bits);
__IO_REG32_BIT(HW_SSP1_DLL_CTRL,          0x800120F0,__READ_WRITE,__hw_ssp_dll_ctrl_bits);
__IO_REG32_BIT(HW_SSP1_STATUS,            0x80012100,__READ      ,__hw_ssp_status_bits);
__IO_REG32_BIT(HW_SSP1_DLL_STS,           0x80012110,__READ      ,__hw_ssp_dll_sts_bits);
__IO_REG32_BIT(HW_SSP1_DEBUG,             0x80012120,__READ      ,__hw_ssp_debug_bits);
__IO_REG32_BIT(HW_SSP1_VERSION,           0x80012130,__READ      ,__hw_ssp_version_bits);

/***************************************************************************
 **
 **  SSP2
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_SSP2_CTRL0,             0x80014000,__READ_WRITE,__hw_ssp_ctrl0_bits);
__IO_REG32_BIT(HW_SSP2_CTRL0_SET,         0x80014004,__WRITE     ,__hw_ssp_ctrl0_bits);
__IO_REG32_BIT(HW_SSP2_CTRL0_CLR,         0x80014008,__WRITE     ,__hw_ssp_ctrl0_bits);
__IO_REG32_BIT(HW_SSP2_CTRL0_TOG,         0x8001400C,__WRITE     ,__hw_ssp_ctrl0_bits);
__IO_REG32_BIT(HW_SSP2_CMD0,              0x80014010,__READ_WRITE,__hw_ssp_cmd0_bits);
__IO_REG32_BIT(HW_SSP2_CMD0_SET,          0x80014014,__WRITE     ,__hw_ssp_cmd0_bits);
__IO_REG32_BIT(HW_SSP2_CMD0_CLR,          0x80014018,__WRITE     ,__hw_ssp_cmd0_bits);
__IO_REG32_BIT(HW_SSP2_CMD0_TOG,          0x8001401C,__WRITE     ,__hw_ssp_cmd0_bits);
__IO_REG32(    HW_SSP2_CMD1,              0x80014020,__READ_WRITE);
__IO_REG32(    HW_SSP2_XFER_SIZE,         0x80014030,__READ_WRITE);
__IO_REG32_BIT(HW_SSP2_BLOCK_SIZE,        0x80014040,__READ_WRITE,__hw_ssp_block_size_bits);
__IO_REG32(    HW_SSP2_COMPREF,           0x80014050,__READ_WRITE);
__IO_REG32(    HW_SSP2_COMPMASK,          0x80014060,__READ_WRITE);
__IO_REG32_BIT(HW_SSP2_TIMING,            0x80014070,__READ_WRITE,__hw_ssp_timing_bits);
__IO_REG32_BIT(HW_SSP2_CTRL1,             0x80014080,__READ_WRITE,__hw_ssp_ctrl1_bits);
__IO_REG32_BIT(HW_SSP2_CTRL1_SET,         0x80014084,__WRITE     ,__hw_ssp_ctrl1_bits);
__IO_REG32_BIT(HW_SSP2_CTRL1_CLR,         0x80014088,__WRITE     ,__hw_ssp_ctrl1_bits);
__IO_REG32_BIT(HW_SSP2_CTRL1_TOG,         0x8001408C,__WRITE     ,__hw_ssp_ctrl1_bits);
__IO_REG32(    HW_SSP2_DATA,              0x80014090,__READ_WRITE);
__IO_REG32(    HW_SSP2_SDRESP0,           0x800140A0,__READ      );
__IO_REG32(    HW_SSP2_SDRESP1,           0x800140B0,__READ      );
__IO_REG32(    HW_SSP2_SDRESP2,           0x800140C0,__READ      );
__IO_REG32(    HW_SSP2_SDRESP3,           0x800140D0,__READ      );
__IO_REG32_BIT(HW_SSP2_DDR_CTRL,          0x800140E0,__READ_WRITE,__hw_ssp_ddr_ctrl_bits);
__IO_REG32_BIT(HW_SSP2_DLL_CTRL,          0x800140F0,__READ_WRITE,__hw_ssp_dll_ctrl_bits);
__IO_REG32_BIT(HW_SSP2_STATUS,            0x80014100,__READ      ,__hw_ssp_status_bits);
__IO_REG32_BIT(HW_SSP2_DLL_STS,           0x80014110,__READ      ,__hw_ssp_dll_sts_bits);
__IO_REG32_BIT(HW_SSP2_DEBUG,             0x80014120,__READ      ,__hw_ssp_debug_bits);
__IO_REG32_BIT(HW_SSP2_VERSION,           0x80014130,__READ      ,__hw_ssp_version_bits);

/***************************************************************************
 **
 **  SSP3
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_SSP3_CTRL0,             0x80016000,__READ_WRITE,__hw_ssp_ctrl0_bits);
__IO_REG32_BIT(HW_SSP3_CTRL0_SET,         0x80016004,__WRITE     ,__hw_ssp_ctrl0_bits);
__IO_REG32_BIT(HW_SSP3_CTRL0_CLR,         0x80016008,__WRITE     ,__hw_ssp_ctrl0_bits);
__IO_REG32_BIT(HW_SSP3_CTRL0_TOG,         0x8001600C,__WRITE     ,__hw_ssp_ctrl0_bits);
__IO_REG32_BIT(HW_SSP3_CMD0,              0x80016010,__READ_WRITE,__hw_ssp_cmd0_bits);
__IO_REG32_BIT(HW_SSP3_CMD0_SET,          0x80016014,__WRITE     ,__hw_ssp_cmd0_bits);
__IO_REG32_BIT(HW_SSP3_CMD0_CLR,          0x80016018,__WRITE     ,__hw_ssp_cmd0_bits);
__IO_REG32_BIT(HW_SSP3_CMD0_TOG,          0x8001601C,__WRITE     ,__hw_ssp_cmd0_bits);
__IO_REG32(    HW_SSP3_CMD1,              0x80016020,__READ_WRITE);
__IO_REG32(    HW_SSP3_XFER_SIZE,         0x80016030,__READ_WRITE);
__IO_REG32_BIT(HW_SSP3_BLOCK_SIZE,        0x80016040,__READ_WRITE,__hw_ssp_block_size_bits);
__IO_REG32(    HW_SSP3_COMPREF,           0x80016050,__READ_WRITE);
__IO_REG32(    HW_SSP3_COMPMASK,          0x80016060,__READ_WRITE);
__IO_REG32_BIT(HW_SSP3_TIMING,            0x80016070,__READ_WRITE,__hw_ssp_timing_bits);
__IO_REG32_BIT(HW_SSP3_CTRL1,             0x80016080,__READ_WRITE,__hw_ssp_ctrl1_bits);
__IO_REG32_BIT(HW_SSP3_CTRL1_SET,         0x80016084,__WRITE     ,__hw_ssp_ctrl1_bits);
__IO_REG32_BIT(HW_SSP3_CTRL1_CLR,         0x80016088,__WRITE     ,__hw_ssp_ctrl1_bits);
__IO_REG32_BIT(HW_SSP3_CTRL1_TOG,         0x8001608C,__WRITE     ,__hw_ssp_ctrl1_bits);
__IO_REG32(    HW_SSP3_DATA,              0x80016090,__READ_WRITE);
__IO_REG32(    HW_SSP3_SDRESP0,           0x800160A0,__READ      );
__IO_REG32(    HW_SSP3_SDRESP1,           0x800160B0,__READ      );
__IO_REG32(    HW_SSP3_SDRESP2,           0x800160C0,__READ      );
__IO_REG32(    HW_SSP3_SDRESP3,           0x800160D0,__READ      );
__IO_REG32_BIT(HW_SSP3_DDR_CTRL,          0x800160E0,__READ_WRITE,__hw_ssp_ddr_ctrl_bits);
__IO_REG32_BIT(HW_SSP3_DLL_CTRL,          0x800160F0,__READ_WRITE,__hw_ssp_dll_ctrl_bits);
__IO_REG32_BIT(HW_SSP3_STATUS,            0x80016100,__READ      ,__hw_ssp_status_bits);
__IO_REG32_BIT(HW_SSP3_DLL_STS,           0x80016110,__READ      ,__hw_ssp_dll_sts_bits);
__IO_REG32_BIT(HW_SSP3_DEBUG,             0x80016120,__READ      ,__hw_ssp_debug_bits);
__IO_REG32_BIT(HW_SSP3_VERSION,           0x80016130,__READ      ,__hw_ssp_version_bits);

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
__IO_REG32_BIT(HW_DIGCTL_EMI_STATUS,              0x8001C040,__READ       ,__hw_digctl_emi_status_bits);
__IO_REG32_BIT(HW_DIGCTL_READ_MARGIN,             0x8001C050,__READ_WRITE ,__hw_digctl_read_margin_bits);
__IO_REG32_BIT(HW_DIGCTL_READ_MARGIN_SET,         0x8001C054,__WRITE      ,__hw_digctl_read_margin_bits);
__IO_REG32_BIT(HW_DIGCTL_READ_MARGIN_CLR,         0x8001C058,__WRITE      ,__hw_digctl_read_margin_bits);
__IO_REG32_BIT(HW_DIGCTL_READ_MARGIN_TOG,         0x8001C05C,__WRITE      ,__hw_digctl_read_margin_bits);
__IO_REG32(    HW_DIGCTL_WRITEONCE,               0x8001C060,__READ_WRITE );
__IO_REG32_BIT(HW_DIGCTL_BIST_CTL,                0x8001C070,__READ_WRITE ,__hw_digctl_bist_ctl_bits);
__IO_REG32_BIT(HW_DIGCTL_BIST_CTL_SET,            0x8001C074,__WRITE      ,__hw_digctl_bist_ctl_bits);
__IO_REG32_BIT(HW_DIGCTL_BIST_CTL_CLR,            0x8001C078,__WRITE      ,__hw_digctl_bist_ctl_bits);
__IO_REG32_BIT(HW_DIGCTL_BIST_CTL_TOG,            0x8001C07C,__WRITE      ,__hw_digctl_bist_ctl_bits);
__IO_REG32_BIT(HW_DIGCTL_BIST_STATUS,             0x8001C080,__READ       ,__hw_digctl_bist_status_bits);
__IO_REG32(    HW_DIGCTL_ENTROPY,                 0x8001C090,__READ       );
__IO_REG32(    HW_DIGCTL_ENTROPY_LATCHED,         0x8001C0A0,__READ       );
__IO_REG32(    HW_DIGCTL_MICROSECONDS,            0x8001C0C0,__READ       );
__IO_REG32(    HW_DIGCTL_DBGRD,                   0x8001C0D0,__READ       );
__IO_REG32(    HW_DIGCTL_DBG,                     0x8001C0E0,__READ       );
__IO_REG32_BIT(HW_DIGCTL_USB_LOOPBACK,            0x8001C100,__READ_WRITE ,__hw_digctl_usb_loopback_bits);
__IO_REG32_BIT(HW_DIGCTL_USB_LOOPBACK_SET,        0x8001C104,__WRITE      ,__hw_digctl_usb_loopback_bits);
__IO_REG32_BIT(HW_DIGCTL_USB_LOOPBACK_CLR,        0x8001C108,__WRITE      ,__hw_digctl_usb_loopback_bits);
__IO_REG32_BIT(HW_DIGCTL_USB_LOOPBACK_TOG,        0x8001C10C,__WRITE      ,__hw_digctl_usb_loopback_bits);
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
__IO_REG32(    HW_DIGCTL_SCRATCH0,                0x8001C280,__READ_WRITE );
__IO_REG32(    HW_DIGCTL_SCRATCH1,                0x8001C290,__READ_WRITE );
__IO_REG32_BIT(HW_DIGCTL_ARMCACHE,                0x8001C2A0,__READ_WRITE ,__hw_digctl_armcache_bits);
__IO_REG32_BIT(HW_DIGCTL_DEBUG_TRAP,              0x8001C2B0,__READ_WRITE ,__hw_digctl_debug_trap_bits);
__IO_REG32_BIT(HW_DIGCTL_DEBUG_TRAP_SET,          0x8001C2B4,__WRITE      ,__hw_digctl_debug_trap_bits);
__IO_REG32_BIT(HW_DIGCTL_DEBUG_TRAP_CLR,          0x8001C2B8,__WRITE      ,__hw_digctl_debug_trap_bits);
__IO_REG32_BIT(HW_DIGCTL_DEBUG_TRAP_TOG,          0x8001C2BC,__WRITE      ,__hw_digctl_debug_trap_bits);
__IO_REG32(    HW_DIGCTL_DEBUG_TRAP_L0_ADDR_LOW,  0x8001C2C0,__READ_WRITE );
__IO_REG32(    HW_DIGCTL_DEBUG_TRAP_L0_ADDR_HIGH, 0x8001C2D0,__READ_WRITE );
__IO_REG32(    HW_DIGCTL_DEBUG_TRAP_L3_ADDR_LOW,  0x8001C2E0,__READ_WRITE );
__IO_REG32(    HW_DIGCTL_DEBUG_TRAP_L3_ADDR_HIGH, 0x8001C2F0,__READ_WRITE );
__IO_REG32(    HW_DIGCTL_FSL,                     0x8001C300,__READ       );
__IO_REG32_BIT(HW_DIGCTL_CHIPID,                  0x8001C310,__READ       ,__hw_digctl_chipid_bits);
__IO_REG32_BIT(HW_DIGCTL_AHB_STATS_SELECT,        0x8001C330,__READ_WRITE ,__hw_digctl_ahb_stats_select_bits);
__IO_REG32(    HW_DIGCTL_L1_AHB_ACTIVE_CYCLES,    0x8001C370,__READ_WRITE );
__IO_REG32(    HW_DIGCTL_L1_AHB_DATA_STALLED,     0x8001C380,__READ_WRITE );
__IO_REG32(    HW_DIGCTL_L1_AHB_DATA_CYCLES,      0x8001C390,__READ_WRITE );
__IO_REG32(    HW_DIGCTL_L2_AHB_ACTIVE_CYCLES,    0x8001C3A0,__READ_WRITE );
__IO_REG32(    HW_DIGCTL_L2_AHB_DATA_STALLED,     0x8001C3B0,__READ_WRITE );
__IO_REG32(    HW_DIGCTL_L2_AHB_DATA_CYCLES,      0x8001C3C0,__READ_WRITE );
__IO_REG32(    HW_DIGCTL_L3_AHB_ACTIVE_CYCLES,    0x8001C3D0,__READ_WRITE );
__IO_REG32(    HW_DIGCTL_L3_AHB_DATA_STALLED,     0x8001C3E0,__READ_WRITE );
__IO_REG32(    HW_DIGCTL_L3_AHB_DATA_CYCLES,      0x8001C3F0,__READ_WRITE );
__IO_REG32_BIT(HW_DIGCTL_MPTE0_LOC,               0x8001C500,__READ_WRITE ,__hw_digctl_mpte_loc_bits);
__IO_REG32_BIT(HW_DIGCTL_MPTE1_LOC,               0x8001C510,__READ_WRITE ,__hw_digctl_mpte_loc_bits);
__IO_REG32_BIT(HW_DIGCTL_MPTE2_LOC,               0x8001C520,__READ_WRITE ,__hw_digctl_mpte_loc_bits);
__IO_REG32_BIT(HW_DIGCTL_MPTE3_LOC,               0x8001C530,__READ_WRITE ,__hw_digctl_mpte_loc_bits);
__IO_REG32_BIT(HW_DIGCTL_MPTE4_LOC,               0x8001C540,__READ_WRITE ,__hw_digctl_mpte_loc_bits);
__IO_REG32_BIT(HW_DIGCTL_MPTE5_LOC,               0x8001C550,__READ_WRITE ,__hw_digctl_mpte_loc_bits);
__IO_REG32_BIT(HW_DIGCTL_MPTE6_LOC,               0x8001C560,__READ_WRITE ,__hw_digctl_mpte_loc_bits);
__IO_REG32_BIT(HW_DIGCTL_MPTE7_LOC,               0x8001C570,__READ_WRITE ,__hw_digctl_mpte_loc_bits);
__IO_REG32_BIT(HW_DIGCTL_MPTE8_LOC,               0x8001C580,__READ_WRITE ,__hw_digctl_mpte_loc_bits);
__IO_REG32_BIT(HW_DIGCTL_MPTE9_LOC,               0x8001C590,__READ_WRITE ,__hw_digctl_mpte_loc_bits);
__IO_REG32_BIT(HW_DIGCTL_MPTE10_LOC,              0x8001C5A0,__READ_WRITE ,__hw_digctl_mpte_loc_bits);
__IO_REG32_BIT(HW_DIGCTL_MPTE11_LOC,              0x8001C5B0,__READ_WRITE ,__hw_digctl_mpte_loc_bits);
__IO_REG32_BIT(HW_DIGCTL_MPTE12_LOC,              0x8001C5C0,__READ_WRITE ,__hw_digctl_mpte_loc_bits);
__IO_REG32_BIT(HW_DIGCTL_MPTE13_LOC,              0x8001C5D0,__READ_WRITE ,__hw_digctl_mpte_loc_bits);
__IO_REG32_BIT(HW_DIGCTL_MPTE14_LOC,              0x8001C5E0,__READ_WRITE ,__hw_digctl_mpte_loc_bits);
__IO_REG32_BIT(HW_DIGCTL_MPTE15_LOC,              0x8001C5F0,__READ_WRITE ,__hw_digctl_mpte_loc_bits);

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
__IO_REG32_BIT(HW_OCOTP_ROM3,                     0x8002C1D0,__READ_WRITE ,__hw_ocotp_rom3_bits);
__IO_REG32_BIT(HW_OCOTP_ROM4,                     0x8002C1E0,__READ_WRITE ,__hw_ocotp_rom4_bits);
__IO_REG32(    HW_OCOTP_ROM5,                     0x8002C1F0,__READ_WRITE );
__IO_REG32(    HW_OCOTP_ROM6,                     0x8002C200,__READ_WRITE );
__IO_REG32_BIT(HW_OCOTP_ROM7,                     0x8002C210,__READ_WRITE ,__hw_ocotp_rom7_bits);
__IO_REG32(    HW_OCOTP_SRK0,                     0x8002C220,__READ_WRITE );
__IO_REG32(    HW_OCOTP_SRK1,                     0x8002C230,__READ_WRITE );
__IO_REG32(    HW_OCOTP_SRK2,                     0x8002C240,__READ_WRITE );
__IO_REG32(    HW_OCOTP_SRK3,                     0x8002C250,__READ_WRITE );
__IO_REG32(    HW_OCOTP_SRK4,                     0x8002C260,__READ_WRITE );
__IO_REG32(    HW_OCOTP_SRK5,                     0x8002C270,__READ_WRITE );
__IO_REG32(    HW_OCOTP_SRK6,                     0x8002C280,__READ_WRITE );
__IO_REG32(    HW_OCOTP_SRK7,                     0x8002C290,__READ_WRITE );
__IO_REG32_BIT(HW_OCOTP_VERSION,                  0x8002C2A0,__READ       ,__hw_ocotp_version_bits);

/***************************************************************************
 **
 **  PERFMON
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_PERFMON_CTRL,                   0x80006000,__READ_WRITE ,__hw_perfmon_ctrl_bits);
__IO_REG32_BIT(HW_PERFMON_CTRL_SET,               0x80006004,__WRITE      ,__hw_perfmon_ctrl_bits);
__IO_REG32_BIT(HW_PERFMON_CTRL_CLR,               0x80006008,__WRITE      ,__hw_perfmon_ctrl_bits);
__IO_REG32_BIT(HW_PERFMON_CTRL_TOG,               0x8000600C,__WRITE      ,__hw_perfmon_ctrl_bits);
__IO_REG32_BIT(HW_PERFMON_MASTER_EN,              0x80006010,__READ_WRITE ,__hw_perfmon_master_en_bits);
__IO_REG32(    HW_PERFMON_TRAP_ADDR_LOW,          0x80006020,__READ_WRITE );
__IO_REG32(    HW_PERFMON_TRAP_ADDR_HIGH,         0x80006030,__READ_WRITE );
__IO_REG32_BIT(HW_PERFMON_LAT_THRESHOLD,          0x80006040,__READ_WRITE ,__hw_perfmon_lat_threshold_bits);
__IO_REG32(    HW_PERFMON_ACTIVE_CYCLE,           0x80006050,__READ       );
__IO_REG32(    HW_PERFMON_TRANSFER_COUNT,         0x80006060,__READ       );
__IO_REG32(    HW_PERFMON_TOTAL_LATENCY,          0x80006070,__READ       );
__IO_REG32(    HW_PERFMON_DATA_COUNT,             0x80006080,__READ       );
__IO_REG32_BIT(HW_PERFMON_MAX_LATENCY,            0x80006090,__READ       ,__hw_perfmon_max_latency_bits);
__IO_REG32_BIT(HW_PERFMON_DEBUG,                  0x800060A0,__READ_WRITE ,__hw_perfmon_debug_bits);
__IO_REG32_BIT(HW_PERFMON_VERSION,                0x800060B0,__READ       ,__hw_perfmon_version_bits);

/***************************************************************************
 **
 **  RTC
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_RTC_CTRL,                       0x80056000,__READ_WRITE ,__hw_rtc_ctrl_bits);
__IO_REG32_BIT(HW_RTC_CTRL_SET,                   0x80056004,__WRITE      ,__hw_rtc_ctrl_bits);
__IO_REG32_BIT(HW_RTC_CTRL_CLR,                   0x80056008,__WRITE      ,__hw_rtc_ctrl_bits);
__IO_REG32_BIT(HW_RTC_CTRL_TOG,                   0x8005600C,__WRITE      ,__hw_rtc_ctrl_bits);
__IO_REG32_BIT(HW_RTC_STAT,                       0x80056010,__READ       ,__hw_rtc_stat_bits);
__IO_REG32(    HW_RTC_MILLISECONDS,               0x80056020,__READ_WRITE );
__IO_REG32(    HW_RTC_SECONDS,                    0x80056030,__READ_WRITE );
__IO_REG32(    HW_RTC_SECONDS_SET,                0x80056034,__WRITE      );
__IO_REG32(    HW_RTC_SECONDS_CLR,                0x80056038,__WRITE      );
__IO_REG32(    HW_RTC_SECONDS_TOG,                0x8005603C,__WRITE      );
__IO_REG32(    HW_RTC_ALARM,                      0x80056040,__READ_WRITE );
__IO_REG32(    HW_RTC_ALARM_SET,                  0x80056044,__WRITE      );
__IO_REG32(    HW_RTC_ALARM_CLR,                  0x80056048,__WRITE      );
__IO_REG32(    HW_RTC_ALARM_TOG,                  0x8005604C,__WRITE      );
__IO_REG32(    HW_RTC_WATCHDOG,                   0x80056050,__READ_WRITE );
__IO_REG32(    HW_RTC_WATCHDOG_SET,               0x80056054,__WRITE      );
__IO_REG32(    HW_RTC_WATCHDOG_CLR,               0x80056058,__WRITE      );
__IO_REG32(    HW_RTC_WATCHDOG_TOG,               0x8005605C,__WRITE      );
__IO_REG32_BIT(HW_RTC_PERSISTENT0,                0x80056060,__READ_WRITE ,__hw_rtc_persistent0_bits);
__IO_REG32_BIT(HW_RTC_PERSISTENT0_SET,            0x80056064,__WRITE      ,__hw_rtc_persistent0_bits);
__IO_REG32_BIT(HW_RTC_PERSISTENT0_CLR,            0x80056068,__WRITE      ,__hw_rtc_persistent0_bits);
__IO_REG32_BIT(HW_RTC_PERSISTENT0_TOG,            0x8005606C,__WRITE      ,__hw_rtc_persistent0_bits);
__IO_REG32(    HW_RTC_PERSISTENT1,                0x80056070,__READ_WRITE );
__IO_REG32(    HW_RTC_PERSISTENT1_SET,            0x80056074,__WRITE      );
__IO_REG32(    HW_RTC_PERSISTENT1_CLR,            0x80056078,__WRITE      );
__IO_REG32(    HW_RTC_PERSISTENT1_TOG,            0x8005607C,__WRITE      );
__IO_REG32(    HW_RTC_PERSISTENT2,                0x80056080,__READ_WRITE );
__IO_REG32(    HW_RTC_PERSISTENT2_SET,            0x80056084,__WRITE      );
__IO_REG32(    HW_RTC_PERSISTENT2_CLR,            0x80056088,__WRITE      );
__IO_REG32(    HW_RTC_PERSISTENT2_TOG,            0x8005608C,__WRITE      );
__IO_REG32(    HW_RTC_PERSISTENT3,                0x80056090,__READ_WRITE );
__IO_REG32(    HW_RTC_PERSISTENT3_SET,            0x80056094,__WRITE      );
__IO_REG32(    HW_RTC_PERSISTENT3_CLR,            0x80056098,__WRITE      );
__IO_REG32(    HW_RTC_PERSISTENT3_TOG,            0x8005609C,__WRITE      );
__IO_REG32(    HW_RTC_PERSISTENT4,                0x800560A0,__READ_WRITE );
__IO_REG32(    HW_RTC_PERSISTENT4_SET,            0x800560A4,__WRITE      );
__IO_REG32(    HW_RTC_PERSISTENT4_CLR,            0x800560A8,__WRITE      );
__IO_REG32(    HW_RTC_PERSISTENT4_TOG,            0x800560AC,__WRITE      );
__IO_REG32(    HW_RTC_PERSISTENT5,                0x800560B0,__READ_WRITE );
__IO_REG32(    HW_RTC_PERSISTENT5_SET,            0x800560B4,__WRITE      );
__IO_REG32(    HW_RTC_PERSISTENT5_CLR,            0x800560B8,__WRITE      );
__IO_REG32(    HW_RTC_PERSISTENT5_TOG,            0x800560BC,__WRITE      );
__IO_REG32_BIT(HW_RTC_DEBUG,                      0x800560C0,__READ_WRITE ,__hw_rtc_debug_bits);
__IO_REG32_BIT(HW_RTC_VERSION,                    0x800560D0,__READ       ,__hw_rtc_version_bits);

/***************************************************************************
 **
 **  TIMROT
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_TIMROT_ROTCTRL,                 0x80068000,__READ_WRITE ,__hw_timrot_rotctrl_bits);
__IO_REG32_BIT(HW_TIMROT_ROTCTRL_SET,             0x80068004,__WRITE      ,__hw_timrot_rotctrl_bits);
__IO_REG32_BIT(HW_TIMROT_ROTCTRL_CLR,             0x80068008,__WRITE      ,__hw_timrot_rotctrl_bits);
__IO_REG32_BIT(HW_TIMROT_ROTCTRL_TOG,             0x8006800C,__WRITE      ,__hw_timrot_rotctrl_bits);
__IO_REG32_BIT(HW_TIMROT_ROTCOUNT,                0x80068010,__READ       ,__hw_timrot_rotcount_bits);
__IO_REG32_BIT(HW_TIMROT_TIMCTRL0,                0x80068020,__READ_WRITE ,__hw_timrot_timctrl_bits);
__IO_REG32_BIT(HW_TIMROT_TIMCTRL0_SET,            0x80068024,__WRITE      ,__hw_timrot_timctrl_bits);
__IO_REG32_BIT(HW_TIMROT_TIMCTRL0_CLR,            0x80068028,__WRITE      ,__hw_timrot_timctrl_bits);
__IO_REG32_BIT(HW_TIMROT_TIMCTRL0_TOG,            0x8006802C,__WRITE      ,__hw_timrot_timctrl_bits);
__IO_REG32(    HW_TIMROT_RUNNING_COUNT0,          0x80068030,__READ       );
__IO_REG32(    HW_TIMROT_FIXED_COUNT0,            0x80068040,__READ_WRITE );
__IO_REG32(    HW_TIMROT_MATCH_COUNT0,            0x80068050,__READ_WRITE );
__IO_REG32_BIT(HW_TIMROT_TIMCTRL1,                0x80068060,__READ_WRITE ,__hw_timrot_timctrl_bits);
__IO_REG32_BIT(HW_TIMROT_TIMCTRL1_SET,            0x80068064,__WRITE      ,__hw_timrot_timctrl_bits);
__IO_REG32_BIT(HW_TIMROT_TIMCTRL1_CLR,            0x80068068,__WRITE      ,__hw_timrot_timctrl_bits);
__IO_REG32_BIT(HW_TIMROT_TIMCTRL1_TOG,            0x8006806C,__WRITE      ,__hw_timrot_timctrl_bits);
__IO_REG32(    HW_TIMROT_RUNNING_COUNT1,          0x80068070,__READ       );
__IO_REG32(    HW_TIMROT_FIXED_COUNT1,            0x80068080,__READ_WRITE );
__IO_REG32(    HW_TIMROT_MATCH_COUNT1,            0x80068090,__READ_WRITE );
__IO_REG32_BIT(HW_TIMROT_TIMCTRL2,                0x800680A0,__READ_WRITE ,__hw_timrot_timctrl_bits);
__IO_REG32_BIT(HW_TIMROT_TIMCTRL2_SET,            0x800680A4,__WRITE      ,__hw_timrot_timctrl_bits);
__IO_REG32_BIT(HW_TIMROT_TIMCTRL2_CLR,            0x800680A8,__WRITE      ,__hw_timrot_timctrl_bits);
__IO_REG32_BIT(HW_TIMROT_TIMCTRL2_TOG,            0x800680AC,__WRITE      ,__hw_timrot_timctrl_bits);
__IO_REG32(    HW_TIMROT_RUNNING_COUNT2,          0x800680B0,__READ       );
__IO_REG32(    HW_TIMROT_FIXED_COUNT2,            0x800680C0,__READ_WRITE );
__IO_REG32(    HW_TIMROT_MATCH_COUNT2,            0x800680D0,__READ_WRITE );
__IO_REG32_BIT(HW_TIMROT_TIMCTRL3,                0x800680E0,__READ_WRITE ,__hw_timrot_timctrl_bits);
__IO_REG32_BIT(HW_TIMROT_TIMCTRL3_SET,            0x800680E4,__WRITE      ,__hw_timrot_timctrl_bits);
__IO_REG32_BIT(HW_TIMROT_TIMCTRL3_CLR,            0x800680E8,__WRITE      ,__hw_timrot_timctrl_bits);
__IO_REG32_BIT(HW_TIMROT_TIMCTRL3_TOG,            0x800680EC,__WRITE      ,__hw_timrot_timctrl_bits);
__IO_REG32(    HW_TIMROT_RUNNING_COUNT4,          0x800680F0,__READ       );
__IO_REG32(    HW_TIMROT_FIXED_COUNT4,            0x80068100,__READ_WRITE );
__IO_REG32(    HW_TIMROT_MATCH_COUNT4,            0x80068110,__READ_WRITE );
__IO_REG32_BIT(HW_TIMROT_VERSION,                 0x80068120,__READ       ,__hw_timrot_version_bits);

/***************************************************************************
 **
 **  DUART
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_UARTDBG_DR,                     0x80074000,__READ_WRITE ,__hw_uartdbg_dr_bits);
__IO_REG32_BIT(HW_UARTDBG_ECR,                    0x80074004,__READ_WRITE ,__hw_uartdbg_ecr_bits);
__IO_REG32_BIT(HW_UARTDBG_FR,                     0x80074018,__READ       ,__hw_uartdbg_fr_bits);
__IO_REG32_BIT(HW_UARTDBG_ILPR,                   0x80074020,__READ_WRITE ,__hw_uartdbg_ilpr_bits);
__IO_REG32_BIT(HW_UARTDBG_IBRD,                   0x80074024,__READ_WRITE ,__hw_uartdbg_ibrd_bits);
__IO_REG32_BIT(HW_UARTDBG_FBRD,                   0x80074028,__READ_WRITE ,__hw_uartdbg_fbrd_bits);
__IO_REG32_BIT(HW_UARTDBG_H,                      0x8007402C,__READ_WRITE ,__hw_uartdbg_h_bits);
__IO_REG32_BIT(HW_UARTDBG_CR,                     0x80074030,__READ_WRITE ,__hw_uartdbg_cr_bits);
__IO_REG32_BIT(HW_UARTDBG_IFLS,                   0x80074034,__READ_WRITE ,__hw_uartdbg_ifls_bits);
__IO_REG32_BIT(HW_UARTDBG_IMSC,                   0x80074038,__READ_WRITE ,__hw_uartdbg_imsc_bits);
__IO_REG32_BIT(HW_UARTDBG_RIS,                    0x8007403C,__READ       ,__hw_uartdbg_ris_bits);
__IO_REG32_BIT(HW_UARTDBG_MIS,                    0x80074040,__READ       ,__hw_uartdbg_mis_bits);
__IO_REG32_BIT(HW_UARTDBG_ICR,                    0x80074044,__WRITE      ,__hw_uartdbg_icr_bits);
__IO_REG32_BIT(HW_UARTDBG_DMACR,                  0x80074048,__READ_WRITE ,__hw_uartdbg_dmacr_bits);

/***************************************************************************
 **
 **  FlexCAN0
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_CAN0_MCR,                   0x80032000,__READ_WRITE,__can_mcr_bits);
__IO_REG32_BIT(HW_CAN0_CTRL,                  0x80032004,__READ_WRITE,__can_ctrl_bits);
__IO_REG32_BIT(HW_CAN0_TIMER,                 0x80032008,__READ_WRITE,__can_timer_bits);
__IO_REG32_BIT(HW_CAN0_RXGMASK,               0x80032010,__READ_WRITE,__can_rxgmask_bits);
__IO_REG32(    HW_CAN0_RX14MASK,              0x80032014,__READ_WRITE);
__IO_REG32(    HW_CAN0_RX15MASK,              0x80032018,__READ_WRITE);
__IO_REG32_BIT(HW_CAN0_ECR,                   0x8003201C,__READ_WRITE,__can_ecr_bits);
__IO_REG32_BIT(HW_CAN0_ESR,                   0x80032020,__READ_WRITE,__can_esr_bits);
__IO_REG32_BIT(HW_CAN0_IMASK2,                0x80032024,__READ_WRITE,__can_imask2_bits);
__IO_REG32_BIT(HW_CAN0_IMASK1,                0x80032028,__READ_WRITE,__can_imask1_bits);
__IO_REG32_BIT(HW_CAN0_IFLAG2,                0x8003202C,__READ_WRITE,__can_iflag2_bits);
__IO_REG32_BIT(HW_CAN0_IFLAG1,                0x80032030,__READ_WRITE,__can_iflag1_bits);
__IO_REG32_BIT(HW_CAN0_GFWR,                  0x80032034,__READ_WRITE,__can_gfwr_bits);
__IO_REG32(    HW_CAN0_MB0_15_BASE_ADDR,      0x80032080,__READ_WRITE);
__IO_REG32(    HW_CAN0_MB16_31_BASE_ADDR,     0x80032180,__READ_WRITE);
__IO_REG32(    HW_CAN0_MB32_63_BASE_ADDR,     0x80032280,__READ_WRITE);
__IO_REG32(    HW_CAN0_RXIMR0_15_BASE_ADDR,   0x80032880,__READ_WRITE);
__IO_REG32(    HW_CAN0_RXIMR16_31_BASE_ADDR,  0x800328C0,__READ_WRITE);
__IO_REG32(    HW_CAN0_RXIMR32_63_BASE_ADDR,  0x80032900,__READ_WRITE);

/***************************************************************************
 **
 **  FlexCAN1
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_CAN1_MCR,                   0x80034000,__READ_WRITE,__can_mcr_bits);
__IO_REG32_BIT(HW_CAN1_CTRL,                  0x80034004,__READ_WRITE,__can_ctrl_bits);
__IO_REG32_BIT(HW_CAN1_TIMER,                 0x80034008,__READ_WRITE,__can_timer_bits);
__IO_REG32_BIT(HW_CAN1_RXGMASK,               0x80034010,__READ_WRITE,__can_rxgmask_bits);
__IO_REG32(    HW_CAN1_RX14MASK,              0x80034014,__READ_WRITE);
__IO_REG32(    HW_CAN1_RX15MASK,              0x80034018,__READ_WRITE);
__IO_REG32_BIT(HW_CAN1_ECR,                   0x8003401C,__READ_WRITE,__can_ecr_bits);
__IO_REG32_BIT(HW_CAN1_ESR,                   0x80034020,__READ_WRITE,__can_esr_bits);
__IO_REG32_BIT(HW_CAN1_IMASK2,                0x80034024,__READ_WRITE,__can_imask2_bits);
__IO_REG32_BIT(HW_CAN1_IMASK1,                0x80034028,__READ_WRITE,__can_imask1_bits);
__IO_REG32_BIT(HW_CAN1_IFLAG2,                0x8003402C,__READ_WRITE,__can_iflag2_bits);
__IO_REG32_BIT(HW_CAN1_IFLAG1,                0x80034030,__READ_WRITE,__can_iflag1_bits);
__IO_REG32_BIT(HW_CAN1_GFWR,                  0x80034034,__READ_WRITE,__can_gfwr_bits);
__IO_REG32(    HW_CAN1_MB0_15_BASE_ADDR,      0x80034080,__READ_WRITE);
__IO_REG32(    HW_CAN1_MB16_31_BASE_ADDR,     0x80034180,__READ_WRITE);
__IO_REG32(    HW_CAN1_MB32_63_BASE_ADDR,     0x80034280,__READ_WRITE);
__IO_REG32(    HW_CAN1_RXIMR0_15_BASE_ADDR,   0x80034880,__READ_WRITE);
__IO_REG32(    HW_CAN1_RXIMR16_31_BASE_ADDR,  0x800348C0,__READ_WRITE);
__IO_REG32(    HW_CAN1_RXIMR32_63_BASE_ADDR,  0x80034900,__READ_WRITE);

/***************************************************************************
 **
 **  ENET-MAC0
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_ENET_MAC_EIR,               0x800F0004,__READ_WRITE,__hw_enet_mac_eir_bits);
__IO_REG32_BIT(HW_ENET_MAC_EIMR,              0x800F0008,__READ_WRITE,__hw_enet_mac_eir_bits);
__IO_REG32_BIT(HW_ENET_MAC_RDAR,              0x800F0010,__READ_WRITE,__hw_enet_mac_rdar_bits);
__IO_REG32_BIT(HW_ENET_MAC_TDAR,              0x800F0014,__READ_WRITE,__hw_enet_mac_tdar_bits);
__IO_REG32_BIT(HW_ENET_MAC_ECR,               0x800F0024,__READ_WRITE,__hw_enet_mac_ecr_bits);
__IO_REG32_BIT(HW_ENET_MAC_MMFR,              0x800F0040,__READ_WRITE,__hw_enet_mac_mmfr_bits);
__IO_REG32_BIT(HW_ENET_MAC_MSCR,              0x800F0044,__READ_WRITE,__hw_enet_mac_mscr_bits);
__IO_REG32_BIT(HW_ENET_MAC_MIBC,              0x800F0064,__READ_WRITE,__hw_enet_mac_mibc_bits);
__IO_REG32_BIT(HW_ENET_MAC_RCR,               0x800F0084,__READ_WRITE,__hw_enet_mac_rcr_bits);
__IO_REG32_BIT(HW_ENET_MAC_TCR,               0x800F00C4,__READ_WRITE,__hw_enet_mac_tcr_bits);
__IO_REG32(    HW_ENET_MAC_PALR,              0x800F00E4,__READ_WRITE);
__IO_REG32_BIT(HW_ENET_MAC_PAUR,              0x800F00E8,__READ_WRITE,__hw_enet_mac_paur_bits);
__IO_REG32_BIT(HW_ENET_MAC_OPD,               0x800F00EC,__READ_WRITE,__hw_enet_mac_opd_bits);
__IO_REG32(    HW_ENET_MAC_IAUR,              0x800F0118,__READ_WRITE);
__IO_REG32(    HW_ENET_MAC_IALR,              0x800F011C,__READ_WRITE);
__IO_REG32(    HW_ENET_MAC_GAUR,              0x800F0120,__READ_WRITE);
__IO_REG32(    HW_ENET_MAC_GALR,              0x800F0124,__READ_WRITE);
__IO_REG32_BIT(HW_ENET_MAC_TFW_SFCR,          0x800F0144,__READ_WRITE,__hw_enet_mac_tfw_sfcr_bits);
__IO_REG32_BIT(HW_ENET_MAC_FRBR,              0x800F014C,__READ_WRITE,__hw_enet_mac_frbr_bits);
__IO_REG32_BIT(HW_ENET_MAC_FRSR,              0x800F0150,__READ_WRITE,__hw_enet_mac_frsr_bits);
__IO_REG32(    HW_ENET_MAC_ERDSR,             0x800F0180,__READ_WRITE);
__IO_REG32(    HW_ENET_MAC_ETDSR,             0x800F0184,__READ_WRITE);
__IO_REG32_BIT(HW_ENET_MAC_EMRBR,             0x800F0188,__READ_WRITE,__hw_enet_mac_emrbr_bits);
__IO_REG32_BIT(HW_ENET_MAC_RX_SECTION_FULL,   0x800F0190,__READ_WRITE,__hw_enet_mac_rx_section_full_bits);
__IO_REG32_BIT(HW_ENET_MAC_RX_SECTION_EMPTY,  0x800F0194,__READ_WRITE,__hw_enet_mac_rx_section_empty_bits);
__IO_REG32_BIT(HW_ENET_MAC_RX_ALMOST_EMPTY,   0x800F0198,__READ_WRITE,__hw_enet_mac_rx_almost_empty_bits);
__IO_REG32_BIT(HW_ENET_MAC_RX_ALMOST_FULL,    0x800F019C,__READ_WRITE,__hw_enet_mac_rx_almost_full_bits);
__IO_REG32_BIT(HW_ENET_MAC_TX_SECTION_EMPTY,  0x800F01A0,__READ_WRITE,__hw_enet_mac_tx_section_empty_bits);
__IO_REG32_BIT(HW_ENET_MAC_TX_ALMOST_EMPTY,   0x800F01A4,__READ_WRITE,__hw_enet_mac_tx_almost_empty_bits);
__IO_REG32_BIT(HW_ENET_MAC_TX_ALMOST_FULL,    0x800F01A8,__READ_WRITE,__hw_enet_mac_tx_almost_full_bits);
__IO_REG32_BIT(HW_ENET_MAC_TX_IPG_LENGTH,     0x800F01AC,__READ_WRITE,__hw_enet_mac_tx_ipg_length_bits);
__IO_REG32_BIT(HW_ENET_MAC_TRUNC_FL,          0x800F01B0,__READ_WRITE,__hw_enet_mac_trunc_fl_bits);
__IO_REG32_BIT(HW_ENET_MAC_IPACCTXCONF,       0x800F01C0,__READ_WRITE,__hw_enet_mac_ipacctxconf_bits);
__IO_REG32_BIT(HW_ENET_MAC_IPACCRXCONF,       0x800F01C4,__READ_WRITE,__hw_enet_mac_ipaccrxconf_bits);
__IO_REG32(    HW_ENET_MAC_RMON_T_DROP,       0x800F0200,__READ      );
__IO_REG32(    HW_ENET_MAC_RMON_T_PACKETS,    0x800F0204,__READ      );
__IO_REG32(    HW_ENET_MAC_RMON_T_BC_PKT,     0x800F0208,__READ      );
__IO_REG32(    HW_ENET_MAC_RMON_T_MC_PKT,     0x800F020C,__READ      );
__IO_REG32(    HW_ENET_MAC_RMON_T_CRC_ALIGN,  0x800F0210,__READ      );
__IO_REG32(    HW_ENET_MAC_RMON_T_UNDERSIZE,  0x800F0214,__READ      );
__IO_REG32(    HW_ENET_MAC_RMON_T_OVERSIZE,   0x800F0218,__READ      );
__IO_REG32(    HW_ENET_MAC_RMON_T_FRAG,       0x800F021C,__READ      );
__IO_REG32(    HW_ENET_MAC_RMON_T_JAB,        0x800F0220,__READ      );
__IO_REG32(    HW_ENET_MAC_RMON_T_COL,        0x800F0224,__READ      );
__IO_REG32(    HW_ENET_MAC_RMON_T_P64,        0x800F0228,__READ      );
__IO_REG32(    HW_ENET_MAC_RMON_T_P65TO127N,  0x800F022C,__READ      );
__IO_REG32(    HW_ENET_MAC_RMON_T_P128TO255N, 0x800F0230,__READ      );
__IO_REG32(    HW_ENET_MAC_RMON_T_P256TO511,  0x800F0234,__READ      );
__IO_REG32(    HW_ENET_MAC_RMON_T_P512TO1023, 0x800F0238,__READ      );
__IO_REG32(    HW_ENET_MAC_RMON_T_P1024TO2047,0x800F023C,__READ      );
__IO_REG32(    HW_ENET_MAC_RMON_T_P_GTE2048,  0x800F0240,__READ      );
__IO_REG32(    HW_ENET_MAC_RMON_T_OCTETS,     0x800F0244,__READ      );
__IO_REG32(    HW_ENET_MAC_IEEE_T_DROP,       0x800F0248,__READ      );
__IO_REG32(    HW_ENET_MAC_IEEE_T_FRAME_OK,   0x800F024C,__READ      );
__IO_REG32(    HW_ENET_MAC_IEEE_T_1COL,       0x800F0250,__READ      );
__IO_REG32(    HW_ENET_MAC_IEEE_T_MCOL,       0x800F0254,__READ      );
__IO_REG32(    HW_ENET_MAC_IEEE_T_DEF,        0x800F0258,__READ      );
__IO_REG32(    HW_ENET_MAC_IEEE_T_LCOL,       0x800F025C,__READ      );
__IO_REG32(    HW_ENET_MAC_IEEE_T_EXCOL,      0x800F0260,__READ      );
__IO_REG32(    HW_ENET_MAC_IEEE_T_MACERR,     0x800F0264,__READ      );
__IO_REG32(    HW_ENET_MAC_IEEE_T_CSERR,      0x800F0268,__READ      );
__IO_REG32(    HW_ENET_MAC_IEEE_T_SQE,        0x800F026C,__READ      );
__IO_REG32(    HW_ENET_MAC_IEEE_T_FDXFC,      0x800F0270,__READ      );
__IO_REG32(    HW_ENET_MAC_IEEE_T_OCTETS_OK,  0x800F0274,__READ      );
__IO_REG32(    HW_ENET_MAC_RMON_R_PACKETS,    0x800F0284,__READ      );
__IO_REG32(    HW_ENET_MAC_RMON_R_BC_PKT,     0x800F0288,__READ      );
__IO_REG32(    HW_ENET_MAC_RMON_R_MC_PKT,     0x800F028C,__READ      );
__IO_REG32(    HW_ENET_MAC_RMON_R_CRC_ALIGN,  0x800F0290,__READ      );
__IO_REG32(    HW_ENET_MAC_RMON_R_UNDERSIZE,  0x800F0294,__READ      );
__IO_REG32(    HW_ENET_MAC_RMON_R_OVERSIZE,   0x800F0298,__READ      );
__IO_REG32(    HW_ENET_MAC_RMON_R_FRAG,       0x800F029C,__READ      );
__IO_REG32(    HW_ENET_MAC_RMON_R_JAB,        0x800F02A0,__READ      );
__IO_REG32(    HW_ENET_MAC_RMON_R_P64,        0x800F02A8,__READ      );
__IO_REG32(    HW_ENET_MAC_RMON_R_P65TO127,   0x800F02AC,__READ      );
__IO_REG32(    HW_ENET_MAC_RMON_R_P128TO255,  0x800F02B0,__READ      );
__IO_REG32(    HW_ENET_MAC_RMON_R_P256TO511,  0x800F02B4,__READ      );
__IO_REG32(    HW_ENET_MAC_RMON_R_P512TO1023, 0x800F02B8,__READ      );
__IO_REG32(    HW_ENET_MAC_RMON_R_P1024TO2047,0x800F02BC,__READ      );
__IO_REG32(    HW_ENET_MAC_RMON_R_P_GTE2048,  0x800F02C0,__READ      );
__IO_REG32(    HW_ENET_MAC_RMON_R_OCTETS,     0x800F02C4,__READ      );
__IO_REG32(    HW_ENET_MAC_IEEE_R_DROP,       0x800F02C8,__READ      );
__IO_REG32(    HW_ENET_MAC_IEEE_R_FRAME_OK,   0x800F02CC,__READ      );
__IO_REG32(    HW_ENET_MAC_IEEE_R_CRC,        0x800F02D0,__READ      );
__IO_REG32(    HW_ENET_MAC_IEEE_R_ALIGN,      0x800F02D4,__READ      );
__IO_REG32(    HW_ENET_MAC_IEEE_R_MACERR,     0x800F02D8,__READ      );
__IO_REG32(    HW_ENET_MAC_IEEE_R_FDXFC,      0x800F02DC,__READ      );
__IO_REG32(    HW_ENET_MAC_IEEE_R_OCTETS_OK,  0x800F02E0,__READ      );
__IO_REG32_BIT(HW_ENET_MAC_ATIME_CTRL,        0x800F0400,__READ_WRITE,__hw_enet_mac_atime_ctrl_bits);
__IO_REG32(    HW_ENET_MAC_ATIME,             0x800F0404,__READ_WRITE);
__IO_REG32(    HW_ENET_MAC_ATIME_EVT_OFFSET,  0x800F0408,__READ_WRITE);
__IO_REG32(    HW_ENET_MAC_ATIME_EVT_PERIOD,  0x800F040C,__READ_WRITE);
__IO_REG32_BIT(HW_ENET_MAC_ATIME_CORR,        0x800F0410,__READ_WRITE,__hw_enet_mac_atime_corr_bits);
__IO_REG32_BIT(HW_ENET_MAC_ATIME_INC,         0x800F0414,__READ_WRITE,__hw_enet_mac_atime_inc_bits);
__IO_REG32(    HW_ENET_MAC_TS_TIMESTAMP,      0x800F0418,__READ      );
__IO_REG32(    HW_ENET_MAC_SMAC_0_0,          0x800F0500,__READ_WRITE);
__IO_REG32(    HW_ENET_MAC_SMAC_0_1,          0x800F0504,__READ_WRITE);
__IO_REG32(    HW_ENET_MAC_SMAC_1_0,          0x800F0508,__READ_WRITE);
__IO_REG32(    HW_ENET_MAC_SMAC_1_1,          0x800F050C,__READ_WRITE);
__IO_REG32(    HW_ENET_MAC_SMAC_2_0,          0x800F0510,__READ_WRITE);
__IO_REG32(    HW_ENET_MAC_SMAC_2_1,          0x800F0514,__READ_WRITE);
__IO_REG32(    HW_ENET_MAC_SMAC_3_0,          0x800F0518,__READ_WRITE);
__IO_REG32(    HW_ENET_MAC_SMAC_3_1,          0x800F051C,__READ_WRITE);
__IO_REG32(    HW_ENET_MAC_COMP_REG_0,        0x800F0600,__READ_WRITE);
__IO_REG32(    HW_ENET_MAC_COMP_REG_1,        0x800F0604,__READ_WRITE);
__IO_REG32(    HW_ENET_MAC_COMP_REG_2,        0x800F0608,__READ_WRITE);
__IO_REG32(    HW_ENET_MAC_COMP_REG_3,        0x800F060C,__READ_WRITE);
__IO_REG32(    HW_ENET_MAC_CAPT_REG_0,        0x800F0640,__READ      );
__IO_REG32(    HW_ENET_MAC_CAPT_REG_1,        0x800F0644,__READ      );
__IO_REG32(    HW_ENET_MAC_CAPT_REG_2,        0x800F0648,__READ      );
__IO_REG32(    HW_ENET_MAC_CAPT_REG_3,        0x800F064C,__READ      );
__IO_REG32_BIT(HW_ENET_MAC_CCB_INT,           0x800F0680,__READ_WRITE,__hw_enet_mac_ccb_int_bits);
__IO_REG32_BIT(HW_ENET_MAC_CCB_INT_MASK,      0x800F0684,__READ_WRITE,__hw_enet_mac_ccb_int_bits);

/***************************************************************************
 **
 **  I2C0
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_I2C0_CTRL0,                 0x80058000,__READ_WRITE ,__hw_i2c_ctrl0_bits);
__IO_REG32_BIT(HW_I2C0_CTRL0_SET,             0x80058004,__WRITE      ,__hw_i2c_ctrl0_bits);
__IO_REG32_BIT(HW_I2C0_CTRL0_CLR,             0x80058008,__WRITE      ,__hw_i2c_ctrl0_bits);
__IO_REG32_BIT(HW_I2C0_CTRL0_TOG,             0x8005800C,__WRITE      ,__hw_i2c_ctrl0_bits);
__IO_REG32_BIT(HW_I2C0_TIMING0,               0x80058010,__READ_WRITE ,__hw_i2c_timing0_bits);
__IO_REG32_BIT(HW_I2C0_TIMING0_SET,           0x80058014,__WRITE      ,__hw_i2c_timing0_bits);
__IO_REG32_BIT(HW_I2C0_TIMING0_CLR,           0x80058018,__WRITE      ,__hw_i2c_timing0_bits);
__IO_REG32_BIT(HW_I2C0_TIMING0_TOG,           0x8005801C,__WRITE      ,__hw_i2c_timing0_bits);
__IO_REG32_BIT(HW_I2C0_TIMING1,               0x80058020,__READ_WRITE ,__hw_i2c_timing1_bits);
__IO_REG32_BIT(HW_I2C0_TIMING1_SET,           0x80058024,__WRITE      ,__hw_i2c_timing1_bits);
__IO_REG32_BIT(HW_I2C0_TIMING1_CLR,           0x80058028,__WRITE      ,__hw_i2c_timing1_bits);
__IO_REG32_BIT(HW_I2C0_TIMING1_TOG,           0x8005802C,__WRITE      ,__hw_i2c_timing1_bits);
__IO_REG32_BIT(HW_I2C0_TIMING2,               0x80058030,__READ_WRITE ,__hw_i2c_timing2_bits);
__IO_REG32_BIT(HW_I2C0_TIMING2_SET,           0x80058034,__WRITE      ,__hw_i2c_timing2_bits);
__IO_REG32_BIT(HW_I2C0_TIMING2_CLR,           0x80058038,__WRITE      ,__hw_i2c_timing2_bits);
__IO_REG32_BIT(HW_I2C0_TIMING2_TOG,           0x8005803C,__WRITE      ,__hw_i2c_timing2_bits);
__IO_REG32_BIT(HW_I2C0_CTRL1,                 0x80058040,__READ_WRITE ,__hw_i2c_ctrl1_bits);
__IO_REG32_BIT(HW_I2C0_CTRL1_SET,             0x80058044,__WRITE      ,__hw_i2c_ctrl1_bits);
__IO_REG32_BIT(HW_I2C0_CTRL1_CLR,             0x80058048,__WRITE      ,__hw_i2c_ctrl1_bits);
__IO_REG32_BIT(HW_I2C0_CTRL1_TOG,             0x8005804C,__WRITE      ,__hw_i2c_ctrl1_bits);
__IO_REG32_BIT(HW_I2C0_STAT,                  0x80058050,__READ       ,__hw_i2c_stat_bits);
__IO_REG32_BIT(HW_I2C0_QUEUECTRL,             0x80058060,__READ_WRITE ,__hw_i2c_queuectrl_bits);
__IO_REG32_BIT(HW_I2C0_QUEUECTRL_SET,         0x80058064,__WRITE      ,__hw_i2c_queuectrl_bits);
__IO_REG32_BIT(HW_I2C0_QUEUECTRL_CLR,         0x80058068,__WRITE      ,__hw_i2c_queuectrl_bits);
__IO_REG32_BIT(HW_I2C0_QUEUECTRL_TOG,         0x8005806C,__WRITE      ,__hw_i2c_queuectrl_bits);
__IO_REG32_BIT(HW_I2C0_QUEUESTAT,             0x80058070,__READ       ,__hw_i2c_queuestat_bits);
__IO_REG32_BIT(HW_I2C0_QUEUECMD,              0x80058080,__READ_WRITE ,__hw_i2c_queuecmd_bits);
__IO_REG32_BIT(HW_I2C0_QUEUECMD_SET,          0x80058084,__WRITE      ,__hw_i2c_queuecmd_bits);
__IO_REG32_BIT(HW_I2C0_QUEUECMD_CLR,          0x80058088,__WRITE      ,__hw_i2c_queuecmd_bits);
__IO_REG32_BIT(HW_I2C0_QUEUECMD_TOG,          0x8005808C,__WRITE      ,__hw_i2c_queuecmd_bits);
__IO_REG32(    HW_I2C0_QUEUEDATA,             0x80058090,__READ       );
__IO_REG32(    HW_I2C0_DATA,                  0x800580A0,__READ_WRITE );
__IO_REG32_BIT(HW_I2C0_DEBUG0,                0x800580B0,__READ_WRITE ,__hw_i2c_debug0_bits);
__IO_REG32_BIT(HW_I2C0_DEBUG0_SET,            0x800580B4,__WRITE      ,__hw_i2c_debug0_bits);
__IO_REG32_BIT(HW_I2C0_DEBUG0_CLR,            0x800580B8,__WRITE      ,__hw_i2c_debug0_bits);
__IO_REG32_BIT(HW_I2C0_DEBUG0_TOG,            0x800580BC,__WRITE      ,__hw_i2c_debug0_bits);
__IO_REG32_BIT(HW_I2C0_DEBUG1,                0x800580C0,__READ_WRITE ,__hw_i2c_debug1_bits);
__IO_REG32_BIT(HW_I2C0_DEBUG1_SET,            0x800580C4,__WRITE      ,__hw_i2c_debug1_bits);
__IO_REG32_BIT(HW_I2C0_DEBUG1_CLR,            0x800580C8,__WRITE      ,__hw_i2c_debug1_bits);
__IO_REG32_BIT(HW_I2C0_DEBUG1_TOG,            0x800580CC,__WRITE      ,__hw_i2c_debug1_bits);
__IO_REG32_BIT(HW_I2C0_VERSION,               0x800580D0,__READ       ,__hw_i2c_version_bits);

/***************************************************************************
 **
 **  I2C1
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_I2C1_CTRL0,                 0x8005A000,__READ_WRITE ,__hw_i2c_ctrl0_bits);
__IO_REG32_BIT(HW_I2C1_CTRL0_SET,             0x8005A004,__WRITE      ,__hw_i2c_ctrl0_bits);
__IO_REG32_BIT(HW_I2C1_CTRL0_CLR,             0x8005A008,__WRITE      ,__hw_i2c_ctrl0_bits);
__IO_REG32_BIT(HW_I2C1_CTRL0_TOG,             0x8005A00C,__WRITE      ,__hw_i2c_ctrl0_bits);
__IO_REG32_BIT(HW_I2C1_TIMING0,               0x8005A010,__READ_WRITE ,__hw_i2c_timing0_bits);
__IO_REG32_BIT(HW_I2C1_TIMING0_SET,           0x8005A014,__WRITE      ,__hw_i2c_timing0_bits);
__IO_REG32_BIT(HW_I2C1_TIMING0_CLR,           0x8005A018,__WRITE      ,__hw_i2c_timing0_bits);
__IO_REG32_BIT(HW_I2C1_TIMING0_TOG,           0x8005A01C,__WRITE      ,__hw_i2c_timing0_bits);
__IO_REG32_BIT(HW_I2C1_TIMING1,               0x8005A020,__READ_WRITE ,__hw_i2c_timing1_bits);
__IO_REG32_BIT(HW_I2C1_TIMING1_SET,           0x8005A024,__WRITE      ,__hw_i2c_timing1_bits);
__IO_REG32_BIT(HW_I2C1_TIMING1_CLR,           0x8005A028,__WRITE      ,__hw_i2c_timing1_bits);
__IO_REG32_BIT(HW_I2C1_TIMING1_TOG,           0x8005A02C,__WRITE      ,__hw_i2c_timing1_bits);
__IO_REG32_BIT(HW_I2C1_TIMING2,               0x8005A030,__READ_WRITE ,__hw_i2c_timing2_bits);
__IO_REG32_BIT(HW_I2C1_TIMING2_SET,           0x8005A034,__WRITE      ,__hw_i2c_timing2_bits);
__IO_REG32_BIT(HW_I2C1_TIMING2_CLR,           0x8005A038,__WRITE      ,__hw_i2c_timing2_bits);
__IO_REG32_BIT(HW_I2C1_TIMING2_TOG,           0x8005A03C,__WRITE      ,__hw_i2c_timing2_bits);
__IO_REG32_BIT(HW_I2C1_CTRL1,                 0x8005A040,__READ_WRITE ,__hw_i2c_ctrl1_bits);
__IO_REG32_BIT(HW_I2C1_CTRL1_SET,             0x8005A044,__WRITE      ,__hw_i2c_ctrl1_bits);
__IO_REG32_BIT(HW_I2C1_CTRL1_CLR,             0x8005A048,__WRITE      ,__hw_i2c_ctrl1_bits);
__IO_REG32_BIT(HW_I2C1_CTRL1_TOG,             0x8005A04C,__WRITE      ,__hw_i2c_ctrl1_bits);
__IO_REG32_BIT(HW_I2C1_STAT,                  0x8005A050,__READ       ,__hw_i2c_stat_bits);
__IO_REG32_BIT(HW_I2C1_QUEUECTRL,             0x8005A060,__READ_WRITE ,__hw_i2c_queuectrl_bits);
__IO_REG32_BIT(HW_I2C1_QUEUECTRL_SET,         0x8005A064,__WRITE      ,__hw_i2c_queuectrl_bits);
__IO_REG32_BIT(HW_I2C1_QUEUECTRL_CLR,         0x8005A068,__WRITE      ,__hw_i2c_queuectrl_bits);
__IO_REG32_BIT(HW_I2C1_QUEUECTRL_TOG,         0x8005A06C,__WRITE      ,__hw_i2c_queuectrl_bits);
__IO_REG32_BIT(HW_I2C1_QUEUESTAT,             0x8005A070,__READ       ,__hw_i2c_queuestat_bits);
__IO_REG32_BIT(HW_I2C1_QUEUECMD,              0x8005A080,__READ_WRITE ,__hw_i2c_queuecmd_bits);
__IO_REG32_BIT(HW_I2C1_QUEUECMD_SET,          0x8005A084,__WRITE      ,__hw_i2c_queuecmd_bits);
__IO_REG32_BIT(HW_I2C1_QUEUECMD_CLR,          0x8005A088,__WRITE      ,__hw_i2c_queuecmd_bits);
__IO_REG32_BIT(HW_I2C1_QUEUECMD_TOG,          0x8005A08C,__WRITE      ,__hw_i2c_queuecmd_bits);
__IO_REG32(    HW_I2C1_QUEUEDATA,             0x8005A090,__READ       );
__IO_REG32(    HW_I2C1_DATA,                  0x8005A0A0,__READ_WRITE );
__IO_REG32_BIT(HW_I2C1_DEBUG0,                0x8005A0B0,__READ_WRITE ,__hw_i2c_debug0_bits);
__IO_REG32_BIT(HW_I2C1_DEBUG0_SET,            0x8005A0B4,__WRITE      ,__hw_i2c_debug0_bits);
__IO_REG32_BIT(HW_I2C1_DEBUG0_CLR,            0x8005A0B8,__WRITE      ,__hw_i2c_debug0_bits);
__IO_REG32_BIT(HW_I2C1_DEBUG0_TOG,            0x8005A0BC,__WRITE      ,__hw_i2c_debug0_bits);
__IO_REG32_BIT(HW_I2C1_DEBUG1,                0x8005A0C0,__READ_WRITE ,__hw_i2c_debug1_bits);
__IO_REG32_BIT(HW_I2C1_DEBUG1_SET,            0x8005A0C4,__WRITE      ,__hw_i2c_debug1_bits);
__IO_REG32_BIT(HW_I2C1_DEBUG1_CLR,            0x8005A0C8,__WRITE      ,__hw_i2c_debug1_bits);
__IO_REG32_BIT(HW_I2C1_DEBUG1_TOG,            0x8005A0CC,__WRITE      ,__hw_i2c_debug1_bits);
__IO_REG32_BIT(HW_I2C1_VERSION,               0x8005A0D0,__READ       ,__hw_i2c_version_bits);

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
__IO_REG32_BIT(HW_PWM_ACTIVE5,                  0x800640B0,__READ_WRITE ,__hw_pwm_activex_bits);
__IO_REG32_BIT(HW_PWM_ACTIVE5_SET,              0x800640B4,__WRITE      ,__hw_pwm_activex_bits);
__IO_REG32_BIT(HW_PWM_ACTIVE5_CLR,              0x800640B8,__WRITE      ,__hw_pwm_activex_bits);
__IO_REG32_BIT(HW_PWM_ACTIVE5_TOG,              0x800640BC,__WRITE      ,__hw_pwm_activex_bits);
__IO_REG32_BIT(HW_PWM_PERIOD5,                  0x800640C0,__READ_WRITE ,__hw_pwm_periodx_bits);
__IO_REG32_BIT(HW_PWM_PERIOD5_SET,              0x800640C4,__WRITE      ,__hw_pwm_periodx_bits);
__IO_REG32_BIT(HW_PWM_PERIOD5_CLR,              0x800640C8,__WRITE      ,__hw_pwm_periodx_bits);
__IO_REG32_BIT(HW_PWM_PERIOD5_TOG,              0x800640CC,__WRITE      ,__hw_pwm_periodx_bits);
__IO_REG32_BIT(HW_PWM_ACTIVE6,                  0x800640D0,__READ_WRITE ,__hw_pwm_activex_bits);
__IO_REG32_BIT(HW_PWM_ACTIVE6_SET,              0x800640D4,__WRITE      ,__hw_pwm_activex_bits);
__IO_REG32_BIT(HW_PWM_ACTIVE6_CLR,              0x800640D8,__WRITE      ,__hw_pwm_activex_bits);
__IO_REG32_BIT(HW_PWM_ACTIVE6_TOG,              0x800640DC,__WRITE      ,__hw_pwm_activex_bits);
__IO_REG32_BIT(HW_PWM_PERIOD6,                  0x800640E0,__READ_WRITE ,__hw_pwm_periodx_bits);
__IO_REG32_BIT(HW_PWM_PERIOD6_SET,              0x800640E4,__WRITE      ,__hw_pwm_periodx_bits);
__IO_REG32_BIT(HW_PWM_PERIOD6_CLR,              0x800640E8,__WRITE      ,__hw_pwm_periodx_bits);
__IO_REG32_BIT(HW_PWM_PERIOD6_TOG,              0x800640EC,__WRITE      ,__hw_pwm_periodx_bits);
__IO_REG32_BIT(HW_PWM_ACTIVE7,                  0x800640F0,__READ_WRITE ,__hw_pwm_activex_bits);
__IO_REG32_BIT(HW_PWM_ACTIVE7_SET,              0x800640F4,__WRITE      ,__hw_pwm_activex_bits);
__IO_REG32_BIT(HW_PWM_ACTIVE7_CLR,              0x800640F8,__WRITE      ,__hw_pwm_activex_bits);
__IO_REG32_BIT(HW_PWM_ACTIVE7_TOG,              0x800640FC,__WRITE      ,__hw_pwm_activex_bits);
__IO_REG32_BIT(HW_PWM_PERIOD7,                  0x80064100,__READ_WRITE ,__hw_pwm_periodx_bits);
__IO_REG32_BIT(HW_PWM_PERIOD7_SET,              0x80064104,__WRITE      ,__hw_pwm_periodx_bits);
__IO_REG32_BIT(HW_PWM_PERIOD7_CLR,              0x80064108,__WRITE      ,__hw_pwm_periodx_bits);
__IO_REG32_BIT(HW_PWM_PERIOD7_TOG,              0x8006410C,__WRITE      ,__hw_pwm_periodx_bits);
__IO_REG32_BIT(HW_PWM_VERSION,                  0x80064110,__READ       ,__hw_pwm_version_bits);

/***************************************************************************
 **
 **  ENET SWI
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_ENET_SWI_REVISION,            0x800F8000,__READ_WRITE ,__hw_enet_swi_revision_bits);
#define HW_ENET_SWI_LOOKUP_MEMORY_START     HW_ENET_SWI_REVISION
__IO_REG32(    HW_ENET_SWI_SCRATCH,             0x800F8004,__READ_WRITE );
__IO_REG32_BIT(HW_ENET_SWI_PORT_ENA,            0x800F8008,__READ_WRITE ,__hw_enet_swi_port_ena_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_VERIFY,         0x800F8010,__READ_WRITE ,__hw_enet_swi_vlan_verify_bits);
__IO_REG32_BIT(HW_ENET_SWI_BCAST_DEFAULT_MASK,  0x800F8014,__READ_WRITE ,__hw_enet_swi_bcast_default_mask_bits);
__IO_REG32_BIT(HW_ENET_SWI_MCAST_DEFAULT_MASK,  0x800F8018,__READ_WRITE ,__hw_enet_swi_mcast_default_mask_bits);
__IO_REG32_BIT(HW_ENET_SWI_INPUT_LEARN_BLOCK,   0x800F801C,__READ_WRITE ,__hw_enet_swi_input_learn_block_bits);
__IO_REG32_BIT(HW_ENET_SWI_MGMT_CONFIG,         0x800F8020,__READ_WRITE ,__hw_enet_swi_mgmt_config_bits);
__IO_REG32_BIT(HW_ENET_SWI_MODE_CONFIG,         0x800F8024,__READ_WRITE ,__hw_enet_swi_mode_config_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_IN_MODE,        0x800F8028,__READ_WRITE ,__hw_enet_swi_vlan_in_mode_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_OUT_MODE,       0x800F802C,__READ_WRITE ,__hw_enet_swi_vlan_out_mode_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_IN_MODE_ENA,    0x800F8030,__READ_WRITE ,__hw_enet_swi_vlan_in_mode_ena_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_TAG_ID,         0x800F8034,__READ_WRITE ,__hw_enet_swi_vlan_tag_id_bits);
__IO_REG32_BIT(HW_ENET_SWI_MIRROR_CONTROL,      0x800F8040,__READ_WRITE ,__hw_enet_swi_mirror_control_bits);
__IO_REG32_BIT(HW_ENET_SWI_MIRROR_EG_MAP,       0x800F8044,__READ_WRITE ,__hw_enet_swi_mirror_eg_map_bits);
__IO_REG32_BIT(HW_ENET_SWI_MIRROR_ING_MAP,      0x800F8048,__READ_WRITE ,__hw_enet_swi_mirror_ing_map_bits);
__IO_REG32(    HW_ENET_SWI_MIRROR_ISRC_0,       0x800F804C,__READ_WRITE );
__IO_REG32_BIT(HW_ENET_SWI_MIRROR_ISRC_1,       0x800F8050,__READ_WRITE ,__hw_enet_swi_mirror_isrc_1_bits);
__IO_REG32(    HW_ENET_SWI_MIRROR_IDST_0,       0x800F8054,__READ_WRITE );
__IO_REG32_BIT(HW_ENET_SWI_MIRROR_IDST_1,       0x800F8058,__READ_WRITE ,__hw_enet_swi_mirror_idst_1_bits);
__IO_REG32(    HW_ENET_SWI_MIRROR_ESRC_0,       0x800F805C,__READ_WRITE );
__IO_REG32_BIT(HW_ENET_SWI_MIRROR_ESRC_1,       0x800F8060,__READ_WRITE ,__hw_enet_swi_mirror_esrc_1_bits);
__IO_REG32(    HW_ENET_SWI_MIRROR_EDST_0,       0x800F8064,__READ_WRITE );
__IO_REG32_BIT(HW_ENET_SWI_MIRROR_EDST_1,       0x800F8068,__READ_WRITE ,__hw_enet_swi_mirror_edst_1_bits);
__IO_REG32_BIT(HW_ENET_SWI_MIRROR_CNT,          0x800F806C,__READ_WRITE ,__hw_enet_swi_mirror_cnt_bits);
__IO_REG32_BIT(HW_ENET_SWI_OQMGR_STATUS,        0x800F8080,__READ_WRITE ,__hw_enet_swi_oqmgr_status_bits);
__IO_REG32_BIT(HW_ENET_SWI_QMGR_MINCELLS,       0x800F8084,__READ_WRITE ,__hw_enet_swi_qmgr_mincells_bits);
__IO_REG32(    HW_ENET_SWI_QMGR_ST_MINCELLS,    0x800F8088,__READ_WRITE );
__IO_REG32_BIT(HW_ENET_SWI_QMGR_CONGEST_STAT,   0x800F808C,__READ       ,__hw_enet_swi_qmgr_congest_stat_bits);
__IO_REG32_BIT(HW_ENET_SWI_QMGR_IFACE_STAT,     0x800F8090,__READ       ,__hw_enet_swi_qmgr_iface_stat_bits);
__IO_REG32_BIT(HW_ENET_SWI_QM_WEIGHTS,          0x800F8094,__READ_WRITE ,__hw_enet_swi_qm_weights_bits);
__IO_REG32_BIT(HW_ENET_SWI_QMGR_MINCELLSP0,     0x800F809C,__READ_WRITE ,__hw_enet_swi_qmgr_mincellsp0_bits);
__IO_REG32_BIT(HW_ENET_SWI_FORCE_FWD_P0,        0x800F80BC,__READ_WRITE ,__hw_enet_swi_force_fwd_p0_bits);
__IO_REG32_BIT(HW_ENET_SWI_PORTSNOOP1,          0x800F80C0,__READ_WRITE ,__hw_enet_swi_portsnoop_bits);
__IO_REG32_BIT(HW_ENET_SWI_PORTSNOOP2,          0x800F80C4,__READ_WRITE ,__hw_enet_swi_portsnoop_bits);
__IO_REG32_BIT(HW_ENET_SWI_PORTSNOOP3,          0x800F80C8,__READ_WRITE ,__hw_enet_swi_portsnoop_bits);
__IO_REG32_BIT(HW_ENET_SWI_PORTSNOOP4,          0x800F80CC,__READ_WRITE ,__hw_enet_swi_portsnoop_bits);
__IO_REG32_BIT(HW_ENET_SWI_PORTSNOOP5,          0x800F80D0,__READ_WRITE ,__hw_enet_swi_portsnoop_bits);
__IO_REG32_BIT(HW_ENET_SWI_PORTSNOOP6,          0x800F80D4,__READ_WRITE ,__hw_enet_swi_portsnoop_bits);
__IO_REG32_BIT(HW_ENET_SWI_PORTSNOOP7,          0x800F80D8,__READ_WRITE ,__hw_enet_swi_portsnoop_bits);
__IO_REG32_BIT(HW_ENET_SWI_PORTSNOOP8,          0x800F80DC,__READ_WRITE ,__hw_enet_swi_portsnoop_bits);
__IO_REG32_BIT(HW_ENET_SWI_IPSNOOP1,            0x800F80E0,__READ_WRITE ,__hw_enet_swi_ipsnoop_bits);
__IO_REG32_BIT(HW_ENET_SWI_IPSNOOP2,            0x800F80E4,__READ_WRITE ,__hw_enet_swi_ipsnoop_bits);
__IO_REG32_BIT(HW_ENET_SWI_IPSNOOP3,            0x800F80E8,__READ_WRITE ,__hw_enet_swi_ipsnoop_bits);
__IO_REG32_BIT(HW_ENET_SWI_IPSNOOP4,            0x800F80EC,__READ_WRITE ,__hw_enet_swi_ipsnoop_bits);
__IO_REG32_BIT(HW_ENET_SWI_IPSNOOP5,            0x800F80F0,__READ_WRITE ,__hw_enet_swi_ipsnoop_bits);
__IO_REG32_BIT(HW_ENET_SWI_IPSNOOP6,            0x800F80F4,__READ_WRITE ,__hw_enet_swi_ipsnoop_bits);
__IO_REG32_BIT(HW_ENET_SWI_IPSNOOP7,            0x800F80F8,__READ_WRITE ,__hw_enet_swi_ipsnoop_bits);
__IO_REG32_BIT(HW_ENET_SWI_IPSNOOP8,            0x800F80FC,__READ_WRITE ,__hw_enet_swi_ipsnoop_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_PRIORITY0,      0x800F8100,__READ_WRITE ,__hw_enet_swi_vlan_priority_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_PRIORITY1,      0x800F8104,__READ_WRITE ,__hw_enet_swi_vlan_priority_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_PRIORITY2,      0x800F8108,__READ_WRITE ,__hw_enet_swi_vlan_priority_bits);
__IO_REG32_BIT(HW_ENET_SWI_IP_PRIORITY,         0x800F8140,__READ_WRITE ,__hw_enet_swi_ip_priority_bits);
__IO_REG32_BIT(HW_ENET_SWI_PRIORITY_CFG0,       0x800F8180,__READ_WRITE ,__hw_enet_swi_priority_cfg_bits);
__IO_REG32_BIT(HW_ENET_SWI_PRIORITY_CFG1,       0x800F8184,__READ_WRITE ,__hw_enet_swi_priority_cfg_bits);
__IO_REG32_BIT(HW_ENET_SWI_PRIORITY_CFG2,       0x800F8188,__READ_WRITE ,__hw_enet_swi_priority_cfg_bits);
__IO_REG32_BIT(HW_ENET_SWI_SYSTEM_TAGINFO0,     0x800F8200,__READ_WRITE ,__hw_enet_swi_system_taginfo_bits);
__IO_REG32_BIT(HW_ENET_SWI_SYSTEM_TAGINFO1,     0x800F8204,__READ_WRITE ,__hw_enet_swi_system_taginfo_bits);
__IO_REG32_BIT(HW_ENET_SWI_SYSTEM_TAGINFO2,     0x800F8208,__READ_WRITE ,__hw_enet_swi_system_taginfo_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_RES_TABLE_0,    0x800F8280,__READ_WRITE ,__hw_enet_swi_vlan_res_table_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_RES_TABLE_1,    0x800F8284,__READ_WRITE ,__hw_enet_swi_vlan_res_table_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_RES_TABLE_2,    0x800F8288,__READ_WRITE ,__hw_enet_swi_vlan_res_table_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_RES_TABLE_3,    0x800F828C,__READ_WRITE ,__hw_enet_swi_vlan_res_table_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_RES_TABLE_4,    0x800F8290,__READ_WRITE ,__hw_enet_swi_vlan_res_table_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_RES_TABLE_5,    0x800F8294,__READ_WRITE ,__hw_enet_swi_vlan_res_table_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_RES_TABLE_6,    0x800F8298,__READ_WRITE ,__hw_enet_swi_vlan_res_table_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_RES_TABLE_7,    0x800F829C,__READ_WRITE ,__hw_enet_swi_vlan_res_table_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_RES_TABLE_8,    0x800F82A0,__READ_WRITE ,__hw_enet_swi_vlan_res_table_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_RES_TABLE_9,    0x800F82A4,__READ_WRITE ,__hw_enet_swi_vlan_res_table_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_RES_TABLE_10,   0x800F82A8,__READ_WRITE ,__hw_enet_swi_vlan_res_table_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_RES_TABLE_11,   0x800F82AC,__READ_WRITE ,__hw_enet_swi_vlan_res_table_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_RES_TABLE_12,   0x800F82B0,__READ_WRITE ,__hw_enet_swi_vlan_res_table_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_RES_TABLE_13,   0x800F82B4,__READ_WRITE ,__hw_enet_swi_vlan_res_table_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_RES_TABLE_14,   0x800F82B8,__READ_WRITE ,__hw_enet_swi_vlan_res_table_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_RES_TABLE_15,   0x800F82BC,__READ_WRITE ,__hw_enet_swi_vlan_res_table_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_RES_TABLE_16,   0x800F82C0,__READ_WRITE ,__hw_enet_swi_vlan_res_table_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_RES_TABLE_17,   0x800F82C4,__READ_WRITE ,__hw_enet_swi_vlan_res_table_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_RES_TABLE_18,   0x800F82C8,__READ_WRITE ,__hw_enet_swi_vlan_res_table_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_RES_TABLE_19,   0x800F82CC,__READ_WRITE ,__hw_enet_swi_vlan_res_table_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_RES_TABLE_20,   0x800F82D0,__READ_WRITE ,__hw_enet_swi_vlan_res_table_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_RES_TABLE_21,   0x800F82D4,__READ_WRITE ,__hw_enet_swi_vlan_res_table_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_RES_TABLE_22,   0x800F82D8,__READ_WRITE ,__hw_enet_swi_vlan_res_table_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_RES_TABLE_23,   0x800F82DC,__READ_WRITE ,__hw_enet_swi_vlan_res_table_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_RES_TABLE_24,   0x800F82E0,__READ_WRITE ,__hw_enet_swi_vlan_res_table_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_RES_TABLE_25,   0x800F82E4,__READ_WRITE ,__hw_enet_swi_vlan_res_table_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_RES_TABLE_26,   0x800F82E8,__READ_WRITE ,__hw_enet_swi_vlan_res_table_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_RES_TABLE_27,   0x800F82EC,__READ_WRITE ,__hw_enet_swi_vlan_res_table_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_RES_TABLE_28,   0x800F82F0,__READ_WRITE ,__hw_enet_swi_vlan_res_table_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_RES_TABLE_29,   0x800F82F4,__READ_WRITE ,__hw_enet_swi_vlan_res_table_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_RES_TABLE_30,   0x800F82F8,__READ_WRITE ,__hw_enet_swi_vlan_res_table_bits);
__IO_REG32_BIT(HW_ENET_SWI_VLAN_RES_TABLE_31,   0x800F82FC,__READ_WRITE ,__hw_enet_swi_vlan_res_table_bits);
__IO_REG32(    HW_ENET_SWI_TOTAL_DISC,          0x800F8300,__READ       );
__IO_REG32(    HW_ENET_SWI_TOTAL_BYT_DISC,      0x800F8304,__READ       );
__IO_REG32(    HW_ENET_SWI_TOTAL_FRM,           0x800F8308,__READ       );
__IO_REG32(    HW_ENET_SWI_TOTAL_BYT_FRM,       0x800F830C,__READ       );
__IO_REG32(    HW_ENET_SWI_ODISC0,              0x800F8310,__READ       );
__IO_REG32(    HW_ENET_SWI_IDISC_VLAN0,         0x800F8314,__READ       );
__IO_REG32(    HW_ENET_SWI_IDISC_UNTAGGED0,     0x800F8318,__READ       );
__IO_REG32(    HW_ENET_SWI_IDISC_BLOCKED0,      0x800F831C,__READ       );
__IO_REG32(    HW_ENET_SWI_ODISC1,              0x800F8320,__READ       );
__IO_REG32(    HW_ENET_SWI_IDISC_VLAN1,         0x800F8324,__READ       );
__IO_REG32(    HW_ENET_SWI_IDISC_UNTAGGED1,     0x800F8328,__READ       );
__IO_REG32(    HW_ENET_SWI_IDISC_BLOCKED1,      0x800F832C,__READ       );
__IO_REG32(    HW_ENET_SWI_ODISC2,              0x800F8330,__READ       );
__IO_REG32(    HW_ENET_SWI_IDISC_VLAN2,         0x800F8334,__READ       );
__IO_REG32(    HW_ENET_SWI_IDISC_UNTAGGED2,     0x800F8338,__READ       );
__IO_REG32(    HW_ENET_SWI_IDISC_BLOCKED2,      0x800F833C,__READ       );
__IO_REG32_BIT(HW_ENET_SWI_EIR,                 0x800F8400,__READ_WRITE ,__hw_enet_swi_eir_bits);
__IO_REG32_BIT(HW_ENET_SWI_EIMR,                0x800F8404,__READ_WRITE ,__hw_enet_swi_eir_bits);
__IO_REG32(    HW_ENET_SWI_ERDSR,               0x800F8408,__READ_WRITE );
__IO_REG32(    HW_ENET_SWI_ETDSR,               0x800F840C,__READ_WRITE );
__IO_REG32_BIT(HW_ENET_SWI_EMRBR,               0x800F8410,__READ_WRITE ,__hw_enet_swi_emrbr_bits);
__IO_REG32(    HW_ENET_SWI_RDAR,                0x800F8414,__READ_WRITE );
__IO_REG32(    HW_ENET_SWI_TDAR,                0x800F8418,__READ_WRITE );
__IO_REG32(    HW_ENET_SWI_LRN_REC_0,           0x800F8500,__READ       );
__IO_REG32_BIT(HW_ENET_SWI_LRN_REC_1,           0x800F8504,__READ       ,__hw_enet_swi_lrn_rec_1_bits);
__IO_REG32_BIT(HW_ENET_SWI_LRN_STATUS,          0x800F8508,__READ       ,__hw_enet_swi_lrn_status_bits);
__IO_REG32(    HW_ENET_SWI_LOOKUP_MEMORY_END,   0x80107FFC,__READ_WRITE );

/***************************************************************************
 **
 **  APPUART0
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_UARTAPP0_CTRL0,               0x8006A000,__READ_WRITE ,__hw_uartapp_ctrl0_bits);
__IO_REG32_BIT(HW_UARTAPP0_CTRL0_SET,           0x8006A004,__WRITE      ,__hw_uartapp_ctrl0_bits);
__IO_REG32_BIT(HW_UARTAPP0_CTRL0_CLR,           0x8006A008,__WRITE      ,__hw_uartapp_ctrl0_bits);
__IO_REG32_BIT(HW_UARTAPP0_CTRL0_TOG,           0x8006A00C,__WRITE      ,__hw_uartapp_ctrl0_bits);
__IO_REG32_BIT(HW_UARTAPP0_CTRL1,               0x8006A010,__READ_WRITE ,__hw_uartapp_ctrl1_bits);
__IO_REG32_BIT(HW_UARTAPP0_CTRL1_SET,           0x8006A014,__WRITE      ,__hw_uartapp_ctrl1_bits);
__IO_REG32_BIT(HW_UARTAPP0_CTRL1_CLR,           0x8006A018,__WRITE      ,__hw_uartapp_ctrl1_bits);
__IO_REG32_BIT(HW_UARTAPP0_CTRL1_TOG,           0x8006A01C,__WRITE      ,__hw_uartapp_ctrl1_bits);
__IO_REG32_BIT(HW_UARTAPP0_CTRL2,               0x8006A020,__READ_WRITE ,__hw_uartapp_ctrl2_bits);
__IO_REG32_BIT(HW_UARTAPP0_CTRL2_SET,           0x8006A024,__WRITE      ,__hw_uartapp_ctrl2_bits);
__IO_REG32_BIT(HW_UARTAPP0_CTRL2_CLR,           0x8006A028,__WRITE      ,__hw_uartapp_ctrl2_bits);
__IO_REG32_BIT(HW_UARTAPP0_CTRL2_TOG,           0x8006A02C,__WRITE      ,__hw_uartapp_ctrl2_bits);
__IO_REG32_BIT(HW_UARTAPP0_LINECTRL,            0x8006A030,__READ_WRITE ,__hw_uartapp_linectrl_bits);
__IO_REG32_BIT(HW_UARTAPP0_LINECTRL_SET,        0x8006A034,__WRITE      ,__hw_uartapp_linectrl_bits);
__IO_REG32_BIT(HW_UARTAPP0_LINECTRL_CLR,        0x8006A038,__WRITE      ,__hw_uartapp_linectrl_bits);
__IO_REG32_BIT(HW_UARTAPP0_LINECTRL_TOG,        0x8006A03C,__WRITE      ,__hw_uartapp_linectrl_bits);
__IO_REG32_BIT(HW_UARTAPP0_LINECTRL2,           0x8006A040,__READ_WRITE ,__hw_uartapp_linectrl2_bits);
__IO_REG32_BIT(HW_UARTAPP0_LINECTRL2_SET,       0x8006A044,__WRITE      ,__hw_uartapp_linectrl2_bits);
__IO_REG32_BIT(HW_UARTAPP0_LINECTRL2_CLR,       0x8006A048,__WRITE      ,__hw_uartapp_linectrl2_bits);
__IO_REG32_BIT(HW_UARTAPP0_LINECTRL2_TOG,       0x8006A04C,__WRITE      ,__hw_uartapp_linectrl2_bits);
__IO_REG32_BIT(HW_UARTAPP0_INTR,                0x8006A050,__READ_WRITE ,__hw_uartapp_intr_bits);
__IO_REG32_BIT(HW_UARTAPP0_INTR_SET,            0x8006A054,__WRITE      ,__hw_uartapp_intr_bits);
__IO_REG32_BIT(HW_UARTAPP0_INTR_CLR,            0x8006A058,__WRITE      ,__hw_uartapp_intr_bits);
__IO_REG32_BIT(HW_UARTAPP0_INTR_TOG,            0x8006A05C,__WRITE      ,__hw_uartapp_intr_bits);
__IO_REG32(    HW_UARTAPP0_DATA,                0x8006A060,__READ_WRITE );
__IO_REG32_BIT(HW_UARTAPP0_STAT,                0x8006A070,__READ_WRITE ,__hw_uartapp_stat_bits);
__IO_REG32_BIT(HW_UARTAPP0_DEBUG,               0x8006A080,__READ       ,__hw_uartapp_debug_bits);
__IO_REG32_BIT(HW_UARTAPP0_VERSION,             0x8006A090,__READ       ,__hw_uartapp_version_bits);
__IO_REG32_BIT(HW_UARTAPP0_AUTOBAUD,            0x8006A0A0,__READ_WRITE ,__hw_uartapp_autobaud_bits);

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
 **  APPUART3
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_UARTAPP3_CTRL0,               0x80070000,__READ_WRITE ,__hw_uartapp_ctrl0_bits);
__IO_REG32_BIT(HW_UARTAPP3_CTRL0_SET,           0x80070004,__WRITE      ,__hw_uartapp_ctrl0_bits);
__IO_REG32_BIT(HW_UARTAPP3_CTRL0_CLR,           0x80070008,__WRITE      ,__hw_uartapp_ctrl0_bits);
__IO_REG32_BIT(HW_UARTAPP3_CTRL0_TOG,           0x8007000C,__WRITE      ,__hw_uartapp_ctrl0_bits);
__IO_REG32_BIT(HW_UARTAPP3_CTRL1,               0x80070010,__READ_WRITE ,__hw_uartapp_ctrl1_bits);
__IO_REG32_BIT(HW_UARTAPP3_CTRL1_SET,           0x80070014,__WRITE      ,__hw_uartapp_ctrl1_bits);
__IO_REG32_BIT(HW_UARTAPP3_CTRL1_CLR,           0x80070018,__WRITE      ,__hw_uartapp_ctrl1_bits);
__IO_REG32_BIT(HW_UARTAPP3_CTRL1_TOG,           0x8007001C,__WRITE      ,__hw_uartapp_ctrl1_bits);
__IO_REG32_BIT(HW_UARTAPP3_CTRL2,               0x80070020,__READ_WRITE ,__hw_uartapp_ctrl2_bits);
__IO_REG32_BIT(HW_UARTAPP3_CTRL2_SET,           0x80070024,__WRITE      ,__hw_uartapp_ctrl2_bits);
__IO_REG32_BIT(HW_UARTAPP3_CTRL2_CLR,           0x80070028,__WRITE      ,__hw_uartapp_ctrl2_bits);
__IO_REG32_BIT(HW_UARTAPP3_CTRL2_TOG,           0x8007002C,__WRITE      ,__hw_uartapp_ctrl2_bits);
__IO_REG32_BIT(HW_UARTAPP3_LINECTRL,            0x80070030,__READ_WRITE ,__hw_uartapp_linectrl_bits);
__IO_REG32_BIT(HW_UARTAPP3_LINECTRL_SET,        0x80070034,__WRITE      ,__hw_uartapp_linectrl_bits);
__IO_REG32_BIT(HW_UARTAPP3_LINECTRL_CLR,        0x80070038,__WRITE      ,__hw_uartapp_linectrl_bits);
__IO_REG32_BIT(HW_UARTAPP3_LINECTRL_TOG,        0x8007003C,__WRITE      ,__hw_uartapp_linectrl_bits);
__IO_REG32_BIT(HW_UARTAPP3_LINECTRL2,           0x80070040,__READ_WRITE ,__hw_uartapp_linectrl2_bits);
__IO_REG32_BIT(HW_UARTAPP3_LINECTRL2_SET,       0x80070044,__WRITE      ,__hw_uartapp_linectrl2_bits);
__IO_REG32_BIT(HW_UARTAPP3_LINECTRL2_CLR,       0x80070048,__WRITE      ,__hw_uartapp_linectrl2_bits);
__IO_REG32_BIT(HW_UARTAPP3_LINECTRL2_TOG,       0x8007004C,__WRITE      ,__hw_uartapp_linectrl2_bits);
__IO_REG32_BIT(HW_UARTAPP3_INTR,                0x80070050,__READ_WRITE ,__hw_uartapp_intr_bits);
__IO_REG32_BIT(HW_UARTAPP3_INTR_SET,            0x80070054,__WRITE      ,__hw_uartapp_intr_bits);
__IO_REG32_BIT(HW_UARTAPP3_INTR_CLR,            0x80070058,__WRITE      ,__hw_uartapp_intr_bits);
__IO_REG32_BIT(HW_UARTAPP3_INTR_TOG,            0x8007005C,__WRITE      ,__hw_uartapp_intr_bits);
__IO_REG32(    HW_UARTAPP3_DATA,                0x80070060,__READ_WRITE );
__IO_REG32_BIT(HW_UARTAPP3_STAT,                0x80070070,__READ_WRITE ,__hw_uartapp_stat_bits);
__IO_REG32_BIT(HW_UARTAPP3_DEBUG,               0x80070080,__READ       ,__hw_uartapp_debug_bits);
__IO_REG32_BIT(HW_UARTAPP3_VERSION,             0x80070090,__READ       ,__hw_uartapp_version_bits);
__IO_REG32_BIT(HW_UARTAPP3_AUTOBAUD,            0x800700A0,__READ_WRITE ,__hw_uartapp_autobaud_bits);

/***************************************************************************
 **
 **  APPUART4
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_UARTAPP4_CTRL0,               0x80072000,__READ_WRITE ,__hw_uartapp_ctrl0_bits);
__IO_REG32_BIT(HW_UARTAPP4_CTRL0_SET,           0x80072004,__WRITE      ,__hw_uartapp_ctrl0_bits);
__IO_REG32_BIT(HW_UARTAPP4_CTRL0_CLR,           0x80072008,__WRITE      ,__hw_uartapp_ctrl0_bits);
__IO_REG32_BIT(HW_UARTAPP4_CTRL0_TOG,           0x8007200C,__WRITE      ,__hw_uartapp_ctrl0_bits);
__IO_REG32_BIT(HW_UARTAPP4_CTRL1,               0x80072010,__READ_WRITE ,__hw_uartapp_ctrl1_bits);
__IO_REG32_BIT(HW_UARTAPP4_CTRL1_SET,           0x80072014,__WRITE      ,__hw_uartapp_ctrl1_bits);
__IO_REG32_BIT(HW_UARTAPP4_CTRL1_CLR,           0x80072018,__WRITE      ,__hw_uartapp_ctrl1_bits);
__IO_REG32_BIT(HW_UARTAPP4_CTRL1_TOG,           0x8007201C,__WRITE      ,__hw_uartapp_ctrl1_bits);
__IO_REG32_BIT(HW_UARTAPP4_CTRL2,               0x80072020,__READ_WRITE ,__hw_uartapp_ctrl2_bits);
__IO_REG32_BIT(HW_UARTAPP4_CTRL2_SET,           0x80072024,__WRITE      ,__hw_uartapp_ctrl2_bits);
__IO_REG32_BIT(HW_UARTAPP4_CTRL2_CLR,           0x80072028,__WRITE      ,__hw_uartapp_ctrl2_bits);
__IO_REG32_BIT(HW_UARTAPP4_CTRL2_TOG,           0x8007202C,__WRITE      ,__hw_uartapp_ctrl2_bits);
__IO_REG32_BIT(HW_UARTAPP4_LINECTRL,            0x80072030,__READ_WRITE ,__hw_uartapp_linectrl_bits);
__IO_REG32_BIT(HW_UARTAPP4_LINECTRL_SET,        0x80072034,__WRITE      ,__hw_uartapp_linectrl_bits);
__IO_REG32_BIT(HW_UARTAPP4_LINECTRL_CLR,        0x80072038,__WRITE      ,__hw_uartapp_linectrl_bits);
__IO_REG32_BIT(HW_UARTAPP4_LINECTRL_TOG,        0x8007203C,__WRITE      ,__hw_uartapp_linectrl_bits);
__IO_REG32_BIT(HW_UARTAPP4_LINECTRL2,           0x80072040,__READ_WRITE ,__hw_uartapp_linectrl2_bits);
__IO_REG32_BIT(HW_UARTAPP4_LINECTRL2_SET,       0x80072044,__WRITE      ,__hw_uartapp_linectrl2_bits);
__IO_REG32_BIT(HW_UARTAPP4_LINECTRL2_CLR,       0x80072048,__WRITE      ,__hw_uartapp_linectrl2_bits);
__IO_REG32_BIT(HW_UARTAPP4_LINECTRL2_TOG,       0x8007204C,__WRITE      ,__hw_uartapp_linectrl2_bits);
__IO_REG32_BIT(HW_UARTAPP4_INTR,                0x80072050,__READ_WRITE ,__hw_uartapp_intr_bits);
__IO_REG32_BIT(HW_UARTAPP4_INTR_SET,            0x80072054,__WRITE      ,__hw_uartapp_intr_bits);
__IO_REG32_BIT(HW_UARTAPP4_INTR_CLR,            0x80072058,__WRITE      ,__hw_uartapp_intr_bits);
__IO_REG32_BIT(HW_UARTAPP4_INTR_TOG,            0x8007205C,__WRITE      ,__hw_uartapp_intr_bits);
__IO_REG32(    HW_UARTAPP4_DATA,                0x80072060,__READ_WRITE );
__IO_REG32_BIT(HW_UARTAPP4_STAT,                0x80072070,__READ_WRITE ,__hw_uartapp_stat_bits);
__IO_REG32_BIT(HW_UARTAPP4_DEBUG,               0x80072080,__READ       ,__hw_uartapp_debug_bits);
__IO_REG32_BIT(HW_UARTAPP4_VERSION,             0x80072090,__READ       ,__hw_uartapp_version_bits);
__IO_REG32_BIT(HW_UARTAPP4_AUTOBAUD,            0x800720A0,__READ_WRITE ,__hw_uartapp_autobaud_bits);

/***************************************************************************
 **
 **  USB Controller 0
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_USBCTRL0_ID,                  0x80080000,__READ       ,__hw_usbctrl_id_bits);
__IO_REG32_BIT(HW_USBCTRL0_HWGENERAL,           0x80080004,__READ       ,__hw_usbctrl_hwgeneral_bits);
__IO_REG32_BIT(HW_USBCTRL0_HWHOST,              0x80080008,__READ       ,__hw_usbctrl_hwhost_bits);
__IO_REG32_BIT(HW_USBCTRL0_HWDEVICE,            0x8008000C,__READ       ,__hw_usbctrl_hwdevice_bits);
__IO_REG32_BIT(HW_USBCTRL0_HWTXBUF,             0x80080010,__READ       ,__hw_usbctrl_hwtxbuf_bits);
__IO_REG32_BIT(HW_USBCTRL0_HWRXBUF,             0x80080014,__READ       ,__hw_usbctrl_hwrxbuf_bits);
__IO_REG32_BIT(HW_USBCTRL0_GPTIMER0LD,          0x80080080,__READ_WRITE ,__hw_usbctrl_gptimerxld_bits);
__IO_REG32_BIT(HW_USBCTRL0_GPTIMER0CTRL,        0x80080084,__READ_WRITE ,__hw_usbctrl_gptimerxctrl_bits);
__IO_REG32_BIT(HW_USBCTRL0_GPTIMER1LD,          0x80080088,__READ_WRITE ,__hw_usbctrl_gptimerxld_bits);
__IO_REG32_BIT(HW_USBCTRL0_GPTIMER1CTRL,        0x8008008C,__READ_WRITE ,__hw_usbctrl_gptimerxctrl_bits);
__IO_REG32_BIT(HW_USBCTRL0_SBUSCFG,             0x80080090,__READ_WRITE ,__hw_usbctrl_sbuscfg_bits);
__IO_REG32_BIT(HW_USBCTRL0_CAPLENGTH,           0x80080100,__READ       ,__hw_usbctrl_caplength_bits);
__IO_REG32_BIT(HW_USBCTRL0_HCSPARAMS,           0x80080104,__READ       ,__hw_usbctrl_hcsparams_bits);
__IO_REG32_BIT(HW_USBCTRL0_HCCPARAMS,           0x80080108,__READ       ,__hw_usbctrl_hccparams_bits);
__IO_REG32_BIT(HW_USBCTRL0_DCIVERSION,          0x80080120,__READ       ,__hw_usbctrl_dciversion_bits);
__IO_REG32_BIT(HW_USBCTRL0_DCCPARAMS,           0x80080124,__READ       ,__hw_usbctrl_dccparams_bits);
__IO_REG32_BIT(HW_USBCTRL0_USBCMD,              0x80080140,__READ_WRITE ,__hw_usbctrl_usbcmd_bits);
__IO_REG32_BIT(HW_USBCTRL0_USBSTS,              0x80080144,__READ_WRITE ,__hw_usbctrl_usbsts_bits);
__IO_REG32_BIT(HW_USBCTRL0_USBINTR,             0x80080148,__READ_WRITE ,__hw_usbctrl_usbintr_bits);
__IO_REG32_BIT(HW_USBCTRL0_FRINDEX,             0x8008014C,__READ_WRITE ,__hw_usbctrl_frindex_bits);
__IO_REG32_BIT(HW_USBCTRL0_PERIODICLISTBASE,    0x80080154,__READ_WRITE ,__hw_usbctrl_periodiclistbase_bits);
#define HW_USBCTRL0_DEVICEADDR      HW_USBCTRL0_PERIODICLISTBASE
#define HW_USBCTRL0_DEVICEADDR_bit  HW_USBCTRL0_PERIODICLISTBASE_bit
__IO_REG32_BIT(HW_USBCTRL0_ASYNCLISTADDR,       0x80080158,__READ_WRITE ,__hw_usbctrl_asynclistaddr_bits);
#define HW_USBCTRL0_ENDPOINTLISTADDR      HW_USBCTRL0_ASYNCLISTADDR
#define HW_USBCTRL0_ENDPOINTLISTADDR_bit  HW_USBCTRL0_ASYNCLISTADDR_bit
__IO_REG32_BIT(HW_USBCTRL0_TTCTRL,              0x8008015C,__READ_WRITE ,__hw_usbctrl_ttctrl_bits);
__IO_REG32_BIT(HW_USBCTRL0_BURSTSIZE,           0x80080160,__READ_WRITE ,__hw_usbctrl_burstsize_bits);
__IO_REG32_BIT(HW_USBCTRL0_TXFILLTUNING,        0x80080164,__READ_WRITE ,__hw_usbctrl_txfilltuning_bits);
__IO_REG32_BIT(HW_USBCTRL0_IC_USB,              0x8008016C,__READ_WRITE ,__hw_usbctrl_ic_usb_bits);
__IO_REG32_BIT(HW_USBCTRL0_ULPI,                0x80080170,__READ_WRITE ,__hw_usbctrl_ulpi_bits);
__IO_REG32_BIT(HW_USBCTRL0_ENDPTNAK,            0x80080178,__READ_WRITE ,__hw_usbctrl_endptnak_bits);
__IO_REG32_BIT(HW_USBCTRL0_ENDPTNAKEN,          0x8008017C,__READ_WRITE ,__hw_usbctrl_endptnaken_bits);
__IO_REG32_BIT(HW_USBCTRL0_PORTSC1,             0x80080184,__READ_WRITE ,__hw_usbctrl_portsc_bits);
__IO_REG32_BIT(HW_USBCTRL0_OTGSC,               0x800801A4,__READ_WRITE ,__hw_usbctrl_otgsc_bits);
__IO_REG32_BIT(HW_USBCTRL0_USBMODE,             0x800801A8,__READ_WRITE ,__hw_usbctrl_usbmode_bits);
__IO_REG32_BIT(HW_USBCTRL0_ENDPTSETUPSTAT,      0x800801AC,__READ_WRITE ,__hw_usbctrl_endptsetupstat_bits);
__IO_REG32_BIT(HW_USBCTRL0_ENDPTPRIME,          0x800801B0,__READ_WRITE ,__hw_usbctrl_endptprime_bits);
__IO_REG32_BIT(HW_USBCTRL0_ENDPTFLUSH,          0x800801B4,__READ_WRITE ,__hw_usbctrl_endptflush_bits);
__IO_REG32_BIT(HW_USBCTRL0_ENDPTSTAT,           0x800801B8,__READ       ,__hw_usbctrl_endptstat_bits);
__IO_REG32_BIT(HW_USBCTRL0_ENDPTCOMPLETE,       0x800801BC,__READ_WRITE ,__hw_usbctrl_endptcomplete_bits);
__IO_REG32_BIT(HW_USBCTRL0_ENDPTCTRL0,          0x800801C0,__READ_WRITE ,__hw_usbctrl_endptctrl0_bits);
__IO_REG32_BIT(HW_USBCTRL0_ENDPTCTRL1,          0x800801C4,__READ_WRITE ,__hw_usbctrl_endptctrl_bits);
__IO_REG32_BIT(HW_USBCTRL0_ENDPTCTRL2,          0x800801C8,__READ_WRITE ,__hw_usbctrl_endptctrl_bits);
__IO_REG32_BIT(HW_USBCTRL0_ENDPTCTRL3,          0x800801CC,__READ_WRITE ,__hw_usbctrl_endptctrl_bits);
__IO_REG32_BIT(HW_USBCTRL0_ENDPTCTRL4,          0x800801D0,__READ_WRITE ,__hw_usbctrl_endptctrl_bits);
__IO_REG32_BIT(HW_USBCTRL0_ENDPTCTRL5,          0x800801D4,__READ_WRITE ,__hw_usbctrl_endptctrl_bits);
__IO_REG32_BIT(HW_USBCTRL0_ENDPTCTRL6,          0x800801D8,__READ_WRITE ,__hw_usbctrl_endptctrl_bits);
__IO_REG32_BIT(HW_USBCTRL0_ENDPTCTRL7,          0x800801DC,__READ_WRITE ,__hw_usbctrl_endptctrl_bits);

/***************************************************************************
 **
 **  USB Controller 1
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_USBCTRL1_ID,                  0x80090000,__READ       ,__hw_usbctrl_id_bits);
__IO_REG32_BIT(HW_USBCTRL1_HWGENERAL,           0x80090004,__READ       ,__hw_usbctrl_hwgeneral_bits);
__IO_REG32_BIT(HW_USBCTRL1_HWHOST,              0x80090008,__READ       ,__hw_usbctrl_hwhost_bits);
__IO_REG32_BIT(HW_USBCTRL1_HWDEVICE,            0x8009000C,__READ       ,__hw_usbctrl_hwdevice_bits);
__IO_REG32_BIT(HW_USBCTRL1_HWTXBUF,             0x80090010,__READ       ,__hw_usbctrl_hwtxbuf_bits);
__IO_REG32_BIT(HW_USBCTRL1_HWRXBUF,             0x80090014,__READ       ,__hw_usbctrl_hwrxbuf_bits);
__IO_REG32_BIT(HW_USBCTRL1_GPTIMER0LD,          0x80090080,__READ_WRITE ,__hw_usbctrl_gptimerxld_bits);
__IO_REG32_BIT(HW_USBCTRL1_GPTIMER0CTRL,        0x80090084,__READ_WRITE ,__hw_usbctrl_gptimerxctrl_bits);
__IO_REG32_BIT(HW_USBCTRL1_GPTIMER1LD,          0x80090088,__READ_WRITE ,__hw_usbctrl_gptimerxld_bits);
__IO_REG32_BIT(HW_USBCTRL1_GPTIMER1CTRL,        0x8009008C,__READ_WRITE ,__hw_usbctrl_gptimerxctrl_bits);
__IO_REG32_BIT(HW_USBCTRL1_SBUSCFG,             0x80090090,__READ_WRITE ,__hw_usbctrl_sbuscfg_bits);
__IO_REG32_BIT(HW_USBCTRL1_CAPLENGTH,           0x80090100,__READ       ,__hw_usbctrl_caplength_bits);
__IO_REG32_BIT(HW_USBCTRL1_HCSPARAMS,           0x80090104,__READ       ,__hw_usbctrl_hcsparams_bits);
__IO_REG32_BIT(HW_USBCTRL1_HCCPARAMS,           0x80090108,__READ       ,__hw_usbctrl_hccparams_bits);
__IO_REG32_BIT(HW_USBCTRL1_DCIVERSION,          0x80090120,__READ       ,__hw_usbctrl_dciversion_bits);
__IO_REG32_BIT(HW_USBCTRL1_DCCPARAMS,           0x80090124,__READ       ,__hw_usbctrl_dccparams_bits);
__IO_REG32_BIT(HW_USBCTRL1_USBCMD,              0x80090140,__READ_WRITE ,__hw_usbctrl_usbcmd_bits);
__IO_REG32_BIT(HW_USBCTRL1_USBSTS,              0x80090144,__READ_WRITE ,__hw_usbctrl_usbsts_bits);
__IO_REG32_BIT(HW_USBCTRL1_USBINTR,             0x80090148,__READ_WRITE ,__hw_usbctrl_usbintr_bits);
__IO_REG32_BIT(HW_USBCTRL1_FRINDEX,             0x8009014C,__READ_WRITE ,__hw_usbctrl_frindex_bits);
__IO_REG32_BIT(HW_USBCTRL1_PERIODICLISTBASE,    0x80090154,__READ_WRITE ,__hw_usbctrl_periodiclistbase_bits);
#define HW_USBCTRL1_DEVICEADDR      HW_USBCTRL1_PERIODICLISTBASE
#define HW_USBCTRL1_DEVICEADDR_bit  HW_USBCTRL1_PERIODICLISTBASE_bit
__IO_REG32_BIT(HW_USBCTRL1_ASYNCLISTADDR,       0x80090158,__READ_WRITE ,__hw_usbctrl_asynclistaddr_bits);
#define HW_USBCTRL1_ENDPOINTLISTADDR      HW_USBCTRL1_ASYNCLISTADDR
#define HW_USBCTRL1_ENDPOINTLISTADDR_bit  HW_USBCTRL1_ASYNCLISTADDR_bit
__IO_REG32_BIT(HW_USBCTRL1_TTCTRL,              0x8009015C,__READ_WRITE ,__hw_usbctrl_ttctrl_bits);
__IO_REG32_BIT(HW_USBCTRL1_BURSTSIZE,           0x80090160,__READ_WRITE ,__hw_usbctrl_burstsize_bits);
__IO_REG32_BIT(HW_USBCTRL1_TXFILLTUNING,        0x80090164,__READ_WRITE ,__hw_usbctrl_txfilltuning_bits);
__IO_REG32_BIT(HW_USBCTRL1_IC_USB,              0x8009016C,__READ_WRITE ,__hw_usbctrl_ic_usb_bits);
__IO_REG32_BIT(HW_USBCTRL1_ULPI,                0x80090170,__READ_WRITE ,__hw_usbctrl_ulpi_bits);
__IO_REG32_BIT(HW_USBCTRL1_ENDPTNAK,            0x80090178,__READ_WRITE ,__hw_usbctrl_endptnak_bits);
__IO_REG32_BIT(HW_USBCTRL1_ENDPTNAKEN,          0x8009017C,__READ_WRITE ,__hw_usbctrl_endptnaken_bits);
__IO_REG32_BIT(HW_USBCTRL1_PORTSC1,             0x80090184,__READ_WRITE ,__hw_usbctrl_portsc_bits);
__IO_REG32_BIT(HW_USBCTRL1_OTGSC,               0x800901A4,__READ_WRITE ,__hw_usbctrl_otgsc_bits);
__IO_REG32_BIT(HW_USBCTRL1_USBMODE,             0x800901A8,__READ_WRITE ,__hw_usbctrl_usbmode_bits);
__IO_REG32_BIT(HW_USBCTRL1_ENDPTSETUPSTAT,      0x800901AC,__READ_WRITE ,__hw_usbctrl_endptsetupstat_bits);
__IO_REG32_BIT(HW_USBCTRL1_ENDPTPRIME,          0x800901B0,__READ_WRITE ,__hw_usbctrl_endptprime_bits);
__IO_REG32_BIT(HW_USBCTRL1_ENDPTFLUSH,          0x800901B4,__READ_WRITE ,__hw_usbctrl_endptflush_bits);
__IO_REG32_BIT(HW_USBCTRL1_ENDPTSTAT,           0x800901B8,__READ       ,__hw_usbctrl_endptstat_bits);
__IO_REG32_BIT(HW_USBCTRL1_ENDPTCOMPLETE,       0x800901BC,__READ_WRITE ,__hw_usbctrl_endptcomplete_bits);
__IO_REG32_BIT(HW_USBCTRL1_ENDPTCTRL0,          0x800901C0,__READ_WRITE ,__hw_usbctrl_endptctrl0_bits);
__IO_REG32_BIT(HW_USBCTRL1_ENDPTCTRL1,          0x800901C4,__READ_WRITE ,__hw_usbctrl_endptctrl_bits);
__IO_REG32_BIT(HW_USBCTRL1_ENDPTCTRL2,          0x800901C8,__READ_WRITE ,__hw_usbctrl_endptctrl_bits);
__IO_REG32_BIT(HW_USBCTRL1_ENDPTCTRL3,          0x800901CC,__READ_WRITE ,__hw_usbctrl_endptctrl_bits);
__IO_REG32_BIT(HW_USBCTRL1_ENDPTCTRL4,          0x800901D0,__READ_WRITE ,__hw_usbctrl_endptctrl_bits);
__IO_REG32_BIT(HW_USBCTRL1_ENDPTCTRL5,          0x800901D4,__READ_WRITE ,__hw_usbctrl_endptctrl_bits);
__IO_REG32_BIT(HW_USBCTRL1_ENDPTCTRL6,          0x800901D8,__READ_WRITE ,__hw_usbctrl_endptctrl_bits);
__IO_REG32_BIT(HW_USBCTRL1_ENDPTCTRL7,          0x800901DC,__READ_WRITE ,__hw_usbctrl_endptctrl_bits);

/***************************************************************************
 **
 **  USB 2.0 PHY 0
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_USBPHY0_PWD,                  0x8007C000,__READ_WRITE ,__hw_usbphy_pwd_bits);
__IO_REG32_BIT(HW_USBPHY0_PWD_SET,              0x8007C004,__WRITE      ,__hw_usbphy_pwd_bits);
__IO_REG32_BIT(HW_USBPHY0_PWD_CLR,              0x8007C008,__WRITE      ,__hw_usbphy_pwd_bits);
__IO_REG32_BIT(HW_USBPHY0_PWD_TOG,              0x8007C00C,__WRITE      ,__hw_usbphy_pwd_bits);
__IO_REG32_BIT(HW_USBPHY0_TX,                   0x8007C010,__READ_WRITE ,__hw_usbphy_tx_bits);
__IO_REG32_BIT(HW_USBPHY0_TX_SET,               0x8007C014,__WRITE      ,__hw_usbphy_tx_bits);
__IO_REG32_BIT(HW_USBPHY0_TX_CLR,               0x8007C018,__WRITE      ,__hw_usbphy_tx_bits);
__IO_REG32_BIT(HW_USBPHY0_TX_TOG,               0x8007C01C,__WRITE      ,__hw_usbphy_tx_bits);
__IO_REG32_BIT(HW_USBPHY0_RX,                   0x8007C020,__READ_WRITE ,__hw_usbphy_rx_bits);
__IO_REG32_BIT(HW_USBPHY0_RX_SET,               0x8007C024,__WRITE      ,__hw_usbphy_rx_bits);
__IO_REG32_BIT(HW_USBPHY0_RX_CLR,               0x8007C028,__WRITE      ,__hw_usbphy_rx_bits);
__IO_REG32_BIT(HW_USBPHY0_RX_TOG,               0x8007C02C,__WRITE      ,__hw_usbphy_rx_bits);
__IO_REG32_BIT(HW_USBPHY0_CTRL,                 0x8007C030,__READ_WRITE ,__hw_usbphy_ctrl_bits);
__IO_REG32_BIT(HW_USBPHY0_CTRL_SET,             0x8007C034,__WRITE      ,__hw_usbphy_ctrl_bits);
__IO_REG32_BIT(HW_USBPHY0_CTRL_CLR,             0x8007C038,__WRITE      ,__hw_usbphy_ctrl_bits);
__IO_REG32_BIT(HW_USBPHY0_CTRL_TOG,             0x8007C03C,__WRITE      ,__hw_usbphy_ctrl_bits);
__IO_REG32_BIT(HW_USBPHY0_STATUS,               0x8007C040,__READ_WRITE ,__hw_usbphy_status_bits);
__IO_REG32_BIT(HW_USBPHY0_DEBUG,                0x8007C050,__READ_WRITE ,__hw_usbphy_debug_bits);
__IO_REG32_BIT(HW_USBPHY0_DEBUG_SET,            0x8007C054,__WRITE      ,__hw_usbphy_debug_bits);
__IO_REG32_BIT(HW_USBPHY0_DEBUG_CLR,            0x8007C058,__WRITE      ,__hw_usbphy_debug_bits);
__IO_REG32_BIT(HW_USBPHY0_DEBUG_TOG,            0x8007C05C,__WRITE      ,__hw_usbphy_debug_bits);
__IO_REG32_BIT(HW_USBPHY0_DEBUG0_STATUS,        0x8007C060,__READ       ,__hw_usbphy_debug0_status_bits);
__IO_REG32_BIT(HW_USBPHY0_DEBUG1,               0x8007C070,__READ_WRITE ,__hw_usbphy_debug1_bits);
__IO_REG32_BIT(HW_USBPHY0_DEBUG1_SET,           0x8007C074,__WRITE      ,__hw_usbphy_debug1_bits);
__IO_REG32_BIT(HW_USBPHY0_DEBUG1_CLR,           0x8007C078,__WRITE      ,__hw_usbphy_debug1_bits);
__IO_REG32_BIT(HW_USBPHY0_DEBUG1_TOG,           0x8007C07C,__WRITE      ,__hw_usbphy_debug1_bits);
__IO_REG32_BIT(HW_USBPHY0_VERSION,              0x8007C080,__READ       ,__hw_usbphy_version_bits);
__IO_REG32_BIT(HW_USBPHY0_IP,                   0x8007C090,__READ_WRITE ,__hw_usbphy_ip_bits);
__IO_REG32_BIT(HW_USBPHY0_IP_SET,               0x8007C094,__WRITE      ,__hw_usbphy_ip_bits);
__IO_REG32_BIT(HW_USBPHY0_IP_CLR,               0x8007C098,__WRITE      ,__hw_usbphy_ip_bits);
__IO_REG32_BIT(HW_USBPHY0_IP_TOG,               0x8007C09C,__WRITE      ,__hw_usbphy_ip_bits);

/***************************************************************************
 **
 **  USB 2.0 PHY 1
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_USBPHY1_PWD,                  0x8007E000,__READ_WRITE ,__hw_usbphy_pwd_bits);
__IO_REG32_BIT(HW_USBPHY1_PWD_SET,              0x8007E004,__WRITE      ,__hw_usbphy_pwd_bits);
__IO_REG32_BIT(HW_USBPHY1_PWD_CLR,              0x8007E008,__WRITE      ,__hw_usbphy_pwd_bits);
__IO_REG32_BIT(HW_USBPHY1_PWD_TOG,              0x8007E00C,__WRITE      ,__hw_usbphy_pwd_bits);
__IO_REG32_BIT(HW_USBPHY1_TX,                   0x8007E010,__READ_WRITE ,__hw_usbphy_tx_bits);
__IO_REG32_BIT(HW_USBPHY1_TX_SET,               0x8007E014,__WRITE      ,__hw_usbphy_tx_bits);
__IO_REG32_BIT(HW_USBPHY1_TX_CLR,               0x8007E018,__WRITE      ,__hw_usbphy_tx_bits);
__IO_REG32_BIT(HW_USBPHY1_TX_TOG,               0x8007E01C,__WRITE      ,__hw_usbphy_tx_bits);
__IO_REG32_BIT(HW_USBPHY1_RX,                   0x8007E020,__READ_WRITE ,__hw_usbphy_rx_bits);
__IO_REG32_BIT(HW_USBPHY1_RX_SET,               0x8007E024,__WRITE      ,__hw_usbphy_rx_bits);
__IO_REG32_BIT(HW_USBPHY1_RX_CLR,               0x8007E028,__WRITE      ,__hw_usbphy_rx_bits);
__IO_REG32_BIT(HW_USBPHY1_RX_TOG,               0x8007E02C,__WRITE      ,__hw_usbphy_rx_bits);
__IO_REG32_BIT(HW_USBPHY1_CTRL,                 0x8007E030,__READ_WRITE ,__hw_usbphy_ctrl_bits);
__IO_REG32_BIT(HW_USBPHY1_CTRL_SET,             0x8007E034,__WRITE      ,__hw_usbphy_ctrl_bits);
__IO_REG32_BIT(HW_USBPHY1_CTRL_CLR,             0x8007E038,__WRITE      ,__hw_usbphy_ctrl_bits);
__IO_REG32_BIT(HW_USBPHY1_CTRL_TOG,             0x8007E03C,__WRITE      ,__hw_usbphy_ctrl_bits);
__IO_REG32_BIT(HW_USBPHY1_STATUS,               0x8007E040,__READ_WRITE ,__hw_usbphy_status_bits);
__IO_REG32_BIT(HW_USBPHY1_DEBUG,                0x8007E050,__READ_WRITE ,__hw_usbphy_debug_bits);
__IO_REG32_BIT(HW_USBPHY1_DEBUG_SET,            0x8007E054,__WRITE      ,__hw_usbphy_debug_bits);
__IO_REG32_BIT(HW_USBPHY1_DEBUG_CLR,            0x8007E058,__WRITE      ,__hw_usbphy_debug_bits);
__IO_REG32_BIT(HW_USBPHY1_DEBUG_TOG,            0x8007E05C,__WRITE      ,__hw_usbphy_debug_bits);
__IO_REG32_BIT(HW_USBPHY1_DEBUG0_STATUS,        0x8007E060,__READ       ,__hw_usbphy_debug0_status_bits);
__IO_REG32_BIT(HW_USBPHY1_DEBUG1,               0x8007E070,__READ_WRITE ,__hw_usbphy_debug1_bits);
__IO_REG32_BIT(HW_USBPHY1_DEBUG1_SET,           0x8007E074,__WRITE      ,__hw_usbphy_debug1_bits);
__IO_REG32_BIT(HW_USBPHY1_DEBUG1_CLR,           0x8007E078,__WRITE      ,__hw_usbphy_debug1_bits);
__IO_REG32_BIT(HW_USBPHY1_DEBUG1_TOG,           0x8007E07C,__WRITE      ,__hw_usbphy_debug1_bits);
__IO_REG32_BIT(HW_USBPHY1_VERSION,              0x8007E080,__READ       ,__hw_usbphy_version_bits);
__IO_REG32_BIT(HW_USBPHY1_IP,                   0x8007E090,__READ_WRITE ,__hw_usbphy_ip_bits);
__IO_REG32_BIT(HW_USBPHY1_IP_SET,               0x8007E094,__WRITE      ,__hw_usbphy_ip_bits);
__IO_REG32_BIT(HW_USBPHY1_IP_CLR,               0x8007E098,__WRITE      ,__hw_usbphy_ip_bits);
__IO_REG32_BIT(HW_USBPHY1_IP_TOG,               0x8007E09C,__WRITE      ,__hw_usbphy_ip_bits);

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
__IO_REG32_BIT(HW_LCDIF_CTRL2,                   0x80030020,__READ_WRITE ,__hw_lcdif_ctrl2_bits);
__IO_REG32_BIT(HW_LCDIF_CTRL2_SET,               0x80030024,__WRITE      ,__hw_lcdif_ctrl2_bits);
__IO_REG32_BIT(HW_LCDIF_CTRL2_CLR,               0x80030028,__WRITE      ,__hw_lcdif_ctrl2_bits);
__IO_REG32_BIT(HW_LCDIF_CTRL2_TOG,               0x8003002C,__WRITE      ,__hw_lcdif_ctrl2_bits);
__IO_REG32_BIT(HW_LCDIF_TRANSFER_COUNT,          0x80030030,__READ_WRITE ,__hw_lcdif_transfer_count_bits);
__IO_REG32(    HW_LCDIF_CUR_BUF,                 0x80030040,__READ_WRITE );
__IO_REG32(    HW_LCDIF_NEXT_BUF,                0x80030050,__READ_WRITE );
__IO_REG32_BIT(HW_LCDIF_TIMING,                  0x80030060,__READ_WRITE ,__hw_lcdif_timing_bits);
__IO_REG32_BIT(HW_LCDIF_VDCTRL0,                 0x80030070,__READ_WRITE ,__hw_lcdif_vdctrl0_bits);
__IO_REG32(    HW_LCDIF_VDCTRL1,                 0x80030080,__READ_WRITE );
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
__IO_REG32_BIT(HW_LCDIF_DATA,                    0x80030180,__READ_WRITE ,__hw_lcdif_data_bits);
__IO_REG32(    HW_LCDIF_BM_ERROR_STAT,           0x80030190,__READ_WRITE );
__IO_REG32(    HW_LCDIF_CRC_STAT,                0x800301A0,__READ_WRITE );
__IO_REG32_BIT(HW_LCDIF_STAT,                    0x800301B0,__READ       ,__hw_lcdif_stat_bits);
__IO_REG32_BIT(HW_LCDIF_VERSION,                 0x800301C0,__READ       ,__hw_lcdif_version_bits);
__IO_REG32_BIT(HW_LCDIF_DEBUG0,                  0x800301D0,__READ       ,__hw_lcdif_debug0_bits);
__IO_REG32_BIT(HW_LCDIF_DEBUG1,                  0x800301E0,__READ       ,__hw_lcdif_debug1_bits);
__IO_REG32(    HW_LCDIF_DEBUG2,                  0x800301F0,__READ       );

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
 **  SAIF0
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_SAIF0_CTRL,                   0x80042000,__READ_WRITE ,__hw_saif_ctrl_bits);
__IO_REG32_BIT(HW_SAIF0_CTRL_SET,               0x80042004,__WRITE      ,__hw_saif_ctrl_bits);
__IO_REG32_BIT(HW_SAIF0_CTRL_CLR,               0x80042008,__WRITE      ,__hw_saif_ctrl_bits);
__IO_REG32_BIT(HW_SAIF0_CTRL_TOG,               0x8004200C,__WRITE      ,__hw_saif_ctrl_bits);
__IO_REG32_BIT(HW_SAIF0_STAT,                   0x80042010,__READ       ,__hw_saif_stat_bits);
__IO_REG32_BIT(HW_SAIF0_STAT_CLR,               0x80042018,__WRITE      ,__hw_saif_stat_bits);
__IO_REG32_BIT(HW_SAIF0_DATA,                   0x80042020,__READ_WRITE ,__hw_saif_data_bits);
__IO_REG32_BIT(HW_SAIF0_DATA_SET,               0x80042024,__WRITE      ,__hw_saif_data_bits);
__IO_REG32_BIT(HW_SAIF0_DATA_CLR,               0x80042028,__WRITE      ,__hw_saif_data_bits);
__IO_REG32_BIT(HW_SAIF0_DATA_TOG,               0x8004202C,__WRITE      ,__hw_saif_data_bits);
__IO_REG32_BIT(HW_SAIF0_VERSION,                0x80042030,__READ       ,__hw_saif_version_bits);

/***************************************************************************
 **
 **  SAIF1
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_SAIF1_CTRL,                   0x80046000,__READ_WRITE ,__hw_saif_ctrl_bits);
__IO_REG32_BIT(HW_SAIF1_CTRL_SET,               0x80046004,__WRITE      ,__hw_saif_ctrl_bits);
__IO_REG32_BIT(HW_SAIF1_CTRL_CLR,               0x80046008,__WRITE      ,__hw_saif_ctrl_bits);
__IO_REG32_BIT(HW_SAIF1_CTRL_TOG,               0x8004600C,__WRITE      ,__hw_saif_ctrl_bits);
__IO_REG32_BIT(HW_SAIF1_STAT,                   0x80046010,__READ       ,__hw_saif_stat_bits);
__IO_REG32_BIT(HW_SAIF1_STAT_CLR,               0x80046018,__WRITE      ,__hw_saif_stat_bits);
__IO_REG32_BIT(HW_SAIF1_DATA,                   0x80046020,__READ_WRITE ,__hw_saif_data_bits);
__IO_REG32_BIT(HW_SAIF1_DATA_SET,               0x80046024,__WRITE      ,__hw_saif_data_bits);
__IO_REG32_BIT(HW_SAIF1_DATA_CLR,               0x80046028,__WRITE      ,__hw_saif_data_bits);
__IO_REG32_BIT(HW_SAIF1_DATA_TOG,               0x8004602C,__WRITE      ,__hw_saif_data_bits);
__IO_REG32_BIT(HW_SAIF1_VERSION,                0x80046030,__READ       ,__hw_saif_version_bits);

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
 **  HSADC
 **
 ***************************************************************************/
__IO_REG32_BIT(HW_HSADC_CTRL0,                    0x80002000,__READ_WRITE ,__hw_hsadc_ctrl0_bits);
__IO_REG32_BIT(HW_HSADC_CTRL0_SET,                0x80002004,__WRITE      ,__hw_hsadc_ctrl0_bits);
__IO_REG32_BIT(HW_HSADC_CTRL0_CLR,                0x80002008,__WRITE      ,__hw_hsadc_ctrl0_bits);
__IO_REG32_BIT(HW_HSADC_CTRL0_TOG,                0x8000200C,__WRITE      ,__hw_hsadc_ctrl0_bits);
__IO_REG32_BIT(HW_HSADC_CTRL1,                    0x80002010,__READ_WRITE ,__hw_hsadc_ctrl1_bits);
__IO_REG32_BIT(HW_HSADC_CTRL1_SET,                0x80002014,__WRITE      ,__hw_hsadc_ctrl1_bits);
__IO_REG32_BIT(HW_HSADC_CTRL1_CLR,                0x80002018,__WRITE      ,__hw_hsadc_ctrl1_bits);
__IO_REG32_BIT(HW_HSADC_CTRL1_TOG,                0x8000201C,__WRITE      ,__hw_hsadc_ctrl1_bits);
__IO_REG32_BIT(HW_HSADC_CTRL2,                    0x80002020,__READ_WRITE ,__hw_hsadc_ctrl2_bits);
__IO_REG32_BIT(HW_HSADC_CTRL2_SET,                0x80002024,__WRITE      ,__hw_hsadc_ctrl2_bits);
__IO_REG32_BIT(HW_HSADC_CTRL2_CLR,                0x80002028,__WRITE      ,__hw_hsadc_ctrl2_bits);
__IO_REG32_BIT(HW_HSADC_CTRL2_TOG,                0x8000202C,__WRITE      ,__hw_hsadc_ctrl2_bits);
__IO_REG32(    HW_HSADC_SEQUENCE_SAMPLES_NUM,     0x80002030,__READ_WRITE );
__IO_REG32(    HW_HSADC_SEQUENCE_SAMPLES_NUM_SET, 0x80002034,__WRITE      );
__IO_REG32(    HW_HSADC_SEQUENCE_SAMPLES_NUM_CLR, 0x80002038,__WRITE      );
__IO_REG32(    HW_HSADC_SEQUENCE_SAMPLES_NUM_TOG, 0x8000203C,__WRITE      );
__IO_REG32(    HW_HSADC_SEQUENCE_NUM,             0x80002040,__READ_WRITE );
__IO_REG32(    HW_HSADC_SEQUENCE_NUM_SET,         0x80002044,__WRITE      );
__IO_REG32(    HW_HSADC_SEQUENCE_NUM_CLR,         0x80002048,__WRITE      );
__IO_REG32(    HW_HSADC_SEQUENCE_NUM_TOG,         0x8000204C,__WRITE      );
__IO_REG32(    HW_HSADC_FIFO_DATA,                0x80002050,__READ       );
__IO_REG32_BIT(HW_HSADC_DBG_INFO0,                0x80002060,__READ       ,__hw_hsadc_dbg_info0_bits);
__IO_REG32(    HW_HSADC_DBG_INFO1,                0x80002070,__READ       );
__IO_REG32(    HW_HSADC_DBG_INFO2,                0x80002080,__READ       );
__IO_REG32_BIT(HW_HSADC_VERSION,                  0x800020B0,__READ       ,__hw_hsadc_version_bits);

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
__IO_REG32_BIT(HW_LRADC_CH7,                    0x800500C0,__READ_WRITE ,__hw_lradc_chx_bits);
__IO_REG32_BIT(HW_LRADC_CH7_SET,                0x800500C4,__WRITE      ,__hw_lradc_chx_bits);
__IO_REG32_BIT(HW_LRADC_CH7_CLR,                0x800500C8,__WRITE      ,__hw_lradc_chx_bits);
__IO_REG32_BIT(HW_LRADC_CH7_TOG,                0x800500CC,__WRITE      ,__hw_lradc_chx_bits);
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
__IO_REG32_BIT(HW_LRADC_THRESHOLD0,             0x80050150,__READ_WRITE ,__hw_lradc_threshold_bits);
__IO_REG32_BIT(HW_LRADC_VERSION,                0x80050170,__READ       ,__hw_lradc_version_bits);

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
 **   MCIX28 APBH DMA channels
 **
 ***************************************************************************/
#define APBH_DMA_SSP0           0
#define APBH_DMA_SSP1           1
#define APBH_DMA_SSP2           2
#define APBH_DMA_SSP3           3
#define APBH_DMA_GPMI0          4
#define APBH_DMA_GPMI1          5
#define APBH_DMA_GPMI2          6
#define APBH_DMA_GPMI3          7
#define APBH_DMA_GPMI4          8
#define APBH_DMA_GPMI5          9
#define APBH_DMA_GPMI6          10
#define APBH_DMA_GPMI7          11
#define APBH_DMA_HSADC          12
#define APBH_DMA_LCDIF          13

/***************************************************************************
 **
 **   MCIX28 APBX DMA channels
 **
 ***************************************************************************/
#define APBX_DMA_AUART4RX       0
#define APBX_DMA_AUART4TX       1
#define APBX_DMA_SPDIFTX        2
#define APBX_DMA_SAIF0          4
#define APBX_DMA_SAIF1          5
#define APBX_DMA_I2C0           6
#define APBX_DMA_I2C1           7
#define APBX_DMA_AUART0RX       8
#define APBX_DMA_AUART0TX       9
#define APBX_DMA_AUART1RX       10
#define APBX_DMA_AUART1TX       11
#define APBX_DMA_AUART2RX       12
#define APBX_DMA_AUART2TX       13
#define APBX_DMA_AUART3RX       14
#define APBX_DMA_AUART3TX       15

/***************************************************************************
 **
 **   MCIX28 interrupt sources
 **
 ***************************************************************************/
#define INT_BATT_BROWNOUT       0             /* Power module battery brownout detect */
#define INT_VDDD_BROWNOUT       1             /* Power module VDDD brownout detect */
#define INT_VDDIO_BROWNOU       2             /* Power module VDDIO brownout detect */
#define INT_VDDA_BROWNOUT       3             /* Power module VDDA brownout detect */
#define INT_VDD5V_DROOP         4             /* 5V Droop */
#define INT_DCDC4P2_BROWNOUT    5             /* 4.2V regulated supply brown-out */
#define INT_VDD5V               6             /* 5V connect or disconnect also OTG 4.2V */
#define INT_CAN0                8             /* CAN 0 */
#define INT_CAN1                9             /* CAN 1 */
#define INT_LRADC_TOUCH        10             /* (Touch Screen) Touch detection */
#define INT_HSADC              13             /* HSADC */
#define INT_LRADC_THRESH0      14             /* LRADC0 Threshold */
#define INT_LRADC_THRESH1      15             /* LRADC1 Threshold */
#define INT_LRADC_CH0          16             /* LRADC Channel 0 conversion complete */
#define INT_LRADC_CH1          17             /* LRADC Channel 1 conversion complete */
#define INT_LRADC_CH2          18             /* LRADC Channel 2 conversion complete */
#define INT_LRADC_CH3          19             /* LRADC Channel 3 conversion complete */
#define INT_LRADC_CH4          20             /* LRADC Channel 4 conversion complete */
#define INT_LRADC_CH5          21             /* LRADC Channel 5 conversion complete */
#define INT_LRADC_CH6          22             /* LRADC Channel 6 conversion complete */
#define INT_LRADC_CH7          23             /* LRADC Channel 7 conversion complete */
#define INT_LRADC_BUTTON0      24             /* LRADC Channel 0 button detection */
#define INT_LRADC_BUTTON1      25             /* LRADC Channel 1 button detection */
#define INT_PERFMON            27             /* Performance monitor */
#define INT_RTC_1MSEC          28             /* RTC 1ms event */
#define INT_RTC_ALARM          29             /* RTC alarm event */
#define INT_COMMS              31             /* JTAG debug communications port */
#define INT_EMI_ERROR          32             /* External memory controller */
#define INT_LCDIF              38             /* LCDIF */
#define INT_PXP                39             /* PXP */
#define INT_BCH                41             /* BCH consolidated */
#define INT_GPMI               42             /* GPMI internal error and status */
#define INT_SPDIF_ERROR        45             /* SPDIF FIFO error */
#define INT_DUART              47             /* Debug UART */
#define INT_TIMER0             48             /* Timer 0 */
#define INT_TIMER1             49             /* Timer 1 */
#define INT_TIMER2             50             /* Timer 2 */
#define INT_TIMER3             51             /* Timer 3 */
#define INT_DCP_VMI            52             /* DCP Channel 0 virtual memory page copy */
#define INT_DCP                53             /* DCP (per channel and CSC) */
#define INT_DCP_SECURE         54             /* DCP secure */
#define INT_SAIF1              58             /* SAIF1 FIFO & Service error */
#define INT_SAIF0              59             /* SAIF0 FIFO & Service error */
#define INT_SPDIF_DMA          66             /* SPDIF DMA channel */
#define INT_I2C0_DMA           68             /* I2C0 DMA channel */
#define INT_I2C1_DMA           69             /* I2C1 DMA channel */
#define INT_AUART0_RX_DMA      70             /* Application UART0 receiver DMA channel */
#define INT_AUART0_TX_DMA      71             /* Application UART0 transmitter DMA channel */
#define INT_AUART1_RX_DMA      72             /* Application UART1 receiver DMA channel */
#define INT_AUART1_TX_DMA      73             /* Application UART1 transmitter DMA channel */
#define INT_AUART2_RX_DMA      74             /* Application UART2 receiver DMA channel */
#define INT_AUART2_TX_DMA      75             /* Application UART2 transmitter DMA channel */
#define INT_AUART3_RX_DMA      76             /* Application UART3 receiver DMA channel */
#define INT_AUART3_TX_DMA      77             /* Application UART3 transmitter DMA channel */
#define INT_AUART4_RX_DMA      78             /* Application UART4 receiver DMA channel */
#define INT_AUART4_TX_DMA      79             /* Application UART4 transmitter DMA channel */
#define INT_SAIF0_DMA          80             /* SAIF0 DMA channel */
#define INT_SAIF1_DMA          81             /* SAIF1 DMA channel */
#define INT_SSP0_DMA           82             /* SSP0 DMA channel */
#define INT_SSP1_DMA           83             /* SSP1 DMA channel */
#define INT_SSP2_DMA           84             /* SSP2 DMA channel */
#define INT_SSP3_DMA           85             /* SSP3 DMA channel */
#define INT_LCDIF_DMA          86             /* LCDIF DMA channel */
#define INT_HSADC_DMA          87             /* HSADC DMA channel */
#define INT_GPMI_DMA           88             /* GPMI DMA channel */
#define INT_DIGCTL_DEBUG_TRAP  89             /* Layer 0 or Layer 3 AHB address access trap */
#define INT_USB1               92             /* USB1 */
#define INT_USB0               93             /* USB0 */
#define INT_USB1_WAKEUP        94             /* UTM1 */
#define INT_USB0_WAKEUP        95             /* UTM0 */
#define INT_SSP0_ERROR         96             /* SSP0 device-level error and status */
#define INT_SSP1_ERROR         97             /* SSP1 device-level error and status */
#define INT_SSP2_ERROR         98             /* SSP2 device-level error and status */
#define INT_SSP3_ERROR         99             /* SSP3 device-level error and status */
#define INT_ENET_SWI          100             /* Switch */
#define INT_ENET_MAC0         101             /* MAC0 */
#define INT_ENET_MAC1         102             /* MAC1 */
#define INT_ENET_MAC0_1588    103             /* 1588 of MAC0 */
#define INT_ENET_MAC1_1588    104             /* 1588 of MAC1 */
#define INT_I2C1_ERROR        110             /* I2C1 device detected errors and line conditions */
#define INT_I2C0_ERROR        111             /* I2C0 device detected errors and line conditions */
#define INT_AUART0            112             /* Application UART0 internal error */
#define INT_AUART1            113             /* Application UART1 internal error */
#define INT_AUART2            114             /* Application UART2 internal error */
#define INT_AUART3            115             /* Application UART3 internal error */
#define INT_AUART4            116             /* Application UART4 internal error */
#define INT_PINCTRL5          122             /* GPIO bank 5 interrupt */
#define INT_PINCTRL4          123             /* GPIO bank 4 interrupt */
#define INT_PINCTRL3          124             /* GPIO bank 3 interrupt */
#define INT_PINCTRL2          125             /* GPIO bank 2 interrupt */
#define INT_PINCTRL1          126             /* GPIO bank 1 interrupt */
#define INT_PINCTRL0          127             /* GPIO bank 0 interrupt */

#endif    /* __MCIX28_H */
