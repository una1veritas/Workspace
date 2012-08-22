/************************************************************************
* Device     : RX/RX600/RX62N
* File Name  : vecttbl.c
* Abstract   : Initialize of Vector Table.
* History    : 1.00  (2010-03-05)  [Hardware Manual Revision : 0.50]
* NOTE       : THIS IS A TYPICAL EXAMPLE.
* Copyright (C) 2009 Renesas Electronics Corporation.
* and Renesas Solutions Corporation. All rights reserved.
*
* Copyright (C) 2011 Tokushu Denshi Kairo Inc.
*
************************************************************************/

/************************************************************************
GCC用の割り込みベクタ & ハンドラ

このファイルは、割り込みベクタと例外ベクタテーブルを定義します。
各ハンドラを書き換えることで、割り込み動作をカスタマイズすることができます。

未定義の割り込みや例外が発生した場合、__STOPというマクロ(実体は何もしない)
が実行されます。STOP_ON_EXCEPTIOというマクロを定義しておくと、
そこでCPUが無限ループして停止するようになっています。

                                                                tokuden
************************************************************************/

#include "intvect.h"
#include "iodefine_gcc62n.h"

#ifdef STOP_ON_EXCEPTION
#define __STOP {while(1);}
#else
#define __STOP 
#endif

#define FIXEDVECT_SECTION    __attribute__ ((section (".fvectors")))
#define RELVECT_SECTION      __attribute__ ((section (".rvectors")))
typedef void (*FUNC_PTR) (void);

#ifdef __cpulsplus
extern "C" {
#endif

FUNC_PTR const Fixed_Vectors[] FIXEDVECT_SECTION = {

//;0xffffffd0  Exception(Supervisor Instruction)
    Excep_SuperVisorInst,
//;0xffffffd4  Reserved
    Dummy,
//;0xffffffd8  Reserved
    Dummy,
//;0xffffffdc  Exception(Undefined Instruction)
    Excep_UndefinedInst,
//;0xffffffe0  Reserved
    Dummy,
//;0xffffffe4  Exception(Floating Point)
    Excep_FloatingPoint,
//;0xffffffe8  Reserved
    Dummy,
//;0xffffffec  Reserved
    Dummy,
//;0xfffffff0  Reserved
    Dummy,
//;0xfffffff4  Reserved
    Dummy,
//;0xfffffff8  NMI
    NonMaskableInterrupt,
//;0xfffffffc  RESET
//;<<VECTOR DATA START (POWER ON RESET)>>
//;Power On Reset PC
PowerON_Reset_PC                                                                                                                             
//;<<VECTOR DATA END (POWER ON RESET)>>
};

FUNC_PTR const Relocatable_Vectors[256] RELVECT_SECTION = {
	Excep_BRK,	// 0
	Dummy,		// 1
	Dummy,		// 2
	Dummy,		// 3
	Dummy,		// 4
	Dummy,		// 5
	Dummy,		// 6
	Dummy,		// 7
	Dummy,		// 8
	Dummy,		// 9
	Dummy,		// 10
	Dummy,		// 11
	Dummy,		// 12
	Dummy,		// 13
	Dummy,		// 14
	Dummy,		// 15
	Excep_BUSERR,		// 16
	Dummy,		// 17
	Dummy,		// 18
	Dummy,		// 19
	Dummy,		// 20
	Excep_FCU_FCUERR,		// 21
	Dummy,		// 22
	Excep_FCU_FRDYI,		// 23
	Dummy,		// 24
	Dummy,		// 25
	Dummy,		// 26
	Excep_ICU_SWINT,		// 27
	Excep_CMTU0_CMT0,		// 28
	Excep_CMTU0_CMT1,		// 29
	Excep_CMTU1_CMT2,		// 30
	Excep_CMTU1_CMT3,		// 31
	Excep_ETHER_EINT,		// 32
	Dummy,		// 33
	Dummy,		// 34
	Dummy,		// 35
	Excep_USB0_D0FIFO0,		// 36
	Excep_USB0_D1FIFO0,		// 37
	Excep_USB0_USBI0,		// 38
	Dummy,		// 39
	Excep_USB1_D0FIFO1,		// 40
	Excep_USB1_D1FIFO1,		// 41
	Excep_USB1_USBI1,		// 42
	Dummy,		// 43
	Excep_RSPI0_SPEI0,		// 44
	Excep_RSPI0_SPRI0,		// 45
	Excep_RSPI0_SPTI0,		// 46
	Excep_RSPI0_SPII0,		// 47
	Excep_RSPI1_SPEI1,		// 48
	Excep_RSPI1_SPRI1,		// 49
	Excep_RSPI1_SPTI1,		// 50
	Excep_RSPI1_SPII1,		// 51
	Dummy,		// 52
	Dummy,		// 53
	Dummy,		// 54
	Dummy,		// 55
	Excep_CAN0_ERS0,		// 56
	Excep_CAN0_RXF0,		// 57
	Excep_CAN0_TXF0,		// 58
	Excep_CAN0_RXM0,		// 59
	Excep_CAN0_TXM0,		// 60
	Dummy,		// 61
	Excep_RTC_PRD,		// 62
	Excep_RTC_CUP,		// 63
	Excep_IRQ0,		// 64
	Excep_IRQ1,		// 65
	Excep_IRQ2,		// 66
	Excep_IRQ3,		// 67
	Excep_IRQ4,		// 68
	Excep_IRQ5,		// 69
	Excep_IRQ6,		// 70
	Excep_IRQ7,		// 71
	Excep_IRQ8,		// 72
	Excep_IRQ9,		// 73
	Excep_IRQ10,		// 74
	Excep_IRQ11,		// 75
	Excep_IRQ12,		// 76
	Excep_IRQ13,		// 77
	Excep_IRQ14,		// 78
	Excep_IRQ15,		// 79
	Dummy,		// 80
	Dummy,		// 81
	Dummy,		// 82
	Dummy,		// 83
	Dummy,		// 84
	Dummy,		// 85
	Dummy,		// 86
	Dummy,		// 87
	Dummy,		// 88
	Dummy,		// 89
	Excep_USB_USBR0,		// 90
	Excep_USB_USBR1,		// 91
	Excep_RTC_ALM,		// 92
	Dummy,		// 93
	Dummy,		// 94
	Dummy,		// 95
	Excep_WDT_WOVI,		// 96
	Dummy,		// 97
	Excep_AD0_ADI0,		// 98
	Excep_AD1_ADI1,		// 99
	Dummy,		// 100
	Dummy,		// 101
	Excep_S12AD_ADI12,		// 102
	Dummy,		// 103
	Dummy,		// 104
	Dummy,		// 105
	Dummy,		// 106
	Dummy,		// 107
	Dummy,		// 108
	Dummy,		// 109
	Dummy,		// 110
	Dummy,		// 111
	Dummy,		// 112
	Dummy,		// 113
	Excep_MTU0_TGIA0,		// 114
	Excep_MTU0_TGIB0,		// 115
	Excep_MTU0_TGIC0,		// 116
	Excep_MTU0_TGID0,		// 117
	Excep_MTU0_TCIV0,		// 118
	Excep_MTU0_TGIE0,		// 119
	Excep_MTU0_TGIF0,		// 120
	Excep_MTU1_TGIA1,		// 121
	Excep_MTU1_TGIB1,		// 122
	Excep_MTU1_TCIV1,		// 123
	Excep_MTU1_TCIU1,		// 124
	Excep_MTU2_TGIA2,		// 125
	Excep_MTU2_TGIB2,		// 126
	Excep_MTU2_TCIV2,		// 127
	Excep_MTU2_TCIU2,		// 128
	Excep_MTU3_TGIA3,		// 129
	Excep_MTU3_TGIB3,		// 130
	Excep_MTU3_TGIC3,		// 131
	Excep_MTU3_TGID3,		// 132
	Excep_MTU3_TCIV3,		// 133
	Excep_MTU4_TGIA4,		// 134
	Excep_MTU4_TGIB4,		// 135
	Excep_MTU4_TGIC4,		// 136
	Excep_MTU4_TGID4,		// 137
	Excep_MTU4_TCIV4,		// 138
	Excep_MTU5_TCIU5,		// 139
	Excep_MTU5_TCIV5,		// 140
	Excep_MTU5_TCIW5,		// 141
	Excep_MTU6_TGIA6,		// 142
	Excep_MTU6_TGIB6,		// 143
	Excep_MTU6_TGIC6,		// 144
	Excep_MTU6_TGID6,		// 145
	Excep_MTU6_TCIV6,		// 146
	Excep_MTU6_TGIE6,		// 147
	Excep_MTU6_TGIF6,		// 148
	Excep_MTU7_TGIA7,		// 149
	Excep_MTU7_TGIB7,		// 150
	Excep_MTU7_TCIV7,		// 151
	Excep_MTU7_TCIU7,		// 152
	Excep_MTU8_TGIA8,		// 153
	Excep_MTU8_TGIB8,		// 154
	Excep_MTU8_TCIV8,		// 155
	Excep_MTU8_TCIU8,		// 156
	Excep_MTU9_TGIA9,		// 157
	Excep_MTU9_TGIB9,		// 158
	Excep_MTU9_TGIC9,		// 159
	Excep_MTU9_TGID9,		// 160
	Excep_MTU9_TCIV9,		// 161
	Excep_MTU10_TGIA10,		// 162
	Excep_MTU10_TGIB10,		// 163
	Excep_MTU10_TGIC10,		// 164
	Excep_MTU10_TGID10,		// 165
	Excep_MTU10_TCIV10,		// 166
	Excep_MTU11_TCIU11,		// 167
	Excep_MTU11_TCIV11,		// 168
	Excep_MTU11_TCIW11,		// 169
	Excep_POE_OEI1,		// 170
	Excep_POE_OEI2,		// 171
	Excep_POE_OEI3,		// 172
	Excep_POE_OEI4,		// 173
	Excep_TMR0_CMI0A,		// 174
	Excep_TMR0_CMI0B,		// 175
	Excep_TMR0_OV0I,		// 176
	Excep_TMR1_CMI1A,		// 177
	Excep_TMR1_CMI1B,		// 178
	Excep_TMR1_OV1I,		// 179
	Excep_TMR2_CMI2A,		// 180
	Excep_TMR2_CMI2B,		// 181
	Excep_TMR2_OV2I,		// 182
	Excep_TMR3_CMI3A,		// 183
	Excep_TMR3_CMI3B,		// 184
	Excep_TMR3_OV3I,		// 185
	Dummy,		// 186
	Dummy,		// 187
	Dummy,		// 188
	Dummy,		// 189
	Dummy,		// 190
	Dummy,		// 191
	Dummy,		// 192
	Dummy,		// 193
	Dummy,		// 194
	Dummy,		// 195
	Dummy,		// 196
	Dummy,		// 197
	Excep_DMACA_DMAC0,		// 198
	Excep_DMACA_DMAC1,		// 199
	Excep_DMACA_DMAC2,		// 200
	Excep_DMACA_DMAC3,		// 201
	Excep_EXDMAC_DMAC0,		// 202
	Excep_EXDMAC_DMAC1,		// 203
	Dummy,		// 204
	Dummy,		// 205
	Dummy,		// 206
	Dummy,		// 207
	Dummy,		// 208
	Dummy,		// 209
	Dummy,		// 210
	Dummy,		// 211
	Dummy,		// 212
	Dummy,		// 213
	Excep_SCI0_ERI0,		// 214
	Excep_SCI0_RXI0,		// 215
	Excep_SCI0_TXI0,		// 216
	Excep_SCI0_TEI0,		// 217
	Excep_SCI1_ERI1,		// 218
	Excep_SCI1_RXI1,		// 219
	Excep_SCI1_TXI1,		// 220
	Excep_SCI1_TEI1,		// 221
	Excep_SCI2_ERI2,		// 222
	Excep_SCI2_RXI2,		// 223
	Excep_SCI2_TXI2,		// 224
	Excep_SCI2_TEI2,		// 225
	Excep_SCI3_ERI3,		// 226
	Excep_SCI3_RXI3,		// 227
	Excep_SCI3_TXI3,		// 228
	Excep_SCI3_TEI3,		// 229
	Dummy,		// 230
	Dummy,		// 231
	Dummy,		// 232
	Dummy,		// 233
	Excep_SCI5_ERI5,		// 234
	Excep_SCI5_RXI5,		// 235
	Excep_SCI5_TXI5,		// 236
	Excep_SCI5_TEI5,		// 237
	Excep_SCI6_ERI6,		// 238
	Excep_SCI6_RXI6,		// 239
	Excep_SCI6_TXI6,		// 240
	Excep_SCI6_TEI6,		// 241
	Dummy,		// 242
	Dummy,		// 243
	Dummy,		// 244
	Dummy,		// 245
	Excep_RIIC0_EEI0,		// 246
	Excep_RIIC0_RXI0,		// 247
	Excep_RIIC0_TXI0,		// 248
	Excep_RIIC0_TEI0,		// 249
	Excep_RIIC1_EEI1,		// 250
	Excep_RIIC1_RXI1,		// 251
	Excep_RIIC1_TXI1,		// 252
	Excep_RIIC1_TEI1,		// 253
	Dummy,		// 254
	Dummy,		// 255
};

#ifdef __cpulsplus
}
#endif

// Exception(Supervisor Instruction)
void Excep_SuperVisorInst(void){
	__STOP
/* brk(); */}

// Exception(Undefined Instruction)
void Excep_UndefinedInst(void){
	__STOP
/* brk(); */}

// Exception(Floating Point)
void Excep_FloatingPoint(void){
	__STOP
/* brk(); */}

// NMI
void NonMaskableInterrupt(void){
	__STOP
/* brk(); */}

// BRK
void Excep_BRK(void){
	__STOP
/* brk(); */}

/////////////////////////
// ここから先はベクタテーブルの本体


// BUSERR
void Excep_BUSERR(void){__STOP}

// FCU_FCUERR
void Excep_FCU_FCUERR(void){__STOP}

// FCU_FRDYI
void Excep_FCU_FRDYI(void){__STOP}

// ICU_SWINT
void Excep_ICU_SWINT(void){__STOP}

// CMTU0_CMT0
//void Excep_CMTU0_CMT0(void){__STOP}

// CMTU0_CMT1
void Excep_CMTU0_CMT1(void){__STOP}

// CMTU1_CMT2
//void Excep_CMTU1_CMT2(void){ }

// CMTU1_CMT3
void Excep_CMTU1_CMT3(void){__STOP}

// ETHER EINT
//void Excep_ETHER_EINT(void){}

// USB0 D0FIFO0
void Excep_USB0_D0FIFO0(void){__STOP}

// USB0 D1FIFO0
void Excep_USB0_D1FIFO0(void){__STOP}

// USB0 USBI0
void Excep_USB0_USBI0(void){__STOP}

// USB1 D0FIFO1
void Excep_USB1_D0FIFO1(void){__STOP}

// USB1 D1FIFO1
void Excep_USB1_D1FIFO1(void){__STOP}

// USB1 USBI1
void Excep_USB1_USBI1(void){__STOP}

// RSPI0 SPEI0
void Excep_RSPI0_SPEI0(void){__STOP}

// RSPI0 SPRI0
void Excep_RSPI0_SPRI0(void){__STOP}

// RSPI0 SPTI0
void Excep_RSPI0_SPTI0(void){__STOP}

// RSPI0 SPII0
void Excep_RSPI0_SPII0(void){__STOP}

// RSPI1 SPEI1
void Excep_RSPI1_SPEI1(void){__STOP}

// RSPI1 SPRI1
void Excep_RSPI1_SPRI1(void){__STOP}

// RSPI1 SPTI1
void Excep_RSPI1_SPTI1(void){__STOP}

// RSPI1 SPII1
void Excep_RSPI1_SPII1(void){__STOP}

// CAN0 ERS0
void Excep_CAN0_ERS0(void){__STOP}

// CAN0 RXF0
void Excep_CAN0_RXF0(void){__STOP}

// CAN0 TXF0
void Excep_CAN0_TXF0(void){__STOP}

// CAN0 RXM0
void Excep_CAN0_RXM0(void){__STOP}

// CAN0 TXM0
void Excep_CAN0_TXM0(void){__STOP}

// RTC PRD
void Excep_RTC_PRD(void){__STOP}

// RTC CUP
void Excep_RTC_CUP(void){__STOP}

// IRQ0
void Excep_IRQ0(void){__STOP}

// IRQ1
void Excep_IRQ1(void){__STOP}

// IRQ2
void Excep_IRQ2(void){__STOP}

// IRQ3
void Excep_IRQ3(void){__STOP}

// IRQ4
void Excep_IRQ4(void){__STOP}

// IRQ5
void Excep_IRQ5(void){__STOP}

// IRQ6
void Excep_IRQ6(void){__STOP}

// IRQ7
void Excep_IRQ7(void){__STOP}

// IRQ8
void Excep_IRQ8(void){__STOP}

// IRQ9
void Excep_IRQ9(void){__STOP}

// IRQ10
void Excep_IRQ10(void){__STOP}

// IRQ11
void Excep_IRQ11(void){__STOP}

// IRQ12
void Excep_IRQ12(void){__STOP}

// IRQ13
void Excep_IRQ13(void){__STOP}

// IRQ14
void Excep_IRQ14(void){__STOP}

// IRQ15
void Excep_IRQ15(void){__STOP}

// USB RESUME USBR0
void Excep_USB_USBR0(void){__STOP}

// USB RESUME USBR1
void Excep_USB_USBR1(void){__STOP}

// RTC ALM
void Excep_RTC_ALM(void){__STOP}

// WDT_WOVI
void Excep_WDT_WOVI(void){__STOP}

// AD0_ADI0
void Excep_AD0_ADI0(void){__STOP}

// AD1_ADI1
void Excep_AD1_ADI1(void){__STOP}

// S12AD ADI12
void Excep_S12AD_ADI12(void){__STOP}

// MTU0 TGIA0
void Excep_MTU0_TGIA0(void){__STOP}

// MTU0 TGIB0
void Excep_MTU0_TGIB0(void){__STOP}

// MTU0 TGIC0
void Excep_MTU0_TGIC0(void){__STOP}

// MTU0 TGID0
void Excep_MTU0_TGID0(void){__STOP}

// MTU0 TCIV0
void Excep_MTU0_TCIV0(void){__STOP}

// MTU0 TGIE0
void Excep_MTU0_TGIE0(void){__STOP}

// MTU0 TGIF0
void Excep_MTU0_TGIF0(void){__STOP}

// MTU1 TGIA1
void Excep_MTU1_TGIA1(void){__STOP}

// MTU1 TGIB1
void Excep_MTU1_TGIB1(void){__STOP}

// MTU1 TCIV1
void Excep_MTU1_TCIV1(void){__STOP}

// MTU1 TCIU1
void Excep_MTU1_TCIU1(void){__STOP}

// MTU2 TGIA2
void Excep_MTU2_TGIA2(void){__STOP}

// MTU2 TGIB2
void Excep_MTU2_TGIB2(void){__STOP}

// MTU2 TCIV2
void Excep_MTU2_TCIV2(void){__STOP}

// MTU2 TCIU2
void Excep_MTU2_TCIU2(void){__STOP}

// MTU3 TGIA3
void Excep_MTU3_TGIA3(void){__STOP}

// MTU3 TGIB3
void Excep_MTU3_TGIB3(void){__STOP}

// MTU3 TGIC3
void Excep_MTU3_TGIC3(void){__STOP}

// MTU3 TGID3
void Excep_MTU3_TGID3(void){__STOP}

// MTU3 TCIV3
void Excep_MTU3_TCIV3(void){__STOP}

// MTU4 TGIA4
void Excep_MTU4_TGIA4(void){__STOP}

// MTU4 TGIB4
void Excep_MTU4_TGIB4(void){__STOP}

// MTU4 TGIC4
void Excep_MTU4_TGIC4(void){__STOP}

// MTU4 TGID4
void Excep_MTU4_TGID4(void){__STOP}

// MTU4 TCIV4
void Excep_MTU4_TCIV4(void){__STOP}

// MTU5 TCIU5
void Excep_MTU5_TCIU5(void){__STOP}

// MTU5 TCIV5
void Excep_MTU5_TCIV5(void){__STOP}

// MTU5 TCIW5
void Excep_MTU5_TCIW5(void){__STOP}

// MTU6 TGIA6
void Excep_MTU6_TGIA6(void){__STOP}

// MTU6 TGIB6
void Excep_MTU6_TGIB6(void){__STOP}

// MTU6 TGIC6
void Excep_MTU6_TGIC6(void){__STOP}

// MTU6 TGID6
void Excep_MTU6_TGID6(void){__STOP}

// MTU6 TCIV6
void Excep_MTU6_TCIV6(void){__STOP}

// MTU6 TGIE6
void Excep_MTU6_TGIE6(void){__STOP}

// MTU6 TGIF6
void Excep_MTU6_TGIF6(void){__STOP}

// MTU7 TGIA7
void Excep_MTU7_TGIA7(void){__STOP}

// MTU7 TGIB7
void Excep_MTU7_TGIB7(void){__STOP}

// MTU7 TCIV7
void Excep_MTU7_TCIV7(void){__STOP}

// MTU7 TCIU7
void Excep_MTU7_TCIU7(void){__STOP}

// MTU8 TGIA8
void Excep_MTU8_TGIA8(void){__STOP}

// MTU8 TGIB8
void Excep_MTU8_TGIB8(void){__STOP}

// MTU8 TCIV8
void Excep_MTU8_TCIV8(void){__STOP}

// MTU8 TCIU8
void Excep_MTU8_TCIU8(void){__STOP}

// MTU9 TGIA9
void Excep_MTU9_TGIA9(void){__STOP}

// MTU9 TGIB9
void Excep_MTU9_TGIB9(void){__STOP}

// MTU9 TGIC9
void Excep_MTU9_TGIC9(void){__STOP}

// MTU9 TGID9
void Excep_MTU9_TGID9(void){__STOP}

// MTU9 TCIV9
void Excep_MTU9_TCIV9(void){__STOP}

// MTU10 TGIA10
void Excep_MTU10_TGIA10(void){__STOP}

// MTU10 TGIB10
void Excep_MTU10_TGIB10(void){__STOP}

// MTU10 TGIC10
void Excep_MTU10_TGIC10(void){__STOP}

// MTU10 TGID10
void Excep_MTU10_TGID10(void){__STOP}

// MTU10 TCIV10
void Excep_MTU10_TCIV10(void){__STOP}

// MTU11 TCIU11
void Excep_MTU11_TCIU11(void){__STOP}

// MTU11 TCIV11
void Excep_MTU11_TCIV11(void){__STOP}

// MTU11 TCIW11
void Excep_MTU11_TCIW11(void){__STOP}

// POE OEI1
void Excep_POE_OEI1(void){__STOP}

// POE OEI1
void Excep_POE_OEI2(void){__STOP}

// POE OEI1
void Excep_POE_OEI3(void){__STOP}

// POE OEI1
void Excep_POE_OEI4(void){__STOP}

// TMR0_CMI0A
void Excep_TMR0_CMI0A(void){__STOP}

// TMR0_CMI0B
void Excep_TMR0_CMI0B(void){__STOP}

// TMR0_OV0I
void Excep_TMR0_OV0I(void){__STOP}

// TMR1_CMI1A
void Excep_TMR1_CMI1A(void){__STOP}

// TMR1_CMI1B
void Excep_TMR1_CMI1B(void){__STOP}

// TMR1_OV1I
void Excep_TMR1_OV1I(void){__STOP}

// TMR2_CMI2A
void Excep_TMR2_CMI2A(void){__STOP}

// TMR2_CMI2B
void Excep_TMR2_CMI2B(void){__STOP}

// TMR2_OV2I
void Excep_TMR2_OV2I(void){__STOP}

// TMR3_CMI3A
void Excep_TMR3_CMI3A(void){__STOP}

// TMR3_CMI3B
void Excep_TMR3_CMI3B(void){__STOP}

// TMR3_OV3I
void Excep_TMR3_OV3I(void){__STOP}

// DMACA DMAC0
void Excep_DMACA_DMAC0(void){__STOP}

// DMAC DMAC1
void Excep_DMACA_DMAC1(void){__STOP}

// DMAC DMAC2
void Excep_DMACA_DMAC2(void){__STOP}

// DMAC DMAC3
void Excep_DMACA_DMAC3(void){__STOP}

// EXDMAC DMAC0
void Excep_EXDMAC_DMAC0(void){__STOP}

// EXDMAC DMAC1
void Excep_EXDMAC_DMAC1(void){__STOP}

// SCI0_ERI0
//void Excep_SCI0_ERI0(void){__STOP}

// SCI0_RXI0
//void Excep_SCI0_RXI0(void){__STOP}

// SCI0_TXI0
//void Excep_SCI0_TXI0(void){__STOP}

// SCI0_TEI0
//void Excep_SCI0_TEI0(void){__STOP}

// SCI1_ERI1
//void Excep_SCI1_ERI1(void){__STOP}

// SCI1_RXI1
//void Excep_SCI1_RXI1(void){__STOP}

// SCI1_TXI1
//void Excep_SCI1_TXI1(void){__STOP}

// SCI1_TEI1
//void Excep_SCI1_TEI1(void){__STOP}

// SCI2_ERI2
void Excep_SCI2_ERI2(void){__STOP}

// SCI2_RXI2
void Excep_SCI2_RXI2(void){__STOP}

// SCI2_TXI2
void Excep_SCI2_TXI2(void){__STOP}

// SCI2_TEI2
void Excep_SCI2_TEI2(void){__STOP}

// SCI3_ERI3
void Excep_SCI3_ERI3(void){__STOP}

// SCI3_RXI3
void Excep_SCI3_RXI3(void){__STOP}

// SCI3_TXI3
void Excep_SCI3_TXI3(void){__STOP}

// SCI3_TEI3
void Excep_SCI3_TEI3(void){__STOP}

// SCI5_ERI5
void Excep_SCI5_ERI5(void){__STOP}

// SCI5_RXI5
void Excep_SCI5_RXI5(void){__STOP}

// SCI5_TXI5
void Excep_SCI5_TXI5(void){__STOP}

// SCI5_TEI5
void Excep_SCI5_TEI5(void){__STOP}

// SCI6_ERI6
void Excep_SCI6_ERI6(void){__STOP}

// SCI6_RXI6
void Excep_SCI6_RXI6(void){__STOP}

// SCI6_TXI6
void Excep_SCI6_TXI6(void){__STOP}

// SCI6_TEI6
void Excep_SCI6_TEI6(void){__STOP}

// RIIC0_EEI0
void Excep_RIIC0_EEI0(void){__STOP}

// RIIC0_RXI0
void Excep_RIIC0_RXI0(void){__STOP}

// RIIC0_TXI0
void Excep_RIIC0_TXI0(void){__STOP}

// RIIC0_TEI0
void Excep_RIIC0_TEI0(void){__STOP}

// RIIC1_EEI1
void Excep_RIIC1_EEI1(void){__STOP}

// RIIC1_RXI1
void Excep_RIIC1_RXI1(void){__STOP}

// RIIC1_TXI1
void Excep_RIIC1_TXI1(void){__STOP}

// RIIC1_TEI1
void Excep_RIIC1_TEI1(void){__STOP}

// Dummy
void Dummy(void){/* brk(); */}
