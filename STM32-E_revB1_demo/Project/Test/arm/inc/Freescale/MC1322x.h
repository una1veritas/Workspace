
#ifndef __MC1322x_H
#define __MC1322x_H

/*Peripheral Memory Map Base addresses*/
#define  MBAR_AIPI   0x80000000  /*32 bit*/
#define  MBAR_GPIO   0x80000000  /*32 bit*/
#define  MBAR_SSI    0x80001000  /*32 bit*/
#define  MBAR_SPI    0x80002000  /*32 bit*/
#define  MBAR_CRM    0x80003000  /*32 bit*/
#define  MBAR_MACA   0x80004000  /*32 bit*/
#define  MBAR_UART1  0x80005000  /*32 bit*/
#define  MBAR_I2C    0x80006000  /*32 bit*/
#define  MBAR_TMR    0x80007000  /*16 bit*/
#define  MBAR_TMR0   0x80007000  /*16 bit*/
#define  MBAR_TMR1   0x80007020  /*16 bit*/
#define  MBAR_TMR2   0x80007040  /*16 bit*/
#define  MBAR_TMR3   0x80007060  /*16 bit*/
#define  MBAR_TMR_EN 0x8000701E  /*16 bit */
#define  MBAR_ASM    0x80008000  /*32 bit*/
#define  MBAR_MODEM  0x80009000  /*32 bit*/
#define  MBAR_RF     0x8000A000  /*32 bit*/
#define  MBAR_UART2  0x8000B000  /*32 bit*/
#define  MBAR_FLASH  0x8000C000  /*32 bit*/
#define  MBAR_ADC    0x8000D000  /*16 bit*/
#define  MBAR_AITC   0x80020000  /*32 bit*/
#define  MBAR_ITC    0x80020000  /*32 bit*/
#define  MBAR_NEX    0x80040000  /*32 bit*/

/* Peripheral Structure Pointer definitions */
#define  GPIO_REGS_P   ((GpioRegs_t*)MBAR_GPIO)
#define  SSI_REGS_P    ((SsiRegs_t*)MBAR_SSI)
#define  SPI_REGS_P    ((SpiRegs_t*)MBAR_SPI)
#define  CRM_REGS_P    ((CrmRegs_t*)MBAR_CRM)
#define  MACA_REGS_P   ((MacaRegs_t*)MBAR_MACA)
#define  UART1_REGS_P  ((UartRegs_t*)MBAR_UART1)
#define  I2C_REGS_P    ((I2cRegs_t*)MBAR_I2C)
#define  TMR_REGS_P    ((TmrRegs_t*)MBAR_TMR)
#define  TMR0_REGS_P   ((TmrRegs_t*)MBAR_TMR0)
#define  TMR1_REGS_P   ((TmrRegs_t*)MBAR_TMR1)
#define  TMR2_REGS_P   ((TmrRegs_t*)MBAR_TMR2)
#define  TMR3_REGS_P   ((TmrRegs_t*)MBAR_TMR3)
#define  TMR_EN_REG_P  ((volatile uint16_t*)MBAR_TMR_EN)
#define  ASM_REGS_P    ((AsmRegs_t*) MBAR_ASM)
#define  MODEM_REGS_P  ((ModemRegs_t*)MBAR_MODEM)
#define  RF_REGS_P     ((RfRegs_t*)MBAR_RF)
#define  UART2_REGS_P  ((UartRegs_t*)MBAR_UART2)
#define  FLASH_REGS_P  ((SpiRegs_t*)MBAR_FLASH)
#define  ADC_REGS_P    ((AdcRegs_t*)MBAR_ADC)
#define  AITC_REGS_P   ((AitcRegs_t*)MBAR_AITC)
#define  ITC_REGS_P    ((ItcRegs_t*)MBAR_ITC)
#define  NEX_REGS_P    ((NexRegs_t*)MBAR_NEX)

/*  extended pointer definition  */
#define GPIO   (*GPIO_REGS_P)
#define SSI    (*SSI_REGS_P)
#define SPI    (*SPI_REGS_P)
#define CRM    (*CRM_REGS_P)
#define MACA   (*MACA_REGS_P)
#define UART1  (*UART1_REGS_P)
#define I2C    (*I2C_REGS_P)
#define TMR0   (*TMR0_REGS_P)
#define TMR1   (*TMR1_REGS_P)
#define TMR2   (*TMR1_REGS_P)
#define TMR3   (*TMR1_REGS_P)
#define TMR_EN (*TMR_EN_REG_P)
#define ASM    (*ASM_REGS_P)
#define MODEM  (*MODEM_REGS_P)
#define RF     (*RF_REGS_P)
#define UART2  (*UART2_REGS_P)
#define FLASH  (*FLASH_REGS_P)
#define ADC    (*ADC_REGS_P)
#define AITC   (*AITC_REGS_P)
#define ITC    (*ITC_REGS_P)
#define NEX    (*NEX_REGS_P)

/* define the registers for the GPIO peripheral */
typedef volatile struct
{  /* Registers */
  unsigned long DirLo;             /*MBAR_GPIO + 0x00*/
  unsigned long DirHi;             /*MBAR_GPIO + 0x04*/
  unsigned long DataLo;            /*MBAR_GPIO + 0x08*/
  unsigned long DataHi;            /*MBAR_GPIO + 0x0C*/
  unsigned long PuEnLo;            /*MBAR_GPIO + 0x10*/
  unsigned long PuEnHi;            /*MBAR_GPIO + 0x14*/
  unsigned long FuncSel0;          /*MBAR_GPIO + 0x18*/
  unsigned long FuncSel1;          /*MBAR_GPIO + 0x1C*/
  unsigned long FuncSel2;          /*MBAR_GPIO + 0x20*/
  unsigned long FuncSel3;          /*MBAR_GPIO + 0x24*/
  unsigned long InputDataSelLo;    /*MBAR_GPIO + 0x28*/
  unsigned long InputDataSelHi;    /*MBAR_GPIO + 0x2C*/
  unsigned long PuSelLo;           /*MBAR_GPIO + 0x30*/
  unsigned long PuSelHi;           /*MBAR_GPIO + 0x34*/
  unsigned long HystEnLo;          /*MBAR_GPIO + 0x38*/
  unsigned long HystEnHi;          /*MBAR_GPIO + 0x3C*/
  unsigned long PuKeepLo;          /*MBAR_GPIO + 0x40*/
  unsigned long PuKeepHi;          /*MBAR_GPIO + 0x44*/
  /* Virtual registers */
  unsigned long DataSetLo;         /*MBAR_GPIO + 0x48*/
  unsigned long DataSetHi;         /*MBAR_GPIO + 0x4C*/
  unsigned long DataResetLo;       /*MBAR_GPIO + 0x50*/
  unsigned long DataResetHi;       /*MBAR_GPIO + 0x54*/
  unsigned long DirSetLo;          /*MBAR_GPIO + 0x58*/
  unsigned long DirSetHi;          /*MBAR_GPIO + 0x5C*/
  unsigned long DirResetLo;        /*MBAR_GPIO + 0x60*/
  unsigned long DirResetHi;        /*MBAR_GPIO + 0x64*/
} GpioRegs_t;

/* define the registers for the CRM peripheral */
typedef volatile struct
{
  unsigned long SysCntl;            /*MBAR_CRM + 0x00*/
  unsigned long WuCntl;             /*MBAR_CRM + 0x04*/
  unsigned long SleepCntl;          /*MBAR_CRM + 0x08*/
  unsigned long BsCntl;             /*MBAR_CRM + 0x0C*/
  unsigned long CopCntl;            /*MBAR_CRM + 0x10*/
  unsigned long CopService;         /*MBAR_CRM + 0x14*/
  unsigned long Status;             /*MBAR_CRM + 0x18*/
  unsigned long ModStatus;          /*MBAR_CRM + 0x1C*/
  unsigned long WuCount;            /*MBAR_CRM + 0x20*/
  unsigned long WuTimeout;          /*MBAR_CRM + 0x24*/
  unsigned long RtcCount;           /*MBAR_CRM + 0x28*/
  unsigned long RtcTimeout;         /*MBAR_CRM + 0x2C*/
  unsigned long reserved;           /*MBAR_CRM + 0x30*/
  unsigned long CalCntl;            /*MBAR_CRM + 0x34*/
  unsigned long CalXtalCnt;         /*MBAR_CRM + 0x38*/
  unsigned long RingOsclCntl;       /*MBAR_CRM + 0x3C*/
  unsigned long XtalCntl;           /*MBAR_CRM + 0x40*/
  unsigned long Xtal32Cntl;         /*MBAR_CRM + 0x44*/
  unsigned long VregCntl;           /*MBAR_CRM + 0x48*/
  unsigned long VregTrim;           /*MBAR_CRM + 0x4C*/
  unsigned long SwRst;              /*MBAR_CRM + 0x50*/
} CrmRegs_t;

/* define the registers for the ITC peripheral */
typedef volatile struct
{
  unsigned long IntCntl;           /*MBAR_ITC + 0x00*/
  unsigned long NiMask;            /*MBAR_ITC + 0x04*/
  unsigned long IntEnNum;          /*MBAR_ITC + 0x08*/
  unsigned long IntDisNum;         /*MBAR_ITC + 0x0C*/
  unsigned long IntEnable;         /*MBAR_ITC + 0x10*/
  unsigned long IntType;           /*MBAR_ITC + 0x14*/
  unsigned long reserved3;         /*MBAR_ITC + 0x18*/
  unsigned long reserved2;         /*MBAR_ITC + 0x1C*/
  unsigned long reserved1;         /*MBAR_ITC + 0x20*/
  unsigned long reserved0;         /*MBAR_ITC + 0x24*/
  unsigned long NiVector;          /*MBAR_ITC + 0x28*/
  unsigned long FiVector;          /*MBAR_ITC + 0x2C*/
  unsigned long IntSrc;            /*MBAR_ITC + 0x30*/
  unsigned long IntFrc;            /*MBAR_ITC + 0x34*/
  unsigned long NiPend;            /*MBAR_ITC + 0x38*/
  unsigned long FiPend;            /*MBAR_ITC + 0x3C*/
} ItcRegs_t, AitcRegs_t;

/* define the registers for the UART  peripherals */
typedef volatile struct
{
  unsigned long Ucon;              /*MBAR_UARTx + 0x00*/
  unsigned long Ustat;             /*MBAR_UARTx + 0x04*/
  unsigned long Udata;             /*MBAR_UARTx + 0x08*/
  unsigned long Urxcon;            /*MBAR_UARTx + 0x0C*/
  unsigned long Utxcon;            /*MBAR_UARTx + 0x10*/
  unsigned long Ucts;              /*MBAR_UARTx + 0x14*/
  unsigned long Ubr;               /*MBAR_UARTx + 0x18*/
} UartRegs_t;

/* define the registers for the SPI  peripherals */
typedef volatile struct
{
  unsigned long TxData;            /*MBAR_SPIx + 0x00*/
  unsigned long RxData;            /*MBAR_SPIx + 0x04*/
  unsigned long ClkCtrl;           /*MBAR_SPIx + 0x08*/
  unsigned long Setup;             /*MBAR_SPIx + 0x0C*/
  unsigned long Status;            /*MBAR_SPIx + 0x10*/
} SpiRegs_t;

/* define the registers for the TIMER  peripherals */
typedef volatile struct
{
  unsigned short Comp1;            /*MBAR_TMRx + 0x00*/
  unsigned short Comp2;            /*MBAR_TMRx + 0x02*/
  unsigned short Capt;             /*MBAR_TMRx + 0x04*/
  unsigned short Load;             /*MBAR_TMRx + 0x06*/
  unsigned short Hold;             /*MBAR_TMRx + 0x08*/
  unsigned short Cntr;             /*MBAR_TMRx + 0x0A*/
  unsigned short Ctrl;             /*MBAR_TMRx + 0x0C*/
  unsigned short StatCtrl;         /*MBAR_TMRx + 0x0E*/
  unsigned short CmpLd1;           /*MBAR_TMRx + 0x10*/
  unsigned short CmpLd2;           /*MBAR_TMRx + 0x12*/
  unsigned short CompStatCtrl;     /*MBAR_TMRx + 0x14*/
  unsigned short reserved0;        /*MBAR_TMRx + 0x16*/
  unsigned short reserved1;        /*MBAR_TMRx + 0x18*/
  unsigned short reserved2;        /*MBAR_TMRx + 0x1A*/
  unsigned short reserved3;        /*MBAR_TMRx + 0x1C*/
  unsigned short reserved4;        /*MBAR_TMRx + 0x1E*/
} TmrRegs_t;

/* define the registers for the ASM peripheral */
typedef volatile struct
{
  unsigned long Key0;             /*MBAR_ASM + 0x00*/
  unsigned long Key1;             /*MBAR_ASM + 0x04*/
  unsigned long Key2;             /*MBAR_ASM + 0x08*/
  unsigned long Key3;             /*MBAR_ASM + 0x0C*/
  unsigned long Data0;            /*MBAR_ASM + 0x10*/
  unsigned long Data1;            /*MBAR_ASM + 0x14*/
  unsigned long Data2;            /*MBAR_ASM + 0x18*/
  unsigned long Data3;            /*MBAR_ASM + 0x1C*/
  unsigned long Ctr0;             /*MBAR_ASM + 0x20*/
  unsigned long Ctr1;             /*MBAR_ASM + 0x24*/
  unsigned long Ctr2;             /*MBAR_ASM + 0x28*/
  unsigned long Ctr3;             /*MBAR_ASM + 0x2C*/
  unsigned long Ctr_result0;      /*MBAR_ASM + 0x30*/
  unsigned long Ctr_result1;      /*MBAR_ASM + 0x34*/
  unsigned long Ctr_result2;      /*MBAR_ASM + 0x38*/
  unsigned long Ctr_result3;      /*MBAR_ASM + 0x3C*/
  unsigned long Cbc_result0;      /*MBAR_ASM + 0x40*/
  unsigned long Cbc_result1;      /*MBAR_ASM + 0x44*/
  unsigned long Cbc_result2;      /*MBAR_ASM + 0x48*/
  unsigned long Cbc_result3;      /*MBAR_ASM + 0x4C*/
  unsigned long Control0;         /*MBAR_ASM + 0x50*/
  unsigned long Control1;         /*MBAR_ASM + 0x54*/
  unsigned long Status;           /*MBAR_ASM + 0x58*/
  unsigned long Undef0;           /*MBAR_ASM + 0x5C  */
  unsigned long Mac0;             /*MBAR_ASM + 0x60*/
  unsigned long Mac1;             /*MBAR_ASM + 0x64*/
  unsigned long Mac2;             /*MBAR_ASM + 0x68*/
  unsigned long Mac3;             /*MBAR_ASM + 0x6C  */
} AsmRegs_t;

/* define the registers for the I2C  peripheral */
typedef volatile struct
{
  unsigned char Address;        /*MBAR_I2C + 0x00     address register*/
  unsigned char dummy0;
  unsigned char dummy1;
  unsigned char dummy2;
  unsigned char FreqDiv;        /*MBAR_I2C + 0x04     frequency divider register*/
  unsigned char dummy3;
  unsigned char dummy4;
  unsigned char dummy5;
  unsigned char Control;        /*MBAR_I2C + 0x08     control register*/
  unsigned char dummy6;
  unsigned char dummy7;
  unsigned char dummy8;
  unsigned char Status;         /*MBAR_I2C + 0x0C     status register*/
  unsigned char dummy9;
  unsigned char dummy10;
  unsigned char dummy11;
  unsigned char Data;           /*MBAR_I2C + 0x10     data register*/
  unsigned char dummy12;
  unsigned char dummy13;
  unsigned char dummy14;
  unsigned char DigitalFilter;  /*MBAR_I2C + 0x14     digital filter sampling rate register*/
  unsigned char dummy15;
  unsigned char dummy16;
  unsigned char dummy17;
  unsigned char ClockEn;        /*MBAR_I2C + 0x18     clock enable register*/
  unsigned char dummy18;
  unsigned char dummy19;
  unsigned char dummy20;
} I2cRegs_t;

/* define the registers for the SSI peripheral */
typedef volatile struct
{
  unsigned long  STX;        /*MBAR_SSI + 0x00     STX   (Transmit Data register)*/
  unsigned long  dummy1;     /*MBAR_SSI + 0x04*/
  unsigned long  SRX;        /*MBAR_SSI + 0x08     SRX   (Receive Data Register)*/
  unsigned long  dummy2;     /*MBAR_SSI + 0x0C*/
  unsigned long  SCR;        /*MBAR_SSI + 0x10     SCR   (Control register)*/
  unsigned long  SISR;       /*MBAR_SSI + 0x14     SISR  (Interrupt status register)*/
  unsigned long  SIER;       /*MBAR_SSI + 0x18     SIER  (Interrupt enable register)*/
  unsigned long  STCR;       /*MBAR_SSI + 0x1C     STCR  (Transmit configuration register)*/
  unsigned long  SRCR;       /*MBAR_SSI + 0x20     SRCR  (Receive configuration register)*/
  unsigned long  STCCR;      /*MBAR_SSI + 0x24     STCCR (Transmit and Receive Clock configuration register)*/
  unsigned long  dummy3;     /*MBAR_SSI + 0x28*/
  unsigned long  SFCSR;      /*MBAR_SSI + 0x2C     SFCSR (FIFO control / status register)*/
  unsigned long  STR;        /*MBAR_SSI + 0x30     STR   (Test register)*/
  unsigned long  SOR;        /*MBAR_SSI + 0x34     SOR   (Option register)*/
  unsigned long  dummy4;     /*MBAR_SSI + 0x38*/
  unsigned long  dummy5;     /*MBAR_SSI + 0x3C*/
  unsigned long  dummy6;     /*MBAR_SSI + 0x40*/
  unsigned long  dummy7;     /*MBAR_SSI + 0x44*/
  unsigned long  STMSK;      /*MBAR_SSI + 0x48     STMSK (Transmit Time Slot mask register)*/
  unsigned long  SRMSK;      /*MBAR_SSI + 0x4C     SRMSK (Receive Time Slot mask register)*/
}SsiRegs_t;

/* define the registers for the ADC  peripheral */
typedef volatile struct
{
  unsigned short Comp0;            /*MBAR_ADC + 0x00     Compare0 register*/
  unsigned short Comp1;            /*MBAR_ADC + 0x02     Compare1 register*/
  unsigned short Comp2;            /*MBAR_ADC + 0x04     Compare2 register*/
  unsigned short Comp3;            /*MBAR_ADC + 0x06     Compare3 register*/
  unsigned short Comp4;            /*MBAR_ADC + 0x08     Compare4 register*/
  unsigned short Comp5;            /*MBAR_ADC + 0x0A     Compare5 register*/
  unsigned short Comp6;            /*MBAR_ADC + 0x0C     Compare6 register*/
  unsigned short Comp7;            /*MBAR_ADC + 0x0E     Compare7 register*/
  unsigned short BattCompOver;     /*MBAR_ADC + 0x10     Battery Voltage upper trip point*/
  unsigned short BattCompUnder;    /*MBAR_ADC + 0x12     Battery Voltage lower trip point*/
  unsigned short Seq1;             /*MBAR_ADC + 0x14     Sequencer1 register*/
  unsigned short Seq2;             /*MBAR_ADC + 0x16     Sequencer2 register*/
  unsigned short Control;          /*MBAR_ADC + 0x18     Control register*/
  unsigned short Triggers;         /*MBAR_ADC + 0x1A     Triggers register*/
  unsigned short Prescale;         /*MBAR_ADC + 0x1C     Prescale register*/
  unsigned short reserved1;        /*MBAR_ADC + 0x1E     reserved*/
  unsigned short FifoRead;         /*MBAR_ADC + 0x20     FIFO Read register*/
  unsigned short FifoCtrl;         /*MBAR_ADC + 0x22     FIFO Control register*/
  unsigned short FifoStatus;       /*MBAR_ADC + 0x24     FIFO Status register*/
  unsigned short reserved2;        /*MBAR_ADC + 0x26     register*/
  unsigned short reserved3;        /*MBAR_ADC + 0x28     register*/
  unsigned short reserved4;        /*MBAR_ADC + 0x2A     register*/
  unsigned short reserved5;        /*MBAR_ADC + 0x2C     register*/
  unsigned short reserved6;        /*MBAR_ADC + 0x2E     register*/
  unsigned short Sr1High;          /*MBAR_ADC + 0x30     Timer1 Sample Rate High Value*/
  unsigned short Sr1Low;           /*MBAR_ADC + 0x32     Timer1 Sample Rate Low Value*/
  unsigned short Sr2High;          /*MBAR_ADC + 0x34     Timer2 Sample Rate High Value*/
  unsigned short Sr2Low;           /*MBAR_ADC + 0x36     Timer2 Sample Rate Low Value*/
  unsigned short OnTime;           /*MBAR_ADC + 0x38     On Time register*/
  unsigned short ConvTime;         /*MBAR_ADC + 0x3A     Convert Time register*/
  unsigned short ClkDiv;           /*MBAR_ADC + 0x3C     Clock divider register*/
  unsigned short reserved7;        /*MBAR_ADC + 0x3E     reserved*/
  unsigned short Override;         /*MBAR_ADC + 0x40     Override register*/
  unsigned short Irq;              /*MBAR_ADC + 0x42     Interrupt register*/
  unsigned short Mode;             /*MBAR_ADC + 0x44     ADC Mode register*/
  unsigned short Adc1Result;       /*MBAR_ADC + 0x46     ADC1 Result register*/
  unsigned short Adc2Result;       /*MBAR_ADC + 0x48     ADC2 Result register*/
} AdcRegs_t;

#endif	/* __MC1322x_H */
