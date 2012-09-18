/*
** ###################################################################
**     Compilers:           ARM Compiler
**                          Freescale C/C++ for Embedded ARM
**                          GNU C Compiler
**                          IAR ANSI C/C++ Compiler for ARM
**
**     Reference manual:    K15 Sub-Family Reference Manual Rev. 1, 9 Feb 2012
**     Version:             rev. 1.0, 2012-02-20
**
**     Abstract:
**         This header file implements peripheral memory map for MKE14D7
**         processor.
**
**     Copyright: 1997 - 2012 Freescale Semiconductor, Inc. All Rights Reserved.
**
**     http:                 www.freescale.com
**     mail:                 support@freescale.com
**
**     Revisions:
**     - rev. 1.0 (2012-02-20)
**         Initial public version.
**
** ###################################################################
*/

/**
 * @file MKE14D7.h
 * @version 1.0
 * @date 2012-02-20
 * @brief Peripheral memory map for MKE14D7
 *
 * This header file implements peripheral memory map for MKE14D7 processor.
 */


/* ----------------------------------------------------------------------------
   -- MCU activation
   ---------------------------------------------------------------------------- */

/* Prevention from multiple including the same memory map */
#if !defined(MCU_MKE14D7)  /* Check if memory map has not been already included */
#define MCU_MKE14D7

/* Check if another memory map has not been also included */
#if (defined(MCU_ACTIVE))
  #error MKE14D7 memory map: There is already included another memory map. Only one memory map can be included.
#endif /* (defined(MCU_ACTIVE)) */
#define MCU_ACTIVE

#include <stdint.h>

/** Memory map major version (memory maps with equal major version number are
 * compatible) */
#define MCU_MEM_MAP_VERSION 0x0100u
/** Memory map minor version */
#define MCU_MEM_MAP_VERSION_MINOR 0x0000u

/**
 * @brief Macro to access a single bit of a peripheral register (bit band region
 *        0x40000000 to 0x400FFFFF) using the bit-band alias region access.
 * @param Reg Register to access.
 * @param Bit Bit number to access.
 * @return Value of the targeted bit in the bit band region.
 */
#define BITBAND_REG(Reg,Bit) (*((uint32_t volatile*)(0x42000000u + (32u*((uint32_t)&(Reg) - (uint32_t)0x40000000u)) + (4u*((uint32_t)(Bit))))))

/* ----------------------------------------------------------------------------
   -- Interrupt vector numbers
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup Interrupt_vector_numbers Interrupt vector numbers
 * @{
 */

/** Interrupt Number Definitions */
typedef enum {
  INT_Initial_Stack_Pointer    = 0,                /**< Initial stack pointer */
  INT_Initial_Program_Counter  = 1,                /**< Initial program counter */
  INT_NMI                      = 2,                /**< Non-maskable interrupt */
  INT_Hard_Fault               = 3,                /**< Hard fault exception */
  INT_Mem_Manage_Fault         = 4,                /**< Memory Manage Fault */
  INT_Bus_Fault                = 5,                /**< Bus fault exception */
  INT_Usage_Fault              = 6,                /**< Usage fault exception */
  INT_Reserved7                = 7,                /**< Reserved interrupt 7 */
  INT_Reserved8                = 8,                /**< Reserved interrupt 8 */
  INT_Reserved9                = 9,                /**< Reserved interrupt 9 */
  INT_Reserved10               = 10,               /**< Reserved interrupt 10 */
  INT_SVCall                   = 11,               /**< A supervisor call exception */
  INT_DebugMonitor             = 12,               /**< Debug Monitor */
  INT_Reserved13               = 13,               /**< Reserved interrupt 13 */
  INT_PendableSrvReq           = 14,               /**< PendSV exception - request for system level service */
  INT_Reserved15               = 15,               /**< Reserved interrupt 15 */
  INT_DMA0                     = 16,               /**< DMA Channel 0 Transfer Complete */
  INT_DMA1                     = 17,               /**< DMA Channel 1 Transfer Complete */
  INT_DMA2                     = 18,               /**< DMA Channel 2 Transfer Complete */
  INT_DMA3                     = 19,               /**< DMA Channel 3 Transfer Complete */
  INT_DMA4                     = 20,               /**< DMA Channel 4 Transfer Complete */
  INT_DMA5                     = 21,               /**< DMA Channel 5 Transfer Complete */
  INT_DMA6                     = 22,               /**< DMA Channel 6 Transfer Complete */
  INT_DMA7                     = 23,               /**< DMA Channel 7 Transfer Complete */
  INT_DMA8                     = 24,               /**< DMA Channel 8 Transfer Complete */
  INT_DMA9                     = 25,               /**< DMA Channel 9 Transfer Complete */
  INT_DMA10                    = 26,               /**< DMA Channel 10 Transfer Complete */
  INT_DMA11                    = 27,               /**< DMA Channel 11 Transfer Complete */
  INT_DMA12                    = 28,               /**< DMA Channel 12 Transfer Complete */
  INT_DMA13                    = 29,               /**< DMA Channel 13 Transfer Complete */
  iEVENT0                      = 30,               /**< iEVENT channel 0 */
  iEVENT1                      = 31,               /**< iEVENT channel 1 */
  INT_DMA_Error                = 32,               /**< DMA Error Interrupt */
  INT_Reserved33               = 33,               /**< Reserved interrupt 33 */
  INT_FTMRA                    = 34,               /**< FTMRA Interrupt */
  INT_Read_Collision           = 35,               /**< Read Collision */
  INT_LVD_LVW                  = 36,               /**< Low Voltage Detect, Low Voltage Warning */
  INT_ICS                      = 37,               /**< ICS */
  INT_Watchdog                 = 38,               /**< WDOG */
  INT_ADC0                     = 39,               /**< ADC0 */
  INT_ADC2                     = 40,               /**< ADC2 */
  INT_SPI0                     = 41,               /**< SPI0 */
  INT_SPI1                     = 42,               /**< SPI1 */
  INT_I2C0                     = 43,               /**< IIC0 interrupt */
  INT_I2C1                     = 44,               /**< IIC1 interrupt */
  INT_UART0_RX_TX              = 45,               /**< UART0 status */
  INT_UART0_ERR                = 46,               /**< UART0 error */
  INT_UART1_RX_TX              = 47,               /**< UART1 status */
  INT_UART1_ERR                = 48,               /**< UART1 error */
  INT_UART2_RX_TX              = 49,               /**< UART2 status */
  INT_UART2_ERR                = 50,               /**< UART2 error */
  INT_UART3_RX_TX              = 51,               /**< UART3 status */
  INT_UART3_ERR                = 52,               /**< UART3 error */
  INT_ADC1                     = 53,               /**< ADC1 */
  INT_ADC3                     = 54,               /**< ADC3 */
  INT_SWI                      = 55,               /**< Software interrupt */
  INT_CMP0                     = 56,               /**< CMP0 */
  INT_CMP1                     = 57,               /**< CMP1 */
  INT_CMP2                     = 58,               /**< CMP2 */
  INT_CMP3                     = 59,               /**< CMP3 */
  INT_FTM0                     = 60,               /**< FTM0 */
  INT_FTM1                     = 61,               /**< FTM1 */
  INT_FTM2                     = 62,               /**< FTM2 */
  INT_FTM3                     = 63,               /**< FTM3 */
  INT_CMT                      = 64,               /**< CMT */
  INT_PDB0                     = 65,               /**< PDB0 */
  INT_PDB1                     = 66,               /**< PDB1 */
  INT_PDB2                     = 67,               /**< PDB2 */
  INT_PDB3                     = 68,               /**< PDB3 */
  INT_PIT0                     = 69,               /**< PIT channel 0 */
  INT_PIT1                     = 70,               /**< PIT channel 1 */
  INT_PIT2                     = 71,               /**< PIT channel 2 */
  INT_PIT3                     = 72,               /**< PIT channel 3 */
  INT_RTC                      = 73,               /**< RTC */
  INT_TSI0                     = 74,               /**< TSI0 */
  INT_PORTA                    = 75,               /**< Port A interrupt */
  INT_PORTB                    = 76,               /**< Port B interrupt */
  INT_PORTC                    = 77,               /**< Port C interrupt */
  INT_PORTD                    = 78,               /**< Port D interrupt */
  INT_PORTE                    = 79                /**< Port E interrupt */
} IRQInterruptIndex;

/**
 * @}
 */ /* end of group Interrupt_vector_numbers */


/* ----------------------------------------------------------------------------
   -- Peripheral type defines
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup Peripheral_defines Peripheral type defines
 * @{
 */


/*
** Start of section using anonymous unions
*/

#if defined(__ARMCC_VERSION)
  #pragma push
  #pragma anon_unions
#elif defined(__CWCC__)
  #pragma push
  #pragma cpp_extensions on
#elif defined(__GNUC__)
  /* anonymous unions are enabled by default */
#elif defined(__IAR_SYSTEMS_ICC__)
  #pragma language=extended
#else
  #error Not supported compiler type
#endif

/* ----------------------------------------------------------------------------
   -- ADC
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup ADC_Peripheral ADC
 * @{
 */

/** ADC - Peripheral register structure */
typedef struct ADC_MemMap {
  uint16_t SC1;                                    /**< Status and Control Register 1, offset: 0x0 */
  uint16_t SC2;                                    /**< Status and Control Register 2, offset: 0x2 */
  uint16_t SC3;                                    /**< Status and Control Register 3, offset: 0x4 */
  uint16_t SC4;                                    /**< Status and Control Register 4, offset: 0x6 */
  uint16_t R;                                      /**< Conversion Result Register, offset: 0x8 */
  uint16_t CVA;                                    /**< Compare Value Register A, offset: 0xA */
  uint16_t CVB;                                    /**< Compare Value Register B, offset: 0xC */
  uint16_t APCTL;                                  /**< Pin Control Register, offset: 0xE */
} volatile *ADC_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- ADC - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup ADC_Register_Accessor_Macros ADC - Register accessor macros
 * @{
 */


/* ADC - Register accessors */
#define ADC_SC1_REG(base)                        ((base)->SC1)
#define ADC_SC2_REG(base)                        ((base)->SC2)
#define ADC_SC3_REG(base)                        ((base)->SC3)
#define ADC_SC4_REG(base)                        ((base)->SC4)
#define ADC_R_REG(base)                          ((base)->R)
#define ADC_CVA_REG(base)                        ((base)->CVA)
#define ADC_CVB_REG(base)                        ((base)->CVB)
#define ADC_APCTL_REG(base)                      ((base)->APCTL)

/**
 * @}
 */ /* end of group ADC_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- ADC Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup ADC_Register_Masks ADC Register Masks
 * @{
 */

/* SC1 Bit Fields */
#define ADC_SC1_ADCH_MASK                        0x1Fu
#define ADC_SC1_ADCH_SHIFT                       0
#define ADC_SC1_ADCH(x)                          (((uint16_t)(((uint16_t)(x))<<ADC_SC1_ADCH_SHIFT))&ADC_SC1_ADCH_MASK)
#define ADC_SC1_ADCO_MASK                        0x20u
#define ADC_SC1_ADCO_SHIFT                       5
#define ADC_SC1_AIEN_MASK                        0x40u
#define ADC_SC1_AIEN_SHIFT                       6
#define ADC_SC1_COCO_MASK                        0x80u
#define ADC_SC1_COCO_SHIFT                       7
/* SC2 Bit Fields */
#define ADC_SC2_FFULL_MASK                       0x4u
#define ADC_SC2_FFULL_SHIFT                      2
#define ADC_SC2_FEMPTY_MASK                      0x8u
#define ADC_SC2_FEMPTY_SHIFT                     3
#define ADC_SC2_ACDSEL_MASK                      0x10u
#define ADC_SC2_ACDSEL_SHIFT                     4
#define ADC_SC2_ACFE_MASK                        0x20u
#define ADC_SC2_ACFE_SHIFT                       5
#define ADC_SC2_ADTRG_MASK                       0x40u
#define ADC_SC2_ADTRG_SHIFT                      6
#define ADC_SC2_ADACT_MASK                       0x80u
#define ADC_SC2_ADACT_SHIFT                      7
/* SC3 Bit Fields */
#define ADC_SC3_ADICLK_MASK                      0x3u
#define ADC_SC3_ADICLK_SHIFT                     0
#define ADC_SC3_ADICLK(x)                        (((uint16_t)(((uint16_t)(x))<<ADC_SC3_ADICLK_SHIFT))&ADC_SC3_ADICLK_MASK)
#define ADC_SC3_MODE_MASK                        0xCu
#define ADC_SC3_MODE_SHIFT                       2
#define ADC_SC3_MODE(x)                          (((uint16_t)(((uint16_t)(x))<<ADC_SC3_MODE_SHIFT))&ADC_SC3_MODE_MASK)
#define ADC_SC3_ADLSMP_MASK                      0x10u
#define ADC_SC3_ADLSMP_SHIFT                     4
#define ADC_SC3_ADIV_MASK                        0x60u
#define ADC_SC3_ADIV_SHIFT                       5
#define ADC_SC3_ADIV(x)                          (((uint16_t)(((uint16_t)(x))<<ADC_SC3_ADIV_SHIFT))&ADC_SC3_ADIV_MASK)
#define ADC_SC3_ADLPC_MASK                       0x80u
#define ADC_SC3_ADLPC_SHIFT                      7
/* SC4 Bit Fields */
#define ADC_SC4_AFDEP_MASK                       0x7u
#define ADC_SC4_AFDEP_SHIFT                      0
#define ADC_SC4_AFDEP(x)                         (((uint16_t)(((uint16_t)(x))<<ADC_SC4_AFDEP_SHIFT))&ADC_SC4_AFDEP_MASK)
#define ADC_SC4_ACFSEL_MASK                      0x20u
#define ADC_SC4_ACFSEL_SHIFT                     5
#define ADC_SC4_ASCANE_MASK                      0x40u
#define ADC_SC4_ASCANE_SHIFT                     6
#define ADC_SC4_DMAEN_MASK                       0x80u
#define ADC_SC4_DMAEN_SHIFT                      7
/* R Bit Fields */
#define ADC_R_ADR_MASK                           0xFFFFu
#define ADC_R_ADR_SHIFT                          0
#define ADC_R_ADR(x)                             (((uint16_t)(((uint16_t)(x))<<ADC_R_ADR_SHIFT))&ADC_R_ADR_MASK)
/* CVA Bit Fields */
#define ADC_CVA_VA_MASK                          0xFFFFu
#define ADC_CVA_VA_SHIFT                         0
#define ADC_CVA_VA(x)                            (((uint16_t)(((uint16_t)(x))<<ADC_CVA_VA_SHIFT))&ADC_CVA_VA_MASK)
/* CVB Bit Fields */
#define ADC_CVB_VB_MASK                          0xFFFFu
#define ADC_CVB_VB_SHIFT                         0
#define ADC_CVB_VB(x)                            (((uint16_t)(((uint16_t)(x))<<ADC_CVB_VB_SHIFT))&ADC_CVB_VB_MASK)
/* APCTL Bit Fields */
#define ADC_APCTL_ADPC0_MASK                     0x1u
#define ADC_APCTL_ADPC0_SHIFT                    0
#define ADC_APCTL_ADPC1_MASK                     0x2u
#define ADC_APCTL_ADPC1_SHIFT                    1
#define ADC_APCTL_ADPC2_MASK                     0x4u
#define ADC_APCTL_ADPC2_SHIFT                    2
#define ADC_APCTL_ADPC3_MASK                     0x8u
#define ADC_APCTL_ADPC3_SHIFT                    3
#define ADC_APCTL_ADPC4_MASK                     0x10u
#define ADC_APCTL_ADPC4_SHIFT                    4
#define ADC_APCTL_ADPC5_MASK                     0x20u
#define ADC_APCTL_ADPC5_SHIFT                    5
#define ADC_APCTL_ADPC6_MASK                     0x40u
#define ADC_APCTL_ADPC6_SHIFT                    6
#define ADC_APCTL_ADPC7_MASK                     0x80u
#define ADC_APCTL_ADPC7_SHIFT                    7
#define ADC_APCTL_ADPC8_MASK                     0x100u
#define ADC_APCTL_ADPC8_SHIFT                    8
#define ADC_APCTL_ADPC9_MASK                     0x200u
#define ADC_APCTL_ADPC9_SHIFT                    9
#define ADC_APCTL_ADPC10_MASK                    0x400u
#define ADC_APCTL_ADPC10_SHIFT                   10
#define ADC_APCTL_ADPC11_MASK                    0x800u
#define ADC_APCTL_ADPC11_SHIFT                   11

/**
 * @}
 */ /* end of group ADC_Register_Masks */


/* ADC - Peripheral instance base addresses */
/** Peripheral ADC0 base pointer */
#define ADC0_BASE_PTR                            ((ADC_MemMapPtr)0x4003B000u)
/** Peripheral ADC1 base pointer */
#define ADC1_BASE_PTR                            ((ADC_MemMapPtr)0x4003C000u)
/** Peripheral ADC2 base pointer */
#define ADC2_BASE_PTR                            ((ADC_MemMapPtr)0x4003D000u)
/** Peripheral ADC3 base pointer */
#define ADC3_BASE_PTR                            ((ADC_MemMapPtr)0x4003E000u)

/* ----------------------------------------------------------------------------
   -- ADC - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup ADC_Register_Accessor_Macros ADC - Register accessor macros
 * @{
 */


/* ADC - Register instance definitions */
/* ADC0 */
#define ADC0_SC1                                 ADC_SC1_REG(ADC0_BASE_PTR)
#define ADC0_SC2                                 ADC_SC2_REG(ADC0_BASE_PTR)
#define ADC0_SC3                                 ADC_SC3_REG(ADC0_BASE_PTR)
#define ADC0_SC4                                 ADC_SC4_REG(ADC0_BASE_PTR)
#define ADC0_R                                   ADC_R_REG(ADC0_BASE_PTR)
#define ADC0_CVA                                 ADC_CVA_REG(ADC0_BASE_PTR)
#define ADC0_CVB                                 ADC_CVB_REG(ADC0_BASE_PTR)
#define ADC0_APCTL                               ADC_APCTL_REG(ADC0_BASE_PTR)
/* ADC1 */
#define ADC1_SC1                                 ADC_SC1_REG(ADC1_BASE_PTR)
#define ADC1_SC2                                 ADC_SC2_REG(ADC1_BASE_PTR)
#define ADC1_SC3                                 ADC_SC3_REG(ADC1_BASE_PTR)
#define ADC1_SC4                                 ADC_SC4_REG(ADC1_BASE_PTR)
#define ADC1_R                                   ADC_R_REG(ADC1_BASE_PTR)
#define ADC1_CVA                                 ADC_CVA_REG(ADC1_BASE_PTR)
#define ADC1_CVB                                 ADC_CVB_REG(ADC1_BASE_PTR)
#define ADC1_APCTL                               ADC_APCTL_REG(ADC1_BASE_PTR)
/* ADC2 */
#define ADC2_SC1                                 ADC_SC1_REG(ADC2_BASE_PTR)
#define ADC2_SC2                                 ADC_SC2_REG(ADC2_BASE_PTR)
#define ADC2_SC3                                 ADC_SC3_REG(ADC2_BASE_PTR)
#define ADC2_SC4                                 ADC_SC4_REG(ADC2_BASE_PTR)
#define ADC2_R                                   ADC_R_REG(ADC2_BASE_PTR)
#define ADC2_CVA                                 ADC_CVA_REG(ADC2_BASE_PTR)
#define ADC2_CVB                                 ADC_CVB_REG(ADC2_BASE_PTR)
#define ADC2_APCTL                               ADC_APCTL_REG(ADC2_BASE_PTR)
/* ADC3 */
#define ADC3_SC1                                 ADC_SC1_REG(ADC3_BASE_PTR)
#define ADC3_SC2                                 ADC_SC2_REG(ADC3_BASE_PTR)
#define ADC3_SC3                                 ADC_SC3_REG(ADC3_BASE_PTR)
#define ADC3_SC4                                 ADC_SC4_REG(ADC3_BASE_PTR)
#define ADC3_R                                   ADC_R_REG(ADC3_BASE_PTR)
#define ADC3_CVA                                 ADC_CVA_REG(ADC3_BASE_PTR)
#define ADC3_CVB                                 ADC_CVB_REG(ADC3_BASE_PTR)
#define ADC3_APCTL                               ADC_APCTL_REG(ADC3_BASE_PTR)

/**
 * @}
 */ /* end of group ADC_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group ADC_Peripheral */


/* ----------------------------------------------------------------------------
   -- AIPS
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup AIPS_Peripheral AIPS
 * @{
 */

/** AIPS - Peripheral register structure */
typedef struct AIPS_MemMap {
  uint32_t MPRA;                                   /**< Master Privilege Register A, offset: 0x0 */
  uint8_t RESERVED_0[28];
  uint32_t PACRA;                                  /**< Peripheral Access Control Register, offset: 0x20 */
  uint32_t PACRB;                                  /**< Peripheral Access Control Register, offset: 0x24 */
  uint32_t PACRC;                                  /**< Peripheral Access Control Register, offset: 0x28 */
  uint32_t PACRD;                                  /**< Peripheral Access Control Register, offset: 0x2C */
  uint8_t RESERVED_1[16];
  uint32_t PACRE;                                  /**< Peripheral Access Control Register, offset: 0x40 */
  uint32_t PACRF;                                  /**< Peripheral Access Control Register, offset: 0x44 */
  uint32_t PACRG;                                  /**< Peripheral Access Control Register, offset: 0x48 */
  uint32_t PACRH;                                  /**< Peripheral Access Control Register, offset: 0x4C */
  uint32_t PACRI;                                  /**< Peripheral Access Control Register, offset: 0x50 */
  uint32_t PACRJ;                                  /**< Peripheral Access Control Register, offset: 0x54 */
  uint32_t PACRK;                                  /**< Peripheral Access Control Register, offset: 0x58 */
  uint32_t PACRL;                                  /**< Peripheral Access Control Register, offset: 0x5C */
  uint32_t PACRM;                                  /**< Peripheral Access Control Register, offset: 0x60 */
  uint32_t PACRN;                                  /**< Peripheral Access Control Register, offset: 0x64 */
  uint32_t PACRO;                                  /**< Peripheral Access Control Register, offset: 0x68 */
  uint32_t PACRP;                                  /**< Peripheral Access Control Register, offset: 0x6C */
  uint8_t RESERVED_2[16];
  uint32_t PACRU;                                  /**< Peripheral Access Control Register U, offset: 0x80 */
} volatile *AIPS_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- AIPS - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup AIPS_Register_Accessor_Macros AIPS - Register accessor macros
 * @{
 */


/* AIPS - Register accessors */
#define AIPS_MPRA_REG(base)                      ((base)->MPRA)
#define AIPS_PACRA_REG(base)                     ((base)->PACRA)
#define AIPS_PACRB_REG(base)                     ((base)->PACRB)
#define AIPS_PACRC_REG(base)                     ((base)->PACRC)
#define AIPS_PACRD_REG(base)                     ((base)->PACRD)
#define AIPS_PACRE_REG(base)                     ((base)->PACRE)
#define AIPS_PACRF_REG(base)                     ((base)->PACRF)
#define AIPS_PACRG_REG(base)                     ((base)->PACRG)
#define AIPS_PACRH_REG(base)                     ((base)->PACRH)
#define AIPS_PACRI_REG(base)                     ((base)->PACRI)
#define AIPS_PACRJ_REG(base)                     ((base)->PACRJ)
#define AIPS_PACRK_REG(base)                     ((base)->PACRK)
#define AIPS_PACRL_REG(base)                     ((base)->PACRL)
#define AIPS_PACRM_REG(base)                     ((base)->PACRM)
#define AIPS_PACRN_REG(base)                     ((base)->PACRN)
#define AIPS_PACRO_REG(base)                     ((base)->PACRO)
#define AIPS_PACRP_REG(base)                     ((base)->PACRP)
#define AIPS_PACRU_REG(base)                     ((base)->PACRU)

/**
 * @}
 */ /* end of group AIPS_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- AIPS Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup AIPS_Register_Masks AIPS Register Masks
 * @{
 */

/* MPRA Bit Fields */
#define AIPS_MPRA_MPL2_MASK                      0x100000u
#define AIPS_MPRA_MPL2_SHIFT                     20
#define AIPS_MPRA_MTW2_MASK                      0x200000u
#define AIPS_MPRA_MTW2_SHIFT                     21
#define AIPS_MPRA_MTR2_MASK                      0x400000u
#define AIPS_MPRA_MTR2_SHIFT                     22
#define AIPS_MPRA_MPL1_MASK                      0x1000000u
#define AIPS_MPRA_MPL1_SHIFT                     24
#define AIPS_MPRA_MTW1_MASK                      0x2000000u
#define AIPS_MPRA_MTW1_SHIFT                     25
#define AIPS_MPRA_MTR1_MASK                      0x4000000u
#define AIPS_MPRA_MTR1_SHIFT                     26
#define AIPS_MPRA_MPL0_MASK                      0x10000000u
#define AIPS_MPRA_MPL0_SHIFT                     28
#define AIPS_MPRA_MTW0_MASK                      0x20000000u
#define AIPS_MPRA_MTW0_SHIFT                     29
#define AIPS_MPRA_MTR0_MASK                      0x40000000u
#define AIPS_MPRA_MTR0_SHIFT                     30
/* PACRA Bit Fields */
#define AIPS_PACRA_TP7_MASK                      0x1u
#define AIPS_PACRA_TP7_SHIFT                     0
#define AIPS_PACRA_WP7_MASK                      0x2u
#define AIPS_PACRA_WP7_SHIFT                     1
#define AIPS_PACRA_SP7_MASK                      0x4u
#define AIPS_PACRA_SP7_SHIFT                     2
#define AIPS_PACRA_TP6_MASK                      0x10u
#define AIPS_PACRA_TP6_SHIFT                     4
#define AIPS_PACRA_WP6_MASK                      0x20u
#define AIPS_PACRA_WP6_SHIFT                     5
#define AIPS_PACRA_SP6_MASK                      0x40u
#define AIPS_PACRA_SP6_SHIFT                     6
#define AIPS_PACRA_TP5_MASK                      0x100u
#define AIPS_PACRA_TP5_SHIFT                     8
#define AIPS_PACRA_WP5_MASK                      0x200u
#define AIPS_PACRA_WP5_SHIFT                     9
#define AIPS_PACRA_SP5_MASK                      0x400u
#define AIPS_PACRA_SP5_SHIFT                     10
#define AIPS_PACRA_TP4_MASK                      0x1000u
#define AIPS_PACRA_TP4_SHIFT                     12
#define AIPS_PACRA_WP4_MASK                      0x2000u
#define AIPS_PACRA_WP4_SHIFT                     13
#define AIPS_PACRA_SP4_MASK                      0x4000u
#define AIPS_PACRA_SP4_SHIFT                     14
#define AIPS_PACRA_TP3_MASK                      0x10000u
#define AIPS_PACRA_TP3_SHIFT                     16
#define AIPS_PACRA_WP3_MASK                      0x20000u
#define AIPS_PACRA_WP3_SHIFT                     17
#define AIPS_PACRA_SP3_MASK                      0x40000u
#define AIPS_PACRA_SP3_SHIFT                     18
#define AIPS_PACRA_TP2_MASK                      0x100000u
#define AIPS_PACRA_TP2_SHIFT                     20
#define AIPS_PACRA_WP2_MASK                      0x200000u
#define AIPS_PACRA_WP2_SHIFT                     21
#define AIPS_PACRA_SP2_MASK                      0x400000u
#define AIPS_PACRA_SP2_SHIFT                     22
#define AIPS_PACRA_TP1_MASK                      0x1000000u
#define AIPS_PACRA_TP1_SHIFT                     24
#define AIPS_PACRA_WP1_MASK                      0x2000000u
#define AIPS_PACRA_WP1_SHIFT                     25
#define AIPS_PACRA_SP1_MASK                      0x4000000u
#define AIPS_PACRA_SP1_SHIFT                     26
#define AIPS_PACRA_TP0_MASK                      0x10000000u
#define AIPS_PACRA_TP0_SHIFT                     28
#define AIPS_PACRA_WP0_MASK                      0x20000000u
#define AIPS_PACRA_WP0_SHIFT                     29
#define AIPS_PACRA_SP0_MASK                      0x40000000u
#define AIPS_PACRA_SP0_SHIFT                     30
/* PACRB Bit Fields */
#define AIPS_PACRB_TP7_MASK                      0x1u
#define AIPS_PACRB_TP7_SHIFT                     0
#define AIPS_PACRB_WP7_MASK                      0x2u
#define AIPS_PACRB_WP7_SHIFT                     1
#define AIPS_PACRB_SP7_MASK                      0x4u
#define AIPS_PACRB_SP7_SHIFT                     2
#define AIPS_PACRB_TP6_MASK                      0x10u
#define AIPS_PACRB_TP6_SHIFT                     4
#define AIPS_PACRB_WP6_MASK                      0x20u
#define AIPS_PACRB_WP6_SHIFT                     5
#define AIPS_PACRB_SP6_MASK                      0x40u
#define AIPS_PACRB_SP6_SHIFT                     6
#define AIPS_PACRB_TP5_MASK                      0x100u
#define AIPS_PACRB_TP5_SHIFT                     8
#define AIPS_PACRB_WP5_MASK                      0x200u
#define AIPS_PACRB_WP5_SHIFT                     9
#define AIPS_PACRB_SP5_MASK                      0x400u
#define AIPS_PACRB_SP5_SHIFT                     10
#define AIPS_PACRB_TP4_MASK                      0x1000u
#define AIPS_PACRB_TP4_SHIFT                     12
#define AIPS_PACRB_WP4_MASK                      0x2000u
#define AIPS_PACRB_WP4_SHIFT                     13
#define AIPS_PACRB_SP4_MASK                      0x4000u
#define AIPS_PACRB_SP4_SHIFT                     14
#define AIPS_PACRB_TP3_MASK                      0x10000u
#define AIPS_PACRB_TP3_SHIFT                     16
#define AIPS_PACRB_WP3_MASK                      0x20000u
#define AIPS_PACRB_WP3_SHIFT                     17
#define AIPS_PACRB_SP3_MASK                      0x40000u
#define AIPS_PACRB_SP3_SHIFT                     18
#define AIPS_PACRB_TP2_MASK                      0x100000u
#define AIPS_PACRB_TP2_SHIFT                     20
#define AIPS_PACRB_WP2_MASK                      0x200000u
#define AIPS_PACRB_WP2_SHIFT                     21
#define AIPS_PACRB_SP2_MASK                      0x400000u
#define AIPS_PACRB_SP2_SHIFT                     22
#define AIPS_PACRB_TP1_MASK                      0x1000000u
#define AIPS_PACRB_TP1_SHIFT                     24
#define AIPS_PACRB_WP1_MASK                      0x2000000u
#define AIPS_PACRB_WP1_SHIFT                     25
#define AIPS_PACRB_SP1_MASK                      0x4000000u
#define AIPS_PACRB_SP1_SHIFT                     26
#define AIPS_PACRB_TP0_MASK                      0x10000000u
#define AIPS_PACRB_TP0_SHIFT                     28
#define AIPS_PACRB_WP0_MASK                      0x20000000u
#define AIPS_PACRB_WP0_SHIFT                     29
#define AIPS_PACRB_SP0_MASK                      0x40000000u
#define AIPS_PACRB_SP0_SHIFT                     30
/* PACRC Bit Fields */
#define AIPS_PACRC_TP7_MASK                      0x1u
#define AIPS_PACRC_TP7_SHIFT                     0
#define AIPS_PACRC_WP7_MASK                      0x2u
#define AIPS_PACRC_WP7_SHIFT                     1
#define AIPS_PACRC_SP7_MASK                      0x4u
#define AIPS_PACRC_SP7_SHIFT                     2
#define AIPS_PACRC_TP6_MASK                      0x10u
#define AIPS_PACRC_TP6_SHIFT                     4
#define AIPS_PACRC_WP6_MASK                      0x20u
#define AIPS_PACRC_WP6_SHIFT                     5
#define AIPS_PACRC_SP6_MASK                      0x40u
#define AIPS_PACRC_SP6_SHIFT                     6
#define AIPS_PACRC_TP5_MASK                      0x100u
#define AIPS_PACRC_TP5_SHIFT                     8
#define AIPS_PACRC_WP5_MASK                      0x200u
#define AIPS_PACRC_WP5_SHIFT                     9
#define AIPS_PACRC_SP5_MASK                      0x400u
#define AIPS_PACRC_SP5_SHIFT                     10
#define AIPS_PACRC_TP4_MASK                      0x1000u
#define AIPS_PACRC_TP4_SHIFT                     12
#define AIPS_PACRC_WP4_MASK                      0x2000u
#define AIPS_PACRC_WP4_SHIFT                     13
#define AIPS_PACRC_SP4_MASK                      0x4000u
#define AIPS_PACRC_SP4_SHIFT                     14
#define AIPS_PACRC_TP3_MASK                      0x10000u
#define AIPS_PACRC_TP3_SHIFT                     16
#define AIPS_PACRC_WP3_MASK                      0x20000u
#define AIPS_PACRC_WP3_SHIFT                     17
#define AIPS_PACRC_SP3_MASK                      0x40000u
#define AIPS_PACRC_SP3_SHIFT                     18
#define AIPS_PACRC_TP2_MASK                      0x100000u
#define AIPS_PACRC_TP2_SHIFT                     20
#define AIPS_PACRC_WP2_MASK                      0x200000u
#define AIPS_PACRC_WP2_SHIFT                     21
#define AIPS_PACRC_SP2_MASK                      0x400000u
#define AIPS_PACRC_SP2_SHIFT                     22
#define AIPS_PACRC_TP1_MASK                      0x1000000u
#define AIPS_PACRC_TP1_SHIFT                     24
#define AIPS_PACRC_WP1_MASK                      0x2000000u
#define AIPS_PACRC_WP1_SHIFT                     25
#define AIPS_PACRC_SP1_MASK                      0x4000000u
#define AIPS_PACRC_SP1_SHIFT                     26
#define AIPS_PACRC_TP0_MASK                      0x10000000u
#define AIPS_PACRC_TP0_SHIFT                     28
#define AIPS_PACRC_WP0_MASK                      0x20000000u
#define AIPS_PACRC_WP0_SHIFT                     29
#define AIPS_PACRC_SP0_MASK                      0x40000000u
#define AIPS_PACRC_SP0_SHIFT                     30
/* PACRD Bit Fields */
#define AIPS_PACRD_TP7_MASK                      0x1u
#define AIPS_PACRD_TP7_SHIFT                     0
#define AIPS_PACRD_WP7_MASK                      0x2u
#define AIPS_PACRD_WP7_SHIFT                     1
#define AIPS_PACRD_SP7_MASK                      0x4u
#define AIPS_PACRD_SP7_SHIFT                     2
#define AIPS_PACRD_TP6_MASK                      0x10u
#define AIPS_PACRD_TP6_SHIFT                     4
#define AIPS_PACRD_WP6_MASK                      0x20u
#define AIPS_PACRD_WP6_SHIFT                     5
#define AIPS_PACRD_SP6_MASK                      0x40u
#define AIPS_PACRD_SP6_SHIFT                     6
#define AIPS_PACRD_TP5_MASK                      0x100u
#define AIPS_PACRD_TP5_SHIFT                     8
#define AIPS_PACRD_WP5_MASK                      0x200u
#define AIPS_PACRD_WP5_SHIFT                     9
#define AIPS_PACRD_SP5_MASK                      0x400u
#define AIPS_PACRD_SP5_SHIFT                     10
#define AIPS_PACRD_TP4_MASK                      0x1000u
#define AIPS_PACRD_TP4_SHIFT                     12
#define AIPS_PACRD_WP4_MASK                      0x2000u
#define AIPS_PACRD_WP4_SHIFT                     13
#define AIPS_PACRD_SP4_MASK                      0x4000u
#define AIPS_PACRD_SP4_SHIFT                     14
#define AIPS_PACRD_TP3_MASK                      0x10000u
#define AIPS_PACRD_TP3_SHIFT                     16
#define AIPS_PACRD_WP3_MASK                      0x20000u
#define AIPS_PACRD_WP3_SHIFT                     17
#define AIPS_PACRD_SP3_MASK                      0x40000u
#define AIPS_PACRD_SP3_SHIFT                     18
#define AIPS_PACRD_TP2_MASK                      0x100000u
#define AIPS_PACRD_TP2_SHIFT                     20
#define AIPS_PACRD_WP2_MASK                      0x200000u
#define AIPS_PACRD_WP2_SHIFT                     21
#define AIPS_PACRD_SP2_MASK                      0x400000u
#define AIPS_PACRD_SP2_SHIFT                     22
#define AIPS_PACRD_TP1_MASK                      0x1000000u
#define AIPS_PACRD_TP1_SHIFT                     24
#define AIPS_PACRD_WP1_MASK                      0x2000000u
#define AIPS_PACRD_WP1_SHIFT                     25
#define AIPS_PACRD_SP1_MASK                      0x4000000u
#define AIPS_PACRD_SP1_SHIFT                     26
#define AIPS_PACRD_TP0_MASK                      0x10000000u
#define AIPS_PACRD_TP0_SHIFT                     28
#define AIPS_PACRD_WP0_MASK                      0x20000000u
#define AIPS_PACRD_WP0_SHIFT                     29
#define AIPS_PACRD_SP0_MASK                      0x40000000u
#define AIPS_PACRD_SP0_SHIFT                     30
/* PACRE Bit Fields */
#define AIPS_PACRE_TP7_MASK                      0x1u
#define AIPS_PACRE_TP7_SHIFT                     0
#define AIPS_PACRE_WP7_MASK                      0x2u
#define AIPS_PACRE_WP7_SHIFT                     1
#define AIPS_PACRE_SP7_MASK                      0x4u
#define AIPS_PACRE_SP7_SHIFT                     2
#define AIPS_PACRE_TP6_MASK                      0x10u
#define AIPS_PACRE_TP6_SHIFT                     4
#define AIPS_PACRE_WP6_MASK                      0x20u
#define AIPS_PACRE_WP6_SHIFT                     5
#define AIPS_PACRE_SP6_MASK                      0x40u
#define AIPS_PACRE_SP6_SHIFT                     6
#define AIPS_PACRE_TP5_MASK                      0x100u
#define AIPS_PACRE_TP5_SHIFT                     8
#define AIPS_PACRE_WP5_MASK                      0x200u
#define AIPS_PACRE_WP5_SHIFT                     9
#define AIPS_PACRE_SP5_MASK                      0x400u
#define AIPS_PACRE_SP5_SHIFT                     10
#define AIPS_PACRE_TP4_MASK                      0x1000u
#define AIPS_PACRE_TP4_SHIFT                     12
#define AIPS_PACRE_WP4_MASK                      0x2000u
#define AIPS_PACRE_WP4_SHIFT                     13
#define AIPS_PACRE_SP4_MASK                      0x4000u
#define AIPS_PACRE_SP4_SHIFT                     14
#define AIPS_PACRE_TP3_MASK                      0x10000u
#define AIPS_PACRE_TP3_SHIFT                     16
#define AIPS_PACRE_WP3_MASK                      0x20000u
#define AIPS_PACRE_WP3_SHIFT                     17
#define AIPS_PACRE_SP3_MASK                      0x40000u
#define AIPS_PACRE_SP3_SHIFT                     18
#define AIPS_PACRE_TP2_MASK                      0x100000u
#define AIPS_PACRE_TP2_SHIFT                     20
#define AIPS_PACRE_WP2_MASK                      0x200000u
#define AIPS_PACRE_WP2_SHIFT                     21
#define AIPS_PACRE_SP2_MASK                      0x400000u
#define AIPS_PACRE_SP2_SHIFT                     22
#define AIPS_PACRE_TP1_MASK                      0x1000000u
#define AIPS_PACRE_TP1_SHIFT                     24
#define AIPS_PACRE_WP1_MASK                      0x2000000u
#define AIPS_PACRE_WP1_SHIFT                     25
#define AIPS_PACRE_SP1_MASK                      0x4000000u
#define AIPS_PACRE_SP1_SHIFT                     26
#define AIPS_PACRE_TP0_MASK                      0x10000000u
#define AIPS_PACRE_TP0_SHIFT                     28
#define AIPS_PACRE_WP0_MASK                      0x20000000u
#define AIPS_PACRE_WP0_SHIFT                     29
#define AIPS_PACRE_SP0_MASK                      0x40000000u
#define AIPS_PACRE_SP0_SHIFT                     30
/* PACRF Bit Fields */
#define AIPS_PACRF_TP7_MASK                      0x1u
#define AIPS_PACRF_TP7_SHIFT                     0
#define AIPS_PACRF_WP7_MASK                      0x2u
#define AIPS_PACRF_WP7_SHIFT                     1
#define AIPS_PACRF_SP7_MASK                      0x4u
#define AIPS_PACRF_SP7_SHIFT                     2
#define AIPS_PACRF_TP6_MASK                      0x10u
#define AIPS_PACRF_TP6_SHIFT                     4
#define AIPS_PACRF_WP6_MASK                      0x20u
#define AIPS_PACRF_WP6_SHIFT                     5
#define AIPS_PACRF_SP6_MASK                      0x40u
#define AIPS_PACRF_SP6_SHIFT                     6
#define AIPS_PACRF_TP5_MASK                      0x100u
#define AIPS_PACRF_TP5_SHIFT                     8
#define AIPS_PACRF_WP5_MASK                      0x200u
#define AIPS_PACRF_WP5_SHIFT                     9
#define AIPS_PACRF_SP5_MASK                      0x400u
#define AIPS_PACRF_SP5_SHIFT                     10
#define AIPS_PACRF_TP4_MASK                      0x1000u
#define AIPS_PACRF_TP4_SHIFT                     12
#define AIPS_PACRF_WP4_MASK                      0x2000u
#define AIPS_PACRF_WP4_SHIFT                     13
#define AIPS_PACRF_SP4_MASK                      0x4000u
#define AIPS_PACRF_SP4_SHIFT                     14
#define AIPS_PACRF_TP3_MASK                      0x10000u
#define AIPS_PACRF_TP3_SHIFT                     16
#define AIPS_PACRF_WP3_MASK                      0x20000u
#define AIPS_PACRF_WP3_SHIFT                     17
#define AIPS_PACRF_SP3_MASK                      0x40000u
#define AIPS_PACRF_SP3_SHIFT                     18
#define AIPS_PACRF_TP2_MASK                      0x100000u
#define AIPS_PACRF_TP2_SHIFT                     20
#define AIPS_PACRF_WP2_MASK                      0x200000u
#define AIPS_PACRF_WP2_SHIFT                     21
#define AIPS_PACRF_SP2_MASK                      0x400000u
#define AIPS_PACRF_SP2_SHIFT                     22
#define AIPS_PACRF_TP1_MASK                      0x1000000u
#define AIPS_PACRF_TP1_SHIFT                     24
#define AIPS_PACRF_WP1_MASK                      0x2000000u
#define AIPS_PACRF_WP1_SHIFT                     25
#define AIPS_PACRF_SP1_MASK                      0x4000000u
#define AIPS_PACRF_SP1_SHIFT                     26
#define AIPS_PACRF_TP0_MASK                      0x10000000u
#define AIPS_PACRF_TP0_SHIFT                     28
#define AIPS_PACRF_WP0_MASK                      0x20000000u
#define AIPS_PACRF_WP0_SHIFT                     29
#define AIPS_PACRF_SP0_MASK                      0x40000000u
#define AIPS_PACRF_SP0_SHIFT                     30
/* PACRG Bit Fields */
#define AIPS_PACRG_TP7_MASK                      0x1u
#define AIPS_PACRG_TP7_SHIFT                     0
#define AIPS_PACRG_WP7_MASK                      0x2u
#define AIPS_PACRG_WP7_SHIFT                     1
#define AIPS_PACRG_SP7_MASK                      0x4u
#define AIPS_PACRG_SP7_SHIFT                     2
#define AIPS_PACRG_TP6_MASK                      0x10u
#define AIPS_PACRG_TP6_SHIFT                     4
#define AIPS_PACRG_WP6_MASK                      0x20u
#define AIPS_PACRG_WP6_SHIFT                     5
#define AIPS_PACRG_SP6_MASK                      0x40u
#define AIPS_PACRG_SP6_SHIFT                     6
#define AIPS_PACRG_TP5_MASK                      0x100u
#define AIPS_PACRG_TP5_SHIFT                     8
#define AIPS_PACRG_WP5_MASK                      0x200u
#define AIPS_PACRG_WP5_SHIFT                     9
#define AIPS_PACRG_SP5_MASK                      0x400u
#define AIPS_PACRG_SP5_SHIFT                     10
#define AIPS_PACRG_TP4_MASK                      0x1000u
#define AIPS_PACRG_TP4_SHIFT                     12
#define AIPS_PACRG_WP4_MASK                      0x2000u
#define AIPS_PACRG_WP4_SHIFT                     13
#define AIPS_PACRG_SP4_MASK                      0x4000u
#define AIPS_PACRG_SP4_SHIFT                     14
#define AIPS_PACRG_TP3_MASK                      0x10000u
#define AIPS_PACRG_TP3_SHIFT                     16
#define AIPS_PACRG_WP3_MASK                      0x20000u
#define AIPS_PACRG_WP3_SHIFT                     17
#define AIPS_PACRG_SP3_MASK                      0x40000u
#define AIPS_PACRG_SP3_SHIFT                     18
#define AIPS_PACRG_TP2_MASK                      0x100000u
#define AIPS_PACRG_TP2_SHIFT                     20
#define AIPS_PACRG_WP2_MASK                      0x200000u
#define AIPS_PACRG_WP2_SHIFT                     21
#define AIPS_PACRG_SP2_MASK                      0x400000u
#define AIPS_PACRG_SP2_SHIFT                     22
#define AIPS_PACRG_TP1_MASK                      0x1000000u
#define AIPS_PACRG_TP1_SHIFT                     24
#define AIPS_PACRG_WP1_MASK                      0x2000000u
#define AIPS_PACRG_WP1_SHIFT                     25
#define AIPS_PACRG_SP1_MASK                      0x4000000u
#define AIPS_PACRG_SP1_SHIFT                     26
#define AIPS_PACRG_TP0_MASK                      0x10000000u
#define AIPS_PACRG_TP0_SHIFT                     28
#define AIPS_PACRG_WP0_MASK                      0x20000000u
#define AIPS_PACRG_WP0_SHIFT                     29
#define AIPS_PACRG_SP0_MASK                      0x40000000u
#define AIPS_PACRG_SP0_SHIFT                     30
/* PACRH Bit Fields */
#define AIPS_PACRH_TP7_MASK                      0x1u
#define AIPS_PACRH_TP7_SHIFT                     0
#define AIPS_PACRH_WP7_MASK                      0x2u
#define AIPS_PACRH_WP7_SHIFT                     1
#define AIPS_PACRH_SP7_MASK                      0x4u
#define AIPS_PACRH_SP7_SHIFT                     2
#define AIPS_PACRH_TP6_MASK                      0x10u
#define AIPS_PACRH_TP6_SHIFT                     4
#define AIPS_PACRH_WP6_MASK                      0x20u
#define AIPS_PACRH_WP6_SHIFT                     5
#define AIPS_PACRH_SP6_MASK                      0x40u
#define AIPS_PACRH_SP6_SHIFT                     6
#define AIPS_PACRH_TP5_MASK                      0x100u
#define AIPS_PACRH_TP5_SHIFT                     8
#define AIPS_PACRH_WP5_MASK                      0x200u
#define AIPS_PACRH_WP5_SHIFT                     9
#define AIPS_PACRH_SP5_MASK                      0x400u
#define AIPS_PACRH_SP5_SHIFT                     10
#define AIPS_PACRH_TP4_MASK                      0x1000u
#define AIPS_PACRH_TP4_SHIFT                     12
#define AIPS_PACRH_WP4_MASK                      0x2000u
#define AIPS_PACRH_WP4_SHIFT                     13
#define AIPS_PACRH_SP4_MASK                      0x4000u
#define AIPS_PACRH_SP4_SHIFT                     14
#define AIPS_PACRH_TP3_MASK                      0x10000u
#define AIPS_PACRH_TP3_SHIFT                     16
#define AIPS_PACRH_WP3_MASK                      0x20000u
#define AIPS_PACRH_WP3_SHIFT                     17
#define AIPS_PACRH_SP3_MASK                      0x40000u
#define AIPS_PACRH_SP3_SHIFT                     18
#define AIPS_PACRH_TP2_MASK                      0x100000u
#define AIPS_PACRH_TP2_SHIFT                     20
#define AIPS_PACRH_WP2_MASK                      0x200000u
#define AIPS_PACRH_WP2_SHIFT                     21
#define AIPS_PACRH_SP2_MASK                      0x400000u
#define AIPS_PACRH_SP2_SHIFT                     22
#define AIPS_PACRH_TP1_MASK                      0x1000000u
#define AIPS_PACRH_TP1_SHIFT                     24
#define AIPS_PACRH_WP1_MASK                      0x2000000u
#define AIPS_PACRH_WP1_SHIFT                     25
#define AIPS_PACRH_SP1_MASK                      0x4000000u
#define AIPS_PACRH_SP1_SHIFT                     26
#define AIPS_PACRH_TP0_MASK                      0x10000000u
#define AIPS_PACRH_TP0_SHIFT                     28
#define AIPS_PACRH_WP0_MASK                      0x20000000u
#define AIPS_PACRH_WP0_SHIFT                     29
#define AIPS_PACRH_SP0_MASK                      0x40000000u
#define AIPS_PACRH_SP0_SHIFT                     30
/* PACRI Bit Fields */
#define AIPS_PACRI_TP7_MASK                      0x1u
#define AIPS_PACRI_TP7_SHIFT                     0
#define AIPS_PACRI_WP7_MASK                      0x2u
#define AIPS_PACRI_WP7_SHIFT                     1
#define AIPS_PACRI_SP7_MASK                      0x4u
#define AIPS_PACRI_SP7_SHIFT                     2
#define AIPS_PACRI_TP6_MASK                      0x10u
#define AIPS_PACRI_TP6_SHIFT                     4
#define AIPS_PACRI_WP6_MASK                      0x20u
#define AIPS_PACRI_WP6_SHIFT                     5
#define AIPS_PACRI_SP6_MASK                      0x40u
#define AIPS_PACRI_SP6_SHIFT                     6
#define AIPS_PACRI_TP5_MASK                      0x100u
#define AIPS_PACRI_TP5_SHIFT                     8
#define AIPS_PACRI_WP5_MASK                      0x200u
#define AIPS_PACRI_WP5_SHIFT                     9
#define AIPS_PACRI_SP5_MASK                      0x400u
#define AIPS_PACRI_SP5_SHIFT                     10
#define AIPS_PACRI_TP4_MASK                      0x1000u
#define AIPS_PACRI_TP4_SHIFT                     12
#define AIPS_PACRI_WP4_MASK                      0x2000u
#define AIPS_PACRI_WP4_SHIFT                     13
#define AIPS_PACRI_SP4_MASK                      0x4000u
#define AIPS_PACRI_SP4_SHIFT                     14
#define AIPS_PACRI_TP3_MASK                      0x10000u
#define AIPS_PACRI_TP3_SHIFT                     16
#define AIPS_PACRI_WP3_MASK                      0x20000u
#define AIPS_PACRI_WP3_SHIFT                     17
#define AIPS_PACRI_SP3_MASK                      0x40000u
#define AIPS_PACRI_SP3_SHIFT                     18
#define AIPS_PACRI_TP2_MASK                      0x100000u
#define AIPS_PACRI_TP2_SHIFT                     20
#define AIPS_PACRI_WP2_MASK                      0x200000u
#define AIPS_PACRI_WP2_SHIFT                     21
#define AIPS_PACRI_SP2_MASK                      0x400000u
#define AIPS_PACRI_SP2_SHIFT                     22
#define AIPS_PACRI_TP1_MASK                      0x1000000u
#define AIPS_PACRI_TP1_SHIFT                     24
#define AIPS_PACRI_WP1_MASK                      0x2000000u
#define AIPS_PACRI_WP1_SHIFT                     25
#define AIPS_PACRI_SP1_MASK                      0x4000000u
#define AIPS_PACRI_SP1_SHIFT                     26
#define AIPS_PACRI_TP0_MASK                      0x10000000u
#define AIPS_PACRI_TP0_SHIFT                     28
#define AIPS_PACRI_WP0_MASK                      0x20000000u
#define AIPS_PACRI_WP0_SHIFT                     29
#define AIPS_PACRI_SP0_MASK                      0x40000000u
#define AIPS_PACRI_SP0_SHIFT                     30
/* PACRJ Bit Fields */
#define AIPS_PACRJ_TP7_MASK                      0x1u
#define AIPS_PACRJ_TP7_SHIFT                     0
#define AIPS_PACRJ_WP7_MASK                      0x2u
#define AIPS_PACRJ_WP7_SHIFT                     1
#define AIPS_PACRJ_SP7_MASK                      0x4u
#define AIPS_PACRJ_SP7_SHIFT                     2
#define AIPS_PACRJ_TP6_MASK                      0x10u
#define AIPS_PACRJ_TP6_SHIFT                     4
#define AIPS_PACRJ_WP6_MASK                      0x20u
#define AIPS_PACRJ_WP6_SHIFT                     5
#define AIPS_PACRJ_SP6_MASK                      0x40u
#define AIPS_PACRJ_SP6_SHIFT                     6
#define AIPS_PACRJ_TP5_MASK                      0x100u
#define AIPS_PACRJ_TP5_SHIFT                     8
#define AIPS_PACRJ_WP5_MASK                      0x200u
#define AIPS_PACRJ_WP5_SHIFT                     9
#define AIPS_PACRJ_SP5_MASK                      0x400u
#define AIPS_PACRJ_SP5_SHIFT                     10
#define AIPS_PACRJ_TP4_MASK                      0x1000u
#define AIPS_PACRJ_TP4_SHIFT                     12
#define AIPS_PACRJ_WP4_MASK                      0x2000u
#define AIPS_PACRJ_WP4_SHIFT                     13
#define AIPS_PACRJ_SP4_MASK                      0x4000u
#define AIPS_PACRJ_SP4_SHIFT                     14
#define AIPS_PACRJ_TP3_MASK                      0x10000u
#define AIPS_PACRJ_TP3_SHIFT                     16
#define AIPS_PACRJ_WP3_MASK                      0x20000u
#define AIPS_PACRJ_WP3_SHIFT                     17
#define AIPS_PACRJ_SP3_MASK                      0x40000u
#define AIPS_PACRJ_SP3_SHIFT                     18
#define AIPS_PACRJ_TP2_MASK                      0x100000u
#define AIPS_PACRJ_TP2_SHIFT                     20
#define AIPS_PACRJ_WP2_MASK                      0x200000u
#define AIPS_PACRJ_WP2_SHIFT                     21
#define AIPS_PACRJ_SP2_MASK                      0x400000u
#define AIPS_PACRJ_SP2_SHIFT                     22
#define AIPS_PACRJ_TP1_MASK                      0x1000000u
#define AIPS_PACRJ_TP1_SHIFT                     24
#define AIPS_PACRJ_WP1_MASK                      0x2000000u
#define AIPS_PACRJ_WP1_SHIFT                     25
#define AIPS_PACRJ_SP1_MASK                      0x4000000u
#define AIPS_PACRJ_SP1_SHIFT                     26
#define AIPS_PACRJ_TP0_MASK                      0x10000000u
#define AIPS_PACRJ_TP0_SHIFT                     28
#define AIPS_PACRJ_WP0_MASK                      0x20000000u
#define AIPS_PACRJ_WP0_SHIFT                     29
#define AIPS_PACRJ_SP0_MASK                      0x40000000u
#define AIPS_PACRJ_SP0_SHIFT                     30
/* PACRK Bit Fields */
#define AIPS_PACRK_TP7_MASK                      0x1u
#define AIPS_PACRK_TP7_SHIFT                     0
#define AIPS_PACRK_WP7_MASK                      0x2u
#define AIPS_PACRK_WP7_SHIFT                     1
#define AIPS_PACRK_SP7_MASK                      0x4u
#define AIPS_PACRK_SP7_SHIFT                     2
#define AIPS_PACRK_TP6_MASK                      0x10u
#define AIPS_PACRK_TP6_SHIFT                     4
#define AIPS_PACRK_WP6_MASK                      0x20u
#define AIPS_PACRK_WP6_SHIFT                     5
#define AIPS_PACRK_SP6_MASK                      0x40u
#define AIPS_PACRK_SP6_SHIFT                     6
#define AIPS_PACRK_TP5_MASK                      0x100u
#define AIPS_PACRK_TP5_SHIFT                     8
#define AIPS_PACRK_WP5_MASK                      0x200u
#define AIPS_PACRK_WP5_SHIFT                     9
#define AIPS_PACRK_SP5_MASK                      0x400u
#define AIPS_PACRK_SP5_SHIFT                     10
#define AIPS_PACRK_TP4_MASK                      0x1000u
#define AIPS_PACRK_TP4_SHIFT                     12
#define AIPS_PACRK_WP4_MASK                      0x2000u
#define AIPS_PACRK_WP4_SHIFT                     13
#define AIPS_PACRK_SP4_MASK                      0x4000u
#define AIPS_PACRK_SP4_SHIFT                     14
#define AIPS_PACRK_TP3_MASK                      0x10000u
#define AIPS_PACRK_TP3_SHIFT                     16
#define AIPS_PACRK_WP3_MASK                      0x20000u
#define AIPS_PACRK_WP3_SHIFT                     17
#define AIPS_PACRK_SP3_MASK                      0x40000u
#define AIPS_PACRK_SP3_SHIFT                     18
#define AIPS_PACRK_TP2_MASK                      0x100000u
#define AIPS_PACRK_TP2_SHIFT                     20
#define AIPS_PACRK_WP2_MASK                      0x200000u
#define AIPS_PACRK_WP2_SHIFT                     21
#define AIPS_PACRK_SP2_MASK                      0x400000u
#define AIPS_PACRK_SP2_SHIFT                     22
#define AIPS_PACRK_TP1_MASK                      0x1000000u
#define AIPS_PACRK_TP1_SHIFT                     24
#define AIPS_PACRK_WP1_MASK                      0x2000000u
#define AIPS_PACRK_WP1_SHIFT                     25
#define AIPS_PACRK_SP1_MASK                      0x4000000u
#define AIPS_PACRK_SP1_SHIFT                     26
#define AIPS_PACRK_TP0_MASK                      0x10000000u
#define AIPS_PACRK_TP0_SHIFT                     28
#define AIPS_PACRK_WP0_MASK                      0x20000000u
#define AIPS_PACRK_WP0_SHIFT                     29
#define AIPS_PACRK_SP0_MASK                      0x40000000u
#define AIPS_PACRK_SP0_SHIFT                     30
/* PACRL Bit Fields */
#define AIPS_PACRL_TP7_MASK                      0x1u
#define AIPS_PACRL_TP7_SHIFT                     0
#define AIPS_PACRL_WP7_MASK                      0x2u
#define AIPS_PACRL_WP7_SHIFT                     1
#define AIPS_PACRL_SP7_MASK                      0x4u
#define AIPS_PACRL_SP7_SHIFT                     2
#define AIPS_PACRL_TP6_MASK                      0x10u
#define AIPS_PACRL_TP6_SHIFT                     4
#define AIPS_PACRL_WP6_MASK                      0x20u
#define AIPS_PACRL_WP6_SHIFT                     5
#define AIPS_PACRL_SP6_MASK                      0x40u
#define AIPS_PACRL_SP6_SHIFT                     6
#define AIPS_PACRL_TP5_MASK                      0x100u
#define AIPS_PACRL_TP5_SHIFT                     8
#define AIPS_PACRL_WP5_MASK                      0x200u
#define AIPS_PACRL_WP5_SHIFT                     9
#define AIPS_PACRL_SP5_MASK                      0x400u
#define AIPS_PACRL_SP5_SHIFT                     10
#define AIPS_PACRL_TP4_MASK                      0x1000u
#define AIPS_PACRL_TP4_SHIFT                     12
#define AIPS_PACRL_WP4_MASK                      0x2000u
#define AIPS_PACRL_WP4_SHIFT                     13
#define AIPS_PACRL_SP4_MASK                      0x4000u
#define AIPS_PACRL_SP4_SHIFT                     14
#define AIPS_PACRL_TP3_MASK                      0x10000u
#define AIPS_PACRL_TP3_SHIFT                     16
#define AIPS_PACRL_WP3_MASK                      0x20000u
#define AIPS_PACRL_WP3_SHIFT                     17
#define AIPS_PACRL_SP3_MASK                      0x40000u
#define AIPS_PACRL_SP3_SHIFT                     18
#define AIPS_PACRL_TP2_MASK                      0x100000u
#define AIPS_PACRL_TP2_SHIFT                     20
#define AIPS_PACRL_WP2_MASK                      0x200000u
#define AIPS_PACRL_WP2_SHIFT                     21
#define AIPS_PACRL_SP2_MASK                      0x400000u
#define AIPS_PACRL_SP2_SHIFT                     22
#define AIPS_PACRL_TP1_MASK                      0x1000000u
#define AIPS_PACRL_TP1_SHIFT                     24
#define AIPS_PACRL_WP1_MASK                      0x2000000u
#define AIPS_PACRL_WP1_SHIFT                     25
#define AIPS_PACRL_SP1_MASK                      0x4000000u
#define AIPS_PACRL_SP1_SHIFT                     26
#define AIPS_PACRL_TP0_MASK                      0x10000000u
#define AIPS_PACRL_TP0_SHIFT                     28
#define AIPS_PACRL_WP0_MASK                      0x20000000u
#define AIPS_PACRL_WP0_SHIFT                     29
#define AIPS_PACRL_SP0_MASK                      0x40000000u
#define AIPS_PACRL_SP0_SHIFT                     30
/* PACRM Bit Fields */
#define AIPS_PACRM_TP7_MASK                      0x1u
#define AIPS_PACRM_TP7_SHIFT                     0
#define AIPS_PACRM_WP7_MASK                      0x2u
#define AIPS_PACRM_WP7_SHIFT                     1
#define AIPS_PACRM_SP7_MASK                      0x4u
#define AIPS_PACRM_SP7_SHIFT                     2
#define AIPS_PACRM_TP6_MASK                      0x10u
#define AIPS_PACRM_TP6_SHIFT                     4
#define AIPS_PACRM_WP6_MASK                      0x20u
#define AIPS_PACRM_WP6_SHIFT                     5
#define AIPS_PACRM_SP6_MASK                      0x40u
#define AIPS_PACRM_SP6_SHIFT                     6
#define AIPS_PACRM_TP5_MASK                      0x100u
#define AIPS_PACRM_TP5_SHIFT                     8
#define AIPS_PACRM_WP5_MASK                      0x200u
#define AIPS_PACRM_WP5_SHIFT                     9
#define AIPS_PACRM_SP5_MASK                      0x400u
#define AIPS_PACRM_SP5_SHIFT                     10
#define AIPS_PACRM_TP4_MASK                      0x1000u
#define AIPS_PACRM_TP4_SHIFT                     12
#define AIPS_PACRM_WP4_MASK                      0x2000u
#define AIPS_PACRM_WP4_SHIFT                     13
#define AIPS_PACRM_SP4_MASK                      0x4000u
#define AIPS_PACRM_SP4_SHIFT                     14
#define AIPS_PACRM_TP3_MASK                      0x10000u
#define AIPS_PACRM_TP3_SHIFT                     16
#define AIPS_PACRM_WP3_MASK                      0x20000u
#define AIPS_PACRM_WP3_SHIFT                     17
#define AIPS_PACRM_SP3_MASK                      0x40000u
#define AIPS_PACRM_SP3_SHIFT                     18
#define AIPS_PACRM_TP2_MASK                      0x100000u
#define AIPS_PACRM_TP2_SHIFT                     20
#define AIPS_PACRM_WP2_MASK                      0x200000u
#define AIPS_PACRM_WP2_SHIFT                     21
#define AIPS_PACRM_SP2_MASK                      0x400000u
#define AIPS_PACRM_SP2_SHIFT                     22
#define AIPS_PACRM_TP1_MASK                      0x1000000u
#define AIPS_PACRM_TP1_SHIFT                     24
#define AIPS_PACRM_WP1_MASK                      0x2000000u
#define AIPS_PACRM_WP1_SHIFT                     25
#define AIPS_PACRM_SP1_MASK                      0x4000000u
#define AIPS_PACRM_SP1_SHIFT                     26
#define AIPS_PACRM_TP0_MASK                      0x10000000u
#define AIPS_PACRM_TP0_SHIFT                     28
#define AIPS_PACRM_WP0_MASK                      0x20000000u
#define AIPS_PACRM_WP0_SHIFT                     29
#define AIPS_PACRM_SP0_MASK                      0x40000000u
#define AIPS_PACRM_SP0_SHIFT                     30
/* PACRN Bit Fields */
#define AIPS_PACRN_TP7_MASK                      0x1u
#define AIPS_PACRN_TP7_SHIFT                     0
#define AIPS_PACRN_WP7_MASK                      0x2u
#define AIPS_PACRN_WP7_SHIFT                     1
#define AIPS_PACRN_SP7_MASK                      0x4u
#define AIPS_PACRN_SP7_SHIFT                     2
#define AIPS_PACRN_TP6_MASK                      0x10u
#define AIPS_PACRN_TP6_SHIFT                     4
#define AIPS_PACRN_WP6_MASK                      0x20u
#define AIPS_PACRN_WP6_SHIFT                     5
#define AIPS_PACRN_SP6_MASK                      0x40u
#define AIPS_PACRN_SP6_SHIFT                     6
#define AIPS_PACRN_TP5_MASK                      0x100u
#define AIPS_PACRN_TP5_SHIFT                     8
#define AIPS_PACRN_WP5_MASK                      0x200u
#define AIPS_PACRN_WP5_SHIFT                     9
#define AIPS_PACRN_SP5_MASK                      0x400u
#define AIPS_PACRN_SP5_SHIFT                     10
#define AIPS_PACRN_TP4_MASK                      0x1000u
#define AIPS_PACRN_TP4_SHIFT                     12
#define AIPS_PACRN_WP4_MASK                      0x2000u
#define AIPS_PACRN_WP4_SHIFT                     13
#define AIPS_PACRN_SP4_MASK                      0x4000u
#define AIPS_PACRN_SP4_SHIFT                     14
#define AIPS_PACRN_TP3_MASK                      0x10000u
#define AIPS_PACRN_TP3_SHIFT                     16
#define AIPS_PACRN_WP3_MASK                      0x20000u
#define AIPS_PACRN_WP3_SHIFT                     17
#define AIPS_PACRN_SP3_MASK                      0x40000u
#define AIPS_PACRN_SP3_SHIFT                     18
#define AIPS_PACRN_TP2_MASK                      0x100000u
#define AIPS_PACRN_TP2_SHIFT                     20
#define AIPS_PACRN_WP2_MASK                      0x200000u
#define AIPS_PACRN_WP2_SHIFT                     21
#define AIPS_PACRN_SP2_MASK                      0x400000u
#define AIPS_PACRN_SP2_SHIFT                     22
#define AIPS_PACRN_TP1_MASK                      0x1000000u
#define AIPS_PACRN_TP1_SHIFT                     24
#define AIPS_PACRN_WP1_MASK                      0x2000000u
#define AIPS_PACRN_WP1_SHIFT                     25
#define AIPS_PACRN_SP1_MASK                      0x4000000u
#define AIPS_PACRN_SP1_SHIFT                     26
#define AIPS_PACRN_TP0_MASK                      0x10000000u
#define AIPS_PACRN_TP0_SHIFT                     28
#define AIPS_PACRN_WP0_MASK                      0x20000000u
#define AIPS_PACRN_WP0_SHIFT                     29
#define AIPS_PACRN_SP0_MASK                      0x40000000u
#define AIPS_PACRN_SP0_SHIFT                     30
/* PACRO Bit Fields */
#define AIPS_PACRO_TP7_MASK                      0x1u
#define AIPS_PACRO_TP7_SHIFT                     0
#define AIPS_PACRO_WP7_MASK                      0x2u
#define AIPS_PACRO_WP7_SHIFT                     1
#define AIPS_PACRO_SP7_MASK                      0x4u
#define AIPS_PACRO_SP7_SHIFT                     2
#define AIPS_PACRO_TP6_MASK                      0x10u
#define AIPS_PACRO_TP6_SHIFT                     4
#define AIPS_PACRO_WP6_MASK                      0x20u
#define AIPS_PACRO_WP6_SHIFT                     5
#define AIPS_PACRO_SP6_MASK                      0x40u
#define AIPS_PACRO_SP6_SHIFT                     6
#define AIPS_PACRO_TP5_MASK                      0x100u
#define AIPS_PACRO_TP5_SHIFT                     8
#define AIPS_PACRO_WP5_MASK                      0x200u
#define AIPS_PACRO_WP5_SHIFT                     9
#define AIPS_PACRO_SP5_MASK                      0x400u
#define AIPS_PACRO_SP5_SHIFT                     10
#define AIPS_PACRO_TP4_MASK                      0x1000u
#define AIPS_PACRO_TP4_SHIFT                     12
#define AIPS_PACRO_WP4_MASK                      0x2000u
#define AIPS_PACRO_WP4_SHIFT                     13
#define AIPS_PACRO_SP4_MASK                      0x4000u
#define AIPS_PACRO_SP4_SHIFT                     14
#define AIPS_PACRO_TP3_MASK                      0x10000u
#define AIPS_PACRO_TP3_SHIFT                     16
#define AIPS_PACRO_WP3_MASK                      0x20000u
#define AIPS_PACRO_WP3_SHIFT                     17
#define AIPS_PACRO_SP3_MASK                      0x40000u
#define AIPS_PACRO_SP3_SHIFT                     18
#define AIPS_PACRO_TP2_MASK                      0x100000u
#define AIPS_PACRO_TP2_SHIFT                     20
#define AIPS_PACRO_WP2_MASK                      0x200000u
#define AIPS_PACRO_WP2_SHIFT                     21
#define AIPS_PACRO_SP2_MASK                      0x400000u
#define AIPS_PACRO_SP2_SHIFT                     22
#define AIPS_PACRO_TP1_MASK                      0x1000000u
#define AIPS_PACRO_TP1_SHIFT                     24
#define AIPS_PACRO_WP1_MASK                      0x2000000u
#define AIPS_PACRO_WP1_SHIFT                     25
#define AIPS_PACRO_SP1_MASK                      0x4000000u
#define AIPS_PACRO_SP1_SHIFT                     26
#define AIPS_PACRO_TP0_MASK                      0x10000000u
#define AIPS_PACRO_TP0_SHIFT                     28
#define AIPS_PACRO_WP0_MASK                      0x20000000u
#define AIPS_PACRO_WP0_SHIFT                     29
#define AIPS_PACRO_SP0_MASK                      0x40000000u
#define AIPS_PACRO_SP0_SHIFT                     30
/* PACRP Bit Fields */
#define AIPS_PACRP_TP7_MASK                      0x1u
#define AIPS_PACRP_TP7_SHIFT                     0
#define AIPS_PACRP_WP7_MASK                      0x2u
#define AIPS_PACRP_WP7_SHIFT                     1
#define AIPS_PACRP_SP7_MASK                      0x4u
#define AIPS_PACRP_SP7_SHIFT                     2
#define AIPS_PACRP_TP6_MASK                      0x10u
#define AIPS_PACRP_TP6_SHIFT                     4
#define AIPS_PACRP_WP6_MASK                      0x20u
#define AIPS_PACRP_WP6_SHIFT                     5
#define AIPS_PACRP_SP6_MASK                      0x40u
#define AIPS_PACRP_SP6_SHIFT                     6
#define AIPS_PACRP_TP5_MASK                      0x100u
#define AIPS_PACRP_TP5_SHIFT                     8
#define AIPS_PACRP_WP5_MASK                      0x200u
#define AIPS_PACRP_WP5_SHIFT                     9
#define AIPS_PACRP_SP5_MASK                      0x400u
#define AIPS_PACRP_SP5_SHIFT                     10
#define AIPS_PACRP_TP4_MASK                      0x1000u
#define AIPS_PACRP_TP4_SHIFT                     12
#define AIPS_PACRP_WP4_MASK                      0x2000u
#define AIPS_PACRP_WP4_SHIFT                     13
#define AIPS_PACRP_SP4_MASK                      0x4000u
#define AIPS_PACRP_SP4_SHIFT                     14
#define AIPS_PACRP_TP3_MASK                      0x10000u
#define AIPS_PACRP_TP3_SHIFT                     16
#define AIPS_PACRP_WP3_MASK                      0x20000u
#define AIPS_PACRP_WP3_SHIFT                     17
#define AIPS_PACRP_SP3_MASK                      0x40000u
#define AIPS_PACRP_SP3_SHIFT                     18
#define AIPS_PACRP_TP2_MASK                      0x100000u
#define AIPS_PACRP_TP2_SHIFT                     20
#define AIPS_PACRP_WP2_MASK                      0x200000u
#define AIPS_PACRP_WP2_SHIFT                     21
#define AIPS_PACRP_SP2_MASK                      0x400000u
#define AIPS_PACRP_SP2_SHIFT                     22
#define AIPS_PACRP_TP1_MASK                      0x1000000u
#define AIPS_PACRP_TP1_SHIFT                     24
#define AIPS_PACRP_WP1_MASK                      0x2000000u
#define AIPS_PACRP_WP1_SHIFT                     25
#define AIPS_PACRP_SP1_MASK                      0x4000000u
#define AIPS_PACRP_SP1_SHIFT                     26
#define AIPS_PACRP_TP0_MASK                      0x10000000u
#define AIPS_PACRP_TP0_SHIFT                     28
#define AIPS_PACRP_WP0_MASK                      0x20000000u
#define AIPS_PACRP_WP0_SHIFT                     29
#define AIPS_PACRP_SP0_MASK                      0x40000000u
#define AIPS_PACRP_SP0_SHIFT                     30
/* PACRU Bit Fields */
#define AIPS_PACRU_TP1_MASK                      0x1000000u
#define AIPS_PACRU_TP1_SHIFT                     24
#define AIPS_PACRU_WP1_MASK                      0x2000000u
#define AIPS_PACRU_WP1_SHIFT                     25
#define AIPS_PACRU_SP1_MASK                      0x4000000u
#define AIPS_PACRU_SP1_SHIFT                     26
#define AIPS_PACRU_TP0_MASK                      0x10000000u
#define AIPS_PACRU_TP0_SHIFT                     28
#define AIPS_PACRU_WP0_MASK                      0x20000000u
#define AIPS_PACRU_WP0_SHIFT                     29
#define AIPS_PACRU_SP0_MASK                      0x40000000u
#define AIPS_PACRU_SP0_SHIFT                     30

/**
 * @}
 */ /* end of group AIPS_Register_Masks */


/* AIPS - Peripheral instance base addresses */
/** Peripheral AIPS base pointer */
#define AIPS_BASE_PTR                            ((AIPS_MemMapPtr)0x40000000u)

/* ----------------------------------------------------------------------------
   -- AIPS - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup AIPS_Register_Accessor_Macros AIPS - Register accessor macros
 * @{
 */


/* AIPS - Register instance definitions */
/* AIPS */
#define AIPS_MPRA                                AIPS_MPRA_REG(AIPS_BASE_PTR)
#define AIPS_PACRA                               AIPS_PACRA_REG(AIPS_BASE_PTR)
#define AIPS_PACRB                               AIPS_PACRB_REG(AIPS_BASE_PTR)
#define AIPS_PACRC                               AIPS_PACRC_REG(AIPS_BASE_PTR)
#define AIPS_PACRD                               AIPS_PACRD_REG(AIPS_BASE_PTR)
#define AIPS_PACRE                               AIPS_PACRE_REG(AIPS_BASE_PTR)
#define AIPS_PACRF                               AIPS_PACRF_REG(AIPS_BASE_PTR)
#define AIPS_PACRG                               AIPS_PACRG_REG(AIPS_BASE_PTR)
#define AIPS_PACRH                               AIPS_PACRH_REG(AIPS_BASE_PTR)
#define AIPS_PACRI                               AIPS_PACRI_REG(AIPS_BASE_PTR)
#define AIPS_PACRJ                               AIPS_PACRJ_REG(AIPS_BASE_PTR)
#define AIPS_PACRK                               AIPS_PACRK_REG(AIPS_BASE_PTR)
#define AIPS_PACRL                               AIPS_PACRL_REG(AIPS_BASE_PTR)
#define AIPS_PACRM                               AIPS_PACRM_REG(AIPS_BASE_PTR)
#define AIPS_PACRN                               AIPS_PACRN_REG(AIPS_BASE_PTR)
#define AIPS_PACRO                               AIPS_PACRO_REG(AIPS_BASE_PTR)
#define AIPS_PACRP                               AIPS_PACRP_REG(AIPS_BASE_PTR)
#define AIPS_PACRU                               AIPS_PACRU_REG(AIPS_BASE_PTR)

/**
 * @}
 */ /* end of group AIPS_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group AIPS_Peripheral */


/* ----------------------------------------------------------------------------
   -- CMP
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup CMP_Peripheral CMP
 * @{
 */

/** CMP - Peripheral register structure */
typedef struct CMP_MemMap {
  uint8_t CR0;                                     /**< CMP Control Register 0, offset: 0x0 */
  uint8_t CR1;                                     /**< CMP Control Register 1, offset: 0x1 */
  uint8_t FPR;                                     /**< CMP Filter Period Register, offset: 0x2 */
  uint8_t SCR;                                     /**< CMP Status and Control Register, offset: 0x3 */
  uint8_t DACCR;                                   /**< DAC Control Register, offset: 0x4 */
  uint8_t MUXCR;                                   /**< MUX Control Register, offset: 0x5 */
  uint8_t MUXPE;                                   /**< MUX Pin Enable Register, offset: 0x6 */
} volatile *CMP_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- CMP - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup CMP_Register_Accessor_Macros CMP - Register accessor macros
 * @{
 */


/* CMP - Register accessors */
#define CMP_CR0_REG(base)                        ((base)->CR0)
#define CMP_CR1_REG(base)                        ((base)->CR1)
#define CMP_FPR_REG(base)                        ((base)->FPR)
#define CMP_SCR_REG(base)                        ((base)->SCR)
#define CMP_DACCR_REG(base)                      ((base)->DACCR)
#define CMP_MUXCR_REG(base)                      ((base)->MUXCR)
#define CMP_MUXPE_REG(base)                      ((base)->MUXPE)

/**
 * @}
 */ /* end of group CMP_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- CMP Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup CMP_Register_Masks CMP Register Masks
 * @{
 */

/* CR0 Bit Fields */
#define CMP_CR0_HYSTCTR_MASK                     0x1u
#define CMP_CR0_HYSTCTR_SHIFT                    0
#define CMP_CR0_FILTER_CNT_MASK                  0x70u
#define CMP_CR0_FILTER_CNT_SHIFT                 4
#define CMP_CR0_FILTER_CNT(x)                    (((uint8_t)(((uint8_t)(x))<<CMP_CR0_FILTER_CNT_SHIFT))&CMP_CR0_FILTER_CNT_MASK)
/* CR1 Bit Fields */
#define CMP_CR1_EN_MASK                          0x1u
#define CMP_CR1_EN_SHIFT                         0
#define CMP_CR1_OPE_MASK                         0x2u
#define CMP_CR1_OPE_SHIFT                        1
#define CMP_CR1_COS_MASK                         0x4u
#define CMP_CR1_COS_SHIFT                        2
#define CMP_CR1_INV_MASK                         0x8u
#define CMP_CR1_INV_SHIFT                        3
#define CMP_CR1_PMODE_MASK                       0x10u
#define CMP_CR1_PMODE_SHIFT                      4
#define CMP_CR1_WE_MASK                          0x40u
#define CMP_CR1_WE_SHIFT                         6
#define CMP_CR1_SE_MASK                          0x80u
#define CMP_CR1_SE_SHIFT                         7
/* FPR Bit Fields */
#define CMP_FPR_FILT_PER_MASK                    0xFFu
#define CMP_FPR_FILT_PER_SHIFT                   0
#define CMP_FPR_FILT_PER(x)                      (((uint8_t)(((uint8_t)(x))<<CMP_FPR_FILT_PER_SHIFT))&CMP_FPR_FILT_PER_MASK)
/* SCR Bit Fields */
#define CMP_SCR_COUT_MASK                        0x1u
#define CMP_SCR_COUT_SHIFT                       0
#define CMP_SCR_CFF_MASK                         0x2u
#define CMP_SCR_CFF_SHIFT                        1
#define CMP_SCR_CFR_MASK                         0x4u
#define CMP_SCR_CFR_SHIFT                        2
#define CMP_SCR_IEF_MASK                         0x8u
#define CMP_SCR_IEF_SHIFT                        3
#define CMP_SCR_IER_MASK                         0x10u
#define CMP_SCR_IER_SHIFT                        4
#define CMP_SCR_DMAEN_MASK                       0x40u
#define CMP_SCR_DMAEN_SHIFT                      6
/* DACCR Bit Fields */
#define CMP_DACCR_VOSEL_MASK                     0x3Fu
#define CMP_DACCR_VOSEL_SHIFT                    0
#define CMP_DACCR_VOSEL(x)                       (((uint8_t)(((uint8_t)(x))<<CMP_DACCR_VOSEL_SHIFT))&CMP_DACCR_VOSEL_MASK)
#define CMP_DACCR_VRSEL_MASK                     0x40u
#define CMP_DACCR_VRSEL_SHIFT                    6
#define CMP_DACCR_DACEN_MASK                     0x80u
#define CMP_DACCR_DACEN_SHIFT                    7
/* MUXCR Bit Fields */
#define CMP_MUXCR_MSEL_MASK                      0x3u
#define CMP_MUXCR_MSEL_SHIFT                     0
#define CMP_MUXCR_MSEL(x)                        (((uint8_t)(((uint8_t)(x))<<CMP_MUXCR_MSEL_SHIFT))&CMP_MUXCR_MSEL_MASK)
#define CMP_MUXCR_PSEL_MASK                      0x30u
#define CMP_MUXCR_PSEL_SHIFT                     4
#define CMP_MUXCR_PSEL(x)                        (((uint8_t)(((uint8_t)(x))<<CMP_MUXCR_PSEL_SHIFT))&CMP_MUXCR_PSEL_MASK)
/* MUXPE Bit Fields */
#define CMP_MUXPE_INPE_MASK                      0x7u
#define CMP_MUXPE_INPE_SHIFT                     0
#define CMP_MUXPE_INPE(x)                        (((uint8_t)(((uint8_t)(x))<<CMP_MUXPE_INPE_SHIFT))&CMP_MUXPE_INPE_MASK)

/**
 * @}
 */ /* end of group CMP_Register_Masks */


/* CMP - Peripheral instance base addresses */
/** Peripheral CMP0 base pointer */
#define CMP0_BASE_PTR                            ((CMP_MemMapPtr)0x40050000u)
/** Peripheral CMP1 base pointer */
#define CMP1_BASE_PTR                            ((CMP_MemMapPtr)0x40051000u)
/** Peripheral CMP2 base pointer */
#define CMP2_BASE_PTR                            ((CMP_MemMapPtr)0x40052000u)
/** Peripheral CMP3 base pointer */
#define CMP3_BASE_PTR                            ((CMP_MemMapPtr)0x40053000u)

/* ----------------------------------------------------------------------------
   -- CMP - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup CMP_Register_Accessor_Macros CMP - Register accessor macros
 * @{
 */


/* CMP - Register instance definitions */
/* CMP0 */
#define CMP0_CR0                                 CMP_CR0_REG(CMP0_BASE_PTR)
#define CMP0_CR1                                 CMP_CR1_REG(CMP0_BASE_PTR)
#define CMP0_FPR                                 CMP_FPR_REG(CMP0_BASE_PTR)
#define CMP0_SCR                                 CMP_SCR_REG(CMP0_BASE_PTR)
#define CMP0_DACCR                               CMP_DACCR_REG(CMP0_BASE_PTR)
#define CMP0_MUXCR                               CMP_MUXCR_REG(CMP0_BASE_PTR)
#define CMP0_MUXPE                               CMP_MUXPE_REG(CMP0_BASE_PTR)
/* CMP1 */
#define CMP1_CR0                                 CMP_CR0_REG(CMP1_BASE_PTR)
#define CMP1_CR1                                 CMP_CR1_REG(CMP1_BASE_PTR)
#define CMP1_FPR                                 CMP_FPR_REG(CMP1_BASE_PTR)
#define CMP1_SCR                                 CMP_SCR_REG(CMP1_BASE_PTR)
#define CMP1_DACCR                               CMP_DACCR_REG(CMP1_BASE_PTR)
#define CMP1_MUXCR                               CMP_MUXCR_REG(CMP1_BASE_PTR)
#define CMP1_MUXPE                               CMP_MUXPE_REG(CMP1_BASE_PTR)
/* CMP2 */
#define CMP2_CR0                                 CMP_CR0_REG(CMP2_BASE_PTR)
#define CMP2_CR1                                 CMP_CR1_REG(CMP2_BASE_PTR)
#define CMP2_FPR                                 CMP_FPR_REG(CMP2_BASE_PTR)
#define CMP2_SCR                                 CMP_SCR_REG(CMP2_BASE_PTR)
#define CMP2_DACCR                               CMP_DACCR_REG(CMP2_BASE_PTR)
#define CMP2_MUXCR                               CMP_MUXCR_REG(CMP2_BASE_PTR)
#define CMP2_MUXPE                               CMP_MUXPE_REG(CMP2_BASE_PTR)
/* CMP3 */
#define CMP3_CR0                                 CMP_CR0_REG(CMP3_BASE_PTR)
#define CMP3_CR1                                 CMP_CR1_REG(CMP3_BASE_PTR)
#define CMP3_FPR                                 CMP_FPR_REG(CMP3_BASE_PTR)
#define CMP3_SCR                                 CMP_SCR_REG(CMP3_BASE_PTR)
#define CMP3_DACCR                               CMP_DACCR_REG(CMP3_BASE_PTR)
#define CMP3_MUXCR                               CMP_MUXCR_REG(CMP3_BASE_PTR)
#define CMP3_MUXPE                               CMP_MUXPE_REG(CMP3_BASE_PTR)

/**
 * @}
 */ /* end of group CMP_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group CMP_Peripheral */


/* ----------------------------------------------------------------------------
   -- CMT
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup CMT_Peripheral CMT
 * @{
 */

/** CMT - Peripheral register structure */
typedef struct CMT_MemMap {
  uint8_t CGH1;                                    /**< CMT Carrier Generator High Data Register 1, offset: 0x0 */
  uint8_t CGL1;                                    /**< CMT Carrier Generator Low Data Register 1, offset: 0x1 */
  uint8_t CGH2;                                    /**< CMT Carrier Generator High Data Register 2, offset: 0x2 */
  uint8_t CGL2;                                    /**< CMT Carrier Generator Low Data Register 2, offset: 0x3 */
  uint8_t OC;                                      /**< CMT Output Control Register, offset: 0x4 */
  uint8_t MSC;                                     /**< CMT Modulator Status and Control Register, offset: 0x5 */
  uint8_t CMD1;                                    /**< CMT Modulator Data Register Mark High, offset: 0x6 */
  uint8_t CMD2;                                    /**< CMT Modulator Data Register Mark Low, offset: 0x7 */
  uint8_t CMD3;                                    /**< CMT Modulator Data Register Space High, offset: 0x8 */
  uint8_t CMD4;                                    /**< CMT Modulator Data Register Space Low, offset: 0x9 */
  uint8_t PPS;                                     /**< CMT Primary Prescaler Register, offset: 0xA */
  uint8_t DMA;                                     /**< CMT Direct Memory Access Register, offset: 0xB */
} volatile *CMT_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- CMT - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup CMT_Register_Accessor_Macros CMT - Register accessor macros
 * @{
 */


/* CMT - Register accessors */
#define CMT_CGH1_REG(base)                       ((base)->CGH1)
#define CMT_CGL1_REG(base)                       ((base)->CGL1)
#define CMT_CGH2_REG(base)                       ((base)->CGH2)
#define CMT_CGL2_REG(base)                       ((base)->CGL2)
#define CMT_OC_REG(base)                         ((base)->OC)
#define CMT_MSC_REG(base)                        ((base)->MSC)
#define CMT_CMD1_REG(base)                       ((base)->CMD1)
#define CMT_CMD2_REG(base)                       ((base)->CMD2)
#define CMT_CMD3_REG(base)                       ((base)->CMD3)
#define CMT_CMD4_REG(base)                       ((base)->CMD4)
#define CMT_PPS_REG(base)                        ((base)->PPS)
#define CMT_DMA_REG(base)                        ((base)->DMA)

/**
 * @}
 */ /* end of group CMT_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- CMT Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup CMT_Register_Masks CMT Register Masks
 * @{
 */

/* CGH1 Bit Fields */
#define CMT_CGH1_PH_MASK                         0xFFu
#define CMT_CGH1_PH_SHIFT                        0
#define CMT_CGH1_PH(x)                           (((uint8_t)(((uint8_t)(x))<<CMT_CGH1_PH_SHIFT))&CMT_CGH1_PH_MASK)
/* CGL1 Bit Fields */
#define CMT_CGL1_PL_MASK                         0xFFu
#define CMT_CGL1_PL_SHIFT                        0
#define CMT_CGL1_PL(x)                           (((uint8_t)(((uint8_t)(x))<<CMT_CGL1_PL_SHIFT))&CMT_CGL1_PL_MASK)
/* CGH2 Bit Fields */
#define CMT_CGH2_SH_MASK                         0xFFu
#define CMT_CGH2_SH_SHIFT                        0
#define CMT_CGH2_SH(x)                           (((uint8_t)(((uint8_t)(x))<<CMT_CGH2_SH_SHIFT))&CMT_CGH2_SH_MASK)
/* CGL2 Bit Fields */
#define CMT_CGL2_SL_MASK                         0xFFu
#define CMT_CGL2_SL_SHIFT                        0
#define CMT_CGL2_SL(x)                           (((uint8_t)(((uint8_t)(x))<<CMT_CGL2_SL_SHIFT))&CMT_CGL2_SL_MASK)
/* OC Bit Fields */
#define CMT_OC_IROPEN_MASK                       0x20u
#define CMT_OC_IROPEN_SHIFT                      5
#define CMT_OC_CMTPOL_MASK                       0x40u
#define CMT_OC_CMTPOL_SHIFT                      6
#define CMT_OC_IROL_MASK                         0x80u
#define CMT_OC_IROL_SHIFT                        7
/* MSC Bit Fields */
#define CMT_MSC_MCGEN_MASK                       0x1u
#define CMT_MSC_MCGEN_SHIFT                      0
#define CMT_MSC_EOCIE_MASK                       0x2u
#define CMT_MSC_EOCIE_SHIFT                      1
#define CMT_MSC_FSK_MASK                         0x4u
#define CMT_MSC_FSK_SHIFT                        2
#define CMT_MSC_BASE_MASK                        0x8u
#define CMT_MSC_BASE_SHIFT                       3
#define CMT_MSC_EXSPC_MASK                       0x10u
#define CMT_MSC_EXSPC_SHIFT                      4
#define CMT_MSC_CMTDIV_MASK                      0x60u
#define CMT_MSC_CMTDIV_SHIFT                     5
#define CMT_MSC_CMTDIV(x)                        (((uint8_t)(((uint8_t)(x))<<CMT_MSC_CMTDIV_SHIFT))&CMT_MSC_CMTDIV_MASK)
#define CMT_MSC_EOCF_MASK                        0x80u
#define CMT_MSC_EOCF_SHIFT                       7
/* CMD1 Bit Fields */
#define CMT_CMD1_MB_MASK                         0xFFu
#define CMT_CMD1_MB_SHIFT                        0
#define CMT_CMD1_MB(x)                           (((uint8_t)(((uint8_t)(x))<<CMT_CMD1_MB_SHIFT))&CMT_CMD1_MB_MASK)
/* CMD2 Bit Fields */
#define CMT_CMD2_MB_MASK                         0xFFu
#define CMT_CMD2_MB_SHIFT                        0
#define CMT_CMD2_MB(x)                           (((uint8_t)(((uint8_t)(x))<<CMT_CMD2_MB_SHIFT))&CMT_CMD2_MB_MASK)
/* CMD3 Bit Fields */
#define CMT_CMD3_SB_MASK                         0xFFu
#define CMT_CMD3_SB_SHIFT                        0
#define CMT_CMD3_SB(x)                           (((uint8_t)(((uint8_t)(x))<<CMT_CMD3_SB_SHIFT))&CMT_CMD3_SB_MASK)
/* CMD4 Bit Fields */
#define CMT_CMD4_SB_MASK                         0xFFu
#define CMT_CMD4_SB_SHIFT                        0
#define CMT_CMD4_SB(x)                           (((uint8_t)(((uint8_t)(x))<<CMT_CMD4_SB_SHIFT))&CMT_CMD4_SB_MASK)
/* PPS Bit Fields */
#define CMT_PPS_PPSDIV_MASK                      0xFu
#define CMT_PPS_PPSDIV_SHIFT                     0
#define CMT_PPS_PPSDIV(x)                        (((uint8_t)(((uint8_t)(x))<<CMT_PPS_PPSDIV_SHIFT))&CMT_PPS_PPSDIV_MASK)
/* DMA Bit Fields */
#define CMT_DMA_DMA_MASK                         0x1u
#define CMT_DMA_DMA_SHIFT                        0

/**
 * @}
 */ /* end of group CMT_Register_Masks */


/* CMT - Peripheral instance base addresses */
/** Peripheral CMT base pointer */
#define CMT_BASE_PTR                             ((CMT_MemMapPtr)0x40042000u)

/* ----------------------------------------------------------------------------
   -- CMT - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup CMT_Register_Accessor_Macros CMT - Register accessor macros
 * @{
 */


/* CMT - Register instance definitions */
/* CMT */
#define CMT_CGH1                                 CMT_CGH1_REG(CMT_BASE_PTR)
#define CMT_CGL1                                 CMT_CGL1_REG(CMT_BASE_PTR)
#define CMT_CGH2                                 CMT_CGH2_REG(CMT_BASE_PTR)
#define CMT_CGL2                                 CMT_CGL2_REG(CMT_BASE_PTR)
#define CMT_OC                                   CMT_OC_REG(CMT_BASE_PTR)
#define CMT_MSC                                  CMT_MSC_REG(CMT_BASE_PTR)
#define CMT_CMD1                                 CMT_CMD1_REG(CMT_BASE_PTR)
#define CMT_CMD2                                 CMT_CMD2_REG(CMT_BASE_PTR)
#define CMT_CMD3                                 CMT_CMD3_REG(CMT_BASE_PTR)
#define CMT_CMD4                                 CMT_CMD4_REG(CMT_BASE_PTR)
#define CMT_PPS                                  CMT_PPS_REG(CMT_BASE_PTR)
#define CMT_DMA                                  CMT_DMA_REG(CMT_BASE_PTR)

/**
 * @}
 */ /* end of group CMT_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group CMT_Peripheral */


/* ----------------------------------------------------------------------------
   -- CRC
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup CRC_Peripheral CRC
 * @{
 */

/** CRC - Peripheral register structure */
typedef struct CRC_MemMap {
  union {                                          /* offset: 0x0 */
    struct {                                         /* offset: 0x0 */
      uint16_t CRCL;                                   /**< CRC_CRCL register., offset: 0x0 */
      uint16_t CRCH;                                   /**< CRC_CRCH register., offset: 0x2 */
    } ACCESS16BIT;
    uint32_t CRC;                                    /**< CRC Data register, offset: 0x0 */
    struct {                                         /* offset: 0x0 */
      uint8_t CRCLL;                                   /**< CRC_CRCLL register., offset: 0x0 */
      uint8_t CRCLU;                                   /**< CRC_CRCLU register., offset: 0x1 */
      uint8_t CRCHL;                                   /**< CRC_CRCHL register., offset: 0x2 */
      uint8_t CRCHU;                                   /**< CRC_CRCHU register., offset: 0x3 */
    } ACCESS8BIT;
  };
  union {                                          /* offset: 0x4 */
    struct {                                         /* offset: 0x4 */
      uint16_t GPOLYL;                                 /**< CRC_GPOLYL register., offset: 0x4 */
      uint16_t GPOLYH;                                 /**< CRC_GPOLYH register., offset: 0x6 */
    } GPOLY_ACCESS16BIT;
    uint32_t GPOLY;                                  /**< CRC Polynomial register, offset: 0x4 */
    struct {                                         /* offset: 0x4 */
      uint8_t GPOLYLL;                                 /**< CRC_GPOLYLL register., offset: 0x4 */
      uint8_t GPOLYLU;                                 /**< CRC_GPOLYLU register., offset: 0x5 */
      uint8_t GPOLYHL;                                 /**< CRC_GPOLYHL register., offset: 0x6 */
      uint8_t GPOLYHU;                                 /**< CRC_GPOLYHU register., offset: 0x7 */
    } GPOLY_ACCESS8BIT;
  };
  union {                                          /* offset: 0x8 */
    uint32_t CTRL;                                   /**< CRC Control register, offset: 0x8 */
    struct {                                         /* offset: 0x8 */
      uint8_t RESERVED_0[3];
      uint8_t CTRLHU;                                  /**< CRC_CTRLHU register., offset: 0xB */
    } CTRL_ACCESS8BIT;
  };
} volatile *CRC_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- CRC - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup CRC_Register_Accessor_Macros CRC - Register accessor macros
 * @{
 */


/* CRC - Register accessors */
#define CRC_CRCL_REG(base)                       ((base)->ACCESS16BIT.CRCL)
#define CRC_CRCH_REG(base)                       ((base)->ACCESS16BIT.CRCH)
#define CRC_CRC_REG(base)                        ((base)->CRC)
#define CRC_CRCLL_REG(base)                      ((base)->ACCESS8BIT.CRCLL)
#define CRC_CRCLU_REG(base)                      ((base)->ACCESS8BIT.CRCLU)
#define CRC_CRCHL_REG(base)                      ((base)->ACCESS8BIT.CRCHL)
#define CRC_CRCHU_REG(base)                      ((base)->ACCESS8BIT.CRCHU)
#define CRC_GPOLYL_REG(base)                     ((base)->GPOLY_ACCESS16BIT.GPOLYL)
#define CRC_GPOLYH_REG(base)                     ((base)->GPOLY_ACCESS16BIT.GPOLYH)
#define CRC_GPOLY_REG(base)                      ((base)->GPOLY)
#define CRC_GPOLYLL_REG(base)                    ((base)->GPOLY_ACCESS8BIT.GPOLYLL)
#define CRC_GPOLYLU_REG(base)                    ((base)->GPOLY_ACCESS8BIT.GPOLYLU)
#define CRC_GPOLYHL_REG(base)                    ((base)->GPOLY_ACCESS8BIT.GPOLYHL)
#define CRC_GPOLYHU_REG(base)                    ((base)->GPOLY_ACCESS8BIT.GPOLYHU)
#define CRC_CTRL_REG(base)                       ((base)->CTRL)
#define CRC_CTRLHU_REG(base)                     ((base)->CTRL_ACCESS8BIT.CTRLHU)

/**
 * @}
 */ /* end of group CRC_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- CRC Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup CRC_Register_Masks CRC Register Masks
 * @{
 */

/* CRCL Bit Fields */
#define CRC_CRCL_CRCL_MASK                       0xFFFFu
#define CRC_CRCL_CRCL_SHIFT                      0
#define CRC_CRCL_CRCL(x)                         (((uint16_t)(((uint16_t)(x))<<CRC_CRCL_CRCL_SHIFT))&CRC_CRCL_CRCL_MASK)
/* CRCH Bit Fields */
#define CRC_CRCH_CRCH_MASK                       0xFFFFu
#define CRC_CRCH_CRCH_SHIFT                      0
#define CRC_CRCH_CRCH(x)                         (((uint16_t)(((uint16_t)(x))<<CRC_CRCH_CRCH_SHIFT))&CRC_CRCH_CRCH_MASK)
/* CRC Bit Fields */
#define CRC_CRC_LL_MASK                          0xFFu
#define CRC_CRC_LL_SHIFT                         0
#define CRC_CRC_LL(x)                            (((uint32_t)(((uint32_t)(x))<<CRC_CRC_LL_SHIFT))&CRC_CRC_LL_MASK)
#define CRC_CRC_LU_MASK                          0xFF00u
#define CRC_CRC_LU_SHIFT                         8
#define CRC_CRC_LU(x)                            (((uint32_t)(((uint32_t)(x))<<CRC_CRC_LU_SHIFT))&CRC_CRC_LU_MASK)
#define CRC_CRC_HL_MASK                          0xFF0000u
#define CRC_CRC_HL_SHIFT                         16
#define CRC_CRC_HL(x)                            (((uint32_t)(((uint32_t)(x))<<CRC_CRC_HL_SHIFT))&CRC_CRC_HL_MASK)
#define CRC_CRC_HU_MASK                          0xFF000000u
#define CRC_CRC_HU_SHIFT                         24
#define CRC_CRC_HU(x)                            (((uint32_t)(((uint32_t)(x))<<CRC_CRC_HU_SHIFT))&CRC_CRC_HU_MASK)
/* CRCLL Bit Fields */
#define CRC_CRCLL_CRCLL_MASK                     0xFFu
#define CRC_CRCLL_CRCLL_SHIFT                    0
#define CRC_CRCLL_CRCLL(x)                       (((uint8_t)(((uint8_t)(x))<<CRC_CRCLL_CRCLL_SHIFT))&CRC_CRCLL_CRCLL_MASK)
/* CRCLU Bit Fields */
#define CRC_CRCLU_CRCLU_MASK                     0xFFu
#define CRC_CRCLU_CRCLU_SHIFT                    0
#define CRC_CRCLU_CRCLU(x)                       (((uint8_t)(((uint8_t)(x))<<CRC_CRCLU_CRCLU_SHIFT))&CRC_CRCLU_CRCLU_MASK)
/* CRCHL Bit Fields */
#define CRC_CRCHL_CRCHL_MASK                     0xFFu
#define CRC_CRCHL_CRCHL_SHIFT                    0
#define CRC_CRCHL_CRCHL(x)                       (((uint8_t)(((uint8_t)(x))<<CRC_CRCHL_CRCHL_SHIFT))&CRC_CRCHL_CRCHL_MASK)
/* CRCHU Bit Fields */
#define CRC_CRCHU_CRCHU_MASK                     0xFFu
#define CRC_CRCHU_CRCHU_SHIFT                    0
#define CRC_CRCHU_CRCHU(x)                       (((uint8_t)(((uint8_t)(x))<<CRC_CRCHU_CRCHU_SHIFT))&CRC_CRCHU_CRCHU_MASK)
/* GPOLYL Bit Fields */
#define CRC_GPOLYL_GPOLYL_MASK                   0xFFFFu
#define CRC_GPOLYL_GPOLYL_SHIFT                  0
#define CRC_GPOLYL_GPOLYL(x)                     (((uint16_t)(((uint16_t)(x))<<CRC_GPOLYL_GPOLYL_SHIFT))&CRC_GPOLYL_GPOLYL_MASK)
/* GPOLYH Bit Fields */
#define CRC_GPOLYH_GPOLYH_MASK                   0xFFFFu
#define CRC_GPOLYH_GPOLYH_SHIFT                  0
#define CRC_GPOLYH_GPOLYH(x)                     (((uint16_t)(((uint16_t)(x))<<CRC_GPOLYH_GPOLYH_SHIFT))&CRC_GPOLYH_GPOLYH_MASK)
/* GPOLY Bit Fields */
#define CRC_GPOLY_LOW_MASK                       0xFFFFu
#define CRC_GPOLY_LOW_SHIFT                      0
#define CRC_GPOLY_LOW(x)                         (((uint32_t)(((uint32_t)(x))<<CRC_GPOLY_LOW_SHIFT))&CRC_GPOLY_LOW_MASK)
#define CRC_GPOLY_HIGH_MASK                      0xFFFF0000u
#define CRC_GPOLY_HIGH_SHIFT                     16
#define CRC_GPOLY_HIGH(x)                        (((uint32_t)(((uint32_t)(x))<<CRC_GPOLY_HIGH_SHIFT))&CRC_GPOLY_HIGH_MASK)
/* GPOLYLL Bit Fields */
#define CRC_GPOLYLL_GPOLYLL_MASK                 0xFFu
#define CRC_GPOLYLL_GPOLYLL_SHIFT                0
#define CRC_GPOLYLL_GPOLYLL(x)                   (((uint8_t)(((uint8_t)(x))<<CRC_GPOLYLL_GPOLYLL_SHIFT))&CRC_GPOLYLL_GPOLYLL_MASK)
/* GPOLYLU Bit Fields */
#define CRC_GPOLYLU_GPOLYLU_MASK                 0xFFu
#define CRC_GPOLYLU_GPOLYLU_SHIFT                0
#define CRC_GPOLYLU_GPOLYLU(x)                   (((uint8_t)(((uint8_t)(x))<<CRC_GPOLYLU_GPOLYLU_SHIFT))&CRC_GPOLYLU_GPOLYLU_MASK)
/* GPOLYHL Bit Fields */
#define CRC_GPOLYHL_GPOLYHL_MASK                 0xFFu
#define CRC_GPOLYHL_GPOLYHL_SHIFT                0
#define CRC_GPOLYHL_GPOLYHL(x)                   (((uint8_t)(((uint8_t)(x))<<CRC_GPOLYHL_GPOLYHL_SHIFT))&CRC_GPOLYHL_GPOLYHL_MASK)
/* GPOLYHU Bit Fields */
#define CRC_GPOLYHU_GPOLYHU_MASK                 0xFFu
#define CRC_GPOLYHU_GPOLYHU_SHIFT                0
#define CRC_GPOLYHU_GPOLYHU(x)                   (((uint8_t)(((uint8_t)(x))<<CRC_GPOLYHU_GPOLYHU_SHIFT))&CRC_GPOLYHU_GPOLYHU_MASK)
/* CTRL Bit Fields */
#define CRC_CTRL_TCRC_MASK                       0x1000000u
#define CRC_CTRL_TCRC_SHIFT                      24
#define CRC_CTRL_WAS_MASK                        0x2000000u
#define CRC_CTRL_WAS_SHIFT                       25
#define CRC_CTRL_FXOR_MASK                       0x4000000u
#define CRC_CTRL_FXOR_SHIFT                      26
#define CRC_CTRL_TOTR_MASK                       0x30000000u
#define CRC_CTRL_TOTR_SHIFT                      28
#define CRC_CTRL_TOTR(x)                         (((uint32_t)(((uint32_t)(x))<<CRC_CTRL_TOTR_SHIFT))&CRC_CTRL_TOTR_MASK)
#define CRC_CTRL_TOT_MASK                        0xC0000000u
#define CRC_CTRL_TOT_SHIFT                       30
#define CRC_CTRL_TOT(x)                          (((uint32_t)(((uint32_t)(x))<<CRC_CTRL_TOT_SHIFT))&CRC_CTRL_TOT_MASK)
/* CTRLHU Bit Fields */
#define CRC_CTRLHU_TCRC_MASK                     0x1u
#define CRC_CTRLHU_TCRC_SHIFT                    0
#define CRC_CTRLHU_WAS_MASK                      0x2u
#define CRC_CTRLHU_WAS_SHIFT                     1
#define CRC_CTRLHU_FXOR_MASK                     0x4u
#define CRC_CTRLHU_FXOR_SHIFT                    2
#define CRC_CTRLHU_TOTR_MASK                     0x30u
#define CRC_CTRLHU_TOTR_SHIFT                    4
#define CRC_CTRLHU_TOTR(x)                       (((uint8_t)(((uint8_t)(x))<<CRC_CTRLHU_TOTR_SHIFT))&CRC_CTRLHU_TOTR_MASK)
#define CRC_CTRLHU_TOT_MASK                      0xC0u
#define CRC_CTRLHU_TOT_SHIFT                     6
#define CRC_CTRLHU_TOT(x)                        (((uint8_t)(((uint8_t)(x))<<CRC_CTRLHU_TOT_SHIFT))&CRC_CTRLHU_TOT_MASK)

/**
 * @}
 */ /* end of group CRC_Register_Masks */


/* CRC - Peripheral instance base addresses */
/** Peripheral CRC base pointer */
#define CRC_BASE_PTR                             ((CRC_MemMapPtr)0x40024000u)

/* ----------------------------------------------------------------------------
   -- CRC - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup CRC_Register_Accessor_Macros CRC - Register accessor macros
 * @{
 */


/* CRC - Register instance definitions */
/* CRC */
#define CRC_CRC                                  CRC_CRC_REG(CRC_BASE_PTR)
#define CRC_CRCL                                 CRC_CRCL_REG(CRC_BASE_PTR)
#define CRC_CRCLL                                CRC_CRCLL_REG(CRC_BASE_PTR)
#define CRC_CRCLU                                CRC_CRCLU_REG(CRC_BASE_PTR)
#define CRC_CRCH                                 CRC_CRCH_REG(CRC_BASE_PTR)
#define CRC_CRCHL                                CRC_CRCHL_REG(CRC_BASE_PTR)
#define CRC_CRCHU                                CRC_CRCHU_REG(CRC_BASE_PTR)
#define CRC_GPOLY                                CRC_GPOLY_REG(CRC_BASE_PTR)
#define CRC_GPOLYL                               CRC_GPOLYL_REG(CRC_BASE_PTR)
#define CRC_GPOLYLL                              CRC_GPOLYLL_REG(CRC_BASE_PTR)
#define CRC_GPOLYLU                              CRC_GPOLYLU_REG(CRC_BASE_PTR)
#define CRC_GPOLYH                               CRC_GPOLYH_REG(CRC_BASE_PTR)
#define CRC_GPOLYHL                              CRC_GPOLYHL_REG(CRC_BASE_PTR)
#define CRC_GPOLYHU                              CRC_GPOLYHU_REG(CRC_BASE_PTR)
#define CRC_CTRL                                 CRC_CTRL_REG(CRC_BASE_PTR)
#define CRC_CTRLHU                               CRC_CTRLHU_REG(CRC_BASE_PTR)

/**
 * @}
 */ /* end of group CRC_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group CRC_Peripheral */


/* ----------------------------------------------------------------------------
   -- CoreDebug
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup CoreDebug_Peripheral CoreDebug
 * @{
 */

/** CoreDebug - Peripheral register structure */
typedef struct CoreDebug_MemMap {
  union {                                          /* offset: 0x0 */
    uint32_t base_DHCSR_Read;                        /**< Debug Halting Control and Status Register, offset: 0x0 */
    uint32_t base_DHCSR_Write;                       /**< Debug Halting Control and Status Register, offset: 0x0 */
  };
  uint32_t base_DCRSR;                             /**< Debug Core Register Selector Register, offset: 0x4 */
  uint32_t base_DCRDR;                             /**< Debug Core Register Data Register, offset: 0x8 */
  uint32_t base_DEMCR;                             /**< Debug Exception and Monitor Control Register, offset: 0xC */
} volatile *CoreDebug_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- CoreDebug - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup CoreDebug_Register_Accessor_Macros CoreDebug - Register accessor macros
 * @{
 */


/* CoreDebug - Register accessors */
#define CoreDebug_base_DHCSR_Read_REG(base)      ((base)->base_DHCSR_Read)
#define CoreDebug_base_DHCSR_Write_REG(base)     ((base)->base_DHCSR_Write)
#define CoreDebug_base_DCRSR_REG(base)           ((base)->base_DCRSR)
#define CoreDebug_base_DCRDR_REG(base)           ((base)->base_DCRDR)
#define CoreDebug_base_DEMCR_REG(base)           ((base)->base_DEMCR)

/**
 * @}
 */ /* end of group CoreDebug_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- CoreDebug Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup CoreDebug_Register_Masks CoreDebug Register Masks
 * @{
 */


/**
 * @}
 */ /* end of group CoreDebug_Register_Masks */


/* CoreDebug - Peripheral instance base addresses */
/** Peripheral CoreDebug base pointer */
#define CoreDebug_BASE_PTR                       ((CoreDebug_MemMapPtr)0xE000EDF0u)

/* ----------------------------------------------------------------------------
   -- CoreDebug - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup CoreDebug_Register_Accessor_Macros CoreDebug - Register accessor macros
 * @{
 */


/* CoreDebug - Register instance definitions */
/* CoreDebug */
#define DHCSR_Read                               CoreDebug_base_DHCSR_Read_REG(CoreDebug_BASE_PTR)
#define DHCSR_Write                              CoreDebug_base_DHCSR_Write_REG(CoreDebug_BASE_PTR)
#define DCRSR                                    CoreDebug_base_DCRSR_REG(CoreDebug_BASE_PTR)
#define DCRDR                                    CoreDebug_base_DCRDR_REG(CoreDebug_BASE_PTR)
#define DEMCR                                    CoreDebug_base_DEMCR_REG(CoreDebug_BASE_PTR)

/**
 * @}
 */ /* end of group CoreDebug_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group CoreDebug_Peripheral */


/* ----------------------------------------------------------------------------
   -- DMA
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup DMA_Peripheral DMA
 * @{
 */

/** DMA - Peripheral register structure */
typedef struct DMA_MemMap {
  uint32_t CR;                                     /**< Control Register, offset: 0x0 */
  uint32_t ES;                                     /**< Error Status Register, offset: 0x4 */
  uint8_t RESERVED_0[4];
  uint32_t ERQ;                                    /**< Enable Request Register, offset: 0xC */
  uint8_t RESERVED_1[4];
  uint32_t EEI;                                    /**< Enable Error Interrupt Register, offset: 0x14 */
  uint8_t CEEI;                                    /**< Clear Enable Error Interrupt Register, offset: 0x18 */
  uint8_t SEEI;                                    /**< Set Enable Error Interrupt Register, offset: 0x19 */
  uint8_t CERQ;                                    /**< Clear Enable Request Register, offset: 0x1A */
  uint8_t SERQ;                                    /**< Set Enable Request Register, offset: 0x1B */
  uint8_t CDNE;                                    /**< Clear DONE Status Bit Register, offset: 0x1C */
  uint8_t SSRT;                                    /**< Set START Bit Register, offset: 0x1D */
  uint8_t CERR;                                    /**< Clear Error Register, offset: 0x1E */
  uint8_t CINT;                                    /**< Clear Interrupt Request Register, offset: 0x1F */
  uint8_t RESERVED_2[4];
  uint32_t INT;                                    /**< Interrupt Request Register, offset: 0x24 */
  uint8_t RESERVED_3[4];
  uint32_t ERR;                                    /**< Error Register, offset: 0x2C */
  uint8_t RESERVED_4[4];
  uint32_t HRS;                                    /**< Hardware Request Status Register, offset: 0x34 */
  uint8_t RESERVED_5[200];
  uint8_t DCHPRI3;                                 /**< Channel n Priority Register, offset: 0x100 */
  uint8_t DCHPRI2;                                 /**< Channel n Priority Register, offset: 0x101 */
  uint8_t DCHPRI1;                                 /**< Channel n Priority Register, offset: 0x102 */
  uint8_t DCHPRI0;                                 /**< Channel n Priority Register, offset: 0x103 */
  uint8_t DCHPRI7;                                 /**< Channel n Priority Register, offset: 0x104 */
  uint8_t DCHPRI6;                                 /**< Channel n Priority Register, offset: 0x105 */
  uint8_t DCHPRI5;                                 /**< Channel n Priority Register, offset: 0x106 */
  uint8_t DCHPRI4;                                 /**< Channel n Priority Register, offset: 0x107 */
  uint8_t DCHPRI11;                                /**< Channel n Priority Register, offset: 0x108 */
  uint8_t DCHPRI10;                                /**< Channel n Priority Register, offset: 0x109 */
  uint8_t DCHPRI9;                                 /**< Channel n Priority Register, offset: 0x10A */
  uint8_t DCHPRI8;                                 /**< Channel n Priority Register, offset: 0x10B */
  uint8_t RESERVED_6[2];
  uint8_t DCHPRI13;                                /**< Channel n Priority Register, offset: 0x10E */
  uint8_t DCHPRI12;                                /**< Channel n Priority Register, offset: 0x10F */
  uint8_t RESERVED_7[3824];
  struct {                                         /* offset: 0x1000, array step: 0x20 */
    uint32_t SADDR;                                  /**< TCD Source Address, array offset: 0x1000, array step: 0x20 */
    uint16_t SOFF;                                   /**< TCD Signed Source Address Offset, array offset: 0x1004, array step: 0x20 */
    uint16_t ATTR;                                   /**< TCD Transfer Attributes, array offset: 0x1006, array step: 0x20 */
    union {                                          /* offset: 0x1008, array step: 0x20 */
      uint32_t NBYTES_MLNO;                            /**< TCD Minor Byte Count (Minor Loop Disabled), array offset: 0x1008, array step: 0x20 */
      uint32_t NBYTES_MLOFFNO;                         /**< TCD Signed Minor Loop Offset (Minor Loop Enabled and Offset Disabled), array offset: 0x1008, array step: 0x20 */
      uint32_t NBYTES_MLOFFYES;                        /**< TCD Signed Minor Loop Offset (Minor Loop and Offset Enabled), array offset: 0x1008, array step: 0x20 */
    };
    uint32_t SLAST;                                  /**< TCD Last Source Address Adjustment, array offset: 0x100C, array step: 0x20 */
    uint32_t DADDR;                                  /**< TCD Destination Address, array offset: 0x1010, array step: 0x20 */
    uint16_t DOFF;                                   /**< TCD Signed Destination Address Offset, array offset: 0x1014, array step: 0x20 */
    union {                                          /* offset: 0x1016, array step: 0x20 */
      uint16_t CITER_ELINKNO;                          /**< TCD Current Minor Loop Link, Major Loop Count (Channel Linking Disabled), array offset: 0x1016, array step: 0x20 */
      uint16_t CITER_ELINKYES;                         /**< TCD Current Minor Loop Link, Major Loop Count (Channel Linking Enabled), array offset: 0x1016, array step: 0x20 */
    };
    uint32_t DLAST_SGA;                              /**< TCD Last Destination Address Adjustment/Scatter Gather Address, array offset: 0x1018, array step: 0x20 */
    uint16_t CSR;                                    /**< TCD Control and Status, array offset: 0x101C, array step: 0x20 */
    union {                                          /* offset: 0x101E, array step: 0x20 */
      uint16_t BITER_ELINKNO;                          /**< TCD Beginning Minor Loop Link, Major Loop Count (Channel Linking Disabled), array offset: 0x101E, array step: 0x20 */
      uint16_t BITER_ELINKYES;                         /**< TCD Beginning Minor Loop Link, Major Loop Count (Channel Linking Enabled), array offset: 0x101E, array step: 0x20 */
    };
  } TCD[14];
} volatile *DMA_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- DMA - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup DMA_Register_Accessor_Macros DMA - Register accessor macros
 * @{
 */


/* DMA - Register accessors */
#define DMA_CR_REG(base)                         ((base)->CR)
#define DMA_ES_REG(base)                         ((base)->ES)
#define DMA_ERQ_REG(base)                        ((base)->ERQ)
#define DMA_EEI_REG(base)                        ((base)->EEI)
#define DMA_CEEI_REG(base)                       ((base)->CEEI)
#define DMA_SEEI_REG(base)                       ((base)->SEEI)
#define DMA_CERQ_REG(base)                       ((base)->CERQ)
#define DMA_SERQ_REG(base)                       ((base)->SERQ)
#define DMA_CDNE_REG(base)                       ((base)->CDNE)
#define DMA_SSRT_REG(base)                       ((base)->SSRT)
#define DMA_CERR_REG(base)                       ((base)->CERR)
#define DMA_CINT_REG(base)                       ((base)->CINT)
#define DMA_INT_REG(base)                        ((base)->INT)
#define DMA_ERR_REG(base)                        ((base)->ERR)
#define DMA_HRS_REG(base)                        ((base)->HRS)
#define DMA_DCHPRI3_REG(base)                    ((base)->DCHPRI3)
#define DMA_DCHPRI2_REG(base)                    ((base)->DCHPRI2)
#define DMA_DCHPRI1_REG(base)                    ((base)->DCHPRI1)
#define DMA_DCHPRI0_REG(base)                    ((base)->DCHPRI0)
#define DMA_DCHPRI7_REG(base)                    ((base)->DCHPRI7)
#define DMA_DCHPRI6_REG(base)                    ((base)->DCHPRI6)
#define DMA_DCHPRI5_REG(base)                    ((base)->DCHPRI5)
#define DMA_DCHPRI4_REG(base)                    ((base)->DCHPRI4)
#define DMA_DCHPRI11_REG(base)                   ((base)->DCHPRI11)
#define DMA_DCHPRI10_REG(base)                   ((base)->DCHPRI10)
#define DMA_DCHPRI9_REG(base)                    ((base)->DCHPRI9)
#define DMA_DCHPRI8_REG(base)                    ((base)->DCHPRI8)
#define DMA_DCHPRI13_REG(base)                   ((base)->DCHPRI13)
#define DMA_DCHPRI12_REG(base)                   ((base)->DCHPRI12)
#define DMA_SADDR_REG(base,index)                ((base)->TCD[index].SADDR)
#define DMA_SOFF_REG(base,index)                 ((base)->TCD[index].SOFF)
#define DMA_ATTR_REG(base,index)                 ((base)->TCD[index].ATTR)
#define DMA_NBYTES_MLNO_REG(base,index)          ((base)->TCD[index].NBYTES_MLNO)
#define DMA_NBYTES_MLOFFNO_REG(base,index)       ((base)->TCD[index].NBYTES_MLOFFNO)
#define DMA_NBYTES_MLOFFYES_REG(base,index)      ((base)->TCD[index].NBYTES_MLOFFYES)
#define DMA_SLAST_REG(base,index)                ((base)->TCD[index].SLAST)
#define DMA_DADDR_REG(base,index)                ((base)->TCD[index].DADDR)
#define DMA_DOFF_REG(base,index)                 ((base)->TCD[index].DOFF)
#define DMA_CITER_ELINKNO_REG(base,index)        ((base)->TCD[index].CITER_ELINKNO)
#define DMA_CITER_ELINKYES_REG(base,index)       ((base)->TCD[index].CITER_ELINKYES)
#define DMA_DLAST_SGA_REG(base,index)            ((base)->TCD[index].DLAST_SGA)
#define DMA_CSR_REG(base,index)                  ((base)->TCD[index].CSR)
#define DMA_BITER_ELINKNO_REG(base,index)        ((base)->TCD[index].BITER_ELINKNO)
#define DMA_BITER_ELINKYES_REG(base,index)       ((base)->TCD[index].BITER_ELINKYES)

/**
 * @}
 */ /* end of group DMA_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- DMA Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup DMA_Register_Masks DMA Register Masks
 * @{
 */

/* CR Bit Fields */
#define DMA_CR_EDBG_MASK                         0x2u
#define DMA_CR_EDBG_SHIFT                        1
#define DMA_CR_ERCA_MASK                         0x4u
#define DMA_CR_ERCA_SHIFT                        2
#define DMA_CR_HOE_MASK                          0x10u
#define DMA_CR_HOE_SHIFT                         4
#define DMA_CR_HALT_MASK                         0x20u
#define DMA_CR_HALT_SHIFT                        5
#define DMA_CR_CLM_MASK                          0x40u
#define DMA_CR_CLM_SHIFT                         6
#define DMA_CR_EMLM_MASK                         0x80u
#define DMA_CR_EMLM_SHIFT                        7
#define DMA_CR_ECX_MASK                          0x10000u
#define DMA_CR_ECX_SHIFT                         16
#define DMA_CR_CX_MASK                           0x20000u
#define DMA_CR_CX_SHIFT                          17
/* ES Bit Fields */
#define DMA_ES_DBE_MASK                          0x1u
#define DMA_ES_DBE_SHIFT                         0
#define DMA_ES_SBE_MASK                          0x2u
#define DMA_ES_SBE_SHIFT                         1
#define DMA_ES_SGE_MASK                          0x4u
#define DMA_ES_SGE_SHIFT                         2
#define DMA_ES_NCE_MASK                          0x8u
#define DMA_ES_NCE_SHIFT                         3
#define DMA_ES_DOE_MASK                          0x10u
#define DMA_ES_DOE_SHIFT                         4
#define DMA_ES_DAE_MASK                          0x20u
#define DMA_ES_DAE_SHIFT                         5
#define DMA_ES_SOE_MASK                          0x40u
#define DMA_ES_SOE_SHIFT                         6
#define DMA_ES_SAE_MASK                          0x80u
#define DMA_ES_SAE_SHIFT                         7
#define DMA_ES_ERRCHN_MASK                       0xF00u
#define DMA_ES_ERRCHN_SHIFT                      8
#define DMA_ES_ERRCHN(x)                         (((uint32_t)(((uint32_t)(x))<<DMA_ES_ERRCHN_SHIFT))&DMA_ES_ERRCHN_MASK)
#define DMA_ES_CPE_MASK                          0x4000u
#define DMA_ES_CPE_SHIFT                         14
#define DMA_ES_ECX_MASK                          0x10000u
#define DMA_ES_ECX_SHIFT                         16
#define DMA_ES_VLD_MASK                          0x80000000u
#define DMA_ES_VLD_SHIFT                         31
/* ERQ Bit Fields */
#define DMA_ERQ_ERQ0_MASK                        0x1u
#define DMA_ERQ_ERQ0_SHIFT                       0
#define DMA_ERQ_ERQ1_MASK                        0x2u
#define DMA_ERQ_ERQ1_SHIFT                       1
#define DMA_ERQ_ERQ2_MASK                        0x4u
#define DMA_ERQ_ERQ2_SHIFT                       2
#define DMA_ERQ_ERQ3_MASK                        0x8u
#define DMA_ERQ_ERQ3_SHIFT                       3
#define DMA_ERQ_ERQ4_MASK                        0x10u
#define DMA_ERQ_ERQ4_SHIFT                       4
#define DMA_ERQ_ERQ5_MASK                        0x20u
#define DMA_ERQ_ERQ5_SHIFT                       5
#define DMA_ERQ_ERQ6_MASK                        0x40u
#define DMA_ERQ_ERQ6_SHIFT                       6
#define DMA_ERQ_ERQ7_MASK                        0x80u
#define DMA_ERQ_ERQ7_SHIFT                       7
#define DMA_ERQ_ERQ8_MASK                        0x100u
#define DMA_ERQ_ERQ8_SHIFT                       8
#define DMA_ERQ_ERQ9_MASK                        0x200u
#define DMA_ERQ_ERQ9_SHIFT                       9
#define DMA_ERQ_ERQ10_MASK                       0x400u
#define DMA_ERQ_ERQ10_SHIFT                      10
#define DMA_ERQ_ERQ11_MASK                       0x800u
#define DMA_ERQ_ERQ11_SHIFT                      11
#define DMA_ERQ_ERQ12_MASK                       0x1000u
#define DMA_ERQ_ERQ12_SHIFT                      12
#define DMA_ERQ_ERQ13_MASK                       0x2000u
#define DMA_ERQ_ERQ13_SHIFT                      13
/* EEI Bit Fields */
#define DMA_EEI_EEI0_MASK                        0x1u
#define DMA_EEI_EEI0_SHIFT                       0
#define DMA_EEI_EEI1_MASK                        0x2u
#define DMA_EEI_EEI1_SHIFT                       1
#define DMA_EEI_EEI2_MASK                        0x4u
#define DMA_EEI_EEI2_SHIFT                       2
#define DMA_EEI_EEI3_MASK                        0x8u
#define DMA_EEI_EEI3_SHIFT                       3
#define DMA_EEI_EEI4_MASK                        0x10u
#define DMA_EEI_EEI4_SHIFT                       4
#define DMA_EEI_EEI5_MASK                        0x20u
#define DMA_EEI_EEI5_SHIFT                       5
#define DMA_EEI_EEI6_MASK                        0x40u
#define DMA_EEI_EEI6_SHIFT                       6
#define DMA_EEI_EEI7_MASK                        0x80u
#define DMA_EEI_EEI7_SHIFT                       7
#define DMA_EEI_EEI8_MASK                        0x100u
#define DMA_EEI_EEI8_SHIFT                       8
#define DMA_EEI_EEI9_MASK                        0x200u
#define DMA_EEI_EEI9_SHIFT                       9
#define DMA_EEI_EEI10_MASK                       0x400u
#define DMA_EEI_EEI10_SHIFT                      10
#define DMA_EEI_EEI11_MASK                       0x800u
#define DMA_EEI_EEI11_SHIFT                      11
#define DMA_EEI_EEI12_MASK                       0x1000u
#define DMA_EEI_EEI12_SHIFT                      12
#define DMA_EEI_EEI13_MASK                       0x2000u
#define DMA_EEI_EEI13_SHIFT                      13
/* CEEI Bit Fields */
#define DMA_CEEI_CEEI_MASK                       0xFu
#define DMA_CEEI_CEEI_SHIFT                      0
#define DMA_CEEI_CEEI(x)                         (((uint8_t)(((uint8_t)(x))<<DMA_CEEI_CEEI_SHIFT))&DMA_CEEI_CEEI_MASK)
#define DMA_CEEI_CAEE_MASK                       0x40u
#define DMA_CEEI_CAEE_SHIFT                      6
#define DMA_CEEI_NOP_MASK                        0x80u
#define DMA_CEEI_NOP_SHIFT                       7
/* SEEI Bit Fields */
#define DMA_SEEI_SEEI_MASK                       0xFu
#define DMA_SEEI_SEEI_SHIFT                      0
#define DMA_SEEI_SEEI(x)                         (((uint8_t)(((uint8_t)(x))<<DMA_SEEI_SEEI_SHIFT))&DMA_SEEI_SEEI_MASK)
#define DMA_SEEI_SAEE_MASK                       0x40u
#define DMA_SEEI_SAEE_SHIFT                      6
#define DMA_SEEI_NOP_MASK                        0x80u
#define DMA_SEEI_NOP_SHIFT                       7
/* CERQ Bit Fields */
#define DMA_CERQ_CERQ_MASK                       0xFu
#define DMA_CERQ_CERQ_SHIFT                      0
#define DMA_CERQ_CERQ(x)                         (((uint8_t)(((uint8_t)(x))<<DMA_CERQ_CERQ_SHIFT))&DMA_CERQ_CERQ_MASK)
#define DMA_CERQ_CAER_MASK                       0x40u
#define DMA_CERQ_CAER_SHIFT                      6
#define DMA_CERQ_NOP_MASK                        0x80u
#define DMA_CERQ_NOP_SHIFT                       7
/* SERQ Bit Fields */
#define DMA_SERQ_SERQ_MASK                       0xFu
#define DMA_SERQ_SERQ_SHIFT                      0
#define DMA_SERQ_SERQ(x)                         (((uint8_t)(((uint8_t)(x))<<DMA_SERQ_SERQ_SHIFT))&DMA_SERQ_SERQ_MASK)
#define DMA_SERQ_SAER_MASK                       0x40u
#define DMA_SERQ_SAER_SHIFT                      6
#define DMA_SERQ_NOP_MASK                        0x80u
#define DMA_SERQ_NOP_SHIFT                       7
/* CDNE Bit Fields */
#define DMA_CDNE_CDNE_MASK                       0xFu
#define DMA_CDNE_CDNE_SHIFT                      0
#define DMA_CDNE_CDNE(x)                         (((uint8_t)(((uint8_t)(x))<<DMA_CDNE_CDNE_SHIFT))&DMA_CDNE_CDNE_MASK)
#define DMA_CDNE_CADN_MASK                       0x40u
#define DMA_CDNE_CADN_SHIFT                      6
#define DMA_CDNE_NOP_MASK                        0x80u
#define DMA_CDNE_NOP_SHIFT                       7
/* SSRT Bit Fields */
#define DMA_SSRT_SSRT_MASK                       0xFu
#define DMA_SSRT_SSRT_SHIFT                      0
#define DMA_SSRT_SSRT(x)                         (((uint8_t)(((uint8_t)(x))<<DMA_SSRT_SSRT_SHIFT))&DMA_SSRT_SSRT_MASK)
#define DMA_SSRT_SAST_MASK                       0x40u
#define DMA_SSRT_SAST_SHIFT                      6
#define DMA_SSRT_NOP_MASK                        0x80u
#define DMA_SSRT_NOP_SHIFT                       7
/* CERR Bit Fields */
#define DMA_CERR_CERR_MASK                       0xFu
#define DMA_CERR_CERR_SHIFT                      0
#define DMA_CERR_CERR(x)                         (((uint8_t)(((uint8_t)(x))<<DMA_CERR_CERR_SHIFT))&DMA_CERR_CERR_MASK)
#define DMA_CERR_CAEI_MASK                       0x40u
#define DMA_CERR_CAEI_SHIFT                      6
#define DMA_CERR_NOP_MASK                        0x80u
#define DMA_CERR_NOP_SHIFT                       7
/* CINT Bit Fields */
#define DMA_CINT_CINT_MASK                       0xFu
#define DMA_CINT_CINT_SHIFT                      0
#define DMA_CINT_CINT(x)                         (((uint8_t)(((uint8_t)(x))<<DMA_CINT_CINT_SHIFT))&DMA_CINT_CINT_MASK)
#define DMA_CINT_CAIR_MASK                       0x40u
#define DMA_CINT_CAIR_SHIFT                      6
#define DMA_CINT_NOP_MASK                        0x80u
#define DMA_CINT_NOP_SHIFT                       7
/* INT Bit Fields */
#define DMA_INT_INT0_MASK                        0x1u
#define DMA_INT_INT0_SHIFT                       0
#define DMA_INT_INT1_MASK                        0x2u
#define DMA_INT_INT1_SHIFT                       1
#define DMA_INT_INT2_MASK                        0x4u
#define DMA_INT_INT2_SHIFT                       2
#define DMA_INT_INT3_MASK                        0x8u
#define DMA_INT_INT3_SHIFT                       3
#define DMA_INT_INT4_MASK                        0x10u
#define DMA_INT_INT4_SHIFT                       4
#define DMA_INT_INT5_MASK                        0x20u
#define DMA_INT_INT5_SHIFT                       5
#define DMA_INT_INT6_MASK                        0x40u
#define DMA_INT_INT6_SHIFT                       6
#define DMA_INT_INT7_MASK                        0x80u
#define DMA_INT_INT7_SHIFT                       7
#define DMA_INT_INT8_MASK                        0x100u
#define DMA_INT_INT8_SHIFT                       8
#define DMA_INT_INT9_MASK                        0x200u
#define DMA_INT_INT9_SHIFT                       9
#define DMA_INT_INT10_MASK                       0x400u
#define DMA_INT_INT10_SHIFT                      10
#define DMA_INT_INT11_MASK                       0x800u
#define DMA_INT_INT11_SHIFT                      11
#define DMA_INT_INT12_MASK                       0x1000u
#define DMA_INT_INT12_SHIFT                      12
#define DMA_INT_INT13_MASK                       0x2000u
#define DMA_INT_INT13_SHIFT                      13
/* ERR Bit Fields */
#define DMA_ERR_ERR0_MASK                        0x1u
#define DMA_ERR_ERR0_SHIFT                       0
#define DMA_ERR_ERR1_MASK                        0x2u
#define DMA_ERR_ERR1_SHIFT                       1
#define DMA_ERR_ERR2_MASK                        0x4u
#define DMA_ERR_ERR2_SHIFT                       2
#define DMA_ERR_ERR3_MASK                        0x8u
#define DMA_ERR_ERR3_SHIFT                       3
#define DMA_ERR_ERR4_MASK                        0x10u
#define DMA_ERR_ERR4_SHIFT                       4
#define DMA_ERR_ERR5_MASK                        0x20u
#define DMA_ERR_ERR5_SHIFT                       5
#define DMA_ERR_ERR6_MASK                        0x40u
#define DMA_ERR_ERR6_SHIFT                       6
#define DMA_ERR_ERR7_MASK                        0x80u
#define DMA_ERR_ERR7_SHIFT                       7
#define DMA_ERR_ERR8_MASK                        0x100u
#define DMA_ERR_ERR8_SHIFT                       8
#define DMA_ERR_ERR9_MASK                        0x200u
#define DMA_ERR_ERR9_SHIFT                       9
#define DMA_ERR_ERR10_MASK                       0x400u
#define DMA_ERR_ERR10_SHIFT                      10
#define DMA_ERR_ERR11_MASK                       0x800u
#define DMA_ERR_ERR11_SHIFT                      11
#define DMA_ERR_ERR12_MASK                       0x1000u
#define DMA_ERR_ERR12_SHIFT                      12
#define DMA_ERR_ERR13_MASK                       0x2000u
#define DMA_ERR_ERR13_SHIFT                      13
/* HRS Bit Fields */
#define DMA_HRS_HRS0_MASK                        0x1u
#define DMA_HRS_HRS0_SHIFT                       0
#define DMA_HRS_HRS1_MASK                        0x2u
#define DMA_HRS_HRS1_SHIFT                       1
#define DMA_HRS_HRS2_MASK                        0x4u
#define DMA_HRS_HRS2_SHIFT                       2
#define DMA_HRS_HRS3_MASK                        0x8u
#define DMA_HRS_HRS3_SHIFT                       3
#define DMA_HRS_HRS4_MASK                        0x10u
#define DMA_HRS_HRS4_SHIFT                       4
#define DMA_HRS_HRS5_MASK                        0x20u
#define DMA_HRS_HRS5_SHIFT                       5
#define DMA_HRS_HRS6_MASK                        0x40u
#define DMA_HRS_HRS6_SHIFT                       6
#define DMA_HRS_HRS7_MASK                        0x80u
#define DMA_HRS_HRS7_SHIFT                       7
#define DMA_HRS_HRS8_MASK                        0x100u
#define DMA_HRS_HRS8_SHIFT                       8
#define DMA_HRS_HRS9_MASK                        0x200u
#define DMA_HRS_HRS9_SHIFT                       9
#define DMA_HRS_HRS10_MASK                       0x400u
#define DMA_HRS_HRS10_SHIFT                      10
#define DMA_HRS_HRS11_MASK                       0x800u
#define DMA_HRS_HRS11_SHIFT                      11
#define DMA_HRS_HRS12_MASK                       0x1000u
#define DMA_HRS_HRS12_SHIFT                      12
#define DMA_HRS_HRS13_MASK                       0x2000u
#define DMA_HRS_HRS13_SHIFT                      13
/* DCHPRI3 Bit Fields */
#define DMA_DCHPRI3_CHPRI_MASK                   0xFu
#define DMA_DCHPRI3_CHPRI_SHIFT                  0
#define DMA_DCHPRI3_CHPRI(x)                     (((uint8_t)(((uint8_t)(x))<<DMA_DCHPRI3_CHPRI_SHIFT))&DMA_DCHPRI3_CHPRI_MASK)
#define DMA_DCHPRI3_DPA_MASK                     0x40u
#define DMA_DCHPRI3_DPA_SHIFT                    6
#define DMA_DCHPRI3_ECP_MASK                     0x80u
#define DMA_DCHPRI3_ECP_SHIFT                    7
/* DCHPRI2 Bit Fields */
#define DMA_DCHPRI2_CHPRI_MASK                   0xFu
#define DMA_DCHPRI2_CHPRI_SHIFT                  0
#define DMA_DCHPRI2_CHPRI(x)                     (((uint8_t)(((uint8_t)(x))<<DMA_DCHPRI2_CHPRI_SHIFT))&DMA_DCHPRI2_CHPRI_MASK)
#define DMA_DCHPRI2_DPA_MASK                     0x40u
#define DMA_DCHPRI2_DPA_SHIFT                    6
#define DMA_DCHPRI2_ECP_MASK                     0x80u
#define DMA_DCHPRI2_ECP_SHIFT                    7
/* DCHPRI1 Bit Fields */
#define DMA_DCHPRI1_CHPRI_MASK                   0xFu
#define DMA_DCHPRI1_CHPRI_SHIFT                  0
#define DMA_DCHPRI1_CHPRI(x)                     (((uint8_t)(((uint8_t)(x))<<DMA_DCHPRI1_CHPRI_SHIFT))&DMA_DCHPRI1_CHPRI_MASK)
#define DMA_DCHPRI1_DPA_MASK                     0x40u
#define DMA_DCHPRI1_DPA_SHIFT                    6
#define DMA_DCHPRI1_ECP_MASK                     0x80u
#define DMA_DCHPRI1_ECP_SHIFT                    7
/* DCHPRI0 Bit Fields */
#define DMA_DCHPRI0_CHPRI_MASK                   0xFu
#define DMA_DCHPRI0_CHPRI_SHIFT                  0
#define DMA_DCHPRI0_CHPRI(x)                     (((uint8_t)(((uint8_t)(x))<<DMA_DCHPRI0_CHPRI_SHIFT))&DMA_DCHPRI0_CHPRI_MASK)
#define DMA_DCHPRI0_DPA_MASK                     0x40u
#define DMA_DCHPRI0_DPA_SHIFT                    6
#define DMA_DCHPRI0_ECP_MASK                     0x80u
#define DMA_DCHPRI0_ECP_SHIFT                    7
/* DCHPRI7 Bit Fields */
#define DMA_DCHPRI7_CHPRI_MASK                   0xFu
#define DMA_DCHPRI7_CHPRI_SHIFT                  0
#define DMA_DCHPRI7_CHPRI(x)                     (((uint8_t)(((uint8_t)(x))<<DMA_DCHPRI7_CHPRI_SHIFT))&DMA_DCHPRI7_CHPRI_MASK)
#define DMA_DCHPRI7_DPA_MASK                     0x40u
#define DMA_DCHPRI7_DPA_SHIFT                    6
#define DMA_DCHPRI7_ECP_MASK                     0x80u
#define DMA_DCHPRI7_ECP_SHIFT                    7
/* DCHPRI6 Bit Fields */
#define DMA_DCHPRI6_CHPRI_MASK                   0xFu
#define DMA_DCHPRI6_CHPRI_SHIFT                  0
#define DMA_DCHPRI6_CHPRI(x)                     (((uint8_t)(((uint8_t)(x))<<DMA_DCHPRI6_CHPRI_SHIFT))&DMA_DCHPRI6_CHPRI_MASK)
#define DMA_DCHPRI6_DPA_MASK                     0x40u
#define DMA_DCHPRI6_DPA_SHIFT                    6
#define DMA_DCHPRI6_ECP_MASK                     0x80u
#define DMA_DCHPRI6_ECP_SHIFT                    7
/* DCHPRI5 Bit Fields */
#define DMA_DCHPRI5_CHPRI_MASK                   0xFu
#define DMA_DCHPRI5_CHPRI_SHIFT                  0
#define DMA_DCHPRI5_CHPRI(x)                     (((uint8_t)(((uint8_t)(x))<<DMA_DCHPRI5_CHPRI_SHIFT))&DMA_DCHPRI5_CHPRI_MASK)
#define DMA_DCHPRI5_DPA_MASK                     0x40u
#define DMA_DCHPRI5_DPA_SHIFT                    6
#define DMA_DCHPRI5_ECP_MASK                     0x80u
#define DMA_DCHPRI5_ECP_SHIFT                    7
/* DCHPRI4 Bit Fields */
#define DMA_DCHPRI4_CHPRI_MASK                   0xFu
#define DMA_DCHPRI4_CHPRI_SHIFT                  0
#define DMA_DCHPRI4_CHPRI(x)                     (((uint8_t)(((uint8_t)(x))<<DMA_DCHPRI4_CHPRI_SHIFT))&DMA_DCHPRI4_CHPRI_MASK)
#define DMA_DCHPRI4_DPA_MASK                     0x40u
#define DMA_DCHPRI4_DPA_SHIFT                    6
#define DMA_DCHPRI4_ECP_MASK                     0x80u
#define DMA_DCHPRI4_ECP_SHIFT                    7
/* DCHPRI11 Bit Fields */
#define DMA_DCHPRI11_CHPRI_MASK                  0xFu
#define DMA_DCHPRI11_CHPRI_SHIFT                 0
#define DMA_DCHPRI11_CHPRI(x)                    (((uint8_t)(((uint8_t)(x))<<DMA_DCHPRI11_CHPRI_SHIFT))&DMA_DCHPRI11_CHPRI_MASK)
#define DMA_DCHPRI11_DPA_MASK                    0x40u
#define DMA_DCHPRI11_DPA_SHIFT                   6
#define DMA_DCHPRI11_ECP_MASK                    0x80u
#define DMA_DCHPRI11_ECP_SHIFT                   7
/* DCHPRI10 Bit Fields */
#define DMA_DCHPRI10_CHPRI_MASK                  0xFu
#define DMA_DCHPRI10_CHPRI_SHIFT                 0
#define DMA_DCHPRI10_CHPRI(x)                    (((uint8_t)(((uint8_t)(x))<<DMA_DCHPRI10_CHPRI_SHIFT))&DMA_DCHPRI10_CHPRI_MASK)
#define DMA_DCHPRI10_DPA_MASK                    0x40u
#define DMA_DCHPRI10_DPA_SHIFT                   6
#define DMA_DCHPRI10_ECP_MASK                    0x80u
#define DMA_DCHPRI10_ECP_SHIFT                   7
/* DCHPRI9 Bit Fields */
#define DMA_DCHPRI9_CHPRI_MASK                   0xFu
#define DMA_DCHPRI9_CHPRI_SHIFT                  0
#define DMA_DCHPRI9_CHPRI(x)                     (((uint8_t)(((uint8_t)(x))<<DMA_DCHPRI9_CHPRI_SHIFT))&DMA_DCHPRI9_CHPRI_MASK)
#define DMA_DCHPRI9_DPA_MASK                     0x40u
#define DMA_DCHPRI9_DPA_SHIFT                    6
#define DMA_DCHPRI9_ECP_MASK                     0x80u
#define DMA_DCHPRI9_ECP_SHIFT                    7
/* DCHPRI8 Bit Fields */
#define DMA_DCHPRI8_CHPRI_MASK                   0xFu
#define DMA_DCHPRI8_CHPRI_SHIFT                  0
#define DMA_DCHPRI8_CHPRI(x)                     (((uint8_t)(((uint8_t)(x))<<DMA_DCHPRI8_CHPRI_SHIFT))&DMA_DCHPRI8_CHPRI_MASK)
#define DMA_DCHPRI8_DPA_MASK                     0x40u
#define DMA_DCHPRI8_DPA_SHIFT                    6
#define DMA_DCHPRI8_ECP_MASK                     0x80u
#define DMA_DCHPRI8_ECP_SHIFT                    7
/* DCHPRI13 Bit Fields */
#define DMA_DCHPRI13_CHPRI_MASK                  0xFu
#define DMA_DCHPRI13_CHPRI_SHIFT                 0
#define DMA_DCHPRI13_CHPRI(x)                    (((uint8_t)(((uint8_t)(x))<<DMA_DCHPRI13_CHPRI_SHIFT))&DMA_DCHPRI13_CHPRI_MASK)
#define DMA_DCHPRI13_DPA_MASK                    0x40u
#define DMA_DCHPRI13_DPA_SHIFT                   6
#define DMA_DCHPRI13_ECP_MASK                    0x80u
#define DMA_DCHPRI13_ECP_SHIFT                   7
/* DCHPRI12 Bit Fields */
#define DMA_DCHPRI12_CHPRI_MASK                  0xFu
#define DMA_DCHPRI12_CHPRI_SHIFT                 0
#define DMA_DCHPRI12_CHPRI(x)                    (((uint8_t)(((uint8_t)(x))<<DMA_DCHPRI12_CHPRI_SHIFT))&DMA_DCHPRI12_CHPRI_MASK)
#define DMA_DCHPRI12_DPA_MASK                    0x40u
#define DMA_DCHPRI12_DPA_SHIFT                   6
#define DMA_DCHPRI12_ECP_MASK                    0x80u
#define DMA_DCHPRI12_ECP_SHIFT                   7
/* SADDR Bit Fields */
#define DMA_SADDR_SADDR_MASK                     0xFFFFFFFFu
#define DMA_SADDR_SADDR_SHIFT                    0
#define DMA_SADDR_SADDR(x)                       (((uint32_t)(((uint32_t)(x))<<DMA_SADDR_SADDR_SHIFT))&DMA_SADDR_SADDR_MASK)
/* SOFF Bit Fields */
#define DMA_SOFF_SOFF_MASK                       0xFFFFu
#define DMA_SOFF_SOFF_SHIFT                      0
#define DMA_SOFF_SOFF(x)                         (((uint16_t)(((uint16_t)(x))<<DMA_SOFF_SOFF_SHIFT))&DMA_SOFF_SOFF_MASK)
/* ATTR Bit Fields */
#define DMA_ATTR_DSIZE_MASK                      0x7u
#define DMA_ATTR_DSIZE_SHIFT                     0
#define DMA_ATTR_DSIZE(x)                        (((uint16_t)(((uint16_t)(x))<<DMA_ATTR_DSIZE_SHIFT))&DMA_ATTR_DSIZE_MASK)
#define DMA_ATTR_DMOD_MASK                       0xF8u
#define DMA_ATTR_DMOD_SHIFT                      3
#define DMA_ATTR_DMOD(x)                         (((uint16_t)(((uint16_t)(x))<<DMA_ATTR_DMOD_SHIFT))&DMA_ATTR_DMOD_MASK)
#define DMA_ATTR_SSIZE_MASK                      0x700u
#define DMA_ATTR_SSIZE_SHIFT                     8
#define DMA_ATTR_SSIZE(x)                        (((uint16_t)(((uint16_t)(x))<<DMA_ATTR_SSIZE_SHIFT))&DMA_ATTR_SSIZE_MASK)
#define DMA_ATTR_SMOD_MASK                       0xF800u
#define DMA_ATTR_SMOD_SHIFT                      11
#define DMA_ATTR_SMOD(x)                         (((uint16_t)(((uint16_t)(x))<<DMA_ATTR_SMOD_SHIFT))&DMA_ATTR_SMOD_MASK)
/* NBYTES_MLNO Bit Fields */
#define DMA_NBYTES_MLNO_NBYTES_MASK              0xFFFFFFFFu
#define DMA_NBYTES_MLNO_NBYTES_SHIFT             0
#define DMA_NBYTES_MLNO_NBYTES(x)                (((uint32_t)(((uint32_t)(x))<<DMA_NBYTES_MLNO_NBYTES_SHIFT))&DMA_NBYTES_MLNO_NBYTES_MASK)
/* NBYTES_MLOFFNO Bit Fields */
#define DMA_NBYTES_MLOFFNO_NBYTES_MASK           0x3FFFFFFFu
#define DMA_NBYTES_MLOFFNO_NBYTES_SHIFT          0
#define DMA_NBYTES_MLOFFNO_NBYTES(x)             (((uint32_t)(((uint32_t)(x))<<DMA_NBYTES_MLOFFNO_NBYTES_SHIFT))&DMA_NBYTES_MLOFFNO_NBYTES_MASK)
#define DMA_NBYTES_MLOFFNO_DMLOE_MASK            0x40000000u
#define DMA_NBYTES_MLOFFNO_DMLOE_SHIFT           30
#define DMA_NBYTES_MLOFFNO_SMLOE_MASK            0x80000000u
#define DMA_NBYTES_MLOFFNO_SMLOE_SHIFT           31
/* NBYTES_MLOFFYES Bit Fields */
#define DMA_NBYTES_MLOFFYES_NBYTES_MASK          0x3FFu
#define DMA_NBYTES_MLOFFYES_NBYTES_SHIFT         0
#define DMA_NBYTES_MLOFFYES_NBYTES(x)            (((uint32_t)(((uint32_t)(x))<<DMA_NBYTES_MLOFFYES_NBYTES_SHIFT))&DMA_NBYTES_MLOFFYES_NBYTES_MASK)
#define DMA_NBYTES_MLOFFYES_MLOFF_MASK           0x3FFFFC00u
#define DMA_NBYTES_MLOFFYES_MLOFF_SHIFT          10
#define DMA_NBYTES_MLOFFYES_MLOFF(x)             (((uint32_t)(((uint32_t)(x))<<DMA_NBYTES_MLOFFYES_MLOFF_SHIFT))&DMA_NBYTES_MLOFFYES_MLOFF_MASK)
#define DMA_NBYTES_MLOFFYES_DMLOE_MASK           0x40000000u
#define DMA_NBYTES_MLOFFYES_DMLOE_SHIFT          30
#define DMA_NBYTES_MLOFFYES_SMLOE_MASK           0x80000000u
#define DMA_NBYTES_MLOFFYES_SMLOE_SHIFT          31
/* SLAST Bit Fields */
#define DMA_SLAST_SLAST_MASK                     0xFFFFFFFFu
#define DMA_SLAST_SLAST_SHIFT                    0
#define DMA_SLAST_SLAST(x)                       (((uint32_t)(((uint32_t)(x))<<DMA_SLAST_SLAST_SHIFT))&DMA_SLAST_SLAST_MASK)
/* DADDR Bit Fields */
#define DMA_DADDR_DADDR_MASK                     0xFFFFFFFFu
#define DMA_DADDR_DADDR_SHIFT                    0
#define DMA_DADDR_DADDR(x)                       (((uint32_t)(((uint32_t)(x))<<DMA_DADDR_DADDR_SHIFT))&DMA_DADDR_DADDR_MASK)
/* DOFF Bit Fields */
#define DMA_DOFF_DOFF_MASK                       0xFFFFu
#define DMA_DOFF_DOFF_SHIFT                      0
#define DMA_DOFF_DOFF(x)                         (((uint16_t)(((uint16_t)(x))<<DMA_DOFF_DOFF_SHIFT))&DMA_DOFF_DOFF_MASK)
/* CITER_ELINKNO Bit Fields */
#define DMA_CITER_ELINKNO_CITER_MASK             0x7FFFu
#define DMA_CITER_ELINKNO_CITER_SHIFT            0
#define DMA_CITER_ELINKNO_CITER(x)               (((uint16_t)(((uint16_t)(x))<<DMA_CITER_ELINKNO_CITER_SHIFT))&DMA_CITER_ELINKNO_CITER_MASK)
#define DMA_CITER_ELINKNO_ELINK_MASK             0x8000u
#define DMA_CITER_ELINKNO_ELINK_SHIFT            15
/* CITER_ELINKYES Bit Fields */
#define DMA_CITER_ELINKYES_CITER_MASK            0x1FFu
#define DMA_CITER_ELINKYES_CITER_SHIFT           0
#define DMA_CITER_ELINKYES_CITER(x)              (((uint16_t)(((uint16_t)(x))<<DMA_CITER_ELINKYES_CITER_SHIFT))&DMA_CITER_ELINKYES_CITER_MASK)
#define DMA_CITER_ELINKYES_LINKCH_MASK           0x1E00u
#define DMA_CITER_ELINKYES_LINKCH_SHIFT          9
#define DMA_CITER_ELINKYES_LINKCH(x)             (((uint16_t)(((uint16_t)(x))<<DMA_CITER_ELINKYES_LINKCH_SHIFT))&DMA_CITER_ELINKYES_LINKCH_MASK)
#define DMA_CITER_ELINKYES_ELINK_MASK            0x8000u
#define DMA_CITER_ELINKYES_ELINK_SHIFT           15
/* DLAST_SGA Bit Fields */
#define DMA_DLAST_SGA_DLASTSGA_MASK              0xFFFFFFFFu
#define DMA_DLAST_SGA_DLASTSGA_SHIFT             0
#define DMA_DLAST_SGA_DLASTSGA(x)                (((uint32_t)(((uint32_t)(x))<<DMA_DLAST_SGA_DLASTSGA_SHIFT))&DMA_DLAST_SGA_DLASTSGA_MASK)
/* CSR Bit Fields */
#define DMA_CSR_START_MASK                       0x1u
#define DMA_CSR_START_SHIFT                      0
#define DMA_CSR_INTMAJOR_MASK                    0x2u
#define DMA_CSR_INTMAJOR_SHIFT                   1
#define DMA_CSR_INTHALF_MASK                     0x4u
#define DMA_CSR_INTHALF_SHIFT                    2
#define DMA_CSR_DREQ_MASK                        0x8u
#define DMA_CSR_DREQ_SHIFT                       3
#define DMA_CSR_ESG_MASK                         0x10u
#define DMA_CSR_ESG_SHIFT                        4
#define DMA_CSR_MAJORELINK_MASK                  0x20u
#define DMA_CSR_MAJORELINK_SHIFT                 5
#define DMA_CSR_ACTIVE_MASK                      0x40u
#define DMA_CSR_ACTIVE_SHIFT                     6
#define DMA_CSR_DONE_MASK                        0x80u
#define DMA_CSR_DONE_SHIFT                       7
#define DMA_CSR_MAJORLINKCH_MASK                 0xF00u
#define DMA_CSR_MAJORLINKCH_SHIFT                8
#define DMA_CSR_MAJORLINKCH(x)                   (((uint16_t)(((uint16_t)(x))<<DMA_CSR_MAJORLINKCH_SHIFT))&DMA_CSR_MAJORLINKCH_MASK)
#define DMA_CSR_BWC_MASK                         0xC000u
#define DMA_CSR_BWC_SHIFT                        14
#define DMA_CSR_BWC(x)                           (((uint16_t)(((uint16_t)(x))<<DMA_CSR_BWC_SHIFT))&DMA_CSR_BWC_MASK)
/* BITER_ELINKNO Bit Fields */
#define DMA_BITER_ELINKNO_BITER_MASK             0x7FFFu
#define DMA_BITER_ELINKNO_BITER_SHIFT            0
#define DMA_BITER_ELINKNO_BITER(x)               (((uint16_t)(((uint16_t)(x))<<DMA_BITER_ELINKNO_BITER_SHIFT))&DMA_BITER_ELINKNO_BITER_MASK)
#define DMA_BITER_ELINKNO_ELINK_MASK             0x8000u
#define DMA_BITER_ELINKNO_ELINK_SHIFT            15
/* BITER_ELINKYES Bit Fields */
#define DMA_BITER_ELINKYES_BITER_MASK            0x1FFu
#define DMA_BITER_ELINKYES_BITER_SHIFT           0
#define DMA_BITER_ELINKYES_BITER(x)              (((uint16_t)(((uint16_t)(x))<<DMA_BITER_ELINKYES_BITER_SHIFT))&DMA_BITER_ELINKYES_BITER_MASK)
#define DMA_BITER_ELINKYES_LINKCH_MASK           0x1E00u
#define DMA_BITER_ELINKYES_LINKCH_SHIFT          9
#define DMA_BITER_ELINKYES_LINKCH(x)             (((uint16_t)(((uint16_t)(x))<<DMA_BITER_ELINKYES_LINKCH_SHIFT))&DMA_BITER_ELINKYES_LINKCH_MASK)
#define DMA_BITER_ELINKYES_ELINK_MASK            0x8000u
#define DMA_BITER_ELINKYES_ELINK_SHIFT           15

/**
 * @}
 */ /* end of group DMA_Register_Masks */


/* DMA - Peripheral instance base addresses */
/** Peripheral DMA base pointer */
#define DMA_BASE_PTR                             ((DMA_MemMapPtr)0x40008000u)

/* ----------------------------------------------------------------------------
   -- DMA - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup DMA_Register_Accessor_Macros DMA - Register accessor macros
 * @{
 */


/* DMA - Register instance definitions */
/* DMA */
#define DMA_CR                                   DMA_CR_REG(DMA_BASE_PTR)
#define DMA_ES                                   DMA_ES_REG(DMA_BASE_PTR)
#define DMA_ERQ                                  DMA_ERQ_REG(DMA_BASE_PTR)
#define DMA_EEI                                  DMA_EEI_REG(DMA_BASE_PTR)
#define DMA_CEEI                                 DMA_CEEI_REG(DMA_BASE_PTR)
#define DMA_SEEI                                 DMA_SEEI_REG(DMA_BASE_PTR)
#define DMA_CERQ                                 DMA_CERQ_REG(DMA_BASE_PTR)
#define DMA_SERQ                                 DMA_SERQ_REG(DMA_BASE_PTR)
#define DMA_CDNE                                 DMA_CDNE_REG(DMA_BASE_PTR)
#define DMA_SSRT                                 DMA_SSRT_REG(DMA_BASE_PTR)
#define DMA_CERR                                 DMA_CERR_REG(DMA_BASE_PTR)
#define DMA_CINT                                 DMA_CINT_REG(DMA_BASE_PTR)
#define DMA_INT                                  DMA_INT_REG(DMA_BASE_PTR)
#define DMA_ERR                                  DMA_ERR_REG(DMA_BASE_PTR)
#define DMA_HRS                                  DMA_HRS_REG(DMA_BASE_PTR)
#define DMA_DCHPRI3                              DMA_DCHPRI3_REG(DMA_BASE_PTR)
#define DMA_DCHPRI2                              DMA_DCHPRI2_REG(DMA_BASE_PTR)
#define DMA_DCHPRI1                              DMA_DCHPRI1_REG(DMA_BASE_PTR)
#define DMA_DCHPRI0                              DMA_DCHPRI0_REG(DMA_BASE_PTR)
#define DMA_DCHPRI7                              DMA_DCHPRI7_REG(DMA_BASE_PTR)
#define DMA_DCHPRI6                              DMA_DCHPRI6_REG(DMA_BASE_PTR)
#define DMA_DCHPRI5                              DMA_DCHPRI5_REG(DMA_BASE_PTR)
#define DMA_DCHPRI4                              DMA_DCHPRI4_REG(DMA_BASE_PTR)
#define DMA_DCHPRI11                             DMA_DCHPRI11_REG(DMA_BASE_PTR)
#define DMA_DCHPRI10                             DMA_DCHPRI10_REG(DMA_BASE_PTR)
#define DMA_DCHPRI9                              DMA_DCHPRI9_REG(DMA_BASE_PTR)
#define DMA_DCHPRI8                              DMA_DCHPRI8_REG(DMA_BASE_PTR)
#define DMA_DCHPRI13                             DMA_DCHPRI13_REG(DMA_BASE_PTR)
#define DMA_DCHPRI12                             DMA_DCHPRI12_REG(DMA_BASE_PTR)
#define DMA_TCD0_SADDR                           DMA_SADDR_REG(DMA_BASE_PTR,0)
#define DMA_TCD0_SOFF                            DMA_SOFF_REG(DMA_BASE_PTR,0)
#define DMA_TCD0_ATTR                            DMA_ATTR_REG(DMA_BASE_PTR,0)
#define DMA_TCD0_NBYTES_MLNO                     DMA_NBYTES_MLNO_REG(DMA_BASE_PTR,0)
#define DMA_TCD0_NBYTES_MLOFFNO                  DMA_NBYTES_MLOFFNO_REG(DMA_BASE_PTR,0)
#define DMA_TCD0_NBYTES_MLOFFYES                 DMA_NBYTES_MLOFFYES_REG(DMA_BASE_PTR,0)
#define DMA_TCD0_SLAST                           DMA_SLAST_REG(DMA_BASE_PTR,0)
#define DMA_TCD0_DADDR                           DMA_DADDR_REG(DMA_BASE_PTR,0)
#define DMA_TCD0_DOFF                            DMA_DOFF_REG(DMA_BASE_PTR,0)
#define DMA_TCD0_CITER_ELINKNO                   DMA_CITER_ELINKNO_REG(DMA_BASE_PTR,0)
#define DMA_TCD0_CITER_ELINKYES                  DMA_CITER_ELINKYES_REG(DMA_BASE_PTR,0)
#define DMA_TCD0_DLASTSGA                        DMA_DLAST_SGA_REG(DMA_BASE_PTR,0)
#define DMA_TCD0_CSR                             DMA_CSR_REG(DMA_BASE_PTR,0)
#define DMA_TCD0_BITER_ELINKNO                   DMA_BITER_ELINKNO_REG(DMA_BASE_PTR,0)
#define DMA_TCD0_BITER_ELINKYES                  DMA_BITER_ELINKYES_REG(DMA_BASE_PTR,0)
#define DMA_TCD1_SADDR                           DMA_SADDR_REG(DMA_BASE_PTR,1)
#define DMA_TCD1_SOFF                            DMA_SOFF_REG(DMA_BASE_PTR,1)
#define DMA_TCD1_ATTR                            DMA_ATTR_REG(DMA_BASE_PTR,1)
#define DMA_TCD1_NBYTES_MLNO                     DMA_NBYTES_MLNO_REG(DMA_BASE_PTR,1)
#define DMA_TCD1_NBYTES_MLOFFNO                  DMA_NBYTES_MLOFFNO_REG(DMA_BASE_PTR,1)
#define DMA_TCD1_NBYTES_MLOFFYES                 DMA_NBYTES_MLOFFYES_REG(DMA_BASE_PTR,1)
#define DMA_TCD1_SLAST                           DMA_SLAST_REG(DMA_BASE_PTR,1)
#define DMA_TCD1_DADDR                           DMA_DADDR_REG(DMA_BASE_PTR,1)
#define DMA_TCD1_DOFF                            DMA_DOFF_REG(DMA_BASE_PTR,1)
#define DMA_TCD1_CITER_ELINKNO                   DMA_CITER_ELINKNO_REG(DMA_BASE_PTR,1)
#define DMA_TCD1_CITER_ELINKYES                  DMA_CITER_ELINKYES_REG(DMA_BASE_PTR,1)
#define DMA_TCD1_DLASTSGA                        DMA_DLAST_SGA_REG(DMA_BASE_PTR,1)
#define DMA_TCD1_CSR                             DMA_CSR_REG(DMA_BASE_PTR,1)
#define DMA_TCD1_BITER_ELINKNO                   DMA_BITER_ELINKNO_REG(DMA_BASE_PTR,1)
#define DMA_TCD1_BITER_ELINKYES                  DMA_BITER_ELINKYES_REG(DMA_BASE_PTR,1)
#define DMA_TCD2_SADDR                           DMA_SADDR_REG(DMA_BASE_PTR,2)
#define DMA_TCD2_SOFF                            DMA_SOFF_REG(DMA_BASE_PTR,2)
#define DMA_TCD2_ATTR                            DMA_ATTR_REG(DMA_BASE_PTR,2)
#define DMA_TCD2_NBYTES_MLNO                     DMA_NBYTES_MLNO_REG(DMA_BASE_PTR,2)
#define DMA_TCD2_NBYTES_MLOFFNO                  DMA_NBYTES_MLOFFNO_REG(DMA_BASE_PTR,2)
#define DMA_TCD2_NBYTES_MLOFFYES                 DMA_NBYTES_MLOFFYES_REG(DMA_BASE_PTR,2)
#define DMA_TCD2_SLAST                           DMA_SLAST_REG(DMA_BASE_PTR,2)
#define DMA_TCD2_DADDR                           DMA_DADDR_REG(DMA_BASE_PTR,2)
#define DMA_TCD2_DOFF                            DMA_DOFF_REG(DMA_BASE_PTR,2)
#define DMA_TCD2_CITER_ELINKNO                   DMA_CITER_ELINKNO_REG(DMA_BASE_PTR,2)
#define DMA_TCD2_CITER_ELINKYES                  DMA_CITER_ELINKYES_REG(DMA_BASE_PTR,2)
#define DMA_TCD2_DLASTSGA                        DMA_DLAST_SGA_REG(DMA_BASE_PTR,2)
#define DMA_TCD2_CSR                             DMA_CSR_REG(DMA_BASE_PTR,2)
#define DMA_TCD2_BITER_ELINKNO                   DMA_BITER_ELINKNO_REG(DMA_BASE_PTR,2)
#define DMA_TCD2_BITER_ELINKYES                  DMA_BITER_ELINKYES_REG(DMA_BASE_PTR,2)
#define DMA_TCD3_SADDR                           DMA_SADDR_REG(DMA_BASE_PTR,3)
#define DMA_TCD3_SOFF                            DMA_SOFF_REG(DMA_BASE_PTR,3)
#define DMA_TCD3_ATTR                            DMA_ATTR_REG(DMA_BASE_PTR,3)
#define DMA_TCD3_NBYTES_MLNO                     DMA_NBYTES_MLNO_REG(DMA_BASE_PTR,3)
#define DMA_TCD3_NBYTES_MLOFFNO                  DMA_NBYTES_MLOFFNO_REG(DMA_BASE_PTR,3)
#define DMA_TCD3_NBYTES_MLOFFYES                 DMA_NBYTES_MLOFFYES_REG(DMA_BASE_PTR,3)
#define DMA_TCD3_SLAST                           DMA_SLAST_REG(DMA_BASE_PTR,3)
#define DMA_TCD3_DADDR                           DMA_DADDR_REG(DMA_BASE_PTR,3)
#define DMA_TCD3_DOFF                            DMA_DOFF_REG(DMA_BASE_PTR,3)
#define DMA_TCD3_CITER_ELINKNO                   DMA_CITER_ELINKNO_REG(DMA_BASE_PTR,3)
#define DMA_TCD3_CITER_ELINKYES                  DMA_CITER_ELINKYES_REG(DMA_BASE_PTR,3)
#define DMA_TCD3_DLASTSGA                        DMA_DLAST_SGA_REG(DMA_BASE_PTR,3)
#define DMA_TCD3_CSR                             DMA_CSR_REG(DMA_BASE_PTR,3)
#define DMA_TCD3_BITER_ELINKNO                   DMA_BITER_ELINKNO_REG(DMA_BASE_PTR,3)
#define DMA_TCD3_BITER_ELINKYES                  DMA_BITER_ELINKYES_REG(DMA_BASE_PTR,3)
#define DMA_TCD4_SADDR                           DMA_SADDR_REG(DMA_BASE_PTR,4)
#define DMA_TCD4_SOFF                            DMA_SOFF_REG(DMA_BASE_PTR,4)
#define DMA_TCD4_ATTR                            DMA_ATTR_REG(DMA_BASE_PTR,4)
#define DMA_TCD4_NBYTES_MLNO                     DMA_NBYTES_MLNO_REG(DMA_BASE_PTR,4)
#define DMA_TCD4_NBYTES_MLOFFNO                  DMA_NBYTES_MLOFFNO_REG(DMA_BASE_PTR,4)
#define DMA_TCD4_NBYTES_MLOFFYES                 DMA_NBYTES_MLOFFYES_REG(DMA_BASE_PTR,4)
#define DMA_TCD4_SLAST                           DMA_SLAST_REG(DMA_BASE_PTR,4)
#define DMA_TCD4_DADDR                           DMA_DADDR_REG(DMA_BASE_PTR,4)
#define DMA_TCD4_DOFF                            DMA_DOFF_REG(DMA_BASE_PTR,4)
#define DMA_TCD4_CITER_ELINKNO                   DMA_CITER_ELINKNO_REG(DMA_BASE_PTR,4)
#define DMA_TCD4_CITER_ELINKYES                  DMA_CITER_ELINKYES_REG(DMA_BASE_PTR,4)
#define DMA_TCD4_DLASTSGA                        DMA_DLAST_SGA_REG(DMA_BASE_PTR,4)
#define DMA_TCD4_CSR                             DMA_CSR_REG(DMA_BASE_PTR,4)
#define DMA_TCD4_BITER_ELINKNO                   DMA_BITER_ELINKNO_REG(DMA_BASE_PTR,4)
#define DMA_TCD4_BITER_ELINKYES                  DMA_BITER_ELINKYES_REG(DMA_BASE_PTR,4)
#define DMA_TCD5_SADDR                           DMA_SADDR_REG(DMA_BASE_PTR,5)
#define DMA_TCD5_SOFF                            DMA_SOFF_REG(DMA_BASE_PTR,5)
#define DMA_TCD5_ATTR                            DMA_ATTR_REG(DMA_BASE_PTR,5)
#define DMA_TCD5_NBYTES_MLNO                     DMA_NBYTES_MLNO_REG(DMA_BASE_PTR,5)
#define DMA_TCD5_NBYTES_MLOFFNO                  DMA_NBYTES_MLOFFNO_REG(DMA_BASE_PTR,5)
#define DMA_TCD5_NBYTES_MLOFFYES                 DMA_NBYTES_MLOFFYES_REG(DMA_BASE_PTR,5)
#define DMA_TCD5_SLAST                           DMA_SLAST_REG(DMA_BASE_PTR,5)
#define DMA_TCD5_DADDR                           DMA_DADDR_REG(DMA_BASE_PTR,5)
#define DMA_TCD5_DOFF                            DMA_DOFF_REG(DMA_BASE_PTR,5)
#define DMA_TCD5_CITER_ELINKNO                   DMA_CITER_ELINKNO_REG(DMA_BASE_PTR,5)
#define DMA_TCD5_CITER_ELINKYES                  DMA_CITER_ELINKYES_REG(DMA_BASE_PTR,5)
#define DMA_TCD5_DLASTSGA                        DMA_DLAST_SGA_REG(DMA_BASE_PTR,5)
#define DMA_TCD5_CSR                             DMA_CSR_REG(DMA_BASE_PTR,5)
#define DMA_TCD5_BITER_ELINKNO                   DMA_BITER_ELINKNO_REG(DMA_BASE_PTR,5)
#define DMA_TCD5_BITER_ELINKYES                  DMA_BITER_ELINKYES_REG(DMA_BASE_PTR,5)
#define DMA_TCD6_SADDR                           DMA_SADDR_REG(DMA_BASE_PTR,6)
#define DMA_TCD6_SOFF                            DMA_SOFF_REG(DMA_BASE_PTR,6)
#define DMA_TCD6_ATTR                            DMA_ATTR_REG(DMA_BASE_PTR,6)
#define DMA_TCD6_NBYTES_MLNO                     DMA_NBYTES_MLNO_REG(DMA_BASE_PTR,6)
#define DMA_TCD6_NBYTES_MLOFFNO                  DMA_NBYTES_MLOFFNO_REG(DMA_BASE_PTR,6)
#define DMA_TCD6_NBYTES_MLOFFYES                 DMA_NBYTES_MLOFFYES_REG(DMA_BASE_PTR,6)
#define DMA_TCD6_SLAST                           DMA_SLAST_REG(DMA_BASE_PTR,6)
#define DMA_TCD6_DADDR                           DMA_DADDR_REG(DMA_BASE_PTR,6)
#define DMA_TCD6_DOFF                            DMA_DOFF_REG(DMA_BASE_PTR,6)
#define DMA_TCD6_CITER_ELINKNO                   DMA_CITER_ELINKNO_REG(DMA_BASE_PTR,6)
#define DMA_TCD6_CITER_ELINKYES                  DMA_CITER_ELINKYES_REG(DMA_BASE_PTR,6)
#define DMA_TCD6_DLASTSGA                        DMA_DLAST_SGA_REG(DMA_BASE_PTR,6)
#define DMA_TCD6_CSR                             DMA_CSR_REG(DMA_BASE_PTR,6)
#define DMA_TCD6_BITER_ELINKNO                   DMA_BITER_ELINKNO_REG(DMA_BASE_PTR,6)
#define DMA_TCD6_BITER_ELINKYES                  DMA_BITER_ELINKYES_REG(DMA_BASE_PTR,6)
#define DMA_TCD7_SADDR                           DMA_SADDR_REG(DMA_BASE_PTR,7)
#define DMA_TCD7_SOFF                            DMA_SOFF_REG(DMA_BASE_PTR,7)
#define DMA_TCD7_ATTR                            DMA_ATTR_REG(DMA_BASE_PTR,7)
#define DMA_TCD7_NBYTES_MLNO                     DMA_NBYTES_MLNO_REG(DMA_BASE_PTR,7)
#define DMA_TCD7_NBYTES_MLOFFNO                  DMA_NBYTES_MLOFFNO_REG(DMA_BASE_PTR,7)
#define DMA_TCD7_NBYTES_MLOFFYES                 DMA_NBYTES_MLOFFYES_REG(DMA_BASE_PTR,7)
#define DMA_TCD7_SLAST                           DMA_SLAST_REG(DMA_BASE_PTR,7)
#define DMA_TCD7_DADDR                           DMA_DADDR_REG(DMA_BASE_PTR,7)
#define DMA_TCD7_DOFF                            DMA_DOFF_REG(DMA_BASE_PTR,7)
#define DMA_TCD7_CITER_ELINKNO                   DMA_CITER_ELINKNO_REG(DMA_BASE_PTR,7)
#define DMA_TCD7_CITER_ELINKYES                  DMA_CITER_ELINKYES_REG(DMA_BASE_PTR,7)
#define DMA_TCD7_DLASTSGA                        DMA_DLAST_SGA_REG(DMA_BASE_PTR,7)
#define DMA_TCD7_CSR                             DMA_CSR_REG(DMA_BASE_PTR,7)
#define DMA_TCD7_BITER_ELINKNO                   DMA_BITER_ELINKNO_REG(DMA_BASE_PTR,7)
#define DMA_TCD7_BITER_ELINKYES                  DMA_BITER_ELINKYES_REG(DMA_BASE_PTR,7)
#define DMA_TCD8_SADDR                           DMA_SADDR_REG(DMA_BASE_PTR,8)
#define DMA_TCD8_SOFF                            DMA_SOFF_REG(DMA_BASE_PTR,8)
#define DMA_TCD8_ATTR                            DMA_ATTR_REG(DMA_BASE_PTR,8)
#define DMA_TCD8_NBYTES_MLNO                     DMA_NBYTES_MLNO_REG(DMA_BASE_PTR,8)
#define DMA_TCD8_NBYTES_MLOFFNO                  DMA_NBYTES_MLOFFNO_REG(DMA_BASE_PTR,8)
#define DMA_TCD8_NBYTES_MLOFFYES                 DMA_NBYTES_MLOFFYES_REG(DMA_BASE_PTR,8)
#define DMA_TCD8_SLAST                           DMA_SLAST_REG(DMA_BASE_PTR,8)
#define DMA_TCD8_DADDR                           DMA_DADDR_REG(DMA_BASE_PTR,8)
#define DMA_TCD8_DOFF                            DMA_DOFF_REG(DMA_BASE_PTR,8)
#define DMA_TCD8_CITER_ELINKNO                   DMA_CITER_ELINKNO_REG(DMA_BASE_PTR,8)
#define DMA_TCD8_CITER_ELINKYES                  DMA_CITER_ELINKYES_REG(DMA_BASE_PTR,8)
#define DMA_TCD8_DLASTSGA                        DMA_DLAST_SGA_REG(DMA_BASE_PTR,8)
#define DMA_TCD8_CSR                             DMA_CSR_REG(DMA_BASE_PTR,8)
#define DMA_TCD8_BITER_ELINKNO                   DMA_BITER_ELINKNO_REG(DMA_BASE_PTR,8)
#define DMA_TCD8_BITER_ELINKYES                  DMA_BITER_ELINKYES_REG(DMA_BASE_PTR,8)
#define DMA_TCD9_SADDR                           DMA_SADDR_REG(DMA_BASE_PTR,9)
#define DMA_TCD9_SOFF                            DMA_SOFF_REG(DMA_BASE_PTR,9)
#define DMA_TCD9_ATTR                            DMA_ATTR_REG(DMA_BASE_PTR,9)
#define DMA_TCD9_NBYTES_MLNO                     DMA_NBYTES_MLNO_REG(DMA_BASE_PTR,9)
#define DMA_TCD9_NBYTES_MLOFFNO                  DMA_NBYTES_MLOFFNO_REG(DMA_BASE_PTR,9)
#define DMA_TCD9_NBYTES_MLOFFYES                 DMA_NBYTES_MLOFFYES_REG(DMA_BASE_PTR,9)
#define DMA_TCD9_SLAST                           DMA_SLAST_REG(DMA_BASE_PTR,9)
#define DMA_TCD9_DADDR                           DMA_DADDR_REG(DMA_BASE_PTR,9)
#define DMA_TCD9_DOFF                            DMA_DOFF_REG(DMA_BASE_PTR,9)
#define DMA_TCD9_CITER_ELINKNO                   DMA_CITER_ELINKNO_REG(DMA_BASE_PTR,9)
#define DMA_TCD9_CITER_ELINKYES                  DMA_CITER_ELINKYES_REG(DMA_BASE_PTR,9)
#define DMA_TCD9_DLASTSGA                        DMA_DLAST_SGA_REG(DMA_BASE_PTR,9)
#define DMA_TCD9_CSR                             DMA_CSR_REG(DMA_BASE_PTR,9)
#define DMA_TCD9_BITER_ELINKNO                   DMA_BITER_ELINKNO_REG(DMA_BASE_PTR,9)
#define DMA_TCD9_BITER_ELINKYES                  DMA_BITER_ELINKYES_REG(DMA_BASE_PTR,9)
#define DMA_TCD10_SADDR                          DMA_SADDR_REG(DMA_BASE_PTR,10)
#define DMA_TCD10_SOFF                           DMA_SOFF_REG(DMA_BASE_PTR,10)
#define DMA_TCD10_ATTR                           DMA_ATTR_REG(DMA_BASE_PTR,10)
#define DMA_TCD10_NBYTES_MLNO                    DMA_NBYTES_MLNO_REG(DMA_BASE_PTR,10)
#define DMA_TCD10_NBYTES_MLOFFNO                 DMA_NBYTES_MLOFFNO_REG(DMA_BASE_PTR,10)
#define DMA_TCD10_NBYTES_MLOFFYES                DMA_NBYTES_MLOFFYES_REG(DMA_BASE_PTR,10)
#define DMA_TCD10_SLAST                          DMA_SLAST_REG(DMA_BASE_PTR,10)
#define DMA_TCD10_DADDR                          DMA_DADDR_REG(DMA_BASE_PTR,10)
#define DMA_TCD10_DOFF                           DMA_DOFF_REG(DMA_BASE_PTR,10)
#define DMA_TCD10_CITER_ELINKNO                  DMA_CITER_ELINKNO_REG(DMA_BASE_PTR,10)
#define DMA_TCD10_CITER_ELINKYES                 DMA_CITER_ELINKYES_REG(DMA_BASE_PTR,10)
#define DMA_TCD10_DLASTSGA                       DMA_DLAST_SGA_REG(DMA_BASE_PTR,10)
#define DMA_TCD10_CSR                            DMA_CSR_REG(DMA_BASE_PTR,10)
#define DMA_TCD10_BITER_ELINKNO                  DMA_BITER_ELINKNO_REG(DMA_BASE_PTR,10)
#define DMA_TCD10_BITER_ELINKYES                 DMA_BITER_ELINKYES_REG(DMA_BASE_PTR,10)
#define DMA_TCD11_SADDR                          DMA_SADDR_REG(DMA_BASE_PTR,11)
#define DMA_TCD11_SOFF                           DMA_SOFF_REG(DMA_BASE_PTR,11)
#define DMA_TCD11_ATTR                           DMA_ATTR_REG(DMA_BASE_PTR,11)
#define DMA_TCD11_NBYTES_MLNO                    DMA_NBYTES_MLNO_REG(DMA_BASE_PTR,11)
#define DMA_TCD11_NBYTES_MLOFFNO                 DMA_NBYTES_MLOFFNO_REG(DMA_BASE_PTR,11)
#define DMA_TCD11_NBYTES_MLOFFYES                DMA_NBYTES_MLOFFYES_REG(DMA_BASE_PTR,11)
#define DMA_TCD11_SLAST                          DMA_SLAST_REG(DMA_BASE_PTR,11)
#define DMA_TCD11_DADDR                          DMA_DADDR_REG(DMA_BASE_PTR,11)
#define DMA_TCD11_DOFF                           DMA_DOFF_REG(DMA_BASE_PTR,11)
#define DMA_TCD11_CITER_ELINKNO                  DMA_CITER_ELINKNO_REG(DMA_BASE_PTR,11)
#define DMA_TCD11_CITER_ELINKYES                 DMA_CITER_ELINKYES_REG(DMA_BASE_PTR,11)
#define DMA_TCD11_DLASTSGA                       DMA_DLAST_SGA_REG(DMA_BASE_PTR,11)
#define DMA_TCD11_CSR                            DMA_CSR_REG(DMA_BASE_PTR,11)
#define DMA_TCD11_BITER_ELINKNO                  DMA_BITER_ELINKNO_REG(DMA_BASE_PTR,11)
#define DMA_TCD11_BITER_ELINKYES                 DMA_BITER_ELINKYES_REG(DMA_BASE_PTR,11)
#define DMA_TCD12_SADDR                          DMA_SADDR_REG(DMA_BASE_PTR,12)
#define DMA_TCD12_SOFF                           DMA_SOFF_REG(DMA_BASE_PTR,12)
#define DMA_TCD12_ATTR                           DMA_ATTR_REG(DMA_BASE_PTR,12)
#define DMA_TCD12_NBYTES_MLNO                    DMA_NBYTES_MLNO_REG(DMA_BASE_PTR,12)
#define DMA_TCD12_NBYTES_MLOFFNO                 DMA_NBYTES_MLOFFNO_REG(DMA_BASE_PTR,12)
#define DMA_TCD12_NBYTES_MLOFFYES                DMA_NBYTES_MLOFFYES_REG(DMA_BASE_PTR,12)
#define DMA_TCD12_SLAST                          DMA_SLAST_REG(DMA_BASE_PTR,12)
#define DMA_TCD12_DADDR                          DMA_DADDR_REG(DMA_BASE_PTR,12)
#define DMA_TCD12_DOFF                           DMA_DOFF_REG(DMA_BASE_PTR,12)
#define DMA_TCD12_CITER_ELINKNO                  DMA_CITER_ELINKNO_REG(DMA_BASE_PTR,12)
#define DMA_TCD12_CITER_ELINKYES                 DMA_CITER_ELINKYES_REG(DMA_BASE_PTR,12)
#define DMA_TCD12_DLASTSGA                       DMA_DLAST_SGA_REG(DMA_BASE_PTR,12)
#define DMA_TCD12_CSR                            DMA_CSR_REG(DMA_BASE_PTR,12)
#define DMA_TCD12_BITER_ELINKNO                  DMA_BITER_ELINKNO_REG(DMA_BASE_PTR,12)
#define DMA_TCD12_BITER_ELINKYES                 DMA_BITER_ELINKYES_REG(DMA_BASE_PTR,12)
#define DMA_TCD13_SADDR                          DMA_SADDR_REG(DMA_BASE_PTR,13)
#define DMA_TCD13_SOFF                           DMA_SOFF_REG(DMA_BASE_PTR,13)
#define DMA_TCD13_ATTR                           DMA_ATTR_REG(DMA_BASE_PTR,13)
#define DMA_TCD13_NBYTES_MLNO                    DMA_NBYTES_MLNO_REG(DMA_BASE_PTR,13)
#define DMA_TCD13_NBYTES_MLOFFNO                 DMA_NBYTES_MLOFFNO_REG(DMA_BASE_PTR,13)
#define DMA_TCD13_NBYTES_MLOFFYES                DMA_NBYTES_MLOFFYES_REG(DMA_BASE_PTR,13)
#define DMA_TCD13_SLAST                          DMA_SLAST_REG(DMA_BASE_PTR,13)
#define DMA_TCD13_DADDR                          DMA_DADDR_REG(DMA_BASE_PTR,13)
#define DMA_TCD13_DOFF                           DMA_DOFF_REG(DMA_BASE_PTR,13)
#define DMA_TCD13_CITER_ELINKNO                  DMA_CITER_ELINKNO_REG(DMA_BASE_PTR,13)
#define DMA_TCD13_CITER_ELINKYES                 DMA_CITER_ELINKYES_REG(DMA_BASE_PTR,13)
#define DMA_TCD13_DLASTSGA                       DMA_DLAST_SGA_REG(DMA_BASE_PTR,13)
#define DMA_TCD13_CSR                            DMA_CSR_REG(DMA_BASE_PTR,13)
#define DMA_TCD13_BITER_ELINKNO                  DMA_BITER_ELINKNO_REG(DMA_BASE_PTR,13)
#define DMA_TCD13_BITER_ELINKYES                 DMA_BITER_ELINKYES_REG(DMA_BASE_PTR,13)

/* DMA - Register array accessors */
#define DMA_SADDR(index)                         DMA_SADDR_REG(DMA_BASE_PTR,index)
#define DMA_SOFF(index)                          DMA_SOFF_REG(DMA_BASE_PTR,index)
#define DMA_ATTR(index)                          DMA_ATTR_REG(DMA_BASE_PTR,index)
#define DMA_NBYTES_MLNO(index)                   DMA_NBYTES_MLNO_REG(DMA_BASE_PTR,index)
#define DMA_NBYTES_MLOFFNO(index)                DMA_NBYTES_MLOFFNO_REG(DMA_BASE_PTR,index)
#define DMA_NBYTES_MLOFFYES(index)               DMA_NBYTES_MLOFFYES_REG(DMA_BASE_PTR,index)
#define DMA_SLAST(index)                         DMA_SLAST_REG(DMA_BASE_PTR,index)
#define DMA_DADDR(index)                         DMA_DADDR_REG(DMA_BASE_PTR,index)
#define DMA_DOFF(index)                          DMA_DOFF_REG(DMA_BASE_PTR,index)
#define DMA_CITER_ELINKNO(index)                 DMA_CITER_ELINKNO_REG(DMA_BASE_PTR,index)
#define DMA_CITER_ELINKYES(index)                DMA_CITER_ELINKYES_REG(DMA_BASE_PTR,index)
#define DMA_DLAST_SGA(index)                     DMA_DLAST_SGA_REG(DMA_BASE_PTR,index)
#define DMA_CSR(index)                           DMA_CSR_REG(DMA_BASE_PTR,index)
#define DMA_BITER_ELINKNO(index)                 DMA_BITER_ELINKNO_REG(DMA_BASE_PTR,index)
#define DMA_BITER_ELINKYES(index)                DMA_BITER_ELINKYES_REG(DMA_BASE_PTR,index)

/**
 * @}
 */ /* end of group DMA_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group DMA_Peripheral */


/* ----------------------------------------------------------------------------
   -- DMAMUX
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup DMAMUX_Peripheral DMAMUX
 * @{
 */

/** DMAMUX - Peripheral register structure */
typedef struct DMAMUX_MemMap {
  uint8_t CHCFG[16];                               /**< Channel Configuration register, array offset: 0x0, array step: 0x1 */
} volatile *DMAMUX_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- DMAMUX - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup DMAMUX_Register_Accessor_Macros DMAMUX - Register accessor macros
 * @{
 */


/* DMAMUX - Register accessors */
#define DMAMUX_CHCFG_REG(base,index)             ((base)->CHCFG[index])

/**
 * @}
 */ /* end of group DMAMUX_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- DMAMUX Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup DMAMUX_Register_Masks DMAMUX Register Masks
 * @{
 */

/* CHCFG Bit Fields */
#define DMAMUX_CHCFG_SOURCE_MASK                 0x3Fu
#define DMAMUX_CHCFG_SOURCE_SHIFT                0
#define DMAMUX_CHCFG_SOURCE(x)                   (((uint8_t)(((uint8_t)(x))<<DMAMUX_CHCFG_SOURCE_SHIFT))&DMAMUX_CHCFG_SOURCE_MASK)
#define DMAMUX_CHCFG_TRIG_MASK                   0x40u
#define DMAMUX_CHCFG_TRIG_SHIFT                  6
#define DMAMUX_CHCFG_ENBL_MASK                   0x80u
#define DMAMUX_CHCFG_ENBL_SHIFT                  7

/**
 * @}
 */ /* end of group DMAMUX_Register_Masks */


/* DMAMUX - Peripheral instance base addresses */
/** Peripheral DMAMUX base pointer */
#define DMAMUX_BASE_PTR                          ((DMAMUX_MemMapPtr)0x40021000u)

/* ----------------------------------------------------------------------------
   -- DMAMUX - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup DMAMUX_Register_Accessor_Macros DMAMUX - Register accessor macros
 * @{
 */


/* DMAMUX - Register instance definitions */
/* DMAMUX */
#define DMAMUX_CHCFG0                            DMAMUX_CHCFG_REG(DMAMUX_BASE_PTR,0)
#define DMAMUX_CHCFG1                            DMAMUX_CHCFG_REG(DMAMUX_BASE_PTR,1)
#define DMAMUX_CHCFG2                            DMAMUX_CHCFG_REG(DMAMUX_BASE_PTR,2)
#define DMAMUX_CHCFG3                            DMAMUX_CHCFG_REG(DMAMUX_BASE_PTR,3)
#define DMAMUX_CHCFG4                            DMAMUX_CHCFG_REG(DMAMUX_BASE_PTR,4)
#define DMAMUX_CHCFG5                            DMAMUX_CHCFG_REG(DMAMUX_BASE_PTR,5)
#define DMAMUX_CHCFG6                            DMAMUX_CHCFG_REG(DMAMUX_BASE_PTR,6)
#define DMAMUX_CHCFG7                            DMAMUX_CHCFG_REG(DMAMUX_BASE_PTR,7)
#define DMAMUX_CHCFG8                            DMAMUX_CHCFG_REG(DMAMUX_BASE_PTR,8)
#define DMAMUX_CHCFG9                            DMAMUX_CHCFG_REG(DMAMUX_BASE_PTR,9)
#define DMAMUX_CHCFG10                           DMAMUX_CHCFG_REG(DMAMUX_BASE_PTR,10)
#define DMAMUX_CHCFG11                           DMAMUX_CHCFG_REG(DMAMUX_BASE_PTR,11)
#define DMAMUX_CHCFG12                           DMAMUX_CHCFG_REG(DMAMUX_BASE_PTR,12)
#define DMAMUX_CHCFG13                           DMAMUX_CHCFG_REG(DMAMUX_BASE_PTR,13)
#define DMAMUX_CHCFG14                           DMAMUX_CHCFG_REG(DMAMUX_BASE_PTR,14)
#define DMAMUX_CHCFG15                           DMAMUX_CHCFG_REG(DMAMUX_BASE_PTR,15)

/* DMAMUX - Register array accessors */
#define DMAMUX_CHCFG(index)                      DMAMUX_CHCFG_REG(DMAMUX_BASE_PTR,index)

/**
 * @}
 */ /* end of group DMAMUX_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group DMAMUX_Peripheral */


/* ----------------------------------------------------------------------------
   -- DWT
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup DWT_Peripheral DWT
 * @{
 */

/** DWT - Peripheral register structure */
typedef struct DWT_MemMap {
  uint32_t CTRL;                                   /**< Control Register, offset: 0x0 */
  uint32_t CYCCNT;                                 /**< Cycle Count Register, offset: 0x4 */
  uint32_t CPICNT;                                 /**< CPI Count Register, offset: 0x8 */
  uint32_t EXCCNT;                                 /**< Exception Overhead Count Register, offset: 0xC */
  uint32_t SLEEPCNT;                               /**< Sleep Count Register, offset: 0x10 */
  uint32_t LSUCNT;                                 /**< LSU Count Register, offset: 0x14 */
  uint32_t FOLDCNT;                                /**< Folded-instruction Count Register, offset: 0x18 */
  uint32_t PCSR;                                   /**< Program Counter Sample Register, offset: 0x1C */
  struct {                                         /* offset: 0x20, array step: 0x10 */
    uint32_t COMP;                                   /**< Comparator Register 0..Comparator Register 3, array offset: 0x20, array step: 0x10 */
    uint32_t MASK;                                   /**< Mask Register 0..Mask Register 3, array offset: 0x24, array step: 0x10 */
    uint32_t FUNCTION;                               /**< Function Register 0..Function Register 3, array offset: 0x28, array step: 0x10 */
    uint8_t RESERVED_0[4];
  } COMPARATOR[4];
  uint8_t RESERVED_0[3952];
  uint32_t PID4;                                   /**< Peripheral Identification Register 4., offset: 0xFD0 */
  uint32_t PID5;                                   /**< Peripheral Identification Register 5., offset: 0xFD4 */
  uint32_t PID6;                                   /**< Peripheral Identification Register 6., offset: 0xFD8 */
  uint32_t PID7;                                   /**< Peripheral Identification Register 7., offset: 0xFDC */
  uint32_t PID0;                                   /**< Peripheral Identification Register 0., offset: 0xFE0 */
  uint32_t PID1;                                   /**< Peripheral Identification Register 1., offset: 0xFE4 */
  uint32_t PID2;                                   /**< Peripheral Identification Register 2., offset: 0xFE8 */
  uint32_t PID3;                                   /**< Peripheral Identification Register 3., offset: 0xFEC */
  uint32_t CID0;                                   /**< Component Identification Register 0., offset: 0xFF0 */
  uint32_t CID1;                                   /**< Component Identification Register 1., offset: 0xFF4 */
  uint32_t CID2;                                   /**< Component Identification Register 2., offset: 0xFF8 */
  uint32_t CID3;                                   /**< Component Identification Register 3., offset: 0xFFC */
} volatile *DWT_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- DWT - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup DWT_Register_Accessor_Macros DWT - Register accessor macros
 * @{
 */


/* DWT - Register accessors */
#define DWT_CTRL_REG(base)                       ((base)->CTRL)
#define DWT_CYCCNT_REG(base)                     ((base)->CYCCNT)
#define DWT_CPICNT_REG(base)                     ((base)->CPICNT)
#define DWT_EXCCNT_REG(base)                     ((base)->EXCCNT)
#define DWT_SLEEPCNT_REG(base)                   ((base)->SLEEPCNT)
#define DWT_LSUCNT_REG(base)                     ((base)->LSUCNT)
#define DWT_FOLDCNT_REG(base)                    ((base)->FOLDCNT)
#define DWT_PCSR_REG(base)                       ((base)->PCSR)
#define DWT_COMP_REG(base,index)                 ((base)->COMPARATOR[index].COMP)
#define DWT_MASK_REG(base,index)                 ((base)->COMPARATOR[index].MASK)
#define DWT_FUNCTION_REG(base,index)             ((base)->COMPARATOR[index].FUNCTION)
#define DWT_PID4_REG(base)                       ((base)->PID4)
#define DWT_PID5_REG(base)                       ((base)->PID5)
#define DWT_PID6_REG(base)                       ((base)->PID6)
#define DWT_PID7_REG(base)                       ((base)->PID7)
#define DWT_PID0_REG(base)                       ((base)->PID0)
#define DWT_PID1_REG(base)                       ((base)->PID1)
#define DWT_PID2_REG(base)                       ((base)->PID2)
#define DWT_PID3_REG(base)                       ((base)->PID3)
#define DWT_CID0_REG(base)                       ((base)->CID0)
#define DWT_CID1_REG(base)                       ((base)->CID1)
#define DWT_CID2_REG(base)                       ((base)->CID2)
#define DWT_CID3_REG(base)                       ((base)->CID3)

/**
 * @}
 */ /* end of group DWT_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- DWT Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup DWT_Register_Masks DWT Register Masks
 * @{
 */


/**
 * @}
 */ /* end of group DWT_Register_Masks */


/* DWT - Peripheral instance base addresses */
/** Peripheral DWT base pointer */
#define DWT_BASE_PTR                             ((DWT_MemMapPtr)0xE0001000u)

/* ----------------------------------------------------------------------------
   -- DWT - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup DWT_Register_Accessor_Macros DWT - Register accessor macros
 * @{
 */


/* DWT - Register instance definitions */
/* DWT */
#define DWT_CTRL                                 DWT_CTRL_REG(DWT_BASE_PTR)
#define DWT_CYCCNT                               DWT_CYCCNT_REG(DWT_BASE_PTR)
#define DWT_CPICNT                               DWT_CPICNT_REG(DWT_BASE_PTR)
#define DWT_EXCCNT                               DWT_EXCCNT_REG(DWT_BASE_PTR)
#define DWT_SLEEPCNT                             DWT_SLEEPCNT_REG(DWT_BASE_PTR)
#define DWT_LSUCNT                               DWT_LSUCNT_REG(DWT_BASE_PTR)
#define DWT_FOLDCNT                              DWT_FOLDCNT_REG(DWT_BASE_PTR)
#define DWT_PCSR                                 DWT_PCSR_REG(DWT_BASE_PTR)
#define DWT_COMP0                                DWT_COMP_REG(DWT_BASE_PTR,0)
#define DWT_MASK0                                DWT_MASK_REG(DWT_BASE_PTR,0)
#define DWT_FUNCTION0                            DWT_FUNCTION_REG(DWT_BASE_PTR,0)
#define DWT_COMP1                                DWT_COMP_REG(DWT_BASE_PTR,1)
#define DWT_MASK1                                DWT_MASK_REG(DWT_BASE_PTR,1)
#define DWT_FUNCTION1                            DWT_FUNCTION_REG(DWT_BASE_PTR,1)
#define DWT_COMP2                                DWT_COMP_REG(DWT_BASE_PTR,2)
#define DWT_MASK2                                DWT_MASK_REG(DWT_BASE_PTR,2)
#define DWT_FUNCTION2                            DWT_FUNCTION_REG(DWT_BASE_PTR,2)
#define DWT_COMP3                                DWT_COMP_REG(DWT_BASE_PTR,3)
#define DWT_MASK3                                DWT_MASK_REG(DWT_BASE_PTR,3)
#define DWT_FUNCTION3                            DWT_FUNCTION_REG(DWT_BASE_PTR,3)
#define DWT_PID4                                 DWT_PID4_REG(DWT_BASE_PTR)
#define DWT_PID5                                 DWT_PID5_REG(DWT_BASE_PTR)
#define DWT_PID6                                 DWT_PID6_REG(DWT_BASE_PTR)
#define DWT_PID7                                 DWT_PID7_REG(DWT_BASE_PTR)
#define DWT_PID0                                 DWT_PID0_REG(DWT_BASE_PTR)
#define DWT_PID1                                 DWT_PID1_REG(DWT_BASE_PTR)
#define DWT_PID2                                 DWT_PID2_REG(DWT_BASE_PTR)
#define DWT_PID3                                 DWT_PID3_REG(DWT_BASE_PTR)
#define DWT_CID0                                 DWT_CID0_REG(DWT_BASE_PTR)
#define DWT_CID1                                 DWT_CID1_REG(DWT_BASE_PTR)
#define DWT_CID2                                 DWT_CID2_REG(DWT_BASE_PTR)
#define DWT_CID3                                 DWT_CID3_REG(DWT_BASE_PTR)

/* DWT - Register array accessors */
#define DWT_COMP(index)                          DWT_COMP_REG(DWT_BASE_PTR,index)
#define DWT_MASK(index)                          DWT_MASK_REG(DWT_BASE_PTR,index)
#define DWT_FUNCTION(index)                      DWT_FUNCTION_REG(DWT_BASE_PTR,index)

/**
 * @}
 */ /* end of group DWT_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group DWT_Peripheral */


/* ----------------------------------------------------------------------------
   -- ETF
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup ETF_Peripheral ETF
 * @{
 */

/** ETF - Peripheral register structure */
typedef struct ETF_MemMap {
  uint32_t FCR;                                    /**< Funnel Control Register, offset: 0x0 */
  uint32_t PCR;                                    /**< Priority Control Register, offset: 0x4 */
  uint8_t RESERVED_0[3812];
  uint32_t ITATBDATA0;                             /**< Integration Register, ITATBDATA0, offset: 0xEEC */
  uint32_t ITATBCTR2;                              /**< Integration Register, ITATBCTR2, offset: 0xEF0 */
  uint32_t ITATBCTR1;                              /**< Integration Register, ITATBCTR1, offset: 0xEF4 */
  uint32_t ITATBCTR0;                              /**< Integration Register, ITATBCTR0, offset: 0xEF8 */
  uint8_t RESERVED_1[4];
  uint32_t ITCTRL;                                 /**< Integration Mode Control Register, offset: 0xF00 */
  uint8_t RESERVED_2[156];
  uint32_t CLAIMSET;                               /**< Claim Tag Set Register, offset: 0xFA0 */
  uint32_t CLAIMCLR;                               /**< Claim Tag Clear Register, offset: 0xFA4 */
  uint8_t RESERVED_3[8];
  uint32_t LAR;                                    /**< Lock Access Register, offset: 0xFB0 */
  uint32_t LSR;                                    /**< Lock Status Register, offset: 0xFB4 */
  uint32_t AUTHSTATUS;                             /**< Authentication Status Register, offset: 0xFB8 */
  uint8_t RESERVED_4[12];
  uint32_t DEVID;                                  /**< Device ID Register, offset: 0xFC8 */
  uint32_t DEVTYPE;                                /**< Device Type Identifier Register, offset: 0xFCC */
  uint32_t PIDR4;                                  /**< Peripheral Identification Register 4, offset: 0xFD0 */
  uint32_t PIDR5;                                  /**< Peripheral Identification Register 5, offset: 0xFD4 */
  uint32_t PIDR6;                                  /**< Peripheral Identification Register 6, offset: 0xFD8 */
  uint32_t PIDR7;                                  /**< Peripheral Identification Register 7, offset: 0xFDC */
  uint32_t PIDR0;                                  /**< Peripheral Identification Register 0, offset: 0xFE0 */
  uint32_t PIDR1;                                  /**< Peripheral Identification Register 1, offset: 0xFE4 */
  uint32_t PIDR2;                                  /**< Peripheral Identification Register 2, offset: 0xFE8 */
  uint32_t PIDR3;                                  /**< Peripheral Identification Register 3, offset: 0xFEC */
  uint32_t CIDR0;                                  /**< Component Identification Register 0, offset: 0xFF0 */
  uint32_t CIDR1;                                  /**< Component Identification Register 1, offset: 0xFF4 */
  uint32_t CIDR2;                                  /**< Component Identification Register 2, offset: 0xFF8 */
  uint32_t CIDR3;                                  /**< Component Identification Register 3, offset: 0xFFC */
} volatile *ETF_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- ETF - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup ETF_Register_Accessor_Macros ETF - Register accessor macros
 * @{
 */


/* ETF - Register accessors */
#define ETF_FCR_REG(base)                        ((base)->FCR)
#define ETF_PCR_REG(base)                        ((base)->PCR)
#define ETF_ITATBDATA0_REG(base)                 ((base)->ITATBDATA0)
#define ETF_ITATBCTR2_REG(base)                  ((base)->ITATBCTR2)
#define ETF_ITATBCTR1_REG(base)                  ((base)->ITATBCTR1)
#define ETF_ITATBCTR0_REG(base)                  ((base)->ITATBCTR0)
#define ETF_ITCTRL_REG(base)                     ((base)->ITCTRL)
#define ETF_CLAIMSET_REG(base)                   ((base)->CLAIMSET)
#define ETF_CLAIMCLR_REG(base)                   ((base)->CLAIMCLR)
#define ETF_LAR_REG(base)                        ((base)->LAR)
#define ETF_LSR_REG(base)                        ((base)->LSR)
#define ETF_AUTHSTATUS_REG(base)                 ((base)->AUTHSTATUS)
#define ETF_DEVID_REG(base)                      ((base)->DEVID)
#define ETF_DEVTYPE_REG(base)                    ((base)->DEVTYPE)
#define ETF_PIDR4_REG(base)                      ((base)->PIDR4)
#define ETF_PIDR5_REG(base)                      ((base)->PIDR5)
#define ETF_PIDR6_REG(base)                      ((base)->PIDR6)
#define ETF_PIDR7_REG(base)                      ((base)->PIDR7)
#define ETF_PIDR0_REG(base)                      ((base)->PIDR0)
#define ETF_PIDR1_REG(base)                      ((base)->PIDR1)
#define ETF_PIDR2_REG(base)                      ((base)->PIDR2)
#define ETF_PIDR3_REG(base)                      ((base)->PIDR3)
#define ETF_CIDR0_REG(base)                      ((base)->CIDR0)
#define ETF_CIDR1_REG(base)                      ((base)->CIDR1)
#define ETF_CIDR2_REG(base)                      ((base)->CIDR2)
#define ETF_CIDR3_REG(base)                      ((base)->CIDR3)

/**
 * @}
 */ /* end of group ETF_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- ETF Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup ETF_Register_Masks ETF Register Masks
 * @{
 */


/**
 * @}
 */ /* end of group ETF_Register_Masks */


/* ETF - Peripheral instance base addresses */
/** Peripheral ETF base pointer */
#define ETF_BASE_PTR                             ((ETF_MemMapPtr)0xE0043000u)

/* ----------------------------------------------------------------------------
   -- ETF - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup ETF_Register_Accessor_Macros ETF - Register accessor macros
 * @{
 */


/* ETF - Register instance definitions */
/* ETF */
#define ETF_FCR                                  ETF_FCR_REG(ETF_BASE_PTR)
#define ETF_PCR                                  ETF_PCR_REG(ETF_BASE_PTR)
#define ETF_ITATBDATA0                           ETF_ITATBDATA0_REG(ETF_BASE_PTR)
#define ETF_ITATBCTR2                            ETF_ITATBCTR2_REG(ETF_BASE_PTR)
#define ETF_ITATBCTR1                            ETF_ITATBCTR1_REG(ETF_BASE_PTR)
#define ETF_ITATBCTR0                            ETF_ITATBCTR0_REG(ETF_BASE_PTR)
#define ETF_ITCTRL                               ETF_ITCTRL_REG(ETF_BASE_PTR)
#define ETF_CLAIMSET                             ETF_CLAIMSET_REG(ETF_BASE_PTR)
#define ETF_CLAIMCLR                             ETF_CLAIMCLR_REG(ETF_BASE_PTR)
#define ETF_LAR                                  ETF_LAR_REG(ETF_BASE_PTR)
#define ETF_LSR                                  ETF_LSR_REG(ETF_BASE_PTR)
#define ETF_AUTHSTATUS                           ETF_AUTHSTATUS_REG(ETF_BASE_PTR)
#define ETF_DEVID                                ETF_DEVID_REG(ETF_BASE_PTR)
#define ETF_DEVTYPE                              ETF_DEVTYPE_REG(ETF_BASE_PTR)
#define ETF_PIDR4                                ETF_PIDR4_REG(ETF_BASE_PTR)
#define ETF_PIDR5                                ETF_PIDR5_REG(ETF_BASE_PTR)
#define ETF_PIDR6                                ETF_PIDR6_REG(ETF_BASE_PTR)
#define ETF_PIDR7                                ETF_PIDR7_REG(ETF_BASE_PTR)
#define ETF_PIDR0                                ETF_PIDR0_REG(ETF_BASE_PTR)
#define ETF_PIDR1                                ETF_PIDR1_REG(ETF_BASE_PTR)
#define ETF_PIDR2                                ETF_PIDR2_REG(ETF_BASE_PTR)
#define ETF_PIDR3                                ETF_PIDR3_REG(ETF_BASE_PTR)
#define ETF_CIDR0                                ETF_CIDR0_REG(ETF_BASE_PTR)
#define ETF_CIDR1                                ETF_CIDR1_REG(ETF_BASE_PTR)
#define ETF_CIDR2                                ETF_CIDR2_REG(ETF_BASE_PTR)
#define ETF_CIDR3                                ETF_CIDR3_REG(ETF_BASE_PTR)

/**
 * @}
 */ /* end of group ETF_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group ETF_Peripheral */


/* ----------------------------------------------------------------------------
   -- EWM
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup EWM_Peripheral EWM
 * @{
 */

/** EWM - Peripheral register structure */
typedef struct EWM_MemMap {
  uint8_t CTRL;                                    /**< Control Register, offset: 0x0 */
  uint8_t SERV;                                    /**< Service Register, offset: 0x1 */
  uint8_t CMPL;                                    /**< Compare Low Register, offset: 0x2 */
  uint8_t CMPH;                                    /**< Compare High Register, offset: 0x3 */
} volatile *EWM_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- EWM - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup EWM_Register_Accessor_Macros EWM - Register accessor macros
 * @{
 */


/* EWM - Register accessors */
#define EWM_CTRL_REG(base)                       ((base)->CTRL)
#define EWM_SERV_REG(base)                       ((base)->SERV)
#define EWM_CMPL_REG(base)                       ((base)->CMPL)
#define EWM_CMPH_REG(base)                       ((base)->CMPH)

/**
 * @}
 */ /* end of group EWM_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- EWM Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup EWM_Register_Masks EWM Register Masks
 * @{
 */

/* CTRL Bit Fields */
#define EWM_CTRL_EWMEN_MASK                      0x1u
#define EWM_CTRL_EWMEN_SHIFT                     0
#define EWM_CTRL_ASSIN_MASK                      0x2u
#define EWM_CTRL_ASSIN_SHIFT                     1
#define EWM_CTRL_INEN_MASK                       0x4u
#define EWM_CTRL_INEN_SHIFT                      2
/* SERV Bit Fields */
#define EWM_SERV_SERVICE_MASK                    0xFFu
#define EWM_SERV_SERVICE_SHIFT                   0
#define EWM_SERV_SERVICE(x)                      (((uint8_t)(((uint8_t)(x))<<EWM_SERV_SERVICE_SHIFT))&EWM_SERV_SERVICE_MASK)
/* CMPL Bit Fields */
#define EWM_CMPL_COMPAREL_MASK                   0xFFu
#define EWM_CMPL_COMPAREL_SHIFT                  0
#define EWM_CMPL_COMPAREL(x)                     (((uint8_t)(((uint8_t)(x))<<EWM_CMPL_COMPAREL_SHIFT))&EWM_CMPL_COMPAREL_MASK)
/* CMPH Bit Fields */
#define EWM_CMPH_COMPAREH_MASK                   0xFFu
#define EWM_CMPH_COMPAREH_SHIFT                  0
#define EWM_CMPH_COMPAREH(x)                     (((uint8_t)(((uint8_t)(x))<<EWM_CMPH_COMPAREH_SHIFT))&EWM_CMPH_COMPAREH_MASK)

/**
 * @}
 */ /* end of group EWM_Register_Masks */


/* EWM - Peripheral instance base addresses */
/** Peripheral EWM base pointer */
#define EWM_BASE_PTR                             ((EWM_MemMapPtr)0x40041000u)

/* ----------------------------------------------------------------------------
   -- EWM - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup EWM_Register_Accessor_Macros EWM - Register accessor macros
 * @{
 */


/* EWM - Register instance definitions */
/* EWM */
#define EWM_CTRL                                 EWM_CTRL_REG(EWM_BASE_PTR)
#define EWM_SERV                                 EWM_SERV_REG(EWM_BASE_PTR)
#define EWM_CMPL                                 EWM_CMPL_REG(EWM_BASE_PTR)
#define EWM_CMPH                                 EWM_CMPH_REG(EWM_BASE_PTR)

/**
 * @}
 */ /* end of group EWM_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group EWM_Peripheral */


/* ----------------------------------------------------------------------------
   -- FB
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup FB_Peripheral FB
 * @{
 */

/** FB - Peripheral register structure */
typedef struct FB_MemMap {
  struct {                                         /* offset: 0x0, array step: 0xC */
    uint32_t CSAR;                                   /**< Chip Select Address Register, array offset: 0x0, array step: 0xC */
    uint32_t CSMR;                                   /**< Chip Select Mask Register, array offset: 0x4, array step: 0xC */
    uint32_t CSCR;                                   /**< Chip Select Control Register, array offset: 0x8, array step: 0xC */
  } CS[6];
  uint8_t RESERVED_0[24];
  uint32_t CSPMCR;                                 /**< Chip Select port Multiplexing Control Register, offset: 0x60 */
} volatile *FB_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- FB - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup FB_Register_Accessor_Macros FB - Register accessor macros
 * @{
 */


/* FB - Register accessors */
#define FB_CSAR_REG(base,index)                  ((base)->CS[index].CSAR)
#define FB_CSMR_REG(base,index)                  ((base)->CS[index].CSMR)
#define FB_CSCR_REG(base,index)                  ((base)->CS[index].CSCR)
#define FB_CSPMCR_REG(base)                      ((base)->CSPMCR)

/**
 * @}
 */ /* end of group FB_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- FB Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup FB_Register_Masks FB Register Masks
 * @{
 */

/* CSAR Bit Fields */
#define FB_CSAR_BA_MASK                          0xFFFF0000u
#define FB_CSAR_BA_SHIFT                         16
#define FB_CSAR_BA(x)                            (((uint32_t)(((uint32_t)(x))<<FB_CSAR_BA_SHIFT))&FB_CSAR_BA_MASK)
/* CSMR Bit Fields */
#define FB_CSMR_V_MASK                           0x1u
#define FB_CSMR_V_SHIFT                          0
#define FB_CSMR_WP_MASK                          0x100u
#define FB_CSMR_WP_SHIFT                         8
#define FB_CSMR_BAM_MASK                         0xFFFF0000u
#define FB_CSMR_BAM_SHIFT                        16
#define FB_CSMR_BAM(x)                           (((uint32_t)(((uint32_t)(x))<<FB_CSMR_BAM_SHIFT))&FB_CSMR_BAM_MASK)
/* CSCR Bit Fields */
#define FB_CSCR_BSTW_MASK                        0x8u
#define FB_CSCR_BSTW_SHIFT                       3
#define FB_CSCR_BSTR_MASK                        0x10u
#define FB_CSCR_BSTR_SHIFT                       4
#define FB_CSCR_BEM_MASK                         0x20u
#define FB_CSCR_BEM_SHIFT                        5
#define FB_CSCR_PS_MASK                          0xC0u
#define FB_CSCR_PS_SHIFT                         6
#define FB_CSCR_PS(x)                            (((uint32_t)(((uint32_t)(x))<<FB_CSCR_PS_SHIFT))&FB_CSCR_PS_MASK)
#define FB_CSCR_AA_MASK                          0x100u
#define FB_CSCR_AA_SHIFT                         8
#define FB_CSCR_BLS_MASK                         0x200u
#define FB_CSCR_BLS_SHIFT                        9
#define FB_CSCR_WS_MASK                          0xFC00u
#define FB_CSCR_WS_SHIFT                         10
#define FB_CSCR_WS(x)                            (((uint32_t)(((uint32_t)(x))<<FB_CSCR_WS_SHIFT))&FB_CSCR_WS_MASK)
#define FB_CSCR_WRAH_MASK                        0x30000u
#define FB_CSCR_WRAH_SHIFT                       16
#define FB_CSCR_WRAH(x)                          (((uint32_t)(((uint32_t)(x))<<FB_CSCR_WRAH_SHIFT))&FB_CSCR_WRAH_MASK)
#define FB_CSCR_RDAH_MASK                        0xC0000u
#define FB_CSCR_RDAH_SHIFT                       18
#define FB_CSCR_RDAH(x)                          (((uint32_t)(((uint32_t)(x))<<FB_CSCR_RDAH_SHIFT))&FB_CSCR_RDAH_MASK)
#define FB_CSCR_ASET_MASK                        0x300000u
#define FB_CSCR_ASET_SHIFT                       20
#define FB_CSCR_ASET(x)                          (((uint32_t)(((uint32_t)(x))<<FB_CSCR_ASET_SHIFT))&FB_CSCR_ASET_MASK)
#define FB_CSCR_EXTS_MASK                        0x400000u
#define FB_CSCR_EXTS_SHIFT                       22
#define FB_CSCR_SWSEN_MASK                       0x800000u
#define FB_CSCR_SWSEN_SHIFT                      23
#define FB_CSCR_SWS_MASK                         0xFC000000u
#define FB_CSCR_SWS_SHIFT                        26
#define FB_CSCR_SWS(x)                           (((uint32_t)(((uint32_t)(x))<<FB_CSCR_SWS_SHIFT))&FB_CSCR_SWS_MASK)
/* CSPMCR Bit Fields */
#define FB_CSPMCR_GROUP5_MASK                    0xF000u
#define FB_CSPMCR_GROUP5_SHIFT                   12
#define FB_CSPMCR_GROUP5(x)                      (((uint32_t)(((uint32_t)(x))<<FB_CSPMCR_GROUP5_SHIFT))&FB_CSPMCR_GROUP5_MASK)
#define FB_CSPMCR_GROUP4_MASK                    0xF0000u
#define FB_CSPMCR_GROUP4_SHIFT                   16
#define FB_CSPMCR_GROUP4(x)                      (((uint32_t)(((uint32_t)(x))<<FB_CSPMCR_GROUP4_SHIFT))&FB_CSPMCR_GROUP4_MASK)
#define FB_CSPMCR_GROUP3_MASK                    0xF00000u
#define FB_CSPMCR_GROUP3_SHIFT                   20
#define FB_CSPMCR_GROUP3(x)                      (((uint32_t)(((uint32_t)(x))<<FB_CSPMCR_GROUP3_SHIFT))&FB_CSPMCR_GROUP3_MASK)
#define FB_CSPMCR_GROUP2_MASK                    0xF000000u
#define FB_CSPMCR_GROUP2_SHIFT                   24
#define FB_CSPMCR_GROUP2(x)                      (((uint32_t)(((uint32_t)(x))<<FB_CSPMCR_GROUP2_SHIFT))&FB_CSPMCR_GROUP2_MASK)
#define FB_CSPMCR_GROUP1_MASK                    0xF0000000u
#define FB_CSPMCR_GROUP1_SHIFT                   28
#define FB_CSPMCR_GROUP1(x)                      (((uint32_t)(((uint32_t)(x))<<FB_CSPMCR_GROUP1_SHIFT))&FB_CSPMCR_GROUP1_MASK)

/**
 * @}
 */ /* end of group FB_Register_Masks */


/* FB - Peripheral instance base addresses */
/** Peripheral FB base pointer */
#define FB_BASE_PTR                              ((FB_MemMapPtr)0x4000C000u)

/* ----------------------------------------------------------------------------
   -- FB - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup FB_Register_Accessor_Macros FB - Register accessor macros
 * @{
 */


/* FB - Register instance definitions */
/* FB */
#define FB_CSAR0                                 FB_CSAR_REG(FB_BASE_PTR,0)
#define FB_CSMR0                                 FB_CSMR_REG(FB_BASE_PTR,0)
#define FB_CSCR0                                 FB_CSCR_REG(FB_BASE_PTR,0)
#define FB_CSAR1                                 FB_CSAR_REG(FB_BASE_PTR,1)
#define FB_CSMR1                                 FB_CSMR_REG(FB_BASE_PTR,1)
#define FB_CSCR1                                 FB_CSCR_REG(FB_BASE_PTR,1)
#define FB_CSAR2                                 FB_CSAR_REG(FB_BASE_PTR,2)
#define FB_CSMR2                                 FB_CSMR_REG(FB_BASE_PTR,2)
#define FB_CSCR2                                 FB_CSCR_REG(FB_BASE_PTR,2)
#define FB_CSAR3                                 FB_CSAR_REG(FB_BASE_PTR,3)
#define FB_CSMR3                                 FB_CSMR_REG(FB_BASE_PTR,3)
#define FB_CSCR3                                 FB_CSCR_REG(FB_BASE_PTR,3)
#define FB_CSAR4                                 FB_CSAR_REG(FB_BASE_PTR,4)
#define FB_CSMR4                                 FB_CSMR_REG(FB_BASE_PTR,4)
#define FB_CSCR4                                 FB_CSCR_REG(FB_BASE_PTR,4)
#define FB_CSAR5                                 FB_CSAR_REG(FB_BASE_PTR,5)
#define FB_CSMR5                                 FB_CSMR_REG(FB_BASE_PTR,5)
#define FB_CSCR5                                 FB_CSCR_REG(FB_BASE_PTR,5)
#define FB_CSPMCR                                FB_CSPMCR_REG(FB_BASE_PTR)

/* FB - Register array accessors */
#define FB_CSAR(index)                           FB_CSAR_REG(FB_BASE_PTR,index)
#define FB_CSMR(index)                           FB_CSMR_REG(FB_BASE_PTR,index)
#define FB_CSCR(index)                           FB_CSCR_REG(FB_BASE_PTR,index)

/**
 * @}
 */ /* end of group FB_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group FB_Peripheral */


/* ----------------------------------------------------------------------------
   -- FMC
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup FMC_Peripheral FMC
 * @{
 */

/** FMC - Peripheral register structure */
typedef struct FMC_MemMap {
  uint32_t PFAPR;                                  /**< Flash Access Protection Register, offset: 0x0 */
  uint32_t PFB0CR;                                 /**< Flash Bank 0 Control Register, offset: 0x4 */
  uint32_t PFB1CR;                                 /**< Flash Bank 1 Control Register, offset: 0x8 */
  uint8_t RESERVED_0[244];
  uint32_t TAGVD[4][4];                            /**< Cache Tag Storage, array offset: 0x100, array step: index*0x10, index2*0x4 */
  uint8_t RESERVED_1[192];
  struct {                                         /* offset: 0x200, array step: index*0x20, index2*0x8 */
    uint32_t DATA_U;                                 /**< Cache Data Storage (upper word), array offset: 0x200, array step: index*0x20, index2*0x8 */
    uint32_t DATA_L;                                 /**< Cache Data Storage (lower word), array offset: 0x204, array step: index*0x20, index2*0x8 */
  } SET[4][4];
} volatile *FMC_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- FMC - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup FMC_Register_Accessor_Macros FMC - Register accessor macros
 * @{
 */


/* FMC - Register accessors */
#define FMC_PFAPR_REG(base)                      ((base)->PFAPR)
#define FMC_PFB0CR_REG(base)                     ((base)->PFB0CR)
#define FMC_PFB1CR_REG(base)                     ((base)->PFB1CR)
#define FMC_TAGVD_REG(base,index,index2)         ((base)->TAGVD[index][index2])
#define FMC_DATA_U_REG(base,index,index2)        ((base)->SET[index][index2].DATA_U)
#define FMC_DATA_L_REG(base,index,index2)        ((base)->SET[index][index2].DATA_L)

/**
 * @}
 */ /* end of group FMC_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- FMC Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup FMC_Register_Masks FMC Register Masks
 * @{
 */

/* PFAPR Bit Fields */
#define FMC_PFAPR_M0AP_MASK                      0x3u
#define FMC_PFAPR_M0AP_SHIFT                     0
#define FMC_PFAPR_M0AP(x)                        (((uint32_t)(((uint32_t)(x))<<FMC_PFAPR_M0AP_SHIFT))&FMC_PFAPR_M0AP_MASK)
#define FMC_PFAPR_M1AP_MASK                      0xCu
#define FMC_PFAPR_M1AP_SHIFT                     2
#define FMC_PFAPR_M1AP(x)                        (((uint32_t)(((uint32_t)(x))<<FMC_PFAPR_M1AP_SHIFT))&FMC_PFAPR_M1AP_MASK)
#define FMC_PFAPR_M2AP_MASK                      0x30u
#define FMC_PFAPR_M2AP_SHIFT                     4
#define FMC_PFAPR_M2AP(x)                        (((uint32_t)(((uint32_t)(x))<<FMC_PFAPR_M2AP_SHIFT))&FMC_PFAPR_M2AP_MASK)
#define FMC_PFAPR_M3AP_MASK                      0xC0u
#define FMC_PFAPR_M3AP_SHIFT                     6
#define FMC_PFAPR_M3AP(x)                        (((uint32_t)(((uint32_t)(x))<<FMC_PFAPR_M3AP_SHIFT))&FMC_PFAPR_M3AP_MASK)
#define FMC_PFAPR_M0PFD_MASK                     0x10000u
#define FMC_PFAPR_M0PFD_SHIFT                    16
#define FMC_PFAPR_M1PFD_MASK                     0x20000u
#define FMC_PFAPR_M1PFD_SHIFT                    17
#define FMC_PFAPR_M2PFD_MASK                     0x40000u
#define FMC_PFAPR_M2PFD_SHIFT                    18
#define FMC_PFAPR_M3PFD_MASK                     0x80000u
#define FMC_PFAPR_M3PFD_SHIFT                    19
/* PFB0CR Bit Fields */
#define FMC_PFB0CR_B0SEBE_MASK                   0x1u
#define FMC_PFB0CR_B0SEBE_SHIFT                  0
#define FMC_PFB0CR_B0IPE_MASK                    0x2u
#define FMC_PFB0CR_B0IPE_SHIFT                   1
#define FMC_PFB0CR_B0DPE_MASK                    0x4u
#define FMC_PFB0CR_B0DPE_SHIFT                   2
#define FMC_PFB0CR_B0ICE_MASK                    0x8u
#define FMC_PFB0CR_B0ICE_SHIFT                   3
#define FMC_PFB0CR_B0DCE_MASK                    0x10u
#define FMC_PFB0CR_B0DCE_SHIFT                   4
#define FMC_PFB0CR_CRC_MASK                      0xE0u
#define FMC_PFB0CR_CRC_SHIFT                     5
#define FMC_PFB0CR_CRC(x)                        (((uint32_t)(((uint32_t)(x))<<FMC_PFB0CR_CRC_SHIFT))&FMC_PFB0CR_CRC_MASK)
#define FMC_PFB0CR_B0MW_MASK                     0x60000u
#define FMC_PFB0CR_B0MW_SHIFT                    17
#define FMC_PFB0CR_B0MW(x)                       (((uint32_t)(((uint32_t)(x))<<FMC_PFB0CR_B0MW_SHIFT))&FMC_PFB0CR_B0MW_MASK)
#define FMC_PFB0CR_S_B_INV_MASK                  0x80000u
#define FMC_PFB0CR_S_B_INV_SHIFT                 19
#define FMC_PFB0CR_CINV_WAY_MASK                 0xF00000u
#define FMC_PFB0CR_CINV_WAY_SHIFT                20
#define FMC_PFB0CR_CINV_WAY(x)                   (((uint32_t)(((uint32_t)(x))<<FMC_PFB0CR_CINV_WAY_SHIFT))&FMC_PFB0CR_CINV_WAY_MASK)
#define FMC_PFB0CR_CLCK_WAY_MASK                 0xF000000u
#define FMC_PFB0CR_CLCK_WAY_SHIFT                24
#define FMC_PFB0CR_CLCK_WAY(x)                   (((uint32_t)(((uint32_t)(x))<<FMC_PFB0CR_CLCK_WAY_SHIFT))&FMC_PFB0CR_CLCK_WAY_MASK)
#define FMC_PFB0CR_B0RWSC_MASK                   0xF0000000u
#define FMC_PFB0CR_B0RWSC_SHIFT                  28
#define FMC_PFB0CR_B0RWSC(x)                     (((uint32_t)(((uint32_t)(x))<<FMC_PFB0CR_B0RWSC_SHIFT))&FMC_PFB0CR_B0RWSC_MASK)
/* PFB1CR Bit Fields */
#define FMC_PFB1CR_B1MW_MASK                     0x60000u
#define FMC_PFB1CR_B1MW_SHIFT                    17
#define FMC_PFB1CR_B1MW(x)                       (((uint32_t)(((uint32_t)(x))<<FMC_PFB1CR_B1MW_SHIFT))&FMC_PFB1CR_B1MW_MASK)
#define FMC_PFB1CR_B1RWSC_MASK                   0xF0000000u
#define FMC_PFB1CR_B1RWSC_SHIFT                  28
#define FMC_PFB1CR_B1RWSC(x)                     (((uint32_t)(((uint32_t)(x))<<FMC_PFB1CR_B1RWSC_SHIFT))&FMC_PFB1CR_B1RWSC_MASK)
/* TAGVD Bit Fields */
#define FMC_TAGVD_valid_MASK                     0x1u
#define FMC_TAGVD_valid_SHIFT                    0
#define FMC_TAGVD_tag_MASK                       0x7FFC0u
#define FMC_TAGVD_tag_SHIFT                      6
#define FMC_TAGVD_tag(x)                         (((uint32_t)(((uint32_t)(x))<<FMC_TAGVD_tag_SHIFT))&FMC_TAGVD_tag_MASK)
/* DATA_U Bit Fields */
#define FMC_DATA_U_data_MASK                     0xFFFFFFFFu
#define FMC_DATA_U_data_SHIFT                    0
#define FMC_DATA_U_data(x)                       (((uint32_t)(((uint32_t)(x))<<FMC_DATA_U_data_SHIFT))&FMC_DATA_U_data_MASK)
/* DATA_L Bit Fields */
#define FMC_DATA_L_data_MASK                     0xFFFFFFFFu
#define FMC_DATA_L_data_SHIFT                    0
#define FMC_DATA_L_data(x)                       (((uint32_t)(((uint32_t)(x))<<FMC_DATA_L_data_SHIFT))&FMC_DATA_L_data_MASK)

/**
 * @}
 */ /* end of group FMC_Register_Masks */


/* FMC - Peripheral instance base addresses */
/** Peripheral FMC base pointer */
#define FMC_BASE_PTR                             ((FMC_MemMapPtr)0x4001F000u)

/* ----------------------------------------------------------------------------
   -- FMC - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup FMC_Register_Accessor_Macros FMC - Register accessor macros
 * @{
 */


/* FMC - Register instance definitions */
/* FMC */
#define FMC_PFAPR                                FMC_PFAPR_REG(FMC_BASE_PTR)
#define FMC_PFB0CR                               FMC_PFB0CR_REG(FMC_BASE_PTR)
#define FMC_PFB1CR                               FMC_PFB1CR_REG(FMC_BASE_PTR)
#define FMC_TAGVDW0S0                            FMC_TAGVD_REG(FMC_BASE_PTR,0,0)
#define FMC_TAGVDW0S1                            FMC_TAGVD_REG(FMC_BASE_PTR,0,1)
#define FMC_TAGVDW0S2                            FMC_TAGVD_REG(FMC_BASE_PTR,0,2)
#define FMC_TAGVDW0S3                            FMC_TAGVD_REG(FMC_BASE_PTR,0,3)
#define FMC_TAGVDW1S0                            FMC_TAGVD_REG(FMC_BASE_PTR,1,0)
#define FMC_TAGVDW1S1                            FMC_TAGVD_REG(FMC_BASE_PTR,1,1)
#define FMC_TAGVDW1S2                            FMC_TAGVD_REG(FMC_BASE_PTR,1,2)
#define FMC_TAGVDW1S3                            FMC_TAGVD_REG(FMC_BASE_PTR,1,3)
#define FMC_TAGVDW2S0                            FMC_TAGVD_REG(FMC_BASE_PTR,2,0)
#define FMC_TAGVDW2S1                            FMC_TAGVD_REG(FMC_BASE_PTR,2,1)
#define FMC_TAGVDW2S2                            FMC_TAGVD_REG(FMC_BASE_PTR,2,2)
#define FMC_TAGVDW2S3                            FMC_TAGVD_REG(FMC_BASE_PTR,2,3)
#define FMC_TAGVDW3S0                            FMC_TAGVD_REG(FMC_BASE_PTR,3,0)
#define FMC_TAGVDW3S1                            FMC_TAGVD_REG(FMC_BASE_PTR,3,1)
#define FMC_TAGVDW3S2                            FMC_TAGVD_REG(FMC_BASE_PTR,3,2)
#define FMC_TAGVDW3S3                            FMC_TAGVD_REG(FMC_BASE_PTR,3,3)
#define FMC_DATAW0S0U                            FMC_DATA_U_REG(FMC_BASE_PTR,0,0)
#define FMC_DATAW0S0L                            FMC_DATA_L_REG(FMC_BASE_PTR,0,0)
#define FMC_DATAW0S1U                            FMC_DATA_U_REG(FMC_BASE_PTR,0,1)
#define FMC_DATAW0S1L                            FMC_DATA_L_REG(FMC_BASE_PTR,0,1)
#define FMC_DATAW0S2U                            FMC_DATA_U_REG(FMC_BASE_PTR,0,2)
#define FMC_DATAW0S2L                            FMC_DATA_L_REG(FMC_BASE_PTR,0,2)
#define FMC_DATAW0S3U                            FMC_DATA_U_REG(FMC_BASE_PTR,0,3)
#define FMC_DATAW0S3L                            FMC_DATA_L_REG(FMC_BASE_PTR,0,3)
#define FMC_DATAW1S0U                            FMC_DATA_U_REG(FMC_BASE_PTR,1,0)
#define FMC_DATAW1S0L                            FMC_DATA_L_REG(FMC_BASE_PTR,1,0)
#define FMC_DATAW1S1U                            FMC_DATA_U_REG(FMC_BASE_PTR,1,1)
#define FMC_DATAW1S1L                            FMC_DATA_L_REG(FMC_BASE_PTR,1,1)
#define FMC_DATAW1S2U                            FMC_DATA_U_REG(FMC_BASE_PTR,1,2)
#define FMC_DATAW1S2L                            FMC_DATA_L_REG(FMC_BASE_PTR,1,2)
#define FMC_DATAW1S3U                            FMC_DATA_U_REG(FMC_BASE_PTR,1,3)
#define FMC_DATAW1S3L                            FMC_DATA_L_REG(FMC_BASE_PTR,1,3)
#define FMC_DATAW2S0U                            FMC_DATA_U_REG(FMC_BASE_PTR,2,0)
#define FMC_DATAW2S0L                            FMC_DATA_L_REG(FMC_BASE_PTR,2,0)
#define FMC_DATAW2S1U                            FMC_DATA_U_REG(FMC_BASE_PTR,2,1)
#define FMC_DATAW2S1L                            FMC_DATA_L_REG(FMC_BASE_PTR,2,1)
#define FMC_DATAW2S2U                            FMC_DATA_U_REG(FMC_BASE_PTR,2,2)
#define FMC_DATAW2S2L                            FMC_DATA_L_REG(FMC_BASE_PTR,2,2)
#define FMC_DATAW2S3U                            FMC_DATA_U_REG(FMC_BASE_PTR,2,3)
#define FMC_DATAW2S3L                            FMC_DATA_L_REG(FMC_BASE_PTR,2,3)
#define FMC_DATAW3S0U                            FMC_DATA_U_REG(FMC_BASE_PTR,3,0)
#define FMC_DATAW3S0L                            FMC_DATA_L_REG(FMC_BASE_PTR,3,0)
#define FMC_DATAW3S1U                            FMC_DATA_U_REG(FMC_BASE_PTR,3,1)
#define FMC_DATAW3S1L                            FMC_DATA_L_REG(FMC_BASE_PTR,3,1)
#define FMC_DATAW3S2U                            FMC_DATA_U_REG(FMC_BASE_PTR,3,2)
#define FMC_DATAW3S2L                            FMC_DATA_L_REG(FMC_BASE_PTR,3,2)
#define FMC_DATAW3S3U                            FMC_DATA_U_REG(FMC_BASE_PTR,3,3)
#define FMC_DATAW3S3L                            FMC_DATA_L_REG(FMC_BASE_PTR,3,3)

/* FMC - Register array accessors */
#define FMC_TAGVD(index,index2)                  FMC_TAGVD_REG(FMC_BASE_PTR,index,index2)
#define FMC_DATA_U(index,index2)                 FMC_DATA_U_REG(FMC_BASE_PTR,index,index2)
#define FMC_DATA_L(index,index2)                 FMC_DATA_L_REG(FMC_BASE_PTR,index,index2)

/**
 * @}
 */ /* end of group FMC_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group FMC_Peripheral */


/* ----------------------------------------------------------------------------
   -- FPB
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup FPB_Peripheral FPB
 * @{
 */

/** FPB - Peripheral register structure */
typedef struct FPB_MemMap {
  uint32_t CTRL;                                   /**< FlashPatch Control Register, offset: 0x0 */
  uint32_t REMAP;                                  /**< FlashPatch Remap Register, offset: 0x4 */
  uint32_t COMP[8];                                /**< FlashPatch Comparator Register 0..FlashPatch Comparator Register 7, array offset: 0x8, array step: 0x4 */
  uint8_t RESERVED_0[4008];
  uint32_t PID4;                                   /**< Peripheral Identification Register 4., offset: 0xFD0 */
  uint32_t PID5;                                   /**< Peripheral Identification Register 5., offset: 0xFD4 */
  uint32_t PID6;                                   /**< Peripheral Identification Register 6., offset: 0xFD8 */
  uint32_t PID7;                                   /**< Peripheral Identification Register 7., offset: 0xFDC */
  uint32_t PID0;                                   /**< Peripheral Identification Register 0., offset: 0xFE0 */
  uint32_t PID1;                                   /**< Peripheral Identification Register 1., offset: 0xFE4 */
  uint32_t PID2;                                   /**< Peripheral Identification Register 2., offset: 0xFE8 */
  uint32_t PID3;                                   /**< Peripheral Identification Register 3., offset: 0xFEC */
  uint32_t CID0;                                   /**< Component Identification Register 0., offset: 0xFF0 */
  uint32_t CID1;                                   /**< Component Identification Register 1., offset: 0xFF4 */
  uint32_t CID2;                                   /**< Component Identification Register 2., offset: 0xFF8 */
  uint32_t CID3;                                   /**< Component Identification Register 3., offset: 0xFFC */
} volatile *FPB_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- FPB - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup FPB_Register_Accessor_Macros FPB - Register accessor macros
 * @{
 */


/* FPB - Register accessors */
#define FPB_CTRL_REG(base)                       ((base)->CTRL)
#define FPB_REMAP_REG(base)                      ((base)->REMAP)
#define FPB_COMP_REG(base,index)                 ((base)->COMP[index])
#define FPB_PID4_REG(base)                       ((base)->PID4)
#define FPB_PID5_REG(base)                       ((base)->PID5)
#define FPB_PID6_REG(base)                       ((base)->PID6)
#define FPB_PID7_REG(base)                       ((base)->PID7)
#define FPB_PID0_REG(base)                       ((base)->PID0)
#define FPB_PID1_REG(base)                       ((base)->PID1)
#define FPB_PID2_REG(base)                       ((base)->PID2)
#define FPB_PID3_REG(base)                       ((base)->PID3)
#define FPB_CID0_REG(base)                       ((base)->CID0)
#define FPB_CID1_REG(base)                       ((base)->CID1)
#define FPB_CID2_REG(base)                       ((base)->CID2)
#define FPB_CID3_REG(base)                       ((base)->CID3)

/**
 * @}
 */ /* end of group FPB_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- FPB Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup FPB_Register_Masks FPB Register Masks
 * @{
 */


/**
 * @}
 */ /* end of group FPB_Register_Masks */


/* FPB - Peripheral instance base addresses */
/** Peripheral FPB base pointer */
#define FPB_BASE_PTR                             ((FPB_MemMapPtr)0xE0002000u)

/* ----------------------------------------------------------------------------
   -- FPB - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup FPB_Register_Accessor_Macros FPB - Register accessor macros
 * @{
 */


/* FPB - Register instance definitions */
/* FPB */
#define FP_CTRL                                  FPB_CTRL_REG(FPB_BASE_PTR)
#define FP_REMAP                                 FPB_REMAP_REG(FPB_BASE_PTR)
#define FP_COMP0                                 FPB_COMP_REG(FPB_BASE_PTR,0)
#define FP_COMP1                                 FPB_COMP_REG(FPB_BASE_PTR,1)
#define FP_COMP2                                 FPB_COMP_REG(FPB_BASE_PTR,2)
#define FP_COMP3                                 FPB_COMP_REG(FPB_BASE_PTR,3)
#define FP_COMP4                                 FPB_COMP_REG(FPB_BASE_PTR,4)
#define FP_COMP5                                 FPB_COMP_REG(FPB_BASE_PTR,5)
#define FP_COMP6                                 FPB_COMP_REG(FPB_BASE_PTR,6)
#define FP_COMP7                                 FPB_COMP_REG(FPB_BASE_PTR,7)
#define FP_PID4                                  FPB_PID4_REG(FPB_BASE_PTR)
#define FP_PID5                                  FPB_PID5_REG(FPB_BASE_PTR)
#define FP_PID6                                  FPB_PID6_REG(FPB_BASE_PTR)
#define FP_PID7                                  FPB_PID7_REG(FPB_BASE_PTR)
#define FP_PID0                                  FPB_PID0_REG(FPB_BASE_PTR)
#define FP_PID1                                  FPB_PID1_REG(FPB_BASE_PTR)
#define FP_PID2                                  FPB_PID2_REG(FPB_BASE_PTR)
#define FP_PID3                                  FPB_PID3_REG(FPB_BASE_PTR)
#define FP_CID0                                  FPB_CID0_REG(FPB_BASE_PTR)
#define FP_CID1                                  FPB_CID1_REG(FPB_BASE_PTR)
#define FP_CID2                                  FPB_CID2_REG(FPB_BASE_PTR)
#define FP_CID3                                  FPB_CID3_REG(FPB_BASE_PTR)

/* FPB - Register array accessors */
#define FPB_COMP(index)                          FPB_COMP_REG(FPB_BASE_PTR,index)

/**
 * @}
 */ /* end of group FPB_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group FPB_Peripheral */


/* ----------------------------------------------------------------------------
   -- FTM
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup FTM_Peripheral FTM
 * @{
 */

/** FTM - Peripheral register structure */
typedef struct FTM_MemMap {
  uint32_t SC;                                     /**< Status And Control, offset: 0x0 */
  uint32_t CNT;                                    /**< Counter, offset: 0x4 */
  uint32_t MOD;                                    /**< Modulo, offset: 0x8 */
  struct {                                         /* offset: 0xC, array step: 0x8 */
    uint32_t CnSC;                                   /**< Channel (n) Status And Control, array offset: 0xC, array step: 0x8 */
    uint32_t CnV;                                    /**< Channel (n) Value, array offset: 0x10, array step: 0x8 */
  } CONTROLS[6];
  uint8_t RESERVED_0[16];
  uint32_t CNTIN;                                  /**< Counter Initial Value, offset: 0x4C */
  uint32_t STATUS;                                 /**< Capture And Compare Status, offset: 0x50 */
  uint32_t MODE;                                   /**< Features Mode Selection, offset: 0x54 */
  uint32_t SYNC;                                   /**< Synchronization, offset: 0x58 */
  uint32_t OUTINIT;                                /**< Initial State For Channels Output, offset: 0x5C */
  uint32_t OUTMASK;                                /**< Output Mask, offset: 0x60 */
  uint32_t COMBINE;                                /**< Function For Linked Channels, offset: 0x64 */
  uint32_t DEADTIME;                               /**< Deadtime Insertion Control, offset: 0x68 */
  uint32_t EXTTRIG;                                /**< FTM External Trigger, offset: 0x6C */
  uint32_t POL;                                    /**< Channels Polarity, offset: 0x70 */
  uint32_t FMS;                                    /**< Fault Mode Status, offset: 0x74 */
  uint32_t FILTER;                                 /**< Input Capture Filter Control, offset: 0x78 */
  uint32_t FLTCTRL;                                /**< Fault Control, offset: 0x7C */
  uint32_t QDCTRL;                                 /**< Quadrature Decoder Control And Status, offset: 0x80 */
  uint32_t CONF;                                   /**< Configuration, offset: 0x84 */
  uint32_t FLTPOL;                                 /**< FTM Fault Input Polarity, offset: 0x88 */
  uint32_t SYNCONF;                                /**< Synchronization Configuration, offset: 0x8C */
  uint32_t INVCTRL;                                /**< FTM Inverting Control, offset: 0x90 */
  uint32_t SWOCTRL;                                /**< FTM Software Output Control, offset: 0x94 */
  uint32_t PWMLOAD;                                /**< FTM PWM Load, offset: 0x98 */
} volatile *FTM_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- FTM - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup FTM_Register_Accessor_Macros FTM - Register accessor macros
 * @{
 */


/* FTM - Register accessors */
#define FTM_SC_REG(base)                         ((base)->SC)
#define FTM_CNT_REG(base)                        ((base)->CNT)
#define FTM_MOD_REG(base)                        ((base)->MOD)
#define FTM_CnSC_REG(base,index)                 ((base)->CONTROLS[index].CnSC)
#define FTM_CnV_REG(base,index)                  ((base)->CONTROLS[index].CnV)
#define FTM_CNTIN_REG(base)                      ((base)->CNTIN)
#define FTM_STATUS_REG(base)                     ((base)->STATUS)
#define FTM_MODE_REG(base)                       ((base)->MODE)
#define FTM_SYNC_REG(base)                       ((base)->SYNC)
#define FTM_OUTINIT_REG(base)                    ((base)->OUTINIT)
#define FTM_OUTMASK_REG(base)                    ((base)->OUTMASK)
#define FTM_COMBINE_REG(base)                    ((base)->COMBINE)
#define FTM_DEADTIME_REG(base)                   ((base)->DEADTIME)
#define FTM_EXTTRIG_REG(base)                    ((base)->EXTTRIG)
#define FTM_POL_REG(base)                        ((base)->POL)
#define FTM_FMS_REG(base)                        ((base)->FMS)
#define FTM_FILTER_REG(base)                     ((base)->FILTER)
#define FTM_FLTCTRL_REG(base)                    ((base)->FLTCTRL)
#define FTM_QDCTRL_REG(base)                     ((base)->QDCTRL)
#define FTM_CONF_REG(base)                       ((base)->CONF)
#define FTM_FLTPOL_REG(base)                     ((base)->FLTPOL)
#define FTM_SYNCONF_REG(base)                    ((base)->SYNCONF)
#define FTM_INVCTRL_REG(base)                    ((base)->INVCTRL)
#define FTM_SWOCTRL_REG(base)                    ((base)->SWOCTRL)
#define FTM_PWMLOAD_REG(base)                    ((base)->PWMLOAD)

/**
 * @}
 */ /* end of group FTM_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- FTM Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup FTM_Register_Masks FTM Register Masks
 * @{
 */

/* SC Bit Fields */
#define FTM_SC_PS_MASK                           0x7u
#define FTM_SC_PS_SHIFT                          0
#define FTM_SC_PS(x)                             (((uint32_t)(((uint32_t)(x))<<FTM_SC_PS_SHIFT))&FTM_SC_PS_MASK)
#define FTM_SC_CLKS_MASK                         0x18u
#define FTM_SC_CLKS_SHIFT                        3
#define FTM_SC_CLKS(x)                           (((uint32_t)(((uint32_t)(x))<<FTM_SC_CLKS_SHIFT))&FTM_SC_CLKS_MASK)
#define FTM_SC_CPWMS_MASK                        0x20u
#define FTM_SC_CPWMS_SHIFT                       5
#define FTM_SC_TOIE_MASK                         0x40u
#define FTM_SC_TOIE_SHIFT                        6
#define FTM_SC_TOF_MASK                          0x80u
#define FTM_SC_TOF_SHIFT                         7
/* CNT Bit Fields */
#define FTM_CNT_COUNT_MASK                       0xFFFFu
#define FTM_CNT_COUNT_SHIFT                      0
#define FTM_CNT_COUNT(x)                         (((uint32_t)(((uint32_t)(x))<<FTM_CNT_COUNT_SHIFT))&FTM_CNT_COUNT_MASK)
/* MOD Bit Fields */
#define FTM_MOD_MOD_MASK                         0xFFFFu
#define FTM_MOD_MOD_SHIFT                        0
#define FTM_MOD_MOD(x)                           (((uint32_t)(((uint32_t)(x))<<FTM_MOD_MOD_SHIFT))&FTM_MOD_MOD_MASK)
/* CnSC Bit Fields */
#define FTM_CnSC_DMA_MASK                        0x1u
#define FTM_CnSC_DMA_SHIFT                       0
#define FTM_CnSC_ELSA_MASK                       0x4u
#define FTM_CnSC_ELSA_SHIFT                      2
#define FTM_CnSC_ELSB_MASK                       0x8u
#define FTM_CnSC_ELSB_SHIFT                      3
#define FTM_CnSC_MSA_MASK                        0x10u
#define FTM_CnSC_MSA_SHIFT                       4
#define FTM_CnSC_MSB_MASK                        0x20u
#define FTM_CnSC_MSB_SHIFT                       5
#define FTM_CnSC_CHIE_MASK                       0x40u
#define FTM_CnSC_CHIE_SHIFT                      6
#define FTM_CnSC_CHF_MASK                        0x80u
#define FTM_CnSC_CHF_SHIFT                       7
/* CnV Bit Fields */
#define FTM_CnV_VAL_MASK                         0xFFFFu
#define FTM_CnV_VAL_SHIFT                        0
#define FTM_CnV_VAL(x)                           (((uint32_t)(((uint32_t)(x))<<FTM_CnV_VAL_SHIFT))&FTM_CnV_VAL_MASK)
/* CNTIN Bit Fields */
#define FTM_CNTIN_INIT_MASK                      0xFFFFu
#define FTM_CNTIN_INIT_SHIFT                     0
#define FTM_CNTIN_INIT(x)                        (((uint32_t)(((uint32_t)(x))<<FTM_CNTIN_INIT_SHIFT))&FTM_CNTIN_INIT_MASK)
/* STATUS Bit Fields */
#define FTM_STATUS_CH0F_MASK                     0x1u
#define FTM_STATUS_CH0F_SHIFT                    0
#define FTM_STATUS_CH1F_MASK                     0x2u
#define FTM_STATUS_CH1F_SHIFT                    1
#define FTM_STATUS_CH2F_MASK                     0x4u
#define FTM_STATUS_CH2F_SHIFT                    2
#define FTM_STATUS_CH3F_MASK                     0x8u
#define FTM_STATUS_CH3F_SHIFT                    3
#define FTM_STATUS_CH4F_MASK                     0x10u
#define FTM_STATUS_CH4F_SHIFT                    4
#define FTM_STATUS_CH5F_MASK                     0x20u
#define FTM_STATUS_CH5F_SHIFT                    5
#define FTM_STATUS_CH6F_MASK                     0x40u
#define FTM_STATUS_CH6F_SHIFT                    6
#define FTM_STATUS_CH7F_MASK                     0x80u
#define FTM_STATUS_CH7F_SHIFT                    7
/* MODE Bit Fields */
#define FTM_MODE_FTMEN_MASK                      0x1u
#define FTM_MODE_FTMEN_SHIFT                     0
#define FTM_MODE_INIT_MASK                       0x2u
#define FTM_MODE_INIT_SHIFT                      1
#define FTM_MODE_WPDIS_MASK                      0x4u
#define FTM_MODE_WPDIS_SHIFT                     2
#define FTM_MODE_PWMSYNC_MASK                    0x8u
#define FTM_MODE_PWMSYNC_SHIFT                   3
#define FTM_MODE_CAPTEST_MASK                    0x10u
#define FTM_MODE_CAPTEST_SHIFT                   4
#define FTM_MODE_FAULTM_MASK                     0x60u
#define FTM_MODE_FAULTM_SHIFT                    5
#define FTM_MODE_FAULTM(x)                       (((uint32_t)(((uint32_t)(x))<<FTM_MODE_FAULTM_SHIFT))&FTM_MODE_FAULTM_MASK)
#define FTM_MODE_FAULTIE_MASK                    0x80u
#define FTM_MODE_FAULTIE_SHIFT                   7
/* SYNC Bit Fields */
#define FTM_SYNC_CNTMIN_MASK                     0x1u
#define FTM_SYNC_CNTMIN_SHIFT                    0
#define FTM_SYNC_CNTMAX_MASK                     0x2u
#define FTM_SYNC_CNTMAX_SHIFT                    1
#define FTM_SYNC_REINIT_MASK                     0x4u
#define FTM_SYNC_REINIT_SHIFT                    2
#define FTM_SYNC_SYNCHOM_MASK                    0x8u
#define FTM_SYNC_SYNCHOM_SHIFT                   3
#define FTM_SYNC_TRIG0_MASK                      0x10u
#define FTM_SYNC_TRIG0_SHIFT                     4
#define FTM_SYNC_TRIG1_MASK                      0x20u
#define FTM_SYNC_TRIG1_SHIFT                     5
#define FTM_SYNC_TRIG2_MASK                      0x40u
#define FTM_SYNC_TRIG2_SHIFT                     6
#define FTM_SYNC_SWSYNC_MASK                     0x80u
#define FTM_SYNC_SWSYNC_SHIFT                    7
/* OUTINIT Bit Fields */
#define FTM_OUTINIT_CH0OI_MASK                   0x1u
#define FTM_OUTINIT_CH0OI_SHIFT                  0
#define FTM_OUTINIT_CH1OI_MASK                   0x2u
#define FTM_OUTINIT_CH1OI_SHIFT                  1
#define FTM_OUTINIT_CH2OI_MASK                   0x4u
#define FTM_OUTINIT_CH2OI_SHIFT                  2
#define FTM_OUTINIT_CH3OI_MASK                   0x8u
#define FTM_OUTINIT_CH3OI_SHIFT                  3
#define FTM_OUTINIT_CH4OI_MASK                   0x10u
#define FTM_OUTINIT_CH4OI_SHIFT                  4
#define FTM_OUTINIT_CH5OI_MASK                   0x20u
#define FTM_OUTINIT_CH5OI_SHIFT                  5
#define FTM_OUTINIT_CH6OI_MASK                   0x40u
#define FTM_OUTINIT_CH6OI_SHIFT                  6
#define FTM_OUTINIT_CH7OI_MASK                   0x80u
#define FTM_OUTINIT_CH7OI_SHIFT                  7
/* OUTMASK Bit Fields */
#define FTM_OUTMASK_CH0OM_MASK                   0x1u
#define FTM_OUTMASK_CH0OM_SHIFT                  0
#define FTM_OUTMASK_CH1OM_MASK                   0x2u
#define FTM_OUTMASK_CH1OM_SHIFT                  1
#define FTM_OUTMASK_CH2OM_MASK                   0x4u
#define FTM_OUTMASK_CH2OM_SHIFT                  2
#define FTM_OUTMASK_CH3OM_MASK                   0x8u
#define FTM_OUTMASK_CH3OM_SHIFT                  3
#define FTM_OUTMASK_CH4OM_MASK                   0x10u
#define FTM_OUTMASK_CH4OM_SHIFT                  4
#define FTM_OUTMASK_CH5OM_MASK                   0x20u
#define FTM_OUTMASK_CH5OM_SHIFT                  5
#define FTM_OUTMASK_CH6OM_MASK                   0x40u
#define FTM_OUTMASK_CH6OM_SHIFT                  6
#define FTM_OUTMASK_CH7OM_MASK                   0x80u
#define FTM_OUTMASK_CH7OM_SHIFT                  7
/* COMBINE Bit Fields */
#define FTM_COMBINE_COMBINE0_MASK                0x1u
#define FTM_COMBINE_COMBINE0_SHIFT               0
#define FTM_COMBINE_COMP0_MASK                   0x2u
#define FTM_COMBINE_COMP0_SHIFT                  1
#define FTM_COMBINE_DECAPEN0_MASK                0x4u
#define FTM_COMBINE_DECAPEN0_SHIFT               2
#define FTM_COMBINE_DECAP0_MASK                  0x8u
#define FTM_COMBINE_DECAP0_SHIFT                 3
#define FTM_COMBINE_DTEN0_MASK                   0x10u
#define FTM_COMBINE_DTEN0_SHIFT                  4
#define FTM_COMBINE_SYNCEN0_MASK                 0x20u
#define FTM_COMBINE_SYNCEN0_SHIFT                5
#define FTM_COMBINE_FAULTEN0_MASK                0x40u
#define FTM_COMBINE_FAULTEN0_SHIFT               6
#define FTM_COMBINE_COMBINE1_MASK                0x100u
#define FTM_COMBINE_COMBINE1_SHIFT               8
#define FTM_COMBINE_COMP1_MASK                   0x200u
#define FTM_COMBINE_COMP1_SHIFT                  9
#define FTM_COMBINE_DECAPEN1_MASK                0x400u
#define FTM_COMBINE_DECAPEN1_SHIFT               10
#define FTM_COMBINE_DECAP1_MASK                  0x800u
#define FTM_COMBINE_DECAP1_SHIFT                 11
#define FTM_COMBINE_DTEN1_MASK                   0x1000u
#define FTM_COMBINE_DTEN1_SHIFT                  12
#define FTM_COMBINE_SYNCEN1_MASK                 0x2000u
#define FTM_COMBINE_SYNCEN1_SHIFT                13
#define FTM_COMBINE_FAULTEN1_MASK                0x4000u
#define FTM_COMBINE_FAULTEN1_SHIFT               14
#define FTM_COMBINE_COMBINE2_MASK                0x10000u
#define FTM_COMBINE_COMBINE2_SHIFT               16
#define FTM_COMBINE_COMP2_MASK                   0x20000u
#define FTM_COMBINE_COMP2_SHIFT                  17
#define FTM_COMBINE_DECAPEN2_MASK                0x40000u
#define FTM_COMBINE_DECAPEN2_SHIFT               18
#define FTM_COMBINE_DECAP2_MASK                  0x80000u
#define FTM_COMBINE_DECAP2_SHIFT                 19
#define FTM_COMBINE_DTEN2_MASK                   0x100000u
#define FTM_COMBINE_DTEN2_SHIFT                  20
#define FTM_COMBINE_SYNCEN2_MASK                 0x200000u
#define FTM_COMBINE_SYNCEN2_SHIFT                21
#define FTM_COMBINE_FAULTEN2_MASK                0x400000u
#define FTM_COMBINE_FAULTEN2_SHIFT               22
#define FTM_COMBINE_COMBINE3_MASK                0x1000000u
#define FTM_COMBINE_COMBINE3_SHIFT               24
#define FTM_COMBINE_COMP3_MASK                   0x2000000u
#define FTM_COMBINE_COMP3_SHIFT                  25
#define FTM_COMBINE_DECAPEN3_MASK                0x4000000u
#define FTM_COMBINE_DECAPEN3_SHIFT               26
#define FTM_COMBINE_DECAP3_MASK                  0x8000000u
#define FTM_COMBINE_DECAP3_SHIFT                 27
#define FTM_COMBINE_DTEN3_MASK                   0x10000000u
#define FTM_COMBINE_DTEN3_SHIFT                  28
#define FTM_COMBINE_SYNCEN3_MASK                 0x20000000u
#define FTM_COMBINE_SYNCEN3_SHIFT                29
#define FTM_COMBINE_FAULTEN3_MASK                0x40000000u
#define FTM_COMBINE_FAULTEN3_SHIFT               30
/* DEADTIME Bit Fields */
#define FTM_DEADTIME_DTVAL_MASK                  0x3Fu
#define FTM_DEADTIME_DTVAL_SHIFT                 0
#define FTM_DEADTIME_DTVAL(x)                    (((uint32_t)(((uint32_t)(x))<<FTM_DEADTIME_DTVAL_SHIFT))&FTM_DEADTIME_DTVAL_MASK)
#define FTM_DEADTIME_DTPS_MASK                   0xC0u
#define FTM_DEADTIME_DTPS_SHIFT                  6
#define FTM_DEADTIME_DTPS(x)                     (((uint32_t)(((uint32_t)(x))<<FTM_DEADTIME_DTPS_SHIFT))&FTM_DEADTIME_DTPS_MASK)
/* EXTTRIG Bit Fields */
#define FTM_EXTTRIG_CH2TRIG_MASK                 0x1u
#define FTM_EXTTRIG_CH2TRIG_SHIFT                0
#define FTM_EXTTRIG_CH3TRIG_MASK                 0x2u
#define FTM_EXTTRIG_CH3TRIG_SHIFT                1
#define FTM_EXTTRIG_CH4TRIG_MASK                 0x4u
#define FTM_EXTTRIG_CH4TRIG_SHIFT                2
#define FTM_EXTTRIG_CH5TRIG_MASK                 0x8u
#define FTM_EXTTRIG_CH5TRIG_SHIFT                3
#define FTM_EXTTRIG_CH0TRIG_MASK                 0x10u
#define FTM_EXTTRIG_CH0TRIG_SHIFT                4
#define FTM_EXTTRIG_CH1TRIG_MASK                 0x20u
#define FTM_EXTTRIG_CH1TRIG_SHIFT                5
#define FTM_EXTTRIG_INITTRIGEN_MASK              0x40u
#define FTM_EXTTRIG_INITTRIGEN_SHIFT             6
#define FTM_EXTTRIG_TRIGF_MASK                   0x80u
#define FTM_EXTTRIG_TRIGF_SHIFT                  7
/* POL Bit Fields */
#define FTM_POL_POL0_MASK                        0x1u
#define FTM_POL_POL0_SHIFT                       0
#define FTM_POL_POL1_MASK                        0x2u
#define FTM_POL_POL1_SHIFT                       1
#define FTM_POL_POL2_MASK                        0x4u
#define FTM_POL_POL2_SHIFT                       2
#define FTM_POL_POL3_MASK                        0x8u
#define FTM_POL_POL3_SHIFT                       3
#define FTM_POL_POL4_MASK                        0x10u
#define FTM_POL_POL4_SHIFT                       4
#define FTM_POL_POL5_MASK                        0x20u
#define FTM_POL_POL5_SHIFT                       5
#define FTM_POL_POL6_MASK                        0x40u
#define FTM_POL_POL6_SHIFT                       6
#define FTM_POL_POL7_MASK                        0x80u
#define FTM_POL_POL7_SHIFT                       7
/* FMS Bit Fields */
#define FTM_FMS_FAULTF0_MASK                     0x1u
#define FTM_FMS_FAULTF0_SHIFT                    0
#define FTM_FMS_FAULTF1_MASK                     0x2u
#define FTM_FMS_FAULTF1_SHIFT                    1
#define FTM_FMS_FAULTF2_MASK                     0x4u
#define FTM_FMS_FAULTF2_SHIFT                    2
#define FTM_FMS_FAULTF3_MASK                     0x8u
#define FTM_FMS_FAULTF3_SHIFT                    3
#define FTM_FMS_FAULTIN_MASK                     0x20u
#define FTM_FMS_FAULTIN_SHIFT                    5
#define FTM_FMS_WPEN_MASK                        0x40u
#define FTM_FMS_WPEN_SHIFT                       6
#define FTM_FMS_FAULTF_MASK                      0x80u
#define FTM_FMS_FAULTF_SHIFT                     7
/* FILTER Bit Fields */
#define FTM_FILTER_CH0FVAL_MASK                  0xFu
#define FTM_FILTER_CH0FVAL_SHIFT                 0
#define FTM_FILTER_CH0FVAL(x)                    (((uint32_t)(((uint32_t)(x))<<FTM_FILTER_CH0FVAL_SHIFT))&FTM_FILTER_CH0FVAL_MASK)
#define FTM_FILTER_CH1FVAL_MASK                  0xF0u
#define FTM_FILTER_CH1FVAL_SHIFT                 4
#define FTM_FILTER_CH1FVAL(x)                    (((uint32_t)(((uint32_t)(x))<<FTM_FILTER_CH1FVAL_SHIFT))&FTM_FILTER_CH1FVAL_MASK)
#define FTM_FILTER_CH2FVAL_MASK                  0xF00u
#define FTM_FILTER_CH2FVAL_SHIFT                 8
#define FTM_FILTER_CH2FVAL(x)                    (((uint32_t)(((uint32_t)(x))<<FTM_FILTER_CH2FVAL_SHIFT))&FTM_FILTER_CH2FVAL_MASK)
#define FTM_FILTER_CH3FVAL_MASK                  0xF000u
#define FTM_FILTER_CH3FVAL_SHIFT                 12
#define FTM_FILTER_CH3FVAL(x)                    (((uint32_t)(((uint32_t)(x))<<FTM_FILTER_CH3FVAL_SHIFT))&FTM_FILTER_CH3FVAL_MASK)
/* FLTCTRL Bit Fields */
#define FTM_FLTCTRL_FAULT0EN_MASK                0x1u
#define FTM_FLTCTRL_FAULT0EN_SHIFT               0
#define FTM_FLTCTRL_FAULT1EN_MASK                0x2u
#define FTM_FLTCTRL_FAULT1EN_SHIFT               1
#define FTM_FLTCTRL_FAULT2EN_MASK                0x4u
#define FTM_FLTCTRL_FAULT2EN_SHIFT               2
#define FTM_FLTCTRL_FAULT3EN_MASK                0x8u
#define FTM_FLTCTRL_FAULT3EN_SHIFT               3
#define FTM_FLTCTRL_FFLTR0EN_MASK                0x10u
#define FTM_FLTCTRL_FFLTR0EN_SHIFT               4
#define FTM_FLTCTRL_FFLTR1EN_MASK                0x20u
#define FTM_FLTCTRL_FFLTR1EN_SHIFT               5
#define FTM_FLTCTRL_FFLTR2EN_MASK                0x40u
#define FTM_FLTCTRL_FFLTR2EN_SHIFT               6
#define FTM_FLTCTRL_FFLTR3EN_MASK                0x80u
#define FTM_FLTCTRL_FFLTR3EN_SHIFT               7
#define FTM_FLTCTRL_FFVAL_MASK                   0xF00u
#define FTM_FLTCTRL_FFVAL_SHIFT                  8
#define FTM_FLTCTRL_FFVAL(x)                     (((uint32_t)(((uint32_t)(x))<<FTM_FLTCTRL_FFVAL_SHIFT))&FTM_FLTCTRL_FFVAL_MASK)
/* QDCTRL Bit Fields */
#define FTM_QDCTRL_QUADEN_MASK                   0x1u
#define FTM_QDCTRL_QUADEN_SHIFT                  0
#define FTM_QDCTRL_TOFDIR_MASK                   0x2u
#define FTM_QDCTRL_TOFDIR_SHIFT                  1
#define FTM_QDCTRL_QUADIR_MASK                   0x4u
#define FTM_QDCTRL_QUADIR_SHIFT                  2
#define FTM_QDCTRL_QUADMODE_MASK                 0x8u
#define FTM_QDCTRL_QUADMODE_SHIFT                3
#define FTM_QDCTRL_PHBPOL_MASK                   0x10u
#define FTM_QDCTRL_PHBPOL_SHIFT                  4
#define FTM_QDCTRL_PHAPOL_MASK                   0x20u
#define FTM_QDCTRL_PHAPOL_SHIFT                  5
#define FTM_QDCTRL_PHBFLTREN_MASK                0x40u
#define FTM_QDCTRL_PHBFLTREN_SHIFT               6
#define FTM_QDCTRL_PHAFLTREN_MASK                0x80u
#define FTM_QDCTRL_PHAFLTREN_SHIFT               7
/* CONF Bit Fields */
#define FTM_CONF_NUMTOF_MASK                     0x1Fu
#define FTM_CONF_NUMTOF_SHIFT                    0
#define FTM_CONF_NUMTOF(x)                       (((uint32_t)(((uint32_t)(x))<<FTM_CONF_NUMTOF_SHIFT))&FTM_CONF_NUMTOF_MASK)
#define FTM_CONF_BDMMODE_MASK                    0xC0u
#define FTM_CONF_BDMMODE_SHIFT                   6
#define FTM_CONF_BDMMODE(x)                      (((uint32_t)(((uint32_t)(x))<<FTM_CONF_BDMMODE_SHIFT))&FTM_CONF_BDMMODE_MASK)
#define FTM_CONF_GTBEEN_MASK                     0x200u
#define FTM_CONF_GTBEEN_SHIFT                    9
#define FTM_CONF_GTBEOUT_MASK                    0x400u
#define FTM_CONF_GTBEOUT_SHIFT                   10
/* FLTPOL Bit Fields */
#define FTM_FLTPOL_FLT0POL_MASK                  0x1u
#define FTM_FLTPOL_FLT0POL_SHIFT                 0
#define FTM_FLTPOL_FLT1POL_MASK                  0x2u
#define FTM_FLTPOL_FLT1POL_SHIFT                 1
#define FTM_FLTPOL_FLT2POL_MASK                  0x4u
#define FTM_FLTPOL_FLT2POL_SHIFT                 2
#define FTM_FLTPOL_FLT3POL_MASK                  0x8u
#define FTM_FLTPOL_FLT3POL_SHIFT                 3
/* SYNCONF Bit Fields */
#define FTM_SYNCONF_HWTRIGMODE_MASK              0x1u
#define FTM_SYNCONF_HWTRIGMODE_SHIFT             0
#define FTM_SYNCONF_CNTINC_MASK                  0x4u
#define FTM_SYNCONF_CNTINC_SHIFT                 2
#define FTM_SYNCONF_INVC_MASK                    0x10u
#define FTM_SYNCONF_INVC_SHIFT                   4
#define FTM_SYNCONF_SWOC_MASK                    0x20u
#define FTM_SYNCONF_SWOC_SHIFT                   5
#define FTM_SYNCONF_SYNCMODE_MASK                0x80u
#define FTM_SYNCONF_SYNCMODE_SHIFT               7
#define FTM_SYNCONF_SWRSTCNT_MASK                0x100u
#define FTM_SYNCONF_SWRSTCNT_SHIFT               8
#define FTM_SYNCONF_SWWRBUF_MASK                 0x200u
#define FTM_SYNCONF_SWWRBUF_SHIFT                9
#define FTM_SYNCONF_SWOM_MASK                    0x400u
#define FTM_SYNCONF_SWOM_SHIFT                   10
#define FTM_SYNCONF_SWINVC_MASK                  0x800u
#define FTM_SYNCONF_SWINVC_SHIFT                 11
#define FTM_SYNCONF_SWSOC_MASK                   0x1000u
#define FTM_SYNCONF_SWSOC_SHIFT                  12
#define FTM_SYNCONF_HWRSTCNT_MASK                0x10000u
#define FTM_SYNCONF_HWRSTCNT_SHIFT               16
#define FTM_SYNCONF_HWWRBUF_MASK                 0x20000u
#define FTM_SYNCONF_HWWRBUF_SHIFT                17
#define FTM_SYNCONF_HWOM_MASK                    0x40000u
#define FTM_SYNCONF_HWOM_SHIFT                   18
#define FTM_SYNCONF_HWINVC_MASK                  0x80000u
#define FTM_SYNCONF_HWINVC_SHIFT                 19
#define FTM_SYNCONF_HWSOC_MASK                   0x100000u
#define FTM_SYNCONF_HWSOC_SHIFT                  20
/* INVCTRL Bit Fields */
#define FTM_INVCTRL_INV0EN_MASK                  0x1u
#define FTM_INVCTRL_INV0EN_SHIFT                 0
#define FTM_INVCTRL_INV1EN_MASK                  0x2u
#define FTM_INVCTRL_INV1EN_SHIFT                 1
#define FTM_INVCTRL_INV2EN_MASK                  0x4u
#define FTM_INVCTRL_INV2EN_SHIFT                 2
#define FTM_INVCTRL_INV3EN_MASK                  0x8u
#define FTM_INVCTRL_INV3EN_SHIFT                 3
/* SWOCTRL Bit Fields */
#define FTM_SWOCTRL_CH0OC_MASK                   0x1u
#define FTM_SWOCTRL_CH0OC_SHIFT                  0
#define FTM_SWOCTRL_CH1OC_MASK                   0x2u
#define FTM_SWOCTRL_CH1OC_SHIFT                  1
#define FTM_SWOCTRL_CH2OC_MASK                   0x4u
#define FTM_SWOCTRL_CH2OC_SHIFT                  2
#define FTM_SWOCTRL_CH3OC_MASK                   0x8u
#define FTM_SWOCTRL_CH3OC_SHIFT                  3
#define FTM_SWOCTRL_CH4OC_MASK                   0x10u
#define FTM_SWOCTRL_CH4OC_SHIFT                  4
#define FTM_SWOCTRL_CH5OC_MASK                   0x20u
#define FTM_SWOCTRL_CH5OC_SHIFT                  5
#define FTM_SWOCTRL_CH6OC_MASK                   0x40u
#define FTM_SWOCTRL_CH6OC_SHIFT                  6
#define FTM_SWOCTRL_CH7OC_MASK                   0x80u
#define FTM_SWOCTRL_CH7OC_SHIFT                  7
#define FTM_SWOCTRL_CH0OCV_MASK                  0x100u
#define FTM_SWOCTRL_CH0OCV_SHIFT                 8
#define FTM_SWOCTRL_CH1OCV_MASK                  0x200u
#define FTM_SWOCTRL_CH1OCV_SHIFT                 9
#define FTM_SWOCTRL_CH2OCV_MASK                  0x400u
#define FTM_SWOCTRL_CH2OCV_SHIFT                 10
#define FTM_SWOCTRL_CH3OCV_MASK                  0x800u
#define FTM_SWOCTRL_CH3OCV_SHIFT                 11
#define FTM_SWOCTRL_CH4OCV_MASK                  0x1000u
#define FTM_SWOCTRL_CH4OCV_SHIFT                 12
#define FTM_SWOCTRL_CH5OCV_MASK                  0x2000u
#define FTM_SWOCTRL_CH5OCV_SHIFT                 13
#define FTM_SWOCTRL_CH6OCV_MASK                  0x4000u
#define FTM_SWOCTRL_CH6OCV_SHIFT                 14
#define FTM_SWOCTRL_CH7OCV_MASK                  0x8000u
#define FTM_SWOCTRL_CH7OCV_SHIFT                 15
/* PWMLOAD Bit Fields */
#define FTM_PWMLOAD_CH0SEL_MASK                  0x1u
#define FTM_PWMLOAD_CH0SEL_SHIFT                 0
#define FTM_PWMLOAD_CH1SEL_MASK                  0x2u
#define FTM_PWMLOAD_CH1SEL_SHIFT                 1
#define FTM_PWMLOAD_CH2SEL_MASK                  0x4u
#define FTM_PWMLOAD_CH2SEL_SHIFT                 2
#define FTM_PWMLOAD_CH3SEL_MASK                  0x8u
#define FTM_PWMLOAD_CH3SEL_SHIFT                 3
#define FTM_PWMLOAD_CH4SEL_MASK                  0x10u
#define FTM_PWMLOAD_CH4SEL_SHIFT                 4
#define FTM_PWMLOAD_CH5SEL_MASK                  0x20u
#define FTM_PWMLOAD_CH5SEL_SHIFT                 5
#define FTM_PWMLOAD_CH6SEL_MASK                  0x40u
#define FTM_PWMLOAD_CH6SEL_SHIFT                 6
#define FTM_PWMLOAD_CH7SEL_MASK                  0x80u
#define FTM_PWMLOAD_CH7SEL_SHIFT                 7
#define FTM_PWMLOAD_LDOK_MASK                    0x200u
#define FTM_PWMLOAD_LDOK_SHIFT                   9

/**
 * @}
 */ /* end of group FTM_Register_Masks */


/* FTM - Peripheral instance base addresses */
/** Peripheral FTM0 base pointer */
#define FTM0_BASE_PTR                            ((FTM_MemMapPtr)0x40029000u)
/** Peripheral FTM1 base pointer */
#define FTM1_BASE_PTR                            ((FTM_MemMapPtr)0x4002A000u)
/** Peripheral FTM2 base pointer */
#define FTM2_BASE_PTR                            ((FTM_MemMapPtr)0x4002B000u)
/** Peripheral FTM3 base pointer */
#define FTM3_BASE_PTR                            ((FTM_MemMapPtr)0x4002C000u)

/* ----------------------------------------------------------------------------
   -- FTM - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup FTM_Register_Accessor_Macros FTM - Register accessor macros
 * @{
 */


/* FTM - Register instance definitions */
/* FTM0 */
#define FTM0_SC                                  FTM_SC_REG(FTM0_BASE_PTR)
#define FTM0_CNT                                 FTM_CNT_REG(FTM0_BASE_PTR)
#define FTM0_MOD                                 FTM_MOD_REG(FTM0_BASE_PTR)
#define FTM0_C0SC                                FTM_CnSC_REG(FTM0_BASE_PTR,0)
#define FTM0_C0V                                 FTM_CnV_REG(FTM0_BASE_PTR,0)
#define FTM0_C1SC                                FTM_CnSC_REG(FTM0_BASE_PTR,1)
#define FTM0_C1V                                 FTM_CnV_REG(FTM0_BASE_PTR,1)
#define FTM0_C2SC                                FTM_CnSC_REG(FTM0_BASE_PTR,2)
#define FTM0_C2V                                 FTM_CnV_REG(FTM0_BASE_PTR,2)
#define FTM0_C3SC                                FTM_CnSC_REG(FTM0_BASE_PTR,3)
#define FTM0_C3V                                 FTM_CnV_REG(FTM0_BASE_PTR,3)
#define FTM0_C4SC                                FTM_CnSC_REG(FTM0_BASE_PTR,4)
#define FTM0_C4V                                 FTM_CnV_REG(FTM0_BASE_PTR,4)
#define FTM0_C5SC                                FTM_CnSC_REG(FTM0_BASE_PTR,5)
#define FTM0_C5V                                 FTM_CnV_REG(FTM0_BASE_PTR,5)
#define FTM0_CNTIN                               FTM_CNTIN_REG(FTM0_BASE_PTR)
#define FTM0_STATUS                              FTM_STATUS_REG(FTM0_BASE_PTR)
#define FTM0_MODE                                FTM_MODE_REG(FTM0_BASE_PTR)
#define FTM0_SYNC                                FTM_SYNC_REG(FTM0_BASE_PTR)
#define FTM0_OUTINIT                             FTM_OUTINIT_REG(FTM0_BASE_PTR)
#define FTM0_OUTMASK                             FTM_OUTMASK_REG(FTM0_BASE_PTR)
#define FTM0_COMBINE                             FTM_COMBINE_REG(FTM0_BASE_PTR)
#define FTM0_DEADTIME                            FTM_DEADTIME_REG(FTM0_BASE_PTR)
#define FTM0_EXTTRIG                             FTM_EXTTRIG_REG(FTM0_BASE_PTR)
#define FTM0_POL                                 FTM_POL_REG(FTM0_BASE_PTR)
#define FTM0_FMS                                 FTM_FMS_REG(FTM0_BASE_PTR)
#define FTM0_FILTER                              FTM_FILTER_REG(FTM0_BASE_PTR)
#define FTM0_FLTCTRL                             FTM_FLTCTRL_REG(FTM0_BASE_PTR)
#define FTM0_QDCTRL                              FTM_QDCTRL_REG(FTM0_BASE_PTR)
#define FTM0_CONF                                FTM_CONF_REG(FTM0_BASE_PTR)
#define FTM0_FLTPOL                              FTM_FLTPOL_REG(FTM0_BASE_PTR)
#define FTM0_SYNCONF                             FTM_SYNCONF_REG(FTM0_BASE_PTR)
#define FTM0_INVCTRL                             FTM_INVCTRL_REG(FTM0_BASE_PTR)
#define FTM0_SWOCTRL                             FTM_SWOCTRL_REG(FTM0_BASE_PTR)
#define FTM0_PWMLOAD                             FTM_PWMLOAD_REG(FTM0_BASE_PTR)
/* FTM1 */
#define FTM1_SC                                  FTM_SC_REG(FTM1_BASE_PTR)
#define FTM1_CNT                                 FTM_CNT_REG(FTM1_BASE_PTR)
#define FTM1_MOD                                 FTM_MOD_REG(FTM1_BASE_PTR)
#define FTM1_C0SC                                FTM_CnSC_REG(FTM1_BASE_PTR,0)
#define FTM1_C0V                                 FTM_CnV_REG(FTM1_BASE_PTR,0)
#define FTM1_C1SC                                FTM_CnSC_REG(FTM1_BASE_PTR,1)
#define FTM1_C1V                                 FTM_CnV_REG(FTM1_BASE_PTR,1)
#define FTM1_CNTIN                               FTM_CNTIN_REG(FTM1_BASE_PTR)
#define FTM1_STATUS                              FTM_STATUS_REG(FTM1_BASE_PTR)
#define FTM1_MODE                                FTM_MODE_REG(FTM1_BASE_PTR)
#define FTM1_SYNC                                FTM_SYNC_REG(FTM1_BASE_PTR)
#define FTM1_OUTINIT                             FTM_OUTINIT_REG(FTM1_BASE_PTR)
#define FTM1_OUTMASK                             FTM_OUTMASK_REG(FTM1_BASE_PTR)
#define FTM1_COMBINE                             FTM_COMBINE_REG(FTM1_BASE_PTR)
#define FTM1_DEADTIME                            FTM_DEADTIME_REG(FTM1_BASE_PTR)
#define FTM1_EXTTRIG                             FTM_EXTTRIG_REG(FTM1_BASE_PTR)
#define FTM1_POL                                 FTM_POL_REG(FTM1_BASE_PTR)
#define FTM1_FMS                                 FTM_FMS_REG(FTM1_BASE_PTR)
#define FTM1_FILTER                              FTM_FILTER_REG(FTM1_BASE_PTR)
#define FTM1_FLTCTRL                             FTM_FLTCTRL_REG(FTM1_BASE_PTR)
#define FTM1_QDCTRL                              FTM_QDCTRL_REG(FTM1_BASE_PTR)
#define FTM1_CONF                                FTM_CONF_REG(FTM1_BASE_PTR)
#define FTM1_FLTPOL                              FTM_FLTPOL_REG(FTM1_BASE_PTR)
#define FTM1_SYNCONF                             FTM_SYNCONF_REG(FTM1_BASE_PTR)
#define FTM1_INVCTRL                             FTM_INVCTRL_REG(FTM1_BASE_PTR)
#define FTM1_SWOCTRL                             FTM_SWOCTRL_REG(FTM1_BASE_PTR)
#define FTM1_PWMLOAD                             FTM_PWMLOAD_REG(FTM1_BASE_PTR)
/* FTM2 */
#define FTM2_SC                                  FTM_SC_REG(FTM2_BASE_PTR)
#define FTM2_CNT                                 FTM_CNT_REG(FTM2_BASE_PTR)
#define FTM2_MOD                                 FTM_MOD_REG(FTM2_BASE_PTR)
#define FTM2_C0SC                                FTM_CnSC_REG(FTM2_BASE_PTR,0)
#define FTM2_C0V                                 FTM_CnV_REG(FTM2_BASE_PTR,0)
#define FTM2_C1SC                                FTM_CnSC_REG(FTM2_BASE_PTR,1)
#define FTM2_C1V                                 FTM_CnV_REG(FTM2_BASE_PTR,1)
#define FTM2_CNTIN                               FTM_CNTIN_REG(FTM2_BASE_PTR)
#define FTM2_STATUS                              FTM_STATUS_REG(FTM2_BASE_PTR)
#define FTM2_MODE                                FTM_MODE_REG(FTM2_BASE_PTR)
#define FTM2_SYNC                                FTM_SYNC_REG(FTM2_BASE_PTR)
#define FTM2_OUTINIT                             FTM_OUTINIT_REG(FTM2_BASE_PTR)
#define FTM2_OUTMASK                             FTM_OUTMASK_REG(FTM2_BASE_PTR)
#define FTM2_COMBINE                             FTM_COMBINE_REG(FTM2_BASE_PTR)
#define FTM2_DEADTIME                            FTM_DEADTIME_REG(FTM2_BASE_PTR)
#define FTM2_EXTTRIG                             FTM_EXTTRIG_REG(FTM2_BASE_PTR)
#define FTM2_POL                                 FTM_POL_REG(FTM2_BASE_PTR)
#define FTM2_FMS                                 FTM_FMS_REG(FTM2_BASE_PTR)
#define FTM2_FILTER                              FTM_FILTER_REG(FTM2_BASE_PTR)
#define FTM2_FLTCTRL                             FTM_FLTCTRL_REG(FTM2_BASE_PTR)
#define FTM2_QDCTRL                              FTM_QDCTRL_REG(FTM2_BASE_PTR)
#define FTM2_CONF                                FTM_CONF_REG(FTM2_BASE_PTR)
#define FTM2_FLTPOL                              FTM_FLTPOL_REG(FTM2_BASE_PTR)
#define FTM2_SYNCONF                             FTM_SYNCONF_REG(FTM2_BASE_PTR)
#define FTM2_INVCTRL                             FTM_INVCTRL_REG(FTM2_BASE_PTR)
#define FTM2_SWOCTRL                             FTM_SWOCTRL_REG(FTM2_BASE_PTR)
#define FTM2_PWMLOAD                             FTM_PWMLOAD_REG(FTM2_BASE_PTR)
/* FTM3 */
#define FTM3_SC                                  FTM_SC_REG(FTM3_BASE_PTR)
#define FTM3_CNT                                 FTM_CNT_REG(FTM3_BASE_PTR)
#define FTM3_MOD                                 FTM_MOD_REG(FTM3_BASE_PTR)
#define FTM3_C0SC                                FTM_CnSC_REG(FTM3_BASE_PTR,0)
#define FTM3_C0V                                 FTM_CnV_REG(FTM3_BASE_PTR,0)
#define FTM3_C1SC                                FTM_CnSC_REG(FTM3_BASE_PTR,1)
#define FTM3_C1V                                 FTM_CnV_REG(FTM3_BASE_PTR,1)
#define FTM3_C2SC                                FTM_CnSC_REG(FTM3_BASE_PTR,2)
#define FTM3_C2V                                 FTM_CnV_REG(FTM3_BASE_PTR,2)
#define FTM3_C3SC                                FTM_CnSC_REG(FTM3_BASE_PTR,3)
#define FTM3_C3V                                 FTM_CnV_REG(FTM3_BASE_PTR,3)
#define FTM3_C4SC                                FTM_CnSC_REG(FTM3_BASE_PTR,4)
#define FTM3_C4V                                 FTM_CnV_REG(FTM3_BASE_PTR,4)
#define FTM3_C5SC                                FTM_CnSC_REG(FTM3_BASE_PTR,5)
#define FTM3_C5V                                 FTM_CnV_REG(FTM3_BASE_PTR,5)
#define FTM3_CNTIN                               FTM_CNTIN_REG(FTM3_BASE_PTR)
#define FTM3_STATUS                              FTM_STATUS_REG(FTM3_BASE_PTR)
#define FTM3_MODE                                FTM_MODE_REG(FTM3_BASE_PTR)
#define FTM3_SYNC                                FTM_SYNC_REG(FTM3_BASE_PTR)
#define FTM3_OUTINIT                             FTM_OUTINIT_REG(FTM3_BASE_PTR)
#define FTM3_OUTMASK                             FTM_OUTMASK_REG(FTM3_BASE_PTR)
#define FTM3_COMBINE                             FTM_COMBINE_REG(FTM3_BASE_PTR)
#define FTM3_DEADTIME                            FTM_DEADTIME_REG(FTM3_BASE_PTR)
#define FTM3_EXTTRIG                             FTM_EXTTRIG_REG(FTM3_BASE_PTR)
#define FTM3_POL                                 FTM_POL_REG(FTM3_BASE_PTR)
#define FTM3_FMS                                 FTM_FMS_REG(FTM3_BASE_PTR)
#define FTM3_FILTER                              FTM_FILTER_REG(FTM3_BASE_PTR)
#define FTM3_FLTCTRL                             FTM_FLTCTRL_REG(FTM3_BASE_PTR)
#define FTM3_QDCTRL                              FTM_QDCTRL_REG(FTM3_BASE_PTR)
#define FTM3_CONF                                FTM_CONF_REG(FTM3_BASE_PTR)
#define FTM3_FLTPOL                              FTM_FLTPOL_REG(FTM3_BASE_PTR)
#define FTM3_SYNCONF                             FTM_SYNCONF_REG(FTM3_BASE_PTR)
#define FTM3_INVCTRL                             FTM_INVCTRL_REG(FTM3_BASE_PTR)
#define FTM3_SWOCTRL                             FTM_SWOCTRL_REG(FTM3_BASE_PTR)
#define FTM3_PWMLOAD                             FTM_PWMLOAD_REG(FTM3_BASE_PTR)

/* FTM - Register array accessors */
#define FTM0_CnSC(index)                         FTM_CnSC_REG(FTM0_BASE_PTR,index)
#define FTM1_CnSC(index)                         FTM_CnSC_REG(FTM1_BASE_PTR,index)
#define FTM2_CnSC(index)                         FTM_CnSC_REG(FTM2_BASE_PTR,index)
#define FTM3_CnSC(index)                         FTM_CnSC_REG(FTM3_BASE_PTR,index)
#define FTM0_CnV(index)                          FTM_CnV_REG(FTM0_BASE_PTR,index)
#define FTM1_CnV(index)                          FTM_CnV_REG(FTM1_BASE_PTR,index)
#define FTM2_CnV(index)                          FTM_CnV_REG(FTM2_BASE_PTR,index)
#define FTM3_CnV(index)                          FTM_CnV_REG(FTM3_BASE_PTR,index)

/**
 * @}
 */ /* end of group FTM_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group FTM_Peripheral */


/* ----------------------------------------------------------------------------
   -- FTMRA
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup FTMRA_Peripheral FTMRA
 * @{
 */

/** FTMRA - Peripheral register structure */
typedef struct FTMRA_MemMap {
  uint8_t FSEC;                                    /**< Flash Security Register, offset: 0x0 */
  uint8_t FCLKDIV;                                 /**< Flash Clock Divider Register, offset: 0x1 */
  uint8_t FECCRIX;                                 /**< Flash ECCR Index Register, offset: 0x2 */
  uint8_t FCCOBIX;                                 /**< Flash Common Command Object Index Register, offset: 0x3 */
  uint8_t FERCNFG;                                 /**< Flash Error Configuration Register, offset: 0x4 */
  uint8_t FCNFG;                                   /**< Flash Configuration Register, offset: 0x5 */
  uint8_t FERSTAT;                                 /**< Flash Error Status Register, offset: 0x6 */
  uint8_t FSTAT;                                   /**< Flash Status Register, offset: 0x7 */
  uint8_t DFPROT;                                  /**< D-Flash Protection Register, offset: 0x8 */
  uint8_t FPROT;                                   /**< P-Flash Protection Register, offset: 0x9 */
  uint8_t FCCOBLO;                                 /**< Flash Common Command Object Low Register, offset: 0xA */
  uint8_t FCCOBHI;                                 /**< Flash Common Command Object High Register, offset: 0xB */
  uint8_t RESERVED_0[2];
  uint8_t FECCRLO;                                 /**< Flash ECC Error Results Low Register, offset: 0xE */
  uint8_t FECCRHI;                                 /**< Flash ECC Error Results High Register, offset: 0xF */
  uint8_t RESERVED_1[1];
  uint8_t FOPT;                                    /**< Flash Option Register, offset: 0x11 */
} volatile *FTMRA_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- FTMRA - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup FTMRA_Register_Accessor_Macros FTMRA - Register accessor macros
 * @{
 */


/* FTMRA - Register accessors */
#define FTMRA_FSEC_REG(base)                     ((base)->FSEC)
#define FTMRA_FCLKDIV_REG(base)                  ((base)->FCLKDIV)
#define FTMRA_FECCRIX_REG(base)                  ((base)->FECCRIX)
#define FTMRA_FCCOBIX_REG(base)                  ((base)->FCCOBIX)
#define FTMRA_FERCNFG_REG(base)                  ((base)->FERCNFG)
#define FTMRA_FCNFG_REG(base)                    ((base)->FCNFG)
#define FTMRA_FERSTAT_REG(base)                  ((base)->FERSTAT)
#define FTMRA_FSTAT_REG(base)                    ((base)->FSTAT)
#define FTMRA_DFPROT_REG(base)                   ((base)->DFPROT)
#define FTMRA_FPROT_REG(base)                    ((base)->FPROT)
#define FTMRA_FCCOBLO_REG(base)                  ((base)->FCCOBLO)
#define FTMRA_FCCOBHI_REG(base)                  ((base)->FCCOBHI)
#define FTMRA_FECCRLO_REG(base)                  ((base)->FECCRLO)
#define FTMRA_FECCRHI_REG(base)                  ((base)->FECCRHI)
#define FTMRA_FOPT_REG(base)                     ((base)->FOPT)

/**
 * @}
 */ /* end of group FTMRA_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- FTMRA Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup FTMRA_Register_Masks FTMRA Register Masks
 * @{
 */

/* FSEC Bit Fields */
#define FTMRA_FSEC_SEC_MASK                      0x3u
#define FTMRA_FSEC_SEC_SHIFT                     0
#define FTMRA_FSEC_SEC(x)                        (((uint8_t)(((uint8_t)(x))<<FTMRA_FSEC_SEC_SHIFT))&FTMRA_FSEC_SEC_MASK)
#define FTMRA_FSEC_RNV_MASK                      0x3Cu
#define FTMRA_FSEC_RNV_SHIFT                     2
#define FTMRA_FSEC_RNV(x)                        (((uint8_t)(((uint8_t)(x))<<FTMRA_FSEC_RNV_SHIFT))&FTMRA_FSEC_RNV_MASK)
#define FTMRA_FSEC_KEYEN_MASK                    0xC0u
#define FTMRA_FSEC_KEYEN_SHIFT                   6
#define FTMRA_FSEC_KEYEN(x)                      (((uint8_t)(((uint8_t)(x))<<FTMRA_FSEC_KEYEN_SHIFT))&FTMRA_FSEC_KEYEN_MASK)
/* FCLKDIV Bit Fields */
#define FTMRA_FCLKDIV_FDIV_MASK                  0x7Fu
#define FTMRA_FCLKDIV_FDIV_SHIFT                 0
#define FTMRA_FCLKDIV_FDIV(x)                    (((uint8_t)(((uint8_t)(x))<<FTMRA_FCLKDIV_FDIV_SHIFT))&FTMRA_FCLKDIV_FDIV_MASK)
#define FTMRA_FCLKDIV_FDIVLD_MASK                0x80u
#define FTMRA_FCLKDIV_FDIVLD_SHIFT               7
/* FECCRIX Bit Fields */
#define FTMRA_FECCRIX_ECCRIX_MASK                0x7u
#define FTMRA_FECCRIX_ECCRIX_SHIFT               0
#define FTMRA_FECCRIX_ECCRIX(x)                  (((uint8_t)(((uint8_t)(x))<<FTMRA_FECCRIX_ECCRIX_SHIFT))&FTMRA_FECCRIX_ECCRIX_MASK)
/* FCCOBIX Bit Fields */
#define FTMRA_FCCOBIX_CCOBIX_MASK                0x7u
#define FTMRA_FCCOBIX_CCOBIX_SHIFT               0
#define FTMRA_FCCOBIX_CCOBIX(x)                  (((uint8_t)(((uint8_t)(x))<<FTMRA_FCCOBIX_CCOBIX_SHIFT))&FTMRA_FCCOBIX_CCOBIX_MASK)
/* FERCNFG Bit Fields */
#define FTMRA_FERCNFG_SFDIE_MASK                 0x1u
#define FTMRA_FERCNFG_SFDIE_SHIFT                0
#define FTMRA_FERCNFG_DFDIE_MASK                 0x2u
#define FTMRA_FERCNFG_DFDIE_SHIFT                1
/* FCNFG Bit Fields */
#define FTMRA_FCNFG_FSFD_MASK                    0x1u
#define FTMRA_FCNFG_FSFD_SHIFT                   0
#define FTMRA_FCNFG_FDFD_MASK                    0x2u
#define FTMRA_FCNFG_FDFD_SHIFT                   1
#define FTMRA_FCNFG_IGNSF_MASK                   0x10u
#define FTMRA_FCNFG_IGNSF_SHIFT                  4
#define FTMRA_FCNFG_ERSAREQ_MASK                 0x20u
#define FTMRA_FCNFG_ERSAREQ_SHIFT                5
#define FTMRA_FCNFG_CCIE_MASK                    0x80u
#define FTMRA_FCNFG_CCIE_SHIFT                   7
/* FERSTAT Bit Fields */
#define FTMRA_FERSTAT_SFDIF_MASK                 0x1u
#define FTMRA_FERSTAT_SFDIF_SHIFT                0
#define FTMRA_FERSTAT_DFDIF_MASK                 0x2u
#define FTMRA_FERSTAT_DFDIF_SHIFT                1
/* FSTAT Bit Fields */
#define FTMRA_FSTAT_MGSTAT_MASK                  0x3u
#define FTMRA_FSTAT_MGSTAT_SHIFT                 0
#define FTMRA_FSTAT_MGSTAT(x)                    (((uint8_t)(((uint8_t)(x))<<FTMRA_FSTAT_MGSTAT_SHIFT))&FTMRA_FSTAT_MGSTAT_MASK)
#define FTMRA_FSTAT_MGBUSY_MASK                  0x8u
#define FTMRA_FSTAT_MGBUSY_SHIFT                 3
#define FTMRA_FSTAT_FPVIOL_MASK                  0x10u
#define FTMRA_FSTAT_FPVIOL_SHIFT                 4
#define FTMRA_FSTAT_ACCERR_MASK                  0x20u
#define FTMRA_FSTAT_ACCERR_SHIFT                 5
#define FTMRA_FSTAT_CCIF_MASK                    0x80u
#define FTMRA_FSTAT_CCIF_SHIFT                   7
/* DFPROT Bit Fields */
#define FTMRA_DFPROT_DPS_MASK                    0x1Fu
#define FTMRA_DFPROT_DPS_SHIFT                   0
#define FTMRA_DFPROT_DPS(x)                      (((uint8_t)(((uint8_t)(x))<<FTMRA_DFPROT_DPS_SHIFT))&FTMRA_DFPROT_DPS_MASK)
#define FTMRA_DFPROT_DPOPEN_MASK                 0x80u
#define FTMRA_DFPROT_DPOPEN_SHIFT                7
/* FPROT Bit Fields */
#define FTMRA_FPROT_FPLS_MASK                    0x3u
#define FTMRA_FPROT_FPLS_SHIFT                   0
#define FTMRA_FPROT_FPLS(x)                      (((uint8_t)(((uint8_t)(x))<<FTMRA_FPROT_FPLS_SHIFT))&FTMRA_FPROT_FPLS_MASK)
#define FTMRA_FPROT_FPLDIS_MASK                  0x4u
#define FTMRA_FPROT_FPLDIS_SHIFT                 2
#define FTMRA_FPROT_FPHS_MASK                    0x18u
#define FTMRA_FPROT_FPHS_SHIFT                   3
#define FTMRA_FPROT_FPHS(x)                      (((uint8_t)(((uint8_t)(x))<<FTMRA_FPROT_FPHS_SHIFT))&FTMRA_FPROT_FPHS_MASK)
#define FTMRA_FPROT_FPHDIS_MASK                  0x20u
#define FTMRA_FPROT_FPHDIS_SHIFT                 5
#define FTMRA_FPROT_RNV_MASK                     0x40u
#define FTMRA_FPROT_RNV_SHIFT                    6
#define FTMRA_FPROT_FPOPEN_MASK                  0x80u
#define FTMRA_FPROT_FPOPEN_SHIFT                 7
/* FCCOBLO Bit Fields */
#define FTMRA_FCCOBLO_CCOBn_MASK                 0xFFu
#define FTMRA_FCCOBLO_CCOBn_SHIFT                0
#define FTMRA_FCCOBLO_CCOBn(x)                   (((uint8_t)(((uint8_t)(x))<<FTMRA_FCCOBLO_CCOBn_SHIFT))&FTMRA_FCCOBLO_CCOBn_MASK)
/* FCCOBHI Bit Fields */
#define FTMRA_FCCOBHI_CCOBn_MASK                 0xFFu
#define FTMRA_FCCOBHI_CCOBn_SHIFT                0
#define FTMRA_FCCOBHI_CCOBn(x)                   (((uint8_t)(((uint8_t)(x))<<FTMRA_FCCOBHI_CCOBn_SHIFT))&FTMRA_FCCOBHI_CCOBn_MASK)
/* FECCRLO Bit Fields */
#define FTMRA_FECCRLO_ECCR_MASK                  0xFFu
#define FTMRA_FECCRLO_ECCR_SHIFT                 0
#define FTMRA_FECCRLO_ECCR(x)                    (((uint8_t)(((uint8_t)(x))<<FTMRA_FECCRLO_ECCR_SHIFT))&FTMRA_FECCRLO_ECCR_MASK)
/* FECCRHI Bit Fields */
#define FTMRA_FECCRHI_ECCR_MASK                  0xFFu
#define FTMRA_FECCRHI_ECCR_SHIFT                 0
#define FTMRA_FECCRHI_ECCR(x)                    (((uint8_t)(((uint8_t)(x))<<FTMRA_FECCRHI_ECCR_SHIFT))&FTMRA_FECCRHI_ECCR_MASK)
/* FOPT Bit Fields */
#define FTMRA_FOPT_NV_MASK                       0xFFu
#define FTMRA_FOPT_NV_SHIFT                      0
#define FTMRA_FOPT_NV(x)                         (((uint8_t)(((uint8_t)(x))<<FTMRA_FOPT_NV_SHIFT))&FTMRA_FOPT_NV_MASK)

/**
 * @}
 */ /* end of group FTMRA_Register_Masks */


/* FTMRA - Peripheral instance base addresses */
/** Peripheral FTMRA base pointer */
#define FTMRA_BASE_PTR                           ((FTMRA_MemMapPtr)0x40039000u)

/* ----------------------------------------------------------------------------
   -- FTMRA - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup FTMRA_Register_Accessor_Macros FTMRA - Register accessor macros
 * @{
 */


/* FTMRA - Register instance definitions */
/* FTMRA */
#define FTMRA_FSEC                               FTMRA_FSEC_REG(FTMRA_BASE_PTR)
#define FTMRA_FCLKDIV                            FTMRA_FCLKDIV_REG(FTMRA_BASE_PTR)
#define FTMRA_FECCRIX                            FTMRA_FECCRIX_REG(FTMRA_BASE_PTR)
#define FTMRA_FCCOBIX                            FTMRA_FCCOBIX_REG(FTMRA_BASE_PTR)
#define FTMRA_FERCNFG                            FTMRA_FERCNFG_REG(FTMRA_BASE_PTR)
#define FTMRA_FCNFG                              FTMRA_FCNFG_REG(FTMRA_BASE_PTR)
#define FTMRA_FERSTAT                            FTMRA_FERSTAT_REG(FTMRA_BASE_PTR)
#define FTMRA_FSTAT                              FTMRA_FSTAT_REG(FTMRA_BASE_PTR)
#define FTMRA_DFPROT                             FTMRA_DFPROT_REG(FTMRA_BASE_PTR)
#define FTMRA_FPROT                              FTMRA_FPROT_REG(FTMRA_BASE_PTR)
#define FTMRA_FCCOBLO                            FTMRA_FCCOBLO_REG(FTMRA_BASE_PTR)
#define FTMRA_FCCOBHI                            FTMRA_FCCOBHI_REG(FTMRA_BASE_PTR)
#define FTMRA_FECCRLO                            FTMRA_FECCRLO_REG(FTMRA_BASE_PTR)
#define FTMRA_FECCRHI                            FTMRA_FECCRHI_REG(FTMRA_BASE_PTR)
#define FTMRA_FOPT                               FTMRA_FOPT_REG(FTMRA_BASE_PTR)

/**
 * @}
 */ /* end of group FTMRA_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group FTMRA_Peripheral */


/* ----------------------------------------------------------------------------
   -- GPIO
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup GPIO_Peripheral GPIO
 * @{
 */

/** GPIO - Peripheral register structure */
typedef struct GPIO_MemMap {
  uint32_t PDOR;                                   /**< Port Data Output Register, offset: 0x0 */
  uint32_t PSOR;                                   /**< Port Set Output Register, offset: 0x4 */
  uint32_t PCOR;                                   /**< Port Clear Output Register, offset: 0x8 */
  uint32_t PTOR;                                   /**< Port Toggle Output Register, offset: 0xC */
  uint32_t PDIR;                                   /**< Port Data Input Register, offset: 0x10 */
  uint32_t PDDR;                                   /**< Port Data Direction Register, offset: 0x14 */
} volatile *GPIO_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- GPIO - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup GPIO_Register_Accessor_Macros GPIO - Register accessor macros
 * @{
 */


/* GPIO - Register accessors */
#define GPIO_PDOR_REG(base)                      ((base)->PDOR)
#define GPIO_PSOR_REG(base)                      ((base)->PSOR)
#define GPIO_PCOR_REG(base)                      ((base)->PCOR)
#define GPIO_PTOR_REG(base)                      ((base)->PTOR)
#define GPIO_PDIR_REG(base)                      ((base)->PDIR)
#define GPIO_PDDR_REG(base)                      ((base)->PDDR)

/**
 * @}
 */ /* end of group GPIO_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- GPIO Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup GPIO_Register_Masks GPIO Register Masks
 * @{
 */

/* PDOR Bit Fields */
#define GPIO_PDOR_PDO_MASK                       0xFFFFFFFFu
#define GPIO_PDOR_PDO_SHIFT                      0
#define GPIO_PDOR_PDO(x)                         (((uint32_t)(((uint32_t)(x))<<GPIO_PDOR_PDO_SHIFT))&GPIO_PDOR_PDO_MASK)
/* PSOR Bit Fields */
#define GPIO_PSOR_PTSO_MASK                      0xFFFFFFFFu
#define GPIO_PSOR_PTSO_SHIFT                     0
#define GPIO_PSOR_PTSO(x)                        (((uint32_t)(((uint32_t)(x))<<GPIO_PSOR_PTSO_SHIFT))&GPIO_PSOR_PTSO_MASK)
/* PCOR Bit Fields */
#define GPIO_PCOR_PTCO_MASK                      0xFFFFFFFFu
#define GPIO_PCOR_PTCO_SHIFT                     0
#define GPIO_PCOR_PTCO(x)                        (((uint32_t)(((uint32_t)(x))<<GPIO_PCOR_PTCO_SHIFT))&GPIO_PCOR_PTCO_MASK)
/* PTOR Bit Fields */
#define GPIO_PTOR_PTTO_MASK                      0xFFFFFFFFu
#define GPIO_PTOR_PTTO_SHIFT                     0
#define GPIO_PTOR_PTTO(x)                        (((uint32_t)(((uint32_t)(x))<<GPIO_PTOR_PTTO_SHIFT))&GPIO_PTOR_PTTO_MASK)
/* PDIR Bit Fields */
#define GPIO_PDIR_PDI_MASK                       0xFFFFFFFFu
#define GPIO_PDIR_PDI_SHIFT                      0
#define GPIO_PDIR_PDI(x)                         (((uint32_t)(((uint32_t)(x))<<GPIO_PDIR_PDI_SHIFT))&GPIO_PDIR_PDI_MASK)
/* PDDR Bit Fields */
#define GPIO_PDDR_PDD_MASK                       0xFFFFFFFFu
#define GPIO_PDDR_PDD_SHIFT                      0
#define GPIO_PDDR_PDD(x)                         (((uint32_t)(((uint32_t)(x))<<GPIO_PDDR_PDD_SHIFT))&GPIO_PDDR_PDD_MASK)

/**
 * @}
 */ /* end of group GPIO_Register_Masks */


/* GPIO - Peripheral instance base addresses */
/** Peripheral PTA base pointer */
#define PTA_BASE_PTR                             ((GPIO_MemMapPtr)0x4007F000u)
/** Peripheral PTB base pointer */
#define PTB_BASE_PTR                             ((GPIO_MemMapPtr)0x4007F040u)
/** Peripheral PTC base pointer */
#define PTC_BASE_PTR                             ((GPIO_MemMapPtr)0x4007F080u)
/** Peripheral PTD base pointer */
#define PTD_BASE_PTR                             ((GPIO_MemMapPtr)0x4007F0C0u)
/** Peripheral PTE base pointer */
#define PTE_BASE_PTR                             ((GPIO_MemMapPtr)0x4007F100u)

/* ----------------------------------------------------------------------------
   -- GPIO - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup GPIO_Register_Accessor_Macros GPIO - Register accessor macros
 * @{
 */


/* GPIO - Register instance definitions */
/* PTA */
#define GPIOA_PDOR                               GPIO_PDOR_REG(PTA_BASE_PTR)
#define GPIOA_PSOR                               GPIO_PSOR_REG(PTA_BASE_PTR)
#define GPIOA_PCOR                               GPIO_PCOR_REG(PTA_BASE_PTR)
#define GPIOA_PTOR                               GPIO_PTOR_REG(PTA_BASE_PTR)
#define GPIOA_PDIR                               GPIO_PDIR_REG(PTA_BASE_PTR)
#define GPIOA_PDDR                               GPIO_PDDR_REG(PTA_BASE_PTR)
/* PTB */
#define GPIOB_PDOR                               GPIO_PDOR_REG(PTB_BASE_PTR)
#define GPIOB_PSOR                               GPIO_PSOR_REG(PTB_BASE_PTR)
#define GPIOB_PCOR                               GPIO_PCOR_REG(PTB_BASE_PTR)
#define GPIOB_PTOR                               GPIO_PTOR_REG(PTB_BASE_PTR)
#define GPIOB_PDIR                               GPIO_PDIR_REG(PTB_BASE_PTR)
#define GPIOB_PDDR                               GPIO_PDDR_REG(PTB_BASE_PTR)
/* PTC */
#define GPIOC_PDOR                               GPIO_PDOR_REG(PTC_BASE_PTR)
#define GPIOC_PSOR                               GPIO_PSOR_REG(PTC_BASE_PTR)
#define GPIOC_PCOR                               GPIO_PCOR_REG(PTC_BASE_PTR)
#define GPIOC_PTOR                               GPIO_PTOR_REG(PTC_BASE_PTR)
#define GPIOC_PDIR                               GPIO_PDIR_REG(PTC_BASE_PTR)
#define GPIOC_PDDR                               GPIO_PDDR_REG(PTC_BASE_PTR)
/* PTD */
#define GPIOD_PDOR                               GPIO_PDOR_REG(PTD_BASE_PTR)
#define GPIOD_PSOR                               GPIO_PSOR_REG(PTD_BASE_PTR)
#define GPIOD_PCOR                               GPIO_PCOR_REG(PTD_BASE_PTR)
#define GPIOD_PTOR                               GPIO_PTOR_REG(PTD_BASE_PTR)
#define GPIOD_PDIR                               GPIO_PDIR_REG(PTD_BASE_PTR)
#define GPIOD_PDDR                               GPIO_PDDR_REG(PTD_BASE_PTR)
/* PTE */
#define GPIOE_PDOR                               GPIO_PDOR_REG(PTE_BASE_PTR)
#define GPIOE_PSOR                               GPIO_PSOR_REG(PTE_BASE_PTR)
#define GPIOE_PCOR                               GPIO_PCOR_REG(PTE_BASE_PTR)
#define GPIOE_PTOR                               GPIO_PTOR_REG(PTE_BASE_PTR)
#define GPIOE_PDIR                               GPIO_PDIR_REG(PTE_BASE_PTR)
#define GPIOE_PDDR                               GPIO_PDDR_REG(PTE_BASE_PTR)

/**
 * @}
 */ /* end of group GPIO_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group GPIO_Peripheral */


/* ----------------------------------------------------------------------------
   -- I2C
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup I2C_Peripheral I2C
 * @{
 */

/** I2C - Peripheral register structure */
typedef struct I2C_MemMap {
  uint8_t A1;                                      /**< I2C Address Register 1, offset: 0x0 */
  uint8_t F;                                       /**< I2C Frequency Divider register, offset: 0x1 */
  uint8_t C1;                                      /**< I2C Control Register 1, offset: 0x2 */
  uint8_t S;                                       /**< I2C Status register, offset: 0x3 */
  uint8_t D;                                       /**< I2C Data I/O register, offset: 0x4 */
  uint8_t C2;                                      /**< I2C Control Register 2, offset: 0x5 */
  uint8_t FLT;                                     /**< I2C Programmable Input Glitch Filter register, offset: 0x6 */
  uint8_t RA;                                      /**< I2C Range Address register, offset: 0x7 */
  uint8_t SMB;                                     /**< I2C SMBus Control and Status register, offset: 0x8 */
  uint8_t A2;                                      /**< I2C Address Register 2, offset: 0x9 */
  uint8_t SLTH;                                    /**< I2C SCL Low Timeout Register High, offset: 0xA */
  uint8_t SLTL;                                    /**< I2C SCL Low Timeout Register Low, offset: 0xB */
} volatile *I2C_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- I2C - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup I2C_Register_Accessor_Macros I2C - Register accessor macros
 * @{
 */


/* I2C - Register accessors */
#define I2C_A1_REG(base)                         ((base)->A1)
#define I2C_F_REG(base)                          ((base)->F)
#define I2C_C1_REG(base)                         ((base)->C1)
#define I2C_S_REG(base)                          ((base)->S)
#define I2C_D_REG(base)                          ((base)->D)
#define I2C_C2_REG(base)                         ((base)->C2)
#define I2C_FLT_REG(base)                        ((base)->FLT)
#define I2C_RA_REG(base)                         ((base)->RA)
#define I2C_SMB_REG(base)                        ((base)->SMB)
#define I2C_A2_REG(base)                         ((base)->A2)
#define I2C_SLTH_REG(base)                       ((base)->SLTH)
#define I2C_SLTL_REG(base)                       ((base)->SLTL)

/**
 * @}
 */ /* end of group I2C_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- I2C Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup I2C_Register_Masks I2C Register Masks
 * @{
 */

/* A1 Bit Fields */
#define I2C_A1_AD_MASK                           0xFEu
#define I2C_A1_AD_SHIFT                          1
#define I2C_A1_AD(x)                             (((uint8_t)(((uint8_t)(x))<<I2C_A1_AD_SHIFT))&I2C_A1_AD_MASK)
/* F Bit Fields */
#define I2C_F_ICR_MASK                           0x3Fu
#define I2C_F_ICR_SHIFT                          0
#define I2C_F_ICR(x)                             (((uint8_t)(((uint8_t)(x))<<I2C_F_ICR_SHIFT))&I2C_F_ICR_MASK)
#define I2C_F_MULT_MASK                          0xC0u
#define I2C_F_MULT_SHIFT                         6
#define I2C_F_MULT(x)                            (((uint8_t)(((uint8_t)(x))<<I2C_F_MULT_SHIFT))&I2C_F_MULT_MASK)
/* C1 Bit Fields */
#define I2C_C1_DMAEN_MASK                        0x1u
#define I2C_C1_DMAEN_SHIFT                       0
#define I2C_C1_WUEN_MASK                         0x2u
#define I2C_C1_WUEN_SHIFT                        1
#define I2C_C1_RSTA_MASK                         0x4u
#define I2C_C1_RSTA_SHIFT                        2
#define I2C_C1_TXAK_MASK                         0x8u
#define I2C_C1_TXAK_SHIFT                        3
#define I2C_C1_TX_MASK                           0x10u
#define I2C_C1_TX_SHIFT                          4
#define I2C_C1_MST_MASK                          0x20u
#define I2C_C1_MST_SHIFT                         5
#define I2C_C1_IICIE_MASK                        0x40u
#define I2C_C1_IICIE_SHIFT                       6
#define I2C_C1_IICEN_MASK                        0x80u
#define I2C_C1_IICEN_SHIFT                       7
/* S Bit Fields */
#define I2C_S_RXAK_MASK                          0x1u
#define I2C_S_RXAK_SHIFT                         0
#define I2C_S_IICIF_MASK                         0x2u
#define I2C_S_IICIF_SHIFT                        1
#define I2C_S_SRW_MASK                           0x4u
#define I2C_S_SRW_SHIFT                          2
#define I2C_S_RAM_MASK                           0x8u
#define I2C_S_RAM_SHIFT                          3
#define I2C_S_ARBL_MASK                          0x10u
#define I2C_S_ARBL_SHIFT                         4
#define I2C_S_BUSY_MASK                          0x20u
#define I2C_S_BUSY_SHIFT                         5
#define I2C_S_IAAS_MASK                          0x40u
#define I2C_S_IAAS_SHIFT                         6
#define I2C_S_TCF_MASK                           0x80u
#define I2C_S_TCF_SHIFT                          7
/* D Bit Fields */
#define I2C_D_DATA_MASK                          0xFFu
#define I2C_D_DATA_SHIFT                         0
#define I2C_D_DATA(x)                            (((uint8_t)(((uint8_t)(x))<<I2C_D_DATA_SHIFT))&I2C_D_DATA_MASK)
/* C2 Bit Fields */
#define I2C_C2_AD_MASK                           0x7u
#define I2C_C2_AD_SHIFT                          0
#define I2C_C2_AD(x)                             (((uint8_t)(((uint8_t)(x))<<I2C_C2_AD_SHIFT))&I2C_C2_AD_MASK)
#define I2C_C2_RMEN_MASK                         0x8u
#define I2C_C2_RMEN_SHIFT                        3
#define I2C_C2_SBRC_MASK                         0x10u
#define I2C_C2_SBRC_SHIFT                        4
#define I2C_C2_HDRS_MASK                         0x20u
#define I2C_C2_HDRS_SHIFT                        5
#define I2C_C2_ADEXT_MASK                        0x40u
#define I2C_C2_ADEXT_SHIFT                       6
#define I2C_C2_GCAEN_MASK                        0x80u
#define I2C_C2_GCAEN_SHIFT                       7
/* FLT Bit Fields */
#define I2C_FLT_FLT_MASK                         0x1Fu
#define I2C_FLT_FLT_SHIFT                        0
#define I2C_FLT_FLT(x)                           (((uint8_t)(((uint8_t)(x))<<I2C_FLT_FLT_SHIFT))&I2C_FLT_FLT_MASK)
#define I2C_FLT_STOPIE_MASK                      0x20u
#define I2C_FLT_STOPIE_SHIFT                     5
#define I2C_FLT_STOPF_MASK                       0x40u
#define I2C_FLT_STOPF_SHIFT                      6
#define I2C_FLT_SHEN_MASK                        0x80u
#define I2C_FLT_SHEN_SHIFT                       7
/* RA Bit Fields */
#define I2C_RA_RAD_MASK                          0xFEu
#define I2C_RA_RAD_SHIFT                         1
#define I2C_RA_RAD(x)                            (((uint8_t)(((uint8_t)(x))<<I2C_RA_RAD_SHIFT))&I2C_RA_RAD_MASK)
/* SMB Bit Fields */
#define I2C_SMB_SHTF2IE_MASK                     0x1u
#define I2C_SMB_SHTF2IE_SHIFT                    0
#define I2C_SMB_SHTF2_MASK                       0x2u
#define I2C_SMB_SHTF2_SHIFT                      1
#define I2C_SMB_SHTF1_MASK                       0x4u
#define I2C_SMB_SHTF1_SHIFT                      2
#define I2C_SMB_SLTF_MASK                        0x8u
#define I2C_SMB_SLTF_SHIFT                       3
#define I2C_SMB_TCKSEL_MASK                      0x10u
#define I2C_SMB_TCKSEL_SHIFT                     4
#define I2C_SMB_SIICAEN_MASK                     0x20u
#define I2C_SMB_SIICAEN_SHIFT                    5
#define I2C_SMB_ALERTEN_MASK                     0x40u
#define I2C_SMB_ALERTEN_SHIFT                    6
#define I2C_SMB_FACK_MASK                        0x80u
#define I2C_SMB_FACK_SHIFT                       7
/* A2 Bit Fields */
#define I2C_A2_SAD_MASK                          0xFEu
#define I2C_A2_SAD_SHIFT                         1
#define I2C_A2_SAD(x)                            (((uint8_t)(((uint8_t)(x))<<I2C_A2_SAD_SHIFT))&I2C_A2_SAD_MASK)
/* SLTH Bit Fields */
#define I2C_SLTH_SSLT_MASK                       0xFFu
#define I2C_SLTH_SSLT_SHIFT                      0
#define I2C_SLTH_SSLT(x)                         (((uint8_t)(((uint8_t)(x))<<I2C_SLTH_SSLT_SHIFT))&I2C_SLTH_SSLT_MASK)
/* SLTL Bit Fields */
#define I2C_SLTL_SSLT_MASK                       0xFFu
#define I2C_SLTL_SSLT_SHIFT                      0
#define I2C_SLTL_SSLT(x)                         (((uint8_t)(((uint8_t)(x))<<I2C_SLTL_SSLT_SHIFT))&I2C_SLTL_SSLT_MASK)

/**
 * @}
 */ /* end of group I2C_Register_Masks */


/* I2C - Peripheral instance base addresses */
/** Peripheral I2C0 base pointer */
#define I2C0_BASE_PTR                            ((I2C_MemMapPtr)0x40047000u)
/** Peripheral I2C1 base pointer */
#define I2C1_BASE_PTR                            ((I2C_MemMapPtr)0x40048000u)

/* ----------------------------------------------------------------------------
   -- I2C - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup I2C_Register_Accessor_Macros I2C - Register accessor macros
 * @{
 */


/* I2C - Register instance definitions */
/* I2C0 */
#define I2C0_A1                                  I2C_A1_REG(I2C0_BASE_PTR)
#define I2C0_F                                   I2C_F_REG(I2C0_BASE_PTR)
#define I2C0_C1                                  I2C_C1_REG(I2C0_BASE_PTR)
#define I2C0_S                                   I2C_S_REG(I2C0_BASE_PTR)
#define I2C0_D                                   I2C_D_REG(I2C0_BASE_PTR)
#define I2C0_C2                                  I2C_C2_REG(I2C0_BASE_PTR)
#define I2C0_FLT                                 I2C_FLT_REG(I2C0_BASE_PTR)
#define I2C0_RA                                  I2C_RA_REG(I2C0_BASE_PTR)
#define I2C0_SMB                                 I2C_SMB_REG(I2C0_BASE_PTR)
#define I2C0_A2                                  I2C_A2_REG(I2C0_BASE_PTR)
#define I2C0_SLTH                                I2C_SLTH_REG(I2C0_BASE_PTR)
#define I2C0_SLTL                                I2C_SLTL_REG(I2C0_BASE_PTR)
/* I2C1 */
#define I2C1_A1                                  I2C_A1_REG(I2C1_BASE_PTR)
#define I2C1_F                                   I2C_F_REG(I2C1_BASE_PTR)
#define I2C1_C1                                  I2C_C1_REG(I2C1_BASE_PTR)
#define I2C1_S                                   I2C_S_REG(I2C1_BASE_PTR)
#define I2C1_D                                   I2C_D_REG(I2C1_BASE_PTR)
#define I2C1_C2                                  I2C_C2_REG(I2C1_BASE_PTR)
#define I2C1_FLT                                 I2C_FLT_REG(I2C1_BASE_PTR)
#define I2C1_RA                                  I2C_RA_REG(I2C1_BASE_PTR)
#define I2C1_SMB                                 I2C_SMB_REG(I2C1_BASE_PTR)
#define I2C1_A2                                  I2C_A2_REG(I2C1_BASE_PTR)
#define I2C1_SLTH                                I2C_SLTH_REG(I2C1_BASE_PTR)
#define I2C1_SLTL                                I2C_SLTL_REG(I2C1_BASE_PTR)

/**
 * @}
 */ /* end of group I2C_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group I2C_Peripheral */


/* ----------------------------------------------------------------------------
   -- ICS
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup ICS_Peripheral ICS
 * @{
 */

/** ICS - Peripheral register structure */
typedef struct ICS_MemMap {
  uint8_t C1;                                      /**< ICS Control Register 1, offset: 0x0 */
  uint8_t C2;                                      /**< ICS Control Register 2, offset: 0x1 */
  uint8_t C3;                                      /**< ICS Control Register 3, offset: 0x2 */
  uint8_t C4;                                      /**< ICS Control Register 4, offset: 0x3 */
  uint8_t S;                                       /**< ICS Status Register, offset: 0x4 */
} volatile *ICS_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- ICS - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup ICS_Register_Accessor_Macros ICS - Register accessor macros
 * @{
 */


/* ICS - Register accessors */
#define ICS_C1_REG(base)                         ((base)->C1)
#define ICS_C2_REG(base)                         ((base)->C2)
#define ICS_C3_REG(base)                         ((base)->C3)
#define ICS_C4_REG(base)                         ((base)->C4)
#define ICS_S_REG(base)                          ((base)->S)

/**
 * @}
 */ /* end of group ICS_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- ICS Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup ICS_Register_Masks ICS Register Masks
 * @{
 */

/* C1 Bit Fields */
#define ICS_C1_IREFSTEN_MASK                     0x1u
#define ICS_C1_IREFSTEN_SHIFT                    0
#define ICS_C1_IRCLKEN_MASK                      0x2u
#define ICS_C1_IRCLKEN_SHIFT                     1
#define ICS_C1_IREFS_MASK                        0x4u
#define ICS_C1_IREFS_SHIFT                       2
#define ICS_C1_RDIV_MASK                         0x38u
#define ICS_C1_RDIV_SHIFT                        3
#define ICS_C1_RDIV(x)                           (((uint8_t)(((uint8_t)(x))<<ICS_C1_RDIV_SHIFT))&ICS_C1_RDIV_MASK)
#define ICS_C1_CLKS_MASK                         0xC0u
#define ICS_C1_CLKS_SHIFT                        6
#define ICS_C1_CLKS(x)                           (((uint8_t)(((uint8_t)(x))<<ICS_C1_CLKS_SHIFT))&ICS_C1_CLKS_MASK)
/* C2 Bit Fields */
#define ICS_C2_FRDIV_MASK                        0xCu
#define ICS_C2_FRDIV_SHIFT                       2
#define ICS_C2_FRDIV(x)                          (((uint8_t)(((uint8_t)(x))<<ICS_C2_FRDIV_SHIFT))&ICS_C2_FRDIV_MASK)
#define ICS_C2_LP_MASK                           0x10u
#define ICS_C2_LP_SHIFT                          4
#define ICS_C2_BDIV_MASK                         0xE0u
#define ICS_C2_BDIV_SHIFT                        5
#define ICS_C2_BDIV(x)                           (((uint8_t)(((uint8_t)(x))<<ICS_C2_BDIV_SHIFT))&ICS_C2_BDIV_MASK)
/* C3 Bit Fields */
#define ICS_C3_SCTRIM_MASK                       0xFFu
#define ICS_C3_SCTRIM_SHIFT                      0
#define ICS_C3_SCTRIM(x)                         (((uint8_t)(((uint8_t)(x))<<ICS_C3_SCTRIM_SHIFT))&ICS_C3_SCTRIM_MASK)
/* C4 Bit Fields */
#define ICS_C4_SCFTRIM_MASK                      0x1u
#define ICS_C4_SCFTRIM_SHIFT                     0
#define ICS_C4_CME_MASK                          0x20u
#define ICS_C4_CME_SHIFT                         5
#define ICS_C4_RLOLIE_MASK                       0x40u
#define ICS_C4_RLOLIE_SHIFT                      6
#define ICS_C4_FLOLIE_MASK                       0x80u
#define ICS_C4_FLOLIE_SHIFT                      7
/* S Bit Fields */
#define ICS_S_CLKST_MASK                         0xCu
#define ICS_S_CLKST_SHIFT                        2
#define ICS_S_CLKST(x)                           (((uint8_t)(((uint8_t)(x))<<ICS_S_CLKST_SHIFT))&ICS_S_CLKST_MASK)
#define ICS_S_IREFST_MASK                        0x10u
#define ICS_S_IREFST_SHIFT                       4
#define ICS_S_RLOCK_MASK                         0x20u
#define ICS_S_RLOCK_SHIFT                        5
#define ICS_S_FLOCK_MASK                         0x40u
#define ICS_S_FLOCK_SHIFT                        6
#define ICS_S_LOLS_MASK                          0x80u
#define ICS_S_LOLS_SHIFT                         7

/**
 * @}
 */ /* end of group ICS_Register_Masks */


/* ICS - Peripheral instance base addresses */
/** Peripheral ICS base pointer */
#define ICS_BASE_PTR                             ((ICS_MemMapPtr)0x40044000u)

/* ----------------------------------------------------------------------------
   -- ICS - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup ICS_Register_Accessor_Macros ICS - Register accessor macros
 * @{
 */


/* ICS - Register instance definitions */
/* ICS */
#define ICS_C1                                   ICS_C1_REG(ICS_BASE_PTR)
#define ICS_C2                                   ICS_C2_REG(ICS_BASE_PTR)
#define ICS_C3                                   ICS_C3_REG(ICS_BASE_PTR)
#define ICS_C4                                   ICS_C4_REG(ICS_BASE_PTR)
#define ICS_S                                    ICS_S_REG(ICS_BASE_PTR)

/**
 * @}
 */ /* end of group ICS_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group ICS_Peripheral */


/* ----------------------------------------------------------------------------
   -- IEVENT
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup IEVENT_Peripheral IEVENT
 * @{
 */

/** IEVENT - Peripheral register structure */
typedef struct IEVENT_MemMap {
  uint32_t DRL;                                    /**< iEvent Data Register: Low, offset: 0x0 */
  uint8_t RESERVED_0[124];
  uint32_t CRL;                                    /**< iEvent Control Register: Low, offset: 0x80 */
  uint8_t RESERVED_1[124];
  struct {                                         /* offset: 0x100, array step: 0x8 */
    uint32_t IMXCR;                                  /**< iEvent Input Mux Configuration Register, array offset: 0x100, array step: 0x8 */
    uint32_t BFECR;                                  /**< iEvent Boolean Function Eva1ation Configuration Register, array offset: 0x104, array step: 0x8 */
  } OUTPUT[4];
} volatile *IEVENT_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- IEVENT - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup IEVENT_Register_Accessor_Macros IEVENT - Register accessor macros
 * @{
 */


/* IEVENT - Register accessors */
#define IEVENT_DRL_REG(base)                     ((base)->DRL)
#define IEVENT_CRL_REG(base)                     ((base)->CRL)
#define IEVENT_IMXCR_REG(base,index)             ((base)->OUTPUT[index].IMXCR)
#define IEVENT_BFECR_REG(base,index)             ((base)->OUTPUT[index].BFECR)

/**
 * @}
 */ /* end of group IEVENT_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- IEVENT Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup IEVENT_Register_Masks IEVENT Register Masks
 * @{
 */

/* DRL Bit Fields */
#define IEVENT_DRL_IN_A0_MASK                    0x1u
#define IEVENT_DRL_IN_A0_SHIFT                   0
#define IEVENT_DRL_IN_B0_MASK                    0x2u
#define IEVENT_DRL_IN_B0_SHIFT                   1
#define IEVENT_DRL_IN_C0_MASK                    0x4u
#define IEVENT_DRL_IN_C0_SHIFT                   2
#define IEVENT_DRL_IN_D0_MASK                    0x8u
#define IEVENT_DRL_IN_D0_SHIFT                   3
#define IEVENT_DRL_Output_FSM0_MASK              0x70u
#define IEVENT_DRL_Output_FSM0_SHIFT             4
#define IEVENT_DRL_Output_FSM0(x)                (((uint32_t)(((uint32_t)(x))<<IEVENT_DRL_Output_FSM0_SHIFT))&IEVENT_DRL_Output_FSM0_MASK)
#define IEVENT_DRL_Ev_Out0_MASK                  0x80u
#define IEVENT_DRL_Ev_Out0_SHIFT                 7
#define IEVENT_DRL_IN_A1_MASK                    0x100u
#define IEVENT_DRL_IN_A1_SHIFT                   8
#define IEVENT_DRL_IN_B1_MASK                    0x200u
#define IEVENT_DRL_IN_B1_SHIFT                   9
#define IEVENT_DRL_IN_C1_MASK                    0x400u
#define IEVENT_DRL_IN_C1_SHIFT                   10
#define IEVENT_DRL_IN_D1_MASK                    0x800u
#define IEVENT_DRL_IN_D1_SHIFT                   11
#define IEVENT_DRL_Output_FSM1_MASK              0x7000u
#define IEVENT_DRL_Output_FSM1_SHIFT             12
#define IEVENT_DRL_Output_FSM1(x)                (((uint32_t)(((uint32_t)(x))<<IEVENT_DRL_Output_FSM1_SHIFT))&IEVENT_DRL_Output_FSM1_MASK)
#define IEVENT_DRL_Ev_Out1_MASK                  0x8000u
#define IEVENT_DRL_Ev_Out1_SHIFT                 15
#define IEVENT_DRL_IN_A2_MASK                    0x10000u
#define IEVENT_DRL_IN_A2_SHIFT                   16
#define IEVENT_DRL_IN_B2_MASK                    0x20000u
#define IEVENT_DRL_IN_B2_SHIFT                   17
#define IEVENT_DRL_IN_C2_MASK                    0x40000u
#define IEVENT_DRL_IN_C2_SHIFT                   18
#define IEVENT_DRL_IN_D2_MASK                    0x80000u
#define IEVENT_DRL_IN_D2_SHIFT                   19
#define IEVENT_DRL_Output_FSM2_MASK              0x700000u
#define IEVENT_DRL_Output_FSM2_SHIFT             20
#define IEVENT_DRL_Output_FSM2(x)                (((uint32_t)(((uint32_t)(x))<<IEVENT_DRL_Output_FSM2_SHIFT))&IEVENT_DRL_Output_FSM2_MASK)
#define IEVENT_DRL_Ev_Out2_MASK                  0x800000u
#define IEVENT_DRL_Ev_Out2_SHIFT                 23
#define IEVENT_DRL_IN_A3_MASK                    0x1000000u
#define IEVENT_DRL_IN_A3_SHIFT                   24
#define IEVENT_DRL_IN_B3_MASK                    0x2000000u
#define IEVENT_DRL_IN_B3_SHIFT                   25
#define IEVENT_DRL_IN_C3_MASK                    0x4000000u
#define IEVENT_DRL_IN_C3_SHIFT                   26
#define IEVENT_DRL_IN_D3_MASK                    0x8000000u
#define IEVENT_DRL_IN_D3_SHIFT                   27
#define IEVENT_DRL_Output_FSM3_MASK              0x70000000u
#define IEVENT_DRL_Output_FSM3_SHIFT             28
#define IEVENT_DRL_Output_FSM3(x)                (((uint32_t)(((uint32_t)(x))<<IEVENT_DRL_Output_FSM3_SHIFT))&IEVENT_DRL_Output_FSM3_MASK)
#define IEVENT_DRL_Ev_Out3_MASK                  0x80000000u
#define IEVENT_DRL_Ev_Out3_SHIFT                 31
/* CRL Bit Fields */
#define IEVENT_CRL_Type0_MASK                    0x3u
#define IEVENT_CRL_Type0_SHIFT                   0
#define IEVENT_CRL_Type0(x)                      (((uint32_t)(((uint32_t)(x))<<IEVENT_CRL_Type0_SHIFT))&IEVENT_CRL_Type0_MASK)
#define IEVENT_CRL_OSE0_MASK                     0x4u
#define IEVENT_CRL_OSE0_SHIFT                    2
#define IEVENT_CRL_DDB0_MASK                     0x8u
#define IEVENT_CRL_DDB0_SHIFT                    3
#define IEVENT_CRL_RO0_MASK                      0x80u
#define IEVENT_CRL_RO0_SHIFT                     7
#define IEVENT_CRL_Type1_MASK                    0x300u
#define IEVENT_CRL_Type1_SHIFT                   8
#define IEVENT_CRL_Type1(x)                      (((uint32_t)(((uint32_t)(x))<<IEVENT_CRL_Type1_SHIFT))&IEVENT_CRL_Type1_MASK)
#define IEVENT_CRL_OSE1_MASK                     0x400u
#define IEVENT_CRL_OSE1_SHIFT                    10
#define IEVENT_CRL_DDB1_MASK                     0x800u
#define IEVENT_CRL_DDB1_SHIFT                    11
#define IEVENT_CRL_RO1_MASK                      0x8000u
#define IEVENT_CRL_RO1_SHIFT                     15
#define IEVENT_CRL_Type2_MASK                    0x30000u
#define IEVENT_CRL_Type2_SHIFT                   16
#define IEVENT_CRL_Type2(x)                      (((uint32_t)(((uint32_t)(x))<<IEVENT_CRL_Type2_SHIFT))&IEVENT_CRL_Type2_MASK)
#define IEVENT_CRL_OSE2_MASK                     0x40000u
#define IEVENT_CRL_OSE2_SHIFT                    18
#define IEVENT_CRL_DDB2_MASK                     0x80000u
#define IEVENT_CRL_DDB2_SHIFT                    19
#define IEVENT_CRL_RO2_MASK                      0x800000u
#define IEVENT_CRL_RO2_SHIFT                     23
#define IEVENT_CRL_Type3_MASK                    0x3000000u
#define IEVENT_CRL_Type3_SHIFT                   24
#define IEVENT_CRL_Type3(x)                      (((uint32_t)(((uint32_t)(x))<<IEVENT_CRL_Type3_SHIFT))&IEVENT_CRL_Type3_MASK)
#define IEVENT_CRL_OSE3_MASK                     0x4000000u
#define IEVENT_CRL_OSE3_SHIFT                    26
#define IEVENT_CRL_DDB3_MASK                     0x8000000u
#define IEVENT_CRL_DDB3_SHIFT                    27
#define IEVENT_CRL_RO3_MASK                      0x80000000u
#define IEVENT_CRL_RO3_SHIFT                     31
/* IMXCR Bit Fields */
#define IEVENT_IMXCR_D_Select_MASK               0xFu
#define IEVENT_IMXCR_D_Select_SHIFT              0
#define IEVENT_IMXCR_D_Select(x)                 (((uint32_t)(((uint32_t)(x))<<IEVENT_IMXCR_D_Select_SHIFT))&IEVENT_IMXCR_D_Select_MASK)
#define IEVENT_IMXCR_C_Select_MASK               0xF00u
#define IEVENT_IMXCR_C_Select_SHIFT              8
#define IEVENT_IMXCR_C_Select(x)                 (((uint32_t)(((uint32_t)(x))<<IEVENT_IMXCR_C_Select_SHIFT))&IEVENT_IMXCR_C_Select_MASK)
#define IEVENT_IMXCR_B_Select_MASK               0xF0000u
#define IEVENT_IMXCR_B_Select_SHIFT              16
#define IEVENT_IMXCR_B_Select(x)                 (((uint32_t)(((uint32_t)(x))<<IEVENT_IMXCR_B_Select_SHIFT))&IEVENT_IMXCR_B_Select_MASK)
#define IEVENT_IMXCR_A_Select_MASK               0xF000000u
#define IEVENT_IMXCR_A_Select_SHIFT              24
#define IEVENT_IMXCR_A_Select(x)                 (((uint32_t)(((uint32_t)(x))<<IEVENT_IMXCR_A_Select_SHIFT))&IEVENT_IMXCR_A_Select_MASK)
/* BFECR Bit Fields */
#define IEVENT_BFECR_PT3_DC_MASK                 0x3u
#define IEVENT_BFECR_PT3_DC_SHIFT                0
#define IEVENT_BFECR_PT3_DC(x)                   (((uint32_t)(((uint32_t)(x))<<IEVENT_BFECR_PT3_DC_SHIFT))&IEVENT_BFECR_PT3_DC_MASK)
#define IEVENT_BFECR_PT3_CC_MASK                 0xCu
#define IEVENT_BFECR_PT3_CC_SHIFT                2
#define IEVENT_BFECR_PT3_CC(x)                   (((uint32_t)(((uint32_t)(x))<<IEVENT_BFECR_PT3_CC_SHIFT))&IEVENT_BFECR_PT3_CC_MASK)
#define IEVENT_BFECR_PT3_BC_MASK                 0x30u
#define IEVENT_BFECR_PT3_BC_SHIFT                4
#define IEVENT_BFECR_PT3_BC(x)                   (((uint32_t)(((uint32_t)(x))<<IEVENT_BFECR_PT3_BC_SHIFT))&IEVENT_BFECR_PT3_BC_MASK)
#define IEVENT_BFECR_PT3_AC_MASK                 0xC0u
#define IEVENT_BFECR_PT3_AC_SHIFT                6
#define IEVENT_BFECR_PT3_AC(x)                   (((uint32_t)(((uint32_t)(x))<<IEVENT_BFECR_PT3_AC_SHIFT))&IEVENT_BFECR_PT3_AC_MASK)
#define IEVENT_BFECR_PT2_DC_MASK                 0x300u
#define IEVENT_BFECR_PT2_DC_SHIFT                8
#define IEVENT_BFECR_PT2_DC(x)                   (((uint32_t)(((uint32_t)(x))<<IEVENT_BFECR_PT2_DC_SHIFT))&IEVENT_BFECR_PT2_DC_MASK)
#define IEVENT_BFECR_PT2_CC_MASK                 0xC00u
#define IEVENT_BFECR_PT2_CC_SHIFT                10
#define IEVENT_BFECR_PT2_CC(x)                   (((uint32_t)(((uint32_t)(x))<<IEVENT_BFECR_PT2_CC_SHIFT))&IEVENT_BFECR_PT2_CC_MASK)
#define IEVENT_BFECR_PT2_BC_MASK                 0x3000u
#define IEVENT_BFECR_PT2_BC_SHIFT                12
#define IEVENT_BFECR_PT2_BC(x)                   (((uint32_t)(((uint32_t)(x))<<IEVENT_BFECR_PT2_BC_SHIFT))&IEVENT_BFECR_PT2_BC_MASK)
#define IEVENT_BFECR_PT2_AC_MASK                 0xC000u
#define IEVENT_BFECR_PT2_AC_SHIFT                14
#define IEVENT_BFECR_PT2_AC(x)                   (((uint32_t)(((uint32_t)(x))<<IEVENT_BFECR_PT2_AC_SHIFT))&IEVENT_BFECR_PT2_AC_MASK)
#define IEVENT_BFECR_PT1_DC_MASK                 0x30000u
#define IEVENT_BFECR_PT1_DC_SHIFT                16
#define IEVENT_BFECR_PT1_DC(x)                   (((uint32_t)(((uint32_t)(x))<<IEVENT_BFECR_PT1_DC_SHIFT))&IEVENT_BFECR_PT1_DC_MASK)
#define IEVENT_BFECR_PT1_CC_MASK                 0xC0000u
#define IEVENT_BFECR_PT1_CC_SHIFT                18
#define IEVENT_BFECR_PT1_CC(x)                   (((uint32_t)(((uint32_t)(x))<<IEVENT_BFECR_PT1_CC_SHIFT))&IEVENT_BFECR_PT1_CC_MASK)
#define IEVENT_BFECR_PT1_BC_MASK                 0x300000u
#define IEVENT_BFECR_PT1_BC_SHIFT                20
#define IEVENT_BFECR_PT1_BC(x)                   (((uint32_t)(((uint32_t)(x))<<IEVENT_BFECR_PT1_BC_SHIFT))&IEVENT_BFECR_PT1_BC_MASK)
#define IEVENT_BFECR_PT1_AC_MASK                 0xC00000u
#define IEVENT_BFECR_PT1_AC_SHIFT                22
#define IEVENT_BFECR_PT1_AC(x)                   (((uint32_t)(((uint32_t)(x))<<IEVENT_BFECR_PT1_AC_SHIFT))&IEVENT_BFECR_PT1_AC_MASK)
#define IEVENT_BFECR_PT0_DC_MASK                 0x3000000u
#define IEVENT_BFECR_PT0_DC_SHIFT                24
#define IEVENT_BFECR_PT0_DC(x)                   (((uint32_t)(((uint32_t)(x))<<IEVENT_BFECR_PT0_DC_SHIFT))&IEVENT_BFECR_PT0_DC_MASK)
#define IEVENT_BFECR_PT0_CC_MASK                 0xC000000u
#define IEVENT_BFECR_PT0_CC_SHIFT                26
#define IEVENT_BFECR_PT0_CC(x)                   (((uint32_t)(((uint32_t)(x))<<IEVENT_BFECR_PT0_CC_SHIFT))&IEVENT_BFECR_PT0_CC_MASK)
#define IEVENT_BFECR_PT0_BC_MASK                 0x30000000u
#define IEVENT_BFECR_PT0_BC_SHIFT                28
#define IEVENT_BFECR_PT0_BC(x)                   (((uint32_t)(((uint32_t)(x))<<IEVENT_BFECR_PT0_BC_SHIFT))&IEVENT_BFECR_PT0_BC_MASK)
#define IEVENT_BFECR_PT0_AC_MASK                 0xC0000000u
#define IEVENT_BFECR_PT0_AC_SHIFT                30
#define IEVENT_BFECR_PT0_AC(x)                   (((uint32_t)(((uint32_t)(x))<<IEVENT_BFECR_PT0_AC_SHIFT))&IEVENT_BFECR_PT0_AC_MASK)

/**
 * @}
 */ /* end of group IEVENT_Register_Masks */


/* IEVENT - Peripheral instance base addresses */
/** Peripheral IEVENT base pointer */
#define IEVENT_BASE_PTR                          ((IEVENT_MemMapPtr)0x40019000u)

/* ----------------------------------------------------------------------------
   -- IEVENT - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup IEVENT_Register_Accessor_Macros IEVENT - Register accessor macros
 * @{
 */


/* IEVENT - Register instance definitions */
/* IEVENT */
#define IEVENT_DRL                               IEVENT_DRL_REG(IEVENT_BASE_PTR)
#define IEVENT_CRL                               IEVENT_CRL_REG(IEVENT_BASE_PTR)
#define IEVENT_IMXCR0                            IEVENT_IMXCR_REG(IEVENT_BASE_PTR,0)
#define IEVENT_BFECR0                            IEVENT_BFECR_REG(IEVENT_BASE_PTR,0)
#define IEVENT_IMXCR1                            IEVENT_IMXCR_REG(IEVENT_BASE_PTR,1)
#define IEVENT_BFECR1                            IEVENT_BFECR_REG(IEVENT_BASE_PTR,1)
#define IEVENT_IMXCR2                            IEVENT_IMXCR_REG(IEVENT_BASE_PTR,2)
#define IEVENT_BFECR2                            IEVENT_BFECR_REG(IEVENT_BASE_PTR,2)
#define IEVENT_IMXCR3                            IEVENT_IMXCR_REG(IEVENT_BASE_PTR,3)
#define IEVENT_BFECR3                            IEVENT_BFECR_REG(IEVENT_BASE_PTR,3)

/* IEVENT - Register array accessors */
#define IEVENT_IMXCR(index)                      IEVENT_IMXCR_REG(IEVENT_BASE_PTR,index)
#define IEVENT_BFECR(index)                      IEVENT_BFECR_REG(IEVENT_BASE_PTR,index)

/**
 * @}
 */ /* end of group IEVENT_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group IEVENT_Peripheral */


/* ----------------------------------------------------------------------------
   -- ITM
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup ITM_Peripheral ITM
 * @{
 */

/** ITM - Peripheral register structure */
typedef struct ITM_MemMap {
  union {                                          /* offset: 0x0 */
    uint32_t STIM_READ[32];                          /**< Stimulus Port Register 0 (for reading)..Stimulus Port Register 31 (for reading), array offset: 0x0, array step: 0x4 */
    uint32_t STIM_WRITE[32];                         /**< Stimulus Port Register 0 (for writing)..Stimulus Port Register 31 (for writing), array offset: 0x0, array step: 0x4 */
  };
  uint8_t RESERVED_0[3456];
  uint32_t TER;                                    /**< Trace Enable Register, offset: 0xE00 */
  uint8_t RESERVED_1[60];
  uint32_t TPR;                                    /**< Trace Privilege Register, offset: 0xE40 */
  uint8_t RESERVED_2[60];
  uint32_t TCR;                                    /**< Trace Control Register, offset: 0xE80 */
  uint8_t RESERVED_3[300];
  uint32_t LAR;                                    /**< Lock Access Register, offset: 0xFB0 */
  uint32_t LSR;                                    /**< Lock Status Register, offset: 0xFB4 */
  uint8_t RESERVED_4[24];
  uint32_t PID4;                                   /**< Peripheral Identification Register 4., offset: 0xFD0 */
  uint32_t PID5;                                   /**< Peripheral Identification Register 5., offset: 0xFD4 */
  uint32_t PID6;                                   /**< Peripheral Identification Register 6., offset: 0xFD8 */
  uint32_t PID7;                                   /**< Peripheral Identification Register 7., offset: 0xFDC */
  uint32_t PID0;                                   /**< Peripheral Identification Register 0., offset: 0xFE0 */
  uint32_t PID1;                                   /**< Peripheral Identification Register 1., offset: 0xFE4 */
  uint32_t PID2;                                   /**< Peripheral Identification Register 2., offset: 0xFE8 */
  uint32_t PID3;                                   /**< Peripheral Identification Register 3., offset: 0xFEC */
  uint32_t CID0;                                   /**< Component Identification Register 0., offset: 0xFF0 */
  uint32_t CID1;                                   /**< Component Identification Register 1., offset: 0xFF4 */
  uint32_t CID2;                                   /**< Component Identification Register 2., offset: 0xFF8 */
  uint32_t CID3;                                   /**< Component Identification Register 3., offset: 0xFFC */
} volatile *ITM_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- ITM - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup ITM_Register_Accessor_Macros ITM - Register accessor macros
 * @{
 */


/* ITM - Register accessors */
#define ITM_STIM_READ_REG(base,index2)           ((base)->STIM_READ[index2])
#define ITM_STIM_WRITE_REG(base,index2)          ((base)->STIM_WRITE[index2])
#define ITM_TER_REG(base)                        ((base)->TER)
#define ITM_TPR_REG(base)                        ((base)->TPR)
#define ITM_TCR_REG(base)                        ((base)->TCR)
#define ITM_LAR_REG(base)                        ((base)->LAR)
#define ITM_LSR_REG(base)                        ((base)->LSR)
#define ITM_PID4_REG(base)                       ((base)->PID4)
#define ITM_PID5_REG(base)                       ((base)->PID5)
#define ITM_PID6_REG(base)                       ((base)->PID6)
#define ITM_PID7_REG(base)                       ((base)->PID7)
#define ITM_PID0_REG(base)                       ((base)->PID0)
#define ITM_PID1_REG(base)                       ((base)->PID1)
#define ITM_PID2_REG(base)                       ((base)->PID2)
#define ITM_PID3_REG(base)                       ((base)->PID3)
#define ITM_CID0_REG(base)                       ((base)->CID0)
#define ITM_CID1_REG(base)                       ((base)->CID1)
#define ITM_CID2_REG(base)                       ((base)->CID2)
#define ITM_CID3_REG(base)                       ((base)->CID3)

/**
 * @}
 */ /* end of group ITM_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- ITM Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup ITM_Register_Masks ITM Register Masks
 * @{
 */


/**
 * @}
 */ /* end of group ITM_Register_Masks */


/* ITM - Peripheral instance base addresses */
/** Peripheral ITM base pointer */
#define ITM_BASE_PTR                             ((ITM_MemMapPtr)0xE0000000u)

/* ----------------------------------------------------------------------------
   -- ITM - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup ITM_Register_Accessor_Macros ITM - Register accessor macros
 * @{
 */


/* ITM - Register instance definitions */
/* ITM */
#define ITM_STIM0_READ                           ITM_STIM_READ_REG(ITM_BASE_PTR,0)
#define ITM_STIM0_WRITE                          ITM_STIM_WRITE_REG(ITM_BASE_PTR,0)
#define ITM_STIM1_READ                           ITM_STIM_READ_REG(ITM_BASE_PTR,1)
#define ITM_STIM1_WRITE                          ITM_STIM_WRITE_REG(ITM_BASE_PTR,1)
#define ITM_STIM2_READ                           ITM_STIM_READ_REG(ITM_BASE_PTR,2)
#define ITM_STIM2_WRITE                          ITM_STIM_WRITE_REG(ITM_BASE_PTR,2)
#define ITM_STIM3_READ                           ITM_STIM_READ_REG(ITM_BASE_PTR,3)
#define ITM_STIM3_WRITE                          ITM_STIM_WRITE_REG(ITM_BASE_PTR,3)
#define ITM_STIM4_READ                           ITM_STIM_READ_REG(ITM_BASE_PTR,4)
#define ITM_STIM4_WRITE                          ITM_STIM_WRITE_REG(ITM_BASE_PTR,4)
#define ITM_STIM5_READ                           ITM_STIM_READ_REG(ITM_BASE_PTR,5)
#define ITM_STIM5_WRITE                          ITM_STIM_WRITE_REG(ITM_BASE_PTR,5)
#define ITM_STIM6_READ                           ITM_STIM_READ_REG(ITM_BASE_PTR,6)
#define ITM_STIM6_WRITE                          ITM_STIM_WRITE_REG(ITM_BASE_PTR,6)
#define ITM_STIM7_READ                           ITM_STIM_READ_REG(ITM_BASE_PTR,7)
#define ITM_STIM7_WRITE                          ITM_STIM_WRITE_REG(ITM_BASE_PTR,7)
#define ITM_STIM8_READ                           ITM_STIM_READ_REG(ITM_BASE_PTR,8)
#define ITM_STIM8_WRITE                          ITM_STIM_WRITE_REG(ITM_BASE_PTR,8)
#define ITM_STIM9_READ                           ITM_STIM_READ_REG(ITM_BASE_PTR,9)
#define ITM_STIM9_WRITE                          ITM_STIM_WRITE_REG(ITM_BASE_PTR,9)
#define ITM_STIM10_READ                          ITM_STIM_READ_REG(ITM_BASE_PTR,10)
#define ITM_STIM10_WRITE                         ITM_STIM_WRITE_REG(ITM_BASE_PTR,10)
#define ITM_STIM11_READ                          ITM_STIM_READ_REG(ITM_BASE_PTR,11)
#define ITM_STIM11_WRITE                         ITM_STIM_WRITE_REG(ITM_BASE_PTR,11)
#define ITM_STIM12_READ                          ITM_STIM_READ_REG(ITM_BASE_PTR,12)
#define ITM_STIM12_WRITE                         ITM_STIM_WRITE_REG(ITM_BASE_PTR,12)
#define ITM_STIM13_READ                          ITM_STIM_READ_REG(ITM_BASE_PTR,13)
#define ITM_STIM13_WRITE                         ITM_STIM_WRITE_REG(ITM_BASE_PTR,13)
#define ITM_STIM14_READ                          ITM_STIM_READ_REG(ITM_BASE_PTR,14)
#define ITM_STIM14_WRITE                         ITM_STIM_WRITE_REG(ITM_BASE_PTR,14)
#define ITM_STIM15_READ                          ITM_STIM_READ_REG(ITM_BASE_PTR,15)
#define ITM_STIM15_WRITE                         ITM_STIM_WRITE_REG(ITM_BASE_PTR,15)
#define ITM_STIM16_READ                          ITM_STIM_READ_REG(ITM_BASE_PTR,16)
#define ITM_STIM16_WRITE                         ITM_STIM_WRITE_REG(ITM_BASE_PTR,16)
#define ITM_STIM17_READ                          ITM_STIM_READ_REG(ITM_BASE_PTR,17)
#define ITM_STIM17_WRITE                         ITM_STIM_WRITE_REG(ITM_BASE_PTR,17)
#define ITM_STIM18_READ                          ITM_STIM_READ_REG(ITM_BASE_PTR,18)
#define ITM_STIM18_WRITE                         ITM_STIM_WRITE_REG(ITM_BASE_PTR,18)
#define ITM_STIM19_READ                          ITM_STIM_READ_REG(ITM_BASE_PTR,19)
#define ITM_STIM19_WRITE                         ITM_STIM_WRITE_REG(ITM_BASE_PTR,19)
#define ITM_STIM20_READ                          ITM_STIM_READ_REG(ITM_BASE_PTR,20)
#define ITM_STIM20_WRITE                         ITM_STIM_WRITE_REG(ITM_BASE_PTR,20)
#define ITM_STIM21_READ                          ITM_STIM_READ_REG(ITM_BASE_PTR,21)
#define ITM_STIM21_WRITE                         ITM_STIM_WRITE_REG(ITM_BASE_PTR,21)
#define ITM_STIM22_READ                          ITM_STIM_READ_REG(ITM_BASE_PTR,22)
#define ITM_STIM22_WRITE                         ITM_STIM_WRITE_REG(ITM_BASE_PTR,22)
#define ITM_STIM23_READ                          ITM_STIM_READ_REG(ITM_BASE_PTR,23)
#define ITM_STIM23_WRITE                         ITM_STIM_WRITE_REG(ITM_BASE_PTR,23)
#define ITM_STIM24_READ                          ITM_STIM_READ_REG(ITM_BASE_PTR,24)
#define ITM_STIM24_WRITE                         ITM_STIM_WRITE_REG(ITM_BASE_PTR,24)
#define ITM_STIM25_READ                          ITM_STIM_READ_REG(ITM_BASE_PTR,25)
#define ITM_STIM25_WRITE                         ITM_STIM_WRITE_REG(ITM_BASE_PTR,25)
#define ITM_STIM26_READ                          ITM_STIM_READ_REG(ITM_BASE_PTR,26)
#define ITM_STIM26_WRITE                         ITM_STIM_WRITE_REG(ITM_BASE_PTR,26)
#define ITM_STIM27_READ                          ITM_STIM_READ_REG(ITM_BASE_PTR,27)
#define ITM_STIM27_WRITE                         ITM_STIM_WRITE_REG(ITM_BASE_PTR,27)
#define ITM_STIM28_READ                          ITM_STIM_READ_REG(ITM_BASE_PTR,28)
#define ITM_STIM28_WRITE                         ITM_STIM_WRITE_REG(ITM_BASE_PTR,28)
#define ITM_STIM29_READ                          ITM_STIM_READ_REG(ITM_BASE_PTR,29)
#define ITM_STIM29_WRITE                         ITM_STIM_WRITE_REG(ITM_BASE_PTR,29)
#define ITM_STIM30_READ                          ITM_STIM_READ_REG(ITM_BASE_PTR,30)
#define ITM_STIM30_WRITE                         ITM_STIM_WRITE_REG(ITM_BASE_PTR,30)
#define ITM_STIM31_READ                          ITM_STIM_READ_REG(ITM_BASE_PTR,31)
#define ITM_STIM31_WRITE                         ITM_STIM_WRITE_REG(ITM_BASE_PTR,31)
#define ITM_TER                                  ITM_TER_REG(ITM_BASE_PTR)
#define ITM_TPR                                  ITM_TPR_REG(ITM_BASE_PTR)
#define ITM_TCR                                  ITM_TCR_REG(ITM_BASE_PTR)
#define ITM_LAR                                  ITM_LAR_REG(ITM_BASE_PTR)
#define ITM_LSR                                  ITM_LSR_REG(ITM_BASE_PTR)
#define ITM_PID4                                 ITM_PID4_REG(ITM_BASE_PTR)
#define ITM_PID5                                 ITM_PID5_REG(ITM_BASE_PTR)
#define ITM_PID6                                 ITM_PID6_REG(ITM_BASE_PTR)
#define ITM_PID7                                 ITM_PID7_REG(ITM_BASE_PTR)
#define ITM_PID0                                 ITM_PID0_REG(ITM_BASE_PTR)
#define ITM_PID1                                 ITM_PID1_REG(ITM_BASE_PTR)
#define ITM_PID2                                 ITM_PID2_REG(ITM_BASE_PTR)
#define ITM_PID3                                 ITM_PID3_REG(ITM_BASE_PTR)
#define ITM_CID0                                 ITM_CID0_REG(ITM_BASE_PTR)
#define ITM_CID1                                 ITM_CID1_REG(ITM_BASE_PTR)
#define ITM_CID2                                 ITM_CID2_REG(ITM_BASE_PTR)
#define ITM_CID3                                 ITM_CID3_REG(ITM_BASE_PTR)

/* ITM - Register array accessors */
#define ITM_STIM_READ(index2)                    ITM_STIM_READ_REG(ITM_BASE_PTR,index2)
#define ITM_STIM_WRITE(index2)                   ITM_STIM_WRITE_REG(ITM_BASE_PTR,index2)

/**
 * @}
 */ /* end of group ITM_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group ITM_Peripheral */


/* ----------------------------------------------------------------------------
   -- MCM
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup MCM_Peripheral MCM
 * @{
 */

/** MCM - Peripheral register structure */
typedef struct MCM_MemMap {
  uint8_t RESERVED_0[8];
  uint16_t PLASC;                                  /**< Crossbar Switch (AXBS) Slave Configuration, offset: 0x8 */
  uint16_t PLAMC;                                  /**< Crossbar Switch (AXBS) Master Configuration, offset: 0xA */
  uint32_t CR;                                     /**< Control Register, offset: 0xC */
  uint8_t RESERVED_1[32];
  uint32_t PID;                                    /**< Process ID register, offset: 0x30 */
} volatile *MCM_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- MCM - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup MCM_Register_Accessor_Macros MCM - Register accessor macros
 * @{
 */


/* MCM - Register accessors */
#define MCM_PLASC_REG(base)                      ((base)->PLASC)
#define MCM_PLAMC_REG(base)                      ((base)->PLAMC)
#define MCM_CR_REG(base)                         ((base)->CR)
#define MCM_PID_REG(base)                        ((base)->PID)

/**
 * @}
 */ /* end of group MCM_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- MCM Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup MCM_Register_Masks MCM Register Masks
 * @{
 */

/* PLASC Bit Fields */
#define MCM_PLASC_ASC_MASK                       0xFFu
#define MCM_PLASC_ASC_SHIFT                      0
#define MCM_PLASC_ASC(x)                         (((uint16_t)(((uint16_t)(x))<<MCM_PLASC_ASC_SHIFT))&MCM_PLASC_ASC_MASK)
/* PLAMC Bit Fields */
#define MCM_PLAMC_AMC_MASK                       0xFFu
#define MCM_PLAMC_AMC_SHIFT                      0
#define MCM_PLAMC_AMC(x)                         (((uint16_t)(((uint16_t)(x))<<MCM_PLAMC_AMC_SHIFT))&MCM_PLAMC_AMC_MASK)
/* CR Bit Fields */
#define MCM_CR_CBRR_MASK                         0x200u
#define MCM_CR_CBRR_SHIFT                        9
#define MCM_CR_SRAMUAP_MASK                      0x3000000u
#define MCM_CR_SRAMUAP_SHIFT                     24
#define MCM_CR_SRAMUAP(x)                        (((uint32_t)(((uint32_t)(x))<<MCM_CR_SRAMUAP_SHIFT))&MCM_CR_SRAMUAP_MASK)
#define MCM_CR_SRAMUWP_MASK                      0x4000000u
#define MCM_CR_SRAMUWP_SHIFT                     26
#define MCM_CR_SRAMLAP_MASK                      0x30000000u
#define MCM_CR_SRAMLAP_SHIFT                     28
#define MCM_CR_SRAMLAP(x)                        (((uint32_t)(((uint32_t)(x))<<MCM_CR_SRAMLAP_SHIFT))&MCM_CR_SRAMLAP_MASK)
#define MCM_CR_SRAMLWP_MASK                      0x40000000u
#define MCM_CR_SRAMLWP_SHIFT                     30
/* PID Bit Fields */
#define MCM_PID_PID_MASK                         0xFFu
#define MCM_PID_PID_SHIFT                        0
#define MCM_PID_PID(x)                           (((uint32_t)(((uint32_t)(x))<<MCM_PID_PID_SHIFT))&MCM_PID_PID_MASK)

/**
 * @}
 */ /* end of group MCM_Register_Masks */


/* MCM - Peripheral instance base addresses */
/** Peripheral MCM base pointer */
#define MCM_BASE_PTR                             ((MCM_MemMapPtr)0xE0080000u)

/* ----------------------------------------------------------------------------
   -- MCM - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup MCM_Register_Accessor_Macros MCM - Register accessor macros
 * @{
 */


/* MCM - Register instance definitions */
/* MCM */
#define MCM_PLASC                                MCM_PLASC_REG(MCM_BASE_PTR)
#define MCM_PLAMC                                MCM_PLAMC_REG(MCM_BASE_PTR)
#define MCM_CR                                   MCM_CR_REG(MCM_BASE_PTR)
#define MCM_PID                                  MCM_PID_REG(MCM_BASE_PTR)

/**
 * @}
 */ /* end of group MCM_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group MCM_Peripheral */


/* ----------------------------------------------------------------------------
   -- MPU
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup MPU_Peripheral MPU
 * @{
 */

/** MPU - Peripheral register structure */
typedef struct MPU_MemMap {
  uint32_t CESR;                                   /**< Control/Error Status Register, offset: 0x0 */
  uint8_t RESERVED_0[12];
  struct {                                         /* offset: 0x10, array step: 0x8 */
    uint32_t EAR;                                    /**< Error Address Register, slave port n, array offset: 0x10, array step: 0x8 */
    uint32_t EDR;                                    /**< Error Detail Register, slave port n, array offset: 0x14, array step: 0x8 */
  } SP[5];
  uint8_t RESERVED_1[968];
  uint32_t WORD[8][4];                             /**< Region Descriptor n, Word 0..Region Descriptor n, Word 3, array offset: 0x400, array step: index*0x10, index2*0x4 */
  uint8_t RESERVED_2[896];
  uint32_t RGDAAC[8];                              /**< Region Descriptor Alternate Access Control n, array offset: 0x800, array step: 0x4 */
} volatile *MPU_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- MPU - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup MPU_Register_Accessor_Macros MPU - Register accessor macros
 * @{
 */


/* MPU - Register accessors */
#define MPU_CESR_REG(base)                       ((base)->CESR)
#define MPU_EAR_REG(base,index)                  ((base)->SP[index].EAR)
#define MPU_EDR_REG(base,index)                  ((base)->SP[index].EDR)
#define MPU_WORD_REG(base,index,index2)          ((base)->WORD[index][index2])
#define MPU_RGDAAC_REG(base,index)               ((base)->RGDAAC[index])

/**
 * @}
 */ /* end of group MPU_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- MPU Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup MPU_Register_Masks MPU Register Masks
 * @{
 */

/* CESR Bit Fields */
#define MPU_CESR_VLD_MASK                        0x1u
#define MPU_CESR_VLD_SHIFT                       0
#define MPU_CESR_NRGD_MASK                       0xF00u
#define MPU_CESR_NRGD_SHIFT                      8
#define MPU_CESR_NRGD(x)                         (((uint32_t)(((uint32_t)(x))<<MPU_CESR_NRGD_SHIFT))&MPU_CESR_NRGD_MASK)
#define MPU_CESR_NSP_MASK                        0xF000u
#define MPU_CESR_NSP_SHIFT                       12
#define MPU_CESR_NSP(x)                          (((uint32_t)(((uint32_t)(x))<<MPU_CESR_NSP_SHIFT))&MPU_CESR_NSP_MASK)
#define MPU_CESR_HRL_MASK                        0xF0000u
#define MPU_CESR_HRL_SHIFT                       16
#define MPU_CESR_HRL(x)                          (((uint32_t)(((uint32_t)(x))<<MPU_CESR_HRL_SHIFT))&MPU_CESR_HRL_MASK)
#define MPU_CESR_SPERR_MASK                      0xF8000000u
#define MPU_CESR_SPERR_SHIFT                     27
#define MPU_CESR_SPERR(x)                        (((uint32_t)(((uint32_t)(x))<<MPU_CESR_SPERR_SHIFT))&MPU_CESR_SPERR_MASK)
/* EAR Bit Fields */
#define MPU_EAR_EADDR_MASK                       0xFFFFFFFFu
#define MPU_EAR_EADDR_SHIFT                      0
#define MPU_EAR_EADDR(x)                         (((uint32_t)(((uint32_t)(x))<<MPU_EAR_EADDR_SHIFT))&MPU_EAR_EADDR_MASK)
/* EDR Bit Fields */
#define MPU_EDR_ERW_MASK                         0x1u
#define MPU_EDR_ERW_SHIFT                        0
#define MPU_EDR_EATTR_MASK                       0xEu
#define MPU_EDR_EATTR_SHIFT                      1
#define MPU_EDR_EATTR(x)                         (((uint32_t)(((uint32_t)(x))<<MPU_EDR_EATTR_SHIFT))&MPU_EDR_EATTR_MASK)
#define MPU_EDR_EMN_MASK                         0xF0u
#define MPU_EDR_EMN_SHIFT                        4
#define MPU_EDR_EMN(x)                           (((uint32_t)(((uint32_t)(x))<<MPU_EDR_EMN_SHIFT))&MPU_EDR_EMN_MASK)
#define MPU_EDR_EACD_MASK                        0xFFFF0000u
#define MPU_EDR_EACD_SHIFT                       16
#define MPU_EDR_EACD(x)                          (((uint32_t)(((uint32_t)(x))<<MPU_EDR_EACD_SHIFT))&MPU_EDR_EACD_MASK)
/* WORD Bit Fields */
#define MPU_WORD_M0UM_MASK                       0x7u
#define MPU_WORD_M0UM_SHIFT                      0
#define MPU_WORD_M0UM(x)                         (((uint32_t)(((uint32_t)(x))<<MPU_WORD_M0UM_SHIFT))&MPU_WORD_M0UM_MASK)
#define MPU_WORD_VLD_MASK                        0x1u
#define MPU_WORD_VLD_SHIFT                       0
#define MPU_WORD_M0SM_MASK                       0x18u
#define MPU_WORD_M0SM_SHIFT                      3
#define MPU_WORD_M0SM(x)                         (((uint32_t)(((uint32_t)(x))<<MPU_WORD_M0SM_SHIFT))&MPU_WORD_M0SM_MASK)
#define MPU_WORD_ENDADDR_MASK                    0xFFFFFFE0u
#define MPU_WORD_ENDADDR_SHIFT                   5
#define MPU_WORD_ENDADDR(x)                      (((uint32_t)(((uint32_t)(x))<<MPU_WORD_ENDADDR_SHIFT))&MPU_WORD_ENDADDR_MASK)
#define MPU_WORD_SRTADDR_MASK                    0xFFFFFFE0u
#define MPU_WORD_SRTADDR_SHIFT                   5
#define MPU_WORD_SRTADDR(x)                      (((uint32_t)(((uint32_t)(x))<<MPU_WORD_SRTADDR_SHIFT))&MPU_WORD_SRTADDR_MASK)
#define MPU_WORD_M1UM_MASK                       0x1C0u
#define MPU_WORD_M1UM_SHIFT                      6
#define MPU_WORD_M1UM(x)                         (((uint32_t)(((uint32_t)(x))<<MPU_WORD_M1UM_SHIFT))&MPU_WORD_M1UM_MASK)
#define MPU_WORD_M1SM_MASK                       0x600u
#define MPU_WORD_M1SM_SHIFT                      9
#define MPU_WORD_M1SM(x)                         (((uint32_t)(((uint32_t)(x))<<MPU_WORD_M1SM_SHIFT))&MPU_WORD_M1SM_MASK)
#define MPU_WORD_M2UM_MASK                       0x7000u
#define MPU_WORD_M2UM_SHIFT                      12
#define MPU_WORD_M2UM(x)                         (((uint32_t)(((uint32_t)(x))<<MPU_WORD_M2UM_SHIFT))&MPU_WORD_M2UM_MASK)
#define MPU_WORD_M2SM_MASK                       0x18000u
#define MPU_WORD_M2SM_SHIFT                      15
#define MPU_WORD_M2SM(x)                         (((uint32_t)(((uint32_t)(x))<<MPU_WORD_M2SM_SHIFT))&MPU_WORD_M2SM_MASK)
#define MPU_WORD_M3UM_MASK                       0x1C0000u
#define MPU_WORD_M3UM_SHIFT                      18
#define MPU_WORD_M3UM(x)                         (((uint32_t)(((uint32_t)(x))<<MPU_WORD_M3UM_SHIFT))&MPU_WORD_M3UM_MASK)
#define MPU_WORD_M3SM_MASK                       0x600000u
#define MPU_WORD_M3SM_SHIFT                      21
#define MPU_WORD_M3SM(x)                         (((uint32_t)(((uint32_t)(x))<<MPU_WORD_M3SM_SHIFT))&MPU_WORD_M3SM_MASK)
#define MPU_WORD_M4WE_MASK                       0x1000000u
#define MPU_WORD_M4WE_SHIFT                      24
#define MPU_WORD_M4RE_MASK                       0x2000000u
#define MPU_WORD_M4RE_SHIFT                      25
#define MPU_WORD_M5WE_MASK                       0x4000000u
#define MPU_WORD_M5WE_SHIFT                      26
#define MPU_WORD_M5RE_MASK                       0x8000000u
#define MPU_WORD_M5RE_SHIFT                      27
#define MPU_WORD_M6WE_MASK                       0x10000000u
#define MPU_WORD_M6WE_SHIFT                      28
#define MPU_WORD_M6RE_MASK                       0x20000000u
#define MPU_WORD_M6RE_SHIFT                      29
#define MPU_WORD_M7WE_MASK                       0x40000000u
#define MPU_WORD_M7WE_SHIFT                      30
#define MPU_WORD_M7RE_MASK                       0x80000000u
#define MPU_WORD_M7RE_SHIFT                      31
/* RGDAAC Bit Fields */
#define MPU_RGDAAC_M0UM_MASK                     0x7u
#define MPU_RGDAAC_M0UM_SHIFT                    0
#define MPU_RGDAAC_M0UM(x)                       (((uint32_t)(((uint32_t)(x))<<MPU_RGDAAC_M0UM_SHIFT))&MPU_RGDAAC_M0UM_MASK)
#define MPU_RGDAAC_M0SM_MASK                     0x18u
#define MPU_RGDAAC_M0SM_SHIFT                    3
#define MPU_RGDAAC_M0SM(x)                       (((uint32_t)(((uint32_t)(x))<<MPU_RGDAAC_M0SM_SHIFT))&MPU_RGDAAC_M0SM_MASK)
#define MPU_RGDAAC_M1UM_MASK                     0x1C0u
#define MPU_RGDAAC_M1UM_SHIFT                    6
#define MPU_RGDAAC_M1UM(x)                       (((uint32_t)(((uint32_t)(x))<<MPU_RGDAAC_M1UM_SHIFT))&MPU_RGDAAC_M1UM_MASK)
#define MPU_RGDAAC_M1SM_MASK                     0x600u
#define MPU_RGDAAC_M1SM_SHIFT                    9
#define MPU_RGDAAC_M1SM(x)                       (((uint32_t)(((uint32_t)(x))<<MPU_RGDAAC_M1SM_SHIFT))&MPU_RGDAAC_M1SM_MASK)
#define MPU_RGDAAC_M2UM_MASK                     0x7000u
#define MPU_RGDAAC_M2UM_SHIFT                    12
#define MPU_RGDAAC_M2UM(x)                       (((uint32_t)(((uint32_t)(x))<<MPU_RGDAAC_M2UM_SHIFT))&MPU_RGDAAC_M2UM_MASK)
#define MPU_RGDAAC_M2SM_MASK                     0x18000u
#define MPU_RGDAAC_M2SM_SHIFT                    15
#define MPU_RGDAAC_M2SM(x)                       (((uint32_t)(((uint32_t)(x))<<MPU_RGDAAC_M2SM_SHIFT))&MPU_RGDAAC_M2SM_MASK)
#define MPU_RGDAAC_M3UM_MASK                     0x1C0000u
#define MPU_RGDAAC_M3UM_SHIFT                    18
#define MPU_RGDAAC_M3UM(x)                       (((uint32_t)(((uint32_t)(x))<<MPU_RGDAAC_M3UM_SHIFT))&MPU_RGDAAC_M3UM_MASK)
#define MPU_RGDAAC_M3SM_MASK                     0x600000u
#define MPU_RGDAAC_M3SM_SHIFT                    21
#define MPU_RGDAAC_M3SM(x)                       (((uint32_t)(((uint32_t)(x))<<MPU_RGDAAC_M3SM_SHIFT))&MPU_RGDAAC_M3SM_MASK)
#define MPU_RGDAAC_M4WE_MASK                     0x1000000u
#define MPU_RGDAAC_M4WE_SHIFT                    24
#define MPU_RGDAAC_M4RE_MASK                     0x2000000u
#define MPU_RGDAAC_M4RE_SHIFT                    25
#define MPU_RGDAAC_M5WE_MASK                     0x4000000u
#define MPU_RGDAAC_M5WE_SHIFT                    26
#define MPU_RGDAAC_M5RE_MASK                     0x8000000u
#define MPU_RGDAAC_M5RE_SHIFT                    27
#define MPU_RGDAAC_M6WE_MASK                     0x10000000u
#define MPU_RGDAAC_M6WE_SHIFT                    28
#define MPU_RGDAAC_M6RE_MASK                     0x20000000u
#define MPU_RGDAAC_M6RE_SHIFT                    29
#define MPU_RGDAAC_M7WE_MASK                     0x40000000u
#define MPU_RGDAAC_M7WE_SHIFT                    30
#define MPU_RGDAAC_M7RE_MASK                     0x80000000u
#define MPU_RGDAAC_M7RE_SHIFT                    31

/**
 * @}
 */ /* end of group MPU_Register_Masks */


/* MPU - Peripheral instance base addresses */
/** Peripheral MPU base pointer */
#define MPU_BASE_PTR                             ((MPU_MemMapPtr)0x4000D000u)

/* ----------------------------------------------------------------------------
   -- MPU - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup MPU_Register_Accessor_Macros MPU - Register accessor macros
 * @{
 */


/* MPU - Register instance definitions */
/* MPU */
#define MPU_CESR                                 MPU_CESR_REG(MPU_BASE_PTR)
#define MPU_EAR0                                 MPU_EAR_REG(MPU_BASE_PTR,0)
#define MPU_EDR0                                 MPU_EDR_REG(MPU_BASE_PTR,0)
#define MPU_EAR1                                 MPU_EAR_REG(MPU_BASE_PTR,1)
#define MPU_EDR1                                 MPU_EDR_REG(MPU_BASE_PTR,1)
#define MPU_EAR2                                 MPU_EAR_REG(MPU_BASE_PTR,2)
#define MPU_EDR2                                 MPU_EDR_REG(MPU_BASE_PTR,2)
#define MPU_EAR3                                 MPU_EAR_REG(MPU_BASE_PTR,3)
#define MPU_EDR3                                 MPU_EDR_REG(MPU_BASE_PTR,3)
#define MPU_EAR4                                 MPU_EAR_REG(MPU_BASE_PTR,4)
#define MPU_EDR4                                 MPU_EDR_REG(MPU_BASE_PTR,4)
#define MPU_RGD0_WORD0                           MPU_WORD_REG(MPU_BASE_PTR,0,0)
#define MPU_RGD0_WORD1                           MPU_WORD_REG(MPU_BASE_PTR,0,1)
#define MPU_RGD0_WORD2                           MPU_WORD_REG(MPU_BASE_PTR,0,2)
#define MPU_RGD0_WORD3                           MPU_WORD_REG(MPU_BASE_PTR,0,3)
#define MPU_RGD1_WORD0                           MPU_WORD_REG(MPU_BASE_PTR,1,0)
#define MPU_RGD1_WORD1                           MPU_WORD_REG(MPU_BASE_PTR,1,1)
#define MPU_RGD1_WORD2                           MPU_WORD_REG(MPU_BASE_PTR,1,2)
#define MPU_RGD1_WORD3                           MPU_WORD_REG(MPU_BASE_PTR,1,3)
#define MPU_RGD2_WORD0                           MPU_WORD_REG(MPU_BASE_PTR,2,0)
#define MPU_RGD2_WORD1                           MPU_WORD_REG(MPU_BASE_PTR,2,1)
#define MPU_RGD2_WORD2                           MPU_WORD_REG(MPU_BASE_PTR,2,2)
#define MPU_RGD2_WORD3                           MPU_WORD_REG(MPU_BASE_PTR,2,3)
#define MPU_RGD3_WORD0                           MPU_WORD_REG(MPU_BASE_PTR,3,0)
#define MPU_RGD3_WORD1                           MPU_WORD_REG(MPU_BASE_PTR,3,1)
#define MPU_RGD3_WORD2                           MPU_WORD_REG(MPU_BASE_PTR,3,2)
#define MPU_RGD3_WORD3                           MPU_WORD_REG(MPU_BASE_PTR,3,3)
#define MPU_RGD4_WORD0                           MPU_WORD_REG(MPU_BASE_PTR,4,0)
#define MPU_RGD4_WORD1                           MPU_WORD_REG(MPU_BASE_PTR,4,1)
#define MPU_RGD4_WORD2                           MPU_WORD_REG(MPU_BASE_PTR,4,2)
#define MPU_RGD4_WORD3                           MPU_WORD_REG(MPU_BASE_PTR,4,3)
#define MPU_RGD5_WORD0                           MPU_WORD_REG(MPU_BASE_PTR,5,0)
#define MPU_RGD5_WORD1                           MPU_WORD_REG(MPU_BASE_PTR,5,1)
#define MPU_RGD5_WORD2                           MPU_WORD_REG(MPU_BASE_PTR,5,2)
#define MPU_RGD5_WORD3                           MPU_WORD_REG(MPU_BASE_PTR,5,3)
#define MPU_RGD6_WORD0                           MPU_WORD_REG(MPU_BASE_PTR,6,0)
#define MPU_RGD6_WORD1                           MPU_WORD_REG(MPU_BASE_PTR,6,1)
#define MPU_RGD6_WORD2                           MPU_WORD_REG(MPU_BASE_PTR,6,2)
#define MPU_RGD6_WORD3                           MPU_WORD_REG(MPU_BASE_PTR,6,3)
#define MPU_RGD7_WORD0                           MPU_WORD_REG(MPU_BASE_PTR,7,0)
#define MPU_RGD7_WORD1                           MPU_WORD_REG(MPU_BASE_PTR,7,1)
#define MPU_RGD7_WORD2                           MPU_WORD_REG(MPU_BASE_PTR,7,2)
#define MPU_RGD7_WORD3                           MPU_WORD_REG(MPU_BASE_PTR,7,3)
#define MPU_RGDAAC0                              MPU_RGDAAC_REG(MPU_BASE_PTR,0)
#define MPU_RGDAAC1                              MPU_RGDAAC_REG(MPU_BASE_PTR,1)
#define MPU_RGDAAC2                              MPU_RGDAAC_REG(MPU_BASE_PTR,2)
#define MPU_RGDAAC3                              MPU_RGDAAC_REG(MPU_BASE_PTR,3)
#define MPU_RGDAAC4                              MPU_RGDAAC_REG(MPU_BASE_PTR,4)
#define MPU_RGDAAC5                              MPU_RGDAAC_REG(MPU_BASE_PTR,5)
#define MPU_RGDAAC6                              MPU_RGDAAC_REG(MPU_BASE_PTR,6)
#define MPU_RGDAAC7                              MPU_RGDAAC_REG(MPU_BASE_PTR,7)

/* MPU - Register array accessors */
#define MPU_EAR(index)                           MPU_EAR_REG(MPU_BASE_PTR,index)
#define MPU_EDR(index)                           MPU_EDR_REG(MPU_BASE_PTR,index)
#define MPU_WORD(index,index2)                   MPU_WORD_REG(MPU_BASE_PTR,index,index2)
#define MPU_RGDAAC(index)                        MPU_RGDAAC_REG(MPU_BASE_PTR,index)

/**
 * @}
 */ /* end of group MPU_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group MPU_Peripheral */


/* ----------------------------------------------------------------------------
   -- NV
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup NV_Peripheral NV
 * @{
 */

/** NV - Peripheral register structure */
typedef struct NV_MemMap {
  uint8_t BACKKEY3;                                /**< Backdoor Comparison Key 3., offset: 0x0 */
  uint8_t BACKKEY2;                                /**< Backdoor Comparison Key 2., offset: 0x1 */
  uint8_t BACKKEY1;                                /**< Backdoor Comparison Key 1., offset: 0x2 */
  uint8_t BACKKEY0;                                /**< Backdoor Comparison Key 0., offset: 0x3 */
  uint8_t BACKKEY7;                                /**< Backdoor Comparison Key 7., offset: 0x4 */
  uint8_t BACKKEY6;                                /**< Backdoor Comparison Key 6., offset: 0x5 */
  uint8_t BACKKEY5;                                /**< Backdoor Comparison Key 5., offset: 0x6 */
  uint8_t BACKKEY4;                                /**< Backdoor Comparison Key 4., offset: 0x7 */
  uint8_t FPROT3;                                  /**< Non-volatile P-Flash Protection 1 - Low Register, offset: 0x8 */
  uint8_t FPROT2;                                  /**< Non-volatile P-Flash Protection 1 - High Register, offset: 0x9 */
  uint8_t FPROT1;                                  /**< Non-volatile P-Flash Protection 0 - Low Register, offset: 0xA */
  uint8_t FPROT0;                                  /**< Non-volatile P-Flash Protection 0 - High Register, offset: 0xB */
  uint8_t FSEC;                                    /**< Non-volatile Flash Security Register, offset: 0xC */
  uint8_t FOPT;                                    /**< Non-volatile Flash Option Register, offset: 0xD */
  uint8_t FEPROT;                                  /**< Non-volatile EERAM Protection Register, offset: 0xE */
  uint8_t FDPROT;                                  /**< Non-volatile D-Flash Protection Register, offset: 0xF */
} volatile *NV_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- NV - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup NV_Register_Accessor_Macros NV - Register accessor macros
 * @{
 */


/* NV - Register accessors */
#define NV_BACKKEY3_REG(base)                    ((base)->BACKKEY3)
#define NV_BACKKEY2_REG(base)                    ((base)->BACKKEY2)
#define NV_BACKKEY1_REG(base)                    ((base)->BACKKEY1)
#define NV_BACKKEY0_REG(base)                    ((base)->BACKKEY0)
#define NV_BACKKEY7_REG(base)                    ((base)->BACKKEY7)
#define NV_BACKKEY6_REG(base)                    ((base)->BACKKEY6)
#define NV_BACKKEY5_REG(base)                    ((base)->BACKKEY5)
#define NV_BACKKEY4_REG(base)                    ((base)->BACKKEY4)
#define NV_FPROT3_REG(base)                      ((base)->FPROT3)
#define NV_FPROT2_REG(base)                      ((base)->FPROT2)
#define NV_FPROT1_REG(base)                      ((base)->FPROT1)
#define NV_FPROT0_REG(base)                      ((base)->FPROT0)
#define NV_FSEC_REG(base)                        ((base)->FSEC)
#define NV_FOPT_REG(base)                        ((base)->FOPT)
#define NV_FEPROT_REG(base)                      ((base)->FEPROT)
#define NV_FDPROT_REG(base)                      ((base)->FDPROT)

/**
 * @}
 */ /* end of group NV_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- NV Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup NV_Register_Masks NV Register Masks
 * @{
 */

/* BACKKEY3 Bit Fields */
#define NV_BACKKEY3_KEY_MASK                     0xFFu
#define NV_BACKKEY3_KEY_SHIFT                    0
#define NV_BACKKEY3_KEY(x)                       (((uint8_t)(((uint8_t)(x))<<NV_BACKKEY3_KEY_SHIFT))&NV_BACKKEY3_KEY_MASK)
/* BACKKEY2 Bit Fields */
#define NV_BACKKEY2_KEY_MASK                     0xFFu
#define NV_BACKKEY2_KEY_SHIFT                    0
#define NV_BACKKEY2_KEY(x)                       (((uint8_t)(((uint8_t)(x))<<NV_BACKKEY2_KEY_SHIFT))&NV_BACKKEY2_KEY_MASK)
/* BACKKEY1 Bit Fields */
#define NV_BACKKEY1_KEY_MASK                     0xFFu
#define NV_BACKKEY1_KEY_SHIFT                    0
#define NV_BACKKEY1_KEY(x)                       (((uint8_t)(((uint8_t)(x))<<NV_BACKKEY1_KEY_SHIFT))&NV_BACKKEY1_KEY_MASK)
/* BACKKEY0 Bit Fields */
#define NV_BACKKEY0_KEY_MASK                     0xFFu
#define NV_BACKKEY0_KEY_SHIFT                    0
#define NV_BACKKEY0_KEY(x)                       (((uint8_t)(((uint8_t)(x))<<NV_BACKKEY0_KEY_SHIFT))&NV_BACKKEY0_KEY_MASK)
/* BACKKEY7 Bit Fields */
#define NV_BACKKEY7_KEY_MASK                     0xFFu
#define NV_BACKKEY7_KEY_SHIFT                    0
#define NV_BACKKEY7_KEY(x)                       (((uint8_t)(((uint8_t)(x))<<NV_BACKKEY7_KEY_SHIFT))&NV_BACKKEY7_KEY_MASK)
/* BACKKEY6 Bit Fields */
#define NV_BACKKEY6_KEY_MASK                     0xFFu
#define NV_BACKKEY6_KEY_SHIFT                    0
#define NV_BACKKEY6_KEY(x)                       (((uint8_t)(((uint8_t)(x))<<NV_BACKKEY6_KEY_SHIFT))&NV_BACKKEY6_KEY_MASK)
/* BACKKEY5 Bit Fields */
#define NV_BACKKEY5_KEY_MASK                     0xFFu
#define NV_BACKKEY5_KEY_SHIFT                    0
#define NV_BACKKEY5_KEY(x)                       (((uint8_t)(((uint8_t)(x))<<NV_BACKKEY5_KEY_SHIFT))&NV_BACKKEY5_KEY_MASK)
/* BACKKEY4 Bit Fields */
#define NV_BACKKEY4_KEY_MASK                     0xFFu
#define NV_BACKKEY4_KEY_SHIFT                    0
#define NV_BACKKEY4_KEY(x)                       (((uint8_t)(((uint8_t)(x))<<NV_BACKKEY4_KEY_SHIFT))&NV_BACKKEY4_KEY_MASK)
/* FPROT3 Bit Fields */
#define NV_FPROT3_PROT_MASK                      0xFFu
#define NV_FPROT3_PROT_SHIFT                     0
#define NV_FPROT3_PROT(x)                        (((uint8_t)(((uint8_t)(x))<<NV_FPROT3_PROT_SHIFT))&NV_FPROT3_PROT_MASK)
/* FPROT2 Bit Fields */
#define NV_FPROT2_PROT_MASK                      0xFFu
#define NV_FPROT2_PROT_SHIFT                     0
#define NV_FPROT2_PROT(x)                        (((uint8_t)(((uint8_t)(x))<<NV_FPROT2_PROT_SHIFT))&NV_FPROT2_PROT_MASK)
/* FPROT1 Bit Fields */
#define NV_FPROT1_PROT_MASK                      0xFFu
#define NV_FPROT1_PROT_SHIFT                     0
#define NV_FPROT1_PROT(x)                        (((uint8_t)(((uint8_t)(x))<<NV_FPROT1_PROT_SHIFT))&NV_FPROT1_PROT_MASK)
/* FPROT0 Bit Fields */
#define NV_FPROT0_PROT_MASK                      0xFFu
#define NV_FPROT0_PROT_SHIFT                     0
#define NV_FPROT0_PROT(x)                        (((uint8_t)(((uint8_t)(x))<<NV_FPROT0_PROT_SHIFT))&NV_FPROT0_PROT_MASK)
/* FSEC Bit Fields */
#define NV_FSEC_SEC_MASK                         0x3u
#define NV_FSEC_SEC_SHIFT                        0
#define NV_FSEC_SEC(x)                           (((uint8_t)(((uint8_t)(x))<<NV_FSEC_SEC_SHIFT))&NV_FSEC_SEC_MASK)
#define NV_FSEC_FSLACC_MASK                      0xCu
#define NV_FSEC_FSLACC_SHIFT                     2
#define NV_FSEC_FSLACC(x)                        (((uint8_t)(((uint8_t)(x))<<NV_FSEC_FSLACC_SHIFT))&NV_FSEC_FSLACC_MASK)
#define NV_FSEC_MEEN_MASK                        0x30u
#define NV_FSEC_MEEN_SHIFT                       4
#define NV_FSEC_MEEN(x)                          (((uint8_t)(((uint8_t)(x))<<NV_FSEC_MEEN_SHIFT))&NV_FSEC_MEEN_MASK)
#define NV_FSEC_KEYEN_MASK                       0xC0u
#define NV_FSEC_KEYEN_SHIFT                      6
#define NV_FSEC_KEYEN(x)                         (((uint8_t)(((uint8_t)(x))<<NV_FSEC_KEYEN_SHIFT))&NV_FSEC_KEYEN_MASK)
/* FOPT Bit Fields */
#define NV_FOPT_LPBOOT_MASK                      0x1u
#define NV_FOPT_LPBOOT_SHIFT                     0
#define NV_FOPT_EZPORT_DIS_MASK                  0x2u
#define NV_FOPT_EZPORT_DIS_SHIFT                 1
/* FEPROT Bit Fields */
#define NV_FEPROT_EPROT_MASK                     0xFFu
#define NV_FEPROT_EPROT_SHIFT                    0
#define NV_FEPROT_EPROT(x)                       (((uint8_t)(((uint8_t)(x))<<NV_FEPROT_EPROT_SHIFT))&NV_FEPROT_EPROT_MASK)
/* FDPROT Bit Fields */
#define NV_FDPROT_DPROT_MASK                     0xFFu
#define NV_FDPROT_DPROT_SHIFT                    0
#define NV_FDPROT_DPROT(x)                       (((uint8_t)(((uint8_t)(x))<<NV_FDPROT_DPROT_SHIFT))&NV_FDPROT_DPROT_MASK)

/**
 * @}
 */ /* end of group NV_Register_Masks */


/* NV - Peripheral instance base addresses */
/** Peripheral FTFL_FlashConfig base pointer */
#define FTFL_FlashConfig_BASE_PTR                ((NV_MemMapPtr)0x400u)

/* ----------------------------------------------------------------------------
   -- NV - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup NV_Register_Accessor_Macros NV - Register accessor macros
 * @{
 */


/* NV - Register instance definitions */
/* FTFL_FlashConfig */
#define NV_BACKKEY3                              NV_BACKKEY3_REG(FTFL_FlashConfig_BASE_PTR)
#define NV_BACKKEY2                              NV_BACKKEY2_REG(FTFL_FlashConfig_BASE_PTR)
#define NV_BACKKEY1                              NV_BACKKEY1_REG(FTFL_FlashConfig_BASE_PTR)
#define NV_BACKKEY0                              NV_BACKKEY0_REG(FTFL_FlashConfig_BASE_PTR)
#define NV_BACKKEY7                              NV_BACKKEY7_REG(FTFL_FlashConfig_BASE_PTR)
#define NV_BACKKEY6                              NV_BACKKEY6_REG(FTFL_FlashConfig_BASE_PTR)
#define NV_BACKKEY5                              NV_BACKKEY5_REG(FTFL_FlashConfig_BASE_PTR)
#define NV_BACKKEY4                              NV_BACKKEY4_REG(FTFL_FlashConfig_BASE_PTR)
#define NV_FPROT3                                NV_FPROT3_REG(FTFL_FlashConfig_BASE_PTR)
#define NV_FPROT2                                NV_FPROT2_REG(FTFL_FlashConfig_BASE_PTR)
#define NV_FPROT1                                NV_FPROT1_REG(FTFL_FlashConfig_BASE_PTR)
#define NV_FPROT0                                NV_FPROT0_REG(FTFL_FlashConfig_BASE_PTR)
#define NV_FSEC                                  NV_FSEC_REG(FTFL_FlashConfig_BASE_PTR)
#define NV_FOPT                                  NV_FOPT_REG(FTFL_FlashConfig_BASE_PTR)
#define NV_FEPROT                                NV_FEPROT_REG(FTFL_FlashConfig_BASE_PTR)
#define NV_FDPROT                                NV_FDPROT_REG(FTFL_FlashConfig_BASE_PTR)

/**
 * @}
 */ /* end of group NV_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group NV_Peripheral */


/* ----------------------------------------------------------------------------
   -- NVIC
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup NVIC_Peripheral NVIC
 * @{
 */

/** NVIC - Peripheral register structure */
typedef struct NVIC_MemMap {
  uint32_t ISER[4];                                /**< Interrupt Set Enable Register n, array offset: 0x0, array step: 0x4 */
  uint8_t RESERVED_0[112];
  uint32_t ICER[4];                                /**< Interrupt Clear Enable Register n, array offset: 0x80, array step: 0x4 */
  uint8_t RESERVED_1[112];
  uint32_t ISPR[4];                                /**< Interrupt Set Pending Register n, array offset: 0x100, array step: 0x4 */
  uint8_t RESERVED_2[112];
  uint32_t ICPR[4];                                /**< Interrupt Clear Pending Register n, array offset: 0x180, array step: 0x4 */
  uint8_t RESERVED_3[112];
  uint32_t IABR[4];                                /**< Interrupt Active bit Register n, array offset: 0x200, array step: 0x4 */
  uint8_t RESERVED_4[240];
  uint8_t IP[106];                                 /**< Interrupt Priority Register n, array offset: 0x300, array step: 0x1 */
  uint8_t RESERVED_5[2710];
  uint32_t STIR[1];                                /**< Software Trigger Interrupt Register, array offset: 0xE00, array step: 0x4 */
} volatile *NVIC_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- NVIC - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup NVIC_Register_Accessor_Macros NVIC - Register accessor macros
 * @{
 */


/* NVIC - Register accessors */
#define NVIC_ISER_REG(base,index)                ((base)->ISER[index])
#define NVIC_ICER_REG(base,index)                ((base)->ICER[index])
#define NVIC_ISPR_REG(base,index)                ((base)->ISPR[index])
#define NVIC_ICPR_REG(base,index)                ((base)->ICPR[index])
#define NVIC_IABR_REG(base,index)                ((base)->IABR[index])
#define NVIC_IP_REG(base,index)                  ((base)->IP[index])
#define NVIC_STIR_REG(base,index)                ((base)->STIR[index])

/**
 * @}
 */ /* end of group NVIC_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- NVIC Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup NVIC_Register_Masks NVIC Register Masks
 * @{
 */

/* ISER Bit Fields */
#define NVIC_ISER_SETENA_MASK                    0xFFFFFFFFu
#define NVIC_ISER_SETENA_SHIFT                   0
#define NVIC_ISER_SETENA(x)                      (((uint32_t)(((uint32_t)(x))<<NVIC_ISER_SETENA_SHIFT))&NVIC_ISER_SETENA_MASK)
/* ICER Bit Fields */
#define NVIC_ICER_CLRENA_MASK                    0xFFFFFFFFu
#define NVIC_ICER_CLRENA_SHIFT                   0
#define NVIC_ICER_CLRENA(x)                      (((uint32_t)(((uint32_t)(x))<<NVIC_ICER_CLRENA_SHIFT))&NVIC_ICER_CLRENA_MASK)
/* ISPR Bit Fields */
#define NVIC_ISPR_SETPEND_MASK                   0xFFFFFFFFu
#define NVIC_ISPR_SETPEND_SHIFT                  0
#define NVIC_ISPR_SETPEND(x)                     (((uint32_t)(((uint32_t)(x))<<NVIC_ISPR_SETPEND_SHIFT))&NVIC_ISPR_SETPEND_MASK)
/* ICPR Bit Fields */
#define NVIC_ICPR_CLRPEND_MASK                   0xFFFFFFFFu
#define NVIC_ICPR_CLRPEND_SHIFT                  0
#define NVIC_ICPR_CLRPEND(x)                     (((uint32_t)(((uint32_t)(x))<<NVIC_ICPR_CLRPEND_SHIFT))&NVIC_ICPR_CLRPEND_MASK)
/* IABR Bit Fields */
#define NVIC_IABR_ACTIVE_MASK                    0xFFFFFFFFu
#define NVIC_IABR_ACTIVE_SHIFT                   0
#define NVIC_IABR_ACTIVE(x)                      (((uint32_t)(((uint32_t)(x))<<NVIC_IABR_ACTIVE_SHIFT))&NVIC_IABR_ACTIVE_MASK)
/* IP Bit Fields */
#define NVIC_IP_PRI0_MASK                        0xFFu
#define NVIC_IP_PRI0_SHIFT                       0
#define NVIC_IP_PRI0(x)                          (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI0_SHIFT))&NVIC_IP_PRI0_MASK)
#define NVIC_IP_PRI1_MASK                        0xFFu
#define NVIC_IP_PRI1_SHIFT                       0
#define NVIC_IP_PRI1(x)                          (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI1_SHIFT))&NVIC_IP_PRI1_MASK)
#define NVIC_IP_PRI2_MASK                        0xFFu
#define NVIC_IP_PRI2_SHIFT                       0
#define NVIC_IP_PRI2(x)                          (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI2_SHIFT))&NVIC_IP_PRI2_MASK)
#define NVIC_IP_PRI3_MASK                        0xFFu
#define NVIC_IP_PRI3_SHIFT                       0
#define NVIC_IP_PRI3(x)                          (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI3_SHIFT))&NVIC_IP_PRI3_MASK)
#define NVIC_IP_PRI4_MASK                        0xFFu
#define NVIC_IP_PRI4_SHIFT                       0
#define NVIC_IP_PRI4(x)                          (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI4_SHIFT))&NVIC_IP_PRI4_MASK)
#define NVIC_IP_PRI5_MASK                        0xFFu
#define NVIC_IP_PRI5_SHIFT                       0
#define NVIC_IP_PRI5(x)                          (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI5_SHIFT))&NVIC_IP_PRI5_MASK)
#define NVIC_IP_PRI6_MASK                        0xFFu
#define NVIC_IP_PRI6_SHIFT                       0
#define NVIC_IP_PRI6(x)                          (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI6_SHIFT))&NVIC_IP_PRI6_MASK)
#define NVIC_IP_PRI7_MASK                        0xFFu
#define NVIC_IP_PRI7_SHIFT                       0
#define NVIC_IP_PRI7(x)                          (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI7_SHIFT))&NVIC_IP_PRI7_MASK)
#define NVIC_IP_PRI8_MASK                        0xFFu
#define NVIC_IP_PRI8_SHIFT                       0
#define NVIC_IP_PRI8(x)                          (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI8_SHIFT))&NVIC_IP_PRI8_MASK)
#define NVIC_IP_PRI9_MASK                        0xFFu
#define NVIC_IP_PRI9_SHIFT                       0
#define NVIC_IP_PRI9(x)                          (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI9_SHIFT))&NVIC_IP_PRI9_MASK)
#define NVIC_IP_PRI10_MASK                       0xFFu
#define NVIC_IP_PRI10_SHIFT                      0
#define NVIC_IP_PRI10(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI10_SHIFT))&NVIC_IP_PRI10_MASK)
#define NVIC_IP_PRI11_MASK                       0xFFu
#define NVIC_IP_PRI11_SHIFT                      0
#define NVIC_IP_PRI11(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI11_SHIFT))&NVIC_IP_PRI11_MASK)
#define NVIC_IP_PRI12_MASK                       0xFFu
#define NVIC_IP_PRI12_SHIFT                      0
#define NVIC_IP_PRI12(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI12_SHIFT))&NVIC_IP_PRI12_MASK)
#define NVIC_IP_PRI13_MASK                       0xFFu
#define NVIC_IP_PRI13_SHIFT                      0
#define NVIC_IP_PRI13(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI13_SHIFT))&NVIC_IP_PRI13_MASK)
#define NVIC_IP_PRI14_MASK                       0xFFu
#define NVIC_IP_PRI14_SHIFT                      0
#define NVIC_IP_PRI14(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI14_SHIFT))&NVIC_IP_PRI14_MASK)
#define NVIC_IP_PRI15_MASK                       0xFFu
#define NVIC_IP_PRI15_SHIFT                      0
#define NVIC_IP_PRI15(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI15_SHIFT))&NVIC_IP_PRI15_MASK)
#define NVIC_IP_PRI16_MASK                       0xFFu
#define NVIC_IP_PRI16_SHIFT                      0
#define NVIC_IP_PRI16(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI16_SHIFT))&NVIC_IP_PRI16_MASK)
#define NVIC_IP_PRI17_MASK                       0xFFu
#define NVIC_IP_PRI17_SHIFT                      0
#define NVIC_IP_PRI17(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI17_SHIFT))&NVIC_IP_PRI17_MASK)
#define NVIC_IP_PRI18_MASK                       0xFFu
#define NVIC_IP_PRI18_SHIFT                      0
#define NVIC_IP_PRI18(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI18_SHIFT))&NVIC_IP_PRI18_MASK)
#define NVIC_IP_PRI19_MASK                       0xFFu
#define NVIC_IP_PRI19_SHIFT                      0
#define NVIC_IP_PRI19(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI19_SHIFT))&NVIC_IP_PRI19_MASK)
#define NVIC_IP_PRI20_MASK                       0xFFu
#define NVIC_IP_PRI20_SHIFT                      0
#define NVIC_IP_PRI20(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI20_SHIFT))&NVIC_IP_PRI20_MASK)
#define NVIC_IP_PRI21_MASK                       0xFFu
#define NVIC_IP_PRI21_SHIFT                      0
#define NVIC_IP_PRI21(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI21_SHIFT))&NVIC_IP_PRI21_MASK)
#define NVIC_IP_PRI22_MASK                       0xFFu
#define NVIC_IP_PRI22_SHIFT                      0
#define NVIC_IP_PRI22(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI22_SHIFT))&NVIC_IP_PRI22_MASK)
#define NVIC_IP_PRI23_MASK                       0xFFu
#define NVIC_IP_PRI23_SHIFT                      0
#define NVIC_IP_PRI23(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI23_SHIFT))&NVIC_IP_PRI23_MASK)
#define NVIC_IP_PRI24_MASK                       0xFFu
#define NVIC_IP_PRI24_SHIFT                      0
#define NVIC_IP_PRI24(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI24_SHIFT))&NVIC_IP_PRI24_MASK)
#define NVIC_IP_PRI25_MASK                       0xFFu
#define NVIC_IP_PRI25_SHIFT                      0
#define NVIC_IP_PRI25(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI25_SHIFT))&NVIC_IP_PRI25_MASK)
#define NVIC_IP_PRI26_MASK                       0xFFu
#define NVIC_IP_PRI26_SHIFT                      0
#define NVIC_IP_PRI26(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI26_SHIFT))&NVIC_IP_PRI26_MASK)
#define NVIC_IP_PRI27_MASK                       0xFFu
#define NVIC_IP_PRI27_SHIFT                      0
#define NVIC_IP_PRI27(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI27_SHIFT))&NVIC_IP_PRI27_MASK)
#define NVIC_IP_PRI28_MASK                       0xFFu
#define NVIC_IP_PRI28_SHIFT                      0
#define NVIC_IP_PRI28(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI28_SHIFT))&NVIC_IP_PRI28_MASK)
#define NVIC_IP_PRI29_MASK                       0xFFu
#define NVIC_IP_PRI29_SHIFT                      0
#define NVIC_IP_PRI29(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI29_SHIFT))&NVIC_IP_PRI29_MASK)
#define NVIC_IP_PRI30_MASK                       0xFFu
#define NVIC_IP_PRI30_SHIFT                      0
#define NVIC_IP_PRI30(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI30_SHIFT))&NVIC_IP_PRI30_MASK)
#define NVIC_IP_PRI31_MASK                       0xFFu
#define NVIC_IP_PRI31_SHIFT                      0
#define NVIC_IP_PRI31(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI31_SHIFT))&NVIC_IP_PRI31_MASK)
#define NVIC_IP_PRI32_MASK                       0xFFu
#define NVIC_IP_PRI32_SHIFT                      0
#define NVIC_IP_PRI32(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI32_SHIFT))&NVIC_IP_PRI32_MASK)
#define NVIC_IP_PRI33_MASK                       0xFFu
#define NVIC_IP_PRI33_SHIFT                      0
#define NVIC_IP_PRI33(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI33_SHIFT))&NVIC_IP_PRI33_MASK)
#define NVIC_IP_PRI34_MASK                       0xFFu
#define NVIC_IP_PRI34_SHIFT                      0
#define NVIC_IP_PRI34(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI34_SHIFT))&NVIC_IP_PRI34_MASK)
#define NVIC_IP_PRI35_MASK                       0xFFu
#define NVIC_IP_PRI35_SHIFT                      0
#define NVIC_IP_PRI35(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI35_SHIFT))&NVIC_IP_PRI35_MASK)
#define NVIC_IP_PRI36_MASK                       0xFFu
#define NVIC_IP_PRI36_SHIFT                      0
#define NVIC_IP_PRI36(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI36_SHIFT))&NVIC_IP_PRI36_MASK)
#define NVIC_IP_PRI37_MASK                       0xFFu
#define NVIC_IP_PRI37_SHIFT                      0
#define NVIC_IP_PRI37(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI37_SHIFT))&NVIC_IP_PRI37_MASK)
#define NVIC_IP_PRI38_MASK                       0xFFu
#define NVIC_IP_PRI38_SHIFT                      0
#define NVIC_IP_PRI38(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI38_SHIFT))&NVIC_IP_PRI38_MASK)
#define NVIC_IP_PRI39_MASK                       0xFFu
#define NVIC_IP_PRI39_SHIFT                      0
#define NVIC_IP_PRI39(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI39_SHIFT))&NVIC_IP_PRI39_MASK)
#define NVIC_IP_PRI40_MASK                       0xFFu
#define NVIC_IP_PRI40_SHIFT                      0
#define NVIC_IP_PRI40(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI40_SHIFT))&NVIC_IP_PRI40_MASK)
#define NVIC_IP_PRI41_MASK                       0xFFu
#define NVIC_IP_PRI41_SHIFT                      0
#define NVIC_IP_PRI41(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI41_SHIFT))&NVIC_IP_PRI41_MASK)
#define NVIC_IP_PRI42_MASK                       0xFFu
#define NVIC_IP_PRI42_SHIFT                      0
#define NVIC_IP_PRI42(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI42_SHIFT))&NVIC_IP_PRI42_MASK)
#define NVIC_IP_PRI43_MASK                       0xFFu
#define NVIC_IP_PRI43_SHIFT                      0
#define NVIC_IP_PRI43(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI43_SHIFT))&NVIC_IP_PRI43_MASK)
#define NVIC_IP_PRI44_MASK                       0xFFu
#define NVIC_IP_PRI44_SHIFT                      0
#define NVIC_IP_PRI44(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI44_SHIFT))&NVIC_IP_PRI44_MASK)
#define NVIC_IP_PRI45_MASK                       0xFFu
#define NVIC_IP_PRI45_SHIFT                      0
#define NVIC_IP_PRI45(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI45_SHIFT))&NVIC_IP_PRI45_MASK)
#define NVIC_IP_PRI46_MASK                       0xFFu
#define NVIC_IP_PRI46_SHIFT                      0
#define NVIC_IP_PRI46(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI46_SHIFT))&NVIC_IP_PRI46_MASK)
#define NVIC_IP_PRI47_MASK                       0xFFu
#define NVIC_IP_PRI47_SHIFT                      0
#define NVIC_IP_PRI47(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI47_SHIFT))&NVIC_IP_PRI47_MASK)
#define NVIC_IP_PRI48_MASK                       0xFFu
#define NVIC_IP_PRI48_SHIFT                      0
#define NVIC_IP_PRI48(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI48_SHIFT))&NVIC_IP_PRI48_MASK)
#define NVIC_IP_PRI49_MASK                       0xFFu
#define NVIC_IP_PRI49_SHIFT                      0
#define NVIC_IP_PRI49(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI49_SHIFT))&NVIC_IP_PRI49_MASK)
#define NVIC_IP_PRI50_MASK                       0xFFu
#define NVIC_IP_PRI50_SHIFT                      0
#define NVIC_IP_PRI50(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI50_SHIFT))&NVIC_IP_PRI50_MASK)
#define NVIC_IP_PRI51_MASK                       0xFFu
#define NVIC_IP_PRI51_SHIFT                      0
#define NVIC_IP_PRI51(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI51_SHIFT))&NVIC_IP_PRI51_MASK)
#define NVIC_IP_PRI52_MASK                       0xFFu
#define NVIC_IP_PRI52_SHIFT                      0
#define NVIC_IP_PRI52(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI52_SHIFT))&NVIC_IP_PRI52_MASK)
#define NVIC_IP_PRI53_MASK                       0xFFu
#define NVIC_IP_PRI53_SHIFT                      0
#define NVIC_IP_PRI53(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI53_SHIFT))&NVIC_IP_PRI53_MASK)
#define NVIC_IP_PRI54_MASK                       0xFFu
#define NVIC_IP_PRI54_SHIFT                      0
#define NVIC_IP_PRI54(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI54_SHIFT))&NVIC_IP_PRI54_MASK)
#define NVIC_IP_PRI55_MASK                       0xFFu
#define NVIC_IP_PRI55_SHIFT                      0
#define NVIC_IP_PRI55(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI55_SHIFT))&NVIC_IP_PRI55_MASK)
#define NVIC_IP_PRI56_MASK                       0xFFu
#define NVIC_IP_PRI56_SHIFT                      0
#define NVIC_IP_PRI56(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI56_SHIFT))&NVIC_IP_PRI56_MASK)
#define NVIC_IP_PRI57_MASK                       0xFFu
#define NVIC_IP_PRI57_SHIFT                      0
#define NVIC_IP_PRI57(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI57_SHIFT))&NVIC_IP_PRI57_MASK)
#define NVIC_IP_PRI58_MASK                       0xFFu
#define NVIC_IP_PRI58_SHIFT                      0
#define NVIC_IP_PRI58(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI58_SHIFT))&NVIC_IP_PRI58_MASK)
#define NVIC_IP_PRI59_MASK                       0xFFu
#define NVIC_IP_PRI59_SHIFT                      0
#define NVIC_IP_PRI59(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI59_SHIFT))&NVIC_IP_PRI59_MASK)
#define NVIC_IP_PRI60_MASK                       0xFFu
#define NVIC_IP_PRI60_SHIFT                      0
#define NVIC_IP_PRI60(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI60_SHIFT))&NVIC_IP_PRI60_MASK)
#define NVIC_IP_PRI61_MASK                       0xFFu
#define NVIC_IP_PRI61_SHIFT                      0
#define NVIC_IP_PRI61(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI61_SHIFT))&NVIC_IP_PRI61_MASK)
#define NVIC_IP_PRI62_MASK                       0xFFu
#define NVIC_IP_PRI62_SHIFT                      0
#define NVIC_IP_PRI62(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI62_SHIFT))&NVIC_IP_PRI62_MASK)
#define NVIC_IP_PRI63_MASK                       0xFFu
#define NVIC_IP_PRI63_SHIFT                      0
#define NVIC_IP_PRI63(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI63_SHIFT))&NVIC_IP_PRI63_MASK)
#define NVIC_IP_PRI64_MASK                       0xFFu
#define NVIC_IP_PRI64_SHIFT                      0
#define NVIC_IP_PRI64(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI64_SHIFT))&NVIC_IP_PRI64_MASK)
#define NVIC_IP_PRI65_MASK                       0xFFu
#define NVIC_IP_PRI65_SHIFT                      0
#define NVIC_IP_PRI65(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI65_SHIFT))&NVIC_IP_PRI65_MASK)
#define NVIC_IP_PRI66_MASK                       0xFFu
#define NVIC_IP_PRI66_SHIFT                      0
#define NVIC_IP_PRI66(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI66_SHIFT))&NVIC_IP_PRI66_MASK)
#define NVIC_IP_PRI67_MASK                       0xFFu
#define NVIC_IP_PRI67_SHIFT                      0
#define NVIC_IP_PRI67(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI67_SHIFT))&NVIC_IP_PRI67_MASK)
#define NVIC_IP_PRI68_MASK                       0xFFu
#define NVIC_IP_PRI68_SHIFT                      0
#define NVIC_IP_PRI68(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI68_SHIFT))&NVIC_IP_PRI68_MASK)
#define NVIC_IP_PRI69_MASK                       0xFFu
#define NVIC_IP_PRI69_SHIFT                      0
#define NVIC_IP_PRI69(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI69_SHIFT))&NVIC_IP_PRI69_MASK)
#define NVIC_IP_PRI70_MASK                       0xFFu
#define NVIC_IP_PRI70_SHIFT                      0
#define NVIC_IP_PRI70(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI70_SHIFT))&NVIC_IP_PRI70_MASK)
#define NVIC_IP_PRI71_MASK                       0xFFu
#define NVIC_IP_PRI71_SHIFT                      0
#define NVIC_IP_PRI71(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI71_SHIFT))&NVIC_IP_PRI71_MASK)
#define NVIC_IP_PRI72_MASK                       0xFFu
#define NVIC_IP_PRI72_SHIFT                      0
#define NVIC_IP_PRI72(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI72_SHIFT))&NVIC_IP_PRI72_MASK)
#define NVIC_IP_PRI73_MASK                       0xFFu
#define NVIC_IP_PRI73_SHIFT                      0
#define NVIC_IP_PRI73(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI73_SHIFT))&NVIC_IP_PRI73_MASK)
#define NVIC_IP_PRI74_MASK                       0xFFu
#define NVIC_IP_PRI74_SHIFT                      0
#define NVIC_IP_PRI74(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI74_SHIFT))&NVIC_IP_PRI74_MASK)
#define NVIC_IP_PRI75_MASK                       0xFFu
#define NVIC_IP_PRI75_SHIFT                      0
#define NVIC_IP_PRI75(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI75_SHIFT))&NVIC_IP_PRI75_MASK)
#define NVIC_IP_PRI76_MASK                       0xFFu
#define NVIC_IP_PRI76_SHIFT                      0
#define NVIC_IP_PRI76(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI76_SHIFT))&NVIC_IP_PRI76_MASK)
#define NVIC_IP_PRI77_MASK                       0xFFu
#define NVIC_IP_PRI77_SHIFT                      0
#define NVIC_IP_PRI77(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI77_SHIFT))&NVIC_IP_PRI77_MASK)
#define NVIC_IP_PRI78_MASK                       0xFFu
#define NVIC_IP_PRI78_SHIFT                      0
#define NVIC_IP_PRI78(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI78_SHIFT))&NVIC_IP_PRI78_MASK)
#define NVIC_IP_PRI79_MASK                       0xFFu
#define NVIC_IP_PRI79_SHIFT                      0
#define NVIC_IP_PRI79(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI79_SHIFT))&NVIC_IP_PRI79_MASK)
#define NVIC_IP_PRI80_MASK                       0xFFu
#define NVIC_IP_PRI80_SHIFT                      0
#define NVIC_IP_PRI80(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI80_SHIFT))&NVIC_IP_PRI80_MASK)
#define NVIC_IP_PRI81_MASK                       0xFFu
#define NVIC_IP_PRI81_SHIFT                      0
#define NVIC_IP_PRI81(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI81_SHIFT))&NVIC_IP_PRI81_MASK)
#define NVIC_IP_PRI82_MASK                       0xFFu
#define NVIC_IP_PRI82_SHIFT                      0
#define NVIC_IP_PRI82(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI82_SHIFT))&NVIC_IP_PRI82_MASK)
#define NVIC_IP_PRI83_MASK                       0xFFu
#define NVIC_IP_PRI83_SHIFT                      0
#define NVIC_IP_PRI83(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI83_SHIFT))&NVIC_IP_PRI83_MASK)
#define NVIC_IP_PRI84_MASK                       0xFFu
#define NVIC_IP_PRI84_SHIFT                      0
#define NVIC_IP_PRI84(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI84_SHIFT))&NVIC_IP_PRI84_MASK)
#define NVIC_IP_PRI85_MASK                       0xFFu
#define NVIC_IP_PRI85_SHIFT                      0
#define NVIC_IP_PRI85(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI85_SHIFT))&NVIC_IP_PRI85_MASK)
#define NVIC_IP_PRI86_MASK                       0xFFu
#define NVIC_IP_PRI86_SHIFT                      0
#define NVIC_IP_PRI86(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI86_SHIFT))&NVIC_IP_PRI86_MASK)
#define NVIC_IP_PRI87_MASK                       0xFFu
#define NVIC_IP_PRI87_SHIFT                      0
#define NVIC_IP_PRI87(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI87_SHIFT))&NVIC_IP_PRI87_MASK)
#define NVIC_IP_PRI88_MASK                       0xFFu
#define NVIC_IP_PRI88_SHIFT                      0
#define NVIC_IP_PRI88(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI88_SHIFT))&NVIC_IP_PRI88_MASK)
#define NVIC_IP_PRI89_MASK                       0xFFu
#define NVIC_IP_PRI89_SHIFT                      0
#define NVIC_IP_PRI89(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI89_SHIFT))&NVIC_IP_PRI89_MASK)
#define NVIC_IP_PRI90_MASK                       0xFFu
#define NVIC_IP_PRI90_SHIFT                      0
#define NVIC_IP_PRI90(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI90_SHIFT))&NVIC_IP_PRI90_MASK)
#define NVIC_IP_PRI91_MASK                       0xFFu
#define NVIC_IP_PRI91_SHIFT                      0
#define NVIC_IP_PRI91(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI91_SHIFT))&NVIC_IP_PRI91_MASK)
#define NVIC_IP_PRI92_MASK                       0xFFu
#define NVIC_IP_PRI92_SHIFT                      0
#define NVIC_IP_PRI92(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI92_SHIFT))&NVIC_IP_PRI92_MASK)
#define NVIC_IP_PRI93_MASK                       0xFFu
#define NVIC_IP_PRI93_SHIFT                      0
#define NVIC_IP_PRI93(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI93_SHIFT))&NVIC_IP_PRI93_MASK)
#define NVIC_IP_PRI94_MASK                       0xFFu
#define NVIC_IP_PRI94_SHIFT                      0
#define NVIC_IP_PRI94(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI94_SHIFT))&NVIC_IP_PRI94_MASK)
#define NVIC_IP_PRI95_MASK                       0xFFu
#define NVIC_IP_PRI95_SHIFT                      0
#define NVIC_IP_PRI95(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI95_SHIFT))&NVIC_IP_PRI95_MASK)
#define NVIC_IP_PRI96_MASK                       0xFFu
#define NVIC_IP_PRI96_SHIFT                      0
#define NVIC_IP_PRI96(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI96_SHIFT))&NVIC_IP_PRI96_MASK)
#define NVIC_IP_PRI97_MASK                       0xFFu
#define NVIC_IP_PRI97_SHIFT                      0
#define NVIC_IP_PRI97(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI97_SHIFT))&NVIC_IP_PRI97_MASK)
#define NVIC_IP_PRI98_MASK                       0xFFu
#define NVIC_IP_PRI98_SHIFT                      0
#define NVIC_IP_PRI98(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI98_SHIFT))&NVIC_IP_PRI98_MASK)
#define NVIC_IP_PRI99_MASK                       0xFFu
#define NVIC_IP_PRI99_SHIFT                      0
#define NVIC_IP_PRI99(x)                         (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI99_SHIFT))&NVIC_IP_PRI99_MASK)
#define NVIC_IP_PRI100_MASK                      0xFFu
#define NVIC_IP_PRI100_SHIFT                     0
#define NVIC_IP_PRI100(x)                        (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI100_SHIFT))&NVIC_IP_PRI100_MASK)
#define NVIC_IP_PRI101_MASK                      0xFFu
#define NVIC_IP_PRI101_SHIFT                     0
#define NVIC_IP_PRI101(x)                        (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI101_SHIFT))&NVIC_IP_PRI101_MASK)
#define NVIC_IP_PRI102_MASK                      0xFFu
#define NVIC_IP_PRI102_SHIFT                     0
#define NVIC_IP_PRI102(x)                        (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI102_SHIFT))&NVIC_IP_PRI102_MASK)
#define NVIC_IP_PRI103_MASK                      0xFFu
#define NVIC_IP_PRI103_SHIFT                     0
#define NVIC_IP_PRI103(x)                        (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI103_SHIFT))&NVIC_IP_PRI103_MASK)
#define NVIC_IP_PRI104_MASK                      0xFFu
#define NVIC_IP_PRI104_SHIFT                     0
#define NVIC_IP_PRI104(x)                        (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI104_SHIFT))&NVIC_IP_PRI104_MASK)
#define NVIC_IP_PRI105_MASK                      0xFFu
#define NVIC_IP_PRI105_SHIFT                     0
#define NVIC_IP_PRI105(x)                        (((uint8_t)(((uint8_t)(x))<<NVIC_IP_PRI105_SHIFT))&NVIC_IP_PRI105_MASK)
/* STIR Bit Fields */
#define NVIC_STIR_INTID_MASK                     0x1FFu
#define NVIC_STIR_INTID_SHIFT                    0
#define NVIC_STIR_INTID(x)                       (((uint32_t)(((uint32_t)(x))<<NVIC_STIR_INTID_SHIFT))&NVIC_STIR_INTID_MASK)

/**
 * @}
 */ /* end of group NVIC_Register_Masks */


/* NVIC - Peripheral instance base addresses */
/** Peripheral NVIC base pointer */
#define NVIC_BASE_PTR                            ((NVIC_MemMapPtr)0xE000E100u)

/* ----------------------------------------------------------------------------
   -- NVIC - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup NVIC_Register_Accessor_Macros NVIC - Register accessor macros
 * @{
 */


/* NVIC - Register instance definitions */
/* NVIC */
#define NVICISER0                                NVIC_ISER_REG(NVIC_BASE_PTR,0)
#define NVICISER1                                NVIC_ISER_REG(NVIC_BASE_PTR,1)
#define NVICISER2                                NVIC_ISER_REG(NVIC_BASE_PTR,2)
#define NVICISER3                                NVIC_ISER_REG(NVIC_BASE_PTR,3)
#define NVICICER0                                NVIC_ICER_REG(NVIC_BASE_PTR,0)
#define NVICICER1                                NVIC_ICER_REG(NVIC_BASE_PTR,1)
#define NVICICER2                                NVIC_ICER_REG(NVIC_BASE_PTR,2)
#define NVICICER3                                NVIC_ICER_REG(NVIC_BASE_PTR,3)
#define NVICISPR0                                NVIC_ISPR_REG(NVIC_BASE_PTR,0)
#define NVICISPR1                                NVIC_ISPR_REG(NVIC_BASE_PTR,1)
#define NVICISPR2                                NVIC_ISPR_REG(NVIC_BASE_PTR,2)
#define NVICISPR3                                NVIC_ISPR_REG(NVIC_BASE_PTR,3)
#define NVICICPR0                                NVIC_ICPR_REG(NVIC_BASE_PTR,0)
#define NVICICPR1                                NVIC_ICPR_REG(NVIC_BASE_PTR,1)
#define NVICICPR2                                NVIC_ICPR_REG(NVIC_BASE_PTR,2)
#define NVICICPR3                                NVIC_ICPR_REG(NVIC_BASE_PTR,3)
#define NVICIABR0                                NVIC_IABR_REG(NVIC_BASE_PTR,0)
#define NVICIABR1                                NVIC_IABR_REG(NVIC_BASE_PTR,1)
#define NVICIABR2                                NVIC_IABR_REG(NVIC_BASE_PTR,2)
#define NVICIABR3                                NVIC_IABR_REG(NVIC_BASE_PTR,3)
#define NVICIP0                                  NVIC_IP_REG(NVIC_BASE_PTR,0)
#define NVICIP1                                  NVIC_IP_REG(NVIC_BASE_PTR,1)
#define NVICIP2                                  NVIC_IP_REG(NVIC_BASE_PTR,2)
#define NVICIP3                                  NVIC_IP_REG(NVIC_BASE_PTR,3)
#define NVICIP4                                  NVIC_IP_REG(NVIC_BASE_PTR,4)
#define NVICIP5                                  NVIC_IP_REG(NVIC_BASE_PTR,5)
#define NVICIP6                                  NVIC_IP_REG(NVIC_BASE_PTR,6)
#define NVICIP7                                  NVIC_IP_REG(NVIC_BASE_PTR,7)
#define NVICIP8                                  NVIC_IP_REG(NVIC_BASE_PTR,8)
#define NVICIP9                                  NVIC_IP_REG(NVIC_BASE_PTR,9)
#define NVICIP10                                 NVIC_IP_REG(NVIC_BASE_PTR,10)
#define NVICIP11                                 NVIC_IP_REG(NVIC_BASE_PTR,11)
#define NVICIP12                                 NVIC_IP_REG(NVIC_BASE_PTR,12)
#define NVICIP13                                 NVIC_IP_REG(NVIC_BASE_PTR,13)
#define NVICIP14                                 NVIC_IP_REG(NVIC_BASE_PTR,14)
#define NVICIP15                                 NVIC_IP_REG(NVIC_BASE_PTR,15)
#define NVICIP16                                 NVIC_IP_REG(NVIC_BASE_PTR,16)
#define NVICIP17                                 NVIC_IP_REG(NVIC_BASE_PTR,17)
#define NVICIP18                                 NVIC_IP_REG(NVIC_BASE_PTR,18)
#define NVICIP19                                 NVIC_IP_REG(NVIC_BASE_PTR,19)
#define NVICIP20                                 NVIC_IP_REG(NVIC_BASE_PTR,20)
#define NVICIP21                                 NVIC_IP_REG(NVIC_BASE_PTR,21)
#define NVICIP22                                 NVIC_IP_REG(NVIC_BASE_PTR,22)
#define NVICIP23                                 NVIC_IP_REG(NVIC_BASE_PTR,23)
#define NVICIP24                                 NVIC_IP_REG(NVIC_BASE_PTR,24)
#define NVICIP25                                 NVIC_IP_REG(NVIC_BASE_PTR,25)
#define NVICIP26                                 NVIC_IP_REG(NVIC_BASE_PTR,26)
#define NVICIP27                                 NVIC_IP_REG(NVIC_BASE_PTR,27)
#define NVICIP28                                 NVIC_IP_REG(NVIC_BASE_PTR,28)
#define NVICIP29                                 NVIC_IP_REG(NVIC_BASE_PTR,29)
#define NVICIP30                                 NVIC_IP_REG(NVIC_BASE_PTR,30)
#define NVICIP31                                 NVIC_IP_REG(NVIC_BASE_PTR,31)
#define NVICIP32                                 NVIC_IP_REG(NVIC_BASE_PTR,32)
#define NVICIP33                                 NVIC_IP_REG(NVIC_BASE_PTR,33)
#define NVICIP34                                 NVIC_IP_REG(NVIC_BASE_PTR,34)
#define NVICIP35                                 NVIC_IP_REG(NVIC_BASE_PTR,35)
#define NVICIP36                                 NVIC_IP_REG(NVIC_BASE_PTR,36)
#define NVICIP37                                 NVIC_IP_REG(NVIC_BASE_PTR,37)
#define NVICIP38                                 NVIC_IP_REG(NVIC_BASE_PTR,38)
#define NVICIP39                                 NVIC_IP_REG(NVIC_BASE_PTR,39)
#define NVICIP40                                 NVIC_IP_REG(NVIC_BASE_PTR,40)
#define NVICIP41                                 NVIC_IP_REG(NVIC_BASE_PTR,41)
#define NVICIP42                                 NVIC_IP_REG(NVIC_BASE_PTR,42)
#define NVICIP43                                 NVIC_IP_REG(NVIC_BASE_PTR,43)
#define NVICIP44                                 NVIC_IP_REG(NVIC_BASE_PTR,44)
#define NVICIP45                                 NVIC_IP_REG(NVIC_BASE_PTR,45)
#define NVICIP46                                 NVIC_IP_REG(NVIC_BASE_PTR,46)
#define NVICIP47                                 NVIC_IP_REG(NVIC_BASE_PTR,47)
#define NVICIP48                                 NVIC_IP_REG(NVIC_BASE_PTR,48)
#define NVICIP49                                 NVIC_IP_REG(NVIC_BASE_PTR,49)
#define NVICIP50                                 NVIC_IP_REG(NVIC_BASE_PTR,50)
#define NVICIP51                                 NVIC_IP_REG(NVIC_BASE_PTR,51)
#define NVICIP52                                 NVIC_IP_REG(NVIC_BASE_PTR,52)
#define NVICIP53                                 NVIC_IP_REG(NVIC_BASE_PTR,53)
#define NVICIP54                                 NVIC_IP_REG(NVIC_BASE_PTR,54)
#define NVICIP55                                 NVIC_IP_REG(NVIC_BASE_PTR,55)
#define NVICIP56                                 NVIC_IP_REG(NVIC_BASE_PTR,56)
#define NVICIP57                                 NVIC_IP_REG(NVIC_BASE_PTR,57)
#define NVICIP58                                 NVIC_IP_REG(NVIC_BASE_PTR,58)
#define NVICIP59                                 NVIC_IP_REG(NVIC_BASE_PTR,59)
#define NVICIP60                                 NVIC_IP_REG(NVIC_BASE_PTR,60)
#define NVICIP61                                 NVIC_IP_REG(NVIC_BASE_PTR,61)
#define NVICIP62                                 NVIC_IP_REG(NVIC_BASE_PTR,62)
#define NVICIP63                                 NVIC_IP_REG(NVIC_BASE_PTR,63)
#define NVICIP64                                 NVIC_IP_REG(NVIC_BASE_PTR,64)
#define NVICIP65                                 NVIC_IP_REG(NVIC_BASE_PTR,65)
#define NVICIP66                                 NVIC_IP_REG(NVIC_BASE_PTR,66)
#define NVICIP67                                 NVIC_IP_REG(NVIC_BASE_PTR,67)
#define NVICIP68                                 NVIC_IP_REG(NVIC_BASE_PTR,68)
#define NVICIP69                                 NVIC_IP_REG(NVIC_BASE_PTR,69)
#define NVICIP70                                 NVIC_IP_REG(NVIC_BASE_PTR,70)
#define NVICIP71                                 NVIC_IP_REG(NVIC_BASE_PTR,71)
#define NVICIP72                                 NVIC_IP_REG(NVIC_BASE_PTR,72)
#define NVICIP73                                 NVIC_IP_REG(NVIC_BASE_PTR,73)
#define NVICIP74                                 NVIC_IP_REG(NVIC_BASE_PTR,74)
#define NVICIP75                                 NVIC_IP_REG(NVIC_BASE_PTR,75)
#define NVICIP76                                 NVIC_IP_REG(NVIC_BASE_PTR,76)
#define NVICIP77                                 NVIC_IP_REG(NVIC_BASE_PTR,77)
#define NVICIP78                                 NVIC_IP_REG(NVIC_BASE_PTR,78)
#define NVICIP79                                 NVIC_IP_REG(NVIC_BASE_PTR,79)
#define NVICIP80                                 NVIC_IP_REG(NVIC_BASE_PTR,80)
#define NVICIP81                                 NVIC_IP_REG(NVIC_BASE_PTR,81)
#define NVICIP82                                 NVIC_IP_REG(NVIC_BASE_PTR,82)
#define NVICIP83                                 NVIC_IP_REG(NVIC_BASE_PTR,83)
#define NVICIP84                                 NVIC_IP_REG(NVIC_BASE_PTR,84)
#define NVICIP85                                 NVIC_IP_REG(NVIC_BASE_PTR,85)
#define NVICIP86                                 NVIC_IP_REG(NVIC_BASE_PTR,86)
#define NVICIP87                                 NVIC_IP_REG(NVIC_BASE_PTR,87)
#define NVICIP88                                 NVIC_IP_REG(NVIC_BASE_PTR,88)
#define NVICIP89                                 NVIC_IP_REG(NVIC_BASE_PTR,89)
#define NVICIP90                                 NVIC_IP_REG(NVIC_BASE_PTR,90)
#define NVICIP91                                 NVIC_IP_REG(NVIC_BASE_PTR,91)
#define NVICIP92                                 NVIC_IP_REG(NVIC_BASE_PTR,92)
#define NVICIP93                                 NVIC_IP_REG(NVIC_BASE_PTR,93)
#define NVICIP94                                 NVIC_IP_REG(NVIC_BASE_PTR,94)
#define NVICIP95                                 NVIC_IP_REG(NVIC_BASE_PTR,95)
#define NVICIP96                                 NVIC_IP_REG(NVIC_BASE_PTR,96)
#define NVICIP97                                 NVIC_IP_REG(NVIC_BASE_PTR,97)
#define NVICIP98                                 NVIC_IP_REG(NVIC_BASE_PTR,98)
#define NVICIP99                                 NVIC_IP_REG(NVIC_BASE_PTR,99)
#define NVICIP100                                NVIC_IP_REG(NVIC_BASE_PTR,100)
#define NVICIP101                                NVIC_IP_REG(NVIC_BASE_PTR,101)
#define NVICIP102                                NVIC_IP_REG(NVIC_BASE_PTR,102)
#define NVICIP103                                NVIC_IP_REG(NVIC_BASE_PTR,103)
#define NVICIP104                                NVIC_IP_REG(NVIC_BASE_PTR,104)
#define NVICIP105                                NVIC_IP_REG(NVIC_BASE_PTR,105)
#define NVICSTIR                                 NVIC_STIR_REG(NVIC_BASE_PTR,0)

/* NVIC - Register array accessors */
#define NVIC_ISER(index)                         NVIC_ISER_REG(NVIC_BASE_PTR,index)
#define NVIC_ICER(index)                         NVIC_ICER_REG(NVIC_BASE_PTR,index)
#define NVIC_ISPR(index)                         NVIC_ISPR_REG(NVIC_BASE_PTR,index)
#define NVIC_ICPR(index)                         NVIC_ICPR_REG(NVIC_BASE_PTR,index)
#define NVIC_IABR(index)                         NVIC_IABR_REG(NVIC_BASE_PTR,index)
#define NVIC_IP(index)                           NVIC_IP_REG(NVIC_BASE_PTR,index)
#define NVIC_STIR(index)                         NVIC_STIR_REG(NVIC_BASE_PTR,index)

/**
 * @}
 */ /* end of group NVIC_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group NVIC_Peripheral */


/* ----------------------------------------------------------------------------
   -- OSC
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup OSC_Peripheral OSC
 * @{
 */

/** OSC - Peripheral register structure */
typedef struct OSC_MemMap {
  uint8_t CR;                                      /**< OSC Control Register, offset: 0x0 */
} volatile *OSC_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- OSC - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup OSC_Register_Accessor_Macros OSC - Register accessor macros
 * @{
 */


/* OSC - Register accessors */
#define OSC_CR_REG(base)                         ((base)->CR)

/**
 * @}
 */ /* end of group OSC_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- OSC Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup OSC_Register_Masks OSC Register Masks
 * @{
 */

/* CR Bit Fields */
#define OSC_CR_OSCINIT_MASK                      0x1u
#define OSC_CR_OSCINIT_SHIFT                     0
#define OSC_CR_HGO_MASK                          0x2u
#define OSC_CR_HGO_SHIFT                         1
#define OSC_CR_OSCOS_MASK                        0x10u
#define OSC_CR_OSCOS_SHIFT                       4
#define OSC_CR_OSCSTEN_MASK                      0x20u
#define OSC_CR_OSCSTEN_SHIFT                     5
#define OSC_CR_OSCEN_MASK                        0x80u
#define OSC_CR_OSCEN_SHIFT                       7

/**
 * @}
 */ /* end of group OSC_Register_Masks */


/* OSC - Peripheral instance base addresses */
/** Peripheral OSC base pointer */
#define OSC_BASE_PTR                             ((OSC_MemMapPtr)0x40045000u)

/* ----------------------------------------------------------------------------
   -- OSC - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup OSC_Register_Accessor_Macros OSC - Register accessor macros
 * @{
 */


/* OSC - Register instance definitions */
/* OSC */
#define OSC_CR                                   OSC_CR_REG(OSC_BASE_PTR)

/**
 * @}
 */ /* end of group OSC_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group OSC_Peripheral */


/* ----------------------------------------------------------------------------
   -- PDB
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup PDB_Peripheral PDB
 * @{
 */

/** PDB - Peripheral register structure */
typedef struct PDB_MemMap {
  uint32_t SC;                                     /**< Status and Control Register, offset: 0x0 */
  uint32_t MOD;                                    /**< Modulus Register, offset: 0x4 */
  uint32_t CNT;                                    /**< Counter Register, offset: 0x8 */
  uint32_t IDLY;                                   /**< Interrupt Delay Register, offset: 0xC */
  struct {                                         /* offset: 0x10, array step: 0x18 */
    uint32_t C1;                                     /**< Channel n Control Register 1, array offset: 0x10, array step: 0x18 */
    uint32_t S;                                      /**< Channel n Status Register, array offset: 0x14, array step: 0x18 */
    uint32_t DLY[4];                                 /**< Channel n Delay 0 Register..Channel n Delay 3 Register, array offset: 0x18, array step: index*0x18, index2*0x4 */
  } CH[1];
  uint8_t RESERVED_0[360];
  uint32_t POEN;                                   /**< Pulse-Out n Enable Register, offset: 0x190 */
  uint32_t PODLY;                                  /**< Pulse-Out n Delay Register, offset: 0x194 */
} volatile *PDB_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- PDB - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup PDB_Register_Accessor_Macros PDB - Register accessor macros
 * @{
 */


/* PDB - Register accessors */
#define PDB_SC_REG(base)                         ((base)->SC)
#define PDB_MOD_REG(base)                        ((base)->MOD)
#define PDB_CNT_REG(base)                        ((base)->CNT)
#define PDB_IDLY_REG(base)                       ((base)->IDLY)
#define PDB_C1_REG(base,index)                   ((base)->CH[index].C1)
#define PDB_S_REG(base,index)                    ((base)->CH[index].S)
#define PDB_DLY_REG(base,index,index2)           ((base)->CH[index].DLY[index2])
#define PDB_POEN_REG(base)                       ((base)->POEN)
#define PDB_PODLY_REG(base)                      ((base)->PODLY)

/**
 * @}
 */ /* end of group PDB_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- PDB Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup PDB_Register_Masks PDB Register Masks
 * @{
 */

/* SC Bit Fields */
#define PDB_SC_LDOK_MASK                         0x1u
#define PDB_SC_LDOK_SHIFT                        0
#define PDB_SC_CONT_MASK                         0x2u
#define PDB_SC_CONT_SHIFT                        1
#define PDB_SC_MULT_MASK                         0xCu
#define PDB_SC_MULT_SHIFT                        2
#define PDB_SC_MULT(x)                           (((uint32_t)(((uint32_t)(x))<<PDB_SC_MULT_SHIFT))&PDB_SC_MULT_MASK)
#define PDB_SC_PDBIE_MASK                        0x20u
#define PDB_SC_PDBIE_SHIFT                       5
#define PDB_SC_PDBIF_MASK                        0x40u
#define PDB_SC_PDBIF_SHIFT                       6
#define PDB_SC_PDBEN_MASK                        0x80u
#define PDB_SC_PDBEN_SHIFT                       7
#define PDB_SC_TRGSEL_MASK                       0xF00u
#define PDB_SC_TRGSEL_SHIFT                      8
#define PDB_SC_TRGSEL(x)                         (((uint32_t)(((uint32_t)(x))<<PDB_SC_TRGSEL_SHIFT))&PDB_SC_TRGSEL_MASK)
#define PDB_SC_PRESCALER_MASK                    0x7000u
#define PDB_SC_PRESCALER_SHIFT                   12
#define PDB_SC_PRESCALER(x)                      (((uint32_t)(((uint32_t)(x))<<PDB_SC_PRESCALER_SHIFT))&PDB_SC_PRESCALER_MASK)
#define PDB_SC_DMAEN_MASK                        0x8000u
#define PDB_SC_DMAEN_SHIFT                       15
#define PDB_SC_SWTRIG_MASK                       0x10000u
#define PDB_SC_SWTRIG_SHIFT                      16
#define PDB_SC_PDBEIE_MASK                       0x20000u
#define PDB_SC_PDBEIE_SHIFT                      17
#define PDB_SC_LDMOD_MASK                        0xC0000u
#define PDB_SC_LDMOD_SHIFT                       18
#define PDB_SC_LDMOD(x)                          (((uint32_t)(((uint32_t)(x))<<PDB_SC_LDMOD_SHIFT))&PDB_SC_LDMOD_MASK)
/* MOD Bit Fields */
#define PDB_MOD_MOD_MASK                         0xFFFFu
#define PDB_MOD_MOD_SHIFT                        0
#define PDB_MOD_MOD(x)                           (((uint32_t)(((uint32_t)(x))<<PDB_MOD_MOD_SHIFT))&PDB_MOD_MOD_MASK)
/* CNT Bit Fields */
#define PDB_CNT_CNT_MASK                         0xFFFFu
#define PDB_CNT_CNT_SHIFT                        0
#define PDB_CNT_CNT(x)                           (((uint32_t)(((uint32_t)(x))<<PDB_CNT_CNT_SHIFT))&PDB_CNT_CNT_MASK)
/* IDLY Bit Fields */
#define PDB_IDLY_IDLY_MASK                       0xFFFFu
#define PDB_IDLY_IDLY_SHIFT                      0
#define PDB_IDLY_IDLY(x)                         (((uint32_t)(((uint32_t)(x))<<PDB_IDLY_IDLY_SHIFT))&PDB_IDLY_IDLY_MASK)
/* C1 Bit Fields */
#define PDB_C1_EN_MASK                           0xFFu
#define PDB_C1_EN_SHIFT                          0
#define PDB_C1_EN(x)                             (((uint32_t)(((uint32_t)(x))<<PDB_C1_EN_SHIFT))&PDB_C1_EN_MASK)
#define PDB_C1_TOS_MASK                          0xFF00u
#define PDB_C1_TOS_SHIFT                         8
#define PDB_C1_TOS(x)                            (((uint32_t)(((uint32_t)(x))<<PDB_C1_TOS_SHIFT))&PDB_C1_TOS_MASK)
/* S Bit Fields */
#define PDB_S_ERR_MASK                           0xFFu
#define PDB_S_ERR_SHIFT                          0
#define PDB_S_ERR(x)                             (((uint32_t)(((uint32_t)(x))<<PDB_S_ERR_SHIFT))&PDB_S_ERR_MASK)
#define PDB_S_CF_MASK                            0xFF0000u
#define PDB_S_CF_SHIFT                           16
#define PDB_S_CF(x)                              (((uint32_t)(((uint32_t)(x))<<PDB_S_CF_SHIFT))&PDB_S_CF_MASK)
/* DLY Bit Fields */
#define PDB_DLY_DLY_MASK                         0xFFFFu
#define PDB_DLY_DLY_SHIFT                        0
#define PDB_DLY_DLY(x)                           (((uint32_t)(((uint32_t)(x))<<PDB_DLY_DLY_SHIFT))&PDB_DLY_DLY_MASK)
/* POEN Bit Fields */
#define PDB_POEN_POEN_MASK                       0xFFu
#define PDB_POEN_POEN_SHIFT                      0
#define PDB_POEN_POEN(x)                         (((uint32_t)(((uint32_t)(x))<<PDB_POEN_POEN_SHIFT))&PDB_POEN_POEN_MASK)
/* PODLY Bit Fields */
#define PDB_PODLY_DLY2_MASK                      0xFFFFu
#define PDB_PODLY_DLY2_SHIFT                     0
#define PDB_PODLY_DLY2(x)                        (((uint32_t)(((uint32_t)(x))<<PDB_PODLY_DLY2_SHIFT))&PDB_PODLY_DLY2_MASK)
#define PDB_PODLY_DLY1_MASK                      0xFFFF0000u
#define PDB_PODLY_DLY1_SHIFT                     16
#define PDB_PODLY_DLY1(x)                        (((uint32_t)(((uint32_t)(x))<<PDB_PODLY_DLY1_SHIFT))&PDB_PODLY_DLY1_MASK)

/**
 * @}
 */ /* end of group PDB_Register_Masks */


/* PDB - Peripheral instance base addresses */
/** Peripheral PDB0 base pointer */
#define PDB0_BASE_PTR                            ((PDB_MemMapPtr)0x40025000u)
/** Peripheral PDB1 base pointer */
#define PDB1_BASE_PTR                            ((PDB_MemMapPtr)0x40026000u)
/** Peripheral PDB2 base pointer */
#define PDB2_BASE_PTR                            ((PDB_MemMapPtr)0x40027000u)
/** Peripheral PDB3 base pointer */
#define PDB3_BASE_PTR                            ((PDB_MemMapPtr)0x40028000u)

/* ----------------------------------------------------------------------------
   -- PDB - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup PDB_Register_Accessor_Macros PDB - Register accessor macros
 * @{
 */


/* PDB - Register instance definitions */
/* PDB0 */
#define PDB0_SC                                  PDB_SC_REG(PDB0_BASE_PTR)
#define PDB0_MOD                                 PDB_MOD_REG(PDB0_BASE_PTR)
#define PDB0_CNT                                 PDB_CNT_REG(PDB0_BASE_PTR)
#define PDB0_IDLY                                PDB_IDLY_REG(PDB0_BASE_PTR)
#define PDB0_CH0C1                               PDB_C1_REG(PDB0_BASE_PTR,0)
#define PDB0_CH0S                                PDB_S_REG(PDB0_BASE_PTR,0)
#define PDB0_CH0DLY0                             PDB_DLY_REG(PDB0_BASE_PTR,0,0)
#define PDB0_CH0DLY1                             PDB_DLY_REG(PDB0_BASE_PTR,0,1)
#define PDB0_CH0DLY2                             PDB_DLY_REG(PDB0_BASE_PTR,0,2)
#define PDB0_CH0DLY3                             PDB_DLY_REG(PDB0_BASE_PTR,0,3)
#define PDB0_POEN                                PDB_POEN_REG(PDB0_BASE_PTR)
#define PDB0_PO0DLY                              PDB_PODLY_REG(PDB0_BASE_PTR)
/* PDB1 */
#define PDB1_SC                                  PDB_SC_REG(PDB1_BASE_PTR)
#define PDB1_MOD                                 PDB_MOD_REG(PDB1_BASE_PTR)
#define PDB1_CNT                                 PDB_CNT_REG(PDB1_BASE_PTR)
#define PDB1_IDLY                                PDB_IDLY_REG(PDB1_BASE_PTR)
#define PDB1_CH0C1                               PDB_C1_REG(PDB1_BASE_PTR,0)
#define PDB1_CH0S                                PDB_S_REG(PDB1_BASE_PTR,0)
#define PDB1_CH0DLY0                             PDB_DLY_REG(PDB1_BASE_PTR,0,0)
#define PDB1_CH0DLY1                             PDB_DLY_REG(PDB1_BASE_PTR,0,1)
#define PDB1_CH0DLY2                             PDB_DLY_REG(PDB1_BASE_PTR,0,2)
#define PDB1_CH0DLY3                             PDB_DLY_REG(PDB1_BASE_PTR,0,3)
#define PDB1_POEN                                PDB_POEN_REG(PDB1_BASE_PTR)
#define PDB1_PO0DLY                              PDB_PODLY_REG(PDB1_BASE_PTR)
/* PDB2 */
#define PDB2_SC                                  PDB_SC_REG(PDB2_BASE_PTR)
#define PDB2_MOD                                 PDB_MOD_REG(PDB2_BASE_PTR)
#define PDB2_CNT                                 PDB_CNT_REG(PDB2_BASE_PTR)
#define PDB2_IDLY                                PDB_IDLY_REG(PDB2_BASE_PTR)
#define PDB2_CH0C1                               PDB_C1_REG(PDB2_BASE_PTR,0)
#define PDB2_CH0S                                PDB_S_REG(PDB2_BASE_PTR,0)
#define PDB2_CH0DLY0                             PDB_DLY_REG(PDB2_BASE_PTR,0,0)
#define PDB2_CH0DLY1                             PDB_DLY_REG(PDB2_BASE_PTR,0,1)
#define PDB2_CH0DLY2                             PDB_DLY_REG(PDB2_BASE_PTR,0,2)
#define PDB2_CH0DLY3                             PDB_DLY_REG(PDB2_BASE_PTR,0,3)
#define PDB2_POEN                                PDB_POEN_REG(PDB2_BASE_PTR)
#define PDB2_PO0DLY                              PDB_PODLY_REG(PDB2_BASE_PTR)
/* PDB3 */
#define PDB3_SC                                  PDB_SC_REG(PDB3_BASE_PTR)
#define PDB3_MOD                                 PDB_MOD_REG(PDB3_BASE_PTR)
#define PDB3_CNT                                 PDB_CNT_REG(PDB3_BASE_PTR)
#define PDB3_IDLY                                PDB_IDLY_REG(PDB3_BASE_PTR)
#define PDB3_CH0C1                               PDB_C1_REG(PDB3_BASE_PTR,0)
#define PDB3_CH0S                                PDB_S_REG(PDB3_BASE_PTR,0)
#define PDB3_CH0DLY0                             PDB_DLY_REG(PDB3_BASE_PTR,0,0)
#define PDB3_CH0DLY1                             PDB_DLY_REG(PDB3_BASE_PTR,0,1)
#define PDB3_CH0DLY2                             PDB_DLY_REG(PDB3_BASE_PTR,0,2)
#define PDB3_CH0DLY3                             PDB_DLY_REG(PDB3_BASE_PTR,0,3)
#define PDB3_POEN                                PDB_POEN_REG(PDB3_BASE_PTR)
#define PDB3_PO0DLY                              PDB_PODLY_REG(PDB3_BASE_PTR)

/* PDB - Register array accessors */
#define PDB0_C1(index)                           PDB_C1_REG(PDB0_BASE_PTR,index)
#define PDB1_C1(index)                           PDB_C1_REG(PDB1_BASE_PTR,index)
#define PDB2_C1(index)                           PDB_C1_REG(PDB2_BASE_PTR,index)
#define PDB3_C1(index)                           PDB_C1_REG(PDB3_BASE_PTR,index)
#define PDB0_S(index)                            PDB_S_REG(PDB0_BASE_PTR,index)
#define PDB1_S(index)                            PDB_S_REG(PDB1_BASE_PTR,index)
#define PDB2_S(index)                            PDB_S_REG(PDB2_BASE_PTR,index)
#define PDB3_S(index)                            PDB_S_REG(PDB3_BASE_PTR,index)
#define PDB0_DLY(index,index2)                   PDB_DLY_REG(PDB0_BASE_PTR,index,index2)
#define PDB1_DLY(index,index2)                   PDB_DLY_REG(PDB1_BASE_PTR,index,index2)
#define PDB2_DLY(index,index2)                   PDB_DLY_REG(PDB2_BASE_PTR,index,index2)
#define PDB3_DLY(index,index2)                   PDB_DLY_REG(PDB3_BASE_PTR,index,index2)

/**
 * @}
 */ /* end of group PDB_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group PDB_Peripheral */


/* ----------------------------------------------------------------------------
   -- PIT
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup PIT_Peripheral PIT
 * @{
 */

/** PIT - Peripheral register structure */
typedef struct PIT_MemMap {
  uint32_t MCR;                                    /**< PIT Module Control Register, offset: 0x0 */
  uint8_t RESERVED_0[220];
  uint32_t LTMR64H;                                /**< PIT Upper Lifetime Timer Register, offset: 0xE0 */
  uint32_t LTMR64L;                                /**< PIT Lower Lifetime Timer Register, offset: 0xE4 */
  uint8_t RESERVED_1[24];
  struct {                                         /* offset: 0x100, array step: 0x10 */
    uint32_t LDVAL;                                  /**< Timer Load Value Register, array offset: 0x100, array step: 0x10 */
    uint32_t CVAL;                                   /**< Current Timer Value Register, array offset: 0x104, array step: 0x10 */
    uint32_t TCTRL;                                  /**< Timer Control Register, array offset: 0x108, array step: 0x10 */
    uint32_t TFLG;                                   /**< Timer Flag Register, array offset: 0x10C, array step: 0x10 */
  } CHANNEL[4];
} volatile *PIT_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- PIT - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup PIT_Register_Accessor_Macros PIT - Register accessor macros
 * @{
 */


/* PIT - Register accessors */
#define PIT_MCR_REG(base)                        ((base)->MCR)
#define PIT_LTMR64H_REG(base)                    ((base)->LTMR64H)
#define PIT_LTMR64L_REG(base)                    ((base)->LTMR64L)
#define PIT_LDVAL_REG(base,index)                ((base)->CHANNEL[index].LDVAL)
#define PIT_CVAL_REG(base,index)                 ((base)->CHANNEL[index].CVAL)
#define PIT_TCTRL_REG(base,index)                ((base)->CHANNEL[index].TCTRL)
#define PIT_TFLG_REG(base,index)                 ((base)->CHANNEL[index].TFLG)

/**
 * @}
 */ /* end of group PIT_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- PIT Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup PIT_Register_Masks PIT Register Masks
 * @{
 */

/* MCR Bit Fields */
#define PIT_MCR_FRZ_MASK                         0x1u
#define PIT_MCR_FRZ_SHIFT                        0
#define PIT_MCR_MDIS_MASK                        0x2u
#define PIT_MCR_MDIS_SHIFT                       1
/* LTMR64H Bit Fields */
#define PIT_LTMR64H_LTH_MASK                     0xFFFFFFFFu
#define PIT_LTMR64H_LTH_SHIFT                    0
#define PIT_LTMR64H_LTH(x)                       (((uint32_t)(((uint32_t)(x))<<PIT_LTMR64H_LTH_SHIFT))&PIT_LTMR64H_LTH_MASK)
/* LTMR64L Bit Fields */
#define PIT_LTMR64L_LTL_MASK                     0xFFFFFFFFu
#define PIT_LTMR64L_LTL_SHIFT                    0
#define PIT_LTMR64L_LTL(x)                       (((uint32_t)(((uint32_t)(x))<<PIT_LTMR64L_LTL_SHIFT))&PIT_LTMR64L_LTL_MASK)
/* LDVAL Bit Fields */
#define PIT_LDVAL_TSV_MASK                       0xFFFFFFFFu
#define PIT_LDVAL_TSV_SHIFT                      0
#define PIT_LDVAL_TSV(x)                         (((uint32_t)(((uint32_t)(x))<<PIT_LDVAL_TSV_SHIFT))&PIT_LDVAL_TSV_MASK)
/* CVAL Bit Fields */
#define PIT_CVAL_TVL_MASK                        0xFFFFFFFFu
#define PIT_CVAL_TVL_SHIFT                       0
#define PIT_CVAL_TVL(x)                          (((uint32_t)(((uint32_t)(x))<<PIT_CVAL_TVL_SHIFT))&PIT_CVAL_TVL_MASK)
/* TCTRL Bit Fields */
#define PIT_TCTRL_TEN_MASK                       0x1u
#define PIT_TCTRL_TEN_SHIFT                      0
#define PIT_TCTRL_TIE_MASK                       0x2u
#define PIT_TCTRL_TIE_SHIFT                      1
#define PIT_TCTRL_CHN_MASK                       0x4u
#define PIT_TCTRL_CHN_SHIFT                      2
/* TFLG Bit Fields */
#define PIT_TFLG_TIF_MASK                        0x1u
#define PIT_TFLG_TIF_SHIFT                       0

/**
 * @}
 */ /* end of group PIT_Register_Masks */


/* PIT - Peripheral instance base addresses */
/** Peripheral PIT base pointer */
#define PIT_BASE_PTR                             ((PIT_MemMapPtr)0x40023000u)

/* ----------------------------------------------------------------------------
   -- PIT - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup PIT_Register_Accessor_Macros PIT - Register accessor macros
 * @{
 */


/* PIT - Register instance definitions */
/* PIT */
#define PIT_MCR                                  PIT_MCR_REG(PIT_BASE_PTR)
#define PIT_LTMR64H                              PIT_LTMR64H_REG(PIT_BASE_PTR)
#define PIT_LTMR64L                              PIT_LTMR64L_REG(PIT_BASE_PTR)
#define PIT_LDVAL0                               PIT_LDVAL_REG(PIT_BASE_PTR,0)
#define PIT_CVAL0                                PIT_CVAL_REG(PIT_BASE_PTR,0)
#define PIT_TCTRL0                               PIT_TCTRL_REG(PIT_BASE_PTR,0)
#define PIT_TFLG0                                PIT_TFLG_REG(PIT_BASE_PTR,0)
#define PIT_LDVAL1                               PIT_LDVAL_REG(PIT_BASE_PTR,1)
#define PIT_CVAL1                                PIT_CVAL_REG(PIT_BASE_PTR,1)
#define PIT_TCTRL1                               PIT_TCTRL_REG(PIT_BASE_PTR,1)
#define PIT_TFLG1                                PIT_TFLG_REG(PIT_BASE_PTR,1)
#define PIT_LDVAL2                               PIT_LDVAL_REG(PIT_BASE_PTR,2)
#define PIT_CVAL2                                PIT_CVAL_REG(PIT_BASE_PTR,2)
#define PIT_TCTRL2                               PIT_TCTRL_REG(PIT_BASE_PTR,2)
#define PIT_TFLG2                                PIT_TFLG_REG(PIT_BASE_PTR,2)
#define PIT_LDVAL3                               PIT_LDVAL_REG(PIT_BASE_PTR,3)
#define PIT_CVAL3                                PIT_CVAL_REG(PIT_BASE_PTR,3)
#define PIT_TCTRL3                               PIT_TCTRL_REG(PIT_BASE_PTR,3)
#define PIT_TFLG3                                PIT_TFLG_REG(PIT_BASE_PTR,3)

/* PIT - Register array accessors */
#define PIT_LDVAL(index)                         PIT_LDVAL_REG(PIT_BASE_PTR,index)
#define PIT_CVAL(index)                          PIT_CVAL_REG(PIT_BASE_PTR,index)
#define PIT_TCTRL(index)                         PIT_TCTRL_REG(PIT_BASE_PTR,index)
#define PIT_TFLG(index)                          PIT_TFLG_REG(PIT_BASE_PTR,index)

/**
 * @}
 */ /* end of group PIT_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group PIT_Peripheral */


/* ----------------------------------------------------------------------------
   -- PMC
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup PMC_Peripheral PMC
 * @{
 */

/** PMC - Peripheral register structure */
typedef struct PMC_MemMap {
  uint8_t SPMSC1;                                  /**< System Power Management Status and Control 1 Register, offset: 0x0 */
  uint8_t SPMSC2;                                  /**< System Power Management Status and Control 2 Register, offset: 0x1 */
} volatile *PMC_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- PMC - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup PMC_Register_Accessor_Macros PMC - Register accessor macros
 * @{
 */


/* PMC - Register accessors */
#define PMC_SPMSC1_REG(base)                     ((base)->SPMSC1)
#define PMC_SPMSC2_REG(base)                     ((base)->SPMSC2)

/**
 * @}
 */ /* end of group PMC_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- PMC Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup PMC_Register_Masks PMC Register Masks
 * @{
 */

/* SPMSC1 Bit Fields */
#define PMC_SPMSC1_BGBE_MASK                     0x1u
#define PMC_SPMSC1_BGBE_SHIFT                    0
#define PMC_SPMSC1_BGBDS_MASK                    0x2u
#define PMC_SPMSC1_BGBDS_SHIFT                   1
#define PMC_SPMSC1_LVDE_MASK                     0x4u
#define PMC_SPMSC1_LVDE_SHIFT                    2
#define PMC_SPMSC1_LVDSE_MASK                    0x8u
#define PMC_SPMSC1_LVDSE_SHIFT                   3
#define PMC_SPMSC1_LVDRE_MASK                    0x10u
#define PMC_SPMSC1_LVDRE_SHIFT                   4
#define PMC_SPMSC1_LVWIE_MASK                    0x20u
#define PMC_SPMSC1_LVWIE_SHIFT                   5
#define PMC_SPMSC1_LVWACK_MASK                   0x40u
#define PMC_SPMSC1_LVWACK_SHIFT                  6
#define PMC_SPMSC1_LVWF_MASK                     0x80u
#define PMC_SPMSC1_LVWF_SHIFT                    7
/* SPMSC2 Bit Fields */
#define PMC_SPMSC2_LVWV_MASK                     0x30u
#define PMC_SPMSC2_LVWV_SHIFT                    4
#define PMC_SPMSC2_LVWV(x)                       (((uint8_t)(((uint8_t)(x))<<PMC_SPMSC2_LVWV_SHIFT))&PMC_SPMSC2_LVWV_MASK)
#define PMC_SPMSC2_LVDV_MASK                     0x40u
#define PMC_SPMSC2_LVDV_SHIFT                    6

/**
 * @}
 */ /* end of group PMC_Register_Masks */


/* PMC - Peripheral instance base addresses */
/** Peripheral PMC base pointer */
#define PMC_BASE_PTR                             ((PMC_MemMapPtr)0x40054000u)

/* ----------------------------------------------------------------------------
   -- PMC - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup PMC_Register_Accessor_Macros PMC - Register accessor macros
 * @{
 */


/* PMC - Register instance definitions */
/* PMC */
#define PMC_SPMSC1                               PMC_SPMSC1_REG(PMC_BASE_PTR)
#define PMC_SPMSC2                               PMC_SPMSC2_REG(PMC_BASE_PTR)

/**
 * @}
 */ /* end of group PMC_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group PMC_Peripheral */


/* ----------------------------------------------------------------------------
   -- PORT
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup PORT_Peripheral PORT
 * @{
 */

/** PORT - Peripheral register structure */
typedef struct PORT_MemMap {
  uint32_t PCR[32];                                /**< Pin Control Register n, array offset: 0x0, array step: 0x4 */
  uint32_t GPCLR;                                  /**< Global Pin Control Low Register, offset: 0x80 */
  uint32_t GPCHR;                                  /**< Global Pin Control High Register, offset: 0x84 */
  uint8_t RESERVED_0[24];
  uint32_t ISFR;                                   /**< Interrupt Status Flag Register, offset: 0xA0 */
  uint8_t RESERVED_1[28];
  uint32_t DFER;                                   /**< Digital Filter Enable Register, offset: 0xC0 */
  uint32_t DFCR;                                   /**< Digital Filter Clock Register, offset: 0xC4 */
  uint32_t DFWR;                                   /**< Digital Filter Width Register, offset: 0xC8 */
} volatile *PORT_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- PORT - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup PORT_Register_Accessor_Macros PORT - Register accessor macros
 * @{
 */


/* PORT - Register accessors */
#define PORT_PCR_REG(base,index)                 ((base)->PCR[index])
#define PORT_GPCLR_REG(base)                     ((base)->GPCLR)
#define PORT_GPCHR_REG(base)                     ((base)->GPCHR)
#define PORT_ISFR_REG(base)                      ((base)->ISFR)
#define PORT_DFER_REG(base)                      ((base)->DFER)
#define PORT_DFCR_REG(base)                      ((base)->DFCR)
#define PORT_DFWR_REG(base)                      ((base)->DFWR)

/**
 * @}
 */ /* end of group PORT_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- PORT Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup PORT_Register_Masks PORT Register Masks
 * @{
 */

/* PCR Bit Fields */
#define PORT_PCR_PE_MASK                         0x2u
#define PORT_PCR_PE_SHIFT                        1
#define PORT_PCR_MUX_MASK                        0x700u
#define PORT_PCR_MUX_SHIFT                       8
#define PORT_PCR_MUX(x)                          (((uint32_t)(((uint32_t)(x))<<PORT_PCR_MUX_SHIFT))&PORT_PCR_MUX_MASK)
#define PORT_PCR_LK_MASK                         0x8000u
#define PORT_PCR_LK_SHIFT                        15
#define PORT_PCR_IRQC_MASK                       0xF0000u
#define PORT_PCR_IRQC_SHIFT                      16
#define PORT_PCR_IRQC(x)                         (((uint32_t)(((uint32_t)(x))<<PORT_PCR_IRQC_SHIFT))&PORT_PCR_IRQC_MASK)
#define PORT_PCR_ISF_MASK                        0x1000000u
#define PORT_PCR_ISF_SHIFT                       24
/* GPCLR Bit Fields */
#define PORT_GPCLR_GPWD_MASK                     0xFFFFu
#define PORT_GPCLR_GPWD_SHIFT                    0
#define PORT_GPCLR_GPWD(x)                       (((uint32_t)(((uint32_t)(x))<<PORT_GPCLR_GPWD_SHIFT))&PORT_GPCLR_GPWD_MASK)
#define PORT_GPCLR_GPWE_MASK                     0xFFFF0000u
#define PORT_GPCLR_GPWE_SHIFT                    16
#define PORT_GPCLR_GPWE(x)                       (((uint32_t)(((uint32_t)(x))<<PORT_GPCLR_GPWE_SHIFT))&PORT_GPCLR_GPWE_MASK)
/* GPCHR Bit Fields */
#define PORT_GPCHR_GPWD_MASK                     0xFFFFu
#define PORT_GPCHR_GPWD_SHIFT                    0
#define PORT_GPCHR_GPWD(x)                       (((uint32_t)(((uint32_t)(x))<<PORT_GPCHR_GPWD_SHIFT))&PORT_GPCHR_GPWD_MASK)
#define PORT_GPCHR_GPWE_MASK                     0xFFFF0000u
#define PORT_GPCHR_GPWE_SHIFT                    16
#define PORT_GPCHR_GPWE(x)                       (((uint32_t)(((uint32_t)(x))<<PORT_GPCHR_GPWE_SHIFT))&PORT_GPCHR_GPWE_MASK)
/* ISFR Bit Fields */
#define PORT_ISFR_ISF_MASK                       0xFFFFFFFFu
#define PORT_ISFR_ISF_SHIFT                      0
#define PORT_ISFR_ISF(x)                         (((uint32_t)(((uint32_t)(x))<<PORT_ISFR_ISF_SHIFT))&PORT_ISFR_ISF_MASK)
/* DFER Bit Fields */
#define PORT_DFER_DFE_MASK                       0xFFFFFFFFu
#define PORT_DFER_DFE_SHIFT                      0
#define PORT_DFER_DFE(x)                         (((uint32_t)(((uint32_t)(x))<<PORT_DFER_DFE_SHIFT))&PORT_DFER_DFE_MASK)
/* DFCR Bit Fields */
#define PORT_DFCR_CS_MASK                        0x1u
#define PORT_DFCR_CS_SHIFT                       0
/* DFWR Bit Fields */
#define PORT_DFWR_FILT_MASK                      0x1Fu
#define PORT_DFWR_FILT_SHIFT                     0
#define PORT_DFWR_FILT(x)                        (((uint32_t)(((uint32_t)(x))<<PORT_DFWR_FILT_SHIFT))&PORT_DFWR_FILT_MASK)

/**
 * @}
 */ /* end of group PORT_Register_Masks */


/* PORT - Peripheral instance base addresses */
/** Peripheral PORTA base pointer */
#define PORTA_BASE_PTR                           ((PORT_MemMapPtr)0x40033000u)
/** Peripheral PORTB base pointer */
#define PORTB_BASE_PTR                           ((PORT_MemMapPtr)0x40034000u)
/** Peripheral PORTC base pointer */
#define PORTC_BASE_PTR                           ((PORT_MemMapPtr)0x40035000u)
/** Peripheral PORTD base pointer */
#define PORTD_BASE_PTR                           ((PORT_MemMapPtr)0x40036000u)
/** Peripheral PORTE base pointer */
#define PORTE_BASE_PTR                           ((PORT_MemMapPtr)0x40037000u)

/* ----------------------------------------------------------------------------
   -- PORT - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup PORT_Register_Accessor_Macros PORT - Register accessor macros
 * @{
 */


/* PORT - Register instance definitions */
/* PORTA */
#define PORTA_PCR0                               PORT_PCR_REG(PORTA_BASE_PTR,0)
#define PORTA_PCR1                               PORT_PCR_REG(PORTA_BASE_PTR,1)
#define PORTA_PCR2                               PORT_PCR_REG(PORTA_BASE_PTR,2)
#define PORTA_PCR3                               PORT_PCR_REG(PORTA_BASE_PTR,3)
#define PORTA_PCR4                               PORT_PCR_REG(PORTA_BASE_PTR,4)
#define PORTA_PCR5                               PORT_PCR_REG(PORTA_BASE_PTR,5)
#define PORTA_PCR6                               PORT_PCR_REG(PORTA_BASE_PTR,6)
#define PORTA_PCR7                               PORT_PCR_REG(PORTA_BASE_PTR,7)
#define PORTA_PCR8                               PORT_PCR_REG(PORTA_BASE_PTR,8)
#define PORTA_PCR9                               PORT_PCR_REG(PORTA_BASE_PTR,9)
#define PORTA_PCR10                              PORT_PCR_REG(PORTA_BASE_PTR,10)
#define PORTA_PCR11                              PORT_PCR_REG(PORTA_BASE_PTR,11)
#define PORTA_PCR12                              PORT_PCR_REG(PORTA_BASE_PTR,12)
#define PORTA_PCR13                              PORT_PCR_REG(PORTA_BASE_PTR,13)
#define PORTA_PCR14                              PORT_PCR_REG(PORTA_BASE_PTR,14)
#define PORTA_PCR15                              PORT_PCR_REG(PORTA_BASE_PTR,15)
#define PORTA_PCR16                              PORT_PCR_REG(PORTA_BASE_PTR,16)
#define PORTA_PCR17                              PORT_PCR_REG(PORTA_BASE_PTR,17)
#define PORTA_PCR18                              PORT_PCR_REG(PORTA_BASE_PTR,18)
#define PORTA_PCR19                              PORT_PCR_REG(PORTA_BASE_PTR,19)
#define PORTA_PCR20                              PORT_PCR_REG(PORTA_BASE_PTR,20)
#define PORTA_PCR21                              PORT_PCR_REG(PORTA_BASE_PTR,21)
#define PORTA_PCR22                              PORT_PCR_REG(PORTA_BASE_PTR,22)
#define PORTA_PCR23                              PORT_PCR_REG(PORTA_BASE_PTR,23)
#define PORTA_PCR24                              PORT_PCR_REG(PORTA_BASE_PTR,24)
#define PORTA_PCR25                              PORT_PCR_REG(PORTA_BASE_PTR,25)
#define PORTA_PCR26                              PORT_PCR_REG(PORTA_BASE_PTR,26)
#define PORTA_PCR27                              PORT_PCR_REG(PORTA_BASE_PTR,27)
#define PORTA_PCR28                              PORT_PCR_REG(PORTA_BASE_PTR,28)
#define PORTA_PCR29                              PORT_PCR_REG(PORTA_BASE_PTR,29)
#define PORTA_PCR30                              PORT_PCR_REG(PORTA_BASE_PTR,30)
#define PORTA_PCR31                              PORT_PCR_REG(PORTA_BASE_PTR,31)
#define PORTA_GPCLR                              PORT_GPCLR_REG(PORTA_BASE_PTR)
#define PORTA_GPCHR                              PORT_GPCHR_REG(PORTA_BASE_PTR)
#define PORTA_ISFR                               PORT_ISFR_REG(PORTA_BASE_PTR)
#define PORTA_DFER                               PORT_DFER_REG(PORTA_BASE_PTR)
#define PORTA_DFCR                               PORT_DFCR_REG(PORTA_BASE_PTR)
#define PORTA_DFWR                               PORT_DFWR_REG(PORTA_BASE_PTR)
/* PORTB */
#define PORTB_PCR0                               PORT_PCR_REG(PORTB_BASE_PTR,0)
#define PORTB_PCR1                               PORT_PCR_REG(PORTB_BASE_PTR,1)
#define PORTB_PCR2                               PORT_PCR_REG(PORTB_BASE_PTR,2)
#define PORTB_PCR3                               PORT_PCR_REG(PORTB_BASE_PTR,3)
#define PORTB_PCR4                               PORT_PCR_REG(PORTB_BASE_PTR,4)
#define PORTB_PCR5                               PORT_PCR_REG(PORTB_BASE_PTR,5)
#define PORTB_PCR6                               PORT_PCR_REG(PORTB_BASE_PTR,6)
#define PORTB_PCR7                               PORT_PCR_REG(PORTB_BASE_PTR,7)
#define PORTB_PCR8                               PORT_PCR_REG(PORTB_BASE_PTR,8)
#define PORTB_PCR9                               PORT_PCR_REG(PORTB_BASE_PTR,9)
#define PORTB_PCR10                              PORT_PCR_REG(PORTB_BASE_PTR,10)
#define PORTB_PCR11                              PORT_PCR_REG(PORTB_BASE_PTR,11)
#define PORTB_PCR12                              PORT_PCR_REG(PORTB_BASE_PTR,12)
#define PORTB_PCR13                              PORT_PCR_REG(PORTB_BASE_PTR,13)
#define PORTB_PCR14                              PORT_PCR_REG(PORTB_BASE_PTR,14)
#define PORTB_PCR15                              PORT_PCR_REG(PORTB_BASE_PTR,15)
#define PORTB_PCR16                              PORT_PCR_REG(PORTB_BASE_PTR,16)
#define PORTB_PCR17                              PORT_PCR_REG(PORTB_BASE_PTR,17)
#define PORTB_PCR18                              PORT_PCR_REG(PORTB_BASE_PTR,18)
#define PORTB_PCR19                              PORT_PCR_REG(PORTB_BASE_PTR,19)
#define PORTB_PCR20                              PORT_PCR_REG(PORTB_BASE_PTR,20)
#define PORTB_PCR21                              PORT_PCR_REG(PORTB_BASE_PTR,21)
#define PORTB_PCR22                              PORT_PCR_REG(PORTB_BASE_PTR,22)
#define PORTB_PCR23                              PORT_PCR_REG(PORTB_BASE_PTR,23)
#define PORTB_PCR24                              PORT_PCR_REG(PORTB_BASE_PTR,24)
#define PORTB_PCR25                              PORT_PCR_REG(PORTB_BASE_PTR,25)
#define PORTB_PCR26                              PORT_PCR_REG(PORTB_BASE_PTR,26)
#define PORTB_PCR27                              PORT_PCR_REG(PORTB_BASE_PTR,27)
#define PORTB_PCR28                              PORT_PCR_REG(PORTB_BASE_PTR,28)
#define PORTB_PCR29                              PORT_PCR_REG(PORTB_BASE_PTR,29)
#define PORTB_PCR30                              PORT_PCR_REG(PORTB_BASE_PTR,30)
#define PORTB_PCR31                              PORT_PCR_REG(PORTB_BASE_PTR,31)
#define PORTB_GPCLR                              PORT_GPCLR_REG(PORTB_BASE_PTR)
#define PORTB_GPCHR                              PORT_GPCHR_REG(PORTB_BASE_PTR)
#define PORTB_ISFR                               PORT_ISFR_REG(PORTB_BASE_PTR)
#define PORTB_DFER                               PORT_DFER_REG(PORTB_BASE_PTR)
#define PORTB_DFCR                               PORT_DFCR_REG(PORTB_BASE_PTR)
#define PORTB_DFWR                               PORT_DFWR_REG(PORTB_BASE_PTR)
/* PORTC */
#define PORTC_PCR0                               PORT_PCR_REG(PORTC_BASE_PTR,0)
#define PORTC_PCR1                               PORT_PCR_REG(PORTC_BASE_PTR,1)
#define PORTC_PCR2                               PORT_PCR_REG(PORTC_BASE_PTR,2)
#define PORTC_PCR3                               PORT_PCR_REG(PORTC_BASE_PTR,3)
#define PORTC_PCR4                               PORT_PCR_REG(PORTC_BASE_PTR,4)
#define PORTC_PCR5                               PORT_PCR_REG(PORTC_BASE_PTR,5)
#define PORTC_PCR6                               PORT_PCR_REG(PORTC_BASE_PTR,6)
#define PORTC_PCR7                               PORT_PCR_REG(PORTC_BASE_PTR,7)
#define PORTC_PCR8                               PORT_PCR_REG(PORTC_BASE_PTR,8)
#define PORTC_PCR9                               PORT_PCR_REG(PORTC_BASE_PTR,9)
#define PORTC_PCR10                              PORT_PCR_REG(PORTC_BASE_PTR,10)
#define PORTC_PCR11                              PORT_PCR_REG(PORTC_BASE_PTR,11)
#define PORTC_PCR12                              PORT_PCR_REG(PORTC_BASE_PTR,12)
#define PORTC_PCR13                              PORT_PCR_REG(PORTC_BASE_PTR,13)
#define PORTC_PCR14                              PORT_PCR_REG(PORTC_BASE_PTR,14)
#define PORTC_PCR15                              PORT_PCR_REG(PORTC_BASE_PTR,15)
#define PORTC_PCR16                              PORT_PCR_REG(PORTC_BASE_PTR,16)
#define PORTC_PCR17                              PORT_PCR_REG(PORTC_BASE_PTR,17)
#define PORTC_PCR18                              PORT_PCR_REG(PORTC_BASE_PTR,18)
#define PORTC_PCR19                              PORT_PCR_REG(PORTC_BASE_PTR,19)
#define PORTC_PCR20                              PORT_PCR_REG(PORTC_BASE_PTR,20)
#define PORTC_PCR21                              PORT_PCR_REG(PORTC_BASE_PTR,21)
#define PORTC_PCR22                              PORT_PCR_REG(PORTC_BASE_PTR,22)
#define PORTC_PCR23                              PORT_PCR_REG(PORTC_BASE_PTR,23)
#define PORTC_PCR24                              PORT_PCR_REG(PORTC_BASE_PTR,24)
#define PORTC_PCR25                              PORT_PCR_REG(PORTC_BASE_PTR,25)
#define PORTC_PCR26                              PORT_PCR_REG(PORTC_BASE_PTR,26)
#define PORTC_PCR27                              PORT_PCR_REG(PORTC_BASE_PTR,27)
#define PORTC_PCR28                              PORT_PCR_REG(PORTC_BASE_PTR,28)
#define PORTC_PCR29                              PORT_PCR_REG(PORTC_BASE_PTR,29)
#define PORTC_PCR30                              PORT_PCR_REG(PORTC_BASE_PTR,30)
#define PORTC_PCR31                              PORT_PCR_REG(PORTC_BASE_PTR,31)
#define PORTC_GPCLR                              PORT_GPCLR_REG(PORTC_BASE_PTR)
#define PORTC_GPCHR                              PORT_GPCHR_REG(PORTC_BASE_PTR)
#define PORTC_ISFR                               PORT_ISFR_REG(PORTC_BASE_PTR)
#define PORTC_DFER                               PORT_DFER_REG(PORTC_BASE_PTR)
#define PORTC_DFCR                               PORT_DFCR_REG(PORTC_BASE_PTR)
#define PORTC_DFWR                               PORT_DFWR_REG(PORTC_BASE_PTR)
/* PORTD */
#define PORTD_PCR0                               PORT_PCR_REG(PORTD_BASE_PTR,0)
#define PORTD_PCR1                               PORT_PCR_REG(PORTD_BASE_PTR,1)
#define PORTD_PCR2                               PORT_PCR_REG(PORTD_BASE_PTR,2)
#define PORTD_PCR3                               PORT_PCR_REG(PORTD_BASE_PTR,3)
#define PORTD_PCR4                               PORT_PCR_REG(PORTD_BASE_PTR,4)
#define PORTD_PCR5                               PORT_PCR_REG(PORTD_BASE_PTR,5)
#define PORTD_PCR6                               PORT_PCR_REG(PORTD_BASE_PTR,6)
#define PORTD_PCR7                               PORT_PCR_REG(PORTD_BASE_PTR,7)
#define PORTD_PCR8                               PORT_PCR_REG(PORTD_BASE_PTR,8)
#define PORTD_PCR9                               PORT_PCR_REG(PORTD_BASE_PTR,9)
#define PORTD_PCR10                              PORT_PCR_REG(PORTD_BASE_PTR,10)
#define PORTD_PCR11                              PORT_PCR_REG(PORTD_BASE_PTR,11)
#define PORTD_PCR12                              PORT_PCR_REG(PORTD_BASE_PTR,12)
#define PORTD_PCR13                              PORT_PCR_REG(PORTD_BASE_PTR,13)
#define PORTD_PCR14                              PORT_PCR_REG(PORTD_BASE_PTR,14)
#define PORTD_PCR15                              PORT_PCR_REG(PORTD_BASE_PTR,15)
#define PORTD_PCR16                              PORT_PCR_REG(PORTD_BASE_PTR,16)
#define PORTD_PCR17                              PORT_PCR_REG(PORTD_BASE_PTR,17)
#define PORTD_PCR18                              PORT_PCR_REG(PORTD_BASE_PTR,18)
#define PORTD_PCR19                              PORT_PCR_REG(PORTD_BASE_PTR,19)
#define PORTD_PCR20                              PORT_PCR_REG(PORTD_BASE_PTR,20)
#define PORTD_PCR21                              PORT_PCR_REG(PORTD_BASE_PTR,21)
#define PORTD_PCR22                              PORT_PCR_REG(PORTD_BASE_PTR,22)
#define PORTD_PCR23                              PORT_PCR_REG(PORTD_BASE_PTR,23)
#define PORTD_PCR24                              PORT_PCR_REG(PORTD_BASE_PTR,24)
#define PORTD_PCR25                              PORT_PCR_REG(PORTD_BASE_PTR,25)
#define PORTD_PCR26                              PORT_PCR_REG(PORTD_BASE_PTR,26)
#define PORTD_PCR27                              PORT_PCR_REG(PORTD_BASE_PTR,27)
#define PORTD_PCR28                              PORT_PCR_REG(PORTD_BASE_PTR,28)
#define PORTD_PCR29                              PORT_PCR_REG(PORTD_BASE_PTR,29)
#define PORTD_PCR30                              PORT_PCR_REG(PORTD_BASE_PTR,30)
#define PORTD_PCR31                              PORT_PCR_REG(PORTD_BASE_PTR,31)
#define PORTD_GPCLR                              PORT_GPCLR_REG(PORTD_BASE_PTR)
#define PORTD_GPCHR                              PORT_GPCHR_REG(PORTD_BASE_PTR)
#define PORTD_ISFR                               PORT_ISFR_REG(PORTD_BASE_PTR)
#define PORTD_DFER                               PORT_DFER_REG(PORTD_BASE_PTR)
#define PORTD_DFCR                               PORT_DFCR_REG(PORTD_BASE_PTR)
#define PORTD_DFWR                               PORT_DFWR_REG(PORTD_BASE_PTR)
/* PORTE */
#define PORTE_PCR0                               PORT_PCR_REG(PORTE_BASE_PTR,0)
#define PORTE_PCR1                               PORT_PCR_REG(PORTE_BASE_PTR,1)
#define PORTE_PCR2                               PORT_PCR_REG(PORTE_BASE_PTR,2)
#define PORTE_PCR3                               PORT_PCR_REG(PORTE_BASE_PTR,3)
#define PORTE_PCR4                               PORT_PCR_REG(PORTE_BASE_PTR,4)
#define PORTE_PCR5                               PORT_PCR_REG(PORTE_BASE_PTR,5)
#define PORTE_PCR6                               PORT_PCR_REG(PORTE_BASE_PTR,6)
#define PORTE_PCR7                               PORT_PCR_REG(PORTE_BASE_PTR,7)
#define PORTE_PCR8                               PORT_PCR_REG(PORTE_BASE_PTR,8)
#define PORTE_PCR9                               PORT_PCR_REG(PORTE_BASE_PTR,9)
#define PORTE_PCR10                              PORT_PCR_REG(PORTE_BASE_PTR,10)
#define PORTE_PCR11                              PORT_PCR_REG(PORTE_BASE_PTR,11)
#define PORTE_PCR12                              PORT_PCR_REG(PORTE_BASE_PTR,12)
#define PORTE_PCR13                              PORT_PCR_REG(PORTE_BASE_PTR,13)
#define PORTE_PCR14                              PORT_PCR_REG(PORTE_BASE_PTR,14)
#define PORTE_PCR15                              PORT_PCR_REG(PORTE_BASE_PTR,15)
#define PORTE_PCR16                              PORT_PCR_REG(PORTE_BASE_PTR,16)
#define PORTE_PCR17                              PORT_PCR_REG(PORTE_BASE_PTR,17)
#define PORTE_PCR18                              PORT_PCR_REG(PORTE_BASE_PTR,18)
#define PORTE_PCR19                              PORT_PCR_REG(PORTE_BASE_PTR,19)
#define PORTE_PCR20                              PORT_PCR_REG(PORTE_BASE_PTR,20)
#define PORTE_PCR21                              PORT_PCR_REG(PORTE_BASE_PTR,21)
#define PORTE_PCR22                              PORT_PCR_REG(PORTE_BASE_PTR,22)
#define PORTE_PCR23                              PORT_PCR_REG(PORTE_BASE_PTR,23)
#define PORTE_PCR24                              PORT_PCR_REG(PORTE_BASE_PTR,24)
#define PORTE_PCR25                              PORT_PCR_REG(PORTE_BASE_PTR,25)
#define PORTE_PCR26                              PORT_PCR_REG(PORTE_BASE_PTR,26)
#define PORTE_PCR27                              PORT_PCR_REG(PORTE_BASE_PTR,27)
#define PORTE_PCR28                              PORT_PCR_REG(PORTE_BASE_PTR,28)
#define PORTE_PCR29                              PORT_PCR_REG(PORTE_BASE_PTR,29)
#define PORTE_PCR30                              PORT_PCR_REG(PORTE_BASE_PTR,30)
#define PORTE_PCR31                              PORT_PCR_REG(PORTE_BASE_PTR,31)
#define PORTE_GPCLR                              PORT_GPCLR_REG(PORTE_BASE_PTR)
#define PORTE_GPCHR                              PORT_GPCHR_REG(PORTE_BASE_PTR)
#define PORTE_ISFR                               PORT_ISFR_REG(PORTE_BASE_PTR)
#define PORTE_DFER                               PORT_DFER_REG(PORTE_BASE_PTR)
#define PORTE_DFCR                               PORT_DFCR_REG(PORTE_BASE_PTR)
#define PORTE_DFWR                               PORT_DFWR_REG(PORTE_BASE_PTR)

/* PORT - Register array accessors */
#define PORTA_PCR(index)                         PORT_PCR_REG(PORTA_BASE_PTR,index)
#define PORTB_PCR(index)                         PORT_PCR_REG(PORTB_BASE_PTR,index)
#define PORTC_PCR(index)                         PORT_PCR_REG(PORTC_BASE_PTR,index)
#define PORTD_PCR(index)                         PORT_PCR_REG(PORTD_BASE_PTR,index)
#define PORTE_PCR(index)                         PORT_PCR_REG(PORTE_BASE_PTR,index)

/**
 * @}
 */ /* end of group PORT_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group PORT_Peripheral */


/* ----------------------------------------------------------------------------
   -- RCM
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup RCM_Peripheral RCM
 * @{
 */

/** RCM - Peripheral register structure */
typedef struct RCM_MemMap {
  uint8_t SRSL;                                    /**< RCM System Reset Status Low Register, offset: 0x0 */
  uint8_t SRSH;                                    /**< RCM System Reset Status High Register, offset: 0x1 */
  uint8_t RESERVED_0[2];
  uint8_t RPFC;                                    /**< RCM RESETb Pin Filter Control Register, offset: 0x4 */
  uint8_t RPFW;                                    /**< RCM RESETb Pin Filter Width Register, offset: 0x5 */
} volatile *RCM_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- RCM - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup RCM_Register_Accessor_Macros RCM - Register accessor macros
 * @{
 */


/* RCM - Register accessors */
#define RCM_SRSL_REG(base)                       ((base)->SRSL)
#define RCM_SRSH_REG(base)                       ((base)->SRSH)
#define RCM_RPFC_REG(base)                       ((base)->RPFC)
#define RCM_RPFW_REG(base)                       ((base)->RPFW)

/**
 * @}
 */ /* end of group RCM_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- RCM Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup RCM_Register_Masks RCM Register Masks
 * @{
 */

/* SRSL Bit Fields */
#define RCM_SRSL_LVD_MASK                        0x2u
#define RCM_SRSL_LVD_SHIFT                       1
#define RCM_SRSL_LOC_MASK                        0x4u
#define RCM_SRSL_LOC_SHIFT                       2
#define RCM_SRSL_COP_MASK                        0x20u
#define RCM_SRSL_COP_SHIFT                       5
#define RCM_SRSL_PIN_MASK                        0x40u
#define RCM_SRSL_PIN_SHIFT                       6
#define RCM_SRSL_POR_MASK                        0x80u
#define RCM_SRSL_POR_SHIFT                       7
/* SRSH Bit Fields */
#define RCM_SRSH_JTAG_MASK                       0x1u
#define RCM_SRSH_JTAG_SHIFT                      0
#define RCM_SRSH_LOCKUP_MASK                     0x2u
#define RCM_SRSH_LOCKUP_SHIFT                    1
#define RCM_SRSH_SW_MASK                         0x4u
#define RCM_SRSH_SW_SHIFT                        2
#define RCM_SRSH_MDMAP_MASK                      0x8u
#define RCM_SRSH_MDMAP_SHIFT                     3
#define RCM_SRSH_SACKERR_MASK                    0x20u
#define RCM_SRSH_SACKERR_SHIFT                   5
/* RPFC Bit Fields */
#define RCM_RPFC_RSTFLTSRW_MASK                  0x3u
#define RCM_RPFC_RSTFLTSRW_SHIFT                 0
#define RCM_RPFC_RSTFLTSRW(x)                    (((uint8_t)(((uint8_t)(x))<<RCM_RPFC_RSTFLTSRW_SHIFT))&RCM_RPFC_RSTFLTSRW_MASK)
#define RCM_RPFC_RSTFLTSS_MASK                   0x4u
#define RCM_RPFC_RSTFLTSS_SHIFT                  2
/* RPFW Bit Fields */
#define RCM_RPFW_RSTFLTSS_MASK                   0x1Fu
#define RCM_RPFW_RSTFLTSS_SHIFT                  0
#define RCM_RPFW_RSTFLTSS(x)                     (((uint8_t)(((uint8_t)(x))<<RCM_RPFW_RSTFLTSS_SHIFT))&RCM_RPFW_RSTFLTSS_MASK)

/**
 * @}
 */ /* end of group RCM_Register_Masks */


/* RCM - Peripheral instance base addresses */
/** Peripheral RCM base pointer */
#define RCM_BASE_PTR                             ((RCM_MemMapPtr)0x40057000u)

/* ----------------------------------------------------------------------------
   -- RCM - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup RCM_Register_Accessor_Macros RCM - Register accessor macros
 * @{
 */


/* RCM - Register instance definitions */
/* RCM */
#define RCM_SRSL                                 RCM_SRSL_REG(RCM_BASE_PTR)
#define RCM_SRSH                                 RCM_SRSH_REG(RCM_BASE_PTR)
#define RCM_RPFC                                 RCM_RPFC_REG(RCM_BASE_PTR)
#define RCM_RPFW                                 RCM_RPFW_REG(RCM_BASE_PTR)

/**
 * @}
 */ /* end of group RCM_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group RCM_Peripheral */


/* ----------------------------------------------------------------------------
   -- RTC
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup RTC_Peripheral RTC
 * @{
 */

/** RTC - Peripheral register structure */
typedef struct RTC_MemMap {
  uint32_t SC;                                     /**< RTC Status and Control Register, offset: 0x0 */
  uint32_t MOD;                                    /**< RTC Modulo Register, offset: 0x4 */
  uint32_t CNT;                                    /**< RTC Counter Register, offset: 0x8 */
} volatile *RTC_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- RTC - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup RTC_Register_Accessor_Macros RTC - Register accessor macros
 * @{
 */


/* RTC - Register accessors */
#define RTC_SC_REG(base)                         ((base)->SC)
#define RTC_MOD_REG(base)                        ((base)->MOD)
#define RTC_CNT_REG(base)                        ((base)->CNT)

/**
 * @}
 */ /* end of group RTC_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- RTC Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup RTC_Register_Masks RTC Register Masks
 * @{
 */

/* SC Bit Fields */
#define RTC_SC_RTCO_MASK                         0x10u
#define RTC_SC_RTCO_SHIFT                        4
#define RTC_SC_DMAE_MASK                         0x20u
#define RTC_SC_DMAE_SHIFT                        5
#define RTC_SC_RTIE_MASK                         0x40u
#define RTC_SC_RTIE_SHIFT                        6
#define RTC_SC_RTIF_MASK                         0x80u
#define RTC_SC_RTIF_SHIFT                        7
#define RTC_SC_RTCPS_MASK                        0x700u
#define RTC_SC_RTCPS_SHIFT                       8
#define RTC_SC_RTCPS(x)                          (((uint32_t)(((uint32_t)(x))<<RTC_SC_RTCPS_SHIFT))&RTC_SC_RTCPS_MASK)
#define RTC_SC_RTCLKS_MASK                       0xC000u
#define RTC_SC_RTCLKS_SHIFT                      14
#define RTC_SC_RTCLKS(x)                         (((uint32_t)(((uint32_t)(x))<<RTC_SC_RTCLKS_SHIFT))&RTC_SC_RTCLKS_MASK)
/* MOD Bit Fields */
#define RTC_MOD_MOD_MASK                         0xFFFFu
#define RTC_MOD_MOD_SHIFT                        0
#define RTC_MOD_MOD(x)                           (((uint32_t)(((uint32_t)(x))<<RTC_MOD_MOD_SHIFT))&RTC_MOD_MOD_MASK)
/* CNT Bit Fields */
#define RTC_CNT_CNT_MASK                         0xFFFFu
#define RTC_CNT_CNT_SHIFT                        0
#define RTC_CNT_CNT(x)                           (((uint32_t)(((uint32_t)(x))<<RTC_CNT_CNT_SHIFT))&RTC_CNT_CNT_MASK)

/**
 * @}
 */ /* end of group RTC_Register_Masks */


/* RTC - Peripheral instance base addresses */
/** Peripheral RTC base pointer */
#define RTC_BASE_PTR                             ((RTC_MemMapPtr)0x4002D000u)

/* ----------------------------------------------------------------------------
   -- RTC - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup RTC_Register_Accessor_Macros RTC - Register accessor macros
 * @{
 */


/* RTC - Register instance definitions */
/* RTC */
#define RTC_SC                                   RTC_SC_REG(RTC_BASE_PTR)
#define RTC_MOD                                  RTC_MOD_REG(RTC_BASE_PTR)
#define RTC_CNT                                  RTC_CNT_REG(RTC_BASE_PTR)

/**
 * @}
 */ /* end of group RTC_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group RTC_Peripheral */


/* ----------------------------------------------------------------------------
   -- SCB
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup SCB_Peripheral SCB
 * @{
 */

/** SCB - Peripheral register structure */
typedef struct SCB_MemMap {
  uint8_t RESERVED_0[8];
  uint32_t ACTLR;                                  /**< Auxiliary Control Register,, offset: 0x8 */
  uint8_t RESERVED_1[3316];
  uint32_t CPUID;                                  /**< CPUID Base Register, offset: 0xD00 */
  uint32_t ICSR;                                   /**< Interrupt Control and State Register, offset: 0xD04 */
  uint32_t VTOR;                                   /**< Vector Table Offset Register, offset: 0xD08 */
  uint32_t AIRCR;                                  /**< Application Interrupt and Reset Control Register, offset: 0xD0C */
  uint32_t SCR;                                    /**< System Control Register, offset: 0xD10 */
  uint32_t CCR;                                    /**< Configuration and Control Register, offset: 0xD14 */
  uint32_t SHPR1;                                  /**< System Handler Priority Register 1, offset: 0xD18 */
  uint32_t SHPR2;                                  /**< System Handler Priority Register 2, offset: 0xD1C */
  uint32_t SHPR3;                                  /**< System Handler Priority Register 3, offset: 0xD20 */
  uint32_t SHCSR;                                  /**< System Handler Control and State Register, offset: 0xD24 */
  uint32_t CFSR;                                   /**< Configurable Fault Status Registers, offset: 0xD28 */
  uint32_t HFSR;                                   /**< HardFault Status register, offset: 0xD2C */
  uint32_t DFSR;                                   /**< Debug Fault Status Register, offset: 0xD30 */
  uint32_t MMFAR;                                  /**< MemManage Address Register, offset: 0xD34 */
  uint32_t BFAR;                                   /**< BusFault Address Register, offset: 0xD38 */
  uint32_t AFSR;                                   /**< Auxiliary Fault Status Register, offset: 0xD3C */
} volatile *SCB_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- SCB - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup SCB_Register_Accessor_Macros SCB - Register accessor macros
 * @{
 */


/* SCB - Register accessors */
#define SCB_ACTLR_REG(base)                      ((base)->ACTLR)
#define SCB_CPUID_REG(base)                      ((base)->CPUID)
#define SCB_ICSR_REG(base)                       ((base)->ICSR)
#define SCB_VTOR_REG(base)                       ((base)->VTOR)
#define SCB_AIRCR_REG(base)                      ((base)->AIRCR)
#define SCB_SCR_REG(base)                        ((base)->SCR)
#define SCB_CCR_REG(base)                        ((base)->CCR)
#define SCB_SHPR1_REG(base)                      ((base)->SHPR1)
#define SCB_SHPR2_REG(base)                      ((base)->SHPR2)
#define SCB_SHPR3_REG(base)                      ((base)->SHPR3)
#define SCB_SHCSR_REG(base)                      ((base)->SHCSR)
#define SCB_CFSR_REG(base)                       ((base)->CFSR)
#define SCB_HFSR_REG(base)                       ((base)->HFSR)
#define SCB_DFSR_REG(base)                       ((base)->DFSR)
#define SCB_MMFAR_REG(base)                      ((base)->MMFAR)
#define SCB_BFAR_REG(base)                       ((base)->BFAR)
#define SCB_AFSR_REG(base)                       ((base)->AFSR)

/**
 * @}
 */ /* end of group SCB_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- SCB Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup SCB_Register_Masks SCB Register Masks
 * @{
 */

/* ACTLR Bit Fields */
#define SCB_ACTLR_DISMCYCINT_MASK                0x1u
#define SCB_ACTLR_DISMCYCINT_SHIFT               0
#define SCB_ACTLR_DISDEFWBUF_MASK                0x2u
#define SCB_ACTLR_DISDEFWBUF_SHIFT               1
#define SCB_ACTLR_DISFOLD_MASK                   0x4u
#define SCB_ACTLR_DISFOLD_SHIFT                  2
/* CPUID Bit Fields */
#define SCB_CPUID_REVISION_MASK                  0xFu
#define SCB_CPUID_REVISION_SHIFT                 0
#define SCB_CPUID_REVISION(x)                    (((uint32_t)(((uint32_t)(x))<<SCB_CPUID_REVISION_SHIFT))&SCB_CPUID_REVISION_MASK)
#define SCB_CPUID_PARTNO_MASK                    0xFFF0u
#define SCB_CPUID_PARTNO_SHIFT                   4
#define SCB_CPUID_PARTNO(x)                      (((uint32_t)(((uint32_t)(x))<<SCB_CPUID_PARTNO_SHIFT))&SCB_CPUID_PARTNO_MASK)
#define SCB_CPUID_VARIANT_MASK                   0xF00000u
#define SCB_CPUID_VARIANT_SHIFT                  20
#define SCB_CPUID_VARIANT(x)                     (((uint32_t)(((uint32_t)(x))<<SCB_CPUID_VARIANT_SHIFT))&SCB_CPUID_VARIANT_MASK)
#define SCB_CPUID_IMPLEMENTER_MASK               0xFF000000u
#define SCB_CPUID_IMPLEMENTER_SHIFT              24
#define SCB_CPUID_IMPLEMENTER(x)                 (((uint32_t)(((uint32_t)(x))<<SCB_CPUID_IMPLEMENTER_SHIFT))&SCB_CPUID_IMPLEMENTER_MASK)
/* ICSR Bit Fields */
#define SCB_ICSR_VECTACTIVE_MASK                 0x1FFu
#define SCB_ICSR_VECTACTIVE_SHIFT                0
#define SCB_ICSR_VECTACTIVE(x)                   (((uint32_t)(((uint32_t)(x))<<SCB_ICSR_VECTACTIVE_SHIFT))&SCB_ICSR_VECTACTIVE_MASK)
#define SCB_ICSR_RETTOBASE_MASK                  0x800u
#define SCB_ICSR_RETTOBASE_SHIFT                 11
#define SCB_ICSR_VECTPENDING_MASK                0x3F000u
#define SCB_ICSR_VECTPENDING_SHIFT               12
#define SCB_ICSR_VECTPENDING(x)                  (((uint32_t)(((uint32_t)(x))<<SCB_ICSR_VECTPENDING_SHIFT))&SCB_ICSR_VECTPENDING_MASK)
#define SCB_ICSR_ISRPENDING_MASK                 0x400000u
#define SCB_ICSR_ISRPENDING_SHIFT                22
#define SCB_ICSR_ISRPREEMPT_MASK                 0x800000u
#define SCB_ICSR_ISRPREEMPT_SHIFT                23
#define SCB_ICSR_PENDSTCLR_MASK                  0x2000000u
#define SCB_ICSR_PENDSTCLR_SHIFT                 25
#define SCB_ICSR_PENDSTSET_MASK                  0x4000000u
#define SCB_ICSR_PENDSTSET_SHIFT                 26
#define SCB_ICSR_PENDSVCLR_MASK                  0x8000000u
#define SCB_ICSR_PENDSVCLR_SHIFT                 27
#define SCB_ICSR_PENDSVSET_MASK                  0x10000000u
#define SCB_ICSR_PENDSVSET_SHIFT                 28
#define SCB_ICSR_NMIPENDSET_MASK                 0x80000000u
#define SCB_ICSR_NMIPENDSET_SHIFT                31
/* VTOR Bit Fields */
#define SCB_VTOR_TBLOFF_MASK                     0xFFFFFF80u
#define SCB_VTOR_TBLOFF_SHIFT                    7
#define SCB_VTOR_TBLOFF(x)                       (((uint32_t)(((uint32_t)(x))<<SCB_VTOR_TBLOFF_SHIFT))&SCB_VTOR_TBLOFF_MASK)
/* AIRCR Bit Fields */
#define SCB_AIRCR_VECTRESET_MASK                 0x1u
#define SCB_AIRCR_VECTRESET_SHIFT                0
#define SCB_AIRCR_VECTCLRACTIVE_MASK             0x2u
#define SCB_AIRCR_VECTCLRACTIVE_SHIFT            1
#define SCB_AIRCR_SYSRESETREQ_MASK               0x4u
#define SCB_AIRCR_SYSRESETREQ_SHIFT              2
#define SCB_AIRCR_PRIGROUP_MASK                  0x700u
#define SCB_AIRCR_PRIGROUP_SHIFT                 8
#define SCB_AIRCR_PRIGROUP(x)                    (((uint32_t)(((uint32_t)(x))<<SCB_AIRCR_PRIGROUP_SHIFT))&SCB_AIRCR_PRIGROUP_MASK)
#define SCB_AIRCR_ENDIANNESS_MASK                0x8000u
#define SCB_AIRCR_ENDIANNESS_SHIFT               15
#define SCB_AIRCR_VECTKEY_MASK                   0xFFFF0000u
#define SCB_AIRCR_VECTKEY_SHIFT                  16
#define SCB_AIRCR_VECTKEY(x)                     (((uint32_t)(((uint32_t)(x))<<SCB_AIRCR_VECTKEY_SHIFT))&SCB_AIRCR_VECTKEY_MASK)
/* SCR Bit Fields */
#define SCB_SCR_SLEEPONEXIT_MASK                 0x2u
#define SCB_SCR_SLEEPONEXIT_SHIFT                1
#define SCB_SCR_SLEEPDEEP_MASK                   0x4u
#define SCB_SCR_SLEEPDEEP_SHIFT                  2
#define SCB_SCR_SEVONPEND_MASK                   0x10u
#define SCB_SCR_SEVONPEND_SHIFT                  4
/* CCR Bit Fields */
#define SCB_CCR_NONBASETHRDENA_MASK              0x1u
#define SCB_CCR_NONBASETHRDENA_SHIFT             0
#define SCB_CCR_USERSETMPEND_MASK                0x2u
#define SCB_CCR_USERSETMPEND_SHIFT               1
#define SCB_CCR_UNALIGN_TRP_MASK                 0x8u
#define SCB_CCR_UNALIGN_TRP_SHIFT                3
#define SCB_CCR_DIV_0_TRP_MASK                   0x10u
#define SCB_CCR_DIV_0_TRP_SHIFT                  4
#define SCB_CCR_BFHFNMIGN_MASK                   0x100u
#define SCB_CCR_BFHFNMIGN_SHIFT                  8
#define SCB_CCR_STKALIGN_MASK                    0x200u
#define SCB_CCR_STKALIGN_SHIFT                   9
/* SHPR1 Bit Fields */
#define SCB_SHPR1_PRI_4_MASK                     0xFFu
#define SCB_SHPR1_PRI_4_SHIFT                    0
#define SCB_SHPR1_PRI_4(x)                       (((uint32_t)(((uint32_t)(x))<<SCB_SHPR1_PRI_4_SHIFT))&SCB_SHPR1_PRI_4_MASK)
#define SCB_SHPR1_PRI_5_MASK                     0xFF00u
#define SCB_SHPR1_PRI_5_SHIFT                    8
#define SCB_SHPR1_PRI_5(x)                       (((uint32_t)(((uint32_t)(x))<<SCB_SHPR1_PRI_5_SHIFT))&SCB_SHPR1_PRI_5_MASK)
#define SCB_SHPR1_PRI_6_MASK                     0xFF0000u
#define SCB_SHPR1_PRI_6_SHIFT                    16
#define SCB_SHPR1_PRI_6(x)                       (((uint32_t)(((uint32_t)(x))<<SCB_SHPR1_PRI_6_SHIFT))&SCB_SHPR1_PRI_6_MASK)
/* SHPR2 Bit Fields */
#define SCB_SHPR2_PRI_11_MASK                    0xFF000000u
#define SCB_SHPR2_PRI_11_SHIFT                   24
#define SCB_SHPR2_PRI_11(x)                      (((uint32_t)(((uint32_t)(x))<<SCB_SHPR2_PRI_11_SHIFT))&SCB_SHPR2_PRI_11_MASK)
/* SHPR3 Bit Fields */
#define SCB_SHPR3_PRI_14_MASK                    0xFF0000u
#define SCB_SHPR3_PRI_14_SHIFT                   16
#define SCB_SHPR3_PRI_14(x)                      (((uint32_t)(((uint32_t)(x))<<SCB_SHPR3_PRI_14_SHIFT))&SCB_SHPR3_PRI_14_MASK)
#define SCB_SHPR3_PRI_15_MASK                    0xFF000000u
#define SCB_SHPR3_PRI_15_SHIFT                   24
#define SCB_SHPR3_PRI_15(x)                      (((uint32_t)(((uint32_t)(x))<<SCB_SHPR3_PRI_15_SHIFT))&SCB_SHPR3_PRI_15_MASK)
/* SHCSR Bit Fields */
#define SCB_SHCSR_MEMFAULTACT_MASK               0x1u
#define SCB_SHCSR_MEMFAULTACT_SHIFT              0
#define SCB_SHCSR_BUSFAULTACT_MASK               0x2u
#define SCB_SHCSR_BUSFAULTACT_SHIFT              1
#define SCB_SHCSR_USGFAULTACT_MASK               0x8u
#define SCB_SHCSR_USGFAULTACT_SHIFT              3
#define SCB_SHCSR_SVCALLACT_MASK                 0x80u
#define SCB_SHCSR_SVCALLACT_SHIFT                7
#define SCB_SHCSR_MONITORACT_MASK                0x100u
#define SCB_SHCSR_MONITORACT_SHIFT               8
#define SCB_SHCSR_PENDSVACT_MASK                 0x400u
#define SCB_SHCSR_PENDSVACT_SHIFT                10
#define SCB_SHCSR_SYSTICKACT_MASK                0x800u
#define SCB_SHCSR_SYSTICKACT_SHIFT               11
#define SCB_SHCSR_USGFAULTPENDED_MASK            0x1000u
#define SCB_SHCSR_USGFAULTPENDED_SHIFT           12
#define SCB_SHCSR_MEMFAULTPENDED_MASK            0x2000u
#define SCB_SHCSR_MEMFAULTPENDED_SHIFT           13
#define SCB_SHCSR_BUSFAULTPENDED_MASK            0x4000u
#define SCB_SHCSR_BUSFAULTPENDED_SHIFT           14
#define SCB_SHCSR_SVCALLPENDED_MASK              0x8000u
#define SCB_SHCSR_SVCALLPENDED_SHIFT             15
#define SCB_SHCSR_MEMFAULTENA_MASK               0x10000u
#define SCB_SHCSR_MEMFAULTENA_SHIFT              16
#define SCB_SHCSR_BUSFAULTENA_MASK               0x20000u
#define SCB_SHCSR_BUSFAULTENA_SHIFT              17
#define SCB_SHCSR_USGFAULTENA_MASK               0x40000u
#define SCB_SHCSR_USGFAULTENA_SHIFT              18
/* CFSR Bit Fields */
#define SCB_CFSR_IACCVIOL_MASK                   0x1u
#define SCB_CFSR_IACCVIOL_SHIFT                  0
#define SCB_CFSR_DACCVIOL_MASK                   0x2u
#define SCB_CFSR_DACCVIOL_SHIFT                  1
#define SCB_CFSR_MUNSTKERR_MASK                  0x8u
#define SCB_CFSR_MUNSTKERR_SHIFT                 3
#define SCB_CFSR_MSTKERR_MASK                    0x10u
#define SCB_CFSR_MSTKERR_SHIFT                   4
#define SCB_CFSR_MLSPERR_MASK                    0x20u
#define SCB_CFSR_MLSPERR_SHIFT                   5
#define SCB_CFSR_MMARVALID_MASK                  0x80u
#define SCB_CFSR_MMARVALID_SHIFT                 7
#define SCB_CFSR_IBUSERR_MASK                    0x100u
#define SCB_CFSR_IBUSERR_SHIFT                   8
#define SCB_CFSR_PRECISERR_MASK                  0x200u
#define SCB_CFSR_PRECISERR_SHIFT                 9
#define SCB_CFSR_IMPRECISERR_MASK                0x400u
#define SCB_CFSR_IMPRECISERR_SHIFT               10
#define SCB_CFSR_UNSTKERR_MASK                   0x800u
#define SCB_CFSR_UNSTKERR_SHIFT                  11
#define SCB_CFSR_STKERR_MASK                     0x1000u
#define SCB_CFSR_STKERR_SHIFT                    12
#define SCB_CFSR_LSPERR_MASK                     0x2000u
#define SCB_CFSR_LSPERR_SHIFT                    13
#define SCB_CFSR_BFARVALID_MASK                  0x8000u
#define SCB_CFSR_BFARVALID_SHIFT                 15
#define SCB_CFSR_UNDEFINSTR_MASK                 0x10000u
#define SCB_CFSR_UNDEFINSTR_SHIFT                16
#define SCB_CFSR_INVSTATE_MASK                   0x20000u
#define SCB_CFSR_INVSTATE_SHIFT                  17
#define SCB_CFSR_INVPC_MASK                      0x40000u
#define SCB_CFSR_INVPC_SHIFT                     18
#define SCB_CFSR_NOCP_MASK                       0x80000u
#define SCB_CFSR_NOCP_SHIFT                      19
#define SCB_CFSR_UNALIGNED_MASK                  0x1000000u
#define SCB_CFSR_UNALIGNED_SHIFT                 24
#define SCB_CFSR_DIVBYZERO_MASK                  0x2000000u
#define SCB_CFSR_DIVBYZERO_SHIFT                 25
/* HFSR Bit Fields */
#define SCB_HFSR_VECTTBL_MASK                    0x2u
#define SCB_HFSR_VECTTBL_SHIFT                   1
#define SCB_HFSR_FORCED_MASK                     0x40000000u
#define SCB_HFSR_FORCED_SHIFT                    30
#define SCB_HFSR_DEBUGEVT_MASK                   0x80000000u
#define SCB_HFSR_DEBUGEVT_SHIFT                  31
/* DFSR Bit Fields */
#define SCB_DFSR_HALTED_MASK                     0x1u
#define SCB_DFSR_HALTED_SHIFT                    0
#define SCB_DFSR_BKPT_MASK                       0x2u
#define SCB_DFSR_BKPT_SHIFT                      1
#define SCB_DFSR_DWTTRAP_MASK                    0x4u
#define SCB_DFSR_DWTTRAP_SHIFT                   2
#define SCB_DFSR_VCATCH_MASK                     0x8u
#define SCB_DFSR_VCATCH_SHIFT                    3
#define SCB_DFSR_EXTERNAL_MASK                   0x10u
#define SCB_DFSR_EXTERNAL_SHIFT                  4
/* MMFAR Bit Fields */
#define SCB_MMFAR_ADDRESS_MASK                   0xFFFFFFFFu
#define SCB_MMFAR_ADDRESS_SHIFT                  0
#define SCB_MMFAR_ADDRESS(x)                     (((uint32_t)(((uint32_t)(x))<<SCB_MMFAR_ADDRESS_SHIFT))&SCB_MMFAR_ADDRESS_MASK)
/* BFAR Bit Fields */
#define SCB_BFAR_ADDRESS_MASK                    0xFFFFFFFFu
#define SCB_BFAR_ADDRESS_SHIFT                   0
#define SCB_BFAR_ADDRESS(x)                      (((uint32_t)(((uint32_t)(x))<<SCB_BFAR_ADDRESS_SHIFT))&SCB_BFAR_ADDRESS_MASK)
/* AFSR Bit Fields */
#define SCB_AFSR_AUXFAULT_MASK                   0xFFFFFFFFu
#define SCB_AFSR_AUXFAULT_SHIFT                  0
#define SCB_AFSR_AUXFAULT(x)                     (((uint32_t)(((uint32_t)(x))<<SCB_AFSR_AUXFAULT_SHIFT))&SCB_AFSR_AUXFAULT_MASK)

/**
 * @}
 */ /* end of group SCB_Register_Masks */


/* SCB - Peripheral instance base addresses */
/** Peripheral SystemControl base pointer */
#define SystemControl_BASE_PTR                   ((SCB_MemMapPtr)0xE000E000u)

/* ----------------------------------------------------------------------------
   -- SCB - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup SCB_Register_Accessor_Macros SCB - Register accessor macros
 * @{
 */


/* SCB - Register instance definitions */
/* SystemControl */
#define SCB_ACTLR                                SCB_ACTLR_REG(SystemControl_BASE_PTR)
#define SCB_CPUID                                SCB_CPUID_REG(SystemControl_BASE_PTR)
#define SCB_ICSR                                 SCB_ICSR_REG(SystemControl_BASE_PTR)
#define SCB_VTOR                                 SCB_VTOR_REG(SystemControl_BASE_PTR)
#define SCB_AIRCR                                SCB_AIRCR_REG(SystemControl_BASE_PTR)
#define SCB_SCR                                  SCB_SCR_REG(SystemControl_BASE_PTR)
#define SCB_CCR                                  SCB_CCR_REG(SystemControl_BASE_PTR)
#define SCB_SHPR1                                SCB_SHPR1_REG(SystemControl_BASE_PTR)
#define SCB_SHPR2                                SCB_SHPR2_REG(SystemControl_BASE_PTR)
#define SCB_SHPR3                                SCB_SHPR3_REG(SystemControl_BASE_PTR)
#define SCB_SHCSR                                SCB_SHCSR_REG(SystemControl_BASE_PTR)
#define SCB_CFSR                                 SCB_CFSR_REG(SystemControl_BASE_PTR)
#define SCB_HFSR                                 SCB_HFSR_REG(SystemControl_BASE_PTR)
#define SCB_DFSR                                 SCB_DFSR_REG(SystemControl_BASE_PTR)
#define SCB_MMFAR                                SCB_MMFAR_REG(SystemControl_BASE_PTR)
#define SCB_BFAR                                 SCB_BFAR_REG(SystemControl_BASE_PTR)
#define SCB_AFSR                                 SCB_AFSR_REG(SystemControl_BASE_PTR)

/**
 * @}
 */ /* end of group SCB_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group SCB_Peripheral */


/* ----------------------------------------------------------------------------
   -- SIM
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup SIM_Peripheral SIM
 * @{
 */

/** SIM - Peripheral register structure */
typedef struct SIM_MemMap {
  uint8_t RESERVED_0[4];
  uint32_t SOPT2;                                  /**< System Options Register 2, offset: 0x4 */
  uint32_t SOPT3;                                  /**< System Options Register 3, offset: 0x8 */
  uint32_t SOPT4;                                  /**< System Options Register 4, offset: 0xC */
  uint32_t SOPT5;                                  /**< System Options Register 5, offset: 0x10 */
  uint32_t SOPT6;                                  /**< System Options Register 6, offset: 0x14 */
  uint32_t SOPT7;                                  /**< System Options Register 7, offset: 0x18 */
  uint8_t RESERVED_1[8];
  uint32_t SDID;                                   /**< System Device Identification Register, offset: 0x24 */
  uint8_t RESERVED_2[16];
  uint32_t SCGC5;                                  /**< System Clock Gating Control Register 5, offset: 0x38 */
  uint32_t SCGC6;                                  /**< System Clock Gating Control Register 6, offset: 0x3C */
  uint32_t SCGC7;                                  /**< System Clock Gating Control Register 7, offset: 0x40 */
  uint32_t CLKDIV1;                                /**< System Clock Divider Register 1, offset: 0x44 */
  uint8_t RESERVED_3[12];
  uint32_t UIDH;                                   /**< Unique Identification Register High, offset: 0x54 */
  uint32_t UIDMH;                                  /**< Unique Identification Register Mid-High, offset: 0x58 */
  uint32_t UIDML;                                  /**< Unique Identification Register Mid Low, offset: 0x5C */
  uint32_t UIDL;                                   /**< Unique Identification Register Low, offset: 0x60 */
} volatile *SIM_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- SIM - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup SIM_Register_Accessor_Macros SIM - Register accessor macros
 * @{
 */


/* SIM - Register accessors */
#define SIM_SOPT2_REG(base)                      ((base)->SOPT2)
#define SIM_SOPT3_REG(base)                      ((base)->SOPT3)
#define SIM_SOPT4_REG(base)                      ((base)->SOPT4)
#define SIM_SOPT5_REG(base)                      ((base)->SOPT5)
#define SIM_SOPT6_REG(base)                      ((base)->SOPT6)
#define SIM_SOPT7_REG(base)                      ((base)->SOPT7)
#define SIM_SDID_REG(base)                       ((base)->SDID)
#define SIM_SCGC5_REG(base)                      ((base)->SCGC5)
#define SIM_SCGC6_REG(base)                      ((base)->SCGC6)
#define SIM_SCGC7_REG(base)                      ((base)->SCGC7)
#define SIM_CLKDIV1_REG(base)                    ((base)->CLKDIV1)
#define SIM_UIDH_REG(base)                       ((base)->UIDH)
#define SIM_UIDMH_REG(base)                      ((base)->UIDMH)
#define SIM_UIDML_REG(base)                      ((base)->UIDML)
#define SIM_UIDL_REG(base)                       ((base)->UIDL)

/**
 * @}
 */ /* end of group SIM_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- SIM Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup SIM_Register_Masks SIM Register Masks
 * @{
 */

/* SOPT2 Bit Fields */
#define SIM_SOPT2_FLLENSTOP4_MASK                0x1u
#define SIM_SOPT2_FLLENSTOP4_SHIFT               0
#define SIM_SOPT2_RESETIFE_MASK                  0x2u
#define SIM_SOPT2_RESETIFE_SHIFT                 1
#define SIM_SOPT2_OBEPADSELA13_MASK              0x30u
#define SIM_SOPT2_OBEPADSELA13_SHIFT             4
#define SIM_SOPT2_OBEPADSELA13(x)                (((uint32_t)(((uint32_t)(x))<<SIM_SOPT2_OBEPADSELA13_SHIFT))&SIM_SOPT2_OBEPADSELA13_MASK)
#define SIM_SOPT2_OBEPADSELA12_MASK              0xC0u
#define SIM_SOPT2_OBEPADSELA12_SHIFT             6
#define SIM_SOPT2_OBEPADSELA12(x)                (((uint32_t)(((uint32_t)(x))<<SIM_SOPT2_OBEPADSELA12_SHIFT))&SIM_SOPT2_OBEPADSELA12_MASK)
#define SIM_SOPT2_OBEPADSELA3_MASK               0x300u
#define SIM_SOPT2_OBEPADSELA3_SHIFT              8
#define SIM_SOPT2_OBEPADSELA3(x)                 (((uint32_t)(((uint32_t)(x))<<SIM_SOPT2_OBEPADSELA3_SHIFT))&SIM_SOPT2_OBEPADSELA3_MASK)
#define SIM_SOPT2_OBEPADSELA2_MASK               0xC00u
#define SIM_SOPT2_OBEPADSELA2_SHIFT              10
#define SIM_SOPT2_OBEPADSELA2(x)                 (((uint32_t)(((uint32_t)(x))<<SIM_SOPT2_OBEPADSELA2_SHIFT))&SIM_SOPT2_OBEPADSELA2_MASK)
#define SIM_SOPT2_OBEPADSELE1_MASK               0x3000u
#define SIM_SOPT2_OBEPADSELE1_SHIFT              12
#define SIM_SOPT2_OBEPADSELE1(x)                 (((uint32_t)(((uint32_t)(x))<<SIM_SOPT2_OBEPADSELE1_SHIFT))&SIM_SOPT2_OBEPADSELE1_MASK)
#define SIM_SOPT2_OBEPADSELE0_MASK               0xC000u
#define SIM_SOPT2_OBEPADSELE0_SHIFT              14
#define SIM_SOPT2_OBEPADSELE0(x)                 (((uint32_t)(((uint32_t)(x))<<SIM_SOPT2_OBEPADSELE0_SHIFT))&SIM_SOPT2_OBEPADSELE0_MASK)
#define SIM_SOPT2_FBSL_MASK                      0x30000u
#define SIM_SOPT2_FBSL_SHIFT                     16
#define SIM_SOPT2_FBSL(x)                        (((uint32_t)(((uint32_t)(x))<<SIM_SOPT2_FBSL_SHIFT))&SIM_SOPT2_FBSL_MASK)
#define SIM_SOPT2_TRACECLKSEL_MASK               0x40000u
#define SIM_SOPT2_TRACECLKSEL_SHIFT              18
#define SIM_SOPT2_RAMSIZE_MASK                   0x200000u
#define SIM_SOPT2_RAMSIZE_SHIFT                  21
#define SIM_SOPT2_FSIZE_MASK                     0xC000000u
#define SIM_SOPT2_FSIZE_SHIFT                    26
#define SIM_SOPT2_FSIZE(x)                       (((uint32_t)(((uint32_t)(x))<<SIM_SOPT2_FSIZE_SHIFT))&SIM_SOPT2_FSIZE_MASK)
#define SIM_SOPT2_TSIEN_MASK                     0x10000000u
#define SIM_SOPT2_TSIEN_SHIFT                    28
#define SIM_SOPT2_MAXCLK_MASK                    0x20000000u
#define SIM_SOPT2_MAXCLK_SHIFT                   29
/* SOPT3 Bit Fields */
#define SIM_SOPT3_FTM0_PDB_MASK                  0xFu
#define SIM_SOPT3_FTM0_PDB_SHIFT                 0
#define SIM_SOPT3_FTM0_PDB(x)                    (((uint32_t)(((uint32_t)(x))<<SIM_SOPT3_FTM0_PDB_SHIFT))&SIM_SOPT3_FTM0_PDB_MASK)
#define SIM_SOPT3_FTM1_PDB_MASK                  0xF0u
#define SIM_SOPT3_FTM1_PDB_SHIFT                 4
#define SIM_SOPT3_FTM1_PDB(x)                    (((uint32_t)(((uint32_t)(x))<<SIM_SOPT3_FTM1_PDB_SHIFT))&SIM_SOPT3_FTM1_PDB_MASK)
#define SIM_SOPT3_FTM2_PDB_MASK                  0xF00u
#define SIM_SOPT3_FTM2_PDB_SHIFT                 8
#define SIM_SOPT3_FTM2_PDB(x)                    (((uint32_t)(((uint32_t)(x))<<SIM_SOPT3_FTM2_PDB_SHIFT))&SIM_SOPT3_FTM2_PDB_MASK)
#define SIM_SOPT3_FTM3_PDB_MASK                  0xF000u
#define SIM_SOPT3_FTM3_PDB_SHIFT                 12
#define SIM_SOPT3_FTM3_PDB(x)                    (((uint32_t)(((uint32_t)(x))<<SIM_SOPT3_FTM3_PDB_SHIFT))&SIM_SOPT3_FTM3_PDB_MASK)
#define SIM_SOPT3_FTM_SYNC0_MASK                 0x10000u
#define SIM_SOPT3_FTM_SYNC0_SHIFT                16
#define SIM_SOPT3_FTM_SYNC1_MASK                 0x20000u
#define SIM_SOPT3_FTM_SYNC1_SHIFT                17
#define SIM_SOPT3_FTM_SYNC2_MASK                 0x40000u
#define SIM_SOPT3_FTM_SYNC2_SHIFT                18
#define SIM_SOPT3_FTM_SYNC3_MASK                 0x80000u
#define SIM_SOPT3_FTM_SYNC3_SHIFT                19
/* SOPT4 Bit Fields */
#define SIM_SOPT4_FTM0FLT0_MASK                  0x1u
#define SIM_SOPT4_FTM0FLT0_SHIFT                 0
#define SIM_SOPT4_FTM0FLT1_MASK                  0x2u
#define SIM_SOPT4_FTM0FLT1_SHIFT                 1
#define SIM_SOPT4_FTM0FLT2_MASK                  0x4u
#define SIM_SOPT4_FTM0FLT2_SHIFT                 2
#define SIM_SOPT4_FTM0FLT3_MASK                  0x8u
#define SIM_SOPT4_FTM0FLT3_SHIFT                 3
#define SIM_SOPT4_FTM3FLT0_MASK                  0x1000u
#define SIM_SOPT4_FTM3FLT0_SHIFT                 12
#define SIM_SOPT4_FTM3FLT1_MASK                  0x2000u
#define SIM_SOPT4_FTM3FLT1_SHIFT                 13
#define SIM_SOPT4_FTM3FLT2_MASK                  0x4000u
#define SIM_SOPT4_FTM3FLT2_SHIFT                 14
#define SIM_SOPT4_FTM3FLT3_MASK                  0x8000u
#define SIM_SOPT4_FTM3FLT3_SHIFT                 15
#define SIM_SOPT4_FTM1CH0SRC_MASK                0x300000u
#define SIM_SOPT4_FTM1CH0SRC_SHIFT               20
#define SIM_SOPT4_FTM1CH0SRC(x)                  (((uint32_t)(((uint32_t)(x))<<SIM_SOPT4_FTM1CH0SRC_SHIFT))&SIM_SOPT4_FTM1CH0SRC_MASK)
#define SIM_SOPT4_FTM2CH0SRC_MASK                0xC00000u
#define SIM_SOPT4_FTM2CH0SRC_SHIFT               22
#define SIM_SOPT4_FTM2CH0SRC(x)                  (((uint32_t)(((uint32_t)(x))<<SIM_SOPT4_FTM2CH0SRC_SHIFT))&SIM_SOPT4_FTM2CH0SRC_MASK)
#define SIM_SOPT4_FTM0CLKSEL_MASK                0x1000000u
#define SIM_SOPT4_FTM0CLKSEL_SHIFT               24
#define SIM_SOPT4_FTM1CLKSEL_MASK                0x2000000u
#define SIM_SOPT4_FTM1CLKSEL_SHIFT               25
#define SIM_SOPT4_FTM2CLKSEL_MASK                0x4000000u
#define SIM_SOPT4_FTM2CLKSEL_SHIFT               26
#define SIM_SOPT4_FTM3CLKSEL_MASK                0x8000000u
#define SIM_SOPT4_FTM3CLKSEL_SHIFT               27
/* SOPT5 Bit Fields */
#define SIM_SOPT5_UART0TXSRC_MASK                0x3u
#define SIM_SOPT5_UART0TXSRC_SHIFT               0
#define SIM_SOPT5_UART0TXSRC(x)                  (((uint32_t)(((uint32_t)(x))<<SIM_SOPT5_UART0TXSRC_SHIFT))&SIM_SOPT5_UART0TXSRC_MASK)
#define SIM_SOPT5_UART0RXSRC_MASK                0xCu
#define SIM_SOPT5_UART0RXSRC_SHIFT               2
#define SIM_SOPT5_UART0RXSRC(x)                  (((uint32_t)(((uint32_t)(x))<<SIM_SOPT5_UART0RXSRC_SHIFT))&SIM_SOPT5_UART0RXSRC_MASK)
#define SIM_SOPT5_UART1TXSRC_MASK                0x30u
#define SIM_SOPT5_UART1TXSRC_SHIFT               4
#define SIM_SOPT5_UART1TXSRC(x)                  (((uint32_t)(((uint32_t)(x))<<SIM_SOPT5_UART1TXSRC_SHIFT))&SIM_SOPT5_UART1TXSRC_MASK)
#define SIM_SOPT5_UART1RXSRC_MASK                0xC0u
#define SIM_SOPT5_UART1RXSRC_SHIFT               6
#define SIM_SOPT5_UART1RXSRC(x)                  (((uint32_t)(((uint32_t)(x))<<SIM_SOPT5_UART1RXSRC_SHIFT))&SIM_SOPT5_UART1RXSRC_MASK)
/* SOPT6 Bit Fields */
#define SIM_SOPT6_CLKDIV_MASK                    0x7u
#define SIM_SOPT6_CLKDIV_SHIFT                   0
#define SIM_SOPT6_CLKDIV(x)                      (((uint32_t)(((uint32_t)(x))<<SIM_SOPT6_CLKDIV_SHIFT))&SIM_SOPT6_CLKDIV_MASK)
#define SIM_SOPT6_CLKOS_MASK                     0xF0u
#define SIM_SOPT6_CLKOS_SHIFT                    4
#define SIM_SOPT6_CLKOS(x)                       (((uint32_t)(((uint32_t)(x))<<SIM_SOPT6_CLKOS_SHIFT))&SIM_SOPT6_CLKOS_MASK)
/* SOPT7 Bit Fields */
#define SIM_SOPT7_ADC0TRGSEL_MASK                0xFu
#define SIM_SOPT7_ADC0TRGSEL_SHIFT               0
#define SIM_SOPT7_ADC0TRGSEL(x)                  (((uint32_t)(((uint32_t)(x))<<SIM_SOPT7_ADC0TRGSEL_SHIFT))&SIM_SOPT7_ADC0TRGSEL_MASK)
#define SIM_SOPT7_ADC1TRGSEL_MASK                0xF0u
#define SIM_SOPT7_ADC1TRGSEL_SHIFT               4
#define SIM_SOPT7_ADC1TRGSEL(x)                  (((uint32_t)(((uint32_t)(x))<<SIM_SOPT7_ADC1TRGSEL_SHIFT))&SIM_SOPT7_ADC1TRGSEL_MASK)
#define SIM_SOPT7_ADC2TRGSEL_MASK                0xF00u
#define SIM_SOPT7_ADC2TRGSEL_SHIFT               8
#define SIM_SOPT7_ADC2TRGSEL(x)                  (((uint32_t)(((uint32_t)(x))<<SIM_SOPT7_ADC2TRGSEL_SHIFT))&SIM_SOPT7_ADC2TRGSEL_MASK)
#define SIM_SOPT7_ADC3TRGSEL_MASK                0xF000u
#define SIM_SOPT7_ADC3TRGSEL_SHIFT               12
#define SIM_SOPT7_ADC3TRGSEL(x)                  (((uint32_t)(((uint32_t)(x))<<SIM_SOPT7_ADC3TRGSEL_SHIFT))&SIM_SOPT7_ADC3TRGSEL_MASK)
#define SIM_SOPT7_CMP0WS_MASK                    0x30000u
#define SIM_SOPT7_CMP0WS_SHIFT                   16
#define SIM_SOPT7_CMP0WS(x)                      (((uint32_t)(((uint32_t)(x))<<SIM_SOPT7_CMP0WS_SHIFT))&SIM_SOPT7_CMP0WS_MASK)
#define SIM_SOPT7_CMP1WS_MASK                    0xC0000u
#define SIM_SOPT7_CMP1WS_SHIFT                   18
#define SIM_SOPT7_CMP1WS(x)                      (((uint32_t)(((uint32_t)(x))<<SIM_SOPT7_CMP1WS_SHIFT))&SIM_SOPT7_CMP1WS_MASK)
#define SIM_SOPT7_CMP2WS_MASK                    0x300000u
#define SIM_SOPT7_CMP2WS_SHIFT                   20
#define SIM_SOPT7_CMP2WS(x)                      (((uint32_t)(((uint32_t)(x))<<SIM_SOPT7_CMP2WS_SHIFT))&SIM_SOPT7_CMP2WS_MASK)
#define SIM_SOPT7_CMP3WS_MASK                    0xC00000u
#define SIM_SOPT7_CMP3WS_SHIFT                   22
#define SIM_SOPT7_CMP3WS(x)                      (((uint32_t)(((uint32_t)(x))<<SIM_SOPT7_CMP3WS_SHIFT))&SIM_SOPT7_CMP3WS_MASK)
#define SIM_SOPT7_ADC0ALTTRGEN_MASK              0x10000000u
#define SIM_SOPT7_ADC0ALTTRGEN_SHIFT             28
#define SIM_SOPT7_ADC1ALTTRGEN_MASK              0x20000000u
#define SIM_SOPT7_ADC1ALTTRGEN_SHIFT             29
#define SIM_SOPT7_ADC2ALTTRGEN_MASK              0x40000000u
#define SIM_SOPT7_ADC2ALTTRGEN_SHIFT             30
#define SIM_SOPT7_ADC3ALTTRGEN_MASK              0x80000000u
#define SIM_SOPT7_ADC3ALTTRGEN_SHIFT             31
/* SDID Bit Fields */
#define SIM_SDID_BOID_MASK                       0xFu
#define SIM_SDID_BOID_SHIFT                      0
#define SIM_SDID_BOID(x)                         (((uint32_t)(((uint32_t)(x))<<SIM_SDID_BOID_SHIFT))&SIM_SDID_BOID_MASK)
#define SIM_SDID_DIEID_MASK                      0x30u
#define SIM_SDID_DIEID_SHIFT                     4
#define SIM_SDID_DIEID(x)                        (((uint32_t)(((uint32_t)(x))<<SIM_SDID_DIEID_SHIFT))&SIM_SDID_DIEID_MASK)
#define SIM_SDID_REVID_MASK                      0xF000u
#define SIM_SDID_REVID_SHIFT                     12
#define SIM_SDID_REVID(x)                        (((uint32_t)(((uint32_t)(x))<<SIM_SDID_REVID_SHIFT))&SIM_SDID_REVID_MASK)
/* SCGC5 Bit Fields */
#define SIM_SCGC5_EWM_MASK                       0x2u
#define SIM_SCGC5_EWM_SHIFT                      1
#define SIM_SCGC5_CMT_MASK                       0x4u
#define SIM_SCGC5_CMT_SHIFT                      2
#define SIM_SCGC5_IIC0_MASK                      0x80u
#define SIM_SCGC5_IIC0_SHIFT                     7
#define SIM_SCGC5_IIC1_MASK                      0x100u
#define SIM_SCGC5_IIC1_SHIFT                     8
#define SIM_SCGC5_UART0_MASK                     0x200u
#define SIM_SCGC5_UART0_SHIFT                    9
#define SIM_SCGC5_UART1_MASK                     0x400u
#define SIM_SCGC5_UART1_SHIFT                    10
#define SIM_SCGC5_UART2_MASK                     0x800u
#define SIM_SCGC5_UART2_SHIFT                    11
#define SIM_SCGC5_UART3_MASK                     0x1000u
#define SIM_SCGC5_UART3_SHIFT                    12
#define SIM_SCGC5_SPI0_MASK                      0x4000u
#define SIM_SCGC5_SPI0_SHIFT                     14
#define SIM_SCGC5_SPI1_MASK                      0x8000u
#define SIM_SCGC5_SPI1_SHIFT                     15
#define SIM_SCGC5_ACMP0_MASK                     0x10000u
#define SIM_SCGC5_ACMP0_SHIFT                    16
#define SIM_SCGC5_ACMP1_MASK                     0x20000u
#define SIM_SCGC5_ACMP1_SHIFT                    17
#define SIM_SCGC5_ACMP2_MASK                     0x40000u
#define SIM_SCGC5_ACMP2_SHIFT                    18
#define SIM_SCGC5_ACMP3_MASK                     0x80000u
#define SIM_SCGC5_ACMP3_SHIFT                    19
/* SCGC6 Bit Fields */
#define SIM_SCGC6_DMAMUX_MASK                    0x2u
#define SIM_SCGC6_DMAMUX_SHIFT                   1
#define SIM_SCGC6_PIT_MASK                       0x8u
#define SIM_SCGC6_PIT_SHIFT                      3
#define SIM_SCGC6_CRC_MASK                       0x10u
#define SIM_SCGC6_CRC_SHIFT                      4
#define SIM_SCGC6_PDB0_MASK                      0x20u
#define SIM_SCGC6_PDB0_SHIFT                     5
#define SIM_SCGC6_PDB1_MASK                      0x40u
#define SIM_SCGC6_PDB1_SHIFT                     6
#define SIM_SCGC6_PDB2_MASK                      0x80u
#define SIM_SCGC6_PDB2_SHIFT                     7
#define SIM_SCGC6_PDB3_MASK                      0x100u
#define SIM_SCGC6_PDB3_SHIFT                     8
#define SIM_SCGC6_FTM0_MASK                      0x200u
#define SIM_SCGC6_FTM0_SHIFT                     9
#define SIM_SCGC6_FTM1_MASK                      0x400u
#define SIM_SCGC6_FTM1_SHIFT                     10
#define SIM_SCGC6_FTM2_MASK                      0x800u
#define SIM_SCGC6_FTM2_SHIFT                     11
#define SIM_SCGC6_FTM3_MASK                      0x1000u
#define SIM_SCGC6_FTM3_SHIFT                     12
#define SIM_SCGC6_RTC_MASK                       0x2000u
#define SIM_SCGC6_RTC_SHIFT                      13
#define SIM_SCGC6_TSI_MASK                       0x10000u
#define SIM_SCGC6_TSI_SHIFT                      16
#define SIM_SCGC6_PORTA_MASK                     0x80000u
#define SIM_SCGC6_PORTA_SHIFT                    19
#define SIM_SCGC6_PORTB_MASK                     0x100000u
#define SIM_SCGC6_PORTB_SHIFT                    20
#define SIM_SCGC6_PORTC_MASK                     0x200000u
#define SIM_SCGC6_PORTC_SHIFT                    21
#define SIM_SCGC6_PORTD_MASK                     0x400000u
#define SIM_SCGC6_PORTD_SHIFT                    22
#define SIM_SCGC6_PORTE_MASK                     0x800000u
#define SIM_SCGC6_PORTE_SHIFT                    23
#define SIM_SCGC6_ADC0_MASK                      0x8000000u
#define SIM_SCGC6_ADC0_SHIFT                     27
#define SIM_SCGC6_ADC1_MASK                      0x10000000u
#define SIM_SCGC6_ADC1_SHIFT                     28
#define SIM_SCGC6_ADC2_MASK                      0x20000000u
#define SIM_SCGC6_ADC2_SHIFT                     29
#define SIM_SCGC6_ADC3_MASK                      0x40000000u
#define SIM_SCGC6_ADC3_SHIFT                     30
/* SCGC7 Bit Fields */
#define SIM_SCGC7_FLEXBUS_MASK                   0x1u
#define SIM_SCGC7_FLEXBUS_SHIFT                  0
#define SIM_SCGC7_DMA_MASK                       0x2u
#define SIM_SCGC7_DMA_SHIFT                      1
#define SIM_SCGC7_MPU_MASK                       0x4u
#define SIM_SCGC7_MPU_SHIFT                      2
#define SIM_SCGC7_IEVT_MASK                      0x8u
#define SIM_SCGC7_IEVT_SHIFT                     3
/* CLKDIV1 Bit Fields */
#define SIM_CLKDIV1_OUTDIV5_MASK                 0x3000u
#define SIM_CLKDIV1_OUTDIV5_SHIFT                12
#define SIM_CLKDIV1_OUTDIV5(x)                   (((uint32_t)(((uint32_t)(x))<<SIM_CLKDIV1_OUTDIV5_SHIFT))&SIM_CLKDIV1_OUTDIV5_MASK)
#define SIM_CLKDIV1_OUTDIV4_MASK                 0x30000u
#define SIM_CLKDIV1_OUTDIV4_SHIFT                16
#define SIM_CLKDIV1_OUTDIV4(x)                   (((uint32_t)(((uint32_t)(x))<<SIM_CLKDIV1_OUTDIV4_SHIFT))&SIM_CLKDIV1_OUTDIV4_MASK)
#define SIM_CLKDIV1_OUTDIV3_MASK                 0x300000u
#define SIM_CLKDIV1_OUTDIV3_SHIFT                20
#define SIM_CLKDIV1_OUTDIV3(x)                   (((uint32_t)(((uint32_t)(x))<<SIM_CLKDIV1_OUTDIV3_SHIFT))&SIM_CLKDIV1_OUTDIV3_MASK)
#define SIM_CLKDIV1_OUTDIV2_MASK                 0x3000000u
#define SIM_CLKDIV1_OUTDIV2_SHIFT                24
#define SIM_CLKDIV1_OUTDIV2(x)                   (((uint32_t)(((uint32_t)(x))<<SIM_CLKDIV1_OUTDIV2_SHIFT))&SIM_CLKDIV1_OUTDIV2_MASK)
#define SIM_CLKDIV1_OUTDIV1_MASK                 0x30000000u
#define SIM_CLKDIV1_OUTDIV1_SHIFT                28
#define SIM_CLKDIV1_OUTDIV1(x)                   (((uint32_t)(((uint32_t)(x))<<SIM_CLKDIV1_OUTDIV1_SHIFT))&SIM_CLKDIV1_OUTDIV1_MASK)
/* UIDH Bit Fields */
#define SIM_UIDH_UID_MASK                        0xFFFFFFFFu
#define SIM_UIDH_UID_SHIFT                       0
#define SIM_UIDH_UID(x)                          (((uint32_t)(((uint32_t)(x))<<SIM_UIDH_UID_SHIFT))&SIM_UIDH_UID_MASK)
/* UIDMH Bit Fields */
#define SIM_UIDMH_UID_MASK                       0xFFFFFFFFu
#define SIM_UIDMH_UID_SHIFT                      0
#define SIM_UIDMH_UID(x)                         (((uint32_t)(((uint32_t)(x))<<SIM_UIDMH_UID_SHIFT))&SIM_UIDMH_UID_MASK)
/* UIDML Bit Fields */
#define SIM_UIDML_UID_MASK                       0xFFFFFFFFu
#define SIM_UIDML_UID_SHIFT                      0
#define SIM_UIDML_UID(x)                         (((uint32_t)(((uint32_t)(x))<<SIM_UIDML_UID_SHIFT))&SIM_UIDML_UID_MASK)
/* UIDL Bit Fields */
#define SIM_UIDL_UID_MASK                        0xFFFFFFFFu
#define SIM_UIDL_UID_SHIFT                       0
#define SIM_UIDL_UID(x)                          (((uint32_t)(((uint32_t)(x))<<SIM_UIDL_UID_SHIFT))&SIM_UIDL_UID_MASK)

/**
 * @}
 */ /* end of group SIM_Register_Masks */


/* SIM - Peripheral instance base addresses */
/** Peripheral SIM base pointer */
#define SIM_BASE_PTR                             ((SIM_MemMapPtr)0x40032000u)

/* ----------------------------------------------------------------------------
   -- SIM - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup SIM_Register_Accessor_Macros SIM - Register accessor macros
 * @{
 */


/* SIM - Register instance definitions */
/* SIM */
#define SIM_SOPT2                                SIM_SOPT2_REG(SIM_BASE_PTR)
#define SIM_SOPT3                                SIM_SOPT3_REG(SIM_BASE_PTR)
#define SIM_SOPT4                                SIM_SOPT4_REG(SIM_BASE_PTR)
#define SIM_SOPT5                                SIM_SOPT5_REG(SIM_BASE_PTR)
#define SIM_SOPT6                                SIM_SOPT6_REG(SIM_BASE_PTR)
#define SIM_SOPT7                                SIM_SOPT7_REG(SIM_BASE_PTR)
#define SIM_SDID                                 SIM_SDID_REG(SIM_BASE_PTR)
#define SIM_SCGC5                                SIM_SCGC5_REG(SIM_BASE_PTR)
#define SIM_SCGC6                                SIM_SCGC6_REG(SIM_BASE_PTR)
#define SIM_SCGC7                                SIM_SCGC7_REG(SIM_BASE_PTR)
#define SIM_CLKDIV1                              SIM_CLKDIV1_REG(SIM_BASE_PTR)
#define SIM_UIDH                                 SIM_UIDH_REG(SIM_BASE_PTR)
#define SIM_UIDMH                                SIM_UIDMH_REG(SIM_BASE_PTR)
#define SIM_UIDML                                SIM_UIDML_REG(SIM_BASE_PTR)
#define SIM_UIDL                                 SIM_UIDL_REG(SIM_BASE_PTR)

/**
 * @}
 */ /* end of group SIM_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group SIM_Peripheral */


/* ----------------------------------------------------------------------------
   -- SMC
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup SMC_Peripheral SMC
 * @{
 */

/** SMC - Peripheral register structure */
typedef struct SMC_MemMap {
  uint8_t PMPROT;                                  /**< SMC Power Mode Protection Register, offset: 0x0 */
  uint8_t PMCTRL;                                  /**< SMC Power Mode Control Register, offset: 0x1 */
  uint8_t RESERVED_0[1];
  uint8_t PMSTAT;                                  /**< SMC Power Mode Status Register, offset: 0x3 */
} volatile *SMC_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- SMC - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup SMC_Register_Accessor_Macros SMC - Register accessor macros
 * @{
 */


/* SMC - Register accessors */
#define SMC_PMPROT_REG(base)                     ((base)->PMPROT)
#define SMC_PMCTRL_REG(base)                     ((base)->PMCTRL)
#define SMC_PMSTAT_REG(base)                     ((base)->PMSTAT)

/**
 * @}
 */ /* end of group SMC_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- SMC Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup SMC_Register_Masks SMC Register Masks
 * @{
 */

/* PMPROT Bit Fields */
#define SMC_PMPROT_AVLP_MASK                     0x20u
#define SMC_PMPROT_AVLP_SHIFT                    5
/* PMCTRL Bit Fields */
#define SMC_PMCTRL_STOPM_MASK                    0x7u
#define SMC_PMCTRL_STOPM_SHIFT                   0
#define SMC_PMCTRL_STOPM(x)                      (((uint8_t)(((uint8_t)(x))<<SMC_PMCTRL_STOPM_SHIFT))&SMC_PMCTRL_STOPM_MASK)
#define SMC_PMCTRL_STOPA_MASK                    0x8u
#define SMC_PMCTRL_STOPA_SHIFT                   3
/* PMSTAT Bit Fields */
#define SMC_PMSTAT_RUN_MASK                      0x1u
#define SMC_PMSTAT_RUN_SHIFT                     0
#define SMC_PMSTAT_STOP_MASK                     0x2u
#define SMC_PMSTAT_STOP_SHIFT                    1
#define SMC_PMSTAT_VLPS_MASK                     0x10u
#define SMC_PMSTAT_VLPS_SHIFT                    4

/**
 * @}
 */ /* end of group SMC_Register_Masks */


/* SMC - Peripheral instance base addresses */
/** Peripheral SMC base pointer */
#define SMC_BASE_PTR                             ((SMC_MemMapPtr)0x40056000u)

/* ----------------------------------------------------------------------------
   -- SMC - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup SMC_Register_Accessor_Macros SMC - Register accessor macros
 * @{
 */


/* SMC - Register instance definitions */
/* SMC */
#define SMC_PMPROT                               SMC_PMPROT_REG(SMC_BASE_PTR)
#define SMC_PMCTRL                               SMC_PMCTRL_REG(SMC_BASE_PTR)
#define SMC_PMSTAT                               SMC_PMSTAT_REG(SMC_BASE_PTR)

/**
 * @}
 */ /* end of group SMC_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group SMC_Peripheral */


/* ----------------------------------------------------------------------------
   -- SPI
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup SPI_Peripheral SPI
 * @{
 */

/** SPI - Peripheral register structure */
typedef struct SPI_MemMap {
  uint8_t C1;                                      /**< SPI control register 1, offset: 0x0 */
  uint8_t C2;                                      /**< SPI control register 2, offset: 0x1 */
  uint8_t BR;                                      /**< SPI baud rate register, offset: 0x2 */
  uint8_t S;                                       /**< SPI status register, offset: 0x3 */
  uint8_t DH;                                      /**< SPI data register high, offset: 0x4 */
  uint8_t DL;                                      /**< SPI data register low, offset: 0x5 */
  uint8_t MH;                                      /**< SPI match register high, offset: 0x6 */
  uint8_t ML;                                      /**< SPI match register low, offset: 0x7 */
  uint8_t C3;                                      /**< SPI control register 3, offset: 0x8 */
  uint8_t CI;                                      /**< SPI clear interrupt register, offset: 0x9 */
} volatile *SPI_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- SPI - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup SPI_Register_Accessor_Macros SPI - Register accessor macros
 * @{
 */


/* SPI - Register accessors */
#define SPI_C1_REG(base)                         ((base)->C1)
#define SPI_C2_REG(base)                         ((base)->C2)
#define SPI_BR_REG(base)                         ((base)->BR)
#define SPI_S_REG(base)                          ((base)->S)
#define SPI_DH_REG(base)                         ((base)->DH)
#define SPI_DL_REG(base)                         ((base)->DL)
#define SPI_MH_REG(base)                         ((base)->MH)
#define SPI_ML_REG(base)                         ((base)->ML)
#define SPI_C3_REG(base)                         ((base)->C3)
#define SPI_CI_REG(base)                         ((base)->CI)

/**
 * @}
 */ /* end of group SPI_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- SPI Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup SPI_Register_Masks SPI Register Masks
 * @{
 */

/* C1 Bit Fields */
#define SPI_C1_LSBFE_MASK                        0x1u
#define SPI_C1_LSBFE_SHIFT                       0
#define SPI_C1_SSOE_MASK                         0x2u
#define SPI_C1_SSOE_SHIFT                        1
#define SPI_C1_CPHA_MASK                         0x4u
#define SPI_C1_CPHA_SHIFT                        2
#define SPI_C1_CPOL_MASK                         0x8u
#define SPI_C1_CPOL_SHIFT                        3
#define SPI_C1_MSTR_MASK                         0x10u
#define SPI_C1_MSTR_SHIFT                        4
#define SPI_C1_SPTIE_MASK                        0x20u
#define SPI_C1_SPTIE_SHIFT                       5
#define SPI_C1_SPE_MASK                          0x40u
#define SPI_C1_SPE_SHIFT                         6
#define SPI_C1_SPIE_MASK                         0x80u
#define SPI_C1_SPIE_SHIFT                        7
/* C2 Bit Fields */
#define SPI_C2_SPC0_MASK                         0x1u
#define SPI_C2_SPC0_SHIFT                        0
#define SPI_C2_SPISWAI_MASK                      0x2u
#define SPI_C2_SPISWAI_SHIFT                     1
#define SPI_C2_RXDMAE_MASK                       0x4u
#define SPI_C2_RXDMAE_SHIFT                      2
#define SPI_C2_BIDIROE_MASK                      0x8u
#define SPI_C2_BIDIROE_SHIFT                     3
#define SPI_C2_MODFEN_MASK                       0x10u
#define SPI_C2_MODFEN_SHIFT                      4
#define SPI_C2_TXDMAE_MASK                       0x20u
#define SPI_C2_TXDMAE_SHIFT                      5
#define SPI_C2_SPIMODE_MASK                      0x40u
#define SPI_C2_SPIMODE_SHIFT                     6
#define SPI_C2_SPMIE_MASK                        0x80u
#define SPI_C2_SPMIE_SHIFT                       7
/* BR Bit Fields */
#define SPI_BR_SPR_MASK                          0xFu
#define SPI_BR_SPR_SHIFT                         0
#define SPI_BR_SPR(x)                            (((uint8_t)(((uint8_t)(x))<<SPI_BR_SPR_SHIFT))&SPI_BR_SPR_MASK)
#define SPI_BR_SPPR_MASK                         0x70u
#define SPI_BR_SPPR_SHIFT                        4
#define SPI_BR_SPPR(x)                           (((uint8_t)(((uint8_t)(x))<<SPI_BR_SPPR_SHIFT))&SPI_BR_SPPR_MASK)
/* S Bit Fields */
#define SPI_S_RFIFOEF_MASK                       0x1u
#define SPI_S_RFIFOEF_SHIFT                      0
#define SPI_S_TXFULLF_MASK                       0x2u
#define SPI_S_TXFULLF_SHIFT                      1
#define SPI_S_TNEAREF_MASK                       0x4u
#define SPI_S_TNEAREF_SHIFT                      2
#define SPI_S_RNFULLF_MASK                       0x8u
#define SPI_S_RNFULLF_SHIFT                      3
#define SPI_S_MODF_MASK                          0x10u
#define SPI_S_MODF_SHIFT                         4
#define SPI_S_SPTEF_MASK                         0x20u
#define SPI_S_SPTEF_SHIFT                        5
#define SPI_S_SPMF_MASK                          0x40u
#define SPI_S_SPMF_SHIFT                         6
#define SPI_S_SPRF_MASK                          0x80u
#define SPI_S_SPRF_SHIFT                         7
/* DH Bit Fields */
#define SPI_DH_Bits_MASK                         0xFFu
#define SPI_DH_Bits_SHIFT                        0
#define SPI_DH_Bits(x)                           (((uint8_t)(((uint8_t)(x))<<SPI_DH_Bits_SHIFT))&SPI_DH_Bits_MASK)
/* DL Bit Fields */
#define SPI_DL_Bits_MASK                         0xFFu
#define SPI_DL_Bits_SHIFT                        0
#define SPI_DL_Bits(x)                           (((uint8_t)(((uint8_t)(x))<<SPI_DL_Bits_SHIFT))&SPI_DL_Bits_MASK)
/* MH Bit Fields */
#define SPI_MH_Bits_MASK                         0xFFu
#define SPI_MH_Bits_SHIFT                        0
#define SPI_MH_Bits(x)                           (((uint8_t)(((uint8_t)(x))<<SPI_MH_Bits_SHIFT))&SPI_MH_Bits_MASK)
/* ML Bit Fields */
#define SPI_ML_Bits_MASK                         0xFFu
#define SPI_ML_Bits_SHIFT                        0
#define SPI_ML_Bits(x)                           (((uint8_t)(((uint8_t)(x))<<SPI_ML_Bits_SHIFT))&SPI_ML_Bits_MASK)
/* C3 Bit Fields */
#define SPI_C3_FIFOMODE_MASK                     0x1u
#define SPI_C3_FIFOMODE_SHIFT                    0
#define SPI_C3_RNFULLIEN_MASK                    0x2u
#define SPI_C3_RNFULLIEN_SHIFT                   1
#define SPI_C3_TNEARIEN_MASK                     0x4u
#define SPI_C3_TNEARIEN_SHIFT                    2
#define SPI_C3_INTCLR_MASK                       0x8u
#define SPI_C3_INTCLR_SHIFT                      3
#define SPI_C3_RNFULLF_MARK_MASK                 0x10u
#define SPI_C3_RNFULLF_MARK_SHIFT                4
#define SPI_C3_TNEAREF_MARK_MASK                 0x20u
#define SPI_C3_TNEAREF_MARK_SHIFT                5
/* CI Bit Fields */
#define SPI_CI_SPRFCI_MASK                       0x1u
#define SPI_CI_SPRFCI_SHIFT                      0
#define SPI_CI_SPTEFCI_MASK                      0x2u
#define SPI_CI_SPTEFCI_SHIFT                     1
#define SPI_CI_RNFULLFCI_MASK                    0x4u
#define SPI_CI_RNFULLFCI_SHIFT                   2
#define SPI_CI_TNEAREFCI_MASK                    0x8u
#define SPI_CI_TNEAREFCI_SHIFT                   3
#define SPI_CI_RXFOF_MASK                        0x10u
#define SPI_CI_RXFOF_SHIFT                       4
#define SPI_CI_TXFOF_MASK                        0x20u
#define SPI_CI_TXFOF_SHIFT                       5
#define SPI_CI_RXFERR_MASK                       0x40u
#define SPI_CI_RXFERR_SHIFT                      6
#define SPI_CI_TXFERR_MASK                       0x80u
#define SPI_CI_TXFERR_SHIFT                      7

/**
 * @}
 */ /* end of group SPI_Register_Masks */


/* SPI - Peripheral instance base addresses */
/** Peripheral SPI0 base pointer */
#define SPI0_BASE_PTR                            ((SPI_MemMapPtr)0x4004E000u)
/** Peripheral SPI1 base pointer */
#define SPI1_BASE_PTR                            ((SPI_MemMapPtr)0x4004F000u)

/* ----------------------------------------------------------------------------
   -- SPI - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup SPI_Register_Accessor_Macros SPI - Register accessor macros
 * @{
 */


/* SPI - Register instance definitions */
/* SPI0 */
#define SPI0_C1                                  SPI_C1_REG(SPI0_BASE_PTR)
#define SPI0_C2                                  SPI_C2_REG(SPI0_BASE_PTR)
#define SPI0_BR                                  SPI_BR_REG(SPI0_BASE_PTR)
#define SPI0_S                                   SPI_S_REG(SPI0_BASE_PTR)
#define SPI0_DH                                  SPI_DH_REG(SPI0_BASE_PTR)
#define SPI0_DL                                  SPI_DL_REG(SPI0_BASE_PTR)
#define SPI0_MH                                  SPI_MH_REG(SPI0_BASE_PTR)
#define SPI0_ML                                  SPI_ML_REG(SPI0_BASE_PTR)
#define SPI0_C3                                  SPI_C3_REG(SPI0_BASE_PTR)
#define SPI0_CI                                  SPI_CI_REG(SPI0_BASE_PTR)
/* SPI1 */
#define SPI1_C1                                  SPI_C1_REG(SPI1_BASE_PTR)
#define SPI1_C2                                  SPI_C2_REG(SPI1_BASE_PTR)
#define SPI1_BR                                  SPI_BR_REG(SPI1_BASE_PTR)
#define SPI1_S                                   SPI_S_REG(SPI1_BASE_PTR)
#define SPI1_DH                                  SPI_DH_REG(SPI1_BASE_PTR)
#define SPI1_DL                                  SPI_DL_REG(SPI1_BASE_PTR)
#define SPI1_MH                                  SPI_MH_REG(SPI1_BASE_PTR)
#define SPI1_ML                                  SPI_ML_REG(SPI1_BASE_PTR)
#define SPI1_C3                                  SPI_C3_REG(SPI1_BASE_PTR)
#define SPI1_CI                                  SPI_CI_REG(SPI1_BASE_PTR)

/**
 * @}
 */ /* end of group SPI_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group SPI_Peripheral */


/* ----------------------------------------------------------------------------
   -- SysTick
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup SysTick_Peripheral SysTick
 * @{
 */

/** SysTick - Peripheral register structure */
typedef struct SysTick_MemMap {
  uint32_t CSR;                                    /**< SysTick Control and Status Register, offset: 0x0 */
  uint32_t RVR;                                    /**< SysTick Reload Value Register, offset: 0x4 */
  uint32_t CVR;                                    /**< SysTick Current Value Register, offset: 0x8 */
  uint32_t CALIB;                                  /**< SysTick Calibration Value Register, offset: 0xC */
} volatile *SysTick_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- SysTick - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup SysTick_Register_Accessor_Macros SysTick - Register accessor macros
 * @{
 */


/* SysTick - Register accessors */
#define SysTick_CSR_REG(base)                    ((base)->CSR)
#define SysTick_RVR_REG(base)                    ((base)->RVR)
#define SysTick_CVR_REG(base)                    ((base)->CVR)
#define SysTick_CALIB_REG(base)                  ((base)->CALIB)

/**
 * @}
 */ /* end of group SysTick_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- SysTick Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup SysTick_Register_Masks SysTick Register Masks
 * @{
 */

/* CSR Bit Fields */
#define SysTick_CSR_ENABLE_MASK                  0x1u
#define SysTick_CSR_ENABLE_SHIFT                 0
#define SysTick_CSR_TICKINT_MASK                 0x2u
#define SysTick_CSR_TICKINT_SHIFT                1
#define SysTick_CSR_CLKSOURCE_MASK               0x4u
#define SysTick_CSR_CLKSOURCE_SHIFT              2
#define SysTick_CSR_COUNTFLAG_MASK               0x10000u
#define SysTick_CSR_COUNTFLAG_SHIFT              16
/* RVR Bit Fields */
#define SysTick_RVR_RELOAD_MASK                  0xFFFFFFu
#define SysTick_RVR_RELOAD_SHIFT                 0
#define SysTick_RVR_RELOAD(x)                    (((uint32_t)(((uint32_t)(x))<<SysTick_RVR_RELOAD_SHIFT))&SysTick_RVR_RELOAD_MASK)
/* CVR Bit Fields */
#define SysTick_CVR_CURRENT_MASK                 0xFFFFFFu
#define SysTick_CVR_CURRENT_SHIFT                0
#define SysTick_CVR_CURRENT(x)                   (((uint32_t)(((uint32_t)(x))<<SysTick_CVR_CURRENT_SHIFT))&SysTick_CVR_CURRENT_MASK)
/* CALIB Bit Fields */
#define SysTick_CALIB_TENMS_MASK                 0xFFFFFFu
#define SysTick_CALIB_TENMS_SHIFT                0
#define SysTick_CALIB_TENMS(x)                   (((uint32_t)(((uint32_t)(x))<<SysTick_CALIB_TENMS_SHIFT))&SysTick_CALIB_TENMS_MASK)
#define SysTick_CALIB_SKEW_MASK                  0x40000000u
#define SysTick_CALIB_SKEW_SHIFT                 30
#define SysTick_CALIB_NOREF_MASK                 0x80000000u
#define SysTick_CALIB_NOREF_SHIFT                31

/**
 * @}
 */ /* end of group SysTick_Register_Masks */


/* SysTick - Peripheral instance base addresses */
/** Peripheral SysTick base pointer */
#define SysTick_BASE_PTR                         ((SysTick_MemMapPtr)0xE000E010u)

/* ----------------------------------------------------------------------------
   -- SysTick - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup SysTick_Register_Accessor_Macros SysTick - Register accessor macros
 * @{
 */


/* SysTick - Register instance definitions */
/* SysTick */
#define SYST_CSR                                 SysTick_CSR_REG(SysTick_BASE_PTR)
#define SYST_RVR                                 SysTick_RVR_REG(SysTick_BASE_PTR)
#define SYST_CVR                                 SysTick_CVR_REG(SysTick_BASE_PTR)
#define SYST_CALIB                               SysTick_CALIB_REG(SysTick_BASE_PTR)

/**
 * @}
 */ /* end of group SysTick_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group SysTick_Peripheral */


/* ----------------------------------------------------------------------------
   -- TPIU
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup TPIU_Peripheral TPIU
 * @{
 */

/** TPIU - Peripheral register structure */
typedef struct TPIU_MemMap {
  uint32_t SSPSR;                                  /**< Supported Parallel Port Size Register, offset: 0x0 */
  uint32_t CSPSR;                                  /**< Current Parallel Port Size Register, offset: 0x4 */
  uint8_t RESERVED_0[8];
  uint32_t ACPR;                                   /**< Asynchronous Clock Prescaler Register, offset: 0x10 */
  uint8_t RESERVED_1[220];
  uint32_t SPPR;                                   /**< Selected Pin Protocol Register, offset: 0xF0 */
  uint8_t RESERVED_2[524];
  uint32_t FFSR;                                   /**< Formatter and Flush Status Register, offset: 0x300 */
  uint32_t FFCR;                                   /**< Formatter and Flush Control Register, offset: 0x304 */
  uint32_t FSCR;                                   /**< Formatter Synchronization Counter Register, offset: 0x308 */
  uint8_t RESERVED_3[3036];
  uint32_t TRIGGER;                                /**< Trigger Register, offset: 0xEE8 */
  uint32_t FIFODATA0;                              /**< FIFODATA0 Register, offset: 0xEEC */
  uint32_t ITATBCTR2;                              /**< Integration Test ATB Control 2 Register, offset: 0xEF0 */
  uint8_t RESERVED_4[4];
  uint32_t ITATBCTR0;                              /**< Integration Test ATB Control 0 Register, offset: 0xEF8 */
  uint32_t FIFODATA1;                              /**< FIFODATA1 Register, offset: 0xEFC */
  uint32_t ITCTRL;                                 /**< Integration Mode Control Register, offset: 0xF00 */
  uint8_t RESERVED_5[156];
  uint32_t CLAIMSET;                               /**< Claim Tag Set Register, offset: 0xFA0 */
  uint32_t CLAIMCLR;                               /**< Claim Tag Clear Register, offset: 0xFA4 */
  uint8_t RESERVED_6[32];
  uint32_t DEVID;                                  /**< TPIU_DEVID Register, offset: 0xFC8 */
  uint8_t RESERVED_7[4];
  uint32_t PID4;                                   /**< Peripheral Identification Register 4., offset: 0xFD0 */
  uint32_t PID5;                                   /**< Peripheral Identification Register 5., offset: 0xFD4 */
  uint32_t PID6;                                   /**< Peripheral Identification Register 6., offset: 0xFD8 */
  uint32_t PID7;                                   /**< Peripheral Identification Register 7., offset: 0xFDC */
  uint32_t PID0;                                   /**< Peripheral Identification Register 0., offset: 0xFE0 */
  uint32_t PID1;                                   /**< Peripheral Identification Register 1., offset: 0xFE4 */
  uint32_t PID2;                                   /**< Peripheral Identification Register 2., offset: 0xFE8 */
  uint32_t PID3;                                   /**< Peripheral Identification Register 3., offset: 0xFEC */
  uint32_t CID0;                                   /**< Component Identification Register 0., offset: 0xFF0 */
  uint32_t CID1;                                   /**< Component Identification Register 1., offset: 0xFF4 */
  uint32_t CID2;                                   /**< Component Identification Register 2., offset: 0xFF8 */
  uint32_t CID4;                                   /**< Component Identification Register 3., offset: 0xFFC */
} volatile *TPIU_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- TPIU - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup TPIU_Register_Accessor_Macros TPIU - Register accessor macros
 * @{
 */


/* TPIU - Register accessors */
#define TPIU_SSPSR_REG(base)                     ((base)->SSPSR)
#define TPIU_CSPSR_REG(base)                     ((base)->CSPSR)
#define TPIU_ACPR_REG(base)                      ((base)->ACPR)
#define TPIU_SPPR_REG(base)                      ((base)->SPPR)
#define TPIU_FFSR_REG(base)                      ((base)->FFSR)
#define TPIU_FFCR_REG(base)                      ((base)->FFCR)
#define TPIU_FSCR_REG(base)                      ((base)->FSCR)
#define TPIU_TRIGGER_REG(base)                   ((base)->TRIGGER)
#define TPIU_FIFODATA0_REG(base)                 ((base)->FIFODATA0)
#define TPIU_ITATBCTR2_REG(base)                 ((base)->ITATBCTR2)
#define TPIU_ITATBCTR0_REG(base)                 ((base)->ITATBCTR0)
#define TPIU_FIFODATA1_REG(base)                 ((base)->FIFODATA1)
#define TPIU_ITCTRL_REG(base)                    ((base)->ITCTRL)
#define TPIU_CLAIMSET_REG(base)                  ((base)->CLAIMSET)
#define TPIU_CLAIMCLR_REG(base)                  ((base)->CLAIMCLR)
#define TPIU_DEVID_REG(base)                     ((base)->DEVID)
#define TPIU_PID4_REG(base)                      ((base)->PID4)
#define TPIU_PID5_REG(base)                      ((base)->PID5)
#define TPIU_PID6_REG(base)                      ((base)->PID6)
#define TPIU_PID7_REG(base)                      ((base)->PID7)
#define TPIU_PID0_REG(base)                      ((base)->PID0)
#define TPIU_PID1_REG(base)                      ((base)->PID1)
#define TPIU_PID2_REG(base)                      ((base)->PID2)
#define TPIU_PID3_REG(base)                      ((base)->PID3)
#define TPIU_CID0_REG(base)                      ((base)->CID0)
#define TPIU_CID1_REG(base)                      ((base)->CID1)
#define TPIU_CID2_REG(base)                      ((base)->CID2)
#define TPIU_CID4_REG(base)                      ((base)->CID4)

/**
 * @}
 */ /* end of group TPIU_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- TPIU Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup TPIU_Register_Masks TPIU Register Masks
 * @{
 */


/**
 * @}
 */ /* end of group TPIU_Register_Masks */


/* TPIU - Peripheral instance base addresses */
/** Peripheral TPIU base pointer */
#define TPIU_BASE_PTR                            ((TPIU_MemMapPtr)0xE0040000u)

/* ----------------------------------------------------------------------------
   -- TPIU - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup TPIU_Register_Accessor_Macros TPIU - Register accessor macros
 * @{
 */


/* TPIU - Register instance definitions */
/* TPIU */
#define TPIU_SSPSR                               TPIU_SSPSR_REG(TPIU_BASE_PTR)
#define TPIU_CSPSR                               TPIU_CSPSR_REG(TPIU_BASE_PTR)
#define TPIU_ACPR                                TPIU_ACPR_REG(TPIU_BASE_PTR)
#define TPIU_SPPR                                TPIU_SPPR_REG(TPIU_BASE_PTR)
#define TPIU_FFSR                                TPIU_FFSR_REG(TPIU_BASE_PTR)
#define TPIU_FFCR                                TPIU_FFCR_REG(TPIU_BASE_PTR)
#define TPIU_FSCR                                TPIU_FSCR_REG(TPIU_BASE_PTR)
#define TPIU_TRIGGER                             TPIU_TRIGGER_REG(TPIU_BASE_PTR)
#define TPIU_FIFODATA0                           TPIU_FIFODATA0_REG(TPIU_BASE_PTR)
#define TPIU_ITATBCTR2                           TPIU_ITATBCTR2_REG(TPIU_BASE_PTR)
#define TPIU_ITATBCTR0                           TPIU_ITATBCTR0_REG(TPIU_BASE_PTR)
#define TPIU_FIFODATA1                           TPIU_FIFODATA1_REG(TPIU_BASE_PTR)
#define TPIU_ITCTRL                              TPIU_ITCTRL_REG(TPIU_BASE_PTR)
#define TPIU_CLAIMSET                            TPIU_CLAIMSET_REG(TPIU_BASE_PTR)
#define TPIU_CLAIMCLR                            TPIU_CLAIMCLR_REG(TPIU_BASE_PTR)
#define TPIU_DEVID                               TPIU_DEVID_REG(TPIU_BASE_PTR)
#define TPIU_PID4                                TPIU_PID4_REG(TPIU_BASE_PTR)
#define TPIU_PID5                                TPIU_PID5_REG(TPIU_BASE_PTR)
#define TPIU_PID6                                TPIU_PID6_REG(TPIU_BASE_PTR)
#define TPIU_PID7                                TPIU_PID7_REG(TPIU_BASE_PTR)
#define TPIU_PID0                                TPIU_PID0_REG(TPIU_BASE_PTR)
#define TPIU_PID1                                TPIU_PID1_REG(TPIU_BASE_PTR)
#define TPIU_PID2                                TPIU_PID2_REG(TPIU_BASE_PTR)
#define TPIU_PID3                                TPIU_PID3_REG(TPIU_BASE_PTR)
#define TPIU_CID0                                TPIU_CID0_REG(TPIU_BASE_PTR)
#define TPIU_CID1                                TPIU_CID1_REG(TPIU_BASE_PTR)
#define TPIU_CID2                                TPIU_CID2_REG(TPIU_BASE_PTR)
#define TPIU_CID3                                TPIU_CID4_REG(TPIU_BASE_PTR)

/**
 * @}
 */ /* end of group TPIU_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group TPIU_Peripheral */


/* ----------------------------------------------------------------------------
   -- UART
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup UART_Peripheral UART
 * @{
 */

/** UART - Peripheral register structure */
typedef struct UART_MemMap {
  uint8_t BDH;                                     /**< UART Baud Rate Registers: High, offset: 0x0 */
  uint8_t BDL;                                     /**< UART Baud Rate Registers: Low, offset: 0x1 */
  uint8_t C1;                                      /**< UART Control Register 1, offset: 0x2 */
  uint8_t C2;                                      /**< UART Control Register 2, offset: 0x3 */
  uint8_t S1;                                      /**< UART Status Register 1, offset: 0x4 */
  uint8_t S2;                                      /**< UART Status Register 2, offset: 0x5 */
  uint8_t C3;                                      /**< UART Control Register 3, offset: 0x6 */
  uint8_t D;                                       /**< UART Data Register, offset: 0x7 */
  uint8_t MA1;                                     /**< UART Match Address Registers 1, offset: 0x8 */
  uint8_t MA2;                                     /**< UART Match Address Registers 2, offset: 0x9 */
  uint8_t C4;                                      /**< UART Control Register 4, offset: 0xA */
  uint8_t C5;                                      /**< UART Control Register 5, offset: 0xB */
  uint8_t ED;                                      /**< UART Extended Data Register, offset: 0xC */
  uint8_t MODEM;                                   /**< UART Modem Register, offset: 0xD */
  uint8_t IR;                                      /**< UART Infrared Register, offset: 0xE */
  uint8_t RESERVED_0[1];
  uint8_t PFIFO;                                   /**< UART FIFO Parameters, offset: 0x10 */
  uint8_t CFIFO;                                   /**< UART FIFO Control Register, offset: 0x11 */
  uint8_t SFIFO;                                   /**< UART FIFO Status Register, offset: 0x12 */
  uint8_t TWFIFO;                                  /**< UART FIFO Transmit Watermark, offset: 0x13 */
  uint8_t TCFIFO;                                  /**< UART FIFO Transmit Count, offset: 0x14 */
  uint8_t RWFIFO;                                  /**< UART FIFO Receive Watermark, offset: 0x15 */
  uint8_t RCFIFO;                                  /**< UART FIFO Receive Count, offset: 0x16 */
} volatile *UART_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- UART - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup UART_Register_Accessor_Macros UART - Register accessor macros
 * @{
 */


/* UART - Register accessors */
#define UART_BDH_REG(base)                       ((base)->BDH)
#define UART_BDL_REG(base)                       ((base)->BDL)
#define UART_C1_REG(base)                        ((base)->C1)
#define UART_C2_REG(base)                        ((base)->C2)
#define UART_S1_REG(base)                        ((base)->S1)
#define UART_S2_REG(base)                        ((base)->S2)
#define UART_C3_REG(base)                        ((base)->C3)
#define UART_D_REG(base)                         ((base)->D)
#define UART_MA1_REG(base)                       ((base)->MA1)
#define UART_MA2_REG(base)                       ((base)->MA2)
#define UART_C4_REG(base)                        ((base)->C4)
#define UART_C5_REG(base)                        ((base)->C5)
#define UART_ED_REG(base)                        ((base)->ED)
#define UART_MODEM_REG(base)                     ((base)->MODEM)
#define UART_IR_REG(base)                        ((base)->IR)
#define UART_PFIFO_REG(base)                     ((base)->PFIFO)
#define UART_CFIFO_REG(base)                     ((base)->CFIFO)
#define UART_SFIFO_REG(base)                     ((base)->SFIFO)
#define UART_TWFIFO_REG(base)                    ((base)->TWFIFO)
#define UART_TCFIFO_REG(base)                    ((base)->TCFIFO)
#define UART_RWFIFO_REG(base)                    ((base)->RWFIFO)
#define UART_RCFIFO_REG(base)                    ((base)->RCFIFO)

/**
 * @}
 */ /* end of group UART_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- UART Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup UART_Register_Masks UART Register Masks
 * @{
 */

/* BDH Bit Fields */
#define UART_BDH_SBR_MASK                        0x1Fu
#define UART_BDH_SBR_SHIFT                       0
#define UART_BDH_SBR(x)                          (((uint8_t)(((uint8_t)(x))<<UART_BDH_SBR_SHIFT))&UART_BDH_SBR_MASK)
#define UART_BDH_RXEDGIE_MASK                    0x40u
#define UART_BDH_RXEDGIE_SHIFT                   6
#define UART_BDH_LBKDIE_MASK                     0x80u
#define UART_BDH_LBKDIE_SHIFT                    7
/* BDL Bit Fields */
#define UART_BDL_SBR_MASK                        0xFFu
#define UART_BDL_SBR_SHIFT                       0
#define UART_BDL_SBR(x)                          (((uint8_t)(((uint8_t)(x))<<UART_BDL_SBR_SHIFT))&UART_BDL_SBR_MASK)
/* C1 Bit Fields */
#define UART_C1_PT_MASK                          0x1u
#define UART_C1_PT_SHIFT                         0
#define UART_C1_PE_MASK                          0x2u
#define UART_C1_PE_SHIFT                         1
#define UART_C1_ILT_MASK                         0x4u
#define UART_C1_ILT_SHIFT                        2
#define UART_C1_WAKE_MASK                        0x8u
#define UART_C1_WAKE_SHIFT                       3
#define UART_C1_M_MASK                           0x10u
#define UART_C1_M_SHIFT                          4
#define UART_C1_RSRC_MASK                        0x20u
#define UART_C1_RSRC_SHIFT                       5
#define UART_C1_UARTSWAI_MASK                    0x40u
#define UART_C1_UARTSWAI_SHIFT                   6
#define UART_C1_LOOPS_MASK                       0x80u
#define UART_C1_LOOPS_SHIFT                      7
/* C2 Bit Fields */
#define UART_C2_SBK_MASK                         0x1u
#define UART_C2_SBK_SHIFT                        0
#define UART_C2_RWU_MASK                         0x2u
#define UART_C2_RWU_SHIFT                        1
#define UART_C2_RE_MASK                          0x4u
#define UART_C2_RE_SHIFT                         2
#define UART_C2_TE_MASK                          0x8u
#define UART_C2_TE_SHIFT                         3
#define UART_C2_ILIE_MASK                        0x10u
#define UART_C2_ILIE_SHIFT                       4
#define UART_C2_RIE_MASK                         0x20u
#define UART_C2_RIE_SHIFT                        5
#define UART_C2_TCIE_MASK                        0x40u
#define UART_C2_TCIE_SHIFT                       6
#define UART_C2_TIE_MASK                         0x80u
#define UART_C2_TIE_SHIFT                        7
/* S1 Bit Fields */
#define UART_S1_PF_MASK                          0x1u
#define UART_S1_PF_SHIFT                         0
#define UART_S1_FE_MASK                          0x2u
#define UART_S1_FE_SHIFT                         1
#define UART_S1_NF_MASK                          0x4u
#define UART_S1_NF_SHIFT                         2
#define UART_S1_OR_MASK                          0x8u
#define UART_S1_OR_SHIFT                         3
#define UART_S1_IDLE_MASK                        0x10u
#define UART_S1_IDLE_SHIFT                       4
#define UART_S1_RDRF_MASK                        0x20u
#define UART_S1_RDRF_SHIFT                       5
#define UART_S1_TC_MASK                          0x40u
#define UART_S1_TC_SHIFT                         6
#define UART_S1_TDRE_MASK                        0x80u
#define UART_S1_TDRE_SHIFT                       7
/* S2 Bit Fields */
#define UART_S2_RAF_MASK                         0x1u
#define UART_S2_RAF_SHIFT                        0
#define UART_S2_LBKDE_MASK                       0x2u
#define UART_S2_LBKDE_SHIFT                      1
#define UART_S2_BRK13_MASK                       0x4u
#define UART_S2_BRK13_SHIFT                      2
#define UART_S2_RWUID_MASK                       0x8u
#define UART_S2_RWUID_SHIFT                      3
#define UART_S2_RXINV_MASK                       0x10u
#define UART_S2_RXINV_SHIFT                      4
#define UART_S2_MSBF_MASK                        0x20u
#define UART_S2_MSBF_SHIFT                       5
#define UART_S2_RXEDGIF_MASK                     0x40u
#define UART_S2_RXEDGIF_SHIFT                    6
#define UART_S2_LBKDIF_MASK                      0x80u
#define UART_S2_LBKDIF_SHIFT                     7
/* C3 Bit Fields */
#define UART_C3_PEIE_MASK                        0x1u
#define UART_C3_PEIE_SHIFT                       0
#define UART_C3_FEIE_MASK                        0x2u
#define UART_C3_FEIE_SHIFT                       1
#define UART_C3_NEIE_MASK                        0x4u
#define UART_C3_NEIE_SHIFT                       2
#define UART_C3_ORIE_MASK                        0x8u
#define UART_C3_ORIE_SHIFT                       3
#define UART_C3_TXINV_MASK                       0x10u
#define UART_C3_TXINV_SHIFT                      4
#define UART_C3_TXDIR_MASK                       0x20u
#define UART_C3_TXDIR_SHIFT                      5
#define UART_C3_T8_MASK                          0x40u
#define UART_C3_T8_SHIFT                         6
#define UART_C3_R8_MASK                          0x80u
#define UART_C3_R8_SHIFT                         7
/* D Bit Fields */
#define UART_D_RT_MASK                           0xFFu
#define UART_D_RT_SHIFT                          0
#define UART_D_RT(x)                             (((uint8_t)(((uint8_t)(x))<<UART_D_RT_SHIFT))&UART_D_RT_MASK)
/* MA1 Bit Fields */
#define UART_MA1_MA_MASK                         0xFFu
#define UART_MA1_MA_SHIFT                        0
#define UART_MA1_MA(x)                           (((uint8_t)(((uint8_t)(x))<<UART_MA1_MA_SHIFT))&UART_MA1_MA_MASK)
/* MA2 Bit Fields */
#define UART_MA2_MA_MASK                         0xFFu
#define UART_MA2_MA_SHIFT                        0
#define UART_MA2_MA(x)                           (((uint8_t)(((uint8_t)(x))<<UART_MA2_MA_SHIFT))&UART_MA2_MA_MASK)
/* C4 Bit Fields */
#define UART_C4_BRFA_MASK                        0x1Fu
#define UART_C4_BRFA_SHIFT                       0
#define UART_C4_BRFA(x)                          (((uint8_t)(((uint8_t)(x))<<UART_C4_BRFA_SHIFT))&UART_C4_BRFA_MASK)
#define UART_C4_M10_MASK                         0x20u
#define UART_C4_M10_SHIFT                        5
#define UART_C4_MAEN2_MASK                       0x40u
#define UART_C4_MAEN2_SHIFT                      6
#define UART_C4_MAEN1_MASK                       0x80u
#define UART_C4_MAEN1_SHIFT                      7
/* C5 Bit Fields */
#define UART_C5_LBKDDMAS_MASK                    0x8u
#define UART_C5_LBKDDMAS_SHIFT                   3
#define UART_C5_ILDMAS_MASK                      0x10u
#define UART_C5_ILDMAS_SHIFT                     4
#define UART_C5_RDMAS_MASK                       0x20u
#define UART_C5_RDMAS_SHIFT                      5
#define UART_C5_TCDMAS_MASK                      0x40u
#define UART_C5_TCDMAS_SHIFT                     6
#define UART_C5_TDMAS_MASK                       0x80u
#define UART_C5_TDMAS_SHIFT                      7
/* ED Bit Fields */
#define UART_ED_PARITYE_MASK                     0x40u
#define UART_ED_PARITYE_SHIFT                    6
#define UART_ED_NOISY_MASK                       0x80u
#define UART_ED_NOISY_SHIFT                      7
/* MODEM Bit Fields */
#define UART_MODEM_TXCTSE_MASK                   0x1u
#define UART_MODEM_TXCTSE_SHIFT                  0
#define UART_MODEM_TXRTSE_MASK                   0x2u
#define UART_MODEM_TXRTSE_SHIFT                  1
#define UART_MODEM_TXRTSPOL_MASK                 0x4u
#define UART_MODEM_TXRTSPOL_SHIFT                2
#define UART_MODEM_RXRTSE_MASK                   0x8u
#define UART_MODEM_RXRTSE_SHIFT                  3
/* IR Bit Fields */
#define UART_IR_TNP_MASK                         0x3u
#define UART_IR_TNP_SHIFT                        0
#define UART_IR_TNP(x)                           (((uint8_t)(((uint8_t)(x))<<UART_IR_TNP_SHIFT))&UART_IR_TNP_MASK)
#define UART_IR_IREN_MASK                        0x4u
#define UART_IR_IREN_SHIFT                       2
/* PFIFO Bit Fields */
#define UART_PFIFO_RXFIFOSIZE_MASK               0x7u
#define UART_PFIFO_RXFIFOSIZE_SHIFT              0
#define UART_PFIFO_RXFIFOSIZE(x)                 (((uint8_t)(((uint8_t)(x))<<UART_PFIFO_RXFIFOSIZE_SHIFT))&UART_PFIFO_RXFIFOSIZE_MASK)
#define UART_PFIFO_RXFE_MASK                     0x8u
#define UART_PFIFO_RXFE_SHIFT                    3
#define UART_PFIFO_TXFIFOSIZE_MASK               0x70u
#define UART_PFIFO_TXFIFOSIZE_SHIFT              4
#define UART_PFIFO_TXFIFOSIZE(x)                 (((uint8_t)(((uint8_t)(x))<<UART_PFIFO_TXFIFOSIZE_SHIFT))&UART_PFIFO_TXFIFOSIZE_MASK)
#define UART_PFIFO_TXFE_MASK                     0x80u
#define UART_PFIFO_TXFE_SHIFT                    7
/* CFIFO Bit Fields */
#define UART_CFIFO_RXUFE_MASK                    0x1u
#define UART_CFIFO_RXUFE_SHIFT                   0
#define UART_CFIFO_TXOFE_MASK                    0x2u
#define UART_CFIFO_TXOFE_SHIFT                   1
#define UART_CFIFO_RXOFE_MASK                    0x4u
#define UART_CFIFO_RXOFE_SHIFT                   2
#define UART_CFIFO_RXFLUSH_MASK                  0x40u
#define UART_CFIFO_RXFLUSH_SHIFT                 6
#define UART_CFIFO_TXFLUSH_MASK                  0x80u
#define UART_CFIFO_TXFLUSH_SHIFT                 7
/* SFIFO Bit Fields */
#define UART_SFIFO_RXUF_MASK                     0x1u
#define UART_SFIFO_RXUF_SHIFT                    0
#define UART_SFIFO_TXOF_MASK                     0x2u
#define UART_SFIFO_TXOF_SHIFT                    1
#define UART_SFIFO_RXOF_MASK                     0x4u
#define UART_SFIFO_RXOF_SHIFT                    2
#define UART_SFIFO_RXEMPT_MASK                   0x40u
#define UART_SFIFO_RXEMPT_SHIFT                  6
#define UART_SFIFO_TXEMPT_MASK                   0x80u
#define UART_SFIFO_TXEMPT_SHIFT                  7
/* TWFIFO Bit Fields */
#define UART_TWFIFO_TXWATER_MASK                 0xFFu
#define UART_TWFIFO_TXWATER_SHIFT                0
#define UART_TWFIFO_TXWATER(x)                   (((uint8_t)(((uint8_t)(x))<<UART_TWFIFO_TXWATER_SHIFT))&UART_TWFIFO_TXWATER_MASK)
/* TCFIFO Bit Fields */
#define UART_TCFIFO_TXCOUNT_MASK                 0xFFu
#define UART_TCFIFO_TXCOUNT_SHIFT                0
#define UART_TCFIFO_TXCOUNT(x)                   (((uint8_t)(((uint8_t)(x))<<UART_TCFIFO_TXCOUNT_SHIFT))&UART_TCFIFO_TXCOUNT_MASK)
/* RWFIFO Bit Fields */
#define UART_RWFIFO_RXWATER_MASK                 0xFFu
#define UART_RWFIFO_RXWATER_SHIFT                0
#define UART_RWFIFO_RXWATER(x)                   (((uint8_t)(((uint8_t)(x))<<UART_RWFIFO_RXWATER_SHIFT))&UART_RWFIFO_RXWATER_MASK)
/* RCFIFO Bit Fields */
#define UART_RCFIFO_RXCOUNT_MASK                 0xFFu
#define UART_RCFIFO_RXCOUNT_SHIFT                0
#define UART_RCFIFO_RXCOUNT(x)                   (((uint8_t)(((uint8_t)(x))<<UART_RCFIFO_RXCOUNT_SHIFT))&UART_RCFIFO_RXCOUNT_MASK)

/**
 * @}
 */ /* end of group UART_Register_Masks */


/* UART - Peripheral instance base addresses */
/** Peripheral UART0 base pointer */
#define UART0_BASE_PTR                           ((UART_MemMapPtr)0x40049000u)
/** Peripheral UART1 base pointer */
#define UART1_BASE_PTR                           ((UART_MemMapPtr)0x4004A000u)
/** Peripheral UART2 base pointer */
#define UART2_BASE_PTR                           ((UART_MemMapPtr)0x4004B000u)
/** Peripheral UART3 base pointer */
#define UART3_BASE_PTR                           ((UART_MemMapPtr)0x4004C000u)

/* ----------------------------------------------------------------------------
   -- UART - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup UART_Register_Accessor_Macros UART - Register accessor macros
 * @{
 */


/* UART - Register instance definitions */
/* UART0 */
#define UART0_BDH                                UART_BDH_REG(UART0_BASE_PTR)
#define UART0_BDL                                UART_BDL_REG(UART0_BASE_PTR)
#define UART0_C1                                 UART_C1_REG(UART0_BASE_PTR)
#define UART0_C2                                 UART_C2_REG(UART0_BASE_PTR)
#define UART0_S1                                 UART_S1_REG(UART0_BASE_PTR)
#define UART0_S2                                 UART_S2_REG(UART0_BASE_PTR)
#define UART0_C3                                 UART_C3_REG(UART0_BASE_PTR)
#define UART0_D                                  UART_D_REG(UART0_BASE_PTR)
#define UART0_MA1                                UART_MA1_REG(UART0_BASE_PTR)
#define UART0_MA2                                UART_MA2_REG(UART0_BASE_PTR)
#define UART0_C4                                 UART_C4_REG(UART0_BASE_PTR)
#define UART0_C5                                 UART_C5_REG(UART0_BASE_PTR)
#define UART0_ED                                 UART_ED_REG(UART0_BASE_PTR)
#define UART0_MODEM                              UART_MODEM_REG(UART0_BASE_PTR)
#define UART0_IR                                 UART_IR_REG(UART0_BASE_PTR)
#define UART0_PFIFO                              UART_PFIFO_REG(UART0_BASE_PTR)
#define UART0_CFIFO                              UART_CFIFO_REG(UART0_BASE_PTR)
#define UART0_SFIFO                              UART_SFIFO_REG(UART0_BASE_PTR)
#define UART0_TWFIFO                             UART_TWFIFO_REG(UART0_BASE_PTR)
#define UART0_TCFIFO                             UART_TCFIFO_REG(UART0_BASE_PTR)
#define UART0_RWFIFO                             UART_RWFIFO_REG(UART0_BASE_PTR)
#define UART0_RCFIFO                             UART_RCFIFO_REG(UART0_BASE_PTR)
/* UART1 */
#define UART1_BDH                                UART_BDH_REG(UART1_BASE_PTR)
#define UART1_BDL                                UART_BDL_REG(UART1_BASE_PTR)
#define UART1_C1                                 UART_C1_REG(UART1_BASE_PTR)
#define UART1_C2                                 UART_C2_REG(UART1_BASE_PTR)
#define UART1_S1                                 UART_S1_REG(UART1_BASE_PTR)
#define UART1_S2                                 UART_S2_REG(UART1_BASE_PTR)
#define UART1_C3                                 UART_C3_REG(UART1_BASE_PTR)
#define UART1_D                                  UART_D_REG(UART1_BASE_PTR)
#define UART1_MA1                                UART_MA1_REG(UART1_BASE_PTR)
#define UART1_MA2                                UART_MA2_REG(UART1_BASE_PTR)
#define UART1_C4                                 UART_C4_REG(UART1_BASE_PTR)
#define UART1_C5                                 UART_C5_REG(UART1_BASE_PTR)
#define UART1_MODEM                              UART_MODEM_REG(UART1_BASE_PTR)
#define UART1_IR                                 UART_IR_REG(UART1_BASE_PTR)
/* UART2 */
#define UART2_BDH                                UART_BDH_REG(UART2_BASE_PTR)
#define UART2_BDL                                UART_BDL_REG(UART2_BASE_PTR)
#define UART2_C1                                 UART_C1_REG(UART2_BASE_PTR)
#define UART2_C2                                 UART_C2_REG(UART2_BASE_PTR)
#define UART2_S1                                 UART_S1_REG(UART2_BASE_PTR)
#define UART2_S2                                 UART_S2_REG(UART2_BASE_PTR)
#define UART2_C3                                 UART_C3_REG(UART2_BASE_PTR)
#define UART2_D                                  UART_D_REG(UART2_BASE_PTR)
#define UART2_MA1                                UART_MA1_REG(UART2_BASE_PTR)
#define UART2_MA2                                UART_MA2_REG(UART2_BASE_PTR)
#define UART2_C4                                 UART_C4_REG(UART2_BASE_PTR)
#define UART2_C5                                 UART_C5_REG(UART2_BASE_PTR)
#define UART2_MODEM                              UART_MODEM_REG(UART2_BASE_PTR)
#define UART2_IR                                 UART_IR_REG(UART2_BASE_PTR)
/* UART3 */
#define UART3_BDH                                UART_BDH_REG(UART3_BASE_PTR)
#define UART3_BDL                                UART_BDL_REG(UART3_BASE_PTR)
#define UART3_C1                                 UART_C1_REG(UART3_BASE_PTR)
#define UART3_C2                                 UART_C2_REG(UART3_BASE_PTR)
#define UART3_S1                                 UART_S1_REG(UART3_BASE_PTR)
#define UART3_S2                                 UART_S2_REG(UART3_BASE_PTR)
#define UART3_C3                                 UART_C3_REG(UART3_BASE_PTR)
#define UART3_D                                  UART_D_REG(UART3_BASE_PTR)
#define UART3_MA1                                UART_MA1_REG(UART3_BASE_PTR)
#define UART3_MA2                                UART_MA2_REG(UART3_BASE_PTR)
#define UART3_C4                                 UART_C4_REG(UART3_BASE_PTR)
#define UART3_C5                                 UART_C5_REG(UART3_BASE_PTR)
#define UART3_MODEM                              UART_MODEM_REG(UART3_BASE_PTR)
#define UART3_IR                                 UART_IR_REG(UART3_BASE_PTR)

/**
 * @}
 */ /* end of group UART_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group UART_Peripheral */


/* ----------------------------------------------------------------------------
   -- WDOG
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup WDOG_Peripheral WDOG
 * @{
 */

/** WDOG - Peripheral register structure */
typedef struct WDOG_MemMap {
  uint16_t STCTRLH;                                /**< Watchdog Status and Control Register High, offset: 0x0 */
  uint16_t STCTRLL;                                /**< Watchdog Status and Control Register Low, offset: 0x2 */
  uint16_t TOVALH;                                 /**< Watchdog Time-out Value Register High, offset: 0x4 */
  uint16_t TOVALL;                                 /**< Watchdog Time-out Value Register Low, offset: 0x6 */
  uint16_t WINH;                                   /**< Watchdog Window Register High, offset: 0x8 */
  uint16_t WINL;                                   /**< Watchdog Window Register Low, offset: 0xA */
  uint16_t REFRESH;                                /**< Watchdog Refresh register, offset: 0xC */
  uint16_t UNLOCK;                                 /**< Watchdog Unlock register, offset: 0xE */
  uint16_t TMROUTH;                                /**< Watchdog Timer Output Register High, offset: 0x10 */
  uint16_t TMROUTL;                                /**< Watchdog Timer Output Register Low, offset: 0x12 */
  uint16_t RSTCNT;                                 /**< Watchdog Reset Count register, offset: 0x14 */
  uint16_t PRESC;                                  /**< Watchdog Prescaler register, offset: 0x16 */
} volatile *WDOG_MemMapPtr;

/* ----------------------------------------------------------------------------
   -- WDOG - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup WDOG_Register_Accessor_Macros WDOG - Register accessor macros
 * @{
 */


/* WDOG - Register accessors */
#define WDOG_STCTRLH_REG(base)                   ((base)->STCTRLH)
#define WDOG_STCTRLL_REG(base)                   ((base)->STCTRLL)
#define WDOG_TOVALH_REG(base)                    ((base)->TOVALH)
#define WDOG_TOVALL_REG(base)                    ((base)->TOVALL)
#define WDOG_WINH_REG(base)                      ((base)->WINH)
#define WDOG_WINL_REG(base)                      ((base)->WINL)
#define WDOG_REFRESH_REG(base)                   ((base)->REFRESH)
#define WDOG_UNLOCK_REG(base)                    ((base)->UNLOCK)
#define WDOG_TMROUTH_REG(base)                   ((base)->TMROUTH)
#define WDOG_TMROUTL_REG(base)                   ((base)->TMROUTL)
#define WDOG_RSTCNT_REG(base)                    ((base)->RSTCNT)
#define WDOG_PRESC_REG(base)                     ((base)->PRESC)

/**
 * @}
 */ /* end of group WDOG_Register_Accessor_Macros */


/* ----------------------------------------------------------------------------
   -- WDOG Register Masks
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup WDOG_Register_Masks WDOG Register Masks
 * @{
 */

/* STCTRLH Bit Fields */
#define WDOG_STCTRLH_WDOGEN_MASK                 0x1u
#define WDOG_STCTRLH_WDOGEN_SHIFT                0
#define WDOG_STCTRLH_CLKSRC_MASK                 0x2u
#define WDOG_STCTRLH_CLKSRC_SHIFT                1
#define WDOG_STCTRLH_IRQRSTEN_MASK               0x4u
#define WDOG_STCTRLH_IRQRSTEN_SHIFT              2
#define WDOG_STCTRLH_WINEN_MASK                  0x8u
#define WDOG_STCTRLH_WINEN_SHIFT                 3
#define WDOG_STCTRLH_ALLOWUPDATE_MASK            0x10u
#define WDOG_STCTRLH_ALLOWUPDATE_SHIFT           4
#define WDOG_STCTRLH_DBGEN_MASK                  0x20u
#define WDOG_STCTRLH_DBGEN_SHIFT                 5
#define WDOG_STCTRLH_STOPEN_MASK                 0x40u
#define WDOG_STCTRLH_STOPEN_SHIFT                6
#define WDOG_STCTRLH_WAITEN_MASK                 0x80u
#define WDOG_STCTRLH_WAITEN_SHIFT                7
#define WDOG_STCTRLH_STNDBYEN_MASK               0x100u
#define WDOG_STCTRLH_STNDBYEN_SHIFT              8
#define WDOG_STCTRLH_TESTWDOG_MASK               0x400u
#define WDOG_STCTRLH_TESTWDOG_SHIFT              10
#define WDOG_STCTRLH_TESTSEL_MASK                0x800u
#define WDOG_STCTRLH_TESTSEL_SHIFT               11
#define WDOG_STCTRLH_BYTESEL_MASK                0x3000u
#define WDOG_STCTRLH_BYTESEL_SHIFT               12
#define WDOG_STCTRLH_BYTESEL(x)                  (((uint16_t)(((uint16_t)(x))<<WDOG_STCTRLH_BYTESEL_SHIFT))&WDOG_STCTRLH_BYTESEL_MASK)
#define WDOG_STCTRLH_DISTESTWDOG_MASK            0x4000u
#define WDOG_STCTRLH_DISTESTWDOG_SHIFT           14
/* STCTRLL Bit Fields */
#define WDOG_STCTRLL_INTFLG_MASK                 0x8000u
#define WDOG_STCTRLL_INTFLG_SHIFT                15
/* TOVALH Bit Fields */
#define WDOG_TOVALH_TOVALHIGH_MASK               0xFFFFu
#define WDOG_TOVALH_TOVALHIGH_SHIFT              0
#define WDOG_TOVALH_TOVALHIGH(x)                 (((uint16_t)(((uint16_t)(x))<<WDOG_TOVALH_TOVALHIGH_SHIFT))&WDOG_TOVALH_TOVALHIGH_MASK)
/* TOVALL Bit Fields */
#define WDOG_TOVALL_TOVALLOW_MASK                0xFFFFu
#define WDOG_TOVALL_TOVALLOW_SHIFT               0
#define WDOG_TOVALL_TOVALLOW(x)                  (((uint16_t)(((uint16_t)(x))<<WDOG_TOVALL_TOVALLOW_SHIFT))&WDOG_TOVALL_TOVALLOW_MASK)
/* WINH Bit Fields */
#define WDOG_WINH_WINHIGH_MASK                   0xFFFFu
#define WDOG_WINH_WINHIGH_SHIFT                  0
#define WDOG_WINH_WINHIGH(x)                     (((uint16_t)(((uint16_t)(x))<<WDOG_WINH_WINHIGH_SHIFT))&WDOG_WINH_WINHIGH_MASK)
/* WINL Bit Fields */
#define WDOG_WINL_WINLOW_MASK                    0xFFFFu
#define WDOG_WINL_WINLOW_SHIFT                   0
#define WDOG_WINL_WINLOW(x)                      (((uint16_t)(((uint16_t)(x))<<WDOG_WINL_WINLOW_SHIFT))&WDOG_WINL_WINLOW_MASK)
/* REFRESH Bit Fields */
#define WDOG_REFRESH_WDOGREFRESH_MASK            0xFFFFu
#define WDOG_REFRESH_WDOGREFRESH_SHIFT           0
#define WDOG_REFRESH_WDOGREFRESH(x)              (((uint16_t)(((uint16_t)(x))<<WDOG_REFRESH_WDOGREFRESH_SHIFT))&WDOG_REFRESH_WDOGREFRESH_MASK)
/* UNLOCK Bit Fields */
#define WDOG_UNLOCK_WDOGUNLOCK_MASK              0xFFFFu
#define WDOG_UNLOCK_WDOGUNLOCK_SHIFT             0
#define WDOG_UNLOCK_WDOGUNLOCK(x)                (((uint16_t)(((uint16_t)(x))<<WDOG_UNLOCK_WDOGUNLOCK_SHIFT))&WDOG_UNLOCK_WDOGUNLOCK_MASK)
/* TMROUTH Bit Fields */
#define WDOG_TMROUTH_TIMEROUTHIGH_MASK           0xFFFFu
#define WDOG_TMROUTH_TIMEROUTHIGH_SHIFT          0
#define WDOG_TMROUTH_TIMEROUTHIGH(x)             (((uint16_t)(((uint16_t)(x))<<WDOG_TMROUTH_TIMEROUTHIGH_SHIFT))&WDOG_TMROUTH_TIMEROUTHIGH_MASK)
/* TMROUTL Bit Fields */
#define WDOG_TMROUTL_TIMEROUTLOW_MASK            0xFFFFu
#define WDOG_TMROUTL_TIMEROUTLOW_SHIFT           0
#define WDOG_TMROUTL_TIMEROUTLOW(x)              (((uint16_t)(((uint16_t)(x))<<WDOG_TMROUTL_TIMEROUTLOW_SHIFT))&WDOG_TMROUTL_TIMEROUTLOW_MASK)
/* RSTCNT Bit Fields */
#define WDOG_RSTCNT_RSTCNT_MASK                  0xFFFFu
#define WDOG_RSTCNT_RSTCNT_SHIFT                 0
#define WDOG_RSTCNT_RSTCNT(x)                    (((uint16_t)(((uint16_t)(x))<<WDOG_RSTCNT_RSTCNT_SHIFT))&WDOG_RSTCNT_RSTCNT_MASK)
/* PRESC Bit Fields */
#define WDOG_PRESC_PRESCVAL_MASK                 0x700u
#define WDOG_PRESC_PRESCVAL_SHIFT                8
#define WDOG_PRESC_PRESCVAL(x)                   (((uint16_t)(((uint16_t)(x))<<WDOG_PRESC_PRESCVAL_SHIFT))&WDOG_PRESC_PRESCVAL_MASK)

/**
 * @}
 */ /* end of group WDOG_Register_Masks */


/* WDOG - Peripheral instance base addresses */
/** Peripheral WDOG base pointer */
#define WDOG_BASE_PTR                            ((WDOG_MemMapPtr)0x4003A000u)

/* ----------------------------------------------------------------------------
   -- WDOG - Register accessor macros
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup WDOG_Register_Accessor_Macros WDOG - Register accessor macros
 * @{
 */


/* WDOG - Register instance definitions */
/* WDOG */
#define WDOG_STCTRLH                             WDOG_STCTRLH_REG(WDOG_BASE_PTR)
#define WDOG_STCTRLL                             WDOG_STCTRLL_REG(WDOG_BASE_PTR)
#define WDOG_TOVALH                              WDOG_TOVALH_REG(WDOG_BASE_PTR)
#define WDOG_TOVALL                              WDOG_TOVALL_REG(WDOG_BASE_PTR)
#define WDOG_WINH                                WDOG_WINH_REG(WDOG_BASE_PTR)
#define WDOG_WINL                                WDOG_WINL_REG(WDOG_BASE_PTR)
#define WDOG_REFRESH                             WDOG_REFRESH_REG(WDOG_BASE_PTR)
#define WDOG_UNLOCK                              WDOG_UNLOCK_REG(WDOG_BASE_PTR)
#define WDOG_TMROUTH                             WDOG_TMROUTH_REG(WDOG_BASE_PTR)
#define WDOG_TMROUTL                             WDOG_TMROUTL_REG(WDOG_BASE_PTR)
#define WDOG_RSTCNT                              WDOG_RSTCNT_REG(WDOG_BASE_PTR)
#define WDOG_PRESC                               WDOG_PRESC_REG(WDOG_BASE_PTR)

/**
 * @}
 */ /* end of group WDOG_Register_Accessor_Macros */


/**
 * @}
 */ /* end of group WDOG_Peripheral */


/*
** End of section using anonymous unions
*/

#if defined(__ARMCC_VERSION)
  #pragma pop
#elif defined(__CWCC__)
  #pragma pop
#elif defined(__GNUC__)
  /* leave anonymous unions enabled */
#elif defined(__IAR_SYSTEMS_ICC__)
  #pragma language=default
#else
  #error Not supported compiler type
#endif

/**
 * @}
 */ /* end of group Peripheral_defines */


/* ----------------------------------------------------------------------------
   -- Backward Compatibility
   ---------------------------------------------------------------------------- */

/**
 * @addtogroup Backward_Compatibility_Symbols Backward Compatibility
 * @{
 */

/* No backward compatibility issues. */

/**
 * @}
 */ /* end of group Backward_Compatibility_Symbols */


#else /* #if !defined(MCU_MKE14D7) */
  /* There is already included the same memory map. Check if it is compatible (has the same major version) */
  #if (MCU_MEM_MAP_VERSION != 0x0100u)
    #if (!defined(MCU_MEM_MAP_SUPPRESS_VERSION_WARNING))
      #warning There are included two not compatible versions of memory maps. Please check possible differences.
    #endif /* (!defined(MCU_MEM_MAP_SUPPRESS_VERSION_WARNING)) */
  #endif /* (MCU_MEM_MAP_VERSION != 0x0100u) */
#endif  /* #if !defined(MCU_MKE14D7) */

/* MKE14D7.h, eof. */
