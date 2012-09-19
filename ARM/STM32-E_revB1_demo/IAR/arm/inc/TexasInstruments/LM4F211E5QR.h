
/****************************************************************************************************//**
 * @file     LM4F211E5QR.h
 *
 * @status   EXPERIMENTAL
 *
 * @brief    CMSIS Cortex-M4 Core Peripheral Access Layer Header File for
 *           default LM4F211E5QR Device Series
 *
 * @version  V7838
 * @date     25. August 2011
 *
 * @note     Generated with SVDConv V2.6 Build 4t  on Thursday, 25.08.2011 17:19:19
 *
 *           from CMSIS SVD File 'lm4f211e5qr.svd.xml' Version 7838,
 *           created on Thursday, 25.08.2011 22:19:18, last modified on Thursday, 25.08.2011 22:19:18
 *
 *******************************************************************************************************/



/** @addtogroup (null)
  * @{
  */

/** @addtogroup LM4F211E5QR
  * @{
  */

#ifndef __LM4F211E5QR_H__
#define __LM4F211E5QR_H__

#ifdef __cplusplus
extern "C" {
#endif 



/********************************************
** Start of section using anonymous unions **
*********************************************/

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


 /* Interrupt Number Definition */

typedef enum {
// -------------------------  Cortex-M4 Processor Exceptions Numbers  -----------------------------
  Reset_IRQn                        = -15,  /*!<   1  Reset Vector, invoked on Power up and warm reset */
  NonMaskableInt_IRQn               = -14,  /*!<   2  Non maskable Interrupt, cannot be stopped or preempted */
  HardFault_IRQn                    = -13,  /*!<   3  Hard Fault, all classes of Fault */
  MemoryManagement_IRQn             = -12,  /*!<   4  Memory Management, MPU mismatch, including Access Violation and No Match */
  BusFault_IRQn                     = -11,  /*!<   5  Bus Fault, Pre-Fetch-, Memory Access Fault, other address/memory related Fault */
  UsageFault_IRQn                   = -10,  /*!<   6  Usage Fault, i.e. Undef Instruction, Illegal State Transition */
  SVCall_IRQn                       = -5,   /*!<  11  System Service Call via SVC instruction */
  DebugMonitor_IRQn                 = -4,   /*!<  12  Debug Monitor                    */
  PendSV_IRQn                       = -2,   /*!<  14  Pendable request for system service */
  SysTick_IRQn                      = -1,   /*!<  15  System Tick Timer                */
// -------------------------  LM4F211E5QR Specific Interrupt Numbers  -----------------------------
} IRQn_Type;


/** @addtogroup Configuration_of_CMSIS
  * @{
  */

/* Processor and Core Peripheral Section */ /* Configuration of the Cortex-M4 Processor and Core Peripherals */

#define __CM4_REV              0x0000       /*!< Cortex-M3 Core Revision               */
#define __MPU_PRESENT             0         /*!< MPU present or not                    */
#define __NVIC_PRIO_BITS          0         /*!< Number of Bits used for Priority Levels */
#define __Vendor_SysTickConfig    0         /*!< Set to 1 if different SysTick Config is used */

    #define __FPU_PRESENT         &i        /*!< FPU present or not                    */
/** @} */ /* End of group Configuration_of_CMSIS */

#include <core_cm4.h>                       /*!< Cortex-M4 processor and core peripherals */
#include "system_LM4F211E5QR.h"             /*!< LM4F211E5QR System                    */

/** @addtogroup Device_Peripheral_Registers
  * @{
  */


// ------------------------------------------------------------------------------------------------
// -----                                       WATCHDOG0                                      -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for WATCHDOG0 peripheral (WATCHDOG0)
  */

typedef struct {                            /*!< WATCHDOG0 Structure                   */
  __IO uint32_t LOAD;                       /*!< Watchdog Load                         */
  __IO uint32_t VALUE;                      /*!< Watchdog Value                        */
  __IO uint32_t CTL;                        /*!< Watchdog Control                      */
  __O  uint32_t ICR;                        /*!< Watchdog Interrupt Clear              */
  __IO uint32_t RIS;                        /*!< Watchdog Raw Interrupt Status         */
  __IO uint32_t MIS;                        /*!< Watchdog Masked Interrupt Status      */
  __I  uint32_t RESERVED0[256];
  __IO uint32_t TEST;                       /*!< Watchdog Test                         */
  __I  uint32_t RESERVED1[505];
  __IO uint32_t LOCK;                       /*!< Watchdog Lock                         */
} WATCHDOG0_Type;


// ------------------------------------------------------------------------------------------------
// -----                                       WATCHDOG1                                      -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for WATCHDOG0 peripheral (WATCHDOG1)
  */

typedef struct {                            /*!< WATCHDOG1 Structure                   */
  __IO uint32_t LOAD;                       /*!< Watchdog Load                         */
  __IO uint32_t VALUE;                      /*!< Watchdog Value                        */
  __IO uint32_t CTL;                        /*!< Watchdog Control                      */
  __O  uint32_t ICR;                        /*!< Watchdog Interrupt Clear              */
  __IO uint32_t RIS;                        /*!< Watchdog Raw Interrupt Status         */
  __IO uint32_t MIS;                        /*!< Watchdog Masked Interrupt Status      */
  __I  uint32_t RESERVED0[256];
  __IO uint32_t TEST;                       /*!< Watchdog Test                         */
  __I  uint32_t RESERVED1[505];
  __IO uint32_t LOCK;                       /*!< Watchdog Lock                         */
} WATCHDOG1_Type;


// ------------------------------------------------------------------------------------------------
// -----                                      GPIO_PORTA                                      -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for GPIO_PORTA peripheral (GPIO_PORTA)
  */

typedef struct {                            /*!< GPIO_PORTA Structure                  */
  __I  uint32_t RESERVED0[255];
  __IO uint32_t DATA;                       /*!< GPIO Data                             */
  __IO uint32_t DIR;                        /*!< GPIO Direction                        */
  __IO uint32_t IS;                         /*!< GPIO Interrupt Sense                  */
  __IO uint32_t IBE;                        /*!< GPIO Interrupt Both Edges             */
  __IO uint32_t IEV;                        /*!< GPIO Interrupt Event                  */
  __IO uint32_t IM;                         /*!< GPIO Interrupt Mask                   */
  __IO uint32_t RIS;                        /*!< GPIO Raw Interrupt Status             */
  __IO uint32_t MIS;                        /*!< GPIO Masked Interrupt Status          */
  __O  uint32_t ICR;                        /*!< GPIO Interrupt Clear                  */
  __IO uint32_t AFSEL;                      /*!< GPIO Alternate Function Select        */
  __I  uint32_t RESERVED1[55];
  __IO uint32_t DR2R;                       /*!< GPIO 2-mA Drive Select                */
  __IO uint32_t DR4R;                       /*!< GPIO 4-mA Drive Select                */
  __IO uint32_t DR8R;                       /*!< GPIO 8-mA Drive Select                */
  __IO uint32_t ODR;                        /*!< GPIO Open Drain Select                */
  __IO uint32_t PUR;                        /*!< GPIO Pull-Up Select                   */
  __IO uint32_t PDR;                        /*!< GPIO Pull-Down Select                 */
  __IO uint32_t SLR;                        /*!< GPIO Slew Rate Control Select         */
  __IO uint32_t DEN;                        /*!< GPIO Digital Enable                   */
  __IO uint32_t LOCK;                       /*!< GPIO Lock                             */
  __I  uint32_t CR;                         /*!< GPIO Commit                           */
  __IO uint32_t AMSEL;                      /*!< GPIO Analog Mode Select               */
  __IO uint32_t PCTL;                       /*!< GPIO Port Control                     */
  __IO uint32_t ADCCTL;                     /*!< GPIO ADC Control                      */
  __IO uint32_t DMACTL;                     /*!< GPIO DMA Control                      */
  __IO uint32_t SI;                         /*!< GPIO Select Interrupt                 */
} GPIO_PORTA_Type;


// ------------------------------------------------------------------------------------------------
// -----                                      GPIO_PORTB                                      -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for GPIO_PORTA peripheral (GPIO_PORTB)
  */

typedef struct {                            /*!< GPIO_PORTB Structure                  */
  __I  uint32_t RESERVED0[255];
  __IO uint32_t DATA;                       /*!< GPIO Data                             */
  __IO uint32_t DIR;                        /*!< GPIO Direction                        */
  __IO uint32_t IS;                         /*!< GPIO Interrupt Sense                  */
  __IO uint32_t IBE;                        /*!< GPIO Interrupt Both Edges             */
  __IO uint32_t IEV;                        /*!< GPIO Interrupt Event                  */
  __IO uint32_t IM;                         /*!< GPIO Interrupt Mask                   */
  __IO uint32_t RIS;                        /*!< GPIO Raw Interrupt Status             */
  __IO uint32_t MIS;                        /*!< GPIO Masked Interrupt Status          */
  __O  uint32_t ICR;                        /*!< GPIO Interrupt Clear                  */
  __IO uint32_t AFSEL;                      /*!< GPIO Alternate Function Select        */
  __I  uint32_t RESERVED1[55];
  __IO uint32_t DR2R;                       /*!< GPIO 2-mA Drive Select                */
  __IO uint32_t DR4R;                       /*!< GPIO 4-mA Drive Select                */
  __IO uint32_t DR8R;                       /*!< GPIO 8-mA Drive Select                */
  __IO uint32_t ODR;                        /*!< GPIO Open Drain Select                */
  __IO uint32_t PUR;                        /*!< GPIO Pull-Up Select                   */
  __IO uint32_t PDR;                        /*!< GPIO Pull-Down Select                 */
  __IO uint32_t SLR;                        /*!< GPIO Slew Rate Control Select         */
  __IO uint32_t DEN;                        /*!< GPIO Digital Enable                   */
  __IO uint32_t LOCK;                       /*!< GPIO Lock                             */
  __I  uint32_t CR;                         /*!< GPIO Commit                           */
  __IO uint32_t AMSEL;                      /*!< GPIO Analog Mode Select               */
  __IO uint32_t PCTL;                       /*!< GPIO Port Control                     */
  __IO uint32_t ADCCTL;                     /*!< GPIO ADC Control                      */
  __IO uint32_t DMACTL;                     /*!< GPIO DMA Control                      */
  __IO uint32_t SI;                         /*!< GPIO Select Interrupt                 */
} GPIO_PORTB_Type;


// ------------------------------------------------------------------------------------------------
// -----                                      GPIO_PORTC                                      -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for GPIO_PORTA peripheral (GPIO_PORTC)
  */

typedef struct {                            /*!< GPIO_PORTC Structure                  */
  __I  uint32_t RESERVED0[255];
  __IO uint32_t DATA;                       /*!< GPIO Data                             */
  __IO uint32_t DIR;                        /*!< GPIO Direction                        */
  __IO uint32_t IS;                         /*!< GPIO Interrupt Sense                  */
  __IO uint32_t IBE;                        /*!< GPIO Interrupt Both Edges             */
  __IO uint32_t IEV;                        /*!< GPIO Interrupt Event                  */
  __IO uint32_t IM;                         /*!< GPIO Interrupt Mask                   */
  __IO uint32_t RIS;                        /*!< GPIO Raw Interrupt Status             */
  __IO uint32_t MIS;                        /*!< GPIO Masked Interrupt Status          */
  __O  uint32_t ICR;                        /*!< GPIO Interrupt Clear                  */
  __IO uint32_t AFSEL;                      /*!< GPIO Alternate Function Select        */
  __I  uint32_t RESERVED1[55];
  __IO uint32_t DR2R;                       /*!< GPIO 2-mA Drive Select                */
  __IO uint32_t DR4R;                       /*!< GPIO 4-mA Drive Select                */
  __IO uint32_t DR8R;                       /*!< GPIO 8-mA Drive Select                */
  __IO uint32_t ODR;                        /*!< GPIO Open Drain Select                */
  __IO uint32_t PUR;                        /*!< GPIO Pull-Up Select                   */
  __IO uint32_t PDR;                        /*!< GPIO Pull-Down Select                 */
  __IO uint32_t SLR;                        /*!< GPIO Slew Rate Control Select         */
  __IO uint32_t DEN;                        /*!< GPIO Digital Enable                   */
  __IO uint32_t LOCK;                       /*!< GPIO Lock                             */
  __I  uint32_t CR;                         /*!< GPIO Commit                           */
  __IO uint32_t AMSEL;                      /*!< GPIO Analog Mode Select               */
  __IO uint32_t PCTL;                       /*!< GPIO Port Control                     */
  __IO uint32_t ADCCTL;                     /*!< GPIO ADC Control                      */
  __IO uint32_t DMACTL;                     /*!< GPIO DMA Control                      */
  __IO uint32_t SI;                         /*!< GPIO Select Interrupt                 */
} GPIO_PORTC_Type;


// ------------------------------------------------------------------------------------------------
// -----                                      GPIO_PORTD                                      -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for GPIO_PORTA peripheral (GPIO_PORTD)
  */

typedef struct {                            /*!< GPIO_PORTD Structure                  */
  __I  uint32_t RESERVED0[255];
  __IO uint32_t DATA;                       /*!< GPIO Data                             */
  __IO uint32_t DIR;                        /*!< GPIO Direction                        */
  __IO uint32_t IS;                         /*!< GPIO Interrupt Sense                  */
  __IO uint32_t IBE;                        /*!< GPIO Interrupt Both Edges             */
  __IO uint32_t IEV;                        /*!< GPIO Interrupt Event                  */
  __IO uint32_t IM;                         /*!< GPIO Interrupt Mask                   */
  __IO uint32_t RIS;                        /*!< GPIO Raw Interrupt Status             */
  __IO uint32_t MIS;                        /*!< GPIO Masked Interrupt Status          */
  __O  uint32_t ICR;                        /*!< GPIO Interrupt Clear                  */
  __IO uint32_t AFSEL;                      /*!< GPIO Alternate Function Select        */
  __I  uint32_t RESERVED1[55];
  __IO uint32_t DR2R;                       /*!< GPIO 2-mA Drive Select                */
  __IO uint32_t DR4R;                       /*!< GPIO 4-mA Drive Select                */
  __IO uint32_t DR8R;                       /*!< GPIO 8-mA Drive Select                */
  __IO uint32_t ODR;                        /*!< GPIO Open Drain Select                */
  __IO uint32_t PUR;                        /*!< GPIO Pull-Up Select                   */
  __IO uint32_t PDR;                        /*!< GPIO Pull-Down Select                 */
  __IO uint32_t SLR;                        /*!< GPIO Slew Rate Control Select         */
  __IO uint32_t DEN;                        /*!< GPIO Digital Enable                   */
  __IO uint32_t LOCK;                       /*!< GPIO Lock                             */
  __I  uint32_t CR;                         /*!< GPIO Commit                           */
  __IO uint32_t AMSEL;                      /*!< GPIO Analog Mode Select               */
  __IO uint32_t PCTL;                       /*!< GPIO Port Control                     */
  __IO uint32_t ADCCTL;                     /*!< GPIO ADC Control                      */
  __IO uint32_t DMACTL;                     /*!< GPIO DMA Control                      */
  __IO uint32_t SI;                         /*!< GPIO Select Interrupt                 */
} GPIO_PORTD_Type;


// ------------------------------------------------------------------------------------------------
// -----                                         SSI0                                         -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for SSI0 peripheral (SSI0)
  */

typedef struct {                            /*!< SSI0 Structure                        */
  __IO uint32_t CR0;                        /*!< SSI Control 0                         */
  __IO uint32_t CR1;                        /*!< SSI Control 1                         */
  __IO uint32_t DR;                         /*!< SSI Data                              */
  __IO uint32_t SR;                         /*!< SSI Status                            */
  __IO uint32_t CPSR;                       /*!< SSI Clock Prescale                    */
  __IO uint32_t IM;                         /*!< SSI Interrupt Mask                    */
  __IO uint32_t RIS;                        /*!< SSI Raw Interrupt Status              */
  __IO uint32_t MIS;                        /*!< SSI Masked Interrupt Status           */
  __O  uint32_t ICR;                        /*!< SSI Interrupt Clear                   */
  __IO uint32_t DMACTL;                     /*!< SSI DMA Control                       */
  __I  uint32_t RESERVED0[1000];
  __IO uint32_t CC;                         /*!< SSI Clock Configuration               */
} SSI0_Type;


// ------------------------------------------------------------------------------------------------
// -----                                         SSI1                                         -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for SSI0 peripheral (SSI1)
  */

typedef struct {                            /*!< SSI1 Structure                        */
  __IO uint32_t CR0;                        /*!< SSI Control 0                         */
  __IO uint32_t CR1;                        /*!< SSI Control 1                         */
  __IO uint32_t DR;                         /*!< SSI Data                              */
  __IO uint32_t SR;                         /*!< SSI Status                            */
  __IO uint32_t CPSR;                       /*!< SSI Clock Prescale                    */
  __IO uint32_t IM;                         /*!< SSI Interrupt Mask                    */
  __IO uint32_t RIS;                        /*!< SSI Raw Interrupt Status              */
  __IO uint32_t MIS;                        /*!< SSI Masked Interrupt Status           */
  __O  uint32_t ICR;                        /*!< SSI Interrupt Clear                   */
  __IO uint32_t DMACTL;                     /*!< SSI DMA Control                       */
  __I  uint32_t RESERVED0[1000];
  __IO uint32_t CC;                         /*!< SSI Clock Configuration               */
} SSI1_Type;


// ------------------------------------------------------------------------------------------------
// -----                                         SSI2                                         -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for SSI0 peripheral (SSI2)
  */

typedef struct {                            /*!< SSI2 Structure                        */
  __IO uint32_t CR0;                        /*!< SSI Control 0                         */
  __IO uint32_t CR1;                        /*!< SSI Control 1                         */
  __IO uint32_t DR;                         /*!< SSI Data                              */
  __IO uint32_t SR;                         /*!< SSI Status                            */
  __IO uint32_t CPSR;                       /*!< SSI Clock Prescale                    */
  __IO uint32_t IM;                         /*!< SSI Interrupt Mask                    */
  __IO uint32_t RIS;                        /*!< SSI Raw Interrupt Status              */
  __IO uint32_t MIS;                        /*!< SSI Masked Interrupt Status           */
  __O  uint32_t ICR;                        /*!< SSI Interrupt Clear                   */
  __IO uint32_t DMACTL;                     /*!< SSI DMA Control                       */
  __I  uint32_t RESERVED0[1000];
  __IO uint32_t CC;                         /*!< SSI Clock Configuration               */
} SSI2_Type;


// ------------------------------------------------------------------------------------------------
// -----                                         SSI3                                         -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for SSI0 peripheral (SSI3)
  */

typedef struct {                            /*!< SSI3 Structure                        */
  __IO uint32_t CR0;                        /*!< SSI Control 0                         */
  __IO uint32_t CR1;                        /*!< SSI Control 1                         */
  __IO uint32_t DR;                         /*!< SSI Data                              */
  __IO uint32_t SR;                         /*!< SSI Status                            */
  __IO uint32_t CPSR;                       /*!< SSI Clock Prescale                    */
  __IO uint32_t IM;                         /*!< SSI Interrupt Mask                    */
  __IO uint32_t RIS;                        /*!< SSI Raw Interrupt Status              */
  __IO uint32_t MIS;                        /*!< SSI Masked Interrupt Status           */
  __O  uint32_t ICR;                        /*!< SSI Interrupt Clear                   */
  __IO uint32_t DMACTL;                     /*!< SSI DMA Control                       */
  __I  uint32_t RESERVED0[1000];
  __IO uint32_t CC;                         /*!< SSI Clock Configuration               */
} SSI3_Type;


// ------------------------------------------------------------------------------------------------
// -----                                         UART0                                        -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for UART0 peripheral (UART0)
  */

typedef struct {                            /*!< UART0 Structure                       */
  __IO uint32_t DR;                         /*!< UART Data                             */
  
  union {
    __IO uint32_t ECR;                      /*!< UART Receive Status/Error Clear       */
    __IO uint32_t RSR;                      /*!< UART Receive Status/Error Clear       */
  } ;
  __I  uint32_t RESERVED0[4];
  __IO uint32_t FR;                         /*!< UART Flag                             */
  __I  uint32_t RESERVED1;
  __IO uint32_t ILPR;                       /*!< UART IrDA Low-Power Register          */
  __IO uint32_t IBRD;                       /*!< UART Integer Baud-Rate Divisor        */
  __IO uint32_t FBRD;                       /*!< UART Fractional Baud-Rate Divisor     */
  __IO uint32_t LCRH;                       /*!< UART Line Control                     */
  __IO uint32_t CTL;                        /*!< UART Control                          */
  __IO uint32_t IFLS;                       /*!< UART Interrupt FIFO Level Select      */
  __IO uint32_t IM;                         /*!< UART Interrupt Mask                   */
  __IO uint32_t RIS;                        /*!< UART Raw Interrupt Status             */
  __IO uint32_t MIS;                        /*!< UART Masked Interrupt Status          */
  __O  uint32_t ICR;                        /*!< UART Interrupt Clear                  */
  __IO uint32_t DMACTL;                     /*!< UART DMA Control                      */
  __I  uint32_t RESERVED2[17];
  __IO uint32_t LCTL;                       /*!< UART LIN Control                      */
  __IO uint32_t LSS;                        /*!< UART LIN Snap Shot                    */
  __IO uint32_t LTIM;                       /*!< UART LIN Timer                        */
  __I  uint32_t RESERVED3[2];
  __IO uint32_t _9BITADDR;                  /*!< UART 9-Bit Self Address               */
  __IO uint32_t _9BITAMASK;                 /*!< UART 9-Bit Self Address Mask          */
  __I  uint32_t RESERVED4[965];
  __IO uint32_t PP;                         /*!< UART Peripheral Properties            */
  __I  uint32_t RESERVED5;
  __IO uint32_t CC;                         /*!< UART Clock Configuration              */
} UART0_Type;


// ------------------------------------------------------------------------------------------------
// -----                                         UART1                                        -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for UART0 peripheral (UART1)
  */

typedef struct {                            /*!< UART1 Structure                       */
  __IO uint32_t DR;                         /*!< UART Data                             */
  
  union {
    __IO uint32_t ECR;                      /*!< UART Receive Status/Error Clear       */
    __IO uint32_t RSR;                      /*!< UART Receive Status/Error Clear       */
  } ;
  __I  uint32_t RESERVED0[4];
  __IO uint32_t FR;                         /*!< UART Flag                             */
  __I  uint32_t RESERVED1;
  __IO uint32_t ILPR;                       /*!< UART IrDA Low-Power Register          */
  __IO uint32_t IBRD;                       /*!< UART Integer Baud-Rate Divisor        */
  __IO uint32_t FBRD;                       /*!< UART Fractional Baud-Rate Divisor     */
  __IO uint32_t LCRH;                       /*!< UART Line Control                     */
  __IO uint32_t CTL;                        /*!< UART Control                          */
  __IO uint32_t IFLS;                       /*!< UART Interrupt FIFO Level Select      */
  __IO uint32_t IM;                         /*!< UART Interrupt Mask                   */
  __IO uint32_t RIS;                        /*!< UART Raw Interrupt Status             */
  __IO uint32_t MIS;                        /*!< UART Masked Interrupt Status          */
  __O  uint32_t ICR;                        /*!< UART Interrupt Clear                  */
  __IO uint32_t DMACTL;                     /*!< UART DMA Control                      */
  __I  uint32_t RESERVED2[17];
  __IO uint32_t LCTL;                       /*!< UART LIN Control                      */
  __IO uint32_t LSS;                        /*!< UART LIN Snap Shot                    */
  __IO uint32_t LTIM;                       /*!< UART LIN Timer                        */
  __I  uint32_t RESERVED3[2];
  __IO uint32_t _9BITADDR;                  /*!< UART 9-Bit Self Address               */
  __IO uint32_t _9BITAMASK;                 /*!< UART 9-Bit Self Address Mask          */
  __I  uint32_t RESERVED4[965];
  __IO uint32_t PP;                         /*!< UART Peripheral Properties            */
  __I  uint32_t RESERVED5;
  __IO uint32_t CC;                         /*!< UART Clock Configuration              */
} UART1_Type;


// ------------------------------------------------------------------------------------------------
// -----                                         UART2                                        -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for UART0 peripheral (UART2)
  */

typedef struct {                            /*!< UART2 Structure                       */
  __IO uint32_t DR;                         /*!< UART Data                             */
  
  union {
    __IO uint32_t ECR;                      /*!< UART Receive Status/Error Clear       */
    __IO uint32_t RSR;                      /*!< UART Receive Status/Error Clear       */
  } ;
  __I  uint32_t RESERVED0[4];
  __IO uint32_t FR;                         /*!< UART Flag                             */
  __I  uint32_t RESERVED1;
  __IO uint32_t ILPR;                       /*!< UART IrDA Low-Power Register          */
  __IO uint32_t IBRD;                       /*!< UART Integer Baud-Rate Divisor        */
  __IO uint32_t FBRD;                       /*!< UART Fractional Baud-Rate Divisor     */
  __IO uint32_t LCRH;                       /*!< UART Line Control                     */
  __IO uint32_t CTL;                        /*!< UART Control                          */
  __IO uint32_t IFLS;                       /*!< UART Interrupt FIFO Level Select      */
  __IO uint32_t IM;                         /*!< UART Interrupt Mask                   */
  __IO uint32_t RIS;                        /*!< UART Raw Interrupt Status             */
  __IO uint32_t MIS;                        /*!< UART Masked Interrupt Status          */
  __O  uint32_t ICR;                        /*!< UART Interrupt Clear                  */
  __IO uint32_t DMACTL;                     /*!< UART DMA Control                      */
  __I  uint32_t RESERVED2[17];
  __IO uint32_t LCTL;                       /*!< UART LIN Control                      */
  __IO uint32_t LSS;                        /*!< UART LIN Snap Shot                    */
  __IO uint32_t LTIM;                       /*!< UART LIN Timer                        */
  __I  uint32_t RESERVED3[2];
  __IO uint32_t _9BITADDR;                  /*!< UART 9-Bit Self Address               */
  __IO uint32_t _9BITAMASK;                 /*!< UART 9-Bit Self Address Mask          */
  __I  uint32_t RESERVED4[965];
  __IO uint32_t PP;                         /*!< UART Peripheral Properties            */
  __I  uint32_t RESERVED5;
  __IO uint32_t CC;                         /*!< UART Clock Configuration              */
} UART2_Type;


// ------------------------------------------------------------------------------------------------
// -----                                         UART3                                        -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for UART0 peripheral (UART3)
  */

typedef struct {                            /*!< UART3 Structure                       */
  __IO uint32_t DR;                         /*!< UART Data                             */
  
  union {
    __IO uint32_t ECR;                      /*!< UART Receive Status/Error Clear       */
    __IO uint32_t RSR;                      /*!< UART Receive Status/Error Clear       */
  } ;
  __I  uint32_t RESERVED0[4];
  __IO uint32_t FR;                         /*!< UART Flag                             */
  __I  uint32_t RESERVED1;
  __IO uint32_t ILPR;                       /*!< UART IrDA Low-Power Register          */
  __IO uint32_t IBRD;                       /*!< UART Integer Baud-Rate Divisor        */
  __IO uint32_t FBRD;                       /*!< UART Fractional Baud-Rate Divisor     */
  __IO uint32_t LCRH;                       /*!< UART Line Control                     */
  __IO uint32_t CTL;                        /*!< UART Control                          */
  __IO uint32_t IFLS;                       /*!< UART Interrupt FIFO Level Select      */
  __IO uint32_t IM;                         /*!< UART Interrupt Mask                   */
  __IO uint32_t RIS;                        /*!< UART Raw Interrupt Status             */
  __IO uint32_t MIS;                        /*!< UART Masked Interrupt Status          */
  __O  uint32_t ICR;                        /*!< UART Interrupt Clear                  */
  __IO uint32_t DMACTL;                     /*!< UART DMA Control                      */
  __I  uint32_t RESERVED2[17];
  __IO uint32_t LCTL;                       /*!< UART LIN Control                      */
  __IO uint32_t LSS;                        /*!< UART LIN Snap Shot                    */
  __IO uint32_t LTIM;                       /*!< UART LIN Timer                        */
  __I  uint32_t RESERVED3[2];
  __IO uint32_t _9BITADDR;                  /*!< UART 9-Bit Self Address               */
  __IO uint32_t _9BITAMASK;                 /*!< UART 9-Bit Self Address Mask          */
  __I  uint32_t RESERVED4[965];
  __IO uint32_t PP;                         /*!< UART Peripheral Properties            */
  __I  uint32_t RESERVED5;
  __IO uint32_t CC;                         /*!< UART Clock Configuration              */
} UART3_Type;


// ------------------------------------------------------------------------------------------------
// -----                                         UART4                                        -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for UART0 peripheral (UART4)
  */

typedef struct {                            /*!< UART4 Structure                       */
  __IO uint32_t DR;                         /*!< UART Data                             */
  
  union {
    __IO uint32_t ECR;                      /*!< UART Receive Status/Error Clear       */
    __IO uint32_t RSR;                      /*!< UART Receive Status/Error Clear       */
  } ;
  __I  uint32_t RESERVED0[4];
  __IO uint32_t FR;                         /*!< UART Flag                             */
  __I  uint32_t RESERVED1;
  __IO uint32_t ILPR;                       /*!< UART IrDA Low-Power Register          */
  __IO uint32_t IBRD;                       /*!< UART Integer Baud-Rate Divisor        */
  __IO uint32_t FBRD;                       /*!< UART Fractional Baud-Rate Divisor     */
  __IO uint32_t LCRH;                       /*!< UART Line Control                     */
  __IO uint32_t CTL;                        /*!< UART Control                          */
  __IO uint32_t IFLS;                       /*!< UART Interrupt FIFO Level Select      */
  __IO uint32_t IM;                         /*!< UART Interrupt Mask                   */
  __IO uint32_t RIS;                        /*!< UART Raw Interrupt Status             */
  __IO uint32_t MIS;                        /*!< UART Masked Interrupt Status          */
  __O  uint32_t ICR;                        /*!< UART Interrupt Clear                  */
  __IO uint32_t DMACTL;                     /*!< UART DMA Control                      */
  __I  uint32_t RESERVED2[17];
  __IO uint32_t LCTL;                       /*!< UART LIN Control                      */
  __IO uint32_t LSS;                        /*!< UART LIN Snap Shot                    */
  __IO uint32_t LTIM;                       /*!< UART LIN Timer                        */
  __I  uint32_t RESERVED3[2];
  __IO uint32_t _9BITADDR;                  /*!< UART 9-Bit Self Address               */
  __IO uint32_t _9BITAMASK;                 /*!< UART 9-Bit Self Address Mask          */
  __I  uint32_t RESERVED4[965];
  __IO uint32_t PP;                         /*!< UART Peripheral Properties            */
  __I  uint32_t RESERVED5;
  __IO uint32_t CC;                         /*!< UART Clock Configuration              */
} UART4_Type;


// ------------------------------------------------------------------------------------------------
// -----                                         UART5                                        -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for UART0 peripheral (UART5)
  */

typedef struct {                            /*!< UART5 Structure                       */
  __IO uint32_t DR;                         /*!< UART Data                             */
  
  union {
    __IO uint32_t ECR;                      /*!< UART Receive Status/Error Clear       */
    __IO uint32_t RSR;                      /*!< UART Receive Status/Error Clear       */
  } ;
  __I  uint32_t RESERVED0[4];
  __IO uint32_t FR;                         /*!< UART Flag                             */
  __I  uint32_t RESERVED1;
  __IO uint32_t ILPR;                       /*!< UART IrDA Low-Power Register          */
  __IO uint32_t IBRD;                       /*!< UART Integer Baud-Rate Divisor        */
  __IO uint32_t FBRD;                       /*!< UART Fractional Baud-Rate Divisor     */
  __IO uint32_t LCRH;                       /*!< UART Line Control                     */
  __IO uint32_t CTL;                        /*!< UART Control                          */
  __IO uint32_t IFLS;                       /*!< UART Interrupt FIFO Level Select      */
  __IO uint32_t IM;                         /*!< UART Interrupt Mask                   */
  __IO uint32_t RIS;                        /*!< UART Raw Interrupt Status             */
  __IO uint32_t MIS;                        /*!< UART Masked Interrupt Status          */
  __O  uint32_t ICR;                        /*!< UART Interrupt Clear                  */
  __IO uint32_t DMACTL;                     /*!< UART DMA Control                      */
  __I  uint32_t RESERVED2[17];
  __IO uint32_t LCTL;                       /*!< UART LIN Control                      */
  __IO uint32_t LSS;                        /*!< UART LIN Snap Shot                    */
  __IO uint32_t LTIM;                       /*!< UART LIN Timer                        */
  __I  uint32_t RESERVED3[2];
  __IO uint32_t _9BITADDR;                  /*!< UART 9-Bit Self Address               */
  __IO uint32_t _9BITAMASK;                 /*!< UART 9-Bit Self Address Mask          */
  __I  uint32_t RESERVED4[965];
  __IO uint32_t PP;                         /*!< UART Peripheral Properties            */
  __I  uint32_t RESERVED5;
  __IO uint32_t CC;                         /*!< UART Clock Configuration              */
} UART5_Type;


// ------------------------------------------------------------------------------------------------
// -----                                         UART6                                        -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for UART0 peripheral (UART6)
  */

typedef struct {                            /*!< UART6 Structure                       */
  __IO uint32_t DR;                         /*!< UART Data                             */
  
  union {
    __IO uint32_t ECR;                      /*!< UART Receive Status/Error Clear       */
    __IO uint32_t RSR;                      /*!< UART Receive Status/Error Clear       */
  } ;
  __I  uint32_t RESERVED0[4];
  __IO uint32_t FR;                         /*!< UART Flag                             */
  __I  uint32_t RESERVED1;
  __IO uint32_t ILPR;                       /*!< UART IrDA Low-Power Register          */
  __IO uint32_t IBRD;                       /*!< UART Integer Baud-Rate Divisor        */
  __IO uint32_t FBRD;                       /*!< UART Fractional Baud-Rate Divisor     */
  __IO uint32_t LCRH;                       /*!< UART Line Control                     */
  __IO uint32_t CTL;                        /*!< UART Control                          */
  __IO uint32_t IFLS;                       /*!< UART Interrupt FIFO Level Select      */
  __IO uint32_t IM;                         /*!< UART Interrupt Mask                   */
  __IO uint32_t RIS;                        /*!< UART Raw Interrupt Status             */
  __IO uint32_t MIS;                        /*!< UART Masked Interrupt Status          */
  __O  uint32_t ICR;                        /*!< UART Interrupt Clear                  */
  __IO uint32_t DMACTL;                     /*!< UART DMA Control                      */
  __I  uint32_t RESERVED2[17];
  __IO uint32_t LCTL;                       /*!< UART LIN Control                      */
  __IO uint32_t LSS;                        /*!< UART LIN Snap Shot                    */
  __IO uint32_t LTIM;                       /*!< UART LIN Timer                        */
  __I  uint32_t RESERVED3[2];
  __IO uint32_t _9BITADDR;                  /*!< UART 9-Bit Self Address               */
  __IO uint32_t _9BITAMASK;                 /*!< UART 9-Bit Self Address Mask          */
  __I  uint32_t RESERVED4[965];
  __IO uint32_t PP;                         /*!< UART Peripheral Properties            */
  __I  uint32_t RESERVED5;
  __IO uint32_t CC;                         /*!< UART Clock Configuration              */
} UART6_Type;


// ------------------------------------------------------------------------------------------------
// -----                                         UART7                                        -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for UART0 peripheral (UART7)
  */

typedef struct {                            /*!< UART7 Structure                       */
  __IO uint32_t DR;                         /*!< UART Data                             */
  
  union {
    __IO uint32_t ECR;                      /*!< UART Receive Status/Error Clear       */
    __IO uint32_t RSR;                      /*!< UART Receive Status/Error Clear       */
  } ;
  __I  uint32_t RESERVED0[4];
  __IO uint32_t FR;                         /*!< UART Flag                             */
  __I  uint32_t RESERVED1;
  __IO uint32_t ILPR;                       /*!< UART IrDA Low-Power Register          */
  __IO uint32_t IBRD;                       /*!< UART Integer Baud-Rate Divisor        */
  __IO uint32_t FBRD;                       /*!< UART Fractional Baud-Rate Divisor     */
  __IO uint32_t LCRH;                       /*!< UART Line Control                     */
  __IO uint32_t CTL;                        /*!< UART Control                          */
  __IO uint32_t IFLS;                       /*!< UART Interrupt FIFO Level Select      */
  __IO uint32_t IM;                         /*!< UART Interrupt Mask                   */
  __IO uint32_t RIS;                        /*!< UART Raw Interrupt Status             */
  __IO uint32_t MIS;                        /*!< UART Masked Interrupt Status          */
  __O  uint32_t ICR;                        /*!< UART Interrupt Clear                  */
  __IO uint32_t DMACTL;                     /*!< UART DMA Control                      */
  __I  uint32_t RESERVED2[17];
  __IO uint32_t LCTL;                       /*!< UART LIN Control                      */
  __IO uint32_t LSS;                        /*!< UART LIN Snap Shot                    */
  __IO uint32_t LTIM;                       /*!< UART LIN Timer                        */
  __I  uint32_t RESERVED3[2];
  __IO uint32_t _9BITADDR;                  /*!< UART 9-Bit Self Address               */
  __IO uint32_t _9BITAMASK;                 /*!< UART 9-Bit Self Address Mask          */
  __I  uint32_t RESERVED4[965];
  __IO uint32_t PP;                         /*!< UART Peripheral Properties            */
  __I  uint32_t RESERVED5;
  __IO uint32_t CC;                         /*!< UART Clock Configuration              */
} UART7_Type;


// ------------------------------------------------------------------------------------------------
// -----                                         I2C0                                         -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for I2C0 peripheral (I2C0)
  */

typedef struct {                            /*!< I2C0 Structure                        */
  __IO uint32_t MSA;                        /*!< I2C Master Slave Address              */
  
  union {
    __IO uint32_t MCS;                      /*!< I2C Master Control/Status             */
    __IO uint32_t MCS;                      /*!< I2C Master Control/Status             */
  } ;
  __IO uint32_t MDR;                        /*!< I2C Master Data                       */
  __IO uint32_t MTPR;                       /*!< I2C Master Timer Period               */
  __IO uint32_t MIMR;                       /*!< I2C Master Interrupt Mask             */
  __IO uint32_t MRIS;                       /*!< I2C Master Raw Interrupt Status       */
  __IO uint32_t MMIS;                       /*!< I2C Master Masked Interrupt Status    */
  __O  uint32_t MICR;                       /*!< I2C Master Interrupt Clear            */
  __IO uint32_t MCR;                        /*!< I2C Master Configuration              */
  __IO uint32_t MCLKOCNT;                   /*!< I2C Master Clock Low Timeout Count    */
  __I  uint32_t RESERVED0;
  __IO uint32_t MBMON;                      /*!< I2C Master Bus Monitor                */
  __I  uint32_t RESERVED1[500];
  __IO uint32_t SOAR;                       /*!< I2C Slave Own Address                 */
  
  union {
    __IO uint32_t SCSR;                     /*!< I2C Slave Control/Status              */
    __IO uint32_t SCSR;                     /*!< I2C Slave Control/Status              */
  } ;
  __IO uint32_t SDR;                        /*!< I2C Slave Data                        */
  __IO uint32_t SIMR;                       /*!< I2C Slave Interrupt Mask              */
  __IO uint32_t SRIS;                       /*!< I2C Slave Raw Interrupt Status        */
  __IO uint32_t SMIS;                       /*!< I2C Slave Masked Interrupt Status     */
  __O  uint32_t SICR;                       /*!< I2C Slave Interrupt Clear             */
  __IO uint32_t SOAR2;                      /*!< I2C Slave Own Address 2               */
  __IO uint32_t SACKCTL;                    /*!< I2C ACK Control                       */
  __I  uint32_t RESERVED2[487];
  __IO uint32_t PP;                         /*!< I2C Peripheral Properties             */
} I2C0_Type;


// ------------------------------------------------------------------------------------------------
// -----                                         I2C1                                         -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for I2C0 peripheral (I2C1)
  */

typedef struct {                            /*!< I2C1 Structure                        */
  __IO uint32_t MSA;                        /*!< I2C Master Slave Address              */
  
  union {
    __IO uint32_t MCS;                      /*!< I2C Master Control/Status             */
    __IO uint32_t MCS;                      /*!< I2C Master Control/Status             */
  } ;
  __IO uint32_t MDR;                        /*!< I2C Master Data                       */
  __IO uint32_t MTPR;                       /*!< I2C Master Timer Period               */
  __IO uint32_t MIMR;                       /*!< I2C Master Interrupt Mask             */
  __IO uint32_t MRIS;                       /*!< I2C Master Raw Interrupt Status       */
  __IO uint32_t MMIS;                       /*!< I2C Master Masked Interrupt Status    */
  __O  uint32_t MICR;                       /*!< I2C Master Interrupt Clear            */
  __IO uint32_t MCR;                        /*!< I2C Master Configuration              */
  __IO uint32_t MCLKOCNT;                   /*!< I2C Master Clock Low Timeout Count    */
  __I  uint32_t RESERVED0;
  __IO uint32_t MBMON;                      /*!< I2C Master Bus Monitor                */
  __I  uint32_t RESERVED1[500];
  __IO uint32_t SOAR;                       /*!< I2C Slave Own Address                 */
  
  union {
    __IO uint32_t SCSR;                     /*!< I2C Slave Control/Status              */
    __IO uint32_t SCSR;                     /*!< I2C Slave Control/Status              */
  } ;
  __IO uint32_t SDR;                        /*!< I2C Slave Data                        */
  __IO uint32_t SIMR;                       /*!< I2C Slave Interrupt Mask              */
  __IO uint32_t SRIS;                       /*!< I2C Slave Raw Interrupt Status        */
  __IO uint32_t SMIS;                       /*!< I2C Slave Masked Interrupt Status     */
  __O  uint32_t SICR;                       /*!< I2C Slave Interrupt Clear             */
  __IO uint32_t SOAR2;                      /*!< I2C Slave Own Address 2               */
  __IO uint32_t SACKCTL;                    /*!< I2C ACK Control                       */
  __I  uint32_t RESERVED2[487];
  __IO uint32_t PP;                         /*!< I2C Peripheral Properties             */
} I2C1_Type;


// ------------------------------------------------------------------------------------------------
// -----                                         I2C2                                         -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for I2C0 peripheral (I2C2)
  */

typedef struct {                            /*!< I2C2 Structure                        */
  __IO uint32_t MSA;                        /*!< I2C Master Slave Address              */
  
  union {
    __IO uint32_t MCS;                      /*!< I2C Master Control/Status             */
    __IO uint32_t MCS;                      /*!< I2C Master Control/Status             */
  } ;
  __IO uint32_t MDR;                        /*!< I2C Master Data                       */
  __IO uint32_t MTPR;                       /*!< I2C Master Timer Period               */
  __IO uint32_t MIMR;                       /*!< I2C Master Interrupt Mask             */
  __IO uint32_t MRIS;                       /*!< I2C Master Raw Interrupt Status       */
  __IO uint32_t MMIS;                       /*!< I2C Master Masked Interrupt Status    */
  __O  uint32_t MICR;                       /*!< I2C Master Interrupt Clear            */
  __IO uint32_t MCR;                        /*!< I2C Master Configuration              */
  __IO uint32_t MCLKOCNT;                   /*!< I2C Master Clock Low Timeout Count    */
  __I  uint32_t RESERVED0;
  __IO uint32_t MBMON;                      /*!< I2C Master Bus Monitor                */
  __I  uint32_t RESERVED1[500];
  __IO uint32_t SOAR;                       /*!< I2C Slave Own Address                 */
  
  union {
    __IO uint32_t SCSR;                     /*!< I2C Slave Control/Status              */
    __IO uint32_t SCSR;                     /*!< I2C Slave Control/Status              */
  } ;
  __IO uint32_t SDR;                        /*!< I2C Slave Data                        */
  __IO uint32_t SIMR;                       /*!< I2C Slave Interrupt Mask              */
  __IO uint32_t SRIS;                       /*!< I2C Slave Raw Interrupt Status        */
  __IO uint32_t SMIS;                       /*!< I2C Slave Masked Interrupt Status     */
  __O  uint32_t SICR;                       /*!< I2C Slave Interrupt Clear             */
  __IO uint32_t SOAR2;                      /*!< I2C Slave Own Address 2               */
  __IO uint32_t SACKCTL;                    /*!< I2C ACK Control                       */
  __I  uint32_t RESERVED2[487];
  __IO uint32_t PP;                         /*!< I2C Peripheral Properties             */
} I2C2_Type;


// ------------------------------------------------------------------------------------------------
// -----                                         I2C3                                         -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for I2C0 peripheral (I2C3)
  */

typedef struct {                            /*!< I2C3 Structure                        */
  __IO uint32_t MSA;                        /*!< I2C Master Slave Address              */
  
  union {
    __IO uint32_t MCS;                      /*!< I2C Master Control/Status             */
    __IO uint32_t MCS;                      /*!< I2C Master Control/Status             */
  } ;
  __IO uint32_t MDR;                        /*!< I2C Master Data                       */
  __IO uint32_t MTPR;                       /*!< I2C Master Timer Period               */
  __IO uint32_t MIMR;                       /*!< I2C Master Interrupt Mask             */
  __IO uint32_t MRIS;                       /*!< I2C Master Raw Interrupt Status       */
  __IO uint32_t MMIS;                       /*!< I2C Master Masked Interrupt Status    */
  __O  uint32_t MICR;                       /*!< I2C Master Interrupt Clear            */
  __IO uint32_t MCR;                        /*!< I2C Master Configuration              */
  __IO uint32_t MCLKOCNT;                   /*!< I2C Master Clock Low Timeout Count    */
  __I  uint32_t RESERVED0;
  __IO uint32_t MBMON;                      /*!< I2C Master Bus Monitor                */
  __I  uint32_t RESERVED1[500];
  __IO uint32_t SOAR;                       /*!< I2C Slave Own Address                 */
  
  union {
    __IO uint32_t SCSR;                     /*!< I2C Slave Control/Status              */
    __IO uint32_t SCSR;                     /*!< I2C Slave Control/Status              */
  } ;
  __IO uint32_t SDR;                        /*!< I2C Slave Data                        */
  __IO uint32_t SIMR;                       /*!< I2C Slave Interrupt Mask              */
  __IO uint32_t SRIS;                       /*!< I2C Slave Raw Interrupt Status        */
  __IO uint32_t SMIS;                       /*!< I2C Slave Masked Interrupt Status     */
  __O  uint32_t SICR;                       /*!< I2C Slave Interrupt Clear             */
  __IO uint32_t SOAR2;                      /*!< I2C Slave Own Address 2               */
  __IO uint32_t SACKCTL;                    /*!< I2C ACK Control                       */
  __I  uint32_t RESERVED2[487];
  __IO uint32_t PP;                         /*!< I2C Peripheral Properties             */
} I2C3_Type;


// ------------------------------------------------------------------------------------------------
// -----                                      GPIO_PORTE                                      -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for GPIO_PORTA peripheral (GPIO_PORTE)
  */

typedef struct {                            /*!< GPIO_PORTE Structure                  */
  __I  uint32_t RESERVED0[255];
  __IO uint32_t DATA;                       /*!< GPIO Data                             */
  __IO uint32_t DIR;                        /*!< GPIO Direction                        */
  __IO uint32_t IS;                         /*!< GPIO Interrupt Sense                  */
  __IO uint32_t IBE;                        /*!< GPIO Interrupt Both Edges             */
  __IO uint32_t IEV;                        /*!< GPIO Interrupt Event                  */
  __IO uint32_t IM;                         /*!< GPIO Interrupt Mask                   */
  __IO uint32_t RIS;                        /*!< GPIO Raw Interrupt Status             */
  __IO uint32_t MIS;                        /*!< GPIO Masked Interrupt Status          */
  __O  uint32_t ICR;                        /*!< GPIO Interrupt Clear                  */
  __IO uint32_t AFSEL;                      /*!< GPIO Alternate Function Select        */
  __I  uint32_t RESERVED1[55];
  __IO uint32_t DR2R;                       /*!< GPIO 2-mA Drive Select                */
  __IO uint32_t DR4R;                       /*!< GPIO 4-mA Drive Select                */
  __IO uint32_t DR8R;                       /*!< GPIO 8-mA Drive Select                */
  __IO uint32_t ODR;                        /*!< GPIO Open Drain Select                */
  __IO uint32_t PUR;                        /*!< GPIO Pull-Up Select                   */
  __IO uint32_t PDR;                        /*!< GPIO Pull-Down Select                 */
  __IO uint32_t SLR;                        /*!< GPIO Slew Rate Control Select         */
  __IO uint32_t DEN;                        /*!< GPIO Digital Enable                   */
  __IO uint32_t LOCK;                       /*!< GPIO Lock                             */
  __I  uint32_t CR;                         /*!< GPIO Commit                           */
  __IO uint32_t AMSEL;                      /*!< GPIO Analog Mode Select               */
  __IO uint32_t PCTL;                       /*!< GPIO Port Control                     */
  __IO uint32_t ADCCTL;                     /*!< GPIO ADC Control                      */
  __IO uint32_t DMACTL;                     /*!< GPIO DMA Control                      */
  __IO uint32_t SI;                         /*!< GPIO Select Interrupt                 */
} GPIO_PORTE_Type;


// ------------------------------------------------------------------------------------------------
// -----                                      GPIO_PORTF                                      -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for GPIO_PORTA peripheral (GPIO_PORTF)
  */

typedef struct {                            /*!< GPIO_PORTF Structure                  */
  __I  uint32_t RESERVED0[255];
  __IO uint32_t DATA;                       /*!< GPIO Data                             */
  __IO uint32_t DIR;                        /*!< GPIO Direction                        */
  __IO uint32_t IS;                         /*!< GPIO Interrupt Sense                  */
  __IO uint32_t IBE;                        /*!< GPIO Interrupt Both Edges             */
  __IO uint32_t IEV;                        /*!< GPIO Interrupt Event                  */
  __IO uint32_t IM;                         /*!< GPIO Interrupt Mask                   */
  __IO uint32_t RIS;                        /*!< GPIO Raw Interrupt Status             */
  __IO uint32_t MIS;                        /*!< GPIO Masked Interrupt Status          */
  __O  uint32_t ICR;                        /*!< GPIO Interrupt Clear                  */
  __IO uint32_t AFSEL;                      /*!< GPIO Alternate Function Select        */
  __I  uint32_t RESERVED1[55];
  __IO uint32_t DR2R;                       /*!< GPIO 2-mA Drive Select                */
  __IO uint32_t DR4R;                       /*!< GPIO 4-mA Drive Select                */
  __IO uint32_t DR8R;                       /*!< GPIO 8-mA Drive Select                */
  __IO uint32_t ODR;                        /*!< GPIO Open Drain Select                */
  __IO uint32_t PUR;                        /*!< GPIO Pull-Up Select                   */
  __IO uint32_t PDR;                        /*!< GPIO Pull-Down Select                 */
  __IO uint32_t SLR;                        /*!< GPIO Slew Rate Control Select         */
  __IO uint32_t DEN;                        /*!< GPIO Digital Enable                   */
  __IO uint32_t LOCK;                       /*!< GPIO Lock                             */
  __I  uint32_t CR;                         /*!< GPIO Commit                           */
  __IO uint32_t AMSEL;                      /*!< GPIO Analog Mode Select               */
  __IO uint32_t PCTL;                       /*!< GPIO Port Control                     */
  __IO uint32_t ADCCTL;                     /*!< GPIO ADC Control                      */
  __IO uint32_t DMACTL;                     /*!< GPIO DMA Control                      */
  __IO uint32_t SI;                         /*!< GPIO Select Interrupt                 */
} GPIO_PORTF_Type;


// ------------------------------------------------------------------------------------------------
// -----                                      GPIO_PORTG                                      -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for GPIO_PORTA peripheral (GPIO_PORTG)
  */

typedef struct {                            /*!< GPIO_PORTG Structure                  */
  __I  uint32_t RESERVED0[255];
  __IO uint32_t DATA;                       /*!< GPIO Data                             */
  __IO uint32_t DIR;                        /*!< GPIO Direction                        */
  __IO uint32_t IS;                         /*!< GPIO Interrupt Sense                  */
  __IO uint32_t IBE;                        /*!< GPIO Interrupt Both Edges             */
  __IO uint32_t IEV;                        /*!< GPIO Interrupt Event                  */
  __IO uint32_t IM;                         /*!< GPIO Interrupt Mask                   */
  __IO uint32_t RIS;                        /*!< GPIO Raw Interrupt Status             */
  __IO uint32_t MIS;                        /*!< GPIO Masked Interrupt Status          */
  __O  uint32_t ICR;                        /*!< GPIO Interrupt Clear                  */
  __IO uint32_t AFSEL;                      /*!< GPIO Alternate Function Select        */
  __I  uint32_t RESERVED1[55];
  __IO uint32_t DR2R;                       /*!< GPIO 2-mA Drive Select                */
  __IO uint32_t DR4R;                       /*!< GPIO 4-mA Drive Select                */
  __IO uint32_t DR8R;                       /*!< GPIO 8-mA Drive Select                */
  __IO uint32_t ODR;                        /*!< GPIO Open Drain Select                */
  __IO uint32_t PUR;                        /*!< GPIO Pull-Up Select                   */
  __IO uint32_t PDR;                        /*!< GPIO Pull-Down Select                 */
  __IO uint32_t SLR;                        /*!< GPIO Slew Rate Control Select         */
  __IO uint32_t DEN;                        /*!< GPIO Digital Enable                   */
  __IO uint32_t LOCK;                       /*!< GPIO Lock                             */
  __I  uint32_t CR;                         /*!< GPIO Commit                           */
  __IO uint32_t AMSEL;                      /*!< GPIO Analog Mode Select               */
  __IO uint32_t PCTL;                       /*!< GPIO Port Control                     */
  __IO uint32_t ADCCTL;                     /*!< GPIO ADC Control                      */
  __IO uint32_t DMACTL;                     /*!< GPIO DMA Control                      */
  __IO uint32_t SI;                         /*!< GPIO Select Interrupt                 */
} GPIO_PORTG_Type;


// ------------------------------------------------------------------------------------------------
// -----                                         PWM0                                         -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for PWM0 peripheral (PWM0)
  */

typedef struct {                            /*!< PWM0 Structure                        */
  __IO uint32_t CTL;                        /*!< PWM Master Control                    */
  __IO uint32_t SYNC;                       /*!< PWM Time Base Sync                    */
  __IO uint32_t ENABLE;                     /*!< PWM Output Enable                     */
  __IO uint32_t INVERT;                     /*!< PWM Output Inversion                  */
  __IO uint32_t FAULT;                      /*!< PWM Output Fault                      */
  __IO uint32_t INTEN;                      /*!< PWM Interrupt Enable                  */
  __IO uint32_t RIS;                        /*!< PWM Raw Interrupt Status              */
  __IO uint32_t ISC;                        /*!< PWM Interrupt Status and Clear        */
  __IO uint32_t STATUS;                     /*!< PWM Status                            */
  __IO uint32_t FAULTVAL;                   /*!< PWM Fault Condition Value             */
  __IO uint32_t ENUPD;                      /*!< PWM Enable Update                     */
  __I  uint32_t RESERVED0[5];
  __IO uint32_t _0_CTL;                     /*!< PWM0 Control                          */
  __IO uint32_t _0_INTEN;                   /*!< PWM0 Interrupt and Trigger Enable     */
  __IO uint32_t _0_RIS;                     /*!< PWM0 Raw Interrupt Status             */
  __IO uint32_t _0_ISC;                     /*!< PWM0 Interrupt Status and Clear       */
  __IO uint32_t _0_LOAD;                    /*!< PWM0 Load                             */
  __IO uint32_t _0_COUNT;                   /*!< PWM0 Counter                          */
  __IO uint32_t _0_CMPA;                    /*!< PWM0 Compare A                        */
  __IO uint32_t _0_CMPB;                    /*!< PWM0 Compare B                        */
  __IO uint32_t _0_GENA;                    /*!< PWM0 Generator A Control              */
  __IO uint32_t _0_GENB;                    /*!< PWM0 Generator B Control              */
  __IO uint32_t _0_DBCTL;                   /*!< PWM0 Dead-Band Control                */
  __IO uint32_t _0_DBRISE;                  /*!< PWM0 Dead-Band Rising-Edge Delay      */
  __IO uint32_t _0_DBFALL;                  /*!< PWM0 Dead-Band Falling-Edge-Delay     */
  __IO uint32_t _0_FLTSRC0;                 /*!< PWM0 Fault Source 0                   */
  __IO uint32_t _0_FLTSRC1;                 /*!< PWM0 Fault Source 1                   */
  __IO uint32_t _0_MINFLTPER;               /*!< PWM0 Minimum Fault Period             */
  __IO uint32_t _1_CTL;                     /*!< PWM1 Control                          */
  __IO uint32_t _1_INTEN;                   /*!< PWM1 Interrupt and Trigger Enable     */
  __IO uint32_t _1_RIS;                     /*!< PWM1 Raw Interrupt Status             */
  __IO uint32_t _1_ISC;                     /*!< PWM1 Interrupt Status and Clear       */
  __IO uint32_t _1_LOAD;                    /*!< PWM1 Load                             */
  __IO uint32_t _1_COUNT;                   /*!< PWM1 Counter                          */
  __IO uint32_t _1_CMPA;                    /*!< PWM1 Compare A                        */
  __IO uint32_t _1_CMPB;                    /*!< PWM1 Compare B                        */
  __IO uint32_t _1_GENA;                    /*!< PWM1 Generator A Control              */
  __IO uint32_t _1_GENB;                    /*!< PWM1 Generator B Control              */
  __IO uint32_t _1_DBCTL;                   /*!< PWM1 Dead-Band Control                */
  __IO uint32_t _1_DBRISE;                  /*!< PWM1 Dead-Band Rising-Edge Delay      */
  __IO uint32_t _1_DBFALL;                  /*!< PWM1 Dead-Band Falling-Edge-Delay     */
  __IO uint32_t _1_FLTSRC0;                 /*!< PWM1 Fault Source 0                   */
  __IO uint32_t _1_FLTSRC1;                 /*!< PWM1 Fault Source 1                   */
  __IO uint32_t _1_MINFLTPER;               /*!< PWM1 Minimum Fault Period             */
  __IO uint32_t _2_CTL;                     /*!< PWM2 Control                          */
  __IO uint32_t _2_INTEN;                   /*!< PWM2 Interrupt and Trigger Enable     */
  __IO uint32_t _2_RIS;                     /*!< PWM2 Raw Interrupt Status             */
  __IO uint32_t _2_ISC;                     /*!< PWM2 Interrupt Status and Clear       */
  __IO uint32_t _2_LOAD;                    /*!< PWM2 Load                             */
  __IO uint32_t _2_COUNT;                   /*!< PWM2 Counter                          */
  __IO uint32_t _2_CMPA;                    /*!< PWM2 Compare A                        */
  __IO uint32_t _2_CMPB;                    /*!< PWM2 Compare B                        */
  __IO uint32_t _2_GENA;                    /*!< PWM2 Generator A Control              */
  __IO uint32_t _2_GENB;                    /*!< PWM2 Generator B Control              */
  __IO uint32_t _2_DBCTL;                   /*!< PWM2 Dead-Band Control                */
  __IO uint32_t _2_DBRISE;                  /*!< PWM2 Dead-Band Rising-Edge Delay      */
  __IO uint32_t _2_DBFALL;                  /*!< PWM2 Dead-Band Falling-Edge-Delay     */
  __IO uint32_t _2_FLTSRC0;                 /*!< PWM2 Fault Source 0                   */
  __IO uint32_t _2_FLTSRC1;                 /*!< PWM2 Fault Source 1                   */
  __IO uint32_t _2_MINFLTPER;               /*!< PWM2 Minimum Fault Period             */
  __IO uint32_t _3_CTL;                     /*!< PWM3 Control                          */
  __IO uint32_t _3_INTEN;                   /*!< PWM3 Interrupt and Trigger Enable     */
  __IO uint32_t _3_RIS;                     /*!< PWM3 Raw Interrupt Status             */
  __IO uint32_t _3_ISC;                     /*!< PWM3 Interrupt Status and Clear       */
  __IO uint32_t _3_LOAD;                    /*!< PWM3 Load                             */
  __IO uint32_t _3_COUNT;                   /*!< PWM3 Counter                          */
  __IO uint32_t _3_CMPA;                    /*!< PWM3 Compare A                        */
  __IO uint32_t _3_CMPB;                    /*!< PWM3 Compare B                        */
  __IO uint32_t _3_GENA;                    /*!< PWM3 Generator A Control              */
  __IO uint32_t _3_GENB;                    /*!< PWM3 Generator B Control              */
  __IO uint32_t _3_DBCTL;                   /*!< PWM3 Dead-Band Control                */
  __IO uint32_t _3_DBRISE;                  /*!< PWM3 Dead-Band Rising-Edge Delay      */
  __IO uint32_t _3_DBFALL;                  /*!< PWM3 Dead-Band Falling-Edge-Delay     */
  __IO uint32_t _3_FLTSRC0;                 /*!< PWM3 Fault Source 0                   */
  __IO uint32_t _3_FLTSRC1;                 /*!< PWM3 Fault Source 1                   */
  __IO uint32_t _3_MINFLTPER;               /*!< PWM3 Minimum Fault Period             */
  __I  uint32_t RESERVED1[432];
  __IO uint32_t _0_FLTSEN;                  /*!< PWM0 Fault Pin Logic Sense            */
  __I  uint32_t _0_FLTSTAT0;                /*!< PWM0 Fault Status 0                   */
  __I  uint32_t _0_FLTSTAT1;                /*!< PWM0 Fault Status 1                   */
  __I  uint32_t RESERVED2[29];
  __IO uint32_t _1_FLTSEN;                  /*!< PWM1 Fault Pin Logic Sense            */
  __I  uint32_t _1_FLTSTAT0;                /*!< PWM1 Fault Status 0                   */
  __I  uint32_t _1_FLTSTAT1;                /*!< PWM1 Fault Status 1                   */
  __I  uint32_t RESERVED3[29];
  __IO uint32_t _2_FLTSEN;                  /*!< PWM2 Fault Pin Logic Sense            */
  __I  uint32_t _2_FLTSTAT0;                /*!< PWM2 Fault Status 0                   */
  __I  uint32_t _2_FLTSTAT1;                /*!< PWM2 Fault Status 1                   */
  __I  uint32_t RESERVED4[29];
  __IO uint32_t _3_FLTSEN;                  /*!< PWM3 Fault Pin Logic Sense            */
  __I  uint32_t _3_FLTSTAT0;                /*!< PWM3 Fault Status 0                   */
  __I  uint32_t _3_FLTSTAT1;                /*!< PWM3 Fault Status 1                   */
} PWM0_Type;


// ------------------------------------------------------------------------------------------------
// -----                                         PWM1                                         -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for PWM0 peripheral (PWM1)
  */

typedef struct {                            /*!< PWM1 Structure                        */
  __IO uint32_t CTL;                        /*!< PWM Master Control                    */
  __IO uint32_t SYNC;                       /*!< PWM Time Base Sync                    */
  __IO uint32_t ENABLE;                     /*!< PWM Output Enable                     */
  __IO uint32_t INVERT;                     /*!< PWM Output Inversion                  */
  __IO uint32_t FAULT;                      /*!< PWM Output Fault                      */
  __IO uint32_t INTEN;                      /*!< PWM Interrupt Enable                  */
  __IO uint32_t RIS;                        /*!< PWM Raw Interrupt Status              */
  __IO uint32_t ISC;                        /*!< PWM Interrupt Status and Clear        */
  __IO uint32_t STATUS;                     /*!< PWM Status                            */
  __IO uint32_t FAULTVAL;                   /*!< PWM Fault Condition Value             */
  __IO uint32_t ENUPD;                      /*!< PWM Enable Update                     */
  __I  uint32_t RESERVED0[5];
  __IO uint32_t _0_CTL;                     /*!< PWM0 Control                          */
  __IO uint32_t _0_INTEN;                   /*!< PWM0 Interrupt and Trigger Enable     */
  __IO uint32_t _0_RIS;                     /*!< PWM0 Raw Interrupt Status             */
  __IO uint32_t _0_ISC;                     /*!< PWM0 Interrupt Status and Clear       */
  __IO uint32_t _0_LOAD;                    /*!< PWM0 Load                             */
  __IO uint32_t _0_COUNT;                   /*!< PWM0 Counter                          */
  __IO uint32_t _0_CMPA;                    /*!< PWM0 Compare A                        */
  __IO uint32_t _0_CMPB;                    /*!< PWM0 Compare B                        */
  __IO uint32_t _0_GENA;                    /*!< PWM0 Generator A Control              */
  __IO uint32_t _0_GENB;                    /*!< PWM0 Generator B Control              */
  __IO uint32_t _0_DBCTL;                   /*!< PWM0 Dead-Band Control                */
  __IO uint32_t _0_DBRISE;                  /*!< PWM0 Dead-Band Rising-Edge Delay      */
  __IO uint32_t _0_DBFALL;                  /*!< PWM0 Dead-Band Falling-Edge-Delay     */
  __IO uint32_t _0_FLTSRC0;                 /*!< PWM0 Fault Source 0                   */
  __IO uint32_t _0_FLTSRC1;                 /*!< PWM0 Fault Source 1                   */
  __IO uint32_t _0_MINFLTPER;               /*!< PWM0 Minimum Fault Period             */
  __IO uint32_t _1_CTL;                     /*!< PWM1 Control                          */
  __IO uint32_t _1_INTEN;                   /*!< PWM1 Interrupt and Trigger Enable     */
  __IO uint32_t _1_RIS;                     /*!< PWM1 Raw Interrupt Status             */
  __IO uint32_t _1_ISC;                     /*!< PWM1 Interrupt Status and Clear       */
  __IO uint32_t _1_LOAD;                    /*!< PWM1 Load                             */
  __IO uint32_t _1_COUNT;                   /*!< PWM1 Counter                          */
  __IO uint32_t _1_CMPA;                    /*!< PWM1 Compare A                        */
  __IO uint32_t _1_CMPB;                    /*!< PWM1 Compare B                        */
  __IO uint32_t _1_GENA;                    /*!< PWM1 Generator A Control              */
  __IO uint32_t _1_GENB;                    /*!< PWM1 Generator B Control              */
  __IO uint32_t _1_DBCTL;                   /*!< PWM1 Dead-Band Control                */
  __IO uint32_t _1_DBRISE;                  /*!< PWM1 Dead-Band Rising-Edge Delay      */
  __IO uint32_t _1_DBFALL;                  /*!< PWM1 Dead-Band Falling-Edge-Delay     */
  __IO uint32_t _1_FLTSRC0;                 /*!< PWM1 Fault Source 0                   */
  __IO uint32_t _1_FLTSRC1;                 /*!< PWM1 Fault Source 1                   */
  __IO uint32_t _1_MINFLTPER;               /*!< PWM1 Minimum Fault Period             */
  __IO uint32_t _2_CTL;                     /*!< PWM2 Control                          */
  __IO uint32_t _2_INTEN;                   /*!< PWM2 Interrupt and Trigger Enable     */
  __IO uint32_t _2_RIS;                     /*!< PWM2 Raw Interrupt Status             */
  __IO uint32_t _2_ISC;                     /*!< PWM2 Interrupt Status and Clear       */
  __IO uint32_t _2_LOAD;                    /*!< PWM2 Load                             */
  __IO uint32_t _2_COUNT;                   /*!< PWM2 Counter                          */
  __IO uint32_t _2_CMPA;                    /*!< PWM2 Compare A                        */
  __IO uint32_t _2_CMPB;                    /*!< PWM2 Compare B                        */
  __IO uint32_t _2_GENA;                    /*!< PWM2 Generator A Control              */
  __IO uint32_t _2_GENB;                    /*!< PWM2 Generator B Control              */
  __IO uint32_t _2_DBCTL;                   /*!< PWM2 Dead-Band Control                */
  __IO uint32_t _2_DBRISE;                  /*!< PWM2 Dead-Band Rising-Edge Delay      */
  __IO uint32_t _2_DBFALL;                  /*!< PWM2 Dead-Band Falling-Edge-Delay     */
  __IO uint32_t _2_FLTSRC0;                 /*!< PWM2 Fault Source 0                   */
  __IO uint32_t _2_FLTSRC1;                 /*!< PWM2 Fault Source 1                   */
  __IO uint32_t _2_MINFLTPER;               /*!< PWM2 Minimum Fault Period             */
  __IO uint32_t _3_CTL;                     /*!< PWM3 Control                          */
  __IO uint32_t _3_INTEN;                   /*!< PWM3 Interrupt and Trigger Enable     */
  __IO uint32_t _3_RIS;                     /*!< PWM3 Raw Interrupt Status             */
  __IO uint32_t _3_ISC;                     /*!< PWM3 Interrupt Status and Clear       */
  __IO uint32_t _3_LOAD;                    /*!< PWM3 Load                             */
  __IO uint32_t _3_COUNT;                   /*!< PWM3 Counter                          */
  __IO uint32_t _3_CMPA;                    /*!< PWM3 Compare A                        */
  __IO uint32_t _3_CMPB;                    /*!< PWM3 Compare B                        */
  __IO uint32_t _3_GENA;                    /*!< PWM3 Generator A Control              */
  __IO uint32_t _3_GENB;                    /*!< PWM3 Generator B Control              */
  __IO uint32_t _3_DBCTL;                   /*!< PWM3 Dead-Band Control                */
  __IO uint32_t _3_DBRISE;                  /*!< PWM3 Dead-Band Rising-Edge Delay      */
  __IO uint32_t _3_DBFALL;                  /*!< PWM3 Dead-Band Falling-Edge-Delay     */
  __IO uint32_t _3_FLTSRC0;                 /*!< PWM3 Fault Source 0                   */
  __IO uint32_t _3_FLTSRC1;                 /*!< PWM3 Fault Source 1                   */
  __IO uint32_t _3_MINFLTPER;               /*!< PWM3 Minimum Fault Period             */
  __I  uint32_t RESERVED1[432];
  __IO uint32_t _0_FLTSEN;                  /*!< PWM0 Fault Pin Logic Sense            */
  __I  uint32_t _0_FLTSTAT0;                /*!< PWM0 Fault Status 0                   */
  __I  uint32_t _0_FLTSTAT1;                /*!< PWM0 Fault Status 1                   */
  __I  uint32_t RESERVED2[29];
  __IO uint32_t _1_FLTSEN;                  /*!< PWM1 Fault Pin Logic Sense            */
  __I  uint32_t _1_FLTSTAT0;                /*!< PWM1 Fault Status 0                   */
  __I  uint32_t _1_FLTSTAT1;                /*!< PWM1 Fault Status 1                   */
  __I  uint32_t RESERVED3[29];
  __IO uint32_t _2_FLTSEN;                  /*!< PWM2 Fault Pin Logic Sense            */
  __I  uint32_t _2_FLTSTAT0;                /*!< PWM2 Fault Status 0                   */
  __I  uint32_t _2_FLTSTAT1;                /*!< PWM2 Fault Status 1                   */
  __I  uint32_t RESERVED4[29];
  __IO uint32_t _3_FLTSEN;                  /*!< PWM3 Fault Pin Logic Sense            */
  __I  uint32_t _3_FLTSTAT0;                /*!< PWM3 Fault Status 0                   */
  __I  uint32_t _3_FLTSTAT1;                /*!< PWM3 Fault Status 1                   */
} PWM1_Type;


// ------------------------------------------------------------------------------------------------
// -----                                         QEI0                                         -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for QEI0 peripheral (QEI0)
  */

typedef struct {                            /*!< QEI0 Structure                        */
  __IO uint32_t CTL;                        /*!< QEI Control                           */
  __IO uint32_t STAT;                       /*!< QEI Status                            */
  __IO uint32_t POS;                        /*!< QEI Position                          */
  __IO uint32_t MAXPOS;                     /*!< QEI Maximum Position                  */
  __IO uint32_t LOAD;                       /*!< QEI Timer Load                        */
  __IO uint32_t TIME;                       /*!< QEI Timer                             */
  __IO uint32_t COUNT;                      /*!< QEI Velocity Counter                  */
  __IO uint32_t SPEED;                      /*!< QEI Velocity                          */
  __IO uint32_t INTEN;                      /*!< QEI Interrupt Enable                  */
  __IO uint32_t RIS;                        /*!< QEI Raw Interrupt Status              */
  __IO uint32_t ISC;                        /*!< QEI Interrupt Status and Clear        */
} QEI0_Type;


// ------------------------------------------------------------------------------------------------
// -----                                         QEI1                                         -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for QEI0 peripheral (QEI1)
  */

typedef struct {                            /*!< QEI1 Structure                        */
  __IO uint32_t CTL;                        /*!< QEI Control                           */
  __IO uint32_t STAT;                       /*!< QEI Status                            */
  __IO uint32_t POS;                        /*!< QEI Position                          */
  __IO uint32_t MAXPOS;                     /*!< QEI Maximum Position                  */
  __IO uint32_t LOAD;                       /*!< QEI Timer Load                        */
  __IO uint32_t TIME;                       /*!< QEI Timer                             */
  __IO uint32_t COUNT;                      /*!< QEI Velocity Counter                  */
  __IO uint32_t SPEED;                      /*!< QEI Velocity                          */
  __IO uint32_t INTEN;                      /*!< QEI Interrupt Enable                  */
  __IO uint32_t RIS;                        /*!< QEI Raw Interrupt Status              */
  __IO uint32_t ISC;                        /*!< QEI Interrupt Status and Clear        */
} QEI1_Type;


// ------------------------------------------------------------------------------------------------
// -----                                        TIMER0                                        -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for TIMER0 peripheral (TIMER0)
  */

typedef struct {                            /*!< TIMER0 Structure                      */
  __IO uint32_t CFG;                        /*!< GPTM Configuration                    */
  __IO uint32_t TAMR;                       /*!< GPTM Timer A Mode                     */
  __IO uint32_t TBMR;                       /*!< GPTM Timer B Mode                     */
  __IO uint32_t CTL;                        /*!< GPTM Control                          */
  __IO uint32_t SYNC;                       /*!< GPTM Synchronize                      */
  __I  uint32_t RESERVED0;
  __IO uint32_t IMR;                        /*!< GPTM Interrupt Mask                   */
  __IO uint32_t RIS;                        /*!< GPTM Raw Interrupt Status             */
  __IO uint32_t MIS;                        /*!< GPTM Masked Interrupt Status          */
  __O  uint32_t ICR;                        /*!< GPTM Interrupt Clear                  */
  __IO uint32_t TAILR;                      /*!< GPTM Timer A Interval Load            */
  __IO uint32_t TBILR;                      /*!< GPTM Timer B Interval Load            */
  __IO uint32_t TAMATCHR;                   /*!< GPTM Timer A Match                    */
  __IO uint32_t TBMATCHR;                   /*!< GPTM Timer B Match                    */
  __IO uint32_t TAPR;                       /*!< GPTM Timer A Prescale                 */
  __IO uint32_t TBPR;                       /*!< GPTM Timer B Prescale                 */
  __IO uint32_t TAPMR;                      /*!< GPTM TimerA Prescale Match            */
  __IO uint32_t TBPMR;                      /*!< GPTM TimerB Prescale Match            */
  __IO uint32_t TAR;                        /*!< GPTM Timer A                          */
  __IO uint32_t TBR;                        /*!< GPTM Timer B                          */
  __IO uint32_t TAV;                        /*!< GPTM Timer A Value                    */
  __IO uint32_t TBV;                        /*!< GPTM Timer B Value                    */
  __IO uint32_t RTCPD;                      /*!< GPTM RTC Predivide                    */
  __IO uint32_t TAPS;                       /*!< GPTM Timer A Prescale Snapshot        */
  __IO uint32_t TBPS;                       /*!< GPTM Timer B Prescale Snapshot        */
  __IO uint32_t TAPV;                       /*!< GPTM Timer A Prescale Value           */
  __IO uint32_t TBPV;                       /*!< GPTM Timer B Prescale Value           */
  __I  uint32_t RESERVED1[981];
  __IO uint32_t PP;                         /*!< GPTM Peripheral Properties            */
} TIMER0_Type;


// ------------------------------------------------------------------------------------------------
// -----                                        TIMER1                                        -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for TIMER0 peripheral (TIMER1)
  */

typedef struct {                            /*!< TIMER1 Structure                      */
  __IO uint32_t CFG;                        /*!< GPTM Configuration                    */
  __IO uint32_t TAMR;                       /*!< GPTM Timer A Mode                     */
  __IO uint32_t TBMR;                       /*!< GPTM Timer B Mode                     */
  __IO uint32_t CTL;                        /*!< GPTM Control                          */
  __IO uint32_t SYNC;                       /*!< GPTM Synchronize                      */
  __I  uint32_t RESERVED0;
  __IO uint32_t IMR;                        /*!< GPTM Interrupt Mask                   */
  __IO uint32_t RIS;                        /*!< GPTM Raw Interrupt Status             */
  __IO uint32_t MIS;                        /*!< GPTM Masked Interrupt Status          */
  __O  uint32_t ICR;                        /*!< GPTM Interrupt Clear                  */
  __IO uint32_t TAILR;                      /*!< GPTM Timer A Interval Load            */
  __IO uint32_t TBILR;                      /*!< GPTM Timer B Interval Load            */
  __IO uint32_t TAMATCHR;                   /*!< GPTM Timer A Match                    */
  __IO uint32_t TBMATCHR;                   /*!< GPTM Timer B Match                    */
  __IO uint32_t TAPR;                       /*!< GPTM Timer A Prescale                 */
  __IO uint32_t TBPR;                       /*!< GPTM Timer B Prescale                 */
  __IO uint32_t TAPMR;                      /*!< GPTM TimerA Prescale Match            */
  __IO uint32_t TBPMR;                      /*!< GPTM TimerB Prescale Match            */
  __IO uint32_t TAR;                        /*!< GPTM Timer A                          */
  __IO uint32_t TBR;                        /*!< GPTM Timer B                          */
  __IO uint32_t TAV;                        /*!< GPTM Timer A Value                    */
  __IO uint32_t TBV;                        /*!< GPTM Timer B Value                    */
  __IO uint32_t RTCPD;                      /*!< GPTM RTC Predivide                    */
  __IO uint32_t TAPS;                       /*!< GPTM Timer A Prescale Snapshot        */
  __IO uint32_t TBPS;                       /*!< GPTM Timer B Prescale Snapshot        */
  __IO uint32_t TAPV;                       /*!< GPTM Timer A Prescale Value           */
  __IO uint32_t TBPV;                       /*!< GPTM Timer B Prescale Value           */
  __I  uint32_t RESERVED1[981];
  __IO uint32_t PP;                         /*!< GPTM Peripheral Properties            */
} TIMER1_Type;


// ------------------------------------------------------------------------------------------------
// -----                                        TIMER2                                        -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for TIMER0 peripheral (TIMER2)
  */

typedef struct {                            /*!< TIMER2 Structure                      */
  __IO uint32_t CFG;                        /*!< GPTM Configuration                    */
  __IO uint32_t TAMR;                       /*!< GPTM Timer A Mode                     */
  __IO uint32_t TBMR;                       /*!< GPTM Timer B Mode                     */
  __IO uint32_t CTL;                        /*!< GPTM Control                          */
  __IO uint32_t SYNC;                       /*!< GPTM Synchronize                      */
  __I  uint32_t RESERVED0;
  __IO uint32_t IMR;                        /*!< GPTM Interrupt Mask                   */
  __IO uint32_t RIS;                        /*!< GPTM Raw Interrupt Status             */
  __IO uint32_t MIS;                        /*!< GPTM Masked Interrupt Status          */
  __O  uint32_t ICR;                        /*!< GPTM Interrupt Clear                  */
  __IO uint32_t TAILR;                      /*!< GPTM Timer A Interval Load            */
  __IO uint32_t TBILR;                      /*!< GPTM Timer B Interval Load            */
  __IO uint32_t TAMATCHR;                   /*!< GPTM Timer A Match                    */
  __IO uint32_t TBMATCHR;                   /*!< GPTM Timer B Match                    */
  __IO uint32_t TAPR;                       /*!< GPTM Timer A Prescale                 */
  __IO uint32_t TBPR;                       /*!< GPTM Timer B Prescale                 */
  __IO uint32_t TAPMR;                      /*!< GPTM TimerA Prescale Match            */
  __IO uint32_t TBPMR;                      /*!< GPTM TimerB Prescale Match            */
  __IO uint32_t TAR;                        /*!< GPTM Timer A                          */
  __IO uint32_t TBR;                        /*!< GPTM Timer B                          */
  __IO uint32_t TAV;                        /*!< GPTM Timer A Value                    */
  __IO uint32_t TBV;                        /*!< GPTM Timer B Value                    */
  __IO uint32_t RTCPD;                      /*!< GPTM RTC Predivide                    */
  __IO uint32_t TAPS;                       /*!< GPTM Timer A Prescale Snapshot        */
  __IO uint32_t TBPS;                       /*!< GPTM Timer B Prescale Snapshot        */
  __IO uint32_t TAPV;                       /*!< GPTM Timer A Prescale Value           */
  __IO uint32_t TBPV;                       /*!< GPTM Timer B Prescale Value           */
  __I  uint32_t RESERVED1[981];
  __IO uint32_t PP;                         /*!< GPTM Peripheral Properties            */
} TIMER2_Type;


// ------------------------------------------------------------------------------------------------
// -----                                        TIMER3                                        -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for TIMER0 peripheral (TIMER3)
  */

typedef struct {                            /*!< TIMER3 Structure                      */
  __IO uint32_t CFG;                        /*!< GPTM Configuration                    */
  __IO uint32_t TAMR;                       /*!< GPTM Timer A Mode                     */
  __IO uint32_t TBMR;                       /*!< GPTM Timer B Mode                     */
  __IO uint32_t CTL;                        /*!< GPTM Control                          */
  __IO uint32_t SYNC;                       /*!< GPTM Synchronize                      */
  __I  uint32_t RESERVED0;
  __IO uint32_t IMR;                        /*!< GPTM Interrupt Mask                   */
  __IO uint32_t RIS;                        /*!< GPTM Raw Interrupt Status             */
  __IO uint32_t MIS;                        /*!< GPTM Masked Interrupt Status          */
  __O  uint32_t ICR;                        /*!< GPTM Interrupt Clear                  */
  __IO uint32_t TAILR;                      /*!< GPTM Timer A Interval Load            */
  __IO uint32_t TBILR;                      /*!< GPTM Timer B Interval Load            */
  __IO uint32_t TAMATCHR;                   /*!< GPTM Timer A Match                    */
  __IO uint32_t TBMATCHR;                   /*!< GPTM Timer B Match                    */
  __IO uint32_t TAPR;                       /*!< GPTM Timer A Prescale                 */
  __IO uint32_t TBPR;                       /*!< GPTM Timer B Prescale                 */
  __IO uint32_t TAPMR;                      /*!< GPTM TimerA Prescale Match            */
  __IO uint32_t TBPMR;                      /*!< GPTM TimerB Prescale Match            */
  __IO uint32_t TAR;                        /*!< GPTM Timer A                          */
  __IO uint32_t TBR;                        /*!< GPTM Timer B                          */
  __IO uint32_t TAV;                        /*!< GPTM Timer A Value                    */
  __IO uint32_t TBV;                        /*!< GPTM Timer B Value                    */
  __IO uint32_t RTCPD;                      /*!< GPTM RTC Predivide                    */
  __IO uint32_t TAPS;                       /*!< GPTM Timer A Prescale Snapshot        */
  __IO uint32_t TBPS;                       /*!< GPTM Timer B Prescale Snapshot        */
  __IO uint32_t TAPV;                       /*!< GPTM Timer A Prescale Value           */
  __IO uint32_t TBPV;                       /*!< GPTM Timer B Prescale Value           */
  __I  uint32_t RESERVED1[981];
  __IO uint32_t PP;                         /*!< GPTM Peripheral Properties            */
} TIMER3_Type;


// ------------------------------------------------------------------------------------------------
// -----                                        TIMER4                                        -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for TIMER0 peripheral (TIMER4)
  */

typedef struct {                            /*!< TIMER4 Structure                      */
  __IO uint32_t CFG;                        /*!< GPTM Configuration                    */
  __IO uint32_t TAMR;                       /*!< GPTM Timer A Mode                     */
  __IO uint32_t TBMR;                       /*!< GPTM Timer B Mode                     */
  __IO uint32_t CTL;                        /*!< GPTM Control                          */
  __IO uint32_t SYNC;                       /*!< GPTM Synchronize                      */
  __I  uint32_t RESERVED0;
  __IO uint32_t IMR;                        /*!< GPTM Interrupt Mask                   */
  __IO uint32_t RIS;                        /*!< GPTM Raw Interrupt Status             */
  __IO uint32_t MIS;                        /*!< GPTM Masked Interrupt Status          */
  __O  uint32_t ICR;                        /*!< GPTM Interrupt Clear                  */
  __IO uint32_t TAILR;                      /*!< GPTM Timer A Interval Load            */
  __IO uint32_t TBILR;                      /*!< GPTM Timer B Interval Load            */
  __IO uint32_t TAMATCHR;                   /*!< GPTM Timer A Match                    */
  __IO uint32_t TBMATCHR;                   /*!< GPTM Timer B Match                    */
  __IO uint32_t TAPR;                       /*!< GPTM Timer A Prescale                 */
  __IO uint32_t TBPR;                       /*!< GPTM Timer B Prescale                 */
  __IO uint32_t TAPMR;                      /*!< GPTM TimerA Prescale Match            */
  __IO uint32_t TBPMR;                      /*!< GPTM TimerB Prescale Match            */
  __IO uint32_t TAR;                        /*!< GPTM Timer A                          */
  __IO uint32_t TBR;                        /*!< GPTM Timer B                          */
  __IO uint32_t TAV;                        /*!< GPTM Timer A Value                    */
  __IO uint32_t TBV;                        /*!< GPTM Timer B Value                    */
  __IO uint32_t RTCPD;                      /*!< GPTM RTC Predivide                    */
  __IO uint32_t TAPS;                       /*!< GPTM Timer A Prescale Snapshot        */
  __IO uint32_t TBPS;                       /*!< GPTM Timer B Prescale Snapshot        */
  __IO uint32_t TAPV;                       /*!< GPTM Timer A Prescale Value           */
  __IO uint32_t TBPV;                       /*!< GPTM Timer B Prescale Value           */
  __I  uint32_t RESERVED1[981];
  __IO uint32_t PP;                         /*!< GPTM Peripheral Properties            */
} TIMER4_Type;


// ------------------------------------------------------------------------------------------------
// -----                                        TIMER5                                        -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for TIMER0 peripheral (TIMER5)
  */

typedef struct {                            /*!< TIMER5 Structure                      */
  __IO uint32_t CFG;                        /*!< GPTM Configuration                    */
  __IO uint32_t TAMR;                       /*!< GPTM Timer A Mode                     */
  __IO uint32_t TBMR;                       /*!< GPTM Timer B Mode                     */
  __IO uint32_t CTL;                        /*!< GPTM Control                          */
  __IO uint32_t SYNC;                       /*!< GPTM Synchronize                      */
  __I  uint32_t RESERVED0;
  __IO uint32_t IMR;                        /*!< GPTM Interrupt Mask                   */
  __IO uint32_t RIS;                        /*!< GPTM Raw Interrupt Status             */
  __IO uint32_t MIS;                        /*!< GPTM Masked Interrupt Status          */
  __O  uint32_t ICR;                        /*!< GPTM Interrupt Clear                  */
  __IO uint32_t TAILR;                      /*!< GPTM Timer A Interval Load            */
  __IO uint32_t TBILR;                      /*!< GPTM Timer B Interval Load            */
  __IO uint32_t TAMATCHR;                   /*!< GPTM Timer A Match                    */
  __IO uint32_t TBMATCHR;                   /*!< GPTM Timer B Match                    */
  __IO uint32_t TAPR;                       /*!< GPTM Timer A Prescale                 */
  __IO uint32_t TBPR;                       /*!< GPTM Timer B Prescale                 */
  __IO uint32_t TAPMR;                      /*!< GPTM TimerA Prescale Match            */
  __IO uint32_t TBPMR;                      /*!< GPTM TimerB Prescale Match            */
  __IO uint32_t TAR;                        /*!< GPTM Timer A                          */
  __IO uint32_t TBR;                        /*!< GPTM Timer B                          */
  __IO uint32_t TAV;                        /*!< GPTM Timer A Value                    */
  __IO uint32_t TBV;                        /*!< GPTM Timer B Value                    */
  __IO uint32_t RTCPD;                      /*!< GPTM RTC Predivide                    */
  __IO uint32_t TAPS;                       /*!< GPTM Timer A Prescale Snapshot        */
  __IO uint32_t TBPS;                       /*!< GPTM Timer B Prescale Snapshot        */
  __IO uint32_t TAPV;                       /*!< GPTM Timer A Prescale Value           */
  __IO uint32_t TBPV;                       /*!< GPTM Timer B Prescale Value           */
  __I  uint32_t RESERVED1[981];
  __IO uint32_t PP;                         /*!< GPTM Peripheral Properties            */
} TIMER5_Type;


// ------------------------------------------------------------------------------------------------
// -----                                        WTIMER0                                       -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for WTIMER0 peripheral (WTIMER0)
  */

typedef struct {                            /*!< WTIMER0 Structure                     */
  __IO uint32_t CFG;                        /*!< GPTM Configuration                    */
  __IO uint32_t TAMR;                       /*!< GPTM Timer A Mode                     */
  __IO uint32_t TBMR;                       /*!< GPTM Timer B Mode                     */
  __IO uint32_t CTL;                        /*!< GPTM Control                          */
  __IO uint32_t SYNC;                       /*!< GPTM Synchronize                      */
  __I  uint32_t RESERVED0;
  __IO uint32_t IMR;                        /*!< GPTM Interrupt Mask                   */
  __IO uint32_t RIS;                        /*!< GPTM Raw Interrupt Status             */
  __IO uint32_t MIS;                        /*!< GPTM Masked Interrupt Status          */
  __O  uint32_t ICR;                        /*!< GPTM Interrupt Clear                  */
  __IO uint32_t TAILR;                      /*!< GPTM Timer A Interval Load            */
  __IO uint32_t TBILR;                      /*!< GPTM Timer B Interval Load            */
  __IO uint32_t TAMATCHR;                   /*!< GPTM Timer A Match                    */
  __IO uint32_t TBMATCHR;                   /*!< GPTM Timer B Match                    */
  __IO uint32_t TAPR;                       /*!< GPTM Timer A Prescale                 */
  __IO uint32_t TBPR;                       /*!< GPTM Timer B Prescale                 */
  __IO uint32_t TAPMR;                      /*!< GPTM TimerA Prescale Match            */
  __IO uint32_t TBPMR;                      /*!< GPTM TimerB Prescale Match            */
  __IO uint32_t TAR;                        /*!< GPTM Timer A                          */
  __IO uint32_t TBR;                        /*!< GPTM Timer B                          */
  __IO uint32_t TAV;                        /*!< GPTM Timer A Value                    */
  __IO uint32_t TBV;                        /*!< GPTM Timer B Value                    */
  __IO uint32_t RTCPD;                      /*!< GPTM RTC Predivide                    */
  __IO uint32_t TAPS;                       /*!< GPTM Timer A Prescale Snapshot        */
  __IO uint32_t TBPS;                       /*!< GPTM Timer B Prescale Snapshot        */
  __IO uint32_t TAPV;                       /*!< GPTM Timer A Prescale Value           */
  __IO uint32_t TBPV;                       /*!< GPTM Timer B Prescale Value           */
  __I  uint32_t RESERVED1[981];
  __IO uint32_t PP;                         /*!< GPTM Peripheral Properties            */
} WTIMER0_Type;


// ------------------------------------------------------------------------------------------------
// -----                                        WTIMER1                                       -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for TIMER0 peripheral (WTIMER1)
  */

typedef struct {                            /*!< WTIMER1 Structure                     */
  __IO uint32_t CFG;                        /*!< GPTM Configuration                    */
  __IO uint32_t TAMR;                       /*!< GPTM Timer A Mode                     */
  __IO uint32_t TBMR;                       /*!< GPTM Timer B Mode                     */
  __IO uint32_t CTL;                        /*!< GPTM Control                          */
  __IO uint32_t SYNC;                       /*!< GPTM Synchronize                      */
  __I  uint32_t RESERVED0;
  __IO uint32_t IMR;                        /*!< GPTM Interrupt Mask                   */
  __IO uint32_t RIS;                        /*!< GPTM Raw Interrupt Status             */
  __IO uint32_t MIS;                        /*!< GPTM Masked Interrupt Status          */
  __O  uint32_t ICR;                        /*!< GPTM Interrupt Clear                  */
  __IO uint32_t TAILR;                      /*!< GPTM Timer A Interval Load            */
  __IO uint32_t TBILR;                      /*!< GPTM Timer B Interval Load            */
  __IO uint32_t TAMATCHR;                   /*!< GPTM Timer A Match                    */
  __IO uint32_t TBMATCHR;                   /*!< GPTM Timer B Match                    */
  __IO uint32_t TAPR;                       /*!< GPTM Timer A Prescale                 */
  __IO uint32_t TBPR;                       /*!< GPTM Timer B Prescale                 */
  __IO uint32_t TAPMR;                      /*!< GPTM TimerA Prescale Match            */
  __IO uint32_t TBPMR;                      /*!< GPTM TimerB Prescale Match            */
  __IO uint32_t TAR;                        /*!< GPTM Timer A                          */
  __IO uint32_t TBR;                        /*!< GPTM Timer B                          */
  __IO uint32_t TAV;                        /*!< GPTM Timer A Value                    */
  __IO uint32_t TBV;                        /*!< GPTM Timer B Value                    */
  __IO uint32_t RTCPD;                      /*!< GPTM RTC Predivide                    */
  __IO uint32_t TAPS;                       /*!< GPTM Timer A Prescale Snapshot        */
  __IO uint32_t TBPS;                       /*!< GPTM Timer B Prescale Snapshot        */
  __IO uint32_t TAPV;                       /*!< GPTM Timer A Prescale Value           */
  __IO uint32_t TBPV;                       /*!< GPTM Timer B Prescale Value           */
  __I  uint32_t RESERVED1[981];
  __IO uint32_t PP;                         /*!< GPTM Peripheral Properties            */
} WTIMER1_Type;


// ------------------------------------------------------------------------------------------------
// -----                                         ADC0                                         -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for ADC0 peripheral (ADC0)
  */

typedef struct {                            /*!< ADC0 Structure                        */
  __IO uint32_t ACTSS;                      /*!< ADC Active Sample Sequencer           */
  __IO uint32_t RIS;                        /*!< ADC Raw Interrupt Status              */
  __IO uint32_t IM;                         /*!< ADC Interrupt Mask                    */
  __IO uint32_t ISC;                        /*!< ADC Interrupt Status and Clear        */
  __IO uint32_t OSTAT;                      /*!< ADC Overflow Status                   */
  __IO uint32_t EMUX;                       /*!< ADC Event Multiplexer Select          */
  __IO uint32_t USTAT;                      /*!< ADC Underflow Status                  */
  __IO uint32_t TSSEL;                      /*!< ADC Trigger Source Select             */
  __IO uint32_t SSPRI;                      /*!< ADC Sample Sequencer Priority         */
  __IO uint32_t SPC;                        /*!< ADC Sample Phase Control              */
  __IO uint32_t PSSI;                       /*!< ADC Processor Sample Sequence Initiate */
  __I  uint32_t RESERVED0;
  __IO uint32_t SAC;                        /*!< ADC Sample Averaging Control          */
  __IO uint32_t DCISC;                      /*!< ADC Digital Comparator Interrupt Status and Clear */
  __I  uint32_t RESERVED1[2];
  __IO uint32_t SSMUX0;                     /*!< ADC Sample Sequence Input Multiplexer Select 0 */
  __IO uint32_t SSCTL0;                     /*!< ADC Sample Sequence Control 0         */
  __IO uint32_t SSFIFO0;                    /*!< ADC Sample Sequence Result FIFO 0     */
  __IO uint32_t SSFSTAT0;                   /*!< ADC Sample Sequence FIFO 0 Status     */
  __IO uint32_t SSOP0;                      /*!< ADC Sample Sequence 0 Operation       */
  __IO uint32_t SSDC0;                      /*!< ADC Sample Sequence 0 Digital Comparator Select */
  __I  uint32_t RESERVED2[2];
  __IO uint32_t SSMUX1;                     /*!< ADC Sample Sequence Input Multiplexer Select 1 */
  __IO uint32_t SSCTL1;                     /*!< ADC Sample Sequence Control 1         */
  __IO uint32_t SSFIFO1;                    /*!< ADC Sample Sequence Result FIFO 1     */
  __IO uint32_t SSFSTAT1;                   /*!< ADC Sample Sequence FIFO 1 Status     */
  __IO uint32_t SSOP1;                      /*!< ADC Sample Sequence 1 Operation       */
  __IO uint32_t SSDC1;                      /*!< ADC Sample Sequence 1 Digital Comparator Select */
  __I  uint32_t RESERVED3[2];
  __IO uint32_t SSMUX2;                     /*!< ADC Sample Sequence Input Multiplexer Select 2 */
  __IO uint32_t SSCTL2;                     /*!< ADC Sample Sequence Control 2         */
  __IO uint32_t SSFIFO2;                    /*!< ADC Sample Sequence Result FIFO 2     */
  __IO uint32_t SSFSTAT2;                   /*!< ADC Sample Sequence FIFO 2 Status     */
  __IO uint32_t SSOP2;                      /*!< ADC Sample Sequence 2 Operation       */
  __IO uint32_t SSDC2;                      /*!< ADC Sample Sequence 2 Digital Comparator Select */
  __I  uint32_t RESERVED4[2];
  __IO uint32_t SSMUX3;                     /*!< ADC Sample Sequence Input Multiplexer Select 3 */
  __IO uint32_t SSCTL3;                     /*!< ADC Sample Sequence Control 3         */
  __IO uint32_t SSFIFO3;                    /*!< ADC Sample Sequence Result FIFO 3     */
  __IO uint32_t SSFSTAT3;                   /*!< ADC Sample Sequence FIFO 3 Status     */
  __IO uint32_t SSOP3;                      /*!< ADC Sample Sequence 3 Operation       */
  __IO uint32_t SSDC3;                      /*!< ADC Sample Sequence 3 Digital Comparator Select */
  __I  uint32_t RESERVED5[786];
  __IO uint32_t DCRIC;                      /*!< ADC Digital Comparator Reset Initial Conditions */
  __I  uint32_t RESERVED6[63];
  __IO uint32_t DCCTL0;                     /*!< ADC Digital Comparator Control 0      */
  __IO uint32_t DCCTL1;                     /*!< ADC Digital Comparator Control 1      */
  __IO uint32_t DCCTL2;                     /*!< ADC Digital Comparator Control 2      */
  __IO uint32_t DCCTL3;                     /*!< ADC Digital Comparator Control 3      */
  __IO uint32_t DCCTL4;                     /*!< ADC Digital Comparator Control 4      */
  __IO uint32_t DCCTL5;                     /*!< ADC Digital Comparator Control 5      */
  __IO uint32_t DCCTL6;                     /*!< ADC Digital Comparator Control 6      */
  __IO uint32_t DCCTL7;                     /*!< ADC Digital Comparator Control 7      */
  __I  uint32_t RESERVED7[8];
  __IO uint32_t DCCMP0;                     /*!< ADC Digital Comparator Range 0        */
  __IO uint32_t DCCMP1;                     /*!< ADC Digital Comparator Range 1        */
  __IO uint32_t DCCMP2;                     /*!< ADC Digital Comparator Range 2        */
  __IO uint32_t DCCMP3;                     /*!< ADC Digital Comparator Range 3        */
  __IO uint32_t DCCMP4;                     /*!< ADC Digital Comparator Range 4        */
  __IO uint32_t DCCMP5;                     /*!< ADC Digital Comparator Range 5        */
  __IO uint32_t DCCMP6;                     /*!< ADC Digital Comparator Range 6        */
  __IO uint32_t DCCMP7;                     /*!< ADC Digital Comparator Range 7        */
  __I  uint32_t RESERVED8[88];
  __IO uint32_t PP;                         /*!< ADC Peripheral Properties             */
  __IO uint32_t PC;                         /*!< ADC Peripheral Configuration          */
  __IO uint32_t CC;                         /*!< ADC Clock Configuration               */
} ADC0_Type;


// ------------------------------------------------------------------------------------------------
// -----                                         ADC1                                         -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for ADC0 peripheral (ADC1)
  */

typedef struct {                            /*!< ADC1 Structure                        */
  __IO uint32_t ACTSS;                      /*!< ADC Active Sample Sequencer           */
  __IO uint32_t RIS;                        /*!< ADC Raw Interrupt Status              */
  __IO uint32_t IM;                         /*!< ADC Interrupt Mask                    */
  __IO uint32_t ISC;                        /*!< ADC Interrupt Status and Clear        */
  __IO uint32_t OSTAT;                      /*!< ADC Overflow Status                   */
  __IO uint32_t EMUX;                       /*!< ADC Event Multiplexer Select          */
  __IO uint32_t USTAT;                      /*!< ADC Underflow Status                  */
  __IO uint32_t TSSEL;                      /*!< ADC Trigger Source Select             */
  __IO uint32_t SSPRI;                      /*!< ADC Sample Sequencer Priority         */
  __IO uint32_t SPC;                        /*!< ADC Sample Phase Control              */
  __IO uint32_t PSSI;                       /*!< ADC Processor Sample Sequence Initiate */
  __I  uint32_t RESERVED0;
  __IO uint32_t SAC;                        /*!< ADC Sample Averaging Control          */
  __IO uint32_t DCISC;                      /*!< ADC Digital Comparator Interrupt Status and Clear */
  __I  uint32_t RESERVED1[2];
  __IO uint32_t SSMUX0;                     /*!< ADC Sample Sequence Input Multiplexer Select 0 */
  __IO uint32_t SSCTL0;                     /*!< ADC Sample Sequence Control 0         */
  __IO uint32_t SSFIFO0;                    /*!< ADC Sample Sequence Result FIFO 0     */
  __IO uint32_t SSFSTAT0;                   /*!< ADC Sample Sequence FIFO 0 Status     */
  __IO uint32_t SSOP0;                      /*!< ADC Sample Sequence 0 Operation       */
  __IO uint32_t SSDC0;                      /*!< ADC Sample Sequence 0 Digital Comparator Select */
  __I  uint32_t RESERVED2[2];
  __IO uint32_t SSMUX1;                     /*!< ADC Sample Sequence Input Multiplexer Select 1 */
  __IO uint32_t SSCTL1;                     /*!< ADC Sample Sequence Control 1         */
  __IO uint32_t SSFIFO1;                    /*!< ADC Sample Sequence Result FIFO 1     */
  __IO uint32_t SSFSTAT1;                   /*!< ADC Sample Sequence FIFO 1 Status     */
  __IO uint32_t SSOP1;                      /*!< ADC Sample Sequence 1 Operation       */
  __IO uint32_t SSDC1;                      /*!< ADC Sample Sequence 1 Digital Comparator Select */
  __I  uint32_t RESERVED3[2];
  __IO uint32_t SSMUX2;                     /*!< ADC Sample Sequence Input Multiplexer Select 2 */
  __IO uint32_t SSCTL2;                     /*!< ADC Sample Sequence Control 2         */
  __IO uint32_t SSFIFO2;                    /*!< ADC Sample Sequence Result FIFO 2     */
  __IO uint32_t SSFSTAT2;                   /*!< ADC Sample Sequence FIFO 2 Status     */
  __IO uint32_t SSOP2;                      /*!< ADC Sample Sequence 2 Operation       */
  __IO uint32_t SSDC2;                      /*!< ADC Sample Sequence 2 Digital Comparator Select */
  __I  uint32_t RESERVED4[2];
  __IO uint32_t SSMUX3;                     /*!< ADC Sample Sequence Input Multiplexer Select 3 */
  __IO uint32_t SSCTL3;                     /*!< ADC Sample Sequence Control 3         */
  __IO uint32_t SSFIFO3;                    /*!< ADC Sample Sequence Result FIFO 3     */
  __IO uint32_t SSFSTAT3;                   /*!< ADC Sample Sequence FIFO 3 Status     */
  __IO uint32_t SSOP3;                      /*!< ADC Sample Sequence 3 Operation       */
  __IO uint32_t SSDC3;                      /*!< ADC Sample Sequence 3 Digital Comparator Select */
  __I  uint32_t RESERVED5[786];
  __IO uint32_t DCRIC;                      /*!< ADC Digital Comparator Reset Initial Conditions */
  __I  uint32_t RESERVED6[63];
  __IO uint32_t DCCTL0;                     /*!< ADC Digital Comparator Control 0      */
  __IO uint32_t DCCTL1;                     /*!< ADC Digital Comparator Control 1      */
  __IO uint32_t DCCTL2;                     /*!< ADC Digital Comparator Control 2      */
  __IO uint32_t DCCTL3;                     /*!< ADC Digital Comparator Control 3      */
  __IO uint32_t DCCTL4;                     /*!< ADC Digital Comparator Control 4      */
  __IO uint32_t DCCTL5;                     /*!< ADC Digital Comparator Control 5      */
  __IO uint32_t DCCTL6;                     /*!< ADC Digital Comparator Control 6      */
  __IO uint32_t DCCTL7;                     /*!< ADC Digital Comparator Control 7      */
  __I  uint32_t RESERVED7[8];
  __IO uint32_t DCCMP0;                     /*!< ADC Digital Comparator Range 0        */
  __IO uint32_t DCCMP1;                     /*!< ADC Digital Comparator Range 1        */
  __IO uint32_t DCCMP2;                     /*!< ADC Digital Comparator Range 2        */
  __IO uint32_t DCCMP3;                     /*!< ADC Digital Comparator Range 3        */
  __IO uint32_t DCCMP4;                     /*!< ADC Digital Comparator Range 4        */
  __IO uint32_t DCCMP5;                     /*!< ADC Digital Comparator Range 5        */
  __IO uint32_t DCCMP6;                     /*!< ADC Digital Comparator Range 6        */
  __IO uint32_t DCCMP7;                     /*!< ADC Digital Comparator Range 7        */
  __I  uint32_t RESERVED8[88];
  __IO uint32_t PP;                         /*!< ADC Peripheral Properties             */
  __IO uint32_t PC;                         /*!< ADC Peripheral Configuration          */
  __IO uint32_t CC;                         /*!< ADC Clock Configuration               */
} ADC1_Type;


// ------------------------------------------------------------------------------------------------
// -----                                         COMP                                         -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for COMP peripheral (COMP)
  */

typedef struct {                            /*!< COMP Structure                        */
  __IO uint32_t ACMIS;                      /*!< Analog Comparator Masked Interrupt Status */
  __IO uint32_t ACRIS;                      /*!< Analog Comparator Raw Interrupt Status */
  __IO uint32_t ACINTEN;                    /*!< Analog Comparator Interrupt Enable    */
  __I  uint32_t RESERVED0;
  __IO uint32_t ACREFCTL;                   /*!< Analog Comparator Reference Voltage Control */
  __I  uint32_t RESERVED1[3];
  __IO uint32_t ACSTAT0;                    /*!< Analog Comparator Status 0            */
  __IO uint32_t ACCTL0;                     /*!< Analog Comparator Control 0           */
  __I  uint32_t RESERVED2[6];
  __IO uint32_t ACSTAT1;                    /*!< Analog Comparator Status 1            */
  __IO uint32_t ACCTL1;                     /*!< Analog Comparator Control 1           */
  __I  uint32_t RESERVED3[990];
  __IO uint32_t PP;                         /*!< Analog Comparator Peripheral Properties */
} COMP_Type;


// ------------------------------------------------------------------------------------------------
// -----                                         CAN0                                         -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for CAN0 peripheral (CAN0)
  */

typedef struct {                            /*!< CAN0 Structure                        */
  __IO uint32_t CTL;                        /*!< CAN Control                           */
  __IO uint32_t STS;                        /*!< CAN Status                            */
  __IO uint32_t ERR;                        /*!< CAN Error Counter                     */
  __IO uint32_t BIT;                        /*!< CAN Bit Timing                        */
  __IO uint32_t INT;                        /*!< CAN Interrupt                         */
  __IO uint32_t TST;                        /*!< CAN Test                              */
  __IO uint32_t BRPE;                       /*!< CAN Baud Rate Prescaler Extension     */
  __I  uint32_t RESERVED0;
  __IO uint32_t IF1CRQ;                     /*!< CAN IF1 Command Request               */
  
  union {
    __IO uint32_t IF1CMSK;                  /*!< CAN IF1 Command Mask                  */
    __IO uint32_t IF1CMSK;                  /*!< CAN IF1 Command Mask                  */
  } ;
  __IO uint32_t IF1MSK1;                    /*!< CAN IF1 Mask 1                        */
  __IO uint32_t IF1MSK2;                    /*!< CAN IF1 Mask 2                        */
  __IO uint32_t IF1ARB1;                    /*!< CAN IF1 Arbitration 1                 */
  __IO uint32_t IF1ARB2;                    /*!< CAN IF1 Arbitration 2                 */
  __IO uint32_t IF1MCTL;                    /*!< CAN IF1 Message Control               */
  __IO uint32_t IF1DA1;                     /*!< CAN IF1 Data A1                       */
  __IO uint32_t IF1DA2;                     /*!< CAN IF1 Data A2                       */
  __IO uint32_t IF1DB1;                     /*!< CAN IF1 Data B1                       */
  __IO uint32_t IF1DB2;                     /*!< CAN IF1 Data B2                       */
  __I  uint32_t RESERVED1[13];
  __IO uint32_t IF2CRQ;                     /*!< CAN IF2 Command Request               */
  
  union {
    __IO uint32_t IF2CMSK;                  /*!< CAN IF2 Command Mask                  */
    __IO uint32_t IF2CMSK;                  /*!< CAN IF2 Command Mask                  */
  } ;
  __IO uint32_t IF2MSK1;                    /*!< CAN IF2 Mask 1                        */
  __IO uint32_t IF2MSK2;                    /*!< CAN IF2 Mask 2                        */
  __IO uint32_t IF2ARB1;                    /*!< CAN IF2 Arbitration 1                 */
  __IO uint32_t IF2ARB2;                    /*!< CAN IF2 Arbitration 2                 */
  __IO uint32_t IF2MCTL;                    /*!< CAN IF2 Message Control               */
  __IO uint32_t IF2DA1;                     /*!< CAN IF2 Data A1                       */
  __IO uint32_t IF2DA2;                     /*!< CAN IF2 Data A2                       */
  __IO uint32_t IF2DB1;                     /*!< CAN IF2 Data B1                       */
  __IO uint32_t IF2DB2;                     /*!< CAN IF2 Data B2                       */
  __I  uint32_t RESERVED2[21];
  __IO uint32_t TXRQ1;                      /*!< CAN Transmission Request 1            */
  __IO uint32_t TXRQ2;                      /*!< CAN Transmission Request 2            */
  __I  uint32_t RESERVED3[6];
  __IO uint32_t NWDA1;                      /*!< CAN New Data 1                        */
  __IO uint32_t NWDA2;                      /*!< CAN New Data 2                        */
  __I  uint32_t RESERVED4[6];
  __IO uint32_t MSG1INT;                    /*!< CAN Message 1 Interrupt Pending       */
  __IO uint32_t MSG2INT;                    /*!< CAN Message 2 Interrupt Pending       */
  __I  uint32_t RESERVED5[6];
  __IO uint32_t MSG1VAL;                    /*!< CAN Message 1 Valid                   */
  __IO uint32_t MSG2VAL;                    /*!< CAN Message 2 Valid                   */
} CAN0_Type;


// ------------------------------------------------------------------------------------------------
// -----                                         CAN1                                         -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for CAN0 peripheral (CAN1)
  */

typedef struct {                            /*!< CAN1 Structure                        */
  __IO uint32_t CTL;                        /*!< CAN Control                           */
  __IO uint32_t STS;                        /*!< CAN Status                            */
  __IO uint32_t ERR;                        /*!< CAN Error Counter                     */
  __IO uint32_t BIT;                        /*!< CAN Bit Timing                        */
  __IO uint32_t INT;                        /*!< CAN Interrupt                         */
  __IO uint32_t TST;                        /*!< CAN Test                              */
  __IO uint32_t BRPE;                       /*!< CAN Baud Rate Prescaler Extension     */
  __I  uint32_t RESERVED0;
  __IO uint32_t IF1CRQ;                     /*!< CAN IF1 Command Request               */
  
  union {
    __IO uint32_t IF1CMSK;                  /*!< CAN IF1 Command Mask                  */
    __IO uint32_t IF1CMSK;                  /*!< CAN IF1 Command Mask                  */
  } ;
  __IO uint32_t IF1MSK1;                    /*!< CAN IF1 Mask 1                        */
  __IO uint32_t IF1MSK2;                    /*!< CAN IF1 Mask 2                        */
  __IO uint32_t IF1ARB1;                    /*!< CAN IF1 Arbitration 1                 */
  __IO uint32_t IF1ARB2;                    /*!< CAN IF1 Arbitration 2                 */
  __IO uint32_t IF1MCTL;                    /*!< CAN IF1 Message Control               */
  __IO uint32_t IF1DA1;                     /*!< CAN IF1 Data A1                       */
  __IO uint32_t IF1DA2;                     /*!< CAN IF1 Data A2                       */
  __IO uint32_t IF1DB1;                     /*!< CAN IF1 Data B1                       */
  __IO uint32_t IF1DB2;                     /*!< CAN IF1 Data B2                       */
  __I  uint32_t RESERVED1[13];
  __IO uint32_t IF2CRQ;                     /*!< CAN IF2 Command Request               */
  
  union {
    __IO uint32_t IF2CMSK;                  /*!< CAN IF2 Command Mask                  */
    __IO uint32_t IF2CMSK;                  /*!< CAN IF2 Command Mask                  */
  } ;
  __IO uint32_t IF2MSK1;                    /*!< CAN IF2 Mask 1                        */
  __IO uint32_t IF2MSK2;                    /*!< CAN IF2 Mask 2                        */
  __IO uint32_t IF2ARB1;                    /*!< CAN IF2 Arbitration 1                 */
  __IO uint32_t IF2ARB2;                    /*!< CAN IF2 Arbitration 2                 */
  __IO uint32_t IF2MCTL;                    /*!< CAN IF2 Message Control               */
  __IO uint32_t IF2DA1;                     /*!< CAN IF2 Data A1                       */
  __IO uint32_t IF2DA2;                     /*!< CAN IF2 Data A2                       */
  __IO uint32_t IF2DB1;                     /*!< CAN IF2 Data B1                       */
  __IO uint32_t IF2DB2;                     /*!< CAN IF2 Data B2                       */
  __I  uint32_t RESERVED2[21];
  __IO uint32_t TXRQ1;                      /*!< CAN Transmission Request 1            */
  __IO uint32_t TXRQ2;                      /*!< CAN Transmission Request 2            */
  __I  uint32_t RESERVED3[6];
  __IO uint32_t NWDA1;                      /*!< CAN New Data 1                        */
  __IO uint32_t NWDA2;                      /*!< CAN New Data 2                        */
  __I  uint32_t RESERVED4[6];
  __IO uint32_t MSG1INT;                    /*!< CAN Message 1 Interrupt Pending       */
  __IO uint32_t MSG2INT;                    /*!< CAN Message 2 Interrupt Pending       */
  __I  uint32_t RESERVED5[6];
  __IO uint32_t MSG1VAL;                    /*!< CAN Message 1 Valid                   */
  __IO uint32_t MSG2VAL;                    /*!< CAN Message 2 Valid                   */
} CAN1_Type;


// ------------------------------------------------------------------------------------------------
// -----                                        WTIMER2                                       -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for TIMER0 peripheral (WTIMER2)
  */

typedef struct {                            /*!< WTIMER2 Structure                     */
  __IO uint32_t CFG;                        /*!< GPTM Configuration                    */
  __IO uint32_t TAMR;                       /*!< GPTM Timer A Mode                     */
  __IO uint32_t TBMR;                       /*!< GPTM Timer B Mode                     */
  __IO uint32_t CTL;                        /*!< GPTM Control                          */
  __IO uint32_t SYNC;                       /*!< GPTM Synchronize                      */
  __I  uint32_t RESERVED0;
  __IO uint32_t IMR;                        /*!< GPTM Interrupt Mask                   */
  __IO uint32_t RIS;                        /*!< GPTM Raw Interrupt Status             */
  __IO uint32_t MIS;                        /*!< GPTM Masked Interrupt Status          */
  __O  uint32_t ICR;                        /*!< GPTM Interrupt Clear                  */
  __IO uint32_t TAILR;                      /*!< GPTM Timer A Interval Load            */
  __IO uint32_t TBILR;                      /*!< GPTM Timer B Interval Load            */
  __IO uint32_t TAMATCHR;                   /*!< GPTM Timer A Match                    */
  __IO uint32_t TBMATCHR;                   /*!< GPTM Timer B Match                    */
  __IO uint32_t TAPR;                       /*!< GPTM Timer A Prescale                 */
  __IO uint32_t TBPR;                       /*!< GPTM Timer B Prescale                 */
  __IO uint32_t TAPMR;                      /*!< GPTM TimerA Prescale Match            */
  __IO uint32_t TBPMR;                      /*!< GPTM TimerB Prescale Match            */
  __IO uint32_t TAR;                        /*!< GPTM Timer A                          */
  __IO uint32_t TBR;                        /*!< GPTM Timer B                          */
  __IO uint32_t TAV;                        /*!< GPTM Timer A Value                    */
  __IO uint32_t TBV;                        /*!< GPTM Timer B Value                    */
  __IO uint32_t RTCPD;                      /*!< GPTM RTC Predivide                    */
  __IO uint32_t TAPS;                       /*!< GPTM Timer A Prescale Snapshot        */
  __IO uint32_t TBPS;                       /*!< GPTM Timer B Prescale Snapshot        */
  __IO uint32_t TAPV;                       /*!< GPTM Timer A Prescale Value           */
  __IO uint32_t TBPV;                       /*!< GPTM Timer B Prescale Value           */
  __I  uint32_t RESERVED1[981];
  __IO uint32_t PP;                         /*!< GPTM Peripheral Properties            */
} WTIMER2_Type;


// ------------------------------------------------------------------------------------------------
// -----                                        WTIMER3                                       -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for TIMER0 peripheral (WTIMER3)
  */

typedef struct {                            /*!< WTIMER3 Structure                     */
  __IO uint32_t CFG;                        /*!< GPTM Configuration                    */
  __IO uint32_t TAMR;                       /*!< GPTM Timer A Mode                     */
  __IO uint32_t TBMR;                       /*!< GPTM Timer B Mode                     */
  __IO uint32_t CTL;                        /*!< GPTM Control                          */
  __IO uint32_t SYNC;                       /*!< GPTM Synchronize                      */
  __I  uint32_t RESERVED0;
  __IO uint32_t IMR;                        /*!< GPTM Interrupt Mask                   */
  __IO uint32_t RIS;                        /*!< GPTM Raw Interrupt Status             */
  __IO uint32_t MIS;                        /*!< GPTM Masked Interrupt Status          */
  __O  uint32_t ICR;                        /*!< GPTM Interrupt Clear                  */
  __IO uint32_t TAILR;                      /*!< GPTM Timer A Interval Load            */
  __IO uint32_t TBILR;                      /*!< GPTM Timer B Interval Load            */
  __IO uint32_t TAMATCHR;                   /*!< GPTM Timer A Match                    */
  __IO uint32_t TBMATCHR;                   /*!< GPTM Timer B Match                    */
  __IO uint32_t TAPR;                       /*!< GPTM Timer A Prescale                 */
  __IO uint32_t TBPR;                       /*!< GPTM Timer B Prescale                 */
  __IO uint32_t TAPMR;                      /*!< GPTM TimerA Prescale Match            */
  __IO uint32_t TBPMR;                      /*!< GPTM TimerB Prescale Match            */
  __IO uint32_t TAR;                        /*!< GPTM Timer A                          */
  __IO uint32_t TBR;                        /*!< GPTM Timer B                          */
  __IO uint32_t TAV;                        /*!< GPTM Timer A Value                    */
  __IO uint32_t TBV;                        /*!< GPTM Timer B Value                    */
  __IO uint32_t RTCPD;                      /*!< GPTM RTC Predivide                    */
  __IO uint32_t TAPS;                       /*!< GPTM Timer A Prescale Snapshot        */
  __IO uint32_t TBPS;                       /*!< GPTM Timer B Prescale Snapshot        */
  __IO uint32_t TAPV;                       /*!< GPTM Timer A Prescale Value           */
  __IO uint32_t TBPV;                       /*!< GPTM Timer B Prescale Value           */
  __I  uint32_t RESERVED1[981];
  __IO uint32_t PP;                         /*!< GPTM Peripheral Properties            */
} WTIMER3_Type;


// ------------------------------------------------------------------------------------------------
// -----                                        WTIMER4                                       -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for TIMER0 peripheral (WTIMER4)
  */

typedef struct {                            /*!< WTIMER4 Structure                     */
  __IO uint32_t CFG;                        /*!< GPTM Configuration                    */
  __IO uint32_t TAMR;                       /*!< GPTM Timer A Mode                     */
  __IO uint32_t TBMR;                       /*!< GPTM Timer B Mode                     */
  __IO uint32_t CTL;                        /*!< GPTM Control                          */
  __IO uint32_t SYNC;                       /*!< GPTM Synchronize                      */
  __I  uint32_t RESERVED0;
  __IO uint32_t IMR;                        /*!< GPTM Interrupt Mask                   */
  __IO uint32_t RIS;                        /*!< GPTM Raw Interrupt Status             */
  __IO uint32_t MIS;                        /*!< GPTM Masked Interrupt Status          */
  __O  uint32_t ICR;                        /*!< GPTM Interrupt Clear                  */
  __IO uint32_t TAILR;                      /*!< GPTM Timer A Interval Load            */
  __IO uint32_t TBILR;                      /*!< GPTM Timer B Interval Load            */
  __IO uint32_t TAMATCHR;                   /*!< GPTM Timer A Match                    */
  __IO uint32_t TBMATCHR;                   /*!< GPTM Timer B Match                    */
  __IO uint32_t TAPR;                       /*!< GPTM Timer A Prescale                 */
  __IO uint32_t TBPR;                       /*!< GPTM Timer B Prescale                 */
  __IO uint32_t TAPMR;                      /*!< GPTM TimerA Prescale Match            */
  __IO uint32_t TBPMR;                      /*!< GPTM TimerB Prescale Match            */
  __IO uint32_t TAR;                        /*!< GPTM Timer A                          */
  __IO uint32_t TBR;                        /*!< GPTM Timer B                          */
  __IO uint32_t TAV;                        /*!< GPTM Timer A Value                    */
  __IO uint32_t TBV;                        /*!< GPTM Timer B Value                    */
  __IO uint32_t RTCPD;                      /*!< GPTM RTC Predivide                    */
  __IO uint32_t TAPS;                       /*!< GPTM Timer A Prescale Snapshot        */
  __IO uint32_t TBPS;                       /*!< GPTM Timer B Prescale Snapshot        */
  __IO uint32_t TAPV;                       /*!< GPTM Timer A Prescale Value           */
  __IO uint32_t TBPV;                       /*!< GPTM Timer B Prescale Value           */
  __I  uint32_t RESERVED1[981];
  __IO uint32_t PP;                         /*!< GPTM Peripheral Properties            */
} WTIMER4_Type;


// ------------------------------------------------------------------------------------------------
// -----                                        WTIMER5                                       -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for TIMER0 peripheral (WTIMER5)
  */

typedef struct {                            /*!< WTIMER5 Structure                     */
  __IO uint32_t CFG;                        /*!< GPTM Configuration                    */
  __IO uint32_t TAMR;                       /*!< GPTM Timer A Mode                     */
  __IO uint32_t TBMR;                       /*!< GPTM Timer B Mode                     */
  __IO uint32_t CTL;                        /*!< GPTM Control                          */
  __IO uint32_t SYNC;                       /*!< GPTM Synchronize                      */
  __I  uint32_t RESERVED0;
  __IO uint32_t IMR;                        /*!< GPTM Interrupt Mask                   */
  __IO uint32_t RIS;                        /*!< GPTM Raw Interrupt Status             */
  __IO uint32_t MIS;                        /*!< GPTM Masked Interrupt Status          */
  __O  uint32_t ICR;                        /*!< GPTM Interrupt Clear                  */
  __IO uint32_t TAILR;                      /*!< GPTM Timer A Interval Load            */
  __IO uint32_t TBILR;                      /*!< GPTM Timer B Interval Load            */
  __IO uint32_t TAMATCHR;                   /*!< GPTM Timer A Match                    */
  __IO uint32_t TBMATCHR;                   /*!< GPTM Timer B Match                    */
  __IO uint32_t TAPR;                       /*!< GPTM Timer A Prescale                 */
  __IO uint32_t TBPR;                       /*!< GPTM Timer B Prescale                 */
  __IO uint32_t TAPMR;                      /*!< GPTM TimerA Prescale Match            */
  __IO uint32_t TBPMR;                      /*!< GPTM TimerB Prescale Match            */
  __IO uint32_t TAR;                        /*!< GPTM Timer A                          */
  __IO uint32_t TBR;                        /*!< GPTM Timer B                          */
  __IO uint32_t TAV;                        /*!< GPTM Timer A Value                    */
  __IO uint32_t TBV;                        /*!< GPTM Timer B Value                    */
  __IO uint32_t RTCPD;                      /*!< GPTM RTC Predivide                    */
  __IO uint32_t TAPS;                       /*!< GPTM Timer A Prescale Snapshot        */
  __IO uint32_t TBPS;                       /*!< GPTM Timer B Prescale Snapshot        */
  __IO uint32_t TAPV;                       /*!< GPTM Timer A Prescale Value           */
  __IO uint32_t TBPV;                       /*!< GPTM Timer B Prescale Value           */
  __I  uint32_t RESERVED1[981];
  __IO uint32_t PP;                         /*!< GPTM Peripheral Properties            */
} WTIMER5_Type;


// ------------------------------------------------------------------------------------------------
// -----                                    GPIO_PORTA_AHB                                    -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for GPIO_PORTA peripheral (GPIO_PORTA_AHB)
  */

typedef struct {                            /*!< GPIO_PORTA_AHB Structure              */
  __I  uint32_t RESERVED0[255];
  __IO uint32_t DATA;                       /*!< GPIO Data                             */
  __IO uint32_t DIR;                        /*!< GPIO Direction                        */
  __IO uint32_t IS;                         /*!< GPIO Interrupt Sense                  */
  __IO uint32_t IBE;                        /*!< GPIO Interrupt Both Edges             */
  __IO uint32_t IEV;                        /*!< GPIO Interrupt Event                  */
  __IO uint32_t IM;                         /*!< GPIO Interrupt Mask                   */
  __IO uint32_t RIS;                        /*!< GPIO Raw Interrupt Status             */
  __IO uint32_t MIS;                        /*!< GPIO Masked Interrupt Status          */
  __O  uint32_t ICR;                        /*!< GPIO Interrupt Clear                  */
  __IO uint32_t AFSEL;                      /*!< GPIO Alternate Function Select        */
  __I  uint32_t RESERVED1[55];
  __IO uint32_t DR2R;                       /*!< GPIO 2-mA Drive Select                */
  __IO uint32_t DR4R;                       /*!< GPIO 4-mA Drive Select                */
  __IO uint32_t DR8R;                       /*!< GPIO 8-mA Drive Select                */
  __IO uint32_t ODR;                        /*!< GPIO Open Drain Select                */
  __IO uint32_t PUR;                        /*!< GPIO Pull-Up Select                   */
  __IO uint32_t PDR;                        /*!< GPIO Pull-Down Select                 */
  __IO uint32_t SLR;                        /*!< GPIO Slew Rate Control Select         */
  __IO uint32_t DEN;                        /*!< GPIO Digital Enable                   */
  __IO uint32_t LOCK;                       /*!< GPIO Lock                             */
  __I  uint32_t CR;                         /*!< GPIO Commit                           */
  __IO uint32_t AMSEL;                      /*!< GPIO Analog Mode Select               */
  __IO uint32_t PCTL;                       /*!< GPIO Port Control                     */
  __IO uint32_t ADCCTL;                     /*!< GPIO ADC Control                      */
  __IO uint32_t DMACTL;                     /*!< GPIO DMA Control                      */
  __IO uint32_t SI;                         /*!< GPIO Select Interrupt                 */
} GPIO_PORTA_AHB_Type;


// ------------------------------------------------------------------------------------------------
// -----                                    GPIO_PORTB_AHB                                    -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for GPIO_PORTA peripheral (GPIO_PORTB_AHB)
  */

typedef struct {                            /*!< GPIO_PORTB_AHB Structure              */
  __I  uint32_t RESERVED0[255];
  __IO uint32_t DATA;                       /*!< GPIO Data                             */
  __IO uint32_t DIR;                        /*!< GPIO Direction                        */
  __IO uint32_t IS;                         /*!< GPIO Interrupt Sense                  */
  __IO uint32_t IBE;                        /*!< GPIO Interrupt Both Edges             */
  __IO uint32_t IEV;                        /*!< GPIO Interrupt Event                  */
  __IO uint32_t IM;                         /*!< GPIO Interrupt Mask                   */
  __IO uint32_t RIS;                        /*!< GPIO Raw Interrupt Status             */
  __IO uint32_t MIS;                        /*!< GPIO Masked Interrupt Status          */
  __O  uint32_t ICR;                        /*!< GPIO Interrupt Clear                  */
  __IO uint32_t AFSEL;                      /*!< GPIO Alternate Function Select        */
  __I  uint32_t RESERVED1[55];
  __IO uint32_t DR2R;                       /*!< GPIO 2-mA Drive Select                */
  __IO uint32_t DR4R;                       /*!< GPIO 4-mA Drive Select                */
  __IO uint32_t DR8R;                       /*!< GPIO 8-mA Drive Select                */
  __IO uint32_t ODR;                        /*!< GPIO Open Drain Select                */
  __IO uint32_t PUR;                        /*!< GPIO Pull-Up Select                   */
  __IO uint32_t PDR;                        /*!< GPIO Pull-Down Select                 */
  __IO uint32_t SLR;                        /*!< GPIO Slew Rate Control Select         */
  __IO uint32_t DEN;                        /*!< GPIO Digital Enable                   */
  __IO uint32_t LOCK;                       /*!< GPIO Lock                             */
  __I  uint32_t CR;                         /*!< GPIO Commit                           */
  __IO uint32_t AMSEL;                      /*!< GPIO Analog Mode Select               */
  __IO uint32_t PCTL;                       /*!< GPIO Port Control                     */
  __IO uint32_t ADCCTL;                     /*!< GPIO ADC Control                      */
  __IO uint32_t DMACTL;                     /*!< GPIO DMA Control                      */
  __IO uint32_t SI;                         /*!< GPIO Select Interrupt                 */
} GPIO_PORTB_AHB_Type;


// ------------------------------------------------------------------------------------------------
// -----                                    GPIO_PORTC_AHB                                    -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for GPIO_PORTA peripheral (GPIO_PORTC_AHB)
  */

typedef struct {                            /*!< GPIO_PORTC_AHB Structure              */
  __I  uint32_t RESERVED0[255];
  __IO uint32_t DATA;                       /*!< GPIO Data                             */
  __IO uint32_t DIR;                        /*!< GPIO Direction                        */
  __IO uint32_t IS;                         /*!< GPIO Interrupt Sense                  */
  __IO uint32_t IBE;                        /*!< GPIO Interrupt Both Edges             */
  __IO uint32_t IEV;                        /*!< GPIO Interrupt Event                  */
  __IO uint32_t IM;                         /*!< GPIO Interrupt Mask                   */
  __IO uint32_t RIS;                        /*!< GPIO Raw Interrupt Status             */
  __IO uint32_t MIS;                        /*!< GPIO Masked Interrupt Status          */
  __O  uint32_t ICR;                        /*!< GPIO Interrupt Clear                  */
  __IO uint32_t AFSEL;                      /*!< GPIO Alternate Function Select        */
  __I  uint32_t RESERVED1[55];
  __IO uint32_t DR2R;                       /*!< GPIO 2-mA Drive Select                */
  __IO uint32_t DR4R;                       /*!< GPIO 4-mA Drive Select                */
  __IO uint32_t DR8R;                       /*!< GPIO 8-mA Drive Select                */
  __IO uint32_t ODR;                        /*!< GPIO Open Drain Select                */
  __IO uint32_t PUR;                        /*!< GPIO Pull-Up Select                   */
  __IO uint32_t PDR;                        /*!< GPIO Pull-Down Select                 */
  __IO uint32_t SLR;                        /*!< GPIO Slew Rate Control Select         */
  __IO uint32_t DEN;                        /*!< GPIO Digital Enable                   */
  __IO uint32_t LOCK;                       /*!< GPIO Lock                             */
  __I  uint32_t CR;                         /*!< GPIO Commit                           */
  __IO uint32_t AMSEL;                      /*!< GPIO Analog Mode Select               */
  __IO uint32_t PCTL;                       /*!< GPIO Port Control                     */
  __IO uint32_t ADCCTL;                     /*!< GPIO ADC Control                      */
  __IO uint32_t DMACTL;                     /*!< GPIO DMA Control                      */
  __IO uint32_t SI;                         /*!< GPIO Select Interrupt                 */
} GPIO_PORTC_AHB_Type;


// ------------------------------------------------------------------------------------------------
// -----                                    GPIO_PORTD_AHB                                    -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for GPIO_PORTA peripheral (GPIO_PORTD_AHB)
  */

typedef struct {                            /*!< GPIO_PORTD_AHB Structure              */
  __I  uint32_t RESERVED0[255];
  __IO uint32_t DATA;                       /*!< GPIO Data                             */
  __IO uint32_t DIR;                        /*!< GPIO Direction                        */
  __IO uint32_t IS;                         /*!< GPIO Interrupt Sense                  */
  __IO uint32_t IBE;                        /*!< GPIO Interrupt Both Edges             */
  __IO uint32_t IEV;                        /*!< GPIO Interrupt Event                  */
  __IO uint32_t IM;                         /*!< GPIO Interrupt Mask                   */
  __IO uint32_t RIS;                        /*!< GPIO Raw Interrupt Status             */
  __IO uint32_t MIS;                        /*!< GPIO Masked Interrupt Status          */
  __O  uint32_t ICR;                        /*!< GPIO Interrupt Clear                  */
  __IO uint32_t AFSEL;                      /*!< GPIO Alternate Function Select        */
  __I  uint32_t RESERVED1[55];
  __IO uint32_t DR2R;                       /*!< GPIO 2-mA Drive Select                */
  __IO uint32_t DR4R;                       /*!< GPIO 4-mA Drive Select                */
  __IO uint32_t DR8R;                       /*!< GPIO 8-mA Drive Select                */
  __IO uint32_t ODR;                        /*!< GPIO Open Drain Select                */
  __IO uint32_t PUR;                        /*!< GPIO Pull-Up Select                   */
  __IO uint32_t PDR;                        /*!< GPIO Pull-Down Select                 */
  __IO uint32_t SLR;                        /*!< GPIO Slew Rate Control Select         */
  __IO uint32_t DEN;                        /*!< GPIO Digital Enable                   */
  __IO uint32_t LOCK;                       /*!< GPIO Lock                             */
  __I  uint32_t CR;                         /*!< GPIO Commit                           */
  __IO uint32_t AMSEL;                      /*!< GPIO Analog Mode Select               */
  __IO uint32_t PCTL;                       /*!< GPIO Port Control                     */
  __IO uint32_t ADCCTL;                     /*!< GPIO ADC Control                      */
  __IO uint32_t DMACTL;                     /*!< GPIO DMA Control                      */
  __IO uint32_t SI;                         /*!< GPIO Select Interrupt                 */
} GPIO_PORTD_AHB_Type;


// ------------------------------------------------------------------------------------------------
// -----                                    GPIO_PORTE_AHB                                    -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for GPIO_PORTA peripheral (GPIO_PORTE_AHB)
  */

typedef struct {                            /*!< GPIO_PORTE_AHB Structure              */
  __I  uint32_t RESERVED0[255];
  __IO uint32_t DATA;                       /*!< GPIO Data                             */
  __IO uint32_t DIR;                        /*!< GPIO Direction                        */
  __IO uint32_t IS;                         /*!< GPIO Interrupt Sense                  */
  __IO uint32_t IBE;                        /*!< GPIO Interrupt Both Edges             */
  __IO uint32_t IEV;                        /*!< GPIO Interrupt Event                  */
  __IO uint32_t IM;                         /*!< GPIO Interrupt Mask                   */
  __IO uint32_t RIS;                        /*!< GPIO Raw Interrupt Status             */
  __IO uint32_t MIS;                        /*!< GPIO Masked Interrupt Status          */
  __O  uint32_t ICR;                        /*!< GPIO Interrupt Clear                  */
  __IO uint32_t AFSEL;                      /*!< GPIO Alternate Function Select        */
  __I  uint32_t RESERVED1[55];
  __IO uint32_t DR2R;                       /*!< GPIO 2-mA Drive Select                */
  __IO uint32_t DR4R;                       /*!< GPIO 4-mA Drive Select                */
  __IO uint32_t DR8R;                       /*!< GPIO 8-mA Drive Select                */
  __IO uint32_t ODR;                        /*!< GPIO Open Drain Select                */
  __IO uint32_t PUR;                        /*!< GPIO Pull-Up Select                   */
  __IO uint32_t PDR;                        /*!< GPIO Pull-Down Select                 */
  __IO uint32_t SLR;                        /*!< GPIO Slew Rate Control Select         */
  __IO uint32_t DEN;                        /*!< GPIO Digital Enable                   */
  __IO uint32_t LOCK;                       /*!< GPIO Lock                             */
  __I  uint32_t CR;                         /*!< GPIO Commit                           */
  __IO uint32_t AMSEL;                      /*!< GPIO Analog Mode Select               */
  __IO uint32_t PCTL;                       /*!< GPIO Port Control                     */
  __IO uint32_t ADCCTL;                     /*!< GPIO ADC Control                      */
  __IO uint32_t DMACTL;                     /*!< GPIO DMA Control                      */
  __IO uint32_t SI;                         /*!< GPIO Select Interrupt                 */
} GPIO_PORTE_AHB_Type;


// ------------------------------------------------------------------------------------------------
// -----                                    GPIO_PORTF_AHB                                    -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for GPIO_PORTA peripheral (GPIO_PORTF_AHB)
  */

typedef struct {                            /*!< GPIO_PORTF_AHB Structure              */
  __I  uint32_t RESERVED0[255];
  __IO uint32_t DATA;                       /*!< GPIO Data                             */
  __IO uint32_t DIR;                        /*!< GPIO Direction                        */
  __IO uint32_t IS;                         /*!< GPIO Interrupt Sense                  */
  __IO uint32_t IBE;                        /*!< GPIO Interrupt Both Edges             */
  __IO uint32_t IEV;                        /*!< GPIO Interrupt Event                  */
  __IO uint32_t IM;                         /*!< GPIO Interrupt Mask                   */
  __IO uint32_t RIS;                        /*!< GPIO Raw Interrupt Status             */
  __IO uint32_t MIS;                        /*!< GPIO Masked Interrupt Status          */
  __O  uint32_t ICR;                        /*!< GPIO Interrupt Clear                  */
  __IO uint32_t AFSEL;                      /*!< GPIO Alternate Function Select        */
  __I  uint32_t RESERVED1[55];
  __IO uint32_t DR2R;                       /*!< GPIO 2-mA Drive Select                */
  __IO uint32_t DR4R;                       /*!< GPIO 4-mA Drive Select                */
  __IO uint32_t DR8R;                       /*!< GPIO 8-mA Drive Select                */
  __IO uint32_t ODR;                        /*!< GPIO Open Drain Select                */
  __IO uint32_t PUR;                        /*!< GPIO Pull-Up Select                   */
  __IO uint32_t PDR;                        /*!< GPIO Pull-Down Select                 */
  __IO uint32_t SLR;                        /*!< GPIO Slew Rate Control Select         */
  __IO uint32_t DEN;                        /*!< GPIO Digital Enable                   */
  __IO uint32_t LOCK;                       /*!< GPIO Lock                             */
  __I  uint32_t CR;                         /*!< GPIO Commit                           */
  __IO uint32_t AMSEL;                      /*!< GPIO Analog Mode Select               */
  __IO uint32_t PCTL;                       /*!< GPIO Port Control                     */
  __IO uint32_t ADCCTL;                     /*!< GPIO ADC Control                      */
  __IO uint32_t DMACTL;                     /*!< GPIO DMA Control                      */
  __IO uint32_t SI;                         /*!< GPIO Select Interrupt                 */
} GPIO_PORTF_AHB_Type;


// ------------------------------------------------------------------------------------------------
// -----                                    GPIO_PORTG_AHB                                    -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for GPIO_PORTA peripheral (GPIO_PORTG_AHB)
  */

typedef struct {                            /*!< GPIO_PORTG_AHB Structure              */
  __I  uint32_t RESERVED0[255];
  __IO uint32_t DATA;                       /*!< GPIO Data                             */
  __IO uint32_t DIR;                        /*!< GPIO Direction                        */
  __IO uint32_t IS;                         /*!< GPIO Interrupt Sense                  */
  __IO uint32_t IBE;                        /*!< GPIO Interrupt Both Edges             */
  __IO uint32_t IEV;                        /*!< GPIO Interrupt Event                  */
  __IO uint32_t IM;                         /*!< GPIO Interrupt Mask                   */
  __IO uint32_t RIS;                        /*!< GPIO Raw Interrupt Status             */
  __IO uint32_t MIS;                        /*!< GPIO Masked Interrupt Status          */
  __O  uint32_t ICR;                        /*!< GPIO Interrupt Clear                  */
  __IO uint32_t AFSEL;                      /*!< GPIO Alternate Function Select        */
  __I  uint32_t RESERVED1[55];
  __IO uint32_t DR2R;                       /*!< GPIO 2-mA Drive Select                */
  __IO uint32_t DR4R;                       /*!< GPIO 4-mA Drive Select                */
  __IO uint32_t DR8R;                       /*!< GPIO 8-mA Drive Select                */
  __IO uint32_t ODR;                        /*!< GPIO Open Drain Select                */
  __IO uint32_t PUR;                        /*!< GPIO Pull-Up Select                   */
  __IO uint32_t PDR;                        /*!< GPIO Pull-Down Select                 */
  __IO uint32_t SLR;                        /*!< GPIO Slew Rate Control Select         */
  __IO uint32_t DEN;                        /*!< GPIO Digital Enable                   */
  __IO uint32_t LOCK;                       /*!< GPIO Lock                             */
  __I  uint32_t CR;                         /*!< GPIO Commit                           */
  __IO uint32_t AMSEL;                      /*!< GPIO Analog Mode Select               */
  __IO uint32_t PCTL;                       /*!< GPIO Port Control                     */
  __IO uint32_t ADCCTL;                     /*!< GPIO ADC Control                      */
  __IO uint32_t DMACTL;                     /*!< GPIO DMA Control                      */
  __IO uint32_t SI;                         /*!< GPIO Select Interrupt                 */
} GPIO_PORTG_AHB_Type;


// ------------------------------------------------------------------------------------------------
// -----                                         I2C4                                         -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for I2C0 peripheral (I2C4)
  */

typedef struct {                            /*!< I2C4 Structure                        */
  __IO uint32_t MSA;                        /*!< I2C Master Slave Address              */
  
  union {
    __IO uint32_t MCS;                      /*!< I2C Master Control/Status             */
    __IO uint32_t MCS;                      /*!< I2C Master Control/Status             */
  } ;
  __IO uint32_t MDR;                        /*!< I2C Master Data                       */
  __IO uint32_t MTPR;                       /*!< I2C Master Timer Period               */
  __IO uint32_t MIMR;                       /*!< I2C Master Interrupt Mask             */
  __IO uint32_t MRIS;                       /*!< I2C Master Raw Interrupt Status       */
  __IO uint32_t MMIS;                       /*!< I2C Master Masked Interrupt Status    */
  __O  uint32_t MICR;                       /*!< I2C Master Interrupt Clear            */
  __IO uint32_t MCR;                        /*!< I2C Master Configuration              */
  __IO uint32_t MCLKOCNT;                   /*!< I2C Master Clock Low Timeout Count    */
  __I  uint32_t RESERVED0;
  __IO uint32_t MBMON;                      /*!< I2C Master Bus Monitor                */
  __I  uint32_t RESERVED1[500];
  __IO uint32_t SOAR;                       /*!< I2C Slave Own Address                 */
  
  union {
    __IO uint32_t SCSR;                     /*!< I2C Slave Control/Status              */
    __IO uint32_t SCSR;                     /*!< I2C Slave Control/Status              */
  } ;
  __IO uint32_t SDR;                        /*!< I2C Slave Data                        */
  __IO uint32_t SIMR;                       /*!< I2C Slave Interrupt Mask              */
  __IO uint32_t SRIS;                       /*!< I2C Slave Raw Interrupt Status        */
  __IO uint32_t SMIS;                       /*!< I2C Slave Masked Interrupt Status     */
  __O  uint32_t SICR;                       /*!< I2C Slave Interrupt Clear             */
  __IO uint32_t SOAR2;                      /*!< I2C Slave Own Address 2               */
  __IO uint32_t SACKCTL;                    /*!< I2C ACK Control                       */
  __I  uint32_t RESERVED2[487];
  __IO uint32_t PP;                         /*!< I2C Peripheral Properties             */
} I2C4_Type;


// ------------------------------------------------------------------------------------------------
// -----                                         I2C5                                         -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for I2C0 peripheral (I2C5)
  */

typedef struct {                            /*!< I2C5 Structure                        */
  __IO uint32_t MSA;                        /*!< I2C Master Slave Address              */
  
  union {
    __IO uint32_t MCS;                      /*!< I2C Master Control/Status             */
    __IO uint32_t MCS;                      /*!< I2C Master Control/Status             */
  } ;
  __IO uint32_t MDR;                        /*!< I2C Master Data                       */
  __IO uint32_t MTPR;                       /*!< I2C Master Timer Period               */
  __IO uint32_t MIMR;                       /*!< I2C Master Interrupt Mask             */
  __IO uint32_t MRIS;                       /*!< I2C Master Raw Interrupt Status       */
  __IO uint32_t MMIS;                       /*!< I2C Master Masked Interrupt Status    */
  __O  uint32_t MICR;                       /*!< I2C Master Interrupt Clear            */
  __IO uint32_t MCR;                        /*!< I2C Master Configuration              */
  __IO uint32_t MCLKOCNT;                   /*!< I2C Master Clock Low Timeout Count    */
  __I  uint32_t RESERVED0;
  __IO uint32_t MBMON;                      /*!< I2C Master Bus Monitor                */
  __I  uint32_t RESERVED1[500];
  __IO uint32_t SOAR;                       /*!< I2C Slave Own Address                 */
  
  union {
    __IO uint32_t SCSR;                     /*!< I2C Slave Control/Status              */
    __IO uint32_t SCSR;                     /*!< I2C Slave Control/Status              */
  } ;
  __IO uint32_t SDR;                        /*!< I2C Slave Data                        */
  __IO uint32_t SIMR;                       /*!< I2C Slave Interrupt Mask              */
  __IO uint32_t SRIS;                       /*!< I2C Slave Raw Interrupt Status        */
  __IO uint32_t SMIS;                       /*!< I2C Slave Masked Interrupt Status     */
  __O  uint32_t SICR;                       /*!< I2C Slave Interrupt Clear             */
  __IO uint32_t SOAR2;                      /*!< I2C Slave Own Address 2               */
  __IO uint32_t SACKCTL;                    /*!< I2C ACK Control                       */
  __I  uint32_t RESERVED2[487];
  __IO uint32_t PP;                         /*!< I2C Peripheral Properties             */
} I2C5_Type;


// ------------------------------------------------------------------------------------------------
// -----                                      FLASH_CTRL                                      -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for FLASH_CTRL peripheral (FLASH_CTRL)
  */

typedef struct {                            /*!< FLASH_CTRL Structure                  */
  __IO uint32_t FMA;                        /*!< Flash Memory Address                  */
  __IO uint32_t FMD;                        /*!< Flash Memory Data                     */
  __IO uint32_t FMC;                        /*!< Flash Memory Control                  */
  __IO uint32_t FCRIS;                      /*!< Flash Controller Raw Interrupt Status */
  __IO uint32_t FCIM;                       /*!< Flash Controller Interrupt Mask       */
  __IO uint32_t FCMISC;                     /*!< Flash Controller Masked Interrupt Status and Clear */
  __I  uint32_t RESERVED0[2];
  __IO uint32_t FMC2;                       /*!< Flash Memory Control 2                */
  __I  uint32_t RESERVED1[3];
  __IO uint32_t FWBVAL;                     /*!< Flash Write Buffer Valid              */
  __I  uint32_t RESERVED2[51];
  __IO uint32_t FWBN;                       /*!< Flash Write Buffer n                  */
  __I  uint32_t RESERVED3[943];
  __IO uint32_t FSIZE;                      /*!< Flash Size                            */
  __IO uint32_t SSIZE;                      /*!< SRAM Size                             */
  __I  uint32_t RESERVED4;
  __IO uint32_t ROMTPSW;                    /*!< ROM Third-Party Software              */
  __I  uint32_t RESERVED5[72];
  __IO uint32_t RMCTL;                      /*!< ROM Control                           */
  __I  uint32_t RESERVED6[55];
  __IO uint32_t BOOTCFG;                    /*!< Boot Configuration                    */
  __I  uint32_t RESERVED7[3];
  __IO uint32_t USERREG0;                   /*!< User Register 0                       */
  __IO uint32_t USERREG1;                   /*!< User Register 1                       */
  __IO uint32_t USERREG2;                   /*!< User Register 2                       */
  __IO uint32_t USERREG3;                   /*!< User Register 3                       */
  __I  uint32_t RESERVED8[4];
  __IO uint32_t FMPRE0;                     /*!< Flash Memory Protection Read Enable 0 */
  __IO uint32_t FMPRE1;                     /*!< Flash Memory Protection Read Enable 1 */
  __I  uint32_t RESERVED9[126];
  __IO uint32_t FMPPE0;                     /*!< Flash Memory Protection Program Enable 0 */
  __IO uint32_t FMPPE1;                     /*!< Flash Memory Protection Program Enable 1 */
} FLASH_CTRL_Type;


// ------------------------------------------------------------------------------------------------
// -----                                        SYSCTL                                        -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for SYSCTL peripheral (SYSCTL)
  */

typedef struct {                            /*!< SYSCTL Structure                      */
  __IO uint32_t DID0;                       /*!< Device Identification 0               */
  __IO uint32_t DID1;                       /*!< Device Identification 1               */
  __IO uint32_t DC0;                        /*!< Device Capabilities 0                 */
  __I  uint32_t RESERVED0;
  __IO uint32_t DC1;                        /*!< Device Capabilities 1                 */
  __IO uint32_t DC2;                        /*!< Device Capabilities 2                 */
  __IO uint32_t DC3;                        /*!< Device Capabilities 3                 */
  __IO uint32_t DC4;                        /*!< Device Capabilities 4                 */
  __IO uint32_t DC5;                        /*!< Device Capabilities 5                 */
  __IO uint32_t DC6;                        /*!< Device Capabilities 6                 */
  __IO uint32_t DC7;                        /*!< Device Capabilities 7                 */
  __IO uint32_t DC8;                        /*!< Device Capabilities 8 ADC Channels    */
  __IO uint32_t PBORCTL;                    /*!< Brown-Out Reset Control               */
  __IO uint32_t LDOPCTL;                    /*!< LDO Power Control                     */
  __I  uint32_t RESERVED1[2];
  __IO uint32_t SRCR0;                      /*!< Software Reset Control 0              */
  __IO uint32_t SRCR1;                      /*!< Software Reset Control 1              */
  __IO uint32_t SRCR2;                      /*!< Software Reset Control 2              */
  __I  uint32_t RESERVED2;
  __IO uint32_t RIS;                        /*!< Raw Interrupt Status                  */
  __IO uint32_t IMC;                        /*!< Interrupt Mask Control                */
  __IO uint32_t MISC;                       /*!< Masked Interrupt Status and Clear     */
  __IO uint32_t RESC;                       /*!< Reset Cause                           */
  __IO uint32_t RCC;                        /*!< Run-Mode Clock Configuration          */
  __I  uint32_t RESERVED3[2];
  __IO uint32_t GPIOHBCTL;                  /*!< GPIO High-Performance Bus Control     */
  __IO uint32_t RCC2;                       /*!< Run-Mode Clock Configuration 2        */
  __I  uint32_t RESERVED4[2];
  __IO uint32_t MOSCCTL;                    /*!< Main Oscillator Control               */
  __I  uint32_t RESERVED5[32];
  __IO uint32_t RCGC0;                      /*!< Run Mode Clock Gating Control Register 0 */
  __IO uint32_t RCGC1;                      /*!< Run Mode Clock Gating Control Register 1 */
  __IO uint32_t RCGC2;                      /*!< Run Mode Clock Gating Control Register 2 */
  __I  uint32_t RESERVED6;
  __IO uint32_t SCGC0;                      /*!< Sleep Mode Clock Gating Control Register 0 */
  __IO uint32_t SCGC1;                      /*!< Sleep Mode Clock Gating Control Register 1 */
  __IO uint32_t SCGC2;                      /*!< Sleep Mode Clock Gating Control Register 2 */
  __I  uint32_t RESERVED7;
  __IO uint32_t DCGC0;                      /*!< Deep Sleep Mode Clock Gating Control Register 0 */
  __IO uint32_t DCGC1;                      /*!< Deep-Sleep Mode Clock Gating Control Register 1 */
  __IO uint32_t DCGC2;                      /*!< Deep Sleep Mode Clock Gating Control Register 2 */
  __I  uint32_t RESERVED8[6];
  __IO uint32_t DSLPCLKCFG;                 /*!< Deep Sleep Clock Configuration        */
  __I  uint32_t RESERVED9;
  __IO uint32_t SYSPROP;                    /*!< System Properties                     */
  __IO uint32_t PIOSCCAL;                   /*!< Precision Internal Oscillator Calibration */
  __I  uint32_t RESERVED10[3];
  __IO uint32_t PLLFREQ0;                   /*!< PLL Frequency 0                       */
  __IO uint32_t PLLFREQ1;                   /*!< PLL Frequency                         */
  __IO uint32_t PLLSTAT;                    /*!< PLL Status                            */
  __I  uint32_t RESERVED11[7];
  __IO uint32_t SLPPWRCFG;                  /*!< Sleep Power Configuration             */
  __IO uint32_t DSLPPWRCFG;                 /*!< Deep-Sleep Power Configuration        */
  __IO uint32_t DC9;                        /*!< Device Capabilities 9 ADC Digital Comparators */
  __I  uint32_t RESERVED12[3];
  __IO uint32_t NVMSTAT;                    /*!< Non-Volatile Memory Information       */
  __I  uint32_t RESERVED13[3];
  __IO uint32_t LDOPCAL;                    /*!< LDO Power Calibration                 */
  __IO uint32_t LDOSPCTL;                   /*!< LDO Sleep Power Control               */
  __IO uint32_t LDOSPCAL;                   /*!< LDO Sleep Power Calibration           */
  __IO uint32_t LDODPCTL;                   /*!< LDO Deep-Sleep Power Control          */
  __IO uint32_t LDODPCAL;                   /*!< LDO Deep-Sleep Power Calibration      */
  __I  uint32_t RESERVED14[2];
  __IO uint32_t SDPMST;                     /*!< Sleep / Deep-Sleep Power Mode Status  */
  __I  uint32_t RESERVED15[76];
  __IO uint32_t PPWD;                       /*!< Watchdog Timer Peripheral Present     */
  __IO uint32_t PPTIMER;                    /*!< Timer Peripheral Present              */
  __IO uint32_t PPGPIO;                     /*!< General-Purpose Input/Output Peripheral Present */
  __IO uint32_t PPDMA;                      /*!< Micro Direct Memory Access Peripheral Present */
  __I  uint32_t RESERVED16;
  __IO uint32_t PPHIB;                      /*!< Hibernation Peripheral Present        */
  __IO uint32_t PPUART;                     /*!< Universal Asynchronous Receiver/Transmitter Peripheral Present */
  __IO uint32_t PPSSI;                      /*!< Synchronous Serial Interface Peripheral Present */
  __IO uint32_t PPI2C;                      /*!< Inter-Integrated Circuit Peripheral Present */
  __I  uint32_t RESERVED17;
  __IO uint32_t PPUSB;                      /*!< Universal Serial Bus Peripheral Present */
  __I  uint32_t RESERVED18[2];
  __IO uint32_t PPCAN;                      /*!< Controller Area Network Peripheral Present */
  __IO uint32_t PPADC;                      /*!< Analog-to-Digital Converter Peripheral Present */
  __IO uint32_t PPACMP;                     /*!< Analog Comparator Peripheral Present  */
  __IO uint32_t PPPWM;                      /*!< Pulse Width Modulator Peripheral Present */
  __IO uint32_t PPQEI;                      /*!< Quadrature Encoder Interface Peripheral Present */
  __IO uint32_t PPLPC;                      /*!< Low Pin Count Interface Peripheral Present */
  __I  uint32_t RESERVED19;
  __IO uint32_t PPPECI;                     /*!< Platform Environment Control Interface Peripheral Present */
  __IO uint32_t PPFAN;                      /*!< FAN Peripheral Present                */
  __IO uint32_t PPEEPROM;                   /*!< EEPROM Peripheral Present             */
  __IO uint32_t PPWTIMER;                   /*!< Wide Timer Peripheral Present         */
  __I  uint32_t RESERVED20[104];
  __IO uint32_t SRWD;                       /*!< Watchdog Timer Software Reset         */
  __IO uint32_t SRTIMER;                    /*!< Timer Software Reset                  */
  __IO uint32_t SRGPIO;                     /*!< General-Purpose Input/Output Software Reset */
  __IO uint32_t SRDMA;                      /*!< Micro Direct Memory Access Software Reset */
  __I  uint32_t RESERVED21;
  __IO uint32_t SRHIB;                      /*!< Hibernation Software Reset            */
  __IO uint32_t SRUART;                     /*!< Universal Asynchronous Receiver/Transmitter Software Reset */
  __IO uint32_t SRSSI;                      /*!< Synchronous Serial Interface Software Reset */
  __IO uint32_t SRI2C;                      /*!< Inter-Integrated Circuit Software Reset */
  __I  uint32_t RESERVED22;
  __IO uint32_t SRUSB;                      /*!< Universal Serial Bus Software Reset   */
  __I  uint32_t RESERVED23[2];
  __IO uint32_t SRCAN;                      /*!< Controller Area Network Software Reset */
  __IO uint32_t SRADC;                      /*!< Analog-to-Digital Converter Software Reset */
  __IO uint32_t SRACMP;                     /*!< Analog Comparator Software Reset      */
  __IO uint32_t SRPWM;                      /*!< Pulse Width Modulator Software Reset  */
  __IO uint32_t SRQEI;                      /*!< Quadrature Encoder Interface Software Reset */
  __IO uint32_t SRLPC;                      /*!< Low Pin Count Interface Software Reset */
  __I  uint32_t RESERVED24;
  __IO uint32_t SRPECI;                     /*!< Platform Environment Control Interface Software Reset */
  __IO uint32_t SRFAN;                      /*!< FAN Software Reset                    */
  __IO uint32_t SREEPROM;                   /*!< EEPROM Software Reset                 */
  __IO uint32_t SRWTIMER;                   /*!< Wide Timer Software Reset             */
  __I  uint32_t RESERVED25[40];
  __IO uint32_t RCGCWD;                     /*!< Watchdog Timer Run Mode Clock Gating Control */
  __IO uint32_t RCGCTIMER;                  /*!< Timer Run Mode Clock Gating Control   */
  __IO uint32_t RCGCGPIO;                   /*!< General-Purpose Input/Output Run Mode Clock Gating Control */
  __IO uint32_t RCGCDMA;                    /*!< Micro Direct Memory Access Run Mode Clock Gating Control */
  __I  uint32_t RESERVED26;
  __IO uint32_t RCGCHIB;                    /*!< Hibernation Run Mode Clock Gating Control */
  __IO uint32_t RCGCUART;                   /*!< Universal Asynchronous Receiver/Transmitter Run Mode Clock Gating Control */
  __IO uint32_t RCGCSSI;                    /*!< Synchronous Serial Interface Run Mode Clock Gating Control */
  __IO uint32_t RCGCI2C;                    /*!< Inter-Integrated Circuit Run Mode Clock Gating Control */
  __I  uint32_t RESERVED27;
  __IO uint32_t RCGCUSB;                    /*!< Universal Serial Bus Run Mode Clock Gating Control */
  __I  uint32_t RESERVED28[2];
  __IO uint32_t RCGCCAN;                    /*!< Controller Area Network Run Mode Clock Gating Control */
  __IO uint32_t RCGCADC;                    /*!< Analog-to-Digital Converter Run Mode Clock Gating Control */
  __IO uint32_t RCGCACMP;                   /*!< Analog Comparator Run Mode Clock Gating Control */
  __IO uint32_t RCGCPWM;                    /*!< Pulse Width Modulator Run Mode Clock Gating Control */
  __IO uint32_t RCGCQEI;                    /*!< Quadrature Encoder Interface Run Mode Clock Gating Control */
  __IO uint32_t RCGCLPC;                    /*!< Low Pin Count Interface Run Mode Clock Gating Control */
  __I  uint32_t RESERVED29;
  __IO uint32_t RCGCPECI;                   /*!< Platform Environment Control Interface Run Mode Clock Gating Control */
  __IO uint32_t RCGCFAN;                    /*!< FAN Run Mode Clock Gating Control     */
  __IO uint32_t RCGCEEPROM;                 /*!< EEPROM Run Mode Clock Gating Control  */
  __IO uint32_t RCGCWTIMER;                 /*!< Wide Timer Run Mode Clock Gating Control */
  __I  uint32_t RESERVED30[40];
  __IO uint32_t SCGCWD;                     /*!< Watchdog Timer Sleep Mode Clock Gating Control */
  __IO uint32_t SCGCTIMER;                  /*!< Timer Sleep Mode Clock Gating Control */
  __IO uint32_t SCGCGPIO;                   /*!< General-Purpose Input/Output Sleep Mode Clock Gating Control */
  __IO uint32_t SCGCDMA;                    /*!< Micro Direct Memory Access Sleep Mode Clock Gating Control */
  __I  uint32_t RESERVED31;
  __IO uint32_t SCGCHIB;                    /*!< Hibernation Sleep Mode Clock Gating Control */
  __IO uint32_t SCGCUART;                   /*!< Universal Asynchronous Receiver/Transmitter Sleep Mode Clock Gating Control */
  __IO uint32_t SCGCSSI;                    /*!< Synchronous Serial Interface Sleep Mode Clock Gating Control */
  __IO uint32_t SCGCI2C;                    /*!< Inter-Integrated Circuit Sleep Mode Clock Gating Control */
  __I  uint32_t RESERVED32;
  __IO uint32_t SCGCUSB;                    /*!< Universal Serial Bus Sleep Mode Clock Gating Control */
  __I  uint32_t RESERVED33[2];
  __IO uint32_t SCGCCAN;                    /*!< Controller Area Network Sleep Mode Clock Gating Control */
  __IO uint32_t SCGCADC;                    /*!< Analog-to-Digital Converter Sleep Mode Clock Gating Control */
  __IO uint32_t SCGCACMP;                   /*!< Analog Comparator Sleep Mode Clock Gating Control */
  __IO uint32_t SCGCPWM;                    /*!< Pulse Width Modulator Sleep Mode Clock Gating Control */
  __IO uint32_t SCGCQEI;                    /*!< Quadrature Encoder Interface Sleep Mode Clock Gating Control */
  __IO uint32_t SCGCLPC;                    /*!< Low Pin Count Interface Sleep Mode Clock Gating Control */
  __I  uint32_t RESERVED34;
  __IO uint32_t SCGCPECI;                   /*!< Platform Environment Control Interface Sleep Mode Clock Gating Control */
  __IO uint32_t SCGCFAN;                    /*!< FAN Sleep Mode Clock Gating Control   */
  __IO uint32_t SCGCEEPROM;                 /*!< EEPROM Sleep Mode Clock Gating Control */
  __IO uint32_t SCGCWTIMER;                 /*!< Wide Timer Sleep Mode Clock Gating Control */
  __I  uint32_t RESERVED35[40];
  __IO uint32_t DCGCWD;                     /*!< Watchdog Timer Deep-Sleep Mode Clock Gating Control */
  __IO uint32_t DCGCTIMER;                  /*!< Timer Deep-Sleep Mode Clock Gating Control */
  __IO uint32_t DCGCGPIO;                   /*!< General-Purpose Input/Output Deep-Sleep Mode Clock Gating Control */
  __IO uint32_t DCGCDMA;                    /*!< Micro Direct Memory Access Deep-Sleep Mode Clock Gating Control */
  __I  uint32_t RESERVED36;
  __IO uint32_t DCGCHIB;                    /*!< Hibernation Deep-Sleep Mode Clock Gating Control */
  __IO uint32_t DCGCUART;                   /*!< Universal Asynchronous Receiver/Transmitter Deep-Sleep Mode Clock Gating Control */
  __IO uint32_t DCGCSSI;                    /*!< Synchronous Serial Interface Deep-Sleep Mode Clock Gating Control */
  __IO uint32_t DCGCI2C;                    /*!< Inter-Integrated Circuit Deep-Sleep Mode Clock Gating Control */
  __I  uint32_t RESERVED37;
  __IO uint32_t DCGCUSB;                    /*!< Universal Serial Bus Deep-Sleep Mode Clock Gating Control */
  __I  uint32_t RESERVED38[2];
  __IO uint32_t DCGCCAN;                    /*!< Controller Area Network Deep-Sleep Mode Clock Gating Control */
  __IO uint32_t DCGCADC;                    /*!< Analog-to-Digital Converter Deep-Sleep Mode Clock Gating Control */
  __IO uint32_t DCGCACMP;                   /*!< Analog Comparator Deep-Sleep Mode Clock Gating Control */
  __IO uint32_t DCGCPWM;                    /*!< Pulse Width Modulator Deep-Sleep Mode Clock Gating Control */
  __IO uint32_t DCGCQEI;                    /*!< Quadrature Encoder Interface Deep-Sleep Mode Clock Gating Control */
  __IO uint32_t DCGCLPC;                    /*!< Low Pin Count Interface Deep-Sleep Mode Clock Gating Control */
  __I  uint32_t RESERVED39;
  __IO uint32_t DCGCPECI;                   /*!< Platform Environment Control Interface Deep-Sleep Mode Clock Gating Control */
  __IO uint32_t DCGCFAN;                    /*!< FAN Deep-Sleep Mode Clock Gating Control */
  __IO uint32_t DCGCEEPROM;                 /*!< EEPROM Deep-Sleep Mode Clock Gating Control */
  __IO uint32_t DCGCWTIMER;                 /*!< Wide Timer Deep-Sleep Mode Clock Gating Control */
  __I  uint32_t RESERVED40[50];
  __IO uint32_t PCUSB;                      /*!< Universal Serial Bus Power Control    */
  __I  uint32_t RESERVED41[2];
  __IO uint32_t PCCAN;                      /*!< Controller Area Network Power Control */
  __I  uint32_t RESERVED42[50];
  __IO uint32_t PRWD;                       /*!< Watchdog Timer Peripheral Ready       */
  __IO uint32_t PRTIMER;                    /*!< Timer Peripheral Ready                */
  __IO uint32_t PRGPIO;                     /*!< General-Purpose Input/Output Peripheral Ready */
  __IO uint32_t PRDMA;                      /*!< Micro Direct Memory Access Peripheral Ready */
  __I  uint32_t RESERVED43;
  __IO uint32_t PRHIB;                      /*!< Hibernation Peripheral Ready          */
  __IO uint32_t PRUART;                     /*!< Universal Asynchronous Receiver/Transmitter Peripheral Ready */
  __IO uint32_t PRSSI;                      /*!< Synchronous Serial Interface Peripheral Ready */
  __IO uint32_t PRI2C;                      /*!< Inter-Integrated Circuit Peripheral Ready */
  __I  uint32_t RESERVED44;
  __IO uint32_t PRUSB;                      /*!< Universal Serial Bus Peripheral Ready */
  __I  uint32_t RESERVED45[2];
  __IO uint32_t PRCAN;                      /*!< Controller Area Network Peripheral Ready */
  __IO uint32_t PRADC;                      /*!< Analog-to-Digital Converter Peripheral Ready */
  __IO uint32_t PRACMP;                     /*!< Analog Comparator Peripheral Ready    */
  __IO uint32_t PRPWM;                      /*!< Pulse Width Modulator Peripheral Ready */
  __IO uint32_t PRQEI;                      /*!< Quadrature Encoder Interface Peripheral Ready */
  __IO uint32_t PRLPC;                      /*!< Low Pin Count Interface Peripheral Ready */
  __I  uint32_t RESERVED46;
  __IO uint32_t PRPECI;                     /*!< Platform Environment Control Interface Peripheral Ready */
  __IO uint32_t PRFAN;                      /*!< FAN Peripheral Ready                  */
  __IO uint32_t PREEPROM;                   /*!< EEPROM Peripheral Ready               */
  __IO uint32_t PRWTIMER;                   /*!< Wide Timer Peripheral Ready           */
} SYSCTL_Type;


// ------------------------------------------------------------------------------------------------
// -----                                         UDMA                                         -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for UDMA peripheral (UDMA)
  */

typedef struct {                            /*!< UDMA Structure                        */
  __IO uint32_t STAT;                       /*!< DMA Status                            */
  __O  uint32_t CFG;                        /*!< DMA Configuration                     */
  __IO uint32_t CTLBASE;                    /*!< DMA Channel Control Base Pointer      */
  __IO uint32_t ALTBASE;                    /*!< DMA Alternate Channel Control Base Pointer */
  __IO uint32_t WAITSTAT;                   /*!< DMA Channel Wait-on-Request Status    */
  __O  uint32_t SWREQ;                      /*!< DMA Channel Software Request          */
  __IO uint32_t USEBURSTSET;                /*!< DMA Channel Useburst Set              */
  __O  uint32_t USEBURSTCLR;                /*!< DMA Channel Useburst Clear            */
  __IO uint32_t REQMASKSET;                 /*!< DMA Channel Request Mask Set          */
  __O  uint32_t REQMASKCLR;                 /*!< DMA Channel Request Mask Clear        */
  __IO uint32_t ENASET;                     /*!< DMA Channel Enable Set                */
  __O  uint32_t ENACLR;                     /*!< DMA Channel Enable Clear              */
  __IO uint32_t ALTSET;                     /*!< DMA Channel Primary Alternate Set     */
  __O  uint32_t ALTCLR;                     /*!< DMA Channel Primary Alternate Clear   */
  __IO uint32_t PRIOSET;                    /*!< DMA Channel Priority Set              */
  __O  uint32_t PRIOCLR;                    /*!< DMA Channel Priority Clear            */
  __I  uint32_t RESERVED0[3];
  __IO uint32_t ERRCLR;                     /*!< DMA Bus Error Clear                   */
  __I  uint32_t RESERVED1[300];
  __IO uint32_t CHASGN;                     /*!< DMA Channel Assignment                */
  __IO uint32_t CHIS;                       /*!< DMA Channel Interrupt Status          */
  __I  uint32_t RESERVED2[2];
  __IO uint32_t CHMAP0;                     /*!< DMA Channel Map Select 0              */
  __IO uint32_t CHMAP1;                     /*!< DMA Channel Map Select 1              */
  __IO uint32_t CHMAP2;                     /*!< DMA Channel Map Select 2              */
  __IO uint32_t CHMAP3;                     /*!< DMA Channel Map Select 3              */
} UDMA_Type;


// ------------------------------------------------------------------------------------------------
// -----                                         NVIC                                         -----
// ------------------------------------------------------------------------------------------------


/**
  * @brief Register map for NVIC peripheral (NVIC)
  */

typedef struct {                            /*!< NVIC Structure                        */
  __I  uint32_t RESERVED0;
  __IO uint32_t INT_TYPE;                   /*!< Interrupt Controller Type Reg         */
  __IO uint32_t ACTLR;                      /*!< Auxiliary Control                     */
  __I  uint32_t RESERVED1;
  __IO uint32_t ST_CTRL;                    /*!< SysTick Control and Status Register   */
  __IO uint32_t ST_RELOAD;                  /*!< SysTick Reload Value Register         */
  __IO uint32_t ST_CURRENT;                 /*!< SysTick Current Value Register        */
  __IO uint32_t ST_CAL;                     /*!< SysTick Calibration Value Reg         */
  __I  uint32_t RESERVED2[56];
  __IO uint32_t EN0;                        /*!< Interrupt 0-31 Set Enable             */
  __IO uint32_t EN1;                        /*!< Interrupt 32-54 Set Enable            */
  __IO uint32_t EN2;                        /*!< Interrupt 64-95 Set Enable            */
  __IO uint32_t EN3;                        /*!< Interrupt 96-127 Set Enable           */
  __IO uint32_t EN4;                        /*!< Interrupt 128-131 Set Enable          */
  __I  uint32_t RESERVED3[27];
  __IO uint32_t DIS0;                       /*!< Interrupt 0-31 Clear Enable           */
  __IO uint32_t DIS1;                       /*!< Interrupt 32-54 Clear Enable          */
  __IO uint32_t DIS2;                       /*!< Interrupt 64-95 Clear Enable          */
  __IO uint32_t DIS3;                       /*!< Interrupt 96-127 Clear Enable         */
  __IO uint32_t DIS4;                       /*!< Interrupt 128-131 Clear Enable        */
  __I  uint32_t RESERVED4[27];
  __IO uint32_t PEND0;                      /*!< Interrupt 0-31 Set Pending            */
  __IO uint32_t PEND1;                      /*!< Interrupt 32-54 Set Pending           */
  __IO uint32_t PEND2;                      /*!< Interrupt 64-95 Set Pending           */
  __IO uint32_t PEND3;                      /*!< Interrupt 96-127 Set Pending          */
  __IO uint32_t PEND4;                      /*!< Interrupt 128-131 Set Pending         */
  __I  uint32_t RESERVED5[27];
  __IO uint32_t UNPEND0;                    /*!< Interrupt 0-31 Clear Pending          */
  __IO uint32_t UNPEND1;                    /*!< Interrupt 32-54 Clear Pending         */
  __IO uint32_t UNPEND2;                    /*!< Interrupt 64-95 Clear Pending         */
  __IO uint32_t UNPEND3;                    /*!< Interrupt 96-127 Clear Pending        */
  __IO uint32_t UNPEND4;                    /*!< Interrupt 128-131 Clear Pending       */
  __I  uint32_t RESERVED6[27];
  __IO uint32_t ACTIVE0;                    /*!< Interrupt 0-31 Active Bit             */
  __IO uint32_t ACTIVE1;                    /*!< Interrupt 32-54 Active Bit            */
  __IO uint32_t ACTIVE2;                    /*!< Interrupt 64-95 Active Bit            */
  __IO uint32_t ACTIVE3;                    /*!< Interrupt 96-127 Active Bit           */
  __IO uint32_t ACTIVE4;                    /*!< Interrupt 128-131 Active Bit          */
  __I  uint32_t RESERVED7[59];
  __IO uint32_t PRI0;                       /*!< Interrupt 0-3 Priority                */
  __IO uint32_t PRI1;                       /*!< Interrupt 4-7 Priority                */
  __IO uint32_t PRI2;                       /*!< Interrupt 8-11 Priority               */
  __IO uint32_t PRI3;                       /*!< Interrupt 12-15 Priority              */
  __IO uint32_t PRI4;                       /*!< Interrupt 16-19 Priority              */
  __IO uint32_t PRI5;                       /*!< Interrupt 20-23 Priority              */
  __IO uint32_t PRI6;                       /*!< Interrupt 24-27 Priority              */
  __IO uint32_t PRI7;                       /*!< Interrupt 28-31 Priority              */
  __IO uint32_t PRI8;                       /*!< Interrupt 32-35 Priority              */
  __IO uint32_t PRI9;                       /*!< Interrupt 36-39 Priority              */
  __IO uint32_t PRI10;                      /*!< Interrupt 40-43 Priority              */
  __IO uint32_t PRI11;                      /*!< Interrupt 44-47 Priority              */
  __IO uint32_t PRI12;                      /*!< Interrupt 48-51 Priority              */
  __IO uint32_t PRI13;                      /*!< Interrupt 52-53 Priority              */
  __IO uint32_t PRI14;                      /*!< Interrupt 56-59 Priority              */
  __IO uint32_t PRI15;                      /*!< Interrupt 60-63 Priority              */
  __IO uint32_t PRI16;                      /*!< Interrupt 64-67 Priority              */
  __IO uint32_t PRI17;                      /*!< Interrupt 68-71 Priority              */
  __IO uint32_t PRI18;                      /*!< Interrupt 72-75 Priority              */
  __IO uint32_t PRI19;                      /*!< Interrupt 76-79 Priority              */
  __IO uint32_t PRI20;                      /*!< Interrupt 80-83 Priority              */
  __IO uint32_t PRI21;                      /*!< Interrupt 84-87 Priority              */
  __IO uint32_t PRI22;                      /*!< Interrupt 88-91 Priority              */
  __IO uint32_t PRI23;                      /*!< Interrupt 92-95 Priority              */
  __IO uint32_t PRI24;                      /*!< Interrupt 96-99 Priority              */
  __IO uint32_t PRI25;                      /*!< Interrupt 100-103 Priority            */
  __IO uint32_t PRI26;                      /*!< Interrupt 104-107 Priority            */
  __IO uint32_t PRI27;                      /*!< Interrupt 108-111 Priority            */
  __IO uint32_t PRI28;                      /*!< Interrupt 112-115 Priority            */
  __IO uint32_t PRI29;                      /*!< Interrupt 116-119 Priority            */
  __IO uint32_t PRI30;                      /*!< Interrupt 120-123 Priority            */
  __IO uint32_t PRI31;                      /*!< Interrupt 124-127 Priority            */
  __IO uint32_t PRI32;                      /*!< Interrupt 128-131 Priority            */
  __I  uint32_t RESERVED8[543];
  __IO uint32_t CPUID;                      /*!< CPU ID Base                           */
  __IO uint32_t INT_CTRL;                   /*!< Interrupt Control and State           */
  __IO uint32_t VTABLE;                     /*!< Vector Table Offset                   */
  __IO uint32_t APINT;                      /*!< Application Interrupt and Reset Control */
  __IO uint32_t SYS_CTRL;                   /*!< System Control                        */
  __IO uint32_t CFG_CTRL;                   /*!< Configuration and Control             */
  __IO uint32_t SYS_PRI1;                   /*!< System Handler Priority 1             */
  __IO uint32_t SYS_PRI2;                   /*!< System Handler Priority 2             */
  __IO uint32_t SYS_PRI3;                   /*!< System Handler Priority 3             */
  __IO uint32_t SYS_HND_CTRL;               /*!< System Handler Control and State      */
  __IO uint32_t FAULT_STAT;                 /*!< Configurable Fault Status             */
  __IO uint32_t HFAULT_STAT;                /*!< Hard Fault Status                     */
  __IO uint32_t DEBUG_STAT;                 /*!< Debug Status Register                 */
  __IO uint32_t MM_ADDR;                    /*!< Memory Management Fault Address       */
  __IO uint32_t FAULT_ADDR;                 /*!< Bus Fault Address                     */
  __I  uint32_t RESERVED9[21];
  __IO uint32_t MPU_TYPE;                   /*!< MPU Type                              */
  __IO uint32_t MPU_CTRL;                   /*!< MPU Control                           */
  __IO uint32_t MPU_NUMBER;                 /*!< MPU Region Number                     */
  __IO uint32_t MPU_BASE;                   /*!< MPU Region Base Address               */
  __IO uint32_t MPU_ATTR;                   /*!< MPU Region Attribute and Size         */
  __IO uint32_t MPU_BASE1;                  /*!< MPU Region Base Address Alias 1       */
  __IO uint32_t MPU_ATTR1;                  /*!< MPU Region Attribute and Size Alias 1 */
  __IO uint32_t MPU_BASE2;                  /*!< MPU Region Base Address Alias 2       */
  __IO uint32_t MPU_ATTR2;                  /*!< MPU Region Attribute and Size Alias 2 */
  __IO uint32_t MPU_BASE3;                  /*!< MPU Region Base Address Alias 3       */
  __IO uint32_t MPU_ATTR3;                  /*!< MPU Region Attribute and Size Alias 3 */
  __I  uint32_t RESERVED10[13];
  __IO uint32_t DBG_CTRL;                   /*!< Debug Control and Status Reg          */
  __IO uint32_t DBG_XFER;                   /*!< Debug Core Reg. Transfer Select       */
  __IO uint32_t DBG_DATA;                   /*!< Debug Core Register Data              */
  __IO uint32_t DBG_INT;                    /*!< Debug Reset Interrupt Control         */
  __I  uint32_t RESERVED11[64];
  __O  uint32_t SW_TRIG;                    /*!< Software Trigger Interrupt            */
} NVIC_Type;



/********************************************
** End of section using anonymous unions   **
*********************************************/

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



// ------------------------------------------------------------------------------------------------
// -----                                 Peripheral memory map                                -----
// ------------------------------------------------------------------------------------------------

#define WATCHDOG0_BASE            0x40000000
#define WATCHDOG1_BASE            0x40001000
#define GPIO_PORTA_BASE           0x40004000
#define GPIO_PORTB_BASE           0x40005000
#define GPIO_PORTC_BASE           0x40006000
#define GPIO_PORTD_BASE           0x40007000
#define SSI0_BASE                 0x40008000
#define SSI1_BASE                 0x40009000
#define SSI2_BASE                 0x4000A000
#define SSI3_BASE                 0x4000B000
#define UART0_BASE                0x4000C000
#define UART1_BASE                0x4000D000
#define UART2_BASE                0x4000E000
#define UART3_BASE                0x4000F000
#define UART4_BASE                0x40010000
#define UART5_BASE                0x40011000
#define UART6_BASE                0x40012000
#define UART7_BASE                0x40013000
#define I2C0_BASE                 0x40020000
#define I2C1_BASE                 0x40021000
#define I2C2_BASE                 0x40022000
#define I2C3_BASE                 0x40023000
#define GPIO_PORTE_BASE           0x40024000
#define GPIO_PORTF_BASE           0x40025000
#define GPIO_PORTG_BASE           0x40026000
#define PWM0_BASE                 0x40028000
#define PWM1_BASE                 0x40029000
#define QEI0_BASE                 0x4002C000
#define QEI1_BASE                 0x4002D000
#define TIMER0_BASE               0x40030000
#define TIMER1_BASE               0x40031000
#define TIMER2_BASE               0x40032000
#define TIMER3_BASE               0x40033000
#define TIMER4_BASE               0x40034000
#define TIMER5_BASE               0x40035000
#define WTIMER0_BASE              0x40036000
#define WTIMER1_BASE              0x40037000
#define ADC0_BASE                 0x40038000
#define ADC1_BASE                 0x40039000
#define COMP_BASE                 0x4003C000
#define CAN0_BASE                 0x40040000
#define CAN1_BASE                 0x40041000
#define WTIMER2_BASE              0x4004C000
#define WTIMER3_BASE              0x4004D000
#define WTIMER4_BASE              0x4004E000
#define WTIMER5_BASE              0x4004F000
#define GPIO_PORTA_AHB_BASE       0x40058000
#define GPIO_PORTB_AHB_BASE       0x40059000
#define GPIO_PORTC_AHB_BASE       0x4005A000
#define GPIO_PORTD_AHB_BASE       0x4005B000
#define GPIO_PORTE_AHB_BASE       0x4005C000
#define GPIO_PORTF_AHB_BASE       0x4005D000
#define GPIO_PORTG_AHB_BASE       0x4005E000
#define I2C4_BASE                 0x400C0000
#define I2C5_BASE                 0x400C1000
#define FLASH_CTRL_BASE           0x400FD000
#define SYSCTL_BASE               0x400FE000
#define UDMA_BASE                 0x400FF000
#define NVIC_BASE                 0xE000E000


// ------------------------------------------------------------------------------------------------
// -----                                Peripheral declaration                                -----
// ------------------------------------------------------------------------------------------------

#define WATCHDOG0                 ((WATCHDOG0_Type          *) WATCHDOG0_BASE)
#define WATCHDOG1                 ((WATCHDOG1_Type          *) WATCHDOG1_BASE)
#define GPIO_PORTA                ((GPIO_PORTA_Type         *) GPIO_PORTA_BASE)
#define GPIO_PORTB                ((GPIO_PORTB_Type         *) GPIO_PORTB_BASE)
#define GPIO_PORTC                ((GPIO_PORTC_Type         *) GPIO_PORTC_BASE)
#define GPIO_PORTD                ((GPIO_PORTD_Type         *) GPIO_PORTD_BASE)
#define SSI0                      ((SSI0_Type               *) SSI0_BASE)
#define SSI1                      ((SSI1_Type               *) SSI1_BASE)
#define SSI2                      ((SSI2_Type               *) SSI2_BASE)
#define SSI3                      ((SSI3_Type               *) SSI3_BASE)
#define UART0                     ((UART0_Type              *) UART0_BASE)
#define UART1                     ((UART1_Type              *) UART1_BASE)
#define UART2                     ((UART2_Type              *) UART2_BASE)
#define UART3                     ((UART3_Type              *) UART3_BASE)
#define UART4                     ((UART4_Type              *) UART4_BASE)
#define UART5                     ((UART5_Type              *) UART5_BASE)
#define UART6                     ((UART6_Type              *) UART6_BASE)
#define UART7                     ((UART7_Type              *) UART7_BASE)
#define I2C0                      ((I2C0_Type               *) I2C0_BASE)
#define I2C1                      ((I2C1_Type               *) I2C1_BASE)
#define I2C2                      ((I2C2_Type               *) I2C2_BASE)
#define I2C3                      ((I2C3_Type               *) I2C3_BASE)
#define GPIO_PORTE                ((GPIO_PORTE_Type         *) GPIO_PORTE_BASE)
#define GPIO_PORTF                ((GPIO_PORTF_Type         *) GPIO_PORTF_BASE)
#define GPIO_PORTG                ((GPIO_PORTG_Type         *) GPIO_PORTG_BASE)
#define PWM0                      ((PWM0_Type               *) PWM0_BASE)
#define PWM1                      ((PWM1_Type               *) PWM1_BASE)
#define QEI0                      ((QEI0_Type               *) QEI0_BASE)
#define QEI1                      ((QEI1_Type               *) QEI1_BASE)
#define TIMER0                    ((TIMER0_Type             *) TIMER0_BASE)
#define TIMER1                    ((TIMER1_Type             *) TIMER1_BASE)
#define TIMER2                    ((TIMER2_Type             *) TIMER2_BASE)
#define TIMER3                    ((TIMER3_Type             *) TIMER3_BASE)
#define TIMER4                    ((TIMER4_Type             *) TIMER4_BASE)
#define TIMER5                    ((TIMER5_Type             *) TIMER5_BASE)
#define WTIMER0                   ((WTIMER0_Type            *) WTIMER0_BASE)
#define WTIMER1                   ((WTIMER1_Type            *) WTIMER1_BASE)
#define ADC0                      ((ADC0_Type               *) ADC0_BASE)
#define ADC1                      ((ADC1_Type               *) ADC1_BASE)
#define COMP                      ((COMP_Type               *) COMP_BASE)
#define CAN0                      ((CAN0_Type               *) CAN0_BASE)
#define CAN1                      ((CAN1_Type               *) CAN1_BASE)
#define WTIMER2                   ((WTIMER2_Type            *) WTIMER2_BASE)
#define WTIMER3                   ((WTIMER3_Type            *) WTIMER3_BASE)
#define WTIMER4                   ((WTIMER4_Type            *) WTIMER4_BASE)
#define WTIMER5                   ((WTIMER5_Type            *) WTIMER5_BASE)
#define GPIO_PORTA_AHB            ((GPIO_PORTA_AHB_Type     *) GPIO_PORTA_AHB_BASE)
#define GPIO_PORTB_AHB            ((GPIO_PORTB_AHB_Type     *) GPIO_PORTB_AHB_BASE)
#define GPIO_PORTC_AHB            ((GPIO_PORTC_AHB_Type     *) GPIO_PORTC_AHB_BASE)
#define GPIO_PORTD_AHB            ((GPIO_PORTD_AHB_Type     *) GPIO_PORTD_AHB_BASE)
#define GPIO_PORTE_AHB            ((GPIO_PORTE_AHB_Type     *) GPIO_PORTE_AHB_BASE)
#define GPIO_PORTF_AHB            ((GPIO_PORTF_AHB_Type     *) GPIO_PORTF_AHB_BASE)
#define GPIO_PORTG_AHB            ((GPIO_PORTG_AHB_Type     *) GPIO_PORTG_AHB_BASE)
#define I2C4                      ((I2C4_Type               *) I2C4_BASE)
#define I2C5                      ((I2C5_Type               *) I2C5_BASE)
#define FLASH_CTRL                ((FLASH_CTRL_Type         *) FLASH_CTRL_BASE)
#define SYSCTL                    ((SYSCTL_Type             *) SYSCTL_BASE)
#define UDMA                      ((UDMA_Type               *) UDMA_BASE)
#define NVIC                      ((NVIC_Type               *) NVIC_BASE)



/** @} */ /* End of group Device_Peripheral_Registers */
/** @} */ /* End of group LM4F211E5QR */
/** @} */ /* End of group (null) */

#ifdef __cplusplus
}
#endif 


#endif  // __LM4F211E5QR_H__

