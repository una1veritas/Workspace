/* ----------------------------------------------------------------------------
 * Copyright (c) 2008 - 2012 Semiconductor Components Industries, LLC (d/b/a ON
 * Semiconductor), All Rights Reserved
 *
 * This code is the property of ON Semiconductor and may not be redistributed
 * in any form without prior written permission from ON Semiconductor.
 * The terms of use and warranty for this code are covered by contractual
 * agreements between ON Semiconductor and the licensee.
 * ----------------------------------------------------------------------------
 * q32m200_nvic.h
 * - NVIC hardware support functions and macros
 * 
 * IMPORTANT: Do not include this file directly. Please use <q32.h> or 
 *            <q32m210.h> instead. 
 * ----------------------------------------------------------------------------
 * $Revision: 53234 $
 * $Date: 2012-05-15 12:18:28 +0200 (ti, 15 maj 2012) $
 * ------------------------------------------------------------------------- */

#ifndef Q32M200_NVIC_H
#define Q32M200_NVIC_H

/* ----------------------------------------------------------------------------
 * If building with a C++ compiler, make all of the definitions in this header
 * have a C binding.
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
extern "C" {
#endif

/* ----------------------------------------------------------------------------
 * NVIC support defines
 * ------------------------------------------------------------------------- */
#define FIRST_EXT_INTERRUPT             INT_WAKEUP

#define NVIC_PRIGROUP_4_0               ((uint32_t)(0x3))
#define NVIC_PRIGROUP_3_1               ((uint32_t)(0x4))
#define NVIC_PRIGROUP_2_2               ((uint32_t)(0x5))
#define NVIC_PRIGROUP_1_3               ((uint32_t)(0x6))
#define NVIC_PRIGROUP_0_4               ((uint32_t)(0x7))

/* ----------------------------------------------------------------------------
 * NVIC support function prototypes
 * ------------------------------------------------------------------------- */

/* Fault Control - applies to MPU, bus and usage faults */
extern void Sys_NVIC_Fault_EnableInt(IRQn_Type intnum);
extern void Sys_NVIC_Fault_DisableInt(IRQn_Type intnum);
extern uint32_t Sys_NVIC_Fault_IsActive(IRQn_Type intnum);
extern uint32_t Sys_NVIC_Fault_IsPending(IRQn_Type intnum);
extern void Sys_NVIC_Fault_SetPending(IRQn_Type intnum);
extern void Sys_NVIC_Fault_ClearPending(IRQn_Type intnum);

/* External (i.e., external to the ARM Cortex-M3) interrupt control */
extern void Sys_NVIC_Int_DisableAllInt(void);
extern void Sys_NVIC_Int_ClearAllPending(void);

/* ----------------------------------------------------------------------------
 * NVIC support inline functions
 * ------------------------------------------------------------------------- */

#define EXT_INTERRUPT_VTABLE_OFFSET 16

/* ----------------------------------------------------------------------------
 * Function      : void Sys_NVIC_MasterEnable()
 * ----------------------------------------------------------------------------
 * Description   : Enable interrupts globally using the PRIMASK register
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_NVIC_MasterEnable()
{
    /* Unmask interrupts using PRIMASK */
    __set_PRIMASK(0x0);
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_NVIC_MasterDisable()
 * ----------------------------------------------------------------------------
 * Description   : Disable interrupts globally using the PRIMASK register. The
 *                 NMI and hard fault exceptions are not disabled.
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_NVIC_MasterDisable()
{
    /* Mask all non-fault interrupts using PRIMASK */
    __set_PRIMASK(0x1);
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_NVIC_MasterFaultEnable()
 * ----------------------------------------------------------------------------
 * Description   : Enable interrupts globally using the FAULTMASK register
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_NVIC_MasterFaultEnable()
{
      /* Unmask all non-fault interrupts using FAULTMASK */
    __set_FAULTMASK(0x0);
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_NVIC_MasterFaultDisable()
 * ----------------------------------------------------------------------------
 * Description   : Disable interrupts globally using the FAULTMASK register.
 *                 Only the NMI exception is not disabled.
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_NVIC_MasterFaultDisable()
{
    /* Mask all interrupts using FAULTMASK */
    __set_FAULTMASK(0x1);
}


/* ----------------------------------------------------------------------------
 * Function      : void Sys_NVIC_Set_BasePriority(uint32_t priority)
 * ----------------------------------------------------------------------------
 * Description   : Set the base priority level. Effectively masks (disables)
 *                 all interrupts of equal or lower priority level.
 * Inputs        : priority - Base priority level
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_NVIC_Set_BasePriority(uint32_t priority)
{
    __set_BASEPRI(priority);
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_NVIC_Get_BasePriority()
 * ----------------------------------------------------------------------------
 * Description   : Get the base priority level.
 * Inputs        : None
 * Outputs       : priority - Base priority level
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_NVIC_Get_BasePriority()
{
    return __get_BASEPRI();
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_NVIC_Set_PriorityGroup(uint32_t group)
 * ----------------------------------------------------------------------------
 * Description   : Set the priority grouping. Grouping is split between
 *                 pre-emptable and sub-priority levels.  Directly calls
 *                 NVIC_SetPriorityGrouping().  Included for backwards 
 *                 compatibility.
 * Inputs        : group - Priority grouping; use NVIC_PRIGROUP_*
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */

static __INLINE void Sys_NVIC_Set_PriorityGroup(uint32_t group)
{
    /* The behaviour is changed here compared to the original Q32M210
     * system library.  The original library assumed that the group 
     * parameter was already shifted by NVIC_APP_INT_CTRL_PRIGROUP_POS; 
     * however, it was then not symmetric with Sys_NVIC_Get_PriorityGroup(), 
     * which included a right shift.  In this library we'll call 
     * NVIC_SetPriorityGrouping() directly, which performs the shift. */
    NVIC_SetPriorityGrouping(group);
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_NVIC_Get_PriorityGroup()
 * ----------------------------------------------------------------------------
 * Description   : Get the priority grouping. Grouping is split between
 *                 pre-emptable and sub-priority levels.  Directly calls
 *                 NVIC_GetPriorityGrouping().  Included for backwards 
 *                 compatibility.
 * Inputs        : None
 * Outputs       : return value - Priority grouping
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_NVIC_Get_PriorityGroup()
{
    return NVIC_GetPriorityGrouping();
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_NVIC_Set_VectTable(uint32_t vect_table)
 * ----------------------------------------------------------------------------
 * Description   : Set the location of the interrupt vector table.
 * Inputs        : vect_table - Base address of interrupt vector table
 * Outputs       : None
 * Assumptions   : The vector table address must be on a 1 kb aligned address
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_NVIC_Set_VectTable(uint32_t vect_table)
{
    SCB->VTOR = vect_table;
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_NVIC_Get_VectTable()
 * ----------------------------------------------------------------------------
 * Description   : Get the location of the interrupt vector table.
 * Inputs        : None
 * Outputs       : return value - Address of vector table; value loaded from
 *                                NVIC_VTABLE_OFFSET
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_NVIC_Get_VectTable()
{
    return SCB->VTOR;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_NVIC_Set_ISRVector(IRQn_Type intnum,
 *                                             uint32_t vector)
 * ----------------------------------------------------------------------------
 * Description   : Set the address of an interrupt service routine vector in
 *                 the interrupt vector table.
 * Inputs        : - intnum - Interrupt vector to set; use INT_*
 *                 - vector - Address of interrupt vector
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_NVIC_Set_ISRVector(IRQn_Type intnum, uint32_t vector)
{
    uint32_t* vtable = (uint32_t*) SCB->VTOR;

    *(vtable + intnum + EXT_INTERRUPT_VTABLE_OFFSET) = vector;
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_NVIC_Get_ISRVector(IRQn_Type intnum)
 * ----------------------------------------------------------------------------
 * Description   : Get the address of an interrupt service routine vector in
 *                 the interrupt vector table.
 * Inputs        : intnum - Interrupt vector to get; use INT_*
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_NVIC_Get_ISRVector(IRQn_Type intnum)
{
    uint32_t* vtable = (uint32_t*) SCB->VTOR;

    return *(vtable + intnum + EXT_INTERRUPT_VTABLE_OFFSET);
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_NVIC_Set_ExceptionPriority(IRQn_Type intnum,
 *                                                     uint8_t priority)
 * ----------------------------------------------------------------------------
 * Description   : Set the priority level for the specified exception. Only
 *                 set the priority for the MPU fault, bus fault, usage fault,
 *                 SVC, debug monitor, PendSV and SYSTICK exceptions.
 * Inputs        : - intnum     - Exception to set the priority for; use *_IRQn
 *                 - priority   - Priority level for the exception
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_NVIC_Set_ExceptionPriority(IRQn_Type intnum,
                                                    uint8_t priority)
{
    /* The original syslib function expected the priority to be pre-encoded/
     * shifted into the form in which the register is set.  This is maintained
     * for backwards compatibility, even though NVIC_SetPriority expects 
     * an unshifted value. */
    NVIC_SetPriority(intnum, priority >> (8 - __NVIC_PRIO_BITS));
}

/* ----------------------------------------------------------------------------
 * Function      : uint8_t
 *                 Sys_NVIC_Get_ExceptionPriority(IRQn_Type intnum)
 * ----------------------------------------------------------------------------
 * Description   : Get the priority level for the specified exception. Only
 *                 get the priority for the MPU fault, bus fault, usage fault,
 *                 SVC, debug monitor, PendSV and SYSTICK exceptions.  Directly
 *                 calls NVIC_GetPriority().  Included for backwards 
 *                 compatibility.
 * Inputs        : intnum - Exception to get the priority for; use *_IRQn
 * Outputs       : return value - Priority level of the specified exception;
 *                                value loaded from the appropriate
 *                                NVIC_*_PRIORITY_BYTE register
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint8_t Sys_NVIC_Get_ExceptionPriority(IRQn_Type intnum)
{
    /* The original syslib function returns the priority byte in its encoded/
     * shifted form.  This is maintained for backwards compatibility.
     */
    return (NVIC_GetPriority(intnum) << (8 - __NVIC_PRIO_BITS)) & 0xff;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_NVIC_Set_InterruptPriority(IRQn_Type intnum,
 *                                                     uint8_t priority)
 * ----------------------------------------------------------------------------
 * Description   : Set the priority level for the specified interrupt. Only
 *                 set the priority for external interrupts.  Directly calls
 *                 NVIC_SetPriority().  Included for backwards compatibility.
 * Inputs        : - intnum - Interrupt to set the priority for; use *_IRQn
 *                 - priority - Priority level for the interrupt
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_NVIC_Set_InterruptPriority(IRQn_Type intnum,
                                                    uint8_t priority)
{
    /* The original syslib function expected the priority to be pre-encoded/
     * shifted into the form in which the register is set.  This is maintained
     * for backwards compatibility, even though NVIC_SetPriority expects 
     * an unshifted value. */
    NVIC_SetPriority(intnum, priority >> (8 - __NVIC_PRIO_BITS));
}

/* ----------------------------------------------------------------------------
 * Function      : uint8_t
 *                 Sys_NVIC_Get_InterruptPriority(IRQn_Type intnum)
 * ----------------------------------------------------------------------------
 * Description   : Get the priority level for the specified interrupt. Only
 *                 get the priority for external interrupts.  Directly calls
 *                 NVIC_GetPriority().  Included for backwards compatibility.
 * Inputs        : intnum - Interrupt to get the priority for; use *_IRQn
 * Outputs       : return value - Priority level of the specified interrupt as
 *                                specified in the NVIC_PRIORITY* register
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint8_t Sys_NVIC_Get_InterruptPriority(IRQn_Type intnum)
{
    /* The original syslib function returns the priority byte in its encoded/
     * shifted form.  This is maintained for backwards compatibility.
     */
    return (NVIC_GetPriority(intnum) << (8 - __NVIC_PRIO_BITS)) & 0xff;
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_NVIC_NMI_IsActive()
 * ----------------------------------------------------------------------------
 * Description   : Return whether the NMI exception is active.
 * Inputs        : None
 * Outputs       : Boolean value indicating whether the NMI exception is active.
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_NVIC_NMI_IsActive()
{
    return  ((SCB->ICSR &
              SCB_ICSR_VECTACTIVE_Msk) == (NonMaskableInt_IRQn + 
                                           EXT_INTERRUPT_VTABLE_OFFSET));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_NVIC_NMI_SetPending()
 * ----------------------------------------------------------------------------
 * Description   : Set the NMI exception to pending. This will cause the NMI to
 *                 execute immediately.
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_NVIC_NMI_SetPending()
{
    SCB->ICSR = (SCB->ICSR | (0x1UL << SCB_ICSR_NMIPENDSET_Pos));
}


/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_NVIC_SVC_IsActive()
 * ----------------------------------------------------------------------------
 * Description   : Return whether the SVC exception is active.
 * Inputs        : None
 * Outputs       : return value - Active state of SVC exception; value loaded
 *                                from SCB->SHCSR
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_NVIC_SVC_IsActive()
{
    return (SCB->SHCSR >> SCB_SHCSR_SVCALLACT_Pos) & 0x1;
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_NVIC_SVC_IsPending()
 * ----------------------------------------------------------------------------
 * Description   : Return whether the SVC exception is pending.
 * Inputs        : None
 * Outputs       : return value - Pending state of SVC exception; value loaded
 *                                from SCB->SHCSR
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_NVIC_SVC_IsPending()
{
    return (SCB->SHCSR >> SCB_SHCSR_SVCALLPENDED_Pos) & 0x1;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_NVIC_SVC_SetPending()
 * ----------------------------------------------------------------------------
 * Description   : Set the SVC exception to pending.
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_NVIC_SVC_SetPending()
{
    SCB->SHCSR = (SCB->SHCSR | (0x1UL << SCB_SHCSR_SVCALLPENDED_Pos));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_NVIC_SVC_ClearPending()
 * ----------------------------------------------------------------------------
 * Description   : Clear the pending status of the SVC exception.
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_NVIC_SVC_ClearPending()
{
    SCB->SHCSR = (SCB->SHCSR & ~(SCB_SHCSR_SVCALLPENDED_Msk));
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_NVIC_PendSV_IsActive()
 * ----------------------------------------------------------------------------
 * Description   : Return whether the PendSV exception is active.
 * Inputs        : None
 * Outputs       : return value - Active state of PendSV exception; value loaded
 *                                from SCB->SHCSR
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_NVIC_PendSV_IsActive()
{
    return (SCB->SHCSR >> SCB_SHCSR_PENDSVACT_Pos) & 0x1;
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_NVIC_PendSV_IsPending()
 * ----------------------------------------------------------------------------
 * Description   : Return whether the PendSV exception is pending.
 * Inputs        : None
 * Outputs       : return value - Pending state of PendSV exception; value
 *                                loaded from SCB->ICSR
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_NVIC_PendSV_IsPending()
{
    return (SCB->ICSR >> SCB_ICSR_PENDSVSET_Pos) & 0x1;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_NVIC_PendSV_SetPending()
 * ----------------------------------------------------------------------------
 * Description   : Set the PendSV exception to pending.
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_NVIC_PendSV_SetPending()
{
    SCB->ICSR = (SCB->ICSR | (0x1UL << SCB_ICSR_PENDSVSET_Pos));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_NVIC_PendSV_ClearPending()
 * ----------------------------------------------------------------------------
 * Description   : Clear the pending status of the PendSV exception.
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_NVIC_PendSV_ClearPending()
{
    SCB->ICSR = (SCB->ICSR | (0x1UL << SCB_ICSR_PENDSVCLR_Pos));
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_NVIC_SYSTICK_IsActive()
 * ----------------------------------------------------------------------------
 * Description   : Return whether the SYSTICK exception is active.
 * Inputs        : None
 * Outputs       : return value - Active state of SYSTICK exception; value
 *                                loaded from SCB->SHCSR
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_NVIC_SYSTICK_IsActive()
{
    return (SCB->SHCSR >> SCB_SHCSR_SYSTICKACT_Pos) & 0x1;
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_NVIC_SYSTICK_IsPending()
 * ----------------------------------------------------------------------------
 * Description   : Return whether the SYSTICK exception is pending.
 * Inputs        : None
 * Outputs       : return value - Pending state of SYSTICK exception; value
 *                                loaded from SCB->ICSR
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_NVIC_SYSTICK_IsPending()
{
    return (SCB->ICSR >> SCB_ICSR_PENDSTSET_Pos) & 0x1;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_NVIC_SYSTICK_SetPending()
 * ----------------------------------------------------------------------------
 * Description   : Set the SYSTICK exception to pending.
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_NVIC_SYSTICK_SetPending()
{
    SCB->ICSR = (SCB->ICSR | (0x1UL << SCB_ICSR_PENDSTSET_Pos));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_NVIC_SYSTICK_ClearPending()
 * ----------------------------------------------------------------------------
 * Description   : Clear the pending status of the SYSTICK exception.
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_NVIC_SYSTICK_ClearPending()
{
    SCB->ICSR = (SCB->ICSR | (0x1UL << SCB_ICSR_PENDSTCLR_Pos));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_NVIC_DebugMon_EnableInt()
 * ----------------------------------------------------------------------------
 * Description   : Enable the debug monitor exception
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_NVIC_DebugMon_EnableInt()
{
    CoreDebug->DEMCR = (CoreDebug->DEMCR |
                       (0x1UL << CoreDebug_DEMCR_MON_EN_Pos));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_NVIC_DebugMon_DisableInt()
 * ----------------------------------------------------------------------------
 * Description   : Disable the debug monitor exception
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_NVIC_DebugMon_DisableInt()
{
    CoreDebug->DEMCR = (CoreDebug->DEMCR & ~(CoreDebug_DEMCR_MON_EN_Msk));
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_NVIC_DebugMon_IsActive()
 * ----------------------------------------------------------------------------
 * Description   : Return whether the debug monitor exception is active.
 * Inputs        : None
 * Outputs       : return value - Active state of debug monitor exception; value
 *                                loaded from SCB->SHCSR
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_NVIC_DebugMon_IsActive()
{
    return (SCB->SHCSR >> SCB_SHCSR_MONITORACT_Pos) & 0x1;
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_NVIC_DebugMon_IsPending()
 * ----------------------------------------------------------------------------
 * Description   : Return whether the debug monitor exception is pending.
 * Inputs        : None
 * Outputs       : return value - Pending state of debug monitor exception;
 *                                value loaded from CoreDebug->DEMCR
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_NVIC_DebugMon_IsPending()
{
    return (CoreDebug->DEMCR >> CoreDebug_DEMCR_MON_PEND_Pos) & 0x1;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_NVIC_DebugMon_SetPending()
 * ----------------------------------------------------------------------------
 * Description   : Set the debug monitor exception to pending.
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_NVIC_DebugMon_SetPending()
{
    CoreDebug->DEMCR = (CoreDebug->DEMCR |
                       (0x1UL << CoreDebug_DEMCR_MON_PEND_Pos));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_NVIC_DebugMon_ClearPending()
 * ----------------------------------------------------------------------------
 * Description   : Clear the pending state of the debug monitor exception.
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_NVIC_DebugMon_ClearPending()
{
    CoreDebug->DEMCR = (CoreDebug->DEMCR & ~(CoreDebug_DEMCR_MON_PEND_Msk));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_NVIC_Int_EnableInt(IRQn_Type intnum)
 * ----------------------------------------------------------------------------
 * Description   : Enable the specified interrupt.  Directly calls
 *                 NVIC_EnableIRQ().  Included for backwards compatibility.
 * Inputs        : intnum - Interrupt to enable; use *_IRQn.  
 * Outputs       : None
 * Assumptions   : The interrupt specified is an external interrupt.
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_NVIC_Int_EnableInt(IRQn_Type intnum)
{
    NVIC_EnableIRQ(intnum);
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_NVIC_Int_DisableInt(IRQn_Type intnum)
 * ----------------------------------------------------------------------------
 * Description   : Disable the specified interrupt.  Directly calls
 *                 NVIC_DisableIRQ().  Included for backwards compatibility.
 * Inputs        : intnum - Interrupt to disable; use *_IRQn.  
 * Outputs       : None
 * Assumptions   : The interrupt specified is an external interrupt.
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_NVIC_Int_DisableInt(IRQn_Type intnum)
{
    NVIC_DisableIRQ(intnum);
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_NVIC_Int_IsActive(IRQn_Type intnum)
 * ----------------------------------------------------------------------------
 * Description   : Return whether the specified interrupt is active.  Directly
 *                 calls NVIC_GetActive().  Included for backwards 
 *                 compatibility.
 * Inputs        : None
 * Outputs       : return value - Active state of the specified interrupt.
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_NVIC_Int_IsActive(IRQn_Type intnum)
{
    return NVIC_GetActive(intnum);
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_NVIC_Int_IsPending(IRQn_Type intnum)
 * ----------------------------------------------------------------------------
 * Description   : Return whether the specified interrupt is pending.  Directly
 *                 calls NVIC_GetPendingIRQ().  Included for backwards
 *                 compatibility.
 * Inputs        : None
 * Outputs       : return value - Pending state of the specified interrupt;
 *                                value loaded from NVIC->ISPR.
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_NVIC_Int_IsPending(IRQn_Type intnum)
{
    return NVIC_GetPendingIRQ(intnum);
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_NVIC_Int_SetPending(IRQn_Type intnum)
 * ----------------------------------------------------------------------------
 * Description   : Set the specified interrupt to pending.  Directly calls
 *                 NVIC_SetPendingIRQ().  Included for backwards compatibility.
 * Inputs        : intnum - Interrupt to set pending; use *_IRQn.
 * Outputs       : None
 * Assumptions   : The interrupt specified is an external interrupt
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_NVIC_Int_SetPending(IRQn_Type intnum)
{
    NVIC_SetPendingIRQ(intnum);
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_NVIC_Int_ClearPending(IRQn_Type intnum)
 * ----------------------------------------------------------------------------
 * Description   : Clear the pending state of the specified interrupt.  Directly
 *                 calls NVIC_ClearPendingIRQ ().  Included for backwards 
 *                 compatibility.
 * Inputs        : intnum - Interrupt to clear the pending state of; use *_IRQn.
 * Outputs       : None
 * Assumptions   : The interrupt specified is an external interrupt
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_NVIC_Int_ClearPending(IRQn_Type intnum)
{
    NVIC_ClearPendingIRQ(intnum);
}

/* ----------------------------------------------------------------------------
 * Close the 'extern "C"' block
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
}
#endif

#endif /* Q32M200_NVIC_H */
