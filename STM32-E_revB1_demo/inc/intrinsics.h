/**************************************************
 *
 * This file declares the ARM intrinsic inline functions.
 *
 * Copyright 1999-2006 IAR Systems. All rights reserved.
 *
 * $Revision: 48618 $
 *
 **************************************************/

#ifndef __INTRINSICS_INCLUDED
#define __INTRINSICS_INCLUDED

#include <ycheck.h>

#ifndef _SYSTEM_BUILD
  #pragma system_include
#endif

/*
 * Check that the correct C compiler is used.
 */

#if ((__TID__ >> 8) & 0x007F) != 79
#error "File intrinsics.h can only be used together with iccarm."
#endif

#ifdef __cplusplus
extern "C" {
#endif

#pragma language=save
#pragma language=extended

__ATTRIBUTES void    __no_operation(void);

__ATTRIBUTES void    __disable_interrupt(void);
__ATTRIBUTES void    __enable_interrupt(void);

typedef unsigned long __istate_t;

__ATTRIBUTES __istate_t __get_interrupt_state(void);
__ATTRIBUTES void __set_interrupt_state(__istate_t);

#ifdef __ARM_PROFILE_M__

/* System control access for Cortex-M cores */
__ATTRIBUTES unsigned long __get_PSR( void );
__ATTRIBUTES unsigned long __get_IPSR( void );
__ATTRIBUTES unsigned long __get_MSP( void );
__ATTRIBUTES void          __set_MSP( unsigned long );
__ATTRIBUTES unsigned long __get_PSP( void );
__ATTRIBUTES void          __set_PSP( unsigned long );
__ATTRIBUTES unsigned long __get_PRIMASK( void );
__ATTRIBUTES void          __set_PRIMASK( unsigned long );
__ATTRIBUTES unsigned long __get_CONTROL( void );
__ATTRIBUTES void          __set_CONTROL( unsigned long );

#ifdef __ARM7__

/* These are only available for v7M */
__ATTRIBUTES unsigned long __get_FAULTMASK( void );
__ATTRIBUTES void          __set_FAULTMASK(unsigned long);
__ATTRIBUTES unsigned long __get_BASEPRI( void );
__ATTRIBUTES void          __set_BASEPRI( unsigned long );

#endif /* __ARM7__ */
#else /* __ARM_PROFILE_M__ */

/* "Old" style intrerrupt control routines */
__ATTRIBUTES void __disable_irq(void);
__ATTRIBUTES void __enable_irq(void);
#endif /* __ARM_PROFILE_M__ */

__ATTRIBUTES void __disable_fiq(void);
__ATTRIBUTES void __enable_fiq(void);


/* ARM-mode intrinsics */

__ATTRIBUTES unsigned long __SWP( unsigned long, unsigned long * );
__ATTRIBUTES unsigned char __SWPB( unsigned char, unsigned char * );

typedef unsigned long __ul;

#define __cpid  unsigned __constrange(0,15)
#define __cpreg unsigned __constrange(0,15)
#define __cpopc unsigned __constrange(0,8)

/*  Co-processor access */
__ATTRIBUTES void          __MCR( __cpid coproc, __cpopc opcode_1, __ul src,
                                 __cpreg CRn, __cpreg CRm, __cpopc opcode_2 );
__ATTRIBUTES unsigned long __MRC( __cpid coproc, __cpopc opcode_1,
                                 __cpreg CRn, __cpreg CRm, __cpopc opcode_2 );
__ATTRIBUTES void          __MCR2( __cpid coproc, __cpopc opcode_1, __ul src,
                                  __cpreg CRn, __cpreg CRm, __cpopc opcode_2 );
__ATTRIBUTES unsigned long __MRC2( __cpid coproc, __cpopc opcode_1,
                                  __cpreg CRn, __cpreg CRm, __cpopc opcode_2 );

/* Load coprocessor register. */
__ATTRIBUTES void __LDC( __cpid coproc, __cpreg CRn, __ul const *src);
__ATTRIBUTES void __LDCL( __cpid coproc, __cpreg CRn, __ul const *src);
__ATTRIBUTES void __LDC2( __cpid coproc, __cpreg CRn, __ul const *src);
__ATTRIBUTES void __LDC2L( __cpid coproc, __cpreg CRn, __ul const *src);

/* Store coprocessor register. */
__ATTRIBUTES void __STC( __cpid coproc, __cpreg CRn, __ul *dst);
__ATTRIBUTES void __STCL( __cpid coproc, __cpreg CRn, __ul *dst);
__ATTRIBUTES void __STC2( __cpid coproc, __cpreg CRn, __ul *dst);
__ATTRIBUTES void __STC2L( __cpid coproc, __cpreg CRn, __ul *dst);

/* Load coprocessor register (noindexed version with coprocessor option). */
__ATTRIBUTES void __LDC_noidx( __cpid coproc, __cpreg CRn, __ul const *src,
                              unsigned __constrange(0,255) option);

__ATTRIBUTES void __LDCL_noidx( __cpid coproc, __cpreg CRn, __ul const *src,
                               unsigned __constrange(0,255) option);

__ATTRIBUTES void __LDC2_noidx( __cpid coproc, __cpreg CRn, __ul const *src,
                               unsigned __constrange(0,255) option);

__ATTRIBUTES void __LDC2L_noidx( __cpid coproc, __cpreg CRn, __ul const *src,
                                unsigned __constrange(0,255) option);

/* Store coprocessor register (version with coprocessor option). */
__ATTRIBUTES void __STC_noidx( __cpid coproc, __cpreg CRn, __ul *dst,
                              unsigned __constrange(0,255) option);

__ATTRIBUTES void __STCL_noidx( __cpid coproc, __cpreg CRn, __ul *dst,
                               unsigned __constrange(0,255) option);

__ATTRIBUTES void __STC2_noidx( __cpid coproc, __cpreg CRn, __ul *dst,
                               unsigned __constrange(0,255) option);

__ATTRIBUTES void __STC2L_noidx( __cpid coproc, __cpreg CRn, __ul *dst,
                                unsigned __constrange(0,255) option);

#ifdef __ARM_PROFILE_M__
/* Status register access, v7M: */
__ATTRIBUTES unsigned long __get_APSR( void );
__ATTRIBUTES void          __set_APSR( unsigned long );
#else /* __ARM_PROFILE_M__ */
/* Status register access */
__ATTRIBUTES unsigned long __get_CPSR( void );
__ATTRIBUTES void          __set_CPSR( unsigned long );
#endif /* __ARM_PROFILE_M__ */

/* Floating-point status and control register access */
__ATTRIBUTES unsigned long __get_FPSCR( void );
__ATTRIBUTES void __set_FPSCR( unsigned long );

/* Architecture v5T, CLZ is also available in Thumb mode for Thumb2 cores */
__ATTRIBUTES unsigned char __CLZ( unsigned long );

/* Architecture v5TE */
#if !defined(__ARM_PROFILE_M__) || defined(__ARM_MEDIA__)
__ATTRIBUTES signed long __QADD( signed long, signed long );
__ATTRIBUTES signed long __QDADD( signed long, signed long );
__ATTRIBUTES signed long __QSUB( signed long, signed long );
__ATTRIBUTES signed long __QDSUB( signed long, signed long );

__ATTRIBUTES signed long __QDOUBLE( signed long );

__ATTRIBUTES int         __QFlag( void );
__ATTRIBUTES void __reset_Q_flag( void );
#endif

__ATTRIBUTES int         __QCFlag( void );
__ATTRIBUTES void __reset_QC_flag( void );

__ATTRIBUTES signed long __SMUL( signed short, signed short );

/* Architecture v6, REV and REVSH are also available in thumb mode */
__ATTRIBUTES unsigned long __REV( unsigned long );
__ATTRIBUTES signed long __REVSH( short );

__ATTRIBUTES unsigned long __REV16( unsigned long );
__ATTRIBUTES unsigned long __RBIT( unsigned long );

__ATTRIBUTES unsigned char  __LDREXB( unsigned char * );
__ATTRIBUTES unsigned short __LDREXH( unsigned short * );
__ATTRIBUTES unsigned long  __LDREX ( unsigned long * );
__ATTRIBUTES unsigned long long __LDREXD( unsigned long long * );

__ATTRIBUTES unsigned long  __STREXB( unsigned char, unsigned char * );
__ATTRIBUTES unsigned long  __STREXH( unsigned short, unsigned short * );
__ATTRIBUTES unsigned long  __STREX ( unsigned long, unsigned long * );
__ATTRIBUTES unsigned long  __STREXD( unsigned long long, unsigned long long * );

__ATTRIBUTES void __CLREX( void );

__ATTRIBUTES void __SEV( void );
__ATTRIBUTES void __WFE( void );
__ATTRIBUTES void __WFI( void );
__ATTRIBUTES void __YIELD( void );

__ATTRIBUTES void __PLI( void const * );
__ATTRIBUTES void __PLD( void const * );
__ATTRIBUTES void __PLDW( void const * );

__ATTRIBUTES unsigned long __SSAT     (unsigned long val,
                                      unsigned int __constrange( 1, 32 ) sat );
__ATTRIBUTES unsigned long __USAT     (unsigned long val,
                                      unsigned int __constrange( 0, 31 ) sat );


#ifdef __ARM_MEDIA__

/* Architecture v6 Media instructions.... */
__ATTRIBUTES unsigned long __SEL( unsigned long op1, unsigned long op2 );

__ATTRIBUTES unsigned long __SADD8    (unsigned long pair1, unsigned long pair2);
__ATTRIBUTES unsigned long __SADD16   (unsigned long pair1, unsigned long pair2);
__ATTRIBUTES unsigned long __SSUB8    (unsigned long pair1, unsigned long pair2);
__ATTRIBUTES unsigned long __SSUB16   (unsigned long pair1, unsigned long pair2);
__ATTRIBUTES unsigned long __SADDSUBX (unsigned long pair1, unsigned long pair2);
__ATTRIBUTES unsigned long __SSUBADDX (unsigned long pair1, unsigned long pair2);

__ATTRIBUTES unsigned long __SHADD8   (unsigned long pair1, unsigned long pair2);
__ATTRIBUTES unsigned long __SHADD16  (unsigned long pair1, unsigned long pair2);
__ATTRIBUTES unsigned long __SHSUB8   (unsigned long pair1, unsigned long pair2);
__ATTRIBUTES unsigned long __SHSUB16  (unsigned long pair1, unsigned long pair2);
__ATTRIBUTES unsigned long __SHADDSUBX(unsigned long pair1, unsigned long pair2);
__ATTRIBUTES unsigned long __SHSUBADDX(unsigned long pair1, unsigned long pair2);

__ATTRIBUTES unsigned long __QADD8    (unsigned long pair1, unsigned long pair2);
__ATTRIBUTES unsigned long __QADD16   (unsigned long pair1, unsigned long pair2);
__ATTRIBUTES unsigned long __QSUB8    (unsigned long pair1, unsigned long pair2);
__ATTRIBUTES unsigned long __QSUB16   (unsigned long pair1, unsigned long pair2);
__ATTRIBUTES unsigned long __QADDSUBX (unsigned long pair1, unsigned long pair2);
__ATTRIBUTES unsigned long __QSUBADDX (unsigned long pair1, unsigned long pair2);

__ATTRIBUTES unsigned long __UADD8    (unsigned long pair1, unsigned long pair2);
__ATTRIBUTES unsigned long __UADD16   (unsigned long pair1, unsigned long pair2);
__ATTRIBUTES unsigned long __USUB8    (unsigned long pair1, unsigned long pair2);
__ATTRIBUTES unsigned long __USUB16   (unsigned long pair1, unsigned long pair2);
__ATTRIBUTES unsigned long __UADDSUBX (unsigned long pair1, unsigned long pair2);
__ATTRIBUTES unsigned long __USUBADDX (unsigned long pair1, unsigned long pair2);

__ATTRIBUTES unsigned long __UHADD8   (unsigned long pair1, unsigned long pair2);
__ATTRIBUTES unsigned long __UHADD16  (unsigned long pair1, unsigned long pair2);
__ATTRIBUTES unsigned long __UHSUB8   (unsigned long pair1, unsigned long pair2);
__ATTRIBUTES unsigned long __UHSUB16  (unsigned long pair1, unsigned long pair2);
__ATTRIBUTES unsigned long __UHADDSUBX(unsigned long pair1, unsigned long pair2);
__ATTRIBUTES unsigned long __UHSUBADDX(unsigned long pair1, unsigned long pair2);

__ATTRIBUTES unsigned long __UQADD8   (unsigned long pair1, unsigned long pair2);
__ATTRIBUTES unsigned long __UQADD16  (unsigned long pair1, unsigned long pair2);
__ATTRIBUTES unsigned long __UQSUB8   (unsigned long pair1, unsigned long pair2);
__ATTRIBUTES unsigned long __UQSUB16  (unsigned long pair1, unsigned long pair2);
__ATTRIBUTES unsigned long __UQADDSUBX(unsigned long pair1, unsigned long pair2);
__ATTRIBUTES unsigned long __UQSUBADDX(unsigned long pair1, unsigned long pair2);

__ATTRIBUTES unsigned long __USAD8(unsigned long x, unsigned long y );
__ATTRIBUTES unsigned long __USADA8(unsigned long x, unsigned long y,
                                   unsigned long acc );

__ATTRIBUTES unsigned long __SSAT16   (unsigned long pair,
                                      unsigned int __constrange( 1, 16 ) sat );
__ATTRIBUTES unsigned long __USAT16   (unsigned long pair,
                                      unsigned int __constrange( 0, 15 ) sat );

__ATTRIBUTES unsigned long __SMUAD(unsigned long x, unsigned long y);
__ATTRIBUTES unsigned long __SMUSD(unsigned long x, unsigned long y);
__ATTRIBUTES unsigned long __SMUADX(unsigned long x, unsigned long y);
__ATTRIBUTES unsigned long __SMUSDX(unsigned long x, unsigned long y);

__ATTRIBUTES unsigned long __SMLAD(unsigned long x, unsigned long y, unsigned long sum);
__ATTRIBUTES unsigned long __SMLSD(unsigned long x, unsigned long y, unsigned long sum);
__ATTRIBUTES unsigned long __SMLADX(unsigned long x, unsigned long y, unsigned long sum);
__ATTRIBUTES unsigned long __SMLSDX(unsigned long x, unsigned long y, unsigned long sum);

__ATTRIBUTES unsigned long long __SMLALD ( unsigned long pair1,
                                          unsigned long pair2,
                                          unsigned long long acc );

__ATTRIBUTES unsigned long long __SMLALDX( unsigned long pair1,
                                          unsigned long pair2,
                                          unsigned long long acc );

__ATTRIBUTES unsigned long long __SMLSLD ( unsigned long pair1,
                                          unsigned long pair2,
                                          unsigned long long acc );

__ATTRIBUTES unsigned long long __SMLSLDX( unsigned long pair1,
                                          unsigned long pair2,
                                          unsigned long long acc );

__ATTRIBUTES unsigned long __PKHBT(unsigned long x,
                                  unsigned long y,
                                  unsigned __constrange(0,31) count);
__ATTRIBUTES unsigned long __PKHTB(unsigned long x,
                                  unsigned long y,
                                  unsigned __constrange(0,32) count);

__ATTRIBUTES unsigned long __SMLABB(unsigned long x,
                                   unsigned long y,
                                   unsigned long acc);
__ATTRIBUTES unsigned long __SMLABT(unsigned long x,
                                   unsigned long y,
                                   unsigned long acc);
__ATTRIBUTES unsigned long __SMLATB(unsigned long x,
                                   unsigned long y,
                                   unsigned long acc);
__ATTRIBUTES unsigned long __SMLATT(unsigned long x,
                                   unsigned long y,
                                   unsigned long acc);
__ATTRIBUTES unsigned long __SMLAWB(unsigned long x,
                                   unsigned long y,
                                   unsigned long acc);
__ATTRIBUTES unsigned long __SMLAWT(unsigned long x,
                                   unsigned long y,
                                   unsigned long acc);

__ATTRIBUTES unsigned long __SMMLA (unsigned long x,
                                   unsigned long y,
                                   unsigned long acc);
__ATTRIBUTES unsigned long __SMMLAR(unsigned long x,
                                   unsigned long y,
                                   unsigned long acc);
__ATTRIBUTES unsigned long __SMMLS (unsigned long x,
                                   unsigned long y,
                                   unsigned long acc);
__ATTRIBUTES unsigned long __SMMLSR(unsigned long x,
                                   unsigned long y,
                                   unsigned long acc);
__ATTRIBUTES unsigned long __SMMUL (unsigned long x, unsigned long y);
__ATTRIBUTES unsigned long __SMMULR(unsigned long x, unsigned long y);

__ATTRIBUTES unsigned long __SMULBB(unsigned long x, unsigned long y);
__ATTRIBUTES unsigned long __SMULBT(unsigned long x, unsigned long y);
__ATTRIBUTES unsigned long __SMULTB(unsigned long x, unsigned long y);
__ATTRIBUTES unsigned long __SMULTT(unsigned long x, unsigned long y);
__ATTRIBUTES unsigned long __SMULWB(unsigned long x, unsigned long y);
__ATTRIBUTES unsigned long __SMULWT(unsigned long x, unsigned long y);

__ATTRIBUTES unsigned long __SXTAB (unsigned long x, unsigned long y);
__ATTRIBUTES unsigned long __SXTAH (unsigned long x, unsigned long y);
__ATTRIBUTES unsigned long __UXTAB (unsigned long x, unsigned long y);
__ATTRIBUTES unsigned long __UXTAH (unsigned long x, unsigned long y);

__ATTRIBUTES unsigned long long __UMAAL(unsigned long x,
                                       unsigned long y,
                                       unsigned long a,
                                       unsigned long b);

__ATTRIBUTES unsigned long long __SMLALBB(unsigned long x,
                                         unsigned long y,
                                         unsigned long long acc);
__ATTRIBUTES unsigned long long __SMLALBT(unsigned long x,
                                         unsigned long y,
                                         unsigned long long acc);
__ATTRIBUTES unsigned long long __SMLALTB(unsigned long x,
                                         unsigned long y,
                                         unsigned long long acc);
__ATTRIBUTES unsigned long long __SMLALTT(unsigned long x,
                                         unsigned long y,
                                         unsigned long long acc);

__ATTRIBUTES unsigned long __UXTB16(unsigned long x);
__ATTRIBUTES unsigned long __UXTAB16(unsigned long acc, unsigned long x);

__ATTRIBUTES unsigned long __SXTB16(unsigned long x);
__ATTRIBUTES unsigned long __SXTAB16(unsigned long acc, unsigned long x);

/*
 * The following intrinsic function aliases are supplied due to instruction
 * set changes made by ARM. All previous mnemonics of the form *ADDSUBX or
 * *SUBADDX have been changed into *ASX and *SAX, respectively.
 * For increased readability and consistency, the suggested intrinsic
 * functions to use are the new short form versions defined below.
 */

#define __SASX  __SADDSUBX
#define __SSAX  __SSUBADDX
#define __SHASX __SHADDSUBX
#define __SHSAX __SHSUBADDX
#define __QASX  __QADDSUBX
#define __QSAX  __QSUBADDX

#define __UASX  __UADDSUBX
#define __USAX  __USUBADDX
#define __UHASX __UHADDSUBX
#define __UHSAX __UHSUBADDX
#define __UQASX __UQADDSUBX
#define __UQSAX __UQSUBADDX
#endif /* __ARM_MEDIA__ */

/* Architecture v7 instructions.... */
__ATTRIBUTES void __DMB(void);
__ATTRIBUTES void __DSB(void);
__ATTRIBUTES void __ISB(void);

#define __fabs(x)  fabs(x)
#define __fabsf(x) fabsf(x)

#pragma language=restore

#ifdef __cplusplus
}
#endif

#endif  /* __INTRINSICS_INCLUDED */
