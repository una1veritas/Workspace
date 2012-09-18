/* setjmp.h standard header wrapper */
/* Copyright 2005-2010 IAR Systems AB. */

#ifndef _SETJMP
#define _SETJMP

#ifndef _SYSTEM_BUILD
  #pragma system_include
#endif

#include <ycheck.h>
#include <yvals.h>

_C_STD_BEGIN

#ifndef __JMP_BUF_ELEMENT_TYPE__
  #ifndef __JMP_BUF_NUM_ELEMENTS__
    #define __JMP_BUF_ELEMENT_TYPE__  char
    #ifndef __JMP_BUF__
      #define __JMP_BUF_NUM_ELEMENTS__  8
    #else
      #define __JMP_BUF_NUM_ELEMENTS__  __JMP_BUF__
    #endif
  #endif
#endif

/* Setup in what type we measure the size of __aeabi_JMP_BUF_SIZE.
   Default is using the element type. */
#ifndef __JMP_BUF_ELEMENT_QUANTITY
  #define __JMP_BUF_ELEMENT_QUANTITY sizeof(__JMP_BUF_ELEMENT_TYPE__)
#endif

#pragma language=save
#pragma language=extended
#if __AEABI_PORTABILITY_INTERNAL_LEVEL
  #ifndef __AEABI_PORTABLE
    #define __AEABI_PORTABLE
  #endif

  typedef __JMP_BUF_ELEMENT_TYPE__ jmp_buf[];

  _C_LIB_DECL
  _DLIB_CONST_ATTR extern int const __aeabi_JMP_BUF_SIZE;
  _END_C_LIB_DECL

#else /* __AEABI_PORTABILITY_INTERNAL_LEVEL */
        /* MULTIBYTE PROPERTIES */

  typedef __JMP_BUF_ELEMENT_TYPE__  jmp_buf[__JMP_BUF_NUM_ELEMENTS__];

  #if _DLIB_USE_RTMODEL_JMP_BUF_NUM_ELEMENTS
    #pragma rtmodel = "__dlib_jmp_buf_num_elements",\
                      _STRINGIFY(__JMP_BUF_NUM_ELEMENTS__)
  #endif

#endif /* __AEABI_PORTABILITY_INTERNAL_LEVEL */
#pragma language=restore

_C_LIB_DECL
__ATTRIBUTES int setjmp(jmp_buf);
__ATTRIBUTES_NORETURN void longjmp(jmp_buf, int) _NO_RETURN;
_END_C_LIB_DECL
_C_STD_END
#endif /* _SETJMP */

#ifdef __cplusplus
  #define setjmp(x)  _CSTD setjmp(x)
#endif

#ifdef _STD_USING
  #if __AEABI_PORTABILITY_INTERNAL_LEVEL
      using _CSTD __aeabi_JMP_BUF_SIZE;
  #endif
  using _CSTD jmp_buf; using _CSTD longjmp;
  using _CSTD setjmp;
#endif /* _STD_USING */


/*
 * Copyright (c) 1992-2009 by P.J. Plauger.  ALL RIGHTS RESERVED.
 * Consult your license regarding permissions and restrictions.
V5.04:0576 */
