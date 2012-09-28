/* xmtx.h internal header */
/* Copyright 2005-2010 IAR Systems AB. */
#ifndef _XMTX
#define _XMTX

#ifndef _SYSTEM_BUILD
  #pragma system_include
#endif

#include <ycheck.h>
#include <yvals.h>

#include <stdlib.h>

_C_LIB_DECL
#if !_MULTI_THREAD
  #define __iar_system_Mtxinit(mtx)
  #define __iar_system_Mtxdst(mtx)
  #define __iar_system_Mtxlock(mtx)
  #define __iar_system_Mtxunlock(mtx)

  #define __iar_file_Mtxinit(mtx)
  #define __iar_file_Mtxdst(mtx)
  #define __iar_file_Mtxlock(mtx)
  #define __iar_file_Mtxunlock(mtx)

  typedef char _Once_t;

  #define _Once(cntrl, func)    if (*(cntrl) == 0) (func)(), *(cntrl) = 2
  #define _ONCE_T_INIT  0
#endif /* _MULTI_THREAD */
_END_C_LIB_DECL
#endif /* _XMTX */

/*
 * Copyright (c) 1992-2009 by P.J. Plauger.  ALL RIGHTS RESERVED.
 * Consult your license regarding permissions and restrictions.
V5.04:0576 */
