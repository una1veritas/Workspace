// new.h standard header -*-c++-*-
// Copyright 2009-2010 IAR Systems AB.
#ifndef _NEW_H_
#define _NEW_H_

#ifndef _SYSTEM_BUILD
  #pragma system_include
#endif

#include <new>

#if _HAS_NAMESPACE
  using std::bad_alloc;
  using std::new_handler;
  using std::nothrow;
  using std::nothrow_t;
  using std::set_new_handler;
#endif
#endif /* _NEW_ */

/*
 * Copyright (c) 1992-2009 by P.J. Plauger.  ALL RIGHTS RESERVED.
 * Consult your license regarding permissions and restrictions.
V5.04:0576 */
