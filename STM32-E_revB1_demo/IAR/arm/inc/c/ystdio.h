/* ystdio.h internal header */
/* Copyright 2009-2010 IAR Systems AB. */
#ifndef _YSTDIO
#define _YSTDIO

#ifndef _SYSTEM_BUILD
  #pragma system_include
#endif

#ifndef _YVALS
  #include <yvals.h>
#endif
_C_STD_BEGIN

#if _DLIB_FILE_DESCRIPTOR
  #ifndef _FD_TYPE
    #define _FD_TYPE      signed char
  #endif  /* _FD_TYPE */

  struct __FILE
  {       /* file control information */
    unsigned short _Mode;
    unsigned char _Lockno;
    _FD_TYPE _Handle;

    /* _Buf points to first byte in buffer. */
    /* _Bend points to one beyond last byte in buffer. */
    /* _Next points to next character to read or write. */
    unsigned char *_Buf, *_Bend, *_Next;
    /* _Rend points to one beyond last byte that can be read. */
    /* _Wend points to one beyond last byte that can be written. */
    /* _Rback points to last pushed back character in _Back. If it has value 
       one beyond the _Back array no pushed back chars exists. */ 
    unsigned char *_Rend, *_Wend, *_Rback;

    /* _WRback points to last pushed back wchar_t in _WBack. If it has value 
       one beyond the _WBack array no pushed back wchar_ts exists. */ 
    _Wchart *_WRback, _WBack[2];

    /* _Rsave holds _Rend if characters have been pushed back. */
    /* _WRend points to one byte beyond last wchar_t that can be read. */
    /* _WWend points to one byte beyond last wchar_t that can be written. */
    unsigned char *_Rsave, *_WRend, *_WWend;

    _Mbstatet _Wstate;
    char *_Tmpnam;
    unsigned char _Back[_MBMAX], _Cbuf;
  };

#endif /* _DLIB_FILE_DESCRIPTOR */

/* File system functions that have debug variants. They are agnostic on 
   whether the library is full or normal. */
_C_LIB_DECL
__ATTRIBUTES int remove(const char *);
__ATTRIBUTES int rename(const char *, const char *);
_END_C_LIB_DECL


_C_STD_END
#endif /* _YSTDIO */

#if defined(_STD_USING) && defined(__cplusplus)
  using _CSTD remove;
  using _CSTD rename; 
#endif /* defined(_STD_USING) && defined(__cplusplus) */

/*
 * Copyright (c) 1992-2009 by P.J. Plauger.  ALL RIGHTS RESERVED.
 * Consult your license regarding permissions and restrictions.
V5.042:0576 */
