/* string.h standard header */
/* Copyright 2009-2010 IAR Systems AB. */
#ifndef _STRING
#define _STRING

#ifndef _SYSTEM_BUILD
  #pragma system_include
#endif

#include <ycheck.h>
#include <yvals.h>
#include <ysizet.h>

#if _DLIB_PRODUCT_STRING
#include <DLib_Product_string.h>
#endif

_C_STD_BEGIN
                /* macros */
#ifndef NULL
  #define NULL  _NULL
#endif /* NULL */

                /* declarations */
_C_LIB_DECL
__EFF_NENW1NW2   __ATTRIBUTES int        memcmp(const void *, const void *,
                                                size_t);
__EFF_NENR1NW2R1 __ATTRIBUTES void *     memcpy(void *_Restrict, 
                                                const void *_Restrict, size_t);
__EFF_NENR1NW2R1 __ATTRIBUTES void *     memmove(void *, const void *, size_t);
__EFF_NENR1R1    __ATTRIBUTES void *     memset(void *, int, size_t);
__EFF_NENW2R1    __ATTRIBUTES char *     strcat(char *_Restrict, 
                                                const char *_Restrict);
__EFF_NENW1NW2   __ATTRIBUTES int        strcmp(const char *, const char *);
__EFF_NW1NW2     __ATTRIBUTES int        strcoll(const char *, const char *);
__EFF_NENR1NW2R1 __ATTRIBUTES char *     strcpy(char *_Restrict, 
                                                const char *_Restrict);
__EFF_NENW1NW2   __ATTRIBUTES size_t     strcspn(const char *, const char *);
                 __ATTRIBUTES char *     strerror(int);
__EFF_NENW1      __ATTRIBUTES size_t     strlen(const char *);
__EFF_NENW2R1    __ATTRIBUTES char *     strncat(char *_Restrict, 
                                                 const char *_Restrict, size_t);
__EFF_NENW1NW2   __ATTRIBUTES int        strncmp(const char *, const char *, 
                                                 size_t);
__EFF_NENR1NW2R1 __ATTRIBUTES char *     strncpy(char *_Restrict, 
                                                 const char *_Restrict, size_t);
__EFF_NENW1NW2   __ATTRIBUTES size_t     strspn(const char *, const char *);
__EFF_NW2        __ATTRIBUTES char *     strtok(char *_Restrict, 
                                                const char *_Restrict);
__EFF_NW2        __ATTRIBUTES size_t     strxfrm(char *_Restrict, 
                                                 const char *_Restrict, size_t);

#if _DLIB_ADD_EXTRA_SYMBOLS
  __EFF_NW1      __ATTRIBUTES char *   strdup(const char *);
  __EFF_NW1NW2   __ATTRIBUTES int      strcasecmp(const char *, const char *);
  __EFF_NW1NW2   __ATTRIBUTES int      strncasecmp(const char *, const char *, 
                                                   size_t);
  __EFF_NENW2    __ATTRIBUTES char *   strtok_r(char *, const char *, char **);
  __EFF_NENW1    __ATTRIBUTES size_t   strnlen(char const *, size_t);
#endif /* _DLIB_ADD_EXTRA_SYMBOLS */

_END_C_LIB_DECL

#ifdef __cplusplus
_EXTERN_CPP
  __EFF_NENW1    __ATTRIBUTES const void *memchr(const void *_S, int _C, 
                                                 size_t _N);
  __EFF_NENW1    __ATTRIBUTES const char *strchr(const char *_S, int _C);
  __EFF_NENW1NW2 __ATTRIBUTES const char *strpbrk(const char *_S, 
                                                  const char *_P);
  __EFF_NENW1    __ATTRIBUTES const char *strrchr(const char *_S, int _C);
  __EFF_NENW1NW2 __ATTRIBUTES const char *strstr(const char *_S, 
                                                 const char *_P);
  __EFF_NENW1    __ATTRIBUTES void *      memchr(void *_S, int _C, size_t _N);
  __EFF_NENW1    __ATTRIBUTES char *      strchr(char *_S, int _C);
  __EFF_NENW1NW2 __ATTRIBUTES char *      strpbrk(char *_S, const char *_P);
  __EFF_NENW1    __ATTRIBUTES char *      strrchr(char *_S, int _C);
  __EFF_NENW1NW2 __ATTRIBUTES char *      strstr(char *_S, const char *_P);
_END_EXTERN_CPP
#else /* !__cplusplus */
  __EFF_NENW1    __ATTRIBUTES void *memchr(const void *_S, int _C, size_t _N);
  __EFF_NENW1    __ATTRIBUTES char *strchr(const char *_S, int _C);
  __EFF_NENW1NW2 __ATTRIBUTES char *strpbrk(const char *_S, const char *_P);
  __EFF_NENW1    __ATTRIBUTES char *strrchr(const char *_S, int _C);
  __EFF_NENW1NW2 __ATTRIBUTES char *strstr(const char *_S, const char *_P);
#endif /* __cplusplus */


#ifndef _NO_DEFINITIONS_IN_HEADER_FILES
                /* Inline definitions. */

#if !__AEABI_PORTABILITY_INTERNAL_LEVEL
                /* The implementations. */
_C_LIB_DECL
__EFF_NENW1    __ATTRIBUTES void *__iar_Memchr(const void *, int, size_t);
__EFF_NENW1    __ATTRIBUTES char *__iar_Strchr(const char *, int);
               __ATTRIBUTES char *__iar_Strerror(int, char *);
__EFF_NENW1NW2 __ATTRIBUTES char *__iar_Strpbrk(const char *, const char *);
__EFF_NENW1    __ATTRIBUTES char *__iar_Strrchr(const char *, int);
__EFF_NENW1NW2 __ATTRIBUTES char *__iar_Strstr(const char *, const char *);
_END_C_LIB_DECL

                /* inlines and overloads, for C and C++ */
  #ifdef __cplusplus
  _EXTERN_CPP
                /* First the const overloads for C++. */
    #pragma inline
    const void *memchr(const void *_S, int _C, size_t _N)
    {
      return (__iar_Memchr(_S, _C, _N));
    }

    #pragma inline
    const char *strchr(const char *_S, int _C)
    {
      return (__iar_Strchr(_S, _C));
    }

    #pragma inline
    const char *strpbrk(const char *_S, const char *_P)
    {
      return (__iar_Strpbrk(_S, _P));
    }

    #pragma inline
    const char *strrchr(const char *_S, int _C)
    {
      return (__iar_Strrchr(_S, _C));
    }

    #pragma inline
    const char *strstr(const char *_S, const char *_P)
    {
      return (__iar_Strstr(_S, _P));
    }
                /* Then the non-const overloads for C++. */
    #pragma inline
    void *memchr(void *_S, int _C, size_t _N)
    {
      return (__iar_Memchr(_S, _C, _N));
    }

    #pragma inline
    char *strchr(char *_S, int _C)
    {
      return (__iar_Strchr(_S, _C));
    }

    #pragma inline
    char *strpbrk(char *_S, const char *_P)
    {
      return (__iar_Strpbrk(_S, _P));
    }

    #pragma inline
    char *strrchr(char *_S, int _C)
    {
      return (__iar_Strrchr(_S, _C));
    }

    #pragma inline
    char *strstr(char *_S, const char *_P)
    {
      return (__iar_Strstr(_S, _P));
    }
  _END_EXTERN_CPP
  #else /* !__cplusplus */
                /* Then the overloads for C. */
    #pragma inline
    void *memchr(const void *_S, int _C, size_t _N)
    {
      return (__iar_Memchr(_S, _C, _N));
    }

    #pragma inline
    char *strchr(const char *_S, int _C)
    {
      return (__iar_Strchr(_S, _C));
    }

    #pragma inline
    char *strpbrk(const char *_S, const char *_P)
    {
      return (__iar_Strpbrk(_S, _P));
    }

    #pragma inline
    char *strrchr(const char *_S, int _C)
    {
      return (__iar_Strrchr(_S, _C));
    }

    #pragma inline
    char *strstr(const char *_S, const char *_P)
    {
      return (__iar_Strstr(_S, _P));
    }
  #endif /* __cplusplus */

  #pragma inline
  char *strerror(int _Err)
  {
    return (__iar_Strerror(_Err, 0));
  }

  #ifdef _STRING_MORE_INLINES

    #ifndef _DLIB_STRING_SKIP_INLINE_MEMCMP
      #pragma inline
      int memcmp(const void *s1, const void *s2, size_t n)
      {       /* compare unsigned char s1[n], s2[n] */
        const unsigned char *su1 = (const unsigned char *)s1;
        const unsigned char *su2 = (const unsigned char *)s2;

        for (; 0 < n; ++su1, ++su2, --n)
          if (*su1 != *su2)
            return (*su1 < *su2 ? -1 : +1);
        return (0);
      }
    #endif /* _DLIB_STRING_SKIP_INLINE_MEMCMP */

    #ifndef _DLIB_STRING_SKIP_INLINE_MEMCPY
      #pragma inline
      void *memcpy(void *_Restrict s1, const void *_Restrict s2, size_t n)
      {       /* copy char s2[n] to s1[n] in any order */
        char *su1 = (char *)s1;
        const char *su2 = (const char *)s2;

        for (; 0 < n; ++su1, ++su2, --n)
          *su1 = *su2;
        return (s1);
      }
    #endif /* _DLIB_STRING_SKIP_INLINE_MEMCPY */

    #ifndef _DLIB_STRING_SKIP_INLINE_MEMSET
      #pragma inline
      void *memset(void *s, int c, size_t n)
      {       /* store c throughout unsigned char s[n] */
        const unsigned char uc = (unsigned char)c;
        unsigned char *su = (unsigned char *)s;

        for (; 0 < n; ++su, --n)
          *su = uc;
        return (s);
      }
    #endif /* _DLIB_STRING_SKIP_INLINE_MEMSET */

    #ifndef _DLIB_STRING_SKIP_INLINE_STRCAT
      #pragma inline
      char *strcat(char *_Restrict s1, const char *_Restrict s2)
      {       /* copy char s2[] to end of s1[] */
        char *s;

        for (s = s1; *s != '\0'; ++s)
          ;                   /* find end of s1[] */
        for (; (*s = *s2) != '\0'; ++s, ++s2)
          ;                   /* copy s2[] to end */
        return (s1);
      }
    #endif /* _DLIB_STRING_SKIP_INLINE_STRCAT */

    #ifndef _DLIB_STRING_SKIP_INLINE_STRCMP
      #pragma inline
      int strcmp(const char *s1, const char *s2)
      {       /* compare unsigned char s1[], s2[] */
        for (; *s1 == *s2; ++s1, ++s2)
          if (*s1 == '\0')
            return (0);
        return (*(unsigned char *)s1 < *(unsigned char *)s2
                ? -1 : +1);
      }
    #endif /* _DLIB_STRING_SKIP_INLINE_STRCMP */

    #ifndef _DLIB_STRING_SKIP_INLINE_STRCPY
      #pragma inline
      char *strcpy(char *_Restrict s1, const char *_Restrict s2)
      {       /* copy char s2[] to s1[] */
        char *s = s1;

        for (s = s1; (*s++ = *s2++) != '\0'; )
          ;
        return (s1);
      }
    #endif /* _DLIB_STRING_SKIP_INLINE_STRCPY */

    #ifndef _DLIB_STRING_SKIP_INLINE_STRCSPN
      #pragma inline
      size_t strcspn(const char *s1, const char *s2)
      {       /* find index of first s1[i] that matches any s2[] */
        const char *sc1, *sc2;

        for (sc1 = s1; *sc1 != '\0'; ++sc1)
          for (sc2 = s2; *sc2 != '\0'; ++sc2)
            if (*sc1 == *sc2)
              return (sc1 - s1);
        return (sc1 - s1);    /* terminating nulls match */
      }
    #endif /* _DLIB_STRING_SKIP_INLINE_STRCSPN */

    #ifndef _DLIB_STRING_SKIP_INLINE_STRLEN
      #pragma inline
      size_t strlen(const char *s)
      {       /* find length of s[] */
        const char *sc;

        for (sc = s; *sc != '\0'; ++sc)
          ;
        return (sc - s);
      }
    #endif /* _DLIB_STRING_SKIP_INLINE_STRLEN */

    #ifndef _DLIB_STRING_SKIP_INLINE_STRNCAT
      #pragma inline
      char *strncat(char *_Restrict s1, const char *_Restrict s2, size_t n)
      {       /* copy char s2[max n] to end of s1[] */
        char *s;

        for (s = s1; *s != '\0'; ++s)
          ;   /* find end of s1[] */
        for (; 0 < n && *s2 != '\0'; --n)
          *s++ = *s2++;       /* copy at most n chars from s2[] */
        *s = '\0';
        return (s1);
      }
    #endif /* _DLIB_STRING_SKIP_INLINE_STRNCAT */

    #ifndef _DLIB_STRING_SKIP_INLINE_STRNCMP
      #pragma inline
      int strncmp(const char *s1, const char *s2, size_t n)
      {       /* compare unsigned char s1[max n], s2[max n] */
        for (; 0 < n; ++s1, ++s2, --n)
          if (*s1 != *s2)
            return (  *(unsigned char *)s1
                    < *(unsigned char *)s2 ? -1 : +1);
          else if (*s1 == '\0')
            return (0);
        return (0);
      }
    #endif /* _DLIB_STRING_SKIP_INLINE_STRNCMP */

    #ifndef _DLIB_STRING_SKIP_INLINE_STRNCPY
      #pragma inline
      char *strncpy(char *_Restrict s1, const char *_Restrict s2, size_t n)
      {       /* copy char s2[max n] to s1[n] */
        char *s;

        for (s = s1; 0 < n && *s2 != '\0'; --n)
          *s++ = *s2++;       /* copy at most n chars from s2[] */
        for (; 0 < n; --n)
          *s++ = '\0';
        return (s1);
      }
    #endif /* _DLIB_STRING_SKIP_INLINE_STRNCPY */

    #ifndef _DLIB_STRING_SKIP_INLINE_STRSPN
      #pragma inline
      size_t strspn(const char *s1, const char *s2)
      {       /* find index of first s1[i] that matches no s2[] */
        const char *sc1, *sc2;

        for (sc1 = s1; *sc1 != '\0'; ++sc1)
          for (sc2 = s2; ; ++sc2)
            if (*sc2 == '\0')
              return (sc1 - s1);
            else if (*sc1 == *sc2)
              break;
        return (sc1 - s1);    /* null doesn't match */
      }
    #endif /* _DLIB_STRING_SKIP_INLINE_STRSPN */

    #ifndef _DLIB_STRING_SKIP_INLINE_MEMCHR
      #pragma inline
      void *__iar_Memchr(const void *s, int c, size_t n)
      {       /* find first occurrence of c in s[n] */
        const unsigned char uc = (unsigned char)c;
        const unsigned char *su = (const unsigned char *)s;

        for (; 0 < n; ++su, --n)
          if (*su == uc)
            return ((void *)su);
        return (0);
      }
    #endif /* _DLIB_STRING_SKIP_INLINE_MEMCHR */

    #ifndef _DLIB_STRING_SKIP_INLINE_STRCHR
      #pragma inline
      char *__iar_Strchr(const char *s, int c)
      {       /* find first occurrence of c in char s[] */
        const char ch = (char)c;

        for (; *s != ch; ++s)
          if (*s == '\0')
            return (0);
        return ((char *)s);
      }
    #endif /* _DLIB_STRING_SKIP_INLINE_STRCHR */

    #ifndef _DLIB_STRING_SKIP_INLINE_STRPBRK
      #pragma inline
      char *__iar_Strpbrk(const char *s1, const char *s2)
      {       /* find index of first s1[i] that matches any s2[] */
        const char *sc1, *sc2;

        for (sc1 = s1; *sc1 != '\0'; ++sc1)
          for (sc2 = s2; *sc2 != '\0'; ++sc2)
            if (*sc1 == *sc2)
              return ((char *)sc1);
        return (0);   /* terminating nulls match */
      }
    #endif /* _DLIB_STRING_SKIP_INLINE_STRPBRK */

    #ifndef _DLIB_STRING_SKIP_INLINE_STRRCHR
      #pragma inline
      char *__iar_Strrchr(const char *s, int c)
      {       /* find last occurrence of c in char s[] */
        const char ch = (char)c;
        const char *sc;

        for (sc = 0; ; ++s)
        {     /* check another char */
          if (*s == ch)
            sc = s;
          if (*s == '\0')
            return ((char *)sc);
        }
      }
    #endif /* _DLIB_STRING_SKIP_INLINE_STRRCHR */

    #ifndef _DLIB_STRING_SKIP_INLINE_STRSTR
      #pragma inline
      char *__iar_Strstr(const char *s1, const char *s2)
      {       /* find first occurrence of s2[] in s1[] */
        if (*s2 == '\0')
          return ((char *)s1);
        for (; (s1 = __iar_Strchr(s1, *s2)) != 0; ++s1)
        {     /* match rest of prefix */
          const char *sc1, *sc2;

          for (sc1 = s1, sc2 = s2; ; )
            if (*++sc2 == '\0')
              return ((char *)s1);
            else if (*++sc1 != *sc2)
              break;
        }
        return (0);
      }
    #endif /* _DLIB_STRING_SKIP_INLINE_STRSTR */

  #endif /* _STRING_MORE_INLINES */
#endif /* !__AEABI_PORTABILITY_INTERNAL_LEVEL */

#endif /* _NO_DEFINITIONS_IN_HEADER_FILES */

_C_STD_END
#endif /* _STRING */

#if defined(_STD_USING) && defined(__cplusplus)
  using _CSTD memchr; using _CSTD memcmp;
  using _CSTD memcpy; using _CSTD memmove; using _CSTD memset;
  using _CSTD strcat; using _CSTD strchr; using _CSTD strcmp;
  using _CSTD strcoll; using _CSTD strcpy; using _CSTD strcspn;
  using _CSTD strerror; using _CSTD strlen; using _CSTD strncat;
  using _CSTD strncmp; using _CSTD strncpy; using _CSTD strpbrk;
  using _CSTD strrchr; using _CSTD strspn; using _CSTD strstr;
  using _CSTD strtok; using _CSTD strxfrm;
  #if _DLIB_ADD_EXTRA_SYMBOLS
    using _CSTD strdup; using _CSTD strcasecmp; using _CSTD strncasecmp; 
    using _CSTD strtok_r; using _CSTD strnlen;
  #endif /* _DLIB_ADD_EXTRA_SYMBOLS */

  #if !__AEABI_PORTABILITY_INTERNAL_LEVEL
    using _CSTD __iar_Memchr; using _CSTD __iar_Strchr; 
    using _CSTD __iar_Strerror; using _CSTD __iar_Strpbrk;
    using _CSTD __iar_Strrchr; using _CSTD __iar_Strstr;
  #endif
#endif /* defined(_STD_USING) && defined(__cplusplus) */

/*
 * Copyright (c) 1992-2009 by P.J. Plauger.  ALL RIGHTS RESERVED.
 * Consult your license regarding permissions and restrictions.
V5.04:0576 */
