/* stdio.h standard header */
/* Copyright 2003-2010 IAR Systems AB.  */
#ifndef _STDIO
#define _STDIO

#ifndef _SYSTEM_BUILD
  #pragma system_include
#endif

#include <ycheck.h>
#include <yvals.h>
#include <ysizet.h>
#include <ystdio.h>

_C_STD_BEGIN

/* Module consistency. */
#pragma rtmodel="__dlib_file_descriptor",_STRINGIFY(_DLIB_FILE_DESCRIPTOR)

                /* macros */
#ifndef NULL
  #define NULL          _NULL
#endif /* NULL */

#if _DLIB_FILE_DESCRIPTOR
  typedef _Filet FILE;
#endif /* _DLIB_FILE_DESCRIPTOR */

#if __AEABI_PORTABILITY_INTERNAL_LEVEL
  #ifndef __AEABI_PORTABLE
    #define __AEABI_PORTABLE
  #endif

  #if _DLIB_FILE_DESCRIPTOR
    _C_LIB_DECL
    _DLIB_DATA_ATTR FILE extern *__aeabi_stdin; 
    _DLIB_DATA_ATTR FILE extern *__aeabi_stdout;
    _DLIB_DATA_ATTR FILE extern *__aeabi_stderr;
    _END_C_LIB_DECL
    #define stdin          _CSTD __aeabi_stdin
    #define stdout         _CSTD __aeabi_stdout
    #define stderr         _CSTD __aeabi_stderr
  #endif /* _DLIB_FILE_DESCRIPTOR */

  _C_LIB_DECL
  _DLIB_CONST_ATTR extern int const __aeabi_IOFBF;
  _DLIB_CONST_ATTR extern int const __aeabi_IOLBF;
  _DLIB_CONST_ATTR extern int const __aeabi_IONBF;
  _DLIB_CONST_ATTR extern int const __aeabi_BUFSIZ;
  _DLIB_CONST_ATTR extern int const __aeabi_FOPEN_MAX;
  _DLIB_CONST_ATTR extern int const __aeabi_TMP_MAX;
  _DLIB_CONST_ATTR extern int const __aeabi_FILENAME_MAX;
  _DLIB_CONST_ATTR extern int const __aeabi_L_tmpnam;
  _END_C_LIB_DECL
  #define _IOFBF          (_CSTD __aeabi_IOFBF)
  #define _IOLBF          (_CSTD __aeabi_IOLBF)
  #define _IONBF          (_CSTD __aeabi_IONBF)
  #define BUFSIZ          (_CSTD __aeabi_BUFSIZ)
  #define FOPEN_MAX       (_CSTD __aeabi_FOPEN_MAX)
  #define TMP_MAX         (_CSTD __aeabi_TMP_MAX)
  #define FILENAME_MAX    (_CSTD __aeabi_FILENAME_MAX)
  #define L_tmpnam        (_CSTD __aeabi_L_tmpnam)

#else /* __AEABI_PORTABILITY_INTERNAL_LEVEL */

  #if _DLIB_FILE_DESCRIPTOR
    _C_LIB_DECL
    _DLIB_DATA_ATTR extern FILE __iar_Stdin;
    _DLIB_DATA_ATTR extern FILE __iar_Stdout;
    _DLIB_DATA_ATTR extern FILE __iar_Stderr;
    _END_C_LIB_DECL
    #define stdin           (&_CSTD __iar_Stdin)
    #define stdout          (&_CSTD __iar_Stdout)
    #define stderr          (&_CSTD __iar_Stderr)
  #endif /* _DLIB_FILE_DESCRIPTOR */

  #define _IOFBF          0
  #define _IOLBF          1
  #define _IONBF          2
  #define BUFSIZ          512
  #ifndef FOPEN_MAX
    #define FOPEN_MAX       8
  #endif
  #define TMP_MAX         256
  #define FILENAME_MAX    260
  #define L_tmpnam        16

#endif /* __AEABI_PORTABILITY_INTERNAL_LEVEL */


#define EOF             (-1)
#define SEEK_SET        0
#define SEEK_CUR        1
#define SEEK_END        2


                /* type definitions */
typedef _Fpost fpos_t;

                /* printf and scanf pragma support */
#pragma language=save
#pragma language=extended

#ifdef _HAS_PRAGMA_PRINTF_ARGS
  #define __PRINTFPR _Pragma("__printf_args") \
                     _Pragma("library_default_requirements _Printf = unknown")
  #define __SCANFPR  _Pragma("__scanf_args") \
                     _Pragma("library_default_requirements _Scanf = unknown")
#else
  #define __PRINTFPR
  #define __SCANFPR
#endif

#if _DLIB_FILE_DESCRIPTOR
                /* declarations */
  _C_LIB_DECL

  __ATTRIBUTES void clearerr(FILE *);
  __ATTRIBUTES int fclose(FILE *);
  __ATTRIBUTES int feof(FILE *);
  __ATTRIBUTES int ferror(FILE *);
  __ATTRIBUTES int fflush(FILE *);
  __ATTRIBUTES int fgetc(FILE *);
  __ATTRIBUTES int fgetpos(FILE *_Restrict, fpos_t *_Restrict);
  __ATTRIBUTES char * fgets(char *_Restrict, int, FILE *_Restrict);
  __ATTRIBUTES FILE * fopen(const char *_Restrict, const char *_Restrict);
  __PRINTFPR __ATTRIBUTES int fprintf(FILE *_Restrict, const char *_Restrict, 
                                      ...);
  __ATTRIBUTES int fputc(int, FILE *);
  __ATTRIBUTES int fputs(const char *_Restrict, FILE *_Restrict);
  __ATTRIBUTES size_t fread(void *_Restrict, size_t, size_t, FILE *_Restrict);
  __ATTRIBUTES FILE * freopen(const char *_Restrict, const char *_Restrict,
                              FILE *_Restrict);
  __SCANFPR __ATTRIBUTES int fscanf(FILE *_Restrict, const char *_Restrict, 
                                    ...);
  __ATTRIBUTES int fseek(FILE *, long, int);
  __ATTRIBUTES int fsetpos(FILE *, const fpos_t *);
  __ATTRIBUTES long ftell(FILE *);
  __ATTRIBUTES size_t fwrite(const void *_Restrict, size_t, size_t, 
                             FILE *_Restrict);

  __ATTRIBUTES void rewind(FILE *);
  __ATTRIBUTES void setbuf(FILE *_Restrict, char *_Restrict);
  __ATTRIBUTES int setvbuf(FILE *_Restrict, char *_Restrict, int, size_t);
  __ATTRIBUTES FILE * tmpfile(void);
  __ATTRIBUTES int ungetc(int, FILE *);
  __PRINTFPR __ATTRIBUTES int vfprintf(FILE *_Restrict, 
                                       const char *_Restrict, __Va_list);
  #if _DLIB_ADD_C99_SYMBOLS
    __SCANFPR  __ATTRIBUTES int vfscanf(FILE *_Restrict, const char *_Restrict,
                                        __Va_list);
  #endif /* _DLIB_ADD_C99_SYMBOLS */

  #if _DLIB_ADD_EXTRA_SYMBOLS
    __ATTRIBUTES FILE * fdopen(_FD_TYPE, const char *);
    __ATTRIBUTES _FD_TYPE fileno(FILE *);
    __ATTRIBUTES int getw(FILE *);
    __ATTRIBUTES int putw(int, FILE *);
  #endif /* _DLIB_ADD_EXTRA_SYMBOLS */

  __ATTRIBUTES int getc(FILE *);
  __ATTRIBUTES int putc(int, FILE *);
  _END_C_LIB_DECL
#endif /* _DLIB_FILE_DESCRIPTOR */

_C_LIB_DECL
             /* Corresponds to fgets(char *, int, stdin); */
__EFF_NR1    __ATTRIBUTES char * __gets(char *, int);
__EFF_NR1    __ATTRIBUTES char * gets(char *);
__EFF_NW1    __ATTRIBUTES void perror(const char *);
__EFF_NW1    __PRINTFPR __ATTRIBUTES int printf(const char *_Restrict, ...);
__EFF_NW1    __ATTRIBUTES int puts(const char *);
__EFF_NW1    __SCANFPR  __ATTRIBUTES int scanf(const char *_Restrict, ...);
__EFF_NR1NW2 __PRINTFPR __ATTRIBUTES int sprintf(char *_Restrict, 
                                                 const char *_Restrict, ...);
__EFF_NW1NW2 __SCANFPR  __ATTRIBUTES int sscanf(const char *_Restrict, 
                                                const char *_Restrict, ...);
             __ATTRIBUTES char * tmpnam(char *);
             /* Corresponds to "ungetc(c, stdout)" */
             __ATTRIBUTES int __ungetchar(int);
__EFF_NW1    __PRINTFPR __ATTRIBUTES int vprintf(const char *_Restrict,
                                                 __Va_list);
#if _DLIB_ADD_C99_SYMBOLS
  __EFF_NW1    __SCANFPR  __ATTRIBUTES int vscanf(const char *_Restrict, 
                                                  __Va_list);
  __EFF_NW1NW2 __SCANFPR  __ATTRIBUTES int vsscanf(const char *_Restrict, 
                                                   const char *_Restrict, 
                                                   __Va_list);
#endif /* _DLIB_ADD_C99_SYMBOLS */
__EFF_NR1NW2  __PRINTFPR __ATTRIBUTES int vsprintf(char *_Restrict, 
                                                   const char *_Restrict,
                                                   __Va_list);
              /* Corresponds to fwrite(p, x, y, stdout); */
__EFF_NW1      __ATTRIBUTES size_t __write_array(const void *, size_t, size_t);
#if _DLIB_ADD_C99_SYMBOLS
  __EFF_NR1NW3 __PRINTFPR __ATTRIBUTES int snprintf(char *_Restrict, size_t, 
                                                    const char *_Restrict, ...);
  __EFF_NR1NW3 __PRINTFPR __ATTRIBUTES int vsnprintf(char *_Restrict, size_t,
                                                     const char *_Restrict, 
                                                     __Va_list);
#endif /* _DLIB_ADD_C99_SYMBOLS */

              __ATTRIBUTES int getchar(void);
              __ATTRIBUTES int putchar(int);

_END_C_LIB_DECL

#pragma language=restore

#ifndef _NO_DEFINITIONS_IN_HEADER_FILES
  #if _DLIB_FILE_DESCRIPTOR
              /* inlines, for C and C++ */
    #pragma inline
    int (getc)(FILE *_Str)
    {
      return fgetc(_Str);
    }

    #pragma inline
    int (putc)(int _C, FILE *_Str)
    {
      return fputc(_C, _Str);
    }
  #endif

#endif /* _NO_DEFINITIONS_IN_HEADER_FILES */

_C_STD_END
#endif /* _STDIO */

#if defined(_STD_USING) && defined(__cplusplus)
  using _CSTD fpos_t;
  using _CSTD getchar; using _CSTD gets; using _CSTD perror;
  using _CSTD putchar;
  using _CSTD printf; using _CSTD puts; 
  using _CSTD scanf; using _CSTD sprintf;
  using _CSTD sscanf; using _CSTD tmpnam;
  using _CSTD vprintf;
  using _CSTD vsprintf;
  #if _DLIB_ADD_C99_SYMBOLS
    using _CSTD snprintf; using _CSTD vsnprintf;
    using _CSTD vscanf; using _CSTD vsscanf;
  #endif /* _DLIB_ADD_C99_SYMBOLS */
  using _CSTD __gets; using _CSTD __ungetchar;

  #if _DLIB_FILE_DESCRIPTOR
    using _CSTD FILE;
    using _CSTD clearerr; using _CSTD fclose; using _CSTD feof;
    using _CSTD ferror; using _CSTD fflush; using _CSTD fgetc;
    using _CSTD fgetpos; using _CSTD fgets; using _CSTD fopen;
    using _CSTD fprintf; using _CSTD fputc; using _CSTD fputs;
    using _CSTD fread; using _CSTD freopen; using _CSTD fscanf;
    using _CSTD fseek; using _CSTD fsetpos; using _CSTD ftell;
    using _CSTD fwrite; using _CSTD getc; using _CSTD putc; 
    using _CSTD rewind; using _CSTD setbuf; using _CSTD setvbuf; 
    using _CSTD tmpfile; using _CSTD ungetc; using _CSTD vfprintf; 
    #if _DLIB_ADD_EXTRA_SYMBOLS
      using _CSTD fdopen; using _CSTD fileno;
      using _CSTD getw; using _CSTD putw;
    #endif /* _DLIB_ADD_EXTRA_SYMBOLS */
    #if _DLIB_ADD_C99_SYMBOLS
      using _CSTD vfscanf;
    #endif /* _DLIB_ADD_C99_SYMBOLS */
  #endif

  #if __AEABI_PORTABILITY_INTERNAL_LEVEL
    #if _DLIB_FILE_DESCRIPTOR
      using _CSTD __aeabi_stdin; using _CSTD __aeabi_stdout; 
      using _CSTD __aeabi_stderr;
    #endif /* _DLIB_FILE_DESCRIPTOR */

    using _CSTD __aeabi_IOFBF; using _CSTD __aeabi_IOLBF; 
    using _CSTD __aeabi_IONBF; using _CSTD __aeabi_BUFSIZ;
    using _CSTD __aeabi_FOPEN_MAX; using _CSTD __aeabi_TMP_MAX;
    using _CSTD __aeabi_FILENAME_MAX; using _CSTD __aeabi_L_tmpnam;
  #endif

  #include <ysizet.h>
  #include <ystdio.h>
#endif /* defined(_STD_USING) && defined(__cplusplus) */

/*
 * Copyright (c) 1992-2002 by P.J. Plauger.  ALL RIGHTS RESERVED.
 * Consult your license regarding permissions and restrictions.
V5.04:0576 */
