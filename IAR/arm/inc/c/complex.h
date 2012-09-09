/* complex.h standard header */
/* Copyright 2001-2010 IAR Systems AB. */
#ifndef _COMPLEX
#define _COMPLEX

#ifndef _SYSTEM_BUILD
  #pragma system_include
#endif

#include <ycheck.h>
#include <ymath.h>
#include <xtgmath.h>
_C_STD_BEGIN

#if _DLIB_ADD_C99_SYMBOLS

                /* MACROS */
#define _Fcomplex               _Fcomplex       /* signal definitions */
#define _DCOMPLEX_(re, im)      _Cbuild(re, im)
#define _FCOMPLEX_(re, im)      _FCbuild(re, im)
#define _LCOMPLEX_(re, im)      _LCbuild(re, im)

#if defined(__cplusplus) 
  #define _Complex_I    _FCbuild(0.0F, 1.0F)

                // TYPES
  #ifndef _C_COMPLEX_T
    #define _C_COMPLEX_T

    typedef struct _C_double_complex
    {       /* double complex */
      double _Val[2];
    } _C_double_complex;

    typedef struct _C_float_complex
    {       /* float complex */
      float _Val[2];
    } _C_float_complex;

    typedef struct _C_ldouble_complex
    {       /* long double complex */
      long double _Val[2];
    } _C_ldouble_complex;
  #endif /* _C_COMPLEX_T */

  _C_LIB_DECL
  typedef _C_double_complex _Dcomplex;
  typedef _C_float_complex _Fcomplex;
  typedef _C_ldouble_complex _Lcomplex;

  __EFF_NE __ATTRIBUTES double cimag(_Dcomplex);
  __EFF_NE __ATTRIBUTES double creal(_Dcomplex);
  __EFF_NE __ATTRIBUTES float cimagf(_Fcomplex);
  __EFF_NE __ATTRIBUTES float crealf(_Fcomplex);
  __EFF_NE __ATTRIBUTES long double cimagl(_Lcomplex);
  __EFF_NE __ATTRIBUTES long double creall(_Lcomplex);

  #pragma inline=forced
  inline double cimag(_Dcomplex _Left)
  {       // get imaginary part
    return (_Left._Val[1]);
  }

  #pragma inline=forced
  inline double creal(_Dcomplex _Left)
  {       // get real part
    return (_Left._Val[0]);
  }

  #pragma inline=forced
  inline float cimagf(_Fcomplex _Left)
  {       // get imaginary part
    return (_Left._Val[1]);
  }

  #pragma inline=forced
  inline float crealf(_Fcomplex _Left)
  {       // get real part
    return (_Left._Val[0]);
  }

  #pragma inline=forced
  inline long double cimagl(_Lcomplex _Left)
  {       // get imaginary part
    return (_Left._Val[1]);
  }

  #pragma inline=forced
  inline long double creall(_Lcomplex _Left)
  {       // get real part
    return (_Left._Val[0]);
  }
  _END_C_LIB_DECL

#else /* defined(__cplusplus) */
                /* TYPES */
  #if 199901L <= __STDC_VERSION__
    typedef double _Complex _Dcomplex;
    typedef float _Complex _Fcomplex;
    typedef long double _Complex _Lcomplex;

    #define complex      _Complex

    #if __EDG__ 
      #define _Complex_I   ((float _Complex)__I__)
    #else /* __EDG__ */
      #define _Complex_I   _FCbuild(0.0F, 1.0F)
    #endif /* __EDG__ */

    #if __EDG__ 
      #define _Cbuild(re, im)       (*(_Dcomplex *)(double []){re, im})
      /* #define _Cmulcc(x, y)        ((x) * (y)) */
      /* #define _Cmulcr(x, y)        ((x) * (y)) */

      #define _FCbuild(re, im)      (*(_Fcomplex *)(float []){re, im})
      /* #define _FCmulcc(x, y)       ((x) * (y)) */
      /* #define _FCmulcr(x, y)       ((x) * (y)) */

      #define _LCbuild(re, im)      (*(_Lcomplex *)(long double []){re, im})
      /* #define _LCmulcc(x, y)       ((x) * (y)) */
      /* #define _LCmulcr(x, y)       ((x) * (y)) */

    #elif 0 < __GNUC__
      /* #define _Cbuild(re, im)      (*(_Dcomplex *)(double []){re, im}) */
      #define _Cmulcc(x, y) ((x) * (y))
      #define _Cmulcr(x, y) ((x) * (y))

      /* #define _FCbuild(re, im)     (*(_Fcomplex *)(float []){re, im}) */
      #define _FCmulcc(x, y)        ((x) * (y))
      #define _FCmulcr(x, y)        ((x) * (y))

      /* #define _LCbuild(re, im)    (*(_Lcomplex *)(long double []){re, im}) */
      #define _LCmulcc(x, y)        ((x) * (y))
      #define _LCmulcr(x, y)        ((x) * (y))
    #endif /* compiler type */
  #else /* 199901L <= __STDC_VERSION__ */
    #ifndef _C_COMPLEX_T
      #define _C_COMPLEX_T

      typedef struct _C_double_complex
      {       /* double complex */
        double _Val[2];
      } _C_double_complex;

      typedef struct _C_float_complex
      {       /* float complex */
        float _Val[2];
      } _C_float_complex;

      typedef struct _C_ldouble_complex
      {       /* long double complex */
        long double _Val[2];
      } _C_ldouble_complex;
    #endif /* _C_COMPLEX_T */

    typedef _C_double_complex _Dcomplex;
    typedef _C_float_complex _Fcomplex;
    typedef _C_ldouble_complex _Lcomplex;

    #define _Complex_I    _FCbuild(0.0F, 1.0F)
  #endif /* 199901L <= __STDC_VERSION__ */

  _C_LIB_DECL
  __EFF_NE __ATTRIBUTES double cimag(_Dcomplex);
  __EFF_NE __ATTRIBUTES double creal(_Dcomplex);
  __EFF_NE __ATTRIBUTES float cimagf(_Fcomplex);
  __EFF_NE __ATTRIBUTES float crealf(_Fcomplex);
  __EFF_NE __ATTRIBUTES long double cimagl(_Lcomplex);
  __EFF_NE __ATTRIBUTES long double creall(_Lcomplex);
  _END_C_LIB_DECL

#endif /* defined(__cplusplus) etc. */

                /* MACROS */

#if __EDG__
  #define _HAS_C99_IMAGINARY_TYPE 1
#endif

#if !defined(__cplusplus)
  #if _HAS_C99_IMAGINARY_TYPE
    #define imaginary      _Imaginary
    #define _Imaginary_I   ((float _Imaginary)_Complex_I)
    #define I      _Imaginary_I
  #else /* _HAS_C99_IMAGINARY_TYPE */
    #define I      _Complex_I
  #endif /* _HAS_C99_IMAGINARY_TYPE */
#endif /* !defined(__cplusplus) */

                /* FUNCTIONS */
_C_LIB_DECL
__EFF_NS __ATTRIBUTES double    cabs(_Dcomplex);
__EFF_NS __ATTRIBUTES _Dcomplex cacos(_Dcomplex);
__EFF_NS __ATTRIBUTES _Dcomplex cacosh(_Dcomplex);
__EFF_NS __ATTRIBUTES double    carg(_Dcomplex);
__EFF_NS __ATTRIBUTES _Dcomplex casin(_Dcomplex);
__EFF_NS __ATTRIBUTES _Dcomplex casinh(_Dcomplex);
__EFF_NS __ATTRIBUTES _Dcomplex catan(_Dcomplex);
__EFF_NS __ATTRIBUTES _Dcomplex catanh(_Dcomplex);
__EFF_NS __ATTRIBUTES _Dcomplex ccos(_Dcomplex);
__EFF_NS __ATTRIBUTES _Dcomplex ccosh(_Dcomplex);
__EFF_NS __ATTRIBUTES _Dcomplex cexp(_Dcomplex);
/* __EFF_NS __ATTRIBUTES double cimag(_Dcomplex); */
/* __EFF_NS __ATTRIBUTES _Dcomplex clog(_Dcomplex); */
__EFF_NS __ATTRIBUTES _Dcomplex clog10(_Dcomplex);
__EFF_NS __ATTRIBUTES _Dcomplex conj(_Dcomplex);
__EFF_NS __ATTRIBUTES _Dcomplex cpow(_Dcomplex, _Dcomplex);
__EFF_NS __ATTRIBUTES _Dcomplex cproj(_Dcomplex);
/* __EFF_NS __ATTRIBUTES double creal(_Dcomplex); */
__EFF_NS __ATTRIBUTES _Dcomplex csin(_Dcomplex);
__EFF_NS __ATTRIBUTES _Dcomplex csinh(_Dcomplex);
__EFF_NS __ATTRIBUTES _Dcomplex csqrt(_Dcomplex);
__EFF_NS __ATTRIBUTES _Dcomplex ctan(_Dcomplex);
__EFF_NS __ATTRIBUTES _Dcomplex ctanh(_Dcomplex);
__EFF_NS __ATTRIBUTES double    norm(_Dcomplex); /* added with TR1 */

__EFF_NS __ATTRIBUTES float     cabsf(_Fcomplex);
__EFF_NS __ATTRIBUTES _Fcomplex cacosf(_Fcomplex);
__EFF_NS __ATTRIBUTES _Fcomplex cacoshf(_Fcomplex);
__EFF_NS __ATTRIBUTES float     cargf(_Fcomplex);
__EFF_NS __ATTRIBUTES _Fcomplex casinf(_Fcomplex);
__EFF_NS __ATTRIBUTES _Fcomplex casinhf(_Fcomplex);
__EFF_NS __ATTRIBUTES _Fcomplex catanf(_Fcomplex);
__EFF_NS __ATTRIBUTES _Fcomplex catanhf(_Fcomplex);
__EFF_NS __ATTRIBUTES _Fcomplex ccosf(_Fcomplex);
__EFF_NS __ATTRIBUTES _Fcomplex ccoshf(_Fcomplex);
__EFF_NS __ATTRIBUTES _Fcomplex cexpf(_Fcomplex);
/* __EFF_NS __ATTRIBUTES float cimagf(_Fcomplex); */
__EFF_NS __ATTRIBUTES _Fcomplex clogf(_Fcomplex);
__EFF_NS __ATTRIBUTES _Fcomplex clog10f(_Fcomplex);
__EFF_NS __ATTRIBUTES _Fcomplex conjf(_Fcomplex);
__EFF_NS __ATTRIBUTES _Fcomplex cpowf(_Fcomplex, _Fcomplex);
__EFF_NS __ATTRIBUTES _Fcomplex cprojf(_Fcomplex);
/* __EFF_NS __ATTRIBUTES float crealf(_Fcomplex); */
__EFF_NS __ATTRIBUTES _Fcomplex csinf(_Fcomplex);
__EFF_NS __ATTRIBUTES _Fcomplex csinhf(_Fcomplex);
__EFF_NS __ATTRIBUTES _Fcomplex csqrtf(_Fcomplex);
__EFF_NS __ATTRIBUTES _Fcomplex ctanf(_Fcomplex);
__EFF_NS __ATTRIBUTES _Fcomplex ctanhf(_Fcomplex);
__EFF_NS __ATTRIBUTES float     normf(_Fcomplex); /* added with TR1 */

__EFF_NS __ATTRIBUTES long double cabsl(_Lcomplex);
__EFF_NS __ATTRIBUTES _Lcomplex   cacosl(_Lcomplex);
__EFF_NS __ATTRIBUTES _Lcomplex   cacoshl(_Lcomplex);
__EFF_NS __ATTRIBUTES long double cargl(_Lcomplex);
__EFF_NS __ATTRIBUTES _Lcomplex   casinl(_Lcomplex);
__EFF_NS __ATTRIBUTES _Lcomplex   casinhl(_Lcomplex);
__EFF_NS __ATTRIBUTES _Lcomplex   catanl(_Lcomplex);
__EFF_NS __ATTRIBUTES _Lcomplex   catanhl(_Lcomplex);
__EFF_NS __ATTRIBUTES _Lcomplex   ccosl(_Lcomplex);
__EFF_NS __ATTRIBUTES _Lcomplex   ccoshl(_Lcomplex);
__EFF_NS __ATTRIBUTES _Lcomplex   cexpl(_Lcomplex);
/* __EFF_NS __ATTRIBUTES long double cimagl(_Lcomplex); */
__EFF_NS __ATTRIBUTES _Lcomplex   clogl(_Lcomplex);
__EFF_NS __ATTRIBUTES _Lcomplex   clog10l(_Lcomplex);
__EFF_NS __ATTRIBUTES _Lcomplex   conjl(_Lcomplex);
__EFF_NS __ATTRIBUTES _Lcomplex   cpowl(_Lcomplex, _Lcomplex);
__EFF_NS __ATTRIBUTES _Lcomplex   cprojl(_Lcomplex);
/* __EFF_NS __ATTRIBUTES long double creall(_Lcomplex); */
__EFF_NS __ATTRIBUTES _Lcomplex   csinl(_Lcomplex);
__EFF_NS __ATTRIBUTES _Lcomplex   csinhl(_Lcomplex);
__EFF_NS __ATTRIBUTES _Lcomplex   csqrtl(_Lcomplex);
__EFF_NS __ATTRIBUTES _Lcomplex   ctanl(_Lcomplex);
__EFF_NS __ATTRIBUTES _Lcomplex   ctanhl(_Lcomplex);
__EFF_NS __ATTRIBUTES long double norml(_Lcomplex);   /* added with TR1 */

__EFF_NE __ATTRIBUTES _Dcomplex _D_FNAME(Cbuild)(double, double);
__EFF_NE __ATTRIBUTES _Dcomplex _D_FNAME(Cmulcc)(_Dcomplex, _Dcomplex);
__EFF_NE __ATTRIBUTES _Dcomplex _D_FNAME(Cmulcr)(_Dcomplex, double);
__EFF_NE __ATTRIBUTES _Dcomplex _D_FNAME(Cdivcc)(_Dcomplex, _Dcomplex);
__EFF_NE __ATTRIBUTES _Dcomplex _D_FNAME(Cdivcr)(_Dcomplex, double);
__EFF_NE __ATTRIBUTES _Dcomplex _D_FNAME(Caddcc)(_Dcomplex, _Dcomplex);
__EFF_NE __ATTRIBUTES _Dcomplex _D_FNAME(Caddcr)(_Dcomplex, double);
__EFF_NE __ATTRIBUTES _Dcomplex _D_FNAME(Csubcc)(_Dcomplex, _Dcomplex);
__EFF_NE __ATTRIBUTES _Dcomplex _D_FNAME(Csubcr)(_Dcomplex, double);

#ifndef _FLOAT_IS_DOUBLE
  __EFF_NE __ATTRIBUTES _Fcomplex _F_FNAME(Cbuild)(float, float);
  __EFF_NE __ATTRIBUTES _Fcomplex _F_FNAME(Cmulcc)(_Fcomplex, _Fcomplex);
  __EFF_NE __ATTRIBUTES _Fcomplex _F_FNAME(Cmulcr)(_Fcomplex, float);
  __EFF_NE __ATTRIBUTES _Fcomplex _F_FNAME(Cdivcc)(_Fcomplex, _Fcomplex);
  __EFF_NE __ATTRIBUTES _Fcomplex _F_FNAME(Cdivcr)(_Fcomplex, float);
  __EFF_NE __ATTRIBUTES _Fcomplex _F_FNAME(Caddcc)(_Fcomplex, _Fcomplex);
  __EFF_NE __ATTRIBUTES _Fcomplex _F_FNAME(Caddcr)(_Fcomplex, float);
  __EFF_NE __ATTRIBUTES _Fcomplex _F_FNAME(Csubcc)(_Fcomplex, _Fcomplex);
  __EFF_NE __ATTRIBUTES _Fcomplex _F_FNAME(Csubcr)(_Fcomplex, float);
#endif /* _FLOAT_IS_DOUBLE */

#ifndef _LONG_DOUBLE_IS_DOUBLE
  __EFF_NE __ATTRIBUTES _Lcomplex _L_FNAME(Cbuild)(long double, long double);
  __EFF_NE __ATTRIBUTES _Lcomplex _L_FNAME(Cmulcc)(_Lcomplex, _Lcomplex);
  __EFF_NE __ATTRIBUTES _Lcomplex _L_FNAME(Cmulcr)(_Lcomplex, long double);
  __EFF_NE __ATTRIBUTES _Lcomplex _L_FNAME(Cdivcc)(_Lcomplex, _Lcomplex);
  __EFF_NE __ATTRIBUTES _Lcomplex _L_FNAME(Cdivcr)(_Lcomplex, long double);
  __EFF_NE __ATTRIBUTES _Lcomplex _L_FNAME(LCaddcc)(_Lcomplex, _Lcomplex);
  __EFF_NE __ATTRIBUTES _Lcomplex _L_FNAME(Caddcr)(_Lcomplex, long double);
  __EFF_NE __ATTRIBUTES _Lcomplex _L_FNAME(Csubcc)(_Lcomplex, _Lcomplex);
  __EFF_NE __ATTRIBUTES _Lcomplex _L_FNAME(Csubcr)(_Lcomplex, long double);
#endif /* _LONG_DOUBLE_IS_DOUBLE */
_END_C_LIB_DECL

#ifdef __cplusplus
  _EXTERN_CPP
  __EFF_NS __ATTRIBUTES _Dcomplex acos(_Dcomplex _Left);
  __EFF_NS __ATTRIBUTES _Dcomplex acosh(_Dcomplex _Left);
  __EFF_NS __ATTRIBUTES _Dcomplex asin(_Dcomplex _Left);
  __EFF_NS __ATTRIBUTES _Dcomplex asinh(_Dcomplex _Left);
  __EFF_NS __ATTRIBUTES _Dcomplex atan(_Dcomplex _Left);
  __EFF_NS __ATTRIBUTES _Dcomplex atanh(_Dcomplex _Left);
  __EFF_NS __ATTRIBUTES _Dcomplex cos(_Dcomplex _Left);
  __EFF_NS __ATTRIBUTES _Dcomplex cosh(_Dcomplex _Left);
  __EFF_NS __ATTRIBUTES _Dcomplex exp(_Dcomplex _Left);
/*  __EFF_NS __ATTRIBUTES _Dcomplex log(_Dcomplex _Left); */
  __EFF_NS __ATTRIBUTES _Dcomplex log10(_Dcomplex _Left);
  __EFF_NS __ATTRIBUTES _Dcomplex pow(_Dcomplex _Left, _Dcomplex _Right);
  __EFF_NS __ATTRIBUTES _Dcomplex sin(_Dcomplex _Left);
  __EFF_NS __ATTRIBUTES _Dcomplex sinh(_Dcomplex _Left);
  __EFF_NS __ATTRIBUTES _Dcomplex sqrt(_Dcomplex _Left);
  __EFF_NS __ATTRIBUTES _Dcomplex tan(_Dcomplex _Left);
  __EFF_NS __ATTRIBUTES _Dcomplex tanh(_Dcomplex _Left);
  __EFF_NS __ATTRIBUTES double    abs(_Dcomplex _Left);
  __EFF_NS __ATTRIBUTES double    arg(_Dcomplex _Left);
  __EFF_NS __ATTRIBUTES double    fabs(_Dcomplex _Left);
  __EFF_NS __ATTRIBUTES double    imag(_Dcomplex _Left);
  __EFF_NS __ATTRIBUTES double    real(_Dcomplex _Left);
  __EFF_NS __ATTRIBUTES _Fcomplex acos(_Fcomplex _Left);
  __EFF_NS __ATTRIBUTES _Fcomplex acosh(_Fcomplex _Left);
  __EFF_NS __ATTRIBUTES _Fcomplex asin(_Fcomplex _Left);
  __EFF_NS __ATTRIBUTES _Fcomplex asinh(_Fcomplex _Left);
  __EFF_NS __ATTRIBUTES _Fcomplex atan(_Fcomplex _Left);
  __EFF_NS __ATTRIBUTES _Fcomplex atanh(_Fcomplex _Left);
  __EFF_NS __ATTRIBUTES _Fcomplex conj(_Fcomplex _Left);
  __EFF_NS __ATTRIBUTES _Fcomplex cos(_Fcomplex _Left);
  __EFF_NS __ATTRIBUTES _Fcomplex cosh(_Fcomplex _Left);
  __EFF_NS __ATTRIBUTES _Fcomplex cproj(_Fcomplex _Left);
  __EFF_NS __ATTRIBUTES _Fcomplex exp(_Fcomplex _Left);
  __EFF_NS __ATTRIBUTES _Fcomplex log(_Fcomplex _Left);
  __EFF_NS __ATTRIBUTES _Fcomplex log10(_Fcomplex _Left);
  __EFF_NS __ATTRIBUTES float     norm(_Fcomplex _Left);
  __EFF_NS __ATTRIBUTES _Fcomplex pow(_Fcomplex _Left, _Fcomplex _Right);
  __EFF_NS __ATTRIBUTES _Fcomplex sin(_Fcomplex _Left);
  __EFF_NS __ATTRIBUTES _Fcomplex sinh(_Fcomplex _Left);
  __EFF_NS __ATTRIBUTES _Fcomplex sqrt(_Fcomplex _Left);
  __EFF_NS __ATTRIBUTES _Fcomplex tan(_Fcomplex _Left);
  __EFF_NS __ATTRIBUTES _Fcomplex tanh(_Fcomplex _Left);
  __EFF_NS __ATTRIBUTES float     abs(_Fcomplex _Left);
  __EFF_NS __ATTRIBUTES float     arg(_Fcomplex _Left);
  __EFF_NS __ATTRIBUTES float     carg(_Fcomplex _Left);
  __EFF_NS __ATTRIBUTES float     cimag(_Fcomplex _Left);
  __EFF_NS __ATTRIBUTES float     creal(_Fcomplex _Left);
  __EFF_NS __ATTRIBUTES float     fabs(_Fcomplex _Left);
  __EFF_NS __ATTRIBUTES float     imag(_Fcomplex _Left);
  __EFF_NS __ATTRIBUTES float     real(_Fcomplex _Left);
  __EFF_NS __ATTRIBUTES _Lcomplex acos(_Lcomplex _Left);
  __EFF_NS __ATTRIBUTES _Lcomplex acosh(_Lcomplex _Left);
  __EFF_NS __ATTRIBUTES _Lcomplex asin(_Lcomplex _Left);
  __EFF_NS __ATTRIBUTES _Lcomplex asinh(_Lcomplex _Left);
  __EFF_NS __ATTRIBUTES _Lcomplex atan(_Lcomplex _Left);
  __EFF_NS __ATTRIBUTES _Lcomplex atanh(_Lcomplex _Left);
  __EFF_NS __ATTRIBUTES _Lcomplex conj(_Lcomplex _Left);
  __EFF_NS __ATTRIBUTES _Lcomplex cos(_Lcomplex _Left);
  __EFF_NS __ATTRIBUTES _Lcomplex cosh(_Lcomplex _Left);
  __EFF_NS __ATTRIBUTES _Lcomplex cproj(_Lcomplex _Left);
  __EFF_NS __ATTRIBUTES _Lcomplex exp(_Lcomplex _Left);
  __EFF_NS __ATTRIBUTES _Lcomplex log(_Lcomplex _Left);
  __EFF_NS __ATTRIBUTES _Lcomplex log10(_Lcomplex _Left);
  __EFF_NS __ATTRIBUTES long double norm(_Lcomplex _Left);
  __EFF_NS __ATTRIBUTES _Lcomplex pow(_Lcomplex _Left, _Lcomplex _Right);
  __EFF_NS __ATTRIBUTES _Lcomplex sin(_Lcomplex _Left);
  __EFF_NS __ATTRIBUTES _Lcomplex sinh(_Lcomplex _Left);
  __EFF_NS __ATTRIBUTES _Lcomplex sqrt(_Lcomplex _Left);
  __EFF_NS __ATTRIBUTES _Lcomplex tan(_Lcomplex _Left);
  __EFF_NS __ATTRIBUTES _Lcomplex tanh(_Lcomplex _Left);
  __EFF_NS __ATTRIBUTES long double abs(_Lcomplex _Left);
  __EFF_NS __ATTRIBUTES long double arg(_Lcomplex _Left);
  __EFF_NS __ATTRIBUTES long double carg(_Lcomplex _Left);
  __EFF_NS __ATTRIBUTES long double cimag(_Lcomplex _Left);
  __EFF_NS __ATTRIBUTES long double creal(_Lcomplex _Left);
  __EFF_NS __ATTRIBUTES long double fabs(_Lcomplex _Left);
  __EFF_NS __ATTRIBUTES long double imag(_Lcomplex _Left);
  __EFF_NS __ATTRIBUTES long double real(_Lcomplex _Left);

        // double complex OVERLOADS
  #pragma inline=forced
  inline _Dcomplex acos(_Dcomplex _Left)
  {       // compute cacos
    return (cacos(_Left));
  }

  #pragma inline=forced
  inline _Dcomplex acosh(_Dcomplex _Left)
  {       // compute cacosh
    return (cacosh(_Left));
  }

  #pragma inline=forced
  inline _Dcomplex asin(_Dcomplex _Left)
  {       // compute casin
    return (casin(_Left));
  }

  #pragma inline=forced
  inline _Dcomplex asinh(_Dcomplex _Left)
  {       // compute casinh
    return (casinh(_Left));
  }

  #pragma inline=forced
  inline _Dcomplex atan(_Dcomplex _Left)
  {       // compute catan
    return (catan(_Left));
  }

  #pragma inline=forced
  inline _Dcomplex atanh(_Dcomplex _Left)
  {       // compute catanh
    return (catanh(_Left));
  }

  #pragma inline=forced
  inline _Dcomplex cos(_Dcomplex _Left)
  {       // compute ccos
    return (ccos(_Left));
  }

  #pragma inline=forced
  inline _Dcomplex cosh(_Dcomplex _Left)
  {       // compute ccosh
    return (ccosh(_Left));
  }

  #pragma inline=forced
  inline _Dcomplex exp(_Dcomplex _Left)
  {       // compute cexp
    return (cexp(_Left));
  }

/*  inline _Dcomplex log(_Dcomplex _Left)
 *  {       // compute clog
 *    return (clog(_Left)); 
 *  }
 */

  #pragma inline=forced
  inline _Dcomplex log10(_Dcomplex _Left)
  {       // compute clog10
    return (clog10(_Left));
  }

  #pragma inline=forced
  inline _Dcomplex pow(_Dcomplex _Left, _Dcomplex _Right)
  {       // compute cpow
    return (cpow(_Left, _Right));
  }

  #pragma inline=forced
  inline _Dcomplex sin(_Dcomplex _Left)
  {       // compute csin
    return (csin(_Left));
  }

  #pragma inline=forced
  inline _Dcomplex sinh(_Dcomplex _Left)
  {       // compute csinh
    return (csinh(_Left));
  }

  #pragma inline=forced
  inline _Dcomplex sqrt(_Dcomplex _Left)
  {       // compute csqrt
    return (csqrt(_Left));
  }

  #pragma inline=forced
  inline _Dcomplex tan(_Dcomplex _Left)
  {       // compute ctan
    return (ctan(_Left));
  }

  #pragma inline=forced
  inline _Dcomplex tanh(_Dcomplex _Left)
  {       // compute ctanh
    return (ctanh(_Left));
  }

  #pragma inline=forced
  inline double abs(_Dcomplex _Left)
  {       // compute cabs
    return (cabs(_Left));
  }

  #pragma inline=forced
  inline double arg(_Dcomplex _Left)
  {       // compute carg
    return (carg(_Left));
  }

  #pragma inline=forced
  inline double fabs(_Dcomplex _Left)
  {       // compute cabs
    return (cabs(_Left));
  }

  #pragma inline=forced
  inline double imag(_Dcomplex _Left)
  {       // compute cimag
    return (cimag(_Left));
  }

  #pragma inline=forced
  inline double real(_Dcomplex _Left)
  {       // compute creal
    return (creal(_Left));
  }

        // float complex OVERLOADS
  #pragma inline=forced
  inline _Fcomplex acos(_Fcomplex _Left)
  {       // compute cacos
    return (cacosf(_Left));
  }

  #pragma inline=forced
  inline _Fcomplex acosh(_Fcomplex _Left)
  {       // compute cacosh
    return (cacoshf(_Left));
  }

  #pragma inline=forced
  inline _Fcomplex asin(_Fcomplex _Left)
  {       // compute casin
    return (casinf(_Left));
  }

  #pragma inline=forced
  inline _Fcomplex asinh(_Fcomplex _Left)
  {       // compute casinh
    return (casinhf(_Left));
  }

  #pragma inline=forced
  inline _Fcomplex atan(_Fcomplex _Left)
  {       // compute catan
    return (catanf(_Left));
  }

  #pragma inline=forced
  inline _Fcomplex atanh(_Fcomplex _Left)
  {       // compute catanh
    return (catanhf(_Left));
  }

  #pragma inline=forced
  inline _Fcomplex conj(_Fcomplex _Left)
  {       // compute conj
    return (conjf(_Left));
  }

  #pragma inline=forced
  inline _Fcomplex cos(_Fcomplex _Left)
  {       // compute ccos
    return (ccosf(_Left));
  }

  #pragma inline=forced
  inline _Fcomplex cosh(_Fcomplex _Left)
  {       // compute ccosh
    return (ccoshf(_Left));
  }

  #pragma inline=forced
  inline _Fcomplex cproj(_Fcomplex _Left)
  {       // compute cproj
    return (cprojf(_Left));
  }

  #pragma inline=forced
  inline _Fcomplex exp(_Fcomplex _Left)
  {       // compute cexp
    return (cexpf(_Left));
  }

  #pragma inline=forced
  inline _Fcomplex log(_Fcomplex _Left)
  {       // compute clog
    return (clogf(_Left));
  }

  #pragma inline=forced
  inline _Fcomplex log10(_Fcomplex _Left)
  {       // compute clog10
    return (clog10f(_Left));
  }

  #pragma inline=forced
  inline float norm(_Fcomplex _Left)
  {       // compute norm -- added with TR1
    return (normf(_Left));
  }

  #pragma inline=forced
  inline _Fcomplex pow(_Fcomplex _Left, _Fcomplex _Right)
  {       // compute cpow
    return (cpowf(_Left, _Right));
  }

  #pragma inline=forced
  inline _Fcomplex sin(_Fcomplex _Left)
  {       // compute csin
    return (csinf(_Left));
  }

  #pragma inline=forced
  inline _Fcomplex sinh(_Fcomplex _Left)
  {       // compute csinh
    return (csinhf(_Left));
  }

  #pragma inline=forced
  inline _Fcomplex sqrt(_Fcomplex _Left)
  {       // compute csqrt
    return (csqrtf(_Left));
  }

  #pragma inline=forced
  inline _Fcomplex tan(_Fcomplex _Left)
  {       // compute ctan
    return (ctanf(_Left));
  }

  #pragma inline=forced
  inline _Fcomplex tanh(_Fcomplex _Left)
  {       // compute ctanh
    return (ctanhf(_Left));
  }

  #pragma inline=forced
  inline float abs(_Fcomplex _Left)
  {       // compute cabs
    return (cabsf(_Left));
  }

  #pragma inline=forced
  inline float arg(_Fcomplex _Left)
  {       // compute carg
    return (cargf(_Left));
  }

  #pragma inline=forced
  inline float carg(_Fcomplex _Left)
  {       // compute carg
    return (cargf(_Left));
  }

  #pragma inline=forced
  inline float cimag(_Fcomplex _Left)
  {       // compute cimag
    return (cimagf(_Left));
  }

  #pragma inline=forced
  inline float creal(_Fcomplex _Left)
  {       // compute creal
    return (crealf(_Left));
  }

  #pragma inline=forced
  inline float fabs(_Fcomplex _Left)
  {       // compute cabs
    return (cabsf(_Left));
  }

  #pragma inline=forced
  inline float imag(_Fcomplex _Left)
  {       // compute cimag
    return (cimagf(_Left));
  }

  #pragma inline=forced
  inline float real(_Fcomplex _Left)
  {       // compute creal
    return (crealf(_Left));
  }

        // long double complex OVERLOADS
  #pragma inline=forced
  inline _Lcomplex acos(_Lcomplex _Left)
  {       // compute cacos
    return (cacosl(_Left));
  }

  #pragma inline=forced
  inline _Lcomplex acosh(_Lcomplex _Left)
  {       // compute cacosh
    return (cacoshl(_Left));
  }

  #pragma inline=forced
  inline _Lcomplex asin(_Lcomplex _Left)
  {       // compute casin
    return (casinl(_Left));
  }

  #pragma inline=forced
  inline _Lcomplex asinh(_Lcomplex _Left)
  {       // compute casinh
    return (casinhl(_Left));
  }

  #pragma inline=forced
  inline _Lcomplex atan(_Lcomplex _Left)
  {       // compute catan
    return (catanl(_Left));
  }

  #pragma inline=forced
  inline _Lcomplex atanh(_Lcomplex _Left)
  {       // compute catanh
    return (catanhl(_Left));
  }

  #pragma inline=forced
  inline _Lcomplex conj(_Lcomplex _Left)
  {       // compute conj
    return (conj(_Left));
  }

  #pragma inline=forced
  inline _Lcomplex cos(_Lcomplex _Left)
  {       // compute ccos
    return (ccosl(_Left));
  }

  #pragma inline=forced
  inline _Lcomplex cosh(_Lcomplex _Left)
  {       // compute ccosh
    return (ccoshl(_Left));
  }

  #pragma inline=forced
  inline _Lcomplex cproj(_Lcomplex _Left)
  {       // compute cproj
    return (cprojl(_Left));
  }

  #pragma inline=forced
  inline _Lcomplex exp(_Lcomplex _Left)
  {       // compute cexp
    return (cexpl(_Left));
  }

  #pragma inline=forced
  inline _Lcomplex log(_Lcomplex _Left)
  {       // compute clog
    return (clogl(_Left));
  }

  #pragma inline=forced
  inline _Lcomplex log10(_Lcomplex _Left)
  {       // compute clog10
    return (clog10l(_Left));
  }

  #pragma inline=forced
  inline long double norm(_Lcomplex _Left)
  {       // compute norm -- added with TR1
    return (norml(_Left));
  }

  #pragma inline=forced
  inline _Lcomplex pow(_Lcomplex _Left, _Lcomplex _Right)
  {       // compute cpow
    return (cpowl(_Left, _Right));
  }

  #pragma inline=forced
  inline _Lcomplex sin(_Lcomplex _Left)
  {       // compute csin
    return (csinl(_Left));
  }

  #pragma inline=forced
  inline _Lcomplex sinh(_Lcomplex _Left)
  {       // compute csinh
    return (csinhl(_Left));
  }

  #pragma inline=forced
  inline _Lcomplex sqrt(_Lcomplex _Left)
  {       // compute csqrt
    return (csqrtl(_Left));
  }

  #pragma inline=forced
  inline _Lcomplex tan(_Lcomplex _Left)
  {       // compute ctan
    return (ctanl(_Left));
  }

  #pragma inline=forced
  inline _Lcomplex tanh(_Lcomplex _Left)
  {       // compute ctanh
    return (ctanhl(_Left));
  }

  #pragma inline=forced
  inline long double abs(_Lcomplex _Left)
  {       // compute cabs
    return (cabsl(_Left));
  }

  #pragma inline=forced
  inline long double arg(_Lcomplex _Left)
  {       // compute carg
    return (cargl(_Left));
  }

  #pragma inline=forced
  inline long double carg(_Lcomplex _Left)
  {       // compute carg
    return (cargl(_Left));
  }

  #pragma inline=forced
  inline long double cimag(_Lcomplex _Left)
  {       // compute cimag
    return (cimagl(_Left));
  }

  #pragma inline=forced
  inline long double creal(_Lcomplex _Left)
  {       // compute creal
    return (creall(_Left));
  }

  #pragma inline=forced
  inline long double fabs(_Lcomplex _Left)
  {       // compute cabs
    return (cabsl(_Left));
  }

  #pragma inline=forced
  inline long double imag(_Lcomplex _Left)
  {       // compute cimag
    return (cimagl(_Left));
  }

  #pragma inline=forced
  inline long double real(_Lcomplex _Left)
  {       // compute creal
    return (creall(_Left));
  }
  _END_EXTERN_CPP
#endif /* __cplusplus */

_C_STD_END

/* GENERIC TEMPLATES */
#if _HAS_GENERIC_TEMPLATES
  _EXTERN_CPP
  _C_STD_BEGIN
  /* TEMPLATE CLASS _Rc_type (ADDITIONS TO <xtgmath.h>) */
  template<> struct _Rc_type<_Fcomplex>
  {       /* determine if type is real or complex */
    typedef char _Type;
  };

  template<> struct _Rc_type<_Dcomplex>
  {       /* determine if type is real or complex */
    typedef char _Type;
  };

  template<> struct _Rc_type<_Lcomplex>
  {       /* determine if type is real or complex */
    typedef char _Type;
  };

  /* TEMPLATE CLASS _Real_type (ADDITIONS TO <xtgmath.h>) */
  template<> struct _Real_type<_Fcomplex>
  {       /* determine equivalent real type */
    typedef float _Type;
  };

  template<> struct _Real_type<_Dcomplex>
  {       /* determine equivalent real type */
    typedef double _Type;
  };

  template<> struct _Real_type<_Lcomplex>
  {       /* determine equivalent real type */
    typedef long double _Type;
  };

  /* TEMPLATE CLASS _Combined_type (ADDITIONS TO <xtgmath.h>) */
  template<> struct _Combined_type<char, float>
  {       /* determine combined type */
    typedef _Fcomplex _Type;
  };

  template<> struct _Combined_type<char, double>
  {       /* determine combined type */
    typedef _Dcomplex _Type;
  };

  template<> struct _Combined_type<char, long double>
  {       /* determine combined type */
    typedef _Lcomplex _Type;
  };

  _TGEN_C0(carg)  /* generic overloads */
  _TGEN_C0(cimag)
  _TGEN_C(conj)
  _TGEN_C(cproj)
  _TGEN_C0(creal)

  _TGEN_C0(arg)   /* added with TR1 */
/*  _TGEN_C(conj) */
  _TGEN_C0(imag)
  _TGEN_C0(norm)
  _TGEN_C0(real)
  _C_STD_END
  _END_EXTERN_CPP

  #include <math.h>      /* define all overloads for complex functions */
#endif /* _HAS_GENERIC_TEMPLATES */

        /* SPECIAL HANDLING FOR clog */
#if _HAS_NAMESPACE
  namespace _iar_clog {
  _C_LIB_DECL
  __EFF_NS __ATTRIBUTES _CSTD _Dcomplex clog(_CSTD _Dcomplex);
  _END_C_LIB_DECL
  }	// namespace _iar_clog

  _EXTERN_CPP
  _C_STD_BEGIN
  __ATTRIBUTES _Dcomplex log(_Dcomplex _Left);

  inline _Dcomplex log(_Dcomplex _Left)
	  {	// compute clog
            return (_iar_clog::clog(_Left));
          }
  _C_STD_END
  _END_EXTERN_CPP

#else /* _HAS_NAMESPACE */
  _C_STD_BEGIN
  _C_LIB_DECL
  __EFF_NS __ATTRIBUTES _Dcomplex clog(_Dcomplex);
  _END_C_LIB_DECL

#ifdef __cplusplus
  _EXTERN_CPP
  __EFF_NS __ATTRIBUTES _Dcomplex log(_Dcomplex _Left);

  #pragma inline=forced
  inline _Dcomplex log(_Dcomplex _Left)
  {       // compute clog
    return (clog(_Left));
  }
  _END_EXTERN_CPP
#endif /* __cplusplus */

_C_STD_END
 #endif /* _HAS_NAMESPACE */

#endif /* _DLIB_ADD_C99_SYMBOLS */
#endif /* _COMPLEX */

#if defined(_STD_USING) && defined(__cplusplus)
#if _DLIB_ADD_C99_SYMBOLS
  using _CSTD _Dcomplex; using _CSTD _Fcomplex; using _CSTD _Lcomplex;
  #if _HAS_NAMESPACE
    using _iar_clog::clog;	/* SPECIAL HANDLING FOR clog */
  #else /* _HAS_NAMESPACE */
    using _CSTD clog;
  #endif /* _HAS_NAMESPACE */

  using _CSTD _D_FNAME(Cbuild); using _CSTD _D_FNAME(Cmulcc); 
  using _CSTD _D_FNAME(Cmulcr); using _CSTD _D_FNAME(Cdivcc); 
  using _CSTD _D_FNAME(Cdivcr); using _CSTD _D_FNAME(Caddcc);
  using _CSTD _D_FNAME(Caddcr); using _CSTD _D_FNAME(Csubcc);
  using _CSTD _D_FNAME(Csubcr);

  using _CSTD _F_FNAME(Cbuild); using _CSTD _F_FNAME(Cmulcc); 
  using _CSTD _F_FNAME(Cmulcr); using _CSTD _F_FNAME(Cdivcc); 
  using _CSTD _F_FNAME(Cdivcr); using _CSTD _F_FNAME(Caddcc);
  using _CSTD _F_FNAME(Caddcr); using _CSTD _F_FNAME(Csubcc); 
  using _CSTD _F_FNAME(Csubcr);

  using _CSTD _L_FNAME(Cbuild); using _CSTD _L_FNAME(Cmulcc); 
  using _CSTD _L_FNAME(Cmulcr); using _CSTD _L_FNAME(Cdivcc); 
  using _CSTD _L_FNAME(Cdivcr); using _CSTD _L_FNAME(Caddcc);
  using _CSTD _L_FNAME(Caddcr); using _CSTD _L_FNAME(Csubcc); 
  using _CSTD _L_FNAME(Csubcr);

  using _CSTD cabs; using _CSTD cacos; using _CSTD cacosh;
  using _CSTD carg; using _CSTD casin; using _CSTD casinh;
  using _CSTD catan; using _CSTD catanh; using _CSTD ccos;
  using _CSTD ccosh; using _CSTD cexp; using _CSTD cimag;
  /* using _CSTD clog; */ using _CSTD conj; using _CSTD cpow;
  using _CSTD cproj; using _CSTD creal; using _CSTD csin;
  using _CSTD csinh; using _CSTD csqrt; using _CSTD ctan;
  using _CSTD ctanh;

  using _CSTD cabsf; using _CSTD cacosf; using _CSTD cacoshf;
  using _CSTD cargf; using _CSTD casinf; using _CSTD casinhf;
  using _CSTD catanf; using _CSTD catanhf; using _CSTD ccosf;
  using _CSTD ccoshf; using _CSTD cexpf; using _CSTD cimagf;
  using _CSTD clogf; using _CSTD conjf; using _CSTD cpowf;
  using _CSTD cprojf; using _CSTD crealf; using _CSTD csinf;
  using _CSTD csinhf; using _CSTD csqrtf; using _CSTD ctanf;
  using _CSTD ctanhf;

  using _CSTD cabsl; using _CSTD cacosl; using _CSTD cacoshl;
  using _CSTD cargl; using _CSTD casinl; using _CSTD casinhl;
  using _CSTD catanl; using _CSTD catanhl; using _CSTD ccosl;
  using _CSTD ccoshl; using _CSTD cexpl; using _CSTD cimagl;
  using _CSTD clogl; using _CSTD conjl; using _CSTD cpowl;
  using _CSTD cprojl; using _CSTD creall; using _CSTD csinl;
  using _CSTD csinhl; using _CSTD csqrtl; using _CSTD ctanl;
  using _CSTD ctanhl;

  using _CSTD abs; using _CSTD acos; using _CSTD acosh;
  using _CSTD arg; using _CSTD imag; using _CSTD real;
  using _CSTD asin; using _CSTD asinh; using _CSTD atan;
  using _CSTD atanh; using _CSTD cos; using _CSTD cosh;
  using _CSTD exp; using _CSTD log; using _CSTD pow;
  using _CSTD sin; using _CSTD sinh; using _CSTD sqrt;
  using _CSTD tan; using _CSTD tanh; using _CSTD fabs;

  using _CSTD log10; using _CSTD norm;    /* added with TR1 */
#endif /* _DLIB_ADD_C99_SYMBOLS */
#endif /* defined(_STD_USING) && defined(__cplusplus) */


/*
 * Copyright (c) 1992-2009 by P.J. Plauger.  ALL RIGHTS RESERVED.
 * Consult your license regarding permissions and restrictions.
V5.04:0576 */
