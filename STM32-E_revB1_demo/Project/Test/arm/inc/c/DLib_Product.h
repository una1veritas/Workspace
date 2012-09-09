#ifndef _DLIB_PRODUCTS_H_
#define _DLIB_PRODUCTS_H_

#ifndef _SYSTEM_BUILD
   #pragma system_include
#endif


/*********************************************************************
*
*       Configuration
*
*********************************************************************/

/* ARM uses the large implementation of DLib */
#define _DLIB_SMALL_TARGET 0

/* This ensures that the standard header file "string.h" includes
 * the Arm-specific file "DLib_Product_string.h". */
#define _DLIB_PRODUCT_STRING 1

/* This ensures that the standard header file "fenv.h" includes
 * the Arm-specific file "DLib_Product_fenv.h". */
#define _DLIB_PRODUCT_FENV 1

/* Max buffer used for swap in qsort */
#define _DLIB_QSORT_BUF_SIZE 128

/* Enable system locking  */
#if !defined(__RWPI__)
#define _DLIB_THREAD_SUPPORT 3
#define _TLS_OBJECT_ATTRIBUTE __absolute
#define _TLS_CONST_DEFINITION
#else
#define _DLIB_THREAD_SUPPORT 2
#endif

/* Enable AEABI support */
#define _DLIB_SUPPORT_FOR_AEABI 1

/* Enable rtmodel for setjump buffer size */
#define _DLIB_USE_RTMODEL_JMP_BUF_NUM_ELEMENTS 1

/* Enable parsing of hex floats */
#define _DLIB_STRTOD_HEX_FLOAT 1

/* Special placement for locale structures when building ropi libraries */
#if defined(__ROPI__)
#define _LOCALE_PLACEMENT_ static
#endif

/* CPP-library uses software floatingpoint interface */
#define __SOFTFP  _Pragma( "type_attribute=__softfp" )

/* Use speedy implementation of floats (simple quad). */
#define _DLIB_SPEEDY_FLOATS 1

/*********************************************************************
*
*       Defines for va_arg & friends.
*
*********************************************************************/

#define _VA_DEFINED

  typedef struct
  {
    char *_Ap;
  } _VA_LIST;

  #define _SIZE_ON_STACK(T) \
          (((sizeof(T) + __VA_STACK_ALIGN__ - 1) / __VA_STACK_ALIGN__) * \
          __VA_STACK_ALIGN__)

  #define _VA_START(ap, A) (ap._Ap = _CSTD __va_start1())
  #define _VA_END(ap)      ((void) 0)

  #define _VA_ARG(ap, T) \
      (  __ALIGNOF__(T) > __VA_STACK_ALIGN__                \
       ? (  (*(long *) &(ap)._Ap) += (__ALIGNOF__(T) - 1),  \
            (*(long *) &(ap)._Ap) &= ~(__ALIGNOF__(T) - 1), \
            (*(*(T **) &(ap)._Ap)++))                       \
       : (_SIZE_ON_STACK(T) == sizeof(T)                    \
            ? (*(*(T **) &(ap)._Ap)++)                      \
            : ((ap)._Ap += _SIZE_ON_STACK(T),               \
                    *(T *) ((ap)._Ap - (_SIZE_ON_STACK(T))))))


#define _DLIB_USE_RTMODEL_JMP_BUF_NUM_ELEMENTS 1

#define _XMATHWRAPPERS_DEF xarmmathwrappers.h

#endif

