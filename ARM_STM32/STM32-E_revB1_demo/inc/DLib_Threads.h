/***************************************************
 *
 * DLib_Threads.h is the library threads manager.
 *
 * Copyright 2004-2010 IAR Systems AB.  
 *
 * This configuration header file sets up how the thread support in the library
 * should work.
 *
 ***************************************************
 *
 * DO NOT MODIFY THIS FILE!
 *
 ***************************************************/

#ifndef _DLIB_THREADS_H
#define _DLIB_THREADS_H

#ifndef _SYSTEM_BUILD
  #pragma system_include
#endif

/*
 * DLib can support a multithreaded environment. The preprocessor symbol 
 * _DLIB_THREAD_SUPPORT governs the support. It can be 0 (no support), 
 * 1 (currently not supported), 2 (locks only), and 3 (simulated TLS and locks).
 */

/*
 * This header sets the following symbols that governs the rest of the
 * library:
 * ------------------------------------------
 * _DLIB_MULTI_THREAD     0 No thread support
 *                        1 Multithread support
 * _DLIB_GLOBAL_VARIABLES 0 Use external TLS interface for the libraries global
 *                          and static variables
 *                        1 Use a lock for accesses to the locale and no 
 *                          security for accesses to other global and static
 *                          variables in the library
 * _DLIB_FILE_OP_LOCKS    0 No file-atomic locks
 *                        1 File-atomic locks

 * _DLIB_COMPILER_TLS     0 No Thread-Local-Storage support in the compiler
 *                        1 Thread-Local-Storage support in the compiler
 * _DLIB_TLS_QUAL         The TLS qualifier, define only if _COMPILER_TLS == 1
 *
 * _DLIB_THREAD_MACRO_SETUP_DONE Whether to use the standard definitions of
 *                               TLS macros defined in xtls.h or the definitions
 *                               are provided here.
 *                        0 Use default macros
 *                        1 Macros defined for xtls.h
 *
 * _DLIB_THREAD_LOCK_ONCE_TYPE
 *                        type for control variable in once-initialization of 
 *                        locks
 * _DLIB_THREAD_LOCK_ONCE_MACRO(control_variable, init_function)
 *                        expression that will be evaluated at each lock access
 *                        to determine if an initialization must be done
 * _DLIB_THREAD_LOCK_ONCE_TYPE_INIT
 *                        initial value for the control variable
 *
 ****************************************************************************
 * Description
 * -----------
 *
 * If locks are to be used (_DLIB_MULTI_THREAD != 0), the following options
 * has to be used in ilink: 
 *   --redirect __iar_Locksyslock=__iar_Locksyslock_mtx
 *   --redirect __iar_Unlocksyslock=__iar_Unlocksyslock_mtx
 *   --redirect __iar_Lockfilelock=__iar_Lockfilelock_mtx
 *   --redirect __iar_Unlockfilelock=__iar_Unlockfilelock_mtx
 *   --keep     __iar_Locksyslock_mtx
 * and, if C++ is used, also:
 *   --redirect __iar_Initdynamicfilelock=__iar_Initdynamicfilelock_mtx
 *   --redirect __iar_Dstdynamicfilelock=__iar_Dstdynamicfilelock_mtx
 *   --redirect __iar_Lockdynamicfilelock=__iar_Lockdynamicfilelock_mtx
 *   --redirect __iar_Unlockdynamicfilelock=__iar_Unlockdynamicfilelock_mtx
 * Xlink uses similar options (-e and -g). The following lock interface must
 * also be implemented: 
 *   typedef void *__iar_Rmtx;                   // Lock info object
 *
 *   void __iar_system_Mtxinit(__iar_Rmtx *);    // Initialize a system lock
 *   void __iar_system_Mtxdst(__iar_Rmtx *);     // Destroy a system lock
 *   void __iar_system_Mtxlock(__iar_Rmtx *);    // Lock a system lock
 *   void __iar_system_Mtxunlock(__iar_Rmtx *);  // Unlock a system lock
 * The interface handles locks for the heap, the locale, the file system
 * structure, the initialization of statics in functions, etc. 
 *
 * The following lock interface is optional to be implemented:
 *   void __iar_file_Mtxinit(__iar_Rmtx *);    // Initialize a file lock
 *   void __iar_file_Mtxdst(__iar_Rmtx *);     // Destroy a file lock
 *   void __iar_file_Mtxlock(__iar_Rmtx *);    // Lock a file lock
 *   void __iar_file_Mtxunlock(__iar_Rmtx *);  // Unlock a file lock
 * The interface handles locks for each file stream.
 * 
 * These three once-initialization symbols must also be defined, if the 
 * default initialization later on in this file doesn't work (done in 
 * DLib_product.h):
 *
 *   _DLIB_THREAD_LOCK_ONCE_TYPE
 *   _DLIB_THREAD_LOCK_ONCE_MACRO(control_variable, init_function)
 *   _DLIB_THREAD_LOCK_ONCE_TYPE_INIT
 *
 * If an external TLS interface is used, the following must
 * be defined:
 *   typedef int __iar_Tlskey_t;
 *   typedef void (*__iar_Tlsdtor_t)(void *);
 *   int __iar_Tlsalloc(__iar_Tlskey_t *, __iar_Tlsdtor_t); 
 *                                                    // Allocate a TLS element
 *   int __iar_Tlsfree(__iar_Tlskey_t);               // Free a TLS element
 *   int __iar_Tlsset(__iar_Tlskey_t, void *);        // Set a TLS element
 *   void *__iar_Tlsget(__iar_Tlskey_t);              // Get a TLS element
 *
 */

/* We don't have a compiler that supports tls declarations */
#define _DLIB_COMPILER_TLS 0
#define _DLIB_TLS_QUAL 

#if _DLIB_THREAD_SUPPORT == 0

  /* No support for threading. */

  #define _DLIB_MULTI_THREAD 0
  #define _DLIB_GLOBAL_VARIABLES 0
  #define _DLIB_FILE_OP_LOCKS 0

#elif _DLIB_THREAD_SUPPORT == 1

  /* Thread support, TLS via external interface, locks on heap and on FILE */

  #define _DLIB_MULTI_THREAD 1
  #define _DLIB_GLOBAL_VARIABLES 0
  #define _DLIB_FILE_OP_LOCKS 1

  /* Must be defined */
  _C_LIB_DECL
  typedef int __iar_Tlskey_t;
  #if !_DLIB_COMPILER_TLS
    typedef void (*__iar_Tlsdtor_t)(void *);
    __ATTRIBUTES int __iar_Tlsalloc(__iar_Tlskey_t *, __iar_Tlsdtor_t);
    __ATTRIBUTES int __iar_Tlsfree(__iar_Tlskey_t);
    __ATTRIBUTES int __iar_Tlsset(__iar_Tlskey_t, void *);
    __ATTRIBUTES void *__iar_Tlsget(__iar_Tlskey_t);
  #endif /* !_DLIB_COMPILER_TLS */
  _END_C_LIB_DECL

#elif _DLIB_THREAD_SUPPORT == 2

  /* Thread support, no TLS, locks on heap and on FILE */

  #define _DLIB_MULTI_THREAD 1
  #define _DLIB_GLOBAL_VARIABLES 1
  #define _DLIB_FILE_OP_LOCKS 1

#elif _DLIB_THREAD_SUPPORT == 3

  /* Thread support, library supports threaded variables in a user specified
     memory area, locks on heap and on FILE */

  /* See Documentation/ThreadsInternal.html for a description. */

  #define _DLIB_MULTI_THREAD 1
  #define _DLIB_GLOBAL_VARIABLES 0
  #define _DLIB_FILE_OP_LOCKS 1

  _C_LIB_DECL
  #define _DLIB_THREAD_MACRO_SETUP_DONE 1

  #if defined(_DLIB_TLS_MEMORY)
    #define _DLIB_TLS_MEMORY_GLUE(x) _GLUE3(_DLIB_TLS_MEMORY,_,x)
  #else
    #define _DLIB_TLS_MEMORY
    #define _DLIB_TLS_MEMORY_GLUE(x) x
  #endif

  #ifndef _DLIB_TLS_REQUIRE_INIT
    #define _DLIB_TLS_REQUIRE_INIT 0
  #endif

  #if _DLIB_TLS_REQUIRE_INIT
    extern void * __cstart_init_tls;
    #define _DLIB_TLS_REQUIRE_CSTART_INIT _Pragma("required=__cstart_init_tls")
  #else
    #define _DLIB_TLS_REQUIRE_CSTART_INIT
  #endif

  #if !defined(_DLIB_TLS_INITIALIZER_MEMORY)
    #define _DLIB_TLS_INITIALIZER_MEMORY _DLIB_TLS_MEMORY
  #endif

  #pragma language=save 
  #pragma language=extended
  __ATTRIBUTES void __iar_dlib_perthread_initialize(void _DLIB_TLS_MEMORY *);
  __ATTRIBUTES void _DLIB_TLS_MEMORY *__iar_dlib_perthread_allocate(void);
  __ATTRIBUTES void __iar_dlib_perthread_destroy(void);
  __ATTRIBUTES void __iar_dlib_perthread_deallocate(void _DLIB_TLS_MEMORY *);

  #ifndef _DLIB_TLS_SEGMENT_DATA
    #define _DLIB_TLS_SEGMENT_DATA "__DLIB_PERTHREAD"
  #endif

  #ifndef _DLIB_TLS_SEGMENT_INIT
    #define _DLIB_TLS_SEGMENT_INIT "__DLIB_PERTHREAD_init"
  #endif

  #pragma segment = _DLIB_TLS_SEGMENT_DATA _DLIB_TLS_MEMORY
  #pragma segment = _DLIB_TLS_SEGMENT_INIT _DLIB_TLS_INITIALIZER_MEMORY


  #define __IAR_DLIB_PERTHREAD_SIZE __segment_size(_DLIB_TLS_SEGMENT_DATA)
  #define __IAR_DLIB_PERTHREAD_INIT_SIZE \
                                    __segment_size(_DLIB_TLS_SEGMENT_INIT)

  #define __IAR_DLIB_PERTHREAD_SYMBOL_OFFSET(symbp) \
    (  (char _DLIB_TLS_MEMORY *) symbp              \
     - (char _DLIB_TLS_MEMORY *) __segment_begin(_DLIB_TLS_SEGMENT_DATA))

  #ifndef _TLS_OBJECT_ATTRIBUTE
    #define _TLS_OBJECT_ATTRIBUTE
  #endif

  #ifndef _TLS_CONST_DEFINITION
    #define _TLS_CONST_DEFINITION const
  #endif

  #ifndef _TLS_CONST_DECLARATION
    #define _TLS_CONST_DECLARATION _TLS_CONST_DEFINITION
  #endif

  #define _TLS_LOCATION_HELPER0(x) #x
  #define _TLS_LOCATION_HELPER1(x) _TLS_LOCATION_HELPER0(location=##x)
  #define _TLS_LOCATION_HELPER2(x) _TLS_LOCATION_HELPER1(x)

  #define _TLS_DEFINITION(scope, type, name)                 \
    _Pragma("language=save") _Pragma("language=extended")    \
    _Pragma(_TLS_LOCATION_HELPER2(_DLIB_TLS_SEGMENT_DATA)) \
    _DLIB_TLS_REQUIRE_CSTART_INIT                            \
    scope _DLIB_TLS_INITIALIZER_MEMORY _TLS_OBJECT_ATTRIBUTE \
      type _TLS_CONST_DEFINITION name \
    _Pragma("language=restore")

  /* The thread-local variable access function */
  void _DLIB_TLS_MEMORY *__iar_dlib_perthread_access(void _DLIB_TLS_MEMORY *);
  #pragma language=restore

  #define _TLS_REFERENCE(type, name) \
    _Pragma("language=save") _Pragma("language=extended") \
    ((type _DLIB_TLS_MEMORY *)       \
     __iar_dlib_perthread_access((_DLIB_TLS_MEMORY char *) &name)) \
    _Pragma("language=restore")

  #define _TLS_REFERENCE_DEF(type, name) \
    _TLS_REFERENCE(type, name)


  #define _TLS_DATA_DECL(type, name) \
    _Pragma("language=save") _Pragma("language=extended") \
    extern _DLIB_TLS_INITIALIZER_MEMORY _TLS_OBJECT_ATTRIBUTE \
      type _TLS_CONST_DECLARATION name \
    _Pragma("language=restore")

  #define _TLS_DEFINE_INIT(scope, type, name) \
    _TLS_DEFINITION(scope, type, name)

  #define _TLS_DEFINE_NO_INIT(scope, type, name) 

  #define _TLS_DATA_DEF(scope, type, name, init) \
    _TLS_DEFINE_INIT(scope, type, name) = init 

  #define _TLS_DATA_DEF_DT(scope, type, name, init, dtor) \
    _TLS_DEFINE_INIT(scope, type, name) = init 
    /* Make sure that each destructor is inserted into _Deallocate_TLS */
  
  #define _TLS_DATA_PTR(type, name) \
    _TLS_REFERENCE_DEF(type, name)

  #define _TLS_ARR_DEF(scope, type, name, elts) \
    _TLS_DEFINITION(scope, type, name)[elts]

  #define _TLS_ARR(type, name) \
    _TLS_REFERENCE_DEF(type, name)

  /* Internal function declarations. */
  #if _DLIB_FULL_LOCALE_SUPPORT
    __ATTRIBUTES void __iar_Locale_lconv_init(void);
  #endif

  _END_C_LIB_DECL

#else
  #error "Thread support erroneously setup"
#endif

  _C_LIB_DECL
  typedef void *__iar_Rmtx;
  _END_C_LIB_DECL
#if _DLIB_THREAD_SUPPORT != 0
  _C_LIB_DECL
  __ATTRIBUTES void __iar_system_Mtxinit(__iar_Rmtx *m);
  __ATTRIBUTES void __iar_system_Mtxdst(__iar_Rmtx *m);
  __ATTRIBUTES void __iar_system_Mtxlock(__iar_Rmtx *m);
  __ATTRIBUTES void __iar_system_Mtxunlock(__iar_Rmtx *m);

  __ATTRIBUTES void __iar_file_Mtxinit(__iar_Rmtx *m);
  __ATTRIBUTES void __iar_file_Mtxdst(__iar_Rmtx *m);
  __ATTRIBUTES void __iar_file_Mtxlock(__iar_Rmtx *m);
  __ATTRIBUTES void __iar_file_Mtxunlock(__iar_Rmtx *m);

  /* Function to destroy the locks. Should be called after atexit and 
     _Close_all. */
  __ATTRIBUTES void __iar_clearlocks(void);


#ifndef _DLIB_THREAD_LOCK_ONCE_TYPE
  #define _DLIB_THREAD_LOCK_ONCE_TYPE      unsigned
  #define _DLIB_THREAD_LOCK_ONCE_MACRO(pCv, ifun) \
                                       { if (*pCv == 0) ifun(); *pCv = 1; }
  #define _DLIB_THREAD_LOCK_ONCE_TYPE_INIT 0
#endif

  _END_C_LIB_DECL
#endif /* _DLIB_THREAD_SUPPORT != 0 */



#if _DLIB_THREAD_SUPPORT >= 1 && _DLIB_THREAD_SUPPORT <= 3
  _C_LIB_DECL

  #if !defined(_DLIB_THREAD_LOCK_ONCE_TYPE)
    #error "Must define _DLIB_THREAD_LOCK_ONCE_TYPE"
  #endif
  #if !defined(_DLIB_THREAD_LOCK_ONCE_MACRO)
    #error "Must define _DLIB_THREAD_LOCK_ONCE_MACRO"
  #endif
  #if !defined(_DLIB_THREAD_LOCK_ONCE_TYPE_INIT)
    #error "Must define _DLIB_THREAD_LOCK_ONCE_TYPE_INIT"
  #endif

  typedef _DLIB_THREAD_LOCK_ONCE_TYPE _Once_t;
  #define _Once _DLIB_THREAD_LOCK_ONCE_MACRO
  #define _ONCE_T_INIT  _DLIB_THREAD_LOCK_ONCE_TYPE_INIT

  _END_C_LIB_DECL
#endif /* _DLIB_THREAD_SUPPORT >= 1 && _DLIB_THREAD_SUPPORT <= 3 */



#define _MULTI_THREAD _DLIB_MULTI_THREAD
#define _GLOBAL_LOCALE _DLIB_GLOBAL_VARIABLES
#define _FILE_OP_LOCKS _DLIB_FILE_OP_LOCKS

#define _COMPILER_TLS _DLIB_COMPILER_TLS
#define _TLS_QUAL _DLIB_TLS_QUAL

#endif /* _DLIB_THREADS_H */

