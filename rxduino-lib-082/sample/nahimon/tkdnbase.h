#ifndef H_TKDNMON_BASE
#define H_TKDNMON_BASE

typedef int BOOL;

#ifndef TRUE
 #define TRUE 1
#endif

#ifndef FALSE
 #define FALSE 0
#endif

#ifndef NULL
 #define NULL ((void *)0)
#endif

#ifndef EOF
 #define EOF (-1)
#endif

// ヒストリバッファの深さ
#define PREVCMD_BUFSIZE 100

#endif
