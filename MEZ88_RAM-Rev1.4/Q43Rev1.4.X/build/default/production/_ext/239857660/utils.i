# 1 "../drivers/utils.c"
# 1 "<built-in>" 1
# 1 "<built-in>" 3
# 295 "<built-in>" 3
# 1 "<command line>" 1
# 1 "<built-in>" 2
# 1 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include/language_support.h" 1 3
# 2 "<built-in>" 2
# 1 "../drivers/utils.c" 2
# 31 "../drivers/utils.c"
# 1 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/stdio.h" 1 3



# 1 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/musl_xc8.h" 1 3
# 5 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/stdio.h" 2 3





# 1 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/features.h" 1 3
# 11 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/stdio.h" 2 3
# 24 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/stdio.h" 3
# 1 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/bits/alltypes.h" 1 3
# 12 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/bits/alltypes.h" 3
typedef void * va_list[1];




typedef void * __isoc_va_list[1];
# 128 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/bits/alltypes.h" 3
typedef unsigned size_t;
# 143 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/bits/alltypes.h" 3
typedef __int24 ssize_t;
# 174 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/bits/alltypes.h" 3
typedef __int24 int24_t;
# 210 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/bits/alltypes.h" 3
typedef __uint24 uint24_t;
# 255 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/bits/alltypes.h" 3
typedef long long off_t;
# 409 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/bits/alltypes.h" 3
typedef struct _IO_FILE FILE;
# 25 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/stdio.h" 2 3
# 52 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/stdio.h" 3
typedef union _G_fpos64_t {
 char __opaque[16];
 double __align;
} fpos_t;

extern FILE *const stdin;
extern FILE *const stdout;
extern FILE *const stderr;





FILE *fopen(const char *restrict, const char *restrict);
FILE *freopen(const char *restrict, const char *restrict, FILE *restrict);
int fclose(FILE *);

int remove(const char *);
int rename(const char *, const char *);

int feof(FILE *);
int ferror(FILE *);
int fflush(FILE *);
void clearerr(FILE *);

int fseek(FILE *, long, int);
long ftell(FILE *);
void rewind(FILE *);

int fgetpos(FILE *restrict, fpos_t *restrict);
int fsetpos(FILE *, const fpos_t *);

size_t fread(void *restrict, size_t, size_t, FILE *restrict);
size_t fwrite(const void *restrict, size_t, size_t, FILE *restrict);

int fgetc(FILE *);
int getc(FILE *);
int getchar(void);





int ungetc(int, FILE *);
int getch(void);

int fputc(int, FILE *);
int putc(int, FILE *);
int putchar(int);





void putch(char);

char *fgets(char *restrict, int, FILE *restrict);

char *gets(char *);


int fputs(const char *restrict, FILE *restrict);
int puts(const char *);

__attribute__((__format__(__printf__, 1, 2)))
int printf(const char *restrict, ...);
__attribute__((__format__(__printf__, 2, 3)))
int fprintf(FILE *restrict, const char *restrict, ...);
__attribute__((__format__(__printf__, 2, 3)))
int sprintf(char *restrict, const char *restrict, ...);
__attribute__((__format__(__printf__, 3, 4)))
int snprintf(char *restrict, size_t, const char *restrict, ...);

__attribute__((__format__(__printf__, 1, 0)))
int vprintf(const char *restrict, __isoc_va_list);
int vfprintf(FILE *restrict, const char *restrict, __isoc_va_list);
__attribute__((__format__(__printf__, 2, 0)))
int vsprintf(char *restrict, const char *restrict, __isoc_va_list);
__attribute__((__format__(__printf__, 3, 0)))
int vsnprintf(char *restrict, size_t, const char *restrict, __isoc_va_list);

__attribute__((__format__(__scanf__, 1, 2)))
int scanf(const char *restrict, ...);
__attribute__((__format__(__scanf__, 2, 3)))
int fscanf(FILE *restrict, const char *restrict, ...);
__attribute__((__format__(__scanf__, 2, 3)))
int sscanf(const char *restrict, const char *restrict, ...);

__attribute__((__format__(__scanf__, 1, 0)))
int vscanf(const char *restrict, __isoc_va_list);
int vfscanf(FILE *restrict, const char *restrict, __isoc_va_list);
__attribute__((__format__(__scanf__, 2, 0)))
int vsscanf(const char *restrict, const char *restrict, __isoc_va_list);

void perror(const char *);

int setvbuf(FILE *restrict, char *restrict, int, size_t);
void setbuf(FILE *restrict, char *restrict);

char *tmpnam(char *);
FILE *tmpfile(void);




FILE *fmemopen(void *restrict, size_t, const char *restrict);
FILE *open_memstream(char **, size_t *);
FILE *fdopen(int, const char *);
FILE *popen(const char *, const char *);
int pclose(FILE *);
int fileno(FILE *);
int fseeko(FILE *, off_t, int);
off_t ftello(FILE *);
int dprintf(int, const char *restrict, ...);
int vdprintf(int, const char *restrict, __isoc_va_list);
void flockfile(FILE *);
int ftrylockfile(FILE *);
void funlockfile(FILE *);
int getc_unlocked(FILE *);
int getchar_unlocked(void);
int putc_unlocked(int, FILE *);
int putchar_unlocked(int);
ssize_t getdelim(char **restrict, size_t *restrict, int, FILE *restrict);
ssize_t getline(char **restrict, size_t *restrict, FILE *restrict);
int renameat(int, const char *, int, const char *);
char *ctermid(char *);







char *tempnam(const char *, const char *);
# 32 "../drivers/utils.c" 2
# 1 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/ctype.h" 1 3
# 10 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/ctype.h" 3
int isalnum(int);
int isalpha(int);
int isblank(int);
int iscntrl(int);
int isdigit(int);
int isgraph(int);
int islower(int);
int isprint(int);
int ispunct(int);
int isspace(int);
int isupper(int);
int isxdigit(int);
int tolower(int);
int toupper(int);
# 48 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/ctype.h" 3
# 1 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/bits/alltypes.h" 1 3
# 421 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/bits/alltypes.h" 3
typedef struct __locale_struct * locale_t;
# 49 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/ctype.h" 2 3

int isalnum_l(int, locale_t);
int isalpha_l(int, locale_t);
int isblank_l(int, locale_t);
int iscntrl_l(int, locale_t);
int isdigit_l(int, locale_t);
int isgraph_l(int, locale_t);
int islower_l(int, locale_t);
int isprint_l(int, locale_t);
int ispunct_l(int, locale_t);
int isspace_l(int, locale_t);
int isupper_l(int, locale_t);
int isxdigit_l(int, locale_t);
int tolower_l(int, locale_t);
int toupper_l(int, locale_t);
# 33 "../drivers/utils.c" 2
# 1 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/stdint.h" 1 3
# 26 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/stdint.h" 3
# 1 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/bits/alltypes.h" 1 3
# 133 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/bits/alltypes.h" 3
typedef unsigned __int24 uintptr_t;
# 148 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/bits/alltypes.h" 3
typedef __int24 intptr_t;
# 164 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/bits/alltypes.h" 3
typedef signed char int8_t;




typedef short int16_t;
# 179 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/bits/alltypes.h" 3
typedef long int32_t;





typedef long long int64_t;
# 194 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/bits/alltypes.h" 3
typedef long long intmax_t;





typedef unsigned char uint8_t;




typedef unsigned short uint16_t;
# 215 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/bits/alltypes.h" 3
typedef unsigned long uint32_t;





typedef unsigned long long uint64_t;
# 235 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/bits/alltypes.h" 3
typedef unsigned long long uintmax_t;
# 27 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/stdint.h" 2 3

typedef int8_t int_fast8_t;

typedef int64_t int_fast64_t;


typedef int8_t int_least8_t;
typedef int16_t int_least16_t;

typedef int24_t int_least24_t;
typedef int24_t int_fast24_t;

typedef int32_t int_least32_t;

typedef int64_t int_least64_t;


typedef uint8_t uint_fast8_t;

typedef uint64_t uint_fast64_t;


typedef uint8_t uint_least8_t;
typedef uint16_t uint_least16_t;

typedef uint24_t uint_least24_t;
typedef uint24_t uint_fast24_t;

typedef uint32_t uint_least32_t;

typedef uint64_t uint_least64_t;
# 148 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/stdint.h" 3
# 1 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/bits/stdint.h" 1 3
typedef int16_t int_fast16_t;
typedef int32_t int_fast32_t;
typedef uint16_t uint_fast16_t;
typedef uint32_t uint_fast32_t;
# 149 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/stdint.h" 2 3
# 34 "../drivers/utils.c" 2
# 1 "../drivers/../drivers/utils.h" 1
# 31 "../drivers/../drivers/utils.h"
void util_hexdump(const char *header, const void *addr, unsigned int size);
void util_hexdump_sum(const char *header, const void *addr, unsigned int size);
void util_addrdump(const char *header, uint32_t addr_offs, const void *addr, unsigned int size);
int util_stricmp(const char *a, const char *b);
# 35 "../drivers/utils.c" 2

void util_hexdump(const char *header, const void *addr, unsigned int size)
{
    char chars[17];
    const uint8_t *buf = addr;
    size = ((size + 15) & ~0xfU);
    for (int i = 0; i < size; i++) {
        if ((i % 16) == 0)
            printf("%s%04x:", header, i);
        printf(" %02x", buf[i]);
        if (0x20 <= buf[i] && buf[i] <= 0x7e) {
            chars[i % 16] = buf[i];
        } else {
            chars[i % 16] = '.';
        }
        if ((i % 16) == 15) {
            chars[16] = '\0';
            printf(" %s\n\r", chars);
        }
    }
}

void util_addrdump(const char *header, uint32_t addr_offs, const void *addr, unsigned int size)
{
    char chars[17];
    const uint8_t *buf = addr;
    size = ((size + 15) & ~0xfU);
    for (unsigned int i = 0; i < size; i++) {
        if ((i % 16) == 0)
            printf("%s%06lx:", header, addr_offs + i);
        printf(" %02x", buf[i]);
        if (0x20 <= buf[i] && buf[i] <= 0x7e) {
            chars[i % 16] = buf[i];
        } else {
            chars[i % 16] = '.';
        }
        if ((i % 16) == 15) {
            chars[16] = '\0';
            printf(" %s\n\r", chars);
        }
    }
}

void util_hexdump_sum(const char *header, const void *addr, unsigned int size)
{
    util_hexdump(header, addr, size);

    uint8_t sum = 0;
    const uint8_t *p = addr;
    for (int i = 0; i < size; i++)
        sum += *p++;
    printf("%s%53s CHECKSUM: %02x\n\r", header, "", sum);
}
# 101 "../drivers/utils.c"
uint16_t chk_leap(uint16_t year) {

 uint16_t n;

 n = 0;
 if( (year%4 == 0) && (year%100 > 0) ) n=1;
 if( year%400==0 ) n=1;
 return(n);
}

static uint16_t leaps(uint16_t year) {

 uint16_t y, n;

 for(n=0,y=1980;y <=year;y++){
  if( chk_leap( y ) ) n++;
 }
 return(n);
}

const uint16_t mtod[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };

static uint16_t jan_to_days(uint16_t month, uint16_t day) {

 uint16_t i, d;

 for( d=0, i=0; i<month-1; i++ ) d += mtod[i];
 d += day;
 return( d );
}

uint16_t days_from_1980(uint16_t year, uint16_t month, uint16_t day) {
 uint16_t d1, d2;

 d1 = leaps(year);
 if ( month == 2 && day == 29 ) d1--;
 d2 = jan_to_days(month, day);
 if (d2 <= 59 && chk_leap( year ) ) d1--;

 return((year - 1980)*365 + d1 + d2 - 1 );
}

uint8_t cnv_bcd(uint8_t bval) {
 union {
  struct {
   uint8_t hex;
   uint8_t bcd;
  } conv ;
  uint16_t buf ;
 } convbcd ;

 uint8_t bitcnt;

 convbcd.buf = 0;
 convbcd.conv.hex = bval;

 for (bitcnt = 0 ; bitcnt < 8 ; bitcnt++) {
  if (((convbcd.conv.bcd & 0x0f) + 0x03) >= 0x08) convbcd.conv.bcd += 0x03;
  if (((convbcd.conv.bcd & 0xf0) + 0x30) >= 0x80) convbcd.conv.bcd += 0x30;
  convbcd.buf <<= 1;
 }
 return convbcd.conv.bcd;
}

uint8_t cnv_byte(uint8_t bval) {

 uint8_t convbin ;

 convbin = ((bval & 0xf0) >> 4) * 10 + (bval & 0x0f) ;
 return convbin ;
}

void cnv_ymd(uint16_t n_date, uint16_t *year, uint16_t *month, uint16_t *date ) {
 uint16_t remain_date;
 uint16_t y, m, d, leaps;

 remain_date = n_date + 1;
 y = 1980;
 for(;;) {
  leaps = chk_leap( y );
  d = 365 + leaps;
  if ( remain_date <= d ) break;
  remain_date -= d;
  y++;
 }
 if (y >= 2000) y-= 2000;
 else y-=1900;

 *year = y;
 m = 0;
 for(;;) {
  d = mtod[m];
  if ( m == 1 ) d += leaps;
  if ( remain_date <= d ) break;
  remain_date -= d;
  m++;
 };
 *month = ++m;
 *date = remain_date;
}
