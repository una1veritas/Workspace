# 1 "../fatfs/ff.c"
# 1 "<built-in>" 1
# 1 "<built-in>" 3
# 295 "<built-in>" 3
# 1 "<command line>" 1
# 1 "<built-in>" 2
# 1 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include/language_support.h" 1 3
# 2 "<built-in>" 2
# 1 "../fatfs/ff.c" 2
# 22 "../fatfs/ff.c"
# 1 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/string.h" 1 3



# 1 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/musl_xc8.h" 1 3
# 5 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/string.h" 2 3





# 1 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/features.h" 1 3
# 11 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/string.h" 2 3
# 25 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/string.h" 3
# 1 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/bits/alltypes.h" 1 3
# 128 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/bits/alltypes.h" 3
typedef unsigned size_t;
# 174 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/bits/alltypes.h" 3
typedef __int24 int24_t;
# 210 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/bits/alltypes.h" 3
typedef __uint24 uint24_t;
# 421 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/bits/alltypes.h" 3
typedef struct __locale_struct * locale_t;
# 26 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/string.h" 2 3

void *memcpy (void *restrict, const void *restrict, size_t);
void *memmove (void *, const void *, size_t);
void *memset (void *, int, size_t);
int memcmp (const void *, const void *, size_t);
void *memchr (const void *, int, size_t);

char *strcpy (char *restrict, const char *restrict);
char *strncpy (char *restrict, const char *restrict, size_t);

char *strcat (char *restrict, const char *restrict);
char *strncat (char *restrict, const char *restrict, size_t);

int strcmp (const char *, const char *);
int strncmp (const char *, const char *, size_t);

int strcoll (const char *, const char *);
size_t strxfrm (char *restrict, const char *restrict, size_t);

char *strchr (const char *, int);
char *strrchr (const char *, int);

size_t strcspn (const char *, const char *);
size_t strspn (const char *, const char *);
char *strpbrk (const char *, const char *);
char *strstr (const char *, const char *);
char *strtok (char *restrict, const char *restrict);

size_t strlen (const char *);

char *strerror (int);




char *strtok_r (char *restrict, const char *restrict, char **restrict);
int strerror_r (int, char *, size_t);
char *stpcpy(char *restrict, const char *restrict);
char *stpncpy(char *restrict, const char *restrict, size_t);
size_t strnlen (const char *, size_t);
char *strdup (const char *);
char *strndup (const char *, size_t);
char *strsignal(int);
char *strerror_l (int, locale_t);
int strcoll_l (const char *, const char *, locale_t);
size_t strxfrm_l (char *restrict, const char *restrict, size_t, locale_t);




void *memccpy (void *restrict, const void *restrict, int, size_t);
# 23 "../fatfs/ff.c" 2
# 1 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/stdio.h" 1 3
# 24 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/stdio.h" 3
# 1 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/bits/alltypes.h" 1 3
# 12 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/bits/alltypes.h" 3
typedef void * va_list[1];




typedef void * __isoc_va_list[1];
# 143 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/bits/alltypes.h" 3
typedef __int24 ssize_t;
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
# 24 "../fatfs/ff.c" 2
# 1 "../fatfs/../fatfs/ff.h" 1
# 29 "../fatfs/../fatfs/ff.h"
# 1 "../fatfs/../fatfs/../fatfs/ffconf.h" 1
# 30 "../fatfs/../fatfs/ff.h" 2
# 48 "../fatfs/../fatfs/ff.h"
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
# 49 "../fatfs/../fatfs/ff.h" 2
typedef unsigned int UINT;
typedef unsigned char BYTE;
typedef uint16_t WORD;
typedef uint32_t DWORD;
typedef uint64_t QWORD;
typedef WORD WCHAR;
# 82 "../fatfs/../fatfs/ff.h"
typedef DWORD FSIZE_t;
typedef DWORD LBA_t;
# 105 "../fatfs/../fatfs/ff.h"
typedef char TCHAR;
# 132 "../fatfs/../fatfs/ff.h"
typedef struct {
 BYTE fs_type;
 BYTE pdrv;
 BYTE ldrv;
 BYTE n_fats;
 BYTE wflag;
 BYTE fsi_flag;
 WORD id;
 WORD n_rootdir;
 WORD csize;
# 152 "../fatfs/../fatfs/ff.h"
 DWORD last_clst;
 DWORD free_clst;
# 163 "../fatfs/../fatfs/ff.h"
 DWORD n_fatent;
 DWORD fsize;
 LBA_t volbase;
 LBA_t fatbase;
 LBA_t dirbase;
 LBA_t database;



 LBA_t winsect;
 BYTE win[512];
} FATFS;





typedef struct {
 FATFS* fs;
 WORD id;
 BYTE attr;
 BYTE stat;
 DWORD sclust;
 FSIZE_t objsize;
# 197 "../fatfs/../fatfs/ff.h"
} FFOBJID;





typedef struct {
 FFOBJID obj;
 BYTE flag;
 BYTE err;
 FSIZE_t fptr;
 DWORD clust;
 LBA_t sect;

 LBA_t dir_sect;
 BYTE* dir_ptr;





 BYTE buf[512];

} FIL;





typedef struct {
 FFOBJID obj;
 DWORD dptr;
 DWORD clust;
 LBA_t sect;
 BYTE* dir;
 BYTE fn[12];






} DIR;





typedef struct {
 FSIZE_t fsize;
 WORD fdate;
 WORD ftime;
 BYTE fattrib;




 TCHAR fname[12 + 1];

} FILINFO;





typedef struct {
 BYTE fmt;
 BYTE n_fat;
 UINT align;
 UINT n_root;
 DWORD au_size;
} MKFS_PARM;





typedef enum {
 FR_OK = 0,
 FR_DISK_ERR,
 FR_INT_ERR,
 FR_NOT_READY,
 FR_NO_FILE,
 FR_NO_PATH,
 FR_INVALID_NAME,
 FR_DENIED,
 FR_EXIST,
 FR_INVALID_OBJECT,
 FR_WRITE_PROTECTED,
 FR_INVALID_DRIVE,
 FR_NOT_ENABLED,
 FR_NO_FILESYSTEM,
 FR_MKFS_ABORTED,
 FR_TIMEOUT,
 FR_LOCKED,
 FR_NOT_ENOUGH_CORE,
 FR_TOO_MANY_OPEN_FILES,
 FR_INVALID_PARAMETER
} FRESULT;
# 304 "../fatfs/../fatfs/ff.h"
FRESULT f_open (FIL* fp, const TCHAR* path, BYTE mode);
FRESULT f_close (FIL* fp);
FRESULT f_read (FIL* fp, void* buff, UINT btr, UINT* br);
FRESULT f_write (FIL* fp, const void* buff, UINT btw, UINT* bw);
FRESULT f_lseek (FIL* fp, FSIZE_t ofs);
FRESULT f_truncate (FIL* fp);
FRESULT f_sync (FIL* fp);
FRESULT f_opendir (DIR* dp, const TCHAR* path);
FRESULT f_closedir (DIR* dp);
FRESULT f_readdir (DIR* dp, FILINFO* fno);
FRESULT f_findfirst (DIR* dp, FILINFO* fno, const TCHAR* path, const TCHAR* pattern);
FRESULT f_findnext (DIR* dp, FILINFO* fno);
FRESULT f_mkdir (const TCHAR* path);
FRESULT f_unlink (const TCHAR* path);
FRESULT f_rename (const TCHAR* path_old, const TCHAR* path_new);
FRESULT f_stat (const TCHAR* path, FILINFO* fno);
FRESULT f_chmod (const TCHAR* path, BYTE attr, BYTE mask);
FRESULT f_utime (const TCHAR* path, const FILINFO* fno);
FRESULT f_chdir (const TCHAR* path);
FRESULT f_chdrive (const TCHAR* path);
FRESULT f_getcwd (TCHAR* buff, UINT len);
FRESULT f_getfree (const TCHAR* path, DWORD* nclst, FATFS** fatfs);
FRESULT f_getlabel (const TCHAR* path, TCHAR* label, DWORD* vsn);
FRESULT f_setlabel (const TCHAR* label);
FRESULT f_forward (FIL* fp, UINT(*func)(const BYTE*,UINT), UINT btf, UINT* bf);
FRESULT f_expand (FIL* fp, FSIZE_t fsz, BYTE opt);
FRESULT f_mount (FATFS* fs, const TCHAR* path, BYTE opt);
FRESULT f_mkfs (const TCHAR* path, const MKFS_PARM* opt, void* work, UINT len);
FRESULT f_fdisk (BYTE pdrv, const LBA_t ptbl[], void* work);
FRESULT f_setcp (WORD cp);
int f_putc (TCHAR c, FIL* fp);
int f_puts (const TCHAR* str, FIL* cp);
int f_printf (FIL* fp, const TCHAR* str, ...);
TCHAR* f_gets (TCHAR* buff, int len, FIL* fp);
# 359 "../fatfs/../fatfs/ff.h"
DWORD get_fattime (void);
# 25 "../fatfs/ff.c" 2
# 1 "../fatfs/../fatfs/diskio.h" 1
# 13 "../fatfs/../fatfs/diskio.h"
typedef BYTE DSTATUS;


typedef enum {
 RES_OK = 0,
 RES_ERROR,
 RES_WRPRT,
 RES_NOTRDY,
 RES_PARERR
} DRESULT;






DSTATUS disk_initialize (BYTE pdrv);
DSTATUS disk_status (BYTE pdrv);
DRESULT disk_read (BYTE pdrv, BYTE* buff, LBA_t sector, UINT count);
DRESULT disk_write (BYTE pdrv, const BYTE* buff, LBA_t sector, UINT count);
DRESULT disk_ioctl (BYTE pdrv, BYTE cmd, void* buff);
# 26 "../fatfs/ff.c" 2
# 465 "../fatfs/ff.c"
static FATFS *FatFs[1];
static WORD Fsid;
# 601 "../fatfs/ff.c"
static const BYTE DbcTbl[] = {0x81, 0x9F, 0xE0, 0xFC, 0x40, 0x7E, 0x80, 0xFC, 0x00, 0x00};
# 619 "../fatfs/ff.c"
static WORD ld_word (const BYTE* ptr)
{
 WORD rv;

 rv = ptr[1];
 rv = rv << 8 | ptr[0];
 return rv;
}

static DWORD ld_dword (const BYTE* ptr)
{
 DWORD rv;

 rv = ptr[3];
 rv = rv << 8 | ptr[2];
 rv = rv << 8 | ptr[1];
 rv = rv << 8 | ptr[0];
 return rv;
}
# 657 "../fatfs/ff.c"
static void st_word (BYTE* ptr, WORD val)
{
 *ptr++ = (BYTE)val; val >>= 8;
 *ptr++ = (BYTE)val;
}

static void st_dword (BYTE* ptr, DWORD val)
{
 *ptr++ = (BYTE)val; val >>= 8;
 *ptr++ = (BYTE)val; val >>= 8;
 *ptr++ = (BYTE)val; val >>= 8;
 *ptr++ = (BYTE)val;
}
# 693 "../fatfs/ff.c"
static int dbc_1st (BYTE c)
{






 if (c >= DbcTbl[0]) {
  if (c <= DbcTbl[1]) return 1;
  if (c >= DbcTbl[2] && c <= DbcTbl[3]) return 1;
 }



 return 0;
}



static int dbc_2nd (BYTE c)
{







 if (c >= DbcTbl[4]) {
  if (c <= DbcTbl[5]) return 1;
  if (c >= DbcTbl[6] && c <= DbcTbl[7]) return 1;
  if (c >= DbcTbl[8] && c <= DbcTbl[9]) return 1;
 }



 return 0;
}
# 1056 "../fatfs/ff.c"
static FRESULT sync_window (
 FATFS* fs
)
{
 FRESULT res = FR_OK;


 if (fs->wflag) {
  if (disk_write(fs->pdrv, fs->win, fs->winsect, 1) == RES_OK) {
   fs->wflag = 0;
   if (fs->winsect - fs->fatbase < fs->fsize) {
    if (fs->n_fats == 2) disk_write(fs->pdrv, fs->win, fs->winsect + fs->fsize, 1);
   }
  } else {
   res = FR_DISK_ERR;
  }
 }
 return res;
}



static FRESULT move_window (
 FATFS* fs,
 LBA_t sect
)
{
 FRESULT res = FR_OK;


 if (sect != fs->winsect) {

  res = sync_window(fs);

  if (res == FR_OK) {
   if (disk_read(fs->pdrv, fs->win, sect, 1) != RES_OK) {
    sect = (LBA_t)0 - 1;
    res = FR_DISK_ERR;
   }
   fs->winsect = sect;
  }
 }
 return res;
}
# 1109 "../fatfs/ff.c"
static FRESULT sync_fs (
 FATFS* fs
)
{
 FRESULT res;


 res = sync_window(fs);
 if (res == FR_OK) {
  if (fs->fs_type == 3 && fs->fsi_flag == 1) {

   memset(fs->win, 0, sizeof fs->win);
   st_word(fs->win + 510, 0xAA55);
   st_dword(fs->win + 0, 0x41615252);
   st_dword(fs->win + 484, 0x61417272);
   st_dword(fs->win + 488, fs->free_clst);
   st_dword(fs->win + 492, fs->last_clst);
   fs->winsect = fs->volbase + 1;
   disk_write(fs->pdrv, fs->win, fs->winsect, 1);
   fs->fsi_flag = 0;
  }

  if (disk_ioctl(fs->pdrv, 0, 0) != RES_OK) res = FR_DISK_ERR;
 }

 return res;
}
# 1145 "../fatfs/ff.c"
static LBA_t clst2sect (
 FATFS* fs,
 DWORD clst
)
{
 clst -= 2;
 if (clst >= fs->n_fatent - 2) return 0;
 return fs->database + (LBA_t)fs->csize * clst;
}
# 1162 "../fatfs/ff.c"
static DWORD get_fat (
 FFOBJID* obj,
 DWORD clst
)
{
 UINT wc, bc;
 DWORD val;
 FATFS *fs = obj->fs;


 if (clst < 2 || clst >= fs->n_fatent) {
  val = 1;

 } else {
  val = 0xFFFFFFFF;

  switch (fs->fs_type) {
  case 1 :
   bc = (UINT)clst; bc += bc / 2;
   if (move_window(fs, fs->fatbase + (bc / ((UINT)512))) != FR_OK) break;
   wc = fs->win[bc++ % ((UINT)512)];
   if (move_window(fs, fs->fatbase + (bc / ((UINT)512))) != FR_OK) break;
   wc |= fs->win[bc % ((UINT)512)] << 8;
   val = (clst & 1) ? (wc >> 4) : (wc & 0xFFF);
   break;

  case 2 :
   if (move_window(fs, fs->fatbase + (clst / (((UINT)512) / 2))) != FR_OK) break;
   val = ld_word(fs->win + clst * 2 % ((UINT)512));
   break;

  case 3 :
   if (move_window(fs, fs->fatbase + (clst / (((UINT)512) / 4))) != FR_OK) break;
   val = ld_dword(fs->win + clst * 4 % ((UINT)512)) & 0x0FFFFFFF;
   break;
# 1224 "../fatfs/ff.c"
  default:
   val = 1;
  }
 }

 return val;
}
# 1240 "../fatfs/ff.c"
static FRESULT put_fat (
 FATFS* fs,
 DWORD clst,
 DWORD val
)
{
 UINT bc;
 BYTE *p;
 FRESULT res = FR_INT_ERR;


 if (clst >= 2 && clst < fs->n_fatent) {
  switch (fs->fs_type) {
  case 1:
   bc = (UINT)clst; bc += bc / 2;
   res = move_window(fs, fs->fatbase + (bc / ((UINT)512)));
   if (res != FR_OK) break;
   p = fs->win + bc++ % ((UINT)512);
   *p = (clst & 1) ? ((*p & 0x0F) | ((BYTE)val << 4)) : (BYTE)val;
   fs->wflag = 1;
   res = move_window(fs, fs->fatbase + (bc / ((UINT)512)));
   if (res != FR_OK) break;
   p = fs->win + bc % ((UINT)512);
   *p = (clst & 1) ? (BYTE)(val >> 4) : ((*p & 0xF0) | ((BYTE)(val >> 8) & 0x0F));
   fs->wflag = 1;
   break;

  case 2:
   res = move_window(fs, fs->fatbase + (clst / (((UINT)512) / 2)));
   if (res != FR_OK) break;
   st_word(fs->win + clst * 2 % ((UINT)512), (WORD)val);
   fs->wflag = 1;
   break;

  case 3:



   res = move_window(fs, fs->fatbase + (clst / (((UINT)512) / 4)));
   if (res != FR_OK) break;
   if (!0 || fs->fs_type != 4) {
    val = (val & 0x0FFFFFFF) | (ld_dword(fs->win + clst * 4 % ((UINT)512)) & 0xF0000000);
   }
   st_dword(fs->win + clst * 4 % ((UINT)512), val);
   fs->wflag = 1;
   break;
  }
 }
 return res;
}
# 1430 "../fatfs/ff.c"
static FRESULT remove_chain (
 FFOBJID* obj,
 DWORD clst,
 DWORD pclst
)
{
 FRESULT res = FR_OK;
 DWORD nxt;
 FATFS *fs = obj->fs;







 if (clst < 2 || clst >= fs->n_fatent) return FR_INT_ERR;


 if (pclst != 0 && (!0 || fs->fs_type != 4 || obj->stat != 2)) {
  res = put_fat(fs, pclst, 0xFFFFFFFF);
  if (res != FR_OK) return res;
 }


 do {
  nxt = get_fat(obj, clst);
  if (nxt == 0) break;
  if (nxt == 1) return FR_INT_ERR;
  if (nxt == 0xFFFFFFFF) return FR_DISK_ERR;
  if (!0 || fs->fs_type != 4) {
   res = put_fat(fs, clst, 0);
   if (res != FR_OK) return res;
  }
  if (fs->free_clst < fs->n_fatent - 2) {
   fs->free_clst++;
   fs->fsi_flag |= 1;
  }
# 1486 "../fatfs/ff.c"
  clst = nxt;
 } while (clst < fs->n_fatent);
# 1515 "../fatfs/ff.c"
 return FR_OK;
}
# 1525 "../fatfs/ff.c"
static DWORD create_chain (
 FFOBJID* obj,
 DWORD clst
)
{
 DWORD cs, ncl, scl;
 FRESULT res;
 FATFS *fs = obj->fs;


 if (clst == 0) {
  scl = fs->last_clst;
  if (scl == 0 || scl >= fs->n_fatent) scl = 1;
 }
 else {
  cs = get_fat(obj, clst);
  if (cs < 2) return 1;
  if (cs == 0xFFFFFFFF) return cs;
  if (cs < fs->n_fatent) return cs;
  scl = clst;
 }
 if (fs->free_clst == 0) return 0;
# 1574 "../fatfs/ff.c"
 {
  ncl = 0;
  if (scl == clst) {
   ncl = scl + 1;
   if (ncl >= fs->n_fatent) ncl = 2;
   cs = get_fat(obj, ncl);
   if (cs == 1 || cs == 0xFFFFFFFF) return cs;
   if (cs != 0) {
    cs = fs->last_clst;
    if (cs >= 2 && cs < fs->n_fatent) scl = cs;
    ncl = 0;
   }
  }
  if (ncl == 0) {
   ncl = scl;
   for (;;) {
    ncl++;
    if (ncl >= fs->n_fatent) {
     ncl = 2;
     if (ncl > scl) return 0;
    }
    cs = get_fat(obj, ncl);
    if (cs == 0) break;
    if (cs == 1 || cs == 0xFFFFFFFF) return cs;
    if (ncl == scl) return 0;
   }
  }
  res = put_fat(fs, ncl, 0xFFFFFFFF);
  if (res == FR_OK && clst != 0) {
   res = put_fat(fs, clst, ncl);
  }
 }

 if (res == FR_OK) {
  fs->last_clst = ncl;
  if (fs->free_clst <= fs->n_fatent - 2) fs->free_clst--;
  fs->fsi_flag |= 1;
 } else {
  ncl = (res == FR_DISK_ERR) ? 0xFFFFFFFF : 1;
 }

 return ncl;
}
# 1659 "../fatfs/ff.c"
static FRESULT dir_clear (
 FATFS *fs,
 DWORD clst
)
{
 LBA_t sect;
 UINT n, szb;
 BYTE *ibuf;


 if (sync_window(fs) != FR_OK) return FR_DISK_ERR;
 sect = clst2sect(fs, clst);
 fs->winsect = sect;
 memset(fs->win, 0, sizeof fs->win);
# 1683 "../fatfs/ff.c"
 {
  ibuf = fs->win; szb = 1;
  for (n = 0; n < fs->csize && disk_write(fs->pdrv, ibuf, sect + n, szb) == RES_OK; n += szb) ;
 }
 return (n == fs->csize) ? FR_OK : FR_DISK_ERR;
}
# 1698 "../fatfs/ff.c"
static FRESULT dir_sdi (
 DIR* dp,
 DWORD ofs
)
{
 DWORD csz, clst;
 FATFS *fs = dp->obj.fs;


 if (ofs >= (DWORD)((0 && fs->fs_type == 4) ? 0x10000000 : 0x200000) || ofs % 32) {
  return FR_INT_ERR;
 }
 dp->dptr = ofs;
 clst = dp->obj.sclust;
 if (clst == 0 && fs->fs_type >= 3) {
  clst = (DWORD)fs->dirbase;
  if (0) dp->obj.stat = 0;
 }

 if (clst == 0) {
  if (ofs / 32 >= fs->n_rootdir) return FR_INT_ERR;
  dp->sect = fs->dirbase;

 } else {
  csz = (DWORD)fs->csize * ((UINT)512);
  while (ofs >= csz) {
   clst = get_fat(&dp->obj, clst);
   if (clst == 0xFFFFFFFF) return FR_DISK_ERR;
   if (clst < 2 || clst >= fs->n_fatent) return FR_INT_ERR;
   ofs -= csz;
  }
  dp->sect = clst2sect(fs, clst);
 }
 dp->clust = clst;
 if (dp->sect == 0) return FR_INT_ERR;
 dp->sect += ofs / ((UINT)512);
 dp->dir = fs->win + (ofs % ((UINT)512));

 return FR_OK;
}
# 1746 "../fatfs/ff.c"
static FRESULT dir_next (
 DIR* dp,
 int stretch
)
{
 DWORD ofs, clst;
 FATFS *fs = dp->obj.fs;


 ofs = dp->dptr + 32;
 if (ofs >= (DWORD)((0 && fs->fs_type == 4) ? 0x10000000 : 0x200000)) dp->sect = 0;
 if (dp->sect == 0) return FR_NO_FILE;

 if (ofs % ((UINT)512) == 0) {
  dp->sect++;

  if (dp->clust == 0) {
   if (ofs / 32 >= fs->n_rootdir) {
    dp->sect = 0; return FR_NO_FILE;
   }
  }
  else {
   if ((ofs / ((UINT)512) & (fs->csize - 1)) == 0) {
    clst = get_fat(&dp->obj, dp->clust);
    if (clst <= 1) return FR_INT_ERR;
    if (clst == 0xFFFFFFFF) return FR_DISK_ERR;
    if (clst >= fs->n_fatent) {

     if (!stretch) {
      dp->sect = 0; return FR_NO_FILE;
     }
     clst = create_chain(&dp->obj, dp->clust);
     if (clst == 0) return FR_DENIED;
     if (clst == 1) return FR_INT_ERR;
     if (clst == 0xFFFFFFFF) return FR_DISK_ERR;
     if (dir_clear(fs, clst) != FR_OK) return FR_DISK_ERR;
     if (0) dp->obj.stat |= 4;




    }
    dp->clust = clst;
    dp->sect = clst2sect(fs, clst);
   }
  }
 }
 dp->dptr = ofs;
 dp->dir = fs->win + ofs % ((UINT)512);

 return FR_OK;
}
# 1807 "../fatfs/ff.c"
static FRESULT dir_alloc (
 DIR* dp,
 UINT n_ent
)
{
 FRESULT res;
 UINT n;
 FATFS *fs = dp->obj.fs;


 res = dir_sdi(dp, 0);
 if (res == FR_OK) {
  n = 0;
  do {
   res = move_window(fs, dp->sect);
   if (res != FR_OK) break;



   if (dp->dir[0] == 0xE5 || dp->dir[0] == 0) {

    if (++n == n_ent) break;
   } else {
    n = 0;
   }
   res = dir_next(dp, 1);
  } while (res == FR_OK);
 }

 if (res == FR_NO_FILE) res = FR_DENIED;
 return res;
}
# 1849 "../fatfs/ff.c"
static DWORD ld_clust (
 FATFS* fs,
 const BYTE* dir
)
{
 DWORD cl;

 cl = ld_word(dir + 26);
 if (fs->fs_type == 3) {
  cl |= (DWORD)ld_word(dir + 20) << 16;
 }

 return cl;
}



static void st_clust (
 FATFS* fs,
 BYTE* dir,
 DWORD cl
)
{
 st_word(dir + 26, (WORD)cl);
 if (fs->fs_type == 3) {
  st_word(dir + 20, (WORD)(cl >> 16));
 }
}
# 2307 "../fatfs/ff.c"
static FRESULT dir_read (
 DIR* dp,
 int vol
)
{
 FRESULT res = FR_NO_FILE;
 FATFS *fs = dp->obj.fs;
 BYTE attr, b;




 while (dp->sect) {
  res = move_window(fs, dp->sect);
  if (res != FR_OK) break;
  b = dp->dir[0];
  if (b == 0) {
   res = FR_NO_FILE; break;
  }
# 2342 "../fatfs/ff.c"
  {
   dp->obj.attr = attr = dp->dir[11] & 0x3F;
# 2364 "../fatfs/ff.c"
   if (b != 0xE5 && b != '.' && attr != 0x0F && (int)((attr & ~0x20) == 0x08) == vol) {
    break;
   }

  }
  res = dir_next(dp, 0);
  if (res != FR_OK) break;
 }

 if (res != FR_OK) dp->sect = 0;
 return res;
}
# 2385 "../fatfs/ff.c"
static FRESULT dir_find (
 DIR* dp
)
{
 FRESULT res;
 FATFS *fs = dp->obj.fs;
 BYTE c;




 res = dir_sdi(dp, 0);
 if (res != FR_OK) return res;
# 2422 "../fatfs/ff.c"
 do {
  res = move_window(fs, dp->sect);
  if (res != FR_OK) break;
  c = dp->dir[0];
  if (c == 0) { res = FR_NO_FILE; break; }
# 2449 "../fatfs/ff.c"
  dp->obj.attr = dp->dir[11] & 0x3F;
  if (!(dp->dir[11] & 0x08) && !memcmp(dp->dir, dp->fn, 11)) break;

  res = dir_next(dp, 0);
 } while (res == FR_OK);

 return res;
}
# 2466 "../fatfs/ff.c"
static FRESULT dir_register (
 DIR* dp
)
{
 FRESULT res;
 FATFS *fs = dp->obj.fs;
# 2543 "../fatfs/ff.c"
 res = dir_alloc(dp, 1);




 if (res == FR_OK) {
  res = move_window(fs, dp->sect);
  if (res == FR_OK) {
   memset(dp->dir, 0, 32);
   memcpy(dp->dir + 0, dp->fn, 11);



   fs->wflag = 1;
  }
 }

 return res;
}
# 2572 "../fatfs/ff.c"
static FRESULT dir_remove (
 DIR* dp
)
{
 FRESULT res;
 FATFS *fs = dp->obj.fs;
# 2599 "../fatfs/ff.c"
 res = move_window(fs, dp->sect);
 if (res == FR_OK) {
  dp->dir[0] = 0xE5;
  fs->wflag = 1;
 }


 return res;
}
# 2618 "../fatfs/ff.c"
static void get_fileinfo (
 DIR* dp,
 FILINFO* fno
)
{
 UINT si, di;






 TCHAR c;



 fno->fname[0] = 0;
 if (dp->sect == 0) return;
# 2734 "../fatfs/ff.c"
 si = di = 0;
 while (si < 11) {
  c = (TCHAR)dp->dir[si++];
  if (c == ' ') continue;
  if (c == 0x05) c = 0xE5;
  if (si == 9) fno->fname[di++] = '.';
  fno->fname[di++] = c;
 }
 fno->fname[di] = 0;


 fno->fattrib = dp->dir[11] & 0x3F;
 fno->fsize = ld_dword(dp->dir + 28);
 fno->ftime = ld_word(dp->dir + 22 + 0);
 fno->fdate = ld_word(dp->dir + 22 + 2);
}
# 2848 "../fatfs/ff.c"
static FRESULT create_name (
 DIR* dp,
 const TCHAR** path
)
{
# 2981 "../fatfs/ff.c"
 BYTE c, d;
 BYTE *sfn;
 UINT ni, si, i;
 const char *p;


 p = *path; sfn = dp->fn;
 memset(sfn, ' ', 11);
 si = i = 0; ni = 8;
# 3003 "../fatfs/ff.c"
 for (;;) {
  c = (BYTE)p[si++];
  if (c <= ' ') break;
  if (((c) == '/' || (c) == '\\')) {
   while (((p[si]) == '/' || (p[si]) == '\\')) si++;
   break;
  }
  if (c == '.' || i >= ni) {
   if (ni == 11 || c != '.') return FR_INVALID_NAME;
   i = 8; ni = 11;
   continue;
  }
# 3024 "../fatfs/ff.c"
  if (dbc_1st(c)) {
   d = (BYTE)p[si++];
   if (!dbc_2nd(d) || i >= ni - 1) return FR_INVALID_NAME;
   sfn[i++] = c;
   sfn[i++] = d;
  } else {
   if (strchr("*+,:;<=>[]|\"\?\x7F", (int)c)) return FR_INVALID_NAME;
   if (((c) >= 'a' && (c) <= 'z')) c -= 0x20;
   sfn[i++] = c;
  }
 }
 *path = &p[si];
 if (i == 0) return FR_INVALID_NAME;

 if (sfn[0] == 0xE5) sfn[0] = 0x05;
 sfn[11] = (c <= ' ' || p[si] <= ' ') ? 0x04 : 0;

 return FR_OK;

}
# 3052 "../fatfs/ff.c"
static FRESULT follow_path (
 DIR* dp,
 const TCHAR* path
)
{
 FRESULT res;
 BYTE ns;
 FATFS *fs = dp->obj.fs;







 {
  while (((*path) == '/' || (*path) == '\\')) path++;
  dp->obj.sclust = 0;
 }
# 3088 "../fatfs/ff.c"
 if ((UINT)*path < ' ') {
  dp->fn[11] = 0x80;
  res = dir_sdi(dp, 0);

 } else {
  for (;;) {
   res = create_name(dp, &path);
   if (res != FR_OK) break;
   res = dir_find(dp);
   ns = dp->fn[11];
   if (res != FR_OK) {
    if (res == FR_NO_FILE) {
     if (0 && (ns & 0x20)) {
      if (!(ns & 0x04)) continue;
      dp->fn[11] = 0x80;
      res = FR_OK;
     } else {
      if (!(ns & 0x04)) res = FR_NO_PATH;
     }
    }
    break;
   }
   if (ns & 0x04) break;

   if (!(dp->obj.attr & 0x10)) {
    res = FR_NO_PATH; break;
   }
# 3123 "../fatfs/ff.c"
   {
    dp->obj.sclust = ld_clust(fs, fs->win + dp->dptr % ((UINT)512));
   }
  }
 }

 return res;
}
# 3139 "../fatfs/ff.c"
static int get_ldnumber (
 const TCHAR** path
)
{
 const TCHAR *tp;
 const TCHAR *tt;
 TCHAR tc;
 int i;
 int vol = -1;





 tt = tp = *path;
 if (!tp) return vol;
 do {
  tc = *tt++;
 } while (!((UINT)(tc) < (0 ? ' ' : '!')) && tc != ':');

 if (tc == ':') {
  i = 1;
  if (((*tp) >= '0' && (*tp) <= '9') && tp + 2 == tt) {
   i = (int)*tp - '0';
  }
# 3177 "../fatfs/ff.c"
  if (i < 1) {
   vol = i;
   *path = tt;
  }
  return vol;
 }
# 3206 "../fatfs/ff.c"
 vol = 0;

 return vol;
}
# 3292 "../fatfs/ff.c"
static UINT check_fs (
 FATFS* fs,
 LBA_t sect
)
{
 WORD w, sign;
 BYTE b;


 fs->wflag = 0; fs->winsect = (LBA_t)0 - 1;
 if (move_window(fs, sect) != FR_OK) return 4;
 sign = ld_word(fs->win + 510);



 b = fs->win[0];
 if (b == 0xEB || b == 0xE9 || b == 0xE8) {
  if (sign == 0xAA55 && !memcmp(fs->win + 82, "FAT32   ", 8)) {
   return 0;
  }

  w = ld_word(fs->win + 11);
  b = fs->win[13];
  if ((w & (w - 1)) == 0 && w >= 512 && w <= 512
   && b != 0 && (b & (b - 1)) == 0
   && ld_word(fs->win + 14) != 0
   && (UINT)fs->win[16] - 1 <= 1
   && ld_word(fs->win + 17) != 0
   && (ld_word(fs->win + 19) >= 128 || ld_dword(fs->win + 32) >= 0x10000)
   && ld_word(fs->win + 22) != 0) {
    return 0;
  }
 }
 return sign == 0xAA55 ? 2 : 3;
}





static UINT find_volume (
 FATFS* fs,
 UINT part
)
{
 UINT fmt, i;
 DWORD mbr_pt[4];


 fmt = check_fs(fs, 0);
 if (fmt != 2 && (fmt >= 3 || part == 0)) return fmt;
# 3368 "../fatfs/ff.c"
 if (0 && part > 4) return 3;
 for (i = 0; i < 4; i++) {
  mbr_pt[i] = ld_dword(fs->win + 446 + i * 16 + 8);
 }
 i = part ? part - 1 : 0;
 do {
  fmt = mbr_pt[i] ? check_fs(fs, mbr_pt[i]) : 3;
 } while (part == 0 && fmt >= 2 && ++i < 4);
 return fmt;
}
# 3386 "../fatfs/ff.c"
static FRESULT mount_volume (
 const TCHAR** path,
 FATFS** rfs,
 BYTE mode
)
{
 int vol;
 FATFS *fs;
 DSTATUS stat;
 LBA_t bsect;
 DWORD tsect, sysect, fasize, nclst, szbfat;
 WORD nrsv;
 UINT fmt;



 *rfs = 0;
 vol = get_ldnumber(path);
 if (vol < 0) return FR_INVALID_DRIVE;


 fs = FatFs[vol];
 if (!fs) return FR_NOT_ENABLED;



 *rfs = fs;

 mode &= (BYTE)~0x01;
 if (fs->fs_type != 0) {
  stat = disk_status(fs->pdrv);
  if (!(stat & 0x01)) {
   if (!0 && mode && (stat & 0x04)) {
    return FR_WRITE_PROTECTED;
   }
   return FR_OK;
  }
 }




 fs->fs_type = 0;
 stat = disk_initialize(fs->pdrv);
 if (stat & 0x01) {
  return FR_NOT_READY;
 }
 if (!0 && mode && (stat & 0x04)) {
  return FR_WRITE_PROTECTED;
 }






 fmt = find_volume(fs, 0);

 if (fmt == 4) return FR_DISK_ERR;
 if (fmt >= 2) return FR_NO_FILESYSTEM;
 bsect = fs->winsect;
# 3513 "../fatfs/ff.c"
 {
  if (ld_word(fs->win + 11) != ((UINT)512)) return FR_NO_FILESYSTEM;

  fasize = ld_word(fs->win + 22);
  if (fasize == 0) fasize = ld_dword(fs->win + 36);
  fs->fsize = fasize;

  fs->n_fats = fs->win[16];
  if (fs->n_fats != 1 && fs->n_fats != 2) return FR_NO_FILESYSTEM;
  fasize *= fs->n_fats;

  fs->csize = fs->win[13];
  if (fs->csize == 0 || (fs->csize & (fs->csize - 1))) return FR_NO_FILESYSTEM;

  fs->n_rootdir = ld_word(fs->win + 17);
  if (fs->n_rootdir % (((UINT)512) / 32)) return FR_NO_FILESYSTEM;

  tsect = ld_word(fs->win + 19);
  if (tsect == 0) tsect = ld_dword(fs->win + 32);

  nrsv = ld_word(fs->win + 14);
  if (nrsv == 0) return FR_NO_FILESYSTEM;


  sysect = nrsv + fasize + fs->n_rootdir / (((UINT)512) / 32);
  if (tsect < sysect) return FR_NO_FILESYSTEM;
  nclst = (tsect - sysect) / fs->csize;
  if (nclst == 0) return FR_NO_FILESYSTEM;
  fmt = 0;
  if (nclst <= 0x0FFFFFF5) fmt = 3;
  if (nclst <= 0xFFF5) fmt = 2;
  if (nclst <= 0xFF5) fmt = 1;
  if (fmt == 0) return FR_NO_FILESYSTEM;


  fs->n_fatent = nclst + 2;
  fs->volbase = bsect;
  fs->fatbase = bsect + nrsv;
  fs->database = bsect + sysect;
  if (fmt == 3) {
   if (ld_word(fs->win + 42) != 0) return FR_NO_FILESYSTEM;
   if (fs->n_rootdir != 0) return FR_NO_FILESYSTEM;
   fs->dirbase = ld_dword(fs->win + 44);
   szbfat = fs->n_fatent * 4;
  } else {
   if (fs->n_rootdir == 0) return FR_NO_FILESYSTEM;
   fs->dirbase = fs->fatbase + fasize;
   szbfat = (fmt == 2) ?
    fs->n_fatent * 2 : fs->n_fatent * 3 / 2 + (fs->n_fatent & 1);
  }
  if (fs->fsize < (szbfat + (((UINT)512) - 1)) / ((UINT)512)) return FR_NO_FILESYSTEM;



  fs->last_clst = fs->free_clst = 0xFFFFFFFF;
  fs->fsi_flag = 0x80;

  if (fmt == 3
   && ld_word(fs->win + 48) == 1
   && move_window(fs, bsect + 1) == FR_OK)
  {
   fs->fsi_flag = 0;
   if (ld_word(fs->win + 510) == 0xAA55
    && ld_dword(fs->win + 0) == 0x41615252
    && ld_dword(fs->win + 484) == 0x61417272)
   {

    fs->free_clst = ld_dword(fs->win + 488);


    fs->last_clst = ld_dword(fs->win + 492);

   }
  }


 }

 fs->fs_type = (BYTE)fmt;
 fs->id = ++Fsid;
# 3605 "../fatfs/ff.c"
 return FR_OK;
}
# 3615 "../fatfs/ff.c"
static FRESULT validate (
 FFOBJID* obj,
 FATFS** rfs
)
{
 FRESULT res = FR_INVALID_OBJECT;


 if (obj && obj->fs && obj->fs->fs_type && obj->id == obj->fs->id) {
# 3635 "../fatfs/ff.c"
  if (!(disk_status(obj->fs->pdrv) & 0x01)) {
   res = FR_OK;
  }

 }
 *rfs = (res == FR_OK) ? obj->fs : 0;
 return res;
}
# 3659 "../fatfs/ff.c"
FRESULT f_mount (
 FATFS* fs,
 const TCHAR* path,
 BYTE opt
)
{
 FATFS *cfs;
 int vol;
 FRESULT res;
 const TCHAR *rp = path;



 vol = get_ldnumber(&rp);
 if (vol < 0) return FR_INVALID_DRIVE;
 cfs = FatFs[vol];

 if (cfs) {
  FatFs[vol] = 0;






  cfs->fs_type = 0;
 }

 if (fs) {
  fs->pdrv = (BYTE)(vol);
# 3702 "../fatfs/ff.c"
  fs->fs_type = 0;
  FatFs[vol] = fs;
 }

 if (opt == 0) return FR_OK;

 res = mount_volume(&path, &fs, 0);
 return res;
}
# 3719 "../fatfs/ff.c"
FRESULT f_open (
 FIL* fp,
 const TCHAR* path,
 BYTE mode
)
{
 FRESULT res;
 DIR dj;
 FATFS *fs;

 DWORD cl, bcs, clst, tm;
 LBA_t sc;
 FSIZE_t ofs;




 if (!fp) return FR_INVALID_OBJECT;


 mode &= 0 ? 0x01 : 0x01 | 0x02 | 0x08 | 0x04 | 0x10 | 0x30;
 res = mount_volume(&path, &fs, mode);
 if (res == FR_OK) {
  dj.obj.fs = fs;
                 ;
  res = follow_path(&dj, path);

  if (res == FR_OK) {
   if (dj.fn[11] & 0x80) {
    res = FR_INVALID_NAME;
   }





  }

  if (mode & (0x08 | 0x10 | 0x04)) {
   if (res != FR_OK) {
    if (res == FR_NO_FILE) {



     res = dir_register(&dj);

    }
    mode |= 0x08;
   }
   else {
    if (dj.obj.attr & (0x01 | 0x10)) {
     res = FR_DENIED;
    } else {
     if (mode & 0x04) res = FR_EXIST;
    }
   }
   if (res == FR_OK && (mode & 0x08)) {
# 3794 "../fatfs/ff.c"
    {

     tm = get_fattime();
     st_dword(dj.dir + 14, tm);
     st_dword(dj.dir + 22, tm);
     cl = ld_clust(fs, dj.dir);
     dj.dir[11] = 0x20;
     st_clust(fs, dj.dir, 0);
     st_dword(dj.dir + 28, 0);
     fs->wflag = 1;
     if (cl != 0) {
      sc = fs->winsect;
      res = remove_chain(&dj.obj, cl, 0);
      if (res == FR_OK) {
       res = move_window(fs, sc);
       fs->last_clst = cl - 1;
      }
     }
    }
   }
  }
  else {
   if (res == FR_OK) {
    if (dj.obj.attr & 0x10) {
     res = FR_NO_FILE;
    } else {
     if ((mode & 0x02) && (dj.obj.attr & 0x01)) {
      res = FR_DENIED;
     }
    }
   }
  }
  if (res == FR_OK) {
   if (mode & 0x08) mode |= 0x40;
   fp->dir_sect = fs->winsect;
   fp->dir_ptr = dj.dir;




  }
# 3847 "../fatfs/ff.c"
  if (res == FR_OK) {
# 3856 "../fatfs/ff.c"
   {
    fp->obj.sclust = ld_clust(fs, dj.dir);
    fp->obj.objsize = ld_dword(dj.dir + 28);
   }



   fp->obj.fs = fs;
   fp->obj.id = fs->id;
   fp->flag = mode;
   fp->err = 0;
   fp->sect = 0;
   fp->fptr = 0;


   memset(fp->buf, 0, sizeof fp->buf);

   if ((mode & 0x20) && fp->obj.objsize > 0) {
    fp->fptr = fp->obj.objsize;
    bcs = (DWORD)fs->csize * ((UINT)512);
    clst = fp->obj.sclust;
    for (ofs = fp->obj.objsize; res == FR_OK && ofs > bcs; ofs -= bcs) {
     clst = get_fat(&fp->obj, clst);
     if (clst <= 1) res = FR_INT_ERR;
     if (clst == 0xFFFFFFFF) res = FR_DISK_ERR;
    }
    fp->clust = clst;
    if (res == FR_OK && ofs % ((UINT)512)) {
     sc = clst2sect(fs, clst);
     if (sc == 0) {
      res = FR_INT_ERR;
     } else {
      fp->sect = sc + (DWORD)(ofs / ((UINT)512));

      if (disk_read(fs->pdrv, fp->buf, fp->sect, 1) != RES_OK) res = FR_DISK_ERR;

     }
    }



   }

  }

               ;
 }

 if (res != FR_OK) fp->obj.fs = 0;

 return res;
}
# 3916 "../fatfs/ff.c"
FRESULT f_read (
 FIL* fp,
 void* buff,
 UINT btr,
 UINT* br
)
{
 FRESULT res;
 FATFS *fs;
 DWORD clst;
 LBA_t sect;
 FSIZE_t remain;
 UINT rcnt, cc, csect;
 BYTE *rbuff = (BYTE*)buff;


 *br = 0;
 res = validate(&fp->obj, &fs);
 if (res != FR_OK || (res = (FRESULT)fp->err) != FR_OK) return res;
 if (!(fp->flag & 0x01)) return FR_DENIED;
 remain = fp->obj.objsize - fp->fptr;
 if (btr > remain) btr = (UINT)remain;

 for ( ; btr > 0; btr -= rcnt, *br += rcnt, rbuff += rcnt, fp->fptr += rcnt) {
  if (fp->fptr % ((UINT)512) == 0) {
   csect = (UINT)(fp->fptr / ((UINT)512) & (fs->csize - 1));
   if (csect == 0) {
    if (fp->fptr == 0) {
     clst = fp->obj.sclust;
    } else {





     {
      clst = get_fat(&fp->obj, fp->clust);
     }
    }
    if (clst < 2) { fp->err = (BYTE)(FR_INT_ERR); return FR_INT_ERR; };
    if (clst == 0xFFFFFFFF) { fp->err = (BYTE)(FR_DISK_ERR); return FR_DISK_ERR; };
    fp->clust = clst;
   }
   sect = clst2sect(fs, fp->clust);
   if (sect == 0) { fp->err = (BYTE)(FR_INT_ERR); return FR_INT_ERR; };
   sect += csect;
   cc = btr / ((UINT)512);
   if (cc > 0) {
    if (csect + cc > fs->csize) {
     cc = fs->csize - csect;
    }
    if (disk_read(fs->pdrv, rbuff, sect, cc) != RES_OK) { fp->err = (BYTE)(FR_DISK_ERR); return FR_DISK_ERR; };






    if ((fp->flag & 0x80) && fp->sect - sect < cc) {
     memcpy(rbuff + ((fp->sect - sect) * ((UINT)512)), fp->buf, ((UINT)512));
    }


    rcnt = ((UINT)512) * cc;
    continue;
   }

   if (fp->sect != sect) {

    if (fp->flag & 0x80) {
     if (disk_write(fs->pdrv, fp->buf, fp->sect, 1) != RES_OK) { fp->err = (BYTE)(FR_DISK_ERR); return FR_DISK_ERR; };
     fp->flag &= (BYTE)~0x80;
    }

    if (disk_read(fs->pdrv, fp->buf, sect, 1) != RES_OK) { fp->err = (BYTE)(FR_DISK_ERR); return FR_DISK_ERR; };
   }

   fp->sect = sect;
  }
  rcnt = ((UINT)512) - (UINT)fp->fptr % ((UINT)512);
  if (rcnt > btr) rcnt = btr;




  memcpy(rbuff, fp->buf + fp->fptr % ((UINT)512), rcnt);

 }

 return FR_OK;
}
# 4016 "../fatfs/ff.c"
FRESULT f_write (
 FIL* fp,
 const void* buff,
 UINT btw,
 UINT* bw
)
{
 FRESULT res;
 FATFS *fs;
 DWORD clst;
 LBA_t sect;
 UINT wcnt, cc, csect;
 const BYTE *wbuff = (const BYTE*)buff;


 *bw = 0;
 res = validate(&fp->obj, &fs);
 if (res != FR_OK || (res = (FRESULT)fp->err) != FR_OK) return res;
 if (!(fp->flag & 0x02)) return FR_DENIED;


 if ((!0 || fs->fs_type != 4) && (DWORD)(fp->fptr + btw) < (DWORD)fp->fptr) {
  btw = (UINT)(0xFFFFFFFF - (DWORD)fp->fptr);
 }

 for ( ; btw > 0; btw -= wcnt, *bw += wcnt, wbuff += wcnt, fp->fptr += wcnt, fp->obj.objsize = (fp->fptr > fp->obj.objsize) ? fp->fptr : fp->obj.objsize) {
  if (fp->fptr % ((UINT)512) == 0) {
   csect = (UINT)(fp->fptr / ((UINT)512)) & (fs->csize - 1);
   if (csect == 0) {
    if (fp->fptr == 0) {
     clst = fp->obj.sclust;
     if (clst == 0) {
      clst = create_chain(&fp->obj, 0);
     }
    } else {





     {
      clst = create_chain(&fp->obj, fp->clust);
     }
    }
    if (clst == 0) break;
    if (clst == 1) { fp->err = (BYTE)(FR_INT_ERR); return FR_INT_ERR; };
    if (clst == 0xFFFFFFFF) { fp->err = (BYTE)(FR_DISK_ERR); return FR_DISK_ERR; };
    fp->clust = clst;
    if (fp->obj.sclust == 0) fp->obj.sclust = clst;
   }



   if (fp->flag & 0x80) {
    if (disk_write(fs->pdrv, fp->buf, fp->sect, 1) != RES_OK) { fp->err = (BYTE)(FR_DISK_ERR); return FR_DISK_ERR; };
    fp->flag &= (BYTE)~0x80;
   }

   sect = clst2sect(fs, fp->clust);
   if (sect == 0) { fp->err = (BYTE)(FR_INT_ERR); return FR_INT_ERR; };
   sect += csect;
   cc = btw / ((UINT)512);
   if (cc > 0) {
    if (csect + cc > fs->csize) {
     cc = fs->csize - csect;
    }
    if (disk_write(fs->pdrv, wbuff, sect, cc) != RES_OK) { fp->err = (BYTE)(FR_DISK_ERR); return FR_DISK_ERR; };







    if (fp->sect - sect < cc) {
     memcpy(fp->buf, wbuff + ((fp->sect - sect) * ((UINT)512)), ((UINT)512));
     fp->flag &= (BYTE)~0x80;
    }


    wcnt = ((UINT)512) * cc;
    continue;
   }






   if (fp->sect != sect &&
    fp->fptr < fp->obj.objsize &&
    disk_read(fs->pdrv, fp->buf, sect, 1) != RES_OK) {
     { fp->err = (BYTE)(FR_DISK_ERR); return FR_DISK_ERR; };
   }

   fp->sect = sect;
  }
  wcnt = ((UINT)512) - (UINT)fp->fptr % ((UINT)512);
  if (wcnt > btw) wcnt = btw;





  memcpy(fp->buf + fp->fptr % ((UINT)512), wbuff, wcnt);
  fp->flag |= 0x80;

 }

 fp->flag |= 0x40;

 return FR_OK;
}
# 4137 "../fatfs/ff.c"
FRESULT f_sync (
 FIL* fp
)
{
 FRESULT res;
 FATFS *fs;
 DWORD tm;
 BYTE *dir;


 res = validate(&fp->obj, &fs);
 if (res == FR_OK) {
  if (fp->flag & 0x40) {

   if (fp->flag & 0x80) {
    if (disk_write(fs->pdrv, fp->buf, fp->sect, 1) != RES_OK) return FR_DISK_ERR;
    fp->flag &= (BYTE)~0x80;
   }


   tm = get_fattime();
# 4189 "../fatfs/ff.c"
   {
    res = move_window(fs, fp->dir_sect);
    if (res == FR_OK) {
     dir = fp->dir_ptr;
     dir[11] |= 0x20;
     st_clust(fp->obj.fs, dir, fp->obj.sclust);
     st_dword(dir + 28, (DWORD)fp->obj.objsize);
     st_dword(dir + 22, tm);
     st_word(dir + 18, 0);
     fs->wflag = 1;
     res = sync_fs(fs);
     fp->flag &= (BYTE)~0x40;
    }
   }
  }
 }

 return res;
}
# 4218 "../fatfs/ff.c"
FRESULT f_close (
 FIL* fp
)
{
 FRESULT res;
 FATFS *fs;


 res = f_sync(fp);
 if (res == FR_OK)

 {
  res = validate(&fp->obj, &fs);
  if (res == FR_OK) {




   fp->obj.fs = 0;




  }
 }
 return res;
}
# 4435 "../fatfs/ff.c"
FRESULT f_lseek (
 FIL* fp,
 FSIZE_t ofs
)
{
 FRESULT res;
 FATFS *fs;
 DWORD clst, bcs;
 LBA_t nsect;
 FSIZE_t ifptr;






 res = validate(&fp->obj, &fs);
 if (res == FR_OK) res = (FRESULT)fp->err;





 if (res != FR_OK) return res;
# 4513 "../fatfs/ff.c"
 {



  if (ofs > fp->obj.objsize && (0 || !(fp->flag & 0x02))) {
   ofs = fp->obj.objsize;
  }
  ifptr = fp->fptr;
  fp->fptr = nsect = 0;
  if (ofs > 0) {
   bcs = (DWORD)fs->csize * ((UINT)512);
   if (ifptr > 0 &&
    (ofs - 1) / bcs >= (ifptr - 1) / bcs) {
    fp->fptr = (ifptr - 1) & ~(FSIZE_t)(bcs - 1);
    ofs -= fp->fptr;
    clst = fp->clust;
   } else {
    clst = fp->obj.sclust;

    if (clst == 0) {
     clst = create_chain(&fp->obj, 0);
     if (clst == 1) { fp->err = (BYTE)(FR_INT_ERR); return FR_INT_ERR; };
     if (clst == 0xFFFFFFFF) { fp->err = (BYTE)(FR_DISK_ERR); return FR_DISK_ERR; };
     fp->obj.sclust = clst;
    }

    fp->clust = clst;
   }
   if (clst != 0) {
    while (ofs > bcs) {
     ofs -= bcs; fp->fptr += bcs;

     if (fp->flag & 0x02) {
      if (0 && fp->fptr > fp->obj.objsize) {
       fp->obj.objsize = fp->fptr;
       fp->flag |= 0x40;
      }
      clst = create_chain(&fp->obj, clst);
      if (clst == 0) {
       ofs = 0; break;
      }
     } else

     {
      clst = get_fat(&fp->obj, clst);
     }
     if (clst == 0xFFFFFFFF) { fp->err = (BYTE)(FR_DISK_ERR); return FR_DISK_ERR; };
     if (clst <= 1 || clst >= fs->n_fatent) { fp->err = (BYTE)(FR_INT_ERR); return FR_INT_ERR; };
     fp->clust = clst;
    }
    fp->fptr += ofs;
    if (ofs % ((UINT)512)) {
     nsect = clst2sect(fs, clst);
     if (nsect == 0) { fp->err = (BYTE)(FR_INT_ERR); return FR_INT_ERR; };
     nsect += (DWORD)(ofs / ((UINT)512));
    }
   }
  }
  if (!0 && fp->fptr > fp->obj.objsize) {
   fp->obj.objsize = fp->fptr;
   fp->flag |= 0x40;
  }
  if (fp->fptr % ((UINT)512) && nsect != fp->sect) {


   if (fp->flag & 0x80) {
    if (disk_write(fs->pdrv, fp->buf, fp->sect, 1) != RES_OK) { fp->err = (BYTE)(FR_DISK_ERR); return FR_DISK_ERR; };
    fp->flag &= (BYTE)~0x80;
   }

   if (disk_read(fs->pdrv, fp->buf, nsect, 1) != RES_OK) { fp->err = (BYTE)(FR_DISK_ERR); return FR_DISK_ERR; };

   fp->sect = nsect;
  }
 }

 return res;
}
# 4599 "../fatfs/ff.c"
FRESULT f_opendir (
 DIR* dp,
 const TCHAR* path
)
{
 FRESULT res;
 FATFS *fs;



 if (!dp) return FR_INVALID_OBJECT;


 res = mount_volume(&path, &fs, 0);
 if (res == FR_OK) {
  dp->obj.fs = fs;
                 ;
  res = follow_path(dp, path);
  if (res == FR_OK) {
   if (!(dp->fn[11] & 0x80)) {
    if (dp->obj.attr & 0x10) {
# 4628 "../fatfs/ff.c"
     {
      dp->obj.sclust = ld_clust(fs, dp->dir);
     }
    } else {
     res = FR_NO_PATH;
    }
   }
   if (res == FR_OK) {
    dp->obj.id = fs->id;
    res = dir_sdi(dp, 0);
# 4648 "../fatfs/ff.c"
   }
  }
               ;
  if (res == FR_NO_FILE) res = FR_NO_PATH;
 }
 if (res != FR_OK) dp->obj.fs = 0;

 return res;
}
# 4665 "../fatfs/ff.c"
FRESULT f_closedir (
 DIR *dp
)
{
 FRESULT res;
 FATFS *fs;


 res = validate(&dp->obj, &fs);
 if (res == FR_OK) {




  dp->obj.fs = 0;




 }
 return res;
}
# 4695 "../fatfs/ff.c"
FRESULT f_readdir (
 DIR* dp,
 FILINFO* fno
)
{
 FRESULT res;
 FATFS *fs;



 res = validate(&dp->obj, &fs);
 if (res == FR_OK) {
  if (!fno) {
   res = dir_sdi(dp, 0);
  } else {
                  ;
   res = dir_read(dp, 0);
   if (res == FR_NO_FILE) res = FR_OK;
   if (res == FR_OK) {
    get_fileinfo(dp, fno);
    res = dir_next(dp, 0);
    if (res == FR_NO_FILE) res = FR_OK;
   }
                ;
  }
 }
 return res;
}
# 4783 "../fatfs/ff.c"
FRESULT f_stat (
 const TCHAR* path,
 FILINFO* fno
)
{
 FRESULT res;
 DIR dj;




 res = mount_volume(&path, &dj.obj.fs, 0);
 if (res == FR_OK) {
                        ;
  res = follow_path(&dj, path);
  if (res == FR_OK) {
   if (dj.fn[11] & 0x80) {
    res = FR_INVALID_NAME;
   } else {
    if (fno) get_fileinfo(&dj, fno);
   }
  }
               ;
 }

 return res;
}
# 4818 "../fatfs/ff.c"
FRESULT f_getfree (
 const TCHAR* path,
 DWORD* nclst,
 FATFS** fatfs
)
{
 FRESULT res;
 FATFS *fs;
 DWORD nfree, clst, stat;
 LBA_t sect;
 UINT i;
 FFOBJID obj;



 res = mount_volume(&path, &fs, 0);
 if (res == FR_OK) {
  *fatfs = fs;

  if (fs->free_clst <= fs->n_fatent - 2) {
   *nclst = fs->free_clst;
  } else {

   nfree = 0;
   if (fs->fs_type == 1) {
    clst = 2; obj.fs = fs;
    do {
     stat = get_fat(&obj, clst);
     if (stat == 0xFFFFFFFF) {
      res = FR_DISK_ERR; break;
     }
     if (stat == 1) {
      res = FR_INT_ERR; break;
     }
     if (stat == 0) nfree++;
    } while (++clst < fs->n_fatent);
   } else {
# 4876 "../fatfs/ff.c"
    {
     clst = fs->n_fatent;
     sect = fs->fatbase;
     i = 0;
     do {
      if (i == 0) {
       res = move_window(fs, sect++);
       if (res != FR_OK) break;
      }
      if (fs->fs_type == 2) {
       if (ld_word(fs->win + i) == 0) nfree++;
       i += 2;
      } else {
       if ((ld_dword(fs->win + i) & 0x0FFFFFFF) == 0) nfree++;
       i += 4;
      }
      i %= ((UINT)512);
     } while (--clst);
    }
   }
   if (res == FR_OK) {
    *nclst = nfree;
    fs->free_clst = nfree;
    fs->fsi_flag |= 1;
   }
  }
 }

 return res;
}
# 4914 "../fatfs/ff.c"
FRESULT f_truncate (
 FIL* fp
)
{
 FRESULT res;
 FATFS *fs;
 DWORD ncl;


 res = validate(&fp->obj, &fs);
 if (res != FR_OK || (res = (FRESULT)fp->err) != FR_OK) return res;
 if (!(fp->flag & 0x02)) return FR_DENIED;

 if (fp->fptr < fp->obj.objsize) {
  if (fp->fptr == 0) {
   res = remove_chain(&fp->obj, fp->obj.sclust, 0);
   fp->obj.sclust = 0;
  } else {
   ncl = get_fat(&fp->obj, fp->clust);
   res = FR_OK;
   if (ncl == 0xFFFFFFFF) res = FR_DISK_ERR;
   if (ncl == 1) res = FR_INT_ERR;
   if (res == FR_OK && ncl < fs->n_fatent) {
    res = remove_chain(&fp->obj, ncl, fp->clust);
   }
  }
  fp->obj.objsize = fp->fptr;
  fp->flag |= 0x40;

  if (res == FR_OK && (fp->flag & 0x80)) {
   if (disk_write(fs->pdrv, fp->buf, fp->sect, 1) != RES_OK) {
    res = FR_DISK_ERR;
   } else {
    fp->flag &= (BYTE)~0x80;
   }
  }

  if (res != FR_OK) { fp->err = (BYTE)(res); return res; };
 }

 return res;
}
# 4964 "../fatfs/ff.c"
FRESULT f_unlink (
 const TCHAR* path
)
{
 FRESULT res;
 FATFS *fs;
 DIR dj, sdj;
 DWORD dclst = 0;







 res = mount_volume(&path, &fs, 0x02);
 if (res == FR_OK) {
  dj.obj.fs = fs;
                 ;
  res = follow_path(&dj, path);
  if (0 && res == FR_OK && (dj.fn[11] & 0x20)) {
   res = FR_INVALID_NAME;
  }



  if (res == FR_OK) {
   if (dj.fn[11] & 0x80) {
    res = FR_INVALID_NAME;
   } else {
    if (dj.obj.attr & 0x01) {
     res = FR_DENIED;
    }
   }
   if (res == FR_OK) {







    {
     dclst = ld_clust(fs, dj.dir);
    }
    if (dj.obj.attr & 0x10) {





     {
      sdj.obj.fs = fs;
      sdj.obj.sclust = dclst;






      res = dir_sdi(&sdj, 0);
      if (res == FR_OK) {
       res = dir_read(&sdj, 0);
       if (res == FR_OK) res = FR_DENIED;
       if (res == FR_NO_FILE) res = FR_OK;
      }
     }
    }
   }
   if (res == FR_OK) {
    res = dir_remove(&dj);
    if (res == FR_OK && dclst != 0) {



     res = remove_chain(&dj.obj, dclst, 0);

    }
    if (res == FR_OK) res = sync_fs(fs);
   }
  }
               ;
 }

 return res;
}
# 5058 "../fatfs/ff.c"
FRESULT f_mkdir (
 const TCHAR* path
)
{
 FRESULT res;
 FATFS *fs;
 DIR dj;
 FFOBJID sobj;
 DWORD dcl, pcl, tm;



 res = mount_volume(&path, &fs, 0x02);
 if (res == FR_OK) {
  dj.obj.fs = fs;
                 ;
  res = follow_path(&dj, path);
  if (res == FR_OK) res = FR_EXIST;
  if (0 && res == FR_NO_FILE && (dj.fn[11] & 0x20)) {
   res = FR_INVALID_NAME;
  }
  if (res == FR_NO_FILE) {
   sobj.fs = fs;
   dcl = create_chain(&sobj, 0);
   res = FR_OK;
   if (dcl == 0) res = FR_DENIED;
   if (dcl == 1) res = FR_INT_ERR;
   if (dcl == 0xFFFFFFFF) res = FR_DISK_ERR;
   tm = get_fattime();
   if (res == FR_OK) {
    res = dir_clear(fs, dcl);
    if (res == FR_OK) {
     if (!0 || fs->fs_type != 4) {
      memset(fs->win + 0, ' ', 11);
      fs->win[0] = '.';
      fs->win[11] = 0x10;
      st_dword(fs->win + 22, tm);
      st_clust(fs, fs->win, dcl);
      memcpy(fs->win + 32, fs->win, 32);
      fs->win[32 + 1] = '.'; pcl = dj.obj.sclust;
      st_clust(fs, fs->win + 32, pcl);
      fs->wflag = 1;
     }
     res = dir_register(&dj);
    }
   }
   if (res == FR_OK) {
# 5116 "../fatfs/ff.c"
    {
     st_dword(dj.dir + 22, tm);
     st_clust(fs, dj.dir, dcl);
     dj.dir[11] = 0x10;
     fs->wflag = 1;
    }
    if (res == FR_OK) {
     res = sync_fs(fs);
    }
   } else {
    remove_chain(&sobj, dcl, 0);
   }
  }
               ;
 }

 return res;
}
# 5142 "../fatfs/ff.c"
FRESULT f_rename (
 const TCHAR* path_old,
 const TCHAR* path_new
)
{
 FRESULT res;
 FATFS *fs;
 DIR djo, djn;
 BYTE buf[0 ? 32 * 2 : 32], *dir;
 LBA_t sect;



 get_ldnumber(&path_new);
 res = mount_volume(&path_old, &fs, 0x02);
 if (res == FR_OK) {
  djo.obj.fs = fs;
                 ;
  res = follow_path(&djo, path_old);
  if (res == FR_OK && (djo.fn[11] & (0x20 | 0x80))) res = FR_INVALID_NAME;





  if (res == FR_OK) {
# 5194 "../fatfs/ff.c"
   {
    memcpy(buf, djo.dir, 32);
    memcpy(&djn, &djo, sizeof (DIR));
    res = follow_path(&djn, path_new);
    if (res == FR_OK) {
     res = (djn.obj.sclust == djo.obj.sclust && djn.dptr == djo.dptr) ? FR_NO_FILE : FR_EXIST;
    }
    if (res == FR_NO_FILE) {
     res = dir_register(&djn);
     if (res == FR_OK) {
      dir = djn.dir;
      memcpy(dir + 13, buf + 13, 32 - 13);
      dir[11] = buf[11];
      if (!(dir[11] & 0x10)) dir[11] |= 0x20;
      fs->wflag = 1;
      if ((dir[11] & 0x10) && djo.obj.sclust != djn.obj.sclust) {
       sect = clst2sect(fs, ld_clust(fs, dir));
       if (sect == 0) {
        res = FR_INT_ERR;
       } else {

        res = move_window(fs, sect);
        dir = fs->win + 32 * 1;
        if (res == FR_OK && dir[1] == '.') {
         st_clust(fs, dir, djn.obj.sclust);
         fs->wflag = 1;
        }
       }
      }
     }
    }
   }
   if (res == FR_OK) {
    res = dir_remove(&djo);
    if (res == FR_OK) {
     res = sync_fs(fs);
    }
   }

  }
               ;
 }

 return res;
}
# 6428 "../fatfs/ff.c"
TCHAR* f_gets (
 TCHAR* buff,
 int len,
 FIL* fp
)
{
 int nc = 0;
 TCHAR *p = buff;
 BYTE s[4];
 UINT rc;
 DWORD dc;
# 6536 "../fatfs/ff.c"
 len -= 1;
 while (nc < len) {
  f_read(fp, s, 1, &rc);
  if (rc != 1) break;
  dc = s[0];
  if (1 == 2 && dc == '\r') continue;
  *p++ = (TCHAR)dc; nc++;
  if (dc == '\n') break;
 }


 *p = 0;
 return nc ? buff : 0;
}





# 1 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/stdarg.h" 1 3







# 1 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/bits/alltypes.h" 1 3
# 9 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/stdarg.h" 2 3

#pragma intrinsic(__va_start)
#pragma intrinsic(__va_arg)

extern void * __va_start(void);
extern void * __va_arg(void *, ...);
# 6556 "../fatfs/ff.c" 2
# 6565 "../fatfs/ff.c"
typedef struct {
 FIL *fp;
 int idx, nchr;






 BYTE buf[64];
} putbuff;




static void putc_bfd (putbuff* pb, TCHAR c)
{
 UINT n;
 int i, nc;
# 6592 "../fatfs/ff.c"
 if (1 == 2 && c == '\n') {
  putc_bfd(pb, '\r');
 }

 i = pb->idx;
 if (i < 0) return;
 nc = pb->nchr;
# 6697 "../fatfs/ff.c"
 pb->buf[i++] = (BYTE)c;


 if (i >= (int)(sizeof pb->buf) - 4) {
  f_write(pb->fp, pb->buf, (UINT)i, &n);
  i = (n == (UINT)i) ? 0 : -1;
 }
 pb->idx = i;
 pb->nchr = nc + 1;
}




static int putc_flush (putbuff* pb)
{
 UINT nw;

 if ( pb->idx >= 0
  && f_write(pb->fp, pb->buf, (UINT)pb->idx, &nw) == FR_OK
  && (UINT)pb->idx == nw) return pb->nchr;
 return -1;
}




static void putc_init (putbuff* pb, FIL* fp)
{
 memset(pb, 0, sizeof (putbuff));
 pb->fp = fp;
}



int f_putc (
 TCHAR c,
 FIL* fp
)
{
 putbuff pb;


 putc_init(&pb, fp);
 putc_bfd(&pb, c);
 return putc_flush(&pb);
}
# 6752 "../fatfs/ff.c"
int f_puts (
 const TCHAR* str,
 FIL* fp
)
{
 putbuff pb;


 putc_init(&pb, fp);
 while (*str) putc_bfd(&pb, *str++);
 return putc_flush(&pb);
}
# 6772 "../fatfs/ff.c"
# 1 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/math.h" 1 3
# 15 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/math.h" 3
# 1 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/bits/alltypes.h" 1 3
# 39 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/bits/alltypes.h" 3
typedef float float_t;




typedef double double_t;
# 16 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/math.h" 2 3
# 42 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/math.h" 3
int __fpclassifyf(float);







int __signbitf(float);
# 59 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/math.h" 3
double acos(double);
float acosf(float);
long double acosl(long double);



double acosh(double);
float acoshf(float);
long double acoshl(long double);



double asin(double);
float asinf(float);
long double asinl(long double);



double asinh(double);
float asinhf(float);
long double asinhl(long double);



double atan(double);
float atanf(float);
long double atanl(long double);



double atan2(double, double);
float atan2f(float, float);
long double atan2l(long double, long double);



double atanh(double);
float atanhf(float);
long double atanhl(long double);



double cbrt(double);
float cbrtf(float);
long double cbrtl(long double);



double ceil(double);
float ceilf(float);
long double ceill(long double);



double copysign(double, double);
float copysignf(float, float);
long double copysignl(long double, long double);



double cos(double);
float cosf(float);
long double cosl(long double);



double cosh(double);
float coshf(float);
long double coshl(long double);



double erf(double);
float erff(float);
long double erfl(long double);



double erfc(double);
float erfcf(float);
long double erfcl(long double);



double exp(double);
float expf(float);
long double expl(long double);



double exp2(double);
float exp2f(float);
long double exp2l(long double);



double expm1(double);
float expm1f(float);
long double expm1l(long double);



double fabs(double);
float fabsf(float);
long double fabsl(long double);



double fdim(double, double);
float fdimf(float, float);
long double fdiml(long double, long double);



double floor(double);
float floorf(float);
long double floorl(long double);



double fma(double, double, double);
float fmaf(float, float, float);
long double fmal(long double, long double, long double);



double fmax(double, double);
float fmaxf(float, float);
long double fmaxl(long double, long double);



double fmin(double, double);
float fminf(float, float);
long double fminl(long double, long double);



double fmod(double, double);
float fmodf(float, float);
long double fmodl(long double, long double);



double frexp(double, int *);
float frexpf(float, int *);
long double frexpl(long double, int *);



double hypot(double, double);
float hypotf(float, float);
long double hypotl(long double, long double);



int ilogb(double);
int ilogbf(float);
int ilogbl(long double);



double ldexp(double, int);
float ldexpf(float, int);
long double ldexpl(long double, int);




double lgamma(double);
float lgammaf(float);
long double lgammal(long double);




long long llrint(double);
long long llrintf(float);
long long llrintl(long double);



long long llround(double);
long long llroundf(float);
long long llroundl(long double);




double log(double);
float logf(float);
long double logl(long double);



double log10(double);
float log10f(float);
long double log10l(long double);



double log1p(double);
float log1pf(float);
long double log1pl(long double);



double log2(double);
float log2f(float);
long double log2l(long double);



double logb(double);
float logbf(float);
long double logbl(long double);



long lrint(double);
long lrintf(float);
long lrintl(long double);



long lround(double);
long lroundf(float);
long lroundl(long double);



double modf(double, double *);
float modff(float, float *);
long double modfl(long double, long double *);



double nan(const char *);
float nanf(const char *);
long double nanl(const char *);



double nearbyint(double);
float nearbyintf(float);
long double nearbyintl(long double);



double nextafter(double, double);
float nextafterf(float, float);
long double nextafterl(long double, long double);



double nexttoward(double, long double);
float nexttowardf(float, long double);
long double nexttowardl(long double, long double);
# 326 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/math.h" 3
double pow(double, double);
__attribute__((nonreentrant)) float powf(float, float);
long double powl(long double, long double);



double remainder(double, double);
float remainderf(float, float);
long double remainderl(long double, long double);



double remquo(double, double, int *);
float remquof(float, float, int *);
long double remquol(long double, long double, int *);



double rint(double);
float rintf(float);
long double rintl(long double);



double round(double);
float roundf(float);
long double roundl(long double);



double scalbln(double, long);
float scalblnf(float, long);
long double scalblnl(long double, long);



double scalbn(double, int);
float scalbnf(float, int);
long double scalbnl(long double, int);



double sin(double);
float sinf(float);
long double sinl(long double);



double sinh(double);
float sinhf(float);
long double sinhl(long double);



double sqrt(double);
float sqrtf(float);
long double sqrtl(long double);



double tan(double);
float tanf(float);
long double tanl(long double);



double tanh(double);
float tanhf(float);
long double tanhl(long double);



double tgamma(double);
float tgammaf(float);
long double tgammal(long double);



double trunc(double);
float truncf(float);
long double truncl(long double);
# 431 "C:\\Program Files\\Microchip\\xc8\\v3.10\\pic\\include\\c99/math.h" 3
extern int signgam;

double j0(double);
double j1(double);
double jn(int, double);

double y0(double);
double y1(double);
double yn(int, double);
# 6773 "../fatfs/ff.c" 2

static int ilog10 (double n)
{
 int rv = 0;

 while (n >= 10) {
  if (n >= 100000) {
   n /= 100000; rv += 5;
  } else {
   n /= 10; rv++;
  }
 }
 while (n < 1) {
  if (n < 0.00001) {
   n *= 100000; rv -= 5;
  } else {
   n *= 10; rv--;
  }
 }
 return rv;
}


static double i10x (int n)
{
 double rv = 1;

 while (n > 0) {
  if (n >= 5) {
   rv *= 100000; n -= 5;
  } else {
   rv *= 10; n--;
  }
 }
 while (n < 0) {
  if (n <= -5) {
   rv /= 100000; n += 5;
  } else {
   rv /= 10; n++;
  }
 }
 return rv;
}


static void ftoa (
 char* buf,
 double val,
 int prec,
 TCHAR fmt
)
{
 int d;
 int e = 0, m = 0;
 char sign = 0;
 double w;
 const char *er = 0;
 const char ds = 1 == 2 ? ',' : '.';


 if (( __fpclassifyf(val) == 0 )) {
  er = "NaN";
 } else {
  if (prec < 0) prec = 6;
  if (val < 0) {
   val = 0 - val; sign = '-';
  } else {
   sign = '+';
  }
  if (( __fpclassifyf(val) == 1 )) {
   er = "INF";
  } else {
   if (fmt == 'f') {
    val += i10x(0 - prec) / 2;
    m = ilog10(val);
    if (m < 0) m = 0;
    if (m + prec + 3 >= 32) er = "OV";
   } else {
    if (val != 0) {
     val += i10x(ilog10(val) - prec) / 2;
     e = ilog10(val);
     if (e > 99 || prec + 7 >= 32) {
      er = "OV";
     } else {
      if (e < -99) e = -99;
      val /= i10x(e);
     }
    }
   }
  }
  if (!er) {
   if (sign == '-') *buf++ = sign;
   do {
    if (m == -1) *buf++ = ds;
    w = i10x(m);
    d = (int)(val / w); val -= d * w;
    *buf++ = (char)('0' + d);
   } while (--m >= -prec);
   if (fmt != 'f') {
    *buf++ = (char)fmt;
    if (e < 0) {
     e = 0 - e; *buf++ = '-';
    } else {
     *buf++ = '+';
    }
    *buf++ = (char)('0' + e / 10);
    *buf++ = (char)('0' + e % 10);
   }
  }
 }
 if (er) {
  if (sign) *buf++ = sign;
  do {
   *buf++ = *er++;
  } while (*er);
 }
 *buf = 0;
}




int f_printf (
 FIL* fp,
 const TCHAR* fmt,
 ...
)
{
 va_list arp;
 putbuff pb;
 UINT i, j, w, f, r;
 int prec;

 QWORD v;



 TCHAR *tp;
 TCHAR tc, pad;
 TCHAR nul = 0;
 char d, str[32];


 putc_init(&pb, fp);

 *arp = __va_start();

 for (;;) {
  tc = *fmt++;
  if (tc == 0) break;
  if (tc != '%') {
   putc_bfd(&pb, tc);
   continue;
  }
  f = w = 0; pad = ' '; prec = -1;
  tc = *fmt++;
  if (tc == '0') {
   pad = '0'; tc = *fmt++;
  } else if (tc == '-') {
   f = 2; tc = *fmt++;
  }
  if (tc == '*') {
   w = (*(int *)__va_arg(*(int **)arp, (int)0));
   tc = *fmt++;
  } else {
   while (((tc) >= '0' && (tc) <= '9')) {
    w = w * 10 + tc - '0';
    tc = *fmt++;
   }
  }
  if (tc == '.') {
   tc = *fmt++;
   if (tc == '*') {
    prec = (*(int *)__va_arg(*(int **)arp, (int)0));
    tc = *fmt++;
   } else {
    prec = 0;
    while (((tc) >= '0' && (tc) <= '9')) {
     prec = prec * 10 + tc - '0';
     tc = *fmt++;
    }
   }
  }
  if (tc == 'l') {
   f |= 4; tc = *fmt++;

   if (tc == 'l') {
    f |= 8; tc = *fmt++;
   }

  }
  if (tc == 0) break;
  switch (tc) {
  case 'b':
   r = 2; break;

  case 'o':
   r = 8; break;

  case 'd':
  case 'u':
   r = 10; break;

  case 'x':
  case 'X':
   r = 16; break;

  case 'c':
   putc_bfd(&pb, (TCHAR)(*(int *)__va_arg(*(int **)arp, (int)0)));
   continue;

  case 's':
   tp = (*(TCHAR* *)__va_arg(*(TCHAR* **)arp, (TCHAR*)0));
   if (!tp) tp = &nul;
   for (j = 0; tp[j]; j++) ;
   if (prec >= 0 && j > (UINT)prec) j = prec;
   for ( ; !(f & 2) && j < w; j++) putc_bfd(&pb, pad);
   while (*tp && prec--) putc_bfd(&pb, *tp++);
   while (j++ < w) putc_bfd(&pb, ' ');
   continue;

  case 'f':
  case 'e':
  case 'E':
   ftoa(str, (*(double *)__va_arg(*(double **)arp, (double)0)), prec, tc);
   for (j = strlen(str); !(f & 2) && j < w; j++) putc_bfd(&pb, pad);
   for (i = 0; str[i]; putc_bfd(&pb, str[i++])) ;
   while (j++ < w) putc_bfd(&pb, ' ');
   continue;

  default:
   putc_bfd(&pb, tc); continue;
  }



  if (f & 8) {
   v = (QWORD)(*(long long *)__va_arg(*(long long **)arp, (long long)0));
  } else if (f & 4) {
   v = (tc == 'd') ? (QWORD)(long long)(*(long *)__va_arg(*(long **)arp, (long)0)) : (QWORD)(*(unsigned long *)__va_arg(*(unsigned long **)arp, (unsigned long)0));
  } else {
   v = (tc == 'd') ? (QWORD)(long long)(*(int *)__va_arg(*(int **)arp, (int)0)) : (QWORD)(*(unsigned int *)__va_arg(*(unsigned int **)arp, (unsigned int)0));
  }
  if (tc == 'd' && (v & 0x8000000000000000)) {
   v = 0 - v; f |= 1;
  }
# 7029 "../fatfs/ff.c"
  i = 0;
  do {
   d = (char)(v % r); v /= r;
   if (d > 9) d += (tc == 'x') ? 0x27 : 0x07;
   str[i++] = d + '0';
  } while (v && i < 32);
  if (f & 1) str[i++] = '-';

  for (j = i; !(f & 2) && j < w; j++) {
   putc_bfd(&pb, pad);
  }
  do {
   putc_bfd(&pb, (TCHAR)str[--i]);
  } while (i);
  while (j++ < w) {
   putc_bfd(&pb, ' ');
  }
 }

 ((void)0);

 return putc_flush(&pb);
}
