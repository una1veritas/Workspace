/* Textcoding Unicode (UTF-8, no BOM), End of Line Windows (CRLF)

bin2wav.c

		V 1.3	www.pocketmuseum.com

2011-12-05	V 1.4	Manfred NOSSWITZ
	Changed to ANSI-C e.g. strupr()
	Command line parser changed to getopt().
	Quiet mode added.
	32bit compilation with gcc (tdm64-1) 4.6.1 (WindowsXP-prof [32bit]): gcc -pedantic -m32 -o xxx xxx.c
	32bit compilation with gcc-4_5-branch revision 167585 (OpenSUSE 11.4 [32bit]): gcc -pedantic -m32 -o xxx xxx.c
	64bit compilation with gcc (tdm64-1) 4.6.1 (Windows7-prof [64bit]): gcc -pedantic -m64 -o xxx xxx.c
	64bit compilation with gcc-4_5-branch revision 167585 (OpenSUSE 11.4 [64bit]): gcc -pedantic -m64 -o xxx xxx.c
	For testing PC-1402 was available only

2013-11-13	V 1.4.1	Torsten Muecker
	Changed for PC-14xx stop bits after last checksum to 11SS, silence added to var 'bit'
	Tested with PC-1401/02, PC-1403H, PC-1475
2013-12-16	V 1.4.2 beta2 Torsten Muecker
	-DAT for a data variable block added, only ONE data variable block (no list) and for PC-1261-1475 is supported
	-VAR for a data variable block without file name, can be appended by wave editor tools to an existing data wave
     How to concatenate wave data files with SoX (example): "sox DAT1.wav VAR2.wav DAT.wav"
2014-04-21	V 1.4.3 beta 12 Torsten Muecker
    - Debug Option "-l" added
    - Compatibility with new Wav2Bin 1.60:VAR-DAT-Files without all Checksum
    - Arguments buffer length protected, Exit code added
    - Support for Multiple Data Blocks added with Wav2Bin 1.6,
      please use debug option DATA_W2B150 for data files from Wav2Bin 1.5 (limitted)
    - Debug option NO_FILE_HEAD added, ident IDENT_DATA_VAR removed
    - PC-1421, PC-1280 and more PCs added
    - Tidy up --help screen
    - Arg default name corrected
    - Grouping of PCs now
    - Relocate code for Data from Convert to HeadToData
    - Remapping of more PC names implemented
    - removed some compiler warnings
    - All EOF-Labels shown in debug mode now
    - RSV-Data tested with Wav2bin
    - OLD_DAT full implemented, 121_DAT experimentally
      Tested with PC-1251
    - Option cspeed for clock modified Pocket Computers with a Hardware SpeedUp
    - DAT, RSV for PC-1500 added, Tested with PC-1500
2014-06-13	V 1.4.4 beta 5c Torsten M�cker
    - program code compression WriteByte(Sum)toBxxWav with MODE
    - checksums improved, when EOF overlap with end of checksum block, double EOF marks removed
    - check for activating seldom NEW/EXT_BAS CLOAD ERR_SUM bug
    - unified end of file mark of all basic images: appended only by Bin2wav
    - checksum before end of file changed for BAS_NEW, BAS_EXT
    - conversion of data variables between PC-1500 and other
    - PC 1211 program and data added
    - replaced fgetpos with ftell
    - conversion of string variable (with debug flag 0x10 )
    - some changes in categorisation of pocket computers
2014-09-30	V 1.4.5 beta 10L Torsten M�cker
    - PC-E series implemented, E_BIN, E_BAS, E_DAT
    - added entry address for E_BIN, added screen --help=l
    - added old arguments conversion to new arguments for backward compatibility with 3rd-party software
    - added option and destination TAP format for emulators, from Olivier De Smet bin2tap 2013-10-05 v1.5
    - implemented G850, Basic, Binary (with data there exist the same problem as with native wave files)
    - more waveform variants triangle, rectangle, trapezoidal with --level 1, 2: 48kHz (44.1 for PC-1500)
    - changed the default waveform to more stable trapezoidal, use -l 1 if you need the old compact format
    - ASCII modus E/G/1600-series (Data, SAVE CAS:)
    - Text modus of GRP_EXT, GRP_E (CSAVE, CLOAD) implemented
    - implemented support for Transfile PC plus SHC files
    - PC-1403 and newer changed to MODE_14
    - implemented PC-1600 DAT,
    - corrected tap-Format for PC-1500
    - 2. --sync for silence separate command line parameter
    - PC-1211: SHC supported (PC-1251)
2014-11-26	V 1.9.8 beta 08f1 Torsten M�cker
    - device parameter: -dINV mirroring all wave forms because of ASCII data
    - precise sync gaps for data
    - entry address for PC-1500,
    - default start address for most PC
    - Waveform 16 kHz with -l3 for Emulator
    - type detection from file extension
    - allow underline in filename and changed spaces at end and begin of 16/G/E series
    - optimized sync length and debug options for 16/G/E series, especially E-ASC
    - End address for all binaries, nbsamp
2015-03-29	V 1.9.9 gamma 4 (3c1) Torsten M�cker
    - debug traces not longer than print for 256 byte a line
    - New image end of transmission more precise
    ToDo: Public TESTs, Help needed
2015-06-08	V 2.0.0 (3c1) Torsten M�cker
2015-11-18	V 2.0.0b      Torsten M�cker
	- Runtime Error with 64bit systems in source code corrected, thanks to H-B. E.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>		/* Command line parser getopt(). */
#include <ctype.h>

                        /* delete "//" in next line to activate debug modus by default */
/* #define DEBUG           1    should be commented for final releases */
#ifdef DEBUG
   #define DEBUG_ARG    "0x00C0"   /* Set to 0xC0 for more output with data variable blocks  */
#else
   #define DEBUG_ARG    "0"
#endif // DEBUG

/* Regular Errors such as EOF and recoverable such as End of sync *MUST* have a lower value then ERR_OK */
#define ERR_NOK        -1   /* also normal EOF */
#define ERR_OK          0

/* Unexpected NOT recoverable error *MUST* have a higher value then ERR_OK */
#define ERR_SYNT        1   /* arguments missed, syntax error */
#define ERR_ARG         3   /* arguments problem, pocket not implemented */
#define ERR_FILE        5   /* File I-O */
#define ERR_FMT         7   /* Source file format or size */
#define ERR_SUM         8   /* Constellation will generate a checksum error on PC */

#define TYPE_NOK        0
#define TYPE_BIN        1
#define TYPE_IMG        2
#define TYPE_VAR        3   /* one variable block without file type and name*/
#define TYPE_DAT        4
#define TYPE_RSV        5   /* For PC-1500, other re-transfer: IMG-compatible */
#define TYPE_ASC        6   /* For PC-E/G/1600 ASCII Data */
#define TYPE_BAS        7   /* For PC-E/G/1600 ASCII Source */
#define TYPE_TXT        8   /* For GRP_NEW, GRP_EXT and GRP_E text modus Image Data */

#define GRP_OLD         0x20
#define GRP_NEW         0x70
#define GRP_EXT         0x72
#define GRP_DAT         0x74
#define GRP_16          0x10    /* PC-1600 */
#define GRP_E           0x05    /* PC-E500 */
#define GRP_G           0x08    /* PC-G850, E2xx */

#define IDENT_PC1211    0x80
#define IDENT_PC121_DAT 0x8F    /* one variable length block only, not tested */
#define IDENT_PC1500    0xA

#define IDENT_PC15_BIN  0xA0
#define IDENT_PC15_BAS  0xA1
#define IDENT_PC15_RSV  0xA2
#define IDENT_PC15_DAT  0xA4
#define IDENT_OLD_BAS   0x20
#define IDENT_OLD_PAS   0x21
#define IDENT_OLD_DAT   0x24
#define IDENT_OLD_BIN   0x26
#define IDENT_NEW_BAS   0x70
#define IDENT_NEW_PAS   0x71
#define IDENT_EXT_BAS   0x72
#define IDENT_EXT_PAS   0x73
#define IDENT_NEW_DAT   0x74    // IDENT_DATA_VAR 0xEF removed
#define IDENT_NEW_BIN   0x76
#define IDENT_UNKNOWN   0x100

#define IDENT_PC16_CAS  0x00    /* PC-1600 ASCII Data or BASIC Image */
#define IDENT_PC16_DAT  0x08    /* Special binary data format PC-1600 */
#define IDENT_E_ASC     0x04    /* ASCII Data or Text Modus BASIC/ASM/C Source */
#define IDENT_E_BAS     0x02    /* Return from Bas2Img also in the format of PC-1475 */
#define IDENT_E_BIN     0x01    /* Binary Data, use --addr parameter 2nd time for entry address */
                                /* also used for Rsv-Data, but different sub id (mode 2) */
#define SYNC_E_HEAD     40      /* additional sync for E-series header block */
#define SYNC_E_DATA     20      /* additional sync for E-series data block */

#define DATA_HEAD_LEN   5       /* length of the header of a data variable element*/
#define DATA_VARIABLE   0xFE00
#define DATA_STD_LEN    8       /* length of a standard data variable element*/
#define DATA_NUM_15     0x88    /* ItemLen = Id of numeric data variable of PC-1500 */

#define DATA_NUM        0x98    /* Internal identity of numeric data variable */
#define DATA_STR        0xA8    /* Internal identity of a string data variable */
#define DATA_UNKNOWN    0x00    /* Variable filled with zeros */

#define DATA_STD_STR    0xF5    /* Standard data variable is a string */
#define DATA_EOF        0x0F    /* End of variable Data Block */

#define EOF_ASC         0x1A    /* End of ASCII transfered files, also CAS: of newer series */
#define EOF_15          0x55    /* End of complete file of PC-1500 */
#define BAS_1500_EOF    0xFF    /* one of two bytes */
#define BAS_NEW_EOF     0xFF    /* one of two bytes */
#define BAS_OLD_EOF     0xF0    /* one byte only */

#define BLK_OLD_SUM     8       /* Transmission Block (plus checksums), old series bas without, data old/new with checksum reset */
#define BLK_OLD         80      /* Transmission Block (plus checksums) of PC-1500 and old series with checksum reset */
#define BLK_NEW         120     /* Transmission Block (plus checksum) of new series 1260-1475 */
#define BLK_E_DAT       256     /* Checksum of series E500 DAT */
#define BLK_E_HEAD      0x30    /* Length of header1 of series E500, next only after the end of a Transmission Block */

#define BGNSAMPLES      44      /* first sample byte */

#define AMP_MID         0x80    /* Sample value for silence (all 8-bit) */
#define AMP_HIGH        0xDA    /* Sample value for amplitude HIGH */
#define AMP_LOW         0x26    /* Sample value for amplitude LOW */
#define AMP_HIGH_E      0xFC    /* Sample value for amplitude HIGH for E-series */
#define AMP_LOW_E       0x04    /* Sample value for amplitude LOW for E-series */

#define ORDER_STD       0
#define ORDER_INV       1
#define ORDER_E         8       /* no nibbles, a byte */

#define BASE_FREQ1      4000
#define BASE_FREQ2      2500    /* PC-1500 */
#define BASE_FREQ3      3000    /* PC-E-series and newer */

#define CHAR_SPACE      0x20
#define CHAR_DOT        0x2E
#define CHAR_COLON      0x3A
#define CHAR_SLASH      0x5C
#define CHAR_UNDERSCORE 0x5F

#define cVL             80		/* Constant value for max. length of a data variable */
#define cLPF 129		        /* Constant value for max. Length of PathFilenames */

#define DATA_W2B150     0x8000 /* debug flag for DAT-IMG from Wav2bin 1.5.0 or older */
#define NO_FILE_HEAD    0x4000 /* debug flag, write without file type and -name, 2.variable block */
#define BAS_EXT_FRMT    0x1000 /* debug flag, use FORMAT of BAS_EXT for E-Series */
#define SYNCL_STD       0x400  /* debug flag, for default sync length like the original */
#define SYNCL_TRM       0x200  /* debug flag, for sync length like the Technical Reference Manual */

#define MODE_B22        0   /* PC-1500 */
#define MODE_B21        1
#define MODE_B20        2
#define MODE_B19        3   /* Old series */
#define MODE_B17        4
#define MODE_B16        5   /* File names of new series an data */
#define MODE_B15        6   /* Variable data body new series */
#define MODE_B14        7   /* PC-1403 and newer, reads also B13 */
#define MODE_B13        8   /* PC-1402 */
#define MODE_B9         9   /* PC-E series and newer */

#define true 1
#define false 0
/*                      used types  */
typedef     int		    bool;
/*			char;	    */
typedef unsigned char	uchar;
// typedef unsigned char   uint8_t;
/*			short;	    */
typedef unsigned short	ushort;
/*			int;	    */
typedef unsigned int	uint;
/*			long;	    */
typedef unsigned long	ulong;

  char  argP[cLPF] = "" ;
  uint  SHCc  = 0 ,     /* Read not from bin, but from Transfile PC plus SHC-format (header included) */
        SHCe  = 0 ;     /* End marks are the SHC Basic image included only*/
  uint  TAPc  = 0 ;     /* Write not to wave format but to emulator tap format (raw bytes)*/
  uint  Qcnt  = 0 ;     /* Quiet, minimal output */
  uint  Scnt = 0 ;      /* sync parameter or sync and sync spaces was defined */

double  speed = 1.00 ;  /* Used it for modified Pocket Computer with a CPU Speed Up switched on */

 ulong  pcId  = 1500 ;  /* ID number of the pocket computer */
ushort  pcgrpId = IDENT_UNKNOWN ;
  bool  cnvstr_upr = false ;

typedef struct {
    FILE*  ptrFd ;
   ushort  ident ;
    uchar  mode ;       /* Bit writing mode and stop bits*/
    uchar  mode_h ;     /* Bit writing mode and stop bits for the file header*/
    ulong  nbSync1 ;    /* sync bits per second */
    ulong  nbSync ;     /* sync bits for intermediate syncs */
    ulong  total ;      /* total bytes written (without checksums) */
    long   total_diff ; /* difference between bytes read and written (without checksums) */
    ulong  count ;      /* Counter of bytes in a block for the checksum */
    ulong  sum ;        /* Checksum calculated */
    ulong  block_len ;  /* Block length for checksum, variable used for E-Series */
    ulong  bitLen ;     /* Wave sample blocks per SHARP bit */
    ulong  debug ;
} FileInfo ;

typedef struct {
      uint  length ;  /* Length of one data variable block or marker for standard data variable block */
     uchar  dim1 ;    /* data variable block array vector columns dim1*/
     uchar  dim2 ;    /* data variable block array matrix lines dim2*/
     uchar  itemLen ; /* Len of one data variable element or marker */
} TypeInfo ;

typedef struct {
    uint  stopb1 ;
    uint  stopb2 ;
} ModeInfo ;

static ModeInfo Mode[] = { /* stop bits of first and second nibble */
    { 6, 6 }, { 5, 6 }, { 5, 5 }, { 4, 5 },
    { 1, 6 }, { 1, 5 }, { 1, 4 }, { 1, 3 }, { 1, 2 }
} ;

static bool bitMirroring = false ;
/* Pattern for WriteBit */
/* Bit low pattern, Bit high pattern, silence, shutdown signal */
/* Amp Max := +     Amp Min := -     Silence := . */

static char* bit1[] = { /* old compressed signal for PC-1210 ... PC-1500 */
    "++--++--++--++--", /* Bit_0 */
    "+-+-+-+-+-+-+-+-", /* Bit_1, sync, waveform: triangle signal */
	"................", /* silence */
	"+---------------"  /* shutdown, depending of freq sound hardware +1 bit */
} ;

static char* bit2[] = {  /* new signal for PC-1210 ... PC-1500, more gain independent */
    "++++----++++----++++----++++----",
    "++--++--++--++--++--++--++--++--",/* Bit_1 trapezoidal signal*/
	"................................",
	"++------------------------------"
} ;

static char* bit3[] = {                /* 48 kHz near rectangle signal for PC-1210 ... PC-1475 */
    ".+++++++++++.-----------.+++++++++++.-----------.+++++++++++.-----------.+++++++++++.-----------",
    ".+++++.-----.+++++.-----.+++++.-----.+++++.-----.+++++.-----.+++++.-----.+++++.-----.+++++.-----",
	"................................................................................................",
    ".+++++.-----------------------------------------------------------------------------------------"
} ;

static char* bit3_15[] = {             /* 44.1 kHz sample rate signal for PC-1500 */
    "...+++++++++++++++...--------------...+++++++++++++++...--------------...+++++++++++++++...--------------...+++++++++++++++...---------------",
    "..+++++++..-------..++++++..-------..+++++++..-------..+++++++..------..+++++++..-------..+++++++..-------..++++++..-------..+++++++..-------",
	".............................................................................................................................................",
	"............................................................................................................................................."
} ;

static char** bit4 = bit2 ; /* 16kHz wave form */

static char* bit4_15[] = {             /* 16 kHz sample rate signal for PC-1500 */
    ".++++++------++++++.------++++++------.++++++------",
    ".+++---+++---+++.---+++---+++---.+++---+++---+++---",
	"...................................................",
    ".+++---+++---+++.---+++---+++---.+++---+++---+++---"
} ;

static char** bit = bit2 ; /* default wave form */

/* The following pattern are used for the PC-E500 and PC-G800 series and especially
   for ASCII Source/Data from the interface dependent have to start with '+' or '-' */

/* PC-E500 series only: smaller files but longer synchronisation than bitE2 needed, Not for G-800 series */
static char* bitE1[] = { /* compressed format, not readable from all PCs */
    "-+",           /* Bit_0, waveform: triangle signal */
    "---++",        /* Bit_1 */
	"..",           /* silence */
    "---+++++++++"  /* header stop bit 1 with signal shutdown */
} ;

/* PC-E500 and newer, G800-series, variable bit length, trapezoidal, stable */
static char* bitE2[] = {
    "--++",                   /* Bit_0 trapezoidal signal */
    "-----+++++",
	"....",
    "-----++++++++++++++++++"
} ;

/* This set writes 48 kHz files. Only *near* rectangle works for G-800 series - threshold needed. */
static char* bitE3[] = {
    ".-------.+++++++",
    ".-------------------.+++++++++++++++++++",
	"................",
    ".-------------------.+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
} ;

/* 16 kHz sample rate signal  */
static char* bitE4[] = {
    "--.++",
    "------.++++++",
	".....",
    "------.++++++++++++++++++++++++"
} ;

static char** bitE = bitE2 ;  /* default wave form for E/G Series*/

static uint bitLE[] = {  /* default values from bitE2, will calculated new in ConvertB. */
    4, 10, 	4,  23
} ;


static ulong CodeOld[] = {
    0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,
    0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,
    0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,
    0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,
    0x11,0x14,0x12,0x15,0x19,0x16,0x1F,0x11,
    0x30,0x31,0x37,0x35,0x1B,0x36,0x4A,0x38,
    0x40,0x41,0x42,0x43,0x44,0x45,0x46,0x47,
    0x48,0x49,0x1D,0x1C,0x33,0x34,0x32,0x13,
    0x1E,0x51,0x52,0x53,0x54,0x55,0x56,0x57,
    0x58,0x59,0x5A,0x5B,0x5C,0x5D,0x5E,0x5F,
    0x60,0x61,0x62,0x63,0x64,0x65,0x66,0x67,
    0x68,0x69,0x6A,0x11,0x11,0x11,0x39,0x11,
    0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,
    0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,
    0x19,0x11,0x1A,0x11,0x11,0x11,0x11,0x11,
    0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11
} ;


void CvShortToStringI (uint  value,
                       char*  ptrStr)
{
    uint  tmp ;

    /* Convert the short value into a String with msb first (INTEL) */
    tmp = value & 0xFF ;
    *ptrStr ++ = (char) tmp ;
    tmp = value >> 8 ;
    *ptrStr ++ = (char) tmp ;
    *ptrStr = 0 ;
}

void CvLongToStringI (ulong  value,
                      char*  ptrStr)
{
    ulong  tmp ;

    /* Convert the 32bit value into a String with msb first (INTEL) */
    tmp = value & 0xFF ;
    *ptrStr ++ = (char) tmp ;
    tmp = (value >> 8) & 0xFF ;
    *ptrStr ++ = (char) tmp ;
    tmp = (value >> 16) & 0xFF ;
    *ptrStr ++ = (char) tmp ;
    tmp = value >> 24 ;
    *ptrStr ++ = (char) tmp ;
    *ptrStr = 0 ;
}


/* String-change UPPER */
char *strupr( char  *string )
{
  int  i = 0;
  while ( ( string[i] = toupper( string[i] ) ) != '\0') ++i;
  return string;
}


/* String-change LOWER */
char *strlor( char  *string )
{
  int  i = 0;
  while ( ( string[i] = tolower( string[i] ) ) != '\0') ++i;
  return string;
}


int WriteStringToFile (char*  ptrStr,
                       FILE**  ptrFd)
{
    int  error ;

    error = fputs (ptrStr, *ptrFd) ;
    if (error == EOF) {
        printf ("%s: Can't write in the file\n", argP) ;
        error = ERR_FILE ;
    }
    else {
        error = ERR_OK ;
    }
    return (error);
}


int WriteLongToFile (ulong  value,
                     FILE**  ptrFd)
{
    char  str[10] ;
     int  ii ;
     int  error ;

    CvLongToStringI (value, str) ;

    for ( ii = 0 ; ii < 4 ; ++ii  ) {
        error = fputc (str[ii], *ptrFd) ;
        if (error == EOF) {
            printf ("%s: Can't write in the file\n", argP) ;
            error = ERR_FILE ;
            break ;
        }
        else
            error = ERR_OK ;
    }
    return (error);
}


int WriteShortToFile (ulong  value,
                      FILE**  ptrFd)
{
    char  str[10] ;
     int  ii ;
     int  error ;

    CvShortToStringI (value, str) ;

    for ( ii = 0 ; ii < 2 ; ++ii ) {
        error = fputc (str[ii], *ptrFd) ;
        if (error == EOF) {
            printf ("%s: Can't write in the file\n", argP) ;
            error = ERR_FILE ;
            break ;
        }
        else
            error = ERR_OK ;
    }
    return (error);
}


int ReadFileLength ( uchar  type,
                    ulong*  ptrLen,
                    FILE**  ptrFd)
{
    long  nbByte ;
     int  inVal ;
     int  error ;
     /* SHCc is the length of a SHC header or zero,
        set in global and ReadHeadFromShc */
     fprintf(stderr,"entered ReadFileLength.\n");

    do {
        *ptrLen = 0 ;

        /* Seek to the end of the source file */
        error = fseek (*ptrFd, 0, SEEK_END) ;
        if (error != ERR_OK) {
            printf ("%s: Can't seek the file\n", argP) ;
            error = ERR_FILE ;
            break ;
        }

        /* Get the length of the source file */
        nbByte = ftell (*ptrFd) ;
        fprintf(stderr,"Got the length of the source file: %ld.\n", nbByte);
        if (nbByte == ERR_NOK) {
            printf ("%s: Can't ftell the file\n", argP) ;
            error = ERR_FILE ;
            break ;
        }

        if (type== TYPE_IMG && nbByte > (long) SHCe) { /* check if EOF marks included in the IMG*/
            error = fseek (*ptrFd, -(long)SHCe-1, SEEK_CUR) ;

            if (error != ERR_OK) {
                printf ("%s: Can't seek the file\n", argP) ;
                error = ERR_FILE ;
                break ;
            }
            inVal = fgetc (*ptrFd) ;
            if (inVal == EOF) {
                    error = ERR_FILE ;
                    break ;
            }
            else if (inVal == BAS_1500_EOF) ++SHCe ; /* mark was included in old Bas2IMG */
        }
        fprintf(stderr,"nbByte = %ld.\n", nbByte);
        fprintf(stderr,"SHCe = %d.\n", SHCe);

        /* Seek to the beginning of the source file or
           after a SHC-Header, if type SHC was used */
        error = fseek (*ptrFd, SHCc, SEEK_SET) ;
        if (error != ERR_OK) {
            printf ("%s: Can't seek the file\n", argP) ;
            error = ERR_FILE ;
            break ;
        }
        nbByte -= ( SHCc + SHCe ); /* SHCe is the length of additional end marks */

        if (nbByte <= 0) {
            printf ("%s: Source file is empty\n", argP) ;
            error = ERR_FILE ;
            *ptrLen = 0 ;
        }
        else
            *ptrLen = nbByte ;

    } while (0) ;
    return (error);
}

/* The setting of Identity is used but length calculation (number of samples) is obsolete now,
   this length calculation is for the old waveform and the identities of older versions only
*/
int LengthAndIdentOfBinWav (uchar  type,
                            ulong  nbByte,
                            ulong  nbSync,
                           ushort*  ptrIdent,
                            ulong*  ptrSamp,
                            ulong*  ptrDebug)
{
    ulong  nbSamp ;
    ulong  debugInfo ;
   ushort  ident ;
   ushort  grpId = pcgrpId ;
   ulong   lmode;  /* byte length with stop bits*/
      int  error = ERR_OK ;

    debugInfo = *ptrDebug ;

    do {
        if (type == TYPE_DAT && pcgrpId != IDENT_PC1500 && pcgrpId != GRP_E && pcgrpId != GRP_G  && pcgrpId != GRP_16) {
        // GRP_16 special data no pre-calculation of samples supported
            if (pcgrpId == IDENT_PC1211) ident = IDENT_PC121_DAT ;
            else if (pcgrpId == GRP_OLD) ident = IDENT_OLD_DAT ;
            else ident = IDENT_NEW_DAT ; /* EXT_DAT identical format */
            grpId = GRP_DAT ; /* pcgrpId local used only */
        }
        switch (grpId) {
        case GRP_DAT :
            if ((debugInfo & NO_FILE_HEAD) == 0) {
                    nbSamp = 1 + 9 ;     /* File type and name, + 1 Checksum incl. */
            }
            else nbSamp = 0 ;
            if ( ident == IDENT_PC121_DAT || ident == IDENT_OLD_DAT )
                nbSamp = (nbSamp + 6) * 19 * 16 ;  /* Var Header = 16 Byte19, Checksum incl. */
            else
                nbSamp = (nbSamp + 6) * 16 * 16 ;  /* Var Header = 16 Byte16, Checksum incl. */

            if ( (debugInfo & DATA_W2B150) > 0 )
                    nbByte -= 6 ;  /* 5 Byte Data Block def + 1 Byte Checksum wav2bin 1.5 */
            else
                    nbByte -= 5 ;  /* 5 Byte Data Block def */

            nbSamp += nbSync * 16 ;
            /* works only for one array block and not for fixed variables, but will corrected at the end*/
            nbSamp += (nbByte + (1 + (nbByte - 1) / 8)) * 15 * 16 ;
            nbSamp += 15 * 16 ;                     /* EOF */
            nbSamp += (1 + 2) * 16 ;        /* End of Transmission 1 Bit more orig. + 2 */
            break;

        case IDENT_PC1500 :
            if (type == TYPE_BIN) {
                ident = IDENT_PC15_BIN ;
                /* Footer = 3 bytes : sum, 0x55 */
                nbSamp = 3 ;
            }
            else if (type == TYPE_RSV) {
                ident = IDENT_PC15_RSV ;
                /* Footer = 3 bytes : sum, 0x55 */
                nbSamp = 3 ;
            }
            else if (type == TYPE_DAT) {
                ident = IDENT_PC15_DAT ;
                /* Header Name + 8 byte, Footer = 3 bytes : sum, 0x55 */
                nbSamp = 11 ; /* missing interim sync, will corrected at end of processing */
            }
            else {
                /* Footer = 4 bytes : 0xFF, sum, 0x55 */
                ident = IDENT_PC15_BAS ;
                nbSamp = 4 ;
            }

            /* Header = 42, Lead = nbByte + 2 * (nbByte / 80) */
            /* One Byte = 22 bits, each Bit = 16 Samples      */
            /* One Sync = one Bit = 16 Samples (2500 Hz)      */
            /* Ident  = 11 bits, 3 gaps: 72+  72+ 70 bits     */

            nbSamp += 42 + nbByte + 2 * (nbByte / 80) ;
            nbSamp *= 22 * 16 ;
            nbSamp += nbSync * 16 ;
            nbSamp += 225 * 16 ;
            if (ident == IDENT_PC15_DAT) nbSamp += 239 * 16 ;
            break ;                             /* 311 - 72, 1.0 sec before first data block */

        case GRP_OLD :
        case IDENT_PC1211 : //ToDo: add length of start and end gaps
            if (type == TYPE_BIN) {
                ident = IDENT_OLD_BIN ;
                nbSamp = 1 + (2 * 9) ;  /* Header = 19 Byte */
            }
            else {
                ident = IDENT_OLD_BAS ;
                if (grpId == IDENT_PC1211) ident = IDENT_PC1211 ;
                nbSamp = 1 + 9 ;        /* Header = 10 Byte */
            }

            /* Footer = 1, Lead = nbByte + (nbByte / 8)  */
            /* One Byte = 19 bits, each Bit = 16 Samples */
            /* One Sync = one Bit = 16 Samples (4000 Hz) */

            nbSamp += 1 + nbByte + (nbByte / 8) ;
            nbSamp *= 19 * 16 ;
            nbSamp += nbSync * 16 ;
            break ;

        case GRP_NEW :
        case GRP_EXT :
            if (type == TYPE_BIN) {
                ident = IDENT_NEW_BIN ;
                nbSamp = (1 + (2 * 9)) * 16 * 16 ;  /* Header = 19 Byte16 */
            }
            else {
                if (grpId == GRP_EXT)
                    ident = IDENT_EXT_BAS ;
                else
                    ident = IDENT_NEW_BAS ;

                nbSamp = (1 + 9) * 16 * 16 ;        /* Header = 10 Byte16 */
            }

            /* Footer = 3 Byte13(14)                        */
            /* Lead = (nbByte + (nbByte / 120)) Byte13      */
            /* Byte = 16 or 13(14) bits, Bit = 16 Samples   */
            /* Sync = one Bit = 16 Samples (4000 Hz)        */
                if (cnvstr_upr) lmode = 13; else lmode = 14 ;
                nbSamp += (nbByte + 3 + (nbByte / 120)) * lmode * 16 ;
                nbSamp += nbSync * 16 ;

	    /* there are 2bits more HIGH at the end of transmission (at least for PC-1402) M. NOSSWITZ */
		/* additional 2bit HIGH or silence at the end of transmission (for PC-14xx ) T. Muecker */

                nbSamp += 64 ;
            break ;

        case GRP_16 :
        case GRP_E  :
        case GRP_G  :
            if (type == TYPE_BIN)
                ident = IDENT_E_BIN ;  /* mode 2 sub id may be Bin or Rsv */
            else if (type == TYPE_IMG || type == TYPE_TXT || (type == TYPE_RSV && grpId == GRP_16))
                ident = IDENT_E_BAS ;
            else if (type == TYPE_ASC){ /* ASCII Data or Source file from Text Modus */
                if (pcgrpId == GRP_16) ident = IDENT_PC16_CAS ; /*only data, for binary use TYPE_IMG */
                else ident = IDENT_E_ASC ;
            }
            else if (type == TYPE_BAS) ident = IDENT_E_ASC ; /* ASCII Source also for PC-1600 */
            else if (type == TYPE_DAT) /* PC-1600 special data similar to PC-1500 */
                ident = IDENT_PC16_DAT ;
            else
                ident  = IDENT_UNKNOWN ;
            nbSamp = 0 ;
            break ;

        default :
            nbSamp = 0 ;
            ident  = IDENT_UNKNOWN ;
            break ;
        }

        if ((debugInfo & 3) == 0) nbSamp<<=1;
        /* other waveforms not calculated */
        else if ((debugInfo & 3) != 1) nbSamp = 0;
        *ptrIdent = ident ;
        *ptrSamp  = nbSamp ;
        // *ptrDebug = debugInfo ;

    } while (0) ;
    return (error);
}


int WriteSampleCountToHeadOfWav (ulong  nbSamp, FileInfo*  ptrFile)
{
    ulong nbSampP ;
//    fpos_t position ;
    long position ;
    int  error ;

    do {

        /* Seek to the Head of the source file */

        error = fseek (ptrFile->ptrFd, 0, SEEK_END) ;
        if (error != ERR_OK) {
            printf ("%s: Can't seek the Wave file\n", argP) ;
            break ;
        }
        if ( (ptrFile->debug & 0x00C0) > 0 ) printf("\n") ;

//        fgetpos(ptrFile->ptrFd, &position) ;
        position = ftell( ptrFile->ptrFd ) ;

        nbSampP = position ;
        if (nbSampP < BGNSAMPLES) break ;

        nbSampP -= BGNSAMPLES ; /* All Header Bytes from WriteHeadToWav */

        if (nbSampP != nbSamp) {
            if (Qcnt == 0 && nbSamp > 0) {
                    if ( (ptrFile->debug & 0x0080) > 0)
                        printf ("Written %lu Samples (%lu estimated)\n", nbSampP, nbSamp) ;
                    else if ( ptrFile->debug != 0)
                        printf ("Written %lu Samples\n", nbSampP) ;
            }
            nbSamp = nbSampP ;

            error = fseek (ptrFile->ptrFd, 4, SEEK_SET) ;
            if (error != ERR_OK) {
                printf ("%s: Can't seek the Wave file\n", argP) ;
                break ;
            }
            error = WriteLongToFile ((nbSamp + BGNSAMPLES - 8), &ptrFile->ptrFd) ;
            if (error != ERR_OK) break ;

            error = fseek (ptrFile->ptrFd, BGNSAMPLES - 4, SEEK_SET) ;
            if (error != ERR_OK) {
                printf ("%s: Can't seek the Wave file\n", argP) ;
                break ;
            }
            error = WriteLongToFile (nbSamp, &ptrFile->ptrFd) ;  /* Nb Samples */
            if (error != ERR_OK) break ;

            error = fseek (ptrFile->ptrFd, 0, SEEK_END) ;
            if (error != ERR_OK) {
                printf ("%s: Can't seek the Wave file\n", argP) ;
                break ;
            }
        }

    } while (0) ;
    return (error);
}


int WriteHeadToWav (ulong  nbSamp,
                    ulong  freq,
                    FileInfo*  ptrFile)
{
    int  error ;

    do {
        error = WriteStringToFile ("RIFF", &ptrFile->ptrFd) ;
        if (error != ERR_OK) break ;

        error = WriteLongToFile ((nbSamp + 36), &ptrFile->ptrFd) ;
        if (error != ERR_OK) break ;

        error = WriteStringToFile ("WAVEfmt ", &ptrFile->ptrFd) ;
        if (error != ERR_OK) break ;

        error = WriteLongToFile (0x10, &ptrFile->ptrFd) ;
        if (error != ERR_OK) break ;

        error = WriteShortToFile (1, &ptrFile->ptrFd) ;      /* PCM */
        if (error != ERR_OK) break ;

        error = WriteShortToFile (1, &ptrFile->ptrFd) ;      /* Mono */
        if (error != ERR_OK) break ;

        error = WriteLongToFile (freq, &ptrFile->ptrFd) ;    /* Samp Freq */
        if (error != ERR_OK) break ;

        error = WriteLongToFile (freq, &ptrFile->ptrFd) ;    /* Byte / sec */
        if (error != ERR_OK) break ;

        error = WriteShortToFile (1, &ptrFile->ptrFd) ;      /* Byte / Samp x Chan */
        if (error != ERR_OK) break ;

        error = WriteShortToFile (8, &ptrFile->ptrFd) ;      /* Bit / Samp */
        if (error != ERR_OK) break ;

        error = WriteStringToFile ("data", &ptrFile->ptrFd) ;
        if (error != ERR_OK) break ;

        error = WriteLongToFile (nbSamp, &ptrFile->ptrFd) ;  /* Nb Samples */
        if (error != ERR_OK) break ;

    } while (0) ;
    return (error);
}


int WriteBitToWav (int  value,
              FileInfo*  ptrFile)
{
     uint  ii, imax = ptrFile->bitLen ;
      int  error ;
    uchar  outb;

    if (TAPc > 0) return (ERR_OK);

    for ( ii = 0 ; ii < imax ; ++ii ) {

        if ( bitMirroring) { /* device or debug option */
            if (bit[value][ii] == '-') outb = AMP_HIGH;
            else if (bit[value][ii] == '+') outb = AMP_LOW;
            else outb = AMP_MID;
        }
        else { /* default */
            if (bit[value][ii] == '+') outb = AMP_HIGH;
            else if (bit[value][ii] == '-') outb = AMP_LOW;
            else outb = AMP_MID;
        }

        error = fputc (outb, ptrFile->ptrFd) ;
        if (error == EOF) {
            printf ("%s: Can't write in the file\n", argP) ;
            error = ERR_NOK ;
            break ;
        }
        else
            error = ERR_OK ;
    }
    return (error);
}


int WriteBitToEWav (int  value,
              FileInfo*  ptrFile)
{
     uint  ii, imax = bitLE[value] ;
      int  error ;
    uchar  outb;

    if (TAPc > 0) return (ERR_OK);

    for ( ii = 0 ; ii < imax ; ++ii ) {

        if ( bitMirroring) { /* device or debug option */
            if (bitE[value][ii] == '-') outb = AMP_HIGH_E;
            else if (bitE[value][ii] == '+') outb = AMP_LOW_E;
            else outb = AMP_MID;
        }
        else {
            if (bitE[value][ii] == '+') outb = AMP_HIGH_E;
            else if (bitE[value][ii] == '-') outb = AMP_LOW_E;
            else outb = AMP_MID;
        }

        error = fputc (outb, ptrFile->ptrFd) ;
        if (error == EOF) {
            printf ("%s: Can't write in the file\n", argP) ;
            error = ERR_NOK ;
            break ;
        }
        else
            error = ERR_OK ;
    }
    return (error);
}


int WriteSyncToWav (ulong  nbSync,
                FileInfo*  ptrFile)
{
    ulong  ii ;
    int    error = ERR_OK ;

    if (TAPc > 0) return (ERR_OK);

    do {
        /* Write the Synchro patern */
        for ( ii = 0 ; ii < nbSync ; ++ii ) {
            error = WriteBitToWav (1, ptrFile) ;
            if (error != ERR_OK) break ;
        }
        if (error != ERR_OK) break ;

    } while (0) ;
    return (error);
}


int WriteSyncToEWav (ulong  nbSync, /* sync bits */
                     ulong  nbSyncS,/* sync bits for space */
                     ulong  nbSync2,/* long/short intro bits */
                 FileInfo*  ptrFile)
{
    ulong  ii ;
    ulong  nbSyncSmin = 5000 ;        /* Minimal sync bits for space */
    int    error = ERR_OK ;

    static uint SyncCnt = 0 ;       /* 0: Header, 1...n: data blocks */

    if (TAPc > 0) return (ERR_OK);

    do {
        if (nbSync2 != SYNC_E_HEAD || pcgrpId == GRP_E || (ptrFile->debug & (SYNCL_TRM | SYNCL_STD))>0 ) {
            /* for following blocks write space, not for first G/16 */

            if (nbSync2 != SYNC_E_HEAD) { /* not before first header block */
                ++SyncCnt ;
                /* Write the stop bit of the last block and pull down the level */
                error = WriteBitToEWav (3, ptrFile) ;
                if (error != ERR_OK) break ;
            }
            // SyncCnt == 0
            else if ((ptrFile->debug & (SYNCL_TRM | SYNCL_STD))>0)
                 nbSyncSmin = ptrFile->nbSync1 *8 ;         /* 8s leading space */
            else nbSyncSmin = ptrFile->nbSync1 *3 + 375 ;   /* E500S series with CE-126P 3s leading space */

            if (pcgrpId == GRP_G ) {
                if (SyncCnt == 1) {
                     if (Scnt<2 && (ptrFile->debug & (SYNCL_TRM | SYNCL_STD))>0)
                                nbSyncSmin = ptrFile->nbSync1 *2 + 375 ; /* Space after header:   2.00 sec + 1/8 */
                }
                else if (SyncCnt > 1) {
                     if (Scnt<2 && (ptrFile->debug & (SYNCL_TRM | SYNCL_STD))>0)
                                nbSyncSmin = ptrFile->nbSync1 *4 + 375 ; /* Space data block:     4.00 sec + 1/8 */
                }
            }
            else if (pcgrpId == GRP_16) {

                if (SyncCnt == 1) {                                                             /*first data*/
                     if (Scnt<2 && (ptrFile->debug & SYNCL_TRM)>0)
                                nbSyncSmin = ptrFile->nbSync1 *3 + 375 ; /* Spc H: 3s, TRef: 2s, +1 for Rmt off/on from wave */
                }
                else if (ptrFile->ident == IDENT_PC16_DAT && SyncCnt > 1 && SyncCnt % 2 == 0 ){ /*data field body*/
                     if (Scnt<2 && (ptrFile->debug & SYNCL_TRM)>0)
                                nbSyncSmin = ptrFile->nbSync1 *9 + 375 ; /* Spc Ds:9s, TRef: 8s, +1 for Rmt off/on + 1/8 */
                }
                else if (SyncCnt > 0) {                                                         /*data field head, ASC */
                     if (Scnt<2 && (ptrFile->debug & SYNCL_TRM)>0)
                                nbSyncSmin = ptrFile->nbSync1 *5 + 375 ; /* Spc D: 5s, TRef: 4s, +1 for Rmt off/on + 1/8 */
                }

                if (Scnt<1 && SyncCnt > 0 && (ptrFile->debug & (SYNCL_TRM | SYNCL_STD))>0 &&
                    nbSync < 11000) nbSync = 11000 ; /* set shorts for syncing after header */
            }
            /* (pcgrpId == GRP_E) */
            else if (ptrFile->ident == IDENT_E_ASC){                                /* 3.x sec GRP_E ASC*/
                if (SyncCnt == 1) {
                    if (Scnt<2 && (ptrFile->debug & (SYNCL_TRM | SYNCL_STD))>0)
                                nbSyncSmin = ptrFile->nbSync1 *3 + 375 ;
                }
                else if (SyncCnt > 0) {
                    if (Scnt<2 && (ptrFile->debug & SYNCL_TRM)>0)
                                nbSyncSmin = 11* ptrFile->nbSync1 /2 + 375 ;        /* 5.5 sec next block */
                }
            }
            else { // pcgrpId == GRP_E and not E_ASC /
                if (SyncCnt > 0) {
                    if (Scnt<2 && (ptrFile->debug & (SYNCL_TRM | SYNCL_STD))>0)
                                nbSyncSmin = ptrFile->nbSync1 *4 ;
                }
                /* if (Scnt != 2 && SyncCnt > 0) {
                    nbSyncSmin = 11* ptrFile->nbSync1 /4 ;           // min. 2.75 sec, > 8000 + tolerance /
                    // if (nbSyncSmin < nbSync ) nbSyncSmin = nbSync ;
                } */
            }
            if (nbSyncS < nbSyncSmin) nbSyncS = nbSyncSmin ;
            /* Write the space (silence) */
            for ( ii = 0 ; ii < nbSyncS ; ++ii ) {
                error = WriteBitToEWav (2, ptrFile) ;
                if (error != ERR_OK) break ;
            }
        }
        if (error != ERR_OK) break ;
        /* Write the Synchro pattern */
        for ( ii = 0 ; ii < nbSync ; ++ii ) {
            error = WriteBitToEWav (0, ptrFile) ;
            if (error != ERR_OK) break ;
        }
        if (error != ERR_OK) break ;

        /* Write the E block Synchro pattern */
        for ( ii = 0 ; ii < nbSync2 ; ++ii ) {
            error = WriteBitToEWav (1, ptrFile) ;
            if (error != ERR_OK) break ;
        }
        if (error != ERR_OK) break ;
        for ( ii = 0 ; ii < nbSync2 ; ++ii ) {
            error = WriteBitToEWav (0, ptrFile) ;
            if (error != ERR_OK) break ;
        }
        if (error != ERR_OK) break ;
        /* Write the E block start bit */
        error = WriteBitToEWav (1, ptrFile) ;

    } while (0) ;
    return (error);
}


int WriteQuaterToTap (uint  value,
                      FileInfo*  ptrFile)
{
      int  error ;

	do {
		error = fputc (0xF0 | value, ptrFile->ptrFd) ;
    	if (error == EOF) {
        	printf ("%s: Can't write in the file\n", argP) ;
        	error = ERR_NOK ;
        	break ;
    	}
    	else
        	error = ERR_OK ;
	} while (0);
    return (error);
}


int WriteQuaterToWav (uint   value,
                      uint   stopBits,
                 FileInfo*   ptrFile)
{
      uint  tmp ;
      uint  ii ;
       int  error ;

    if (TAPc > 0) return (WriteQuaterToTap (value, ptrFile));

    do {
        error = WriteBitToWav (0, ptrFile) ;
        if (error != ERR_OK) break ;

        for ( ii = 0 ; ii < 4 ; ++ii ) {
            tmp = 1 << ii ;
            if ( (value & tmp) == 0 )
                error = WriteBitToWav (0, ptrFile) ;
            else
                error = WriteBitToWav (1, ptrFile) ;

            if (error != ERR_OK) break ;
        }
        if (error != ERR_OK) break ;

        for ( ii = 0 ; ii < stopBits ; ++ii ) {
            error = WriteBitToWav (1, ptrFile) ;
            if (error != ERR_OK) break ;
        }
        if (error != ERR_OK) break ;

    } while (0) ;
    return (error);
}


int WriteByteToTap    ( ulong  value,
                        uchar  order,
                    FileInfo*  ptrFile)
{
  ulong  msq, lsq ;
    int  error ;

    do {
        if ( pcgrpId == IDENT_PC1500 ) {
            lsq = value & 0x0F ;
            msq = (value >> 4) & 0x0F ;

            if (order == ORDER_INV) {
                error = WriteQuaterToTap (lsq, ptrFile) ;
                if (error != ERR_OK) break ;
                error = WriteQuaterToTap (msq, ptrFile) ;
        	}
            else {
                error = WriteQuaterToTap (msq, ptrFile) ;
                if (error != ERR_OK) break ;
                error = WriteQuaterToTap (lsq, ptrFile) ;
            }
        }
        else {
            error = fputc (value, ptrFile->ptrFd) ;
            if (error == EOF) {
                printf ("%s: Can't write in the file\n", argP) ;
                error = ERR_NOK ;
                break ;
            }
            else
                error = ERR_OK ;
        }

	} while (0);
    return (error);
}


int WriteByteToEWav (ulong value,
                 FileInfo* ptrFile)
{
      uint  tmp ;
      uint  ii ;
      int  error ;

    do {
        /* write the start bit */
        error = WriteBitToEWav (1, ptrFile) ;
        if (error != ERR_OK) break ;

        tmp = 0x80 ;
        for ( ii = 0 ; ii < 8 ; ++ii ) {
            if ( (value & tmp) == 0 )
                error = WriteBitToEWav (0, ptrFile) ;
            else
                error = WriteBitToEWav (1, ptrFile) ;
            if (error != ERR_OK) break ;
            tmp >>= 1 ;
        }
        if (error != ERR_OK) break ;
        /* no stop bits */

    } while (0) ;
    return (error);
}


int WriteByteToWav (ulong  value,
                    uchar  order,
                    uchar  mode,
                FileInfo*  ptrFile)
{
      uint  lsq ;
      uint  msq ;
      int  error ;

    if (TAPc > 0) return (WriteByteToTap (value, order, ptrFile)) ;

    if (order == ORDER_E) return (WriteByteToEWav (value, ptrFile)) ;

    do {
        lsq = value & 0x0F ;
        msq = (value >> 4) & 0x0F ;
        if (order == ORDER_INV) {
            error = WriteQuaterToWav (lsq, Mode[mode].stopb1, ptrFile) ;
            if (error != ERR_OK) break ;
            error = WriteQuaterToWav (msq, Mode[mode].stopb2, ptrFile) ;
            if (error != ERR_OK) break ;
        }
        else {
            error = WriteQuaterToWav (msq, Mode[mode].stopb1, ptrFile) ;
            if (error != ERR_OK) break ;
            error = WriteQuaterToWav (lsq, Mode[mode].stopb2, ptrFile) ;
            if (error != ERR_OK) break ;
        }

    } while (0) ;
    return (error);
}

/*
int WriteIdentToB22Wav (ulong  value, FileInfo*  ptrFile)
replaced by WriteQuater
*/
/*  WriteByteToB22Wav*/
int WriteByteTo15Wav (ulong value, FileInfo* ptrFile)
{
    return( WriteByteToWav(value, ORDER_INV, ptrFile->mode, ptrFile)) ;
}

/*
int WriteByteToB19Wav (ulong  value, FileInfo*  ptrFile)
replaced by WriteByte
int WriteByteToB16Wav (ulong  value, FileInfo*  ptrFile)
replaced by WriteByte
int WriteByteToB13Wav (ulong  value, FileInfo*  ptrFile)
replaced by WriteByte
int WriteByteToDataWav (ulong  value,
                       FileInfo*  ptrFile)
    byte was swapped before because of uniform Checksum calculation
replaced by WriteByte
*/

int CheckSumB1 (  ulong  Byte,
              FileInfo*  ptrFile)
{
    ushort sum ;

    /* Update the checksum */
        sum = ptrFile->sum + ((Byte & 0xF0) >> 4) ;
        if (sum > 0xFF) {
            ++sum ;
            sum &= 0xFF ;
            }
        ptrFile->sum = (sum + (Byte & 0x0F)) & 0xFF ;

    return (0);
}


int CheckSumE (  ulong  Byte,
              FileInfo*  ptrFile)
{
    uint tmp, ii ;

    /* Update the checksum */
    tmp = 0x80 ;
    for ( ii = 0 ; ii < 8 ; ++ii, tmp >>= 1 )
        if ( (Byte & tmp) != 0 ) ++ ptrFile->sum ;

    return (0);
}


int WriteByteSumToWav (ulong  value,
                       uchar  order,
                       uchar  mode,
                       FileInfo*  ptrFile)
{
    int  error ;

    do {

        if ( (ptrFile->debug & 0x0040) > 0) {
            printf(" %02X", (uchar) value);
            if ( ptrFile->total %0x100 == 0xFF ) printf("\n");
        }
        error = WriteByteToWav (value, order, mode, ptrFile) ;
        if (error != ERR_OK) break ;

        if (mode == MODE_B22) ptrFile->sum += value ;
        else if (mode == MODE_B9) CheckSumE (value, ptrFile) ;
        else CheckSumB1 (value, ptrFile) ;

        ++ptrFile->count ;
        ++ptrFile->total ;

        switch (mode) {
        case MODE_B22 :
            if ( ptrFile->count >= BLK_OLD ) {

                if ( (ptrFile->debug & 0x0040) > 0 )
                    printf(" (%04X)", (uint) ptrFile->sum);

                /* Write the checksum */
                error = WriteByteToWav (ptrFile->sum >> 8 & 0xFF, order, mode, ptrFile) ;
                if (error != ERR_OK) break ;
                error = WriteByteToWav (ptrFile->sum & 0xFF, order, mode, ptrFile) ;
                if (error != ERR_OK) break ;

                ptrFile->count = 0 ;
                ptrFile->sum   = 0 ;
            }
            break ;

        case MODE_B21 :
        case MODE_B20 :
        case MODE_B19 :
            if ( (ptrFile->count % BLK_OLD_SUM) == 0) {

                if ( (ptrFile->debug & 0x0040) > 0 )
                    printf(" (%02X)", (uchar) ptrFile->sum);

                /* Write the checksum */
                error = WriteByteToWav (ptrFile->sum, order, mode, ptrFile) ;
                if (error != ERR_OK) break ;

                if ( ptrFile->count >= BLK_OLD ) {
                    ptrFile->count = 0 ;
                    ptrFile->sum   = 0 ;
                    // if (pcgrpId==IDENT_PC1211) error = WriteSyncToWav (1803, ptrFile) ; //DATA not
                    if (ptrFile->ident == IDENT_PC1211) /* default 1803 bits, data not */
                        error = WriteSyncToWav (ptrFile->nbSync, ptrFile) ;
                }
            }
            break ;

        case MODE_B17 :
        case MODE_B16 :
        case MODE_B15 :
            if ( ptrFile->count >= BLK_OLD_SUM) {

                if ( (ptrFile->debug & 0x0040) > 0 )
                    printf(" (%02X)", (uchar) ptrFile->sum);

                /* Write the checksum */
                error = WriteByteToWav (ptrFile->sum, order, mode, ptrFile) ;
                if (error != ERR_OK) break ;

                ptrFile->count = 0 ;
                ptrFile->sum   = 0 ;
            }
            break ;

        case MODE_B14 :
        case MODE_B13 :
            if ( ptrFile->count >= BLK_NEW) {

                if ( (ptrFile->debug & 0x0040) > 0 )
                    printf(" (%02X)", (uchar) ptrFile->sum);

                /* Write the checksum */
                error = WriteByteToWav (ptrFile->sum, order, mode, ptrFile) ;
                if (error != ERR_OK) break ;

                ptrFile->count = 0 ;
                ptrFile->sum   = 0 ;
            }
            break ;

        case MODE_B9 : /* PC-E/G/1600 */
            if ( ptrFile->count >= ptrFile->block_len ) {

                if ( (ptrFile->debug & 0x0040) > 0 )
                    printf(" (%04X)", (uint) ptrFile->sum & 0xFFFF);

                /* Write the checksum */
                error = WriteByteToWav (ptrFile->sum >> 8 & 0xFF, order, mode, ptrFile) ;
                if (error != ERR_OK) break ;
                error = WriteByteToWav (ptrFile->sum & 0xFF, order, mode, ptrFile) ;
                if (error != ERR_OK) break ;

                ptrFile->count = 0 ;
                ptrFile->sum   = 0 ;
            }
            break ;

        default :
            printf ("%s: Unknown Mode\n", argP) ;
            break ;
        }

    } while (0) ;
    return (error);
}
/*  WriteByteSumToB22Wav */
int WriteByteSumTo15Wav (ulong value, FileInfo* ptrFile)
{
    return( WriteByteSumToWav(value, ORDER_INV, ptrFile->mode, ptrFile)) ;
}

int WriteByteSumTo156Wav (ulong value, FileInfo* ptrFile)
{
    uchar order;
    if (ptrFile->mode == MODE_B9) order = ORDER_E ; else order = ORDER_INV ;

    return( WriteByteSumToWav(value, order, ptrFile->mode, ptrFile)) ;
}

/* int WriteByteSumToB19Wav (ulong value, FileInfo* ptrFile)
{
    return( WriteByteSumToWav(value, ORDER_STD, MODE_B19, ptrFile)) ;
} */
/* int WriteByteSumToB16Wav (ulong value, FileInfo* ptrFile)
{
    return( WriteByteSumToWav(value, ORDER_STD, MODE_B16, ptrFile)) ;
} */
/* int WriteByteSumToB13Wav (ulong value, FileInfo* ptrFile)
{
    return( WriteByteSumToWav(value, ORDER_STD, MODE_B13, ptrFile)) ;
} */

/* Data Blocks and Data Header are not swapped */
/* Byte swapped before checksum with uniform calculation */

ulong SwapByte (ulong byte)
{
    return ( (byte >> 4) + (byte << 4 & 0xF0) );
}


int WriteByteSumToDataWav (ulong  value, uchar  mode,
                          FileInfo*  ptrFile)
{
      int  error ;

    do {
        if ( (ptrFile->debug & 0x0040) > 0 )
            printf(" %02X", (uchar) value);

        /* byte swapped before because of uniform checksum calculation */
        value = SwapByte(value) ;
        /* Write the byte */
        error = WriteByteToWav (value, ORDER_STD, mode, ptrFile) ;
        if (error != ERR_OK) break ;

        /* Update the checksum */
        CheckSumB1 (value, ptrFile) ;

        /* Update an check the byte counter */
        if ( ++ptrFile->count >= BLK_OLD_SUM) {

            /* Write the checksum */
            if ( (ptrFile->debug & 0x0040) > 0 )
                printf(" (%02X)", (uchar) ptrFile->sum);

            error = WriteByteToWav (ptrFile->sum, ORDER_STD, mode, ptrFile) ;
            if (error != ERR_OK) break ;

            ptrFile->count = 0 ;
            ptrFile->sum = 0 ;
        }

    } while (0) ;
    return (error);
}

/* File name and header for PC-1500 */
/*  WriteHeadToB22Wav */
int WriteHeadTo15Wav (char*  ptrName,
                       ulong  addr,		// address == 0x40C5 ?
                       ulong  eaddr, 	// entry address
                       ulong  size,  	// buffer size
                       uchar  type,
                       FileInfo*  ptrFile)
{
     uint  ii ;
    ulong  len ;
    ulong  tmpL ;
     char  tmpS[20] ;
      int  error ;

    do {
        /* Search the length */
        tmpL = strlen (ptrName) ;
        if (tmpL > 16)
            tmpL = 16 ;

        for ( ii = 0 ; ii < tmpL ; ++ii )
            tmpS[ii] = ptrName[ii] ;
        for ( ii = tmpL ; ii < 16 ; ++ii )
            tmpS[ii] = 0 ;

        if (Qcnt == 0) printf ("Save name    : %s\n", tmpS) ;

        /* Write the Header */
        ptrFile->count = 0 ;
        ptrFile->sum   = 0 ;
        for ( ii = 0x10 ; ii < 0x18 ; ++ii ) {
            error = WriteByteSumTo15Wav (ii, ptrFile) ;
            if (error != ERR_OK) break ;
        }
        if (error != ERR_OK) break ;

        /* Write the Sub-Ident */
        if (type == TYPE_DAT)
            tmpL = 0x04 ;
        else if (type == TYPE_RSV)
            tmpL = 0x02 ;
        else if (type == TYPE_BIN)
            tmpL = 0x00 ;
        else // TYPE_IMG
            tmpL = 0x01 ;

        error = WriteByteSumTo15Wav (tmpL, ptrFile) ;
        if (error != ERR_OK) break ;

        /* Write the Name */
        for ( ii = 0 ; ii < 16 ; ++ii ) {
            error = WriteByteSumTo15Wav (tmpS[ii], ptrFile) ;
            if (error != ERR_OK) break ;
        }
        if (error != ERR_OK) break ;

        /* Write 9 null bytes */
        for ( ii = 0 ; ii < 9 ; ++ii ) {
            error = WriteByteSumTo15Wav (0, ptrFile) ;
            if (error != ERR_OK) break ;
        }
        if (error != ERR_OK) break ;

        /* Write the address */
        if (type==TYPE_IMG && addr==0) addr = 0xC5 ; /* RSV length before BASIC program */

        tmpL = (addr >> 8) & 0xFF ;
        error = WriteByteSumTo15Wav (tmpL, ptrFile) ;
        if (error != ERR_OK) break ;

        tmpL = addr & 0xFF ;
        error = WriteByteSumTo15Wav (tmpL, ptrFile) ;
        if (error != ERR_OK) break ;

        /* Write the Buffer Size */
        if (type == TYPE_DAT)
            len = 0 ;
        else if (type == TYPE_BIN || type == TYPE_RSV)
            len = size - 1 ;
        else
            len = size ;

        fprintf(stderr,"wonder buffer size = %ld\n", size);

        tmpL = (len >> 8) & 0xFF ;
        error = WriteByteSumTo15Wav (tmpL, ptrFile) ;
        if (error != ERR_OK) break ;

        tmpL = len & 0xFF ;
        error = WriteByteSumTo15Wav (tmpL, ptrFile) ;
        if (error != ERR_OK) break ;

        /* Write the entry address */
        if (type == TYPE_BIN) {
            // if (Acnt<2) eaddr = 0xFFFF ;
            tmpL = (eaddr >> 8) & 0xFF ;
            error = WriteByteSumTo15Wav (tmpL, ptrFile) ;
            if (error != ERR_OK) break ;

            tmpL = eaddr & 0xFF ;
            error = WriteByteSumTo15Wav (tmpL, ptrFile) ;
            if (error != ERR_OK) break ;
        }
        else {
            tmpL = 0x00 ;

            error = WriteByteSumTo15Wav (tmpL, ptrFile) ;
            if (error != ERR_OK) break ;

            error = WriteByteSumTo15Wav (tmpL, ptrFile) ;
            if (error != ERR_OK) break ;
        }

        /* Write the checksum */
        tmpL = (ptrFile->sum >> 8) & 0xFF ;
        error = WriteByteTo15Wav (tmpL, ptrFile) ;
        if (error != ERR_OK) break ;

        tmpL = ptrFile->sum & 0xFF ;
        error = WriteByteTo15Wav (tmpL, ptrFile) ;
        if (error != ERR_OK) break ;

        if ( (ptrFile->debug & 0x0040) > 0 )
            printf(" (%04X)", (uint) ptrFile->sum);

        if (type == TYPE_DAT)
            error = WriteSyncToWav (ptrFile->nbSync, ptrFile) ;
        else
            error = WriteSyncToWav (75, ptrFile) ;
        if (error != ERR_OK) break ;

        ptrFile->count = 0 ;
        ptrFile->sum   = 0 ;

        if ( (ptrFile->debug & 0x00C0) > 0) printf("\n") ;

    } while (0) ;
    return (error);
}


/* File name and header for PC-E Series and newer */
int WriteHeadToEWav (char*  ptrName,
                     char*  ptrDstExt,
                     ulong  addr,
                     ulong  eaddr,
                     ulong  size,
                     ulong  nbSync,
                     ulong  nbSyncS,
                     uchar  type,
                 FileInfo*  ptrFile)
{
    ulong  haddr ; // , eaddr = 0xFFFFFF
   ushort  ident ;
    uchar  mode ;
     uint  ii, imax ;
    ulong  len ;
    ulong  tmpL ;
    ulong  tmpH[20] ;
     char  tmpS[20] ;
      int  error ;

    ident = ptrFile->ident ;
    mode  = ptrFile->mode_h ;

    ptrFile->block_len = BLK_E_HEAD ;
    ptrFile->count = 0 ;
    ptrFile->sum   = 0 ;

    if (ident != IDENT_E_BIN) eaddr = 0; /* 2. separate bin2wav parameter -addr, no entry: 0xFFFFFF */

    do {
        /*write the tap code */
        error = WriteByteSumToWav ( (ulong) ident, ORDER_E, mode, ptrFile) ;
        if (error != ERR_OK) break ;

        /* Search the name length */
        if ( pcgrpId == GRP_16) imax = 16 ; else imax = 9 ;
        tmpL = strlen (ptrName) ;
        if (tmpL > imax) tmpL = imax ;
        for ( ii = 0 ; ii < tmpL ; ++ii )
            tmpS[ii] = ptrName[ii] ;
        if ( pcgrpId == GRP_16) {
            for ( ii = tmpL ; ii < 16 ; ++ii ) tmpS[ii] = 0 ;
        }
        else {
            for ( ii = tmpL ; ii < 16 ; ++ii ) tmpS[ii] = 0x20 ;
        }
        /* ASCII: BAS, TXT */
        if (type == TYPE_BAS && pcgrpId != GRP_G)  for ( ii = 0 ; ii < 3 ; ++ii ) tmpS[13 + ii] = ptrDstExt[ii] ;
        if (type == TYPE_ASC && pcgrpId == GRP_16) for ( ii = 0 ; ii < 3 ; ++ii ) tmpS[13 + ii] = 0x20 ;

        /* Write the Name */
        tmpS[16] = 0 ;
        if (Qcnt == 0) printf ("Save name    : %s\n", tmpS) ;
        for ( ii = 0 ; ii < 16 ; ++ii ) {
            error = WriteByteSumToWav (tmpS[ii], ORDER_E, mode, ptrFile) ;
            if (error != ERR_OK) break ;
        }
        if (error != ERR_OK) break ;

        if ( ident == IDENT_PC16_CAS || (ident == IDENT_E_ASC && pcgrpId == GRP_16) )
             error = WriteByteSumToWav (0, ORDER_E, mode, ptrFile) ;
        else error = WriteByteSumToWav (0x0D, ORDER_E, mode, ptrFile) ;
        if (error != ERR_OK) break ;
        /* Write the Header */

        /* Add Length of BAS Header with start mark + end mark */
        if (ident == IDENT_E_BAS) {
            if ( pcgrpId == GRP_E)
                len = size + 19 + 1 ;
            else if ( pcgrpId == GRP_G)
                len = size + 12 + 0 ;
            else // pcgrpId == GRP_16
                len = size ;
        }
        else if (ident == IDENT_E_ASC || ident == IDENT_PC16_CAS || ident == IDENT_PC16_DAT)
            len = 0 ;
        else if (ident == IDENT_E_BIN)
            len = size ;
        else
            len = size ;

        /* Write the Buffer Size of the Data Block */
        tmpL = len & 0xFF ;
        error = WriteByteSumToWav (tmpL, ORDER_E, mode, ptrFile) ;
        if (error != ERR_OK) break ;

        tmpL = (len >> 8) & 0xFF ;
        error = WriteByteSumToWav (tmpL, ORDER_E, mode, ptrFile) ;
        if (error != ERR_OK) break ;

        tmpL = addr & 0xFF ;
        error = WriteByteSumToWav (tmpL, ORDER_E, mode, ptrFile) ;
        if (error != ERR_OK) break ;

        tmpL = (addr >> 8) & 0xFF ;
        error = WriteByteSumToWav (tmpL, ORDER_E, mode, ptrFile) ;
        if (error != ERR_OK) break ;

        tmpL = eaddr & 0xFF ;
        error = WriteByteSumToWav (tmpL, ORDER_E, mode, ptrFile) ;
        if (error != ERR_OK) break ;

        tmpL = (eaddr >> 8) & 0xFF ;
        error = WriteByteSumToWav (tmpL, ORDER_E, mode, ptrFile) ;
        if (error != ERR_OK) break ;

        /* Write the Sub-Ident (mode 2) */
        if (pcgrpId == GRP_16 && type == TYPE_RSV ) {
            tmpL = 0x02 ;
            ptrFile->ident = IDENT_E_BIN ;
        }
        else if ( ident == IDENT_E_BAS ||
                 (pcgrpId == GRP_G && type == TYPE_BAS ) )
            tmpL = 0x01 ;
        else if ((ident == IDENT_E_ASC && pcgrpId == GRP_E) ||
                  ident == IDENT_PC16_DAT)
            tmpL = 0x04 ; // GRP_E, IDENT_PC16_DAT
        else   /* All Binary or Data GRP_G */
            tmpL = 0x00 ;
        error = WriteByteSumToWav (tmpL, ORDER_E, mode, ptrFile) ;
        if (error != ERR_OK) break ;

        for ( ii = 0 ; ii < 4 ; ++ii ) { /* Date+Time PC-1600: Mon Day Hour Min */
            if ( ii < 2 && pcgrpId == GRP_16) /* write a valid date */
                  error = WriteByteSumToWav (1, ORDER_E, mode, ptrFile) ;
            else  error = WriteByteSumToWav (0, ORDER_E, mode, ptrFile) ;
            if (error != ERR_OK) break ;
        }
        if (error != ERR_OK) break ;

        tmpL = (len >> 16) & 0xFF ;
        error = WriteByteSumToWav (tmpL, ORDER_E, mode, ptrFile) ;
        if (error != ERR_OK) break ;

        tmpL = (addr >> 16) & 0xFF ;
        error = WriteByteSumToWav (tmpL, ORDER_E, mode, ptrFile) ;
        if (error != ERR_OK) break ;

        tmpL = (eaddr >> 16) & 0xFF ;
        error = WriteByteSumToWav (tmpL, ORDER_E, mode, ptrFile) ;
        if (error != ERR_OK) break ;

        for ( ii = 0 ; ii < 16 ; ++ii ) {
            error = WriteByteSumToWav (0, ORDER_E, mode, ptrFile) ;
            if (error != ERR_OK) break ;
        }
        if (error != ERR_OK) break ;


        /* finish header block and prepare next block */
        mode  = ptrFile->mode ;

        /* Write the block stop bit, space, sync and start bit */
        error = WriteSyncToEWav (nbSync, nbSyncS, SYNC_E_DATA, ptrFile) ;
        if (error != ERR_OK) break ;

        if (ident == IDENT_E_ASC || ident == IDENT_PC16_CAS) len = BLK_E_DAT ;
        else if (ident == IDENT_PC16_DAT) len = DATA_HEAD_LEN ;
        ptrFile->block_len = len ;
        ptrFile->count = 0 ;
        ptrFile->sum   = 0 ;

        if (ident == IDENT_E_BAS && pcgrpId != GRP_16) {
        /* write the 2.part of the internal BASIC file header */

            for ( ii = 0 ; ii < 20 ; ++ii ) tmpH[ii] = 0 ;
            tmpH[0] = 0xFF ;

            if (pcgrpId == GRP_E) {
                tmpH[4] = 0x34 ; // '4'

                haddr = len + 33 ;
                tmpH[7] = haddr & 0xFF ;
                tmpH[8] = haddr >> 8 & 0xFF ;
                tmpH[9] = haddr >> 16 & 0xFF ;

                tmpH[18] = 0x0D ;
                imax = 19;
            }
            else  { // pcgrpId == GRP_G
                tmpH[2] = 0x03 ;
                imax = 12;
            }
            if (type == TYPE_TXT) tmpH[2] = 0x08 ;

            for ( ii = 0 ; ii < imax ; ++ii ) {
                error = WriteByteSumToWav (tmpH[ii], ORDER_E, mode, ptrFile) ;
                if (error != ERR_OK) break ;
            }
            if (error != ERR_OK) break ;
        }
        if ( (ptrFile->debug & 0x00C0) > 0) printf("\n") ;

    } while (0) ;
    return (error);
}


/* File name for New and Old BASIC */
int WriteSaveNameToWav (char*  ptrName,
                        uchar  mode,
                    FileInfo*  ptrFile)
{
     uint  ii ;
    ulong  byte ;
    ulong  tmpL ;
//    char*  ptrDot ;
     char  tmpS[10] ;
      int  error ;

    do {
        /* Uppercase the name is done in main if needed */
        tmpL = strlen (ptrName) ;
        if (tmpL > 7)
            tmpL = 7 ;

        for ( ii = 0 ; ii < tmpL ; ++ii )
            tmpS[ii] = ptrName[ii] ;
        tmpS[tmpL] = 0 ;
        if (Qcnt == 0) printf ("Save name    : %s\n", tmpS) ;

        tmpL = 7 - tmpL ;
        for ( ii = 0 ; ii < tmpL ; ++ii )
            tmpS[ii] = 0 ;

        for ( ii = tmpL ; ii < 7 ; ++ii ) {
            byte = (ulong) ptrName[6 - ii] ;

            switch (mode) {
            case MODE_B19 :
            case MODE_B20 :

                if (byte < 0x80)
                    byte = CodeOld[byte] ;
                else
                    byte = CodeOld[0] ;
                break ;

            default :

                if (byte >= 0x80)
                    byte = 0x20 ;
                break ;
            }
            tmpS[ii] = (char) SwapByte(byte) ;
        }
        tmpS[7] = 0x5F ;

        /* Write the Name */
        ptrFile->count = 0 ;
        ptrFile->sum   = 0 ;
        for ( ii = 0 ; ii < 8 ; ++ii ) {
            error = WriteByteSumToWav (tmpS[ii], ORDER_STD, mode, ptrFile) ;
            if (error != ERR_OK) break ;
        }
        if ( (ptrFile->debug & 0x0040) > 0 ) printf(":Name - Bytes was printed swapped.\n");

        if (ptrFile->ident == IDENT_PC1211)
            error = WriteSyncToWav (151, ptrFile) ;
        else if (ptrFile->ident == IDENT_PC121_DAT)
            error = WriteSyncToWav (111, ptrFile) ;

        ptrFile->count = 0 ;
        ptrFile->sum   = 0 ;

    } while (0) ;
    return (error);
}

/*
int WriteSaveNameToB19Wav (char*  ptrName, FileInfo*  ptrFile)
replaced by WriteSaveNameToWav

int WriteSaveNameToB16Wav (char*  ptrName, FileInfo*  ptrFile)
replaced by WriteSaveNameToWav
*/

/* Head of Binary Data for New and Old series */
int WriteHeadToBinWav (ulong  addr,
                       ulong  size,
                       uchar  mode,
                   FileInfo*  ptrFile)
{
      int  ii ;
    ulong  len ;
    ulong  tmpL ;
      int  error ;

    do {
        if (Qcnt == 0)
	{
		printf ("Start Address: 0x%04X\n", (uint) addr);
        printf ("End   Address: 0x%04X, Length: %d bytes\n", (uint) (addr + size -1), (uint) size);
	}

        ptrFile->count = 0 ;
        ptrFile->sum   = 0 ;
        for ( ii = 0 ; ii < 4 ; ++ii ) {
            error = WriteByteSumToWav (0, ORDER_STD, mode, ptrFile) ;
            if (error != ERR_OK) break ;
        }

        /* Write the Addresse */
        tmpL = ((addr >> 4) & 0xF0) + (addr >> 12) ;
        error = WriteByteSumToWav (tmpL, ORDER_STD, mode, ptrFile) ;
        if (error != ERR_OK) break ;

        tmpL = ((addr << 4) & 0xF0) + ((addr >> 4) & 0x0F) ;
        error = WriteByteSumToWav (tmpL, ORDER_STD, mode, ptrFile) ;
        if (error != ERR_OK) break ;

        /* Write the Length */
        len = size - 1 ;
        tmpL = ((len >> 4) & 0xF0) + (len >> 12) ;
        error = WriteByteSumToWav (tmpL, ORDER_STD, mode, ptrFile) ;
        if (error != ERR_OK) break ;

        tmpL = ((len << 4) & 0xF0) + ((len >> 4) & 0x0F) ;
        error = WriteByteSumToWav (tmpL, ORDER_STD, mode, ptrFile) ;
        if (error != ERR_OK) break ;

        ptrFile->count = 0 ;
        ptrFile->sum   = 0 ;

    } while (0) ;
    return (error);
}

/*
int WriteHeadToB19BinWav (ulong  addr, ulong  size, FileInfo*  ptrFile)
replaced by WriteHeadToBinWav

int WriteHeadToB16BinWav (ulong  addr, ulong  size, FileInfo*  ptrFile)
replaced by WriteHeadToBinWav
*/

int DetectDataType ( uchar* ptrItemType,
                      uint  tmpDim,
                     FILE*  srcFd)
 {
    uint  ii ;
     int  inVal, inVal2 ;
     int  error = ERR_OK ;

    *ptrItemType = DATA_UNKNOWN ;

    for ( ii = 0 ; ii < tmpDim ; ii += DATA_STD_LEN ){
        if ( ii > 0 ) {
            error = fseek (srcFd, DATA_STD_LEN-3, SEEK_CUR) ; /* Begin of next item */
            if (error != ERR_OK) break ;
        }
        inVal = fgetc (srcFd) ;
        if (inVal == EOF) break ;

        if (inVal > 0x7F && inVal < 0xA0 ) { /* isNumeric */
            *ptrItemType = DATA_NUM ;
            break ;
        }
        else if (inVal > 0x0F && inVal < 0x90 ) { /* isString */
            *ptrItemType = DATA_STR ;
            break ;
        }
        inVal2 = fgetc (srcFd) ;
        if (inVal2 == EOF) break ;
        if (inVal == 0 && inVal2 > 0 ) { /* isNumeric with Exp 0X */
            *ptrItemType = DATA_NUM ;
            break ;
        }
        inVal2 = fgetc (srcFd) ;
        if (inVal2 == EOF) break ;
        if (inVal == 0 && inVal2 > 0 ) { /* isNumeric with Exp 0X */
            *ptrItemType = DATA_NUM ;
            break ;
        }
    }
    if (error != ERR_OK) {
        printf ("\n%s:DetectData8 - Can't seek/read the source file\n", argP) ;
        error = ERR_FILE ;
    }
    return (error);
}

int WriteHeadToDataWav (
                     TypeInfo*  ptrSrcHead,
                     TypeInfo*  ptrDstHead,
                        uchar*  ptrItemLen, /* Real length of an item */
                        uchar*  ptrItemType,
                        ulong*  ptrPosEnd,  /* will set to point to the last byte of the data block */
                        ulong*  ptrNbByte,  /*Number of bytes to write*/
                     FileInfo*  ptrFile,
                         FILE*  srcFd)
{
       ulong  length = 0 ;      /* Length of defined variable block or ident of fixed variable block */
       ulong  tmpH, tmpL ;
      // fpos_t  position ;
        long  position ;
         int  inVal ;
         int  error = ERR_OK ;

    do {
        /* Read the length of data block */
        inVal = fgetc (srcFd) ;
        if (inVal == EOF) break ;
        if (SHCc > 0) inVal = SwapByte(inVal);
        tmpH = (uint)inVal ;
        if ( (ptrFile->debug & 0x00C0) == 0x0080 )
            printf(" Length:%02X", (uchar) inVal) ; /* Data block LenH or type */

        inVal = fgetc (srcFd) ;
        if (inVal == EOF) break ;
        if (SHCc > 0) inVal = SwapByte(inVal);
        tmpL = (uint)inVal ;
        if ( (ptrFile->debug & 0x00C0) == 0x0080 )
            printf("%02X", (uchar) inVal) ;     /* Data block LenL */

        length = ((tmpH << 8) + tmpL) ;
        ptrSrcHead->length = length ;

//        fgetpos(srcFd, &position) ;
        position = ftell( srcFd ) ;
        if (position <= 0) {
            printf ("\n%s:datahead - Can't get position in the source file\n", argP) ;
            return ( ERR_FILE ) ;
        }

        if (length == DATA_VARIABLE) {  /* Fixed variables, block of undefined type and length, itemLen=8 */
            *ptrPosEnd = (ulong) position + ( DATA_HEAD_LEN -2 + 0xFF * DATA_STD_LEN ); /* set to max: A(1)...A(255) */
            /* ConvertBin will find the mark DATA_EOF at block end */
        }
        else {                                    /* Array or simple variables, with defined header block */
            *ptrPosEnd = (ulong) position + length -1 ;   /* Last Data Byte, if not fixed variables block */
        }
        if ( (ptrFile->debug & 0x00C0) == 0x0080 )
            printf(" EOB:%d ", (uint) *ptrPosEnd) ;   /* Data End of block */

        /* Write the length and variable head */
        if (Qcnt == 0) {
            if (length == DATA_VARIABLE)
                printf (" Variable length block") ;
            else
                printf (" Data block, length: %d bytes\n", (uint) length + 2) ;
        }

        if (ptrFile->ident != IDENT_PC121_DAT) ptrFile->count = ( BLK_OLD_SUM - DATA_HEAD_LEN ) ;	/* 8-5= 3 bytes less for checksum */
        else ptrFile->count = 0 ;
        ptrFile->sum   = 0 ;

        /* Read and write dim1, dim2, itemLen, 6. Byte is NOT a var type but is
           also a checksum that is included by WAV2BIN 1.5.0 into var data block */

            inVal = fgetc (srcFd) ;
            if (inVal == EOF) break ;
            if (SHCc > 0) inVal = SwapByte(inVal);
            ptrSrcHead->dim1 = (uint)inVal ;

            inVal = fgetc (srcFd) ;
            if (inVal == EOF) break ;
            if (SHCc > 0) inVal = SwapByte(inVal);
            ptrSrcHead->dim2 = (uint)inVal ;

            inVal = fgetc (srcFd) ;
            if (inVal == EOF) break ;
            if (SHCc > 0) inVal = SwapByte(inVal);
            ptrSrcHead->itemLen = (uint)inVal ;

            if (ptrSrcHead->itemLen == DATA_NUM_15 && /* numeric from PC-1500, itemLen=8 */
               ((ptrFile->debug & 0x4) > 0 || ptrFile->ident == IDENT_PC121_DAT )) { /* convert it to standard variable */
                if (Qcnt == 0) printf (" Convert of numeric data format from PC-1500 to Standard variable block\n") ;
                tmpH = DATA_VARIABLE >> 8 ;
                tmpL = DATA_VARIABLE & 0xFF ;
                ptrDstHead->length = DATA_VARIABLE ;
                ptrDstHead->dim1 = 0 ;
                ptrDstHead->dim2 = 0 ;
                ptrDstHead->itemLen = 0 ;
                -- ptrFile->total_diff ; /* block end mark not included in PC-1500 source but for total write counter */
            }
            else {
                ptrDstHead->length = ptrSrcHead->length ;
                ptrDstHead->dim1 = ptrSrcHead->dim1 ;
                ptrDstHead->dim2 = ptrSrcHead->dim2 ;
                if (ptrSrcHead->itemLen == DATA_NUM_15) {
                    if (Qcnt == 0) printf (" Convert of numeric data format from PC-1500 to PC-%lu\n", pcId) ;
                    ptrDstHead->itemLen = DATA_STD_LEN ;
                }
                else ptrDstHead->itemLen = ptrSrcHead->itemLen ;
            }
            if (ptrDstHead->length == DATA_VARIABLE )
                *ptrItemLen = DATA_STD_LEN ;               /* real length */
            else {
                *ptrItemLen = ptrDstHead->itemLen ;
                if ( length != (ptrDstHead->dim1 +1)* (ptrDstHead->dim2 +1)* (*ptrItemLen)+ (ulong) (DATA_HEAD_LEN -2))
                    printf (" Check the data block offset: %lu differs from length for DIM (%i,%i)*%i\n", length,
                            ptrDstHead->dim1, ptrDstHead->dim2, *ptrItemLen ) ;
                // ToDo More Tests with ptrFile->total_diff
                if ( (ulong) (position) + length + ((ptrFile->debug & DATA_W2B150)>0? 1:0) > *ptrNbByte + SHCc) {
                    printf (" Data variable block length %lu exceeds the end of file\n", length) ;
                    error = ERR_FMT ;
                }
            }
            if ( *ptrItemLen < 1 || *ptrItemLen > cVL ) {
                printf (" Data variable items length %i is not supported: %u\n", cVL, *ptrItemLen) ;
                error = ERR_FMT ;
            }
            if (error != ERR_OK) break ;

            //ToDo check double precision
            /* Item length 8 is undefined if numeric from from PC-1234 or string with length 8 from all PC */
            if (ptrSrcHead->itemLen == DATA_STD_LEN) { /* unknown if is string or numeric from PC-1234 */

                error = DetectDataType( ptrItemType, length - DATA_HEAD_LEN + 2, srcFd) ;
                if (error != ERR_OK) break ;

                error = fseek (srcFd, position -2 + DATA_HEAD_LEN, SEEK_SET) ; /*rewind to position in header */
                if (error != ERR_OK) {
                    printf ("\n%s:datahead-8 - Can't seek the file: %ld\n", argP, position) ;
                    error = ERR_FILE ;
                    break ;
                }
            }

            if (ptrFile->ident != IDENT_PC121_DAT) {
                error = WriteByteSumToDataWav (tmpH, MODE_B16, ptrFile) ;
                if (error != ERR_OK) break ;
                error = WriteByteSumToDataWav (tmpL, MODE_B16, ptrFile) ;
                if (error != ERR_OK) break ;

                if ( (ptrFile->debug & 0x00C0) == 0x0080 )
                    printf(" Dim1:%02X ", (uchar) ptrDstHead->dim1) ;
                error = WriteByteSumToDataWav (ptrDstHead->dim1, MODE_B16, ptrFile) ;
                if (error != ERR_OK) break ;

                if ( (ptrFile->debug & 0x00C0) == 0x0080 )
                    printf(" Dim2:%02X ", (uchar) ptrDstHead->dim2) ;
                error = WriteByteSumToDataWav (ptrDstHead->dim2, MODE_B16, ptrFile) ;
                if (error != ERR_OK) break ;

                if ( (ptrFile->debug & 0x00C0) == 0x0080 )
                    printf(" Item/length:%02X ", (uchar) ptrDstHead->itemLen) ;
                error = WriteByteSumToDataWav (ptrDstHead->itemLen, MODE_B16, ptrFile) ;
                if (error != ERR_OK) break ;

                if ( (ptrFile->debug & DATA_W2B150) > 0 ) { 	/* checksum was in Wav2Bin 1.5 binary Pos 6 */
                    ++ ptrFile->total_diff ; /* old checksum not written and not included in total write counter */
                    if (inVal != EOF) inVal = fgetc (srcFd) ;   /* checksum will be written by WriteByteSumTo automatically */
                }
            }
            if (inVal == EOF) break ;

    } while (0) ;
    if (error == ERR_OK && inVal == EOF ) error = ERR_NOK ;

    return (error);
}


int WriteHeadTo156DataWav (
                        TypeInfo*  ptrSrcHead,
                        TypeInfo*  ptrDstHead,
                           uchar*  ptrItemLen,
                           uchar*  ptrItemType,
                           ulong*  ptrPosEnd,
                           ulong*  ptrNbByte,
                        FileInfo*  ptrFile,
                            FILE*  srcFd)
{
       ulong  length = 0 ;      /* Length of defined variable block */
       ulong  tmpH, tmpL ;
       ushort dim1 ;
//      fpos_t  position ;
        long  position ;
         int  inVal;
         int  error = ERR_OK ;

    do {
        /* Read the length of data block */
        inVal = fgetc (srcFd) ;
        if (inVal == EOF) break ;
        if (SHCc > 0) inVal = SwapByte(inVal); /* from other series */
        tmpH = (uint)inVal ;
        if ( (ptrFile->debug & 0x00C0) == 0x0080 )
            printf(" LenHL:%02X", (uchar) inVal) ;  /* Data Block Len H */

        inVal = fgetc (srcFd) ;
        if (inVal == EOF) break ;
        if (SHCc > 0) inVal = SwapByte(inVal); /* from other series */
        tmpL = (uint)inVal ;
        if ( (ptrFile->debug & 0x00C0) == 0x0080 )
            printf("%02X", (uchar) inVal) ;         /* Data Block Len L */

        length = ((tmpH << 8) + tmpL) ;
        ptrSrcHead->length = length ;

//        fgetpos(srcFd, &position) ;
        position = ftell( srcFd ) ;
        if (position <= 0) {
            printf ("\n%s:datahead15 - Can't get position in the source file\n", argP) ;
            return ( ERR_FILE ) ;
        }

        if (length == DATA_VARIABLE) { /* Fixed variables from series PC-1234, block of undefined length, itemLen=8 */
            /*search from head+1 +8: for DATA_EOF */
            dim1 = 0 ;
            length = DATA_HEAD_LEN-2 ;
            error = fseek (srcFd, DATA_HEAD_LEN-2+1, SEEK_CUR) ; /* Begin of first item + 1 */
            if (error == ERR_OK) {
                    do {
                        length += DATA_STD_LEN ;
                        error = fseek (srcFd, DATA_STD_LEN-1, SEEK_CUR) ; /* Begin of next item */
                        if (error != ERR_OK) break ;
                        inVal = fgetc (srcFd) ;
                        if (inVal == EOF) break ;
                        if (SHCc > 0) inVal = SwapByte(inVal); /* from other series */
                        if (inVal == DATA_EOF ) break ;

                    } while ( dim1++ < 0xFF ) ;
                    if ( dim1 > 0xFF) {
                        printf ("\n%s:datahead15 - Illegal dimension or missing end mark of standard variable block\n", argP) ;
                        error = ERR_FMT ;
                        break ;
                    }
                    else ptrDstHead->dim1 = dim1 & 0xFF ;
            }
            if (error != ERR_OK) {
                    printf ("\n%s:datahead15 - Can't seek/read the source file\n", argP) ;
                    error = ERR_FILE ;
                    break ;
            }
            else {
                tmpH = length >> 8 ;
                tmpL = length & 0xFF ;
            }
            error = fseek (srcFd, position, SEEK_SET) ; /*rewind to position in header */
            if (error != ERR_OK) {
                printf ("\n%s:datahead15 - Can't seek the file: %ld\n", argP, (long) position) ;
                error = ERR_FILE ;
                break ;
            }
            /* Length calculated for conversion to array or PC-1500 numeric standard variables
               with defined header block */
        }
        ptrDstHead->length = length ;
        *ptrPosEnd = (ulong) position + length -1 ;     /* Last data byte */

        if ( (ptrFile->debug & 0x00C0) == 0x0080 )
            printf(" EOB:%d ", (uint) *ptrPosEnd) ;   /* Data End of block */

        /* Write the length and variable head */
        if (Qcnt == 0) printf (" Data block, length: %d bytes\n", (uint) length + 2) ;

        if ( ptrFile->ident == IDENT_PC16_DAT) {
            ptrFile->block_len = DATA_HEAD_LEN ;
            ptrFile->count = 0 ;
        }
        else ptrFile->count = ( BLK_OLD - DATA_HEAD_LEN ) ;	/* 80-5= 75 bytes less than B22Wav */
        ptrFile->sum   = 0 ;

        /* Read and write dim1, dim2, itemLen */

            inVal = fgetc (srcFd) ;
            if (inVal == EOF) break ;
            if (SHCc > 0) inVal = SwapByte(inVal); /* from other series */
            ptrSrcHead->dim1 = (uint)inVal ;

            inVal = fgetc (srcFd) ;
            if (inVal == EOF) break ;
            if (SHCc > 0) inVal = SwapByte(inVal); /* from other series */
            ptrSrcHead->dim2 = (uint)inVal ;

            inVal = fgetc (srcFd) ;
            if (inVal == EOF) break ;
            if (SHCc > 0) inVal = SwapByte(inVal); /* from other series */
            ptrSrcHead->itemLen = (uint)inVal ;

            if (ptrSrcHead->length == DATA_VARIABLE) { /* from PC-1234, num expected */
                ptrDstHead->dim2 = 0 ;
                ptrDstHead->itemLen = DATA_NUM_15 ;
                (*ptrPosEnd)++ ; /* End mark DATA_EOF included */
                if ( Qcnt==0) printf (" Convert numeric values of other standard variable format to PC-%lu\n", pcId) ;
            } // end if: data_variable
            else {
                ptrDstHead->dim1 = ptrSrcHead->dim1 ;
                ptrDstHead->dim2 = ptrSrcHead->dim2 ;
                //ToDo check double precision
                /* Item length 8 is undefined if numeric from from PC-1234 or string with length 8 from all PC */
                if (ptrSrcHead->itemLen == DATA_STD_LEN) { /* unknown if is string or numeric from PC-1234 */

                        error = DetectDataType( ptrItemType, length - DATA_HEAD_LEN + 2, srcFd) ;
                        if (error != ERR_OK) break ;

                        error = fseek (srcFd, position -2 + DATA_HEAD_LEN, SEEK_SET) ; /*rewind to position in header */
                        if (error != ERR_OK) {
                            printf ("\n%s:datahead15-8 - Can't seek the file: %ld\n", argP, position) ;
                            error = ERR_FILE ;
                            break ;
                        }
                        if  ( *ptrItemType == DATA_UNKNOWN ) { /* only empty items found */
                            if ((ptrFile->debug & 0x8) > 0 ) ptrDstHead->itemLen = DATA_NUM_15 ; /* define numeric 0-array from PC-1234 */
                            else ptrDstHead->itemLen = ptrSrcHead->itemLen ;              /* define 0-string data from PC-1500 or other */
                        }
                        else if (*ptrItemType == DATA_NUM) ptrDstHead->itemLen = DATA_NUM_15 ; /* Convert numeric data from other PC */
                        else ptrDstHead->itemLen = ptrSrcHead->itemLen ; /*DATA_STD_LEN*/

                        if ( ptrDstHead->itemLen == DATA_NUM_15 )
                            if ( Qcnt==0) printf (" Convert other numeric variable format to PC-%lu\n", pcId) ;

                    } // end if: DATA_STD_LEN
                    else ptrDstHead->itemLen = ptrSrcHead->itemLen ;   /* expect numeric data from PC-1500 or string data */
            } // end if: no data_variable

            if (ptrDstHead->itemLen == DATA_NUM_15 ) *ptrItemLen = DATA_STD_LEN ; /* real length */
            else *ptrItemLen = ptrDstHead->itemLen ;
            if ( length != (ptrDstHead->dim1 +1)* (ptrDstHead->dim2 +1)* (*ptrItemLen)+ (ulong) (DATA_HEAD_LEN -2))
                    printf (" Check the data block offset: %lu differs from length for DIM (%i,%i)*%i\n", length,
                            ptrDstHead->dim1, ptrDstHead->dim2, *ptrItemLen ) ;
            //ToDo More tests, -1 or not
            if ( (ulong) position + length > *ptrNbByte + SHCc ) {
                printf (" Data variable block length %lu exceeds the dist %lu to end of file\n", length, *ptrNbByte + SHCc - (ulong) position ) ;
                error = ERR_FMT ;
            }
            if ( *ptrItemLen < 1 || *ptrItemLen > cVL ) {
                printf (" Data variable items length %i is not supported: %u\n", cVL, *ptrItemLen) ;
                error = ERR_FMT ;
            }
            if (error != ERR_OK) break ;

            error = WriteByteSumTo156Wav (tmpH, ptrFile) ;
            if (error != ERR_OK) break ;
            error = WriteByteSumTo156Wav (tmpL, ptrFile) ;
            if (error != ERR_OK) break ;

            if ( (ptrFile->debug & 0x00C0) == 0x0080 )
                printf(" Dim1:%02X ", (uchar) ptrDstHead->dim1) ;
            error = WriteByteSumTo156Wav (ptrDstHead->dim1, ptrFile) ;
            if (error != ERR_OK) break ;

            if ( (ptrFile->debug & 0x00C0) == 0x0080 )
                printf(" Dim2:%02X ", (uchar) ptrDstHead->dim2) ;
            error = WriteByteSumTo156Wav (ptrDstHead->dim2, ptrFile) ;
            if (error != ERR_OK) break ;

            if ( (ptrFile->debug & 0x00C0) == 0x0080 )
                printf(" Item/length:%02X ", (uchar) ptrDstHead->itemLen) ;
            error = WriteByteSumTo156Wav (ptrDstHead->itemLen, ptrFile) ;
            if (error != ERR_OK) break ;

            if ( ptrFile->ident == IDENT_PC16_DAT) {
                /* for checksum */
                ptrFile->block_len = length - 3 ;
                ptrFile->count = 0 ;
            }

    } while (0) ;
    if (error == ERR_OK && inVal == EOF ) error = ERR_NOK ;

    return (error);
}

/*  WriteFooterToB22BinWav */
int WriteFooterTo15Wav (uchar  type,
                    FileInfo*  ptrFile)
{
    ulong  sum ;
      int  error ;

    do {

        if (type != TYPE_DAT) { /* Last checksum of DAT was written inside the footer of the data block */

            if (type == TYPE_IMG) {
                    /* BAS_1500_EOF is not included in images from Wav2bin */
                    error = WriteByteSumTo15Wav (BAS_1500_EOF, ptrFile) ;
                    if (error != ERR_OK) break ;
            }
            if (ptrFile->count > 0) {   /* Last checksum has not written by WriteByteSum before */
                sum = (ptrFile->sum >> 8) & 0xFF ;
                error = WriteByteTo15Wav (sum, ptrFile) ;
                if (error != ERR_OK) break ;

                sum = ptrFile->sum & 0xFF ;
                error = WriteByteTo15Wav (sum, ptrFile) ;
                if (error != ERR_OK) break ;

                if ( (ptrFile->debug & 0x0040) > 0 )
                printf(" (%04X)", (uint) ptrFile->sum);
            }
        }

        error = WriteSyncToWav (72, ptrFile) ;
        if (error != ERR_OK) break ;

        error = WriteByteTo15Wav ( EOF_15 , ptrFile) ;
        if (error != ERR_OK) break ;

        if ( (ptrFile->debug & 0x0040) > 0 )
            printf(" (%02X)", (uchar) EOF_15 );

        error = WriteSyncToWav (70, ptrFile) ;
        if (error != ERR_OK) break ;

    } while (0) ;
    return (error);
}

/* WriteFooterToB13BinWav */
int WriteFooterToNewWav (FileInfo*  ptrFile)
{
      int  error ;

    do {
        ptrFile->count = 0 ; /* no checksum writing from here until the end */

        error = WriteByteSumToWav(BAS_NEW_EOF, ORDER_STD, ptrFile->mode, ptrFile) ;
        // error = WriteByteSumToB13Wav (BAS_NEW_EOF, ptrFile) ;
        if (error != ERR_OK) break ;

        error = WriteByteToWav(BAS_NEW_EOF, ORDER_STD, ptrFile->mode, ptrFile) ;
        if (error != ERR_OK) break ;

        if ( (ptrFile->debug & 0x00C0) > 0 )
            printf(" EOF:%02X", (uchar) BAS_NEW_EOF);

        error = WriteByteToWav(ptrFile->sum, ORDER_STD, ptrFile->mode, ptrFile) ;
        if (error != ERR_OK) break ;

        if ( (ptrFile->debug & 0x0040) > 0 )
            printf(" (%02X)", (uchar) ptrFile->sum);

	/* there are 2bits more HIGH at the end of transmission (at least for PC-1402) M. NOSSWITZ */

	error = WriteBitToWav (1, ptrFile) ;
	if (error != ERR_OK) break ;

	error = WriteBitToWav (1, ptrFile) ;
	if (error != ERR_OK) break ;

	/* This puts 2 bits of silence (or 2 HIGH bits alternatively) to the end of the wave file. */
	/* CLOAD does not accept any sound, that could be interpreted as a start bit,              */
    /* during post-processing. Original CSAVE switches the signal low ms after the            */
    /* end of transmission, before the motor of the cassette recorder is switched off.        */
    /* This level out is visible in the CSAVE audio signal after the last bit. T. Muecker     */

	error = WriteBitToWav (3, ptrFile) ;
	if (error != ERR_OK) break ;

	error = WriteBitToWav (2, ptrFile) ;
	if (error != ERR_OK) break ;

    } while (0) ;
    return (error);
}


int WriteFooterToDataWav (uint  type, /* data type or length */
                          FileInfo*  ptrFile)
{
    int  error ;

    do {

        if (ptrFile->count != 0) {

            if ( (ptrFile->debug & 0x0040) > 0 )
                printf(" (%02X)", (uchar) ptrFile->sum);

            error = WriteByteToWav(ptrFile->sum, ORDER_STD, ptrFile->mode, ptrFile) ;
            if (error != ERR_OK) break ;

            ptrFile->count = 0 ;
        }

        if (type == DATA_VARIABLE) {

            error = WriteSyncToWav (97, ptrFile) ;  /* ~ 0.2017 sec = 101 ( - 4 standard from WriteBitToDataWave ) */
            if (error != ERR_OK) break ;

            error = WriteByteToWav(BAS_OLD_EOF, ORDER_STD, ptrFile->mode, ptrFile) ; //WriteToTAP need this way
//            error = WriteByteToWav(DATA_EOF, ORDER_INV, MODE_B15, ptrFile) ;  //WriteToTAP NOT works this way
            if (error != ERR_OK) break ;

            if ( (ptrFile->debug & 0x00C0) > 0 )
                printf(" EOB:%02X", (uchar) DATA_EOF);

            error = WriteBitToWav (1, ptrFile) ;
            if (error != ERR_OK) break ;

            error = WriteBitToWav (1, ptrFile) ;
            if (error != ERR_OK) break ;
        }

	error = WriteBitToWav (1, ptrFile) ;
	if (error != ERR_OK) break ;

	/* 2bits more than original to prevent sound interpretation as a start
       bit at the end the wave file, more - see WriteFooterToB13BinWav */

	error = WriteBitToWav (1, ptrFile) ;
	if (error != ERR_OK) break ;

	error = WriteBitToWav (1, ptrFile) ;
	if (error != ERR_OK) break ;

    if ( (ptrFile->debug & 0x00C0) > 0 ) printf("\n");

    } while (0) ;
    return (error);
}

/* Footer of one data block only */
int WriteFooterTo15DataWav (FileInfo*  ptrFile)
{
    ulong  sum ;
//    int  error ;
    int  error = 0;

    do {

        if (ptrFile->count != 0) {

            if ( (ptrFile->debug & 0x0040) > 0 )
                printf(" (%04X)", (uint) ptrFile->sum);

            /* Write the checksum */
            sum = (ptrFile->sum >> 8) & 0xFF ;
            error = WriteByteTo15Wav (sum, ptrFile) ;
            if (error != ERR_OK) break ;

            sum = ptrFile->sum & 0xFF ;
            error = WriteByteTo15Wav (sum, ptrFile) ;
            if (error != ERR_OK) break ;

            ptrFile->count = 0 ;
            ptrFile->sum = 0 ;
        }

    } while (0) ;

    return (error);
}


int WriteFooterToEWav (uchar   type,
                    FileInfo*  ptrFile)
{
       ulong  nbSync ;
        uint  ii ;
         int  error ;

    do {
        /* write stop bit and pull down the level at the end of the E block */
        error = WriteBitToEWav (3, ptrFile) ;
        if (error != ERR_OK) break ;

        if ( type == TYPE_ASC || type == TYPE_DAT || (ptrFile->debug & (SYNCL_TRM | SYNCL_STD))>0 ) // type == TYPE_BAS ||
             nbSync = ptrFile->nbSync1 *4 +375 ; /* 4+ sec for ASC-BAS, ASC-DAT, DAT_16 */
        else
             nbSync = ptrFile->nbSync1 /4 ; /* IMG, BIN, RSV can end with silence of 0.25 sec only */

        for ( ii = 0 ; ii < nbSync ; ++ii ) {
            error = WriteBitToEWav (2, ptrFile) ;
            if (error != ERR_OK) break ;
        }

    } while (0) ;

    return (error);
}


void conv_old2asc( uchar *str, int len )

{
    int ii ;
    uchar asc, old ;

    for ( ii = 0 ; ii < len ; ++ii  ) {
        old=str[ii] ;
        if (old == 0 || old == DATA_STD_STR) break ;
        asc = old ;

        if ((old > 63 && old < 74 ) || /* Numbers */
            (old > 80 && old < 107 ))  /* upper chars */
            asc = old - 16 ;
        else {
            if (old == 17 ) asc = ' ' ;
            if (old == 18 ) asc = 34  ;
            if (old == 19 ) asc = '?' ;
            if (old == 20 ) asc = '!' ;
            if (old == 21 ) asc = 35  ;
            if (old == 22 ) asc = '%' ;
            if (old == 24 ) asc = '$' ;
            if (old == 27 ) asc = ',' ;
            if (old == 28 ) asc = ';' ;
            if (old == 29 ) asc = ':' ;
            if (old == 30 ) asc = '@' ;
            if (old == 31 ) asc = '&' ;
            if (old == 48 ) asc = '(' ;
            if (old == 49 ) asc = ')' ;
            if (old == 50 ) asc = '>' ;
            if (old == 51 ) asc = '<' ;
            if (old == 52 ) asc = '=' ;
            if (old == 53 ) asc = '+' ;
            if (old == 54 ) asc = '-' ;
            if (old == 55 ) asc = '*' ;
            if (old == 56 ) asc = '/' ;
            if (old == 57 ) asc = '^' ;
            if (old == 74 ) asc = '.' ;
            if (old == 75 ) asc = 'E' ;
            if (old == 77 ) asc = '~' ;
            if (old == 78 ) asc = '_' ;
        }
        str[ii] = (uchar)asc ;
    }
}


void conv_asc2old( uchar *str, int len )

{
    int ii ;
    uchar asc, old ;

    for ( ii = 0 ; ii < len ; ++ii  ) {
        asc=str[ii] ;
        if (asc == 0 || old == DATA_STD_STR) break ;
        old = asc ;

        if ((asc > 47 && asc < 58 ) || /* Numbers */
            (asc > 64 && asc < 91 ))   /* upper chars */
            old = asc + 16 ;
        else {
            if (asc == ' ' ) old = 17 ;
            if (asc == 34 )  old = 18 ;
            if (asc == '?' ) old = 19 ;
            if (asc == '!' ) old = 20 ;
            if (asc == 35 )  old = 21 ;
            if (asc == '%' ) old = 22 ;
            if (asc == '$' ) old = 24 ;
            if (asc == ',' ) old = 27 ;
            if (asc == ';' ) old = 28 ;
            if (asc == ':' ) old = 29 ;
            if (asc == '@' ) old = 30 ;
            if (asc == '&' ) old = 31 ;
            if (asc == '(' ) old = 48 ;
            if (asc == ')' ) old = 49 ;
            if (asc == '>' ) old = 50 ;
            if (asc == '<' ) old = 51 ;
            if (asc == '=' ) old = 52 ;
            if (asc == '+' ) old = 53 ;
            if (asc == '-' ) old = 54 ;
            if (asc == '*' ) old = 55 ;
            if (asc == '/' ) old = 56 ;
            if (asc == '^' ) old = 57 ;
            if (asc == '.' ) old = 74 ;
            if (asc == '~' ) old = 77 ;
            if (asc == '_' ) old = 78 ;

            if (asc > 96 && asc < 123) old = asc - 16 ; /* lower chars */
        }
        str[ii] = (uchar)old ;
    }
}


int ConvertDataVariableItem (
                        uchar*  VarItem,
                     TypeInfo*  ptrSrcHead,
                     TypeInfo*  ptrDstHead,
                        uchar   ItemLen,
                        uchar   ItemType,
                        ulong   debug)
{
        uchar   tmpB, tmpE, Sign, Exponent ;
        bool    ExpNegative ;
        int     ii, error = ERR_OK ;

        /* Convert from PC-1500 to other */
        if ( ptrSrcHead->itemLen == DATA_NUM_15  &&
            (ptrDstHead->itemLen == DATA_STD_LEN || ptrDstHead->length == DATA_VARIABLE)) {

            /* Convert exponent from binary to BCD and shift sign */
            Sign = ( VarItem [1] & 0xF0 ) >> 4 ;

            ExpNegative = VarItem [0]  > 0x80 ;
            if (ExpNegative) {
                Exponent = 99 - (0xFF - VarItem [0] ) ;
                tmpE = 0x90 ;
            }
            else {
                Exponent = VarItem [0] ;
                tmpE = 0 ;
            }
            VarItem [0] = tmpE | Exponent / 10 ;
            VarItem [1] = Sign | Exponent % 10 << 4 ;

            if (ptrDstHead->length == DATA_VARIABLE){
                /* bring bytes in reverse order */
                for ( ii = 0 ; ii < ItemLen / 2 ; ++ii ) {
                    tmpB = VarItem[ ii ] ;
                    tmpE = VarItem[ ItemLen-1 - ii] ;
                    VarItem[ ii ] = tmpE ;
                    VarItem[ ItemLen-1 - ii] = tmpB ;
                }
            }
        }

        /* Convert from other to PC-1500 */
        else if (ptrDstHead->itemLen == DATA_NUM_15  &&
                (ptrSrcHead->itemLen == DATA_STD_LEN || ptrSrcHead->length == DATA_VARIABLE)) {

            if (ptrSrcHead->length == DATA_VARIABLE){
                if ( VarItem[ItemLen - 1] == DATA_STD_STR ) { /*set string variable to 0*/
                    for ( ii = 0 ; ii < ItemLen ; ++ii ) {
                        VarItem[ ii ] = 0 ;
                    }
                }
                else { /* bring bytes in reverse order */
                    for ( ii = 0 ; ii < ItemLen / 2 ; ++ii ) {
                        tmpB = VarItem[ ii ] ;
                        tmpE = VarItem[ ItemLen-1 - ii] ;
                        VarItem[ ii ] = tmpE ;
                        VarItem[ ItemLen-1 - ii] = tmpB ;
                    }
                }
            }
            /* Convert exponent from BCD to binary and shift sign */
            Sign = ( VarItem [1] & 0x0F ) << 4 ;
            Exponent = ((VarItem [1] & 0xF0 ) >> 4 ) + ( VarItem [0] & 0x0F ) * 10 ;

            ExpNegative = ( VarItem [0] & 0xF0 ) > 0x80 ;
            if (ExpNegative) Exponent = 0xFF - (99 - Exponent ) ;

            VarItem [0] = Exponent ;
            VarItem [1] = Sign ;
        }

        /* Convert strings */
        else if ((debug & 0x10) > 0 &&  /*convert char code */
                ptrDstHead->itemLen != DATA_NUM_15 && ptrSrcHead->itemLen != DATA_NUM_15 ) {

                if ( pcgrpId == IDENT_PC1500) {
                    /* convert string from OLD_BAS to PC-1500 */
                    conv_old2asc ( VarItem, ItemLen ) ;
                }
                else if ( ItemType == DATA_STR ||
                          ( ptrSrcHead->itemLen != DATA_STD_LEN  && //ToDo check double precision
                            ptrSrcHead->length  != DATA_VARIABLE  ) ||
                          ( ptrSrcHead->length  == DATA_VARIABLE &&
                            VarItem[ItemLen -1] == DATA_STD_STR   )  )  {
                    /* convert strings between OLD_BAS */
                    if ( pcgrpId == GRP_OLD || pcgrpId == IDENT_PC1211 ) conv_asc2old( VarItem, ItemLen );
                    else conv_old2asc ( VarItem, ItemLen );
                }
        }

    return (error);
}

int ConvertBinToWav (char*  ptrSrcFile,
                     char*  ptrDstFile,
                     char*  ptrDstExt,
                     uchar  type,
                     ulong  addr,
                     ulong  eaddr,
                     double sync,
                     double syncS,
                     char*  ptrName,
                     ulong  debug)
{
    FileInfo  info ;
        FILE  *srcFd ;
         int  inVal ;
       ulong  nbSamp ;
       ulong  freq ;
       ulong  base_freq ;   /* SHARP audio default base frequency for sync bits */
       ulong  nbSync, nbSyncS ;
         int  error, error_tmp ;
//      fpos_t  position;
        long  position;
       ulong  nbByte, limit ;
	   ulong  pos_end ;     /* last byte of a data variable block */
       uchar  itemLen ;     /* variable item real length */
       uchar  itemType ;    /* variable item type */
    TypeInfo  srcHead,
              dstHead ;     /* Variable Header */
       uchar  varItem[cVL] ;
        uint  ii ;

    do {
        info.ptrFd = NULL ;
        info.ident = IDENT_UNKNOWN ;
        info.total_diff = 0 ;
        info.total = 0 ;
        info.count = 0 ;
        info.sum   = 0 ;
        info.bitLen= strlen (bit[0]) ; /* with constant bit length, not for E/G */
        info.debug = debug ;
        srcFd      = NULL ;
        error      = ERR_OK ;

        /* Open the destination file */
        info.ptrFd = fopen (ptrDstFile, "wb") ;
        if (info.ptrFd == NULL) {
            printf ("%s: Can't open the destination file: %s\n", argP, ptrDstFile) ;
            error = ERR_FILE ;
            break ;
        }

        /* Open the source file */
        srcFd = fopen (ptrSrcFile, "rb") ;
        if (srcFd == NULL) {
            printf ("%s: Can't open the source file: %s\n", argP, ptrSrcFile) ;
            error = ERR_FILE ;
            break ;
        }

        error = ReadFileLength (type, &nbByte, &srcFd) ;
        if (error != ERR_OK) break ;

        if ((info.debug & 0x0080) > 0) printf (" File length total: %d bytes\n", (uint) nbByte);

        if (pcgrpId == GRP_E ) limit = 0x100000 ;
        else if (pcgrpId == GRP_16 ) limit = 0x14000 ;
        else limit = 0x10000 ;
        if (nbByte > limit) {
            printf ("%s: Source file contains more than %lu bytes\n", argP, limit) ;
            error = ERR_FMT ;
            break ;
        }
        if (pcgrpId == GRP_16 ) limit = 0x100000 ;
        if ( (addr + nbByte) > limit) {
            printf ("%s: (Address + Size) greater than %lu bytes\n", argP, limit) ;
            error = ERR_FMT ;
            break ;
        }
        if (type == TYPE_RSV) { /* from PC-1500 or PC-1600 only */
                if (pcgrpId == IDENT_PC1500) {
                    limit = 190 ;
                    if ( limit == nbByte +1 ) {
                        ++nbByte;
                        if (Qcnt ==0) printf ("%s: Reserve data are from PC-1600!\n", argP) ;
                    }
                }
                else if (pcgrpId == GRP_16 ) {
                    limit = 189 ;
                    if ( limit == nbByte -1 ) {
                        --nbByte;
                        if (Qcnt ==0) printf ("%s: Reserve data are from PC-1500 or Mode 1!\n", argP) ;
                    }
                }
                if (nbByte > limit) {
                    printf ("%s: Reserve data greater than 189+%lu bytes\n", argP, limit-189) ;
                    error = ERR_FMT ;
                    break ;
                }
                else if (nbByte != limit) {
                    printf ("%s: Warning: Reserve data shorter than 189+%lu bytes\n", argP, limit-189) ;
                }
        }

        if (pcgrpId == IDENT_PC1500 ) {
            base_freq = BASE_FREQ2 ;
            freq = base_freq * info.bitLen/16 ;
            if (freq == 22500) freq = 22050 ; /* bit3_15: 2% slower but 44,1 kHz better supported with sound hardware */
            else if (freq == 7968) freq = 8000 ; /* bit4_15: slightly faster but exactly 16 kHz */
        }
        else if (pcgrpId == GRP_E || pcgrpId == GRP_G || pcgrpId == GRP_16) {

            for ( ii = 0 ; ii < 4; ++ii ) {bitLE [ii] = (uint) strlen (bitE[ii]); }
            base_freq = BASE_FREQ3 ;
            freq = base_freq * bitLE[0]/2 ;
            if (freq == 7500) freq = 8000 ; /* bitE4 16 kHz */
        }
        else { /* common types with 4 kHz synchronisation signal */
            base_freq = BASE_FREQ1 ;
            freq = base_freq * info.bitLen/16 ;
        }

        if (pcgrpId == GRP_E || pcgrpId == GRP_G || pcgrpId == GRP_16) {
            if (freq == base_freq && sync < 3) sync = 3; /* min. 3 sec GRP_E with asymmetric waveform 1*/

            info.nbSync1 = base_freq ;

            if (pcgrpId == GRP_E) { // && (type == TYPE_ASC || type == TYPE_BAS)) {
                if (sync < 2.125) nbSync = 17 * info.nbSync1 /8 ;   /* min. ca. 2.125 sec */
                else nbSync = (ulong)(sync*256) * info.nbSync1 /256 ;
            }
            else if (sync < 1.875) nbSync = 15 * info.nbSync1 /8 ;  /* min. ca. 1.875 sec */
            else nbSync = (ulong)(sync*256) * info.nbSync1 /256 ;   /* > 5000 + tolerance */

            if (syncS < 2.125) nbSyncS = 17 * info.nbSync1 /8 ;     /* 1. Spc after header > 2 s */
            else nbSyncS = (ulong)(syncS*256) * info.nbSync1 /256 ; /* min. ca. 2.125 sec */

            if (Scnt<1 || (info.debug & (SYNCL_STD | SYNCL_TRM)) > 0) {
                if (pcgrpId == GRP_16 && nbSync < 10000) nbSync = 10000 ; /* for first sync only */
                if (pcgrpId == GRP_E && (type == TYPE_ASC || type == TYPE_BAS)  &&  nbSync < 11250)
                                                                                    nbSync = 11250 ;
            }
            info.nbSync = nbSync ;
        }
        else { /* PC-1211 to PC-1500 */
            info.nbSync1 = (base_freq +4 ) /8 ;
            info.nbSync = nbSync = (ulong)(sync*256) * info.nbSync1 /256 ;
        }

        /* Calculate the length and Ident of the destination file */
        error = LengthAndIdentOfBinWav (type, nbByte, nbSync, &info.ident, &nbSamp, &info.debug) ;
        if (error != ERR_OK) break ;

        /* Write the header of the destination WAV file */
        if (TAPc > 0) {
            if (Qcnt == 0) printf ("File format  : forced to emulator tap format (no wav)\n") ;
        }
        else
            error = WriteHeadToWav (nbSamp, (ulong) (speed * freq * 2), &info) ;
        if (error != ERR_OK) break ;

        /* Write the synchro pattern for N times and set length of interim syncs  */
        if (Qcnt == 0) printf ("Synchro size : %lu bits\n", nbSync);

        if (pcgrpId == GRP_E || pcgrpId == GRP_G || pcgrpId == GRP_16) {
            /* for E500S (+ CE-126P) also spaces at start of file, wait until remote relays is switched on */
             error = WriteSyncToEWav (nbSync, nbSyncS, SYNC_E_HEAD, &info) ;

            if (pcgrpId == GRP_E        /* min. ca. 2.375 sec for first data block */
                    && sync < 2.5)  {   /* sync = 2.5 ; for first data block after header only */
                    nbSync = 10 * info.nbSync1 /4 ;
                    /* info.nbSync not changed: min. ca. 2.125 sec */
            }
            /* allow shorter syncs and spaces for interim syncs */
            else if ((info.debug & (SYNCL_STD | SYNCL_TRM)) == 0) {
                nbSyncS = (ulong)(syncS*256) * info.nbSync1 /256 ;

                if (sync < 1.75) nbSync =  7 * info.nbSync1 /4 ; /* min. ca. 1.7 sec */
                else nbSync = (ulong)(sync*256) * info.nbSync1 /256 ;
                info.nbSync = nbSync ;
            }
            // moved to WriteSync. else 16_DAT 5.0 sec, 16_ASC 5...5.175 sec, G8_ASC 3.0839
        }
        else { /* PC-1211 to PC-1500 */
            error = WriteSyncToWav (nbSync, &info) ;

            /* TRM PC-1600 P.124 for PC-1500 first sync 1.260: Not so found! (-6 stop b: 1.2408) */
            if (info.ident == IDENT_PC15_DAT && sync > 1.0068)  /* 1.026 - 6 stop bits*/
                                         info.nbSync = (1007 * info.nbSync1)/1000;
            /* Standard sync between data blocks is: 6 ms + 2000 ms + 6 ms +? */
            else if (type == TYPE_DAT && sync > 2.009)  /* 2.017 - 4 stop bits*/
                                         info.nbSync = (2009 * info.nbSync1)/1000;
        }
        if (error != ERR_OK) break ;

        if (Qcnt == 0) {
            printf ("Wave format  : 0x%02X <- ", (uchar) info.ident) ;
            if (type == TYPE_BIN)
                printf ("Binary for CLOAD M, PC-%lu\n", pcId) ;
            else if (type == TYPE_RSV)
                printf ("ReSerVe data for CLOAD in RSV Mode, PC-%lu\n", pcId) ;
            else if (type == TYPE_DAT)
                printf ("Special binary data for INPUT#, PC-%lu\n", pcId) ;
            else if (type == TYPE_ASC) /* PC-E/G/1600 */
                printf ("ASCII data for INPUT# (or LOAD CAS:) PC-%lu\n", pcId) ;
            else if (type == TYPE_BAS) /* PC-E/G/1600 */
                printf ("ASCII BASIC source for Text menu or LOAD CAS: PC-%lu\n", pcId) ;
            else if (type == TYPE_TXT) /* GRP_EXT, GRP_E */
                printf ("Basic image for CLOAD in TEXT modus, PC-%lu\n", pcId) ;
            else { /*TYPE_IMG */
                if (pcgrpId == IDENT_PC1500 || pcgrpId == GRP_16 || pcgrpId == GRP_E || pcgrpId == GRP_G)
                     printf ("Basic image with intermediate code, PC-%lu\n", pcId) ;
                else printf ("Basic (or RSV) image with intermediate code, PC-%lu\n", pcId) ;
            }
        }

        /* No file header for TYPE_VAR, following data_variable block */
        if ((info.debug & NO_FILE_HEAD) == 0) {

            if ( (info.debug & 0x00C0) > 0 && Qcnt != 0 )
                printf(" FileID:%02X   ", (uchar) info.ident) ;     /* File ID */

            if (pcgrpId == IDENT_PC1500) {
                info.mode = info.mode_h = MODE_B22 ;

                /* Write the TAP code */
                error = WriteQuaterToWav (IDENT_PC1500, 6, &info) ;
                if (error != ERR_OK) break ;

                /* Write the name, size, addresses */
                error = WriteHeadTo15Wav (ptrName, addr, eaddr, nbByte, type, &info) ;
                if (error != ERR_OK) break ;

                if ((type == TYPE_BIN) && (Qcnt == 0)) {
                    printf ("Start Address: 0x%04X\n", (uint) addr);
                    printf ("End   Address: 0x%04X, Length: %d bytes\n", (uint) (addr + nbByte -1), (uint) nbByte);
                    if (eaddr < 0xFFFF) printf ("Entry Address: 0x%04X\n", (uint) eaddr);
                }
            }  // end if PC-1500
            else if (pcgrpId == GRP_E || pcgrpId == GRP_G || pcgrpId == GRP_16) {
                info.mode = info.mode_h = MODE_B9 ;

                /* Write the Name and Header */
                error = WriteHeadToEWav (ptrName, ptrDstExt, addr, eaddr, nbByte, nbSync, nbSyncS, type, &info) ;
                if (error != ERR_OK) break ;

                if ((type == TYPE_BIN) && (Qcnt == 0)) {
                    printf ("Start Address: 0x%06X\n", (uint) addr);
                    printf ("End   Address: 0x%06X, Length: %d bytes\n", (uint) (addr + nbByte -1), (uint) nbByte);
                    if (eaddr < 0xFFFFFF) printf ("Entry Address: 0x%06X\n", (uint) eaddr);
                }
            }
            else { // PC-121x ... PC-1475
                switch (info.ident) { /* Header Mode */
                case IDENT_PC1211 :
                    info.mode = info.mode_h = MODE_B20 ;
                    break ;

                case IDENT_PC121_DAT :
                case IDENT_OLD_BAS :
                case IDENT_OLD_DAT :
                case IDENT_OLD_BIN :
                    info.mode = info.mode_h = MODE_B19 ;
                    break ;

                case IDENT_NEW_BAS :
                case IDENT_EXT_BAS :
                case IDENT_NEW_DAT :
                case IDENT_NEW_BIN :
                    info.mode = info.mode_h = MODE_B16 ;
                    break ;

                default :
                    printf ("%s: Unknown Ident\n", argP) ;
                    info.mode = info.mode_h = MODE_B21 ;
                    break ;
                }
                /* Write the TAPE code */
                error = WriteByteToWav ( (ulong) info.ident, ORDER_STD, info.mode_h, &info) ;
                if (error != ERR_OK) break ;

                /* Write the Name */
                error = WriteSaveNameToWav (ptrName, info.mode_h, &info) ;
                if (error != ERR_OK) break ;

                switch (info.ident) { /* Body Data Mode */
                case IDENT_PC1211 :
                    info.mode = MODE_B20 ;
                    break ;

                case IDENT_PC121_DAT :
                case IDENT_OLD_BAS :
                case IDENT_OLD_BIN :
                    info.mode = MODE_B19 ;
                    break ;

                case IDENT_OLD_DAT :
                case IDENT_NEW_DAT :
                    info.mode = MODE_B15 ;
                    break ;

                case IDENT_EXT_BAS :

                case IDENT_NEW_BAS :
                case IDENT_NEW_BIN :
                    if (cnvstr_upr) info.mode = MODE_B13 ; /*older part of new series*/
                    else  info.mode = MODE_B14 ;     /*new series and extended series*/
                    break ;

                default :
                    printf ("%s: Unknown Ident\n", argP) ;
                    info.mode = MODE_B21 ;
                    break ;
                }
            }  // PC-121x ... PC-1475
            if (error != ERR_OK) break ;
        }
        info.total = 0 ; /* count bytes of body only */

        switch (info.ident) { /* header was written, write all data now*/
        case IDENT_PC15_BAS :
            /* Write the datas */
            do {
                inVal = fgetc (srcFd) ;
                if (inVal == EOF) break ;

                /* BAS_1500_EOF was included in images from Bas2img but should be no more*/
                if ( inVal == BAS_1500_EOF && info.total == nbByte ) { // nbByte - 1, changed in ReadFL
                    if (Qcnt == 0) printf
                        ("\nEnd of file mark %02X should not be included in the image\n", inVal) ;
                    /* EOF mark will be written by WriteFooter */
                    break ;
                }

                error = WriteByteSumTo15Wav ( (uint) inVal, &info) ;
                if (error != ERR_OK) break ;

            } while (1) ;
            if (error != ERR_OK) break ;

            /* Write the END code */
            error = WriteFooterTo15Wav (type, &info) ;

            break ; // IDENT_PC15_BAS

        case IDENT_PC15_RSV :
        case IDENT_PC15_BIN :
            /* Write the datas */
            do {
                inVal = fgetc (srcFd) ;
                if (inVal == EOF) {
                    if (info.ident == IDENT_PC15_RSV && info.total +1 == nbByte)
                        inVal = 0 ; /* append 0 for RSV from PC-1600 Mode 0*/
                    else break ;
                }
                error = WriteByteSumTo15Wav ( (uint) inVal, &info) ;
                if (error != ERR_OK) break ;

            } while (1) ;
            if (error != ERR_OK) break ;

            /* Write the END code */
            error = WriteFooterTo15Wav (type, &info) ;

            break ; // IDENT_PC15_BIN, IDENT_PC15_RSV

        case IDENT_OLD_BAS :
        case IDENT_PC1211 :
            /* Write the datas */
            do {
                inVal = fgetc (srcFd) ;
                if (inVal == EOF) break ;

                if ( inVal == BAS_OLD_EOF && info.total + 1 == nbByte ) {
                    if (Qcnt == 0 && SHCc == 0) printf
                        ("\nEnd of file mark %02X should not be included in the image\n", inVal) ;
                    break ;
                }

                error = WriteByteSumToWav((uint) inVal, ORDER_STD, info.mode, &info) ;
//                error = WriteByteSumToB19Wav ( (uint) inVal, &info) ;
                if (error != ERR_OK) break ;

            } while (1) ;
            if (error != ERR_OK) break ;

            /* Write the END code */
            error = WriteByteToWav (BAS_OLD_EOF, ORDER_STD, info.mode, &info) ;
            if ( (info.debug & 0x00C0) > 0 )
                printf(" EOF:%02X", (uchar) BAS_OLD_EOF);

            if ( info.ident == IDENT_PC1211) {

                error = WriteBitToWav (2, &info) ;
                if (error != ERR_OK) break ;
                error = WriteSyncToWav (38, &info) ;
            }
            break ;  // IDENT_OLD_BAS, IDENT_PC1211

        case IDENT_OLD_BIN :
            /* Write the addresse and length */
            error = WriteHeadToBinWav (addr, nbByte, info.mode_h, &info) ;
            if (error != ERR_OK) break ;

            /* Write the datas */
            do {
                inVal = fgetc (srcFd) ;
                if (inVal == EOF) break ;

                error = WriteByteSumToWav ( (uint) inVal, ORDER_STD, info.mode, &info) ;
                if (error != ERR_OK) break ;

            } while (1) ;
            if (error != ERR_OK) break ;

            /* Write the END code */
            error = WriteByteToWav (BAS_OLD_EOF, ORDER_STD, info.mode, &info) ;
            if ( (info.debug & 0x00C0) > 0 )
                printf(" EOF:%02X", (uchar) BAS_OLD_EOF);

            break ; // IDENT_OLD_BIN

        case IDENT_NEW_BAS :
        case IDENT_EXT_BAS :
            /* Write the datas */
//            info.mode = MODE_B13 ; /*PC-1403 and newer should be MODE_14 */
            /* the older simple algorithm seems to work as well, but this is now, what the PC does originally */
            for ( ii = 0 ; ii < nbByte - 1 ; ++ii ) {
                inVal = fgetc (srcFd) ;
                if (inVal == EOF) break ;

                if ( inVal == BAS_NEW_EOF ) {
                    if (info.count + 1 == BLK_NEW && info.sum == 0xE1) { /* Constellation will generate 2-times BAS_NEW_EOF */
                        printf ("\nERROR %i at %lu. byte, usually the low byte of a BASIC line number\n", ERR_SUM, info.total) ;
                        printf ("This binary constellation activates the CLOAD bug of this series. The line\n") ;
                        printf ("number must be changed or minor changes done in the BASIC text before.\n") ;
                        /* Seldom Bug in CLOAD, for PC-1402/(01) at known ROM address: 40666 */
                        if ((info.debug & 0x800) == 0 ) {
                            error = ERR_SUM ;
                            break ;
                        }
                    }
                }
                error = WriteByteSumToWav ( (uint) inVal, ORDER_STD, info.mode, &info) ;
                if (error != ERR_OK) break ;
            }
            if (error != ERR_OK) break ;

            inVal = fgetc (srcFd) ; /* Read the last byte before EOF mark */
            if (inVal == EOF) break ;

            if (inVal == BAS_NEW_EOF) {
                        /* EOF mark should not be included for this file type normally*/
                        if (Qcnt == 0) printf ("End of File mark %i should not be included in the image\n", inVal) ;
                        /* if end of block, then an additional checksum would be written, but this does work anyhow */
            }
            else {
                if ( (info.debug & 0x0040) > 0 ) printf(" %02X", (uchar) inVal);
                error = WriteByteToWav ( (uint) inVal, ORDER_STD, info.mode, &info) ;
                if (error != ERR_OK) break ;
                CheckSumB1 ((uint) inVal, &info) ; /* never write the checksum before BAS_NEW_EOF */
                ++info.total ;
                ++info.count ; /* for debug purposes only, WriteFooter will reset it */
            }

            /* Write the END code */
            error = WriteFooterToNewWav (&info) ;

            break ; // IDENT_NEW_BAS, IDENT_EXT_BAS

        case IDENT_NEW_BIN :
            /* Write the address and length */
            error = WriteHeadToBinWav (addr, nbByte, info.mode_h, &info) ;
            if (error != ERR_OK) break ;

            /* Write the datas */
            /* the older simple algorithm seems to work as well, but this is now, what the PC does originally */
            for ( ii = 0 ; ii < nbByte - 1 ; ++ii ) {
                inVal = fgetc (srcFd) ;
                if (inVal == EOF) break ;

                error = WriteByteSumToWav ( (uint) inVal, ORDER_STD, info.mode, &info) ;
                if (error != ERR_OK) break ;
            }
            if (error != ERR_OK) break ;

            inVal = fgetc (srcFd) ; /* Read the last byte before EOF mark */
            if (inVal == EOF) break ;

            if ( (info.debug & 0x0040) > 0 ) printf(" %02X", (uchar) inVal);
            error = WriteByteToWav ( (uint) inVal, ORDER_STD, info.mode, &info) ;
            if (error != ERR_OK) break ;
            CheckSumB1 ( (uint) inVal, &info) ; /* never write the checksum before BAS_NEW_EOF */
            ++info.total ;
            ++info.count ; /* for debug purposes only, WriteFooter will reset it */

            /* Write the END code */
            error = WriteFooterToNewWav (&info) ;

            break ; // IDENT_NEW_BIN

        case IDENT_PC121_DAT :
        case IDENT_OLD_DAT :
        case IDENT_NEW_DAT :
            /* Do multiple data variable list*/
            do {
                /* One variable block*/
                error = WriteHeadToDataWav ( &srcHead, &dstHead, &itemLen, &itemType, &pos_end, &nbByte, &info, srcFd) ;
                /* Read, write, print info, reset checksum */
                if (error != ERR_OK) break ;

                if ( (info.debug & 0x0060) == 0x0020 )
                    printf("\nPos Byte\n") ;     /* Head of Data Row */

                /* Write the data */
                do {
                    for ( ii = 0 ; ii < itemLen; ++ii ) {
//                        fgetpos(srcFd, &position) ;
                        position = ftell( srcFd ) ;
                        if (position <= 0) {
                            printf ("\n%s:convert - Can't get position in the source file\n", argP) ;
                            return ( ERR_FILE ) ;
                        }
                        inVal = fgetc (srcFd) ;
                        if (inVal == EOF) break ;
                        if (SHCc > 0) inVal = SwapByte(inVal);
                        if ( ii == 0 && inVal == DATA_EOF && srcHead.length == DATA_VARIABLE ) break ;
                        varItem[ii] = (uint) inVal ;
                    }
                    if (inVal == EOF) {
                            printf ("\n%s:convert - file ended with an incomplete data variable block\n", argP) ;
                            return (ERR_FMT) ;
                    }
                    if ((srcHead.length == DATA_VARIABLE) && (info.count == 0 )&& (inVal == DATA_EOF)) break ;
                    /* Mark will written by WriteFooter if needed */

                    /* Convert numeric items from PC-1500 to other series */
                    error = ConvertDataVariableItem (varItem, &srcHead, &dstHead, itemLen, itemType, info.debug) ;

                    for ( ii = 0 ; ii < itemLen; ++ii ) {

                        if ( (info.debug & 0x0060) == 0x0020 )
                            printf("%ld %u \n", position, (uint) varItem[ii]) ;
                        error = WriteByteSumToDataWav ( (ulong) varItem[ii], info.mode, &info) ;
                        if (error != ERR_OK) break ;
                    }

                    if ( info.ident == IDENT_PC121_DAT) {
                        error = WriteSyncToWav (111, &info) ;
                        if (error != ERR_OK) break ;
                    }

                } while ( (ulong) position < pos_end && inVal != EOF && error == ERR_OK) ;
                if (error != ERR_OK) break ;

                /* Multiple data variable list was not supported with Ver. 1.4.2, option -VAR needed */
                /* Checksum was included between multiple Variable Blocks if itemLen<>8 and Wav2bin 1.5.0 */
                if ( (ulong) position >= pos_end && (info.debug & DATA_W2B150) > 0 &&
                                                    (srcHead.itemLen & ( DATA_STD_LEN - 1 ) ) == 0 ) {
                    inVal = fgetc (srcFd) ;
                    break ;
                }

                /* Write the END code of one or all data blocks */
                error = WriteFooterToDataWav (dstHead.length, &info) ;
                if (error != ERR_OK) break ;

                if (info.ident == IDENT_PC121_DAT) break ; /* one variable length block only */

                /* Write the synchronization pattern for the next data block */
                if (inVal != EOF && info.total + info.total_diff < nbByte)
                                                            error = WriteSyncToWav (info.nbSync, &info) ;

            } while (inVal != EOF && error == ERR_OK) ;

            break ; // IDENT_NEW_DAT, IDENT_OLD_DAT, IDENT_PC121_DAT

        case IDENT_PC16_DAT :

            /* Switch info.block_len between DATA_HEAD_LEN and BLK_E_DAT for envelope */
            /* content is similar to IDENT_PC15_DAT */
            /* see Tech Ref. Man. PC-1600 Page 117-121 */

        case IDENT_PC15_DAT :

            /* Do multiple data variable list*/
            do {
                /* One variable block*/
                /* Read, write, print info, reset checksum */
                error = WriteHeadTo156DataWav ( &srcHead, &dstHead, &itemLen, &itemType, &pos_end, &nbByte, &info, srcFd) ;
                if (error != ERR_OK) break ;
                if (info.ident == IDENT_PC16_DAT) error = WriteSyncToEWav (nbSync, nbSyncS, SYNC_E_DATA, &info) ;
                else if (info.ident == IDENT_PC15_DAT) error = WriteSyncToWav (73, &info) ;
                if (error != ERR_OK) break ;

                if ( (info.debug & 0x0060) == 0x0020 )
                    printf("\nPos Byte\n") ;     /* Head of Data Row */

                /* Write the data */
                do {
                    for ( ii = 0 ; ii < itemLen; ++ii ) {
//                        fgetpos(srcFd, &position) ;
                        position = ftell( srcFd ) ;
                        if (position <= 0) {
                            printf ("\n%s:convert - Can't get position in the source file\n", argP) ;
                            return ( ERR_FILE ) ;
                        }
                        inVal = fgetc (srcFd) ;
                        if (inVal == EOF) break ;
                        if (SHCc > 0) inVal = SwapByte(inVal); /* data from other series are swapped */
                        if ( ii == 0 && srcHead.length == DATA_VARIABLE && inVal == DATA_EOF) {
                            ++info.total_diff;  /* block end not written and not included in total write counter */
                            break ;
                        }
                        varItem[ii] = (uint) inVal ;
                    }
                    if (inVal == EOF) {
                            printf ("\n%s:convert - file ended with an incomplete data variable block\n", argP) ;
                            return (ERR_FMT) ;
                    }
                    if (srcHead.length == DATA_VARIABLE && (ulong) position >= pos_end && inVal == DATA_EOF) break ;
                    /* Mark will written by WriteFooter if needed */

                    /* Convert numeric items (single precision only) from other series to PC-1500 */
                    error = ConvertDataVariableItem (varItem, &srcHead, &dstHead, itemLen, itemType, info.debug) ;

                    for ( ii = 0 ; ii < itemLen; ++ii ) {

                        if ( (info.debug & 0x0060) == 0x0020 )
                            printf("%d %d \n", (uint) position, varItem[ii]) ;
                        error = WriteByteSumTo156Wav ((ulong) varItem[ii], &info) ;
                        if (error != ERR_OK) break ;
                    }
                //ToDo More tests, pos_end
                } while ( (ulong) position < pos_end && inVal != EOF && error == ERR_OK) ;
                if (error != ERR_OK) break ;

                if (info.ident == IDENT_PC15_DAT) {
                /* Write the END code of one data block */
                    error = WriteFooterTo15DataWav (&info) ;
                    if (error != ERR_OK) break ;

                    /* Write the synchro patern for the next data block */
                    if (inVal != EOF && info.total + info.total_diff < nbByte)
                                                            error = WriteSyncToWav (info.nbSync, &info) ;
                }
                else if (info.ident == IDENT_PC16_DAT) {
                    if (inVal != EOF && info.total + info.total_diff < nbByte)
                                     error = WriteSyncToEWav (info.nbSync, nbSyncS, SYNC_E_DATA, &info) ;
                }
                if (error != ERR_OK) break ;

            } while (inVal != EOF && error == ERR_OK) ;
            if (error > ERR_OK) break ;

            /* Write the END code */
            if (info.ident == IDENT_PC15_DAT) error = WriteFooterTo15Wav (type, &info) ;
            else if (info.ident == IDENT_PC16_DAT) error = WriteFooterToEWav (type, &info) ;

            break ; // IDENT_PC15_DAT, IDENT_PC16_DAT

        case IDENT_PC16_CAS :
        case IDENT_E_ASC :

            ii = 0 ;
            do {
                inVal = fgetc (srcFd) ;

                if (inVal == EOF) {
                    if ( ii == 0 )      { inVal = 0x0D ; ++ii; }
                    else if ( ii == 1 ) { inVal = 0x0A ; ++ii; }
                    else if ( ii == 2 ) { inVal = EOF_ASC ; ++ii; }
                    else if ( ii == 3 )   inVal = 0x0 ;
                }
                else if ( inVal == 0x0D && ii == 0 ) ++ii ;
                else if ( inVal == 0x0A && ii == 1 ) ++ii ;
                else    ii = 0 ;

                if (info.count == 0 && info.total > 0) {
                    /* Write the block stop bit, space, sync and start bit */
                    error = WriteSyncToEWav (info.nbSync, nbSyncS, SYNC_E_DATA, &info) ;
                    if (error != ERR_OK) break ;
                }
                /* Write the datas */
                error = WriteByteSumToWav ( (uint) inVal, ORDER_E, info.mode, &info) ;
                if (error != ERR_OK) break ;

            } while (info.count > 0 || ii < 3 ) ;
            if (error != ERR_OK) break ;

        error = WriteFooterToEWav (type, &info) ;

        break ; // IDENT_E_ASC, IDENT_PC16_CAS

        case IDENT_E_BIN  :
        case IDENT_E_BAS  : /* or G16 RSV */

            do {
                inVal = fgetc (srcFd) ;
                if (inVal == EOF) break ;

                /* Write the datas */
                error = WriteByteSumToWav ( (uint) inVal, ORDER_E, info.mode, &info) ;
                if (error != ERR_OK) break ;

            } while (1) ;
            if (error != ERR_OK) break ;

            if ( info.ident == IDENT_E_BAS && pcgrpId == GRP_E) { /* not GRP_G, not GRP_16 */

                error = WriteByteSumToWav ( BAS_1500_EOF, ORDER_E, info.mode, &info) ;
                if (error != ERR_OK) break ;
            }
            error = WriteFooterToEWav (type, &info) ;

            break ; // IDENT_E_BIN, IDENT_E_BAS

        default :
            printf ("%s:Convert: Unknown Ident\n", argP) ;
            error = ERR_ARG ;
            break ;
        } // end switch ident

    } while (0) ;
    error_tmp = error ;

    /* Close the source file */
    if (srcFd != NULL) {
        error = fclose (srcFd) ;
        if (error != ERR_OK) {
            printf ("%s: Can't close the source file\n", argP) ;
            error_tmp = ERR_FILE ;
        }
    }

    /* correct the count of samples, for multi data blocks needed */
        if ( TAPc == 0 && (error == ERR_OK || error == ERR_NOK)) {
            error = WriteSampleCountToHeadOfWav (nbSamp, &info) ;
            if (error != ERR_OK) {
                printf ("%s: Can't change the sample count inside the head of the destination file\n", argP) ;
                error_tmp = ERR_FILE ;
            }
        }

    /* Close the destination file */
    if (info.ptrFd != NULL) {
        error = fclose (info.ptrFd) ;
        if (error != ERR_OK) {
            printf ("%s: Can't close the destination file\n", argP) ;
            error_tmp = ERR_FILE ;
        }
    }
    error = error_tmp ;
    return (error);
}

/*Read Header data from SHC-File,  sets SHCc and pcgrpId also */
int ReadHeadFromShc (char*  ptrSrcFile,
                     uchar* ptrType,
                     ulong* ptrAddr,
                     char*  ptrName )
{
        FILE  *srcFd ;
         int  inVal, ii ;
       ulong  byte ;
      ushort  ident ;
       uchar  type = TYPE_NOK ;
       uchar  tmpS[8] ;
        long  position;
         int  error_tmp, error = ERR_NOK ;

    do {
        /* 1. Open the source file */
        srcFd = fopen (ptrSrcFile, "rb") ;
        if (srcFd == NULL) {            // ToDo: Check Path length and more '.'
            printf ("%s: Can't open the source file: %s\n", argP, ptrSrcFile) ;
            error = ERR_FILE ;
            break ;
        }

        inVal = fgetc (srcFd) ;
        if (inVal == EOF) break ;
        else ident = inVal ;
        error = ERR_OK ;

        switch (ident) {

        case IDENT_OLD_BAS :
            pcgrpId = GRP_OLD ; /*or IDENT_PC1211 encapsulated */
            type = TYPE_IMG ;
            break ;

        case IDENT_OLD_DAT :
            pcgrpId = GRP_OLD ; /*or IDENT_PC1211 encapsulated */
            type = TYPE_DAT ;
            break ;

        case IDENT_OLD_BIN :
            pcgrpId = GRP_OLD ;
            type = TYPE_BIN ;
            break ;

        case IDENT_NEW_BIN :
            pcgrpId = GRP_NEW ; /*or GRP_EXT */
            type = TYPE_BIN ;
            break ;

        case IDENT_NEW_DAT :
            pcgrpId = GRP_NEW ; /*or GRP_EXT */
            type = TYPE_DAT ;
            break ;

        case IDENT_NEW_BAS :
            pcgrpId = GRP_NEW ;
            type = TYPE_IMG ;
            break ;

        case IDENT_EXT_BAS :
            pcgrpId = GRP_EXT ;
            type = TYPE_IMG ;
            break ;

        default :    /* Password or IDENT_PC1500 are NOT supported */
            printf ("%s: Unsupported Identity %i of the SHC file\n", argP, ident) ;
            error = ERR_FMT ;
            break ;
        }
        if ( error != ERR_OK ) break;
        *ptrType = type ;

        for ( ii = 0 ; ii < 8 ; ++ii ) { /* no checksum included*/

            inVal = fgetc (srcFd) ;
            if (inVal == EOF) error = ERR_FMT ;
            else byte = SwapByte( (ulong) inVal) ;
            if (error != ERR_OK || ii == 7 ) break ;

            tmpS[6 - ii] = byte ;
        }
        if ( byte != 0xF5 || ii < 7 ) {
            printf("\n%s: Unexpected byte %lu in SHC file name header, position %i\n", argP, byte, ii + 1 ) ;
            error = ERR_FMT ;
        }
        if ( error != ERR_OK ) break;

        if (pcgrpId == GRP_OLD ) conv_old2asc( tmpS, 7) ;
        if (strlen(ptrName)==0) for ( ii = 0 ; ii < 8 ; ++ii ) ptrName[ii] = tmpS[ii] ;

        if (type == TYPE_BIN ) {
            for ( ii = 0 ; ii < 8 ; ++ii ) { /* no checksum included*/
                inVal = fgetc (srcFd) ;
                if (inVal == EOF) error = ERR_FMT ;
                else byte = SwapByte( (ulong) inVal) ;
                if (error != ERR_OK) break ;
                tmpS[ii] = byte;
            }
            if (error != ERR_OK) break ;

            *ptrAddr   = tmpS[4] ;
            *ptrAddr   = (*ptrAddr << 8) + tmpS[5] ;
            /* length will new calculated from real length */
        }

        /* Get the length of the SHC-Header */
        position = ftell (srcFd) ;
        if (position < ERR_OK) {
            printf ("%s: Can't ftell the file\n", argP) ;
            error = ERR_FILE ;
            break ;
        }
        else {
            SHCc = position ; /* SHC header length */
            if (type == TYPE_IMG) { /* End marks included in SHC image */
                if ( pcgrpId == GRP_NEW || pcgrpId == GRP_EXT ) SHCe = 2;
                else  SHCe = 1;
            }
            else if (type == TYPE_BIN) {
                SHCe = 1;
            }
            error = ERR_OK ;
        }

    } while (0) ;
    error_tmp = error ;

    /* Close the source file */
    if (srcFd != NULL) {
        error = fclose (srcFd) ;
        if (error != ERR_OK) {
            printf ("%s: Can't close the source file\n", argP) ;
            error_tmp = ERR_FILE ;
        }
    }
    error = error_tmp ;
    return (error);
}


void PrintHelp (char* argH)  /* 1         2         3         4         5         6         7         8 */
{
  if (strcmp (argH, "l")==0 || strcmp (argH, "level")==0 || strcmp (argH, "debug")==0) {
                       /* 1         2         3         4         5         6         7         8 */
              /* 12345678901234567890123456789012345678901234567890123456789012345678901234567890 */
	printf ("-d, --device=TYPE : INV interface with inverting level converter (mirror)\n") ;
	printf ("-l, --level=VALUE : Option bits and Print debug traces\n") ;
	printf ("                    a hexadecimal integer (0x____) or sum of it\n") ;
	printf (" Waveform and frequency (default sample rate is 4* base frequency):\n") ;
	printf ("    1   Force triangle waveform for base frequency (old compact format)\n") ;
	printf ("    2   Force wave with 48 kHz (PC-1500: 44.1) near rectangle waveform\n") ;
	printf ("    3   Force wave with sample rate of 16 kHz for emulator and so on\n\n") ;
	printf (" Convert Data variables between series:\n") ;
	printf (" 0x04   Convert PC-1500/1600 numeric data to other PC standard variable,\n") ;
	printf ("        otherwise to numeric array,\n") ;
	printf (" 0x08   Data for PC-1500/1600 of length 8 are numeric data from other PC\n") ;
	printf (" 0x10   Convert Strings between ASCII code and Old Basic Code\n\n") ;
	printf (" 0x1000 Use tape format of PC-1475 (slow) for E5-series CLOAD@ of old images\n") ;
	printf (" 0x4000 Write no file header, have to merge data blocks manually\n") ;
	printf (" 0x8000 Data variable block is from Wav2Bin 1.5 or version before\n") ;
	printf (" 0x800  Write also, if checksum bug will be activated (not readable)\n") ;
	printf (" 0x400  Write long synchronisation like the original, 0x200 like TRM\n\n") ;
	printf (" 0x80   Print some global infos more, for all bytes and (sums): 0xC0\n") ;
	printf (" 0x40   Print all bytes and (Sum_calculated) - see also Wav2bin\n") ;
	printf (" 0x20   Position and byte list, for data only\n") ;
	printf (" For more options - see the source code") ;
    /* Debug Bin2wav -l 0x20=Data only Pos/Byte 0x40=Byte 0x80=only some global Infos,
            DATA_W2B150 = 0x8000 for older IMG-DAT-Files with some Checksum,
            NO_FILE_HEAD = 0x4000 for variable blocks
            BAS_EXT_FRMT = 0x1000 debug flag, use FORMAT of BAS_EXT for E-Series
            SYNCL_STD 0x400 Use default sync, syncS, SYNCL_TRM 0x200 values like TRM for PC-16/G/E
            0x800 No ERR_SUM
            0x4 PC-1234: convert PC-1500 numeric to PC-1234 standard variable, else to numeric array
            0x8 PC-1500: ItemLen 8 = numeric from PC-1234, else text l=8 from PC-1500
            0x10 Convert Strings between Old Basic Code and ASCII code
   */
  }
  else {
              /* 12345678901234567890123456789012345678901234567890123456789012345678901234567890 */
	printf ("\nUsage: %s [Options] SrcFile(.typ) [DstFile(.wav/tap)]\n", argP) ;
	printf ("SrcFile          : Binary image file (usually created by BAS2IMG or WAV2BIN)\n") ;
	printf ("DstFile          : WAVe file (default: SrcFile.wav) or tap file\n") ;
//	printf ("Options:\n") ;
	printf ("-t, --type=TYPE   : Source file type\n") ;  /* type=VAR is as dat without file name */
	printf ("                     img  BASIC-program binary image (default) txt Text modus\n") ;
	printf ("                     bin  Binary assembly program or data      shc Transfile PC\n") ;
	printf ("                     dat  Data variable blocks (binary data)   asc ASCII Data\n") ;
	printf ("                     rsv  ReSerVe data (binary image)          bas ASCII Source\n") ;
	printf ("-p, --pc=NUMBER   : Sharp pocket computer, currently available 1211, 1245, 1251\n") ;
	printf ("                     1261, 1280, 1350, 1360, 1401, 1402, 1403, 1421, 1450, 1460\n") ;
	printf ("                     1475, 1600, E500, E220, G850 and more (default: 1500)\n") ;
	printf ("-c, --cspeed=VALUE: Ratio of CPU frequency to original (use it with a modified\n") ;
	printf ("                     Pocket Computer with speedup switched on, 0.94 to 2.7)\n") ;
	printf ("-a, --addr=VALUE  : 1. Start address, needed for BIN type, 2. Entry address\n") ;
	printf ("                     0 to 65535 or 0xFFFF, E500:0xFFFFFF (dflt: Manual. 2. no)\n") ;
	printf ("-s, --sync=VALUE  : Synchronisation duration, expressed in seconds, 2. Space\n") ;
	printf ("                     0.5 to 9 (default: 0.5 or minimum for the PC and waveform)\n") ;
	printf ("-nNAME, --name=   : Sharp file name (7 characters max, 16 for the PC-1500, E:8)\n") ;
	printf ("                     (default: DstFile without extension, nor path)\n") ;
	printf ("-q, --quiet       : Quiet mode (minimal display output)\n") ;
	printf ("    --tap         : Destination file: Emulator tap byte format (not wave file)\n") ;
	printf ("    --version     : Display version information\n") ;
	printf ("    --help        : Display this information,  --help=l : show option screen\n") ;
    printf ("-l, --level=VALUE : Print debug traces, more options, see help option screen") ;
  }
  #ifdef __APPLE__
    /* Mac specific here, not for _WIN32 */
    printf ("\n") ;
  #endif
  #ifdef __linux__
    /* For Linux shell */
    printf ("\n") ;
  #endif
  exit( EXIT_SUCCESS ); /* no arguments were passed */
}


void PrintVersion (void)
{   char argPU[cLPF] = "" ;
	strcpy(argPU, argP) ;
	printf ("%s (%s) version: 2.0.0\n", argP, strupr(argPU) ) ;
	printf ("Author: Pocket -> www.pocketmuseum.com\n") ; /* Please do not remove */
	printf ("        2013-2015 Torsten Muecker\n") ;       /* Please do not remove */
	printf ("        for complete list see the manual and the source code\n") ;
	printf ("This is free software. There is NO warranty;\n") ;
    printf ("not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n") ;
	exit( EXIT_SUCCESS ); /* no arguments were passed */
}


void MoreInfo (int error)
{	printf("%s: '%s --help' gives you more information\n", argP, argP);
	exit( error ); /* no arguments were passed */
}


int main ( int argc, char **argv )  /* (int argc, char* argv[]) */
{	/* 0=SrcFile 1=[DstFile] 2=[-t] 3=[-p] 4=[-d] 5=[-c] 6=[-a](start) 7=[-a](entry) 8=[-s] 9=[-s]space 10=[-n] 11=[-h] 12=[-l] 13=[-l|] */
	  char  argD[14][cLPF] = { "", "", "img", "1500", "std", "1.00", "0x000000", "0xFFFFFF", "0.5", "2.125", "", "", DEBUG_ARG, "0" } ;
	  char  argS[cLPF] = "", *ptrToken, *ptrErr, tmp ;
	  uint  FILEcnt = 0, Tcnt = 0, PCcnt = 0, Dcnt = 0, Ccnt = 0, Acnt = 0, Ncnt = 0, Hcnt = 0, Lcnt = 0 ; // to global: Qcnt, TAPc, Scnt
	 uchar  type ;
    ushort  grpId = pcgrpId ; // to global: pcId, pcgrpId
	 ulong  addr, eaddr, debug = 0 ;
    double  sync, syncS ;
	   int  option_index, i, j, k, l, error = ERR_OK, c = 0 ;
static int  longval ;
      bool  new_arg = false, old_arg = false ; // to global: cnvstr_upr = false

 const int  Token_Nb = 5 ;
     char*  oldToken[] = { "PC:", "T:", "A:", "S:", "N:" } ; /* TOKEN_NB */
     char*  newToken[] = { "-p" , "-t", "-a", "-s", "-n" } ; /* strlen 2 only */

        struct option long_options[] =
	{
		{"type",    required_argument, 0,        't'},
		{"pc",	    required_argument, 0,        'p'},
		{"device",	optional_argument, 0,        'd'}, /*option device needed, for inverting interfaces*/
		{"cspeed",	required_argument, 0,        'c'},
		{"addr",    required_argument, 0,        'a'},
		{"sync",    required_argument, 0,        's'},
		{"name",    optional_argument, 0,        'n'}, /* spaces as delimiter not allowed with opt_arg */
		{"tap",     no_argument,       0,        'r'},
		{"quiet",   no_argument,       0,        'q'},
        {"level",   required_argument, 0,        'l'},
		{"version", no_argument,       &longval, 'v'}, /* long option only */
		{"help",    optional_argument, &longval, 'h'}, /* long option only, delimiter must `=` */
	        {0, 0, 0, 0}
	};

    /* ProgramName */
    if      (strrchr (argv[0], '\\')) strncpy(argP, 1 + strrchr (argv[0], '\\'), cLPF-1);  /* Windows path separator '\' */
    else if (strrchr (argv[0], '/')) strncpy(argP, 1 + strrchr (argv[0], '/'), cLPF-1);    /* Linux   path separator '/' */
    else strncpy(argP, argv[0], cLPF-1);
    if ( strrchr (argP, '.')) *(strrchr (argP, '.')) = '\0';                      /* Extension separator '.'    */


    /* check, if the old argument format is used */
    for (i = 1; i < argc; ++i) {// 1. argument is program
        if ( *argv[i] == '-' ) {
                new_arg = true ;
                break ;
        }
        strncpy (argS, argv[i], cLPF -1) ;
        strupr(argS) ;
        ptrToken = strstr (argS, "PC:") ; /* PC: */
        if (ptrToken == argS) {
                old_arg = true ;
                break ;
        }
        if ( strcmp(argS, "T:IMG") == 0 ) old_arg = true ;
        if ( strcmp(argS, "T:BIN") == 0 ) old_arg = true ;
    }
    if ( !new_arg && old_arg) {
        printf("%s: Old format of arguments was detected", argP);
        for (i = 2; i < argc; ++i) { // 1. argument is program, 2. a file name
            strncpy (argS, argv[i], cLPF -1) ;
            strupr(argS) ;
            for ( j = 0 ; j < Token_Nb ; ++j ) { // old TOKEN_NB /
                ptrToken = strstr (argS, oldToken[j]) ;
                if (ptrToken == argS) { // replace on old argument token /
                    for ( k = 0 ; k < 2 ; ++k ) { // new_token length 2 /
                        argv[i][k] = newToken[j][k] ;
                    }  // next char
                    k = 2 ;
                    l = strlen (oldToken[j]) ;
                    if (l > k) { // shift argument content to left /
                        do { argv[i][k++] = argv[i][l]; }
                        while (argv[i][l++]!= 0) ;
                    }
                    ++c;
                    break ;
                } // END if old token found
            } // next old token
        } // next argv
        printf(" and %i arguments converted!\n", c);
    } // END if old argv

    do
    {
        error = ERR_OK ;
        addr  = 0 ;

        while (1) {

            /* getopt_long stores the option index here. */
            option_index = 0;

            c = getopt_long (argc, argv, "t:p:d::c:a:s:n::rql:vh::", long_options, &option_index);

            /* Detect the end of the options. */
            if (c == -1) break;

            switch (c)
            {
            case 't': strncpy( argD[2], optarg, cLPF-1); ++Tcnt; break;
            case 'p': strncpy( argD[3], optarg, cLPF-1); ++PCcnt; break;
            // case 'd': strncpy( argD[4], optarg, cLPF-1); ++Dcnt; break;
            case 'd': { if ( optarg != 0 ) strncpy( argD[4], optarg, cLPF-1);
                    ++Dcnt; break; }
            case 'c': strncpy( argD[5], optarg, cLPF-1); ++Ccnt; break;
            case 'a': { if (Acnt < 2) strncpy( argD[6+Acnt], optarg, cLPF-1);
                    ++Acnt; break; }
            case 's': { if (Scnt < 2) strncpy( argD[8+Scnt], optarg, cLPF-1);
                    ++Scnt; break; }
            case 'n': { if ( optarg != 0 ) strncpy( argD[10], optarg, cLPF-1);
                    ++Ncnt; break; }
            case 'r': ++TAPc; break;
            case 'q': ++Qcnt; break;
            case 'l': { if (Lcnt < 2) strncpy( argD[12+Lcnt], optarg, cLPF-1);
                    ++Lcnt; break; }
            case 0:
                switch (longval) {
                case 'v': PrintVersion (); break;
                // case 'h': PrintHelp (); break;
                case 'h': { if ( optarg != 0 ) strncpy( argD[11], optarg, cLPF-1);
                        ++Hcnt; break; }
                } break;
            case '?':
                    printf("%s: Unknown argument for '%s'\n", argP, argP);
            default : MoreInfo (ERR_SYNT); break;
            }
        }

        if (optind < argc) { /* get non-option ARGV-elements */
            while (optind < argc) {
                strncpy(argD[FILEcnt!=0], argv[optind++], cLPF-1);
                ++FILEcnt;
            }
        }

        if ((FILEcnt > 2) || (Tcnt > 1) || (PCcnt > 1) || (Dcnt > 1) || (Ccnt > 1) || (Acnt > 2) ||
               (Scnt > 2) || (Ncnt > 1) || (TAPc > 1)  || (Qcnt > 1) || (Lcnt > 2) || (Hcnt > 1)) {
                printf("%s: To much arguments of same type for '%s'\n", argP, argP);
                MoreInfo (ERR_SYNT);
        }

        if ( Hcnt == 1 ) {
            (void) strlor (argD[11]) ;
            PrintHelp (argD[11]) ;
            break;
        }

        if (FILEcnt < 1) { printf("%s: Missing Operand after '%s'\n", argP, argP); MoreInfo (ERR_SYNT); }

        if (FILEcnt == 2){
            ptrToken = strrchr (argD[1], '.') ;
            if (ptrToken != NULL) {
                strncpy (argS, ptrToken, cLPF -1) ;
                strlor(argS) ;
                if (strcmp (argS, ".tap") == 0) {
                    if (TAPc==0 && Qcnt==0) printf("%s: Switched output format from wav to --tap implicitly\n", argP);
                    TAPc |= 1;
                }
            }
        }
        ptrToken = strrchr (argD[0], '.') ;
        if (FILEcnt == 1) {
            if (ptrToken != NULL) strncat (argD[1], argD[0], strrchr (argD[0], '.') - argD[0] ); /* GetSrcFile */
            else strncpy (argD[1], argD[0], cLPF -1) ;
            if (TAPc > 0) strcat (argD[1], ".tap" );
            else strcat (argD[1], ".wav" );  /* DstFile=SrcFile.wav */
        }
        if (ptrToken != NULL) {
            (void) strncpy (argS, ptrToken + 1, cLPF -1) ;
            (void) strupr (argS) ;
        }
        else strcpy (argS, "");

        (void) strupr (argD[2]) ;
        type = TYPE_NOK ;

        if (Tcnt == 0) {
            if (strcmp (argS, "ASC") == 0) type = TYPE_ASC ;
            else if (strcmp (argS, "BAS") == 0) type = TYPE_BAS ;
            else if (strcmp (argS, "RSV") == 0) type = TYPE_RSV ;
            else if (strcmp (argS, "BIN") == 0) type = TYPE_BIN ;
            else if (strcmp (argS, "DAT") == 0) type = TYPE_DAT ;
            if (Qcnt == 0 && type != TYPE_NOK) printf ("%s: Source file --type=%s was set automatically\n", argP, argS) ;
        }
        if (Tcnt > 0 || type == TYPE_NOK) {

            if ((strcmp (argS, "SHC") == 0 && Tcnt == 0) ||
                strcmp (argD[2], "SHC") == 0) { /* Read from SHC-format, header included */
                error = ReadHeadFromShc (argD[0], &type, &addr, argD[10] ); /* sets global SHCc, SHCe, pcgrpId also */
                if (error != ERR_OK) break ;
            }
            else if (strcmp (argD[2], "IMG") == 0) type = TYPE_IMG ; /* default argument*/
            else if (strcmp (argD[2], "TXT") == 0) type = TYPE_TXT ;
            else if (strcmp (argD[2], "ASC") == 0) type = TYPE_ASC ;
            else if (strcmp (argD[2], "BAS") == 0) type = TYPE_BAS ;
            else if (strcmp (argD[2], "RSV") == 0) type = TYPE_RSV ;
            else if (strcmp (argD[2], "BIN") == 0) type = TYPE_BIN ;
            else if (strcmp (argD[2], "DAT") == 0) type = TYPE_DAT ;

            else if (strcmp (argD[2], "VAR") == 0) /* Variable DATA BLOCK without file type and -name */
                type = TYPE_VAR ;
        }
        if (type == TYPE_NOK) {
            printf ("%s: Source file type %s is not valid\n", argP, argD[2]) ;
            MoreInfo (ERR_SYNT);
            break ;
        }

        /* Convert debug in a long */
        debug = (ulong) strtol (argD[12], &ptrErr, 0) ;
        if (*ptrErr != 0) {
            debug = 0 ;
            printf ("%s: Convert debug level number from '%s' is not valid\n", argP, argD[12]) ;
            MoreInfo (ERR_ARG);
        }
        debug = debug | (ulong) strtol (argD[13], &ptrErr, 0) ;
        if (*ptrErr != 0) {
            debug = 0 ;
            printf ("%s: Convert debug level number2 from '%s' is not valid\n", argP, argD[13]) ;
            MoreInfo (ERR_ARG);
        }

        i = 3 ;
        /* Compare the PC Ident to the allowed tokens */
        if (strlen (argD[i]) == 0) {
            pcId = 1500 ;      /* default pcId */
        }
        else {
            strupr (argD[i]) ;
                 if (strcmp (argD[i], "1100")   == 0) strcpy (argD[i],  "1245") ;
            else if (strcmp (argD[i], "PA-500") == 0) strcpy (argD[i],  "1150") ;
            else if (strcmp (argD[i], "EL-6300")== 0) strcpy (argD[i],  "1150") ;
            else if (strcmp (argD[i], "6300")   == 0) strcpy (argD[i],  "1150") ;
            else if (strcmp (argD[i], "1110")   == 0) strcpy (argD[i],  "1245") ;
            else if (strcmp (argD[i], "1140")   == 0) strcpy (argD[i],  "1150") ;
            else if (strcmp (argD[i], "1210")   == 0) strcpy (argD[i],  "1211") ;
            else if (strcmp (argD[i], "1210H")  == 0) strcpy (argD[i],  "1211") ;
            else if (strcmp (argD[i], "1212")   == 0) strcpy (argD[i],  "1211") ;
            else if (strcmp (argD[i], "1246S")  == 0) strcpy (argD[i],  "1248") ;
            else if (strcmp (argD[i], "1246DB") == 0) strcpy (argD[i],  "1248") ;
            else if (strcmp (argD[i], "1248DB") == 0) strcpy (argD[i],  "1248") ;
            else if (strcmp (argD[i], "LAMBDA2")== 0) strcpy (argD[i],  "1248") ;
            else if (strcmp (argD[i], "RAMUDA10")== 0) strcpy (argD[i], "1248") ;
            else if (strcmp (argD[i], "1250A")  == 0) strcpy (argD[i],  "1250") ;
            else if (strcmp (argD[i], "1251H")  == 0) strcpy (argD[i],  "1251") ;
            else if (strcmp (argD[i], "1252H")  == 0) strcpy (argD[i],  "1251") ;
            else if (strcmp (argD[i], "1252")   == 0) strcpy (argD[i],  "1251") ;
            else if (strcmp (argD[i], "1253H")  == 0) strcpy (argD[i],  "1251") ;
            else if (strcmp (argD[i], "1253")   == 0) strcpy (argD[i],  "1251") ;
            else if (strcmp (argD[i], "1270")   == 0) strcpy (argD[i],  "1248") ;
            else if (strcmp (argD[i], "1260H")  == 0) strcpy (argD[i],  "1260") ;
            else if (strcmp (argD[i], "1260J")  == 0) strcpy (argD[i],  "1260") ;
            else if (strcmp (argD[i], "1261J")  == 0) strcpy (argD[i],  "1261") ;
            else if (strcmp (argD[i], "1262J")  == 0) strcpy (argD[i],  "1262") ;
            else if (strcmp (argD[i], "1285")   == 0) strcpy (argD[i],  "1280") ;
            else if (strcmp (argD[i], "1350J")  == 0) strcpy (argD[i],  "1350") ;
            else if (strcmp (argD[i], "1360J")  == 0) strcpy (argD[i],  "1360") ;
            else if (strcmp (argD[i], "1360K")  == 0) strcpy (argD[i],  "1360") ;
            else if (strcmp (argD[i], "1365")   == 0) strcpy (argD[i],  "1360") ;
            else if (strcmp (argD[i], "1365K")  == 0) strcpy (argD[i],  "1360") ;
            else if (strcmp (argD[i], "1403H")  == 0) strcpy (argD[i],  "1403") ;
            else if (strcmp (argD[i], "1404G")  == 0) strcpy (argD[i],  "1401") ;
            else if (strcmp (argD[i], "1405G")  == 0) strcpy (argD[i],  "1401") ;
            else if (strcmp (argD[i], "1415G")  == 0) strcpy (argD[i],  "1401") ;
            else if (strcmp (argD[i], "1416G")  == 0) strcpy (argD[i],  "1440") ;
            else if (strcmp (argD[i], "1417G")  == 0) strcpy (argD[i],  "1445") ;
            else if (strcmp (argD[i], "1450J")  == 0) strcpy (argD[i],  "1450") ;
            else if (strcmp (argD[i], "1460J")  == 0) strcpy (argD[i],  "1460") ;
            else if (strcmp (argD[i], "1470U")  == 0) strcpy (argD[i],  "1475") ;
            else if (strcmp (argD[i], "1475J")  == 0) strcpy (argD[i],  "1475") ;
            else if (strcmp (argD[i], "1500D")  == 0) strcpy (argD[i],  "1500") ;
            else if (strcmp (argD[i], "1500J")  == 0) strcpy (argD[i],  "1500") ;
            else if (strcmp (argD[i], "1500A")  == 0) strcpy (argD[i],  "1501") ;
            // else if (strcmp (argD[i], "1501")   == 0) strcpy (argD[i],  "1500") ;
            else if (strcmp (argD[i], "150")    == 0) strcpy (argD[i],  "1500") ;    /* CE-150 */
            else if (strcmp (argD[i], "2500")   == 0) strcpy (argD[i],  "1350") ;
            else if (strcmp (argD[i], "EL-5400")== 0) strcpy (argD[i],  "1430") ;
            else if (strcmp (argD[i], "5400")   == 0) strcpy (argD[i],  "1430") ;
            else if (strcmp (argD[i], "EL-5500III")==0) strcpy (argD[i],"1403") ;
            else if (strcmp (argD[i], "5500III")== 0) strcpy (argD[i],  "1403") ;
            else if (strcmp (argD[i], "EL-5500II")== 0) strcpy (argD[i],"1401") ;
            else if (strcmp (argD[i], "5500II") == 0) strcpy (argD[i],  "1401") ;
            else if (strcmp (argD[i], "EL-5500")== 0) strcpy (argD[i],  "1401") ;
            else if (strcmp (argD[i], "5500")   == 0) strcpy (argD[i],  "1401") ;
            else if (strcmp (argD[i], "EL-5510")== 0) strcpy (argD[i],  "1421") ;
            else if (strcmp (argD[i], "5510")   == 0) strcpy (argD[i],  "1421") ;
            else if (strcmp (argD[i], "EL-5520")== 0) strcpy (argD[i],  "1450") ;
            else if (strcmp (argD[i], "5520")   == 0) strcpy (argD[i],  "1450") ;
            else if (strcmp (argD[i], "PTA-4000+16")== 0) strcpy (argD[i], "1500") ; /* Hiradas Technika */
            else if (strcmp (argD[i], "PTA-4000")== 0) strcpy (argD[i], "1500") ;
            else if (strcmp (argD[i], "MC-2200") == 0) strcpy (argD[i], "1245") ;    /* Seiko */
            else if (strcmp (argD[i], "2200")   == 0) strcpy (argD[i],  "1245") ;
            else if (strcmp (argD[i], "34")     == 0) strcpy (argD[i],  "1250") ;    /* Tandy*/
            else if (strcmp (argD[i], "31")     == 0) strcpy (argD[i],  "1250") ;
            else if (strcmp (argD[i], "TRS-80PC-1")==0) strcpy (argD[i],"1210") ;
            else if (strcmp (argD[i], "1")      == 0) strcpy (argD[i],  "1210") ;
            else if (strcmp (argD[i], "TRS-80PC-2")==0) strcpy (argD[i],"1500") ;
            else if (strcmp (argD[i], "2")      == 0) strcpy (argD[i],  "1500") ;
            else if (strcmp (argD[i], "TRS-80PC-3A")==0)strcpy (argD[i],"1251") ;
            else if (strcmp (argD[i], "TRS-80PC-3")==0) strcpy (argD[i],"1251") ;
            else if (strcmp (argD[i], "TRS-80PC-8")==0) strcpy (argD[i],"1246") ;
            else if (strcmp (argD[i], "3A")     == 0) strcpy (argD[i],  "1251") ;
            else if (strcmp (argD[i], "3")      == 0) strcpy (argD[i],  "1251") ;
            else if (strcmp (argD[i], "8")      == 0) strcpy (argD[i],  "1246") ;
            else if (strcmp (argD[i], "1600K")  == 0
                 ||  strcmp (argD[i], "1605K")  == 0
                 ||  strcmp (argD[i], "1600P")  == 0) {              /* (1609) CE-1600P, PC-1600 Mode 0 */
/*               ||  strcmp (argD[i], "1600")   == 0) {
                 if (Qcnt == 0) printf ("\n%s: Only the compatibility mode 1 'PC-1500' is supported with PC-%s.\n", argP, argD[i]) ;
                                                      strcpy (argD[i],  "1500") ; */
                                                      strcpy (argD[i],  "1600") ;
            }
            else if (strcmp (argD[i], "1600M1") == 0) strcpy (argD[i],  "1500") ;    /* (1601) CE-150, PC-1600 Mode 1 */
            else if (strcmp (argD[i], "E220")   == 0
                 ||  strcmp (argD[i], "220")    == 0
                 ||  strcmp (argD[i], "E200")   == 0
                 ||  strcmp (argD[i], "200")    == 0
                                                    ) {
                 if (type == TYPE_IMG && (debug & BAS_EXT_FRMT)>0 ) {
                    if (Qcnt == 0) { printf ("\n%s: Use 'CLOAD@' to read this tape format of PC-1460 at PC-%s.\n", argP, argD[i]) ;
                                     printf ("         You MUST create such an BASIC image with Bas2img --pc=1460 implicitly!\n") ;
                                     printf ("         Old tape format selected for PC-E200 series, for files from Bas2img.\n") ;
                    }
                                                      strcpy (argD[i],  "1460") ; /* NOT the native tape format of PC-E2  */
                 }
                 else {
                    if (type == TYPE_IMG && debug > 0x1F && Qcnt == 0) {
                                     printf ("\n%s: If the source file was made by Bas2img and syntax errors are found\n", argP) ;
                                     printf ("          you can convert it with Text editor to 'TEXT' and back to 'BASIC'.\n\n");
                    }
                                                      strcpy (argD[i],   "220") ;
                 }
            }
            else if (strcmp (argD[i], "G801")   == 0
                 ||  strcmp (argD[i], "801")    == 0
                 ||  strcmp (argD[i], "G802")   == 0
                 ||  strcmp (argD[i], "802")    == 0
                 ||  strcmp (argD[i], "G803")   == 0
                 ||  strcmp (argD[i], "803")    == 0
                 ||  strcmp (argD[i], "G805")   == 0
                 ||  strcmp (argD[i], "805")    == 0
                 ||  strcmp (argD[i], "G811")   == 0
                 ||  strcmp (argD[i], "811")    == 0
                 ||  strcmp (argD[i], "G813")   == 0
                 ||  strcmp (argD[i], "813")    == 0
                 ||  strcmp (argD[i], "G815")   == 0
                 ||  strcmp (argD[i], "815")    == 0
                 ||  strcmp (argD[i], "G820")   == 0
                 ||  strcmp (argD[i], "820")    == 0
                 ||  strcmp (argD[i], "G830")   == 0
                 ||  strcmp (argD[i], "830")    == 0
                 ||  strcmp (argD[i], "G850VS") == 0
                 ||  strcmp (argD[i], "G850V")  == 0
                 ||  strcmp (argD[i], "G850S")  == 0
                 ||  strcmp (argD[i], "G850")   == 0
                 ||  strcmp (argD[i], "850")    == 0
                                                    ) {
                    if (Qcnt == 0 && debug > 0x1F && type == TYPE_IMG ) {
                                     printf ("\n%s: If the source file was made by Bas2img and syntax errors are found\n", argP) ;
                                     printf ("          you can convert it with Text editor to 'TEXT' and back to 'BASIC'.\n\n");
                    }
                                                      strcpy (argD[i],   "850") ;
            }
            else if (strcmp (argD[i], "E500S")  == 0
                 ||  strcmp (argD[i], "E500")   == 0
                 ||  strcmp (argD[i], "500")    == 0
                 ||  strcmp (argD[i], "E550")   == 0
                 ||  strcmp (argD[i], "550")    == 0
                 ||  strcmp (argD[i], "E650")   == 0
                 ||  strcmp (argD[i], "650")    == 0
                 ||  strcmp (argD[i], "U6000II")== 0
                 ||  strcmp (argD[i], "U6000")  == 0
                 ||  strcmp (argD[i], "6000")   == 0
                 ||  strcmp (argD[i], "1490UII")== 0
                 ||  strcmp (argD[i], "1490U")  == 0
                 ||  strcmp (argD[i], "1490")   == 0
                 ||  strcmp (argD[i], "1480U")  == 0
                 ||  strcmp (argD[i], "1480")   == 0
                                                    ) {
                 if (type == TYPE_IMG && (debug & BAS_EXT_FRMT)>0 ) {
                    if (Qcnt == 0) { printf ("\n%s: Use 'CLOAD @' to read this format of PC-1475 on PC-%s.\n", argP, argD[i]) ;
                                     printf ("         There are no particular options for Bas2img necessary.\n") ;
                                     printf ("         Old tape format was selected for PC-E500, for files from Bas2img.\n") ;
                    }
                                                      strcpy (argD[i],  "1475") ; /* NOT the native tape format of PC-E500  */
                 }
                    /*  BASE_FREQ3 2990/1230 Hz (0/1), 2 transmissions/bit with variable length (1T) 1A 1T 1A (1T),
                        Byte = 1 start bit1 + data bit 7...0,
                        1. Header Block = Sync_b0/40b1/40b0/1b1/File ID + Filename + header = 30B + CSum, Pause,
                        see also SHARP: Technical Reference Manual PC-E500, P64-66
                        2. Data Block = Sync_b0/20b1/20b0/1b1/ FF + 7 Header Bytes + 10 Bytes 0, 0D, Program Data
                        Block length 256? bytes ... 0D FF cs 1A */
                 else {
                    if (type == TYPE_IMG && Qcnt == 0) {
                                     printf ("\n%s: If the source file was made by Bas2img you have to switch to 'TEXT',\n", argP) ;
                                     printf ("          back to 'BASIC' on PC-%s or use old tape format with '-l 0x%05X'.\n", argD[i], BAS_EXT_FRMT);
                                     printf ("          You can transfer Text lines with -t TXT for Bas2Img and here IMG.\n\n");
                    }
                                                      strcpy (argD[i],   "500") ; /* The native tape format of PC-E500  */
                 }
            }
            pcId = (ulong) strtol (argD[i], &ptrErr, 0) ;
            if (pcId == 0) {
                printf ("%s: Pocket computer %s is not valid\n", argP, argD[i]);
                MoreInfo (ERR_ARG);    // exit (ERR_SYNT) ;
                break ;
            }
        }
        switch (pcId) {
        case 1211 :
            {   if (Acnt==0 && addr==0 && type == TYPE_BIN) addr = 0xFFFF;
                grpId=IDENT_PC1211; cnvstr_upr = true ; break; }

        case 1150 :
        case 1246 :
        case 1247 :
        case 1248 :
            if (Acnt==0 && addr==0 && type == TYPE_BIN) addr = 0xFFFF;
        case 1245 :
        case 1250 :
        case 1251 :
        case 1255 :
            {   if (Acnt==0 && addr==0 && type == TYPE_BIN) addr = 0xB830;
                grpId=GRP_OLD; cnvstr_upr = true ; break; }

        case 1430 :
        case 1431 :
            if (Acnt==0 && addr==0 && type == TYPE_BIN) addr = 0xFFFF;
        case 1421 :
        case 1440 : /*Memory map unknown*/
        case 1401 :
            if (Acnt==0 && addr==0 && type == TYPE_BIN) addr = 0x3800;
        case 1402 :
            if (Acnt==0 && addr==0 && type == TYPE_BIN) addr = 0x2000;
            cnvstr_upr = true ;

        case 1260 :
            if (Acnt==0 && addr==0 && type == TYPE_BIN) addr = 0x5880;
        case 1261 :
        case 1262 :
            if (Acnt==0 && addr==0 && type == TYPE_BIN) addr = 0x4080;
        case 1350 :
        case 1450 :
            if (Acnt==0 && addr==0 && type == TYPE_BIN) addr = 0x2030;
        case 1445 : /*Memory map and SML unknown*/
        case 1403 :
        case 1425 :
        case 1460 :
            if (Acnt==0 && addr==0 && type == TYPE_BIN) addr = 0x8030;
            grpId = GRP_NEW ;
            break ;

        case 1280 :
        case 1360 :
        case 1475 :
            if (Acnt==0 && addr==0 && type == TYPE_BIN) addr = 0x8030;
            grpId = GRP_EXT ;
            break ;

        case 1501 :
            if (Acnt==0 && addr==0 && type == TYPE_BIN) addr = 0x7C01;
            pcId = 1500 ;
        case 1500 :
            if (Acnt==0 && addr==0 && type == TYPE_BIN) addr = 0x40C5;
            grpId = IDENT_PC1500 ;
            break ;

        case 1600 :
            if (Acnt==0 && addr==0 && type == TYPE_BIN) addr = 0xC0C5;
            grpId = GRP_16 ;
            break ;

        case  500 :
            if (Acnt==0 && addr==0 && type == TYPE_BIN) addr = 0xBE000;
            grpId = GRP_E ;
            break ;

        case  220 :
        case  850 :
            if (Acnt==0 && addr==0 && type == TYPE_BIN) addr = 0x0100;
            grpId = GRP_G ;
            break ;

        default :
            /* Newest of G-Series, G850V and newer */
            printf ("%s: Pocket computer %s is not implemented\n", argP, argD[3]) ;
            // MoreInfo (ERR_ARG);
            error = ERR_ARG ;
            // break ;
        }
        if (error != ERR_OK) break ;

        if (pcgrpId == IDENT_UNKNOWN) pcgrpId = grpId ;
        else if ( SHCc > 0 && pcgrpId != grpId ) { /* grpId from --pc parameter is different from SHC file pcgrpId*/

            if (pcgrpId == GRP_NEW && grpId == GRP_EXT && /* ID for both groups */
                (type == TYPE_BIN || type == TYPE_DAT)) pcgrpId = grpId ;

            else if ( type == TYPE_BIN ) {
                /* Data may be universally valid or nonsense mostly */
                if (Qcnt == 0) printf
                    ("%s: Type mismatch between PC-%s and Binary SHC format. Data may be not valid.\n", argP, argD[3]) ;
                pcgrpId = grpId ;
            }
            else if (grpId == IDENT_PC1211 && pcgrpId == GRP_OLD ) { //  && type != TYPE_BIN
                /* PC-1211 can use SHC data of PC-1251 without data header */
                pcgrpId = grpId ;
            }
            else if (type == TYPE_DAT && (grpId==IDENT_PC1500 || grpId==GRP_16)){
                /* Limited Conversion of other SHC data into PC-1500 data */
                if (Qcnt ==0) printf ("%s: Data is converted by SHC file format to Pocket computer %s\n", argP, argD[3]) ;
                if (pcgrpId == GRP_OLD) debug |= 0x10 ;
                debug |= 8 ;
                pcgrpId = grpId ;
            }
            else if (type == TYPE_DAT && (pcgrpId == GRP_OLD || grpId == GRP_OLD ) ) {
                /* convert ASCII data */
                if (Qcnt ==0) printf ("%s: Strings are converted to Pocket computer %s\n", argP, argD[3]) ;
                debug |= 0x10 ;
                pcgrpId = grpId ;
            }
            else if (type == TYPE_IMG && pcgrpId == GRP_NEW && (grpId == GRP_E || pcId == 220 ) ) {
                if (Qcnt ==0) printf ("%s: SHC of PC-1460 is used for PC-%s, use CLOAD@ to transfer!\n", argP, argD[3]) ;
                debug |= BAS_EXT_FRMT ;
                grpId = pcgrpId ;
            }
            else if (type == TYPE_IMG && pcgrpId == GRP_EXT && (grpId == GRP_E || pcgrpId == GRP_G) ) {
                if (Qcnt ==0) printf ("%s: SHC PC-1475 used for PC-%s, convert it via TEXT mode after transfer!\n", argP, argD[3]) ;
                pcgrpId = grpId ;
            }
            else if ( grpId != IDENT_UNKNOWN && PCcnt > 0 ) {
                printf ("%s: Type mismatch between Pocket computer %s and SHC file format\n", argP, argD[3]) ;
                MoreInfo (ERR_ARG);
            }
        }

        if ( type == TYPE_DAT && (pcgrpId == GRP_E || pcgrpId == GRP_G)) { /* Ascii files used for data */
            type = TYPE_ASC ;
            if (Qcnt == 0 )
                printf ("%s: Source file type %s for group of PC-%lu was changed to 'asc'.\n", argP, argD[2], pcId) ;
        }
        if ( type == TYPE_TXT) {
            if ( pcgrpId == GRP_16 ) type = TYPE_BAS ; /* for comments saved with SAVE* CAS: */
            else if ( pcgrpId != GRP_E && pcgrpId != GRP_EXT && pcgrpId != GRP_NEW) {
                printf ("%s: Source file type %s is for TEXT mode, not valid for group of PC-%lu \n", argP, argD[2], pcId) ;
                if (pcgrpId == GRP_G){
                    printf ("%s: Text MENU needs BAS type! Type is switched from %s to BAS now.\n", argP, argD[2]) ;
                    type = TYPE_BAS ;
                }
                else {
                    MoreInfo (ERR_ARG);
                    break ;
                }
            }
            else if ( pcgrpId == GRP_E ) { /* That CSAVEd from TEXT modus may not CLOADable*/
                printf ("%s: Source file type %s is not compatible with CLOAD for group of PC-E%lu\n", argP, argD[2], pcId) ;
                if ( (debug & 0x800) ==0 ){
                    printf ("%s: Option -l 0x800 not found: Type is switched from %s to IMG now.\n", argP, argD[2]) ;
                    type = TYPE_IMG ;
                }
            }
        }
        if ( (type == TYPE_BAS || type == TYPE_ASC) && !(pcgrpId == GRP_16 || pcgrpId == GRP_E || pcgrpId == GRP_G)) {
            printf ("%s: Src type %s (Text Menu or device CAS:)is not valid for group of PC-%lu\n", argP, argD[2], pcId) ;
            MoreInfo (ERR_ARG);
            break ;
        }
        if ((type == TYPE_ASC  && (pcgrpId == GRP_E || (pcgrpId == GRP_G && pcId >= 800))) ||
            (type == TYPE_BAS  &&  pcgrpId == GRP_E) ) { /* That SAVEd to CAS: may not LOADable*/
            if (Qcnt == 0) {
                    printf ("%s: Source type %s may not loadable from device CAS: for PC-G/E%lu\n", argP, argD[2], pcId);
                    printf ("         Try high sound level. ");
                    if (pcgrpId == GRP_E) printf ("Some interfaces need --device=INV and some not!\n");
                    else printf ("\n         Try first with a tape recorder, if your PC-G%lu can read this format.\n", pcId);
            }
        }
        if ( (type == TYPE_RSV) && (pcgrpId == GRP_E || pcgrpId == GRP_G)) {
            printf ("%s: Src type %s (ReSerVe data) is not valid for group of PC-%lu\n", argP, argD[2], pcId) ;
            MoreInfo (ERR_ARG);
            break ;
        }
        if ( (type == TYPE_BIN) && (pcgrpId == IDENT_PC1211 || (1245 < pcId && pcId < 1249) ||
                                    1140 == pcId || pcId == 1150 || pcId == 1430 || pcId == 1431)) {
            printf ("%s: Src type %s (Binary data) is not valid for 4-bit CPU of PC-%lu\n", argP, argD[2], pcId) ;
            MoreInfo (ERR_ARG);
            break ;
        }
        strlor(argD[4]) ;
        if (strcmp (argD[4], "std") == 0 || strcmp (argD[4], "150") == 0 ||
            strcmp (argD[4], "126") == 0 || strcmp (argD[4], "124") == 0 )
            error = ERR_OK ;
        else if (strcmp (argD[4], "inv") == 0 || strcmp (argD[4], "mfe") == 0)
            bitMirroring = true ;
        else if (strcmp (argD[4], "tap") == 0)
            {if (TAPc == 0) TAPc = 1; }
        else {
            printf ("%s: Property of target device '%s' is not valid\n", argP, argD[4]) ;
            MoreInfo (ERR_ARG);
        }

        /* Choose the waveform */
        switch (debug & 0x3) {
        case 1 :
            if (Qcnt == 0 ) printf ("%s: Waveform of base frequency forced to old compact triangle.\n", argP) ;
            bit = bit1 ;    /* default wave form */
            if (pcgrpId != GRP_G)
                bitE = bitE1 ;  /* asymmetric compressed wave form for E-Series*/
            else  printf ("%s: Waveform not readable with E200/G800 series, using standard.\n", argP) ;
            break ;

        case 2 :
            if (pcgrpId == IDENT_PC1500) {
                bit = bit3_15 ;
                if (Qcnt == 0 ) printf ("%s: Wave frequency forced to 44.1 kHz, near rectangle.\n", argP) ;
            }
            else {
                bit = bit3 ;
                if (Qcnt == 0 ) printf ("%s: Wave frequency forced to 48 kHz, near rectangle.\n", argP) ;
            }
            bitE = bitE3 ;
            break ;

        case 3 :
            if (Qcnt == 0 ) printf ("%s: Wave frequency forced to 16 kHz.\n", argP) ;
            if (pcgrpId == IDENT_PC1500) {
                bit = bit4_15 ;
            }
            else {
                bit = bit4 ;
            }
            bitE = bitE4 ;
            break ;

        /* case 3 : // default but mirrored /
            if (Qcnt == 0 ) printf ("%s: Waveform of base frequency forced to mirrored trapezoidal.\n", argP) ;
            bit = bit2 ;    // trapezoidal wave form /
            bitE = bitE2 ;  // default wave form for E/G Series /
            bitMirroring = !bitMirroring ; // + - mirrored /
            break ; */

        // default : see global definition
        }
        if (Qcnt == 0 && bitMirroring ) printf ("%s: Waveform will be mirrored for inverting interfaces.\n", argP) ;

        /* Check the range of CPU (or cassette) speed factor from modified hardware */
        speed = strtod (argD[5], &ptrErr) ;
        if ((*ptrErr != 0) || 0.939 > speed || speed > 2.701) {
                printf ("%s: A ratio of CPU frequency to an unmodified %s is not supported.\n", argP, argD[5]) ;
                MoreInfo (ERR_ARG);
        }
        else if (Qcnt == 0 && (float) speed != 1.0 ) printf ("%s: Option cspeed, ratio to original CPU frequency: %1.2f\n", argP, speed) ;

        if ( type == TYPE_RSV && pcgrpId != IDENT_PC1500 && pcgrpId != GRP_16 ) type = TYPE_IMG ;

        /* Convert the Address in a ulong if necessary */
        if (type == TYPE_BIN || type == TYPE_RSV) {
            if ( Acnt > 0 || (SHCc == 0 && addr == 0) ) addr = (ulong) strtol (argD[6], &ptrErr, 0) ;
            if (*ptrErr != 0) { /* was (*ptrErr != NULL) */
                tmp = *ptrErr ;
                *ptrErr = 0 ;
                printf ("%s: Start address is not valid '%s -> ", argP, argD[6]) ;
                *ptrErr = tmp ;
                printf ("%s'\n", ptrErr) ;
                MoreInfo (ERR_ARG);
                break ;
            }
            if (pcgrpId == GRP_E || pcgrpId == GRP_G || pcgrpId == GRP_16 || pcgrpId == IDENT_PC1500) {

                if ( addr > 0xFFFFFF || (addr > 0xFFFF &&
                                        (pcgrpId == GRP_G || pcgrpId == IDENT_PC1500))) {
                    printf ("%s: Start address '%s' is out of range\n", argP, argD[6]) ;
                    MoreInfo (ERR_ARG);
                    break ;
                }

                eaddr = (ulong) strtol (argD[7], &ptrErr, 0) ;
                if (*ptrErr != 0) { /* was (*ptrErr != NULL) */
                    tmp = *ptrErr ;
                    *ptrErr = 0 ;
                    printf ("%s: Entry address is not valid '%s -> ", argP, argD[7]) ;
                    *ptrErr = tmp ;
                    printf ("%s'\n", ptrErr) ;
                    MoreInfo (ERR_ARG);
                    break ;
                }
                if (eaddr == 0xFFFFFF && pcgrpId == IDENT_PC1500) eaddr = 0xFFFF;
                if ( eaddr > 0xFFFFFF || (eaddr > 0xFFFF && eaddr < 0xFFFFFF &&
                                         (pcgrpId == GRP_G || pcgrpId == IDENT_PC1500))) {
                    printf ("%s: Entry address '%s' is out of range\n", argP, argD[7]) ;
                    MoreInfo (ERR_ARG);
                    break ;
                }
                if (pcgrpId == GRP_G && eaddr != 0xFFFFFF && Qcnt == 0)
                    printf ("%s: No autostart, type> MON and *G%04X to start this program!\n", argP, (uint) (eaddr & 0xFFFF)) ;
                /* if ( pcgrpId == GRP_16 &&
                     type == TYPE_RSV && addr == 0) addr = 0xC008 ;  RSV addr PC-1600 */
            }
            else {
                if ( addr > 0xFFFF ) { // || (addr < 0)
                    printf ("%s: Start address '%s' is out of range\n", argP, argD[6]) ;
                    MoreInfo (ERR_ARG);
                    break ;
                }
                if (type == TYPE_RSV && addr == 0) addr = 0x4008 ; /* RSV addr PC-1500 */
            }
            if (type==TYPE_BIN && addr==0 && Qcnt == 0)
                    printf ("%s: Load address '0' found. Use CLOAD M with the correct start address!\n", argP);
        }

        /* Convert the Sync length in a double */
        sync = strtod (argD[8], &ptrErr) ;
        if (*ptrErr != 0) { /* was (*ptrErr != NULL) */
            tmp = *ptrErr ;
            *ptrErr = 0 ;
            printf ("%s: Sync length is not valid '%s -> ", argP, argD[8]) ;
            *ptrErr = tmp ;
            printf ("%s'\n", ptrErr) ;
            MoreInfo (ERR_ARG);
            break ;
        }
        syncS = strtod (argD[9], &ptrErr) ;
        if (*ptrErr != 0) { /* was (*ptrErr != NULL) */
            tmp = *ptrErr ;
            *ptrErr = 0 ;
            printf ("%s: Sync space (silence) length is not valid '%s -> ", argP, argD[9]) ;
            *ptrErr = tmp ;
            printf ("%s'\n", ptrErr) ;
            MoreInfo (ERR_ARG);
            break ;
        }

        if ( Scnt == 0 ) {      /* change default sync length */
            if ((debug & (SYNCL_STD | SYNCL_TRM))>0 ) {
                switch (pcgrpId) {
                    case GRP_G :
                    case GRP_16 :
                        sync = 3.5 ; break ;
                    case GRP_E :
                        sync = 4.0 ; break ;
                    case IDENT_PC1211 :
                        sync = 5.0 ; break ;
                    default :
                        sync = 8.0 ; break ;
                }
            }
            else if (pcgrpId == GRP_G || pcgrpId == GRP_16 ||  /* min 1.7 sec */
                     pcgrpId == GRP_E) sync = 2.125 ; /* change default value */
        }
        if  ( Scnt < 2 && ( pcgrpId == GRP_E || pcgrpId == GRP_16 ||
                            pcgrpId == GRP_G) ) {
            if ((debug & (SYNCL_STD | SYNCL_TRM))==0 )
                Scnt = 3 ; /* use default values from Bin2wav, not from TRM */
        }

        if ( (sync < 0.4999) ||
             (sync > 10.0001) ) {
            printf ("%s: Sync length '%s' is out of range 0.5 - 10\n", argP, argD[8]) ;
            MoreInfo (ERR_ARG);
            break ;
        }
        if ( (syncS < 1.6999) ||
             (syncS > 10.0001) ) {
            printf ("%s: Sync space (silence) length '%s' is out of range 1.7 - 10\n", argP, argD[9]) ;
            MoreInfo (ERR_ARG);
            break ;
        }
        /* Check if the name-option is used and no name is defined */
        if ( ( Ncnt == 1 ) && ( strlen (argD[10]) == 0 ) ) {
            /* Search the last colon position */
            ptrToken = strrchr (argD[1], CHAR_COLON) ;
            if (ptrToken == NULL)
                (void) strcpy (argS, argD[1]) ;
            else
                (void) strcpy (argS, ptrToken + 1) ;

            /* Search the last slash position */
            ptrToken = strrchr (argS, CHAR_SLASH) ;
            if (ptrToken == NULL)
                (void) strcpy (argD[10], argS) ;
            else
                (void) strcpy (argD[10], ptrToken + 1) ;

            /* Search the dot position */
            ptrToken = strchr (argD[10], CHAR_DOT) ;
            if (ptrToken != NULL)
                *ptrToken = 0 ;
        }
        if (pcgrpId != GRP_16 && pcgrpId != GRP_G && pcgrpId != GRP_E ) {
            /* Replace the underscore with space in the name */
            ptrToken = strchr (argD[10], CHAR_UNDERSCORE) ;
            while (ptrToken != NULL) {
                *ptrToken = CHAR_SPACE ;
                ptrToken = strchr (argD[10], CHAR_UNDERSCORE) ;
            }
        }
        if (pcgrpId == GRP_16 && strlen (argD[10]) > 0 ) {
            if (type == TYPE_ASC || type == TYPE_BAS) strupr(argD[10]) ;
            else if (type == TYPE_IMG) strncat (argD[10], ".BAS", cLPF - 1 - strlen (argD[10]) ) ;
            else if (type == TYPE_BIN) strncat (argD[10], ".BIN", cLPF - 1 - strlen (argD[10]) ) ;
        }
        // TAPc>0 special needs for tap-format could be implemented here

        if (Qcnt != 0 ) debug &= (DATA_W2B150 | NO_FILE_HEAD | BAS_EXT_FRMT | SYNCL_STD | SYNCL_TRM | 0x1F ) ;
        if (type == TYPE_VAR) {
            debug |= NO_FILE_HEAD ;
            type = TYPE_DAT ;
        }
        else if (cnvstr_upr == true) (void) strupr (argD[10]) ;          /* Uppercase the name*/
                                        /*cnvstr_upr used to adapt the byte writing mode also*/
        /* Call the Convert function */
        error = ConvertBinToWav (argD[0], argD[1], argD[2], type, addr, eaddr, sync, syncS, argD[10], debug) ;

    } while (0) ;
    if (error != ERR_OK && error != ERR_NOK) {
            if (debug != 0) printf ("Exit with error %d\n", error) ;
            return (error) ; // For debugging set here a breakpoint and in MoreInfo
    }
    else return (EXIT_SUCCESS) ;
}
