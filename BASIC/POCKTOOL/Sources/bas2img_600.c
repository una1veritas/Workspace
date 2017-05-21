/* Textcoding Unicode (UTF-8, no BOM), End of Line Windows (CRLF)

bas2img.c

2004-08-15	0.9-4.0 Joachim BOHS
	von ver 0.9 hochgezählt in v1.0 wegen UPRSTR(inputline) soll nicht die
	ganze zeile umfassen, "comments" and  "strings" sollen bleiben.
	noch zu ändern uprstr nur auf nicht "" anwenden.
	von ver 1.0 hochgezählt in v2.0 wegen USING fehlte
	von ver 2.0 hochgezählt in v3.0 wegen REM nutzte nicht line_buffer2
	von ver 3.0 hochgezählt in v4.0 wegen spaces am end of line is error in PC1500

2010-03-12	V 5.0	Norbert ROLL
	Umgestellt auf ANSI-C, u.a. strupr() ersetzt; DOS-Fkt. entfernt; zusätzliche Fehlermeldungen,
	Ausgabe auf stdout statt auf stderr; im BASIC-Text CRLF, CR oder LF möglich (Win/Mac/Unix)
	Ausgabe der eingelesenen Zeilen als Fortschrittsanzeige (z.B. bei Benutzung von DOSBox (0.73))
	gcc 4.0.1 (Mac OSX 10.4.11 auf Intel Core Duo für OSX ab 10.1 auf Intel 32 Bit und PowerPC 32 Bit)
		gcc -Wall -arch i386 -arch ppc -mmacosx-version-min=10.1 bas2img.c -o bas2img
	gcc 3.4.5-20060117-3 (MinGW, Win XP Home Edition 2002 auf Intel Celeron M für 32-Bit DOS-Box/Win)
		gcc -Wall bas2img.c -o bas2img.exe
	gcc 4.4.2 (DJGPP, Win XP Home Edition 2002 auf Intel Celeron M für 16-Bit DOS-Box mit DPMI / Win)
		gcc -Wall bas2img.c -o bas2img.exe
	dmc 8.42n (Digital Mars, Win XP Home Edition 2002 auf Intel Celeron M für 16-Bit DOS ab 8088)
		dmc -ms -d -w- -0 bas2img.c -o bas2img.exe

2011-12-05	V 5.1	Manfred NOSSWITZ
	Changed to ANSI-C e.g. strupr()
	Expanded to PC-1245, 1251 (OLD-BASIC); 1261, 1350, 1360, 1401, 1402, 1403, 1450, 1475 (NEW-BASIC).
	Ignore spaces in the beginning of lines and after REM-Commands for NEW-BASIC.
	Last character in OLD-BASIC- and NEW-BASIC-IMG-File is 0x0D.
	Last characters in 1500-IMG-File are 0x0D, 0xFF.
	Translation of π (PI), √ (SQR) and ¥ (Yen). (utf-8 <-> Sharp-ASCII)
	Command line parser changed to getopt().
	Undefined behavior of strcpy( x, x + 1 ) when source and destination strings overlap, solved.
	Functions strupr, shift_left, del_spaces and replace_str tested in 64bit-systems.
	32bit compilation with gcc (tdm64-1) 4.6.1 (WindowsXP-prof [32bit]): gcc -pedantic -m32 -o xxx xxx.c
	32bit compilation with gcc-4_5-branch revision 167585 (OpenSUSE 11.4 [32bit]): gcc -pedantic -m32 -o xxx xxx.c
	64bit compilation with gcc (tdm64-1) 4.6.1 (Windows7-prof [64bit]): gcc -pedantic -m64 -o xxx xxx.c
	64bit compilation with gcc-4_5-branch revision 167585 (OpenSUSE 11.4 [64bit]): gcc -pedantic -m64 -o xxx xxx.c
	For testing PC-1402 was available only.

2013-10-05	v 5.2	Olivier De Smet
	Added 1475 tokens

2013-11-25	v 5.2.1b	Torsten Muecker
	Added more 1360 tokens, special chars, moved 1360 to NEW_BAS2
	Delete colons following the line numbers (for compatibility with other listings)
2014-01-12	v 5.2.1b	Torsten Muecker
	Tokens corrected for NEW_BAS2, LET i.e. ,now EXT_BAS as Wav2bin
	Special chars in Strings and one-char-functions PI, SQR corrected for PC-14xx
	Comment lines beginning with ' are allowed now and will not be transfered
	Sources from TransFile PC (.SHA) with special DOS chars allowed now -
	put "." as first char in the first line to convert, rest of first line is ignored
	Leading artifact in ASC-Files from some serial transfers are ignored now
2014-09-01	v 5.3.0 beta 12b	Torsten Mücker
	Support for OLD-BAS (i.e. 1251) completed, line, conv_asc2old
	Convert lower chars in string constants to upper for PCs, that don´t support lower chars
	Separate tokens for PC-1421 added,
	2-Byte tokens for PC-1403, 1460, 1445, 1440, 1425 added (tokenL=1 and tokenL=2 mixed)
	PC-1280 and more PCs added,
	Help screen changed
	Tokens for PC-E500 added
	Handling for old Exp sign €/[E] and equal placeholders [SQR][PI][Y] for all PC groups added
	For OLD_BAS Handling of >=, <=, <> corrected
	more and precise handling of special chars, card symbols added, [INS][FUL] added
	if DOS-US Chars for √, π, € found, always will converted
    Image line maximal length check implemented
	Existing Limitation for EXT_BAS:
        NO compile of fixed numeric line number to 0x1F + binary format HL will done,
        recommended for pocketcom: Switch to "TEXT" and recompile with "BASIC" will do it
    Tokens for PC-E500S added
	Existing Limitation for E/G-BAS:
        NO compile of fixed numeric line number and mark distances, no compile of fixed numbers, no relative branch distances,
        MUST done on pocketcom: Switch to "TEXT" and recompile with "BASIC" will do it
    additional EOF mark for PC-1500 removed
    some changes in categorisation of pocket computers
    Tokens for E/G-series up to G850S and 2. REM ID "'"
    Debug levels added, no line output without l option, better to see errors and warnings,
    added old arguments conversion to new arguments for backward compatibility with 3rd-party software
    added minor tokens for old BASIC
    minor token replacements for PC-G800 series
    token for PC-1600 series added, support for merged files with linenb 99999,
    adapted check of line length
2014-09-21	v 5.3.1 beta 2b	Torsten Mücker
    --type option added, output of BASIC Text modus lines and ASCII files
    reconstruct all special chars from wav2bin
    compile fixed line numbers for PC-1600, GRP_EXT and newer series
    changes spaces after ASCII line number
    check compatibility with type ASC and TXT
2014-11-26	v 5.9.8 beta 09 Torsten Mücker
    more support of merged program files
    post-processing of G series for RemIdChar
    shortcuts for tokens
    more checks for length of tokens and lines
    E-series labels not tokenized, G850 label without :
    PC-121x tokens separated from old BASIC
2015-03-29	v 5.9.9 gamma 4 (3c1) Torsten Mücker
    PC-1421 i, n
    Debug option for preprocessor and upper cases
    Illegal line number sequence for New BASIC without auto merge mark with l=0x800
    ToDo: Public TESTs, Help needed
2015-06-08	V 6.0.0 (c4 + tokens of c4a, c4b) Torsten Mücker
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>		/* Command line parser getopt(). */
#include <ctype.h>		/* Test Characters */
/* #include <locale.h>     Problem with strupr and special chars xFA-xFE */

#define DEBUG_ARG       "0"

#define TYPE_NOK        0
#define TYPE_IMG        2
#define TYPE_ASC        6   /* For PC-E/G/1600 ASCII Data */
#define TYPE_TXT        8   /* For GRP_NEW, GRP_EXT and GRP_E text modus Image Data */

#define GRP_16          0x10    /* PC-1600 */
#define GRP_E           0x05    /* E500-series: PC-E500... for tokens and line length only */
#define GRP_G           0x08    /* G800 series: PC-G8xx, E200, E220 */

#define IDENT_UNKNOWN   0x00
#define IDENT_OLD_BAS   0x20	/* End Of File character 0x00 */
#define IDENT_NEW_BAS   0x70	/* End Of File character 0x0D */
#define IDENT_EXT_BAS   0x72	/* End Of File character 0x0D */
#define IDENT_E_BAS     0x02    /* Only extended format of PC-1475 supported */

#define IDENT_PC15_BAS  0xA1	/* End Of File characters 0x0D, 0xFF removed */
#define IDENT_PC16_BAS  0x11    /* Mode 1 ID is IDENT_E_BAS, real Mode 2 ID is 01 */

#define BAS_EXT_CODE    0xFE    /* Extended 2-Byte only */
#define BAS_EXT_LINE_NB 0x1F    /* E- or extended fixed line numbers are binary coded */

#define MERGE_MARK      99999   /* Number with empty line for mark of merged programs */

#define TOKEN_GEN        0      /* for GRP_EXT and newer line number processing */
#define TOKEN_LBL        1      /* Line number may follow this token */
#define TOKEN_LST        2      /* ON, ... */
#define TOKEN_REM        4
#define TOKEN_COL        8      /* Colon before token need */

#define true 1
#define false 0

#define cLL  512		/* Constant value for max. Length of Lines */ // old: 256
#define cLPF 129		/* Constant value for max. Length of PathFilenames */
#define cLC  255		/* Constant value for max. Length of Commands */ //old: 80

#define SOURCE_BAS  0	/* BAS default source file written by wav2bin */
#define SOURCE_SHA  1	/* SHA Source file from "TransFile PC" with one header line */

#define ERR_NOK    -1   /* also normal EOF */
#define ERR_OK      0
#define ERR_SYNT    1
#define ERR_ARG     3   /* pocket not implemented, argument problem */
#define ERR_LINE    4   /* error with line numbers, old definition: 255 */
#define ERR_FILE    5   /* File I-O */
#define ERR_MEM     6   /* Line to long, Memory */

#define COLON       0x3A

/*                      used types  def. bits    Win32 Lin32 Win64 Lin64 */
typedef int		bool;
/*			char;	    8             8     8     8     8    */
typedef unsigned char	uchar;
/*	        short;	    at least 16  16    16    16    16    */
typedef unsigned short	ushort;
/*			int;	    at least 16  32    32    32    32    */
typedef unsigned int	uint;
/*	long			    at least 32  32    32    32    64    */
typedef unsigned long	ulong;

 char  argP[cLPF] = "" ;
uchar  ident = 0x00 ;

uint   pcId =  0 ;      /* moved from main to support more token tables for one Id, PC-1421 */
ushort  pcgrpId = IDENT_UNKNOWN ;
bool shortcuts = true ; /* replace also shortcuts (P. = PRINT ...) with tokens */

uint tokenL =  0 ;      /* moved from main to support tokens with mixed length (0) */
uint ll_Img = 80 ;      /* Constant value for max. length of BASIC IMAGE lines,
                           minus length of line Nb text to stay editable,
                           PC-E500 series lower 255 */

/* is a token of BASIC intermediate code, used chars A upto Z and .$# */
uint istoken( char *befehl )
{
   char tokenstr[cLC] = "";

   if (strlen(befehl)== 0) return (0);

   if (shortcuts) {
    if ( !strcmp( tokenstr, "AS." ) )     strcpy (tokenstr, "ASC") ;
    else if ( !strcmp( befehl, "AE." ) )  strcpy (tokenstr, "AER") ;
    else if ( !strcmp( befehl, "AR." ) )  strcpy (tokenstr, "AREAD") ;
    else if ( !strcmp( befehl, "AU." ) )  strcpy (tokenstr, "AUTOGOTO") ;
    else if ( !strcmp( befehl, "B." ) )   strcpy (tokenstr, "BEEP") ;
    else if ( !strcmp( befehl, "BE." ) )  strcpy (tokenstr, "BEEP") ;
    else if ( !strcmp( befehl, "CA." ) )  strcpy (tokenstr, "CALL") ;
    else if ( !strcmp( befehl, "CH." ) )  strcpy (tokenstr, "CHR$") ;
    else if ( !strcmp( befehl, "CHA." ) ) strcpy (tokenstr, "CHAIN") ;
    else if ( !strcmp( befehl, "CI." ) )  strcpy (tokenstr, "CIRCLE") ;
    else if ( !strcmp( befehl, "CL." ) )  strcpy (tokenstr, "CLEAR") ;
    else if ( !strcmp( befehl, "CLOS." ) )strcpy (tokenstr, "CLOSE") ;
    else if ( !strcmp( befehl, "COL." ) ) strcpy (tokenstr, "COLOR") ;
    else if ( !strcmp( befehl, "CONS." ) )strcpy (tokenstr, "CONSOLE") ;
    else if ( !strcmp( befehl, "COP." ) ) strcpy (tokenstr, "COPY") ;
    else if ( !strcmp( befehl, "CR." ) )  strcpy (tokenstr, "CROTATE") ;
    else if ( !strcmp( befehl, "CS." ) )  strcpy (tokenstr, "CSAVE") ;
    else if ( !strcmp( befehl, "CSI." ) ) strcpy (tokenstr, "CSIZE") ;
    else if ( !strcmp( befehl, "CU." ) )  strcpy (tokenstr, "CURSOR") ;
    else if ( !strcmp( befehl, "D." ) )   strcpy (tokenstr, "DIM") ;
    else if ( !strcmp( befehl, "DA." ) )  strcpy (tokenstr, "DATA") ;
    else if ( !strcmp( befehl, "DE." ) )  strcpy (tokenstr, "DEGREE") ;
    else if ( !strcmp( befehl, "DEF." ) ) strcpy (tokenstr, "DEFDBL") ;
    else if ( !strcmp( befehl, "DEFS." ) )strcpy (tokenstr, "DEFSNG") ;
    else if ( !strcmp( befehl, "DS." ) )  strcpy (tokenstr, "DSKF") ;
    else if ( !strcmp( befehl, "E." ) )   strcpy (tokenstr, "END") ;
    else if ( !strcmp( befehl, "EL." ) )  strcpy (tokenstr, "ELSE") ;
    else if ( !strcmp( befehl, "F." ) )   strcpy (tokenstr, "FOR") ;
    else if ( !strcmp( befehl, "FR." ) )  strcpy (tokenstr, "FRE") ;
    else if ( !strcmp( befehl, "ER." ) )  strcpy (tokenstr, "ERASE") ;
    else if ( !strcmp( befehl, "ERR." ) ) strcpy (tokenstr, "ERROR") ;
    else if ( !strcmp( befehl, "EO." ) )  strcpy (tokenstr, "EOF") ;
    else if ( !strcmp( befehl, "G." ) )   strcpy (tokenstr, "GOTO") ;
    else if ( !strcmp( befehl, "GO." ) )  strcpy (tokenstr, "GOTO") ;
    else if ( !strcmp( befehl, "GC." ) )  strcpy (tokenstr, "GCURSOR") ;
    else if ( !strcmp( befehl, "GL." ) )  strcpy (tokenstr, "GLCURSOR") ;
    else if ( !strcmp( befehl, "GOS." ) ) strcpy (tokenstr, "GOSUB") ;
    else if ( !strcmp( befehl, "GP." ) )  strcpy (tokenstr, "GPRINT") ;
    else if ( !strcmp( befehl, "GR." ) )  strcpy (tokenstr, "GRAD") ;
    else if ( !strcmp( befehl, "GRAP." ) )strcpy (tokenstr, "GRAPH") ;
    else if ( !strcmp( befehl, "H." ) )   strcpy (tokenstr, "HEX") ;
    else if ( !strcmp( befehl, "I." ) )   strcpy (tokenstr, "INPUT") ;
    else if ( !strcmp( befehl, "IN." ) )  strcpy (tokenstr, "INPUT") ;
    else if ( !strcmp( befehl, "INI." ) ) strcpy (tokenstr, "INIT") ;
    else if ( !strcmp( befehl, "K." ) )   strcpy (tokenstr, "KILL") ;
    else if ( !strcmp( befehl, "KE." ) )  strcpy (tokenstr, "KEY") ;
    else if ( !strcmp( befehl, "INK." ) ) strcpy (tokenstr, "INKEY$") ;
    else if ( !strcmp( befehl, "LE." ) )  strcpy (tokenstr, "LET") ;
    else if ( !strcmp( befehl, "LEF." ) ) strcpy (tokenstr, "LEFT$") ;
    else if ( !strcmp( befehl, "LF." ) )  strcpy (tokenstr, "LFILES") ;
    else if ( !strcmp( befehl, "LLIN." ) )strcpy (tokenstr, "LLINE") ;
    else if ( !strcmp( befehl, "LIN." ) ) strcpy (tokenstr, "LINE") ;
    else if ( !strcmp( befehl, "LO." ) )  strcpy (tokenstr, "LOCATE") ;
    else if ( !strcmp( befehl, "LOC." ) ) strcpy (tokenstr, "LOCATE") ;
    else if ( !strcmp( befehl, "LP." ) )  strcpy (tokenstr, "LPRINT") ;
    else if ( !strcmp( befehl, "LS." ) )  strcpy (tokenstr, "LSET") ;
    else if ( !strcmp( befehl, "LT." ) )  strcpy (tokenstr, "LTEXT") ;
    else if ( !strcmp( befehl, "M." ) )   strcpy (tokenstr, "MEM") ;
    else if ( !strcmp( befehl, "MD." ) )  strcpy (tokenstr, "MDF") ;
    else if ( !strcmp( befehl, "MER." ) ) strcpy (tokenstr, "MERGE") ;
    else if ( !strcmp( befehl, "MI." ) )  strcpy (tokenstr, "MID$") ;
    else if ( !strcmp( befehl, "N." ) )   strcpy (tokenstr, "NEXT") ;
    else if ( !strcmp( befehl, "NE." ) )  strcpy (tokenstr, "NEXT") ;
    else if ( !strcmp( befehl, "NA." ) )  strcpy (tokenstr, "NAME") ;
    else if ( !strcmp( befehl, "O." ) )   strcpy (tokenstr, "ON") ;
    else if ( !strcmp( befehl, "OP." ) )  strcpy (tokenstr, "OPEN") ;
    else if ( !strcmp( befehl, "P." ) )   strcpy (tokenstr, "PRINT") ;
    else if ( !strcmp( befehl, "PR." ) )  strcpy (tokenstr, "PRINT") ;
    else if ( !strcmp( befehl, "PAU." ) ) strcpy (tokenstr, "PAUSE") ;
    else if ( !strcmp( befehl, "PAI." ) ) strcpy (tokenstr, "PAINT") ;
    else if ( !strcmp( befehl, "PE." ) )  strcpy (tokenstr, "PEEK") ;
    else if ( !strcmp( befehl, "PRE." ) ) strcpy (tokenstr, "PRESET") ;
    else if ( !strcmp( befehl, "PS." ) )  strcpy (tokenstr, "PSET") ;
    else if ( !strcmp( befehl, "RA." ) )  strcpy (tokenstr, "RANDOM") ;
    else if ( !strcmp( befehl, "RAD." ) ) strcpy (tokenstr, "RADIAN") ;
    else if ( !strcmp( befehl, "RE." ) )  strcpy (tokenstr, "RETURN") ;
    else if ( !strcmp( befehl, "REA." ) ) strcpy (tokenstr, "READ") ;
    else if ( !strcmp( befehl, "RES." ) ) strcpy (tokenstr, "RESTORE") ;
    else if ( !strcmp( befehl, "RESU." ) )strcpy (tokenstr, "RESUME") ;
    else if ( !strcmp( befehl, "RI." ) )  strcpy (tokenstr, "RIGHT$") ;
    else if ( !strcmp( befehl, "RL." ) )  strcpy (tokenstr, "RLINE") ;
    else if ( !strcmp( befehl, "RN." ) )  strcpy (tokenstr, "RND") ;
    else if ( !strcmp( befehl, "RS." ) )  strcpy (tokenstr, "RSET") ;
    else if ( !strcmp( befehl, "S." ) )   strcpy (tokenstr, "STOP") ;
    else if ( !strcmp( befehl, "ST." ) )  strcpy (tokenstr, "STOP") ;
    else if ( !strcmp( befehl, "SE." ) )  strcpy (tokenstr, "SET") ;
    else if ( !strcmp( befehl, "SO." ) )  strcpy (tokenstr, "SORGN") ;
    else if ( !strcmp( befehl, "STR." ) ) strcpy (tokenstr, "STR$") ;
    else if ( !strcmp( befehl, "STE." ) ) strcpy (tokenstr, "STEP") ;
    else if ( !strcmp( befehl, "T." ) )   strcpy (tokenstr, "THEN") ;
    else if ( !strcmp( befehl, "TH." ) )  strcpy (tokenstr, "THEN") ;
    else if ( !strcmp( befehl, "TR." ) )  strcpy (tokenstr, "TRON") ;
    else if ( !strcmp( befehl, "TROF." ) )strcpy (tokenstr, "TROFF") ;
    else if ( !strcmp( befehl, "U." ) )   strcpy (tokenstr, "USING") ;
    else if ( !strcmp( befehl, "US." ) )  strcpy (tokenstr, "USING") ;
    else if ( !strcmp( befehl, "V." ) )   strcpy (tokenstr, "VAL") ;
    else if ( !strcmp( befehl, "W." ) )   strcpy (tokenstr, "WAIT") ;
    else if ( !strcmp( befehl, "WA." ) )  strcpy (tokenstr, "WAIT") ;
    else strncpy (tokenstr, befehl, cLC-1) ;
  }

  if ( ident == IDENT_OLD_BAS )
  {
	if ( !strcmp( tokenstr, "AREAD" ) )    		return 0xDC;
	if ( !strcmp( tokenstr, "ABS" ) )    		return 0xAA;
	if ( !strcmp( tokenstr, "ACS" ) )    		return 0xA4;
	if ( !strcmp( tokenstr, "ASN" ) )    		return 0xA3;
	if ( !strcmp( tokenstr, "ATN" ) )    		return 0xA5;
	if ( !strcmp( tokenstr, "BEEP" ) )    		return 0xDB;
	if ( !strcmp( tokenstr, "CHAIN" ) )    		return 0xD9;
	if ( !strcmp( tokenstr, "CLEAR" ) )    		return 0xC5;
	if ( !strcmp( tokenstr, "CLOAD" ) )    		return 0xB7;
	if ( !strcmp( tokenstr, "CONT" ) )    		return 0xB4;
	if ( !strcmp( tokenstr, "COS" ) )    		return 0xA1;
	if ( !strcmp( tokenstr, "CSAVE" ) )    		return 0xB6;
	if ( !strcmp( tokenstr, "DEBUG" ) )    		return 0xB5;
	if ( !strcmp( tokenstr, "DEGREE" ) )    	return 0xC4;
	if ( !strcmp( tokenstr, "DEG" ) )    		return 0xAC;
	if ( !strcmp( tokenstr, "DMS" ) )    		return 0xAD;
	if ( !strcmp( tokenstr, "END" ) )    		return 0xD4;
	if ( !strcmp( tokenstr, "EXP" ) )    		return 0xA6;
	if ( !strcmp( tokenstr, "FOR" ) )    		return 0xD1;
	if ( !strcmp( tokenstr, "GOSUB" ) )    		return 0xD8;
	if ( !strcmp( tokenstr, "GOTO" ) )   		return 0xD7;
	if ( !strcmp( tokenstr, "GRAD" ) )    		return 0xC0;
	if ( !strcmp( tokenstr, "IF" ) )    		return 0xD0;
	if ( !strcmp( tokenstr, "INPUT" ) )    		return 0xC2;
	if ( !strcmp( tokenstr, "INT" ) )    		return 0xA9;
	if ( !strcmp( tokenstr, "LN" ) )    		return 0xA7;
	if ( !strcmp( tokenstr, "LET" ) )    		return 0xD2;
	if ( !strcmp( tokenstr, "LIST" ) )    		return 0xB3;
	if ( !strcmp( tokenstr, "LOG" ) )    		return 0xA8;
	if ( !strcmp( tokenstr, "MEM" ) )    		return 0xB2;
	if ( !strcmp( tokenstr, "NEXT" ) )    		return 0xD5;
	if ( !strcmp( tokenstr, "NEW" ) )    		return 0xB1;
	if ( !strcmp( tokenstr, "PAUSE" ) )    		return 0xDA;
	if ( !strcmp( tokenstr, "PRINT" ) )    		return 0xC1;
	if ( !strcmp( tokenstr, "RADIAN" ) )    	return 0xC3;
	if ( !strcmp( tokenstr, "REM" ) )    		return 0xD3;
	if ( !strcmp( tokenstr, "RETURN" ) )    	return 0xDE;
	if ( !strcmp( tokenstr, "RUN" ) )    		return 0xB0;
	if ( !strcmp( tokenstr, "SGN" ) )    		return 0xAB;
	if ( !strcmp( tokenstr, "SIN" ) )    		return 0xA0;
	if ( !strcmp( tokenstr, "STEP" ) )    		return 0x91;
	if ( !strcmp( tokenstr, "STOP" ) )    		return 0xD6;
	if ( !strcmp( tokenstr, "TAN" ) )    		return 0xA2;
	if ( !strcmp( tokenstr, "THEN" ) )    		return 0x92;
	if ( !strcmp( tokenstr, "TO" ) )    		return 0x90;
	if ( !strcmp( tokenstr, "USING" ) )    		return 0xDD;

    if ( pcId < 1210 || 1212 < pcId ) {
        if ( !strcmp( tokenstr, "AND" ) )    		return 0x81;
        if ( !strcmp( tokenstr, "ASC" ) )    		return 0x7D;
        if ( !strcmp( tokenstr, "CALL" ) )    		return 0xC9;
        if ( !strcmp( tokenstr, "CHR$" ) )    		return 0x88;
        if ( !strcmp( tokenstr, "CLOSE" ) )      	return 0x9D;
        if ( !strcmp( tokenstr, "COM$" ) )    		return 0x89;
        if ( !strcmp( tokenstr, "DATA" ) )    		return 0xCB;
        if ( !strcmp( tokenstr, "DIM" ) )    		return 0xCA;
        if ( !strcmp( tokenstr, "ERROR" ) )    		return 0x96;
        if ( !strcmp( tokenstr, "INKEY$" ) )    	return 0x8A;
        if ( !strcmp( tokenstr, "INSTAT" ) )    	return 0xBF;
        if ( !strcmp( tokenstr, "KEY" ) )    		return 0x99;
        if ( !strcmp( tokenstr, "LEFT$" ) )    		return 0x8C;
        if ( !strcmp( tokenstr, "LEN" ) )    		return 0x7F;
        if ( !strcmp( tokenstr, "LLIST" ) )    		return 0xBC;
        if ( !strcmp( tokenstr, "LPRINT" ) )    	return 0x9F;
        if ( !strcmp( tokenstr, "MERGE" ) )    		return 0xB8;
        if ( !strcmp( tokenstr, "MID$" ) )    		return 0x8E;
        if ( !strcmp( tokenstr, "NOT" ) )    		return 0x86;
        if ( !strcmp( tokenstr, "OFF" ) )    		return 0xCD;
        if ( !strcmp( tokenstr, "ON" ) )    		return 0xCC;
        if ( !strcmp( tokenstr, "OPEN" ) )      	return 0x9C;
        if ( !strcmp( tokenstr, "OR" ) )    		return 0x85;
        if ( !strcmp( tokenstr, "OUTSTAT" ) )    	return 0xBE;
        if ( !strcmp( tokenstr, "PASS" ) )    		return 0xBB;
        if ( !strcmp( tokenstr, "PEEK" ) )    		return 0xAF;
        if ( !strcmp( tokenstr, "PI" ) )    		return 0xBD;
        if ( !strcmp( tokenstr, "POKE" ) )    		return 0xCE;
        if ( !strcmp( tokenstr, "RANDOM" ) )    	return 0x93;
        if ( !strcmp( tokenstr, "READ" ) )    		return 0xCF;
        if ( !strcmp( tokenstr, "RESTORE" ) )    	return 0xDF;
        if ( !strcmp( tokenstr, "RIGHT$" ) )    	return 0x8D;
        if ( !strcmp( tokenstr, "RND" ) )    		return 0xAE;
        if ( !strcmp( tokenstr, "ROM" ) )    		return 0x9E;
        if ( !strcmp( tokenstr, "SETCOM" ) )    	return 0x9B;
        if ( !strcmp( tokenstr, "SQR" ) )    		return 0x87;
        if ( !strcmp( tokenstr, "STR$" ) )    		return 0x8B;
        if ( !strcmp( tokenstr, "TROFF" ) )    		return 0xBA;
        if ( !strcmp( tokenstr, "TRON" ) )    		return 0xB9;
        if ( !strcmp( tokenstr, "VAL" ) )    		return 0x7E;
        if ( !strcmp( tokenstr, "WAIT" ) )  		return 0x95;
	}
	return 0;
  }
  else if ( ident == IDENT_NEW_BAS ) {
    if (pcId == 1421) {
        if ( !strcmp( tokenstr, "ACC" ) )  		return 0xB8;
        if ( !strcmp( tokenstr, "ARMT" ) ) 		return 0xB9;
        if ( !strcmp( tokenstr, "COMP" ) )  	return 0xBA;
        if ( !strcmp( tokenstr, "MDF" ) )  		return 0xBB;
        if ( !strcmp( tokenstr, "EFF" ) )  		return 0xBC;
        if ( !strcmp( tokenstr, "APR" ) )  		return 0xBD;
        if ( !strcmp( tokenstr, "DAYSII" ) )  	return 0xBF;
        if ( !strcmp( tokenstr, "DAYSI" ) )  	return 0xBE;
        if ( !strcmp( tokenstr, "BGNON" ) )  	return 0xCE;
        if ( !strcmp( tokenstr, "BGNOFF" ) )  	return 0xCF;
        if ( !strcmp( tokenstr, "ERASE" ) )  	return 0xE5;
        if ( !strcmp( tokenstr, "FIN" ) )   	return 0xE6;
        if ( !strcmp( tokenstr, "CST" ) )  	    return 0xEA;
        if ( !strcmp( tokenstr, "SEL" ) )  	    return 0xEB;
        if ( !strcmp( tokenstr, "MAR" ) )  	    return 0xEC;
        if ( !strcmp( tokenstr, "MU" ) )  	    return 0xED;
        if ( !strcmp( tokenstr, "PV" ) )  	    return 0xEE;
        if ( !strcmp( tokenstr, "FV" ) )  	    return 0xEF;
        if ( !strcmp( tokenstr, "PMT" ) )  	    return 0xF0;
        if ( !strcmp( tokenstr, "NPV" ) )  	    return 0xF1;
        if ( !strcmp( tokenstr, "IRR" ) )  	    return 0xF2;
        if ( !strcmp( tokenstr, "PRN" ) )  	    return 0xF3;
        if ( !strcmp( tokenstr, "INTE" ) )  	return 0xF4;
        if ( !strcmp( tokenstr, "BAL" ) )  	    return 0xF5;
        if ( !strcmp( tokenstr, "SPRN" ) ) 	    return 0xF6;
        if ( !strcmp( tokenstr, "SINTE" ) )     return 0xF7;
        if ( !strcmp( tokenstr, "NI" ) )  	    return 0xF8;
        if ( !strcmp( tokenstr, "CFI" ) )  	    return 0xF9;
    }
    if (pcId == 1401) {
        if ( !strcmp( tokenstr, "DEC" ) )     	return 0x84; // PC-1401, 1.version
        if ( !strcmp( tokenstr, "FAC" ) )     	return 0x90; // PC-1401, 1.version
    }
	if ( !strcmp( tokenstr, "ABS" ) )  		return 0x99;
	if ( !strcmp( tokenstr, "ACS" ) )  		return 0x9E;
	if ( !strcmp( tokenstr, "AHC" ) )  		return 0x8E;
	if ( !strcmp( tokenstr, "AHS" ) )  		return 0x8D;
	if ( !strcmp( tokenstr, "AHT" ) )  		return 0x8F;
	if ( !strcmp( tokenstr, "AND" ) )  		return 0xA1;
	if ( !strcmp( tokenstr, "AREAD" ) )  	return 0xE1;
	if ( !strcmp( tokenstr, "ASC" ) )  		return 0xA4;
	if ( !strcmp( tokenstr, "ASN" ) )  		return 0x9D;
	if ( !strcmp( tokenstr, "ATN" ) )  		return 0x9F;
	if ( !strcmp( tokenstr, "BASIC" ) )  	return 0xEC;
	if ( !strcmp( tokenstr, "BEEP" ) )  	return 0xC4;
	if ( !strcmp( tokenstr, "CALL" ) )  	return 0xCC;
	if ( !strcmp( tokenstr, "CHAIN" ) )  	return 0xE5;
	if ( !strcmp( tokenstr, "CHR$" ) )  	return 0xA8;
	if ( !strcmp( tokenstr, "CLEAR" ) )  	return 0xC9;
	if ( !strcmp( tokenstr, "CLOAD" ) )  	return 0xB7;
	if ( !strcmp( tokenstr, "CLOSE" ) )  	return 0xBC;
	if ( !strcmp( tokenstr, "CLS" ) )  		return 0xCE;
	if ( !strcmp( tokenstr, "CONSOLE" ) )  	return 0xBF;
	if ( !strcmp( tokenstr, "CONT" ) )  	return 0xB2;
	if ( !strcmp( tokenstr, "COS" ) )  		return 0x96;
	if ( !strcmp( tokenstr, "CSAVE" ) )  	return 0xB6;
	if ( !strcmp( tokenstr, "CUR" ) )  		return 0x89;
	if ( !strcmp( tokenstr, "CURSOR" ) )  	return 0xCF;
	if ( !strcmp( tokenstr, "DATA" ) )  	return 0xDC;
	if ( !strcmp( tokenstr, "DECI" ) )  	return 0x84;
	if ( !strcmp( tokenstr, "DEG" ) )  		return 0x9B;
	if ( !strcmp( tokenstr, "DEGREE" ) )  	return 0xC1;
	if ( !strcmp( tokenstr, "DIM" ) )  		return 0xCB;
	if ( !strcmp( tokenstr, "DMS" ) )  		return 0x9C;
	if ( !strcmp( tokenstr, "END" ) )  		return 0xD8;
	if ( !strcmp( tokenstr, "EQU#" ) )  	return 0xB9;
	if ( !strcmp( tokenstr, "EXP" ) )  		return 0x93;
	if ( !strcmp( tokenstr, "FACT" ) )  	return 0x90;
	if ( !strcmp( tokenstr, "FOR" ) )  		return 0xD5;
	if ( !strcmp( tokenstr, "GCURSOR" ) )  	return 0xE6;
	if ( !strcmp( tokenstr, "GOSUB" ) )  	return 0xE0;
	if ( !strcmp( tokenstr, "GOTO" ) )  	return 0xC6;
	if ( !strcmp( tokenstr, "GPRINT" ) )  	return 0xE7;
	if ( !strcmp( tokenstr, "GRAD" ) )  	return 0xC3;
	if ( !strcmp( tokenstr, "HCS" ) )  		return 0x8B;
	if ( !strcmp( tokenstr, "HEX" ) )  		return 0x85;
	if ( !strcmp( tokenstr, "HSN" ) )  		return 0x8A;
	if ( !strcmp( tokenstr, "HTN" ) )  		return 0x8C;
	if ( !strcmp( tokenstr, "IF" ) )  		return 0xD4;
	if ( !strcmp( tokenstr, "INKEY$" ) )  	return 0xAD;
	if ( !strcmp( tokenstr, "INPUT" ) )  	return 0xDF;
	if ( !strcmp( tokenstr, "INT" ) )  		return 0x98;
	if ( !strcmp( tokenstr, "LEFT$" ) )  	return 0xAB;
	if ( !strcmp( tokenstr, "LEN" ) )  		return 0xA6;
	if ( !strcmp( tokenstr, "LET" ) )  		return 0xD6;
	if ( !strcmp( tokenstr, "LINE" ) )  	return 0xE8;
	if ( !strcmp( tokenstr, "LIST" ) )  	return 0xB4;
	if ( !strcmp( tokenstr, "LLIST" ) )  	return 0xB5;
	if ( !strcmp( tokenstr, "LN" ) )  		return 0x91;
	if ( !strcmp( tokenstr, "LOAD" ) )  	return 0xBE;
	if ( !strcmp( tokenstr, "LOG" ) )  		return 0x92;
	if ( !strcmp( tokenstr, "LPRINT" ) )  	return 0xE2;
	if ( !strcmp( tokenstr, "MDF" ) )  		return 0x80;
	if ( !strcmp( tokenstr, "MEM" ) )  		return 0xAF;
	if ( !strcmp( tokenstr, "MEM#" ) )  	return 0xBA;
	if ( !strcmp( tokenstr, "MERGE" ) )  	return 0xB8;
	if ( !strcmp( tokenstr, "MID$" ) )  	return 0xAA;
	if ( !strcmp( tokenstr, "NEW" ) )  		return 0xB1;
	if ( !strcmp( tokenstr, "NEXT" ) )  	return 0xD9;
	if ( !strcmp( tokenstr, "NOT" ) )  		return 0xA3;
	if ( !strcmp( tokenstr, "ON" ) )  		return 0xD3;
	if ( !strcmp( tokenstr, "OPEN" ) )  	return 0xBB;
	if ( !strcmp( tokenstr, "OPEN$" ) )  	return 0xEE;
	if ( !strcmp( tokenstr, "OR" ) )  		return 0xA2;
	if ( !strcmp( tokenstr, "PASS" ) )  	return 0xB3;
	if ( !strcmp( tokenstr, "PAUSE" ) )  	return 0xDD;
	if ( !strcmp( tokenstr, "PEEK" ) )  	return 0xA7;
	if ( !strcmp( tokenstr, "PI" ) )  		return 0xAE;
	if ( !strcmp( tokenstr, "POINT" ) )  	return 0xE9;
	if ( !strcmp( tokenstr, "POKE" ) )  	return 0xCD;
	if ( !strcmp( tokenstr, "POL" ) )  		return 0x82;
	if ( !strcmp( tokenstr, "PRESET" ) )  	return 0xEB;
	if ( !strcmp( tokenstr, "PRINT" ) )  	return 0xDE;
	if ( !strcmp( tokenstr, "PSET" ) )  	return 0xEA;
	if ( !strcmp( tokenstr, "RADIAN" ) )  	return 0xC2;
	if ( !strcmp( tokenstr, "RANDOM" ) )  	return 0xC0;
	if ( !strcmp( tokenstr, "RCP" ) )  		return 0x87;
	if ( !strcmp( tokenstr, "READ" ) )  	return 0xDB;
	if ( !strcmp( tokenstr, "REC" ) )  		return 0x81;
	if ( !strcmp( tokenstr, "REM" ) )  		return 0xD7;
	if ( !strcmp( tokenstr, "RESTORE" ) )  	return 0xE4;
	if ( !strcmp( tokenstr, "RETURN" ) )  	return 0xE3;
	if ( !strcmp( tokenstr, "RIGHT$" ) )  	return 0xAC;
	if ( !strcmp( tokenstr, "RND" ) )  		return 0xA0;
	if ( !strcmp( tokenstr, "ROT" ) )  		return 0x83;
	if ( !strcmp( tokenstr, "RUN" ) )  		return 0xB0;
	if ( !strcmp( tokenstr, "SAVE" ) )  	return 0xBD;
	if ( !strcmp( tokenstr, "SGN" ) )  		return 0x9A;
	if ( !strcmp( tokenstr, "SIN" ) )  		return 0x95;
	if ( !strcmp( tokenstr, "SQR" ) )  		return 0x94;
	if ( !strcmp( tokenstr, "SQU" ) )  		return 0x88;
	if ( !strcmp( tokenstr, "STEP" ) )  	return 0xD1;
	if ( !strcmp( tokenstr, "STOP" ) )  	return 0xDA;
	if ( !strcmp( tokenstr, "STR$" ) )  	return 0xA9;
	if ( !strcmp( tokenstr, "TAN" ) )  		return 0x97;
	if ( !strcmp( tokenstr, "TEN" ) )  		return 0x86;
	if ( !strcmp( tokenstr, "TEXT" ) )  	return 0xED;
	if ( !strcmp( tokenstr, "THEN" ) )  	return 0xD2;
	if ( !strcmp( tokenstr, "TO" ) )  		return 0xD0;
	if ( !strcmp( tokenstr, "TROFF" ) )  	return 0xC8;
	if ( !strcmp( tokenstr, "TRON" ) ) 		return 0xC7;
	if ( !strcmp( tokenstr, "USING" ) ) 	return 0xCA;
	if ( !strcmp( tokenstr, "VAL" ) ) 		return 0xA5;
	if ( !strcmp( tokenstr, "WAIT" ) )		return 0xC5;

    /* pcId 1403, 1425 ... 1445, 1460 */
    if (tokenL == 0) {
        if ( !strcmp( tokenstr, "APPEND" ) )  	return 0xEFE1;
        if ( !strcmp( tokenstr, "AS" ) )  	    return 0xEFE2;
        if ( !strcmp( tokenstr, "BDS" ) )  	    return 0xEF8C;
        if ( !strcmp( tokenstr, "BIN" ) )  	    return 0xEF84;
        if ( !strcmp( tokenstr, "COPY" ) )  	return 0xEFB6;
        if ( !strcmp( tokenstr, "CONVERT" ) )  	return 0xEFB7;
        if ( !strcmp( tokenstr, "CSI" ) )  	    return 0xEF5A;
        if ( !strcmp( tokenstr, "LOR" ) )  	    return 0xEFC9;
        if ( !strcmp( tokenstr, "CIRCLE" ) )  	return 0xEFCC;
        if ( !strcmp( tokenstr, "CROTATE" ) )  	return 0xEFCB;
        if ( !strcmp( tokenstr, "DSKF" ) )  	return 0xEF81;
        if ( !strcmp( tokenstr, "DELETE" ) )  	return 0xEFA9;
        if ( !strcmp( tokenstr, "EOF" ) )  	    return 0xEF80;
        if ( !strcmp( tokenstr, "ERASE" ) )  	return 0xEFC0;
        if ( !strcmp( tokenstr, "FILES" ) )  	return 0xEFB0;
        if ( !strcmp( tokenstr, "FDS" ) )  	    return 0xEF9B;
        if ( !strcmp( tokenstr, "GRAPH" ) )  	return 0xEFC6;
        if ( !strcmp( tokenstr, "GLCURSOR" ) )  return 0xEFC4;
        if ( !strcmp( tokenstr, "INIT" ) )  	return 0xEFB2;
        if ( !strcmp( tokenstr, "KILL" ) )  	return 0xEFB3;
        if ( !strcmp( tokenstr, "LOF" ) )  	    return 0xEF82;
        if ( !strcmp( tokenstr, "LOC" ) )  	    return 0xEF83;
        if ( !strcmp( tokenstr, "LFILES" ) )  	return 0xEFB1;
        if ( !strcmp( tokenstr, "LTEXT" ) )  	return 0xEFC5;
        if ( !strcmp( tokenstr, "LF" ) )  	    return 0xEFC7;
        if ( !strcmp( tokenstr, "LLINE" ) )  	return 0xEFC2;
        if ( !strcmp( tokenstr, "NAME" ) )  	return 0xEFB4;
        if ( !strcmp( tokenstr, "NCR" ) )  	    return 0xEF86;
        if ( !strcmp( tokenstr, "NPR" ) )  	    return 0xEF87;
        if ( !strcmp( tokenstr, "NDS" ) )  	    return 0xEF98;
        if ( !strcmp( tokenstr, "OUTPUT" ) )    return 0xEFE0;
        if ( !strcmp( tokenstr, "OCT" ) )  	    return 0xEF85;
        if ( !strcmp( tokenstr, "PAINT" ) )  	return 0xEFCD;
        if ( !strcmp( tokenstr, "PND" ) )  	    return 0xEF88;
        if ( !strcmp( tokenstr, "PTD" ) )  	    return 0xEF89;
        if ( !strcmp( tokenstr, "PXD" ) )  	    return 0xEF8A;
        if ( !strcmp( tokenstr, "PFD" ) )  	    return 0xEF8B;
        if ( !strcmp( tokenstr, "PDS" ) )  	    return 0xEF8D;
        if ( !strcmp( tokenstr, "RENUM" ) )  	return 0xEFA8;
        if ( !strcmp( tokenstr, "RLINE" ) )  	return 0xEFC3;
        if ( !strcmp( tokenstr, "SET" ) )     	return 0xEFB5;
        if ( !strcmp( tokenstr, "SORGN" ) )  	return 0xEFCA;
        if ( !strcmp( tokenstr, "TDS" ) )  	    return 0xEF99;
        if ( !strcmp( tokenstr, "XDS" ) )  	    return 0xEF9A;
        if ( !strcmp( tokenstr, "XOR" ) )  	    return 0xEFA1;
    }
//	else
    return 0;
  }
  else if ( ident == IDENT_EXT_BAS || ident == IDENT_E_BAS  )
  {
    if (pcgrpId == GRP_G) {
        if ( !strcmp( tokenstr, "PAUSE" ) )  	return 0xFE60; // replaced by PRINT
        if ( !strcmp( tokenstr, "HEX" ) )  		return 0xFEF2; // replaced by HEX$
        if ( !strcmp( tokenstr, "BLOAD" ) )  	return 0xFE16; // G850V replaces CLOAD
        if ( !strcmp( tokenstr, "BSAVE" ) )  	return 0xFE20; // G850V replaces CSAVE
    }
    if (pcgrpId == GRP_E || pcgrpId == GRP_G) {
        if ( !strcmp( tokenstr, "AER" ) )  		return 0xFEBE; // E500
        if ( !strcmp( tokenstr, "AUTO" ) )  	return 0xFE1A; // E500
        if ( !strcmp( tokenstr, "BDATA$" ) )  	return 0xFE0C; // E500
        if ( !strcmp( tokenstr, "BTEXT$" ) )  	return 0xFE0B; // E500
        if ( !strcmp( tokenstr, "CASE" ) )  	return 0xFE7D; // E500S
        if ( !strcmp( tokenstr, "DEFAULT" ) )  	return 0xFE7E; // E500S
        if ( !strcmp( tokenstr, "ELSE" ) )  	return 0xFE76; // E500
        if ( !strcmp( tokenstr, "ENDIF" ) )  	return 0xFE4D; // E500S
        if ( !strcmp( tokenstr, "ENDSWITCH" ) ) return 0xFE7F; // E500S
        if ( !strcmp( tokenstr, "EVAL" ) )  	return 0xFEA7; // E500
        if ( !strcmp( tokenstr, "FRE" ) )     	return 0xFEAF; // E500 = MEM
        if ( !strcmp( tokenstr, "KEY" ) )     	return 0xFE79; // E500
        if ( !strcmp( tokenstr, "LOCATE" ) )    return 0xFE51; // E500 = CURSOR
        if ( !strcmp( tokenstr, "MEM$" ) )  	return 0xFE0D; // E500
        if ( !strcmp( tokenstr, "MON" ) )  		return 0xFE0F; // E220
        if ( !strcmp( tokenstr, "RANDOMIZE" ) ) return 0xFE25; // E500 = RANDOM
        if ( !strcmp( tokenstr, "REPEAT" ) )  	return 0xFE4E; // E500S
        if ( !strcmp( tokenstr, "RESERVED" ) )  return 0xFE00; // E500
        if ( !strcmp( tokenstr, "RESUME" ) )  	return 0xFE77; // E500
        if ( !strcmp( tokenstr, "SWITCH" ) )  	return 0xFE7C; // E500S
        if ( !strcmp( tokenstr, "UNTIL" ) )  	return 0xFE4F; // E500S
        if ( !strcmp( tokenstr, "WEND" ) )  	return 0xFE7B; // E500S
        if ( !strcmp( tokenstr, "WHILE" ) )  	return 0xFE7A; // E500S
    }
	if ( !strcmp( tokenstr, "ABS" ) )  		return 0xFE99;
	if ( !strcmp( tokenstr, "ACS" ) )  		return 0xFE9E;
	if ( !strcmp( tokenstr, "AHC" ) )  		return 0xFE8E;
	if ( !strcmp( tokenstr, "AHS" ) )  		return 0xFE8D;
	if ( !strcmp( tokenstr, "AHT" ) )  		return 0xFE8F;
	if ( !strcmp( tokenstr, "AND" ) )  		return 0xFEA1;
	if ( !strcmp( tokenstr, "ASN" ) )  		return 0xFE9D;
	if ( !strcmp( tokenstr, "ATN" ) )  		return 0xFE9F;
	if ( !strcmp( tokenstr, "COS" ) )  		return 0xFE96;
	if ( !strcmp( tokenstr, "CUB" ) )  		return 0xFEBF;
	if ( !strcmp( tokenstr, "CUR" ) )  		return 0xFE89;
	if ( !strcmp( tokenstr, "DECI" ) )  	return 0xFE84;
	if ( !strcmp( tokenstr, "DEG" ) )  		return 0xFE9B;
	if ( !strcmp( tokenstr, "DMS" ) )  		return 0xFE9C;
	if ( !strcmp( tokenstr, "EXP" ) )  		return 0xFE93;
	if ( !strcmp( tokenstr, "FACT" ) )  	return 0xFE90;
	if ( !strcmp( tokenstr, "HCS" ) )  		return 0xFE8B;
	if ( !strcmp( tokenstr, "HEX" ) )  		return 0xFE85;
	if ( !strcmp( tokenstr, "HSN" ) )  		return 0xFE8A;
	if ( !strcmp( tokenstr, "HTN" ) )  		return 0xFE8C;
	if ( !strcmp( tokenstr, "INT" ) )  		return 0xFE98;
	if ( !strcmp( tokenstr, "LN" ) )  		return 0xFE91;
	if ( !strcmp( tokenstr, "LOG" ) )  		return 0xFE92;
	if ( !strcmp( tokenstr, "LOF" ) )  		return 0xFEB2;
	if ( !strcmp( tokenstr, "LOC" ) )  		return 0xFEB3;
	if ( !strcmp( tokenstr, "NCR" ) )  		return 0xFEB6;
	if ( !strcmp( tokenstr, "NPR" ) )  		return 0xFEB7;
	if ( !strcmp( tokenstr, "OR" ) )  		return 0xFEA2;
	if ( !strcmp( tokenstr, "PI" ) )  		return 0xFEAE;
	if ( !strcmp( tokenstr, "POL" ) )  		return 0xFE82;
	if ( !strcmp( tokenstr, "RCP" ) )  		return 0xFE87;
	if ( !strcmp( tokenstr, "REC" ) )  		return 0xFE81;
	if ( !strcmp( tokenstr, "RND" ) )  		return 0xFEA0;
	if ( !strcmp( tokenstr, "ROT" ) )  		return 0xFE83;
	if ( !strcmp( tokenstr, "SGN" ) )  		return 0xFE9A;
	if ( !strcmp( tokenstr, "SIN" ) )  		return 0xFE95;
	if ( !strcmp( tokenstr, "SQR" ) )  		return 0xFE94;
	if ( !strcmp( tokenstr, "SQU" ) )  		return 0xFE88;
	if ( !strcmp( tokenstr, "TAN" ) )  		return 0xFE97;
	if ( !strcmp( tokenstr, "TEN" ) )  		return 0xFE86;
	if ( !strcmp( tokenstr, "XOR" ) )  		return 0xFEA5;

	if ( !strcmp( tokenstr, "AKCNV$" ) )  	return 0xFEE0;
	if ( !strcmp( tokenstr, "APPEND" ) )  	return 0xFE72;
	if ( !strcmp( tokenstr, "AREAD" ) )  	return 0xFE63;
	if ( !strcmp( tokenstr, "ARUN" ) )  	return 0xFE74;
	if ( !strcmp( tokenstr, "AS" ) )	  	return 0xFE73;
	if ( !strcmp( tokenstr, "ASC" ) )  		return 0xFED0;
	if ( !strcmp( tokenstr, "AUTOGOTO" ) )  return 0xFE75;
	if ( !strcmp( tokenstr, "BASIC" ) )  	return 0xFE36;
	if ( !strcmp( tokenstr, "BEEP" ) )  	return 0xFE29;
	if ( !strcmp( tokenstr, "CALL" ) )  	return 0xFE31;
	if ( !strcmp( tokenstr, "CHAIN" ) )  	return 0xFE67;
	if ( !strcmp( tokenstr, "CHR$" ) )  	return 0xFEF0;
	if ( !strcmp( tokenstr, "CIRCLE" ) )  	return 0xFE6F;
	if ( !strcmp( tokenstr, "CLEAR" ) )  	return 0xFE2E;
	if ( !strcmp( tokenstr, "CLOAD" ) )  	return 0xFE16;
	if ( !strcmp( tokenstr, "CLOSE" ) )  	return 0xFE22;
	if ( !strcmp( tokenstr, "CLS" ) )  		return 0xFE50;
	if ( !strcmp( tokenstr, "COLOR" ) )  	return 0xFE44;
	if ( !strcmp( tokenstr, "CONSOLE" ) )  	return 0xFE24;
	if ( !strcmp( tokenstr, "CONT" ) )  	return 0xFE12;
	if ( !strcmp( tokenstr, "CONVERT" ) )  	return 0xFE1E;
	if ( !strcmp( tokenstr, "COPY" ) )  	return 0xFE3D;
	if ( !strcmp( tokenstr, "CROTATE" ) )  	return 0xFE6E;
	if ( !strcmp( tokenstr, "CSAVE" ) )  	return 0xFE20;
	if ( !strcmp( tokenstr, "CSIZE" ) )  	return 0xFE43;
	if ( !strcmp( tokenstr, "CURSOR" ) )  	return 0xFE51;
	if ( !strcmp( tokenstr, "DATA" ) )  	return 0xFE5E;
	if ( !strcmp( tokenstr, "DEFDBL" ) )  	return 0xFE46;
	if ( !strcmp( tokenstr, "DEFSNG" ) )  	return 0xFE47;
	if ( !strcmp( tokenstr, "DEGREE" ) )  	return 0xFE26;
	if ( !strcmp( tokenstr, "DELETE" ) )  	return 0xFE1B;
	if ( !strcmp( tokenstr, "DIM" ) )  		return 0xFE30;
	if ( !strcmp( tokenstr, "DSKF" ) )  	return 0xFEB1;
	if ( !strcmp( tokenstr, "END" ) )  		return 0xFE5A;
	if ( !strcmp( tokenstr, "EOF" ) )  		return 0xFEB0;
	if ( !strcmp( tokenstr, "ERASE" ) )  	return 0xFE3A;
	if ( !strcmp( tokenstr, "ERL" ) )  		return 0xFEC1;
	if ( !strcmp( tokenstr, "ERN" ) )  		return 0xFEC0;
	if ( !strcmp( tokenstr, "ERROR" ) )  	return 0xFE78;
	if ( !strcmp( tokenstr, "FIELD" ) )  	return 0xFE48;
	if ( !strcmp( tokenstr, "FILES" ) )  	return 0xFE1C;
	if ( !strcmp( tokenstr, "FOR" ) )  		return 0xFE57;
	if ( !strcmp( tokenstr, "GET" ) )  		return 0xFE4A;
	if ( !strcmp( tokenstr, "GLCURSOR" ) )  return 0xFE6C;
	if ( !strcmp( tokenstr, "GCURSOR" ) )  	return 0xFE68;
	if ( !strcmp( tokenstr, "GOSUB" ) )  	return 0xFE62;
	if ( !strcmp( tokenstr, "GOTO" ) )  	return 0xFE2B;
	if ( !strcmp( tokenstr, "GPRINT" ) )  	return 0xFE33;
	if ( !strcmp( tokenstr, "GRAD" ) )  	return 0xFE28;
	if ( !strcmp( tokenstr, "GRAPH" ) )  	return 0xFE41;
	if ( !strcmp( tokenstr, "HEX$" ) )  	return 0xFEF2;
	if ( !strcmp( tokenstr, "IF" ) )  		return 0xFE56;
	if ( !strcmp( tokenstr, "INIT" ) )  	return 0xFE1D;
	if ( !strcmp( tokenstr, "INKEY$" ) )  	return 0xFEE9;
	if ( !strcmp( tokenstr, "INPUT" ) )  	return 0xFE61;
	if ( !strcmp( tokenstr, "JIS$" ) )  	return 0xFEE2;
	if ( !strcmp( tokenstr, "KILL" ) )  	return 0xFE3C;
	if ( !strcmp( tokenstr, "KLEN" ) )  	return 0xFED3;
	if ( !strcmp( tokenstr, "KMID$" ) )  	return 0xFEED;
	if ( !strcmp( tokenstr, "KLEFT$" ) )  	return 0xFEEE;
	if ( !strcmp( tokenstr, "KRIGHT$" ) )  	return 0xFEEF;
	if ( !strcmp( tokenstr, "KACNV$" ) )  	return 0xFEE1;
	if ( !strcmp( tokenstr, "LEFT$" ) )  	return 0xFEEB;
	if ( !strcmp( tokenstr, "LEN" ) )  		return 0xFED2;
	if ( !strcmp( tokenstr, "LET" ) )  		return 0xFE58;
	if ( !strcmp( tokenstr, "LF" ) )  		return 0xFE42;
	if ( !strcmp( tokenstr, "LFILES" ) )  	return 0xFE3B;
	if ( !strcmp( tokenstr, "LINE" ) )  	return 0xFE69;
	if ( !strcmp( tokenstr, "LIST" ) )  	return 0xFE14;
	if ( !strcmp( tokenstr, "LLINE" ) )  	return 0xFE6A;
	if ( !strcmp( tokenstr, "LLIST" ) )  	return 0xFE15;
	if ( !strcmp( tokenstr, "LOAD" ) )  	return 0xFE18;
	if ( !strcmp( tokenstr, "LPRINT" ) )  	return 0xFE64;
	if ( !strcmp( tokenstr, "LSET" ) )  	return 0xFE4B;
	if ( !strcmp( tokenstr, "LTEXT" ) )  	return 0xFE40;
	if ( !strcmp( tokenstr, "MDF" ) )  		return 0xFE80;
	if ( !strcmp( tokenstr, "MEM" ) )  		return 0xFEAF;
	if ( !strcmp( tokenstr, "MERGE" ) )  	return 0xFE17;
	if ( !strcmp( tokenstr, "MID$" ) )  	return 0xFEEA;
	if ( !strcmp( tokenstr, "NAME" ) )  	return 0xFE3E;
	if ( !strcmp( tokenstr, "NEW" ) )  		return 0xFE11;
	if ( !strcmp( tokenstr, "NEXT" ) )  	return 0xFE5B;
	if ( !strcmp( tokenstr, "NOT" ) )  		return 0xFEA3;
	if ( !strcmp( tokenstr, "ON" ) )  		return 0xFE55;
	if ( !strcmp( tokenstr, "OPEN" ) )  	return 0xFE21;
	if ( !strcmp( tokenstr, "OPEN$" ) )  	return 0xFEE8;
	if ( !strcmp( tokenstr, "OUTPUT" ) )  	return 0xFE71;
	if ( !strcmp( tokenstr, "PAINT" ) )  	return 0xFE70;
	if ( !strcmp( tokenstr, "PASS" ) )  	return 0xFE13;
	if ( !strcmp( tokenstr, "PAUSE" ) )  	return 0xFE5F;
	if ( !strcmp( tokenstr, "PEEK" ) )  	return 0xFEA4;
	if ( !strcmp( tokenstr, "POINT" ) )  	return 0xFEAD;
	if ( !strcmp( tokenstr, "POKE" ) )  	return 0xFE32;
	if ( !strcmp( tokenstr, "PRESET" ) )  	return 0xFE35;
	if ( !strcmp( tokenstr, "PRINT" ) )  	return 0xFE60;
	if ( !strcmp( tokenstr, "PUT" ) )  		return 0xFE49;
	if ( !strcmp( tokenstr, "PSET" ) )  	return 0xFE34;
	if ( !strcmp( tokenstr, "RADIAN" ) )  	return 0xFE27;
	if ( !strcmp( tokenstr, "RANDOM" ) )  	return 0xFE25;
	if ( !strcmp( tokenstr, "READ" ) )  	return 0xFE5D;
	if ( !strcmp( tokenstr, "RENUM" ) )  	return 0xFE19;
	if ( !strcmp( tokenstr, "REM" ) )  		return 0xFE59;
	if ( !strcmp( tokenstr, "RESTORE" ) )  	return 0xFE66;
	if ( !strcmp( tokenstr, "RETURN" ) )  	return 0xFE65;
	if ( !strcmp( tokenstr, "RIGHT$" ) ) 	return 0xFEEC;
	if ( !strcmp( tokenstr, "RLINE" ) )  	return 0xFE6B;
	if ( !strcmp( tokenstr, "RSET" ) )  	return 0xFE4C;
	if ( !strcmp( tokenstr, "RUN" ) )  		return 0xFE10;
	if ( !strcmp( tokenstr, "SAVE" ) )  	return 0xFE23;
	if ( !strcmp( tokenstr, "SET" ) )  		return 0xFE3F;
	if ( !strcmp( tokenstr, "SORGN" ) )  	return 0xFE6D;
	if ( !strcmp( tokenstr, "STEP" ) )  	return 0xFE53;
	if ( !strcmp( tokenstr, "STOP" ) )  	return 0xFE5C;
	if ( !strcmp( tokenstr, "STR$" ) )  	return 0xFEF1;
	if ( !strcmp( tokenstr, "TEXT" ) )  	return 0xFE37;
	if ( !strcmp( tokenstr, "THEN" ) )  	return 0xFE54;
	if ( !strcmp( tokenstr, "TO" ) )  		return 0xFE52;
	if ( !strcmp( tokenstr, "TROFF" ) )  	return 0xFE2D;
	if ( !strcmp( tokenstr, "TRON" ) ) 		return 0xFE2C;
	if ( !strcmp( tokenstr, "USING" ) ) 	return 0xFE2F;
	if ( !strcmp( tokenstr, "VAL" ) ) 		return 0xFED1;
	if ( !strcmp( tokenstr, "WAIT" ) )		return 0xFE2A;
	if ( !strcmp( tokenstr, "WIDTH" ) )		return 0xFE38;
	else
        return 0;
  }
  else if ( ident == IDENT_PC15_BAS || ident == IDENT_PC16_BAS )
  {
    if (ident == IDENT_PC16_BAS) {
        if ( !strcmp( tokenstr, "ACNV$" ) )  	return 0xF2F6;
        if ( !strcmp( tokenstr, "ADIN" ) )  	return 0xF280;
        if ( !strcmp( tokenstr, "AIN" ) )  	    return 0xF25A;
        if ( !strcmp( tokenstr, "ALARM$" ) )  	return 0xF25C;
        if ( !strcmp( tokenstr, "AOFF" ) )  	return 0xF2BC;
        if ( !strcmp( tokenstr, "APPEND" ) )  	return 0xF2BF;
        if ( !strcmp( tokenstr, "AS" ) )  	    return 0xF2BD;
        if ( !strcmp( tokenstr, "AUTO" ) )  	return 0xF2B6;
        if ( !strcmp( tokenstr, "BLOAD" ) )  	return 0xF290;
        if ( !strcmp( tokenstr, "BSAVE" ) )  	return 0xF291;
        if ( !strcmp( tokenstr, "CALL" ) )  	return 0xF282; // PC_15 Mode XCALL
        if ( !strcmp( tokenstr, "CLOSE" ) )  	return 0xF292;
        if ( !strcmp( tokenstr, "COM" ) )  	    return 0xF2A3;
        if ( !strcmp( tokenstr, "COPY" ) )  	return 0xF293;
        if ( !strcmp( tokenstr, "DATE$" ) )  	return 0xF257;
        if ( !strcmp( tokenstr, "DELETE" ) )  	return 0xF2B9;
        if ( !strcmp( tokenstr, "DSKF" ) )  	return 0xF274;
        if ( !strcmp( tokenstr, "ELSE" ) )  	return 0xF283;
        if ( !strcmp( tokenstr, "EOF" ) )  	    return 0xF271;
        if ( !strcmp( tokenstr, "ERASE" ) )  	return 0xF2B7;
        if ( !strcmp( tokenstr, "FILES" ) )  	return 0xF098;
        if ( !strcmp( tokenstr, "HEX$" ) )  	return 0xF265;
        if ( !strcmp( tokenstr, "INIT" ) )  	return 0xF294;
        if ( !strcmp( tokenstr, "INP" ) )  	    return 0xF266;
        if ( !strcmp( tokenstr, "INSTR" ) )  	return 0xF267;
        if ( !strcmp( tokenstr, "JS$" ) )  	    return 0xF268;
        if ( !strcmp( tokenstr, "KBUFF$" ) )  	return 0xF284;
        if ( !strcmp( tokenstr, "KCNV$" ) )  	return 0xF263;
        if ( !strcmp( tokenstr, "KEFT$" ) )  	return 0xF269;
        if ( !strcmp( tokenstr, "KEN" ) )  	    return 0xF26A;
        if ( !strcmp( tokenstr, "KEYSTAT" ) )  	return 0xF286;
        if ( !strcmp( tokenstr, "KEY" ) )  	    return 0xF285;
        if ( !strcmp( tokenstr, "KID$" ) )  	return 0xF26B;
        if ( !strcmp( tokenstr, "KIGHT$" ) )  	return 0xF26C;
        if ( !strcmp( tokenstr, "KILL" ) )  	return 0xF287;
        if ( !strcmp( tokenstr, "KN$" ) )  	    return 0xF264;
        if ( !strcmp( tokenstr, "LCURSOR" ) )  	return 0xF0A5; // changed PC_15
        if ( !strcmp( tokenstr, "LFILES" ) )  	return 0xF0A0;
        if ( !strcmp( tokenstr, "LINE" ) )  	return 0xF099; // changed PC_15
        if ( !strcmp( tokenstr, "LLINE" ) )  	return 0xF0B7; // from PC-15 LINE
        if ( !strcmp( tokenstr, "LOAD" ) )  	return 0xF295;
        if ( !strcmp( tokenstr, "LOC" ) )  	    return 0xF272;
        if ( !strcmp( tokenstr, "LOF" ) )  	    return 0xF273;
        if ( !strcmp( tokenstr, "MAXFILES" ) )  return 0xF288;
        if ( !strcmp( tokenstr, "MODE" ) )  	return 0xF2B3;
        if ( !strcmp( tokenstr, "MOD" ) )  	    return 0xF250;
        if ( !strcmp( tokenstr, "NAME" ) )  	return 0xF297;
        if ( !strcmp( tokenstr, "OPEN" ) )  	return 0xF296;
        if ( !strcmp( tokenstr, "OUTPUT" ) )  	return 0xF2BE;
        if ( !strcmp( tokenstr, "OUT" ) )  	    return 0xF28A;
        if ( !strcmp( tokenstr, "PAPER" ) )  	return 0xE381;
        if ( !strcmp( tokenstr, "PASS" ) )  	return 0xF2B8;
        if ( !strcmp( tokenstr, "PCONSOLE" ) )  return 0xF2B1;
        if ( !strcmp( tokenstr, "PEEK#" ) )  	return 0xF26E; // changed PC-15
        if ( !strcmp( tokenstr, "PEEK" ) )  	return 0xF26D; // changed PC-15
        if ( !strcmp( tokenstr, "PHONE" ) )     return 0xF2A0;
        if ( !strcmp( tokenstr, "PITCH" ) )     return 0xF0A4;
        if ( !strcmp( tokenstr, "POKE" ) )  	return 0xF28C; // changed PC-15, POKE# ?
        if ( !strcmp( tokenstr, "POWER" ) )  	return 0xF28B;
        if ( !strcmp( tokenstr, "PRESET" ) )  	return 0xF09A;
        if ( !strcmp( tokenstr, "PSET" ) )  	return 0xF09B;
        if ( !strcmp( tokenstr, "PZONE" ) )  	return 0xF2B4;
        if ( !strcmp( tokenstr, "RCVSTAT" ) )  	return 0xF2A4;
        if ( !strcmp( tokenstr, "RENUM" ) )  	return 0xF2B5;
        if ( !strcmp( tokenstr, "RESUME" ) )  	return 0xF28D;
        if ( !strcmp( tokenstr, "RETI" ) )  	return 0xF28E;
        if ( !strcmp( tokenstr, "RXD$" ) )  	return 0xF256;
        if ( !strcmp( tokenstr, "SAVE" ) )  	return 0xF299;
        if ( !strcmp( tokenstr, "SET" ) )  	    return 0xF298;
        if ( !strcmp( tokenstr, "SNDBRK" ) )  	return 0xF2A1;
        if ( !strcmp( tokenstr, "SNDSTAT" ) )  	return 0xF2A2;
        if ( !strcmp( tokenstr, "TAB" ) )  	    return 0xE683; // from PC-15 LCURCOR
        if ( !strcmp( tokenstr, "TIME$" ) )  	return 0xF258;
        if ( !strcmp( tokenstr, "TITLE" ) )  	return 0xF2BA;
        if ( !strcmp( tokenstr, "WAKE$" ) )  	return 0xF261;
        if ( !strcmp( tokenstr, "WIDTH" ) )  	return 0xF087;
        if ( !strcmp( tokenstr, "XCALL" ) )  	return 0xF18A; // from PC-15
        if ( !strcmp( tokenstr, "XOR" ) )  	    return 0xF251;
        if ( !strcmp( tokenstr, "XPEEK#" ) )  	return 0xF16E; // from PC-15
        if ( !strcmp( tokenstr, "XPEEK" ) )  	return 0xF16F; // from PC-15
        if ( !strcmp( tokenstr, "XPOKE#" ) )  	return 0xF1A0; // from PC-15
        if ( !strcmp( tokenstr, "XPOKE" ) )  	return 0xF1A1; // from PC-15
    }
    {   /* Dr. Schetter BMC MC-12(A) */
        if ( !strcmp( tokenstr, "AUTORANGE" ) ) return 0xE384;
        if ( !strcmp( tokenstr, "BUFFER" ) )    return 0xE363;
        if ( !strcmp( tokenstr, "BUFLEN" ) )    return 0xE350;
        if ( !strcmp( tokenstr, "BUFINIT" ) )   return 0xE3A1;
        if ( !strcmp( tokenstr, "BUFNUM" ) )    return 0xE351;
        if ( !strcmp( tokenstr, "BUFREAD" ) )   return 0xE3A3;
        if ( !strcmp( tokenstr, "BUFOPEN" ) )   return 0xE3A4;
        if ( !strcmp( tokenstr, "BUFWRITE" ) )  return 0xE3A5;
        if ( !strcmp( tokenstr, "BUFRANGE" ) )  return 0xE364;
        if ( !strcmp( tokenstr, "CHA" ) )       return 0xE361;
        if ( !strcmp( tokenstr, "COM" ) )       return 0xE3AE;
        if ( !strcmp( tokenstr, "DBUFINIT" ) )  return 0xE3AB;
        if ( !strcmp( tokenstr, "INCHA" ) )     return 0xE386;
        if ( !strcmp( tokenstr, "INIT" ) )      return 0xE390;
        if ( !strcmp( tokenstr, "INSCAN" ) )    return 0xE3AA;
        if ( !strcmp( tokenstr, "INFUNCTION" ) )return 0xE399;
        if ( !strcmp( tokenstr, "LOADBUFFER" ) )return 0xE3A6;
        if ( !strcmp( tokenstr, "MCON" ) )      return 0xE38A;
        if ( !strcmp( tokenstr, "MCOFF" ) )     return 0xE38B;
        if ( !strcmp( tokenstr, "MULTIMETER" ) )return 0xE39B;
        if ( !strcmp( tokenstr, "OUTCHA" ) )    return 0xE382;
        if ( !strcmp( tokenstr, "OUTSCREEN" ) ) return 0xE393;
        if ( !strcmp( tokenstr, "PREHIST" ) )   return 0xE355;
        if ( !strcmp( tokenstr, "POSITION" ) )  return 0xE356;
        if ( !strcmp( tokenstr, "PLOT" ) )      return 0xE39A;
        if ( !strcmp( tokenstr, "RANGE" ) )     return 0xE362;
        if ( !strcmp( tokenstr, "RELAY" ) )     return 0xE39D;
        if ( !strcmp( tokenstr, "ROM#" ) )      return 0xE357;
        if ( !strcmp( tokenstr, "SWITCH" ) )    return 0xE380;
        if ( !strcmp( tokenstr, "SCANTIME" ) )  return 0xE353;
        if ( !strcmp( tokenstr, "SCREEN" ) )    return 0xE381; // TRM PC-1600: PAPER
        if ( !strcmp( tokenstr, "SELECT" ) )    return 0xE3A9;
        if ( !strcmp( tokenstr, "SETCOM" ) )    return 0xE3AF;
        if ( !strcmp( tokenstr, "SETRANGE" ) )  return 0xE389;
        if ( !strcmp( tokenstr, "SETTRIGGER" ) )return 0xE395;
        if ( !strcmp( tokenstr, "SETSCANTIME") )return 0xE396;
        if ( !strcmp( tokenstr, "SETPREHIST" ) )return 0xE397;
        if ( !strcmp( tokenstr, "SETFUNCTION") )return 0xE398;
        if ( !strcmp( tokenstr, "SLEEP" ) )     return 0xE39E;
        if ( !strcmp( tokenstr, "TRIGGER" ) )   return 0xE354;
        if ( !strcmp( tokenstr, "TRANSREC" ) )  return 0xE39C;
        if ( !strcmp( tokenstr, "POSTLOAD" ) )  return 0xF0FC;
        if ( !strcmp( tokenstr, "RVSLOAD" ) )   return 0xF0FB;
    }
    {   /* Tramsoft Tools */
        /* Tool1 V2.0    */
        if ( !strcmp( tokenstr, "APPEND" ) )    return 0xF0C0;
        if ( !strcmp( tokenstr, "CHANGE" ) )    return 0xF0C1;
        if ( !strcmp( tokenstr, "DELETE" ) )    return 0xF0C2;
        if ( !strcmp( tokenstr, "ERASE" ) )     return 0xF0C3;
        if ( !strcmp( tokenstr, "FIND" ) )      return 0xF0C4;
        if ( !strcmp( tokenstr, "KEEP" ) )      return 0xF0C5;
        if ( !strcmp( tokenstr, "LINK" ) )      return 0xF0CA;
        if ( !strcmp( tokenstr, "PLIST" ) )     return 0xF0C6;
        if ( !strcmp( tokenstr, "PLAST" ) )     return 0xF0C9;
        if ( !strcmp( tokenstr, "PROGRAM" ) )   return 0xF0C7;
        if ( !strcmp( tokenstr, "RENUMBER" ) )  return 0xF0C8;
        if ( !strcmp( tokenstr, "SPLIT" ) )     return 0xF0CB;
        /* Tool2 V1.0    */
        if ( !strcmp( tokenstr, "FLOAD" ) )     return 0xE180;
        if ( !strcmp( tokenstr, "FSAVE" ) )     return 0xE181;
        if ( !strcmp( tokenstr, "FCHAIN" ) )    return 0xE182;
        if ( !strcmp( tokenstr, "VERIFY" ) )    return 0xE183;
        /* Tool3 V1.5    */
        if ( !strcmp( tokenstr, "CLR" ) )       return 0xE2C0;
        if ( !strcmp( tokenstr, "DEC" ) )       return 0xF070;
        if ( !strcmp( tokenstr, "ERL" ) )       return 0xF053;
        if ( !strcmp( tokenstr, "ERN" ) )       return 0xF052;
        if ( !strcmp( tokenstr, "FRC" ) )       return 0xE271;
        if ( !strcmp( tokenstr, "FRE" ) )       return 0xE250;
        if ( !strcmp( tokenstr, "HEX$" ) )      return 0xF071;
        if ( !strcmp( tokenstr, "INSTR" ) )     return 0xE273;
        if ( !strcmp( tokenstr, "PGM" ) )       return 0xE251;
        if ( !strcmp( tokenstr, "PSIZE" ) )     return 0xE252;
        if ( !strcmp( tokenstr, "PURGE" ) )     return 0xE2C1;
        if ( !strcmp( tokenstr, "REDIM" ) )     return 0xE2C2;
        if ( !strcmp( tokenstr, "RESUME" ) )    return 0xE2C3;
        if ( !strcmp( tokenstr, "STRING$" ) )   return 0xE272;
        if ( !strcmp( tokenstr, "SWAP" ) )      return 0xE2C4;
        if ( !strcmp( tokenstr, "VKEEP" ) )     return 0xE253;
        if ( !strcmp( tokenstr, "VLIST" ) )     return 0xE2C5;
        /* Tool4 V1.0    */
        if ( !strcmp( tokenstr, "AVGX" ) )      return 0xE350;
        if ( !strcmp( tokenstr, "AVGY" ) )      return 0xE351;
        if ( !strcmp( tokenstr, "CLEN" ) )      return 0xE360;
        if ( !strcmp( tokenstr, "CONVL$" ) )    return 0xE363;
        if ( !strcmp( tokenstr, "CONVS$" ) )    return 0xE364;
        if ( !strcmp( tokenstr, "CONV" ) )      return 0xE362;
        if ( !strcmp( tokenstr, "CORR" ) )      return 0xE352;
        if ( !strcmp( tokenstr, "ENTER" ) )     return 0xE3D0;
        if ( !strcmp( tokenstr, "ELAST" ) )     return 0xE35F;
        if ( !strcmp( tokenstr, "ELINE" ) )     return 0xE361;
        if ( !strcmp( tokenstr, "ELIST$" ) )    return 0xE365;
        if ( !strcmp( tokenstr, "GRA" ) )       return 0xE353;
        if ( !strcmp( tokenstr, "STATON" ) )    return 0xE3C0;
        if ( !strcmp( tokenstr, "STATOFF" ) )   return 0xE3C1;
        if ( !strcmp( tokenstr, "STATCLR" ) )   return 0xE3C2;
        if ( !strcmp( tokenstr, "STATIN" ) )    return 0xE3C3;
        if ( !strcmp( tokenstr, "STATOUT" ) )   return 0xE3C4;
        if ( !strcmp( tokenstr, "STAT" ) )      return 0xE370;
        if ( !strcmp( tokenstr, "SDNX" ) )      return 0xE354;
        if ( !strcmp( tokenstr, "SDNY" ) )      return 0xE355;
        if ( !strcmp( tokenstr, "SDVX" ) )      return 0xE357;
        if ( !strcmp( tokenstr, "SDVY" ) )      return 0xE358;
        if ( !strcmp( tokenstr, "STX" ) )       return 0xE371;
        if ( !strcmp( tokenstr, "STY" ) )       return 0xE372;
        if ( !strcmp( tokenstr, "SEG" ) )       return 0xE356;
    }
	if ( !strcmp( tokenstr, "ABS" ) )		return 0xF170;
	if ( !strcmp( tokenstr, "ACS" ) )		return 0xF174;
	if ( !strcmp( tokenstr, "AND" ) )		return 0xF150;
	if ( !strcmp( tokenstr, "AREAD" ) )		return 0xF180;
	if ( !strcmp( tokenstr, "ARUN" ) )		return 0xF181;
	if ( !strcmp( tokenstr, "ASC" ) )		return 0xF160;
	if ( !strcmp( tokenstr, "ASN" ) )		return 0xF173;
	if ( !strcmp( tokenstr, "ATN" ) )		return 0xF175;
	if ( !strcmp( tokenstr, "BEEP" ) )		return 0xF182;
	if ( !strcmp( tokenstr, "BREAK" ) )		return 0xF0B3;
	if ( !strcmp( tokenstr, "CALL" ) )		return 0xF18A;
	if ( !strcmp( tokenstr, "CHAIN" ) )		return 0xF0B2;
	if ( !strcmp( tokenstr, "CHR$" ) )		return 0xF163;
	if ( !strcmp( tokenstr, "CLEAR" ) )		return 0xF187;
	if ( !strcmp( tokenstr, "CLOAD" ) )		return 0xF089;
	if ( !strcmp( tokenstr, "CLS" ) )		return 0xF088;
	if ( !strcmp( tokenstr, "COM$" ) )		return 0xE858;
	if ( !strcmp( tokenstr, "CONSOLE" ) )	return 0xF0B1;
	if ( !strcmp( tokenstr, "CONT" ) )		return 0xF183;
	if ( !strcmp( tokenstr, "COLOR" ) )		return 0xF0B5;
	if ( !strcmp( tokenstr, "COS" ) )		return 0xF17E;
	if ( !strcmp( tokenstr, "CSAVE" ) )		return 0xF095;
	if ( !strcmp( tokenstr, "CSIZE" ) )		return 0xE680;
	if ( !strcmp( tokenstr, "CURSOR" ) )	return 0xF084;
	if ( !strcmp( tokenstr, "DATA" ) )		return 0xF18D;
	if ( !strcmp( tokenstr, "DEG" ) )		return 0xF165;
	if ( !strcmp( tokenstr, "DEGREE" ) )	return 0xF18C;
	if ( !strcmp( tokenstr, "DEV$" ) )		return 0xE857;
	if ( !strcmp( tokenstr, "DIM" ) )		return 0xF18B;
	if ( !strcmp( tokenstr, "DMS" ) )		return 0xF166;
	if ( !strcmp( tokenstr, "DTE" ) )		return 0xE884;
	if ( !strcmp( tokenstr, "END" ) )		return 0xF18E;
//	if ( !strcmp( tokenstr, "ERL" ) )		return 0xF053;
//	if ( !strcmp( tokenstr, "ERN" ) )		return 0xF052;
	if ( !strcmp( tokenstr, "ERROR" ) )		return 0xF1B4;
	if ( !strcmp( tokenstr, "EXP" ) )		return 0xF178;
	if ( !strcmp( tokenstr, "FEED" ) )		return 0xF0B0;
	if ( !strcmp( tokenstr, "FOR" ) )		return 0xF1A5;
	if ( !strcmp( tokenstr, "GCURSOR" ) )	return 0xF093;
	if ( !strcmp( tokenstr, "GLCURSOR" ) )	return 0xE682;
	if ( !strcmp( tokenstr, "GOSUB" ) )		return 0xF194;
	if ( !strcmp( tokenstr, "GOTO" ) )		return 0xF192;
	if ( !strcmp( tokenstr, "GPRINT" ) )	return 0xF09F;
	if ( !strcmp( tokenstr, "GRAD" ) )		return 0xF186;
	if ( !strcmp( tokenstr, "GRAPH" ) )		return 0xE681;
	if ( !strcmp( tokenstr, "IF" ) )		return 0xF196;
	if ( !strcmp( tokenstr, "INKEY$" ) )	return 0xF15C;
	if ( !strcmp( tokenstr, "INPUT" ) )		return 0xF091;
	if ( !strcmp( tokenstr, "INSTAT" ) )	return 0xE859;
	if ( !strcmp( tokenstr, "INT" ) )		return 0xF171;
	if ( !strcmp( tokenstr, "LCURSOR" ) )	return 0xE683;
	if ( !strcmp( tokenstr, "LEFT$" ) )		return 0xF17A;
	if ( !strcmp( tokenstr, "LEN" ) )		return 0xF164;
	if ( !strcmp( tokenstr, "LET" ) )		return 0xF198;
	if ( !strcmp( tokenstr, "LF" ) )		return 0xF0B6;
	if ( !strcmp( tokenstr, "LINE" ) )		return 0xF0B7;
	if ( !strcmp( tokenstr, "LIST" ) )		return 0xF090;
	if ( !strcmp( tokenstr, "LLIST" ) )		return 0xF0B8;
	if ( !strcmp( tokenstr, "LN" ) )		return 0xF176;
	if ( !strcmp( tokenstr, "LOCK" ) )		return 0xF1B5;
	if ( !strcmp( tokenstr, "LOG" ) )		return 0xF177;
	if ( !strcmp( tokenstr, "LPRINT" ) )	return 0xF0B9;
	if ( !strcmp( tokenstr, "MEM" ) )		return 0xF158;
	if ( !strcmp( tokenstr, "MERGE" ) )		return 0xF08F;
	if ( !strcmp( tokenstr, "MID$" ) )		return 0xF17B;
	if ( !strcmp( tokenstr, "NEW" ) )		return 0xF19B;
	if ( !strcmp( tokenstr, "NEXT" ) )		return 0xF19A;
	if ( !strcmp( tokenstr, "NOT" ) )		return 0xF16D;
	if ( !strcmp( tokenstr, "OFF" ) )		return 0xF19E;
	if ( !strcmp( tokenstr, "ON" ) )		return 0xF19C;
	if ( !strcmp( tokenstr, "OPN" ) )		return 0xF19D;
	if ( !strcmp( tokenstr, "OR" ) )		return 0xF151;
	if ( !strcmp( tokenstr, "OUTSTAT" ) )	return 0xE880;
//	if ( !strcmp( tokenstr, "P    " ) )		return 0xF1A3;
	if ( !strcmp( tokenstr, "PAUSE" ) )		return 0xF1A2;
	if ( !strcmp( tokenstr, "PEEK" ) )		return 0xF16F;
	if ( !strcmp( tokenstr, "PEEK#" ) )		return 0xF16E;
	if ( !strcmp( tokenstr, "PI" ) )		return 0xF15D;
	if ( !strcmp( tokenstr, "POINT" ) )		return 0xF168;
	if ( !strcmp( tokenstr, "POKE" ) )		return 0xF1A1;
	if ( !strcmp( tokenstr, "POKE#" ) )		return 0xF1A0;
	if ( !strcmp( tokenstr, "PRINT" ) )		return 0xF097;
	if ( !strcmp( tokenstr, "RADIAN" ) )	return 0xF1AA;
	if ( !strcmp( tokenstr, "RANDOM" ) )	return 0xF1A8;	/* F0A8 */
	if ( !strcmp( tokenstr, "READ" ) )		return 0xF1A6;
	if ( !strcmp( tokenstr, "REM" ) )		return 0xF1AB;
	if ( !strcmp( tokenstr, "RESTORE" ) )	return 0xF1A7;
	if ( !strcmp( tokenstr, "RETURN" ) )	return 0xF199;
	if ( !strcmp( tokenstr, "RIGHT$" ) )	return 0xF172;
	if ( !strcmp( tokenstr, "RINKEY$" ) )	return 0xE85A;
	if ( !strcmp( tokenstr, "RLINE" ) )		return 0xF0BA;
	if ( !strcmp( tokenstr, "RMT" ) )		return 0xE7A9;
	if ( !strcmp( tokenstr, "RND" ) )		return 0xF17C;
	if ( !strcmp( tokenstr, "ROTATE" ) )	return 0xE685;
	if ( !strcmp( tokenstr, "RUN" ) )		return 0xF1A4;
	if ( !strcmp( tokenstr, "SETCOM" ) )	return 0xE882;
	if ( !strcmp( tokenstr, "SETDEV" ) )	return 0xE886;
	if ( !strcmp( tokenstr, "SGN" ) )		return 0xF179;
	if ( !strcmp( tokenstr, "SIN" ) )		return 0xF17D;
	if ( !strcmp( tokenstr, "SORGN" ) )		return 0xE684;
	if ( !strcmp( tokenstr, "SPACE$" ) )	return 0xF061;
	if ( !strcmp( tokenstr, "SQR" ) )		return 0xF16B;
	if ( !strcmp( tokenstr, "STATUS" ) )	return 0xF167;
	if ( !strcmp( tokenstr, "STEP" ) )		return 0xF1AD;
	if ( !strcmp( tokenstr, "STOP" ) )		return 0xF1AC;
	if ( !strcmp( tokenstr, "STR$" ) )		return 0xF161;
	if ( !strcmp( tokenstr, "TAB" ) )		return 0xF0BB;
	if ( !strcmp( tokenstr, "TAN" ) )		return 0xF17F;
	if ( !strcmp( tokenstr, "TERMINAL" ) )	return 0xE883;
	if ( !strcmp( tokenstr, "TEST" ) )		return 0xF0BC;
	if ( !strcmp( tokenstr, "TEXT" ) )		return 0xE686;
	if ( !strcmp( tokenstr, "THEN" ) )		return 0xF1AE;
	if ( !strcmp( tokenstr, "TIME" ) )		return 0xF15B;
	if ( !strcmp( tokenstr, "TO" ) )		return 0xF1B1;
	if ( !strcmp( tokenstr, "TRANSMIT" ) )	return 0xE885;
	if ( !strcmp( tokenstr, "TROFF" ) )		return 0xF1B0;
	if ( !strcmp( tokenstr, "TRON" ) )		return 0xF1AF;
    if ( !strcmp( tokenstr, "UNLOCK" ) )  	return 0xF1B6;
	if ( !strcmp( tokenstr, "USING" ) )		return 0xF085;
	if ( !strcmp( tokenstr, "VAL" ) )		return 0xF162;
	if ( !strcmp( tokenstr, "WAIT" ) )		return 0xF1B3;
	if ( !strcmp( tokenstr, "ZONE" ) )		return 0xF0B4;
	else
        return 0;
  }
return 0;
}

uint TokenType (uint token)
{
    if ( ident == IDENT_EXT_BAS || ident == IDENT_E_BAS  ) {

        switch (token) {

        case 0xFE55 : /* ON */
            return TOKEN_LST ;

        case 0xFE1B : /* DELETE */
        case 0xFE15 : /* LLIST */
        case 0xFE19 : /* RENUM */
        case 0xFE67 : /* CHAIN */
            return (TOKEN_LBL | TOKEN_LST) ;

        case 0xFE74 : /* ARUN */
        case 0xFE1A : /* AUTO, E500 */
        case 0xFE75 : /* AUTOGOTO */
        case 0xFE62 : /* GOSUB */
        case 0xFE2B : /* GOTO */
        case 0xFE14 : /* LIST */
        case 0xFE66 : /* RESTORE */
        case 0xFE77 : /* RESUME */
        case 0xFE10 : /* RUN */
        case 0xFE54 : /* THEN */
            return TOKEN_LBL ;

        case 0xFE76 : /* ELSE, E500, PC-G with colon before */
            return (TOKEN_LBL | TOKEN_COL)  ;

        case 0xFE59 : /* REM */
            return TOKEN_REM ;

        default:
            return TOKEN_GEN ;
        }
    }
    else if ( ident == IDENT_PC16_BAS ) {

        switch (token) {

        case 0xF19C : /* ON */
            return TOKEN_LST ;

        case 0xF2B9 : /* DELETE */
        case 0xF0B8 : /* LLIST */
        case 0xF2B5 : /* RENUM */
        case 0xF0B2 : /* CHAIN */
            return (TOKEN_LBL | TOKEN_LST) ;

        case 0xF181 : /* ARUN */
        case 0xF2B6 : /* AUTO */
        case 0xF283 : /* ELSE */
        case 0xF194 : /* GOSUB */
        case 0xF192 : /* GOTO */
        case 0xF090 : /* LIST */
        case 0xF1A7 : /* RESTORE */
        case 0xF28D : /* RESUME */
        case 0xF1A4 : /* RUN */
        case 0xF1AE : /* THEN */
            return TOKEN_LBL ;

        case 0xF1AB : /* REM */
            return TOKEN_REM ;

        default:
            return TOKEN_GEN ;
        }
    }
    else
        return TOKEN_GEN;
}

void conv_asc2old( char *str, int len ) // IDENT_OLD_BAS
{
    int ii ;
    uchar asc, old ;

//    len = strlen (str) ;

    for ( ii = 2 ; ii < len ; ii++  ) {
        asc=str[ii] ;
        if (asc == 0) break ;
        old = asc ;

        if ((asc > 47 && asc < 58 ) || // Numbers
            (asc > 64 && asc < 91 ))   // upper chars
            old = asc + 16 ;
        else {
            if (asc == 0x60) old = 75 ; // Placeholder for Exp
            if (asc == 0x80) old = 76 ; // Placeholder for FullFrame
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

            // if (asc == 'e' ) old = 75 ;
            else if (asc > 96 && asc < 123) old = asc - 16 ; // lower chars
        }
        str[ii] = (uchar)old ;
    }
}

/* String-change UPPER, special SHARP chars are converted by strupr to false code, undone elsewhere */
char *strupr( char *string )
{
  int  i = 0;
  while ( ( string[i] = toupper( string[i] ) ) != '\0') i++;
  return string;
}

/* String-change LOWER */
char *strlor( char  *string )
{
  int  i = 0;
  while ( ( string[i] = tolower( string[i] ) ) != '\0') ++i;
  return string;
}

/* String-shift left */
char *shift_left( char *string )
{
  int  i = 0;
  while ( ( string[i] = string[i+1] ) != '\0') i++;
  return string;
}


/* String-delete leading spaces */
char *del_spaces( char *string )
{
  while ( isspace( (uchar) *string ) ) shift_left(string);
  return string;
}

/* String-delete one leading colon */
char *del_colon( char *string )
{
  if ( string[0]==COLON ) shift_left(string);
  return string;
}

/* From: http://bytes.com/topic/c/answers/223500-how-replace-substring-string-using-c
   In string "str" replace all occurrences of "orig" with "repl" */
char *replace_str2( const char *str, const char *orig, const char *repl )
{
    char  *ret, *r;
   const  char *p, *q;
  size_t  len_str = strlen(str);
  size_t  len_orig = strlen(orig);
  size_t  len_repl = strlen(repl);
  size_t  count;

  for(count = 0, p = str; (p = strstr(p, orig)); p += len_orig)
  count++;

  ret = malloc(count * (len_repl - len_orig) + len_str + 1);
  if(!ret)
  return NULL;

  for(r = ret, p = str; (q = strstr(p, orig)); p = q + len_orig) {
  count = q - p;
  memcpy(r, p, count);
  r += count;
  strcpy(r, repl);
  r += len_repl;
  }

  strcpy(r, p);
  return ret;
}


/* String-replace (for utf-8 characters also) */
void replace_str( char *str, char *orig, char *repl )
{
  strcpy( str, replace_str2(str, orig, repl));
}


/* returns one line from file without the system's native newline sequence (CRLF, CR, LF, ...) */
char *getlineF( char *retstring, ulong maxchars, FILE *datei )
{
	ulong  cin = 0;
	 char  igot = 0;
	 char  cgot = 0;
 	 bool  ineol = false;
	 bool  afteol = false;

	while ( ( igot != EOF ) && ( afteol != true ) && ( cin < maxchars ) )
	{
		igot = getc( datei );
		if ( igot != EOF )
		{
			cgot = igot;
			if ( ( cgot != '\x0D' ) && ( cgot != '\x0A' ) )
			{
				if (ineol != true)
				{
					retstring[cin++] = cgot;
				}
				else
				{
					ungetc( igot, datei );
					afteol = true;
				}
			}
			else
			{
				/* retstring[cin] = '\0'; Manfred Nosswitz 2011 */
				ineol = true;
			}
		}
	}
	retstring[cin] = '\0'; /* Finish string */
	/* printf("Zeile %u\n",__LINE__); */
	/* printf("|%s|\n",retstring); */
	if ( ( cin == 0 ) && ( igot == EOF ) )
	  return NULL;
	else
	  return retstring;
}

/*Insert needed colons before Token or Rem-Char of G-series */
int PostProcess_G  ( char  REMidC,
                     char* out_line,
                     uint* ptrI_line )
{
     char  line_buffer[cLL];

     uchar  byte ;
       int  error = ERR_OK;
      uint  ii, i_in_line=3, i_out_line=3 , /* start after the line number and length */
            token, token_type ;
     bool   string_auf = false,
            rem_line   = false,
            is_colon = false,
            pre_colon ;

    out_line[*ptrI_line] = '\x00';
    for (ii = 0; ii <= *ptrI_line; ++ii) line_buffer[ii] = out_line[ii] ;

    while ( i_in_line < *ptrI_line && error == ERR_OK ) {

        byte = (uchar) line_buffer[i_in_line++] ;
        out_line[i_out_line++] = byte ;
        if (i_out_line == cLL) {
                error = ERR_MEM ;
                break ;
        }
        pre_colon = is_colon ;
        if ( byte == ':' ) is_colon = true ;
        else is_colon = false ;

        if ( byte == '"' )string_auf = !string_auf;
        else if ( byte == REMidC && REMidC != 0) rem_line = true ;

        if ( !pre_colon && byte == REMidC ) {
                if (i_out_line+1 >= cLL) {
                        error = ERR_MEM ;
                        break ;
                }
                else { /* Insert a colon */
                        out_line[i_out_line-1] = ':' ;
                        out_line[i_out_line++] = byte ;
                }
        } // End if RemIdC

        if ( rem_line || string_auf )
            /* read and write string or a REM line */
            while ( i_in_line < *ptrI_line ) {

                byte = (uchar) line_buffer[i_in_line++] ;
                out_line[i_out_line++] = byte ;
                if (i_out_line == cLL) {
                        error = ERR_MEM ;
                        break ;
                }
                if ( !rem_line && byte == '"' ){
                    string_auf = !string_auf;
                    break;
                }
            }
            if (error != ERR_OK) break ;

        /* Is a token ? */
        if ( i_in_line+1 < *ptrI_line && byte == BAS_EXT_CODE ) {

            token = byte << 8;
            byte =  (uchar) line_buffer[i_in_line++] ;
            out_line[i_out_line++] = byte ;

            token |= byte ;
            token_type = TokenType (token) ;

            if (i_in_line >= *ptrI_line ) break ;

            if ((token_type & TOKEN_REM) != 0 ) rem_line = true ;
            else {
                if ( !pre_colon && (token_type & TOKEN_COL) != 0 ) {
                    if (i_out_line+1 >= cLL) {
                            error = ERR_MEM ;
                            break ;
                    }
                    else { /* Insert a colon */
                        out_line[i_out_line-2] = ':' ;
                        out_line[i_out_line-1] = BAS_EXT_CODE ;
                        out_line[i_out_line++] = byte ;
                    }
                } // End if TOKEN_COL
            } // End if no TOKEN_REM
        } // End if no TOKEN
    } // End of line

    if ( error != ERR_OK) for (ii = 0; ii <= *ptrI_line; ++ii) out_line[ii] = line_buffer[ii] ; /* cancel changes */
    else *ptrI_line = i_out_line ;

    return (error) ;
}
/* Convert of E series fixed Ascii numbers to BCD format SHOULD be done here in one step
   with line numbers because not to do mistakes with token ID later, but NOT implemented until now */

/* Convert fixed line numbers (jump targets) from ASCII to the intermediate format */
int CompileFixedJumpNb( char  REMidC,
                        char* out_line,
                        uint* ptrI_line )
{
     char  line_buffer[cLL],
           line_nbr[cLL] ,
           *ptrErr;

     uchar  byte ;
       int  error = ERR_OK;
      uint  ii, i_in_line=3, i_out_line=3 , /* start after the line number and length */
            line_nbr_len, token, token_type ;
    ulong   zeilennummer ;
     uint   bracket_level = 0 ;
     bool   string_auf = false,
            rem_line   = false,
            token_lst  = false ;

    out_line[*ptrI_line] = '\x00';
    for (ii = 0; ii <= *ptrI_line; ++ii) line_buffer[ii] = out_line[ii] ;
    // strncpy (line_buffer, out_line, cLL -1) ;

    while ( i_in_line < *ptrI_line && error == ERR_OK ) {

        byte = (uchar) line_buffer[i_in_line++] ;
        out_line[i_out_line++] = byte ;
        if (i_out_line == cLL) {
                error = ERR_MEM ;
                break ;
        }
        if ( byte == ':' ) {
                token_lst  = false ;
                bracket_level = 0 ;
        }
        else if ( byte == '(' ) ++bracket_level;
        else if ( byte == ')' && bracket_level >0) --bracket_level;
        else if ( byte == '"' )string_auf = !string_auf;
        else if ( byte == REMidC && REMidC != 0) rem_line = true ;

        if ( rem_line || string_auf )
            /* read and write string or a REM line */
            while ( i_in_line < *ptrI_line ) {

                byte = (uchar) line_buffer[i_in_line++] ;
                out_line[i_out_line++] = byte ;
                if (i_out_line == cLL) {
                        error = ERR_MEM ;
                        break ;
                }
                if ( !rem_line && byte == '"' ){
                    string_auf = !string_auf;
                    break;
                }
            }
            if (error != ERR_OK) break ;

        /* Is a token ? */
        if ((i_in_line+1 < *ptrI_line && (
            (   byte == BAS_EXT_CODE    && pcgrpId != GRP_16) ||
            (0xEF < byte && byte < 0xF3 && pcgrpId == GRP_16) ||
            (0xE5 < byte && byte < 0xE9 && pcgrpId == GRP_16) ||
            (   byte == 0xE3            && pcgrpId == GRP_16)  )) ||
            (   byte == ',' && token_lst && bracket_level ==0 )    ) {

            /* Is not list of line numbers */
            if ( byte != ',' && i_in_line < *ptrI_line ) {

                token = byte << 8;
                byte =  (uchar) line_buffer[i_in_line++] ;
                out_line[i_out_line++] = byte ;

                token |= byte ;
                token_type = TokenType (token) ;
            }
            if (i_in_line >= *ptrI_line ) break ;

            if ((token_type & TOKEN_REM) != 0 ) rem_line = true ;
            else {
                if ((token_type & TOKEN_LST) != 0 ) token_lst = true ;
                if ((token_type & TOKEN_LBL) != 0 ) {

                    if (( pcgrpId == GRP_16 && i_out_line+4 >= cLL) ||
                        ( pcgrpId != GRP_16 && i_out_line+3 >= cLL)  ) {
                            error = ERR_MEM ;
                            break ;
                    }
                    strcpy(line_nbr, "") ;
                    line_nbr_len = 0;
                    byte =  (uchar) line_buffer[i_in_line] ;

                    while ('0' <= byte && byte <='9') { /* fixed line number ASCII chars */
                        line_nbr[line_nbr_len] = byte ;
                        byte = (uchar) line_buffer[++line_nbr_len + i_in_line] ;
                    }
                    line_nbr[line_nbr_len] = 0 ; /* str end */

                    if ( byte == 0 || byte == ':' ||
                    (byte == ',' && token_lst == true )) {
                        /* convert line number from ASCII to binary */
                        zeilennummer = strtoul( line_nbr, &ptrErr, 0) ;
                        if (*ptrErr != 0) {
                            printf ("%s: Line number %s is not valid\n", argP, line_nbr) ;
                            error = ERR_LINE ;
                        }
                        else if ( zeilennummer > 65279) {
                            printf ("%s: Line number %s is to large\n", argP, line_nbr) ;
                            error = ERR_LINE ;
                        }
                        else if (line_nbr_len >0) {
                            /* write the converted line number */
                            out_line[i_out_line++] = BAS_EXT_LINE_NB ;
                            out_line[i_out_line++] = (uchar) (zeilennummer >>8 );
                            out_line[i_out_line++] = (uchar) (zeilennummer & 0xFF );
                            if (pcgrpId == GRP_16) out_line[i_out_line++] = 0 ;

                            i_in_line += line_nbr_len ;
                        }
                    } // Is a fixed number, no expression
                } // End if TOKEN_LBL
            } // End if no TOKEN_REM
        } // End if no TOKEN
    } // End of line
    // if ( error != ERR_OK) strncpy (out_line, line_buffer,  cLL -1) ; /* cancel changes */
    if ( error != ERR_OK) for (ii = 0; ii <= *ptrI_line; ++ii) out_line[ii] = line_buffer[ii] ; /* cancel changes */

    else *ptrI_line = i_out_line ;

    return (error) ;
}


void PrintHelp (void)  /* 1         2         3         4         5         6         7         8 */
{             /* 12345678901234567890123456789012345678901234567890123456789012345678901234567890 */
	printf ("Usage: %s [Options] SrcFile [DstFile]\n", argP) ;
	printf ("SrcFile         : BASIC-program text file\n") ;
	printf ("DstFile         : Binary image file (default: SrcFile.img or .asc)\n") ;
	printf ("Options:\n") ;
	printf ("-p, --pc=NUMBER : Sharp pocket computer, currently supported\n") ;
	printf ("                   1150, 1211, 1245, 1248, 1251, 1261, 1280, 1350, 1360, 1401\n") ;
	printf ("                   1402, 1403, 1421, 1425, 1430, 1445, 1450, 1460, 1475, 1500\n") ;
	printf ("                   1600, E220, G850 and more (default: 1500),\n") ;
	printf ("                    Only at E500 series must be used the commands 'TEXT'\n") ;
	printf ("                    and then 'BASIC' after the transfer of an img.\n\n") ;
	printf ("-t, --type=TYPE : destination file type (default: img)\n") ;
	printf ("                  img  BASIC Program binary image with intermediate code\n") ;
	printf ("                  txt  TEXT mode image,  asc  ASCII file (for Text Menu Cmt)\n\n") ;
	printf ("-q, --quiet     : Quiet mode (minimal display output)\n") ;
	printf ("    --help      : Display this information\n") ;
	printf ("    --version   : Display version information\n") ;
    printf ("-l, --level=SUM : 1     Don't compile fixed line numbers (in line)\n") ;
    printf ("                : 2     Append missing apostrophes at end of line\n") ;
    printf ("                : 4     Don't replace shortcuts(.) with commands\n") ;
    printf ("                : 8     Don't convert to upper case, 0x10 Disable preprocessor\n") ;
    printf ("                : 0x80 (0x20) Print lines in,        0x40 Print values out\n") ;
    printf ("                : 0x800 Depress some line errors, result may not editable") ;
	/*                 debug : 0x10  Deactivate preprocessor with special chars conversion */
    #ifdef __APPLE__
        /* Mac specific here, test __linux__ not for _WIN32 shell */
        printf ("\n") ;
    #endif
    #ifdef __linux__
        /* Test __linux__ */
        printf ("\n") ;
    #endif
    exit( EXIT_SUCCESS );
}


void PrintVersion (void)
{   char  argPU[cLPF] = "" ;
	strcpy(argPU, argP) ;
	printf ("%s (%s) version: 6.0.0\n", argP, strupr(argPU) ) ;
	printf ("Author: Pocket -> www.pocketmuseum.com\n") ; /* Please do not remove */
	printf ("        2013-2015 Torsten Muecker\n") ;       /* Please do not remove */
	printf ("        for complete list see the manual and the source code\n") ;
	printf ("This is free software. There is NO warranty;\n") ;
    printf ("not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n") ;
	exit( EXIT_SUCCESS ); /* no arguments were passed */
}


void MoreInfo (int error)
{	printf("%s: '%s --help' gives you more information\n", argP, argP) ;
	exit( error ); /* no arguments were passed */
}


int main( int argc, char **argv )  /* (int argc, char* argv[]) */
{               /* 0=SrcFile 1=[DstFile] 2=[-t] 3=[-p] 4=[-l] */
        char  argD[5][cLPF] = { "", "", "img", "1500", DEBUG_ARG }, \
	          line_buffer[cLL] = "", line_buffer2[cLL] = "", line_nbr[cLL] = "", \
              out_line[cLL] = "", befehl[cLC] = "", merke[cLC] = "", \
		      *ptrErr;                                  /* befehl: command, merke: note */
        char  argS[cLPF] = "", *ptrToken ,
              spec_str[]= "[FF]", spec_chr[]= "\xFF";
	    bool  string_auf = false, delREMspc = false, cnvstr_upr = false;
	    uint  ii, i_token = 0, i_out_line = 0, nbByte = 0, tmpl = 0, inline_len2 = 0, \
	          FILEcnt = 0, Tcnt = 0, PCcnt = 0, Qcnt = 0, Lcnt = 0, REMid = 0, \
	          line_cnt = 0, src_type = SOURCE_BAS,      // pcId = 0, tokenL = 0, moved to global
	          line_nbr_len, token_type ;
       uchar  type ;
        char  REMidC = 0 ;                              /* char for 2. REM ID */
	   ulong  pre_zeilennr = 0, zeilennummer = 0;       /* zeilennummer: line number */
	    FILE  *dateiein, *dateiaus;                     /* dateiein: File In, dateiaus: File Out */
       ulong  debug = 0 ;
	     int  option_index, i, j, k, l, error = ERR_OK, c = 0 ;
  static int  longval;
        bool  new_arg = false, old_arg = false, InsertMergeMark = false ;

   const int  Token_Nb = 3 ;
       char*  oldToken[] = { "PC:", "/Q", "/?" } ; /* TOKEN_NB */
       char*  newToken[] = { "-p" , "-q", "-h" } ; /* strlen 2 only */

	struct option long_options[] =
	{
		{"type",    required_argument, 0,         't'},
		{"pc",	    required_argument, NULL,      'p'},
		{"quiet",   no_argument,       NULL,      'q'},
        {"level",   required_argument, NULL,      'l'},
		{"version", no_argument,       &longval,  'v'},  /* long option only */
		{"help",    no_argument,       NULL,      'h'},
	        {0, 0, 0, 0}
	};

	/* ProgramName */
	if      (strrchr (argv[0], '\\')) strcpy(argP, 1 + strrchr (argv[0], '\\'));  /* Windows path separator '\' */
	else if (strrchr (argv[0], '/')) strcpy(argP, 1 + strrchr (argv[0], '/'));    /* Linux   path separator '/' */
	else strcpy(argP, argv[0]);
	if ( strrchr (argP, '.')) *(strrchr (argP, '.')) = '\0';                      /* Extension separator '.' */


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
        if ( strcmp(argS, "/Q") == 0 ) old_arg = true ;
        if ( strcmp(argS, "/?") == 0 ) old_arg = true ;
    }
    if ( !new_arg && old_arg) {
        printf("%s: Old format of arguments was detected", argP);
        for (i = 1; i < argc; ++i) { // 1. argument is program
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


	while (1) {

	/* getopt_long stores the option index here. */
        option_index = 0;

        c = getopt_long (argc, argv, "t:p:ql:vh", long_options, &option_index);

        /* Detect the end of the options. */
        if (c == -1)  break;

	switch (c)
          {
		case 't': strncpy( argD[2], optarg, cLPF-1); ++Tcnt; break;
		case 'p': strncpy( argD[3], optarg, cLPF-1); ++PCcnt; break;
		case 'q': ++Qcnt; break;
		case 'l': strncpy( argD[4], optarg, cLPF-1); ++Lcnt; break;
		case 'h': PrintHelp (); break;
		case 0:
		  switch (longval) {
		    case 'v': PrintVersion (); break;
		    // case 'h': PrintHelp (); break;
		  } break;
        case '?':
                  printf("%s: Unknown argument for '%s'\n", argP, argP);
		default : MoreInfo (ERR_SYNT); break;
          }
	}

	if (optind < argc) /* get non-option ARGV-elements */
    {
      while (optind < argc) {
	    strncpy(argD[FILEcnt!=0], argv[optind++], cLPF-1);
	    ++FILEcnt;
	  }
    }

	if ((FILEcnt > 2) || (Tcnt > 1) || (PCcnt > 1) || (Qcnt > 1) || (Lcnt > 1)) {
                            printf("%s: Operand error in '%s'\n", argP, argP); MoreInfo (ERR_SYNT); }
	if (FILEcnt < 1) { printf("%s: Missing Operand after '%s'\n", argP, argP); MoreInfo (ERR_SYNT); }

    (void) strlor (argD[2]) ;
    type = TYPE_NOK ;

    if (strcmp (argD[2], "img") == 0) type = TYPE_IMG ;
    else if (strcmp (argD[2], "txt") == 0) type = TYPE_TXT ;
    else if (strcmp (argD[2], "asc") == 0) type = TYPE_ASC ;
    else if (strcmp (argD[2], "bas") == 0) type = TYPE_ASC ;

    if (type == TYPE_NOK) {
        printf ("%s: Source file type %s is not valid\n", argP, argD[2]) ;
        MoreInfo (ERR_SYNT);
    }
    if (type != TYPE_IMG) {
        if ( Qcnt==0) printf ("%s: Tokenization is depressed for file type %s\n", argP, argD[2]) ;
    }

	if (FILEcnt == 1) {
		if ( strrchr (argD[0], '.') != NULL) strncat (argD[1], argD[0], strrchr(argD[0], '.') - argD[0] ); /* GetSrcFile */
		else strncpy (argD[1], argD[0], cLPF -1) ;
        if (type == TYPE_ASC) strcat (argD[1], ".asc" );
        else strcat (argD[1], ".img" );
	}

    /* Convert debug in a long */
    debug = (ulong) strtol (argD[4], &ptrErr, 0) ;
    if (*ptrErr != 0) {
        debug = 0 ;
        printf ("%s: Convert debug level number from '%s' is not valid\n", argP, argD[4]) ;
        MoreInfo (ERR_ARG);
    }
    if ((debug & 0x04) >0) shortcuts = false ;

	/* Check PC Ident */
        i = 3 ;
        if (strlen (argD[i]) == 0) {
            pcId = 1500 ;      // default pcId
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
            else if (strcmp (argD[i], "1404G")  == 0) strcpy (argD[i],  "1430") ;
            else if (strcmp (argD[i], "1405G")  == 0) strcpy (argD[i],  "1430") ;
            else if (strcmp (argD[i], "1415G")  == 0) strcpy (argD[i],  "1430") ;
            else if (strcmp (argD[i], "1416G")  == 0) strcpy (argD[i],  "1440") ;
            else if (strcmp (argD[i], "1417G")  == 0) strcpy (argD[i],  "1445") ;
            else if (strcmp (argD[i], "1450J")  == 0) strcpy (argD[i],  "1450") ;
            else if (strcmp (argD[i], "1460J")  == 0) strcpy (argD[i],  "1460") ;
            else if (strcmp (argD[i], "1470U")  == 0) strcpy (argD[i],  "1475") ;
            else if (strcmp (argD[i], "1475J")  == 0) strcpy (argD[i],  "1475") ;
            else if (strcmp (argD[i], "1500A")  == 0) strcpy (argD[i],  "1500") ;
            else if (strcmp (argD[i], "1500D")  == 0) strcpy (argD[i],  "1500") ;
            else if (strcmp (argD[i], "1500J")  == 0) strcpy (argD[i],  "1500") ;
            else if (strcmp (argD[i], "1501")   == 0) strcpy (argD[i],  "1500") ;
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
            else if (strcmp (argD[i], "PTA-4000+16")== 0) strcpy (argD[i], "1500") ; // Hiradas Technika
            else if (strcmp (argD[i], "PTA-4000")== 0) strcpy (argD[i], "1500") ;
            else if (strcmp (argD[i], "MC-2200") == 0) strcpy (argD[i], "1245") ;    // Seiko
            else if (strcmp (argD[i], "2200")   == 0) strcpy (argD[i],  "1245") ;
            else if (strcmp (argD[i], "34")     == 0) strcpy (argD[i],  "1250") ;    // Tandy
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
                 ||  strcmp (argD[i], "1600P")  == 0                 /* (1609) CE-1600P, PC-1600 Mode 0 */
                 ||  strcmp (argD[i], "1600M1") == 0                 /* (1601) CE-150, PC-1600 Mode 1 */
                 ||  strcmp (argD[i], "1600")   == 0) {
/*                   if (Qcnt == 0)  { printf ("\n%s: Fixed line numbers are not compiled to binary.\n", argP) ;
                                     printf ("         For maximal speed you have to edit this lines on PC-1600.\n\n") ;
                   }
                   if (Qcnt == 0) printf ("\n%s: Only the BASIC of 'PC-1500' is supported with PC-%s.\n", argP, argD[i]) ;
                                                      strcpy (argD[i],  "1500") ; */
                                                      pcgrpId = GRP_16 ;
                                                      REMidC = '\'' ;
                                                      strcpy (argD[i],  "1600") ;
            }
            else if (strcmp (argD[i], "E220")   == 0
                 ||  strcmp (argD[i], "220")    == 0
                 ||  strcmp (argD[i], "E200")   == 0
                 ||  strcmp (argD[i], "200")    == 0

                 ||  strcmp (argD[i], "G801")   == 0
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
/*                 if (type == TYPE_IMG && debug !=0 &&
                     Qcnt == 0) { printf ("\n%s: A simplified format is used for PC-%s. You can switch the PC\n", argP, argD[i]) ;
                                  printf ("         after transfer to the [TEXT] Editor Basic, Text, then to Basic.\n") ;
                 } */
                                                      ll_Img = 255 ;
                                                      REMidC = '\'' ;
                                                      pcgrpId = GRP_G ;
                                                      strcpy (argD[i],  "850") ;
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
                 if (type == TYPE_IMG &&
                     Qcnt == 0) { printf ("\n%s: A simplified format is used for PC-%s. If you use the native tape\n", argP, argD[i]) ;
                                  printf ("         format with Bin2wav (default) then you MUST switch the PC after\n") ;
                                  printf ("         the transfer of a IMG or TXT to 'TEXT' and back to 'BASIC' mode.\n") ;
                                  printf ("         Alternatively, the ASCII format may be used (slower, --device).\n") ;
                                  printf ("         You also can convert with Bin2wav into PC-1475 tape format (slow).\n") ;
                                  /* Decimal numbers has to be included as unsigned 0x1D + 7- or 12-byte BCD-format,
                                     also fixed line number 0x1F, jumps 0x1E, labels 0x1C in 2-byte binary format HL
                                     see also Dr. Joachim Stange: Systemhandbuch PC-E500, Ch. 6, publisher Fischel */
                 }
                                                      ll_Img = 255 ;
                                                      REMidC = '\'' ;
                                                      pcgrpId = GRP_E ;
                                                      strcpy (argD[i],  "500") ;
            }

            pcId = (uint) strtol (argD[i], &ptrErr, 0) ;
            if (*ptrErr != 0) {
                printf ("%s: Pocket computer %s is not valid\n", argP, argD[i]);
                MoreInfo (ERR_ARG);  // exit (ERR_SYNT) ;
            }
        }
	switch (pcId) {
        case 1211 : // IDENT_PC1211
        case 1245 : // IDENT_OLD_BAS
        case 1246 :
        case 1247 :
        case 1248 :
        case 1250 :
        case 1251 :
        case 1255 :
            // cnvstr_upr done by conv_asc2old ;
        case 1150 :
            {    ident=IDENT_OLD_BAS;  tokenL=1;  REMid=0xD3;   delREMspc=true;   } break; // delREMspc=false;

        case 1401 : // IDENT_NEW_BAS
        case 1402 :
        case 1421 :
        case 1430 :
        case 1431 :
            cnvstr_upr=true ;
            // hereafter PCs supports lower cases, cnvstr_upr=false
        case 1260 :
        case 1261 :
        case 1262 :
        case 1350 :
        case 1450 :
            tokenL=1 ;
            {    ident=IDENT_NEW_BAS; REMid=0xD7; delREMspc=true;  } break;

            // hereafter PCs use a variable token length, tokenL=0
        case 1440 :
        case 1445 :
            cnvstr_upr=true ;
            // hereafter PCs supports lower cases, cnvstr_upr=false
        case 1403 :
        case 1425 :
        case 1460 :
            {    ident=IDENT_NEW_BAS; REMid=0xD7; delREMspc=true;  } break;

        case  500 : // GRP was set before
            if (type == TYPE_IMG && Qcnt == 0)
                printf ("%s: Switch to 'TEXT' and back to 'BASIC' to compile jumps and numbers!\n\n", argP) ;
        case  850 :

        case 1280 : // IDENT_EXT_BAS
        case 1360 :
        case 1475 :
            {    ident=IDENT_EXT_BAS;  tokenL=2;  REMid=0xFE59; delREMspc=true;  }

            if (pcgrpId == GRP_E || pcgrpId == GRP_G ) ident=IDENT_E_BAS;
            // else if (type == TYPE_IMG && Qcnt == 0)
            //    printf ("         Short lines can transfered directly to TEXT modus with --type=txt.\n") ;

            break;

        case 1500 : // IDENT_PC15_BAS
            {    ident=IDENT_PC15_BAS; tokenL=2;  REMid=0xF1AB; delREMspc=true; } break;
        case 1600 :
            {    ident=IDENT_PC16_BAS; tokenL=2;  REMid=0xF1AB; delREMspc=true; } break;
	default:
		{
        // PC-1600, E-Series, G-Series
		printf("%s: Pocket computer PC-%u not implemented until now\n", argP, pcId );
		MoreInfo (ERR_ARG);
		}
	}

    if ( type == TYPE_TXT) {
        if ( pcgrpId == GRP_16 ) type = TYPE_ASC ; /* for comment lines saved with SAVE* CAS: */
        else if ( pcgrpId != GRP_E && ident != IDENT_EXT_BAS && ident != IDENT_NEW_BAS) {
            printf ("%s: Destination file type %s is for TEXT mode, not valid for group of PC-%u \n", argP, argD[2], pcId) ;
            if (pcgrpId == GRP_G){
                printf ("%s: Text MENU Cmt loads BAS type! Type is switched from %s to BAS now.\n", argP, argD[2]) ;
                type = TYPE_ASC ;
            }
            else {
                MoreInfo (ERR_ARG);
            }
        }
        else if ( pcgrpId == GRP_E ) { /* That CSAVEd from TEXT modus may not CLOADable*/
            printf ("%s: File type %s has to CLOAD as Bin2wav type IMG for group of PC-E%u\n", argP, argD[2], pcId) ;
        }
    }
    if ( (type == TYPE_ASC) && !(pcgrpId == GRP_16 || pcgrpId == GRP_E || pcgrpId == GRP_G)) {
        printf ("%s: Src type %s (Text Menu or device CAS:)is not valid for group of PC-%u\n", argP, argD[2], pcId) ;
        MoreInfo (ERR_ARG);
    }
    if (Qcnt == 0) {
            if (type== TYPE_IMG) printf ("Target file: BASIC image, format of PC-%u\n", pcId) ;
            else if (type== TYPE_TXT) printf ("Target file: BASIC TEXT modus image, format of PC-%u\n", pcId) ;
            else if (type== TYPE_ASC) printf ("Target file: BASIC ASCII (Text Menu Cmt/ LOAD CAS:), format of PC-%u\n", pcId) ;
    }

	/* Function CountBits uses <limits.h> */
/*	printf("CHAR_BIT  = %d\n", 8 * sizeof( char ) );
	printf("SHORT_BIT = %d\n", 8 * sizeof( short ) );
	printf("INT_BIT   = %d\n", 8 * sizeof( int ) );
	printf("LONG_BIT  = %d\n", 8 * sizeof( long ) );
*/
/*	if ( ( dateiein = fopen( argD[0], "rt" ) ) != NULL )  */      /* end-of-line sequence is system-dependent! */
	if ( ( dateiein = fopen( argD[0], "rb" ) ) != NULL )
	{
		if (Qcnt == 0 && debug !=0 ) printf("File %s was opened\n", argD[0] );
		/* Get the length of the source file */
		fseek( dateiein, 0, SEEK_END ) ;   /* set file position indicator to end */
		nbByte = ftell( dateiein ) ;       /* get file position indicator */
		fseek( dateiein, 0, SEEK_SET ) ;   /* set file position indicator to start */
		if (Qcnt == 0) printf("File size of %s is %u bytes\n\n", argD[0], nbByte );
		dateiaus = fopen( argD[1], "wb" ); /* write in binary mode */

        /* Begin of line processing */

      		/* while ( fgets( line_buffer, cLL, dateiein ) != NULL ) */
		while ( getlineF( line_buffer, cLL, dateiein ) != NULL )
		{  /* get one line - eine zeile rausholen */

            /* Begin of preprocessor of a line, starting main processing of a line  */
		    string_auf = false ; // string_open

			if (++line_cnt == 1) {
                /* starting first line in text with a '.' is usable to convert more chars from DOS-US */
                if((uchar) *line_buffer == '.') {
                    /*found src_type SHA"  */
                    src_type = SOURCE_SHA ;
                    continue ; /* ignore the header line of a SHA-file */
                }
                /* remove artifact: First byte in file is "\xF8" if serial transfer was done with some hardware */
                if ((uchar) *line_buffer >= 0xF8) shift_left( line_buffer ) ;
			}
			del_spaces( line_buffer );  /* Delete leading spaces */
			if (strlen (line_buffer) == 0) continue ; /* empty line */

            /* Progress (Fortschrittsanzeige) DOSBox 0.73,  Win XP 900 MHz */
			if (Qcnt == 0 && (debug & 0x80) >0 ) {
                    printf("%s\n", line_buffer );
			}
			/* Ignore comment lines in BASIC source, offline comments are allowed now */
			if ((uchar) *line_buffer == '\'') continue ; /* not transfered, next line */

			if (Qcnt == 0 && (debug & 0x20) >0 ) {
			/* Show content of line_buffer as hex values */
                i = 0;
                printf ("i<");
                while ( line_buffer[i] != '\0') { printf ("%02X ", (uchar) line_buffer[i]); i++; }
                    printf ("%02X", (uchar) line_buffer[i]);
                printf (">\n") ;
			}

            /* Begin of the preprocessor */
            if ((debug & 0x10) == 0) {
                /* replaces DOS-US graphic chars, often in older Listings found, if this would */
                /* conflict with usable UTF-chars then separate option for bas2img --utf8 would needed */
                replace_str(line_buffer, "\xFB", "[SQR]") ; /* .SHA  √ <-- xFB */
                replace_str(line_buffer, "\xE3", "[PI]") ;  /* .SHA  π <-- xE3 */
                replace_str(line_buffer, "\x90", "[E]") ;   /* .SHA  € <-- x90 */

                /* tested with "TransFile PC 5.55 D" and PC-1401/1251 */
                if (src_type == SOURCE_SHA) {
                    replace_str(line_buffer, "\x9D", "[Y]") ;   /* .SHA  ¥ <-- x90 */
                    if (ident == IDENT_OLD_BAS ) {
                        replace_str(line_buffer, "\xF0", "[10]") ; /* Space Frame */
                        replace_str(line_buffer, "\xDB", "[4C]") ; /* FullFrame rectangle */
                    }
                    else if (ident == IDENT_NEW_BAS && cnvstr_upr == true ) {
                        replace_str(line_buffer, "\xF0", "[FD]") ; /* Space Frame */
                        replace_str(line_buffer, "\xDB", "[FE]") ; /* FullFrame rectangle */
                    }
                    else if (ident == IDENT_NEW_BAS || ident == IDENT_EXT_BAS ) {
                        replace_str(line_buffer, "\xF0", "[FA]") ; /* Space Frame */
                        replace_str(line_buffer, "\xDB", "[F9]") ; /* FullFrame rectangle */
                        replace_str(line_buffer, "\x06", "[F5]") ; /* Card symbols ♠ ♥ ♦ ♣ */
                        replace_str(line_buffer, "\x03", "[F6]") ;
                        replace_str(line_buffer, "\x04", "[F7]") ;
                        replace_str(line_buffer, "\x05", "[F8]") ;
                    }
                    else if (ident == IDENT_PC15_BAS ) {
                        replace_str(line_buffer, "\xF0", "[27]") ; /* Space Frame */
                        replace_str(line_buffer, "\xDB", "[7F]") ; /* FullFrame rectangle */
                    }
//                  else if (ident == IDENT_PC16_BAS || ident == IDENT__BAS ) {
//                  }
                } // end if SHA

                i = 0;

                switch (ident) {
                case IDENT_OLD_BAS :

                    replace_str(line_buffer, "Π", "\x19");
                    replace_str(line_buffer, "π", "\x19");
                    replace_str(line_buffer, "[PI]", "\x19");
                    replace_str(line_buffer, "[SQR]", "\x1A");
                    replace_str(line_buffer, "√", "\x1A");
                    replace_str(line_buffer, "[E]", "\x60"); // Placeholder for Exp before conv_asc2old
                    replace_str(line_buffer, "€", "\x60");
                    replace_str(line_buffer, "[Y]", "\x17");
                    replace_str(line_buffer, "¥", "\x17");
                    replace_str(line_buffer, "□", "\x10");
                    replace_str(line_buffer, "[INS]", "\x10");
                    replace_str(line_buffer, "■", "\x80");// Placeholder for FullFrame before conv_asc2old
                    replace_str(line_buffer, "[FUL]", "\x80");// Placeholder
                    replace_str(line_buffer, "[4C]", "\x80"); // Placeholder FUL
                    /* reconstruct special chars generated by wav2bin from placeholders - moved to end of section*/
/*                  replace_str(line_buffer, "[10]", "\x10");
                    replace_str(line_buffer, "[3A]", "\x3A"); // -3F Japanese chars
*/
                    while ( line_buffer[i] != '\0' ) {
                        if ( (uchar) line_buffer[i] > 0x80 ) {
                            printf ("%s: Character 0x%02X is not valid\n", argP, (uchar) line_buffer[i]);
                        }
                        i++;
                    }
                    replace_str(line_buffer, ">=", "\x82");  // istoken does not work for this
                    replace_str(line_buffer, "<=", "\x83");
                    replace_str(line_buffer, "<>", "\x84");
                    replace_str(line_buffer, "[S]", "\x9B"); // S ETCOM
                    replace_str(line_buffer, "[O]", "\x9C"); // O PEN
                    replace_str(line_buffer, "[C]", "\x9D"); // C LOSE

                    break ;

                case IDENT_EXT_BAS :
                case IDENT_NEW_BAS :

                    /* replaces inside strings for line_buffer2 */
                    if (ident == IDENT_NEW_BAS && cnvstr_upr == true ) {
                        replace_str(line_buffer, "□", "\xFD") ;     // U+25A1
                        replace_str(line_buffer, "■", "\xFE") ;     // U+25A0
                        replace_str(line_buffer, "[INS]", "\xFD") ;
                        replace_str(line_buffer, "[FUL]", "\xFE") ;
                        if (pcId == 1421) {
                            replace_str(line_buffer, " i ", "i") ;
                            replace_str(line_buffer, " n ", "n") ;
                        }
                    }
                    else {
                        replace_str(line_buffer, "□", "\xFA") ;
                        replace_str(line_buffer, "■", "\xF9") ;
                        replace_str(line_buffer, "[INS]", "\xFA") ;
                        replace_str(line_buffer, "[FUL]", "\xF9") ;
                    }
                    replace_str(line_buffer, "♠", "\xF5");
                    replace_str(line_buffer, "♥", "\xF6");
                    replace_str(line_buffer, "♦", "\xF7");
                    replace_str(line_buffer, "♣", "\xF8");
                    replace_str(line_buffer, "¥", "\x5C");
                    replace_str(line_buffer, "Π", "\xFB");
                    replace_str(line_buffer, "π", "\xFB");
                    replace_str(line_buffer, "√", "\xFC");      // U+221A
                    replace_str(line_buffer, "€", "\x45");
                    /* reconstruct special chars generated by wav2bin */
                    replace_str(line_buffer, "[PI]", "\xFB");
                    replace_str(line_buffer, "[SQR]", "\xFC");
                    replace_str(line_buffer, "[Y]", "\x5C");
                    replace_str(line_buffer, "[E]", "\x45");
/*                  replace_str(line_buffer, "[5C]", "\x5C");   // Yen
                    replace_str(line_buffer, "[F5]", "\xF5"); - FE  */

                    while ( line_buffer[i] != '\0') {
                        if ( ( line_buffer[i] < 0 ) && !(( line_buffer[i] >= '\xF5' ) && ( line_buffer[i] <= '\xFE' )) ) {
                            printf ("%s: Character 0x%02X is not valid\n", argP, (uchar) line_buffer[i]);
                        }
                        i++;
                    }
                    break ;

                case IDENT_PC15_BAS :

                    replace_str(line_buffer, "□", "\x27");
                    replace_str(line_buffer, "[INS]", "\x27");
/*                  replace_str(line_buffer, "[27]", "\x27");
                    replace_str(line_buffer, "[7F]", "\x7F"); */
                    replace_str(line_buffer, "■", "\x7F");
                    replace_str(line_buffer, "[FUL]", "\x7F");
                    replace_str(line_buffer, "√", "\x5B");
                    replace_str(line_buffer, "¥", "\x5C");
                    replace_str(line_buffer, "Π", "\x5D");
                    replace_str(line_buffer, "π", "\x5D");
                    replace_str(line_buffer, "€", "\x45");
                    /* reconstruct special chars generated by wav2bin */
                    replace_str(line_buffer, "[PI]", "\x5D");
                    replace_str(line_buffer, "[SQR]", "\x5B");
                    replace_str(line_buffer, "[Y]", "\x5C");
                    replace_str(line_buffer, "[E]", "\x45");

                    while ( line_buffer[i] != '\0') {
                        if ( line_buffer[i] < 0 ) {
                            printf ("%s: Character 0x%02X is not valid\n", argP, (uchar) line_buffer[i]);
                        }
                        i++;
                    }
                    break ;

                case IDENT_PC16_BAS :
                case IDENT_E_BAS :

                    replace_str(line_buffer, "□", "\xF0") ;
                    replace_str(line_buffer, "■", "\xDB") ;
                    replace_str(line_buffer, "[INS]", "\xF0") ;
                    replace_str(line_buffer, "[FUL]", "\xDB") ;

                    replace_str(line_buffer, "√", "\xFB"); // SQR
                    replace_str(line_buffer, "¥", "\\");
                    replace_str(line_buffer, "Π", "PI ");
                    replace_str(line_buffer, "π", "\xE3");
                    replace_str(line_buffer, "€", "E");
                    /* reconstruct special chars generated by wav2bin */
                    replace_str(line_buffer, "[PI]", "PI ");
                    replace_str(line_buffer, "[SQR]", "SQR ");
                    replace_str(line_buffer, "[Y]", "\\");
                    replace_str(line_buffer, "[E]", "E");

                } // end switch ident

                /* replace all placeholders for special chars of one byte only */
                // ToDo: replace placeholders for special chars of two bytes /
                for ( ii = 0 ; ii < 0x100 ; ++ii) {
                    if (strchr(line_buffer, '[') == NULL) break ;

                    tmpl = ii>>4 ;
                    if (tmpl > 9) spec_str[1] = tmpl + 0x37 ; else spec_str[1] = tmpl | 0x30 ;
                    tmpl = ii & 0x0F ;
                    if (tmpl > 9) spec_str[2] = tmpl + 0x37 ; else spec_str[2] = tmpl | 0x30 ;
                    spec_chr[0] = ii ;
                    replace_str(line_buffer, spec_str, spec_chr);
                }
            }
            /* End of preprocessor of a line, starting main processing of a line  */
			inline_len2 = strlen (line_buffer);

			/* Get linenumber */
			while ( isdigit( (uchar) *line_buffer ) )
			{
				strncat( line_nbr, line_buffer, 1 );
				shift_left( line_buffer );
			}
			strcat( line_nbr, "\0" );  /* Finish string */

			/* line_buffer2 protects text constants in BASIC Lines */
			// if (!cnvstr_upr)
            strcpy( line_buffer2, line_buffer );
            // printf("DEBUG: %s\n", setlocale(LC_CTYPE, "C"));
            if ((debug & 8) == 0) {
                strupr( line_buffer );

                if (pcId == 1421) /* special financial variables */
                    for ( ii = strlen( line_buffer ) ; ii > 0 ; ) {
                        --ii;
                        if (line_buffer2[ii]=='i') line_buffer[ii] = 'i' ;
                        if (line_buffer2[ii]=='n') line_buffer[ii] = 'n' ;
                    }

                /* Special chars are converted by strupr to false code, undo it */
                if ( ident == IDENT_NEW_BAS || ident == IDENT_EXT_BAS ) {
                    replace_str(line_buffer, "\xDB", "\xFB");   // PI
                    replace_str(line_buffer, "\xDC", "\xFC");   // SQR
                    if (cnvstr_upr) {                               // for text constants
                        replace_str(line_buffer, "\xDA", "\xFA");   // ~
                        replace_str(line_buffer, "\xDD", "\xFD");   // Frame
                        replace_str(line_buffer, "\xDE", "\xFE");   // Rect filled
                    }
                }
                else if ( ident == IDENT_PC16_BAS || ident == IDENT_E_BAS ) {
                    replace_str(line_buffer, "\xDB", "\xFB");   // SQR E437
                }
                if (cnvstr_upr) strcpy( line_buffer2, line_buffer );
            }

			zeilennummer = strtoul ( line_nbr, &ptrErr, 0) ;
			if (*ptrErr != 0) {
			  printf ("%s: Line number %s is not valid\n", argP, line_nbr ) ;
              error = ERR_LINE ;
              break ;
			}
			line_nbr_len = strlen (line_nbr);

			if ( zeilennummer )
			{
                strcpy( out_line, "" );
                i_out_line = 0;

                if (type == TYPE_ASC) {
                    strcpy( out_line, line_nbr );
                    i_out_line = line_nbr_len;
                }
				else if (! ( pre_zeilennr < zeilennummer ) ) {
					printf("%s: Merged line number %lu: not higher than previous line number %lu\n",
                            argP, zeilennummer, pre_zeilennr );
					// printf("%s: Previous line number: %lu\n", argP, pre_zeilennr );
					// printf("%s: Actual line number: %lu\n", argP, zeilennummer );
					if ( pcgrpId == GRP_E || pcgrpId == GRP_G) {
                        printf("%s: This line number sequence is illegal for group of %i\n", argP, pcId);
                        if ((debug & 0x800) == 0 ) {
                            error = ERR_LINE ;
                            break ;
                        }
                    }
                    else if (ident != IDENT_OLD_BAS && (debug & 0x800) == 0) InsertMergeMark = true;
                }
				strcpy( line_nbr, "" );

				if ( ident == IDENT_OLD_BAS )
				{
					if ( zeilennummer > 999 )
					{
                      if ( zeilennummer == MERGE_MARK) continue ; /* not used in OLD_BAS read next line*/

					  printf("%s: Line number: %lu higher than 999 (MAX line number)\n", argP, zeilennummer );
					  if ((debug & 0x800) == 0) {
                            error = ERR_LINE ;
                            break ;
                        }
					}
                    if (type != TYPE_ASC) {
                        out_line[0] = (uchar) ( zeilennummer / 100 | 0xE0 );
                        i_out_line++;
                        tmpl = zeilennummer - (100 * out_line[0]);
                        out_line[1] = (uchar) ( ((tmpl / 10 ) << 4) | (tmpl % 10) );
                        i_out_line++;
					}
				}
				else
				{
					if ( zeilennummer == MERGE_MARK || InsertMergeMark ) { /* merged basic programs */

                        if ( zeilennummer == MERGE_MARK && (pcgrpId == GRP_E || pcgrpId == GRP_G))
                                                  continue ; /* not support in E_BAS, read next line*/
                        if (type != TYPE_ASC) {
                            out_line[0] = '\xFF'; /* Start mark of a merged basic program block */
                            i_out_line++;

                            if (Qcnt == 0 && (debug & 0x40) >0 ) printf ("o> %02X<\n", (uchar) out_line[0]);
                        }
                        else if (Qcnt == 0 && (debug & 0x40) >0) printf ("o> %s<\n", out_line);

                        fwrite( out_line, sizeof( char ), i_out_line, dateiaus );

                        if (InsertMergeMark) {
                            InsertMergeMark = false ;
                            if (Qcnt ==0) printf ("%s: Merging line %lu was missed but mark inserted.\n",
                                                  argP, (ulong) MERGE_MARK);
                            /* process the actual line*/
                            strcpy( out_line, "" );
                            i_out_line = 0;
                        }
                        else { // MERGE_MARK
                            pre_zeilennr = 0;
                            continue ; /* read next line*/
                        }

					}
					if ( zeilennummer > 65279 ) {
					  printf("%s: Line number: %lu higher than 65279 (MAX line number)\n", argP, zeilennummer );
                      error = ERR_LINE ;
                      break ;
					}
                    if (type != TYPE_ASC) {
                        out_line[0] = (uchar) (zeilennummer / 256 );
                        i_out_line++;
                        out_line[1] = (uchar) (zeilennummer);
                        i_out_line++;

                        out_line[2] = '\xFF'; /* Placeholder for length of Binary image command line */
                        i_out_line++;         /* T. Muecker - not in images of type OLD_BAS */
                    }
				}
                pre_zeilennr = zeilennummer;

                if (inline_len2 > ll_Img && Qcnt==0)  /* long source line, text modus may fail */
                      printf ("%s: The length of line number %lu: is %u (for the text editor).\n",
                              argP, zeilennummer, inline_len2) ;

				/* Delete colon following the line number */
                del_spaces( line_buffer );
                del_spaces( line_buffer2 );
                del_colon( line_buffer );
				del_colon( line_buffer2 );

				if (type == TYPE_TXT || type == TYPE_ASC ) {
					del_spaces( line_buffer );
					del_spaces( line_buffer2 );
                    if (type == TYPE_ASC && pcgrpId != GRP_G) {
                        out_line[i_out_line++] = 0x20;
					}
                    do
                    {
                        memcpy( &out_line[i_out_line], line_buffer2, 1 );
                        i_out_line++;
                        shift_left( line_buffer );
                        shift_left( line_buffer2 );
                    } while ( !( *line_buffer == '\0' ) );
				}
                else  { /*TYPE_IMG*/
                  if (ident == IDENT_E_BAS) token_type = TOKEN_LBL ; /* force check for label */

				  do  /* Process the body of a line */
				  {  /* Delete leading spaces */
                    if (ident == IDENT_E_BAS) {
                        del_spaces( line_buffer );
                        del_spaces( line_buffer2 );
                        if ( *line_buffer == '*' && (token_type & TOKEN_LBL) != 0 ) { /* Label of E-Series */
                            do
                            {
                                memcpy( &out_line[i_out_line], line_buffer, 1 );
                                i_out_line++;
                                shift_left( line_buffer );
                                shift_left( line_buffer2 );
                              /* Definition, which characters may be included in a label */
                            } while ( !( *line_buffer == ' ' || *line_buffer == ':' || /* G850 also without : */
                                         *line_buffer == ',' || *line_buffer == REMidC ||
                                         *line_buffer == '"' || *line_buffer == '\0' )  &&
                                    ((*line_buffer >= '@' && *line_buffer <= 'Z') ||
                                     (*line_buffer >= '0' && *line_buffer <= '9') ||
                                      *line_buffer == '[' || *line_buffer == ']'  ||
                                      *line_buffer == '?' || *line_buffer == '_'   )     );
                        }
                    }
					del_spaces( line_buffer );
					del_spaces( line_buffer2 );
					/* Comment OR $-out */
					if ( *line_buffer == '"' )
					{
						string_auf = !string_auf;
						if ( string_auf )
						{
							do
							{
								memcpy( &out_line[i_out_line], line_buffer2, 1 );
								i_out_line++;
								shift_left( line_buffer );
								shift_left( line_buffer2 );
							} while ( !( *line_buffer == '"' || *line_buffer == '\0' ) );
						}
						else
						{
							do
							{
								memcpy( &out_line[i_out_line], line_buffer, 1 );
								i_out_line++;
								shift_left( line_buffer );
								shift_left( line_buffer2 );
							} while ( !( *line_buffer == '"' || *line_buffer == '\0' ) );
						}
						if ( *line_buffer == '"' )
						{
							string_auf = !string_auf;
							memcpy( &out_line[i_out_line], line_buffer, 1 );
							i_out_line++;
							shift_left( line_buffer );
							shift_left( line_buffer2 );
						}
					}
					if ( *line_buffer == ':' )
					{
						memcpy( &out_line[i_out_line], line_buffer, 1 );
						i_out_line++;
						shift_left( line_buffer );
						shift_left( line_buffer2 );
					}
                        /* Definition, which characters may be included in a token */
					/* while ( !( *line_buffer == '=' || *line_buffer == ':' || *line_buffer == REMidC ||\
						   *line_buffer == '"' || *line_buffer == '\0' || *line_buffer == ' ' ||\
						   *line_buffer == ',' || *line_buffer == ';') ) */
					while ( ( *line_buffer >= 'A' && *line_buffer <= 'Z' ) ||\
                              *line_buffer == '$' || *line_buffer == '#' ||\
						    ( *line_buffer == '.' && shortcuts ) )
					{
						if (strlen(befehl) < cLC-1) strncat( befehl, line_buffer, 1 );
						else {
                            if (Qcnt==0)
                                printf ("%s: A word with excessive length found in line with number %lu.\n",
                                        argP, zeilennummer) ;
                            break ;
						}
						shift_left( line_buffer );
						shift_left( line_buffer2 );
					}
					if ( strlen( befehl ) )
					{
						i_token = istoken( befehl ); // 1. try
                        if ( i_token > 0) token_type = TokenType (i_token); else if (strlen(befehl)>0) token_type = 0;
						if ( i_token == REMid )  /* REM */
						{
							strcpy( befehl, "" );
							if  ( tokenL == 2 || i_token > 255 ) {  /* 2Byte-Token */
							out_line[i_out_line] = (uchar) ( i_token / 256 );
							i_out_line++; }
							out_line[i_out_line] = (uchar) i_token;
							i_out_line++;
							i_token = 0;
							if  ( delREMspc ) {  /* Delete spaces after REM-Command */
							del_spaces( line_buffer );
							del_spaces( line_buffer2 ); }
							while ( !( *line_buffer == '\0' ) )
							{
                                out_line[i_out_line] = *line_buffer2;
								i_out_line++;
								shift_left( line_buffer );
								shift_left( line_buffer2 );
							}
						}
						do /* Forward search */
						{
							while ( i_token == 0 && strlen( befehl ) > 2 )
							{
								i_token = istoken( befehl );  // 2. try
                                if ( i_token > 0) token_type = TokenType (i_token); else if (strlen(befehl)>0) token_type = 0;
								if ( i_token == REMid )  /* REM */
								{
									strcpy( befehl, "" );
									if  ( tokenL == 2 || i_token > 255 ) {  /* 2Byte-Token */
									out_line[i_out_line] = (uchar) ( i_token / 256 );
									i_out_line++; }
									out_line[i_out_line] = (uchar) i_token;
									i_out_line++;
									i_token = 0;
									if  ( delREMspc ) {  /* Delete spaces after REM-Command */
									del_spaces( line_buffer );
									del_spaces( line_buffer2 ); }
									while ( !( *line_buffer == '\0' ) )
									{
                                        out_line[i_out_line] = *line_buffer;
                                        i_out_line++;
										shift_left( line_buffer );
										shift_left( line_buffer2 );
									}
								}
								if ( i_token )
								{
									strcpy( befehl, "" );
									if  ( tokenL == 2 || i_token > 255 ) {  /* 2Byte-Token */
									out_line[i_out_line] = (uchar) ( i_token / 256 );
									i_out_line++; }
									out_line[i_out_line] = (uchar) i_token;
									i_out_line++;
									i_token = 0;
								}
								else
								{   /* Memorize the extra characters, stored in reverse order */
									strcat( merke, befehl + strlen( befehl ) - 1 );
									/* shorten the possible command string */
									*( befehl + strlen( befehl ) - 1 ) = '\0';
								}
								i_token = istoken( befehl ); // 3. try
                                if ( i_token > 0) token_type = TokenType (i_token); else if (strlen(befehl)>0) token_type = 0;
								if ( i_token == REMid )  /* REM */
								{
									strcpy( befehl, "" );
									if  ( tokenL == 2 || i_token > 255 ) {  /* 2Byte-Token */
									out_line[i_out_line] = (uchar) ( i_token / 256 );
									i_out_line++; }
									out_line[i_out_line] = (uchar) i_token;
									i_out_line++;
									i_token = 0;
									if  ( delREMspc ) {  /* Delete spaces after REM-Command */
									del_spaces( line_buffer );
									del_spaces( line_buffer2 ); }
									while ( strlen( merke ) )
									{
										strcat( befehl, ( merke + strlen( merke ) - 1 ) );
										*( merke + strlen( merke ) - 1 ) = '\0';
									}
									memcpy( &out_line[i_out_line], befehl, strlen( befehl ) );
									i_out_line += strlen( befehl );
									strcpy( befehl, "" );
									while ( !( *line_buffer == '\0' ) )
									{
										out_line[i_out_line] = *line_buffer2;
										i_out_line++;
										shift_left( line_buffer );
										shift_left( line_buffer2 );
									}
								}
								if ( i_token )
								{
									strcpy( befehl, "" );
									if  ( tokenL == 2 || i_token > 255 ) {  /* 2Byte-Token */
									out_line[i_out_line] = (uchar) ( i_token / 256 );
									i_out_line++; }
									out_line[i_out_line] = (uchar) i_token;
									i_out_line++;
									while ( strlen( merke ) )
									{
										strcat( befehl, ( merke + strlen( merke ) - 1 ) );
										*( merke + strlen( merke ) - 1 ) = '\0';
									}
									i_token = 0;
								}
							}
							while ( strlen ( merke ) )
							{
								strcat( befehl, ( merke + strlen( merke ) - 1 ) );
								*( merke + strlen( merke ) - 1 ) = '\0';
							}
							i_token = istoken( befehl ); // 4. try
                            if ( i_token > 0) token_type = TokenType (i_token); else if (strlen(befehl)>0) token_type = 0;
							if ( i_token )
							{
								if ( i_token == REMid )  /* REM */
								{
									strcpy( befehl, "" );
									if  ( tokenL == 2 || i_token > 255 ) {  /* 2Byte-Token */
									out_line[i_out_line] = (uchar) ( i_token / 256 );
									i_out_line++; }
									out_line[i_out_line] = (uchar) i_token;
									i_out_line++;
									i_token = 0;
									if  ( delREMspc ) {  /* Delete spaces after REM-Command */
									del_spaces( line_buffer );
									del_spaces( line_buffer2 ); }
									while( strlen( merke ) )
									{
										strcat( befehl, ( merke + strlen( merke ) - 1 ) );
										*( merke + strlen( merke ) - 1 ) = '\0';
									}
									memcpy( &out_line[i_out_line], befehl, strlen( befehl ) );
									i_out_line += strlen( befehl );
									strcpy( befehl, "" );
									while( !( *line_buffer == '\0' ) )
									{
										out_line[i_out_line] = *line_buffer;
										i_out_line++;
										shift_left( line_buffer );
										shift_left( line_buffer2 );
									}
								}
								else
								{
									strcpy( befehl, "" );
									if  ( tokenL == 2 || i_token > 255 ) {  /* 2Byte-Token */
									out_line[i_out_line] = (uchar) ( i_token / 256 );
									i_out_line++; }
									out_line[i_out_line] = (uchar) i_token;
									i_out_line++;
									i_token = 0;
								}
							}
							if ( strlen( befehl) && !i_token )
							{
								out_line[i_out_line] = *befehl;
								i_out_line++;
								shift_left( befehl );
							}
						} while ( strlen( befehl ) > 2 && !i_token );
						i_token = istoken( befehl ); // 5. try
                        if ( i_token > 0) token_type = TokenType (i_token); else if (strlen(befehl)>0) token_type = 0;
						if ( i_token )
						{
							if ( i_token == REMid )  /* REM */
							{
								strcpy( befehl, "" );
								if  ( tokenL == 2 || i_token > 255 ) {  /* 2Byte-Token */
								out_line[i_out_line] = (uchar) ( i_token / 256 );
								i_out_line++; }
								out_line[i_out_line] = (uchar) i_token;
								i_out_line++;
								i_token = 0;
								if  ( delREMspc ) {  /* Delete spaces after REM-Command */
								del_spaces( line_buffer );
								del_spaces( line_buffer2 ); }
								while ( strlen( merke ) )
								{
									strcat( befehl, ( merke + strlen( merke ) - 1 ) );
									*( merke + strlen( merke ) - 1 ) = 0;
								}
								memcpy( &out_line[i_out_line], befehl, strlen( befehl ) );
								i_out_line += strlen( befehl );
								strcpy( befehl, "" );
								while ( !( *line_buffer == '\0' ) )
								{
									out_line[i_out_line] = *line_buffer;
									i_out_line++;
									shift_left( line_buffer );
									shift_left( line_buffer2 );
								}
							}
							else
							{
								strcpy( befehl, "" );
								if  ( tokenL == 2 || i_token > 255 ) {  /* 2Byte-Token */
								out_line[i_out_line] = (uchar) ( i_token / 256 );
								i_out_line++; }
								out_line[i_out_line] = (uchar) i_token;
								i_out_line++;
								i_token = 0;
							}
						}
					}
					else /* strlen(befehl) == 0, No token */
					{
						i_token = 0;
						if ( !( *line_buffer == '\0' || *line_buffer == ' ' ) )
						{
                            if ( *line_buffer == REMidC ) {
                                while ( !( *line_buffer == '\0' ) )
                                {
                                    out_line[i_out_line] = *line_buffer2;
                                    i_out_line++;
                                    shift_left( line_buffer );
                                    shift_left( line_buffer2 );
                                }
                            }
                            else {
                                if (*line_buffer == ',' ) token_type = TOKEN_LBL;
                                else token_type = 0; /* for E_BAS labels only */

                                out_line[i_out_line] = *line_buffer;
                                i_out_line++;
                                shift_left( line_buffer );
                                shift_left( line_buffer2 );
                            }
						}
						else token_type = 0;
					}
					if ( i_token == REMid )  /* REM */
					{
						strcpy( befehl, "" );
						if  ( tokenL == 2 || i_token > 255 ) {  /* 2Byte-Token */
						out_line[i_out_line] = (uchar) ( i_token / 256 );
						i_out_line++; }
						out_line[i_out_line] = (uchar) i_token;
						i_out_line++;
						i_token = 0;
						if  ( delREMspc ) {  /* Delete spaces after REM-Command */
						del_spaces( line_buffer );
						del_spaces( line_buffer2 ); }
						while ( strlen( merke ) )
						{
							strcat( befehl, ( merke + strlen( merke ) - 1 ) );
							*( merke + strlen( merke ) - 1 ) = '\0';
						}
						memcpy( &out_line[i_out_line], befehl, strlen( befehl ) );
						i_out_line += strlen( befehl );
						strcpy( befehl, "" );
						while ( !( *line_buffer == '\0' ) )
						{
							out_line[i_out_line] = *line_buffer;
							i_out_line++;
							shift_left( line_buffer );
							shift_left( line_buffer2 );
						}
					}
					if ( i_token )
					{
						if  ( tokenL == 2 || i_token > 255 ) {  /* 2Byte-Token */
						out_line[i_out_line] = (uchar) ( i_token / 256 );
						i_out_line++; }
						out_line[i_out_line] = (uchar) i_token;
						i_out_line++;
					}
					else
					{
						if ( strlen( befehl ) )
						{
							memcpy( &out_line[i_out_line], befehl, strlen( befehl ) );
							i_out_line += strlen( befehl );
							strcpy( befehl, "" );
						}
					}
				  } while ( !( *line_buffer == '\0' ) ); /* until End Of Line */
				  if ( string_auf && debug & 0x02 ) /* Better readable source but longer lines */
				  {
					out_line[i_out_line] = '"';
					i_out_line++;
					string_auf = !string_auf;
				  }

                  /* Post-processing for RemIdChar and ELSE of G series, no more at moment, and
                     the Compile of fixed numeric line number to binary format for 16/G/E/EXT_BAS,
                     see Wav2bin ReadLineFromBasWav, "case IDENT_E_BAS" */

                   if ( pcgrpId == GRP_G ) {
                        error = PostProcess_G( REMidC, out_line, &i_out_line ) ;
                        if (error == ERR_MEM)
                            printf ("\n WARNING: Post-processed line would be to long, line: %lu\n\n", zeilennummer) ;
                   }
                   /* Post-processing of E-500 Series 1. step (REMidC together with PostProcess_G, placeholder for jump marks),
                      COULD be implemented here,      2. step (all numeric constants to BCD together with CompileFixedJumpNb),
                                                      3. step (calculate jump marks): at end of file */

                   if ( (debug & 0x01)==0 && ident != IDENT_OLD_BAS && ident != IDENT_NEW_BAS && ident != IDENT_PC15_BAS ) {

                        error = CompileFixedJumpNb( REMidC, out_line, &i_out_line ) ;
                        if (error == ERR_MEM)
                            printf ("\n WARNING: Compiled line would be to long, line: %lu\n\n", zeilennummer) ;
                        else if (error == ERR_LINE)
                            printf ("\n WARNING: Illegal line numbers found in line: %lu\n\n", zeilennummer) ;
                   }


                } // if End TYPE_IMG
                if (type == TYPE_ASC) {
                    /* In ASCII file each End Of Line is indicated by 0x0D 0x0A*/
                    out_line[i_out_line] = '\x0D'; /* Usage of '\r' could be system dependent! */
                    i_out_line++;
                    out_line[i_out_line] = '\x0A';
                    i_out_line++;
                    tmpl = i_out_line ;
                    if (tmpl > ll_Img && (debug & 0x800) == 0) {
                            printf ("\n WARNING: ASCII line length %i to long - not editable, line: %lu\n\n", tmpl, zeilennummer) ;
                    }
                    if (tmpl > 0xFF) {
                            error = ERR_MEM ;
                            break ;
                    }
                }
				else if (ident == IDENT_OLD_BAS ) {
                    out_line[i_out_line] = '\x00';
                    conv_asc2old(out_line, i_out_line) ; /* convert ASCII chars to OLD chars */
                    i_out_line++;
                    tmpl = i_out_line - 2 ;
                    if (tmpl > ll_Img - line_nbr_len && (debug & 0x800) == 0)
                        printf ("\n WARNING: Image line length %i to long - not editable, line: %lu\n\n", tmpl, zeilennummer) ;
                    if (tmpl > 0xFF) {
                            error = ERR_MEM ;
                            break ;
                    }
                }
                else {
                    /* In Binary image file each End Of Line is indicated by 0x0D */
                    out_line[i_out_line] = '\x0D'; /* Usage of '\r' could be system dependent! */
                    i_out_line++;
                    tmpl = i_out_line - 3 ;
                    if (tmpl > ll_Img - line_nbr_len && (debug & 0x800) == 0) {
                        if (type == TYPE_TXT) {
                            printf ("\n ERROR %i: Image line length %i is to long, line: %lu\n", ERR_MEM, tmpl, zeilennummer) ;
                            error = ERR_MEM ;
                            break ;
                        }
                        else
                            printf ("\n WARNING: Image line length %i to long - not editable, line: %lu\n\n", tmpl, zeilennummer) ;
                    }
                    if (tmpl > 0xFF) {
                            error = ERR_MEM ;
                            break ;
                    }
                    out_line[2] = (uchar) tmpl ; /* Length of Binary image command line */
                }
                if (Qcnt == 0 && (debug & 0x40) >0 ) {
                    /* Show content of line_buffer as hex values */
                    ii = 0;
                    printf ("o>");
                    for ( ii = 0 ; ii < i_out_line ; ++ii)
                        printf (" %02X", (uchar) out_line[ii]);
                    printf ("<\n") ;
                }
				fwrite( out_line, sizeof( char ), i_out_line, dateiaus );
			} /* if ( zeilennummer ) : line number */
		} /* while ( getlineF( line_buffer, cLL, dateiein ) != NULL ) */

        /* EOF mark for PC-1500 from image removed for compatibility of this image with images from Wav2bin and
           for uniformity with the other images, Bin2wav also does append it (now: if it was not found before) */
		/* if (ident == IDENT_PC15_BAS )  {
		  out_line[0] = '\xFF'; // In Binary image file of PC-1500 a 0xFF was additional used as End Of File indicator /
		  fwrite( out_line, sizeof( char ), 1, dateiaus );  } */

		fclose( dateiaus );
		if (Qcnt == 0)  printf("File %s was written\n", argD[1] );
		fclose( dateiein );
	} /* if ( ( dateiein = fopen( argD[0], "rb" ) ) != NULL ) */
	else /* open with error : öffnen fehlerhaft */
	{
		error = ERR_FILE ;
		printf ("%s: Can't open the source file: %s\n", argP, argD[0]);
	}
    if (error != ERR_OK && error != ERR_NOK) {
            if (debug != 0) printf ("Exit with error %d\n", error) ;
            return (error) ;
    }
    else return (EXIT_SUCCESS) ;
}
