/* vim: set noexpandtab ai ts=4 sw=4 tw=4:

	as.c - part of as9 6809 assembler
*/

#include <stdio.h>
#include <stdlib.h>

#include <string.h>
#include <unistd.h>
#include <ctype.h>

#include "as.h"
#include "util.h"
#include "output.h"
#include "ffwd.h"
#include "do9.h"
#include "symtab.h"
#include "pseudo.h"


/* global vars */
int srcarg;
char cur_file[64];	// current filename
int     Line_num =0;            /* current line number          */
int     Err_count =0;           /* total number of errors       */
char    Line[MAXBUF] = {0};     /* input line buffer            */
char    Label[MAXLAB] = {0};    /* label on current line        */
char    Op[MAXOP] = {0};        /* opcode mnemonic on current line      */
char    Operand[MAXBUF] = {0};  /* remainder of line after op           */
char    *Optr =0;               /* pointer into current Operand field   */
int     Result =0;              /* result of expression evaluation      */
int     Force_word =0;          /* Result should be a word when set     */
int     Force_byte =0;          /* Result should be a byte when set     */
int     Pc =0;                  /* Program Counter              */
int     Old_pc =0;              /* Program Counter at beginning */

int     Last_sym =0;            /* result of last lookup        */

int     Pass =0;                /* Current pass #               */
int     N_files =0;             /* Number of files to assemble  */
FILE    *Fd =0;                 /* Current input file structure */
int     Cfn =0;                 /* Current file number 1...n    */
int     Ffn =0;                 /* forward ref file #           */
int     F_ref =0;               /* next line with forward ref   */
char    **Argv =0;              /* pointer to file names        */

int     E_total =0;             /* total # bytes for one line   */
int     E_bytes[E_LIMIT] = {0}; /* Emitted held bytes           */
int     E_pc =0;                /* Pc at beginning of collection*/

int     Lflag = 0;              /* listing flag 0=nolist, 1=list*/

int     P_force = 0;            /* force listing line to include Old_pc */
int     P_total =0;             /* current number of bytes collected    */
int     P_bytes[P_LIMIT] = {0}; /* Bytes collected for listing  */

int     Cflag = 0;              /* cycle count flag */
int     Cycles = 0;             /* # of cycles per instruction  */
long    Ctotal = 0;             /* # of cycles seen so far */
int     Sflag = 0;              /* symbol table flag, 0=no symbol */
int     N_page = 0;             /* new page flag */
int     Page_num = 2;           /* page number */
int     CREflag = 0;            /* cross reference table flag */


FILE    *Objfil =0;             /* object file's file descriptor*/
char    Obj_name[64];

struct  nlist * root;

/* end of global vars */

static char *rcs_id = "Mirage asm09 2012 <gordon@gjcp.net>";

void PrintHelp (char *pszName);
/*
 *	as ---	cross assembler main program
 */
int main(int argc, char **argv) {
	//char	**np;
	char	*i;
	//int	j = 0;
	int c, srcarg;

	if(argc < 2){
	   PrintHelp (argv[0]);
	   exit(1);
	}
	Argv = argv;
	initialize();

	while ((c = getopt (argc, argv, "lscxo:")) != -1)
		switch (c) {
			case 'o':
				strncpy(Obj_name, optarg, 64);
				break;
			case 'l':
				Lflag = 1;
				break;
			case 's':
				Sflag = 1;
				break;
			case 'c':
				Cflag = 1;
				break;
			case 'x':
				CREflag = 1;
				break;
			case '?':
				if (optopt == 'o') {
					fprintf (stderr, "Option -%c requires an argument.\n", optopt);
				}
				if (isprint (optopt)) {
					fprintf (stderr, "Unknown option `-%c'.\n", optopt);
				} else {
					fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
				}
				return 1;
			default:
				abort ();
           }

	root = NULL;
	N_files = argc - optind;

	if (strlen(Obj_name)==0) {
		// copy first filename as object name
		strcpy(Obj_name,argv[optind]);
		i = strrchr(Obj_name, '.');
		strncpy(i, ".s19", 4);
	}

	if( (Objfil = fopen(Obj_name,"w")) == NULL) fatal("Can't create object file");

	// first pass
	srcarg = optind;  // get the first filename
	while (srcarg < argc) {
		Fd = fopen(argv[srcarg], "r");
		strncpy(cur_file, argv[srcarg], 64);	// save filename for warnings etc
		Line_num = 0; /* reset line number */
		if (!Fd) {
			printf("as: cannot open source file %s\n", argv[srcarg]);
		} else {
			make_pass();
			fclose(Fd);
		}
		srcarg++;
	}

	if (Err_count) {
		printf("error ocurred %d time. Exit.\n", Err_count);
		free_symtab(root);
		exit(Err_count);
	}
	
	Pass++;
	re_init();
	srcarg = optind;

	while (srcarg < argc) {
		Fd = fopen(argv[srcarg], "r");
		strncpy(cur_file, argv[srcarg], 64);	// save filename for warnings etc
		Line_num = 0; /* reset line number */		

		if (!Fd) {
			printf("as: cannot open source file %s\n", argv[srcarg]);
		} else {
			make_pass();
			fclose(Fd);
		}
		
		if (Sflag == 1) {
			printf ("\f");
			stable (root);
		}
		if (CREflag == 1) {
			printf ("\f");
			cross (root);
		}
		srcarg++;
	}

	free_symtab(root);

	fprintf(Objfil,"S9030000FC\n"); /* at least give a decent ending */
    if (Err_count) {
        fprintf(stderr, "error ocurred %d time.\n", Err_count);
        exit(Err_count);
    } else {
        return 0;
    }
}

void PrintHelp (char *pszName)
{
  fprintf (stderr, "%s:  assembler for Motorola MPUs\n", pszName);
  fprintf (stderr, "  Usage:    %s file1 file2 -option1 -option2...\n", pszName);
  fprintf (stderr, "  Options:  l - generate listing\n");
  fprintf (stderr, "            c - cycle count on\n");
  fprintf (stderr, "            s - symbol table on\n");
  fprintf (stderr, "            x - cross reference flag\n");
  fprintf (stderr, "            o [file] - specify output filename\n");
  //fprintf (stderr, "            h  - this listing\n");
  //fprintf (stderr, "            V  - print version information\n");
  fprintf (stderr, "  Version:  %s\n", rcs_id);
}

void initialize(void)
{

	//int	i = 0;

#ifdef DEBUG
	printf("Initializing\n");
#endif
	Err_count = 0;
	Pc	  = 0;
	Pass	  = 1;
	Lflag	  = 0;
	Cflag	  = 0;
	Ctotal	  = 0;
	Sflag	  = 0;
	CREflag   = 0;
	N_page	  = 0;
	Line[MAXBUF-1] = NEWLINE;
	fwdinit();	/* forward ref init */
	localinit();	/* target machine specific init. */
}

void re_init(void)
{
#ifdef DEBUG
	printf("Reinitializing\n");
#endif
	Pc	= 0;
	E_total = 0;
	P_total = 0;
	Ctotal	= 0;
	N_page	= 0;
	fwdreinit();
}

void make_pass(void) {
	//char	*fgets();

#ifdef DEBUG
	printf("Pass %d\n", Pass);
#endif
	while (fgets(Line, MAXBUF - 1, Fd) != (char*) NULL) {
		Line_num++;
		P_force = 0; /* No force unless bytes emitted */
		N_page = 0;
		if (parse_line())
			process();
		if (Pass == 2 && Lflag && !N_page)
			print_line();
		P_total = 0; /* reset byte count */
		Cycles = 0; /* and per instruction cycle count */
	}
	f_record();
}


/*
 *	parse_line --- split input line into label, op and operand
 */
int parse_line(void)
{
	register char *ptrfrm = Line;
	register char *ptrto = Label;
	//char	*skip_white();

	//if ( *ptrfrm == '*' || *ptrfrm == '\n' )
	if ( *ptrfrm == '*' || *ptrfrm == ';' || *ptrfrm == '\n' )
		return (0);	/* a comment line */

	while( delim(*ptrfrm)== NO ) {
		*ptrto++ = *ptrfrm++;
	}
	if(*--ptrto != ':')ptrto++;     /* allow trailing : */
	*ptrto = EOS;

	ptrfrm = skip_white(ptrfrm);

	ptrto = Op;
	while( delim(*ptrfrm) == NO)
		*ptrto++ = mapdn(*ptrfrm++);
	*ptrto = EOS;

	ptrfrm = skip_white(ptrfrm);

	ptrto = Operand;
	while( *ptrfrm != NEWLINE )
		*ptrto++ = *ptrfrm++;
	*ptrto = EOS;

#ifdef DEBUG
	printf("Label-%s-\n",Label);
	printf("Op----%s-\n",Op);
	printf("Operand-%s-\n",Operand);
#endif
	return(1);
}

/*
 *	process --- determine mnemonic class and act on it
 */
void process(void)
{
	register struct oper *i;
	// struct oper *mne_look();
	char tmp[32];

	Old_pc = Pc;		/* setup `old' program counter */
	Optr = Operand; 	/* point to beginning of operand field */

	if (*Op == EOS) {		/* no mnemonic */
		if(*Label != EOS)
			install(Label, Pc, 0); 	// <-- lacking the 3rd arg; assumes the default val of override is 0
	} else if( (i = mne_look(Op))== NULL) {
		sprintf(tmp,"Unrecognized Mnemonic %.8s", Op);
		error(tmp);
	} else if( i->class == PSEUDO ) {
		do_pseudo(i->opcode);
	} else {
		if ( *Label )
			install(Label,Pc, 0); 	// <-- lacking the 4rd arg; assumes the default val of override is 0
		if (Cflag)
			Cycles = i->cycles;
		do_op(i->opcode,i->class);
		if (Cflag)
			Ctotal += Cycles;
	}
}
