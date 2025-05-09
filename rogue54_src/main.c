/*
 * Rogue: Exploring the Dungeons of Doom
 * Copyright (C) 1980-1983, 1985, 1999 Michael Toy, Ken Arnold and Glenn Wichman
 * All rights reserved.
 *
 * See the file LICENSE.TXT for full copyright and licensing information.
 *
 * @(#)main.c	4.22 (Berkeley) 02/05/99
 */

#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <time.h>
#include <curses.h>
#include "rogue.h"
/* {{ */
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#include "extern.h"

#define NUMSCORES 10

void rs_read_int(FILE *, int *);
void rs_read_string(FILE *, char *, int);

char cfgfile[256];
/* }} */

/*
 * main:
 *	The main program, of course
 */
int
main(int argc, char **argv)
{
    char *env;
    time_t lowtime;
/* {{ */
    time_t now;
    int i;
    FILE *fptr;
    char null_buf[MAXSTR+100];

#ifdef SCOREFILE
#ifdef _WIN32
    char scorefilebuf[32], *scorefile;
    FILE *fptmp;
    sprintf(scorefilebuf, "R:\\%s", SCOREFILE);
    scorefile = (char *)scorefilebuf;
    for ( ; (*scorefile) <= 'Z'; (*scorefile)++) {
	fptmp = fopen(scorefile, "r+");
	if (fptmp != NULL)  break;
    }
    fcloseall();
    if ((*scorefile) > 'Z') {
	scorefile += 3;
    }
#else
    char *scorefile = SCOREFILE;
#endif

    if ((fptr = fopen(scorefile, "r")) == NULL)
    {
	if ((fptr = fopen(scorefile, "w")) == NULL)
	{
	    fprintf(stderr, "Could not open %s for writing\n", scorefile);
	    fflush(stderr);
	}
	else
	{
	    for (i = 0; i < MAXSTR+100; i++)
		null_buf[i] = '\0';
	    for (i = 0; i < NUMSCORES; i++)
		fwrite(null_buf, 1, 1124, fptr);
	}
    }
    fclose(fptr);
#endif

    whoami[0] = '\0';
#ifdef _WIN32
/*    sprintf(cfgfile, "%s\\rogue54.cfg", getenv("APPDATA"));	*/
    sprintf(cfgfile, "rogue54.cfg");
#else
    sprintf(cfgfile, "%s/rogue54.cfg", getenv("HOME"));
#endif
    if ((fptr = fopen(cfgfile, "r")) != NULL)
    {
	rs_read_int(fptr, &terse);
	rs_read_int(fptr, &fight_flush);
	rs_read_int(fptr, &jump);
	rs_read_int(fptr, &see_floor);
	rs_read_int(fptr, &passgo);
	rs_read_int(fptr, &tombstone);
	rs_read_int(fptr, &inv_type);
	rs_read_string(fptr, whoami, MAXSTR);
	rs_read_string(fptr, fruit, MAXSTR);
	fclose(fptr);
    }
/* }} */

    md_init();

/* {{ */
    if (argc >= 3)
	if (strcmp(argv[1], "-u") == 0
	||  strcmp(argv[1], "-n") == 0)
	    strncpy(whoami, argv[2], 80-1);
/* }} */

#ifdef MASTER
    /*
     * Check to see if he is a wizard
     */
    if (argc >= 2 && argv[1][0] == '\0')
	if (strcmp(PASSWD, md_crypt(md_getpass("wizard's password: "), "mT")) == 0)
	{
	    wizard = TRUE;
	    player.t_flags |= SEEMONST;
	    argv++;
	    argc--;
	}

#endif

    /*
     * get home and options from environment
     */

    strcpy(home, md_gethomedir());

/*	if (strlen(home) > MAXSTR - strlen("rogue.save") - 1) */
/* {{ */
	if (strlen(home) > MAXSTR - strlen("rogue54.sav") - 1)
/* }} */
		*home = 0;

    strcpy(file_name, home);
/*    strcat(file_name, "rogue.save"); */
/* {{ */
    strcat(file_name, "rogue54.sav");
/* }} */

    if ((env = getenv("ROGUEOPTS")) != NULL)
	parse_opts(env);
/*    if (env == NULL || whoami[0] == '\0')	*/
    if (whoami[0] == '\0')
	strucpy(whoami, md_getusername(), strlen(md_getusername()));
    lowtime = time(NULL);
    if (getenv("SEED") != NULL)
    {
	dnum = atoi(getenv("SEED"));
	noscore = 1;
    }
    else
	dnum = (unsigned int) lowtime + md_getpid();
    seed = dnum;

    open_score();

	/*
     * Drop setuid/setgid after opening the scoreboard file.
     */

    md_normaluser();

    /*
     * check for print-score option
     */

	md_normaluser(); /* we drop any setgid/setuid priveldges here */

    if (argc == 2)
    {
	if (strcmp(argv[1], "-s") == 0)
	{
	    noscore = TRUE;
	    score(0, -1, 0);
	    exit(0);
	}
	else if (strcmp(argv[1], "-d") == 0)
	{
	    dnum = rnd(100);	/* throw away some rnd()s to break patterns */
	    while (--dnum)
		rnd(100);
	    purse = rnd(100) + 1;
	    level = rnd(100) + 1;
	    initscr();
	    getltchars();
	    death(death_monst());
	    exit(0);
	}
    }

    init_check();			/* check for legal startup */
    if (argc == 2)
	if (!restore(argv[1]))	/* Note: restore will never return */
	    my_exit(1);
/* {{ */
    if (whoami[0] == '?')
    {
	printf("Rogue's Name? ");
	fgets(whoami, 80-1, stdin);
	for (i=0; whoami[i] >= ' '; i++)
	    /* nothing */ ;
	whoami[i] = '\0';
	if (whoami[0] == '\0')
	    strcpy(whoami, "Rogue");
    }
/* }} */
#ifdef MASTER
    if (wizard)
	printf("Hello %s, welcome to dungeon #%d", whoami, dnum);
    else
#endif
	printf("Hello %s, just a moment while I dig the dungeon...", whoami);
    fflush(stdout);
/* {{ */
#ifdef _WIN32
    Sleep(3000);
#else
    sleep(3);
#endif
/* }} */

    initscr();				/* Start up cursor package */
    init_probs();			/* Set up prob tables for objects */
    init_player();			/* Set up initial player stats */
    init_names();			/* Set up names of scrolls */
    init_colors();			/* Set up colors of potions */
    init_stones();			/* Set up stone settings of rings */
    init_materials();			/* Set up materials of wands */
    setup();

    /*
     * The screen must be at least NUMLINES x NUMCOLS
     */
    if (LINES < NUMLINES || COLS < NUMCOLS)
    {
	printf("\nSorry, the screen must be at least %dx%d\n", NUMLINES, NUMCOLS);
	endwin();
	my_exit(1);
    }

    /*
     * Set up windows
     */
    hw = newwin(LINES, COLS, 0, 0);
    idlok(stdscr, TRUE);
    idlok(hw, TRUE);
#ifdef MASTER
    noscore = wizard;
#endif
    new_level();			/* Draw current level */
    /*
     * Start up daemons and fuses
     */
    start_daemon(runners, 0, AFTER);
    start_daemon(doctor, 0, AFTER);
    fuse(swander, 0, WANDERTIME, AFTER);
    start_daemon(stomach, 0, AFTER);
    playit();
    return(0);
}

/*
 * endit:
 *	Exit the program abnormally.
 */

void
endit(int sig)
{
    NOOP(sig);
    fatal("Okay, bye bye!\n");
}

/*
 * fatal:
 *	Exit the program, printing a message.
 */

void
fatal(const char *s)
{
    mvaddstr(LINES - 2, 0, s);
    refresh();
    endwin();
    my_exit(0);
}

/*
 * rnd:
 *	Pick a very random number.
 */
int
rnd(int range)
{
    return range == 0 ? 0 : abs((int) RN) % range;
}

/*
 * roll:
 *	Roll a number of dice
 */
int
roll(int number, int sides)
{
    int dtotal = 0;

    while (number--)
	dtotal += rnd(sides)+1;
    return dtotal;
}

/*
 * tstp:
 *	Handle stop and start signals
 */

void
tstp(int ignored)
{
    int y, x;
    int oy, ox;

	NOOP(ignored);

    /*
     * leave nicely
     */
    getyx(curscr, oy, ox);
    mvcur(0, COLS - 1, LINES - 1, 0);
    endwin();
    resetltchars();
    fflush(stdout);
	md_tstpsignal();

    /*
     * start back up again
     */
	md_tstpresume();
    raw();
    noecho();
    keypad(stdscr,1);
    playltchars();
    clearok(curscr, TRUE);
    wrefresh(curscr);
    getyx(curscr, y, x);
    mvcur(y, x, oy, ox);
    fflush(stdout);
    curscr->_cury = oy;
    curscr->_curx = ox;
}

/*
 * playit:
 *	The main loop of the program.  Loop until the game is over,
 *	refreshing things and looking at the proper times.
 */

void
playit(void)
{
    char *opts;

    /*
     * set up defaults for slow terminals
     */

    if (baudrate() <= 1200)
    {
	terse = TRUE;
	jump = TRUE;
	see_floor = FALSE;
    }

    if (md_hasclreol())
	inv_type = INV_CLEAR;

    /*
     * parse environment declaration of options
     */
    if ((opts = getenv("ROGUEOPTS")) != NULL)
	parse_opts(opts);

    oldpos = hero;
    oldrp = roomin(&hero);
    while (playing)
	command();			/* Command execution */
    endit(0);
}

/*
 * quit:
 *	Have player make certain, then exit.
 */

void
quit(int sig)
{
    int oy, ox;

    NOOP(sig);

    /*
     * Reset the signal in case we got here via an interrupt
     */
    if (!q_comm)
	mpos = 0;
    getyx(curscr, oy, ox);
    msg("really quit?");
    if (readchar() == 'y')
    {
	signal(SIGINT, leave);
	clear();
	mvprintw(LINES - 2, 0, "You quit with %d gold pieces", purse);
	move(LINES - 1, 0);
	refresh();
	score(purse, 1, 0);
	my_exit(0);
    }
    else
    {
	move(0, 0);
	clrtoeol();
	status();
	move(oy, ox);
	refresh();
	mpos = 0;
	count = 0;
	to_death = FALSE;
    }
}

/*
 * leave:
 *	Leave quickly, but curteously
 */

void
leave(int sig)
{
    static char buf[BUFSIZ];

    NOOP(sig);

    setbuf(stdout, buf);	/* throw away pending output */

    if (!isendwin())
    {
	mvcur(0, COLS - 1, LINES - 1, 0);
	endwin();
    }

    putchar('\n');
    my_exit(0);
}

/*
 * shell:
 *	Let them escape for a while
 */

void
shell(void)
{
    /*
     * Set the terminal back to original mode
     */
    move(LINES-1, 0);
    refresh();
    endwin();
    resetltchars();
    putchar('\n');
    in_shell = TRUE;
    after = FALSE;
    fflush(stdout);
    /*
     * Fork and do a shell
     */
    md_shellescape();

    noecho();
    raw();
    keypad(stdscr,1);
    playltchars();
    in_shell = FALSE;
    clearok(stdscr, TRUE);
}

/*
 * my_exit:
 *	Leave the process properly
 */

void
my_exit(int st)
{
    resetltchars();
    exit(st);
}
