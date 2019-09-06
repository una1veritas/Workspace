/*
 * main.cpp
 *
 *  Created on: 2016/04/03
 *  Modified on: 2019/09/06
 *      Author: sin
 */

#include <stdlib.h>
#include <ctype.h>

#include <ncurses/ncurses.h>

typedef unsigned int uint;

struct NCursesWindow {
	WINDOW * mainwin;
	uint maxrow, maxcol;

	enum {
		RAW_MODE = 1,
		ECHO = 2,
		CURSOR_VISIBLE = 4,
		NO_DELAY = 8,
	};

	NCursesWindow() {
		mainwin = initscr();
		if ( this ) {
			getmaxyx(stdscr, maxrow, maxcol);
		}
	}

	bool operator()(void) {
		return mainwin != NULL;
	}

	void set_mode(const uint flags) {
		if ( !(flags & RAW_MODE) )
			cbreak(); 			/* not raw mode but read key immediately */
		if ( !(flags & ECHO) )
			noecho(); 			/* do not echo the input key char */
		if ( !(flags & CURSOR_VISIBLE) )
			curs_set(0);		/* 0 ... set cursor invisible */
		if ( flags & NO_DELAY )
			nodelay(stdscr, TRUE); /* getch do not wait keypress */

	}

	int print(const char * str) {
		return printw(str);
	}


};

int main(const int argc, const char **argv)
{
	WINDOW * mainwin;
	int row, col;
	char ch = ' ';
	int counter;

	mainwin = initscr();			/* Start curses mode 		  */
	if ( mainwin == NULL ) {
		printf("failed ncurses mode.\n");
		return EXIT_FAILURE;
	}

	getmaxyx(stdscr,row,col); 	/* get the size of stdscr */
	cbreak(); 			/* not raw mode but read key immediately */
	noecho(); 			/* do not echo the input key char */
	curs_set(0);		/* 0 ... set cursor invisible */

	printw("Hello World !!!");	/* Print Hello World		  */
	mvprintw(1,0,"My world is %d x %d.",row, col);

	mvprintw(4,5,"Type 'Q' or 'q' to exit.");
	refresh();			/* Print it on to the real screen */

	nodelay(stdscr, TRUE); /* getch do not wait keypress */
	counter = 0;
	while (1) {
		move(3,0);
		attrset(A_REVERSE);
		printw("counter = %d",counter);
		attroff(A_REVERSE);

		ch = getch();			/* Wait for user input */
		if ( ch != -1 ) { /* 255 if there is no input. */
			if ( !isprint(ch) )
				ch = ' ';
			mvprintw(4,0,"[%c]",ch);
		}
		refresh();


		if ( ch == 'Q' || ch == 'q' )
			break;

		counter++;
	}
	delwin(mainwin);
	endwin();			/* End curses mode		  */
	refresh();

	return EXIT_SUCCESS;
}
