/*
 * main.c
 *
 *  Created on: 2016/04/03
 *      Author: sin
 */

#include <ncurses.h>
#include <ctype.h>

int main(void)
{
	int row, col;
	char ch = ' ';
	int counter;

	initscr();			/* Start curses mode 		  */
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
	endwin();			/* End curses mode		  */

	return 0;
}
