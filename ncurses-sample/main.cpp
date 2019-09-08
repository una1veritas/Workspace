/*
 * main.cpp
 *
 *  Created on: 2016/04/03
 *  Modified on: 2019/09/06
 *      Author: sin
 */

#include <stdlib.h>
#include <ctype.h>

#include <ncurses.h>

typedef unsigned int uint;

struct NCursesWindow {
	WINDOW * mainwin;
	uint maxrow, maxcol;

	enum {
		RAW_MODE = 1,
		ECHO_INPUT = 2,
		CURSOR_VISIBLE = 4,
		NO_DELAY = 8,
		KEYPAD = 16,
		FUNCTION_MASK = RAW_MODE | ECHO_INPUT | CURSOR_VISIBLE | NO_DELAY | KEYPAD,
	};

	NCursesWindow() {
		mainwin = initscr();
		if ( *this ) {
			getmaxyx(stdscr, maxrow, maxcol);
		}
	}

	~NCursesWindow() {
		delwin(mainwin);
		endwin();			/* End curses mode		  */
		refresh();
	}

	explicit operator bool() const {
		return mainwin != NULL;
	}

	void buffered_input(const bool yes) {
	if ( yes )
		nocbreak();
	else
		cbreak();	/* not raw mode but read key immediately */
	}

	void echo_input(const bool yes) {
		if ( yes )
			echo();
		else
			noecho(); 			/* do not echo the input key char */
	}

	void cursor_visible(const bool yes) {
		if ( yes )
			curs_set(1);
		else
			curs_set(0);		/* 0 ... set cursor invisible */
	}

	void keypress_delay(const bool yes) {
		if ( yes )
			nodelay(stdscr, TRUE); /* getch do not wait keypress */
		else
			nodelay(stdscr, FALSE);
	}

	void use_keypad(const bool yes) {
		if ( yes )
			keypad(stdscr, TRUE);
		else
			keypad(stdscr,FALSE);
	}

	int print(const char * str) {
		return printw(str);
	}


};

int main(const int argc, const char **argv)
{
	NCursesWindow mainwin;
//	WINDOW * mainwin;
//	int row, col;
	char ch = ' ';
	int counter;

//	mainwin = initscr();			/* Start curses mode 		  */
	if ( !mainwin ) {
		printf("failed ncurses mode.\n");
		return EXIT_FAILURE;
	}

	//getmaxyx(stdscr,row,col); 	/* get the size of stdscr */
	//mainwin.buffered_input(false);
	cbreak(); 			/* not raw mode but read key immediately */
	mainwin.echo_input(false); //noecho(); 			/* do not echo the input key char */
	mainwin.use_keypad(true); //	keypad(stdscr,TRUE);
	mainwin.cursor_visible(true); //curs_set(0);		/* 0 ... set cursor invisible */

	//mainwin.set_mode(NCursesWindow::RAW_MODE | NCursesWindow::NO_DELAY | NCursesWindow::KEYPAD,
	//		NCursesWindow::RAW_MODE | NCursesWindow::NO_DELAY | NCursesWindow::CURSOR_VISIBLE | NCursesWindow::KEYPAD );

	printw("Hello World !!!");	/* Print Hello World		  */
	mvprintw(1,0,"My world is %d x %d.",mainwin.maxrow, mainwin.maxcol);

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

	//delwin(mainwin);
	//endwin();			/* End curses mode		  */
	//refresh();

	return EXIT_SUCCESS;
}
