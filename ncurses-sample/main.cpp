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

struct NCWindow {
	WINDOW * mainwin;
	uint width, height;

	enum {
		BUFFERED = 1<<0,
		NO_BUFFERED = 1<<1,
		ECHO = 1<<2,
		NO_ECHO = 1<<3,
		CURSOR_VISIBLE = 1<<4,
		CURSOR_INVISIBLE = 1<<5,
		NO_DELAY = 1<<6,
		DELAY = 1<<7,
		KEYPAD = 1<<8,
		NO_KEYPAD = 1<<8,
	};

	enum {
		KEYPAD_DOWN = 0x102,
		KEYPAD_UP,
		KEYPAD_LEFT,
		KEYPAD_RIGHT,
	};

	NCWindow() {
		mainwin = initscr();
		if ( *this ) {
			getmaxyx(stdscr, height,  width);
		}
	}

	~NCWindow() {
		delwin(mainwin);
		endwin();			/* End curses mode		  */
		refresh();
	}

	explicit operator bool() const {
		return mainwin != NULL;
	}

	void screen_mode(const uint flags) {
		if (flags & BUFFERED)
			nocbreak();
		else if (flags & NO_BUFFERED)
			cbreak(); 	/* raw mode, read key immediately */
		if (flags & ECHO)
			echo();
		else if (flags & NO_ECHO)
			noecho(); 	/* do not echo the input key char */
		if (flags & CURSOR_VISIBLE)
			curs_set(1);
		else if (flags & CURSOR_INVISIBLE)
			curs_set(0); 	/* 0 ... set cursor invisible */
		if (flags & NO_DELAY)
			nodelay(stdscr, TRUE);
		else if (flags & DELAY)
			nodelay(stdscr, FALSE);
		if (flags & KEYPAD)
			keypad(stdscr, TRUE);
		else if (flags & NO_KEYPAD)
			keypad(stdscr,FALSE);
	}

	int print(const char * str) {
		return printw(str);
	}

};


#define range(x,y,z)  ((x) > (y)? (x) : ((y) > (z)? (z) : (y)))

int main(const int argc, const char **argv)
{
	NCWindow mainwin;
//	WINDOW * mainwin; int row, col;

	if ( !mainwin ) {
		printf("failed ncurses mode.\n");
		return EXIT_FAILURE;
	}

	mainwin.screen_mode(NCWindow::NO_BUFFERED | NCWindow::NO_ECHO | NCWindow::NO_DELAY
			|NCWindow::KEYPAD | NCWindow::CURSOR_INVISIBLE);

	printw("Hello World !!!");	/* Print Hello World		  */
	mvprintw(1,0,"My world is %d x %d.",mainwin.width, mainwin.height);

	mvprintw(4,5,"Type 'Q' or 'q' to exit.");
	refresh();			/* Print it on to the real screen */

	int ch = '@', t;
	bool update = true;
	int counter = 0;
	int posrow = 5, poscol = 0;
	while ( true ) {
		move(3,0);
		attrset(A_REVERSE);
		printw("counter = %d",counter);
		attroff(A_REVERSE);

		t = getch();			/* scan a pressed key */
		if ( t != -1 ) { /* 255 if there is no input. */
			update = true;
		}
		if ( update ) {
			if (t >= NCWindow::KEYPAD_DOWN && t <= NCWindow::KEYPAD_RIGHT) {
				mvprintw(posrow, poscol, "    ");
				switch (t) {
				case NCWindow::KEYPAD_DOWN:
					posrow += 1;
					break;
				case NCWindow::KEYPAD_UP:
					posrow -= 1;
					break;
				case NCWindow::KEYPAD_LEFT:
					poscol -= 1;
					break;
				case NCWindow::KEYPAD_RIGHT:
					poscol += 1;
				}
				posrow = range(5, posrow, (int)mainwin.height-1);
				poscol = range(0, poscol, (int)mainwin.width-4);
			} else if ( isprint(t) ) {
				ch = t;
			}
			mvprintw(posrow, poscol,"[%c]", ch);
			refresh();
			update = false;
		}

		if ( ch == 'Q' || ch == 'q' )
			break;

		counter++;
	}

	//delwin(mainwin);
	//endwin();			/* End curses mode		  */
	//refresh();

	return EXIT_SUCCESS;
}
