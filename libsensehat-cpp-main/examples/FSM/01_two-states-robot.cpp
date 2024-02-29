/* File: 01-two-states-robot.cpp
 * Author: Philippe Latu
 * Source: https://github.com/platu/libsensehat-cpp
 *
 * This example program is a minimalistic robot simulator with two states:
 * - state RUN: the robot moves forward
 * - state STOP: the robot stops
 * Keyboard keys are used to change the state of the robot.
 * - 'r' key: RUN state
 * - 's' key: STOP state
 * - 'q' key: quit the program
 *
 * These functions are not part of the Sense Hat library. They are intended to
 * be used by students who wish to drive a robot with the arrow keys of the
 * console.
 */

#include <console_io.h>
#include <sensehat.h>

#include <iostream>
#include <thread>  // sleep_for, sleep_until

using namespace std;
using namespace std::this_thread;  // sleep_for, sleep_until
using namespace std::chrono;	   // system_clock, milliseconds

enum STATE { RUN, STOP };

// Main program
int main() {
	STATE current_state, next_state;
	bool quit;
	int key_count, time, cycle_count;
	char c;

	if (senseInit()) {
		std::cout << "-------------------------------" << endl
				  << "Sense Hat initialization Ok." << endl;

		//---------------------------------------------------------------------
		// Initializations
		current_state = STOP;
		next_state = STOP;
		quit = false;
		key_count = 0;
		time = 0;
		cycle_count = 0;
		clearConsole();	 // Clear the console
		gotoxy(5, 4);	 // Move cursor to column 5, raw 4
		std::cout << '>' << ends;
		gotoxy(7, 4);  // Move cursor to column 7, raw 4

		//---------------------------------------------------------------------
		// Main task
		do {
			// ---------------------------------------------------------------
			// Event detection
			key_count = keypressed();

			// check if a single key is pressed. We have an event!
			if (key_count == 1) {
				c = std::cin.get();
				gotoxy(7, 4);  // Move cursor to column 7, raw 4
				clearEOL();
				std::cout << "key = [" << c << "]";	 // Display the character

				// -----------------------------------------------------------
				// STATE machine evolution
				switch (toupper(c)) {
					case 'R':
						next_state = RUN;
						break;
					case 'S':
						next_state = STOP;
						break;
					case 'Q':
						quit = true;
						break;
				}

				// -----------------------------------------------------------
				// State transition and Robot action
				if (next_state != current_state) {
					gotoxy(5, 7);  // Move cursor to column 5, raw 7
					clearEOL();
					std::cout << "Robot state: ";
					switch (next_state) {
						case RUN:
							std::cout << "RUN";
							// TODO: move the robot forward

							break;
						case STOP:
							std::cout << "STOP";
							// TODO: stop the robot

							break;
					}
					current_state = next_state;
				}

				// Reset the cursor position and key count
				gotoxy(7, 4);  // Move cursor to column 7, raw 4
				key_count = 0;
			}
			// ---------------------------------------------------------------
			// Wait 20 ms before next iteration
			sleep_until(system_clock::now() + milliseconds(20));
			cycle_count++;

			// Print time every 50 cycles of 20 ms
			// This is the background task which illustrates the non-blocking
			// use of event driven functions
			if (cycle_count == 50) {
				cycle_count = 0;
				time += 1;
				gotoxy(30, 4);  // Move cursor to column 30, raw 4
				std::cout << "time = " << time << "s";
				clearEOL();
			}

		} while (!quit);
		std::cout << endl;

		senseShutdown();
		std::cout << "-------------------------------" << endl
				  << "Sense Hat shut down." << endl;
	}

	return EXIT_SUCCESS;
}
