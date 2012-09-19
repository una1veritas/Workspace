	*** OLIMEX demo project for the STM32-P207 ***

1. Requirements
	- STM32-P207 demo board by Olimex
	- Compatible debugger, for example ARM-JTAG-EW
	- IAR EW v6.20 or later
	- RS232 port and a terminal program (Hyper Terminal, PuTTY, etc.)
	- microSD/MMC card
	
2. Description
	The demo demonstrates the functionality of the SD/MMC card slot present on the board. Card size is shown on the terminal when a card is inserted.
	
3. How to use this demo
	+ Open the workspace file: .\DemoSoft\STM32-P207_DEMOS.eww
	+ Press F7 to compile the project.
	+ Conect the debugger to the PC and to the target board. Supply the board as needed.
	+ Press Ctrl+D to download the executble to the target.
	+ Connect the RS232 cable to the board and start your terminal program with the following settings: 115200-8N1
	+ Start debugging (F5)
	+ Follow the instructions on the terminal.
		
4. Support
	http://www.iar.com/
	http://www.olimex.com/dev/
