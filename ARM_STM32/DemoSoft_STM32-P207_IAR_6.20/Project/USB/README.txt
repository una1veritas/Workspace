	*** OLIMEX demo project for the STM32-P207 ***

1. Requirements
	- STM32-P207 demo board by Olimex
	- Compatible debugger, for example ARM-JTAG-EW
	- IAR EW v6.20 or later
	- miniUSB cable

2. Description
	The program implements a USB HID mouse. When a mini USB is inserted in the OGT connector a mouse is initialized. In most cases no driver from the PC side are needed. Movement is implemented with the joystick, its center position acting as a left key press. TAMPER button also acts as a left mouse key and WKUP button as the right key.
	
3. How to use this demo
	+ Open the workspace file: .\DemoSoft\STM32-P207_DEMOS.eww
	+ Press F7 to compile the project.
	+ Conect the debugger to the PC and to the target board. Supply the board as needed.
	+ Press Ctrl+D to download the executble to the target.
	+ Start debugging (F5)
		
4. Support
	http://www.iar.com/
	http://www.olimex.com/dev/
