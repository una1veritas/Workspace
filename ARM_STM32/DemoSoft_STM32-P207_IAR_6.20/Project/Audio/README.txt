	*** OLIMEX demo project for the STM32-P207 ***

1. Requirements
	- STM32-P207 demo board by Olimex
	- Compatible debugger, for example ARM-JTAG-EW
	- IAR EW v6.20 or later

2. Description
	The program demonstrates the functionality of the I2S 2channel audio circuit CS4344 present on the board. Starting the demo produces alternating audio signals (left channel, right channel) that can be heard with headphones.
	
3. How to use this demo
	+ Open the workspace file: .\DemoSoft\STM32-P207_DEMOS.eww
	+ Press F7 to compile the project.
	+ Conect the debugger to the PC and to the target board. Supply the board as needed.
	+ Press Ctrl+D to download the executble to the target.
	+ Start debugging (F5)
		
4. Support
	http://www.iar.com/
	http://www.olimex.com/dev/
