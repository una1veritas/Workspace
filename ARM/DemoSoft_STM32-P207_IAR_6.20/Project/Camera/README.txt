	*** OLIMEX demo project for the STM32-P207 ***

1. Requirements
	- STM32-P207 demo board by Olimex
	- Compatible debugger, for example ARM-JTAG-EW
	- IAR EW v6.20 or later

2. Description
	The program demonstrates the functionality of the Samsung E700 camera, the external 512kB SRAM chip and the color LCD display. The program acts as a very simple and crude video capture device taht streams its output to the LCD display. The SRAM is used as a buffer for both the capture and processed image data from the camera.
	
3. How to use this demo
	+ Open the workspace file: .\DemoSoft\STM32-P207_DEMOS.eww
	+ Press F7 to compile the project.
	+ Conect the debugger to the PC and to the target board. Supply the board as needed.
	+ Press Ctrl+D to download the executble to the target.
	+ Start debugging (F5)
		
4. Support
	http://www.iar.com/
	http://www.olimex.com/dev/
