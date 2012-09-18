	*** OLIMEX demo project for the STM32-P207 ***

1. Requirements
	- STM32-P207 demo board by Olimex
	- Compatible debugger, for example ARM-JTAG-EW
	- IAR EW v6.20 or later
	- RS232 port and a terminal program (Hyper Terminal, PuTTY, etc.)
	- LAN cable and a router/switch
	
2. Description
	The program demonstrates the functionality of the onboard Ethernet present on the board. The demo starts a simple WEB server as part of the uIP stack. Default IP address of the board is 192.168.0.100 as can be seen from the information displayed on the teriminal.
	
3. How to use this demo
	+ Open the workspace file: .\DemoSoft\STM32-P207_DEMOS.eww
	+ Press F7 to compile the project.
	+ Conect the debugger to the PC and to the target board. Supply the board as needed.
	+ Press Ctrl+D to download the executble to the target.
	+ Connect the RS232 cable to the board (RS232_1) and start your terminal program with the following settings: 115200-8N1
	+ Start debugging (F5)
	+ Follow the instructions on the terminal.
		
4. Support
	http://www.iar.com/
	http://www.olimex.com/dev/
