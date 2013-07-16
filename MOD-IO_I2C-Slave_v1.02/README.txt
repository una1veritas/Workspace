	*** OLIMEX demo project for MOD-IO ***

1. Description
	This demo requires an I2C slave device to interface the following peripherals on the board:
	- Digital outputs O1, O2, O3, O4. 
	- Digital inputs IN1, IN2, IN3, IN4. 
	- Analogue inputs AIN1, AIN2, AIN3, AIN4. 

	There is actually nothing specific about the I2C protocol. Default address of the slave is 0b1011000 (0x58). When addressed, the device acknowledges reception with an ACK flag set to 0 to indicate its presence. Besides, you can issue the following commands:
	
	1) Set states of the digital outputs on the board. The board features four relay outputs that can be set together with one command. The command should have the following 3 byte format:
	
	************************************
	S aaaaaaaW cccccccc 0000dddd P
	************************************
	
	Where 
	S - start condition
	aaaaaaa - slave address of the board
	W - write mode, should be 0
	cccccccc - command code, should be 0x10
	dddd - bitmap of the output states, i.e. bit0 corresponds to REL1, bi1 to REL2 and so on. '1' switches the relay ON, '0' switches to OFF state.
	P - Stop condition

	2) Get states of the digital inputs of the board. The board features four optoisolated inputs read together with one command. The command should have the following format:
	
	************************************
	S aaaaaaaW cccccccc P S aaaaaaaR 0000dddd P
	************************************
	
	Where 
	S - start condition
	aaaaaaa - slave address of the board
	W - write mode, should be 0
	cccccccc - command code, should be 0x20
	P - Stop condition
	R - read mode, should be 1	
	dddd - bitmap of the input states received from the MOD-IO board, i.e. bit0 corresponds to IN1, bit1 to IN2 and so on. '1' means that power is applied to the optocoupler, '0' means the opposite.

	Note: Successive readings from the board without reissuing the command code will not get an updated value of the ports (i.e. the user will read the same value) until another command is issued.
	
		2) Get the voltage applied to one of the analogue inputs of the board. The board features four 10bit resolution analogue inputs (input voltages from 0 - 3.3V) and each of them is read with a separate command. Command should have the following common format:
	
	************************************
	S aaaaaaaW cccccccc P S aaaaaaaR dddddddd 000000dd P
	************************************
	
	Where 
	S - start condition
	aaaaaaa - slave address of the board
	W - write mode, should be 0
	cccccccc - command code, should be 0x30 for AIN1, 0x31 for AIN2, 0x31 for AIN3, 0x31 for AIN4.
	P - Stop condition
	R - read mode, should be 1	
	dddddddd 000000dd – Little Endian (LSB:MSB) 10bit binary encoded value corresponding to the input voltage. Range is 0 - 0x3FF and voltage on the pin is calculated using the following simple formula: voltage = (3.3 / 1024) * (readvalue) [Volts]
	
	Note: Successive readings from the board without reissuing the command code will not get an updated value of the voltage (i.e. the user will read the same value) until another command is issued.
	
	4) Set new slave address to the board. The board ships with default 7bit address 0x58 that can be changed to any other 7bit value in order for the host to interface more than 1 device connected on the bus at the same time. Change is stored in EEPROM and thus is permanent between power cycles. Changing the address requires the following command format:
	
	************************************
	S aaaaaaaW cccccccc 0ddddddd P
	************************************
	
	Where 
	S - start condition
	aaaaaaa - slave address of the board (the default or the old address of the board)
	W - write mode, should be 0
	cccccccc - command code, should be 0xF0
	ddddddd - new 7bit address to update
	P - Stop condition
	
	NB!! To protect the device from accidental address updates the user should hold the onboard button pressed (not the RESET button!) while issuing the command. Successful update is indicated with the onboard status LED being contently lit for 2-3 seconds. Address is immediately updated so the board will not respond to its old address any more.
	
	IMPORTANT: The default address of the board could be restored if the onboard button is held pressed at power up for more than 4 seconds. This situation is indicated by the onboard LED blinking fast for the timeout period. When the fast blinking ends default address is restored. 

2. Support
	http://www.olimex.com/dev/
	http://www.atmel.com/

3. Revision history
	23 July 2012 - v1.02 - fixed docuemnt info about default address
	21 Nov 2011 - V1.01 - changed the default slave address due to incompatability with Duinomite boards
	10 Nov 2011 - v1.00 - initial release
