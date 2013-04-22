//============================================================================
// Name        : E_Rotor.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
using namespace std;

/*
Rotor #	ABCDEFGHIJKLMNOPQRSTUVWXYZ	Date Introduced	Model Name & Number
I	EKMFLGDQVZNTOWYHXUSPAIBRCJ	1930	Enigma I
II	AJDKSIRUXBLHWTMCQGZNPYFVOE	1930	Enigma I
III	BDFHJLCPRTXVZNYEIWGAKMUSQO	1930	Enigma I
IV	ESOVPZJAYQUIRHXLNFTGKDCMWB	DEC 1938	M3 Army
V	VZBRGITYUPSDNHLXAWMJQOFECK	DEC 1938	M3 Army
VI	JPGVOUMFYQBENHZRDKASXLICTW	1939	M3 & M4 Naval (FEB 1942)
VII	NZJHGRCXMYSWBOUFAIVLPEKQDT	1939	M3 & M4 Naval (FEB 1942)
VIII	FKQHTLXOCBJSPDZRAMEWNIUYGV	1939	M3 & M4 Naval (FEB 1942)
*/

/*
 *
Rotor	Notch	Effect
I	Q	If rotor steps from Q to R, the next rotor is advanced
II	E	If rotor steps from E to F, the next rotor is advanced
III	V	If rotor steps from V to W, the next rotor is advanced
IV	J	If rotor steps from J to K, the next rotor is advanced
V	Z	If rotor steps from Z to A, the next rotor is advanced
VI, VII, VIII	Z+M	if rotor steps from Z to A, or from M to N the next rotor is advanced
 *
 */

struct Rotor {
	char map[31];
	char notch;
};

const Rotor rotor[][32] = {
		{"EKMFLGDQVZNTOWYHXUSPAIBRCJ",'Q'},
		{"AJDKSIRUXBLHWTMCQGZNPYFVOE",'E'},
		{"BDFHJLCPRTXVZNYEIWGAKMUSQO",'V'},
		{"ESOVPZJAYQUIRHXLNFTGKDCMWB",'J'},
		{"VZBRGITYUPSDNHLXAWMJQOFECK",'Z'},
		"JPGVOUMFYQBENHZRDKASXLICTW",
		"NZJHGRCXMYSWBOUFAIVLPEKQDT",
		"FKQHTLXOCBJSPDZRAMEWNIUYGV",
};

int main() {
	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!
	return 0;
}
