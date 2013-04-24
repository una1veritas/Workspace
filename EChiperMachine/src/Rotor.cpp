//============================================================================
// Name        : E_Rotor.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
using namespace std;

#include <cstring>
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
Rotor	Notch	Effect
I	Q	If rotor steps from Q to R, the next rotor is advanced
II	E	If rotor steps from E to F, the next rotor is advanced
III	V	If rotor steps from V to W, the next rotor is advanced
IV	J	If rotor steps from J to K, the next rotor is advanced
V	Z	If rotor steps from Z to A, the next rotor is advanced
VI, VII, VIII	Z+M	if rotor steps from Z to A, or from M to N the next rotor is advanced
 */

struct Rotor {
/*	static const char * rotor1[] = {"I","EKMFLGDQVZNTOWYHXUSPAIBRCJ","Q"};
			{"II", "AJDKSIRUXBLHWTMCQGZNPYFVOE","E" },
			{"III", "BDFHJLCPRTXVZNYEIWGAKMUSQO","V"},
			{"IV", "ESOVPZJAYQUIRHXLNFTGKDCMWB","J"},
			{"V", "VZBRGITYUPSDNHLXAWMJQOFECK","Z"},
			{"VI", "JPGVOUMFYQBENHZRDKASXLICTW","MZ"},
			{"VII", "NZJHGRCXMYSWBOUFAIVLPEKQDT","MZ"},
			{"VIII", "FKQHTLXOCBJSPDZRAMEWNIUYGV","MZ"},
	};
*/
private:
	char name[8];
	char wire[26];
	char notch[26];
	unsigned int ringsetting;

public:

	Rotor(void) {}
	Rotor(const char * label, const char * w, const char *n) {
		strcpy(name, label);
		ringsetting = 0;
		memcpy(wire, w, 26);
		strcpy(notch, n);
	}

	Rotor(const Rotor & org) {
		strcpy(name, org.name);
		ringsetting = org.ringsetting;
		memcpy(wire, org.wire, 26);
		strcpy(notch, org.notch);
	}

	char transfer(const char c) {
		return wire[(c - 'A' + ringsetting) % 26];
	}

	bool willCarry(const char c) {
		int i;
		for (i = 0; notch[i]; i++) {
			if ( c == notch[i] )
				break;
		}
		return notch[i] != 0;
	}
};

class RotorUnit {

	Rotor rotors[3], reflector;
	unsigned int position[3];

public:
	RotorUnit(void) {
		rotors[0] = Rotor("III", "BDFHJLCPRTXVZNYEIWGAKMUSQO","V");
		rotors[1] = Rotor("II","AJDKSIRUXBLHWTMCQGZNPYFVOE","E");
		rotors[2] = Rotor("I","EKMFLGDQVZNTOWYHXUSPAIBRCJ","Q");
		position[0] = 0;
		position[1] = 0;
		position[2] = 0;
		reflector = Rotor("WideB","YRUHQSLDPXNGOKMIEBFZCWVJAT","");
	}

	char translate(const char c) {
		unsigned int offset = toupper(c);
		bool carry;
		// make motion
		for(int r = 0; r < 3; r++) {
			if ( r == 0 ) {
				if ( rotors[r].willCarry(position[r]+'A') )
					carry = true;
				else
					carry = false;
				position[r]++;
				position[r] %= 26;
			} else if ( carry ) {
				if ( rotors[r].willCarry(position[r]+'A') )
					carry = true;
				else
					carry = false;
				position[r]++;
				position[r] %= 26;
			}
			// translate to reflector
			offset = rotors[r].transfer(offset+position[r]);
		}
		for(int r = 3; r > 0; r--) {
			cerr << "[" <<  (char)(position[r-1] + 'A') << "]";
		}
		cerr << endl;
		return offset;
	}

	char reflect(const char c) {
		unsigned int offset = toupper(c);
		return reflector.transfer(offset);
	}
};


int main() {
	RotorUnit rotors1_2_3;

	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!

	char str[] = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
	char tmp;
	for(int i = 0; str[i]; i++) {
		tmp = rotors1_2_3.translate(str[i]);
		cout << rotors1_2_3.reflect(tmp);
	}
	cout << endl;

	return 0;
}
