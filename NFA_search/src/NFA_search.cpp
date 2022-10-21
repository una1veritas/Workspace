//============================================================================
// Name        : NFA_search.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <map>
#include <bitset>

using namespace std;

class NFA {
	static constexpr int N_STATES = 32;
	static constexpr int N_ALPHABET = 8;
	static constexpr char alphabet[] = "*-b=#+";

	bitset<N_STATES> state;
	map<uint64_t, bitset<N_STATES>> transfunc;
	static constexpr bitset<N_STATES> initial_state{0b1};
	bitset<N_STATES> finals;

public:
	static ostream & printStateOn(ostream & out, bitset<N_STATES> & s) {
		out << "{";
		int cnt = 0;
		for(int i = 0; i < N_STATES; ++i) {
			if (cnt) out << ", ";
			out << s[i];
			++cnt;
		}
		out << "}";
		return out;
	}

	NFA() : state(0), transfunc(), finals(0){ }

// ユーティリティ
#define char2state(x)  (tolower(x) - 0x30)
#define state2char(x)  ((x) + 0x30)


	void reset() {
		state = initial_state;
	}

	bitset<N_STATES> transfer(char a) {
		bitset<N_STATES> next = 0;
		uint64_t index = (state.to_ullong()<< 8) | (0x7f & a);
		if (transfunc.contains(index))
			return transfunc[index];
		for(int i = 0; i < N_STATES; ++i) {
			if (state[i] != 0) {
				uint64_t singlestateindex = ((1<<i)<<8) | (0x7f & a);
				if ( transfunc.contains(singlestateindex) ) /* if defined */
					next |= transfunc[singlestateindex];
				//else /* if omitted, go to and self-loop in the ghost state. */
			}
		}
		state = next;
		return next;
	}

	int accepting() {
		return (finals & current).size() != 0;
	}

	ostream & printOn(ostream & out) {
		out << "sate " << state.to_string('0', '1');
		return out;
	}


	int run(char * inputstr) {
		char * ptr = inputstr;
		char buf[128];
		printf("run on '%s' :\n", ptr);
		nfa_reset(mp);
		printf("     -> %s", bset64_str(mp->current, buf));
		for ( ; *ptr; ++ptr) {
			nfa_transfer(mp, *ptr);
			printf(", -%c-> %s", *ptr, bset64_str(mp->current, buf));
		}
		if ( nfa_accepting(mp) ) {
			printf(", \naccepted.\n");
			fflush(stdout);
			return STATE_IS_FINAL;
		} else {
			printf(", \nrejected.\n");
			fflush(stdout);
			return STATE_IS_NOT_FINAL;
		}
	}
};



int command_arguments(int , char ** , char ** , char * , char ** , char *);

int main(int argc, char **argv) {
	char * delta = "0a01,0b0,0c0,1a0,1b02,1c0,2a0,2b03,2c0,3a3,3b3,3c3", initial = '0', *finals = "3";
	char input_buff[1024] = "acabaccababbacbbac";
	if ( command_arguments(argc, argv, &delta, &initial, &finals, input_buff) )
		return 1;

	nfa M;
	//printf("M is using %0.2f Kbytes.\n\n", (double)(sizeof(M)/1024) );
	nfa_define(&M, delta, initial, finals);
	nfa_print(&M);
	if (strlen(input_buff))
		nfa_run(&M, input_buff);
	else {
		printf("Type an input as a line, or quit by the empty line.\n");
		fflush(stdout);
		/* 標準入力から一行ずつ，入力文字列として走らせる */
		while( fgets(input_buff, 1023, stdin) ) {
			char * p;
			for(p = input_buff; *p != '\n' && *p != '\r' && *p != 0; ++p) ;
			*p = '\0'; /* 行末の改行は消す */
			if (!strlen(input_buff))
				break;
			nfa_run(&M, input_buff);
		}
	}
	printf("bye.\n");
	return 0;
}

int command_arguments(int argc, char * argv[], char ** delta, char * initial, char ** finals, char * input) {
	if (argc > 1) {
		if (strcmp(argv[1], "-h") == 0 ) {
			printf("usage: command \"transition triples\" \"initial state\" \"final states\" (\"input string\")\n");
			printf("example: dfa.exe \"%s\" \"%c\" \"%s\"\n\n", *delta, *initial, *finals);
			return 1;
		} else if (argc == 4 || argc == 5 ) {
			*delta = argv[1]; *initial = argv[2][0]; *finals = argv[3];
			if (argc == 5 )
				strcpy(input, argv[4]);
			else
				input[0] = '\0';
		} else {
			printf("Illegal number of arguments.\n");
			return 1;
		}
	} else {
		printf("define M by built-in example: \"%s\" \"%c\" \"%s\"\n", *delta, *initial, *finals);
		printf("(Use 'command -h' to get a help message.)\n\n");
	}
	return 0;
}
