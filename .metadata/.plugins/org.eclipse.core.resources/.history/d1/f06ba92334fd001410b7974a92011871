//============================================================================
// Name        : tm.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

//#include <stdio.h>
//#include <conio.h>
#include <unistd.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <time.h>
//#include <windows.h>
#include <cmath>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <set>
#include <map>
#include <vector>
#include <stack>

using namespace std;

struct Tuple {
	string current, next; // state[0]:現状態、state[1]:次状態
//	char alph[4];		// alph[0]:入力用テープの読み込み、alph[1]:作業用テープの読み込み、
			// alph[2]:書き込み、alph[3]:書き込み
	string read, write, headding; // hder[0]:入力用テープのヘッドの動き、hder[1]:作業用テープの

	Tuple(int tapes) {
		current = "";
		next = "";
		read = string(tapes, ' ');
		write = string(tapes, ' ');
		headding = string(tapes, ' ');
	}

	// input & output ;
	friend ostream & operator <<(ostream & stream, const Tuple & obj) {
		stream << obj.current << ", " << obj.read << ", " << obj.next << ", " << obj.write << ", " << obj.headding;
		return stream;
	}

};

struct TuringMachine {
public:
	static const char BLANK = '_';
	static const char SPECIAL_DONTCARE = '*';
	static const char SPECIAL_THESAME = '*';

private:
	string::iterator * head;
	string * tape;
	string state;

public:
	vector<Tuple> table;
	unsigned int tapes;
	set<string> acceptingStates;

	int step;

	TuringMachine() {
		tapes = 1;
		//step = 0;
		//answ = 0;
	}

public:
	void maketable(char[]);
	void simulate(string, string[]);
	void print(int); //string state);
	bool searchin(string state, char oninput, char onwork);

};

// メイン関数
int main(int argc, char * argv[]) {

	TuringMachine tm;

	unsigned int i; //,v;			// v:mapのキー
	string inputTape;

	// テープ初期化
	if (argc == 3)
		inputTape = argv[2];

	cout << "Version 12.0419" << endl << endl;

	// TMファイルを読み込み、状態遷移表を作成する
	tm.maketable(argv[1]);

// 遷移関数の表示
	cout << "---Transition table---" << endl;
	for (i = 0; i < tm.table.size(); i++)
		cout << tm.table[i].current << ' ' << tm.table[i].read[0] << ' '
				<< tm.table[i].read[1] << " -> " << tm.table[i].next << " ("
				<< tm.table[i].write[0] << ", " << tm.table[i].headding[0]
				<< "), (" << tm.table[i].write[1] << ", "
				<< tm.table[i].headding[1] << ") " << endl;
	cout << "---Table end---" << endl;
	cout << "Accepting states: ";
	for (set<string>::iterator ep = tm.acceptingStates.begin();
			ep != tm.acceptingStates.end(); ep++) {
		cout << *ep << ", ";
	}
	cout << endl;
	// シミュレート実行
	cout
	//		<< " Redo      -> 'r'" << endl
	<< " Go next step         -> '[return]'" << endl
//		<< " Undo      -> 'u'" << endl
			<< " Continue until halt  -> 'c'" << endl
//		<< " Traverse  -> 't'" << endl
			<< " Exit                 -> 'e'" << endl << endl;

	string * workingTapes = new string[tm.tapes];
	for(unsigned int i = 1; i < tm.tapes; i++) {
		workingTapes[i] += tm.BLANK;
	}
	tm.simulate(inputTape, workingTapes);

	delete[] workingTapes;
	return 0;
}

// 状態遷移表のチェックと行数の確認
void TuringMachine::maketable(char fname[]) {

	ifstream fin(fname);
//	int i;
	istringstream strin;
	string buff;
	int v;
	//int k = 0;

	// inspecting the number of tapes
	string dummy;
	char tapealphabet;
	unsigned int c = 0;
	set<string> states;

	getline(fin, buff, '\n');
	strin.str(buff);
	strin.clear();
	for (c = 0; !strin.eof(); c++) {
		strin >> dummy;
	}
//	cerr << "columns " << columns << endl;
	tapes = (c - 2) / 3;

	set<char> * tapeAlphabets = new set<char>[tapes]; // = new set<char>[k];
	fin.clear();
	fin.seekg(0, std::ios::beg);
	while (!fin.eof()) {
		if (getline(fin, buff, '\n') == 0)
			continue;
		//	cout << "'" << buff << "'" << buff.empty() << endl;
		strin.str(buff);
		strin.clear();
		strin >> dummy;
		if (dummy.empty())
			continue;
		//	cout << dummy << endl;
		states.insert(dummy);
		for (c = 0; c < tapes; c++) {
			strin >> tapealphabet;
			if ( tapealphabet == TuringMachine::SPECIAL_DONTCARE || tapealphabet == TuringMachine::SPECIAL_THESAME )
				continue;
			tapeAlphabets[c].insert(tapealphabet);
		}
		strin >> dummy;
		states.insert(dummy);
		//	cout << dummy << endl;
		for (c = 0; c < tapes; c++) {
			strin >> tapealphabet;
			strin >> dummy; // head motion
			if ( tapealphabet == TuringMachine::SPECIAL_DONTCARE || tapealphabet == TuringMachine::SPECIAL_THESAME )
				continue;
			tapeAlphabets[c].insert(tapealphabet);
		}
	}

	cout << "This is " << tapes << " tape machine w/ states ";
	for (set<string>::iterator p = states.begin(); p != states.end(); p++) {
		cout << *p << ", ";
	}
	cout << endl;
	cout << "and, for each tape, alphabet is as follows: " << endl;
	for (unsigned int c = 0; c < tapes; c++) {
		cout << "Tape " << c + 1 << " alphabet =";
		cout << " {";
		for (set<char>::iterator p = tapeAlphabets[c].begin();
				p != tapeAlphabets[c].end(); p++) {
			cout << *p << ", ";
		}
		cout << "}, ";
	}
	cout << endl;

	delete[] tapeAlphabets;
	//
	//
	fin.clear();
	fin.seekg(0, std::ios::beg);
	v = 0;
	bool skipremaining = false;
	while (!fin.eof()) {
		if ( skipremaining ) break;
		getline(fin, buff, '\n');
		strin.str(buff);
		strin.clear();
		if (string(buff).empty())
			continue;

		table.push_back(Tuple(tapes));
		for (c = 0; c < tapes; c++) {
			table.back().read += " ";
			table.back().write += " ";
		}
		strin >> table.back().current;
		if (table.back().current[0] == '!') {
			table.back().current = (string) table.back().current.substr(1,
					table.back().current.size());
			acceptingStates.insert(table.back().current);
		}
		for (c = 0; c < tapes; c++) {
			table.back().read += " ";
			strin >> table.back().read[c];
		}
		strin >> table.back().next;
		if (table.back().next[0] == '!') {
			table.back().next = (string) table.back().next.substr(1,
					table.back().next.size());
			acceptingStates.insert(table.back().next);
		}
		for (c = 0; c < tapes; c++) {
			table.back().write += " ";
			table.back().headding += " ";
			strin >> table.back().write[c] >> table.back().headding[c];
		}
		//cerr << table[table.size()].read[0] << ", " << table[table.size()].read[1] << endl;
		if (fin.eof())
			break;
		if (states.count(table.back().current) == 0)
			cerr << "Error!!" << endl << flush;
		// テープ記号がアルファベットもしくは数字かをチェック。
		for (c = 0; c < tapes; c++) {
			if (!(isgraph(table.back().read[c])
					&& isgraph(table.back().write[c]))) {
				cout << "table-" << table.size() << ":Improper format." << endl;
				skipremaining = true;
				break; //exit(1);
			}
			if (!(table.back().headding[c] == 'R'
					|| table.back().headding[c] == 'L'
					|| table.back().headding[c] == 'N')) {
				cout << "table-" << table.size()
						<< ":Improper head motion specification." << endl;
				exit(1);
			}
		}
		// 遷移がR,L,Nのいずれかになっているかをチェック。
//		table.size()++;
	}
	fin.close();

	return;

}

// tapeを読み状態遷移を実行する関数
void TuringMachine::simulate(string input, string work[]) {
	// adrs:テープのヘッダの位置、s:現状態、step:ステップ数
	unsigned int i;
	int searchOffset, /* adrs[2],*//* s=0, */step = 0; //,undo;
	char c = 'n';
	string acc;
//	time_t t1,t2;
	map<string, int>::iterator sitr;
	map<char, int>::iterator hitr;

	int headd;

	tape = new string[tapes];
	head = new string::iterator[tapes];

	tape[0] = input;
	head[0] = tape[0].begin();
	//cerr << head[0] << "," << input.begin()<< endl;
	for (i = 1; i < tapes; i++) {
		tape[i] = work[i];
		head[i] = tape[i].begin();
	}
	state = table[0].current; //the initial state

	// 初期状態の印字
	print(step);
	// 乱数の初期化
	srand(time(NULL));
	// 状態遷移を行う
	while (true) {
		searchOffset = rand() % table.size();
		for (i = 0; i < table.size(); i++) {
			Tuple & currentTuple = table[(searchOffset + i) % table.size()];
			if (currentTuple.current == state) {
				unsigned int tn;
				//cerr << currentTuple << endl;
				string expectstr(""), headstr("");
				for (tn = 0; tn < tapes; tn++) {
					if ( currentTuple.read[tn] == TuringMachine::SPECIAL_DONTCARE ) {
						expectstr += *(head[tn]);
					} else {
						expectstr += currentTuple.read[tn];
					}
					headstr += *(head[tn]);
				}
				//cerr << expectstr << " - " << headstr << endl;
				if ( headstr == expectstr)
					break;
			}
		}
		if ( (unsigned ) i == table.size())
			break;
		// preserve the row-in-table number of the tuple
		i = (searchOffset + i) % table.size();
		if (c == 'r') {
			c = 'n';
		} else if (c != 's' && c != 'c') {
			cout << ">>" << endl;
			c = (char) getchar();
		}
		switch (c) {
		case 'e':
			exit(0);
		case 'c':
			//Sleep(750);
			usleep(250000);
			break;
		}

		// データの書き換え
		for (unsigned int k = 0; k < tapes; k++) {
			if (table[i].write[k] == TuringMachine::SPECIAL_THESAME) {
				//*head[k] = *head[k];
				// implements this by don't touch
			} else {
				*head[k] = table[i].write[k];
			}
			switch (table[i].headding[k]) {
			case 'R':
			case 'r':
				headd = +1;
				break;
			case 'L':
			case 'l':
				headd = -1;
				break;
			default: // 'N' or 'n'
				headd = 0;
				break;
			}
			if (head[k] == tape[k].begin() && headd == -1) {
				tape[k] = string("_") + tape[k];
				head[k] = tape[k].begin();
			} else {
				head[k] = head[k] + headd;
			}
			if (head[k] == tape[k].end()) {
				tape[k] += "_";
				head[k] = tape[k].end();
				head[k]--;
			}
			state = table[i].next;
		}
		step++;
		print(step);
		// undo,redo
	}
	cout << endl << "The Machine has halted at the state '" << state << "' and " << endl;
	if (acceptingStates.find(state) != acceptingStates.end()) {
		cout << "accepted ";
	} else {
		cout << "rejected ";
	}
	cout << "input '" << input << "'." << endl << endl;

	if (step == 0)
		exit(0);
}

// ステップ毎の状態を表示する関数
void TuringMachine::print(int step) { //string state){
	string::iterator h;

	cout << endl << "Step: " << step << ", ";
	if (acceptingStates.find(state) != acceptingStates.end())
		cout << "Accepting ";
	cout << "State: " << state << endl;
	// 入力用テープの表示
	cout << "Input tape: " << endl;
	for (h = tape[0].begin() - 1; h != tape[0].end(); h++) {
		if (h + 1 != tape[0].begin())
			cout << *h;
		if (h + 1 == head[0]) {
			cout << "[";
		} else if (h == head[0]) {
			cout << "]";
		} else {
			cout << " ";
		}
	}
	cout << endl;

	// 作業用テープの表示
	cout << "Working tape:";
	for (unsigned int tn = 1; tn < tapes; tn++) {
		cout << endl;
		for (h = tape[tn].begin() - 1; h != tape[tn].end(); h++) {
			if (h + 1 != tape[tn].begin())
				cout << *h;
			if (h + 1 == head[tn]) {
				cout << "[";
			} else if (h == head[tn]) {
				cout << "]";
			} else {
				cout << " ";
			}
		}
	}
	cout << endl;

}

bool TuringMachine::searchin(string s, char in, char wk) {

	for (unsigned int i = 0; i < table.size(); i++)
		if (table[i].current == s && table[i].read[0] == in
				&& table[i].read[1] == wk)
			return true;
	return false;
}
