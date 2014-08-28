//============================================================================
// Name        : NetlistGraph.cpp
// Author      : Sin
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

using namespace std;

int main(int argc, char * argv[]) {

	ifstream ifreader;
	stringstream sstream("", ios_base::app | ios_base::out);
	string lines, token;
	const int bufsize = 128;
	char buf[bufsize];

	if ( argc == 1 ) {
		cerr << "Give me the file name as the argument." << endl;
		return 1;
	}

	ifreader.open(argv[1], ios::in);

	if ( !ifreader ) {
		cout.flush();
		cerr << "Couldn't open the file " << argv[1] << ". " << endl;
				return 1;
	}
	for(int cnt = 0; cnt < 6; cnt++) {
		ifreader.getline(buf, bufsize);
	}
	while ( !ifreader.eof() ) {
		int len;
		sstream.clear();
		lines.clear();
		do {
			ifreader.getline(buf, bufsize);
			len = strlen(buf);
			if ( len == 0 ) {
				cout << "-------" << endl;
				continue;
			}
			lines.append(buf);
			lines.append("\n");
		} while (len > 0);
		sstream.str(lines);
		sstream >> token;
		cout << "cnode: " << token << endl;
		while (!sstream.eof()) {
			sstream >> token;
			cout << '"' << token << "\", "<< endl;
		}
		cout << "-----" << endl;
	}
	ifreader.close();

	return 0;
}
