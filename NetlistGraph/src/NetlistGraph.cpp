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
	stringstream sstream;
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
		string str;
		str.clear();
		do {
			ifreader.getline(buf, bufsize);
			len = strlen(buf);
			if ( len == 0 ) {
				continue;
			}
			str.append(buf);
			str.append(";");
		} while (len > 0);

		sstream.clear();
		sstream.str(str);

		string netname, part, pad, pin, sheet;
		stringstream line;
		netname.clear();
		sstream.getline(buf, bufsize, ';');
		if ( sstream >> netname ) {
			do {
				line.str(buf);
				part.clear(); pad.clear(); pin.clear(); sheet.clear();
				line >> part >> pad >> pin >> sheet;
				cout << "net name = " << netname.c_str() << ", " << pad.c_str() << ", " << pin.c_str() <<
						", " << sheet.c_str() << endl;
				line.clear();
			} while ( sstream.getline(buf, bufsize, ';') );
		}
		cout << endl;
	}
	ifreader.close();

	return 0;
}
