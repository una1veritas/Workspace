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

#include <vector>
#include <map>
#include <algorithm>

using namespace std;

struct Part {
	string label;
	string value;
	map<string,string> pads;

	Part(void) : label(""), pads() {}
	Part(const string & name) : label(name), pads() { }

	void add(string & pin, string & padname) {
		pads[pin] = padname;
	}

	string & pad(const string & pin) {
		return pads[pin];
	}

	friend ostream & operator<<(ostream & st, Part & me) {
		st << me.label << "[";
		for(map<string,string>::iterator i = me.pads.begin(); i != me.pads.end(); ) {
			cout << i->first;
			++i;
			if ( i != me.pads.end() )
				cout <<  ",";
		}
		cout << "]";
		return st;
	}
};

struct Net {
	string label;
	vector< pair<Part&,string&> > link;

	Net(void) { }
	Net(const string & name) : label(name), link() {  }

	void add(Part & part, string & pin) {
		link.push_back(pair<Part&,string&>(part,pin) );
	}

	friend ostream & operator<<(ostream & st, Net & me) {
		st << me.label << "[";
		for( vector<pair<Part&,string&> >::iterator i = me.link.begin();
			i != me.link.end() ; ) {
			pair<Part&,string&> & p = *i;
			st << p.first.label << ":" << p.second << "(" << p.first.pad(p.second) << ")";
			++i;
			if ( i != me.link.end() )
				st << ", ";
		}
		st << "] ";
		st << " ";
		return st;
	}

};

int main(int argc, char * argv[]) {

	ifstream ifreader;
	stringstream sstream;

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

	string buf;

	for(int cnt = 0; cnt < 8; cnt++) {
		std::getline(ifreader, buf);
	}

	string str, token, netname, part, pad, pin, sheet;
	stringstream line;
	int linenum = 0;
	map<string,Part> parts;
	map<string,Net> nets;
	while ( !ifreader.eof() ) {
		std::getline(ifreader, buf);
		if ( buf.length() == 0 ) {
			// the end of the last net
			linenum = 0;
			continue;
		}
		line.str(buf);
		line.clear();
		if ( linenum == 0 ) {
			line >> netname;
			if( nets.find(netname) == nets.end() ) {
				nets[netname] = Net(netname);
			}
		}
		if ( !line.eof() ) {
			line >> part >> pad >> pin >> sheet;
			if ( parts.find(part) == parts.end() ) {
				parts[part] = Part(part);
			}

			//cout << "part " << p << " id, pin " << id << ", " << pin << endl;
			Part & p = parts[part];
			p.add(pad, pin);
			nets[netname].add(p, pad);
			//cout << netname << "<-->" << part << ":" << pad << " (" << pin << "," << sheet << ")" << endl;
			//cout << nets[netname] << endl;
		}
		linenum++;
	}
	ifreader.close();

	cout << endl << "Bill of Materials: " << endl;
	for( map<string,Part>::iterator i = parts.begin(); i != parts.end(); ++i ) {
		cout << i->second << endl;
	}

	cout << endl << "Netlist:" << endl;
	for( map<string,Net>::iterator i = nets.begin(); i != nets.end(); ++i ) {
		cout << i->second << endl;
	}

	return 0;
}
