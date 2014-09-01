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
	string val, dev, pkg;
	map<string,string> pads;

	Part(void) : label(""), val(""), dev(""), pkg(""), pads() {}
	Part(const string & name) : label(name), val(""), dev(""), pkg(""), pads() { }

	void setProperty(const string & v, const string & d, const string & p) {
		val = v;
		dev = d;
		pkg = p;
	}
	string & value(void) { return val; }
	void add(string & pin, string & padname) {
		pads[pin] = padname;
	}

	string & pad(const string & pin) {
		return pads[pin];
	}

	ostream & printOn(ostream & st) {
		st << label << "[";
		for(map<string,string>::iterator i = pads.begin(); i != pads.end(); ) {
			cout << i->first;
			++i;
			if ( i != pads.end() )
				cout <<  ",";
		}
		cout << "]";
		if ( !val.empty() ) {
			cout << " " << val;
		}
		if ( !dev.empty() ) {
			cout << " " << dev;
		}
		if ( !pkg.empty() ) {
			cout << " " << pkg;
		}
		return st;
	}
	friend ostream & operator<<(ostream & st, Part & me) { return me.printOn(st); }
};

struct Net {
	string label;
	typedef pair<Part&,string> conn;
	vector< conn > link;

	Net(void) { }
	Net(const string & name) : label(name), link() {  }

	void add(Part & part, string & pin) {
		link.push_back(conn(part,pin) );
	}

	ostream & printOn(ostream & st);
	friend ostream & operator<<(ostream & st, Net & me) {
		return me.printOn(st);
	}

};

ostream & Net::printOn(ostream & st) {
	st << label << "[";
	for( vector<conn>::iterator i = link.begin(); i != link.end() ; ) {
		conn p = *i;
		st << p.first.label << ":" << p.second << "(" << p.first.pad(p.second) << ")";
		++i;
		if ( i != link.end() )
			st << ", ";
	}
	st << "] ";
	st << " ";
	return st;
}

void readNetlist(map<string,Net> &, map<string,Part> &, ifstream &, ifstream &);

int main(int argc, char * argv[]) {

	ifstream inetlist;
	ifstream ipartlist;
	map<string, Net> nets;
	map<string, Part> parts;

	if ( argc < 2 ) {
		cerr << "Give me the file name as the argument." << endl;
		return 1;
	}

	inetlist.open(argv[1], ios::in);
	if ( argc >= 3 ) {
		ipartlist.open(argv[2], ios::in);
	}

	if ( !inetlist ) {
		cout.flush();
		cerr << "Couldn't open the file " << argv[1] << ". " << endl;
		return 1;
	}

	readNetlist(nets, parts, inetlist, ipartlist);
	inetlist.close();

	cout << endl << "Parts: " << endl;
	for( map<string,Part>::iterator i = parts.begin(); i != parts.end(); ++i ) {
		cout << i->second << endl;
	}

	cout << endl << "Netlist:" << endl;
	for( map<string,Net>::iterator i = nets.begin(); i != nets.end(); ++i ) {
		cout << i->second << endl;
	}

	return 0;
}

void readNetlist(map<string,Net> & nets, map<string,Part> & parts,
		ifstream & netlist, ifstream & partlist) {

	string buf;
	stringstream line;
	string tmp[6];

	if ( partlist.is_open() ) {
		//cout << "partlist is opened." << endl;
		for(int cnt = 0; cnt < 10 && getline(partlist, buf); cnt++);
		while ( !partlist.eof() ) {
			getline(partlist, buf);
			for(int i = 0; i < buf.length(); ++i) {
				if ( (unsigned char)buf[i] == 0xb5 )
					buf[i] = 'u';
			}
			line.str(buf);
			line.clear();
			// Part     Value          Device      Package  Library    Sheet
			line >> tmp[0] >> tmp[1] >> tmp[2] >> tmp[3] >> tmp[4] >> tmp[5];
			//cout << tmp[0] << ',' << tmp[1] << ',' << tmp[2] << ',' << tmp[3] << endl;
			parts[tmp[0]] = Part(tmp[0]);
			parts[tmp[0]].setProperty(tmp[1], tmp[2], tmp[3]);
		}
	} else {
		//cout << "skip partlist." << endl;
	}

	for(int cnt = 0; cnt < 8; cnt++) {
		getline(netlist, buf);
	}

	string str, netname, part, pad, pin, sheet;
	int linenum = 0;
	while ( !netlist.eof() ) {
		getline(netlist, buf);
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

}
