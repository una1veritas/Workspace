#include <iostream>
#include <iomanip>
#include <fstream>
#include <cctype>
#include <cstring>

using namespace std; // std:: 名前空間を省略して使用

typedef uint16_t uint16;

bool check_signature(ifstream & ifile, const char * sigstr, const int length);
bool get_uint16(ifstream & ifile, uint16 & val);

struct SMFScore {
	uint16 format, tracks, division;
	uint16 status;

	enum {
		ERROR_HEADER = 1<<0,
		ERROR_MTHD = 1<<1,
		ERROR_MTRK = 1<<2,
	};

	SMFScore(ifstream & ifs) : format(0), tracks(0), division(0), status(0) {
		unsigned char t[18];
		status = 0;
		if ( !ifs.read((char*) t, 18) ) {
			status |= ERROR_HEADER;
			return;
		}
		if ( memcmp(t, "MThd", 4) != 0 ) {
			status |= ERROR_MTHD;
			return;
		}
		if ( memcmp(t+14, "MTrk", 4) != 0 ) {
			status |= ERROR_MTRK;
			return;
		}
		format = t[8]<<8 | t[9];
		tracks = t[10]<<8 | t[11];
		division = t[12]<<8 | t[13];
	}

	friend ostream & operator<<(ostream & ost, const SMFScore & score) {
		ost << score.format << ", " << score.tracks << ", " << score.division;
		return ost;
	}
};
/*
	private static int parseVarLenInt(InputStream stream) throws IOException {
		int oneByte, value = 0; // for storing unsigned byte
		while ( (oneByte = stream.read()) != -1 ) {
			value = value << 7;
			value += oneByte & 0x7f;
			if ( (oneByte & 0x80) == 0 )
				break;
		}
		if ( oneByte == -1 )
			return -1;
		return value;
	}
*/

int main(int argc, char **argv) {
	ifstream infile;

	infile.open("panzerlied.mid", (ios::in | ios::binary) );

	if ( !infile ) {
		cerr << "失敗" << endl;
		return -1;
	}

	SMFScore smfscore(infile);
	cout << smfscore << endl;

	infile.close();
	return 0;
}

bool check_signature(ifstream & ifile, const char * sigstr, const int length) {
	char binstr[length];
	if ( !ifile.read(binstr, length) )
		return false;
	if ( memcmp(sigstr, binstr, length) != 0 )
		return false;
	return true;
}

bool get_uint16(ifstream & ifile, uint16 & val) {
	char t[2];
	if ( !ifile.read(t, 2) )
		return false;
	val = (unsigned char) t[0];
	val <<= 8;
	val |= (unsigned char) t[1];
	return true;
}
