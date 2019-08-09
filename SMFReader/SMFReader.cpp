#include <iostream>
#include <iomanip>
#include <fstream>
#include <cctype>
#include <cstring>

typedef uint16_t uint16;

struct SMFScore {
	uint16 format, tracks, division;
	uint16 status;

	enum {
		ERROR_HEADER = 1<<0,
		ERROR_MTHD = 1<<1,
		ERROR_MTRK = 1<<2,
	};

	static void read_event(std::ifstream & ifs) {
		unsigned int deltat = SMFScore::read_varlenint(ifs);
	}

	static unsigned int read_varlenint(std::ifstream & ifs) {
		char tbyte;
		unsigned int val = 0;
		while ( ifs.get(tbyte) ) {
			val <<= 7;
			val |= 0x07f & tbyte;
			if ( (tbyte & 0x80) == 0 )
				break;
		}
		return val;
	}

	SMFScore(std::ifstream & ifs) : format(0), tracks(0), division(0), status(0) {
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

	friend std::ostream & operator<<(std::ostream & ost, const SMFScore & score) {
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
	std::ifstream infile;

	infile.open(argv[1], (std::ios::in | std::ios::binary) );

	if ( !infile ) {
		std::cerr << "失敗" << std::endl;
		return -1;
	}

	SMFScore smfscore(infile);
	std::cout << smfscore << std::endl;
	char buf[4];
	unsigned int deltat;
	unsigned int len;
	unsigned char tbyte;
	infile.read(buf,4);
	deltat = (unsigned int) SMFScore::read_varlenint(infile);
	tbyte = (unsigned char) infile.get();
	std::cout << "delta time = " << deltat << ", ";
	if ( (tbyte & 0x80) != 0 ) {
		// status byte
		std::cout << "status byte: ";
		switch (tbyte) {
		case 0xf0:
			// system exclusive event
			std::cout << "system exclusive event, ";
			break;
		case 0xf7:
			// escaped system exclusive event
			std::cout << "secaped system exclusive event, ";
			break;
		case 0xff:
			// meta event
			std::cout << "meta event, ";
			tbyte = (unsigned char) infile.get(); // event type
			len = SMFScore::read_varlenint(infile); // event size
			if (tbyte == 0x2f) {
				// may be 'end of track'
				// len is always assumed to be zero.
				//score.add(new MusicalNote(-1, -1, -1));
				std::cout << "end of track, ";
				break; // go to the next track.
			} else {
				std::cout << "len = " << len << ", ";
				do {
					infile.get();
				} while (--len);
			}
		}
	} else {
		std::cout << "???";
	}
	std::cout << std::endl;
	infile.close();
	return 0;
}
