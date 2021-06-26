#include <iostream>
#include <cstring>

struct StringDecimal {
	char * digits;
	int length;
	int afterdp;

	StringDecimal(const char * str) {
		length = strlen(str);
		if (length > 0) {
			digits = new char[length];
			strcpy(digits, str);
		} else {
			length = strlen("0");
			digits =  new char[length];
			strcpy(digits,"0");
		}
		for(int i = 0; i < length; ++i) {
			if (str[i] == '.') {
				afterdp = length - i - 1;
				break;
			}
		}
	}

	~StringDecimal() {
		delete [] digits;
	}

	StringDecimal frac() {
		char zerop[2+afterdp];
		strcpy(zerop, "0.");
		strcat(zerop, digits+(length - afterdp) );
		return StringDecimal(zerop);
	}

	friend std::ostream & operator<<(std::ostream & out, const StringDecimal & sd) {
		out << sd.digits;
		return out;
	}
};

int main(const int argc, const char **argv) {
	std::cout << "Hi." << std::endl;
	StringDecimal a(argv[1]), b(argv[2]);
	std::cout << a << ", " << b << ", frac(b) = " << b.frac() << std::endl;
	return 0;
}
