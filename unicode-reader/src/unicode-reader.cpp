//============================================================================
// Name        : unicode-reader.cpp
// Author      : Sin Shimozono
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <fstream>
//#include <iterator>
#include <unordered_map>
#include <vector>
#include <math.h>

using namespace std;

long read_utf8(std::istreambuf_iterator<char> & iter) {
	unsigned char firstbyte = *iter;
	if ( iter == istreambuf_iterator<char>() )
		return -1;
	++iter;
	if ( (firstbyte & 0x80) == 0 )
		return firstbyte;
	long code = firstbyte;
	firstbyte <<= 1;
	while ( firstbyte & 0x80 ) {
		code <<= 8;
		code |= (unsigned char)(*iter);
		++iter;
		firstbyte <<= 1;
	}
	return code;
}

int read_byte(std::istreambuf_iterator<char> & iter) {
	unsigned char firstbyte = *iter;
	if ( iter == istreambuf_iterator<char>() )
		return -1;
	++iter;
	return firstbyte;
}

int main(const int argc, const char * argv[]) {
	unordered_map<long, int> hist_utf8, hist_byte;
	if (argc != 2) {
		cerr << "usage: command filename[cr]" << endl;
		exit(1);
	}
	cout << "input file: \"" << argv[1] << "\"" << endl;
	ifstream inputf(argv[1], ios::binary);

	cout << "utf-8 stats" << endl;
	for (istreambuf_iterator<char> iter = istreambuf_iterator<char>(inputf);
			iter != istreambuf_iterator<char>(); ){
		int code = read_byte(iter);
		if (hist_byte.count(code)) {
			hist_byte[code] += 1;
		} else {
			hist_byte[code] = 1;
		}
		//cout << std::hex << code << endl;
	}

	inputf.clear();
	inputf.seekg(0);
	for (istreambuf_iterator<char> iter = istreambuf_iterator<char>(inputf);
			iter != istreambuf_iterator<char>(); ){
		long code = read_utf8(iter);
		if (hist_utf8.count(code)) {
			hist_utf8[code] += 1;
		} else {
			hist_utf8[code] = 1;
		}
		//cout << std::hex << code << endl;
	}
	inputf.close();

	vector<pair<long, int> > codes(hist_utf8.cbegin(), hist_utf8.cend());
	sort(codes.begin(), codes.end());
	long sum = 0;
	for(auto i = codes.begin(); i != codes.end(); ++i) {
		cout << std::hex << i->first << ": " << std::dec << i->second << endl;
		sum += i->second;
	}
	cout << endl;

	double rate, h = 0.0;
	for(auto i = codes.begin(); i != codes.end(); ++i) {
		rate = (double) i->second / sum;
		h += - rate * log2(rate);
	}
	cout << "entropy in utf-8: " << h << endl << endl;

	codes.clear();
	std::copy(hist_byte.cbegin(), hist_byte.cend(), std::back_inserter(codes));
	sort(codes.begin(), codes.end());
	sum = 0;
	for(auto i = codes.begin(); i != codes.end(); ++i) {
		cout << std::hex << i->first << ": " << std::dec << i->second << endl;
		sum += i->second;
	}
	cout << endl;

	h = 0.0;
	for(auto i = codes.begin(); i != codes.end(); ++i) {
		rate = (double) i->second / sum;
		h += - rate * log2(rate);
	}
	cout << "entropy in byte: " << h << endl;

	return 0;
}
