/*
 * reader.cpp
 *
 *  Created on: 2017/06/09
 *      Author: sin
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>

using namespace std;

int main(const int argc, const char * argv[]) {
	cout << "Hello." << endl;

	std::ifstream binfile(argv[1], std::ios::binary);

	if ( ! binfile )
		return -1;

	std::vector<char> buff( (std::istreambuf_iterator<char>(binfile)),
			(std::istreambuf_iterator<char>()) );

	binfile.close();

	long addr;
	for( addr = 0; addr < buff.size(); addr++) {
		if ( (addr & 0x0f) == 0 ) {
			std::cout << std::endl;
			std::cout << std::setw(4) << std::setfill('0') << std::hex << addr << ": ";
		}
		std::cout << std::setw(2) << std::hex << ((unsigned int) buff[addr] & 0xff) << " ";
	}
	std::cout << std::endl;

	return 0;
}
