//
//
// cardsetextra.cc - トランプカードの集合型(C++版)テストプログラム
//	作者: (あなたの名前); 日付: (完成した日付)
//
#include <iostream>
#include "cardset.h"

//
// main() - トランプカードの集合型テストプログラム
//

int main (int argc, char * const argv[]) {
	
    // insert code here...
    //std::cout << "Hello, World!\n";
    //return 0;
	
		Card c;
		CardSet cs;
		
		std::cout << cs << "\n\n";
		// 入力がエラーになるまで指定したカードを入れる
		std::cout << "insert: c = ? ";
		while(true) {
			std::cin >> c;
			if (! c.isValid())
				break;
			if(cs.insert(c))
				std::cout << "\tinsert error\n";
			std::cout << "insert: c = ? ";
		}
		std::cout << cs << "\n\n";
		
		// 入力がエラーになるまで指定したカードを除く
		std::cout << "remove: c = ? ";
		while(true) {
			std::cin >> c;
			if (! c.isValid())
				break;
			if(cs.remove(c))
				std::cout << "\tremove error\n";
			std::cout << "remove: c = ? ";
		}		
		std::cout << cs << "\n";
		
		return 0;
}
