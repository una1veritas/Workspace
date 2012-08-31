//
//
// cardsetextra.cc - �ȥ��ץ����ɤν��緿(C++��)�ƥ��ȥץ����
//	���: (���ʤ���̾��); ����: (������������)
//
#include <iostream>
#include "cardset.h"

//
// main() - �ȥ��ץ����ɤν��緿�ƥ��ȥץ����
//

int main (int argc, char * const argv[]) {
	
    // insert code here...
    //std::cout << "Hello, World!\n";
    //return 0;
	
		Card c;
		CardSet cs;
		
		std::cout << cs << "\n\n";
		// ���Ϥ����顼�ˤʤ�ޤǻ��ꤷ�������ɤ������
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
		
		// ���Ϥ����顼�ˤʤ�ޤǻ��ꤷ�������ɤ����
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
