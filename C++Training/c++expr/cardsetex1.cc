//
// cardsetex1.cc - �ȥ��ץ����ɤν��緿(C++��)�ƥ��ȥץ����
//	���: (���ʤ���̾��); ����: (������������)
//
#include <stdio.h>
#include "cardset.h"

//
// main() - �ȥ��ץ����ɤν��緿�ƥ��ȥץ����
//
int main(void)
{
	Card c;
	CardSet cs;

	cs.print();
// ���Ϥ����顼�ˤʤ�ޤǻ��ꤷ�������ɤ������
	printf("insert: c = ? ");
	while(!c.scan()) {
		if(cs.insert(c))
			printf("\tinsert error\n");
		printf("insert: c = ? ");
	}
	cs.print();
// ���Ϥ����顼�ˤʤ�ޤǻ��ꤷ�������ɤ����
	printf("remove: c = ? ");
	while(!c.scan()) {
		if(cs.remove(c))
			printf("\tremove error\n");
		printf("remove: c = ? ");
	}
	cs.print();
	
	return 0;
}
