//
// cardsetex1.cc - �g�����v�J�[�h�̏W���^(C++��)�e�X�g�v���O����
//	���: (���Ȃ��̖��O); ���t: (�����������t)
//
#include <stdio.h>
#include "Card.h"
#include "CardSet.h"

//
// main() - �g�����v�J�[�h�̏W���^�e�X�g�v���O����
//
int main(void)
{
	Card c;
	CardSet cs;

	cs.print();
// ���͂��G���[�ɂȂ�܂Ŏw�肵���J�[�h������
	printf("insert: c = ? ");
	while(!c.scan()) {
		if(cs.insert(c))
			printf("�_tinsert error�_n");
		printf("insert: c = ? ");
	}
	cs.print();
// ���͂��G���[�ɂȂ�܂Ŏw�肵���J�[�h������
	printf("remove: c = ? ");
	while(!c.scan()) {
		if(cs.remove(c))
			printf("�_tremove error�_n");
		printf("remove: c = ? ");
	}
	cs.print();
	
	return 0;
}
