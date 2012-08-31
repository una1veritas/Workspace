//
// babanuki1.cc - �Х�ȴ���ץ����(C++��)
//	���: (���ʤ���̾��); ����: (������������)
//
#include "babastate.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

//
// main() - �Х�ȴ���ץ����
//
int main(void)
{
	time_t seed;
// ���ȯ���Υ�����
// (�ǥХå�����Ʊ�������ȯ�������������ϡ��ʲ���2�Ԥ��������)
	time(&seed);
	srandom(seed);

	BabaState bs;	// �Х�ȴ���ξ���

	bs.print();

// ��λ���ˤʤ�ޤǥ����ɤ����³����
	int from, to = 0; // to �� from ���饫���ɤ���
	while(1) {
	// ���� from �� to ��õ��
		for(to = (to + 1) % BabaState::numplayer;
		 bs.isfinished(to);
		 to = (to + 1) % BabaState::numplayer)
			;
		for(from = (to + BabaState::numplayer - 1) % BabaState::numplayer;
		 bs.isfinished(from);
		 from = (from + BabaState::numplayer - 1) % BabaState::numplayer)
			;
// ��λȽ��
		if(from == to)	// ������ν�λ���(1�Ͱʳ��Ͼ夬�ä�)
			break;
// from ���� to �˥����ɤ��Ϥ�
		bs.move(from, to);
		printf("# �ץ졼�� %d ����ץ졼�� %d �˥����ɤ��Ϥ�\n",			 from, to);
		bs.print();
	}

	printf("\n### FINISHED ###\n");

	return 0;
}
