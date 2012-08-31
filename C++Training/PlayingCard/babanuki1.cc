//
// babanuki1.cc - �o�o�����v���O����(C++��)
//	���: (���Ȃ��̖��O); ���t: (�����������t)
//
#include "babastate.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

//
// main() - �o�o�����v���O����
//
int main(void)
{
	time_t seed;
// ���������̃V�[�h
// (�f�o�b�O���œ��������𔭐������������́C�ȉ���2�s����������)
	time(&seed);
	srandom(seed);

	BabaState bs;	// �o�o�����̏��

	bs.print();

// �I�������ɂȂ�܂ŃJ�[�h������������
	int from, to = 0; // to �� from ����J�[�h�����
	while(1) {
	// ���� from �� to ��T��
		for(to = (to + 1) % BabaState::numplayer;
		 bs.isfinished(to);
		 to = (to + 1) % BabaState::numplayer)
			;
		for(from = (to + BabaState::numplayer - 1) % BabaState::numplayer;
		 bs.isfinished(from);
		 from = (from + BabaState::numplayer - 1) % BabaState::numplayer)
			;
// �I������
		if(from == to)	// �Q�[���̏I������(1�l�ȊO�͏オ����)
			break;
// from ���� to �ɃJ�[�h���n��
		bs.move(from, to);
		printf("# �v���[�� %d ����v���[�� %d �ɃJ�[�h���n��_n",			 from, to);
		bs.print();
	}

	printf("�_n### FINISHED ###�_n");

	return 0;
}
