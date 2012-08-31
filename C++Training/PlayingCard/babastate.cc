//
// babastate.cc - �o�o�����̏�Ԍ^(C++��)
//	���: (���Ȃ��̖��O); ���t: (�����������t)
//
#include "Card.h"
#include "CardSet.h"
#include "babastate.h"
#include <stdio.h>

//
// BabaState::reset() - ���Z�b�g(�ŏ��ɃJ�[�h��z������Ԃɂ���)
//
void BabaState::reset(void)
{
// �e�v���[���̃t���O�𖢗��ɂ��C���������ɂ���
	for(int i = 0; i < numplayer; i++) {
		finished[i] = false;
		hand[i].makeempty();
	}
// �f�b�N(1�Z�b�g)�̃J�[�h�� initcs �ɓ����
	CardSet initcs;
	initcs.makedeck();
// �����Ȃ�܂� initcs ���疳��ׂɃJ�[�h���Ƃ� plr �Ԃ̃v���[���ɔz��
	int plr = 0;	// �z�鑊��̃v���[��
	Card c;		// �z��J�[�h
	while(!initcs.pickup(&c)) {
	// ���ɔz��ꂽ�J�[�h�Ɠ����ԍ��������
	// ���x�̃J�[�h�Ɗ��ɔz��ꂽ�J�[�h�Ƃ̗������̂Ă�
	// ������΍��x�̃J�[�h�������̎�ɉ�����
		if(hand[plr].remove(c.gnumber()))
			hand[plr].insert(c);
	// plr �����̃v���[���ɂ���
		if(++plr >= numplayer)
			plr = 0;
	}
}

//
// BabaState::move() - to �Ԃ̃v���[���� from �Ԃ̃v���[������J�[�h�����
//
bool BabaState::move(int from, int to)
{
	Card c;	// �������J�[�h
// from �̎肩��1�����D����� from �ɃJ�[�h�������Ȃ������ۂ����m�F
	if(hand[from].pickup(&c))
		return true;	// from �ɂ̓J�[�h������
	if(hand[from].isempty())
		finished[from] = true;
// ������J�[�h�Ɠ����ԍ��� to �̎�ɂ����
// ���x�̃J�[�h�ƌ��������J�[�h�Ƃ̗������̂Ă�
// ������΍��x�̃J�[�h�������̎�ɉ�����
// ����� to �ɃJ�[�h�������Ȃ������ۂ����m�F
	if(hand[to].remove(c.gnumber()))
		hand[to].insert(c);
	if(hand[to].isempty())
		finished[to] = true;

	return false;
}

//
// BabaState::print() - ���g�̏�Ԃ�W���o�͂ɏo�͂���
//
void BabaState::print(void)
{
	for(int i = 0; i < numplayer; i++) {
		printf("PLAYER %d ", i);
		hand[i].print();
	}
}
