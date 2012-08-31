//
// babastate.cc - �Х�ȴ���ξ��ַ�(C++��)
//	���: (���ʤ���̾��); ����: (������������)
//
#include "babastate.h"
#include <stdio.h>

//
// BabaState::reset() - �ꥻ�å�(�ǽ�˥����ɤ��ۤä����֤ˤ���)
//
void BabaState::reset(void)
{
// �ƥץ졼��Υե饰��̤λ�ˤ������������ˤ���
	for(int i = 0; i < numplayer; i++) {
		finished[i] = false;
		hand[i].makeempty();
	}
// �ǥå�(1���å�)�Υ����ɤ� initcs �������
	CardSet initcs;
	initcs.makedeck();
// ̵���ʤ�ޤ� initcs ����̵��٤˥����ɤ�Ȥ� plr �֤Υץ졼����ۤ�
	int plr = 0;	// �ۤ����Υץ졼��
	Card c;		// �ۤ륫����
	while(!initcs.pickup(&c)) {
	// �����ۤ�줿�����ɤ�Ʊ���ֹ椬�����
	// ���٤Υ����ɤȴ����ۤ�줿�����ɤȤ�ξ����ΤƤ�
	// ̵����к��٤Υ����ɤ�ʬ�μ�˲ä���
		if(hand[plr].remove(c.gnumber()))
			hand[plr].insert(c);
	// plr �򼡤Υץ졼��ˤ���
		if(++plr >= numplayer)
			plr = 0;
	}
}

//
// BabaState::move() - to �֤Υץ졼�䤬 from �֤Υץ졼�䤫�饫���ɤ���
//
bool BabaState::move(int from, int to)
{
	Card c;	// ư����������
// from �μ꤫��1���롥����� from �˥����ɤ�̵���ʤä����ݤ����ǧ
	if(hand[from].pickup(&c))
		return true;	// from �ˤϥ����ɤ�̵��
	if(hand[from].isempty())
		finished[from] = true;
// ��ä������ɤ�Ʊ���ֹ椬 to �μ�ˤ����
// ���٤Υ����ɤȸ��Ĥ��ä������ɤȤ�ξ����ΤƤ�
// ̵����к��٤Υ����ɤ�ʬ�μ�˲ä���
// ����� to �˥����ɤ�̵���ʤä����ݤ����ǧ
	if(hand[to].remove(c.gnumber()))
		hand[to].insert(c);
	if(hand[to].isempty())
		finished[to] = true;

	return false;
}

//
// BabaState::print() - ���Ȥξ��֤�ɸ����Ϥ˽��Ϥ���
//
void BabaState::print(void)
{
	for(int i = 0; i < numplayer; i++) {
		printf("PLAYER %d ", i);
		hand[i].print();
	}
}
