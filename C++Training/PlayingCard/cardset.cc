//
// cardset.cc - �g�����v�J�[�h�̏W���^(C++��)
//	���: (���Ȃ��̖��O); ���t: (�����������t)
//
#include <stdlib.h>
#include <stdio.h>
#include "cardset.h"


//
// Card::scan() - �W���o�͂��玩�g�ɓ��͂���(true: �G���[; false: ����I��)
//
bool Card::scan(void)
{
	char buf[BUFSIZ];
	char* suitname[] = { "spade", "diamond", "heart", "club" };
// 4�g�̂����ꂩ�Ȃ�ԍ������͂���
	if(scanf("%s", buf) < 1)
		return true;
	for(int s = SUIT_SPADE; s <= SUIT_CLUB; s++)
		if(!strcmp(buf, suitname[s])) {
			suit = s;
			if(scanf("%d", &number) < 1)
				return true;
			if(number < 1 || number > 13)
				return true;
			return false;
		}
// joker �͂��̂܂�(number �� 0 �Ƃ���)
	if(!strcmp(buf, "joker")) {
		suit = SUIT_JOKER;
		number = 0;
		return false;
	}

	return true;	// �G���[
}

//
// Card::print() - ���g�̒l��W���o�͂ɏo�͂���
//
void Card::print(void)
{
	char* suitname[] = { "spade", "diamond", "heart", "club" };

	if(suit < SUIT_JOKER)
		printf("[%s %d]", suitname[suit], number);
	else if(suit == SUIT_JOKER)
		printf("[joker]");
}

//
// CardSet::locate() - �����ł� target �̃J�[�h�̈ʒu��Ԃ�(-1: �Ȃ�)
//
int CardSet::locate(Card target)
{
	for(int i = 0; i < numcard; i++)
		if(target.equal(cdat[i]))
			return i;

	return -1;	// ������Ȃ�����
}

//
// CardSet::locate() - �����ł� num �Ԃ̃J�[�h�̈ʒu��Ԃ�(-1: �Ȃ�)
//
int CardSet::locate(int number)
{
	for(int i = 0; i < numcard; i++)
		if(number == cdat[i].gnumber())
			return i;

	return -1;	// ������Ȃ�����
}

//
// CardSet::makedeck() - ���g�ɑS����(maxnumcard ����)�J�[�h������
//
void CardSet::makedeck(void)
{
	Card c;
	int suit, num;

	for(suit = SUIT_SPADE; suit <= SUIT_CLUB; suit++)
		for(num = 1; num <= 13; num++) {
			c.set(suit, num);
			insert(c);
		}
	c.set(SUIT_JOKER, 0);
	insert(c);
}

//
// CardSet::pickup() - ���g���� targetpos ���ڂ̃J�[�h������ *ret �ɕԂ�
//	targetpos �� -1 �̂Ƃ��̓����_���ɑI��(true: ���s; false: ����)
//
bool CardSet::pickup(Card* ret, int targetpos /* = -1 */)
{
	if(numcard == 0)
		return true;
	if(targetpos < 0)
		targetpos = random() % numcard;
	else
		targetpos %= numcard;

	*ret = cdat[targetpos];
	remove(*ret);

	return false;
}

//
// CardSet::insert() - ���g�� newcard �̃J�[�h������(true: ���s; false: ����)
//
bool CardSet::insert(Card newcard)
{
	if(locate(newcard) >= 0)
		return true;	// ���ɂ���
// �Ō�ɒǉ�
	cdat[numcard] = newcard;
	numcard++;

	return false;
}

//
// CardSet::remove() - ���g���� target �̃J�[�h������(true: ���s; false: ����)
//
bool CardSet::remove(Card target)
{
	int pos;

// �����J�[�h�̈ʒu�����߂�
	if((pos = locate(target)) < 0)
		return true;	// target �̃J�[�h�͖���
// 1���O�ɋl�߂�
	for(int i = pos + 1; i < numcard; i++)
		cdat[i-1] = cdat[i];
	numcard--;

	return false;
}

//
// CardSet::remove() - ���g���� num �Ԃ̃J�[�h������(true: ���s; false: ����)
//
bool CardSet::remove(int number)
{
	int pos;

// �����J�[�h�̈ʒu�����߂�
	if((pos = locate(number)) < 0)
		return true;	// num �Ԃ̃J�[�h�͖���
// 1���O�ɋl�߂�
	for(int i = pos + 1; i < numcard; i++)
		cdat[i-1] = cdat[i];
	numcard--;

	return false;
}

//
// CardSet::print() - ���g�̏�Ԃ�W���o�͂ɏo�͂���
//
void CardSet::print(void)
{
	printf("((CARDSET))�_n");
	if(numcard == 0) {
		printf("�_tno card�_n");
		return;
	}
	for(int i = 0; i < numcard; i++) {
		printf("�_t");
		cdat[i].print();
		printf("�_n");
	}
}

}
