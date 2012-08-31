//
// cardset.cc - �ȥ��ץ����ɤν��緿(C++��)
//	���: (���ʤ���̾��); ����: (������������)
//
#include <stdlib.h>
#include <stdio.h>
#include "cardset.h"


//
// Card::scan() - ɸ����Ϥ��鼫�Ȥ����Ϥ���(true: ���顼; false: ���ｪλ)
//
bool Card::scan(void)
{
	char buf[BUFSIZ];
	char* suitname[] = { "spade", "diamond", "heart", "club" };
// 4�ȤΤ����줫�ʤ��ֹ�����Ϥ���
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
// joker �Ϥ��Τޤ�(number �� 0 �Ȥ���)
	if(!strcmp(buf, "joker")) {
		suit = SUIT_JOKER;
		number = 0;
		return false;
	}

	return true;	// ���顼
}

//
// Card::print() - ���Ȥ��ͤ�ɸ����Ϥ˽��Ϥ���
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
// CardSet::locate() - �����Ǥ� target �Υ����ɤΰ��֤��֤�(-1: �ʤ�)
//
int CardSet::locate(Card target)
{
	for(int i = 0; i < numcard; i++)
		if(target.equal(cdat[i]))
			return i;

	return -1;	// ���Ĥ���ʤ��ä�
}

//
// CardSet::locate() - �����Ǥ� num �֤Υ����ɤΰ��֤��֤�(-1: �ʤ�)
//
int CardSet::locate(int number)
{
	for(int i = 0; i < numcard; i++)
		if(number == cdat[i].gnumber())
			return i;

	return -1;	// ���Ĥ���ʤ��ä�
}

//
// CardSet::makedeck() - ���Ȥ�������(maxnumcard ���)�����ɤ������
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
// CardSet::pickup() - ���Ȥ��� targetpos ���ܤΥ����ɤ���� *ret ���֤�
//	targetpos �� -1 �ΤȤ��ϥ����������(true: ����; false: ����)
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
// CardSet::insert() - ���Ȥ� newcard �Υ����ɤ������(true: ����; false: ����)
//
bool CardSet::insert(Card newcard)
{
	if(locate(newcard) >= 0)
		return true;	// ���ˤ���
// �Ǹ���ɲ�
	cdat[numcard] = newcard;
	numcard++;

	return false;
}

//
// CardSet::remove() - ���Ȥ��� target �Υ����ɤ����(true: ����; false: ����)
//
bool CardSet::remove(Card target)
{
	int pos;

// ���������ɤΰ��֤����
	if((pos = locate(target)) < 0)
		return true;	// target �Υ����ɤ�̵��
// 1�Ĥ������˵ͤ��
	for(int i = pos + 1; i < numcard; i++)
		cdat[i-1] = cdat[i];
	numcard--;

	return false;
}

//
// CardSet::remove() - ���Ȥ��� num �֤Υ����ɤ����(true: ����; false: ����)
//
bool CardSet::remove(int number)
{
	int pos;

// ���������ɤΰ��֤����
	if((pos = locate(number)) < 0)
		return true;	// num �֤Υ����ɤ�̵��
// 1�Ĥ������˵ͤ��
	for(int i = pos + 1; i < numcard; i++)
		cdat[i-1] = cdat[i];
	numcard--;

	return false;
}

//
// CardSet::print() - ���Ȥξ��֤�ɸ����Ϥ˽��Ϥ���
//
void CardSet::print(void)
{
	printf("((CARDSET))\n");
	if(numcard == 0) {
		printf("\tno card\n");
		return;
	}
	for(int i = 0; i < numcard; i++) {
		printf("\t");
		cdat[i].print();
		printf("\n");
	}
}

}
