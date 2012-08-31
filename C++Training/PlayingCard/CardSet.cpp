//
// cardset.cc - �g�����v�J�[�h�̏W���^(C++��)
//	���: (���Ȃ��̖��O); ���t: (�����������t)
//
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string>

#include "Card.h"
#include "CardSet.h"


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
		if(number == cdat[i].getrank())
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

	for(suit = c.SUIT_SPADE; suit <=c.SUIT_CLUB; suit++)
		for(num = 1; num <= 13; num++) {
			c.set(suit, num);
			insert(c);
		}
	c.set(c.SUIT_JOKER, 0);
	insert(c);
}


Card CardSet::at(int i) {
	return cdat[i];
}

Card CardSet::operator[](int i) {
	return at(i);
}


//
// CardSet::pickup() - ���g���� targetpos ���ڂ̃J�[�h������ *ret �ɕԂ�
//	targetpos �� -1 �̂Ƃ��̓����_���ɑI��(true: ����; false: ���s)
//
bool CardSet::pickup(Card* ret, int targetpos /* = -1 */)
{
	if(numcard == 0)
		return false;
	if(targetpos < 0)
		targetpos = random() % numcard;
	else
		targetpos %= numcard;

	*ret = cdat[targetpos];
	remove(*ret);

	return true;
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

bool CardSet::insert(CardSet & cards) {
	for(int i = 0; i < cards.numcard; i++) {
		insert(cards.cdat[i]);
	}
	return true;
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
	//printf("((CARDSET))\n\t");
	if(numcard == 0) {
		printf("no cards");
		return;
	}
	for(int i = 0; i < numcard; i++) {
		//printf("\t");
		cdat[i].print();
		printf(", ");
	}
	printf("\n");
}

std::string CardSet::printString() const {
	std::string tmp;
	if(numcard == 0) {
		tmp += "(no card)";
		return tmp;
	}
	for(int i = 0; i < numcard; i++) {
		tmp += cdat[i].printString();
		tmp += " ";
	}
	return tmp;
}


  void CardSet::shuffle(void) {
    int t, i, j;
    Card swap;
    time_t seed;
    
    time(&seed);
    srandom(seed);
    for (t = 0; t < 100; t++) {
      i = random() % numcard;
      j = ((random() % (numcard-1)) + 1 + i) % numcard;
      swap = cdat[i];
      cdat[i] = cdat[j];
      cdat[j] = swap;
    }
  }
