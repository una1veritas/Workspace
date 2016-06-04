/* 
 * File:   Group3.cpp
 * Author: yuuya
 *
 * Created on 2015/05/17, 20:34
 */
#include <groups2015/Group3.h>
#include <iostream>
#include <string>
 
#include "Card.h"
#include "CardSet.h"
#include "Player.h"
 
using namespace grp2015;

// �Q�[�����n�߂�ɂ�����K�v�ȏ�����������D
void Group3::ready() {
  memory.makeEmpty(); //memory.clear();
  trump.makeEmpty(); //trump.clear();
  InitArray();
}
 
// �J�[�h���o���v�l������g�ݍ��ށD
bool Group3::follow(const GameStatus & gstat, CardSet & cards) {
  CardSet pile(gstat.pile);
  Card tmp;
  int i, count;
 
  cards.makeEmpty();        
  HandOrder();
  PairFlag();
  Check();
 
 
 
  //  std::cout << "( " << hand << " )" << std::endl; // �f�o�b�N�p�F�\��        
  // �f�o�b�N�p
  /*  printf("pair:( ");
  for(i = 0; i < N; i++)
    printf("%2d", pairlist[i]);
  printf(" )\n");
         
  printf("discard:( ");
  for(i = 0; i < N; i++)
    printf("%2d", discard[i]);
  printf(" )\n");
  */
     
  // �̂ĎD�I���̏���
  switch(pile.size()) {
    // �ŏ��̃^�[���̂Ƃ�
  case 0:
    // �D�揇��: 1�� > 2�� > ����ȏ�̖����@�Ŏ̂Ă�
    // 1��
    for(i = 0; i < hand.size(); i++) {
      if(pairlist[i] == 1) {
	if(Field(hand[i].getNumber(), 3, 8)) {
	  PF_Sort(i);
	  hand.pickup(tmp, i);
	  cardlist[tmp.getSuit()][tmp.getNumber()] = getID() + 1;
	  cards.insert(tmp);
	  return true;
	}
      }
    }
    // 2��
    for(i = 0; i < hand.size(); i++) {
      if(pairlist[i] == 2 && Field(hand[i].getNumber(), 3, 8)) {
	count = pairlist[i] - 1;
	while(count >= 0) {
	  PF_Sort(i + count);
	  hand.pickup(tmp, i + count);
	  cardlist[tmp.getSuit()][tmp.getNumber()] = getID() + 1;
	  cards.insert(tmp);
	  count--;
	}
	return true;
      }
    }
    // ����ȏ�̖���
    for(i = 0; i < hand.size(); i++) {
      if(pairlist[i] > 2 && Field(hand[i].getNumber(), 3, 8)) {
	count = pairlist[i] - 1;
	while(count >= 0) {
	  PF_Sort(i + count);
	  hand.pickup(tmp, i + count);
	  cardlist[tmp.getSuit()][tmp.getNumber()] = getID() + 1;
	  cards.insert(tmp);
	  count--;
	}
	return true;
      }
    }
                 
    // ������9�ȏ�̂Ƃ�
    count = pairlist[0] - 1;
    while(count >= 0) {
      PF_Sort(0 + count);
      hand.pickup(tmp, 0 + count);
      cardlist[tmp.getSuit()][tmp.getNumber()] = getID() + 1;
      cards.insert(tmp);
      count--;
    }
    return true;
                 
    // 1���o���̂Ƃ�
  case 1:
    for(i = 0; i < hand.size(); i++)
      if(cardGreaterThan(hand[i],pile[0]) && pairlist[i] == 1) {
	if(Stronger(i) >= hand.size() - 6) {
	  PF_Sort(i);
	  hand.pickup(tmp, i);
	  cardlist[tmp.getSuit()][tmp.getNumber()] = getID() + 1;
	  cards.insert(tmp);
	  return true;
	}
	else {
	  if(hand.size() < 6 || Active(gstat)) {
	    if(HaveJoker() && hand.size() < 5) {
	      PF_Sort(hand.size() -1);
	      hand.pickup(tmp, hand.size() - 1);
	      cardlist[tmp.getSuit()][tmp.getNumber()] = getID() + 1;
	      cards.insert(tmp);
	      return true;
	    }
	    else {
	      PF_Sort(i);
	      hand.pickup(tmp, i);
	      cardlist[tmp.getSuit()][tmp.getNumber()] = getID() + 1;
	      cards.insert(tmp);
	      return true;
	    }
	  }
	}
      }
    return false;
                 
    // �������̂Ƃ�
  default:
    for(i = 0; i <= hand.size() - pile.size(); i++) {
      if(cardGreaterThan(hand[i],pile[0]) && pairlist[i] == pile.size()) {
	if(Stronger(i) >= hand.size() - 6) {
	  count = pile.size() - 1;
	  while(count >= 0) {
	    PF_Sort(i + count);
	    hand.pickup(tmp, i + count);
	    cardlist[tmp.getSuit()][tmp.getNumber()] = getID() + 1;
	    cards.insert(tmp);
	    count--;
	  }
	  return true;
	}
	else { 
	  if(hand.size() < 6 || Active(gstat)) {
	    if(hand[i].getNumber() == 2 && hand.size() - pairlist[i] == 1) {
	      count = pile.size() - 1;
	      while(count >= 0) {
		PF_Sort(i + count);
		hand.pickup(tmp, i + count);
		cardlist[tmp.getSuit()][tmp.getNumber()] = getID() + 1;
		cards.insert(tmp);
		count--;
	      }
	      return true;
	    }
	    count = pile.size() - 1;
	    while(count >= 0) {
	      PF_Sort(i + count);
	      hand.pickup(tmp, i + count);
	      cardlist[tmp.getSuit()][tmp.getNumber()] = getID() + 1;
	      cards.insert(tmp);
	      count--;
	    }
	    return true;
	  }
	}
      }
    }
    return false;
  } 
  return false;
}
 
// �̂Ă��J�[�h�Z�b�g���̏󋵊m�F������
bool Group3::approve(const GameStatus & gstat) {
  CardSet pile(gstat.pile);
  Card c;
  int pilesize = pile.size();
  if(pile.isEmpty() != 1)
    for(int i = 0; i < pilesize; i++){
      pile.pickup(c, 0);
      cardlist[c.getSuit()][c.getNumber()] = gstat.leaderIndex + 1;
    }
  return true;
}
 
 
/*
 *�@�v�l�����p�̊֐��Q
 */
// ��D�̃\�[�e�B���O (�ア���ɕ��ёւ�)
void Group3::HandOrder(void) {
  int i, j;
  Card tmp;
  for(i = 0; i < hand.size() - 1; i++)
    for(j = i + 1; j <hand.size(); j++)
      if(cardGreaterThan(hand[i],hand[j])) {
	tmp = hand[i];
	hand[i] = hand[j];
	hand[j] = tmp;
      }
}
 
// �y�A�̃t���O���Ǘ�����
void Group3::PairFlag(void) {
  int i, j, count = 1;
  for(i = 0; i < hand.size(); ) {
    for(j = i + 1; j < hand.size(); j++) {
      if(hand[i].getNumber() == hand[j].getNumber())
	count++;
      else
	break;
    }
    for(j = 0; j < count; j++)
      pairlist[i + j] = count;
    i = i + count;
    count = 1;
  }
}
 
// �y�A�̃t���O�z��̑���i�K�v�Ȃ��z��̕�����0�������j
void Group3::PF_Sort(int pos){
  int i;
  for(i = pos; i < hand.size() - 1; i++)
    pairlist[i] = pairlist[i + 1];
  for(i = hand.size() - 1; i < N; i++)
    pairlist[i] = 0;
}
 
// lower <= num <= upper �����藧��
bool Group3::Field(int num, int lower, int upper) {
  if((lower + 10) % 13 <= (num + 10) % 13 && (num + 10) % 13 <= (upper + 10) % 13)
    return true;
  return false;
}
 
// ����̎�D��5���ȉ��ɂȂ������Ƃ������t���O�i��p�؂�ւ��j
bool Group3::Active(const GameStatus & gstat) {
  int i;
  for(i = 0; i < gstat.numPlayers ; i++) {
    if(gstat.numCards[i] <= 5)
      return true;
  }
  return false;
}
 
// ���m�F�p�A�̂ĎD�m�F�p�̔z��̏�����
void Group3::InitArray(void) {
  int i, j;
  for(i = 0; i < N; i++) {
    pairlist[i] = 0;
    discard[i] = 0;
  }
  for(i = 0; i < 6; i++)
    for(j = 0; j < 15; j++)
      cardlist[i][j] = 0;
}
 
// �̂ĎD�̖����������m�F����
void Group3::Check(void) {
  int i, j;
  // �d�����Đ����邽�ߏ��������Ă���l����꒼��
  for(i = 0; i < N; i++)
    discard[i] = 0;
  // �ē���
  for(i = 0; i < 6; i++)
    for(j = 0; j < 15; j++)
      if(cardlist[i][j] != 0)
	discard[j]++;
}
 
// �W���[�J�[����ɏo�Ă��邩�itrue: �o�Ă��Ȃ� false: �o�Ă���j
bool Group3::JokerFlag(void) {
  if(cardlist[4][0] == 0)
    return true;
  else
    return false;
}
 
// �������W���[�J�[�������Ă��邩
bool Group3::HaveJoker(void) {
  if(hand[hand.size() - 1].getNumber() == 0)
    return true;
  return false;
}
// ���g�̏o���J�[�h��苭���J�[�h���������邩
int Group3::Stronger(int pos) {
  int i, counter = 0;
  for(i = pos + 1; i < hand.size(); i++)
    if(cardGreaterThan(hand[i], hand[pos]))
      counter++;
  return counter;
}
