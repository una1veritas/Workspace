/* 
 * File:   Group3.h
 * Author: yuuya
 *
 * Created on 2015/05/17, 20:34
 */
#include "Player.h"

namespace grp2015 {

#define N 14
 
class Group3 : public Player {
  CardSet memory;
  CardSet trump;
 
 private:
  int pairlist[N]; 
  int cardlist[6][15];
  int discard[N];
  /* int maxNumber; */
 
 public:
 Group3(const char * name = "Little John") : Player(name) {
       
  }
 
  /*
   * �O���[�v�Ŏ��Ԃ��쐬���v�l������ǉ�����֐��D
   */
  // �Q�[�����n�߂�ɂ�����K�v�ȏ�����������D
  virtual void ready();
  // �J�[�h���o���v�l������g�ݍ��ށD
  virtual bool follow(const GameStatus &, CardSet &);
 
  virtual bool approve(const GameStatus &);
   
  /*
   *�@�v�l�����p�̊֐��Q
   */
  // ��D�̃\�[�e�B���O
  void HandOrder(void);
  // �y�A�̃t���O���Ǘ�����
  void PairFlag(void);
  // �y�A�̃t���O�z��̑���
  void PF_Sort(int);
  // �͈͓��̃J�[�h���ǂ���
  bool Field(int, int, int);
  // ����̎�D��5���ȉ��ɂȂ������Ƃ������t���O�i��p�؂�ւ��j
  bool Active(const GameStatus &);
  // ���m�F�p�A�̂ĎD�m�F�p�̔z��̏�����
  void InitArray(void);
  // �̂ĎD�̖����������m�F����
  void Check(void);
  // �W���[�J�[����ɏo�Ă��邩
  bool JokerFlag(void);

  bool HaveJoker(void);
  // ���g�̏o���J�[�h��苭���J�[�h���������邩
  int Stronger(int);
};

} // namespace
