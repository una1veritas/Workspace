//
// cardset.h - �g�����v�J�[�h�̏W���^(C++��)
//	���: (���Ȃ��̖��O); ���t: (�����������t)
//
#ifndef CARDSET_H
#define CARDSET_H
//
// CardSet - �g�����v�J�[�h�̏W���^
//
class CardSet {
// ��`�E�錾
public:
	const static int maxnumcard = 53;	// �J�[�h����
// �����o�ϐ�
private:
	int numcard;		// ���݂̏W�����̃J�[�h��
	Card cdat[maxnumcard];	// �J�[�h�̃f�[�^
// �����o�֐�
private:
	int locate(Card target);
		// �����ł� target �̃J�[�h�̈ʒu��Ԃ�(-1: �Ȃ�)
	int locate(int num);
		// �����ł� num �Ԃ̃J�[�h�̈ʒu��Ԃ�(-1: �Ȃ�)
public:
	CardSet(void)		{ makeempty(); }
		// �f�t�H���g�R���X�g���N�^(�����l��W��)
	void makeempty(void)	{ numcard = 0 ; }
		// ���g����W���ɂ���
	bool isempty(void)	{ return numcard == 0; }
		// ���g����W�����ۂ��̔��� (true: ��; false: ���)
	int size() { return numcard; }
	
	Card at(int);
	Card operator[](int);
	
	void makedeck(void);
		// ���g�ɑS����(maxnumcard ����)�J�[�h������
	bool pickup(Card* ret, int targetpos = -1);
		// ���g���� targetpos ���ڂ̃J�[�h������ *ret �ɕԂ�
		// targetpos �� -1 �̂Ƃ��̓����_���ɑI��
		// (true: ���s; false: ����)
	bool insert(Card newcard);
		// ���g�� newcard �̃J�[�h������(true: ���s; false: ����)
	bool insert(CardSet & cards);

	bool remove(Card target);
		// ���g���� target �̃J�[�h������(true: ���s; false: ����)
	bool remove(int num);
		// ���g���� num �Ԃ̃J�[�h������(true: ���s; false: ����)
	void print(void);
		// ���g�̏�Ԃ�W���o�͂ɏo�͂���
	std::string printString() const;
	
	void shuffle(void);
	  
	// Streaming
	//
	friend std::ostream& operator<<(std::ostream& out, const CardSet & c) {
		out << c.printString();
		return out;
	}
};

#endif
