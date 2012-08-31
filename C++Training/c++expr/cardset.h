//
// cardset.h - �ȥ��ץ����ɤν��緿(C++��)
//	���: (���ʤ���̾��); ����: (������������)
//
#ifndef CARDSET_H
#define CARDSET_H

// �ȥ��פ���(suit)�Υ�����
enum {
	SUIT_SPADE,
	SUIT_DIAMOND,
	SUIT_HEART,
	SUIT_CLUB,
	SUIT_JOKER
};

//
// Card - �ȥ��ץ����ɷ�
//
class Card {
// �����ѿ�
private:
	int suit;	// ��
	int number;	// �ֹ�
// ���дؿ�
public:
	Card(void)	{ }
		// �ǥե���ȥ��󥹥ȥ饯��(���������)
	void set(int st, int num)	{ suit = st; number = num; }
		// ���Ȥ˻��ꤷ���Ȥ��ֹ�������
	bool equal(Card tgt)
		{ return suit == tgt.suit && number == tgt.number; }
		// ���Ȥ� tgt ��Ʊ�����ݤ���Ƚ�� (true: Ʊ; false: ��)
	int gnumber(void)	{ return number; }
	int gsuit(void)		{ return suit; }
		// ��������
	bool scan(void);
		// ɸ����Ϥ��鼫�Ȥ����Ϥ���(true: ���顼; false: ���ｪλ)
	void print(void);
		// ���Ȥ��ͤ�ɸ����Ϥ˽��Ϥ���
};

//
// CardSet - �ȥ��ץ����ɤν��緿
//
class CardSet {
// ��������
public:
	const static int maxnumcard = 53;	// ���������
// �����ѿ�
private:
	int numcard;		// ���ߤν�����Υ����ɿ�
	Card cdat[maxnumcard];	// �����ɤΥǡ���
// ���дؿ�
private:
	int locate(Card target);
		// �����Ǥ� target �Υ����ɤΰ��֤��֤�(-1: �ʤ�)
	int locate(int num);
		// �����Ǥ� num �֤Υ����ɤΰ��֤��֤�(-1: �ʤ�)
public:
	CardSet(void)		{ makeempty(); }
		// �ǥե���ȥ��󥹥ȥ饯��(����Ͷ�����)
	void makeempty(void)	{ numcard = 0 ; }
		// ���Ȥ������ˤ���
	bool isempty(void)	{ return numcard == 0; }
		// ���Ȥ������礫�ݤ���Ƚ�� (true: ��; false: ���)
	void makedeck(void);
		// ���Ȥ�������(maxnumcard ���)�����ɤ������
	bool pickup(Card* ret, int targetpos = -1);
		// ���Ȥ��� targetpos ���ܤΥ����ɤ���� *ret ���֤�
		// targetpos �� -1 �ΤȤ��ϥ����������
		// (true: ����; false: ����)
	bool insert(Card newcard);
		// ���Ȥ� newcard �Υ����ɤ������(true: ����; false: ����)
	bool remove(Card target);
		// ���Ȥ��� target �Υ����ɤ����(true: ����; false: ����)
	bool remove(int num);
		// ���Ȥ��� num �֤Υ����ɤ����(true: ����; false: ����)
	void print(void);
		// ���Ȥξ��֤�ɸ����Ϥ˽��Ϥ���
		
	// Streaming
	//
	friend std::ostream& operator<<(std::ostream& out, const CardSet & c);
	
};

#endif
