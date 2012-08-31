//
// cardset.h - �ȥ��ץ����ɤν��緿(C++��)
//	���: (���ʤ���̾��); ����: (������������)
//
#ifndef CARDSET_H
#define CARDSET_H

//
// Card - �ȥ��ץ����ɷ�
//
class Card {
	// ���饹���
	// �ȥ��פ���(suit)�Υ�����
	public:
	enum {
		SUIT_SPADE,
		SUIT_DIAMOND,
		SUIT_HEART,
		SUIT_CLUB,
		SUIT_JOKER
	};
	
// �����ѿ�
private:
	int suit;	// ��
	int number;	// �ֹ�
// ���дؿ�
public:
	// �ǥե���ȥ��󥹥ȥ饯��(���������)
	Card(void)	{ }
	// �Ȥ��ֹ�����ꤹ��
	void set(int st, int num) {
		suit = st;
		number = num; 
	}
	
	// ���Ȥ� tgt �Υ����ɤ��ȡ��ֹ椬��������Ƚ�� (true: Ʊ; false: ��)
	// �ǡ����Ȥ���Ʊ�����֥������Ȥ��ɤ����ǤϤʤ���
	bool equal(Card tgt) { 
		return (suit == tgt.suit) && (number == tgt.number); 
	}
	
	bool isValid() {
		if ( ((SUIT_SPADE <= suit) && (suit <= SUIT_CLUB)) 
			 && (1 <= number && (number <= 13)) )
			return true;
		else if (suit == SUIT_JOKER)
			return true;
		return false;
	}

	// ��������
	int gnumber(void) {
		return number;
	}
	
	int gsuit(void)	{
		return suit;
	}

	// ɸ����Ϥ��鼫�Ȥ����Ϥ���(true: ���顼; false: ���ｪλ)
	//bool scan(void);
	
	// ���ȥ꡼�फ�饫���ɤ��ͤ��ߡ����åȤ��롥���ϥ��顼�ΤȤ��ϡ�
	// isValid() �� false ���֤���
	friend std::istream& operator>>(std::istream& in, Card & c) {
		char* suitname[] = { "spade", "diamond", "heart", "club" };
		char buf[BUFSIZ];
		int num;
		in >> buf;
		in >> num;
		c.set(0,0);
		for(int s = c.SUIT_SPADE; s <= c.SUIT_CLUB; s++) {
			if(!strcmp(buf, suitname[s])) {
				c.set(s, num);
			}
		}
		return in;
	}
	
	// ���Ȥ��ͤ�ɸ����Ϥ˽��Ϥ���
	//void print(void);
	
	// ���ȥ꡼��˥����ɤ��ͤ�񤭽Ф���
	friend std::ostream& operator<<(std::ostream& out, const Card & c) { 
		char* suitname[] = { "spade", "diamond", "heart", "club" };
		out << "[";
		if (c.suit < Card::SUIT_JOKER)
				out << suitname[c.suit] << " " << c.number;
		else //if(suit == Card::SUIT_JOKER)
			out << "Joker";
		out << "]" ; 
		return out; 
	}
	
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
	//void print(void);
		// ���Ȥξ��֤�ɸ����Ϥ˽��Ϥ���
		
	// Streaming
	//
	friend std::ostream& operator<<(std::ostream& out, const CardSet & c);
	
};

#endif
