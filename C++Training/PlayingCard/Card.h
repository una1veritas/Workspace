//
// Card - �g�����v�J�[�h�^
//

class Card {
	// �����o�ϐ�
	private:
		int suit;	// �g
		int rank;	// �ԍ�
		
	// �����o�֐�
	public:
	// �g�����v�̑g(suit)�̃R�[�h
	enum SUIT {
		SUIT_SPADE,
		SUIT_DIAMOND,
		SUIT_HEART,
		SUIT_CLUB,
		SUIT_JOKER
	};

	Card(void)	{ return; }
		// �f�t�H���g�R���X�g���N�^(�����l�s��)

	void set(int s, int r)
		{ suit = s; rank = r; return; }
		// ���g�Ɏw�肵���g�Ɣԍ�������

	bool equal(Card tgt)
		{ return suit == tgt.suit && rank == tgt.rank; }
		// ���g�� tgt ���������ۂ��̔��� (true: ��; false: ��)

	int getrank(void)	{ return rank; }

	int getsuit(void)		{ return suit; }
		// �A�N�Z�T

	bool isJoker() { return suit == SUIT_JOKER; }
	bool isGreaterThan(Card c);

	bool scan(void);
		// �W���o�͂��玩�g�ɓ��͂���(true: �G���[; false: ����I��)

	void print(void);
		// ���g�̒l��W���o�͂ɏo�͂���
	std::string printString() const;
};
