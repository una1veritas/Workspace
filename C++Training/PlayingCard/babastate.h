//
// babastate.h - �Х�ȴ���ξ��ַ�(C++��)
//	���: (���ʤ���̾��); ����: (������������)
//
#ifndef BABASTATE_H
#define BABASTATE_H

//#include "cardset.h"

//
// BabaState - �Х�ȴ���ξ��ַ�
//
class BabaState {
// ��������
public:
	const int numplayer = 5;	// �ץ졼���
// �����ѿ�
private:
	bool finished[numplayer]; // �ƥץ졼��ν�λ���ݤ��Υե饰
	CardSet hand[numplayer];  // �ƥץ졼��λ�����
// ���дؿ�
public:
	BabaState(void)	{ reset(); }
		// �ǥե���ȥ��󥹥ȥ饯��(����ͤϥ����ɤ��ۤä�����)
	void reset(void);
		// �ꥻ�å�(�ǽ�˥����ɤ��ۤä����֤ˤ���)
	bool isfinished(int plr)	{ return finished[plr]; }
		// plr �֤Υץ졼�䤬�夬�ä�(��λ����)���ݤ���Ƚ��
		// (true: �夬�ä�; false: ̤λ)
	bool move(int from, int to);
		// to �֤Υץ졼�䤬 from �֤Υץ졼�䤫�饫���ɤ���
		// (true: ����; false: ����)
	void print(void);
		// ���Ȥξ��֤�ɸ����Ϥ˽��Ϥ���
};

#endif
