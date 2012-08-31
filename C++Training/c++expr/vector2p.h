//
// vector2p.h - 2�����٥��ȥ뷿(C++��)
//	���: (���ʤ���̾��); ����: (������������)
//
#ifndef VECTOR2P_H
#define VECTOR2P_H

//
// Vector2 - 2�����٥��ȥ뷿
//
class Vector2 {
// �����ѿ�
private:
	double x;	// x��
	double y;	// y��
// ���дؿ�
public:
	Vector2(void)	{ }
		// �ǥե���ȥ��󥹥ȥ饯��(���������)
	Vector2(double x0, double y0);
		// x0, y0 ��Ϳ������� (x0, y0) �����륳�󥹥ȥ饯��
	Vector2 add(Vector2 u);
		// ���ȤΥ٥��ȥ�ȥ٥��ȥ� u �Ȥ��¤����
	Vector2 sub(Vector2 u);
		// ���ȤΥ٥��ȥ�ȥ٥��ȥ� u �Ȥκ������
	void scan(void);
		// �٥��ȥ���ͤ�ɸ�����Ϥ��鼫�Ȥ����Ϥ���
	void print(void);
		// ���ȤΥ٥��ȥ���ͤ�ɸ����Ϥ˽��Ϥ���
};

#endif
