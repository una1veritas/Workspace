//
// matrix22.h - 2x2����(C++��)
//	���: (���ʤ���̾��); ����: (������������)
//
#ifndef MATRIX22_H
#define MATRIX22_H

//
// Matrix22 - 2x2����
//
class Matrix22 {
// �����ѿ�
private:
	double u[2][2];	// ��ʬ
// ���дؿ�
public:
	Matrix22(void)	{ }
		// �ǥե���ȥ��󥹥ȥ饯��(���������)
	Matrix22(double m00, double m01, double m10, double m11)
		{ u[0][0] = m00; u[0][1] = m01; u[1][0] = m10; u[1][1] = m11; }
		// ����ʬ��Ϳ���������������륳�󥹥ȥ饯��
	Matrix22 add(Matrix22 m);
		// ���Ȥι���ȹ��� m �Ȥ��¤����
	Matrix22 sub(Matrix22 m);
		// ���Ȥι���ȹ��� m �Ȥκ������
	Matrix22 mult(Matrix22 m);
		// ���Ȥι���ȹ��� m �Ȥ��Ѥ����
	void scan(void);
		// ������ͤ�ɸ�����Ϥ��鼫�Ȥ����Ϥ���
	void print(void);
		// ���Ȥι�����ͤ�ɸ����Ϥ˽��Ϥ���
};

#endif
