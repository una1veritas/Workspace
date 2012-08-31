//
// matrix22.cc - 2x2����(C++��)
//	���: (���ʤ���̾��); ����: (������������)
//
#include <stdio.h>
#include "matrix22.h"

//
// Matrix22::add() - ���Ȥι���ȹ��� m �Ȥ��¤����
//
Matrix22 Matrix22::add(Matrix22 m)
{
	Matrix22 ret;
	int i, j;

	for(i = 0; i < 2; i++)
		for(j = 0; j < 2; j++)
			ret.u[i][j] = u[i][j] + m.u[i][j];

	return ret;
}

//
// Matrix22::sub() - ���Ȥι���ȹ��� m �Ȥκ������
//
Matrix22 Matrix22::sub(Matrix22 m)
{
	Matrix22 ret;
	int i, j;

	for(i = 0; i < 2; i++)
		for(j = 0; j < 2; j++)
			ret.u[i][j] = u[i][j] - m.u[i][j];

	return ret;
}

//
// Matrix22::mult() - ���Ȥι���ȹ��� m �Ȥ��Ѥ����
//
Matrix22 Matrix22::mult(Matrix22 m)
{
	Matrix22 ret;
	int i, j, k;

	for(i = 0; i < 2; i++)
		for(j = 0; j < 2; j++) {
			ret.u[i][j] = 0;
			for(k = 0; k < 2; k++)
				ret.u[i][j] += u[i][k] * m.u[k][j];
		}

	return ret;
}

//
// Matrix22::scan() - ������ͤ�ɸ�����Ϥ��鼫�Ȥ����Ϥ���
//
void Matrix22::scan(void)
{
	int i, j;

	for(i = 0; i < 2; i++)
		for(j = 0; j < 2; j++)
			scanf("%lf", &u[i][j]);
}

//
// Matrix22::print() - ���Ȥι�����ͤ�ɸ����Ϥ˽��Ϥ���
//
void Matrix22::print(void)
{
	int i, j;

	for(i = 0; i < 2; i++) {
		printf("|\t");
		for(j = 0; j < 2; j++)
			printf("%f\t", u[i][j]);
		printf("|\n");
	}
}
