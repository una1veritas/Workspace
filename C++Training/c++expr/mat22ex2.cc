//
// mat22ex1.cc - 2x2����(C++��)�ƥ��ȥץ����
//	���: (���ʤ���̾��); ����: (������������)
//
#include <stdio.h>
#include "matrix22.h"
#include "vector2p.h"

//
// main() - �ƥ��ȥץ����
//
int main(void)
{
	Matrix22 a, b, e;

// a, b ������
	printf("a = ? ");
	a.scan();
	printf("b = ? ");
	b.scan();
// a + b �����
	printf("a + b = \n");
	(a.add(b)).print();
	printf("\n");;
// a - b �����
	printf("a - b = \n");
	(a.sub(b)).print();
	printf("\n");;
// a * b �����
	printf("a * b = \n");
	(a.mult(b)).print();
	printf("\n");
// a + ñ�̹��� �����
	e = Matrix22(1, 0, 0, 1);
	printf("a + E = \n");
	(a.add(e)).print();
	printf("\n");;

	Vector2 u, v;
	
	// a, b ������
	printf("u = ? ");
	u.scan();
	printf("v = ? ");
	v.scan();
	// a + b �����
	printf("u + v = ");
	(u.add(v)).print();
	printf("\n");
	
	return 0;
}
