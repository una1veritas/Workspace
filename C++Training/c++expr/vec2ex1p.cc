//
// vec2ex1p.cc - 2�����٥��ȥ뷿(C++��)�ƥ��ȥץ����
//	���: (���ʤ���̾��); ����: (������������)
//
#include <stdio.h>
#include "vector2p.h"

//
// main() - �ƥ��ȥץ����
//
int main(void)
{
	Vector2 a, b;

// a, b ������
	printf("a = ? ");
	a.scan();
	printf("b = ? ");
	b.scan();
// a + b �����
	printf("a + b = ");
	(a.add(b)).print();
	printf("\n");;
// a - b �����
	printf("a - b = ");
	(a.sub(b)).print();
	printf("\n");;
// a + (2, 1) �����
	printf("a + (2,1) = ");
	(a.add(Vector2(2, 1))).print();
	printf("\n");
// (5, 5) - b �����
	printf("(5,5) - b = ");
	(Vector2(5, 5).sub(b)).print();
	printf("\n");

	return 0;
}
