/*
 * vec2ex1c.c - 2�����٥��ȥ뷿(C��)�ƥ��ȥץ����
 *	���: (���ʤ���̾��); ����: (������������)
 */
#include <stdio.h>
#include "vector2c.h"

/*
 * main() - �ƥ��ȥץ����
 */
int main(void)
{
	struct vector2 a, b;

/* a, b ������ */
	printf("a = ? ");
	scanv2(&a);
	printf("b = ? ");
	scanv2(&b);
/* a + b ����� */
	printf("a + b = ");
	printv2(addv2(a, b));
	printf("\n");;
/* a - b ����� */
	printf("a - b = ");
	printv2(subv2(a, b));
	printf("\n");;
/* a + (2, 1) ����� */
	printf("a + (2,1) = ");
	printv2(addv2(a, constv2(2, 1)));
	printf("\n");
/* (5, 5) - b ����� */
	printf("(5,5) - b = ");
	printv2(subv2(constv2(5, 5), b));
	printf("\n");

	return 0;
}
