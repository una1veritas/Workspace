/*
 * vector2c.c - 2�����٥��ȥ뷿(C��)
 *	���: (���ʤ���̾��); ����: (������������)
 */
#include <stdio.h>
#include "vector2c.h"

/*
 * constv2() - x0, y0 ��Ϳ���ơ���� (x0, y0) ������
 */
struct vector2 constv2(double x0, double y0)
{
	struct vector2 ret;

	ret.x = x0;
	ret.y = y0;

	return ret;
}

/*
 * addv2() - �٥��ȥ� u �� v ���� u + v �����
 */
struct vector2 addv2(struct vector2 u, struct vector2 v)
{
	struct vector2 ret;

	ret.x = u.x + v.x;
	ret.y = u.y + v.y;

	return ret;
}

/*
 * subv2() - �٥��ȥ� u �� v �κ� u - v �����
 */
struct vector2 subv2(struct vector2 u, struct vector2 v)
{
	struct vector2 ret;

	ret.x = u.x - v.x;
	ret.y = u.y - v.y;

	return ret;
}

/*
 * scanv2() - �٥��ȥ���ͤ�ɸ�����Ϥ��� *pu �����Ϥ���
 */
void scanv2(struct vector2* pu)
{
	scanf("%lf %lf", &pu->x, &pu->y);
}

/*
 * printv2() - �٥��ȥ� u ���ͤ�ɸ����Ϥ˽��Ϥ���
 */
void printv2(struct vector2 u)
{
	printf("( %f %f )", u.x, u.y);
}
