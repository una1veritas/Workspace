/*
 * vector2c.h - 2�����٥��ȥ뷿(C��)
 *	���: (���ʤ���̾��); ����: (������������)
 */
#ifndef VECTOR2C_H
#define VECTOR2C_H

/* �ǡ����������� */
struct vector2 { /* 2�����٥��ȥ뷿 */
	double x;	/* x�� */
	double y;	/* y�� */
};

/* �ץ�ȥ�������� */
struct vector2 constv2(double x0, double y0);
	/* x0, y0 ��Ϳ���ơ���� (x0, y0) ������ */
struct vector2 addv2(struct vector2 u, struct vector2 v);
	/* �٥��ȥ� u �� v ���� u + v ����� */
struct vector2 subv2(struct vector2 u, struct vector2 v);
	/* �٥��ȥ� u �� v �κ� u - v ����� */
void scanv2(struct vector2* pu);
	/* �٥��ȥ���ͤ�ɸ�����Ϥ��� *pu �����Ϥ��� */
void printv2(struct vector2 u);
	/* �٥��ȥ� u ���ͤ�ɸ����Ϥ˽��Ϥ��� */

#endif
