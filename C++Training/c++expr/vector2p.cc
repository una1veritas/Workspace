//
// vector2p.cc - 2�����٥��ȥ뷿(C++��)
//	���: (���ʤ���̾��); ����: (������������)
//
#include <stdio.h>
#include "vector2p.h"

//
// Vector2::Vector2() - x0, y0 ��Ϳ������� (x0, y0) �����륳�󥹥ȥ饯��
//
Vector2::Vector2(double x0, double y0)
{
	x = x0;
	y = y0;
}

//
// Vector2::add() - ���ȤΥ٥��ȥ�ȥ٥��ȥ� u �Ȥ��¤����
//
Vector2 Vector2::add(Vector2 u)
{
	Vector2 ret;

	ret.x = x + u.x;
	ret.y = y + u.y;

	return ret;
}

//
// Vector2::sub() - ���ȤΥ٥��ȥ�ȥ٥��ȥ� u �Ȥκ������
//
Vector2 Vector2::sub(Vector2 u)
{
	Vector2 ret;

	ret.x = x - u.x;
	ret.y = y - u.y;

	return ret;
}

//
// Vector2::scan() - �٥��ȥ���ͤ�ɸ�����Ϥ��鼫�Ȥ����Ϥ���
//
void Vector2::scan(void)
{
	scanf("%lf %lf", &x, &y);
}

//
// Vector2::print() - ���ȤΥ٥��ȥ���ͤ�ɸ����Ϥ˽��Ϥ���
//
void Vector2::print(void)
{
	printf("( %f %f )", x, y);
}
