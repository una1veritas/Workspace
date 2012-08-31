/*
 * vector2c.c - 2次元ベクトル型(C版)
 *	作者: (あなたの名前); 日付: (完成した日付)
 */
#include <stdio.h>
#include "vector2c.h"

/*
 * constv2() - x0, y0 を与えて，定数 (x0, y0) を得る
 */
struct vector2 constv2(double x0, double y0)
{
	struct vector2 ret;

	ret.x = x0;
	ret.y = y0;

	return ret;
}

/*
 * addv2() - ベクトル u と v の和 u + v を求める
 */
struct vector2 addv2(struct vector2 u, struct vector2 v)
{
	struct vector2 ret;

	ret.x = u.x + v.x;
	ret.y = u.y + v.y;

	return ret;
}

/*
 * subv2() - ベクトル u と v の差 u - v を求める
 */
struct vector2 subv2(struct vector2 u, struct vector2 v)
{
	struct vector2 ret;

	ret.x = u.x - v.x;
	ret.y = u.y - v.y;

	return ret;
}

/*
 * scanv2() - ベクトルの値を標準入力から *pu に入力する
 */
void scanv2(struct vector2* pu)
{
	scanf("%lf %lf", &pu->x, &pu->y);
}

/*
 * printv2() - ベクトル u の値を標準出力に出力する
 */
void printv2(struct vector2 u)
{
	printf("( %f %f )", u.x, u.y);
}
