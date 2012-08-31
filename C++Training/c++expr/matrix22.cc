//
// matrix22.cc - 2x2行列型(C++版)
//	作者: (あなたの名前); 日付: (完成した日付)
//
#include <stdio.h>
#include "matrix22.h"

//
// Matrix22::add() - 自身の行列と行列 m との和を求める
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
// Matrix22::sub() - 自身の行列と行列 m との差を求める
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
// Matrix22::mult() - 自身の行列と行列 m との積を求める
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
// Matrix22::scan() - 行列の値を標準入力から自身に入力する
//
void Matrix22::scan(void)
{
	int i, j;

	for(i = 0; i < 2; i++)
		for(j = 0; j < 2; j++)
			scanf("%lf", &u[i][j]);
}

//
// Matrix22::print() - 自身の行列の値を標準出力に出力する
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
