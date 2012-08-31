/*
 * vector2c.h - 2次元ベクトル型(C版)
 *	作者: (あなたの名前); 日付: (完成した日付)
 */
#ifndef VECTOR2C_H
#define VECTOR2C_H

/* データ定義・宣言 */
struct vector2 { /* 2次元ベクトル型 */
	double x;	/* x値 */
	double y;	/* y値 */
};

/* プロトタイプ宣言 */
struct vector2 constv2(double x0, double y0);
	/* x0, y0 を与えて，定数 (x0, y0) を得る */
struct vector2 addv2(struct vector2 u, struct vector2 v);
	/* ベクトル u と v の和 u + v を求める */
struct vector2 subv2(struct vector2 u, struct vector2 v);
	/* ベクトル u と v の差 u - v を求める */
void scanv2(struct vector2* pu);
	/* ベクトルの値を標準入力から *pu に入力する */
void printv2(struct vector2 u);
	/* ベクトル u の値を標準出力に出力する */

#endif
