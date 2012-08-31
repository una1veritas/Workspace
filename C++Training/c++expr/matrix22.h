//
// matrix22.h - 2x2行列型(C++版)
//	作者: (あなたの名前); 日付: (完成した日付)
//
#ifndef MATRIX22_H
#define MATRIX22_H

//
// Matrix22 - 2x2行列型
//
class Matrix22 {
// メンバ変数
private:
	double u[2][2];	// 成分
// メンバ関数
public:
	Matrix22(void)	{ }
		// デフォルトコンストラクタ(初期値不定)
	Matrix22(double m00, double m01, double m10, double m11)
		{ u[0][0] = m00; u[0][1] = m01; u[1][0] = m10; u[1][1] = m11; }
		// 各成分を与えて定数行列を得るコンストラクタ
	Matrix22 add(Matrix22 m);
		// 自身の行列と行列 m との和を求める
	Matrix22 sub(Matrix22 m);
		// 自身の行列と行列 m との差を求める
	Matrix22 mult(Matrix22 m);
		// 自身の行列と行列 m との積を求める
	void scan(void);
		// 行列の値を標準入力から自身に入力する
	void print(void);
		// 自身の行列の値を標準出力に出力する
};

#endif
