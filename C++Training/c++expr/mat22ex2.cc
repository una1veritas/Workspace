//
// mat22ex1.cc - 2x2行列型(C++版)テストプログラム
//	作者: (あなたの名前); 日付: (完成した日付)
//
#include <stdio.h>
#include "matrix22.h"
#include "vector2p.h"

//
// main() - テストプログラム
//
int main(void)
{
	Matrix22 a, b, e;

// a, b を入力
	printf("a = ? ");
	a.scan();
	printf("b = ? ");
	b.scan();
// a + b を出力
	printf("a + b = \n");
	(a.add(b)).print();
	printf("\n");;
// a - b を出力
	printf("a - b = \n");
	(a.sub(b)).print();
	printf("\n");;
// a * b を出力
	printf("a * b = \n");
	(a.mult(b)).print();
	printf("\n");
// a + 単位行列 を出力
	e = Matrix22(1, 0, 0, 1);
	printf("a + E = \n");
	(a.add(e)).print();
	printf("\n");;

	Vector2 u, v;
	
	// a, b を入力
	printf("u = ? ");
	u.scan();
	printf("v = ? ");
	v.scan();
	// a + b を出力
	printf("u + v = ");
	(u.add(v)).print();
	printf("\n");
	
	return 0;
}
