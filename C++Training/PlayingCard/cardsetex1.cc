//
// cardsetex1.cc - トランプカードの集合型(C++版)テストプログラム
//	作者: (あなたの名前); 日付: (完成した日付)
//
#include <stdio.h>
#include "Card.h"
#include "CardSet.h"

//
// main() - トランプカードの集合型テストプログラム
//
int main(void)
{
	Card c;
	CardSet cs;

	cs.print();
// 入力がエラーになるまで指定したカードを入れる
	printf("insert: c = ? ");
	while(!c.scan()) {
		if(cs.insert(c))
			printf("＼tinsert error＼n");
		printf("insert: c = ? ");
	}
	cs.print();
// 入力がエラーになるまで指定したカードを除く
	printf("remove: c = ? ");
	while(!c.scan()) {
		if(cs.remove(c))
			printf("＼tremove error＼n");
		printf("remove: c = ? ");
	}
	cs.print();
	
	return 0;
}
