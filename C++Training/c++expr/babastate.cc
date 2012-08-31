//
// babastate.cc - ババ抜きの状態型(C++版)
//	作者: (あなたの名前); 日付: (完成した日付)
//
#include "babastate.h"
#include <stdio.h>

//
// BabaState::reset() - リセット(最初にカードを配った状態にする)
//
void BabaState::reset(void)
{
// 各プレーヤのフラグを未了にし，持ち手を空にする
	for(int i = 0; i < numplayer; i++) {
		finished[i] = false;
		hand[i].makeempty();
	}
// デック(1セット)のカードを initcs に入れる
	CardSet initcs;
	initcs.makedeck();
// 無くなるまで initcs から無作為にカードをとり plr 番のプレーヤに配る
	int plr = 0;	// 配る相手のプレーヤ
	Card c;		// 配るカード
	while(!initcs.pickup(&c)) {
	// 既に配られたカードと同じ番号があれば
	// 今度のカードと既に配られたカードとの両方を捨てる
	// 無ければ今度のカードを自分の手に加える
		if(hand[plr].remove(c.gnumber()))
			hand[plr].insert(c);
	// plr を次のプレーヤにする
		if(++plr >= numplayer)
			plr = 0;
	}
}

//
// BabaState::move() - to 番のプレーヤが from 番のプレーヤからカードを取る
//
bool BabaState::move(int from, int to)
{
	Card c;	// 動かすカード
// from の手から1枚取る．これで from にカードが無くなったか否かを確認
	if(hand[from].pickup(&c))
		return true;	// from にはカードが無い
	if(hand[from].isempty())
		finished[from] = true;
// 取ったカードと同じ番号が to の手にあれば
// 今度のカードと見つかったカードとの両方を捨てる
// 無ければ今度のカードを自分の手に加える
// これで to にカードが無くなったか否かを確認
	if(hand[to].remove(c.gnumber()))
		hand[to].insert(c);
	if(hand[to].isempty())
		finished[to] = true;

	return false;
}

//
// BabaState::print() - 自身の状態を標準出力に出力する
//
void BabaState::print(void)
{
	for(int i = 0; i < numplayer; i++) {
		printf("PLAYER %d ", i);
		hand[i].print();
	}
}
