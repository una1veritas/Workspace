/* Group 7 */

#define _STDOUT_DEBUG

#ifdef _MBCS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <iostream>
#include <string>
#include <vector>
//#include <fstream>

#include "Card.h"
#include "CardSet.h"
#include "Player.h"

#include "Group7.h"

const double Group7::EVALUTION_TABLE[14][2] = {
	{ 8.0f	,1.0f	},	//Joker
	{ 4.0f	,3.0f	},	//1
	{ 5.0f	,3.0f	},	//2
	{ -3.5f	,3.0f	},	//3
	{ -3.0f	,4.0f	},	//4
	{ -2.5f	,3.0f	},	//5
	{ -2.0f	,2.0f	},	//6
	{ -1.5f	,1.0f	},	//7
	{ -1.0f	,-0.1f	},	//8
	{ -0.5f	,-0.5f	},	//9
	{ 0.1f	,5.0f	},	//10
	{ 1.0f	,2.0f	},	//J
	{ 2.0f	,3.0f	},	//Q
	{ 3.0f	,3.0f	}	//K
};

Group7::Group7(const char * name) : Player(name), playernum(-1), all_result(0.0f), playnum(0), log_flag(false) {
	for (int i = 0; i < 10;i++) this->chaku[i] = 0;
}

Group7::Group7(const bool log_flag, const char * name) : Player(name), playernum(-1), all_result(0.0f), playnum(0)
{
	this->log_flag = log_flag;
	for (int i = 0; i < 10; i++) this->chaku[i] = 0;
}

void Group7::ready() {


//#ifdef _STDOUT_DEBUG
	std::cout << "*** Start the game - player : " << this->playerName() << "." << std::endl;
	
	//ログ出力
	if(!this->log_flag) this->logger.newFile();
	this->logger.write("ログ取得を開始\n************************************\n最初の手札一覧\n");
//#endif

	this->table.clear();

	this->sort();

	//それぞれの初期化
	if (this->playernum != -1) {
		//-1に初期化する前に終わった＝ビリ
		this->all_result += this->playernum;
		this->chaku[this->playernum - 1]++;
		this->playernum = -1;
	}
	this->playnum++;

//#ifdef _STDOUT_DEBUG
	//今の手札
	std::ofstream * ofs = this->logger.getOutputStream();
	this->hand.printOn(*ofs);
	this->logger.write("\n\n************************************\n");
//#endif




}

bool Group7::follow(const GameStatus & gstat, CardSet & s) {
	CardSet pile(gstat.pile);
	CardSet mycards;
	Card card;

	if (this->playernum == -1) this->playernum = gstat.numPlayers;

	this->sort();
//#ifdef _STDOUT_DEBUG
	this->logger.write("\n************************************\n###現在のゲームステータス\n");
	std::ofstream * ost = this->logger.getOutputStream();
	std::cout << std::endl << gstat << std::endl;
	gstat.printOn(*ost);
	if (gstat.pile.isEmpty()) this->logger.write("自分がリーダーです。\n");
	*ost << "\n###現在の手札\n";
	std::cout << std::endl << this->hand << std::endl;
	ost = this->logger.getOutputStream();
	hand.printOn(*ost);
	ost->close();
//#endif

/*
	std::cout << "average:: " << average(hand) << std::endl;
	std::cout << "hand:: " << inHand() << std::endl;
*/
	//数の計測
	int table_left[14];
	int mycard[14];
	Group7::getCardNum(this->table,table_left);
	Group7::getCardNum(this->hand,mycard);

	//計測が終了
//#ifdef _STDOUT_DEBUG
	this->logger.write("\n\n###現在の場と、自分の手札状況\n");
	this->logger.write("-------------------------\n");
	this->logger.write("|カード||場残|手札||差分|\n");
	this->logger.write("-------------------------\n");
	std::cout << "card nums::" << std::endl;
	for (int i = 0; i < 14; i++) {
		std::string str = "" + Group7::to_string(i) + "\t   ";
		switch (i) {
		case 1:	str =	"Ace   "; break;
		case 11: str =	"Jack  "; break;
		case 12: str =	"Queen "; break;
		case 13: str =	"King  "; break;
		case 0: str =	"Joker "; break;
		}
		std::cout << str << ":\t" << table_left[i] << "\t" << mycard[i] << std::endl;
		this->logger.write("|" + str + "|| " + Group7::to_string((i == 0?1:4) - table_left[i]) + "\t | " + Group7::to_string(mycard[i]) + "  || " + Group7::to_string((i == 0 ? 1 : 4) - mycard[i] - table_left[i]) + "\t|\n");
	}
	this->logger.write("-------------------------\n");
//#endif

//#ifdef _STDOUT_DEBUG
	this->logger.write("\n\n###手札の評価\n");
//#endif

	double evalutionA = Group7::calcEvalutionA(this->hand);
	double evalutionB = Group7::calcEvalutionB(gstat, gstat.turnIndex);

//#ifdef _STDOUT_DEBUG
	this->logger.write("評価値A:" + Group7::to_string(evalutionA) + "\n");
	this->logger.write("評価値B:" + Group7::to_string(evalutionB) + "\n");
//#endif

	this->logger.write("###思考ログ\n");
	//自分が親の時
		int min_pw = -1;	//最弱
		int min2_pw = -1;	//次に弱い
		int max_pw = -1;
		for (int i = 3; i <= 13; i++) {
			if (mycard[i] > 0) {
				if (min_pw == -1) min_pw = i;
				else if (min2_pw == -1) {
					min2_pw = i;
					break;
				}
				max_pw = i;
			}
		}
		if (min2_pw == -1) {
			if (mycard[1] > 0) {
				if (min_pw == -1) min_pw = 1;
				else if (min2_pw == -1) min2_pw = 1;
			}
			if (mycard[2] > 0) {
				if (min_pw == -1) min_pw = 2;
				else if (min2_pw == -1) min2_pw = 2;
			}
			if (mycard[0] > 0) {
				if (min_pw == -1) min_pw = 0;
				else if (min2_pw == -1) min2_pw = 0;
			}
		}

		this->logger.write("一番弱いのは" + Group7::to_string(min_pw) + "、次に弱いのは" + Group7::to_string(min2_pw) + "\n");



	if (gstat.pile.isEmpty()) {
		//自分が親の時
		this->logger.write("自分は今回リーダー。\n");
		if (this->has2orJkr()) {
			this->logger.write("一番弱いのをキープして、２番目を捨てる\n");
			Group7::pickupCardfromNum(&this->hand, min2_pw, mycard[min2_pw], &s);
		}
		else {
			this->logger.write("一番弱いのを捨てる\n");
			Group7::pickupCardfromNum(&this->hand, min_pw, mycard[min_pw], &s);
		}
	}
	else {
		this->logger.write("自分は場よりも強いカードを出さなければならない…\n");
		bool jk2 = this->has2orJkr();
		if (jk2)this->logger.write("でも今はJokerもしくは2を2枚以上持っている…。\n");
		bool ismin = true;
		switch (gstat.pile[0].getNumber()) {

		case 3:	//出されたカードが3	
			if (this->patternA(gstat, mycard, table_left, 4, &ismin, evalutionA, evalutionB, s)) break;
 
		case 4:	//出されたカードが4
			if (this->patternA(gstat, mycard, table_left, 5, &ismin, evalutionA, evalutionB, s)) break;

		case 5:	//出されたカードが5
			if (this->patternA(gstat, mycard, table_left, 6, &ismin, evalutionA, evalutionB, s)) break;

		case 6://出されたカードが6
			if (this->patternA(gstat, mycard, table_left, 7, &ismin, evalutionA, evalutionB, s)) break;

		case 7://出されたカードが7
			if (this->patternA(gstat, mycard, table_left, 8, &ismin, evalutionA, evalutionB, s)) break;

		case 8://出されたカードが8
			if (this->patternA(gstat, mycard, table_left, 9, &ismin, evalutionA, evalutionB, s)) break;

		case 9://出されたカードが9
			if (this->patternA(gstat, mycard, table_left, 10, &ismin, evalutionA, evalutionB, s)) break;

		case 10://出されたカードが10
			if (this->patternB(gstat, mycard, table_left, 11, &ismin, evalutionA, evalutionB, s)) break;

		case 11://Jack
			if (this->patternB(gstat, mycard, table_left, 12, &ismin, evalutionA, evalutionB, s)) break;

		case 12://Queen
			if (this->patternB(gstat, mycard, table_left, 13, &ismin, evalutionA, evalutionB, s)) break;


		case 13://King
			if (mycard[1] > 0) {
				this->logger.write("手元に一応出せるカード1がある。\n");
				//出し方が難しいぞ

				if (gstat.pile.size() == 1 && mycard[1] == 1) {
					//1枚だけ
					this->logger.write("単体で出せと。\n");
					if ((table_left[2] + mycard[2]) >= 4 && table_left[0] == 1) {
						this->logger.write("Jkrはすでに出てて、かつ2も出る可能性は無いから絶対にリーダーになれる\n");
						this->logger.write("自分の手にどれぐらい残ってるか\n");
						if (this->isShuban(evalutionB)) {
							this->logger.write("もう割と残ってなさそうなので、ここは1を捨てて小さいのを捨てよう\n");
							//捨てる（判定緩めで）
							Group7::pickupCardfromNum(&this->hand, 1, gstat.pile.size(), &s);
							break;
						}
						else {
							this->logger.write("まだまだ序盤だし、一回パスして様子見\n");
						}
					}
					else {
						if (this->isShuban(evalutionB)) {
							this->logger.write("終盤。もう割と残ってなさそうなので、ここは1を捨てて小さいのを捨てよう\n");
							//捨てる（判定緩めで）
							Group7::pickupCardfromNum(&this->hand, 1, gstat.pile.size(), &s);
							break;
						}
						else this->logger.write("まだまだ序盤だし、一回パスして様子見\n");
					}

				}
				else if (mycard[1] > gstat.pile.size()) {
					//ここは要求されてる数よりも多く持っている。					
					if (mycard[1] == gstat.pile.size()) {
						//捨てるべきか
						if (mycard[2] > 0) {
							this->logger.write("2がまだ残ってるならいいかな、1を捨てよう\n");
							Group7::pickupCardfromNum(&this->hand, 1, gstat.pile.size(), &s);
							break;
						}
						else if (table_left[2] == 4 && this->hand.size() < (54.0f / this->playernum / Vp3)) {
							this->logger.write("2がすべて出払っていて終盤なので、1を捨てよう\n");
							Group7::pickupCardfromNum(&this->hand, 1, gstat.pile.size(), &s);
							break;
						}

					}
					else {
						this->logger.write("捨てても1がまだ残るなんて最高じゃないか、良いだろう、捨てよう\n");
						Group7::pickupCardfromNum(&this->hand, 1, gstat.pile.size(), &s);
						break;
					}

				}
				else if (mycard[0] >= 1) {
					this->logger.write("枚数的に足りないがJkrが使える。さて、どうする。\n");
					//外部関数に処理を委ねる。他のダブルも同じ。
					if (this->awayJkrWith(1, gstat.pile.size(), this->hand.size()) || (this->hand.size() < (54.0f / this->playernum / Vp3) || evalutionB < -1.0)) {
						//ＯＫらしい
						this->logger.write("終盤だからJkr使って捨てていく\n");
						Group7::pickupCardfromNum(&this->hand, 1, gstat.pile.size() - 1, &s);
						this->hand.remove(0);
						Card c;
						c.set(Card::SUIT_JOKER, 0);
						s.insert(c);	//Jkrの追加
						break;
					}
					else {
						this->logger.write("まだJkrを使う時ではない\n");
					}
				}
				else {
					this->logger.write("枚数的に足りない。Jkrも使えない。パスしかなさそう。\n");
				}
			}

		case 1:	//Ace
			if (mycard[2] > 0) {
				this->logger.write("手元に一応出せるカード2がある。\n");
				//出し方が難しいぞ

				if (gstat.pile.size() == 1 && mycard[2] == 1) {
					//1枚だけ
					this->logger.write("単体で出せと。\n");
					if (table_left[0] == 1) {
						this->logger.write("Jkrはすでに出てるから絶対にリーダーになれる\n");
						this->logger.write("自分の手にどれぐらい残ってるか\n");
						if (evalutionB <= -1.0 || this->hand.size() < (54.0f / this->playernum / Vp3)) {
							this->logger.write("もう割と残ってなさそうなので、ここは2を捨てて小さいのを捨てよう\n");
							//捨てる（判定緩めで）
							Group7::pickupCardfromNum(&this->hand, 2, gstat.pile.size(), &s);
							break;
						}
						else {
							this->logger.write("まだまだ序盤だし、一回パスして様子見\n");
						}
					}
					else {
						if (evalutionB <= -1.0 || this->hand.size() < (54.0f / this->playernum / Vp3)) {
							this->logger.write("終盤。もう割と残ってなさそうなので、ここは2を捨てて小さいのを捨てよう\n");
							//捨てる（判定緩めで）
							Group7::pickupCardfromNum(&this->hand, 2, gstat.pile.size(), &s);
							break;
						}else this->logger.write("まだまだ序盤だし、一回パスして様子見\n");
					}
				}
				else if (mycard[2] == gstat.pile.size()) {
					//ここは2枚以上を出すように言われている
					//それ以上持ってるが、要求されてる数きっちりしか持ってない
					//2を崩すということは、一番小さいのを捨てないといけない。
					//ここでevalutionAを確認
					if (evalutionA > 0.0f) {
						this->logger.write("評価値Aは良い。下に溜まってないということだから、2の2枚攻めでリーダーになろう。\n");
						//捨てて一番小さいのを捨てよう！
						Group7::pickupCardfromNum(&this->hand, 2, gstat.pile.size(), &s);
						break;
					}
					else {
						this->logger.write("評価値Aが悪い。下に溜まっているということ。序盤か？終盤か？\n");
						if (this->hand.size() < (54.0f / this->playernum / Vp3)) {
							this->logger.write("自分の手札的には終盤だ。ここはガツガツ勝負に出ねば。\n");
							//捨てる（判定緩めで）
							Group7::pickupCardfromNum(&this->hand, 2, gstat.pile.size(), &s);
							break;
						}
						else {
							this->logger.write("まだまだ序盤だし、一回パスして様子見\n");
						}
					}
				}
				else if (mycard[2] > gstat.pile.size()) {
					//ここは要求されてる数よりも多く持っている。
					this->logger.write("捨てても2がまだ残るなんて最高じゃないか、良いだろう、捨ててリーダーになろう\n");
					Group7::pickupCardfromNum(&this->hand, 2, gstat.pile.size(), &s);
					break;
				}
				else if (mycard[0] >= 1) {
					this->logger.write("枚数的に足りないがJkrが使える。さて、どうする。\n");
					//外部関数に処理を委ねる。他のダブルも同じ。
					if (this->awayJkrWith(2, gstat.pile.size(), this->hand.size()) || (this->hand.size() < (54.0f / this->playernum / Vp3) || evalutionB < -1.0)) {
						//ＯＫらしい
						this->logger.write("終盤だからJkr使って捨てていく\n");
						Group7::pickupCardfromNum(&this->hand, 2, gstat.pile.size() - 1, &s);
						this->hand.remove(0);
						Card c;
						c.set(Card::SUIT_JOKER, 0);
						s.insert(c);	//Jkrの追加
						break;
					}
					else {
						this->logger.write("まだJkrを使う時ではない\n");
					}
				}
				else {
					this->logger.write("枚数的に足りない。Jkrも使えない。パスしかなさそう。\n");
				}


		case 2:
			if (mycard[0] > 0) {
				this->logger.write("Jkrがあるけど…\n");
				//Jkrと何か連続が一つあるなら出す
				int p = 0;
				for (int i = 1; i < 14; i++) {
					if (mycard[i] > 0) p++;
				}
				if (p == 1) {
					this->logger.write("勝ち確定\n");
					Group7::pickupCardfromNum(&this->hand, 0, gstat.pile.size(), &s);
					break;
				}
				else {
					this->logger.write("単体でJkrを出すのはもったいなさすぎる\n");
				}
			}

		case 0:	//Jkr
			//パスしかない
			this->logger.write("パスです\n");
			break;
			}
		}
	}


	/*
	if (s.isEmpty()) {
		this->logger.write("思考放棄☆\n");
		if (cardStrength(pile) < average(hand)) {
			std::cout << "Hummm..." << std::endl;
			findSmallestAcceptable(pile, mycards);
			if (mycards.size() > 0) {
				hand.removeAll(mycards);
				s.insertAll(mycards);
				this->table.insertAll(s);
			}
		}
		else {
			hand.pickup(card, -1); // とにかく一枚選ぶ．
			std::cout << "try " << card << std::endl;
			s.insert(card);
			if (cardsStrongerThan(s, pile)) {
				//return true;
			}
			else {
				// やっぱやめる
				hand.insertAll(s);
				s.makeEmpty();
			}
		}
	}*/

	this->logger.write("\n###結果捨てるカード\n");
	if (s.isEmpty()) {
		this->logger.write("パスします\n");
	}
	else {
//		std::ofstream * ost2 = this->logger.getOutputStream();
		std::cout << std::endl << s << std::endl;
		s.printOn(*ost);
		ost->close();
	}

	if (this->hand.isEmpty()) {
		//上がり
		//何番目か
		this->logger.write("\n\n上がりました:" + Group7::to_string(this->playernum - gstat.numPlayers + 1) + "番目です\n");
		this->all_result += this->playernum - gstat.numPlayers + 1;
		this->chaku[this->playernum - gstat.numPlayers]++;
		this->playernum = -1;
	}

	this->logger.write("\n\n************************************\n");
	return true;
}

bool Group7::approve(const GameStatus & gstat){

	if (this->playernum == -1) this->playernum = gstat.numPlayers;

	this->table.insertAll(gstat.pile);

//#ifdef _STDOUT_DEBUG
//	std::cout << "APPROVE : " << gstat.pile << std::endl;
	//this->logger.write("
	if (gstat.pile.isEmpty()) this->logger.write("次のプレイヤーはパスしました\n");
	else {
		this->logger.write("次のプレイヤーが捨てました:");
		gstat.pile.printOn(*this->logger.getOutputStream());
		this->logger.write("\n");
	}
	std::ofstream * ofs = this->logger.getOutputStream();
	
	gstat.printOn(*ofs);
	this->logger.write("\n");
//#endif
	return true;
}

void Group7::result(const int pnum) {
	//結果を格納
	//テスト用

	//最後ビリだった場合を考える
	if (this->playernum != -1) {
		//-1に初期化する前に終わった＝ビリ
		this->all_result += this->playernum;
		this->chaku[this->playernum - 1]++;
		this->playernum = -1;
	}

	this->logger.newFile();
	std::ofstream * ofs = this->logger.getOutputStream();

	*ofs << "###勝率" << std::endl;
	*ofs << "試合数:" << this->playnum << std::endl;
	*ofs << "着平均:" << (this->all_result / this->playnum) << std::endl;
	*ofs << "勝率:" << (1.00 - this->all_result / this->playnum / pnum) << std::endl << std::endl;
	
	for (int i = 0; i < 10; i++) {
		*ofs << (i+1) << "位:" << this->chaku[i] << std::endl;
	}

	ofs->close();
}



double Group7::calcEvalutionA(const CardSet cset)
{
	//評価を行う
	//Ａ評価
	int num[14];
	
	Group7::getCardNum(cset,num);

	double ret = 0.0f;

	for (int i = 0; i < 14; i++) {
		if (num[i] > 0)	ret += Group7::EVALUTION_TABLE[i][0] + Group7::EVALUTION_TABLE[i][0] * (num[i] - 1) * Group7::EVALUTION_TABLE[i][1];
	}
	ret /= static_cast<double>(cset.size());
	return ret;
}

double Group7::calcEvalutionB(const GameStatus & gstat, const int my_id)
{
	//評価を行う
	//B評価
	double average_ev = 0.0f;
	for (int i = 0; i < gstat.numPlayers; i++) {
		if (i != my_id) {
			average_ev += gstat.numCards[i];
		}
		else {
			average_ev -= gstat.numCards[i] * static_cast<double>(gstat.numPlayers - 1);
		}
	}
	
	average_ev /= static_cast<double>(gstat.numPlayers - 1);
	//average_ev -= gstat.numCards[


	return average_ev;
}

void Group7::getCardNum(const CardSet & cset,int dst[])
{
	for (int i = 0; i < 14; i++) dst[i] = 0;

	for (int i = 0; i < cset.size(); i++) {
		if (cset[i].isJoker()) dst[0]++;
		else dst[cset[i].getNumber()]++;
	}

}

bool Group7::has2orJkr(void)
{
	int handnum[14];
	Group7::getCardNum(this->hand,handnum);
	if (handnum[2] >= 2 || handnum[0] >= 1) return true;
	return false;
}

void Group7::pickupCardfromNum(CardSet * from, const int number, const int num, CardSet * dst)
{
	if (num > 0) {
		for (int i = 0; i < from->size(); i++) {
			if ((*from)[i].getNumber() == number) {
				Card p = (*from)[i];
				from->remove(p);
				dst->insert(p);
				Group7::pickupCardfromNum(from, number, num - 1, dst);
				break;
			}
		}
	}
}

bool Group7::isShuban(const double evalutionB)
{
  int tmp[14];
  
	Group7::getCardNum(this->hand,tmp);
	int p = 0;
	for(int i = 0;i < 14;i++) p += (tmp[i] > 0? 1 : 0);
	return evalutionB <= -1.0 || p < (54.0f / this->playernum / Vp3);
}

bool Group7::awayJkrWith(const int number, const int num, const int hand_size)
{
	//numには要求されている数
	//実際に持っているのはnum-1である。

	if (num == 1) {
		this->logger.write("****program error");
		return false;
	}

	//単純に、J以上ならJkrを一緒に出していいことにしよう
	bool res = false;
	switch (num) {
	case 2:
		if (number > VpJk2) res = true;
		break;
	case 3:
		if (number > VpJk3) res = true;
		break;
	case 4:
		if (number > VpJk4) res = true;
		break;
	default:
		this->logger.write("****program error");
		return false;
	}
	if (res) this->logger.write("Jkrをワイルドカードとして使用");
	else this->logger.write("Jkrはまだ温存しよう");

	return res;
}

bool Group7::patternA(const GameStatus gstat, const int mycard[], const int table_left[], const int number, bool * ismin, const double evalutionA, const double evalutionB, CardSet & s)
{
//	bool jk2 = this->has2orJkr();
	if (mycard[number] > 0) {	//4のカードがあった
		this->logger.write("手元に一応出せるカード" + Group7::to_string(number) + "がある。\n");
		if (false) {
			*ismin = false;
			this->logger.write("一番小さいカードはこれだが、Jkrと2を信じてこれは取っておこう。\n");
		}
		else {
			if (mycard[number] == gstat.pile.size()) {
				this->logger.write("場と同じ枚数だけ持ってる。これを出してみよう。\n");
				//leave all 4
				Group7::pickupCardfromNum(&this->hand, number, gstat.pile.size(), &s);
				return true;
			}
			else if (mycard[number] > gstat.pile.size()) {
				this->logger.write("必要とされてる数以上持ってる…。どうする？\n");
				if (evalutionB > -1.0) {
					this->logger.write("数の評価値Bが-1.0より大きいのでペアで出したい。今回は次のものを探す。\n");
				}
				else {
					//余計かもしれない
					this->logger.write("数の評価値Bが-1.0以下だ…。ひどいな、出来るだけ捨てたい。\n");
					this->logger.write("ペア崩してでもとりあえずパスは避けたい\n");
					Group7::pickupCardfromNum(&this->hand, number, gstat.pile.size(), &s);
					return true;
				}
			}
			else {
				this->logger.write("でもカードが足りない。\n");
				if (mycard[0] >= 1) {
					this->logger.write("Jkrがあった。使っていいか？\n");
					//外部関数に処理を委ねる。他のダブルも同じ。
					if (this->awayJkrWith(number, gstat.pile.size(), this->hand.size()) || (this->hand.size() < (54.0f / this->playernum / Vp3) || evalutionB < -1.0)) {
						//ＯＫらしい
						this->logger.write("終盤だからJkr使って捨てていく\n");
						Group7::pickupCardfromNum(&this->hand, number, gstat.pile.size() - 1, &s);
						this->hand.remove(0);
						Card c;
						c.set(Card::SUIT_JOKER, 0);
						s.insert(c);	//Jkrの追加
						return true;
					}
					else {
						this->logger.write("Jkrはあったが使うに値しない場面\n");
					}
				}else this->logger.write("Jkrもねえや\n");
			}
		}
	}
	return false;
}


bool Group7::patternB(const GameStatus gstat, const int mycard[], const int table_left[], const int number, bool * ismin, const double evalutionA, const double evalutionB, CardSet & s)
{
	if (mycard[number] > 0) {
				this->logger.write("手元に一応出せるカード" + Group7::to_string(number) + "がある。\n");
				//出し方が難しいぞ

				if (gstat.pile.size() == mycard[number]) {
					if(this->isShuban(evalutionB)){
					  Group7::pickupCardfromNum(&this->hand, number, gstat.pile.size(), &s);
					return true;
					}

				}
				else if (mycard[number] > gstat.pile.size()) {
					//ここは要求されてる数よりも多く持っている。					
					if (mycard[number] == gstat.pile.size()) {
						//捨てるべきか
						if (mycard[2] > 0) {
							this->logger.write("まだ残ってるならいいかな、" + Group7::to_string(number) + "を捨てよう\n");
							Group7::pickupCardfromNum(&this->hand, number, gstat.pile.size(), &s);
							return true;
						}
						else if (table_left[2] == 4 && this->isShuban(evalutionB)) {
							this->logger.write("2がすべて出払っていて終盤なので、1を捨てよう\n");
							Group7::pickupCardfromNum(&this->hand, number, gstat.pile.size(), &s);
							return true;
						}

					}
					else {
						this->logger.write("捨てても" + Group7::to_string(number) + "がまだ残るなんて最高じゃないか、良いだろう、捨てよう\n");
						Group7::pickupCardfromNum(&this->hand, number, gstat.pile.size(), &s);
						return true;
					}

				}
				else if (mycard[0] >= 1) {
					this->logger.write("枚数的に足りないがJkrが使える。さて、どうする。\n");
					//外部関数に処理を委ねる。他のダブルも同じ。
					if (this->awayJkrWith(number, gstat.pile.size(), this->hand.size()) || this->isShuban(evalutionB)) {
						//ＯＫらしい
						this->logger.write("終盤だからJkr使って捨てていく\n");
						Group7::pickupCardfromNum(&this->hand,number, gstat.pile.size() - 1, &s);
						this->hand.remove(0);
						Card c;
						c.set(Card::SUIT_JOKER, 0);
						s.insert(c);	//Jkrの追加
						return true;
					}
					else {
						this->logger.write("まだJkrを使う時ではない\n");
					}
				}
				else {
					this->logger.write("枚数的に足りない。Jkrも使えない。パスしかなさそう。\n");
				}
			}


return false;
}


std::string Group7::to_string(int p)
{
	char dst[128];
	sprintf(dst, "%d", p);
	return std::string(dst);
}

std::string Group7::to_string(double p)
{
	char dst[128];
	sprintf(dst, "%f", p);
	return std::string(dst);
}





/*
 * 順序関係 compareCards を使うナイーヴ naive なソート．
 * 枚数は少ないので，効率は気にしない．
 */
void Group7::sort(bool ascending) {
	for(int i = 0; i+1 < hand.size(); i++) {
		for(int j = i+1; j < hand.size(); j++) {
			if ( (ascending && cardGreaterThan(hand[i], hand[j]))
					|| (!ascending && cardLessThan(hand[i], hand[j])) ) {
				Card t = hand[i];
				hand[i] = hand[j];
				hand[j] = t;
			}
		}
	}
}

//   Returns true if and only if the left CardSet is either correct and stronger than
// the right one, or the right one is an illegal set.
//
bool Group7::cardsStrongerThan(const CardSet & left, const CardSet & right) {
	int leftRank, rightRank;

	// regarded as "pass"
	if (left.isEmpty() )
		return false;

	// left is an illegal set
	if (!checkRankUniqueness(left))
		return false;
	if ( left.size() >= 5 )
		return true;

	// left always wins
	if ( right.isEmpty() )
		return true;

	// right is an illegal set.
	if (!checkRankUniqueness(right))
		return true;
	if ( right.size() >= 5 )
		return true;

	// the number of cards of the left set must be match to that of the right one.
	if ( left.size() != right.size() )
		return false;


	leftRank = cardStrength(left);
	rightRank = cardStrength(right);

	if ( leftRank > rightRank )
		return true;
	else
		return false;
}

int Group7::cardStrength(const CardSet & cs) {
	int i;
	if ( cs.isEmpty() )
		return 0;

	if ( cs.size() == 1 && cs[0].isJoker() ) {
		return cardStrength(cs[0]);
	}
  	for (i = 0; i < cs.size(); i++) {
	  if (!cs[i].isJoker()) {
		  break;
	  }
	}
	return cardStrength(cs[i]);
}

bool Group7::checkRankUniqueness(const CardSet & cs) {
	int rank = 0;

	if (cs.size() == 0)
		return false;

	if ( cs.size() == 1 && cs[0].isJoker() )
		return true;

	for (int i = 0; i < cs.size(); i++) {
	  if (cs[i].isJoker() )
		  continue;  // Jkrをスキップ
	  if ( rank == 0 ) {
		  rank = cs[i].getNumber();
	  } else if ( rank != cs[i].getNumber() ) {
	    return false;
	  }
	}
	return true;
}

double Group7::average(const CardSet & cset) {
	double sum = 0;
	if ( cset.size() == 0 )
		return sum;
	for(int i = 0; i < cset.size(); i++) {
		sum += cardStrength(cset[i]);
	}
	return sum/cset.size();
}

int Group7::cardStrength(const Card & c) {
	if ( c.isJoker() )
		return 18;
  	if ( c.getNumber() <= 2 )
  		return c.getNumber() + 13;
	return c.getNumber();
}

CardSet & Group7::findSmallestAcceptable(const CardSet & cs, CardSet & mycs) {
	// assumes the hand is sorted in ascending order
	mycs.makeEmpty();
	int cssize = cs.size();
	int csstrength = cardStrength(cs);
	int i, n;
	if ( cssize > 1 )
		std::cout << "multiple cards!!!" << std::endl;
	for(i = 0; i < hand.size(); ) {
		if ( cardStrength(hand[i]) <= csstrength ) {
			++i;
			continue;
		}
		for(n = 1; i + n < hand.size(); ) {
			if ( hand[i+n].isJoker()
				|| (cardStrength(hand[i + n]) == cardStrength(hand[i])) ) {
				++n;
				continue;
			}
			break;
		}
		if ( cssize <= n ) {
			if ( cssize != 0 )
				n = cssize;
			for(int j = i; j < i + n; j++)
				mycs.insert(hand[j]);
			return mycs;
		}
		i += n;
	}
	return mycs;   // empty card set.
}



//**********************************************************
//* Logger


Logger::Logger() : ofs(), logname("default.log") {

}

void Logger::newFile(void) {

	//出力ファイル名を決定
	time_t timer;
	struct tm *date;
	char str[256];

	timer = time(NULL);          /* 経過時間を取得 */
	date = localtime(&timer);    /* 経過時間を時間を表す構造体 date に変換 */

	int ptm = 0;
	std::ifstream ifs;
	do {
		ifs.close();
		strftime(str, 255, "logs/%Y_%B_%d_%A__%p%I_%M_%Slog", date);
		this->logname = std::string(str);
		if (ptm == 0) this->logname += ".log";
		else this->logname = this->logname + "_" + Group7::to_string(ptm) + ".log";
#ifdef _STDOUT_DEBUG
		std::cout << "Log file:" << str << std::endl;
#endif
		ifs.open(this->logname.c_str(), std::ios::out);
		ptm++;
	} while (ifs);
}

void Logger::write(std::string message) {
#ifdef _STDOUT_DEBUG
	if (!this->ofs.is_open()) this->ofs.open(this->logname.c_str(), std::ios::app);
	this->ofs.write(message.c_str(), message.length());
	this->ofs.close();
#endif
}

void Logger::writeC(const CardSet & mycardset, const CardSet & leave, const GameStatus & gstat) {

}

std::ofstream * Logger::getOutputStream()
{
#ifdef _STDOUT_DEBUG
	if (!this->ofs.is_open()) this->ofs.open(this->logname.c_str(), std::ios::app);
#endif
	return &this->ofs;
}


Logger::~Logger()
{
}

