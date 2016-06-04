/* Group 7 Program */


#pragma once

#ifndef Group7_H
#define Group7_H

#include "../Player.h"

#include <fstream>

class Logger {
private:
	std::ofstream ofs;
	std::string logname;
public:
	Logger();

	void newFile();

	void write(std::string message);

	void writeC(const CardSet & mycardset, const CardSet & leave, const GameStatus & gstat);
	
	std::ofstream * getOutputStream();

	~Logger();

};








class Group7 : public Player {
private:

	CardSet table;	//場

	Logger logger;

	int playernum; //最初のプレイヤー人数

	const static double EVALUTION_TABLE[14][2];

	double all_result;
	int playnum;
	bool log_flag;

	int chaku[10];

public:
	Group7(const char * name = "Group7 Ultra");
	Group7(const bool log_flag, const char * name = "Group7 Ultra");

	/*
	 * グループで実態を作成し思考処理を追加する関数．
	 */
	 // ゲームを始めるにあたり必要な初期化をする．
	virtual void ready();
	// カードを出す思考処理を組み込む．
	virtual bool follow(const GameStatus &, CardSet &);

	virtual bool approve(const GameStatus &);

	void result(const int pnum);

	static std::string to_string(int p);
	static std::string to_string(double p);
private:
	//オリジナル
	
	/**
	 * 評価値Aを計算する
	 * @param cset 計算するカードセット
	 * @returns 計算された評価値A
	 */
	static double calcEvalutionA(const CardSet cset);


	/**
	* 評価値Bを計算する
	* @param cset 計算するゲームステータス
	* @returns 計算された評価値B
	*/
	static double calcEvalutionB(const GameStatus & gstat, const int my_id);

	/**
	 * 手札をランクごとに分析する
	 * @param cset 計算するカードセット
	 * @returns 集計された手札の配列。0がJoker、そのあと1～13までそれぞれの数
	 */
	static void getCardNum(const CardSet & cset, int dst[]);

	bool has2orJkr(void);

	static void pickupCardfromNum(CardSet * from, const int number, const int num, CardSet * dst);

#define Vp3 1.4 //終盤判定に使う。最初の手札枚数の1/Vp3になった時に終盤
	bool isShuban(const double evalutionB);

#define VpJk2 13 //Jkrをワイルドカードとして2枚で出すときに一緒になるカードの最低値
#define VpJk3 11 //Jkrをワイルドカードとして3枚で出すときに一緒になるカードの最低値
#define VpJk4 8 //Jkrをワイルドカードとして4枚で出すときに一緒になるカードの最低値
	bool awayJkrWith(const int number, const int num, const int hand_size);

	bool patternA(const GameStatus gstat, const int mycard[], const int table_left[], const int number, bool * ismin, const double evalutionA, const double evalutionB, CardSet & s);
	bool patternB(const GameStatus gstat, const int mycard[], const int table_left[], const int number, bool * ismin, const double evalutionA, const double evalutionB, CardSet & s);

  /*
   * 思考処理を実装するのに使うユーティリティ関数は，自由につくってよい．
   * たとえば手札のソート順のもとでのソート．
   * カード一枚どうしの比較は Player クラスから継承
   */
	void sort(bool ascending = true);
	// CardSet 比較用ツール．基本 Dealer の使っている関数とおなじ．
	static bool cardsStrongerThan(const CardSet & left, const CardSet & right);
	static bool checkRankUniqueness(const CardSet & cs);
	static int cardStrength(const Card & c);
	static int cardStrength(const CardSet & cs);

	double average(const CardSet & );
	CardSet & findSmallestAcceptable(const CardSet & cs, CardSet & mycs);
};

#endif
