/**********************************
 *                                *
 * LZB Encode Operation           *
 *                                *
 *	         Kouichi Kitamura *
 *                                *
 **********************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tree.h"
#include "bitio.h"
#include "c-lzb.h"
#include "e-lzb.h"
#include "bin_code.h"

/********************************************** 入出力操作 */

FILE *fp_in, *fp_out;
Tree *T;
int flush_need = 0; /* ファイル出力時のbit掃き捨ての有無 */

int get_c() {
	static int chr = 0;
	if (chr != EOF)
		chr = getc(fp_in);
	return (chr);
}

void put_code(int h, int p, int lng, int part) { /* LZBフォーマットの2値出力 */
	bit_out(fp_out, h);
	if (h == 0)
		fit_code(p, 256);
	else {
		pos_code2(p, part);
		leng_code(lng - 3);
	}
}

/********************************************** リング buffer 操作 */

int buf[BUF_SIZE];
int connect_buf[BUF_SIZE];
int buf_point = 0, buf_leng = 0;
/* buf_point 配列 buf[] の絶対位置 */
/* buf_leng  配列 buf[] の入力記号数 */

int get_buf(int pos) {
	if (pos >= buf_leng)
		return (-1);
	if (buf_leng == BUF_SIZE)
		pos = (pos + buf_point) % BUF_SIZE;
	return (buf[pos]);
}

int put_buf(int c) { /* buf_leng を返す */
	buf[buf_point] = c;
	buf_point = (buf_point + 1) % BUF_SIZE;
	buf_leng = (buf_leng < BUF_SIZE) ? buf_leng + 1 : BUF_SIZE;
	return (buf_leng);
}

void fill_buf(int c) { /* buffer の残りを c で埋め尽くす */
	while (buf_leng < BUF_SIZE)
		put_buf(c);
}

int get_real_pos(int pos) { /* buffer 上の位置 pos の配列上の位置を返す */
	if (pos >= buf_leng)
		return (-1);
	if (buf_leng == BUF_SIZE)
		pos = (pos + buf_point) % BUF_SIZE;
	return (pos);
}

int get_ring_pos(int pos) { /* 配列上の位置 pos の buffer 上の位置を返す */
	if (buf_leng == BUF_SIZE)
		pos = (BUF_SIZE + pos - buf_point) % BUF_SIZE;
	return (pos);
}

int get_buf_leng() { /* buffer の要素数を返す */
	return (buf_leng);
}

/********************************************** buffer の仕切り 操作 */

int part_pos = 0; /* 仕切りの位置 */

int move_part(int num) { /* 窓枠の移動数 を返す */
	int dif, i, dum = 0, dum1 = 0;
	for (i = 1; i < num; i++)
		set_tree(part_pos + i, &dum, &dum1);
	part_pos += num;
	dif = part_pos - REG_PART_POS;
	if (dif > 0) {
		part_pos = REG_PART_POS;
	}
	return (dif);
}

int get_search_code(int pos) { /* pos(0~)番目の未符号化記号 を返す */
	return (get_buf(part_pos + pos));
}

int get_part() { /* 仕切りの位置を返す */
	return (part_pos);
}

/********************************************** buffer 操作 */

int buf_remain; /* buffer 内の記号数 */

void pre_set_buf() {
	int chr;
	chr = get_c();
	if (chr != EOF)
		buf_remain = put_buf(chr);
	else
		put_buf(-2);
}

int set_buf() { /* buffer に新たに記号を入力 入力終了かどうかを返す*/
	int chr;
	chr = get_c();
	if (chr != EOF)
		buf_remain = put_buf(chr);
	else {
		put_buf(-2);
		buf_remain--;
	}
	return (chr);
}

void init_buf() { /* buffer の初期化 buffer いっぱいに入力する */
	int i;
	for (i = 0; i < BUF_SIZE; i++)
		pre_set_buf();
}

void mod_buf(int num) { /* buffer の更新操作 */
	int i, shift;
	shift = move_part(num);
	if (shift > 0)
		del_tree(shift);
	for (i = 0; i < shift; i++)
		set_buf();
}

int do_cont() { /* 符号化 終了判定 */
	if (get_buf(get_part() + 3) != -2)
		return (1);
	else
		return (0);
}

/********************************************** 最長一致記号列探索 */

int now_match_leng = 0;

int search(int pos0, int pos1, int *lng) { /* 一致記号数探索 */
	int p0, p1, pe = get_buf_leng() - 4, f = 0;

	p0 = get_ring_pos(pos0);
	p1 = get_ring_pos(pos1);
	(*lng) = 0;

	while (get_buf(p0) == get_buf(p1) && p1 < pe
			&& (now_match_leng + (*lng)) < MAX_MATCH_LENG) {
		p0++;
		p1++;
		(*lng)++;
	}
	if (p1 == pe)
		f = 1; /* 未符号化bufferを超えても一致 */
	else if ((now_match_leng + (*lng)) == MAX_MATCH_LENG)
		f = 2; /* 上限を超えても一致 */
	return (f);
}

int set_tree(int pos, int *match_pos, int *match_leng) { /* 4文字列を2分木に挿入 */
// unused var.    static int ps_back = 0;
	int i, ret = 1;
// unused var.    int old_ret = -1;
	int ps = 0, pe;
// unused var. int f;
// unused var.    int leng = 0, lng = 0;
	unsigned long x1 = (unsigned long) 0;
	unsigned char x2 = 0;

	pe = get_real_pos(pos);

	for (i = 0; i < 3; i++)
		x1 = x1 * (unsigned long) 256 + (unsigned long) get_buf(pos + i);
	x2 = (unsigned char) get_buf(pos + 3);

	insert(x1, x2, &T, &ps, pe, &ret, match_leng);

	if ((*match_leng) == 4)
		connect_buf[ret] = pe;
	connect_buf[pe] = -1;

	(*match_pos) = ps;
	return (0);
}

void encode(int ps, int pe) { /* 最長一致記号列探索 */
	int f, ff, match_pos, leng = 0, lng = 0, dum = 0, dum1 = 0, p_pos;

	now_match_leng = 0;
	f = search(ps, pe, &leng);
	match_pos = get_ring_pos(ps);
	switch (f) {
	case 0: /* 最長一致がバッファ内で収まる */
		ps = connect_buf[ps];
		while (ps != pe) {
			ff = search(ps, pe, &lng);
			if (leng <= lng) {
				leng = lng;
				match_pos = get_ring_pos(ps);
			}
			ps = connect_buf[ps];
		}
		put_code(1, match_pos, leng, get_part());
		mod_buf(leng);
		break;

	case 1: /* 最長一致が未符号化bufferを超える */
		p_pos = get_part();
		now_match_leng = leng;
		ps = (ps + leng) % BUF_SIZE;
		pe = (pe + leng) % BUF_SIZE;
		mod_buf(leng);
		while ((ff = search(ps, pe, &lng)) != 0) {
			leng += lng;
			now_match_leng += lng;
			ps = (ps + lng) % BUF_SIZE;
			pe = (pe + lng) % BUF_SIZE;
			set_tree(get_part(), &dum, &dum1);
			mod_buf(lng);
			if (ff == 2) {
				put_code(1, match_pos, leng, p_pos);
				p_pos = get_part();
				leng = 0;
				now_match_leng = 0;
				match_pos = get_ring_pos(ps);
			}
		}
		leng += lng;
		put_code(1, match_pos, leng, p_pos);
		if (lng != 0)
			set_tree(get_part(), &dum, &dum1);
		mod_buf(lng);
		break;

	case 2: /* 一致長が上限を超える */
		put_code(1, match_pos, leng, get_part());
		ps = (ps + leng) % BUF_SIZE;
		pe = (pe + leng) % BUF_SIZE;
		mod_buf(leng);
		p_pos = get_part();
		match_pos = get_ring_pos(ps);
		leng = 0;
		while ((ff = search(ps, pe, &lng)) != 0) {
			leng += lng;
			now_match_leng += lng;
			ps = (ps + lng) % BUF_SIZE;
			pe = (pe + lng) % BUF_SIZE;
			set_tree(get_part(), &dum, &dum1);
			mod_buf(lng);
			if (ff == 2) {
				put_code(1, match_pos, leng, p_pos);
				p_pos = get_part();
				leng = 0;
				now_match_leng = 0;
				match_pos = get_ring_pos(ps);
			}
		}
		leng += lng;
		put_code(1, match_pos, leng, p_pos);
		if (lng != 0)
			set_tree(get_part(), &dum, &dum1);
		mod_buf(lng);
		break;
	}
}

void del_tree(int shift) { /* 2分木からbuffer 先頭4記号の列をshift 個削除する */
	unsigned long x1 = 0;
	unsigned char x2 = 0;
	int i, j, next;

	for (i = 0; i < shift; i++) {
		for (j = 0; j < 3; j++)
			x1 = x1 * 256 + (unsigned long) get_buf(i + j);
		x2 = (unsigned char) get_buf(i + 3);

		next = connect_buf[get_real_pos(i)];
		connect_buf[get_real_pos(i)] = -2;

		delete(x1, x2, &T, next);

		x1 = 0;
	}
}

/********************************************** 符号化 */

int main(int argc, char *argv[]) {
	char fname_out[32];
	int i = 0, chr;
	// unused var: f, ff;
	int s_c, match_pos = 0, match_leng;
	// unused var: lng, dum = 0, p;
	T = NULL;

	if ( argc < 2 ) {
		printf("PARAMETER ERROR\n");
		exit(1);
	}
	if ( (fp_in = fopen(argv[1], "rb")) == 0 ) {
		printf("CAN'T OPEN %s\n", argv[1]);
		exit(1);
	}
	if ( argc >= 3 ) {
		strncpy(fname_out, argv[2], 32);
		fname_out[31] = (char) 0;
	} else {
		strncpy(fname_out, argv[1], 28);
		fname_out[28] = (char) 0;
		strcat(fname_out, ".lzb");
	}
	if ( (fp_out = fopen(fname_out, "wb")) == 0 ) {
		printf("CAN'T CREATE %s\n", fname_out);
		exit(1);
	}

	init_buf();

	set_tree(0, &match_pos, &match_leng);
	put_code(0, get_buf(0), 0, 0); /* 最初の記号は無条件に記号の符号化 */
	mod_buf(1);

	while (do_cont()) {

		match_leng = 0; /* 最長一致記号数 */
		now_match_leng = 0;
		match_pos = 0; /* 最長一致記号位置 */

		s_c = get_search_code(0); /* s_c は 検索記号 */

		set_tree(get_part(), &match_pos, &match_leng);
		/* 仕切りの記号を符号化 */

		if (match_leng < 4) { /* 不一致符号化 */
			put_code(0, s_c, 0, get_part());
			mod_buf(1);
		} else
			encode(match_pos, get_real_pos(get_part()));
	}
	i = 0; /* 有効一致記号数未満の残り */
	while ((chr = get_search_code(i++)) != -2) {
		bit_out(fp_out, 0);
		fit_code(chr, 256);
	}

	if (flush_need != 0)
		flush(fp_out); /* bit の掃き捨て */

	fclose(fp_in);
	fclose(fp_out);
}
