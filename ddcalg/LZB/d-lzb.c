
/**********************************
 *                                *
 * LZB Decode Operation           *
 *                                *
 *               Kouichi Kitamura *
 *                                *
 **********************************/

#include <stdio.h>
#include <stdlib.h>

#include "c-lzb.h"
#include "d-lzb.h"
#include "bitio.h"
#include "bin_code.h"

/********************************************** 入出力操作 */
//FILE *fp_in, *fp_out;
int flush_need;
FILE *fp_in;
FILE *fp_out;

int get_head_bit()
{				/* 先頭１bit 読込み */
    return (bit_in(fp_in));
}

int get_match_pos(int part)
{				/* 一致列位置読込み 引数は符号化時の位置範囲 */
    int filt = 1;
    int code = 0, bit;

    while (filt < part) {
	filt = filt << 1;
	bit = bit_in(fp_in);
	code = (code << 1) + bit;
    }
    return (code);
}

int get_match_pos2(int part)
{
    int filt = 1;
    int code = 0; // unused var , bit;
    while ((filt << 1) < part) {
	filt = filt << 1;
	code = (code << 1) + bit_in(fp_in);
    }
    filt = filt << 1;
    if (code >= filt - part) {
	code = (code << 1) + bit_in(fp_in);
	code = code - filt + part;
    }
    return (code);
}

int get_match_leng()
{				/* 一致列長 読込み */
    int i, leng = 0, code = 1, bit;
    while (bit_in(fp_in) == 0)
	leng++;
    for (i = 0; i < leng; i++) {
	bit = bit_in(fp_in);
	code = (code << 1) + bit;
    }
    return (code + 3);
}

int get_chr_code()
{				/* 記号 読込み */
    int i, bit, code = 0;
    for (i = 0; i < 8; i++) {
	bit = bit_in(fp_in);
	if (bit == EOF)
	    return (-1);
	else
	    code = (code << 1) + bit;
    }
    return (code);
}

void put_txt(int chr)
{				/* 復号記号 出力操作 */
    putc(chr, fp_out);
}

/********************************************** リング buffer 操作 */

int buf[BUF_SIZE];
int start_pos = 0, end_pos = 0;

int put_buf(int chr)
{				/* バッファの残り数を返す */
    int rem;
    buf[end_pos] = chr;
    end_pos = (end_pos + 1) % BUF_SIZE;
    rem = start_pos - end_pos;
    if (rem < 0)
	rem = BUF_SIZE + rem;
    return (rem);
}

int get_buf(int num)
{				/* num番目のデータを返す */
    int chr = -1;
    int pos = (start_pos + num) % BUF_SIZE;
    int leng = (BUF_SIZE + end_pos - start_pos) % BUF_SIZE;
    if (leng == 0)
	leng = BUF_SIZE;
    if (num < leng)
	chr = buf[pos];
    return (chr);
}

void shift_buf(int num)
{				/* バッファの内容を左へnum個シフトする */
    start_pos = (start_pos + num) % BUF_SIZE;
}

/********************************************** buffer の仕切り 操作 */

int part_pos = 0;		/* 仕切りの位置 */

int move_part(int num)
{				
    /* 仕切りを右へnum個移動する 移動できなかった個数を返す*/
    int dif, i; //unused var , dum = 0;
    for (i = 0; i < num; i++)
	put_txt(get_buf(part_pos + i));
    part_pos += num;
    dif = part_pos - REG_PART_POS;
    if (dif > 0) {
	part_pos = REG_PART_POS;
	shift_buf(dif);
    }
    return (dif);
}

int get_search_code(int pos)
{				/* pos(0~)番目の未符号化文字を返す */
    return (get_buf(part_pos + pos));
}

int get_part()
{				/* 仕切りの位置を返す */
    return (part_pos);
}

/********************************************** 復号化 */

int decode()
{
    int head, pos, leng, chr, i, rem, lng, sft;

    head = get_head_bit();

    if (head == -1)
	return (-1);		/* 読込み終了なら復号化終了 */
    else if (head == 0) {
	chr = get_chr_code();	/* 記号の復号 */
	if (chr == -1)
	    return (-1);
	rem = put_buf(chr);
	move_part(1);
    } else {
	pos = get_match_pos2(get_part());	/* 一致記号列の復号 */
	leng = get_match_leng();
	lng = leng;
	sft = 0;

	for (i = 0; i < leng; i++) {
	    rem = put_buf(get_buf(pos + i));
	    if (rem == 0) {
		move_part(sft);
		pos = pos - sft;
		lng = lng - sft;
		sft = 0;
	    }
	    sft++;
	}
	move_part(lng);
    }
    return (1);
}

int main(int argc, char *argv[]) {

	if (argc < 3) {
		printf("PARAMETER ERROR\n");
		exit(1);
	}
	if ((fp_in = fopen(argv[1], "rb")) == 0) {
		printf("CAN'T OPEN %s\n", argv[1]);
		exit(1);
	}
	if ((fp_out = fopen(argv[2], "wb")) == 0) {
		printf("CAN'T CREATE %s\n", argv[2]);
		exit(1);
	}
	while (decode() != -1)
		;

	fclose(fp_in);
	fclose(fp_out);
}
