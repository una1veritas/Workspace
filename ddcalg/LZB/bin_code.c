
/**************************************
 *                                    *
 * Binary encoding Operation          *
 *                                    *
 *                   Kouichi Kitamura *
 *                                    *
 **************************************/

#include <stdio.h>

#include "bitio.h"

#include "bin_code.h"

/*
extern int flush_need;
// extern FILE *fp_in,
extern FILE *fp_out;
*/

int I_log(int num, int *filt)
{				/* 整数用対数 log2 num を返す filt=2^(log2 num) */
    int i = 0;
    (*filt) = 1;
    for (i = 0; (*filt) <= num; i++)
	*filt = (*filt) << 1;
    return (i);
}

void fit_code(int code, int filt)
{				/* filt 通りのbit幅で codeを符号化 */
    int bit; // unused var: , f = 0;

    filt = filt >> 1;
    while (filt > 0) {
	bit = ((code & filt) == 0) ? 0 : 1;
	flush_need = bit_out(fp_out, bit);
	filt = filt >> 1;
    }
}

void pos_code(int code, int p_pos)
{				/* 一致位置の符号化 */
    static int filt = 1;
    int leng = p_pos;
    while (filt < leng)
	filt = filt << 1;
    fit_code(code, filt);
}

void pos_code2(int code, int p_pos)
{				/* 一致位置の符号化2 */
    static int filt = 1;
    int leng;

    leng = p_pos;
    while (filt < leng)
	filt = filt << 1;

    if (code < filt - leng)
	fit_code(code, filt / 2);
    else
	fit_code(code + filt - leng, filt);
}

void leng_code(int code)
{				/* 一致記号数の符号化 */
    int i, ret;
    int leng;

    leng = I_log(code, &ret);
    for (i = 1; i < leng; i++)
	flush_need = bit_out(fp_out, 0);
    fit_code(code, ret);
}
