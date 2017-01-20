/**************************************
 *                                    *
 * Bit I/O  Operation                 *
 *                                    *
 *                   Kouichi Kitamura *
 *                                    *
 **************************************/

#include <stdio.h>
#include "bitio.h"

int bit_in(FILE * fp)
{
    /* ファイルから１byte読込み１bitずつ返す */
    /*          ファイル終了なら　－１を返す */

    static int filt = 1, byte = 0;
    int bit;

    if (filt == 1) {
	if ((byte = getc(fp)) != EOF)
	    filt = 0x80;
	else
	    return (-1);
    } else
	filt = filt >> 1;
    bit = ((byte & filt) == 0) ? 0 : 1;
    return (bit);
}

int bit_out(FILE * fp, int bit)
{
    /* １bitずつのファイル出力 */
    /* 実際は１byteになった時点で書き出す */
    /* 溜まっているbit数を返す */

    static int cnt = 0, byte = 0;

    byte = bit + (byte << 1);
    if (cnt == 7) {
	putc(byte, fp);
	cnt = 0;
	byte = 0;
    } else
	cnt++;
    return (cnt);
}

void flush(FILE * fp)
{
    /* 読込み終了時に残っているbitを掃き出す */
    while (bit_out(fp, 0) != 0);
}
