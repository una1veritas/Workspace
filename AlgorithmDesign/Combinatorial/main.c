/*
 * main.c
 *
 *  Created on: 2011/11/30
 *      Author: sin
 */
#include <stdio.h>
#include <string.h> /* strlen, strcmp 用 */
#include <stdlib.h>  /* atoi 用 */

typedef char Char255[256];

#define LN "\r\n"

int main(int argc, char * argv[]) {
	Char255 tmp;
	Char255 itemName[128];
	unsigned int itemPrice[128];
	int itemCount;

	fprintf(stdout, "商品名と価格を，半角スペースか改行で区切って入力して，");
	fprintf(stdout, LN);
	fprintf(stdout, "終わったら '^D' か，終了アイテム 'end.' を入力して．");
	fprintf(stdout, "まちがったら 'back.' を入力して，まえにもどってやりなおし．");
	fprintf(stdout, LN);
	fprintf(stdout, "でも１２７個までにしないとオーバーフローエラーになるから．");
	fprintf(stdout, LN);

	int num;
	for (num = 0; ; num++) {
		fprintf(stdout, "%d個目: ",num+1);
		fscanf(stdin,"%s", itemName[num]);
		if ( strlen(itemName[num]) == 0
				|| strcmp(itemName[num],"end.") == 0) break;
		if ( strcmp(itemName[num],"back.") == 0) {
			num--;
			continue;
		}
		fprintf(stdout, "価格: ");
		fscanf(stdin,"%s", tmp);
		itemPrice[num] = atoi(tmp);
		if ( itemPrice[num] <= 0 ) {
			fprintf(stdout,"値段 %d 円？ありえない．商品名からやり直し．", itemPrice[num]);
			fprintf(stdout, LN);
			num--;
			continue;
		}
		fprintf(stdout,"%d 円ですね．", itemPrice[num]);
		fprintf(stdout, LN);
	}
	itemCount = num;

	fprintf(stdout, LN);
	fprintf(stdout, "おさらい．");
	fprintf(stdout, LN);
	for(num = 0; num < itemCount; num++) {
		fprintf(stdout, "No. %d %d円 %s", num+1, itemPrice[num], itemName[num]);
		fprintf(stdout, LN);
	}
	fprintf(stdout, "が入力されました．");
	fprintf(stdout, LN);
	//
	return 0;
}
