/*
 * main.c
 *
 *  Created on: 2011/11/30
 *      Author: sin
 */
#include <stdio.h>
#include <string.h> /* strlen, strcmp �p */
#include <stdlib.h>  /* atoi �p */

typedef char Char255[256];

#define LN "\r\n"

int main(int argc, char * argv[]) {
	Char255 tmp;
	Char255 itemName[128];
	unsigned int itemPrice[128];
	int itemCount;

	fprintf(stdout, "���i���Ɖ��i���C���p�X�y�[�X�����s�ŋ�؂��ē��͂��āC");
	fprintf(stdout, LN);
	fprintf(stdout, "�I������� '^D' ���C�I���A�C�e�� 'end.' ����͂��āD");
	fprintf(stdout, "�܂��������� 'back.' ����͂��āC�܂��ɂ��ǂ��Ă��Ȃ����D");
	fprintf(stdout, LN);
	fprintf(stdout, "�ł��P�Q�V�܂łɂ��Ȃ��ƃI�[�o�[�t���[�G���[�ɂȂ邩��D");
	fprintf(stdout, LN);

	int num;
	for (num = 0; ; num++) {
		fprintf(stdout, "%d��: ",num+1);
		fscanf(stdin,"%s", itemName[num]);
		if ( strlen(itemName[num]) == 0
				|| strcmp(itemName[num],"end.") == 0) break;
		if ( strcmp(itemName[num],"back.") == 0) {
			num--;
			continue;
		}
		fprintf(stdout, "���i: ");
		fscanf(stdin,"%s", tmp);
		itemPrice[num] = atoi(tmp);
		if ( itemPrice[num] <= 0 ) {
			fprintf(stdout,"�l�i %d �~�H���肦�Ȃ��D���i�������蒼���D", itemPrice[num]);
			fprintf(stdout, LN);
			num--;
			continue;
		}
		fprintf(stdout,"%d �~�ł��ˁD", itemPrice[num]);
		fprintf(stdout, LN);
	}
	itemCount = num;

	fprintf(stdout, LN);
	fprintf(stdout, "�����炢�D");
	fprintf(stdout, LN);
	for(num = 0; num < itemCount; num++) {
		fprintf(stdout, "No. %d %d�~ %s", num+1, itemPrice[num], itemName[num]);
		fprintf(stdout, LN);
	}
	fprintf(stdout, "�����͂���܂����D");
	fprintf(stdout, LN);
	//
	return 0;
}
