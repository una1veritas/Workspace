// linecount01.c  2011-09-24  Hatada
// 指定拡張子のファイルを指定ディレクトリ下で再帰探索し、パス名、更新日時、行数、サイズを一覧表示する
// linecount01 c:/mh/www .c;.h;.java;
#include <io.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

int totalFiles = 0, totalLines = 0, totalSize = 0;
time_t last = 0;	// 対象ファイル中一番最近の更新日時

int lineCount1(char* filename) {
    int c, lines = 0;
    FILE* fp = fopen(filename, "r");
    while ((c = fgetc(fp)) != EOF) {
	if (c == '\n') lines++;
    }
    fclose(fp);
    return lines;
}

char *getDate(time_t *time) {
    static char datetime[20];
    struct tm *date = localtime(time); 
    sprintf(datetime, "%d/%02d/%02d %2d:%02d:%02d", 
	date->tm_year+1900, date->tm_mon+1, date->tm_mday, 
	date->tm_hour, date->tm_min, date->tm_sec);
    return datetime;
}

void lineCount(char *dir, char *exts) {
    struct _finddata_t fdata;
    int fh;
    char path[260], *pExt, ext[16];

    sprintf(path, "%s/*.*", dir);
    if ((fh = _findfirst(path, &fdata)) == -1) return;
    do {
	pExt = strchr(fdata.name, '.');		// 拡張子
	sprintf(ext, "%s;", pExt!=NULL ? pExt : "");
    	sprintf(path, "%s/%s", dir, fdata.name);
	if (fdata.attrib & _A_SUBDIR) {
	    if (strcmp(fdata.name, ".") != 0 && strcmp(fdata.name, "..") != 0) {
	        lineCount(path, exts);
	    }	// カレントディレクトリと親ディレクトリは除外する
	} else if (strstr(exts, ext) != NULL) {	// 拡張子の照合
	    int lines = fdata.size > 0 ? lineCount1(path) : 0;
	    if (lines > 0) {
            	printf("%d,%s,%s,%d,%d\n", ++totalFiles, path, 
			getDate(&fdata.time_write), lines, fdata.size);
	    	totalLines += lines;
	    	totalSize += fdata.size;
		if (fdata.time_write > last) last = fdata.time_write;
	    }
	}
    } while (_findnext(fh, &fdata) == 0);
    _findclose(fh);
}

int main(int argc, char* argv[]) {
    lineCount(argv[1], argv[2]);
    printf("合計,%d,%s,%d,%d\n", totalFiles, getDate(&last), totalLines, totalSize);
}
