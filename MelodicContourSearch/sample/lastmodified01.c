// lastmodified01.c  2011-09-24  Hatada
// �w��f�B���N�g�����őS�t�@�C�����ċA�T�����A��ԍŋ߂̍X�V������\������
// lastmodified01 c:/mh/www .c;.h;.java;
#include <io.h>
#include <stdio.h>
#include <time.h>

time_t last = 0;	// �Ώۃt�@�C������ԍŋ߂̍X�V����

char *formatDate(time_t *time) {
    static char datetime[20];
    struct tm *date = localtime(time); 
    sprintf(datetime, "%d/%02d/%02d %2d:%02d:%02d", 
	date->tm_year+1900, date->tm_mon+1, date->tm_mday, 
	date->tm_hour, date->tm_min, date->tm_sec);
    return datetime;
}

void scan(char *dir) {
    struct _finddata_t fdata;
    int fh;
    char path[260];

    sprintf(path, "%s/*.*", dir);
    if ((fh = _findfirst(path, &fdata)) == -1) return;
    do {
    	sprintf(path, "%s/%s", dir, fdata.name);
	if (fdata.attrib & _A_SUBDIR) {	// �f�B���N�g��
	    if (strcmp(fdata.name, ".") != 0 && strcmp(fdata.name, "..") != 0) {
	        scan(path);
	    }	// �J�����g�f�B���N�g���Ɛe�f�B���N�g���͏��O����
	} else {			// �t�@�C��
	    if (fdata.time_write > last) last = fdata.time_write;
	}
    } while (_findnext(fh, &fdata) == 0);
    _findclose(fh);
}

int main(int argc, char* argv[]) {
    scan(argv[1]);
    printf("%s\n", formatDate(&last));
}
