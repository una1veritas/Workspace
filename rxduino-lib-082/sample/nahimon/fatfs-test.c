// RX62NのGCCサンプルプログラム
// イーサネット
// (C)Copyright 2011 特殊電子回路

// 使い方
// このプログラムのファイル名をmain.cに変更して、makeしてください。

// 特電HAL
#include "tkdn_hal.h"
#include "tkdnbase.h"

#include <stdio.h>
#include <string.h>

// C言語で書かれたライブラリのヘッダファイルをインクルードする
#ifdef __cplusplus
extern "C" {
#endif

#include "ff.h"

void	disk_timerproc (void);
void	disk_ins (int ins);

#ifdef __cplusplus
}
#endif

FATFS fatfs;			/* File system object */
int found_files;

FRESULT scan_files(const char *path)
{
    FRESULT res;
    FILINFO fno;
    DIR dir;
//	int i;
	char tmp[64];

    res = f_opendir(&dir, path);                       /* ディレクトリを開く */
    if (res == FR_OK) {
//        i = strlen(path);
        for (;;) {
          res = f_readdir(&dir, &fno);                   /* ディレクトリ項目を1個読み出す */
            if (res != FR_OK || fno.fname[0] == 0) break;  /* エラーまたは項目無しのときは抜ける */
            if (fno.fname[0] == '.') continue;             /* ドットエントリは無視 */
			if(fno.fattrib & AM_DIR) sci_putc('D');
			else                     sci_putc(' ');
			if(fno.fattrib & AM_RDO) sci_putc('R');
			else                     sci_putc(' ');
			if(fno.fattrib & AM_HID) sci_putc('H');
			else                     sci_putc(' ');
			if(fno.fattrib & AM_SYS) sci_putc('S');
			else                     sci_putc(' ');
			if(fno.fattrib & AM_ARC) sci_putc('A');
			else                     sci_putc(' ');
			sci_putc(' ');

			if(1980 + (fno.fdate >> 9) - 2011 + 23 >= 1)
			{
	            printf("平成%2d年%2d月%2d日 ", 1980 + (fno.fdate >> 9) - 2011 + 23,
	                   (fno.fdate >> 5) & 15 , fno.fdate & 31);
			}
			else
			{
	            printf("昭和%2d年%2d月%2d日 ", 1980 + (fno.fdate >> 9) - 2011 + 23 + 63,
	                   (fno.fdate >> 5) & 15 , fno.fdate & 31);
			}

            printf("%2d:%02d:%02d ", (fno.ftime >> 11),
                   (fno.ftime >> 5) & 63 , (fno.ftime & 31) << 1);

            printf("%10ld ", fno.fsize);

            sci_puts(path);
            sci_puts("/");
            sci_puts(fno.fname);
            sci_puts("\n");
            found_files++;
        }
    }
    return res;
}

void timer_interrupt_function(void)
{
	disk_timerproc();
}

void fatfs_init()
{
	timer_regist_userfunc(timer_interrupt_function); // ユーザの割り込み関数を登録する
	f_mount(0, &fatfs);		/* Register volume work area (never fails) */	
}

void fatfs_dir(const char *path) {
	sci_puts("ファイルの一覧を表\示します\n");
	found_files = 0;
	if(*path) scan_files(path);
	else      scan_files("");
	
	printf("\n%d個のファイルが見つかりました。\n",found_files);
}

void fatfs_cd(const char *path) {
	const int BUFFER_SIZE = 256;
	char buffer[BUFFER_SIZE];
	if(!path || !*path) {
		sci_puts("ルートディレクトリに移動します");
		if(f_chdir("/") != FR_OK) {
			sci_puts("失敗\n");
		}
	}
	else {
		f_getcwd(buffer,255);
		sci_puts("カレントディレクトリを");
		sci_puts(path);
		sci_puts("に移動します。\n");
		if(f_chdir(path) != FR_OK) {
			sci_puts("失敗\n");
		}
	}
}

void fatfs_type(const char *filename) {
	const int BUFFER_SIZE = 512;
	char buffer[BUFFER_SIZE];

	if(!filename  || !*filename) {
		sci_puts("ファイル名が指定されていません\n");
		return;
	}
	sci_puts("ファイルを表\示します\n");
	sci_puts("\n--開始--\n");

	FIL fil;
	f_open(&fil,filename,FA_READ);

	unsigned int bytes;
	while(f_read(&fil,buffer,BUFFER_SIZE,&bytes) == FR_OK)
	{
		int i;
		for(i=0;i<bytes;i++) sci_putc(buffer[i]);
		if(bytes != BUFFER_SIZE) break;
		if(sci_rxcount()) break;
	}

	f_close(&fil);
	sci_puts("\n--終了--\n");
}

void fatfs_getcwd(char *buffer,int maxlen) {
	f_getcwd(buffer,maxlen);
}

#if 0
	while(1)
	{
		const int BUFFER_SIZE = 512;
		char line[32];
		char buffer[BUFFER_SIZE];
		unsigned int bytes;

		sci_puts("(RX62N)");
		f_getcwd(buffer,255);
		sci_puts(buffer);
		sci_puts("$");
		
		sci_gets(line,31);

		char *cmd = strtok(line," ");
		char *param = strtok(0," ");

		if(!stricmp(cmd,"ins"))
		{
			disk_ins(0);
			timer_wait_ms(100);
			disk_ins(1);
			sci_puts("ディスクの交換が行れたことを通知しました\n");
			continue;
		}

		if(!stricmp(cmd,"help"))
		{
			sci_puts(" ins             ディスクを交換を通知します\n");
			sci_puts(" dir             ファイルの一覧を表\示します\n");
			sci_puts(" cd     dirname  カレントディレクトリを移動します\n");
			sci_puts(" del    filename ファイルを削除します\n");
			sci_puts(" type   filename ファイルの内容を表\示します\n");
			sci_puts(" read   filename ファイルを読み出します\n");
			sci_puts(" create filename ファイルを作成します\n");
			continue;
		}

		if(!stricmp(cmd,"create"))
		{
			if(!param)
			{
				sci_puts("ファイル名が指定されていません\n");
				continue;
			}
			sci_puts("ファイルを生成します\n");
			sci_puts("\n--開始--\n");
			FIL fil;
			f_open(&fil,param,FA_CREATE_ALWAYS | FA_WRITE);

			int p = 0;
			while(1)
			{
				if(sci_rxcount() == 0) continue;
				char c = sci_getc();
				if(c == 0x03)
				{
					f_write(&fil,buffer,p,&bytes);
					p = 0;
					break;
				}
				
				sci_putc(c);
				buffer[p++] = c;
				
				if(p == BUFFER_SIZE)
				{
					f_write(&fil,buffer,p,&bytes);
					p = 0;
				}
			}

			f_close(&fil);
			sci_puts("\n--終了--\n");
			continue;
		}

		if(!stricmp(cmd,"type"))
		{
		}

		if(!stricmp(cmd,"read"))
		{
			if(!param)
			{
				sci_puts("ファイル名が指定されていません\n");
				continue;
			}
			sci_puts("ファイルを読み出します\n");
			sci_puts("\n--開始--\n");
			int start_ms = timer_get_ms();
			int total_size = 0;
			FIL fil;
			f_open(&fil,param,FA_READ);

			while(f_read(&fil,buffer,BUFFER_SIZE,&bytes) == FR_OK)
			{
				total_size += bytes;
				if(bytes != BUFFER_SIZE) break;
				if(sci_rxcount()) break;
			}

			f_close(&fil);
			int end_ms = timer_get_ms();
			sci_puts("--終了--\n");

			sprintf(buffer,"%d bytes を %d msで読み込みました。",
				total_size,end_ms - start_ms);
			sci_puts(buffer);

			if(end_ms != start_ms)
			{
				sprintf(buffer,"速度は %dkB/sec です。\n",
				total_size / (end_ms - start_ms));
				sci_puts(buffer);
			}
			else
			{
				sci_puts("ファイルサイズが小さいため速度が測定できません\n");
			}

			continue;
		}

		if(!stricmp(cmd,"del"))
		{
			if(!param)
			{
				sci_puts("ファイル名が指定されていません\n");
				continue;
			}
			sprintf(buffer,"ファイル %s を削除します\n",param);
			sci_puts(buffer);
			
			f_unlink(param);
			continue;
		}

		if(!stricmp(cmd,"cd"))
		{
		}

		if(!stricmp(cmd,"dir"))
		{
			sci_puts("ファイルの一覧を表\示します\n");
			found_files = 0;

			if(param) {
				scan_files(param);
			}
			else {
				scan_files("0:");
			}
		
			sprintf(buffer,"\n%d個のファイルが見つかりました。\n",found_files);
			sci_puts(buffer);
			continue;
		}
	}
}
#endif

DWORD get_fattime (void)
{
	/* No RTC feature provided. Return a fixed value 2011/1/29 0:00:00 */
	return	  ((DWORD)(2011 - 1980) << 25)	/* Y */
			| ((DWORD)1  << 21)				/* M */
			| ((DWORD)29 << 16)				/* D */
			| ((DWORD)12 << 11)				/* H */
			| ((DWORD)34 << 5)				/* M */
			| ((DWORD)56 >> 1);				/* S */
}
