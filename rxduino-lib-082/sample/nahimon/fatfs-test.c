// RX62N��GCC�T���v���v���O����
// �C�[�T�l�b�g
// (C)Copyright 2011 ����d�q��H

// �g����
// ���̃v���O�����̃t�@�C������main.c�ɕύX���āAmake���Ă��������B

// ���dHAL
#include "tkdn_hal.h"
#include "tkdnbase.h"

#include <stdio.h>
#include <string.h>

// C����ŏ����ꂽ���C�u�����̃w�b�_�t�@�C�����C���N���[�h����
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

    res = f_opendir(&dir, path);                       /* �f�B���N�g�����J�� */
    if (res == FR_OK) {
//        i = strlen(path);
        for (;;) {
          res = f_readdir(&dir, &fno);                   /* �f�B���N�g�����ڂ�1�ǂݏo�� */
            if (res != FR_OK || fno.fname[0] == 0) break;  /* �G���[�܂��͍��ږ����̂Ƃ��͔����� */
            if (fno.fname[0] == '.') continue;             /* �h�b�g�G���g���͖��� */
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
	            printf("����%2d�N%2d��%2d�� ", 1980 + (fno.fdate >> 9) - 2011 + 23,
	                   (fno.fdate >> 5) & 15 , fno.fdate & 31);
			}
			else
			{
	            printf("���a%2d�N%2d��%2d�� ", 1980 + (fno.fdate >> 9) - 2011 + 23 + 63,
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
	timer_regist_userfunc(timer_interrupt_function); // ���[�U�̊��荞�݊֐���o�^����
	f_mount(0, &fatfs);		/* Register volume work area (never fails) */	
}

void fatfs_dir(const char *path) {
	sci_puts("�t�@�C���̈ꗗ��\\�����܂�\n");
	found_files = 0;
	if(*path) scan_files(path);
	else      scan_files("");
	
	printf("\n%d�̃t�@�C����������܂����B\n",found_files);
}

void fatfs_cd(const char *path) {
	const int BUFFER_SIZE = 256;
	char buffer[BUFFER_SIZE];
	if(!path || !*path) {
		sci_puts("���[�g�f�B���N�g���Ɉړ����܂�");
		if(f_chdir("/") != FR_OK) {
			sci_puts("���s\n");
		}
	}
	else {
		f_getcwd(buffer,255);
		sci_puts("�J�����g�f�B���N�g����");
		sci_puts(path);
		sci_puts("�Ɉړ����܂��B\n");
		if(f_chdir(path) != FR_OK) {
			sci_puts("���s\n");
		}
	}
}

void fatfs_type(const char *filename) {
	const int BUFFER_SIZE = 512;
	char buffer[BUFFER_SIZE];

	if(!filename  || !*filename) {
		sci_puts("�t�@�C�������w�肳��Ă��܂���\n");
		return;
	}
	sci_puts("�t�@�C����\\�����܂�\n");
	sci_puts("\n--�J�n--\n");

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
	sci_puts("\n--�I��--\n");
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
			sci_puts("�f�B�X�N�̌������s�ꂽ���Ƃ�ʒm���܂���\n");
			continue;
		}

		if(!stricmp(cmd,"help"))
		{
			sci_puts(" ins             �f�B�X�N��������ʒm���܂�\n");
			sci_puts(" dir             �t�@�C���̈ꗗ��\\�����܂�\n");
			sci_puts(" cd     dirname  �J�����g�f�B���N�g�����ړ����܂�\n");
			sci_puts(" del    filename �t�@�C�����폜���܂�\n");
			sci_puts(" type   filename �t�@�C���̓��e��\\�����܂�\n");
			sci_puts(" read   filename �t�@�C����ǂݏo���܂�\n");
			sci_puts(" create filename �t�@�C�����쐬���܂�\n");
			continue;
		}

		if(!stricmp(cmd,"create"))
		{
			if(!param)
			{
				sci_puts("�t�@�C�������w�肳��Ă��܂���\n");
				continue;
			}
			sci_puts("�t�@�C���𐶐����܂�\n");
			sci_puts("\n--�J�n--\n");
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
			sci_puts("\n--�I��--\n");
			continue;
		}

		if(!stricmp(cmd,"type"))
		{
		}

		if(!stricmp(cmd,"read"))
		{
			if(!param)
			{
				sci_puts("�t�@�C�������w�肳��Ă��܂���\n");
				continue;
			}
			sci_puts("�t�@�C����ǂݏo���܂�\n");
			sci_puts("\n--�J�n--\n");
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
			sci_puts("--�I��--\n");

			sprintf(buffer,"%d bytes �� %d ms�œǂݍ��݂܂����B",
				total_size,end_ms - start_ms);
			sci_puts(buffer);

			if(end_ms != start_ms)
			{
				sprintf(buffer,"���x�� %dkB/sec �ł��B\n",
				total_size / (end_ms - start_ms));
				sci_puts(buffer);
			}
			else
			{
				sci_puts("�t�@�C���T�C�Y�����������ߑ��x������ł��܂���\n");
			}

			continue;
		}

		if(!stricmp(cmd,"del"))
		{
			if(!param)
			{
				sci_puts("�t�@�C�������w�肳��Ă��܂���\n");
				continue;
			}
			sprintf(buffer,"�t�@�C�� %s ���폜���܂�\n",param);
			sci_puts(buffer);
			
			f_unlink(param);
			continue;
		}

		if(!stricmp(cmd,"cd"))
		{
		}

		if(!stricmp(cmd,"dir"))
		{
			sci_puts("�t�@�C���̈ꗗ��\\�����܂�\n");
			found_files = 0;

			if(param) {
				scan_files(param);
			}
			else {
				scan_files("0:");
			}
		
			sprintf(buffer,"\n%d�̃t�@�C����������܂����B\n",found_files);
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
