/*******************************************************************************
* RXduinoライブラリ & 特電HAL
* 
* このソフトウェアは特殊電子回路株式会社によって開発されたものであり、当社製品の
* サポートとして提供されます。このライブラリは当社製品および当社がライセンスした
* 製品に対して使用することができます。
* このソフトウェアはあるがままの状態で提供され、内容および動作についての保障はあ
* りません。弊社はファイルの内容および実行結果についていかなる責任も負いません。
* お客様は、お客様の製品開発のために当ソフトウェアのソースコードを自由に参照し、
* 引用していただくことができます。
* このファイルを単体で第三者へ開示・再配布・貸与・譲渡することはできません。
* コンパイル・リンク後のオブジェクトファイル(ELF ファイルまたはMOT,SRECファイル)
* であって、デバッグ情報が削除されている場合は第三者に再配布することができます。
* (C) Copyright 2011-2012 TokushuDenshiKairo Inc. 特殊電子回路株式会社
* http://rx.tokudenkairo.co.jp/
*******************************************************************************/

/*
 *      Author: R.Naitou
 *      Author: 特殊電子回路(株)
 */

#include "ff.h"
#include "sdmmc.h"

#ifdef __cplusplus
	extern "C" {
#endif
void	disk_timerproc (void);
void	disk_ins (int ins);
#ifdef __cplusplus
	}
#endif

#include <string.h>

static FATFS fatfs;
static FIL   fatfs_file;
static DIR   fatfs_dir;
int count;

static void timer_func(void)
{
	disk_ins(digitalRead(PIN_P15) ? 0 : 1);
	disk_timerproc();
}

SDMMC::SDMMC()
{
	sdmmc_handle = &fatfs;

	gpio_set_pinmode(PIN_SPI_CS0,1);
	gpio_write_port(PIN_SPI_CS0,1);
	gpio_set_pinmode(PIN_SPI_CS1,1);
	gpio_write_port(PIN_SPI_CS1,1);
	gpio_set_pinmode(PIN_SPI_CS2,1);
	gpio_write_port(PIN_SPI_CS2,1);
	gpio_set_pinmode(PIN_SPI_CS3,1);
	gpio_write_port(PIN_SPI_CS3,1);
}

void SDMMC::insert(bool ins)
{
	disk_ins(ins ? 1 : 0);
}

void SDMMC::begin()
{
	begin(0);
}

void SDMMC::begin(int cspin)
{
	timer_regist_disktimer(timer_func); // ユーザの割り込み関数を登録する
	f_mount(0, (FATFS *)sdmmc_handle);
	insert(0);
	insert(1);
	pinMode(PIN_P15 , INPUT);
}

bool SDMMC::exists(const char *filename)
{
	FIL fil;
	FRESULT result = f_open(&fil, filename, FA_READ);
	f_close(&fil);
	if(result == FR_OK) return true;
	return false;
}

bool SDMMC::mkdir(const char *filename)
{
	FRESULT result = f_mkdir(filename);

	if(result == FR_OK) return true;
	return false;
}

File SDMMC::open(const char *filename, FILE_MODE mode)
{
	FILINFO info;
	int found = 0; // 0:none 1:file 2:dir

	FRESULT result;
	result = f_stat (filename, &info);
	if(result == FR_OK) 
	{
		found = 1; // 既にファイルが存在する
	}
	else
	{
		// みつからなければディレクトリを探す
		DIR tmpdir;
		result = f_opendir(&tmpdir, filename);
		if(result == FR_OK) 
		{
			found = 2; // ディレクトリがみつかった
			// ファイルライトでディレクトリが見つかったらエラー
			if(mode == FILE_WRITE) return File(NULL,0,"",false,this);
		}
		else
		{
			// ファイルリードで、見つからなければエラー
			if(mode == FILE_READ) return File(NULL,0,"",false,this);
		}
	}

	if(found == 2) // ディレクトリだ
	{
		DIR *pdir = &fatfs_dir;
		result= f_opendir(pdir, filename);
		return File(pdir,0,filename,true,this);
	}
	else
	{
		FIL *pfil = &fatfs_file;
		if(mode == FILE_READ)
		{
			result= f_open(pfil, filename, FA_READ);
			if(result == FR_OK)
			{
				return File(pfil,info.fsize,filename,false,this);
			}
		}

		if(mode == FILE_WRITE)
		{
			if(found == 1)
			{
				result= f_open(pfil, filename, FA_WRITE);
			}
			else
			{
				result= f_open(pfil, filename, FA_CREATE_ALWAYS | FA_WRITE);
			}

			if(result == FR_OK)
			{
				if(found) f_lseek(pfil,info.fsize); // 末尾に移動
				return File(pfil,info.fsize,filename,false,this);
			}
		}
	}

	return File(NULL,0,"",false,this);
}

bool SDMMC::remove(const char *filename)
{
	FRESULT result = f_unlink(filename);

	if(result == FR_OK) return true;
	return false;
}

bool SDMMC::rename(const char *oldname, const char *newname)
{
	FRESULT result = f_rename(oldname, newname);

	if(result == FR_OK) return true;
	return false;
}

bool SDMMC::rmdir(const char *filename)
{
	FRESULT result = f_unlink(filename);

	if(result == FR_OK) return true;
	return false;
}

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

//////////////////////////////////////////////////////////

File::File(void *handle,int size,const char *name,bool isdir,SDMMC *parent)
{
	this->handle = handle;
	this->isdir = isdir;
	this->filesize = size;
	strncpy(this->filename,name,15);
	this->filename[15] = '\0';
	this->parent = parent;
	rbuf_rptr = 0;
	rbuf_datacount = 0;
	eof = false;
	rpos = 0;
}

bool File::available()
{
	if(!handle) return false;
	if(eof) return false;
	if(rbuf_rptr >= rbuf_datacount)
	{
		if(!readbuf()) return 0;
		rbuf_rptr = 0;
	}

	return true;
}

void File::close()
{
	if(!handle) return;
	FIL *pfil = (FIL *)handle;
	f_close(pfil);
	handle = NULL;
}

char *File::name()
{
	return filename;
}

unsigned long File::size()
{
	return filesize;
}

unsigned char File::read()
{
	unsigned char uc;
	if(!handle) return 0;
	if(eof) return 0;
	if(rbuf_rptr >= rbuf_datacount)
	{
		if(!readbuf()) return 0;
		rbuf_rptr = 0;
	}

	uc = rbuf[rbuf_rptr++];
	rpos++;
	return uc;
}

// 1バイト書き込み
void File::write(unsigned char c)
{
	unsigned int bytes;
	if(!handle) return;

	f_write((FIL *)handle , &c, 1, &bytes);
}

// バッファ書き込み
void File::write(unsigned char *buf,int len)
{
	unsigned int bytes;
	if(!handle) return;

	f_write((FIL *)handle , buf, len, &bytes);
}

void File::write(const char *str)
{
	unsigned int bytes;
	if(!handle) return;
	char *p = (char *)str;
	while(*p)
	{
		f_write((FIL *)handle , p, 1, &bytes);
		p++;
	}
}

void File::print(int val, PRINT_TYPE print_type)
{
	unsigned long tmpval;
	char buf[33];
	int i;

	if(print_type == BYTE){
		write((char)(val & 0xff));
		return;
	}

	if(print_type == DEC) {
		tmpval = val;
		buf[11] = '\0';
		if(val < 0)
		{
			write('-');
			tmpval = -val;
		}

		for(i = 10; i > 0 ; i--)
		{
			buf[i] = (tmpval % 10) | 0x30;
			tmpval = tmpval / 10;
			if(tmpval == 0) break;
		}
		write(&buf[i]);
	}
	if(print_type == OCT) {
		tmpval = val;

		buf[12] = '\0';
		for(i = 11; i > 0 ; i--)
		{
			buf[i] = (tmpval & 7) + '0';
			tmpval >>= 3;
			if(tmpval == 0) break;
		}
		write(&buf[i]);
	}
	if(print_type == HEX) {
		tmpval = val;

		buf[9] = '\0';
		for(i = 8; i > 0 ; i--)
		{
			if((tmpval & 15) >= 10) buf[i] = (tmpval & 15) + 'A' - 10;
			else                    buf[i] = (tmpval & 15) + '0';
			tmpval >>= 4;
			if(tmpval == 0) break;
		}
		write(&buf[i]);
	}
	if(print_type == BIN) {
		tmpval = val;

		buf[32] = '\0';
		for(i = 32; i > 0 ; i--)
		{
			buf[i] = (tmpval & 1) ? '1' : '0';
			if(tmpval == 0) break;
		}
		write(&buf[i]);
	}
}

void File::println(int val, PRINT_TYPE print_type)
{
	print(val,print_type);
	write('\r');
	write('\n');
}

void File::print(double val, int fpdigit)
{
	write("<浮動小数点のprintは対応していません>");
}

void File::println(double val, int fpdigit)
{
	print(val,fpdigit);
	write('\r');
	write('\n');
}

void File::print(const char *str)
{
	write(str);
}

void File::println(const char *str)
{
	write(str);
	write('\r');
	write('\n');
}

bool File::readbuf()
{
	unsigned int bytes = FILE_BUFFER_SIZE;
	if(f_read((FIL *)handle , rbuf, FILE_BUFFER_SIZE, &bytes) == FR_OK)
	{
		rbuf_datacount = bytes;
	}
	else
	{
		rbuf_datacount = 0;
	}

	if(rbuf_datacount == 0)
	{
		eof = true;
		return false;
	}
	return true;
}

bool File::seek(unsigned long pos)
{
	FRESULT result;
	if(!handle) return false;

	result = f_lseek((FIL *)handle ,pos); // 末尾に移動
	if(result == FR_OK)
	{
		rpos = pos;
		return true;
	}
	return false;
}

void File::flush()
{
	if(!handle) return;

	f_sync((FIL *)handle);
}

unsigned long File::position()
{
	if(!handle) return 0;
	return rpos;
}

bool File::isDirectory()
{
	if(!handle) return false;
	return isdir;
}

File File::openNextFile()
{
	FILINFO info;
	if(!handle) return File(NULL,0,"",0,parent);
	if(!isdir) return File(NULL,0,"",0,parent); // それはファイルだ

	DIR *pdir = (DIR *)handle;
	f_readdir (pdir, &info);
	
	Serial.println("----");
	if(!(info.fattrib & AM_DIR))
	{
		if(parent) return parent->open(info.fname,FILE_READ);
	}

	return File(NULL,0,"",0,parent);
}

void File::rewindDirectory()
{
	FILINFO info;
	if(!handle) return;
	if(!isdir) return;
	parent->open(filename, FILE_READ);
}
