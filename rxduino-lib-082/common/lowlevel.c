// RX62NのGCCサンプルプログラム
// 低レベルI/Oのサポート このルーチンを使うとprintfやmallocが使えるようになる
// 
// (C)Copyright 2011 特殊電子回路

#include <stdio.h>
#include <errno.h>
#include <sys/stat.h> 
#include "tkdn_sci.h"

char * _sbrk (int adj) {
  extern char    end;
  static char *  heap = & end;
  char *         rv = heap;

  heap += adj;
  return rv;
}

char * sbrk (int) __attribute__((weak, alias ("_sbrk")));

int open(const char *pathname, int flags, mode_t mode)
{
	return 1;
}

ssize_t read(int fd, void *buf, size_t count)
{
	// この関数はcountが0でないなら、0を返してはいけない
	// 0はファイルの終わりを意味する。
	// 読み出すデータがない場合は、待たなければならない
	size_t bytes;
	int i;
	unsigned char *cbuf = (unsigned char *)buf;

	if(count == 0) return 0;
	if(fd != 0)
	{
		errno = EBADF;
		return -1; // fd が有効なファイル・ディスクリプターでない
	}
	
	do
	{
		bytes  = sci_rxcount();
	} while(bytes == 0);
	if(bytes == 0)
	{
		errno = EAGAIN; // ブロッキングしないでね
		return -1; // fd が有効なファイル・ディスクリプターでない
	}

	if(bytes > count) bytes = count;

	count = 0;
	char prev = 0;
	for(i=0;i<bytes;i++)
	{
		char c = sci_getc();
		if((prev == 0x0d) && (c == 0x0a)) // CR+LF受信
		{
			cbuf[count] = 0x0a;	count++;
			prev = 0;
			continue;
		}
		if((prev != 0x0d) && (c == 0x0a)) // LF単独受信
		{
			cbuf[count] = 0x0a;	count++;
			prev = 0;
			continue;
		}
		if((i == bytes-1) && (c == 0x0d)) // CR単独受信
		{
			cbuf[count] = 0x0a;	count++;
			prev = 0;
			continue;
		}
		cbuf[count] = c;	count++;
		prev = c;
	}

	return count;
}

ssize_t write(int fd, void *buf, size_t count)
{
	int len = count;
	char *p = (char *)buf;

	if(fd == 1) // STDOUT
	{
		while(len)
		{
			if(sci_putc(*p) == 0) continue;
			len--;
			p++;
		}
		return count; // 成功
	}
	if(fd == 2) // STDERR
	{
		while(len)
		{
			if(sci_putc(*p) == 0) continue;
			len--;
			p++;
		}
		return count; // 成功
	}
//	errno = EBADF;
	return -1;
}

off_t lseek(int fd, off_t offset, int whence)
{
	return 0;
}

int isatty(int fd)
{
	return 1;
}

int fstat(int fd, struct stat *buf)
{
	if(fd == 0)
	{
		buf->st_dev = 327681;
		buf->st_ino = 600457966;
		buf->st_mode = 0x21b6;
		buf->st_nlink = 1;
		buf->st_uid = 0;
		buf->st_gid = 0;
		buf->st_rdev = 327681;
		buf->st_size = 0;
		buf->st_blksize = 65536;
		buf->st_blocks = 0;
		return 0;
	}
		
	if((fd == 1) || (fd == 2)) return 0; // 成功
	return -1; // 失敗
}

int close(int fd)
{
	return 0;
}
