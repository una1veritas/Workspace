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

#ifndef	__H_RXDUINO_SDMMC
#define	__H_RXDUINO_SDMMC

#include "rxduino.h"

#ifdef __cplusplus
	extern "C" {
#endif

#define FILE_BUFFER_SIZE 512

#define FILE_READ  0x01
#define FILE_WRITE 0x02

typedef int FILE_MODE;

class SDMMC;

class File
{
private:
	unsigned char rbuf[FILE_BUFFER_SIZE];
	int rbuf_rptr;
	int rbuf_datacount;
	bool eof;
	int rpos;

	char wbuf[FILE_BUFFER_SIZE];

	bool readbuf(); // ファイルからメモリ上のバッファへ読み込む

	void *handle;
	int filesize;
	char filename[16];
	bool isdir;
	SDMMC *parent; 

public:

	typedef enum {
		NONE,	/*!< 指定無し */
		BYTE,	/*!< アスキーコードでの表示 */
		BIN,	/*!< 2進数での表示 */
		OCT,	/*!< 8進数での表示 */
		DEC,	/*!< 10進数での表示 */
		HEX		/*!< 16進数での表示 */
	} PRINT_TYPE;

	// ファイルを閉じる。(最後に必ず呼び出すこと！)
	void close();

	// 読み出せるデータがあるか
	bool available();

	// 1バイト読み込み
	unsigned char read();

	// 1バイト書き込み
	void write(unsigned char c);

	// バッファ書き込み
	void write(const char *str);
	void write(unsigned char *buf,int len);


	void flush();
	unsigned char peek();
	unsigned long position();

	bool seek(unsigned long pos);

	void print(int val, PRINT_TYPE print_type=DEC);
	void print(double val, int fpdigit=2);
	void print(const char *str);
	void println(int val, PRINT_TYPE print_type=DEC);
	void println(double val, int fpdigit=2);
	void println(const char *str);

	// ファイルのサイズ
	unsigned long size();

	// ファイルの名前
	char *name();

	bool isDirectory();
	File openNextFile();
	void rewindDirectory();

	// 以下の関数やメンバ変数には、ユーザは操作しないこと
	File(void *handle,int size,const char *name,bool isdir,SDMMC *parent);
	operator int() {if(!handle) return 0; return 1;}
	operator bool() {if(!handle) return false; return true;}
	
};

// SDカードやMMCカードを操作する
class SDMMC 
{
private:
	void *sdmmc_handle;

public:

	SDMMC();
	void begin();
	void begin(int cspin);
	void insert(bool ins);
	bool exists(const char *filename);
	bool mkdir(const char *filename);
	File open(const char *filename, FILE_MODE mode = FILE_READ);
	bool remove(const char *filename);
	bool rename(const char *oldname, const char *newname);
	bool rmdir(const char *filename);
};

#ifdef __cplusplus
	}
#endif

#endif
