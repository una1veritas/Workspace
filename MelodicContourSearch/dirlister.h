/*
 * dirlister.h
 *
 *  Created on: 2020/03/01
 *      Author: Sin Shimozono
 */

#ifndef DIRLISTER_H_
#define DIRLISTER_H_

#include <cstdlib>
#include <io.h>

struct dirlister {
	const char * path;
	struct _finddata_t fdata;
	intptr_t fh;

	dirlister(const char * pstr) : path(pstr), fh(0) { }

	~dirlister() {
		_findclose(fh);
	}

	bool get_next_entry() {
		if ( fh == 0 ) {
			fh = _findfirst(path, &fdata);
			return (fh != -1);
		}
		return ( _findnext(fh, &fdata) == 0 );
	}

	bool entry_is_dir() const {
		return ((fdata.attrib & _A_SUBDIR) != 0 );
	}

	const char * entry_name() const {
		 return fdata.name;
	}

/*
 *
 * unsigned attrib;        // ファイル属性。
time_t   time_write;    // 最新ファイル書き込み時刻
_fsize_t size;          // ファイルの長さ (バイト数)
char     name[260];     // ファイルまたはディレクトリの名前
 *
 * _A_HIDDEN(0x02) // 隠しファイル
_A_NORMAL(0x00) // 通常ファイル
_A_RDONLY(0x01) // 読み取り専用
_A_SUBDIR(0x10) // サブディレクトリ
_A_SYSTEM(0x04) // システム ファイル。
 */
};

#endif /* DIRLISTER_H_ */
