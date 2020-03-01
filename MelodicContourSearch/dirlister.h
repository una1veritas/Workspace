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

#include <string>
#include <deque>
#include <regex>

class dirlister {
	struct lister {
		const std::string dirpath;
		struct _finddata_t fdata;
		intptr_t fhandl;
		bool opened;

		lister(const std::string path) : dirpath(path), fhandl(0), opened(false) {}
	};
	const std::regex fpat;
	std::deque<lister> spath;

public:
	dirlister(const std::string & basedir, const std::regex pattern) :fpat(pattern) {
		std::string path = basedir + "/*.*";
		spath.push_back(lister(path));
	}

	~dirlister() {
		while( ! finished() )
			close_dir();
	}

	void close_dir() {
		_findclose(spath.back().fhandl);
		spath.pop_back();
	}

	bool finished() {
		return spath.empty();
	}

	bool get_next_entry() {
		bool result;
		while ( !spath.empty() ) {
			if ( ! spath.back().opened ) {
				spath.back().fhandl = _findfirst(spath.back().dirpath.c_str(), &spath.back().fdata);
				spath.back().opened = true;
				result = (spath.back().fhandl != -1);
			} else {
				result = (_findnext(spath.back().fhandl, &spath.back().fdata) == 0);
			}
			if ( ! result ) {
				if ( !spath.empty() ) {
					close_dir();
				}
				continue;
			}
			if ( entry_is_dir() ) {
				if ( entry_name() == "." || entry_name() == ".." ) {
					continue;
				}
				std::cout << "enter dir: " << entry_name() << std::endl;

			}
			if ( std::regex_match(entry_name(), fpat) )
				return true;
		}
		return false;
	}

	bool operator++(int) {
		return _findnext(spath.back().fhandl, &spath.back().fdata) == 0;
	}

	bool entry_is_dir() const {
		return ((spath.back().fdata.attrib & _A_SUBDIR) != 0 );
	}

	const std::string entry_name() const {
		 return std::string(spath.back().fdata.name);
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
