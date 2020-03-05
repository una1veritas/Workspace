/*
 * dirlister.h
 *
 *  Created on: 2020/03/01
 *      Author: Sin Shimozono
 */

#ifndef DIRLISTER_WIN_H_
#define DIRLISTER_WIN_H_

#include <cstdlib>
#include <io.h>

#include <string>
#include <deque>
#include <regex>

/*
 * unsigned attrib;        // ファイル属性。
 * time_t   time_write;    // 最新ファイル書き込み時刻
 * _fsize_t size;          // ファイルの長さ (バイト数)
 * char     name[260];     // ファイルまたはディレクトリの名前
 *
 * _A_HIDDEN(0x02) // 隠しファイル
 * _A_NORMAL(0x00) // 通常ファイル
 * _A_RDONLY(0x01) // 読み取り専用
 * _A_SUBDIR(0x10) // サブディレクトリ
 * _A_SYSTEM(0x04) // システム ファイル。
 *
 */

class dirlister {
	struct lister {
		const std::string cdir;
		struct _finddata_t fdata;
		intptr_t fhandl;
		bool opened;

		lister(const std::string path) : cdir(path), fhandl(0), opened(false) {}
	};
	std::deque<lister> spath;

public:
	dirlister(std::string basedir) {
		if ( basedir.back() == '/' )
			basedir.pop_back();
		spath.push_back(lister(basedir));
	}

	~dirlister() {
		while( ! finished() )
			close_dir();
	}

	void close_dir() {
		findclose(spath.back().fhandl);
		spath.pop_back();
	}

	bool finished() {
		return spath.empty();
	}

	bool get_next_entry(const std::regex & fnpattern, const bool skipdotfile = true) {
		bool result;
		std::string path;
		while ( !spath.empty() ) {
			if ( ! spath.back().opened ) {
				path = spath.back().cdir + "/*.*";
				spath.back().fhandl = _findfirst(path.c_str(), &spath.back().fdata);
				spath.back().opened = true;
				result = (spath.back().fhandl != -1);
			} else {
				result = (_findnext(spath.back().fhandl, &spath.back().fdata) == 0);
			}
			if ( ! result ) {
				if ( !spath.empty() ) {
					//std::cout << "exit dir" << std::endl;
					close_dir();
				}
				continue;
			}
			if ( skipdotfile && entry_name()[0] == '.' ) {
				continue;
			}
			if ( entry_is_dir() ) {
				if ( entry_name() == "." || entry_name() == ".." ) {
					continue;
				}
				path = entry_name();
				//std::cout << "enter dir: " << path << std::endl;
				spath.push_back(lister(path));
				continue;
			}
			if ( std::regex_match(entry_name(), fnpattern) )
				return true;
		}
		return false;
	}

	bool entry_is_dir() const {
		return ((spath.back().fdata.attrib & _A_SUBDIR) != 0 );
	}

	const std::string entry_name() const {
		 return std::string(spath.back().fdata.name);
	}

	const std::string entry_fullpath() const {
		std::string tmp = "";
		for (auto i : spath) {
			if ( tmp.size() > 0 )
				tmp += "/";
			tmp += i.cdir ;
		}
		if ( tmp.size() > 0 )
			return tmp + "/" + entry_name();
		return entry_name();
	}
};

#endif /* DIRLISTER_WIN_H_ */
