/*
 * dirlister.h
 *
 *  Created on: 2020/03/05
 *      Author: sin
 */

#ifndef DIRLISTER_H_
#define DIRLISTER_H_

#include <iostream>
#include <cstdlib>

#include <string>
#include <deque>
#include <regex>

#if defined(__linux__) || defined(__MACH__)
/*
#include "dirlister_unix.h"
*/

#include <dirent.h>

class dirlister {
	struct lister {
		const std::string cdir;
		DIR * dir;
		dirent * entry;
		bool opened;

		lister(const std::string path) : cdir(path), opened(false) {
			dir = NULL;
			entry = NULL;
		}
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
		closedir(spath.back().dir);
		spath.pop_back();
	}
/*
	bool open_dir() {
		if ( ! spath.back().opened ) {
			spath.back().dir = opendir(fullpath().c_str());
			//std::cout << "opendir " << spath.back().cdir <<  std::endl;
			spath.back().opened = (spath.back().dir != NULL);
		}
		return spath.back().opened;
	}
*/
	bool first_or_next_entry() {
		if ( ! spath.back().opened ) {
			spath.back().dir = opendir(fullpath().c_str());
			//std::cout << "opendir " << spath.back().cdir <<  std::endl;
			spath.back().opened = (spath.back().dir != NULL);
			if ( ! spath.back().opened ) {
				/* dir open failed. */
				return false;
			}
		}
		spath.back().entry = readdir(spath.back().dir);
		return (spath.back().entry != NULL);
	}

	bool finished() {
		return spath.empty();
	}

	bool entry_is_dir() const {
		return spath.back().entry->d_type == DT_DIR;
		//return ((spath.back().fdata.attrib & _A_SUBDIR) != 0 );
	}

	const std::string entry_name() const {
		return std::string(spath.back().entry->d_name);
		//return std::string(spath.back().fdata.name);
	}

	const std::string entry_fullpath() const {
		return fullpath() + "/" + entry_name();
	}

	const std::string fullpath() const {
		std::string tmp("");
		for(auto i : spath) {
			if ( tmp.size() > 0 )
				tmp += "/";
			tmp += i.cdir;
		}
		return tmp;
	}

	bool get_next_entry(const std::regex & fnpattern, const bool skipdotfile = true) {
		bool result;
		std::string path;
		while ( !spath.empty() ) {
			result = first_or_next_entry();
			if ( ! result ) {
				if ( !spath.empty() ) {
					//std::cout << "exit dir: " << spath.back().dirpath << std::endl;
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
				//path = spath.back().cdir + "/" + entry_name();
				//std::cout << "enter dir: " << path << std::endl;
				/* enter dir */
				spath.push_back(lister(entry_name()));
				continue;
			}
			if ( std::regex_match(entry_name(), fnpattern) ) {
				return true;
			}
		}
		return false;
	}


};

#elif defined(__WIN64)
/* #include "dirlister_win.h"
*/
#include <io.h>
#endif

class dirlister {
	struct lister {
		const std::string cdir;
#if defined(__linux__) || defined(__MACH__)
		DIR * dir;
		dirent * entry;
#elif defined(__WIN64)
		struct _finddata_t fdata;
		intptr_t fhandl;
#endif
		bool opened;

#if defined(__linux__) || defined(__MACH__)
		lister(const std::string path) : cdir(path), opened(false) {
			dir = NULL;
			entry = NULL;
		}
#elif defined(__WIN64)
		lister(const std::string path) : cdir(path), fhandl(0), opened(false) {}
#endif
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

#if defined(__linux__) || defined(__MACH__)
	bool first_or_next_entry() {
		if ( ! spath.back().opened ) {
			spath.back().dir = opendir(fullpath().c_str());
			//std::cout << "opendir " << spath.back().cdir <<  std::endl;
			spath.back().opened = (spath.back().dir != NULL);
			if ( ! spath.back().opened ) {
				/* dir open failed. */
				return false;
			}
		}
		spath.back().entry = readdir(spath.back().dir);
		return (spath.back().entry != NULL);
	}

	void close_dir() {
		closedir(spath.back().dir);
		spath.pop_back();
	}

	bool entry_is_dir() const {
		return spath.back().entry->d_type == DT_DIR;
		//return ((spath.back().fdata.attrib & _A_SUBDIR) != 0 );
	}

	const std::string entry_name() const {
		return std::string(spath.back().entry->d_name);
		//return std::string(spath.back().fdata.name);
	}
#elif defined(__WIN64)

	bool first_or_next_entry() {
		std::string path;
		if (!spath.back().opened) {
			path = fullpath() + "/*.*";
			spath.back().fhandl = _findfirst(path.c_str(), &spath.back().fdata);
			spath.back().opened = (spath.back().fhandl != -1);
			if ( ! spath.back().opened ) {
				std::cerr << "open failed " << path << std::endl;
			}
			return spath.back().opened;
		} else {
			return (_findnext(spath.back().fhandl, &spath.back().fdata) == 0);
		}
	}

	void close_dir() {
		_findclose(spath.back().fhandl);
		spath.pop_back();
	}

	bool entry_is_dir() const {
		return ((spath.back().fdata.attrib & _A_SUBDIR) != 0 );
	}

	const std::string entry_name() const {
		 return std::string(spath.back().fdata.name);
	}
#endif

	bool finished() {
		return spath.empty();
	}

	const std::string fullpath() const;

	bool get_next_entry(const std::regex & fnpattern, const bool skipdotfile = true);

};


#endif /* DIRLISTER_H_ */
