#ifndef _DIRLISTER_UNIX_H_
#define _DIRLISTER_UNIX_H_

#include <iostream>
#include <cstdlib>
#include <deque>
#include <string>
#include <regex>

#include <dirent.h>
/* dirent.h
#define DT_UNKNOWN       0
#define DT_FIFO          1
#define DT_CHR           2
#define DT_DIR           4
#define DT_BLK           6
#define DT_REG           8
#define DT_LNK          10
#define DT_SOCK         12
#define DT_WHT          14
*/

struct dirlister {
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

#endif
