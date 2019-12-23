#ifndef _DIRECTORYLISTER_H_
#define _DIRECTORYLISTER_H_

#include <iostream>
#include <cstdlib>
#include <deque>
#include <regex>
#include <dirent.h>

struct DirectoryLister {
	typedef std::pair<std::string,DIR*> pathdir;
	std::deque<pathdir> pathdirs;
	dirent * lastentry;
/* dirent.h で定義される
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

	DirectoryLister(const char *path) : pathdirs(), lastentry(NULL) {
		std::string rootpath(path);
		if ( rootpath.size() != 0 and rootpath[rootpath.size()-1] == '/' )
			rootpath.pop_back();
		DIR *dp = opendir(rootpath.c_str());
		if (dp != NULL) {
			// if dp == NULL then an error has occurred.
			pathdirs.push_back(pathdir(rootpath, dp));
		}
	}

	bool operator()() const {
		return !pathdirs.empty();
	}

	dirent * get_next_file(const char * regpat = ".*") {
		std::regex regexpr(regpat);
		return get_next_file(regexpr);
	}

	dirent * get_next_file(const std::regex & fpattern) {
		std::string subdir;
		for(;;) {
			lastentry = readdir(pathdirs.back().second);
			if ( lastentry == NULL ) {
				closedir(pathdirs.back().second);
				pathdirs.pop_back();
				if ( pathdirs.empty() )
					break;
				continue;
			} else if ( entry_isdir() ) {
				if ( strcmp(entry_name(),".") == 0 or strcmp(entry_name(),"..") == 0 )
					continue;
				subdir.clear();
				subdir += entry_basepath();
				subdir += '/';
				subdir += entry_name();
				DIR * dp = opendir(subdir.c_str());
				//std::cout << " subdir = '" << subdir.c_str() << "'" << std::endl;
				if ( dp == NULL ) {
					std::cerr << "error: directory '" << subdir.c_str() << "' open failed." << std::endl;
					return NULL;
				} else {
					pathdirs.push_back(pathdir(subdir, dp));
					continue;
				}
			} else if ( entry_isreg() ) {
				if ( std::regex_match(entry_name(), fpattern) ) {
					break;
				}
				continue;
			} else {
				//std::cout << "unknown d_type " << node_type() << std::endl;
				continue;
			}
		}
		return lastentry;
	}

	int entry_type() const {
		if ( lastentry == NULL )
			return DT_UNKNOWN;
		return lastentry->d_type;
	}

	bool entry_isdir() const {
		return entry_type() == DT_DIR;
	}

	bool entry_isreg() const {
		return entry_type() == DT_REG;
	}

	const char * entry_name() const {
		return lastentry->d_name;
	}

	const char * entry_basepath() const {
		return pathdirs.back().first.c_str();
	}

	const std::string entry_path() const {
		std::string t;
		t += entry_basepath();
		t += "/";
		t += entry_name();
		return t;
	}
};

#endif
