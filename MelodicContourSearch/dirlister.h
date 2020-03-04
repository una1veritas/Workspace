#ifndef _DIRLISTER_H_
#define _DIRLISTER_H_

#include <iostream>
#include <cstdlib>
#include <deque>
#include <string>
#include <regex>

#if defined(__linux__) || defined(__MACH__)
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
#endif

struct dirlister {
	struct lister {
		const std::string dirpath;
#if defined(__linux__) || defined(__MACH__)
		DIR * dir;
		dirent * entry;
#elif defined(__WIN64)
		struct _finddata_t fdata;
		intptr_t fhandl;
#endif
		bool opened;

		lister(const std::string path) : dirpath(path), opened(false) {
#if defined(__linux__) || defined(__MACH__)
			dir = NULL;
			entry = NULL;
#elif defined(__WIN64)
			fhandl = NULL;
#endif
		}
	};
	std::deque<lister> spath;

	dirlister(const std::string basedir) {
		std::string path = basedir;
		if (path.back() == '/')
			path.pop_back();
		spath.push_back(lister(path));
	}

	~dirlister() {
		while( ! finished() )
			close_dir();
	}

	void close_dir() {
#if defined(__linux__) || defined(__MACH__)
		closedir(spath.back().dir);
#endif
		spath.pop_back();
	}

	bool finished() {
		return spath.empty();
	}

	bool entry_is_dir() const {
#if defined(__linux__) || defined(__MACH__)
		return spath.back().entry->d_type == DT_DIR;
#endif
		//return ((spath.back().fdata.attrib & _A_SUBDIR) != 0 );
	}

	const std::string entry_name() const {
#if defined(__linux__) || defined(__MACH__)
		return std::string(spath.back().entry->d_name);
#endif
		//return std::string(spath.back().fdata.name);
	}

	const std::string entry_fullpath() const {
		std::string tmp = "";
		/*
		auto spathptr = spath.begin();
		if ( spathptr->dirpath != "." ) {
			tmp += spathptr->dirpath + "/";
		}
		++spathptr;
		while ( spathptr != spath.end() ) {
			tmp += spathptr->dirpath + "/";
			++spathptr;
			//std::cout << tmp << " > ";
		}
		*/
		for(auto elem : spath) {
			if ( elem.dirpath == "." )
				continue;
			tmp += elem.dirpath + "/";
			std::cout << tmp << " > ";
		}
		return tmp + entry_name();
	}

	bool get_next_entry(const std::regex & fnpattern, const bool skipdotfile = true) {
		bool result;
		std::string path;
		while ( !spath.empty() ) {
			if ( ! spath.back().opened ) {
#if defined(__linux__) || defined(__MACH__)
				spath.back().dir = opendir(spath.back().dirpath.c_str());
				std::cout << "opendir " << spath.back().dirpath <<  std::endl;
				spath.back().opened = true;
				result = (spath.back().dir != NULL);
				if ( result ) {
					spath.back().entry = readdir(spath.back().dir);
				}
#elif defined(__WIN64)
				path = spath.back().dirpath + "/*.*";
				spath.back().fhandl = _findfirst(path.c_str(), &spath.back().fdata);
				spath.back().opened = true;
				result = (spath.back().fhandl != -1);
#endif
			} else {
#if defined(__linux__) || defined(__MACH__)
				std::cout << "readdir " << spath.back().dirpath << std::endl;
				spath.back().entry = readdir(spath.back().dir);
				result = (spath.back().entry != NULL);
				/*
				if ( result ) {
					std::cout << "entry_name " << entry_name() << std::endl;
				} else {
					std::cout << "result " << result << std::endl;
				}
				*/
#elif defined(__WIN64)
				result = (_findnext(spath.back().fhandl, &spath.back().fdata) == 0);
#endif
			}
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
				path = spath.back().dirpath + "/" + entry_name();
				std::cout << "enter dir: " << path << std::endl;
				spath.push_back(lister(path));
				continue;
			}
			if ( std::regex_match(entry_name(), fnpattern) ) {
				return true;
			}
		}
		return false;
	}

	/*
	dirent * get_next_file(const std::regex & fpattern) {
		std::string subdir;
		for(;;) {
			entry = readdir(spath.back().second);
			if ( entry == NULL ) {
				closedir(spath.back().second);
				spath.pop_back();
				if ( spath.empty() )
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
					spath.push_back(pathdir(subdir, dp));
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
		return entry;
	}
	*/

};

#endif
