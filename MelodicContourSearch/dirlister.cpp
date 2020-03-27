/*
 * dirlister.cpp
 *
 *  Created on: 2020/03/17
 *      Author: Sin Shimozono
 */

#include "dirlister.h"

const std::string dirlister::fullpath() const {
	std::string tmp = "";
	for (auto i : spath) {
		if ( tmp.size() > 0 )
			tmp += "/";
		tmp += i.cdir ;
	}
	return tmp;
}

bool dirlister::get_next_entry(const std::regex &fnpattern, const bool skipdotfile) {
	bool result;
	std::string path;
	while (!spath.empty()) {
		result = first_or_next_entry();
		if (!result) {
			if (!spath.empty()) {
				//std::cout << "exit dir" << std::endl;
				close_dir();
			}
			continue;
		}
		if (skipdotfile && entry_name()[0] == '.') {
			continue;
		}
		if (entry_is_dir()) {
			if (entry_name() == "." || entry_name() == "..") {
				continue;
			}
			//path = entry_name();
			//std::cout << "enter dir: " << path << std::endl;
			spath.push_back(lister(entry_name()));
			continue;
		}
		if (std::regex_match(entry_name(), fnpattern))
			return true;
	}
	return false;
}

