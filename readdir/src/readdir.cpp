#include <iostream>
#include <cstdlib>
#include <dirent.h>

int main(int argc, char *argv[]) {
	if (argc == 1) {
		std::cout << "give me a path." << std::endl;
		exit(1);
	}
	char *path = argv[1];

	DIR *dp;       // �f�B���N�g���ւ̃|�C���^
	dirent *entry; // readdir() �ŕԂ����G���g���[�|�C���g

	dp = opendir(path);
	if (dp == NULL)
		exit(1);
	do {
		entry = readdir(dp);
		if (entry != NULL)
			std::cout << path << entry->d_name << std::endl;
	} while (entry != NULL);
	return 0;
}
