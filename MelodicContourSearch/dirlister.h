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
 * unsigned attrib;        // �t�@�C�������B
time_t   time_write;    // �ŐV�t�@�C���������ݎ���
_fsize_t size;          // �t�@�C���̒��� (�o�C�g��)
char     name[260];     // �t�@�C���܂��̓f�B���N�g���̖��O
 *
 * _A_HIDDEN(0x02) // �B���t�@�C��
_A_NORMAL(0x00) // �ʏ�t�@�C��
_A_RDONLY(0x01) // �ǂݎ���p
_A_SUBDIR(0x10) // �T�u�f�B���N�g��
_A_SYSTEM(0x04) // �V�X�e�� �t�@�C���B
 */
};

#endif /* DIRLISTER_H_ */
