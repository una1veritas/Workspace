#ifndef FATFSTEST_H
#define FATFSTEST_H

// C����ŏ����ꂽ���C�u�����̃w�b�_�t�@�C�����C���N���[�h����
#ifdef __cplusplus
extern "C" {
#endif

void fatfs_init();
void fatfs_dir(const char *path);
void fatfs_cd(const char *path);
void fatfs_type(const char *filename);
void fatfs_loadmot(const char *filename);
void fatfs_getcwd(char *buffer,int maxlen);

#ifdef __cplusplus
}
#endif

#endif
