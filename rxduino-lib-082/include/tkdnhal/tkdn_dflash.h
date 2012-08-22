/*******************************************************************************
* RXduino���C�u���� & ���dHAL
* 
* ���̃\�t�g�E�F�A�͓���d�q��H������Ђɂ���ĊJ�����ꂽ���̂ł���A���А��i��
* �T�|�[�g�Ƃ��Ē񋟂���܂��B���̃��C�u�����͓��А��i����ѓ��Ђ����C�Z���X����
* ���i�ɑ΂��Ďg�p���邱�Ƃ��ł��܂��B
* ���̃\�t�g�E�F�A�͂��邪�܂܂̏�ԂŒ񋟂���A���e����ѓ���ɂ��Ă̕ۏ�͂�
* ��܂���B���Ђ̓t�@�C���̓��e����ю��s���ʂɂ��Ă����Ȃ�ӔC�������܂���B
* ���q�l�́A���q�l�̐��i�J���̂��߂ɓ��\�t�g�E�F�A�̃\�[�X�R�[�h�����R�ɎQ�Ƃ��A
* ���p���Ă����������Ƃ��ł��܂��B
* ���̃t�@�C����P�̂ő�O�҂֊J���E�Ĕz�z�E�ݗ^�E���n���邱�Ƃ͂ł��܂���B
* �R���p�C���E�����N��̃I�u�W�F�N�g�t�@�C��(ELF �t�@�C���܂���MOT,SREC�t�@�C��)
* �ł����āA�f�o�b�O��񂪍폜����Ă���ꍇ�͑�O�҂ɍĔz�z���邱�Ƃ��ł��܂��B
* (C) Copyright 2011-2012 TokushuDenshiKairo Inc. ����d�q��H�������
* http://rx.tokudenkairo.co.jp/
*******************************************************************************/

#ifndef __H_TKDN_DFLASH
#define __H_TKDN_DFLASH

#include "tkdn_dflash.h"

#ifdef __cplusplus
extern "C" {
#endif

//��������������������������������������������������
//  ���[�U���J�X�^�}�C�Y����ꏊ�͂���܂���
//��������������������������������������������������

TKDN_HAL
int tkdn_dflash_write(unsigned long offset,unsigned char *buf,int len);

TKDN_HAL
int tkdn_dflash_read(unsigned long offset,unsigned char *buf,int len);

TKDN_HAL
int tkdn_dflash_erase(unsigned long offset,int len);

// �t���b�V���������݃��[�h���I������
TKDN_HAL
void tkdn_dflash_terminate(void);

// �u�����N�`�F�b�N �u�����N�Ȃ�1 �������܂�Ă����0 �G���[�Ȃ�-1
TKDN_HAL
int tkdn_dflash_blank(unsigned long offset,int len);

#ifdef __cplusplus
 }
#endif

#endif // __H_TKDN_TIMER
