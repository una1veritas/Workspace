#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

/* #include <mysql.h> */

#include "mysql_io.h"

#define MAXNAME  100
#define DESIRE_LIMIT   4

/* ���s���[�h�̐ݒ� */
typedef struct Config_ {
	int MODE;  /* ���s���[�h�Bprintconfig�֐����Q�ƁB */
	int WIDTH; /* �������쐬���T���@���s���� */
	int DEPTH; /* �T���̐[���̌��E */
	int SAVE;  /* �ێ�������̐� */
	int SEED;  /* �����_���֐���seed */
} Config;

/* �v�]�ᔽ�̃y�i���e�B�̏d�� */
typedef struct Weight_ {
	int PREFER;
	int R_WEEKOVER;  /* ��΍u�t�̈�T�Ԃ̒S������ */
	int R_WEEKUNDER; /* ��΍u�t�̈�T�Ԃ̒S���s�� */
	int R_DAYSOVER;   /* ��΍u�t�̒S���������� */
	int R_DAYSUNDER;  /* ��΍u�t�̒S�������s�� */
//	int R_DAYOVER;   /* ��΍u�t�̒S�����̒S������ */
//	int R_DAYUNDER;  /* ��΍u�t�̒S�����̒S���s�� */
//	int R_DAYS;      /* ��΍u�t�̕������S�� */
	int R_HOLE;      /* ��΍u�t�̋󂫎��� */
	int R_DESIRE;    /* ��΍u�t�̒S���s���x */
	int N_WEEKOVER;  /* ���΍u�t�̈�T�Ԃ̒S������ */
	int N_WEEKUNDER; /* ���΍u�t�̈�T�Ԃ̒S���s�� */
	int N_DAYSOVER;   /* ���΍u�t�̒S���������� */
	int N_DAYSUNDER;  /* ���΍u�t�̒S�������s�� */
//	int N_DAYOVER;   /* ���΍u�t�̒S�����̒S������ */
//	int N_DAYUNDER;  /* ���΍u�t�̒S�����̒S���s�� */
//	int N_DAYS;      /* ���΍u�t�̕������S�� */
	int N_HOLE;      /* ���΍u�t�̋󂫎��� */
	int N_DESIRE;    /* ���΍u�t�̒S���s���x */
//	int J_N;         /* �w�肳�ꂽ�N���X�̑g�����{��u�`�^�p��u�`�ɓ񕪉�����Ă��Ȃ� */
//	int NRGOVER;     /* ���΍u�t�S�̂̒S������ */
//	int NRGUNDER;    /* ���΍u�t�S�̂̒S���s�� */
} Weight;

/*  */
typedef struct Num_ {
	int week;   /* ���� */
	int period; /* ����̍u�`���Ԃ̎�ށi�����̐��j */
	int time;   /* week * period */
	int lctnum; /* �u�`�̐� */
	int clsnum; /* �N���X�̐� */
//	int blgnum; /* ���{��u�`�^�p��u�`�ɓ񕪉������ׂ��N���X�̑g�̐� */
	int tchnum; /* �u�t�̐� */
	int qualificationnum;
} Num;

/* �u�` */
typedef struct Lecture_ {
//  char name[MAXNAME]; /* �u�`�� */
  int class;          /* ������N���X */
  int *avail;         /* �e���Ԃɂ�����J�u�\�^�s�\ */
	int qualification;  /* 0:Japanese based, 1:English based, 2:don't care*/
} Lecture;

/* �u�t */
typedef struct Teacher_ {
	int id;
//  char name[MAXNAME]; /* ���O */
  int *desire; /* �e���Ԃ̒S����]�x�i0�̏ꍇ�͒S���s�\�j */
  int regular; /* 0:��΍u�t 1:���΍u�t */
  int *qualification;  /* [0]:Japanese based, [1]:English based, [2]:don't care*/
  int week_up; /* ��T�Ԃ̒S����� */
  int week_lw; /* ��T�Ԃ̒S������ */
  int days_up;  /* �S�������̏�� */
  int days_lw;  /* �S�������̉��� */
//  int day_up;  /* �S�����ɂ�����S�����̏�� */
//  int day_lw;  /* �S�����ɂ�����S�����̉��� */
	int wage;
} Teacher;

/* �N���X */
typedef struct Class_ {
	int id;
  int num;   /* ������u�`�̐� */
  int *lcts; /* ������u�`�̔ԍ��̔z�� */
	int qualification_id;
} Class;

/* ���{��u�`�^�p��u�`�ɓ񕪉������ׂ��N���X�̑g */
//typedef struct Bilingual_ {
//  int cls1; /* �N���X�̔ԍ� */
//  int cls2; /* �N���X�̔ԍ� */
//} Bilingual;
  
/* ���Ԋ��i���j */
typedef int ***Timetable; /* �u�`lct������time�ɍu�ttch���J�u����:[lct][time][tch]=1 */

/* ����ᔽ */
typedef struct Offence_ {
	int null_lct;         /* �u�`���J�u����Ȃ� */
	int lct_overlap;      /* ����u�`�������̎��ԂɊJ�u����� */
	int class_separation; /* ����N���X�ɑ�����N���X�������ɊJ�u����Ȃ� */
	int tch_overlap;      /* ����u�t�������ɕ����̍u�`��S������ */
	int not_qualified;		/* �l�C�e�B�u�i��l�C�e�B�u�j�����{��x�[�X�i�p��x�[�X�j�̍u�`��S�� */
	int total;            /* ���v */
} Offence;

/* �v�]�ᔽ */
typedef struct Penalty_ {
	int *prefer;			/* �J�u�s���x */
	int *weekover;  /* ��T�Ԃ̒S������ */
	int *weekunder; /* ��T�Ԃ̒S���s�� */
	int *daysover;  /* �S���������� */
	int *daysunder; /* �S�������s�� */
//	int **dayover;  /* �S�����̒S������ */
//	int **dayunder; /* �S�����̒S���s�� */
//	int *days;      /* �������S�� */
	int *hole;      /* �󂫎��� */
	int *desire;    /* �S���s���x */
//	int j_n;        /* �w�肳�ꂽ�N���X�̑g�����{��u�`�^�p��u�`�ɓ񕪉�����Ă��Ȃ� */
//	int nrgover;    /* ���΍u�t�S�̂̒S������ */
//	int nrgunder;   /* ���΍u�t�S�̂̒S���s�� */
	int total;      /* ���v */
} Penalty;

/*  */
typedef struct Utility_ {
	int *lct_openat;
	int *lct_takenby;
	int **tch_takelct;
} Utility;


/*�t�@�C��config������s���[�h��ǂݍ��݁C�t�@�C��weight����v�]�ᔽ�y�i���e�B�̏d�݂�ǂݍ��݁C
�@�t�@�C��data����C���X�^���X�f�[�^��ǂݍ��݁C�z��̃��������m�ۂ���*/
void read_and_ready(
			FILE **fpi1, FILE **fpi2, /*FILE **fpi3,*/ FILE **fpi5, FILE **fpi6, FILE **fpi7, 
			Config *config, Weight *weight, Num *num, 
			Lecture **lct, Class **cls, /*Bilingual **blg,*/ Teacher **tch, /*int *nrgl_lects,*/ int **period_id, int **day_id, int **day,
			Timetable *f, Timetable *new_f, Penalty *penalty, Utility *utility, 
			Timetable **best_sol, int **best_off, int **best_val);

/*���s���[�h��\��*/
void printconfig(FILE *fpo, Config config);

/*�v�]�ᔽ�y�i���e�B�̒l��\��*/
void printweight(FILE *fpo, Weight weight);

/*�C���X�^���X��\��*/
void printdata(FILE *fpo, Num num, Lecture *lct, Class *cls, /*Bilingual *blg,*/ Teacher *tch/*, int nrgl_lects*/);

/*�t�@�C��sol�������ǂݍ���*/
void readsol(FILE *fpi4, Num num, Timetable f);

/*����\��*/
void printsol(FILE *fpo, Num num, Lecture *lct, Teacher *tch, int *day, Timetable f);

/* �������쐬 */
void initsol(Num num, Lecture *lct, Teacher *tch, Class *cls, Timetable f);

/*  */
void set_utility(Num num, Timetable f, Utility *utility);

/*�u�tTch������Time�ɒS������u�`�̐�*/
int takelct(Num num, int Tch, int Time, Timetable f);

/*�u�`Lct������Time�ɒS������u�t�̐l��*/
int taketch(Num num, int Lct, int Time, Timetable f);

/* ���̐���ᔽ����ޕʂɐ����ĕێ� */
int count_offence(Num num, Lecture *lct, Class *cls, Teacher *tch, Timetable f, Offence *offence);

/* ���̐���ᔽ����ޕʂɕ\�� */
void print_offence(FILE *fpo, Offence offence);

/* ���̕]���l��v�]�ʂɐ����ĕێ����A���v��Ԃ� */
int valuate(
			Weight weight, Num num, 
			Lecture *lct, Class *cls, /*Bilingual *blg,*/ Teacher *tch, /*int nrgl_lects,*/ int *day,
			Timetable f, Penalty *penalty);

/* �u�`Lct���J�u����鎞�ԁB�J�u����Ȃ��ꍇ-1��Ԃ� */
int openat(Num num, int Lct, Timetable f);

/* �u�`Lct��S������u�t�B�J�u����Ȃ��ꍇ-1��Ԃ� */
int takenby(Num num, int Lct, Timetable f);

/* ���̕]���l��v�]�ʂɕ\�� */
void printval(
			FILE *fpo, Weight weight, Num num,
			Lecture *lct, Class *cls, /*Bilingual *blg,*/ Teacher *tch, /*int nrgl_lects, */
			Timetable f, Penalty penalty);

/* �Ǐ��T�� */
int local_search(
			Config config, Weight weight, Num num,
			Lecture *lct, Class *cls, /*Bilingual *blg,*/ Teacher *tch, /*int nrgl_lects,*/ int *day, 
			Timetable f, Timetable new_f, Offence *offence, Penalty *penalty, Utility *utility);

/*�ߖTsearch��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���1��Ԃ��B�Ȃ����0��Ԃ��B*/
int search_move(
			Weight weight, Num num,
			Lecture *lct, Class *cls, /*Bilingual *blg,*/ Teacher *tch, /*int nrgl_lects,*/ int *day, 
			Timetable f, Timetable new_f, Offence *offence, Penalty *penalty, Utility utility);

/* ��f�ɋߖT����move(cls1, time1, cls2, time2)����������new_f���쐬 */
void make_moved_sol(
			Num num, Lecture *lct, Class *cls, Teacher *tch, 
			int cls1, int time1, int cls2, int time2, Timetable f, Timetable new_f, Utility utility);

/* ��f�ɋߖT����move(cls1, time1, cls2, time2)�������� */
void operate_move(Num num, Lecture *lct, Class *cls, Teacher *tch, int cls1, int time1, int cls2, int time2, Timetable f, Utility utility);

/* �ߖTchange��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���1��Ԃ��B�Ȃ����0��Ԃ��B */
int search_change(
			Weight weight, Num num,
			Lecture *lct, Class *cls, /*Bilingual *blg,*/ Teacher *tch, /*int nrgl_lects,*/ int *day, 
			Timetable f, Timetable new_f, Offence *offence, Penalty *penalty, Utility utility);

/* ��f�ɋߖT����change(Lct, Tch_n)����������new_f���쐬 */
void make_changed_sol(Num num, int Lct, int Time, int Tch_c, int Tch_n, Timetable f, Timetable new_f);

/* ��f�ɋߖT����change(Lct, Tch_n)�������� */
void operate_change(Num num, int Lct, int Time, int Tch_c, int Tch_n, Timetable f);

/* �ߖTswap��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���1��Ԃ��B�Ȃ����0��Ԃ��B */
int search_swap(
			Weight weight, Num num,
			Lecture *lct, Class *cls, /*Bilingual *blg,*/ Teacher *tch, /*int nrgl_lects,*/ int *day, 
			Timetable f, Timetable new_f, Offence *offence, Penalty *penalty, Utility utility);

/* ��f�ɋߖT����swap(lct1, lct2)����������new_f���쐬 */
void make_swapped_sol(
			Num num, int lct1, int time1, int tch1, int lct2, int time2, int tch2, Timetable f, Timetable new_f);

/* ��f�ɋߖT����swap(lct1, lct2)�������� */
void operate_swap(Num num, int lct1, int time1, int tch1, int lct2, int time2, int tch2, Timetable f);

/*�]���l���ǂ�����ێ�*/
void save_best(Config config, Num num, Timetable f, int off, int val, Timetable *best_sol, int *best_off, int *best_val);

void sort_best(Config config, Num num, Timetable *best_sol, int *best_off, int *best_val);

void output_timetable(FILE **fpo2, Num num, Lecture *lct, Class *cls, Teacher *tch, int *period_id, int *day_id, Timetable f);

void output_best_timetables(Config config, Num num, Lecture *lct, Class *cls, Teacher *tch, int *period_id, int *day_id, Timetable *best_sol);

/* �������̊J���A�t�@�C���̃N���[�Y */
void free_memory();


int main() 
{
  FILE *fpi1;/*���s���[�h���i�[�����t�@�C���ւ̃|�C���^*/
	FILE *fpi2;/*�d�݂��i�[�����t�@�C���ւ̃|�C���^*/
//	FILE *fpi3;/**/
	FILE *fpi4;/*�����i�[�����t�@�C���ւ̃|�C���^*/
	FILE *fpi5;/*�u�`�̃C���X�^���X�f�[�^���i�[�����t�@�C���ւ̃|�C���^*/
	FILE *fpi6;/*�u�t�̃C���X�^���X�f�[�^���i�[�����t�@�C���ւ̃|�C���^*/
	FILE *fpi7;/*�u�`���Ԃ̃C���X�^���X�f�[�^���i�[�����t�@�C���ւ̃|�C���^*/
	FILE *fpo;/*�o�̓t�@�C���ւ̃|�C���^*/
	FILE *fpo2;/**/
	Config config;/*���s���[�h*/
	Weight weight;/*�v�]�ᔽ�y�i���e�B�̏d��*/
	Num num;/**/
  Lecture *lct;/*�u�`�̔z��*/
  Class *cls;/*�N���X�̔z��*/
//  Bilingual *blg;/**/
  Teacher *tch;/*�u�t�̔z��*/
	int *period_id;
	int *day;
	int *day_id;
//  int nrgl_lects;/*���΍u�t�S�̂̒S����*/
  Timetable f, new_f;/*���C�Ǐ��T���p�̉�*/
	Offence offence;/*����ᔽ*/
	Penalty penalty;/*�v�]�ᔽ�y�i���e�B*/
	Utility utility;/**/
	Timetable *best_sol;/**/
  int *best_off, *best_val;/**/

	int i, j, d;

	pid_t pid;
	pid_t pwait;
	int status;

	pid = fork();
	if (pid == -1)
		{
			fprintf(stderr, "Cannot fork \n");
			exit(1);
		}
	else if (pid == 0) //�q�v���Z�X
		{
			fprintf(stdout, "         \t  \t                                  \n");
			exit(0);
		}
	else	//�e�v���Z�X
{
	sleep(3);
  read_and_ready(
				&fpi1, &fpi2, /*&fpi3,*/ &fpi5, &fpi6, &fpi7, &config, &weight, &num, 
		    &lct, &cls, /*&blg,*/ &tch, /*&nrgl_lects,*/ &period_id, &day_id, &day, &f, &new_f, &penalty, &utility,
				&best_sol, &best_off, &best_val);

  fpo = fopen("/export/public/home/yukiko/public/a/timetables/result", "w");
	printconfig(fpo, config);
//	printconfig(stderr, config);
	printweight(fpo, weight);
  printdata(fpo, num, lct, cls, /*blg,*/ tch/*, nrgl_lects*/);

  if (config.MODE == 0)
    {
      for (i=0; i<config.WIDTH; i++)
				{
				  initsol(num, lct, tch, cls, f);
					d = local_search(config, weight, num, lct, cls, /*blg,*/ tch, /*nrgl_lects,*/ day, f, new_f, &offence, &penalty, &utility);
					count_offence(num, lct, cls, tch, f, &offence);
					valuate(weight, num, lct, cls, /*blg,*/ tch, /*nrgl_lects,*/ day, f,	&penalty);
//					fprintf(stderr, "made #%2d  (%3d moved)  offence:%6d  value:%6d\n", i, d, offence.total, penalty.total);
					save_best(config, num, f, offence.total, penalty.total, best_sol, best_off, best_val);
				}
			sort_best(config, num, best_sol, best_off, best_val);
      for (i=0; i<config.SAVE; i++)
				if (best_off[i] >= 0)
	 	 			{
						fprintf(fpo, "saved #%d ", i);
   				  printsol(fpo, num, lct, tch, day, best_sol[i]);
						count_offence(num, lct, cls, tch, best_sol[i], &offence);
						print_offence(fpo, offence);
						valuate(weight, num, lct, cls, /*blg,*/ tch, /*nrgl_lects,*/ day, best_sol[i],	&penalty);
						printval(fpo,	weight, num, lct, cls, /*blg,*/ tch, /*nrgl_lects,*/ best_sol[i],	penalty);
	  			}
//			fpo2 = fopen("/export/public/home/yukiko/public/a/timetables/csv_timetable", "w");
//			output_timetable(fpo2, num, lct, cls, tch, period_id, day_id, best_sol[0]);
			output_best_timetables(config, num, lct, cls, tch, period_id, day_id, best_sol);
    }

  else if (config.MODE == 1)
    {
			fpi4 = fopen("/export/public/home/yukiko/public/a/sol","r");
      readsol(fpi4, num, f);
			fclose(fpi4);
      printsol(fpo, num, lct, tch, day, f);
			count_offence(num, lct, cls, tch, f, &offence);
			print_offence(fpo, offence);
			valuate(weight, num, lct, cls, /*blg,*/ tch, /*nrgl_lects,*/ day, f,	&penalty);
			printval(fpo,	weight, num, lct, cls, /*blg,*/ tch, /*nrgl_lects,*/ f,	penalty);
			d = local_search(config, weight, num, lct, cls, /*blg,*/ tch, /*nrgl_lects,*/ day, f, new_f, &offence, &penalty, &utility);
			printsol(fpo, num, lct, tch, day, f);
			count_offence(num, lct, cls, tch, f, &offence);
			print_offence(fpo, offence);
			valuate(weight, num, lct, cls, /*blg,*/ tch, /*nrgl_lects,*/ day, f,	&penalty);
			printval(fpo,	weight, num, lct, cls, /*blg,*/ tch, /*nrgl_lects,*/ f,	penalty);
    }

  else if (config.MODE == 2)
    {
			fpi4 = fopen("/export/public/home/yukiko/public/a/sol","r");
      readsol(fpi4, num, f);
			fclose(fpi4);
      printsol(fpo, num, lct, tch, day, f);
			count_offence(num, lct, cls, tch, f, &offence);
			print_offence(fpo, offence);
			valuate(weight, num, lct, cls, /*blg,*/ tch, /*nrgl_lects,*/ day, f,	&penalty);
			printval(fpo,	weight, num, lct, cls, /*blg,*/ tch, /*nrgl_lects,*/ f,	penalty);
    }

  else if (config.MODE == 3)
    {
			fpo2 = fopen("/export/public/home/yukiko/public/a/timetables/csv_timetable", "w");
		  initsol(num, lct, tch, cls, f);
			d = local_search(config, weight, num, lct, cls, /*blg,*/ tch, /*nrgl_lects,*/ day, f, new_f, &offence, &penalty, &utility);
			count_offence(num, lct, cls, tch, f, &offence);
			valuate(weight, num, lct, cls, /*blg,*/ tch, /*nrgl_lects,*/ day, f,	&penalty);
//			fprintf(stderr, "The timetable is maked. (moved %3d times from initial solution)  offence:%6d  value:%6d\n",
//							 d, offence.total, penalty.total);
   		printsol(fpo, num, lct, tch, day, f);
			count_offence(num, lct, cls, tch, f, &offence);
			print_offence(fpo, offence);
			valuate(weight, num, lct, cls, /*blg,*/ tch, /*nrgl_lects,*/ day, f,	&penalty);
			printval(fpo,	weight, num, lct, cls, /*blg,*/ tch, /*nrgl_lects,*/ f,	penalty);
			output_timetable(&fpo2, num, lct, cls, tch, period_id, day_id, f);
			fclose(fpo2);
    }


  //free_memory();


  fclose(fpo);


	return 0;

}

}


/*�t�@�C��config������s���[�h��ǂݍ��݁C�t�@�C��weight����v�]�ᔽ�y�i���e�B��ǂݍ��݁C
�@�t�@�C��data����C���X�^���X�f�[�^�̓ǂݍ��݁C�z��̃��������m�ۂ���*/
void read_and_ready(
			FILE **fpi1, FILE **fpi2, /*FILE **fpi3,*/ FILE **fpi5, FILE **fpi6, FILE **fpi7, 
			Config *config, Weight *weight, Num *num, 
			Lecture **lct, Class **cls, /*Bilingual **blg,*/ Teacher **tch, /*int *nrgl_lects,*/ int **period_id, int **day_id, int **day,
			Timetable *f, Timetable *new_f, Penalty *penalty, Utility *utility, 
			Timetable **best_sol, int **best_off, int **best_val)
{
  int i, j, k, l;
	int tmp[11];
	char tmps[1000];
	int *hash;
	int *cls_qual;
	int **cls_avail;
	int *rhash_period;
	int *rhash_cls;
	int *rhash_tch;
	int *rhash_day;

	int *hash_qual;
	int *rhash_qual;
	int row_t;
	int row_p;
	int max;

  *fpi1 = fopen("/export/public/home/yukiko/public/a/config", "r");

//	*fpi3 = fopen("/export/public/home/yukiko/public/a/numbers", "r");
	*fpi5 = fopen("/export/public/home/yukiko/public/a/task_relation", "r");
	*fpi6 = fopen("/export/public/home/yukiko/public/a/processor_relation", "r");
	*fpi7 = fopen("/export/public/home/yukiko/public/a/period_relation", "r");

  fscanf(*fpi1, "%d %d %d %d %d", &(config->MODE), &(config->WIDTH), &(config->DEPTH), &(config->SAVE), &(config->SEED));
	srand(config->SEED);


//period_relation, task_relation, processor_relation����񑖍�����clsnum, time, tchnum�𐔂���
	fseek(*fpi7, 0L, SEEK_SET);
	fscanf(*fpi7, "%s\n", tmps);
	max = -1;
	while (fscanf(*fpi7, "%d,%d\n", &(tmp[0]), &(tmp[1])) != EOF) 
		if (max < tmp[0]) max = tmp[0];
	hash = (int *)malloc(sizeof(int) * (max + 1));
	for (j = 0; j < max+1; j++)
		hash[j] = -1;
	fseek(*fpi7, 0L, SEEK_SET);
	fscanf(*fpi7, "%s\n", tmps);
	while(fscanf(*fpi7, "%d,%d\n", &(tmp[0]), &(tmp[1])) != EOF) hash[tmp[0]]++;
	(*num).time = 0;
	for (j = 0; j < max+1; j++) if (hash[j] > -1) (*num).time++;

	rhash_period = (int *)malloc(sizeof(int) * (max + 1));
	for (j=0; j<max+1; j++) rhash_period[j] = -1;
	k = 0;
	for (j=0; j<max+1; j++) 
		if (hash[j] > -1) 
			{
				rhash_period[j] = k;
				k++;
			}


	(*period_id) = (int *)malloc(sizeof(int) * (*num).time);
	for (j=0; j<max+1; j++) if (rhash_period[j] >= 0) (*period_id)[rhash_period[j]] = j;
	free(hash);


	fseek(*fpi7, 0L, SEEK_SET);
	fscanf(*fpi7, "%s\n", tmps);
	max = -1;
	while (fscanf(*fpi7, "%d,%d\n", &(tmp[0]), &(tmp[1])) != EOF) 
		if (max < tmp[1]) max = tmp[1];
	hash = (int *)malloc(sizeof(int) * (max + 1));
	for (j = 0; j < max+1; j++)
		hash[j] = -1;
	fseek(*fpi7, 0L, SEEK_SET);
	fscanf(*fpi7, "%s\n", tmps);
	while(fscanf(*fpi7, "%d,%d\n", &(tmp[0]), &(tmp[1])) != EOF) hash[tmp[1]]++;
	(*num).week = 0;
	for (j = 0; j < max+1; j++) if (hash[j] > -1) (*num).week++;

	rhash_day = (int *)malloc(sizeof(int) * (max + 1));
	for (j=0; j<max+1; j++) rhash_day[j] = -1;
	k = 0;
	for (j=0; j<max+1; j++) 
		if (hash[j] > -1) 
			{
				rhash_day[j] = k;
				k++;
			}
	free(hash);



	fseek(*fpi7, 0L, SEEK_SET);
	fscanf(*fpi7, "%s\n", tmps);
	*day = (int *)malloc(sizeof(int) * (*num).time);
	*day_id = (int *)malloc(sizeof(int) * (*num).time);
	for (i=0; i<(*num).time; i++)
		{
			fscanf(*fpi7, "%d,%d\n", &(tmp[0]), &(tmp[1]));
			(*day)[rhash_period[tmp[0]]] = rhash_day[tmp[1]];
			(*day_id)[rhash_period[tmp[0]]] = tmp[1];
		}





	fseek(*fpi5, 0L, SEEK_SET);
	fscanf(*fpi5, "%s\n", tmps);
	max = -1;
	while (fscanf(*fpi5, "%d,%d,%d,%d,%d,%d,%d\n", 
		  		&(tmp[0]), &(tmp[1]), &(tmp[2]), &(tmp[3]), &(tmp[4]), &(tmp[5]), &(tmp[6])) != EOF)
		if (max < tmp[0]) max = tmp[0];
	hash = (int *)malloc(sizeof(int) * (max + 1));
	for (j = 0; j < max+1; j++)
		hash[j] = -1;
	fseek(*fpi5, 0L, SEEK_SET);
	fscanf(*fpi5, "%s\n", tmps);
	while(fscanf(*fpi5, "%d,%d,%d,%d,%d,%d,%d\n", 
				&(tmp[0]), &(tmp[1]), &(tmp[2]), &(tmp[3]), &(tmp[4]), &(tmp[5]), &(tmp[6])) != EOF)
		hash[tmp[0]]++;
	(*num).clsnum = 0;
	for (j = 0; j < max+1; j++) if (hash[j] > -1) (*num).clsnum++;

	rhash_cls = (int *)malloc(sizeof(int) * (max + 1));
	for (j=0; j<max+1; j++) rhash_cls[j] = -1;
	k = 0;
	for (j=0; j<max+1; j++) 
		if (hash[j] > -1) 
			{
				rhash_cls[j] = k;
				k++;
			}
	free(hash);


	fseek(*fpi6, 0L, SEEK_SET);
	fscanf(*fpi6, "%s\n", tmps);
	max = -1;
	while (fscanf(*fpi6, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n", 
					&tmp[0], &tmp[1], &tmp[2], &tmp[3], 
					&tmp[4], &tmp[5], &tmp[6], &tmp[7], 
					&tmp[8], &tmp[9], &tmp[10]) != EOF)
		if (max < tmp[0]) max = tmp[0];
	hash = (int *)malloc(sizeof(int) * (max + 1));
	for (j = 0; j < max+1; j++)
		hash[j] = -1;
	fseek(*fpi6, 0L, SEEK_SET);
	fscanf(*fpi6, "%s\n", tmps);
	while(fscanf(*fpi6, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n", 
					&tmp[0], &tmp[1], &tmp[2], &tmp[3], 
					&tmp[4], &tmp[5], &tmp[6], &tmp[7], 
					&tmp[8], &tmp[9], &tmp[10]) != EOF)
		hash[tmp[0]]++;
	(*num).tchnum = 0;
	for (j = 0; j < max+1; j++) if (hash[j] > -1) (*num).tchnum++;

	rhash_tch = (int *)malloc(sizeof(int) * (max + 1));
	for (j=0; j<max+1; j++) rhash_tch[j] = -1;
	k = 0;
	for (j=0; j<max+1; j++) 
		if (hash[j] > -1) 
			{
				rhash_tch[j] = k;
				k++;
			}
	free(hash);


	max = -1;
	fseek(*fpi5, 0L, SEEK_SET);
	fscanf(*fpi5, "%s\n", tmps);
	while (fscanf(*fpi5, "%d,%d,%d,%d,%d,%d,%d\n", 
		  		&(tmp[0]), &(tmp[1]), &(tmp[2]), &(tmp[3]), &(tmp[4]), &(tmp[5]), &(tmp[6])) != EOF)
		if (max < tmp[2]) max = tmp[2];
	fseek(*fpi6, 0L, SEEK_SET);
	fscanf(*fpi6, "%s\n", tmps);
	while (fscanf(*fpi6, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n", 
					&tmp[0], &tmp[1], &tmp[2], &tmp[3], 
					&tmp[4], &tmp[5], &tmp[6], &tmp[7], 
					&tmp[8], &tmp[9], &tmp[10]) != EOF)
		if (max < tmp[2]) max = tmp[2];
	hash = (int *)malloc(sizeof(int) * (max + 1));
	for (j = 0; j < max+1; j++)
		hash[j] = -1;
	fseek(*fpi5, 0L, SEEK_SET);
	fscanf(*fpi5, "%s\n", tmps);
	while (fscanf(*fpi5, "%d,%d,%d,%d,%d,%d,%d\n", 
		  		&(tmp[0]), &(tmp[1]), &(tmp[2]), &(tmp[3]), &(tmp[4]), &(tmp[5]), &(tmp[6])) != EOF)
		hash[tmp[2]]++;
	fseek(*fpi6, 0L, SEEK_SET);
	fscanf(*fpi6, "%s\n", tmps);
	while(fscanf(*fpi6, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n", 
					&tmp[0], &tmp[1], &tmp[2], &tmp[3], 
					&tmp[4], &tmp[5], &tmp[6], &tmp[7], 
					&tmp[8], &tmp[9], &tmp[10]) != EOF)
		hash[tmp[2]]++;
	(*num).qualificationnum = 0;
	for (j = 0; j < max+1; j++) if (hash[j] > -1) (*num).qualificationnum++;

	rhash_qual = (int *)malloc(sizeof(int) * (max + 1));
	for (j=0; j<max+1; j++) rhash_qual[j] = -1;
	k = 0;
	for (j=0; j<max+1; j++) 
		if (hash[j] > -1) 
			{
				rhash_qual[j] = k;
				k++;
			}
	free(hash);



//�������m�ہC�f�[�^�i�[
  *cls = (Class *)malloc(sizeof(Class) * (*num).clsnum);
	cls_qual = (int *)malloc(sizeof(int) * (*num).clsnum);
	cls_avail = (int **)malloc(sizeof(int *) * (*num).clsnum);
	for (i=0; i<(*num).clsnum; i++)
		{
			cls_avail[i] = (int *)malloc(sizeof(int) * (*num).time);
			for (j=0; j<(*num).time; j++)
				cls_avail[i][j] = 0;
		}

	fseek(*fpi5, 0L, SEEK_SET);
	fscanf(*fpi5, "%s\n", tmps);
	while (fscanf(*fpi5, "%d,%d,%d,%d,%d,%d,%d\n", 
						&(tmp[0]), &(tmp[1]), &(tmp[2]), &(tmp[3]), &(tmp[4]), &(tmp[5]), &(tmp[6])) != EOF)
		{
			(*cls)[rhash_cls[tmp[0]]].id = tmp[0];
			(*cls)[rhash_cls[tmp[0]]].num = tmp[3];
			cls_qual[rhash_cls[tmp[0]]] = rhash_qual[tmp[2]];
			(*cls)[rhash_cls[tmp[0]]].qualification_id = tmp[2];
			cls_avail[rhash_cls[tmp[0]]][rhash_period[tmp[1]]] = tmp[6];
		}

	(*num).lctnum = 0;
  for (i=0; i<(*num).clsnum; i++)
		{
    	(*cls)[i].lcts = (int *)malloc(sizeof(int) * (*cls)[i].num);
			(*num).lctnum += (*cls)[i].num;
		}

  *lct = (Lecture *)malloc(sizeof(Lecture) * (*num).lctnum);
	for (i=0; i<(*num).lctnum; i++)
		{
    	(*lct)[i].avail = (int *)malloc(sizeof(int) * (*num).time);
			for (j=0; j<(*num).time; j++)
				(*lct)[i].avail[j] = 0;
		}

	i = 0;
	for (j=0; j<(*num).clsnum; j++)
		for (k=0; k<(*cls)[j].num; k++)
			{
				(*cls)[j].lcts[k] = i;
				(*lct)[i].class = j;
				(*lct)[i].qualification = cls_qual[j];
				for (l=0; l<(*num).time; l++)
					(*lct)[i].avail[l] = cls_avail[j][l];
				i++;
			}
	free(cls_qual);
	for (i=0; i<(*num).clsnum; i++)
		free(cls_avail[i]);
	free(cls_avail);

  *tch = (Teacher *)malloc(sizeof(Teacher) * (*num).tchnum);
	for (i=0; i<num->tchnum; i++)
		{
			(*tch)[i].desire = (int *)malloc(sizeof(int) * (*num).time);
			for (j=0; j<(*num).time; j++)
				(*tch)[i].desire[j] = 0;
			(*tch)[i].qualification = (int *)malloc(sizeof(int) * (*num).qualificationnum);
			for (j=0; j<(*num).qualificationnum; j++)
				(*tch)[i].qualification[j] = 0;
		}

	fseek(*fpi6, 0L, SEEK_SET);
	fscanf(*fpi6, "%s\n", tmps);
	while (fscanf(*fpi6, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n", 
				&tmp[0], &tmp[1], &tmp[2], &tmp[3], 
				&tmp[4], &tmp[5], &tmp[6], &tmp[7], 
				&tmp[8], &tmp[9], &tmp[10]) != EOF)
		{
			(*tch)[rhash_tch[tmp[0]]].id = tmp[0];
			(*tch)[rhash_tch[tmp[0]]].desire[rhash_period[tmp[1]]] = tmp[10];
			(*tch)[rhash_tch[tmp[0]]].qualification[rhash_qual[tmp[2]]] = 1;
			(*tch)[rhash_tch[tmp[0]]].regular = tmp[3] - 1;
			(*tch)[rhash_tch[tmp[0]]].week_lw = tmp[4];
			(*tch)[rhash_tch[tmp[0]]].week_up = tmp[5];
			(*tch)[rhash_tch[tmp[0]]].days_lw = tmp[6];
			(*tch)[rhash_tch[tmp[0]]].days_up = tmp[7];
			(*tch)[rhash_tch[tmp[0]]].wage = tmp[8];
		}
			
	fclose(*fpi1);

//	fclose(*fpi3);
	fclose(*fpi5);
	fclose(*fpi6);
	fclose(*fpi7);

	// Getting penalty weights from db.
	/*
	association assoc[] = 
	  {{"PREFER", 0},
	   {"R_WEEKOVER", 0}, 
	   {"R_WEEKUNDER", 0}, 
	   {"R_DAYSOVER", 0}, 
	   {"R_DAYSUNDER", 0},
	   {"R_HOLE", 0}, 
	   {"R_DESIRE", 0}, 
	   {"N_WEEKOVER", 0}, 
	   {"N_WEEKUNDER", 0}, 
	   {"N_DAYSOVER", 0}, 
	   {"N_DAYSUNDER", 0}, 
	   {"N_HOLE", 0}, 
	   {"N_DESIRE", 0}, 
	   {NULL, 0}};
	read_from_db("lilith.daisy.ai.kyutech.ac.jp","sin","qpm4nz","schedule",
		     "name", "weight", "penalties", assoc);
	weight->PREFER = (int) get_value_from_assoc(assoc, "PREFER");
	weight->R_WEEKOVER = (int) get_value_from_assoc(assoc, "R_WEEKOVER");
	weight->R_WEEKUNDER = (int) get_value_from_assoc(assoc, "R_WEEKUNDER");
	weight->R_DAYSOVER = (int) get_value_from_assoc(assoc, "R_DAYSOVER");
	weight->R_DAYSUNDER = (int) get_value_from_assoc(assoc, "R_DAYSUNDER");
	weight->R_HOLE = (int) get_value_from_assoc(assoc, "R_HOLE");
	weight->R_DESIRE = (int) get_value_from_assoc(assoc, "R_DESIRE");
	weight->N_WEEKOVER = (int) get_value_from_assoc(assoc, "N_WEEKOVER");
	weight->N_WEEKUNDER = (int) get_value_from_assoc(assoc, "N_WEEKUNDER");
	weight->N_DAYSOVER = (int) get_value_from_assoc(assoc, "N_DAYSOVER");
	weight->N_DAYSUNDER = (int) get_value_from_assoc(assoc, "N_DAYSUNDER");
	weight->N_HOLE = (int) get_value_from_assoc(assoc, "N_HOLE");
	weight->N_DESIRE = (int) get_value_from_assoc(assoc, "N_DESIRE");
	//
	//
	*/

	*f = (int ***)malloc(sizeof(int **) * (*num).lctnum);
	*new_f = (int ***)malloc(sizeof(int **) * (*num).lctnum);
	for (i=0; i<(*num).lctnum; i++)
		{
			(*f)[i] = (int **)malloc(sizeof(int *) * (*num).time);
			(*new_f)[i] = (int **)malloc(sizeof(int *) * (*num).time);
			for (j=0; j<(*num).time; j++)
				{
					(*f)[i][j] = (int *)malloc(sizeof(int) * (*num).tchnum);
					(*new_f)[i][j] = (int *)malloc(sizeof(int) * (*num).tchnum);
					for (k=0; k<(*num).tchnum; k++)
						{
							(*f)[i][j][k] = 0;
							(*new_f)[i][j][k] = 0;
						}
				}
		}
	(*penalty).prefer = (int *)malloc(sizeof(int) * (*num).clsnum);
	(*penalty).weekover = (int *)malloc(sizeof(int) * (*num).tchnum);
	(*penalty).weekunder = (int *)malloc(sizeof(int) * (*num).tchnum);
	(*penalty).daysover = (int *)malloc(sizeof(int) * (*num).tchnum);
	(*penalty).daysunder = (int *)malloc(sizeof(int) * (*num).tchnum);
//	(*penalty).dayover = (int **)malloc(sizeof(int *) * (*num).tchnum);
//	(*penalty).dayunder = (int **)malloc(sizeof(int *) * (*num).tchnum);
//	(*penalty).days = (int *)malloc(sizeof(int) * (*num).tchnum);
	(*penalty).hole = (int *)malloc(sizeof(int) * (*num).tchnum);
	(*penalty).desire = (int *)malloc(sizeof(int) * (*num).tchnum);
/*	for (i=0; i<(*num).tchnum; i++)
		{
			(*penalty).dayover[i] = (int *)malloc(sizeof(int) * (*num).week);
			(*penalty).dayunder[i] = (int *)malloc(sizeof(int) * (*num).week);
		}
*/
	(*utility).lct_openat = (int *)malloc(sizeof(int) * (*num).lctnum);
	(*utility).lct_takenby = (int *)malloc(sizeof(int) * (*num).lctnum);
	(*utility).tch_takelct = (int **)malloc(sizeof(int *) * (*num).tchnum);
	for (i=0; i<(*num).tchnum; i++)
		(*utility).tch_takelct[i] = (int *)malloc(sizeof(int) * (*num).time);
	(*best_sol) = (int ****)malloc(sizeof(int ***) * (*config).SAVE);
	for (i=0; i<(*config).SAVE; i++)
		{
			(*best_sol)[i] = (int ***)malloc(sizeof(int **) * (*num).lctnum);
			for (j=0; j<(*num).lctnum; j++)
				{
					(*best_sol)[i][j] = (int **)malloc(sizeof(int *) * (*num).time);
					for (k=0; k<(*num).time; k++)
						{
							(*best_sol)[i][j][k] = (int *)malloc(sizeof(int) * (*num).tchnum);
							for (l=0; l<(*num).tchnum; l++)
								(*best_sol)[i][j][k][l] = 0;
						}
				}
		}
	(*best_off) = (int *)malloc(sizeof(int) * (*config).SAVE);
	for (i=0; i<(*config).SAVE; i++)
		(*best_off)[i] = -1;
	(*best_val) = (int *)malloc(sizeof(int) * (*config).SAVE);
	for (i=0; i<(*config).SAVE; i++)
		(*best_val)[i] = -1;
}


/*���s���[�h��\��*/
void printconfig(FILE *fpo, Config config)
{
	fprintf(fpo, "CONFIG:\n\n");
	switch (config.MODE) 
		{
			case 0: fprintf(fpo, "Running Mode 0: [readdata -> initsol -> localsearch -> printsol]\n");break;
			case 1: fprintf(fpo, "Running Mode 1: [readdata -> readsol -> localsearch -> printsol]\n");break;
			case 2: fprintf(fpo, "Running Mode 2: [readdata -> readsol -> printsol]\n");break;
			case 3: fprintf(fpo, "Running Mode 3: test running mode.\n");break;
			default: fprintf(stderr, "Error: Illegal mode. Select from 0-3.\n");break;
		}
	fprintf(fpo, "search Width: %d\n", config.WIDTH);
	fprintf(fpo, "search Depth: %d\n", config.DEPTH);
	fprintf(fpo, "Save the best %d solution.\n", config.SAVE);
	fprintf(fpo, "Seed of random function: %d\n\n", config.SEED);
}

/*�v�]�ᔽ�y�i���e�B�̒l��\��*/
void printweight(FILE *fpo, Weight weight)
{
	fprintf(fpo, "WEIGHT:\n\n");
	fprintf(fpo, "PREFER: %d\n", weight.PREFER);
	fprintf(fpo, "R_WEEKOVER: %d, R_WEEKUNDER: %d, R_DAYSOVER: %d, R_DAYSUNDER: %d, \n", 
								weight.R_WEEKOVER, weight.R_WEEKUNDER, weight.R_DAYSOVER, weight.R_DAYSUNDER);
//	fprintf(fpo, "R_WEEKOVER: %d, R_WEEKUNDER: %d, R_DAYOVER: %d, R_DAYUNDER: %d, \n", 
//								weight.R_WEEKOVER, weight.R_WEEKUNDER, weight.R_DAYOVER, weight.R_DAYUNDER);
	fprintf(fpo, "R_HOLE: %d, R_DESIRE: %d,\n",
								weight.R_HOLE, weight.R_DESIRE);
//	fprintf(fpo, "R_DAYS: %d, R_HOLE: %d, R_DESIRE: %d,\n",
//								weight.R_DAYS, weight.R_HOLE, weight.R_DESIRE);
	fprintf(fpo, "N_WEEKOVER: %d, N_WEEKUNDER: %d, N_DAYSOVER: %d, N_DAYSUNDER: %d, \n",
								weight.N_WEEKOVER, weight.N_WEEKUNDER, weight.N_DAYSOVER, weight.N_DAYSUNDER);
//	fprintf(fpo, "N_WEEKOVER: %d, N_WEEKUNDER: %d, N_DAYOVER: %d, N_DAYUNDER: %d, \n",
//								weight.N_WEEKOVER, weight.N_WEEKUNDER, weight.N_DAYOVER, weight.N_DAYUNDER);
	fprintf(fpo, "N_HOLE: %d, N_DESIRE: %d,\n",
								weight.N_HOLE, weight.N_DESIRE);
//	fprintf(fpo, "N_DAYS: %d, N_HOLE: %d, N_DESIRE: %d,\n",
//								weight.N_DAYS, weight.N_HOLE, weight.N_DESIRE);
//	fprintf(fpo, "J_N: %d, NRGOVER: %d, NRGUNDER:%d\n\n",
//								weight.J_N, weight.NRGOVER, weight.NRGUNDER);
	fprintf(fpo, "\n");
}


/*�C���X�^���X��\��*/
void printdata(
			FILE *fpo, Num num,
	    Lecture *lct, Class *cls, /*Bilingual *blg,*/ Teacher *tch/*, int nrgl_lects*/)
{
	int i, j;
/*
  fprintf(fpo, "INSTANCE:\n\n");
  fprintf(fpo, "%2d sections exist:\n", num.lctnum);
  fprintf(fpo, "ID  class name       |    Monday    |   Tuesday    |  Wednesday   |  Thursday    |    Friday    |\n");
  fprintf(fpo, "                      ");
  for (i=0; i<num.time; i++) fprintf(fpo, "%2d ",i);
  fprintf(fpo, "\n");
  for (i=0; i<num.lctnum; i++)    
    {
      fprintf(fpo, "#%-2d  %2d   %-10s: ", i, lct[i].class, lct[i].name);
      for (j=0; j<num.time; j++)
				if (lct[i].avail[j] > 0)
	  			fprintf(fpo, "%2d ", lct[i].avail[j]);
				else 
	  			fprintf(fpo, " _ ");
      fprintf(fpo, "\n");
    }
  fprintf(fpo, "\n");
  fprintf(fpo, "%2d teachers exist:\n", num.tchnum);
  fprintf(fpo, "               |    Monday    |   Tuesday    |  Wednesday   |  Thursday    |    Friday    |\n");
  fprintf(fpo, "                ");
  for (i=0; i<num.time; i++) fprintf(fpo, "%2d ",i);
  fprintf(fpo, "( w^, w_, d^, d_, r_n, j_n )");
  fprintf(fpo, "\n");
  for (i=0; i<num.tchnum; i++)    
    {
      fprintf(fpo, "#%-2d %-10s: ", i, tch[i].name);
      for (j=0; j<num.time; j++)
				if (tch[i].desire[j])
				  fprintf(fpo, "%2d ", tch[i].desire[j]);
				else
				  fprintf(fpo, " _ ");
      fprintf(fpo, "( %2d, %2d, %2d, %2d,", tch[i].week_up, tch[i].week_lw, tch[i].day_up, tch[i].day_lw);
      if (tch[i].regular == 0)
				fprintf(fpo, "  r ,");
      else
				fprintf(fpo, "  n ,");
      if (tch[i].native == 0)
				fprintf(fpo, "  j  )\n");
      else
				fprintf(fpo, "  n  )\n");
    }
  fprintf(fpo, "\n");
*/
/*  fprintf(fpo, "%2d pair of classes exist:\n", num.blgnum);
  for (i=0; i<num.blgnum; i++)
    {
      fprintf(fpo, "( %d{ ", blg[i].cls1);
      for (j=0; j<cls[blg[i].cls1].num; j++)
				fprintf(fpo, "#%d ", cls[blg[i].cls1].lcts[j]);
      fprintf(fpo, "}, %d{ ", blg[i].cls2);
      for (j=0; j<cls[blg[i].cls2].num; j++)
				fprintf(fpo, "#%d ", cls[blg[i].cls2].lcts[j]);
      fprintf(fpo, "} )\n");
    }
  fprintf(fpo, "\n");
  fprintf(fpo, "number of sections taken charge of not regular teachers : %d\n", nrgl_lects);
  fprintf(fpo, "\n");*/


  fprintf(fpo, "INSTANCE:\n\n");
  fprintf(fpo, "%2d sections exist:\n", num.lctnum);
  fprintf(fpo, "ID  class type |    Monday    |   Tuesday    |  Wednesday   |  Thursday    |    Friday    |\n");
//  fprintf(fpo, "ID  class type name       |    Monday    |   Tuesday    |  Wednesday   |  Thursday    |    Friday    |\n");
  fprintf(fpo, "                ");
  for (i=0; i<num.time; i++) fprintf(fpo, "%2d ",i);
  fprintf(fpo, "\n");
  for (i=0; i<num.lctnum; i++)    
    {
      fprintf(fpo, "#%-2d  %2d    %2d   ", i, lct[i].class, lct[i].qualification);
//      fprintf(fpo, "#%-2d  %2d   %4d  ", i, lct[i].class, lct[i].qualification, lct[i].name);
      for (j=0; j<num.time; j++)
				if (lct[i].avail[j] > 0)
	  			fprintf(fpo, "%2d ", lct[i].avail[j]);
				else 
	  			fprintf(fpo, " _ ");
      fprintf(fpo, "\n");
    }
  fprintf(fpo, "\n");
  fprintf(fpo, "%2d teachers exist:\n", num.tchnum);
  fprintf(fpo, "ID  |    Monday    |   Tuesday    |  Wednesday   |  Thursday    |    Friday    |\n");
  fprintf(fpo, "     ");
  for (i=0; i<num.time; i++) fprintf(fpo, "%2d ",i);
  fprintf(fpo, "( w^, w_, d^, d_, r_n, J_E, wg)");
  fprintf(fpo, "\n");
  for (i=0; i<num.tchnum; i++)    
    {
      fprintf(fpo, "#%-2d  ", i);
      for (j=0; j<num.time; j++)
				if (tch[i].desire[j])
				  fprintf(fpo, "%2d ", tch[i].desire[j]);
				else
				  fprintf(fpo, " _ ");
      fprintf(fpo, "( %2d, %2d, %2d, %2d,", tch[i].week_up, tch[i].week_lw, tch[i].days_up, tch[i].days_lw);
      if (tch[i].regular == 0)
				fprintf(fpo, "  r ,");
      else
				fprintf(fpo, "  n ,");
      if (tch[i].qualification[0])
				fprintf(fpo, "  J");
			else
				fprintf(fpo, "   ");
      if (tch[i].qualification[1])
				fprintf(fpo, "E, ");
			else
				fprintf(fpo, " , ");
			fprintf(fpo, "%2d)\n", tch[i].wage);
    }
  fprintf(fpo, "\n");
}



/*�t�@�C��sol�������ǂݍ���*/
void readsol(FILE *fpi4, Num num, Timetable f)
{
  int i, lc, tc, tm;
	for (lc=0; lc<num.lctnum; lc++)
		for (tm=0; tm<num.time; tm++)
			for (tc=0; tc<num.tchnum; tc++)
				f[lc][tm][tc] = 0;
  for (i=0; i<num.lctnum; i++)
    {
      fscanf(fpi4, "%d %d %d", &lc, &tm, &tc);
      f[lc][tm][tc] = 1;
    }
}

/*����\��*/
void printsol(FILE *fpo, Num num, Lecture *lct, Teacher *tch, int *day, Timetable f)
{
  int i, j, k, l;
	int check, tmp;
	int wc, *c;
  c = (int *)malloc(sizeof(int) * num.week);

  fprintf(fpo, "TIMETABLE:\n\n");
//  fprintf(fpo, "ID  class type name       |    Monday    |   Tuesday    |  Wednesday   |  Thursday    |    Friday    |\n");
  fprintf(fpo, "ID  class type |    Monday    |   Tuesday    |  Wednesday   |  Thursday    |    Friday    |\n");
  fprintf(fpo, "                ");
  for (i=0; i<num.time; i++) fprintf(fpo, "%2d ",i);
  fprintf(fpo, "\n");
  for (i=0; i<num.lctnum; i++)
    {
      fprintf(fpo, "#%-2d: %2d   %3d   ", i, lct[i].class, lct[i].qualification);
 //     fprintf(fpo, "#%-2d  %2d   %4d %-10s: ", i, lct[i].class, lct[i].qualification, lct[i].name);
			tmp = -1;
			for (j=0; j<num.time; j++)
				{
					check = 0;
					for (k=0; k<num.tchnum; k++) 
						if (f[i][j][k]) 
							{
								check++;
								tmp = k;
							}
					if (check) fprintf(fpo, "%2d ", tmp);
					else fprintf(fpo, " _ ");
				}
//			if (tmp >= 0) fprintf(fpo, " %s", tch[tmp].name);
			fprintf(fpo, "\n");
		}
  fprintf(fpo, "\n");
  fprintf(fpo, "    |    Monday    |   Tuesday    |  Wednesday   |  Thursday    |    Friday    |\n");
  fprintf(fpo, "     ");
  for (i=0; i<num.time; i++) fprintf(fpo, "%2d ", i);
  fprintf(fpo, " ( Mo, Tu, We, Th, Fr )\n");
  for (i=0; i<num.tchnum; i++)
    {
      fprintf(fpo, "#%-2d: ", i);
/*      wc = 0;
      for (j=0; j<num.week; j++)
				{
	  			c[j] = 0;
	  			for (k=0; k<num.period; k++)
						{
							tmp = -1;
	    				for (l=0; l<num.lctnum; l++)
								if (f[l][j*num.period+k][i])
									{
										c[j]++;
										wc++;
										tmp = l;
									}
							if (tmp >= 0) fprintf(fpo, "%2d ", tmp);
							else fprintf(fpo, " _ ");
						}
				}
*/
			wc = 0;
			for (j=0; j<num.week; j++) c[j] = 0;
			for (j=0; j<num.time; j++)
				{
					tmp = -1;
					for (k=0; k<num.lctnum; k++)
						if (f[k][j][i])
							{
								wc++;
								c[day[j]]++;
								tmp = k;
							}
					if (tmp >= 0) fprintf(fpo, "%2d ", tmp);
					else fprintf(fpo, " _ ");
				}
      fprintf(fpo, " ( %2d", c[0]);
      for (j=1; j<num.week; j++)
				fprintf(fpo, ", %2d", c[j]);
      fprintf(fpo, " ): %d\n", wc);
    }
  fprintf(fpo, "\n");


	free(c);
}


/* �������쐬 */
void initsol(Num num, Lecture *lct, Teacher *tch, Class *cls, Timetable f)
{
	int i, j, k, r, tmp, check;
	int rest;
	int *assigned;
	int rand_lct, rand_time, rand_tch, rand_cls;

	for (i=0; i<num.lctnum; i++)
		for (j=0; j<num.time; j++)
			for (k=0; k<num.tchnum; k++)
				f[i][j][k] = 0;
	assigned = (int *)malloc(sizeof(int) * num.lctnum);
	for (i=0; i<num.lctnum; i++) assigned[i] = 0;
	rest = num.lctnum;
	while (rest)
		{
			r = rand() % rest;
			for (i=0; i<=r; i++)
				if (assigned[i]) r++;
			rand_lct = r;
			tmp = 0;
			for (i=0; i<num.time; i++)
				if (lct[rand_lct].avail[i]) tmp++;
			r = rand() % tmp;
			for (i=0; i<=r; i++)
				if (!lct[rand_lct].avail[i]) r++;
			rand_time = r;
			rand_cls = lct[rand_lct].class;
			for (i=0; i<cls[rand_cls].num; i++)
				{
					tmp = 0;
					for (j=0; j<num.tchnum; j++)
						if (tch[j].desire[rand_time] && 
								tch[j].qualification[lct[rand_lct].qualification] &&
								!takelct(num, j, rand_time, f)) tmp++;
					if (tmp) 
						{
							r = rand() % tmp;
							for (j=0; j<=r; j++)
								if (!(tch[j].desire[rand_time] && 
											tch[j].qualification[lct[rand_lct].qualification] &&
											!takelct(num, j, rand_time, f))) r++;
							rand_tch = r;
							f[cls[rand_cls].lcts[i]][rand_time][rand_tch] = 1;
						}
					assigned[cls[rand_cls].lcts[i]]++;
					rest--;
				}
		}
	free(assigned);
}


void set_utility(Num num, Timetable f, Utility *utility)
{
	int i, j, k;
	for (i=0; i<num.lctnum; i++)
		{
			(*utility).lct_openat[i] = -1;
			(*utility).lct_takenby[i] = -1;
		}
	for (i=0; i<num.tchnum; i++)
		for (j=0; j<num.time; j++)
			(*utility).tch_takelct[i][j] = -1;
	for (i=0; i<num.lctnum; i++)
		for (j=0; j<num.time; j++)
			for (k=0; k<num.tchnum; k++)
				if (f[i][j][k])
					{
						(*utility).lct_openat[i] = j;
						(*utility).lct_takenby[i] = k;
						(*utility).tch_takelct[k][j] = i;
					}
}



/*�u�tTch������Time�ɒS������u�`�̐�*/
int takelct(Num num, int Tch, int Time, Timetable f)
{
	int i, count=0;
	for (i=0; i<num.lctnum; i++)
		if (f[i][Time][Tch]) count++;
	return count;
}

/*�u�`Lct������Time�ɒS������u�t�̐l��*/
int taketch(Num num, int Lct, int Time, Timetable f)
{
	int i, count=0;
	for (i=0; i<num.tchnum; i++)
		if (f[Lct][Time][i]) count++;
	return count;
}



/* ���̐���ᔽ����ޕʂɐ����ĕێ����A���v��Ԃ� */
int count_offence(Num num, Lecture *lct, Class *cls, Teacher *tch, Timetable f, Offence *offence)
{
	int i, j, k, l;
	int tmp, *open;

	(*offence).null_lct = 0;
	(*offence).lct_overlap = 0;
	(*offence).class_separation = 0;
	(*offence).tch_overlap = 0;
	(*offence).not_qualified = 0;
	open = (int *)malloc(sizeof(int) * num.time);

	for (i=0; i<num.lctnum; i++)
		{
			tmp = 0;
			for (j=0; j<num.time; j++)
				tmp += taketch(num, i, j, f);
			if (tmp==0) (*offence).null_lct++;
			else if (tmp > 1) (*offence).lct_overlap += tmp-1;
		}

	for (i=0; i<num.clsnum; i++)
		{
			for (j=0; j<num.time; j++) open[j] = 0;
			for (j=0; j<cls[i].num; j++)
				{
					if ((tmp = openat(num, cls[i].lcts[j], f)) >= 0) open[tmp]++;
				}
			tmp = 0;
			for (j=0; j<num.time; j++)
				if (open[j]) tmp++;
			if (tmp) (*offence).class_separation += tmp-1;
		}

	free(open);
	for (i=0; i<num.tchnum; i++)
		for (j=0; j<num.time; j++)
			if ((tmp = takelct(num, i, j, f)) > 1) (*offence).tch_overlap += tmp-1;
	for (i=0; i<num.lctnum; i++)
		for (j=0; j<num.time; j++)
			for (k=0; k<num.tchnum; k++)
				if (f[i][j][k] && !(tch[k].qualification[lct[i].qualification]))
					(*offence).not_qualified += 1;

	(*offence).total = (*offence).null_lct + (*offence).lct_overlap + (*offence).class_separation + 
											(*offence).tch_overlap + (*offence).not_qualified;
	return (*offence).total;
}


/* ����ᔽ����ޕʂɕ\�� */
void print_offence(FILE *fpo, Offence offence)
{
	fprintf(fpo, "OFFENCE:\n");
	fprintf(fpo, "null lecture:     %d\n", offence.null_lct);
	fprintf(fpo, "lecture overlap:  %d\n", offence.lct_overlap);
	fprintf(fpo, "class separation: %d\n", offence.class_separation);
	fprintf(fpo, "teacher overlap:  %d\n", offence.tch_overlap);
	fprintf(fpo, "not qualified:    %d\n", offence.not_qualified);
	fprintf(fpo, "total:            %d\n\n", offence.total);
}


/* ���̕]���l��v�]�ʂɐ����ĕێ����A���v��Ԃ� */
int valuate(
			Weight weight, Num num, 
			Lecture *lct, Class *cls, /*Bilingual *blg,*/ Teacher *tch, /*int nrgl_lects,*/ int *day, 
			Timetable f, Penalty *penalty)
{
	int i, j, k, l;
	int tmp_w, tmp_d, last, nrg_count = 0;
	int count_days;
	int t, j1, n1, j2, n2, tmp1, tmp2;
	int total = 0;
	int *c;
	int last_day, last_period;
	int check;

	c = (int *)malloc(sizeof(int) * num.week);

	for (i=0; i<num.clsnum; i++)
		(*penalty).prefer[i] = 0;
	for (i=0; i<num.clsnum; i++)
		{
			check = 0;
			for (j=0; j<num.time; j++)
				{
					for (k=0; k<num.tchnum; k++)
						{
							for (l=0; l<cls[i].num; l++)
								if (f[cls[i].lcts[l]][j][k])
									if (lct[cls[i].lcts[l]].avail[j])
										{
											(*penalty).prefer[i] += (DESIRE_LIMIT - lct[cls[i].lcts[l]].avail[j] - 1) * weight.PREFER;
											check++;
											break;
										}
							if (check) break;
						}
					if (check) break;
				}
		}
	for (i=0; i<num.tchnum; i++)
		{
			(*penalty).weekover[i] = 0;
			(*penalty).weekunder[i] = 0;
			(*penalty).daysover[i] = 0;
			(*penalty).daysunder[i] = 0;

/*			for (j=0; j<num.week; j++)
				{
					(*penalty).dayover[i][j] = 0;
					(*penalty).dayunder[i][j] = 0;
				}
			(*penalty).days[i] = 0;
*/
			(*penalty).hole[i] = 0;
			(*penalty).desire[i] = 0;
		}
//	(*penalty).j_n = 0;
//	(*penalty).nrgover = 0;
//	(*penalty).nrgunder = 0;
/*
	for (i=0; i<num.tchnum; i++)
		{
			if (tch[i].regular == 0)   //��΍u�t
				{
					tmp_w = 0;
					count_days = 0;
					last = -1;
					for (j=0; j<num.week; j++)
						{
							tmp_d = 0;
							for (k=0; k<num.period; k++)
								if (t = takelct(num, i, j*num.period+k, f))
									{
										tmp_w += t;
										tmp_d += t;
										if (last >= 0)
											if (last/num.period < j)
													count_days += 1;
//												(*penalty).days[i] += weight.R_DAYS;
											else if ((k - last%num.period) > 1)
												(*penalty).hole[i] += ((k - last%num.period) - 1) * weight.R_HOLE;
										(*penalty).desire[i] += (DESIRE_LIMIT - tch[i].desire[j*num.period+k]-1) * weight.R_DESIRE;
										last = j*num.period+k;
									}
*/
	for (i=0; i<num.tchnum; i++)
		{
			for (j=0; j<num.week; j++) c[j] = 0;
			if (tch[i].regular == 0)
				{
					tmp_w = 0;
					last_day = -1;
					last_period = -1;
					for (j=0; j<num.time; j++)
						if (t = takelct(num, i, j, f))
							{
								tmp_w += t;
								c[day[j]] += t;
								if (last_day == day[j])
									(*penalty).hole[i] += (j - last_period - 1) * weight.R_HOLE;
								(*penalty).desire[i] += (DESIRE_LIMIT - tch[i].desire[j] - 1) * weight.R_DESIRE;
								last_day = day[j];
								last_period = j;
							}
					count_days = 0;
					for (j=0; j<num.week; j++)
						if (c[j]) count_days++;

/*
							if (tmp_d)
								if (tmp_d > tch[i].day_up)
									(*penalty).dayover[i][j] = (tmp_d - tch[i].day_up) * weight.R_DAYOVER;
								else if (tmp_d < tch[i].day_lw)
									(*penalty).dayunder[i][j] = (tch[i].day_lw - tmp_d) * weight.R_DAYUNDER;
*/
//						}
//					if (last) count_days++;
					if (tmp_w > tch[i].week_up)
						(*penalty).weekover[i] = (tmp_w - tch[i].week_up) * weight.R_WEEKOVER;
					else if (tmp_w < tch[i].week_lw)
						(*penalty).weekunder[i] = (tch[i].week_lw - tmp_w) * weight.R_WEEKUNDER;
					if (count_days > tch[i].days_up)
						(*penalty).daysover[i] = (count_days - tch[i].days_up) * weight.R_DAYSOVER;
					else if (count_days < tch[i].days_lw)
						(*penalty).daysunder[i] = (tch[i].days_lw - count_days) * weight.R_DAYSUNDER;
				}
/*
			else   //���΍u�t
				{
					tmp_w = 0;
					count_days = 0;
					last = -1;
					for (j=0; j<num.week; j++)
						{
							tmp_d = 0;
							for (k=0; k<num.period; k++)
								if (t = takelct(num, i, j*num.period+k, f))
									{
										tmp_w += t;
										tmp_d += t;
										nrg_count++;
										if (last >= 0)
											if (last/num.period < j)
													count_days += 1;
//												(*penalty).days[i] += weight.N_DAYS;
											else if ((k - last%num.period) > 1)
												(*penalty).hole[i] += ((k - last%num.period) - 1) * weight.N_HOLE;
										(*penalty).desire[i] += (DESIRE_LIMIT - tch[i].desire[j*num.period+k] - 1) * weight.N_DESIRE;
										last = j*num.period+k;
									}
*/
			else //���΍u�t
				{
					tmp_w = 0;
					last_day = -1;
					last_period = -1;
					for (j=0; j<num.time; j++)
						if (t = takelct(num, i, j, f))
							{
								tmp_w += t;
								c[day[j]] += t;
								if (last_day == day[j])
									(*penalty).hole[i] += (j - last_period - 1) * weight.N_HOLE;
								(*penalty).desire[i] += (DESIRE_LIMIT - tch[i].desire[j] - 1) * weight.N_DESIRE;
								last_day = day[j];
								last_period = j;
							}
					count_days = 0;
					for (j=0; j<num.week; j++)
						if (c[j]) count_days++;
/*
							if (tmp_d)
								if (tmp_d > tch[i].day_up)
									(*penalty).dayover[i][j] = (tmp_d - tch[i].day_up) * weight.N_DAYOVER;
								else if (tmp_d < tch[i].day_lw)
									(*penalty).dayunder[i][j] = (tch[i].day_lw - tmp_d) * weight.N_DAYUNDER;
*/
//						}
//					if (last) count_days++;
					if (tmp_w > tch[i].week_up)
						(*penalty).weekover[i] = (tmp_w - tch[i].week_up) * weight.N_WEEKOVER;
					else if (tmp_w < tch[i].week_lw)
						(*penalty).weekunder[i] = (tch[i].week_lw - tmp_w) * weight.N_WEEKUNDER;
					if (count_days > tch[i].days_up)
						(*penalty).daysover[i] = (count_days - tch[i].days_up) * weight.N_DAYSOVER;
					else if (count_days < tch[i].days_lw)
						(*penalty).daysunder[i] = (tch[i].days_lw - count_days) * weight.N_DAYSUNDER;
				}
		}
/*
	if (nrg_count > nrgl_lects)
		(*penalty).nrgover = (nrg_count - nrgl_lects) * weight.NRGOVER;
	else if (nrg_count < nrgl_lects)
		(*penalty).nrgunder = (nrgl_lects - nrg_count) * weight.NRGUNDER;
	for (i=0; i<num.blgnum; i++)
		{
			j1 = 0;
			n1 = 0;
			j2 = 0;
			n2 = 0;
			for (j=0; j<cls[blg[i].cls1].num; j++)
				{
					t = takenby(num, cls[blg[i].cls1].lcts[j], f);
					if (t >= 0)
						if (tch[t].native == 0)
							j1++;
						else
							n1++;
				}
			for (j=0; j<cls[blg[i].cls2].num; j++)
				{
					t = takenby(num, cls[blg[i].cls2].lcts[j], f);
					if (t >= 0)
						if (tch[t].native == 0)
							j2++;
						else
							n2++;
				}
			tmp1 = (cls[blg[i].cls1].num - j1) + n1 + j2 + (cls[blg[i].cls2].num - n2);
			tmp2 = j1 + (cls[blg[i].cls1].num - n1) + (cls[blg[i].cls2].num - j2) + n2;
			if (tmp1 <= tmp2) (*penalty).j_n += tmp1 * weight.J_N;
			else (*penalty).j_n += tmp2 * weight.J_N;
		}
*/
	(*penalty).total = 0;
	for (i=0; i<num.clsnum; i++)
		(*penalty).total += (*penalty).prefer[i];
	for (i=0; i<num.tchnum; i++)
		{
			(*penalty).total += (*penalty).weekover[i];
			(*penalty).total += (*penalty).weekunder[i];
			(*penalty).total += (*penalty).daysover[i];
			(*penalty).total += (*penalty).daysunder[i];
/*
			for (j=0; j<num.week; j++)
				{
					(*penalty).total += (*penalty).dayover[i][j];
					(*penalty).total += (*penalty).dayunder[i][j];
				}
			(*penalty).total += (*penalty).days[i];
*/
			(*penalty).total += (*penalty).hole[i];
			(*penalty).total += (*penalty).desire[i];
		}
//	(*penalty).total += (*penalty).j_n + (*penalty).nrgover + (*penalty).nrgunder;
	free(c);



	return (*penalty).total;
}


/* �u�`Lct���J�u����鎞�� */
int openat(Num num, int Lct, Timetable f)
{
	int i, j, t = -1;
	for (i=0; i<num.time; i++)
		for (j=0; j<num.tchnum; j++)
			{
				if (f[Lct][i][j]) t = i;
			}
	return t;
}


/* �u�`Lct��S������u�t */
int takenby(Num num, int Lct, Timetable f)
{
	int i, j, t = -1;
	for (i=0; i<num.time; i++)
		for (j=0; j<num.tchnum; j++)
			{
				if (f[Lct][i][j]) t=j;
			}
	return t;
}


/* ���̕]���l��v�]�ʂɕ\�� */
void printval(
			FILE *fpo,
			Weight weight, Num num,
			Lecture *lct, Class *cls, /*Bilingual *blg,*/ Teacher *tch, /*int nrgl_lects,*/ 
			Timetable f, Penalty penalty)
{
	int i, j, k;
	int dov, dun;
	fprintf(fpo, "VALUE:\n");
	fprintf(fpo, "PREFER: ");
	for (i=0; i<num.clsnum; i++)
		fprintf(fpo, "#%d:%d(%d), ", i, penalty.prefer[i], penalty.prefer[i]/weight.PREFER);
	fprintf(fpo, "\n\n");

	fprintf(fpo, "#ID:  weekover weekunder daysover daysunder hole  desire  total\n");

//	fprintf(fpo, "#ID name        weekover weekunder dayover dayunder   days    hole  desire  total\n");
	for (i=0; i<num.tchnum; i++)
		{
/*			dov = 0;
			dun = 0;
			for (j=0; j<num.week; j++)
				{
					dov += penalty.dayover[i][j];
					dun += penalty.dayunder[i][j];
				}
*/
			if (tch[i].regular == 0)
				fprintf(fpo, "#%-2d:  %3d(%2d)   %3d(%2d) %3d(%2d)  %3d(%2d)  %3d(%2d)  %3d    %4d\n",
							i, 
							penalty.weekover[i], penalty.weekover[i]/weight.R_WEEKOVER, 
							penalty.weekunder[i], penalty.weekunder[i]/weight.R_WEEKUNDER,
							penalty.daysover[i], penalty.daysover[i]/weight.R_DAYSOVER,
							penalty.daysunder[i], penalty.daysunder[i]/weight.R_DAYSUNDER,
//							dov, dov/weight.R_DAYOVER, dun, dun/weight.R_DAYUNDER, 
//							penalty.days[i], (penalty.days[i]/weight.R_DAYS)+1, 
							penalty.hole[i], penalty.hole[i]/weight.R_HOLE, 
							penalty.desire[i], 
							penalty.weekover[i] + penalty.weekunder[i] + penalty.daysover[i] + penalty.daysunder[i] + 
								penalty.hole[i] + penalty.desire[i]);
//							penalty.weekover[i] + penalty.weekunder[i] + dov + dun + penalty.days[i] + penalty.hole[i] + penalty.desire[i]);
			else
				fprintf(fpo, "#%-2d:  %3d(%2d)   %3d(%2d) %3d(%2d)  %3d(%2d)  %3d(%2d)  %3d    %4d\n",
							i, 
							penalty.weekover[i], penalty.weekover[i]/weight.N_WEEKOVER, 
							penalty.weekunder[i], penalty.weekunder[i]/weight.N_WEEKUNDER,
							penalty.daysover[i], penalty.daysover[i]/weight.N_DAYSOVER,
							penalty.daysunder[i], penalty.daysunder[i]/weight.N_DAYSUNDER,
//							dov, dov/weight.N_DAYOVER, dun, dun/weight.N_DAYUNDER, 
//							penalty.days[i], (penalty.days[i]/weight.N_DAYS)+1, 
							penalty.hole[i], penalty.hole[i]/weight.N_HOLE, 
							penalty.desire[i],
							penalty.weekover[i] + penalty.weekunder[i] + penalty.daysover[i] + penalty.daysunder[i] + 
								penalty.hole[i] + penalty.desire[i]);
//							penalty.weekover[i] + penalty.weekunder[i] + dov + dun + penalty.days[i] + penalty.hole[i] + penalty.desire[i]);
		}
//	fprintf(fpo, "j_n = %d(%d)\n", penalty.j_n, penalty.j_n/weight.J_N);
//	fprintf(fpo, "nrgover  = %d(%d)\n", penalty.nrgover, penalty.nrgover/weight.NRGOVER);
//	fprintf(fpo, "nrgunder = %d(%d)\n", penalty.nrgunder, penalty.nrgunder/weight.NRGUNDER);
	fprintf(fpo, "total: %d\n\n", penalty.total);
}


/* �Ǐ��T�� */
int local_search(
			Config config, Weight weight, Num num,
			Lecture *lct, Class *cls, /*Bilingual *blg,*/ Teacher *tch, /*int nrgl_lects,*/ int *day, 
			Timetable f, Timetable new_f, Offence *offence, Penalty *penalty, Utility *utility)
{
	int i, j;
	for (i=0; i<config.DEPTH; i++)
		{
			count_offence(num, lct, cls, tch, f, offence);
//			fprintf(stderr, "offence:%5d,  ", (*offence).total);
			valuate(weight, num, lct, cls, /*blg,*/ tch, /*nrgl_lects,*/ day, f,	penalty);
//			fprintf(stderr, "val:%5d\n", (*penalty).total);
			set_utility(num, f, utility);
			if (!search_move(weight, num, lct, cls, /*blg,*/ tch, /*nrgl_lects,*/ day, f, new_f, offence, penalty, *utility) &&
					!search_change(weight, num, lct, cls, /*blg,*/ tch, /*nrgl_lects,*/ day, f, new_f, offence, penalty, *utility) &&
					!search_swap(weight, num, lct, cls, /*blg,*/ tch, /*nrgl_lects,*/ day, f, new_f, offence, penalty, *utility))
				break;
		}
	return i;
}



/*�ߖTmove��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���1��Ԃ��B�Ȃ����0��Ԃ��B*/
int search_move(
			Weight weight, Num num,
			Lecture *lct, Class *cls, /*Bilingual *blg,*/ Teacher *tch, /*int nrgl_lects,*/ int *day, 
			Timetable f, Timetable new_f, Offence *offence, Penalty *penalty, Utility utility)
{
	int i, j, k, l;
	int check = 0;
	int min_i, min_j, min_k, min_l;
	int min_off, min_val, new_off, new_val;

//fprintf(stderr, "searching in neighbor \"move\"...\n");

	min_off = count_offence(num, lct, cls, tch, f, offence);
	min_val = valuate(weight, num, lct, cls, /*blg,*/ tch, /*nrgl_lects,*/ day, f, penalty);

	for (i=0; i<num.clsnum; i++)
		for (j=0; j<num.time; j++)
			if (lct[cls[i].lcts[0]].avail[j])
				for (k=0; k<num.clsnum; k++)
					if (i != k)
						for (l=0; l<num.time; l++)
							if (lct[cls[k].lcts[0]].avail[l])
								{
									make_moved_sol(num, lct, cls, tch, i, j, k, l, f, new_f, utility);
									new_off = count_offence(num, lct, cls, tch, new_f, offence);
									new_val = valuate(weight, num, lct, cls, /*blg,*/ tch, /*nrgl_lects,*/ day, new_f, penalty);
									if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
										{
											min_i = i;
											min_j = j;
											min_k = k;
											min_l = l;
											min_off = new_off;
											min_val = new_val;
											check++;
										}
								}
	if (check) 
		{
			operate_move(num, lct, cls, tch, min_i, min_j, min_k, min_l, f, utility);
			return 1;
		}
	else return 0;
}


/* ��f�ɋߖT����move(cls1, time1, cls2, time2)�����������ւ̃|�C���^��Ԃ� */
void make_moved_sol(
			Num num, Lecture *lct, Class *cls, Teacher *tch, 
			int cls1, int time1, int cls2, int time2, Timetable f, Timetable new_f, Utility utility)
{
	int i, j, k, t;

	for (i=0; i<num.lctnum; i++)
		for (j=0; j<num.time; j++)
			for (k=0; k<num.tchnum; k++)
				new_f[i][j][k] = f[i][j][k];
	for (i=0; i<cls[cls1].num; i++)
		if ((t = utility.lct_takenby[cls[cls1].lcts[i]]) >= 0)
			new_f[cls[cls1].lcts[i]][utility.lct_openat[cls[cls1].lcts[i]]][t] = 0;
	for (i=0; i<cls[cls2].num; i++)
		if ((t = utility.lct_takenby[cls[cls2].lcts[i]]) >= 0)
			new_f[cls[cls2].lcts[i]][utility.lct_openat[cls[cls2].lcts[i]]][t] = 0;
	i = 0;
	for (j = 0; (i<cls[cls1].num && j<num.tchnum); j++)
		if (tch[j].desire[time1] && !takelct(num, j, time1, new_f) &&
				tch[j].qualification[lct[cls[cls1].lcts[i]].qualification])
			{
				new_f[cls[cls1].lcts[i]][time1][j] = 1;
				i++;
			}
	i = 0;
	for (j = 0; (i<cls[cls2].num && j<num.tchnum); j++)
		if (tch[j].desire[time2] && !takelct(num, j, time2, new_f) &&
				tch[j].qualification[lct[cls[cls2].lcts[i]].qualification])
			{
				new_f[cls[cls2].lcts[i]][time2][j] = 1;
				i++;
			}
}


/* ��f�ɋߖT����move(cls1, time1, cls2, time2)�������� */
void operate_move(Num num, Lecture *lct, Class *cls, Teacher *tch, int cls1, int time1, int cls2, int time2, Timetable f, Utility utility)
{
	int i, j, k, t;

	for (i=0; i<cls[cls1].num; i++)
		if ((t = utility.lct_takenby[cls[cls1].lcts[i]]) >= 0)
			f[cls[cls1].lcts[i]][utility.lct_openat[cls[cls1].lcts[i]]][t] = 0;
	for (i=0; i<cls[cls2].num; i++)
		if ((t = utility.lct_takenby[cls[cls2].lcts[i]]) >= 0)
			f[cls[cls2].lcts[i]][utility.lct_openat[cls[cls2].lcts[i]]][t] = 0;
	i = 0;
	for (j = 0; (i<cls[cls1].num && j<num.tchnum); j++)
		if (tch[j].desire[time1] && !takelct(num, j, time1, f) && 
				tch[j].qualification[lct[cls[cls1].lcts[i]].qualification])
			{
				f[cls[cls1].lcts[i]][time1][j] = 1;
				i++;
			}
	i = 0;
	for (j = 0; (i<cls[cls2].num && j<num.tchnum); j++)
		if (tch[j].desire[time2] && !takelct(num, j, time2, f) && 
				tch[j].qualification[lct[cls[cls2].lcts[i]].qualification])
			{
				f[cls[cls2].lcts[i]][time2][j] = 1;
				i++;
			}
}



/* �ߖTchange��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���1��Ԃ��B�Ȃ����0��Ԃ��B */
int search_change(
			Weight weight, Num num,
			Lecture *lct, Class *cls, /*Bilingual *blg,*/ Teacher *tch, /*int nrgl_lects,*/int *day, 
			Timetable f, Timetable new_f, Offence *offence, Penalty *penalty, Utility utility)
{
	int i, j, tc, tm;
	int check = 0;
	int min_i, min_j;
	int min_off, min_val, new_off, new_val;

//fprintf(stderr, "searching in neighbor \"change\"...\n");

	min_off = count_offence(num, lct, cls, tch, f, offence);
	min_val = valuate(weight, num, lct, cls, /*blg,*/ tch, /*nrgl_lects,*/ day, f, penalty);
	for (i=0; i<num.lctnum; i++)
		if ((tc = utility.lct_takenby[i]) >= 0)
			for (j=0; j<num.tchnum; j++)
				if (tc != j)
					{
						tm = utility.lct_openat[i];
						if ( (tch[j].desire[tm] > 0) && (utility.tch_takelct[j][tm] < 0) &&
									tch[j].qualification[lct[i].qualification])

							{
								make_changed_sol(num, i, tm, tc, j, f, new_f);
								new_off = count_offence(num, lct, cls, tch, new_f, offence);
								new_val = valuate(weight, num, lct, cls, /*blg,*/ tch, /*nrgl_lects,*/ day, new_f, penalty);
								if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
									{
										min_i = i;
										min_j = j;
										min_off = new_off;
										min_val = new_val;
										check++;
									}
							}
					}
	if (check) 
		{
			operate_change(num, min_i, utility.lct_openat[min_i], utility.lct_takenby[min_i], min_j, f);
			return 1;
		}
	else return 0;
}



void make_changed_sol(
			Num num, int Lct, int Time, int Tch_c, int Tch_n, Timetable f, Timetable new_f)
{
	int i, j, k;

	for (i=0; i<num.lctnum; i++)
		for (j=0; j<num.time; j++)
			for (k=0; k<num.tchnum; k++)
				new_f[i][j][k] = f[i][j][k];

	new_f[Lct][Time][Tch_c] = 0;
	new_f[Lct][Time][Tch_n] = 1;
}


void operate_change(Num num, int Lct, int Time, int Tch_c, int Tch_n, Timetable f)
{
	f[Lct][Time][Tch_c] = 0;
	f[Lct][Time][Tch_n] = 1;
}


/* �ߖTswap��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���1��Ԃ��B�Ȃ����0��Ԃ��B */
int search_swap(
			Weight weight, Num num,
			Lecture *lct, Class *cls, /*Bilingual *blg,*/ Teacher *tch, /*int nrgl_lects,*/ int *day,
			Timetable f, Timetable new_f, Offence *offence, Penalty *penalty, Utility utility)
{
	int i, j, tc1, tm1, tc2, tm2;
	int check = 0;
	int min_i, min_j;
	int min_off, min_val, new_off, new_val;

//fprintf(stderr, "searching in neighbor \"swap\"...\n");

	min_off = count_offence(num, lct, cls, tch, f, offence);
	min_val = valuate(weight, num, lct, cls, /*blg,*/ tch, /*nrgl_lects,*/ day, f, penalty);
	for (i=0; i<num.lctnum; i++)
		if ((tc1 = utility.lct_takenby[i]) >= 0)
			for (j=0; j<num.lctnum; j++)
				if (i != j)
					if ((tc2 = utility.lct_takenby[j]) >= 0)
						if (tc1 != tc2)
							{
								tm1 = utility.lct_openat[i];
								tm2 = utility.lct_openat[j];
								if ( (tch[tc1].desire[tm2] > 0) && (utility.tch_takelct[tc1][tm2] < 0) &&
										 tch[tc2].qualification[lct[i].qualification] &&
										 (tch[tc2].desire[tm1] > 0) && (utility.tch_takelct[tc2][tm1] < 0) &&
										 tch[tc1].qualification[lct[j].qualification] )
									{
										make_swapped_sol(num, i, tm1, tc1, j, tm2, tc2, f, new_f);
										new_off = count_offence(num, lct, cls, tch, new_f, offence);
										new_val = valuate(weight, num, lct, cls, /*blg,*/ tch, /*nrgl_lects,*/ day, new_f, penalty);
										if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
											{
												min_i = i;
												min_j = j;
												min_off = new_off;
												min_val = new_val;
												check++;
											}
									}
							}
	if (check) 
		{
			operate_swap(num, min_i, utility.lct_openat[min_i], utility.lct_takenby[min_i], 
												min_j, utility.lct_openat[min_j], utility.lct_takenby[min_j], f);
			return 1;
		}
	else return 0;
}


void make_swapped_sol(
			Num num, int lct1, int time1, int tch1, int lct2, int time2, int tch2, Timetable f, Timetable new_f)
{
	int i, j, k;

	for (i=0; i<num.lctnum; i++)
		for (j=0; j<num.time; j++)
			for (k=0; k<num.tchnum; k++)
				new_f[i][j][k] = f[i][j][k];

	new_f[lct1][time1][tch1] = 0;
	new_f[lct2][time2][tch2] = 0;
	new_f[lct1][time1][tch2] = 1;
	new_f[lct2][time2][tch1] = 1;
}


void operate_swap(
			Num num, int lct1, int time1, int tch1, int lct2, int time2, int tch2, Timetable f)
{
	f[lct1][time1][tch1] = 0;
	f[lct2][time2][tch2] = 0;
	f[lct1][time1][tch2] = 1;
	f[lct2][time2][tch1] = 1;
}


void save_best(Config config, Num num, Timetable f, int off, int val, Timetable *best_sol, int *best_off, int *best_val)
{
	int i, j, k, wst = -1, wst_off, wst_val;

	for (i=0; i<config.SAVE; i++)
		if ((wst < 0) || (best_off[i] < 0) || 
				((wst_off >= 0) && (best_off[i] > wst_off)) || 
				((wst_off >= 0) && (best_off[i] == wst_off) && (best_val[i] > wst_val)))
			{
				wst = i;
				wst_off = best_off[i];
				wst_val = best_val[i];
			}
	if ( (wst_off < 0) || ((off < wst_off) || ((off == wst_off) && (val < wst_val))) )
		{
			best_off[wst] = off;
			best_val[wst] = val;
			for (i=0; i<num.lctnum; i++)
				for (j=0; j<num.time; j++)
					for (k=0; k<num.tchnum; k++)
						best_sol[wst][i][j][k] = f[i][j][k];
		}
}


void sort_best(Config config, Num num, Timetable *best_sol, int *best_off, int *best_val)
{
	int i, j, k, l;
	int *sorted;
	int *rest;
	Timetable *sorted_sol;
	int *sorted_off;
	int *sorted_val;
	int lowest;
	int lowest_off;
	int lowest_val;

	sorted = (int *)malloc(sizeof(int) * config.SAVE);
	rest = (int *)malloc(sizeof(int) * config.SAVE);
	sorted_sol = (Timetable *)malloc(sizeof(Timetable) * config.SAVE);
	sorted_off = (int *)malloc(sizeof(int) * config.SAVE);
	sorted_val = (int *)malloc(sizeof(int) * config.SAVE);
	for (i=0; i<config.SAVE; i++)
		{
			rest[i] = 1;
			sorted_sol[i] = (Timetable)malloc(sizeof(int **) * num.lctnum);
			for (j=0; j<num.lctnum; j++)
				{
					sorted_sol[i][j] = (int **)malloc(sizeof(int *) * num.time);
					for (k=0; k<num.time; k++)
						sorted_sol[i][j][k] = (int *)malloc(sizeof(int) * num.tchnum);
				}
			sorted_off[i] = -1;
			sorted_val[i] = -1;
		}

	for (i=0; i<config.SAVE; i++)
		{
			lowest = -1;
			for (j=0; j<config.SAVE; j++)
				{
					if (rest[j])
						if ((lowest < 0) || (best_off[j] < lowest_off) || 
								((best_off[j] == lowest_off) && (best_val[j] < lowest_val)))
							{
								lowest_off = best_off[j];
								lowest_val = best_val[j];
								lowest = j;
							}
				}
			sorted[i] = lowest;
			rest[lowest] = 0;
		}
	for (i=0; i<config.SAVE; i++)
		{
			sorted_off[i] = best_off[sorted[i]];
			sorted_val[i] = best_val[sorted[i]];
			for (j=0; j<num.lctnum; j++)
				for (k=0; k<num.time; k++)
					for (l=0; l<num.tchnum; l++)
						sorted_sol[i][j][k][l] = best_sol[sorted[i]][j][k][l];
		}
	for (i=0; i<config.SAVE; i++)
		{
			best_off[i] = sorted_off[i];
			best_val[i] = sorted_val[i];
			for (j=0; j<num.lctnum; j++)
				for (k=0; k<num.time; k++)
					for (l=0; l<num.tchnum; l++)
						best_sol[i][j][k][l] = sorted_sol[i][j][k][l];
		}
}


void output_timetable(FILE **fpo2, Num num, Lecture *lct, Class *cls, Teacher *tch, int *period_id, int *day_id, Timetable f)
{
	int i, j, k, l;
	fprintf(*fpo2, "task_id,period_id,processor_id,qualification_id,required_processors_lb,required_processors_ub,");
	fprintf(*fpo2, "employment,total_periods_lb,total_periods_ub,total_days_lb,total_days_ub,");
	fprintf(*fpo2, "wage_level,day_id,preferred_level_task,preferred_level_proc\n");
	for (i=0; i<num.lctnum; i++)
		for (j=0; j<num.time; j++)
			for (k=0; k<num.tchnum; k++)
				if (f[i][j][k])
					{
						fprintf(*fpo2, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
									cls[lct[i].class].id, period_id[j], tch[k].id, cls[lct[i].class].qualification_id, cls[lct[i].class].num, cls[lct[i].class].num,
									tch[k].regular, tch[k].week_lw, tch[k].week_up,
									tch[k].days_lw, tch[k].days_up, tch[k].wage, day_id[j],
									lct[i].avail[j], tch[k].desire[j]);
					}
}


void output_best_timetables(Config config, Num num, Lecture *lct, Class *cls, Teacher *tch, int *period_id, int *day_id, Timetable *best_sol)
{
	int i, j, k, n, len, a, b, c;
	char *filename;
	FILE *fp;

	filename = (char *)malloc(sizeof(char) * 100);
	for (i=0; i<config.SAVE; i++)
		{
			strcpy(filename, "/export/public/home/yukiko/public/a/timetables/timetable");
			len = strlen(filename);
			n = 0;
			for (j=10; i/j>0; j*=10) n++;
			k = n;
			for (j=0; j<=n; j++) 
				{
					filename[len + j] = '0' + (i % (int)pow(10, (double)(k+1))) / (int)pow(10, (double)k);
					k--;
				}
			filename[len + j] = '\0';
			fp = fopen(filename, "w");
	fprintf(fp, "task_id,period_id,processor_id,qualification_id,required_processors_lb,required_processors_ub,");
	fprintf(fp, "employment,total_periods_lb,total_periods_ub,total_days_lb,total_days_ub,");
	fprintf(fp, "wage_level,day_id,preferred_level_task,preferred_level_proc\n");
	for (a=0; a<num.lctnum; a++)
		for (b=0; b<num.time; b++)
			for (c=0; c<num.tchnum; c++)
				if (best_sol[i][a][b][c])
					{
						fprintf(fp, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
									cls[lct[a].class].id, period_id[b], tch[c].id, cls[lct[a].class].qualification_id, cls[lct[a].class].num, cls[lct[a].class].num,
									tch[c].regular, tch[c].week_lw, tch[c].week_up,
									tch[c].days_lw, tch[c].days_up, tch[c].wage, day_id[b],
									lct[a].avail[b], tch[c].desire[b]);
					}

//			output_timetable(&fp, num, lct, cls, tch, period_id, day_id, best_sol[i]);
			fclose(fp);
		}
}

