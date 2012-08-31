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

/* 実行モードの設定 */
typedef struct Config_ {
	int MODE;  /* 実行モード。printconfig関数を参照。 */
	int WIDTH; /* 初期解作成→探索　を行う回数 */
	int DEPTH; /* 探索の深さの限界 */
	int SAVE;  /* 保持する解の数 */
	int SEED;  /* ランダム関数のseed */
} Config;

/* 要望違反のペナルティの重み */
typedef struct Weight_ {
	int PREFER;
	int R_WEEKOVER;  /* 常勤講師の一週間の担当超過 */
	int R_WEEKUNDER; /* 常勤講師の一週間の担当不足 */
	int R_DAYSOVER;   /* 常勤講師の担当日数超過 */
	int R_DAYSUNDER;  /* 常勤講師の担当日数不足 */
//	int R_DAYOVER;   /* 常勤講師の担当日の担当超過 */
//	int R_DAYUNDER;  /* 常勤講師の担当日の担当不足 */
//	int R_DAYS;      /* 常勤講師の複数日担当 */
	int R_HOLE;      /* 常勤講師の空き時間 */
	int R_DESIRE;    /* 常勤講師の担当不満度 */
	int N_WEEKOVER;  /* 非常勤講師の一週間の担当超過 */
	int N_WEEKUNDER; /* 非常勤講師の一週間の担当不足 */
	int N_DAYSOVER;   /* 非常勤講師の担当日数超過 */
	int N_DAYSUNDER;  /* 非常勤講師の担当日数不足 */
//	int N_DAYOVER;   /* 非常勤講師の担当日の担当超過 */
//	int N_DAYUNDER;  /* 非常勤講師の担当日の担当不足 */
//	int N_DAYS;      /* 非常勤講師の複数日担当 */
	int N_HOLE;      /* 非常勤講師の空き時間 */
	int N_DESIRE;    /* 非常勤講師の担当不満度 */
//	int J_N;         /* 指定されたクラスの組が日本語講義／英語講義に二分化されていない */
//	int NRGOVER;     /* 非常勤講師全体の担当超過 */
//	int NRGUNDER;    /* 非常勤講師全体の担当不足 */
} Weight;

/*  */
typedef struct Num_ {
	int week;   /* 日数 */
	int period; /* 一日の講義時間の種類（時限の数） */
	int time;   /* week * period */
	int lctnum; /* 講義の数 */
	int clsnum; /* クラスの数 */
//	int blgnum; /* 日本語講義／英語講義に二分化されるべきクラスの組の数 */
	int tchnum; /* 講師の数 */
	int qualificationnum;
} Num;

/* 講義 */
typedef struct Lecture_ {
//  char name[MAXNAME]; /* 講義名 */
  int class;          /* 属するクラス */
  int *avail;         /* 各時間における開講可能／不可能 */
	int qualification;  /* 0:Japanese based, 1:English based, 2:don't care*/
} Lecture;

/* 講師 */
typedef struct Teacher_ {
	int id;
//  char name[MAXNAME]; /* 名前 */
  int *desire; /* 各時間の担当希望度（0の場合は担当不可能） */
  int regular; /* 0:常勤講師 1:非常勤講師 */
  int *qualification;  /* [0]:Japanese based, [1]:English based, [2]:don't care*/
  int week_up; /* 一週間の担当上限 */
  int week_lw; /* 一週間の担当下限 */
  int days_up;  /* 担当日数の上限 */
  int days_lw;  /* 担当日数の下限 */
//  int day_up;  /* 担当日における担当数の上限 */
//  int day_lw;  /* 担当日における担当数の下限 */
	int wage;
} Teacher;

/* クラス */
typedef struct Class_ {
	int id;
  int num;   /* 属する講義の数 */
  int *lcts; /* 属する講義の番号の配列 */
	int qualification_id;
} Class;

/* 日本語講義／英語講義に二分化されるべきクラスの組 */
//typedef struct Bilingual_ {
//  int cls1; /* クラスの番号 */
//  int cls2; /* クラスの番号 */
//} Bilingual;
  
/* 時間割（解） */
typedef int ***Timetable; /* 講義lctを時間timeに講師tchが開講する:[lct][time][tch]=1 */

/* 制約違反 */
typedef struct Offence_ {
	int null_lct;         /* 講義が開講されない */
	int lct_overlap;      /* 同一講義が複数の時間に開講される */
	int class_separation; /* 同一クラスに属するクラスが同時に開講されない */
	int tch_overlap;      /* 同一講師が同時に複数の講義を担当する */
	int not_qualified;		/* ネイティブ（非ネイティブ）が日本語ベース（英語ベース）の講義を担当 */
	int total;            /* 合計 */
} Offence;

/* 要望違反 */
typedef struct Penalty_ {
	int *prefer;			/* 開講不満度 */
	int *weekover;  /* 一週間の担当超過 */
	int *weekunder; /* 一週間の担当不足 */
	int *daysover;  /* 担当日数超過 */
	int *daysunder; /* 担当日数不足 */
//	int **dayover;  /* 担当日の担当超過 */
//	int **dayunder; /* 担当日の担当不足 */
//	int *days;      /* 複数日担当 */
	int *hole;      /* 空き時間 */
	int *desire;    /* 担当不満度 */
//	int j_n;        /* 指定されたクラスの組が日本語講義／英語講義に二分化されていない */
//	int nrgover;    /* 非常勤講師全体の担当超過 */
//	int nrgunder;   /* 非常勤講師全体の担当不足 */
	int total;      /* 合計 */
} Penalty;

/*  */
typedef struct Utility_ {
	int *lct_openat;
	int *lct_takenby;
	int **tch_takelct;
} Utility;


/*ファイルconfigから実行モードを読み込み，ファイルweightから要望違反ペナルティの重みを読み込み，
　ファイルdataからインスタンスデータを読み込み，配列のメモリを確保する*/
void read_and_ready(
			FILE **fpi1, FILE **fpi2, /*FILE **fpi3,*/ FILE **fpi5, FILE **fpi6, FILE **fpi7, 
			Config *config, Weight *weight, Num *num, 
			Lecture **lct, Class **cls, /*Bilingual **blg,*/ Teacher **tch, /*int *nrgl_lects,*/ int **period_id, int **day_id, int **day,
			Timetable *f, Timetable *new_f, Penalty *penalty, Utility *utility, 
			Timetable **best_sol, int **best_off, int **best_val);

/*実行モードを表示*/
void printconfig(FILE *fpo, Config config);

/*要望違反ペナルティの値を表示*/
void printweight(FILE *fpo, Weight weight);

/*インスタンスを表示*/
void printdata(FILE *fpo, Num num, Lecture *lct, Class *cls, /*Bilingual *blg,*/ Teacher *tch/*, int nrgl_lects*/);

/*ファイルsolから解を読み込む*/
void readsol(FILE *fpi4, Num num, Timetable f);

/*解を表示*/
void printsol(FILE *fpo, Num num, Lecture *lct, Teacher *tch, int *day, Timetable f);

/* 初期解作成 */
void initsol(Num num, Lecture *lct, Teacher *tch, Class *cls, Timetable f);

/*  */
void set_utility(Num num, Timetable f, Utility *utility);

/*講師Tchが時間Timeに担当する講義の数*/
int takelct(Num num, int Tch, int Time, Timetable f);

/*講義Lctを時間Timeに担当する講師の人数*/
int taketch(Num num, int Lct, int Time, Timetable f);

/* 解の制約違反を種類別に数えて保持 */
int count_offence(Num num, Lecture *lct, Class *cls, Teacher *tch, Timetable f, Offence *offence);

/* 解の制約違反を種類別に表示 */
void print_offence(FILE *fpo, Offence offence);

/* 解の評価値を要望別に数えて保持し、合計を返す */
int valuate(
			Weight weight, Num num, 
			Lecture *lct, Class *cls, /*Bilingual *blg,*/ Teacher *tch, /*int nrgl_lects,*/ int *day,
			Timetable f, Penalty *penalty);

/* 講義Lctが開講される時間。開講されない場合-1を返す */
int openat(Num num, int Lct, Timetable f);

/* 講義Lctを担当する講師。開講されない場合-1を返す */
int takenby(Num num, int Lct, Timetable f);

/* 解の評価値を要望別に表示 */
void printval(
			FILE *fpo, Weight weight, Num num,
			Lecture *lct, Class *cls, /*Bilingual *blg,*/ Teacher *tch, /*int nrgl_lects, */
			Timetable f, Penalty penalty);

/* 局所探索 */
int local_search(
			Config config, Weight weight, Num num,
			Lecture *lct, Class *cls, /*Bilingual *blg,*/ Teacher *tch, /*int nrgl_lects,*/ int *day, 
			Timetable f, Timetable new_f, Offence *offence, Penalty *penalty, Utility *utility);

/*近傍searchを探索し、改善解があればその中で最良のものに移動し1を返す。なければ0を返す。*/
int search_move(
			Weight weight, Num num,
			Lecture *lct, Class *cls, /*Bilingual *blg,*/ Teacher *tch, /*int nrgl_lects,*/ int *day, 
			Timetable f, Timetable new_f, Offence *offence, Penalty *penalty, Utility utility);

/* 解fに近傍操作move(cls1, time1, cls2, time2)を加えた解new_fを作成 */
void make_moved_sol(
			Num num, Lecture *lct, Class *cls, Teacher *tch, 
			int cls1, int time1, int cls2, int time2, Timetable f, Timetable new_f, Utility utility);

/* 解fに近傍操作move(cls1, time1, cls2, time2)を加える */
void operate_move(Num num, Lecture *lct, Class *cls, Teacher *tch, int cls1, int time1, int cls2, int time2, Timetable f, Utility utility);

/* 近傍changeを探索し、改善解があればその中で最良のものに移動し1を返す。なければ0を返す。 */
int search_change(
			Weight weight, Num num,
			Lecture *lct, Class *cls, /*Bilingual *blg,*/ Teacher *tch, /*int nrgl_lects,*/ int *day, 
			Timetable f, Timetable new_f, Offence *offence, Penalty *penalty, Utility utility);

/* 解fに近傍操作change(Lct, Tch_n)を加えた解new_fを作成 */
void make_changed_sol(Num num, int Lct, int Time, int Tch_c, int Tch_n, Timetable f, Timetable new_f);

/* 解fに近傍操作change(Lct, Tch_n)を加える */
void operate_change(Num num, int Lct, int Time, int Tch_c, int Tch_n, Timetable f);

/* 近傍swapを探索し、改善解があればその中で最良のものに移動し1を返す。なければ0を返す。 */
int search_swap(
			Weight weight, Num num,
			Lecture *lct, Class *cls, /*Bilingual *blg,*/ Teacher *tch, /*int nrgl_lects,*/ int *day, 
			Timetable f, Timetable new_f, Offence *offence, Penalty *penalty, Utility utility);

/* 解fに近傍操作swap(lct1, lct2)を加えた解new_fを作成 */
void make_swapped_sol(
			Num num, int lct1, int time1, int tch1, int lct2, int time2, int tch2, Timetable f, Timetable new_f);

/* 解fに近傍操作swap(lct1, lct2)を加える */
void operate_swap(Num num, int lct1, int time1, int tch1, int lct2, int time2, int tch2, Timetable f);

/*評価値が良い解を保持*/
void save_best(Config config, Num num, Timetable f, int off, int val, Timetable *best_sol, int *best_off, int *best_val);

void sort_best(Config config, Num num, Timetable *best_sol, int *best_off, int *best_val);

void output_timetable(FILE **fpo2, Num num, Lecture *lct, Class *cls, Teacher *tch, int *period_id, int *day_id, Timetable f);

void output_best_timetables(Config config, Num num, Lecture *lct, Class *cls, Teacher *tch, int *period_id, int *day_id, Timetable *best_sol);

/* メモリの開放、ファイルのクローズ */
void free_memory();


int main() 
{
  FILE *fpi1;/*実行モードを格納したファイルへのポインタ*/
	FILE *fpi2;/*重みを格納したファイルへのポインタ*/
//	FILE *fpi3;/**/
	FILE *fpi4;/*解を格納したファイルへのポインタ*/
	FILE *fpi5;/*講義のインスタンスデータを格納したファイルへのポインタ*/
	FILE *fpi6;/*講師のインスタンスデータを格納したファイルへのポインタ*/
	FILE *fpi7;/*講義時間のインスタンスデータを格納したファイルへのポインタ*/
	FILE *fpo;/*出力ファイルへのポインタ*/
	FILE *fpo2;/**/
	Config config;/*実行モード*/
	Weight weight;/*要望違反ペナルティの重み*/
	Num num;/**/
  Lecture *lct;/*講義の配列*/
  Class *cls;/*クラスの配列*/
//  Bilingual *blg;/**/
  Teacher *tch;/*講師の配列*/
	int *period_id;
	int *day;
	int *day_id;
//  int nrgl_lects;/*非常勤講師全体の担当数*/
  Timetable f, new_f;/*解，局所探索用の解*/
	Offence offence;/*制約違反*/
	Penalty penalty;/*要望違反ペナルティ*/
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
	else if (pid == 0) //子プロセス
		{
			fprintf(stdout, "         \t  \t                                  \n");
			exit(0);
		}
	else	//親プロセス
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


/*ファイルconfigから実行モードを読み込み，ファイルweightから要望違反ペナルティを読み込み，
　ファイルdataからインスタンスデータの読み込み，配列のメモリを確保する*/
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


//period_relation, task_relation, processor_relationを一回走査してclsnum, time, tchnumを数える
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



//メモリ確保，データ格納
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


/*実行モードを表示*/
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

/*要望違反ペナルティの値を表示*/
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


/*インスタンスを表示*/
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



/*ファイルsolから解を読み込む*/
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

/*解を表示*/
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


/* 初期解作成 */
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



/*講師Tchが時間Timeに担当する講義の数*/
int takelct(Num num, int Tch, int Time, Timetable f)
{
	int i, count=0;
	for (i=0; i<num.lctnum; i++)
		if (f[i][Time][Tch]) count++;
	return count;
}

/*講義Lctを時間Timeに担当する講師の人数*/
int taketch(Num num, int Lct, int Time, Timetable f)
{
	int i, count=0;
	for (i=0; i<num.tchnum; i++)
		if (f[Lct][Time][i]) count++;
	return count;
}



/* 解の制約違反を種類別に数えて保持し、合計を返す */
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


/* 制約違反を種類別に表示 */
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


/* 解の評価値を要望別に数えて保持し、合計を返す */
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
			if (tch[i].regular == 0)   //常勤講師
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
			else   //非常勤講師
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
			else //非常勤講師
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


/* 講義Lctが開講される時間 */
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


/* 講義Lctを担当する講師 */
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


/* 解の評価値を要望別に表示 */
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


/* 局所探索 */
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



/*近傍moveを探索し、改善解があればその中で最良のものに移動し1を返す。なければ0を返す。*/
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


/* 解fに近傍操作move(cls1, time1, cls2, time2)を加えた解へのポインタを返す */
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


/* 解fに近傍操作move(cls1, time1, cls2, time2)を加える */
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



/* 近傍changeを探索し、改善解があればその中で最良のものに移動し1を返す。なければ0を返す。 */
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


/* 近傍swapを探索し、改善解があればその中で最良のものに移動し1を返す。なければ0を返す。 */
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

