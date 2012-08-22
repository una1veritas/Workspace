/*******************************************************************************
* RXduinoライブラリ & 特電HAL
* 
* このソフトウェアは特殊電子回路株式会社によって開発されたものであり、当社製品の
* サポートとして提供されます。このライブラリは当社製品および当社がライセンスした
* 製品に対して使用することができます。
* このソフトウェアはあるがままの状態で提供され、内容および動作についての保障はあ
* りません。弊社はファイルの内容および実行結果についていかなる責任も負いません。
* お客様は、お客様の製品開発のために当ソフトウェアのソースコードを自由に参照し、
* 引用していただくことができます。
* このファイルを単体で第三者へ開示・再配布・貸与・譲渡することはできません。
* コンパイル・リンク後のオブジェクトファイル(ELF ファイルまたはMOT,SRECファイル)
* であって、デバッグ情報が削除されている場合は第三者に再配布することができます。
* (C) Copyright 2011-2012 TokushuDenshiKairo Inc. 特殊電子回路株式会社
* http://rx.tokudenkairo.co.jp/
*******************************************************************************/

#ifndef TKDN_SERVO_H_
#define TKDN_SERVO_H_

#include "tkdn_hal.h"

#define MIN_PULSE_WIDTH       544
#define MAX_PULSE_WIDTH       2400

#define SERVO_MIN() (MIN_PULSE_WIDTH - min * 4)  // minimum value in uS for this servo
#define SERVO_MAX() (MAX_PULSE_WIDTH - max * 4)  // maximum value in uS for this servo

#ifdef __cplusplus
extern "C" {
#endif

// サーボ構造体
typedef struct servo_str
{
	int  pinnum;
	int  ontime;
	int  angle_value;
	int  min_pulse_width;
	int  max_pulse_width;
} servo_str;

typedef struct servo_str servo_t;

// サーボの初期化
// servo 初期化したいサーボ構造体へのポインタ
// min = 0°のときのON時間(us)
// max = 180°のときのON時間(us)
// minとmaxの両方に同じ値を入れると、デフォルト値(544,2400)が採用される
TKDN_HAL
void servo_attach(servo_t *servo, int pin,int min,int max);

// 指定された角度に相当する幅のパルスをサーボモータに送る
// 引数angleが上記のminの値を超えた場合は、ON時間(us単位)と解釈され、通常のPWM動作になる
TKDN_HAL
void servo_write(servo_t *servo, int angle);

// 指定されたON時間(us単位)、通常のPWM動作を行う
TKDN_HAL
void servo_write_us(servo_t *servo,int us);

// 最後に指定された角度を読み出す
TKDN_HAL
int  servo_read(servo_t *servo);

// サーボの動作を止め、割り込みを解除する
TKDN_HAL
void servo_detach(servo_t *servo);

#ifdef __cplusplus
 }
#endif

#endif /* TKDN_SERVO_H_ */
