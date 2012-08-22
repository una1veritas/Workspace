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

#ifndef TKDN_I2C_H_
#define TKDN_I2C_H_

#ifdef __cplusplus
	extern "C" {
#endif

typedef struct i2c_private i2c_info;

/*! @enum	I2C_ACK_STATE
	@brief	I2Cでバイトデータを送った際にスレーブからの値を返す
*/
typedef enum {
	ACK = 0,	/*!< ACK(入力バイトは承認された) */
	NAK			/*!< NAK(入力バイトは承認されなかった) */
} I2C_ACK_STATE;

/*********************************************************************//**
 * @brief		I2Cを初期化する.
 * 				I2C専用ハードウェアを使用していないのでボード上の任意のピンをI2C通信用に指定できます。
 * @param[in]	sda データ入出力ピン
 * @param[in]	scl クロック入出力ピン
 *
 * @return		i2c_private
 **********************************************************************/
i2c_info *i2c_init(int sda, int scl,int address_if_im_slave);

/*********************************************************************//**
 * @brief		指定したアドレスのI2Cスレーブに対して送信処理を始める。
 * 				この関数の実行後、i2c_write、i2c_end_transmissionで送信を実行します。
 * @param[in]	i2c I2Cの各種パラメータ用構造体(i2c_info)
 * @param[in]	address スレーブアドレス
 *
 * @return		i2c_info
 **********************************************************************/
void i2c_begin_transmission(i2c_info *i2c,unsigned char address);

/*********************************************************************//**
 * @brief		スレーブデバイスがマスタからのリクエストに応じてデータを送信するときと、
 * 				マスタがスレーブに送信するデータをキューに入れるときに使用する。
 * 				i2c_begin_transmissionとi2c_end_transmissionの間で実行します。
 * @param[in]	i2c I2Cの各種パラメータ用構造体(i2c_info)
 * @param[in]	data 配列(char*)
 * @param[in]	nbytes 送信するバイト数(unsigned char)
 *
 * @return		I2C_ACK_STATE
 * 				- 0 : ACK
 * 				- 1 : NAK
 **********************************************************************/
I2C_ACK_STATE i2c_write(i2c_info *i2c,unsigned char* data, unsigned char nbytes);

/*********************************************************************//**
 * @brief		マスタがスレーブからのデータを受信するときに使用する。
 * 				i2c_begin_transmissionとi2c_end_transmissionの間で実行します。
 * @param[in]	i2c I2Cの各種パラメータ用構造体(i2c_info)
 * @param[in]	data 受信する配列(char*)
 * @param[in]	nbytes 受信するバイト数(unsigned char)
 *
 * @return		I2C_ACK_STATE
 * 				- 0 : ACK
 * 				- 1 : NAK
 **********************************************************************/
I2C_ACK_STATE i2c_read(i2c_info *i2c, char* data, unsigned char nbytes);

/*********************************************************************//**
 * @brief		スレーブデバイスに対する送信を完了する
 * @param[in]	i2c I2Cの各種パラメータ用構造体(i2c_info)
 *
 * @return		(unsigned char)
 * 				- 0 : 成功
 * 				- 1 : 送ろうとしたデータが送信バッファのサイズを超えた
 * 				- 2 : スレーブ・アドレスを送信し、NACKを受信した
 * 				- 3 : データ・バイトを送信し、NACKを受信した
 * 				- 4 : その他のエラー
 **********************************************************************/
unsigned char i2c_end_transmission(i2c_info *i2c);

/*********************************************************************//**
 * @brief		スレーブデバイスに対する受信を要求する
 * @param[in]	i2c I2Cの各種パラメータ用構造体(i2c_info)
 * @param[in]	data 受信する配列(char*)
 * @param[in]	nbytes 受信要求するバイト数(unsigned char)
 *
 * @return		実際に受信できたバイト数(unsigned char)
 **********************************************************************/
unsigned char i2c_request_from(i2c_info *i2c,unsigned char address, unsigned char* data, unsigned char counter);


#ifdef __cplusplus
	}
#endif

#endif /* TKDN_I2C_H_ */
