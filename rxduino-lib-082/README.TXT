＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
       RXduino ＆ 特電HAL   ソースコード＆サンプルコード アーカイブ
       Version 0.82 2012/6/6 beta

     (C) Copyright 2011-2012 特殊電子回路株式会社 All rights reserved.
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★   ご注意！ このプログラム中のライブラリは究極のRaXino用に   ★
★   ビルドされています。他のボードで使う場合には、make use_***** を      ★
★   【第4章参照】実行してから、再度ビルドしてください                    ★
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

■１．概要
このアーカイブは、特電RX62N/RX63N用ライブラリ(HAL)と、RXduinoに関するものです。
HALとRXduinoのすべてのソースコードと、サンプルアプリケーションを含みます。

究極のRX62Nボードや、RX-MEGAをサポート付きでご購入された方は、ソースコードが利用できます。
そうでない方も、ライブラリ(*.a,*.o)を利用できます。

現在は、RaXinoと究極のRX62Nボードに対応しています。
検証が行われているのはRaXinoのみで、究極のRX62Nボードは未検証です。
RX-MEGAへは未対応です。

■２．ディレクトリ構成

lib    --+  各種のスタティックライブラリ(*.a)が納められています
         +- libfatfs.a         RX-ELF-GCC用に移植されたFatFSライブラリです
         +- libliquidcrystal.a キャラクタ液晶ライブラリです
         +- libnokialcd.a      Arduino用のNokiaLCDライブラリです
         +- liboled.a          MARY拡張基板のOLEDライブラリです
         +- librxduino.a       RXduinoライブラリです
         +- libtkdnhal.a       特電HALライブラリです
         +- start.o            スタートアップコードです
         +- gcc/               GCCやnewlibで使用されるライブラリです
         +- fatfs/             RX-ELF-GCC用に移植されたFatFSのソースです
         +- liquidcrystal/     キャラクタ液晶ライブラリのソースです
         +- nokialcd/          Arduino用のNokiaLCDライブラリのソースです
         +- oled/              MARY拡張基板のOLEDライブラリのソースです
         +- tgtbrd/            各ターゲットボード用のライブラリ(バイナリ)です

sample --+  各種のサンプルアプリケーションが納められています
         +- adc                RX62Nの内蔵ADCの操作サンプルです
         +- adbtest            AndroidとADBモードで接続するためのサンプルです(New!)
         +- AE-GYRO-SMD        秋月電子の圧電振動ジャイロの動作サンプルです
         +- ethercapture       イーサネットの通信をキャプチャしてダンプします
         +- fatfs              SD/MMCカードの動作サンプルです
         +- gpio_hal           HAL経由でGPIOを叩くサンプルです
         +- gpio_rxduino       RXduino経由でGPIOを叩くサンプルです
         +- KXM52              加速度センサKXM52の動作サンプルです
         +- liquidcrystal      キャラクタ液晶の動作サンプルです
         +- math_and_printf    printfや数学関数を使うサンプルです
         +- nokialcd           ArduinoのNokiaLCDに長方形をいっぱい描くサンプルです
         +- oled               MARYのOLED基板に長方形をいっぱい描くサンプルです
         +- rtc                RX62Nの内蔵RTCの動作サンプルです
         +- serial_hal         HAL経由でシリアルポートにアクセスするサンプルです
         +- serial_rxduino     RXduino経由でシリアルポートにアクセスするサンプルです
         +- fractal/           MARY拡張基板のOLEDにフラクタルを描画します(★新規)
         +- tcpip/             RX62NのTCP/IPサンプルです(★新規)
         +- nahimon/           RX62N用のモニタ・ローダプログラムです
         +- pwm/               ソフトウェアPWMのサンプルです
         +- tone/              音を鳴らすサンプルです
         +- multisci/          複数のSCIチャネルを同時に使うデモです

common --+  割り込みベクタなど共通して使われるものが入っています

include -+  インクルードファイルが納められています
         +- lib               各種ライブラリのインクルードファイルが納められています
         +- rxduino           RXduinoのインクルードファイルが納められています
         +- tkdnhal           HALのインクルードファイルが納められています

core   --+  RXduinoや特電HALのソースコードです
         +- rxduino           RXduinoのソースコードが納められています
         +- tkdnhal           HALのソースコードが納められています

■３．ユーザアプリケーションの作成方法

まず最初に、RXduinoアプリにするか、特電HALアプリにするかを決めます。

RXduinoアプリはC++で、特電HALアプリはCで記述しますが、
特電HALアプリをC++で記述しても構いません。
また、RXduinoアプリから特電HALの関数を呼び出すこともできるので、
厳密に考える必要はありません。

◆３．１ 特電HALアプリを作る場合
ユーザプログラムの先頭で、tkdn_hal.hをインクルードしてください。
#include <tkdn_hal.h>

プログラムは通常のプログラムと同じく、int main()から開始します。
コンパイルは、
rx-elf-gcc -Wall -g -I./include/tkdnhal -c $(TARGET).c -o $(TARGET).o
で行います。
デバイスライブラリを使う場合は
-I./include/lib 
を追加してください。

リンクはコマンドが大変長くなりますので、makefileを活用してください。

rx-elf-ld -Map $(TARGET).map lib/start.o sample/common/intvect.o $(TARGET).o -T sample/common/rx62n_ram_standalone.ld -Llib/gcc -Llib -lfatfs -ltkdnhal -lstdc++ -lsupc++ -lc -lsim -lgcc -lm -nostartfiles -o $(TARGET).elf

上記のコマンドで$(TARGET)はユーザが作成するアプリケーションの名前に置き換えてください。

このコマンドを簡単に解説します。
・-Map $(TARGET).mapは、リンクの結果どのアドレスに何が配置されたかを記述したファイルを出力します。
・lib/start.oは、スタートアップルーチンです。リセット後に実行するコードが書かれています。
・sample/common/intvect.oは、割り込みベクタと、割り込みハンドラです。例外も処理します。
・$(TARGET).oは、ユーザの作ったプログラムをリンクすることを指示します。$(TARGET).oは、sample/common/intvect.oの次、すなわち各種ライブラリの前に書きます。
・-T sample/common/rx62n_ram_standalone.ldは、RAMアプリ用のリンカスクリプトを指定します。
・-Lの各コマンドは、ライブラリを探すパスを指定します。
・-lfatfs -lliquidcrystal -lnokialcd -loledは、各種のデバイスライブラリを使用する際に指定します。FATFSや液晶などを使わなければ削除しても構いません。
・-ltkdnhalは、特電HALのライブラリをリンクすることを指示します。
・-lstdc++ -lsupc++ -lc -lsim -lgcc -lmは、標準的なライブラリをリンクします。
・-nostartfilesは、標準のスタートアップルーチンを使わないことを指示します。

このコマンドを実行すると、$(TARGET).elfが生成されます。

◆３．２ Rxduinoアプリを作る場合
ユーザプログラムの先頭で、rxduino.hをインクルードしてください。
#include <rxduino.h>

プログラムはsetup()から始まり、loop()を繰り返します。
コンパイルは、
rx-elf-gcc -Wall -g -I./include/rxduino -I./include/tkdnhal -c $(TARGET).cpp -o $(TARGET).o
で行います。デバイスライブラリを使う場合は
-I./include/lib 
を追加してください。

リンクはコマンドが大変長くなりますので、makefileを活用してください。

rx-elf-ld -Map $(TARGET).map lib/start.o sample/common/intvect.o $(TARGET).o -T sample/common/rx62n_ram_standalone.ld -Llib/gcc -Llib -lfatfs -loled -lliquidcrystal -lnokialcd -lrxduino -ltkdnhal -lstdc++ -lsupc++ -lc -lsim -lgcc -lm -nostartfiles -o $(TARGET).elf
特電HALのときと似ていますが、-lrxduinoが追加されています。

◆３．３ ユーザアプリの簡単な作り方
① このディレクトリに、cまたはcppの拡張子でプログラムを作ります。
② ユーザアプリケーションをコンパイルするには、rxbuild ファイル名.cpp とタイプします。

■４．ライブラリの切り替え
本ライブラリとディレクトリは、究極のRX62Nボード(RevA版)用に構築されています。
他のボードで使う場合は、以下のコマンドを実行してください。

#  make use_raxino       RaXino用のユーザアプリをビルドできるように環境をセットする
#  make use_rx62n0_sdram 究極RX62Nボード初期版(SDRAM使用)のアプリをビルドできるように。
#  make use_rx62n0_mmc   究極RX62Nボード初期版(SDカード使用)のアプリをビルドできるように。
#  make use_rx62n        究極RX62NボードRevA版のアプリをビルドできるように環境をセットする
#  make use_rxmega       RXMEGA用のアプリをビルドできるように環境をセットする

これで、ライブラリが切り替わります。

■５．ライブラリの再ビルド
ライブラリ(特電HALとRXduino)のソースを書き換えたりなどして、再度、ライブラリをビルドしたい場合には、
make lib_rxduino
とタイプします。ライブラリ(HAL、RXduino、周辺デバイスライブラリ)と、サンプルアプリケーションが再ビルドされます。

ライブラリとサンプルアプリケーションを一度クリーンアップするには、
make libclean
とタイプします。

■６．サンプルプロジェクトの再ビルド

特定のサンプルプロジェクトをコンパイルするには、
make -C sample/gpio/
のようにしてください。

特定のサンプルプロジェクトをROM化用にコンパイルするには、
make -C sample/ethercapture/rom
のようにしてください。

■７．更新履歴
●Version 0.82 平成24年6月6日
【全般】
・UTF-8化
・ROM化の際の.dataセクションを0x00001000番地に移動 (printfの%xのルーチンと被っていたため)
・動作安定化のための様々な調整
・ビルド用スクリプト rxbuild の提供

【ライブラリ】
・Arduino互換のServoライブラリの追加
・Arduino互換のMMC(または例外的実装であるSD)カードライブラリの追加
・Arduino互換のSPIライブラリをここに移動
・Arduino互換のLiquidcrystalライブラリここに移動
・NokiaLCDライブラリを一時的に対象外とする
・OLEDライブラリを一時的に対象外とする
・Arduino互換のEEPROMライブラリを作りかけ（まだ動かさないこと）

【RXduino】
・analogReadが効くようになった (070ではanalogRead不具合)
・void analogWriteFrequency(int Hz) 関数追加
・pinModeまたはdigitalWriteでPWM停止
・複数のシリアルポートが使用できるようになった
  Serial Serial1 Serial2 Serial3
・Serial.setDefault()実行で、printfの出力をSerialに出力することができるようになった
・SPIライブラリの分離
・Liquidcrystalライブラリの分離

【特電HAL】
・disk用タイマ割り込みルーチンとユーザ用タイマ割り込みルーチンの分離
・USBの文字化け対策
・toneの音質改善
・シリアルポート関係(sci,USB仮想COM)に、peek関数とflush関数を追加
・servo関数を追加

●Version 0.70 平成24年5月19日
・RX63N対応
・常にスーパーバイザモードで動作させるようになった
・ボードのGPIOの構成やメモリ容量などをbrd_***.hとして分離した
  (tkdn_gpio.cやtkdn_gpio.hがボードに依存しないようになった)
・RXduinoライブラリにおける更新点
  ・PROGMEMマクロ実装
  ・analogWriteでPWM出力ができるようになった
  ・shiftIn,ShiftOutの実装
  ・tone関数の実装
  ・attachInterruptとinterrupts関数の実装
  ・wiring互換の低レベル関数実装
  ・算術演算関数実装
・特電HALにおける更新点
  ・シリアルポートが複数チャネル開けるようになった
  ・pwmの実装
  ・tone関数の実装
  ・ピン割り込み機能とグローバル割り込み許可/禁止
  ・数学関数のsqrtを呼び出した際にリンカでエラーが出る問題を防止

●Version 0.66 平成24年4月5日
・割り込み関係のバグフィックス
・USBの割り込み方法を変更
・ハードウェア初期化関数を2回呼んでも問題ないようにした
・nahimon  helpコマンドを作成
・nahimon  SDRMA上にMOTファイルをロードし、実行できるようにした

●Version 0.65 平成24年3月25日
・USBホストライブラリを作成
・AndroidとADBモードで接続するためのサンプルアプリを作成
・nahimonがUSBホスト対応

●Version 0.64 平成24年2月7日
・USB CDCライブラリで、送信する文字が稀に化ける現象を修正
・UARTライブラリで、SCI2とSCI6が使用できるようにした
・様々なボード用のライブラリを用意して、切り替えることができるようにした

