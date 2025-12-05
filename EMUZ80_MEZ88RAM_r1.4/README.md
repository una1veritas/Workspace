# MEZ88_RAM Firmware Rev1.4

# ファームウェアRev1.4での変更点
  ## 1. MS-DOS Ver3.10のサポート
    MS-DOSをVer2.11からVer3.10へバージョンアップしました。
    これによって、FAT16のディスクイメージを扱うことが出来、
    大容量のディスクイメージをサポートします。
    DISKD.DSKはFAT16の30MBのディスクイメージです。

  ## 2. Diskイメージの対応強化
    ファームウェアRev1.3では、ディスクイメージが1.4MBと9.84MB固定でした。
    Rev1.4では起動時にディスクイメージのFAT情報を読み取り、MS-DOSのBPB情報を設定します。
    したがって、ネット上にあるアーカイブ上のディスクイメージをDOSDISKSにコピー
    してファイル名をDRIVEA.DSK～DRIVED.DSKに変更すれば、MS-DOS起動時に自動的に
    認識されます。
    ただし、セクターサイズが512バイトのものに限ります。

  ## 3. DRIVED.DSKに[MS-DOS Ver4.0](https://github.com/microsoft/MS-DOS/tree/main/v4.0)の開発環境

    公開されているMS-DOS Ver4.0のソースコードの中には、Ver4.0をビルド出来る
    強力な開発環境が提供されています。
    DRIVEDのディスクイメージには、この開発環境を収録しました。

# インストール方法

# [MEZ88_RAM Firmware Rev1.3](https://github.com/akih-san/MEZ88_RAM/tree/Rev1.3)を参照してください。
