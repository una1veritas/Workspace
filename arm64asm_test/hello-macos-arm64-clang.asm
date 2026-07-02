.global _main
.align 2

_main:
    ; 1. write() システムコールを呼び出す
    mov x0, #1              ; 第1引数: stdout (ファイルディスクリプタ1)
    adrp x1, message@PAGE         ; 第2引数: 表示する文字列のメモリアドレス
    add x1, x1, message@PAGEOFF
    mov x2, #13             ; 第3引数: 文字列の長さ (13バイト)
    mov x16, #4             ; システムコール番号 4 (write)
    svc #0                  ; 割り込み実行 call kernel

    ; 2. exit() システムコールを呼び出す
    mov x0, #0              ; 第1引数: 終了ステータス 0
    mov x16, #1             ; システムコール番号 1 (exit)
    svc #0                  ; 割り込み実行

message:
    .asciz "Hello, World\n"