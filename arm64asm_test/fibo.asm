// my-fib.S

    .section __TEXT,__text
    .p2align 2

// fib
// n番目のフィボナッチ数を返す
    .globl _fib
_fib:
    // スタックフレームの確保
    sub sp, sp, #32
    // リンクレジスタとフレームポインタを退避
    stp x29, x30, [sp, #16]

    // n をスタックへ保存
    str w0, [sp, #0]

    // n < 2 か否かを判定
    cmp w0, #2
    blt FIB_LT_2
    b FIB_GE_2
FIB_LT_2:
    // n < 2 の場合は n をそのまま返す
    b FIB_FIN
FIB_GE_2:
    // f(n-1) を計算
    ldr w0, [sp, #0]
    sub w0, w0, #1
    bl _fib

    // f(n-1) の結果をスタックへ保存
    str w0, [sp, #4]

    // f(n-2) を計算
    ldr w0, [sp, #0]
    sub w0, w0, #2
    bl _fib

    // f(n-1) + f(n-2) の結果を w0 へセット
    ldr w9, [sp, #4]
    add w0, w0, w9
FIB_FIN:
    // 退避していたリンクレジスタとスタックフレームを復帰
    ldp x29, x30, [sp, #16]
    // スタックフレームを解放
    add sp, sp, #32
    ret

// main
    .globl _main
_main:
    sub sp, sp, #16
    stp x29, x30, [sp, #0]

    // 10番目のフィボナッチ数を計算
    mov w0, #10
    bl _fib

    ldp x29, x30, [sp, #0]
    add sp, sp, #16
    ret