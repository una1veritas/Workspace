TARGET = hello
ASMSRC = hello.asm
NASMSRC = hello.nasm

clang:
    clang -o $TARGET $ASMSRC

nasm:
    nasm -f macho64 -o $TARGET.o $NASMSRC
    clang -o $TARGET $TARGET.o
