//============================================================================
// Name        : GMP_test.cpp
// Author      : Sin Shimozono
// Version     :
// Copyright   : GPL
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>

#include <gmpxx.h> // GMPヘッダのインクルード

using namespace std;

int main() {
	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!

    // 多倍長整数変数の宣言
    mpz_t a, b, sum;

    // 変数の初期化（計算前に必ず必要）
    mpz_init(a);
    mpz_init(b);
    mpz_init(sum);

    // 文字列から巨大な数値をセットする（10進数）
    mpz_set_str(a, "123456789012345678901234567890", 10);
    mpz_set_str(b, "987654321098765432109876543210", 10);

    // 足し算の実行: sum = a + b
    mpz_add(sum, a, b);

    // 結果の出力（10進数）
    gmp_printf("a + b = %Zd\n", sum);

    mpz_clear(a);
    mpz_clear(b);
    mpz_clear(sum);

    //

    mpq_t x, y, result;

    mpq_init(x);
    mpq_init(y);
    mpq_init(result);

    // Set value as strings (Format: "numerator/denominator")
    mpq_set_str(x, "2/3", 10);
    mpq_set_str(y, "3/4", 10);

    // Always canonicalize after setting or modifying numerators/denominators directly
    mpq_canonicalize(x);
    mpq_canonicalize(y);

    // Multiply fractions: result = a * b  (2/3 * 3/4 = 6/12 = 1/2)
    mpq_mul(result, x, y);

    // Output rational number
    gmp_printf("Multiplication (2/3 * 3/4): %Qd\n", result);

    // メモリの解放（最後に必ず必要）
    mpq_clear(x);
    mpq_clear(y);
    mpq_clear(result);

    return 0;
}
