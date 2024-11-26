#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>

#include <openssl/sha.h>

#include "hash.h"

int main() {
  std::string message = "it is helloworld,and after tomorrow day.";
  {
    auto start = std::chrono::system_clock::now();

    unsigned char digest[SHA256_DIGEST_LENGTH];

    SHA256_CTX sha_ctx;
    SHA256_Init(&sha_ctx);  // コンテキストを初期化
    SHA256_Update(&sha_ctx, message.c_str(),
                  message.size());   // message を入力にする
    SHA256_Final(digest, &sha_ctx);  // digest に出力

    auto end = std::chrono::system_clock::now();
    double elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    std::cout << elapsed << " nano seconds " << std::endl;
    for (size_t i = 0; i < sizeof(digest); ++i) {
      std::cout << std::setfill('0') << std::setw(2) << std::hex
                << static_cast<unsigned int>(digest[i]);
      // printf("%x", digest[i]);
    }
    std::cout << std::endl;
    // 処理
  }
  {
    auto start = std::chrono::system_clock::now();
    using namespace sha;
    auto hash = Hash<sha256>::createHash(message);
    auto [seed, hashArray] = hash();
    // std::cout << "seed is " << seed << std::endl;

    auto end = std::chrono::system_clock::now();
    double elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    std::cout << elapsed << " nano seconds " << std::endl;
    for (size_t i = 0; i < hashArray.size(); ++i) {
      std::cout << std::hex << std::setfill('0') << std::setw(8)
                << hashArray[i];
    }
    std::cout << std::endl;
  }

  {
    std::string message = "";
    auto start = std::chrono::system_clock::now();

    unsigned char hash[SHA512_DIGEST_LENGTH];
    SHA512_CTX c;
    SHA512_Init(&c);
    SHA512_Update(&c, message.c_str(), message.size());
    SHA512_Final(hash, &c);
    auto end = std::chrono::system_clock::now();
    double elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    std::cout << elapsed << " nano seconds " << std::endl;
    std::cout << "size is " << sizeof(hash) << std::endl;
    for (size_t i = 0; i < sizeof(hash); ++i) {
      std::cout << std::setfill('0') << std::setw(2) << std::hex
                << static_cast<unsigned int>(hash[i]);
      // printf("%x", digest[i]);
    }
    std::cout << std::endl;
    // 処理
  }
  {
    auto start = std::chrono::system_clock::now();
    using namespace sha;
    auto hash = Hash<sha512>::createHash("");
    auto [seed, hashArray] = hash();
    // std::cout << "seed is " << seed << std::endl;

    auto end = std::chrono::system_clock::now();
    double elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    std::cout << elapsed << " nano seconds " << std::endl;
    std::cout << "size is " << hashArray.size() << std::endl;
    for (size_t i = 0; i < hashArray.size(); ++i) {
      std::cout << std::setfill('0') << std::setw(16) << std::hex
                << hashArray[i] << std::endl;
    }
    std::cout << std::endl;
  }
  return 0;
}
