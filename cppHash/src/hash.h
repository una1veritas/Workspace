#ifndef __HASH_H__
#define __HASH_H__

#include <vector>
#include <algorithm>
#include <array>
#include <bitset>


namespace sha {
struct sha256 {
};
struct sha512 {
};
template<typename Tag>
class Hash;
template<>
class Hash<sha256> {
public:
	using Type = unsigned int;
	std::string seed;
	std::vector<uint8_t> messages;
	size_t N;
	static constexpr size_t BLOCK_SIZE = 64;
	static constexpr std::array<Type, 64> K = { 0x428a2f98UL, 0x71374491UL,
			0xb5c0fbcfUL, 0xe9b5dba5UL, 0x3956c25bUL, 0x59f111f1UL,
			0x923f82a4UL, 0xab1c5ed5UL, 0xd807aa98UL, 0x12835b01UL,
			0x243185beUL, 0x550c7dc3UL, 0x72be5d74UL, 0x80deb1feUL,
			0x9bdc06a7UL, 0xc19bf174UL, 0xe49b69c1UL, 0xefbe4786UL,
			0x0fc19dc6UL, 0x240ca1ccUL, 0x2de92c6fUL, 0x4a7484aaUL,
			0x5cb0a9dcUL, 0x76f988daUL, 0x983e5152UL, 0xa831c66dUL,
			0xb00327c8UL, 0xbf597fc7UL, 0xc6e00bf3UL, 0xd5a79147UL,
			0x06ca6351UL, 0x14292967UL, 0x27b70a85UL, 0x2e1b2138UL,
			0x4d2c6dfcUL, 0x53380d13UL, 0x650a7354UL, 0x766a0abbUL,
			0x81c2c92eUL, 0x92722c85UL, 0xa2bfe8a1UL, 0xa81a664bUL,
			0xc24b8b70UL, 0xc76c51a3UL, 0xd192e819UL, 0xd6990624UL,
			0xf40e3585UL, 0x106aa070UL, 0x19a4c116UL, 0x1e376c08UL,
			0x2748774cUL, 0x34b0bcb5UL, 0x391c0cb3UL, 0x4ed8aa4aUL,
			0x5b9cca4fUL, 0x682e6ff3UL, 0x748f82eeUL, 0x78a5636fUL,
			0x84c87814UL, 0x8cc70208UL, 0x90befffaUL, 0xa4506cebUL,
			0xbef9a3f7UL, 0xc67178f2UL };
	std::array<Type, 8> H = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
			0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };

public:
	Type Rot(Type v, int n) {
		return (v << n) | (v >> (32 - n));
	}
	Type Shr(Type v, int n) {
		return (v >> n);
	}
	Type Sigma0(Type x) {
		return Rot(x, 30) ^ Rot(x, 19) ^ Rot(x, 10);
	}
	Type Sigma1(Type x) {
		return Rot(x, 26) ^ Rot(x, 21) ^ Rot(x, 7);
	}
	Type sigma0(Type x) {
		return Rot(x, 25) ^ Rot(x, 14) ^ Shr(x, 3);
	}
	Type sigma1(Type x) {
		return Rot(x, 15) ^ Rot(x, 13) ^ Shr(x, 10);
	}
	Type Ch(Type x, Type y, Type z) {
		return (x & y) ^ (~x & z);
	}
	Type Maj(Type x, Type y, Type z) {
		return (x & y) ^ (y & z) ^ (x & z);
	}
	auto padding(const std::string &input) {
		size_t size = input.size();
		size_t padding_size = ((size + 9 + BLOCK_SIZE - 1) / BLOCK_SIZE)
				* BLOCK_SIZE;
		std::vector<uint8_t> ret(padding_size);
		for (size_t i = 0; i < size; i++) {
			ret[i] = static_cast<uint8_t>(input[i]);
		}
		ret[size] = 0x80;
		size *= 8;
		ret[padding_size - 4] = static_cast<Type>(size >> 24) & 0xff;
		ret[padding_size - 3] = static_cast<Type>(size >> 16) & 0xff;
		ret[padding_size - 2] = static_cast<Type>(size >> 8) & 0xff;
		ret[padding_size - 1] = static_cast<Type>(size) & 0xff;
		return ret;
	}
	Type loadMessge(const std::vector<uint8_t> &arr, int index) {
		Type load { };
		load |= static_cast<Type>(arr[index]) << 24;
		load |= static_cast<Type>(arr[index + 1]) << 16;
		load |= static_cast<Type>(arr[index + 2]) << 8;
		load |= static_cast<Type>(arr[index + 3]);
		return load;
	}
	Hash(const std::string &str) :
			seed(str), messages(padding(str)) {
	}
	auto operator()() {
		size_t size = messages.size();
		for (size_t num = 0; num < size; num += 64) {
			Type T1 { }, T2 { }, s0 { }, s1 { };
			std::array<Type, 16> X;
			std::array<Type, 8> arr;
			for (int i = 0; i < 8; ++i) {
				arr[i] = H[i];
			}
			auto& [a, b, c, d, e, f, g, h] = arr;
			for (int t = 0; t < 64; ++t) {
				if (t < 16) {
					T1 = X[t] = loadMessge(messages, t * 4);
				} else {
					T1 = X[t & 0xf] += sigma0(X[(t + 1) & 0x0f])
							+ sigma1(X[(t + 14) & 0x0f]) + X[(t + 9) & 0xf];
				}
				T1 += h + Sigma1(e) + Ch(e, f, g) + K[t];
				T2 = Sigma0(a) + Maj(a, b, c);
				h = g;
				g = f;
				f = e;
				e = d + T1;
				d = c;
				c = b;
				b = a;
				a = T1 + T2;
			}

			for (int i = 0; i < 8; ++i) {
				H[i] += arr[i];
			}
		}
		return std::make_pair(seed, H);
	}
	static Hash createHash(const std::string &str) {
		Hash ret(str);
		return ret;
	}
};

template<>
class Hash<sha512> {
public:
	using Type = uint64_t;
	std::string seed;
	std::vector<uint8_t> messages;
	size_t N;
	static constexpr size_t BLOCK_SIZE = 128;
	static constexpr std::array<Type, 80> K = { 0x428a2f98d728ae22,
			0x7137449123ef65cd, 0xb5c0fbcfec4d3b2f, 0xe9b5dba58189dbbc,
			0x3956c25bf348b538, 0x59f111f1b605d019, 0x923f82a4af194f9b,
			0xab1c5ed5da6d8118, 0xd807aa98a3030242, 0x12835b0145706fbe,
			0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2, 0x72be5d74f27b896f,
			0x80deb1fe3b1696b1, 0x9bdc06a725c71235, 0xc19bf174cf692694,
			0xe49b69c19ef14ad2, 0xefbe4786384f25e3, 0x0fc19dc68b8cd5b5,
			0x240ca1cc77ac9c65, 0x2de92c6f592b0275, 0x4a7484aa6ea6e483,
			0x5cb0a9dcbd41fbd4, 0x76f988da831153b5, 0x983e5152ee66dfab,
			0xa831c66d2db43210, 0xb00327c898fb213f, 0xbf597fc7beef0ee4,
			0xc6e00bf33da88fc2, 0xd5a79147930aa725, 0x06ca6351e003826f,
			0x142929670a0e6e70, 0x27b70a8546d22ffc, 0x2e1b21385c26c926,
			0x4d2c6dfc5ac42aed, 0x53380d139d95b3df, 0x650a73548baf63de,
			0x766a0abb3c77b2a8, 0x81c2c92e47edaee6, 0x92722c851482353b,
			0xa2bfe8a14cf10364, 0xa81a664bbc423001, 0xc24b8b70d0f89791,
			0xc76c51a30654be30, 0xd192e819d6ef5218, 0xd69906245565a910,
			0xf40e35855771202a, 0x106aa07032bbd1b8, 0x19a4c116b8d2d0c8,
			0x1e376c085141ab53, 0x2748774cdf8eeb99, 0x34b0bcb5e19b48a8,
			0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb, 0x5b9cca4f7763e373,
			0x682e6ff3d6b2b8a3, 0x748f82ee5defb2fc, 0x78a5636f43172f60,
			0x84c87814a1f0ab72, 0x8cc702081a6439ec, 0x90befffa23631e28,
			0xa4506cebde82bde9, 0xbef9a3f7b2c67915, 0xc67178f2e372532b,
			0xca273eceea26619c, 0xd186b8c721c0c207, 0xeada7dd6cde0eb1e,
			0xf57d4f7fee6ed178, 0x06f067aa72176fba, 0x0a637dc5a2c898a6,
			0x113f9804bef90dae, 0x1b710b35131c471b, 0x28db77f523047d84,
			0x32caab7b40c72493, 0x3c9ebe0a15c9bebc, 0x431d67c49c100d4c,
			0x4cc5d4becb3e42b6, 0x597f299cfc657e2a, 0x5fcb6fab3ad6faec,
			0x6c44198c4a475817 };
	std::array<Type, 8> H = { 0x6a09e667f3bcc908, 0xbb67ae8584caa73b,
			0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1, 0x510e527fade682d1,
			0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179 };

public:
	Type Rot(Type v, int n) {
		return (v >> n) | (v << (64 - n));
	}
	Type Shr(Type v, int n) {
		return (v >> n);
	}
	Type Sigma0(Type x) {
		return Rot(x, 28) ^ Rot(x, 34) ^ Rot(x, 39);
	}
	Type Sigma1(Type x) {
		return Rot(x, 14) ^ Rot(x, 18) ^ Rot(x, 41);
	}
	Type sigma0(Type x) {
		return Rot(x, 1) ^ Rot(x, 8) ^ Shr(x, 7);
	}
	Type sigma1(Type x) {
		return Rot(x, 19) ^ Rot(x, 61) ^ Shr(x, 6);
	}
	Type Ch(Type x, Type y, Type z) {
		return (x & y) ^ (~x & z);
	}
	Type Maj(Type x, Type y, Type z) {
		return (x & y) ^ (y & z) ^ (x & z);
	}
	auto padding(const std::string &input) {
		size_t size = input.size();
		size_t padding_size = ((size + 17 + BLOCK_SIZE - 1) / BLOCK_SIZE)
				* BLOCK_SIZE;
		std::vector<uint8_t> ret(padding_size);
		for (int i = 0; i < size; i++) {
			ret[i] = static_cast<uint8_t>(input[i]);
		}
		ret[size] = 0x80;
		size *= 8;
		ret[padding_size - 8] = static_cast<Type>(size >> 56) & 0xff;
		ret[padding_size - 7] = static_cast<Type>(size >> 48) & 0xff;
		ret[padding_size - 6] = static_cast<Type>(size >> 40) & 0xff;
		ret[padding_size - 5] = static_cast<Type>(size >> 32) & 0xff;
		ret[padding_size - 4] = static_cast<Type>(size >> 24) & 0xff;
		ret[padding_size - 3] = static_cast<Type>(size >> 16) & 0xff;
		ret[padding_size - 2] = static_cast<Type>(size >> 8) & 0xff;
		ret[padding_size - 1] = static_cast<Type>(size) & 0xff;
		// for (auto& a : ret) {
		//   std::cout << std::setfill('0') << std::setw(2) << std::hex
		//             << std::bitset<8>(a) << std::endl;
		// }
		std::cout.setf(std::ios::dec, std::ios::basefield);
		std::cout << "size is " << ret.size() * 8 << std::endl;
		return ret;
	}
	Type loadMessge(const std::vector<uint8_t> &arr, int index) {
		Type load { };
		load |= static_cast<Type>(arr[index]) << 56;
		load |= static_cast<Type>(arr[index + 1]) << 48;
		load |= static_cast<Type>(arr[index + 2]) << 40;
		load |= static_cast<Type>(arr[index + 3]) << 32;
		load |= static_cast<Type>(arr[index + 4]) << 24;
		load |= static_cast<Type>(arr[index + 5]) << 16;
		load |= static_cast<Type>(arr[index + 6]) << 8;
		load |= static_cast<Type>(arr[index + 7]);
		return load;
	}
	Hash(const std::string &str) :
			seed(str), messages(padding(str)) {
	}
	auto operator()() {
		size_t size = messages.size();
		for (int num = 0; num < size; num += 128) {
			Type T1 { }, T2 { }, s0 { }, s1 { };
			std::array<Type, 16> X;
			std::array<Type, 8> arr;
			for (int i = 0; i < 8; ++i) {
				arr[i] = H[i];
			}
			auto& [a, b, c, d, e, f, g, h] = arr;
			for (int t = 0; t < 80; ++t) {
				if (t < 16) {
					T1 = X[t] = loadMessge(messages, t * 8);
				} else {
					T1 = X[t & 0xf] += sigma0(X[(t + 1) & 0x0f])
							+ sigma1(X[(t + 14) & 0x0f]) + X[(t + 9) & 0xf];
				}
				T1 += h + Sigma1(e) + Ch(e, f, g) + K[t];
				T2 = Sigma0(a) + Maj(a, b, c);
				h = g;
				g = f;
				f = e;
				e = d + T1;
				d = c;
				c = b;
				b = a;
				a = T1 + T2;
			}

			std::cout << "num is " << num << std::endl;
			for (int i = 0; i < 8; ++i) {
				H[i] += arr[i];
				std::cout << std::setfill('0') << std::setw(16) << std::hex
						<< arr[i] << std::endl;
			}
		}
		return std::make_pair(seed, H);
	}

	static Hash createHash(const std::string &str) {
		Hash ret(str);
		return ret;
	}
};
}  // namespace sha

#endif /* __HASH_H__ */
