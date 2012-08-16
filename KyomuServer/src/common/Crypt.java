package common;
import java.awt.*;
import java.util.*;
import java.io.*;
import java.net.*;
import java.lang.*;
import java.awt.event.*;

public class Crypt {
  static long[] ary = new long[4];
  
  public Crypt() {  }
  
  private long SBA(long[] sb, long v) {
    long ret = 0;
    int i, offset = (int)v / 8, address = (int)v % 8;

    for (i = 0; i < 8; i++) {
      ret |= ((sb[offset] >> ((7 - address) * 8)) & 0x00000000000000ff) <<
	(7 - i) * 8;
      address++;
      if (address >= 8) {
	offset++; address = 0;
      }
    }
    return ret;
  }

  private long[] _ufc_doit(long l1, long l2, long r1, long r2, long itr)
    { 
      int i, k;
      long l, r, s;
      
      l = ((l1 << 32) | l2);
      r = ((r1 << 32) | r2);
      
      while(itr > 0) {
	itr--;
	k = 0;
	for(i = 7 ; i >= 0; i--) {
	  s = _ufc_keytab[k++] ^ r;
	  l ^= SBA(_ufc_sb3, (s >>  0) & 0xffff);
	  l ^= SBA(_ufc_sb2, (s >> 16) & 0xffff);
	  l ^= SBA(_ufc_sb1, (s >> 32) & 0xffff);
	  l ^= SBA(_ufc_sb0, (s >> 48) & 0xffff);
	  
	  s = _ufc_keytab[k++] ^ l;
	  r ^= SBA(_ufc_sb3, (s >>  0) & 0xffff);
	  r ^= SBA(_ufc_sb2, (s >> 16) & 0xffff);
	  r ^= SBA(_ufc_sb1, (s >> 32) & 0xffff);
	  r ^= SBA(_ufc_sb0, (s >> 48) & 0xffff);
	} 
	s=l; l=r; r=s;
      }
      
      ary[0] = l >> 32; ary[1] = l & 0xffffffff;
      ary[2] = r >> 32; ary[3] = r & 0xffffffff;
      return ary;
    }

  static int[] pc1 = { 
    57, 49, 41, 33, 25, 17,  9,  1, 58, 50, 42, 34, 26, 18,
    10,  2, 59, 51, 43, 35, 27, 19, 11,  3, 60, 52, 44, 36,
    63, 55, 47, 39, 31, 23, 15,  7, 62, 54, 46, 38, 30, 22,
    14,  6, 61, 53, 45, 37, 29, 21, 13,  5, 28, 20, 12,  4
    };
  
  static int[] rots = { 
    1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1 
    };
  
  static int[] pc2 = { 
    14, 17, 11, 24,  1,  5,  3, 28, 15,  6, 21, 10,
    23, 19, 12,  4, 26,  8, 16,  7, 27, 20, 13,  2,
    41, 52, 31, 37, 47, 55, 30, 40, 51, 45, 33, 48,
    44, 49, 39, 56, 34, 53, 46, 42, 50, 36, 29, 32
    };
  
  static int[] esel = { 
    32,  1,  2,  3,  4,  5,  4,  5,  6,  7,  8,  9,
    8,  9, 10, 11, 12, 13, 12, 13, 14, 15, 16, 17,
    16, 17, 18, 19, 20, 21, 20, 21, 22, 23, 24, 25,
    24, 25, 26, 27, 28, 29, 28, 29, 30, 31, 32,  1
    };
  
  static int[] e_inverse = new int[64];

  static int[] perm32 = {
    16,  7, 20, 21, 29, 12, 28, 17,  1, 15, 23, 26,  5, 18, 31, 10,
    2,   8, 24, 14, 32, 27,  3,  9, 19, 13, 30,  6, 22, 11,  4, 25
    };
  
  static int[][][] sbox = {
    { { 14,  4, 13,  1,  2, 15, 11,  8,  3, 10,  6, 12,  5,  9,  0,  7 },
      {  0, 15,  7,  4, 14,  2, 13,  1, 10,  6, 12, 11,  9,  5,  3,  8 },
      {  4,  1, 14,  8, 13,  6,  2, 11, 15, 12,  9,  7,  3, 10,  5,  0 },
      { 15, 12,  8,  2,  4,  9,  1,  7,  5, 11,  3, 14, 10,  0,  6, 13 }
    },
    
    { { 15,  1,  8, 14,  6, 11,  3,  4,  9,  7,  2, 13, 12,  0,  5, 10 },
      {  3, 13,  4,  7, 15,  2,  8, 14, 12,  0,  1, 10,  6,  9, 11,  5 },
      {  0, 14,  7, 11, 10,  4, 13,  1,  5,  8, 12,  6,  9,  3,  2, 15 },
      { 13,  8, 10,  1,  3, 15,  4,  2, 11,  6,  7, 12,  0,  5, 14,  9 }
    },
    
    { { 10,  0,  9, 14,  6,  3, 15,  5,  1, 13, 12,  7, 11,  4,  2,  8 },
      { 13,  7,  0,  9,  3,  4,  6, 10,  2,  8,  5, 14, 12, 11, 15,  1 },
      { 13,  6,  4,  9,  8, 15,  3,  0, 11,  1,  2, 12,  5, 10, 14,  7 },
      {  1, 10, 13,  0,  6,  9,  8,  7,  4, 15, 14,  3, 11,  5,  2, 12 }
    },
    
    { {  7, 13, 14,  3,  0,  6,  9, 10,  1,  2,  8,  5, 11, 12,  4, 15 },
      { 13,  8, 11,  5,  6, 15,  0,  3,  4,  7,  2, 12,  1, 10, 14,  9 },
      { 10,  6,  9,  0, 12, 11,  7, 13, 15,  1,  3, 14,  5,  2,  8,  4 },
      {  3, 15,  0,  6, 10,  1, 13,  8,  9,  4,  5, 11, 12,  7,  2, 14 }
    },
    
    { {  2, 12,  4,  1,  7, 10, 11,  6,  8,  5,  3, 15, 13,  0, 14,  9 },
      { 14, 11,  2, 12,  4,  7, 13,  1,  5,  0, 15, 10,  3,  9,  8,  6 },
      {  4,  2,  1, 11, 10, 13,  7,  8, 15,  9, 12,  5,  6,  3,  0, 14 },
      { 11,  8, 12,  7,  1, 14,  2, 13,  6, 15,  0,  9, 10,  4,  5,  3 }
    },
    
    { { 12,  1, 10, 15,  9,  2,  6,  8,  0, 13,  3,  4, 14,  7,  5, 11 },
      { 10, 15,  4,  2,  7, 12,  9,  5,  6,  1, 13, 14,  0, 11,  3,  8 },
      {  9, 14, 15,  5,  2,  8, 12,  3,  7,  0,  4, 10,  1, 13, 11,  6 },
      {  4,  3,  2, 12,  9,  5, 15, 10, 11, 14,  1,  7,  6,  0,  8, 13 }
    },
    
    { {  4, 11,  2, 14, 15,  0,  8, 13,  3, 12,  9,  7,  5, 10,  6,  1 },
      { 13,  0, 11,  7,  4,  9,  1, 10, 14,  3,  5, 12,  2, 15,  8,  6 },
      {  1,  4, 11, 13, 12,  3,  7, 14, 10, 15,  6,  8,  0,  5,  9,  2 },
      {  6, 11, 13,  8,  1,  4, 10,  7,  9,  5,  0, 15, 14,  2,  3, 12 }
    },
    
    { { 13,  2,  8,  4,  6, 15, 11,  1, 10,  9,  3, 14,  5,  0, 12,  7 },
      {  1, 15, 13,  8, 10,  3,  7,  4, 12,  5,  6, 11,  0, 14,  9,  2 },
      {  7, 11,  4,  1,  9, 12, 14,  2,  0,  6, 10, 13, 15,  3,  5,  8 },
      {  2,  1, 14,  7,  4, 10,  8, 13, 15, 12,  9,  0,  3,  5,  6, 11 }
    }
  };
  
  static int[] initial_perm = { 
    58, 50, 42, 34, 26, 18, 10,  2, 60, 52, 44, 36, 28, 20, 12, 4,
    62, 54, 46, 38, 30, 22, 14,  6, 64, 56, 48, 40, 32, 24, 16, 8,
    57, 49, 41, 33, 25, 17,  9,  1, 59, 51, 43, 35, 27, 19, 11, 3,
    61, 53, 45, 37, 29, 21, 13,  5, 63, 55, 47, 39, 31, 23, 15, 7
    };
  
  static int[] final_perm = {
    40,  8, 48, 16, 56, 24, 64, 32, 39,  7, 47, 15, 55, 23, 63, 31,
    38,  6, 46, 14, 54, 22, 62, 30, 37,  5, 45, 13, 53, 21, 61, 29,
    36,  4, 44, 12, 52, 20, 60, 28, 35,  3, 43, 11, 51, 19, 59, 27,
    34,  2, 42, 10, 50, 18, 58, 26, 33,  1, 41,  9, 49, 17, 57, 25
    };
  
  long[] _ufc_keytab = new long[16];

  private byte ascii_to_bin(byte c) {
    return (byte)(c >= 'a' ? (c - 59) : c >='A'? (c - 53) : c -'.');
  }

  private byte bin_to_ascii(byte c) {
    return (byte)(c >= 38 ? (c - 38 + 'a') : 
		  (c >= 12 ? (c - 12 + 'A'): c + '.'));
  }
  
  private long BITMASK(int i) {
    return ((1L << (11L- (long)i % 12L + 3L)) << ( (long)i < 12L ? 16L : 0L));
  }
  
  long[] _ufc_sb0 = new long[4096], _ufc_sb1 = new long[4096];
  long[] _ufc_sb2 = new long[4096], _ufc_sb3 = new long[4096];
  
  private long[] sb(int n) {
    long[] ret = null;
    switch (n) {
    case 0:
      ret = _ufc_sb0; break;
    case 1:
      ret = _ufc_sb1; break;
    case 2:
      ret = _ufc_sb2; break;
    case 3:
      ret = _ufc_sb3; break;
    }
    return ret;
  }    
  
  static long[][][] eperm32tab = new long[4][256][2];
  static long[] do_pc1 = new long[2048];
  static long[] do_pc2 = new long[1024];
  static long[][][] efp = new long[16][64][2];
  
  static char[] bytemask  = {
    (char)0x80, (char)0x40, (char)0x20, (char)0x10,
    (char)0x08, (char)0x04, (char)0x02, (char)0x01
    };

  static long[] longmask = {
    0x80000000, 0x40000000, 0x20000000, 0x10000000,
    0x08000000, 0x04000000, 0x02000000, 0x01000000,
    0x00800000, 0x00400000, 0x00200000, 0x00100000,
    0x00080000, 0x00040000, 0x00020000, 0x00010000,
    0x00008000, 0x00004000, 0x00002000, 0x00001000,
    0x00000800, 0x00000400, 0x00000200, 0x00000100,
    0x00000080, 0x00000040, 0x00000020, 0x00000010,
    0x00000008, 0x00000004, 0x00000002, 0x00000001
    };
  
  private void pr_bits(long[] a, int n){
    long i, j, t, tmp;
    n /= 8;
    for(i = 0; i < n; i++) {
      tmp=0;
      for(j = 0; j < 8; j++) {
	t=8*i+j;
	tmp|=(a[(int)(t/24)] & BITMASK((int)(t % 24))) != 0 ? bytemask[(int)j]:0;
      }
      System.out.println(tmp);
    }
    System.out.println(" ");
  }

  private void set_bits(long v, long[] b)
  { 
    long i;
    b[0] = 0;
    for(i = 0; i < 24; i++) {
      if((v & longmask[(int)(8 + i)]) != 0)
	b[0] |= BITMASK((int)i);
    }
  }

  static int initialized = 0;
  
  private int s_lookup(int i, int s) {
    return sbox[(i)][(((s)>>4) & 0x2)|((s) & 0x1)][((s)>>1) & 0xf];
  }
  
  public void init_des(){ 
    int comes_from_bit;
    int bit, sg, ii, jj, kk;
    long j;
    long mask1, mask2;

    for(bit = 0; bit < 56; bit++) {
      comes_from_bit  = pc1[bit] - 1;
      mask1 = bytemask[comes_from_bit % 8 + 1];
      mask2 = longmask[bit % 28 + 4];
      for(j = 0; j < 128; j++) {
	if((j & mask1) != 0) 
	  do_pc1[(comes_from_bit / 8) * 256 + (bit / 28) * 128 + (int)j] |=
	    mask2;
      }
    }

    for(bit = 0; bit < 48; bit++) {
      comes_from_bit  = pc2[bit] - 1;
      mask1 = bytemask[comes_from_bit % 7 + 1];
      mask2 = BITMASK(bit % 24);
      for(j = 0; j < 128; j++) {
	if((j & mask1) != 0)
	  do_pc2[(comes_from_bit / 7) * 128 +(int)j] |= mask2;
      }
    }

    for (ii = 0; ii < 4; ii++)
      for (jj = 0; jj < 256; jj++)
	for (kk = 0; kk < 2; kk++)
	  eperm32tab[ii][jj][kk] = 0;

    for(bit = 0; bit < 48; bit++) {
      long comes_from;
      
      comes_from = perm32[esel[bit]-1]-1;
      mask1      = bytemask[(int)(comes_from % 8)];
	
      for(j = 255; j >= 0; j--) {
	if((j & mask1) != 0)
	  eperm32tab[(int)(comes_from / 8)][(int)j][bit / 24] |= BITMASK(bit % 24);
      }
    }
    
    for(sg = 0; sg < 4; sg++) {
      int j1, j2;
      int s1, s2;
      long[] sbb;
      
      for(j1 = 0; j1 < 64; j1++) {
	s1 = s_lookup(2 * sg, j1);
	for(j2 = 0; j2 < 64; j2++) {
	  long to_permute, inx;
	  
	  s2         = s_lookup(2 * sg + 1, j2);
	  to_permute = (((long)s1 << 4)  | 
			(long)s2) << (24 - 8 * (long)sg);
	  
	  inx = ((j1 << 6)  | j2);
	  sbb = sb(sg);
	  sbb[(int)inx]  = 
	    ((long)eperm32tab[0][(int)((to_permute >> 24) & 0xff)][0] << 32) |
	      (long)eperm32tab[0][(int)((to_permute >> 24) & 0xff)][1];
	  sbb[(int)inx] |=
	    ((long)eperm32tab[1][(int)((to_permute >> 16) & 0xff)][0] << 32) |
	      (long)eperm32tab[1][(int)((to_permute >> 16) & 0xff)][1];
  	  sbb[(int)inx] |= 
	    ((long)eperm32tab[2][(int)((to_permute >>  8) & 0xff)][0] << 32) |
	      (long)eperm32tab[2][(int)((to_permute >>  8) & 0xff)][1];
	  sbb[(int)inx] |=
	    ((long)eperm32tab[3][(int)((to_permute)       & 0xff)][0] << 32) |
	      (long)eperm32tab[3][(int)((to_permute)       & 0xff)][1];
	}
      }
    }  

    for(bit = 47; bit >= 0; bit--) {
      e_inverse[esel[bit] - 1     ] = bit;
      e_inverse[esel[bit] - 1 + 32] = bit + 48;
    }
    
    for (ii = 0; ii < 16; ii++)
      for (jj = 0; jj < 64; jj++)
	for (kk = 0; kk < 2; kk++)
	  efp[ii][jj][kk] = 0;

    for(bit = 0; bit < 64; bit++) {
      int o_bit, o_long;
      long word_value;
      int comes_from_f_bit, comes_from_e_bit;
      int comes_from_word, bit_within_word;
      o_long = bit / 32; /* 0..1  */
      o_bit  = bit % 32; /* 0..31 */
      comes_from_f_bit = final_perm[bit] - 1;         /* 0..63 */
      comes_from_e_bit = e_inverse[comes_from_f_bit]; /* 0..95 */
      comes_from_word  = comes_from_e_bit / 6;        /* 0..15 */
      bit_within_word  = comes_from_e_bit % 6;        /* 0..5  */

      mask1 = longmask[bit_within_word + 26];
      mask2 = longmask[o_bit];
      
      for(word_value = 63; word_value >= 0; word_value--) {
	if((word_value & mask1) != 0)
	  efp[comes_from_word][(int)word_value][o_long] |= mask2;
      }
    }
    initialized++;
  }

  static void shuffle_sb(long[] k, long saltbits) { 
    long j;
    long x;
    int i = 0;
    for(j = 4095; j >= 0; j--) {
      x = ((k[i] >> 32) ^ k[i]) & (long)saltbits;
      k[i++] ^= (x << 32) | x;
    }
  }

  static byte[] current_salt = {(byte)'&', (byte)'&', (byte)0}; 
  static long current_saltbits = 0;
  static int direction = 0;

  public void setup_salt(byte[] s) {
    long i, j, saltbits;
    
    if(initialized == 0)
      init_des();
    
    if(s[0] == current_salt[0] && s[1] == current_salt[1])
      return;
    current_salt[0] = s[0]; current_salt[1] = s[1];
    saltbits = 0;
    for(i = 0; i < 2; i++) {
      long c = (long)ascii_to_bin(s[(int)i]);
      if(c < 0 || c > 63)
	c = 0;
      for(j = 0; j < 6; j++) {
	if(((c >> j) & 0x1) != 0)
	  saltbits |= BITMASK((int)(6 * i + j));
      }
    }
    shuffle_sb(_ufc_sb0, current_saltbits ^ saltbits); 
    shuffle_sb(_ufc_sb1, current_saltbits ^ saltbits);
    shuffle_sb(_ufc_sb2, current_saltbits ^ saltbits);
    shuffle_sb(_ufc_sb3, current_saltbits ^ saltbits);

    current_saltbits = saltbits;
  }

  public void ufc_mk_keytab(byte[] key) { 
    long v1, v2;
    int i, ky = 0, k2 = 0, k1 = 0;
    long v;

    v1 = v2 = 0;
    for(i = 7; i >= 0; i--) {
      v1 |= do_pc1[(int)((key[ky] & 0x7f) + k1)];   k1 += 128;
      v2 |= do_pc1[(int)((key[ky++] & 0x7f) + k1)]; k1 += 128;
    }
    for(i = 0; i < 16; i++) {
      k1 = 0;
      
      v1 = (v1 << rots[i]) | (v1 >> (28 - rots[i]));
      v  = do_pc2[(int)(((v1 >> 21) & 0x7f) + k1)]; k1 += 128;
      v |= do_pc2[(int)(((v1 >> 14) & 0x7f) + k1)]; k1 += 128;
      v |= do_pc2[(int)(((v1 >>  7) & 0x7f) + k1)]; k1 += 128;
      v |= do_pc2[(int)(((v1      ) & 0x7f) + k1)]; k1 += 128;
      v <<= 32;
      v2 = (v2 << rots[i]) | (v2 >> (28 - rots[i]));
      v |= do_pc2[(int)(((v2 >> 21) & 0x7f) + k1)]; k1 += 128;
      v |= do_pc2[(int)(((v2 >> 14) & 0x7f) + k1)]; k1 += 128;
      v |= do_pc2[(int)(((v2 >>  7) & 0x7f) + k1)]; k1 += 128;
      v |= do_pc2[(int)(((v2      ) & 0x7f) + k1)];

      _ufc_keytab[k2++] = v;
    }

    direction = 0;
  }
  
  long[] _ufc_dofinalperm(long l1, long l2, long r1, long r2){
    long v1, v2, x;
    long[] ary = new long[2];
    
    x = (l1 ^ l2) & current_saltbits; l1 ^= x; l2 ^= x;
    x = (r1 ^ r2) & current_saltbits; r1 ^= x; r2 ^= x;

    v1=v2=0; l1 >>= 3; l2 >>= 3; r1 >>= 3; r2 >>= 3;

    v1 |= efp[15][(int)( r2         & 0x3f)][0]; 
    v2 |= efp[15][(int)( r2 & 0x3f)][1];
    v1 |= efp[14][(int)((r2 >>= 6)  & 0x3f)][0]; 
    v2 |= efp[14][(int)( r2 & 0x3f)][1];
    v1 |= efp[13][(int)((r2 >>= 10) & 0x3f)][0]; 
    v2 |= efp[13][(int)( r2 & 0x3f)][1];
    v1 |= efp[12][(int)((r2 >>= 6)  & 0x3f)][0]; 
    v2 |= efp[12][(int)( r2 & 0x3f)][1];

    v1 |= efp[11][(int)( r1         & 0x3f)][0]; 
    v2 |= efp[11][(int)( r1 & 0x3f)][1];
    v1 |= efp[10][(int)((r1 >>= 6)  & 0x3f)][0]; 
    v2 |= efp[10][(int)( r1 & 0x3f)][1];
    v1 |= efp[ 9][(int)((r1 >>= 10) & 0x3f)][0]; 
    v2 |= efp[ 9][(int)( r1 & 0x3f)][1];
    v1 |= efp[ 8][(int)((r1 >>= 6)  & 0x3f)][0]; 
    v2 |= efp[ 8][(int)( r1 & 0x3f)][1];

    v1 |= efp[ 7][(int)( l2         & 0x3f)][0]; 
    v2 |= efp[ 7][(int)( l2 & 0x3f)][1];
    v1 |= efp[ 6][(int)((l2 >>= 6)  & 0x3f)][0]; 
    v2 |= efp[ 6][(int)( l2 & 0x3f)][1];
    v1 |= efp[ 5][(int)((l2 >>= 10) & 0x3f)][0]; 
    v2 |= efp[ 5][(int)( l2 & 0x3f)][1];
    v1 |= efp[ 4][(int)((l2 >>= 6)  & 0x3f)][0]; 
    v2 |= efp[ 4][(int)( l2 & 0x3f)][1];

    v1 |= efp[ 3][(int)( l1         & 0x3f)][0]; 
    v2 |= efp[ 3][(int)( l1 & 0x3f)][1];
    v1 |= efp[ 2][(int)((l1 >>= 6)  & 0x3f)][0]; 
    v2 |= efp[ 2][(int)( l1 & 0x3f)][1];
    v1 |= efp[ 1][(int)((l1 >>= 10) & 0x3f)][0]; 
    v2 |= efp[ 1][(int)( l1 & 0x3f)][1];
    v1 |= efp[ 0][(int)((l1 >>= 6)  & 0x3f)][0]; 
    v2 |= efp[ 0][(int)( l1 & 0x3f)][1];

    ary[0] = v1; ary[1] = v2;
    return ary;
  }

  public byte[] output_conversion(long v1, long v2, byte[] salt) { 
    byte[] outbuf = new byte[14];
    int i, s, shf;
      
    outbuf[0] = salt[0];
    outbuf[1] = salt[1] != 0 ? salt[1] : salt[0];

    for(i = 0; i < 5; i++) {
      shf = (26 - 6 * i); /* to cope with MSC compiler bug */
      outbuf[i + 2] = bin_to_ascii((byte)((v1 >> shf) & 0x3f));
    }
    s  = (int)((v2 & 0xf) << 2);
    v2 = ((long)(((int)v2 >>> 2)) | ((v1 & 0x3) << 30));

    for(i = 5; i < 10; i++) {
      shf = (56 - 6 * i);
      outbuf[i + 2] = bin_to_ascii((byte)((v2 >> shf) & 0x3f));
    }
    
    outbuf[12] = bin_to_ascii((byte)s);
    outbuf[13] = 0;
    
    return outbuf;
  }
  
  public String crypt(String key, String salt){
    long[] s;
    byte[] ktab = new byte[9], ktab_buf = new byte[9],salt_bytes = new byte[50];
    int i;

    salt_bytes = salt.getBytes();
    setup_salt(salt_bytes);
    for (i = 0; i < 9; i++)
      ktab[i] = 0;

    if (key.length() >= 8)
      ktab = key.getBytes();
    else {
      //key.getBytes(0, key.length(), ktab, 0);   <- before
      ktab_buf = key.getBytes();
      System.arraycopy(ktab_buf, 0, ktab, 0, ktab_buf.length);
      ktab_buf = null;
    }

    ufc_mk_keytab(ktab); 
    s = _ufc_doit((long)0, (long)0, 
		  (long)0, (long)0, (long)25);
    s = _ufc_dofinalperm(s[0], s[1], s[2], s[3]);
    return ConvertString(output_conversion(s[0], s[1], salt_bytes));
  }
  
  private String ConvertString(byte[] b) {
    int i;
    StringBuffer sb = new StringBuffer("");

    for (i = 0; i < b.length; i++)
      if (b[i] == 0) 
	break;
      else
	sb.append((char)b[i]);
    return new String(sb);
  }

  public static void main(String arg[]){
    Crypt c = new Crypt();
    System.out.println(c);
  }

}










