// RX62NとGCCで、printfやscanfを使うサンプル
// 特殊電子回路㈱

// このプログラムをコンパイルして、リンクするには、
// ../common/lowlevel.oをリンクする必要があります。
// ../common/lowlevel.oをリンクしないと、printfなどの出力が別のところに出力されてしまい、
// シリアルポートに出てきません

#include <tkdn_hal.h>
#include <stdio.h>
#include <math.h>

int main()
{
//	int i;
//	double pi = 3.1415926535;
	
	// SCIを初期化して、\n→\r\nの自動変換を有効にする
    sci_init(SCI_AUTO,38400); 
    sci_convert_crlf(CRLF_CRLF,CRLF_CRLF); // \nを\r\nに変換 

	printf("--------------------------------------------------------\n");
	printf("RX62N BASE emvironment \n");
	printf("Compiled at %s %s\n",__DATE__,__TIME__);
	printf("--------------------------------------------------------\n");
	printf("★gccからprintf、scanf、mallocなどが使えるようになった\n");

	while(1)
	{
		double A,B;
		char dummy[128];
		printf("\n浮動小数点数 Aを入力してください ");
		if(scanf("%lf",&A) == 0)
		{
			scanf("%s",dummy);
			printf(dummy);
			continue;
		}
		printf("%f",A);
		printf("\n浮動小数点数 Bを入力してください ");
		if(scanf("%lf",&B) == 0)
		{
			scanf("%s",dummy);
			printf(dummy);
			continue;
		}
		printf("%f",B);
		printf("\n");

		printf("A     = %f\n",A);
		printf("B     = %f\n",B);
		printf("A + B = %f\n",A + B);
		printf("A - B = %f\n",A - B);
		printf("A * B = %f\n",A * B);
		printf("A / B = %f\n",A / B);
		printf("sin(A) = %f\n",sin(A));
		printf("cos(B) = %f\n",cos(B));
		printf("A^B = %f\n",pow(A,B));
	}

	setvbuf(stdout, 0, _IONBF, 0);
}

