// RX62N��GCC�ŁAprintf��scanf���g���T���v��
// ����d�q��H��

// ���̃v���O�������R���p�C�����āA�����N����ɂ́A
// ../common/lowlevel.o�������N����K�v������܂��B
// ../common/lowlevel.o�������N���Ȃ��ƁAprintf�Ȃǂ̏o�͂��ʂ̂Ƃ���ɏo�͂���Ă��܂��A
// �V���A���|�[�g�ɏo�Ă��܂���

#include <tkdn_hal.h>
#include <stdio.h>
#include <math.h>

int main()
{
//	int i;
//	double pi = 3.1415926535;
	
	// SCI�����������āA\n��\r\n�̎����ϊ���L���ɂ���
    sci_init(SCI_AUTO,38400); 
    sci_convert_crlf(CRLF_CRLF,CRLF_CRLF); // \n��\r\n�ɕϊ� 

	printf("--------------------------------------------------------\n");
	printf("RX62N BASE emvironment \n");
	printf("Compiled at %s %s\n",__DATE__,__TIME__);
	printf("--------------------------------------------------------\n");
	printf("��gcc����printf�Ascanf�Amalloc�Ȃǂ��g����悤�ɂȂ���\n");

	while(1)
	{
		double A,B;
		char dummy[128];
		printf("\n���������_�� A����͂��Ă������� ");
		if(scanf("%lf",&A) == 0)
		{
			scanf("%s",dummy);
			printf(dummy);
			continue;
		}
		printf("%f",A);
		printf("\n���������_�� B����͂��Ă������� ");
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

