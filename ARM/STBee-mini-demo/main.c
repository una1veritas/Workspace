/*
 *	STBee Mini LED�_���T���v��
 *	2010/8 Strawberry Linux Co.,Ltd.
 *
 *	���荞�݂������g��Ȃ��ł��V���v����LED�_���T���v���ł��B
 *
 *
 */


// I/O�Ȃǂ̑S�Ă�#define������܂��B
#include "stm32f10x_conf.h"



// �󃋁[�v�ŃE�F�C�g���郋�[�`��
void Delay(volatile unsigned long delay)
{ 
	while(delay) delay--;
}


/*************************************************************************
 * Function Name: main
 * Parameters: none
 * Return: Int32U
 *
 * Description: The main subroutine
 *
 *************************************************************************/
int main(void)
{
	GPIO_InitTypeDef GPIO_InitStructure;
	int i;
	
	// STM32�̏����� �N���b�N�ݒ�
	SystemInit();
	NVIC_SetVectorTable(0x3000, 0);
	
	// JTAG�𖳌��ɂ��܂��B
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_AFIO , ENABLE);
	AFIO->MAPR = _BV(26);
	
	// GPIO A, B, C�|�[�g��L���ɂ��܂�
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA, ENABLE);
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB, ENABLE);
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOC, ENABLE);
	
	
	// �|�[�g�̏�����(PA13, PA15���o�͂�)
	GPIO_InitStructure.GPIO_Pin = _BV(13) | _BV(15);
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOA, &GPIO_InitStructure);

	// �I���{�[�hLED�����݂ɓ_�ł�����
	GPIOA->ODR = 0;
	uint32_t cnt = 0;
	uint32_t tmp;
	while(1){
	  tmp = cnt;
	  cnt++;
	  if ( (tmp^cnt) & 2 ) {
	    GPIOA->ODR ^= _BV(13);
	  }
	  if ( (tmp^cnt) & 8 ) {
	    GPIOA->ODR ^= _BV(15);
	  }
	  Delay(500000);
	}

}
