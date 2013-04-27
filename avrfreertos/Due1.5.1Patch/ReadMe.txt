Please use Arduino 1.5.2 or newer since it doesn't require a patch.

This folder has a patched version of wiring.c. 

You must replace wiring.c in this directory with the patched version.

arduino-1.5.1r2/hardware/arduino/sam/cores/arduino

The patched version replaces SysTick_Handler with the following:

//------------------------------------------------------------------------------
// Modified for RTOS libraries.
extern int sysTickHook(void);
/**
 * SysTick hook
 *
 * This function is called from SysTick handler, before the default
 * handler provided by Arduino.
 */
static int __false() {
	// Return false
	return 0;
}
int sysTickHook(void) __attribute__ ((weak, alias("__false")));
/*
 * Cortex-M3 Systick IT handler
 */
void SysTick_Handler( void )
{
	if (sysTickHook()) return;  // Mod for RTOS
	
	tickReset();

	// Increment tick count each ms
	TimeTick_Increment() ;
}
//------------------------------------------------------------------------------

Future versions for Arduino Due should have sysTickHook() already installed.