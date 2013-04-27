/**
 * \file
 * \brief FreeRTOS for AVR Arduino
 */
#include <FreeRTOS_AVR.h>
/** current begining of heap */
extern char *__brkval;
/** \return free heap size */
size_t freeHeap() {
#if defined(CORE_TEENSY) || (ARDUINO > 103 && ARDUINO != 151)
  return (char*)RAMEND - __brkval +1;
#else  // CORE_TEENSY
/** initial begining of heap */
 extern char *__malloc_heap_start;
  return (char*)RAMEND - (__brkval ? __brkval : __malloc_heap_start) + 1;
#endif  // CORE_TEENSY
}
