// Simple demo of three threads
// LED blink thread, print thread, and main thread
#include <ChibiOS_AVR.h>

const uint8_t LED_PIN = 13;

volatile uint32_t count = 0;

// remember thread pointers
Thread* tp1;
Thread* tp2;
//------------------------------------------------------------------------------
// thread 1 - high priority for blinking LED
// 64 byte stack beyond task switch and interrupt needs
static WORKING_AREA(waThread1, 64);

static msg_t Thread1(void *arg) {
  pinMode(LED_PIN, OUTPUT);
  while (!chThdShouldTerminate()) {
    digitalWrite(LED_PIN, HIGH);
    chThdSleepMilliseconds(50);
    digitalWrite(LED_PIN, LOW);
    chThdSleepMilliseconds(150);
  }
  return 0;
}
//------------------------------------------------------------------------------
// thread 2 - print main thread count every second
// 200 byte stack beyond task switch and interrupt needs
static WORKING_AREA(waThread2, 200);

static msg_t Thread2(void *arg) {

  Serial.println("Type any character for stack use");

  // print count every second
  while (!Serial.available()) {
    Serial.println(count);
    count = 0;
    chThdSleepMilliseconds(1000);
  }
  // Terminate the LED thread
  chThdTerminate(tp1);

  // print memory use
  Serial.println();
  Serial.println("Memory use");
  Serial.println("Area,Size,Unused");
  Serial.print("Thread 1,");
  
  // size of stack for thread 1
  Serial.print(sizeof(waThread1) - sizeof(Thread));
  Serial.write(',');
  
  // unused stack for thread 1
  Serial.println(chUnusedStack(waThread1, sizeof(waThread1)));
  
  Serial.print("Thread 2,");
  
  // size of stack for thread 2
  Serial.print(sizeof(waThread2) - sizeof(Thread));
  Serial.write(',');

  // unused stack for thread 2
  Serial.println(chUnusedStack(waThread2, sizeof(waThread2)));

  // print stats for heap/main thread area
  Serial.print("Heap/Main,");
  Serial.print(chHeapMainSize());
  Serial.print(",");
  Serial.println(chUnusedHeapMain());
  
  // end task
  return 0;
}
//------------------------------------------------------------------------------
void setup() {
  Serial.begin(9600);
  // wait for USB Serial
  while (!Serial) {}
  
  // read any input
  delay(200);
  while (Serial.read() >= 0) {}

  chBegin(mainThread);
  // chBegin never returns, main thread continues with mainThread()
  while(1) {}
}
//------------------------------------------------------------------------------
// main thread runs at NORMALPRIO
void mainThread() {

  // start blink thread
  tp1 = chThdCreateStatic(waThread1, sizeof(waThread1),
                          NORMALPRIO + 2, Thread1, NULL);

  // start print thread
  tp2 = chThdCreateStatic(waThread2, sizeof(waThread2),
                          NORMALPRIO + 1, Thread2, NULL);

  // increment counter
  while (1) {
    // must insure increment is atomic in case of context switch for print
    // should use mutex for longer critical sections
    noInterrupts();
    count++;
    interrupts();
  }
}
//------------------------------------------------------------------------------
void loop() {
 // not used
}