// Illustration of Rate Monotonic Scheduling from Liu and Layland paper
//
// Rate Monotonic Scheduling for a set of repeating tasks gives higher
// priority to a task with a smaller period.
//
// Theorem Liu and Layland 1973. Given a preemptive, fixed priority scheduler
// and a finite set of repeating tasks T = {T1; T2; ...; Tn} with associated
// periods {p1; p2 ...; pn} and no precedence constraints, if any priority
// assignment yields a feasible schedule, then the rate monotonic
// priority assignment yields a feasible schedule.
//
// Liu and Layland also derived a bound on CPU utilization that guarantees
// there will be a feasible Rate Monotonic Schedule when a set of n tasks
// have CPU utilization less than the bound.
//
// The Liu Layland bound = 100*n*(2^(1/n) - 1) in percent.  For large n
// this approaches ln(2) or 69.3%.  The extra CPU time can be used by
// lower priority tasks that do not have hard deadlines.
//
// Note that it may be possible to run a given set of tasks with higher CPU
// utilization, depending on task parameters.  The Liu Layland bound works
// for every set of tasks independent of task parameters.
//
#include <ChibiOS_ARM.h>
//------------------------------------------------------------------------------
struct task_t {
  uint16_t period;
  uint16_t cpu;
  uint16_t priority;
};
task_t tasks1[] = {{10, 5, 2}, {15, 6, 1}};
task_t tasks2[] = {{10, 5, 2}, {15, 4, 1}};
task_t tasks3[] = {{10, 3, 3}, {13, 4, 2}, {17, 4, 1}};
task_t tasks4[] = {{10, 3, 3}, {13, 4, 2}, {17, 2, 1}};
task_t* taskList[] = {tasks1, tasks2, tasks3, tasks4};
int taskCount[] = {2, 2, 3, 3};

int nTask;  // number of tasks to run
task_t* tasks;  // list of tasks to run
//------------------------------------------------------------------------------
#ifdef __AVR__
const size_t STK_SIZE = 100;
#else  // __AVR__
const size_t STK_SIZE = 200;
#endif  // __AVR__
WORKING_AREA(waThd2, STK_SIZE);
WORKING_AREA(waThd3, STK_SIZE);
WORKING_AREA(waThd4, STK_SIZE);
// first task will be main thread - allow up to four tasks
stkalign_t* waMem[] = {0, waThd2, waThd3, waThd4};
size_t waSize[] = {0, sizeof(waThd2), sizeof(waThd3), sizeof(waThd4)};
//------------------------------------------------------------------------------
// override IDE definition to prevent errors
void printTask(task_t* task);
void done(const char* msg, task_t* task, systime_t now);
//------------------------------------------------------------------------------
// Liu Layland bound = 100*n*(2^(1/n) - 1) in percent
float LiuLayland[] = {100, 82.84271247, 77.97631497, 75.682846, 74.3491775};
//------------------------------------------------------------------------------
// dummy CPU utilization functions
#ifdef __AVR__
const unsigned int CAL_GUESS = 3000;
const float TICK_USEC = 1024;
#else  // __AVR__
const unsigned int CAL_GUESS = 17000;
const float TICK_USEC = 1000;
#endif  // __AVR__

static unsigned int cal = CAL_GUESS;

void burnCPU(uint16_t ticks) {
  while (ticks--) {
    for (unsigned int i = 0; i < cal; i++) {
      asm("nop");
    }
  }
}
void calibrate() {
  uint32_t t = micros();
  burnCPU(1000);
  t = micros() - t;
  cal = (TICK_USEC*1000*cal)/t;
}
//------------------------------------------------------------------------------
// print helpers
void printTask(task_t* task) {
    Serial.print(task->period);
    Serial.write(',');
    Serial.print(task->cpu);
    Serial.write(',');
    Serial.println(task->priority);
}
// gate keeper
BSEMAPHORE_DECL(doneSem, 0);

void done(const char* msg, task_t* task, systime_t now) {
  // only allow first task to print status
  chBSemWait(&doneSem);

  Serial.println(msg);
  Serial.print("Tick: ");
  Serial.println(now);
  Serial.print("Task: ");
  printTask(task);
  Serial.print("unusedStack: ");
  Serial.print(chUnusedHeapMain());
  for ( int i = 1; i < nTask; i++) {
    Serial.write(' ');
    Serial.print(chUnusedStack(waMem[i], waSize[i]));
  }
  Serial.println();
  while(1);
}
//------------------------------------------------------------------------------
// start tasks at 1000 ticks
const systime_t startTime = 1000;

// test runs for 3000 ticks
const systime_t finishTime = 4000;

// task code
msg_t taskFcn(void* arg) {
  systime_t wakeTime = startTime;
  uint16_t period = ((task_t*)arg)->period;
  uint16_t cpu = ((task_t*)arg)->cpu;

  while (1) {
    chThdSleepUntil(wakeTime);

    // use cpu ticks of execution time
    burnCPU(cpu);

    systime_t now = chTimeNow();

    // check for success
    if (now >= finishTime) {
      done("Success", (task_t*)arg, now);
    }

    // next wake time
    wakeTime += period;

    // check for failure
    if (now >= wakeTime) {
      done("Missed Deadline", (task_t*)arg, now);
    }
  }
}
//------------------------------------------------------------------------------


void setup() {

  int c;  // Serial input

  Serial.begin(9600);
  // wait for USB Serial
  while (!Serial) {}

  Serial.println("Rate Monotonic Scheduling Examples.");
  Serial.println("Cases 1 and 3 should fail");
  Serial.println("Cases 2 and 4 should succeed");
  Serial.println();

  // get input
  while (1) {
    while (Serial.read() >= 0) {}
    Serial.print("Enter number [1-4] ");
    while ((c = Serial.read()) < 0) {}
    Serial.println((char)c);
    if (c < '1' || c > '4') {
      Serial.println("Invalid input");
      continue;
    }
    c -= '1';
    tasks = taskList[c];
    nTask = taskCount[c];
    break;
  }
  Serial.print("calibrating CPU: ");

  // insure no interrupts from Serial
  Serial.flush();
  delay(100);
  calibrate();

  // check calibration accuracy
  uint32_t t = micros();
  burnCPU(1000);
  Serial.println(micros() -t);

  // start ChibiOS
  chBegin(mainThread);

  // shouldn't return
  while(1) {}
}
//------------------------------------------------------------------------------
void mainThread() {
  float cpuUse = 0;  // total cpu utilization for set of tasks

  Serial.println("Starting tasks - period and CPU in ticks");
  Serial.println("Period,CPU,Priority");

  // start tasks
  for (int i = 0; i < nTask; i++) {
    printTask(&tasks[i]);
    cpuUse += tasks[i].cpu/(float)tasks[i].period;

    // mainThread will become first task
    if (i == 0) continue;

    chThdCreateStatic(waMem[i], waSize[i], NORMALPRIO + tasks[i].priority,
                      taskFcn, (void*)&tasks[i]);
  }

  Serial.print("CPU use %: ");
  Serial.println(cpuUse*100);
  Serial.print("Liu and Layland bound %: ");
  Serial.println(LiuLayland[nTask - 1]);

  // main thread becomes first task
  chThdSetPriority(NORMALPRIO + tasks[0].priority);
  taskFcn(&tasks[0]);
}
//------------------------------------------------------------------------------
void loop() {
  // not used
}