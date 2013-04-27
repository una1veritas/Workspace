
#include "cmsis_os.h"


/* Convert from CMSIS type osPriority to FreeRTOS priority number */
static unsigned portBASE_TYPE makeFreeRtosPriority (osPriority priority)
{
    unsigned portBASE_TYPE fpriority = tskIDLE_PRIORITY;

    if (priority != osPriorityError) {
        fpriority += (priority - osPriorityIdle);
    }

    return fpriority;
}


/* Convert from FreeRTOS priority number to CMSIS type osPriority */
#if INCLUDE_uxTaskPriorityGet
static osPriority makeCmsisPriority (unsigned portBASE_TYPE fpriority)
{
    osPriority priority = osPriorityError;

    if ((fpriority - tskIDLE_PRIORITY) <= (osPriorityRealtime - osPriorityIdle)) {
        priority = (osPriority)((int)osPriorityIdle + (int)(fpriority - tskIDLE_PRIORITY));
    }

    return priority;
}
#endif


/* Determine whether we are in thread mode or handler mode. */
#if defined (__CC_ARM) && defined (__TARGET_ARCH_4T)    /* ARM Compiler for ARM7TDMI */
static __asm int inHandlerMode (void)
{
    mrs     r0, cpsr
    ands    r0, #0x0F       /* Test for user mode */
    sub     r0, #0x0F       /* Test for system mode */

    bx      lr
}
#elif defined (__GNUC__) && defined (__ARM_ARCH_4T__)   /* GNU compiler for ARM7TDMI */
static inline int inHandlerMode (void) __attribute__((always_inline));
static int inHandlerMode (void)
{
    int result;

    __ASM volatile ("mrs %0, cpsr \r\n"
                 "\t ands %0, #0x0F \r\n"       /* Test for user mode */
                 "\t sub %0, #0x0F \r\n"        /* Test for system mode */
                 : "=r" (result) );
    return result;
}
#else
static int inHandlerMode (void)
{
    return __get_IPSR() != 0;
}
#endif



//  ==== Kernel Control Functions ====

/// Start the RTOS Kernel with executing the specified thread.
/// \param[in]     thread_def    thread definition referenced with \ref osThread.
/// \param[in]     argument      pointer that is passed to the thread function as start argument.
/// \return status code that indicates the execution status of the function.
/// \note MUST REMAIN UNCHANGED: \b osKernelStart shall be consistent in every CMSIS-RTOS.
osStatus osKernelStart (osThreadDef_t *thread_def, void *argument)
{
    (void) argument;

    osThreadCreate(thread_def, argument);
    vTaskStartScheduler();

    return osOK;
}


/// Check if the RTOS kernel is already started.
/// \note MUST REMAIN UNCHANGED: \b osKernelRunning shall be consistent in every CMSIS-RTOS.
/// \return 0 RTOS is not started, 1 RTOS is started.
int32_t osKernelRunning(void);


//  ==== Thread Management ====


/// Create a thread and add it to Active Threads and set it to state READY.
/// \param[in]     thread_def    thread definition referenced with \ref osThread.
/// \param[in]     argument      pointer that is passed to the thread function as start argument.
/// \return thread ID for reference by other functions or NULL in case of error.
/// \note MUST REMAIN UNCHANGED: \b osThreadCreate shall be consistent in every CMSIS-RTOS.
osThreadId osThreadCreate (osThreadDef_t *thread_def, void *argument)
{
    xTaskHandle handle;
    uint32_t stackSize;
    (void) argument;


    stackSize = thread_def->stacksize ? thread_def->stacksize / 4 : configMINIMAL_STACK_SIZE;
    xTaskCreate((pdTASK_CODE)thread_def->pthread,
                (const signed portCHAR *)thread_def->name,
                stackSize,
                argument,
                makeFreeRtosPriority(thread_def->tpriority),
                &handle);

    return handle;
}


/// Return the thread ID of the current running thread.
/// \return thread ID for reference by other functions or NULL in case of error.
/// \note MUST REMAIN UNCHANGED: \b osThreadGetId shall be consistent in every CMSIS-RTOS.
osThreadId osThreadGetId (void)
{
    return xTaskGetCurrentTaskHandle();
}


/// Terminate execution of a thread and remove it from Active Threads.
/// \param[in]     thread_id   thread ID obtained by \ref osThreadCreate or \ref osThreadGetId.
/// \return status code that indicates the execution status of the function.
/// \note MUST REMAIN UNCHANGED: \b osThreadTerminate shall be consistent in every CMSIS-RTOS.
osStatus osThreadTerminate (osThreadId thread_id)
{
    vTaskDelete(thread_id);

    return osOK;
}


/// Pass control to next thread that is in state \b READY.
/// \return status code that indicates the execution status of the function.
/// \note MUST REMAIN UNCHANGED: \b osThreadYield shall be consistent in every CMSIS-RTOS.
osStatus osThreadYield (void)
{
    taskYIELD();

    return osOK;
}


/// Change priority of an active thread.
/// \param[in]     thread_id     thread ID obtained by \ref osThreadCreate or \ref osThreadGetId.
/// \param[in]     priority      new priority value for the thread function.
/// \return status code that indicates the execution status of the function.
/// \note MUST REMAIN UNCHANGED: \b osThreadSetPriority shall be consistent in every CMSIS-RTOS.
osStatus osThreadSetPriority (osThreadId thread_id, osPriority priority)
{
#if INCLUDE_vTaskPrioritySet
    vTaskPrioritySet(thread_id, makeFreeRtosPriority(priority));
#else
    (void) thread_id;
    (void) priority;
#endif

    return osOK;
}


/// Get current priority of an active thread.
/// \param[in]     thread_id     thread ID obtained by \ref osThreadCreate or \ref osThreadGetId.
/// \return current priority value of the thread function.
/// \note MUST REMAIN UNCHANGED: \b osThreadGetPriority shall be consistent in every CMSIS-RTOS.
osPriority osThreadGetPriority (osThreadId thread_id)
{
#if INCLUDE_uxTaskPriorityGet
    return makeCmsisPriority(uxTaskPriorityGet(thread_id));
#else
    (void) thread_id;

    return osPriorityNormal;
#endif
}



//  ==== Generic Wait Functions ====

/// Wait for Timeout (Time Delay)
/// \param[in]     millisec      time delay value
/// \return status code that indicates the execution status of the function.
osStatus osDelay (uint32_t millisec)
{
#if INCLUDE_vTaskDelay
    /* The resulting delay must be a *minimum* of "millisec" milliseconds.
     * - We are at an unknown position within a tick period. The next tick interrupt may therefore
     *   occur at any time, resulting in an uncertainty of one tick period.
     *   --> Offset +1
     */

    portTickType ticks;
    ticks = 1;                                          /* Compensate for unknown start position */
    ticks += (millisec * configTICK_RATE_HZ) / 1000;    /* Wanted delay (rounded down!) */
    if ((millisec * configTICK_RATE_HZ) % 1000) {
        ++ticks;                                        /* Round up to guarantee minimum delay */
    }

    vTaskDelay(ticks);

    return osOK;
#else
    (void) millisec;

    return osErrorResource;
#endif
}


#if (defined (osFeature_Wait)  &&  (osFeature_Wait != 0))     // Generic Wait available

/// Wait for Signal, Message, Mail, or Timeout
/// \param[in] millisec          timeout value or 0 in case of no time-out
/// \return event that contains signal, message, or mail information or error code.
/// \note MUST REMAIN UNCHANGED: \b osWait shall be consistent in every CMSIS-RTOS.
osEvent osWait (uint32_t millisec);

#endif  // Generic Wait available


//  ==== Timer Management Functions ====

static void _osTimerCallbackFreeRTOS (xTimerHandle handle)
{
    osTimerDef_t *timer = (osTimerDef_t *)(pvTimerGetTimerID(handle));

    timer->ptimer(timer->custom->argument);
}


/// Create a timer.
/// \param[in]     timer_def     timer object referenced with \ref osTimer.
/// \param[in]     type          osTimerOnce for one-shot or osTimerPeriodic for periodic behavior.
/// \param[in]     argument      argument to the timer call back function.
/// \return timer ID for reference by other functions or NULL in case of error.
/// \note MUST REMAIN UNCHANGED: \b osTimerCreate shall be consistent in every CMSIS-RTOS.
osTimerId osTimerCreate (osTimerDef_t *timer_def, os_timer_type type, void *argument)
{
    timer_def->custom->argument = argument;

    return xTimerCreate((const signed portCHAR *)"",
                        1,  //Set later when timer is started
                        (type == osTimerPeriodic) ? pdTRUE : pdFALSE,
                        (void *)timer_def,
                        _osTimerCallbackFreeRTOS
                        );
}



/// Start or restart a timer.
/// \param[in]     timer_id      timer ID obtained by \ref osTimerCreate.
/// \param[in]     millisec      time delay value of the timer.
/// \return status code that indicates the execution status of the function.
/// \note MUST REMAIN UNCHANGED: \b osTimerStart shall be consistent in every CMSIS-RTOS.
osStatus osTimerStart (osTimerId timer_id, uint32_t millisec)
{
    portBASE_TYPE taskWoken = pdFALSE;
    osStatus result = osOK;
    portTickType ticks = millisec / portTICK_RATE_MS;
    if (ticks == 0) {
        ticks = 1;
    }

    if (inHandlerMode()) {
        if (xTimerChangePeriodFromISR(timer_id, ticks, &taskWoken) == pdPASS) {
            xTimerStartFromISR(timer_id, &taskWoken);
            portEND_SWITCHING_ISR(taskWoken);
        }
    }
    else {
        //TODO: add timeout support
        if (xTimerChangePeriod(timer_id, ticks, 0) != pdPASS) {
            result = osErrorOS;
        }
        else {
            if (xTimerStart(timer_id, 0) != pdPASS) {
                result = osErrorOS;
            }
        }
    }

    return result;
}



/// Stop the timer.
/// \param[in]     timer_id      timer ID obtained by \ref osTimerCreate.
/// \return status code that indicates the execution status of the function.
/// \note MUST REMAIN UNCHANGED: \b osTimerStop shall be consistent in every CMSIS-RTOS.
osStatus osTimerStop (osTimerId timer_id)
{
    portBASE_TYPE taskWoken = pdFALSE;
    osStatus result = osOK;

    if (inHandlerMode()) {
        xTimerStopFromISR(timer_id, &taskWoken);
        portEND_SWITCHING_ISR(taskWoken);
    }
    else {
        if (xTimerStop(timer_id, 0) != pdPASS) {    //TODO: add timeout support
            result = osErrorOS;
        }
    }

    return result;
}



//  ==== Signal Management ====

/// Set the specified Signal Flags of an active thread.
/// \param[in]     thread_id     thread ID obtained by \ref osThreadCreate or \ref osThreadGetId.
/// \param[in]     signals       specifies the signal flags of the thread that should be set.
/// \return previous signal flags of the specified thread or 0x80000000 in case of incorrect parameters.
/// \note MUST REMAIN UNCHANGED: \b osSignalSet shall be consistent in every CMSIS-RTOS.
int32_t osSignalSet (osThreadId thread_id, int32_t signal);

/// Clear the specified Signal Flags of an active thread.
/// \param[in]     thread_id     thread ID obtained by \ref osThreadCreate or \ref osThreadGetId.
/// \param[in]     signals       specifies the signal flags of the thread that shall be cleared.
/// \return previous signal flags of the specified thread or 0x80000000 in case of incorrect parameters.
/// \note MUST REMAIN UNCHANGED: \b osSignalClear shall be consistent in every CMSIS-RTOS.
int32_t osSignalClear (osThreadId thread_id, int32_t signal);

/// Get Signal Flags status of an active thread.
/// \param[in]     thread_id     thread ID obtained by \ref osThreadCreate or \ref osThreadGetId.
/// \return previous signal flags of the specified thread or 0x80000000 in case of incorrect parameters.
/// \note MUST REMAIN UNCHANGED: \b osSignalGet shall be consistent in every CMSIS-RTOS.
int32_t osSignalGet (osThreadId thread_id);

/// Wait for one or more Signal Flags to become signaled for the current \b RUNNING thread.
/// \param[in]     signals       wait until all specified signal flags set or 0 for any single signal flag.
/// \param[in]     millisec      timeout value or 0 in case of no time-out.
/// \return event flag information or error code.
/// \note MUST REMAIN UNCHANGED: \b osSignalWait shall be consistent in every CMSIS-RTOS.
osEvent osSignalWait (int32_t signals, uint32_t millisec);


//  ==== Mutex Management ====


/// Create and Initialize a Mutex object
/// \param[in]     mutex_def     mutex definition referenced with \ref osMutex.
/// \return mutex ID for reference by other functions or NULL in case of error.
/// \note MUST REMAIN UNCHANGED: \b osMutexCreate shall be consistent in every CMSIS-RTOS.
osMutexId osMutexCreate (osMutexDef_t *mutex_def)
{
    (void) mutex_def;

    return xSemaphoreCreateMutex();
}



/// Wait until a Mutex becomes available
/// \param[in]     mutex_id      mutex ID obtained by \ref osMutexCreate.
/// \param[in]     millisec      timeout value or 0 in case of no time-out.
/// \return status code that indicates the execution status of the function.
/// \note MUST REMAIN UNCHANGED: \b osMutexWait shall be consistent in every CMSIS-RTOS.
osStatus osMutexWait (osMutexId mutex_id, uint32_t millisec)
{
    portTickType ticks;


    if (mutex_id == NULL) {
        return osErrorParameter;
    }

    ticks = 0;
    if (millisec == osWaitForever) {
        ticks = portMAX_DELAY;
    }
    else if (millisec != 0) {
        ticks = millisec / portTICK_RATE_MS;
        if (ticks == 0) {
            ticks = 1;
        }
    }

    if (inHandlerMode()) {
        return osErrorISR;
    }

    if (xSemaphoreTake(mutex_id, ticks) != pdTRUE) {
        return osErrorOS;
    }

    return osOK;
}



/// Release a Mutex that was obtained by \ref osMutexWait
/// \param[in]     mutex_id      mutex ID obtained by \ref osMutexCreate.
/// \return status code that indicates the execution status of the function.
/// \note MUST REMAIN UNCHANGED: \b osMutexRelease shall be consistent in every CMSIS-RTOS.
osStatus osMutexRelease (osMutexId mutex_id)
{
    osStatus result = osOK;
    portBASE_TYPE taskWoken = pdFALSE;


    if (inHandlerMode()) {
        if (xSemaphoreGiveFromISR(mutex_id, &taskWoken) != pdTRUE) {
            result = osErrorOS;
        }
        portEND_SWITCHING_ISR(taskWoken);
    }
    else {
        if (xSemaphoreGive(mutex_id) != pdTRUE) {
            result = osErrorOS;
        }
    }

    return result;
}


//  ==== Semaphore Management Functions ====

#if (defined (osFeature_Semaphore)  &&  (osFeature_Semaphore != 0))     // Semaphore available

/// Create and Initialize a Semaphore object used for managing resources
/// \param[in]     semaphore_def semaphore definition referenced with \ref osSemaphore.
/// \param[in]     count         number of available resources.
/// \return semaphore ID for reference by other functions or NULL in case of error.
/// \note MUST REMAIN UNCHANGED: \b osSemaphoreCreate shall be consistent in every CMSIS-RTOS.
osSemaphoreId osSemaphoreCreate (osSemaphoreDef_t *semaphore_def, int32_t count)
{
    osSemaphoreId sema;
    (void) semaphore_def;

    if (count == 1) {
        vSemaphoreCreateBinary(sema);
        return sema;
    }

    return xSemaphoreCreateCounting(count, count);
}



/// Wait until a Semaphore token becomes available
/// \param[in]     semaphore_id  semaphore object referenced with \ref osSemaphore.
/// \param[in]     millisec      timeout value or 0 in case of no time-out.
/// \return number of available tokens, or -1 in case of incorrect parameters.
/// \note MUST REMAIN UNCHANGED: \b osSemaphoreWait shall be consistent in every CMSIS-RTOS.
int32_t osSemaphoreWait (osSemaphoreId semaphore_id, uint32_t millisec)
{
    portTickType ticks;


    if (semaphore_id == NULL) {
        return -1;
    }

    ticks = 0;
    if (millisec == osWaitForever) {
        ticks = portMAX_DELAY;
    }
    else if (millisec != 0) {
        ticks = millisec / portTICK_RATE_MS;
        if (ticks == 0) {
            ticks = 1;
        }
    }

    if (inHandlerMode()) {
        return -1;
    }

    if (xSemaphoreTake(semaphore_id, ticks) != pdTRUE) {
        return 0;
    }

    return 1;
}


/// Release a Semaphore token
/// \param[in]     semaphore_id  semaphore object referenced with \ref osSemaphore.
/// \return status code that indicates the execution status of the function.
/// \note MUST REMAIN UNCHANGED: \b osSemaphoreRelease shall be consistent in every CMSIS-RTOS.
osStatus osSemaphoreRelease (osSemaphoreId semaphore_id)
{
    osStatus result = osOK;
    portBASE_TYPE taskWoken = pdFALSE;


    if (inHandlerMode()) {
        if (xSemaphoreGiveFromISR(semaphore_id, &taskWoken) != pdTRUE) {
            result = osErrorOS;
        }
        portEND_SWITCHING_ISR(taskWoken);
    }
    else {
        if (xSemaphoreGive(semaphore_id) != pdTRUE) {
            result = osErrorOS;
        }
    }

    return result;
}



#endif     // Semaphore available

//  ==== Memory Pool Management Functions ====

#if (defined (osFeature_Pool)  &&  (osFeature_Pool != 0))  // Memory Pool Management available

//TODO
//This is a primitive and inefficient wrapper around the existing FreeRTOS memory management.
//A better implementation will have to modify heap_x.c!


typedef struct os_pool_cb {
    void *pool;
    uint8_t *markers;
    uint32_t pool_sz;
    uint32_t item_sz;
    uint32_t currentIndex;
} os_pool_cb_t;


/// \brief Access a Memory Pool definition.
/// \param         name          name of the memory pool
/// \note CAN BE CHANGED: The parameter to \b osPool shall be consistent but the
///       macro body is implementation specific in every CMSIS-RTOS.
#define osPool(name) \
&os_pool_def_##name

/// Create and Initialize a memory pool
/// \param[in]     pool_def      memory pool definition referenced with \ref osPool.
/// \return memory pool ID for reference by other functions or NULL in case of error.
/// \note MUST REMAIN UNCHANGED: \b osPoolCreate shall be consistent in every CMSIS-RTOS.
osPoolId osPoolCreate (osPoolDef_t *pool_def)
{
    osPoolId thePool;
    int itemSize = 4 * ((pool_def->item_sz + 3) / 4);
    uint32_t i;

    /* First have to allocate memory for the pool control block. */
    thePool = pvPortMalloc(sizeof(os_pool_cb_t));
    if (thePool) {
        thePool->pool_sz = pool_def->pool_sz;
        thePool->item_sz = itemSize;
        thePool->currentIndex = 0;

        /* Memory for markers */
        thePool->markers = pvPortMalloc(pool_def->pool_sz);
        if (thePool->markers) {
            /* Now allocate the pool itself. */
            thePool->pool = pvPortMalloc(pool_def->pool_sz * itemSize);

            if (thePool->pool) {
                for (i = 0; i < pool_def->pool_sz; i++) {
                    thePool->markers[i] = 0;
                }
            }
            else {
                vPortFree(thePool->markers);
                vPortFree(thePool);
                thePool = NULL;
            }
        }
        else {
            vPortFree(thePool);
            thePool = NULL;
        }
    }

    return thePool;
}


/// Allocate a memory block from a memory pool
/// \param[in]     pool_id       memory pool ID obtain referenced with \ref osPoolCreate.
/// \return address of the allocated memory block or NULL in case of no memory available.
/// \note MUST REMAIN UNCHANGED: \b osPoolAlloc shall be consistent in every CMSIS-RTOS.
void *osPoolAlloc (osPoolId pool_id)
{
    int dummy;
    void *p = NULL;
    uint32_t i;
    uint32_t index;

    if (inHandlerMode()) {
        dummy = portSET_INTERRUPT_MASK_FROM_ISR();
    }
    else {
        vPortEnterCritical();
    }

    for (i = 0; i < pool_id->pool_sz; i++) {
        index = pool_id->currentIndex + i;
        if (index >= pool_id->pool_sz) {
            index = 0;
        }

        if (pool_id->markers[index] == 0) {
            pool_id->markers[index] = 1;
            p = (void *)((uint32_t)(pool_id->pool) + (index * pool_id->item_sz));
            pool_id->currentIndex = index;
            break;
        }
    }

    if (inHandlerMode()) {
        portCLEAR_INTERRUPT_MASK_FROM_ISR(dummy);
    }
    else {
        vPortExitCritical();
    }

    return p;
}


/// Allocate a memory block from a memory pool and set memory block to zero
/// \param[in]     pool_id       memory pool ID obtain referenced with \ref osPoolCreate.
/// \return address of the allocated memory block or NULL in case of no memory available.
/// \note MUST REMAIN UNCHANGED: \b osPoolCAlloc shall be consistent in every CMSIS-RTOS.
void *osPoolCAlloc (osPoolId pool_id)
{
    void *p = osPoolAlloc(pool_id);

//TODO clear memory

    return p;
}


/// Return an allocated memory block back to a specific memory pool
/// \param[in]     pool_id       memory pool ID obtain referenced with \ref osPoolCreate.
/// \param[in]     block         address of the allocated memory block that is returned to the memory pool.
/// \return status code that indicates the execution status of the function.
/// \note MUST REMAIN UNCHANGED: \b osPoolFree shall be consistent in every CMSIS-RTOS.
osStatus osPoolFree (osPoolId pool_id, void *block)
{
    uint32_t index;

    if (pool_id == NULL) {
        return osErrorParameter;
    }

    if (block == NULL) {
        return osErrorParameter;
    }

    if (block < pool_id->pool) {
        return osErrorParameter;
    }

    index = (uint32_t)block - (uint32_t)(pool_id->pool);
    if (index % pool_id->item_sz) {
        return osErrorParameter;
    }
    index = index / pool_id->item_sz;
    if (index >= pool_id->pool_sz) {
        return osErrorParameter;
    }

    pool_id->markers[index] = 0;

    return osOK;
}


#endif   // Memory Pool Management available


//  ==== Message Queue Management Functions ====

#if (defined (osFeature_MessageQ)  &&  (osFeature_MessageQ != 0))     // Message Queues available

/// Create and Initialize a Message Queue.
/// \param[in]     queue_def     queue definition referenced with \ref osMessageQ.
/// \param[in]     thread_id     thread ID (obtained by \ref osThreadCreate or \ref osThreadGetId) or NULL.
/// \return message queue ID for reference by other functions or NULL in case of error.
/// \note MUST REMAIN UNCHANGED: \b osMessageCreate shall be consistent in every CMSIS-RTOS.
osMessageQId osMessageCreate (osMessageQDef_t *queue_def, osThreadId thread_id)
{
    (void) thread_id;

    return xQueueCreate(queue_def->queue_sz, queue_def->item_sz);
}


/// Put a Message to a Queue.
/// \param[in]     queue_id      message queue ID obtained with \ref osMessageCreate.
/// \param[in]     info          message information.
/// \param[in]     millisec      timeout value or 0 in case of no time-out.
/// \return status code that indicates the execution status of the function.
/// \note MUST REMAIN UNCHANGED: \b osMessagePut shall be consistent in every CMSIS-RTOS.
osStatus osMessagePut (osMessageQId queue_id, uint32_t info, uint32_t millisec)
{
    portBASE_TYPE taskWoken = pdFALSE;
    portTickType ticks;

    ticks = millisec / portTICK_RATE_MS;
    if (ticks == 0) {
        ticks = 1;
    }

    if (inHandlerMode()) {
        if (xQueueSendFromISR(queue_id, (const void *)info, &taskWoken) != pdTRUE) {
            return osErrorOS;
        }
        portEND_SWITCHING_ISR(taskWoken);
    }
    else {
        if (xQueueSend(queue_id, (const void *)info, ticks) != pdTRUE) {
            return osErrorOS;
        }
    }

    return osOK;
}



/// Get a Message or Wait for a Message from a Queue.
/// \param[in]     queue_id      message queue ID obtained with \ref osMessageCreate.
/// \param[in]     millisec      timeout value or 0 in case of no time-out.
/// \return event information that includes status code.
/// \note MUST REMAIN UNCHANGED: \b osMessageGet shall be consistent in every CMSIS-RTOS.
osEvent osMessageGet (osMessageQId queue_id, uint32_t millisec);

#endif     // Message Queues available


//  ==== Mail Queue Management Functions ====

#if (defined (osFeature_MailQ)  &&  (osFeature_MailQ != 0))     // Mail Queues available


typedef struct os_mailQ_cb {
    osMailQDef_t *queue_def;
    xQueueHandle handle;
    osPoolId pool;
} os_mailQ_cb_t;


/// Create and Initialize mail queue
/// \param[in]     queue_def     reference to the mail queue definition obtain with \ref osMailQ
/// \param[in]     thread_id     thread ID (obtained by \ref osThreadCreate or \ref osThreadGetId) or NULL.
/// \return mail queue ID for reference by other functions or NULL in case of error.
/// \note MUST REMAIN UNCHANGED: \b osMailCreate shall be consistent in every CMSIS-RTOS.
osMailQId osMailCreate (osMailQDef_t *queue_def, osThreadId thread_id)
{
    osPoolDef_t pool_def = {queue_def->queue_sz, queue_def->item_sz};
    (void) thread_id;


    /* Create a mail queue control block */
    *(queue_def->cb) = pvPortMalloc(sizeof(struct os_mailQ_cb));
    if (*(queue_def->cb) == NULL) {
        return NULL;
    }
    (*(queue_def->cb))->queue_def = queue_def;

    /* Create a queue in FreeRTOS */
    (*(queue_def->cb))->handle = xQueueCreate(queue_def->queue_sz, sizeof(void *));
    if ((*(queue_def->cb))->handle == NULL) {
        vPortFree(*(queue_def->cb));
        return NULL;
    }

    /* Create a mail pool */
    (*(queue_def->cb))->pool = osPoolCreate(&pool_def);
    if ((*(queue_def->cb))->pool == NULL) {
        //TODO: Delete queue. How to do it in FreeRTOS?
        vPortFree(*(queue_def->cb));
        return NULL;
    }

    return *(queue_def->cb);
}



/// Allocate a memory block from a mail
/// \param[in]     queue_id      mail queue ID obtained with \ref osMailCreate.
/// \param[in]     millisec      timeout value or 0 in case of no time-out
/// \return pointer to memory block that can be filled with mail or NULL in case error.
/// \note MUST REMAIN UNCHANGED: \b osMailAlloc shall be consistent in every CMSIS-RTOS.
void *osMailAlloc (osMailQId queue_id, uint32_t millisec)
{
    void *p;
    (void) millisec;


    if (queue_id == NULL) {
        return NULL;
    }

    p = osPoolAlloc(queue_id->pool);

    return p;
}



/// Allocate a memory block from a mail and set memory block to zero
/// \param[in]     queue_id      mail queue ID obtained with \ref osMailCreate.
/// \param[in]     millisec      timeout value or 0 in case of no time-out
/// \return pointer to memory block that can shall filled with mail or NULL in case error.
/// \note MUST REMAIN UNCHANGED: \b osMailCAlloc shall be consistent in every CMSIS-RTOS.
void *osMailCAlloc (osMailQId queue_id, uint32_t millisec)
{
    uint32_t i;
    void *p = osMailAlloc(queue_id, millisec);

    if (p) {
        for (i = 0; i < sizeof(queue_id->queue_def->item_sz); i++) {
            ((uint8_t *)p)[i] = 0;
        }
    }

    return p;
}



/// Put a mail to a queue
/// \param[in]     queue_id      mail queue ID obtained with \ref osMailCreate.
/// \param[in]     mail          memory block previously allocated with \ref osMailAlloc or \ref osMailCAlloc.
/// \return status code that indicates the execution status of the function.
/// \note MUST REMAIN UNCHANGED: \b osMailPut shall be consistent in every CMSIS-RTOS.
osStatus osMailPut (osMailQId queue_id, void *mail)
{
    portBASE_TYPE taskWoken;


    if (queue_id == NULL) {
        return osErrorParameter;
    }

    taskWoken = pdFALSE;

    if (inHandlerMode()) {
        if (xQueueSendFromISR(queue_id->handle, &mail, &taskWoken) != pdTRUE) {
            return osErrorOS;
        }
        portEND_SWITCHING_ISR(taskWoken);
    }
    else {
        if (xQueueSend(queue_id->handle, &mail, 0) != pdTRUE) {  //TODO where to get timeout value?
            return osErrorOS;
        }
    }

    return osOK;
}



/// Get a mail from a queue
/// \param[in]     queue_id      mail queue ID obtained with \ref osMailCreate.
/// \param[in]     millisec      timeout value or 0 in case of no time-out
/// \return event that contains mail information or error code.
/// \note MUST REMAIN UNCHANGED: \b osMailGet shall be consistent in every CMSIS-RTOS.
osEvent osMailGet (osMailQId queue_id, uint32_t millisec)
{
    portBASE_TYPE taskWoken;
    portTickType ticks;
    osEvent event;

    event.def.mail_id = queue_id;

    if (queue_id == NULL) {
        event.status = osErrorParameter;
        return event;
    }

    taskWoken = pdFALSE;

    ticks = 0;
    if (millisec == osWaitForever) {
        ticks = portMAX_DELAY;
    }
    else if (millisec != 0) {
        ticks = millisec / portTICK_RATE_MS;
        if (ticks == 0) {
            ticks = 1;
        }
    }

    if (inHandlerMode()) {
        if (xQueueReceiveFromISR(queue_id->handle, &event.value.p, &taskWoken) == pdTRUE) {
            /* We have mail */
            event.status = osEventMail;
        }
        else {
            event.status = osOK;
        }
        portEND_SWITCHING_ISR(taskWoken);
    }
    else {
        if (xQueueReceive(queue_id->handle, &event.value.p, ticks) == pdTRUE) {
            /* We have mail */
            event.status = osEventMail;
        }
        else {
            event.status = (ticks == 0) ? osOK : osEventTimeout;
        }
    }

    return event;
}



/// Free a memory block from a mail
/// \param[in]     queue_id      mail queue ID obtained with \ref osMailCreate.
/// \param[in]     mail          pointer to the memory block that was obtained with \ref osMailGet.
/// \return status code that indicates the execution status of the function.
/// \note MUST REMAIN UNCHANGED: \b osMailFree shall be consistent in every CMSIS-RTOS.
osStatus osMailFree (osMailQId queue_id, void *mail)
{
    if (queue_id == NULL) {
        return osErrorParameter;
    }

    osPoolFree(queue_id->pool, mail);

    return osOK;
}

#endif  // Mail Queues available





#if 1//defined (__TARGET_ARCH_4T)              /* ARM7TDMI */

#else                                       /* Cortex-M */

extern void SVC_Handler (void);
extern void PendSV_Handler (void);
extern void SysTick_Handler (void);

extern void vPortSVCHandler (void);
extern void xPortPendSVHandler (void);
extern void xPortSysTickHandler (void);


  #if defined (__GNUC__)
/* Because of the way the FreeRTOS SVC handler is implemented,
 * we must make sure that we do not manipulate the stack or the LR register here.
 * It may otherwise happen depending on optimization level!
 */
__attribute__((naked)) void SVC_Handler (void);
void SVC_Handler (void)
{
    __asm (
        " b.w vPortSVCHandler \n\t"
    );
}

__attribute__((naked)) void SysTick_Handler (void);
void SysTick_Handler (void)
{
    __asm (
        " b.w xPortSysTickHandler \n\t"
    );
}

__attribute__((naked)) void PendSV_Handler (void);
void PendSV_Handler (void)
{
    __asm (
        " b.w xPortPendSVHandler \n\t"
    );
}

  #else

void SVC_Handler (void)
{
    vPortSVCHandler();
}

void SysTick_Handler (void)
{
    xPortSysTickHandler();
}

void PendSV_Handler (void)
{
    xPortPendSVHandler();
}

  #endif

#endif


