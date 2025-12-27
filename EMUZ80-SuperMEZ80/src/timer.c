/*
 * Copyright (c) 2023 @hanyazou
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <supermez80.h>
#include <stdio.h>

static timer_t *timer_root = NULL;

void timer_run(void)
{
    uint32_t tick;
    board_tick(&tick);

    while (timer_root != NULL && timer_root->tick_expire <= tick) {
        timer_expire(timer_root);
    }
}

void timer_set_absolute(timer_t *timer, timer_callback_t callback, uint32_t tick)
{
    timer_t **tpp = &timer_root;
    timer_cancel(timer);
    timer->tick_expire = tick;
    timer->callback = callback;
    while (*tpp != NULL) {
        if (tick <= (*tpp)->tick_expire) {
            timer->next = (*tpp)->next;
            *tpp = timer;
            return;
        }
        tpp = &((*tpp)->next);
    }
    timer->next = NULL;
    *tpp = timer;
}

void timer_set_relative(timer_t *timer, timer_callback_t callback, unsigned int timer_ms)
{
    uint32_t tick;
    board_tick(&tick);
    tick += ((uint32_t)timer_ms + (1000 / BOARD_TICK_HZ - 1)) * BOARD_TICK_HZ / 1000;
    timer_set_absolute(timer, callback, tick);
}

int timer_cancel(timer_t *timer)
{
    timer_t **tpp = &timer_root;
    while (*tpp != NULL) {
        if (*tpp == timer) {
            *tpp = timer->next;
            timer->next = NULL;  // fail safe
            return 1;
        }
        tpp = &((*tpp)->next);
    }
    return 0;
}

int timer_expire(timer_t *timer)
{
    int result = timer_cancel(timer);
    if (result) {
        timer->callback(timer);
    }
    return result;
}
