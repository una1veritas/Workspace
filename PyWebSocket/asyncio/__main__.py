'''
Created on 2025/04/22

@author: sin
'''

import asyncio
import time
import sys

async def say_after(delay, what):
    await asyncio.sleep(delay)
    print(what)

async def main():
    print(f"started at {time.strftime('%X')}")

    await say_after(1, 'hello')
    await say_after(2, 'world')

    print(f"finished at {time.strftime('%X')}")

print(sys.version)
asyncio.run(main())
