'''
Created on 2025/04/22

@author: sin
'''
'''
Created on 2025/04/04

@author: sin
'''

import asyncio
import websockets

'''declare function as asynchronous'''
async def handler(websocket):
    print(f"New connection from {websocket.remote_address}")
    try:
        async for message in websocket:
            print(f"Received message: {message}")
            await websocket.send(f"Echo: {message}")
    except websockets.exceptions.ConnectionClosed as e:
        print(f"Connection closed: {e}")

async def main():
    '''wait until server is started.'''
    server = await websockets.serve(handler, "0.0.0.0", 8765)
    print("Server started on ws://0.0.0.0:8765")

    '''wait until server is closed.'''
    await server.wait_closed()

if __name__ == '__main__':
    asyncio.run(main())

