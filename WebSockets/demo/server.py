'''
Created on 2025/04/03

@author: sin
'''

import asyncio
import websockets

async def handler(websocket):
    print(f"New connection from {websocket.remote_address}")
    try:
        async for message in websocket:
            print(f"Received message: {message}")
            await websocket.send(f"Echo: {message}")
    except websockets.exceptions.ConnectionClosed as e:
        print(f"Connection closed: {e}")

async def main():
    server = await websockets.serve(handler, "0.0.0.0", 8765)
    print("Server started on ws://0.0.0.0:8765")
    await server.wait_closed()

asyncio.run(main())