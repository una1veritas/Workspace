'''
Created on 2025/04/03

@author: sin
'''

import http.server
import socketserver

# Define the handler to use for HTTP requests
Handler = http.server.SimpleHTTPRequestHandler

# Define the port on which the server will listen
PORT = 8000

# Create the server object, binding to PORT
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving on port {PORT}")
    # Serve until the server is stopped
    httpd.serve_forever()
