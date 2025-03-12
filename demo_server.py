#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 06:03:11 2025

@author: revolabs
"""

import socket
import struct

def handle_client(client_socket):
    """
    Handle communication with a connected client.
    For any received data, send back a hard-coded response.
    """
    try:
        while True:
            data = client_socket.recv(1024)
            if not data:
                # No more data from the client.
                break

            print("Received from client:", data.decode('utf-8', errors='ignore'))

            # Hard-coded response data: a comma-separated string.
            response_data = "1.0,2.0,3.0,4.0,5.0,6.0"
            encoded_data = response_data.encode('utf-8')
            data_length = len(encoded_data)

            # Create a header:
            # - 24 bytes of padding (for example, use 'X' to fill these bytes)
            # - 4 bytes representing the length of the data using struct.pack
            header = b'X' * 24 + struct.pack('i', data_length)

            # Combine header and data.
            full_response = header + encoded_data

            # Send the full response back to the client.
            client_socket.sendall(full_response)
            print("Sent hard-coded response to client.")
    except Exception as e:
        print("Error handling client:", e)
    finally:
        client_socket.close()
        print("Closed client connection.")

def server_main():
    """
    Start the server, bind it to a port, and listen for incoming connections.
    """
    server_ip = "127.0.0.1"  # Listen on all available interfaces.
    server_port = 5051    # Use the same port as specified in the client code.
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        server_socket.bind((server_ip, server_port))
        server_socket.listen(5)  # Allow up to 5 queued connections.
        print(f"Server listening on {server_ip}:{server_port}")

        while True:
            client_socket, client_address = server_socket.accept()
            print(f"Accepted connection from {client_address}")
            # For test purposes, handle one client at a time.
            handle_client(client_socket)
    except Exception as e:
        print("Server error:", e)
    finally:
        server_socket.close()
        print("Server socket closed.")

if __name__ == "__main__":
    server_main()
