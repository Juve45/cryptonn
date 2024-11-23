from mlsocket import MLSocket

import numpy as np

HOST = '127.0.0.1'
PORT = 65432

# Make an ndarray
data = np.array([1, 2, 3, 4])


# Send data
with MLSocket() as s:
    s.connect((HOST, PORT)) # Connect to the port and host
    s.send(data) # After sending the data, it will wait until it receives the reponse from the server
    s.send(np.array([99,84,22])) # Same
    print(s.recv(1024))