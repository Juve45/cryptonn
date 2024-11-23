from mlsocket import MLSocket
import numpy as np
HOST = '127.0.0.1'
PORT = 65432

with MLSocket() as s:
	s.bind((HOST, PORT))
	s.listen()
	conn, address = s.accept()

	with conn:
		data = conn.recv(1024) # This will block until it receives all the data send by the client, with the step size of 1024 bytes.
		clf = conn.recv(1024) # Same
		conn.send(np.array([99,84,22]))
print(data)
print(clf)