import numpy as np

def stack(data):
	result = None
	if len(data) == 0:
		result = None
	elif len(data) == 1:
		result = data[0]
	else:
		result = np.vstack(tuple(data))

	return result