import util
import numpy as np

def main():
	########## BBOX overlap test ##########
	region_bbox = np.array([50, 50, 100, 100])
	gt_bboxes = np.array([[75, 25, 125, 75], # 0.14
						[50, 200, 100, 250], # 0 
						[50, 50, 100, 100] # 1
						])
	out = util.computeOverlap(region_bbox, gt_bboxes)
	assert(np.abs(out[0] - 0.14935926) <= 1e-7)
	assert(np.abs(out[1] - 0) <= 1e-7)
	assert(np.abs(out[2] - 1) <= 1e-7)
	print "Passed all tests."

if __name__ == '__main__':
	main()