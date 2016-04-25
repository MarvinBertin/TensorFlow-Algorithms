from __future__ import print_function
import tensorflow as tf
import numpy as np


from hmm import HMMNumpy, HMMTensorflow

def main():
	p0 = np.array([0.6, 0.4])

	emi = np.array([[0.5, 0.1],
	                [0.4, 0.3],
	                [0.1, 0.6]])

	trans = np.array([[0.7, 0.3],
	                  [0.4, 0.6]])
	states = {0:'Healthy', 1:'Fever'}
	obs = {0:'normal', 1:'cold', 2:'dizzy'}

	obs_seq = np.array([0, 1, 2]) 

	def dptable(V):
	    print(" ".join(("%10d" % i) for i in range(V.shape[0])))
	    for i, y in enumerate(pathScores.T):
	        print("%.7s: " % states[i] +" ".join("%.7s" % ("%f" % yy) for yy in y))

	print()
	print("TensorFlow Example: ")

	tf_model = HMMTensorflow(trans, p0)

	y = emi[obs_seq]
	tf_s_graph, tf_scores_graph = tf_model.viterbi_decode(y, len(y))
	tf_s = tf.Session().run(tf_s_graph)
	print("Most likely States: ", [obs[s] for s in tf_s])

	tf_scores = [tf_scores_graph[0]]
	tf_scores.extend([tf.Session().run(g) for g in tf_scores_graph[1:]])
	pathScores = np.array(np.exp(tf_scores))
	dptable(pathScores)

	print()
	print("numpy Example: ")
	np_model = HMMNumpy(trans, p0)

	y = emi[obs_seq]
	np_states, np_scores = np_model.viterbi_decode(y)
	print("Most likely States: ",[obs[s] for s in np_states])
	pathScores = np.array(np.exp(np_scores))
	dptable(pathScores)

if __name__ == "__main__":
	main()
