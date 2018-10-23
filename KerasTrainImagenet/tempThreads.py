#exec(open("tempThreads.py").read())

import threading, time

class AugSequence:

	def prepDataAsync(self):
		time.sleep(1)
		print ("preparing data")

	def __init__(self):
		a=1
		
	
	def first(self):
		self.th = threading.Thread(target=self.prepDataAsync)
		self.th.start()
		print ("thread started")
	
	def second(self):
		self.th.join()
		del self.th
		print ("thread joined")
	