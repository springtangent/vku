import time

class FPSCounter:
	"""
	print_interval is the number of seconds between prints
	"""
	def __init__(self, print_interval=5.0):
		super().__init__()
		self.t0 = self._get_time()
		self.print_interval = print_interval
		self.next_print = self.print_interval
		self.tick_count = 0

	def _get_time(self):
		return time.monotonic_ns()

	def tick(self):
		t = self._get_time()
		dt = t - self.t0
		dt = dt*1e-9
		self.tick_count += 1
		if dt > self.next_print:
			print("fps: ", self.tick_count/dt)
			self.next_print += self.print_interval
		return dt
