
from itertools import chain
from typing import Iterable
from collections import OrderedDict

HEADLINE_INTERVAL = 20

def logger(loggables: OrderedDict, interval: int):
	counter = 0
	def print_row(items: Iterable[str]):
		print(' '.join(item.ljust(12) for item in items))
	def log(iteration):
		nonlocal counter
		if not counter%(HEADLINE_INTERVAL*interval):
			print_row(chain(['i'], loggables.keys()))
		counter += 1
		if not iteration%interval:
			values = ('%.2E'%loggable.eval() for loggable in loggables.values())
			print_row(chain(['%d'%iteration], values))
	return log
