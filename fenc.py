from charm.toolbox.pairinggroup import PairingGroup,ZR,G1,G2,GT,pair
from charm.toolbox.integergroup import IntegerGroup
import numpy as np
from memory_profiler import profile 

# trials = 10
# group = PairingGroup("SS1024")
# g = group.random(G1)
# h = group.random(G1)
# i = group.random(G2)



# gh1 = pair(g**45, h**11)
# gh2 = pair(g, h) ** (11*45)

# if (gh1 == gh2):
# 	print("ok")
# else:
# 	print("Nok")

# assert group.InitBenchmark(), "failed to initialize benchmark"
# group.StartBenchmark(["Mul", "Exp", "Pair", "Granular"])
# for a in range(trials):
#     j = g * h
#     k = i ** group.random(ZR)
#     t = (j ** group.random(ZR)) / h
#     n = pair(h, i)
# group.EndBenchmark()

# msmtDict = group.GetGeneralBenchmarks()
# granDict = group.GetGranularBenchmarks()
# print("<=== General Benchmarks ===>")
# print("Results  := ", msmtDict)
# print("<=== Granular Benchmarks ===>")
# print("G1 mul   := ", granDict["Mul"][G1])
# print("G2 exp   := ", granDict["Exp"][G2])



class FEBO:

	def setup(self):
		
		self.group = PairingGroup("SS1024")
		s = self.group.random(ZR)
		g = self.group.random(G1)
		h = g ** s 

		self.msk = {"s" : s}
		self.mpk = {"g" : g, "h" : h}

	def key_derive(self, cmt, delta, y):

		s = self.msk['s']
		g = self.mpk['g']

		if delta == '+':
			skf = cmt ** s * g ** (-y)
		elif delta == '-':
			skf = cmt ** s * g ** y
		elif delta == '*':
			skf = cmt ** (s * y)
		else:
			skf = cmt ** (s * y ** (-1))
		return skf

	def encrypt(self, x):

		r = self.group.random(ZR)
		g = self.mpk['g']
		h = self.mpk['h']

		cmt = g ** r
		ct = h ** r * g ** x

		return {'cmt' : cmt, 'ct' : ct}

	def decrypt(self, skf, ct, delta, y):

		if delta == '+' or delta == '-':
			return ct / skf
		elif delta == '*':
			return ct ** y / skf
		else:
			return ct ** (y ** (-1)) / skf
		



def discrete_log_bf(group, g, ge):
	for i in range(1500):
		if g ** i == ge:
			return i
		ii = - i + group.order()
		if g ** ii == ge:
			return -i

	raise Exception('logarithm exponent is too big')

	



class FEIP:

	def precalc_log(self):
		order = 10 ** 7
		self.m = int(np.sqrt(order)) + 1
		self.baby_steps = {}

		ai = self.g ** 0
		for i in range(self.m + 1):
			self.baby_steps[ai] = i
			ai *= self.g 


	# @profile
	def discrete_log(self, b):
		#todo precalc discretelog 
		
		am = self.g ** self.m
		giant_step = b
		giant_step_2 = b
		for i in range (self.m + 1):
			if giant_step in self.baby_steps:
				bs = self.baby_steps[giant_step]
				result = bs + i * self.m
				return result
			giant_step = giant_step / am
			if giant_step_2 in self.baby_steps:
				bs = self.baby_steps[giant_step_2]
				result = bs - i * self.m
				return result
			giant_step_2 = giant_step_2 * am


		raise Exception('logarithm exponent is too big')



	def setup(self, n):

		self.group = PairingGroup("MNT224")

		s = [self.group.random(ZR) for i in range(n)]
		self.g = self.group.random(G1)
		h = [self.g ** i for i in s]
		self.precalc_log()

		self.msk = {"s" : s}
		self.mpk = {"g" : self.g, "h" : h}

	def key_derive(self, y):

		s = self.msk['s']
		# print("s", s, range(len(s) - 1))
		ans = s[-1] * y[-1]

		# print("ans", ans, flush=True)
		for i in range(len(s) - 1):
			ans += s[i] * y[i]

		return ans

	def encrypt(self, x):

		r = self.group.random(ZR)
		g = self.mpk['g']
		h = self.mpk['h']

		ct0 = g ** r
		cti = [h[i] ** r * g ** x[i] for i in range(len(x))]

		return {'ct0' : ct0, 'cti' : cti}

	def decrypt(self, ct, skf, y):
		g = self.mpk['g']
		# print("y", y)
		ct0 = ct['ct0']
		cti = ct['cti']
		prod = 1 # ct[0] ** y[0] / (ct0)
		for i in range(len(y)):
			if y[i] < 0:
				y[i] += self.group.order()
			prod *= cti[i] ** y[i]
		prod /= ct0 ** skf

		return self.discrete_log(prod)



	
