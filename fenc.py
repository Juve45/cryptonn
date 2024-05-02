from charm.toolbox.pairinggroup import PairingGroup,ZR,G1,G2,GT,pair
import numpy as np
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
		


obj = FEBO()
obj.setup()
enc = obj.encrypt(10)
key = obj.key_derive(enc['cmt'], '-', 4)
dec = obj.decrypt(key, enc['ct'], '-', 4)
# print(dec)

# print(obj.mpk['g'] ** 6)



def discrete_log_bf(group, g, ge):
	for i in range(1500):
		if g ** i == ge:
			return i
		ii = - i + group.order()
		if g ** ii == ge:
			return -i

	raise Exception('logarithm exponent is too big')

def discrete_log(group, a, b):
	order = 10 ** 6
	m = int(np.sqrt(order)) + 1
	baby_steps = {}

	for i in range(m + 1):
		baby_steps[g ** i] = i 

	am = 1 / (a ** m)
	giant_step = b
	for i in range (m + 1):
		if giant_step in baby_steps:
			bs = baby_steps[giant_step]
			result = bs + i * m
			return result
		giant_step = giant_step * am
	

group = PairingGroup("MNT224")
g = group.random(G1)
ge = g ** 10009
print("result", discrete_log(group, g, ge))

class FEIP:

	def setup(self, n):

		self.group = PairingGroup("MNT224")
		s = [self.group.random(ZR) for i in range(n)]
		g = self.group.random(G1)
		h = [g ** i for i in s]

		self.msk = {"s" : s}
		self.mpk = {"g" : g, "h" : h}

	def key_derive(self, y):

		s = self.msk['s']
		print("s", s, range(len(s) - 1))
		ans = s[-1] * y[-1]

		print("ans", ans, flush=True)
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

		return discrete_log_bf(self.group, g, prod)


a = FEIP()

a.setup(3)
ct = a.encrypt([1, 2, 3])
skf = a.key_derive([2, 2, 1])
r = a.decrypt(ct, skf, [2, 2, 1])
print(r)

	
