#Python code to take .raw files and generate .info files

import numpy as np
import networkx as nx
import multiprocessing as mp
from scipy.sparse.linalg import svds
from scipy.special import gamma, gammainc
from scipy.misc import comb
from itertools import combinations_with_replacement, combinations
import glob
import argparse
import time

##################
### READ/WRITE ###
##################
#read information from a .raw file and return a dictionary of community memberships, 
#a matrix of adjacency probabilities, and an edge list
def readRaw(fileName):
  (C, P, E, S, pVal, diameter) = ({}, [], [], -1, -1, -1)
  with open(fileName, 'r') as f:
    sect = 0
    for line in f:
      if line[0]=='#': sect += 1
      elif sect==1: #test stat
        l = line.strip().split(':')
        S = float(l[1].split(',')[0])
        pVal = float(l[1].split(',')[1])
      elif sect==2: #get diameter
        l = line.strip().split(':')
        diameter = int(l[1])
      elif sect==3: #communities
        l = line.strip().split(':')
        C[int(l[0])] = map(int, l[1].split(','))
      elif sect==4: #params
        l = eval(line.strip())
        P.append(l)
      elif sect==5: #edge list
        l = line.strip().split('\t')
        E.append((int(l[0]), int(l[1])))
  return (C, P, E, S, pVal, diameter)

#write information to a .info file in the format #Test Stat followed by goodness of fit information based on https://arxiv.org/abs/1412.4857,
##Graph quantities follows by the diameter of the graph, #Communities followed by (community id): (comma seperated list of members), #Adjacency
#Probabilities followed by rows of the matrix P, #Edge List followed by edges each on its own line, #Adjacency Matrix followed by the rows of
#the adjacency matrix, #Distance Matrix followed by the rows of the distance matrix, #Distance Time followed by the time it took to generate the distance matrix
def writeInfo(C, P, E, A, D, elapsed, diameter, outFile):
  with open(outFile, 'w') as o:
    o.write('#Test Stat\n')
    S = testStat(A, P, C)
    pVal = pValueTW(0.05/2., approx=True)
    o.write('Stat, p-value, result:'+str(S)+','+str(pVal)+','+('reject\n' if S>=pVal else 'accept\n'))
    o.write('#Graph Quantities\n')
    o.write('Diameter:'+str(diameter)+'\n')
    o.write('#Communities\n')
    for c in sorted(C.keys()): o.write(str(c)+':'+','.join(map(str, sorted(C[c])))+'\n')
    o.write('#Adjacency Probabilities\n')
    for row in P: o.write(str(row)+'\n')
    o.write('#Edge List\n')
    for (u,v) in E: o.write(str(u)+'\t'+str(v)+'\n')
    o.write('#Adjacency Matrix\n')
    for row in A: o.write(str(row)+'\n')
    o.write('#Distance Matrix\n')
    for row in D: o.write(str(row)+'\n')
    o.write('#Distance Time\n')
    o.write(str(elapsed)+'\n')

#read information from a .info file to extract communities, adjacency probabilities, edge list, adjacency matrix,
#distance matrix, and test stastic information
def readInfo(fileName):
  (C, P, E, A, D, elapsed, S, pVal, diameter) = ({}, [], [], [], [], -1, -1, -1, -1)
  with open(fileName, 'r') as f:
    sect = 0
    for line in f:
      if line[0]=='#': sect += 1
      elif sect==1: #test stat info
        l = line.strip().split(':')
        S = float(l[1].split(',')[0])
        pVal = float(l[1].split(',')[1])
      elif sect==2: #get diameter
        l = line.strip().split(':')
        diameter = int(l[1])
      elif sect==3: #communities
        l = line.strip().split(':')
        C[int(l[0])] = map(int, l[1].split(','))
      elif sect==4: #params
        l = eval(line.strip())
        P.append(l)
      elif sect==5: #edge list
        l = line.strip().split('\t')
        E.append((int(l[0]), int(l[1])))
      elif sect==6: #adjacency matrix
        l = eval(line.strip().replace('inf', 'np.inf'))
        A.append(l)
      elif sect==7: #distance matrix
        l = eval(line.strip().replace('inf', 'np.inf'))
        D.append(l)
      elif sect==8: #distance matrix time
        l = float(line.strip())
        elapsed = l
  return (C, P, E, A, D, elapsed, S, pVal, diameter)    

#################
### SBM SYNTH ###
#################
#generate a synthetic network according the SBM with the given parameters
#the number of nodes in each community is !!!!!!
def genSBM(C, P, N):
  G = nx.Graph()
  origN = float(sum(len(C[c]) for c in C))
  comSizes = [int(np.around(N * len(C[i]) / origN)) for i in xrange(len(C))]
  synthC = {i:range(sum(comSizes[:i]), sum(comSizes[:i+1])) for i in xrange(len(comSizes))}
  G.add_nodes_from(range(sum(comSizes)))
  for (i,j) in combinations(xrange(sum(len(synthC[c]) for c in synthC)), 2): 
    comI = [c for c in synthC if i in synthC[c]][0]
    comJ = [c for c in synthC if j in synthC[c]][0]
    if np.random.choice([True, False], p=[P[comI][comJ], 1-P[comI][comJ]], size=1): G.add_edge(i, j)
  return (synthC, G)

#######################
### ESTIMATE PARAMS ###
#######################
#estimate the adjacency probabilities based on community membership and edges in a given graph
def estimateP(C, G):
  counts = [[0 for _ in xrange(len(C))] for _ in xrange(len(C))]
#  for c in C: print(c, C[c])
  for (x,y) in G.edges():
#    print('  estimate p edge coms', x, y, [c for c in C if x in C[c]], [c for c in C if y in C[c]])
    xCom = [c for c in C if x in C[c]][0]
    yCom = [c for c in C if y in C[c]][0]
    if xCom==yCom: counts[xCom][yCom] += 1
    else:
      counts[xCom][yCom] += 1
      counts[yCom][xCom] += 1
  for i in xrange(len(counts)):
    for j in xrange(i, len(counts[i])):
      prob = counts[i][j] / float(comb(len(C[i]), 2) if i==j else len(C[i])*len(C[j]))
      counts[i][j] = prob
      counts[j][i] = prob
  return counts

######################
### TEST STATISTIC ###
######################
#
#A - an adjacency matrix
#P - P[i][j] is the probability that a node from community i is adjacent to a node in community j
#C - dictionary community -> member nodes
def testStat(A, P, C):
  nodeComs = genComDict(C)
  n = len(A)
  Ap = [[(A[i][j] - P[nodeComs[i]][nodeComs[j]]) / np.sqrt((n-1)*P[nodeComs[i]][nodeComs[j]]*(1-P[nodeComs[i]][nodeComs[j]]))
         if nodeComs[i]!=nodeComs[j] else 0.0 for j in xrange(len(A[i]))] for i in xrange(len(A))]
  try:
    maxSV = svds(Ap, k=1, return_singular_vectors=False)[0]
#    print('mm', maxSV)
    S = np.power(n, 2./3)*(maxSV-2)
    return S
  except:
    return 9999

#
def genComDict(C):
  D = {}
  for c in C:
    for n in C[c]:
      D[n] = c
  return D

###################
### TRACY-WIDOM ###
###################
#
def pValueTW(S, approx=True):
  if approx:
    k = 46.446
    theta = 0.186054
    alpha = 9.84801
    cdf = lambda x: gammainc(k, (x+alpha/theta))
    val = cdf(S)
#    print('val', val, 'S', S)
    return (1-val) #val if val<0.5 else (1-val) #we want upper quantile
  else: #use a table of values? from mathworks, need account
    return -1

############
### MAIN ###
############
#for all .raw files, generate some number of synthetic networks with the same parameters
#the number of nodes in the synthetic networks may be different from the number in the original
def genSynthNetworks(N=-1, num=1):
  for fileName in glob.glob('*.raw'):
    curNum = len(glob.glob('../'+fileName.split('.')[0]+'_*.info'))
    (C, P, E, _, _, _) = readRaw(fileName)
    N = sum(len(C[c]) for c in C) if N==-1 else N
    for i in xrange(curNum, num, 1):
      outFile = '../'+'.'.join(fileName.split('.')[:-1]) + '_' + str(i) + '.info'
      print('*******', outFile, '*********')
      (synthC, G) = genSBM(C, P, N)
      numNodes = len(G.nodes())
      print('   adjacency matrix')
      A = [[(1 if G.has_edge(x, y) else 0) if x!=y else -1 for y in xrange(numNodes)] for x in xrange(numNodes)]
      print('   distance matrix')
      D = nx.floyd_warshall(G)
      #make the distance matrix ints.... instead of inf when not connected, use -2 (-1 is diagonal in adj matrix... just to avoid confusion)
      #D = [[(D[x][y] if y in D[x] else np.inf) for y in xrange(numNodes)] for x in xrange(numNodes)]
      D = [[(int(D[x][y]) if y in D[x] else -2) for y in xrange(numNodes)] for x in xrange(numNodes)]
#      print(len(G.nodes()), sum(len(synthC[c]) for c in synthC))
      synthP = estimateP(synthC, G)
      synthE = G.edges() #guaranteed to be no self loops or multiedges based on construction
      print('   write')
      writeInfo(synthC, synthP, synthE, A, D, outFile)
      print('   done')

#
def genSynthNetworksMP(network='polblogs', N=-1, num=1, procs=1):
  (C, P, E, _, _, _) = readRaw(network+'.raw')
  done = [int(f.split('.')[-2].split('_')[-1]) for f in glob.glob('../SynthNetworks/'+network+'_*.info')]
  jobs = [(network, C, P, E, i, N) for i in xrange(num) if num not in done]
#  curNum = len(glob.glob('../SynthNetworks/'+network+'_'+str(N)+'_*.info'))
#  jobs = [(network, C, P, E, i, N) for i in xrange(curNum, num, 1)]

  pool = mp.Pool(processes=procs)
  results = pool.map_async(genNetworks, jobs)
  results = results.get()
  pool.close()
  pool.join()

#
def genNetworks((network, C, P, E, i, N)):
  np.random.seed()
  N = sum(len(C[c]) for c in C) if N==-1 else N
  outFile = '../SynthNetworks/'+network+ '_' + str(N) + '_' + str(i) + '.info'
  (synthC, G) = genSBM(C, P, N)
  numNodes = len(G.nodes())
  print('   adjacency matrix', network, i)
  A = [[(1 if G.has_edge(x, y) else 0) if x!=y else -1 for y in xrange(numNodes)] for x in xrange(numNodes)]
  print('   distance matrix', network, i)
  (D, elapsed) = ([], -1)
  if N<=5000:
    start = time.clock()
    D = nx.floyd_warshall(G)
    #make the distance matrix ints.... instead of inf when not connected, use -2 (-1 is diagonal in adj matrix... just to avoid confusion)
    #D = [[(D[x][y] if y in D[x] else np.inf) for y in xrange(numNodes)] for x in xrange(numNodes)]
    D = [[(int(D[x][y]) if y in D[x] else -2) for y in xrange(numNodes)] for x in xrange(numNodes)]
    elapsed = time.clock() - start
  synthP = estimateP(synthC, G)
  synthE = G.edges() #guaranteed to be no self loops or multiedges based on construction
  print('   diameter', 'num nodes', numNodes)
  diameter = nx.diameter(G) if nx.is_connected(G) else -1
  print('   write', network, i)
  writeInfo(synthC, synthP, synthE, A, D, elapsed, diameter, outFile)
  print('   done', network, i)

#
def rerun(network):
  (C, P, E, _, _, _) = readRaw(network+'.raw')
  jobs = []
  for f in glob.glob('../SynthNetworks/'+network+'_*.info'):
    num = int(f.split('.')[-2].split('_')[-1])
    (_, _, _, _, _, elapsed, S, pVal, diameter) = readInfo(f)
    if elapsed==-1:
      jobs.append((network, C, P, E, num, N))
      print('incomplete', num)
  for i,job in enumerate(jobs):
    print(i)
    genNetworks(job)

#
if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Generate Synthetic Networks')
  parser.add_argument('--network', type=str, default='polblogs', required=False,
                      help='name of the network to consider. this is the prefix of .raw files assciated with the network')
  parser.add_argument('--size', type=int, default=2000, required=False,
                      help='the approximate number of nodes in the synthetic networks')
  args = parser.parse_args()


  print('NETWORK: ', args.network, 'SIZE: ', args.size)

  num = 30
  N = args.size
#  N = 2000
#  genSynthNetworks(N=N, num=num)

  #svds did not converge in some cases, seemed to break multithreading
#  network = args.network
#  rerun(network)

#  (C, P, E, _, _, _) = readRaw(network+'.raw')
#  curNum = len(glob.glob('../SynthNetworks/'+network+'_'+str(N)+'_*.info'))
#  jobs = [(network, C, P, E, i, N) for i in xrange(curNum, num, 1)]
#  for i,job in enumerate(jobs):
#    print(i)
#    genNetworks(job)

  procs = 16
  genSynthNetworksMP(network=args.network, N=N, num=num, procs=procs)


