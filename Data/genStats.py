#Python code to collect information regarding resolving set size and run time for several
#algorithms on a given set of networks 

import argparse
import numpy as np
from scipy.misc import comb
import multiprocessing as mp
from metricDimension import approxMetricDim
from itertools import combinations_with_replacement
import glob
import pickle
import time

##################
### READ/WRITE ###
##################
#write a dictionary using the pickle function
#input: d - dictionary to write to a file
#       outFile - file in which to save the dictionary
def writeDict(d, outFile):
  with open(outFile, 'wb') as o:
    pickle.dump(d, o, protocol=pickle.HIGHEST_PROTOCOL)

#read a dictionary from a pickled file
#input: inFile - file where the dictionary is saved
#return: the dictionary saved in the given file
def readDict(inFile):
  d = {}
  with open(inFile, 'rb') as f:
    d = pickle.load(f)
  return d

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
        l = float(line.strip()) if ',' not in line else tuple([float(x) for x in line.strip().split(',')])
        elapsed = l
  return (C, P, E, A, D, elapsed, S, pVal, diameter)

############################
### PREORDER COMMUNITIES ###
############################
#
def preorderComs(M, C, P, alpha=-1):
  N = float(len(M))
  if alpha<=0: alpha = 1/(2.*N)
  greedyList = greedyKList(C, P, alpha=alpha)
  orderC = {c:sorted(C[c], key=lambda col: colCollisionProb(col, M, C, P, greedyList)) for c in C} #instead of ordering after an order over communities has been chosen, maybe do before? ie let k_i = 0?
  kList = [0 for _ in C]
  R = []
  while not checkResolving(R, M):
    nextCom = getMinCom(C, P, kList)
    kList[nextCom] += 1
    col = orderC[nextCom][0]
    R.append(col)
    orderC[nextCom] = orderC[nextCom][1:]
  return R

#
def colCollisionProb(col, M, C, P, kList):
  colCom = getCom(col, C)
  s = 0
  for (x,y) in combinations_with_replacement(xrange(len(C)), 2):
    weight = (len(C[x])-kList[x])*(len(C[y])-kList[y]) if x!=y else comb(len(C[x])-kList[x], 2)
    prod = [np.power(P[x][a]*P[y][a]+(1-P[x][a])*(1-P[y][a]), kList[a] if a!=colCom else max([0,kList[a]-1])) for a in xrange(len(C))]
    Px = (sum(1 for row in C[x] if M[row][col]==1))/float(len(C[x])) #(num 1s in col for row)/(total rows in com)
    Py = (sum(1 for row in C[y] if M[row][col]==1))/float(len(C[y]))
    prod += [Px*Py+(1.-Px)*(1.-Py)]
    prod = reduce(lambda a,b: a*b, prod)
    s += weight*prod
  return s

############################
### RECOMPUTE NODE ORDER ###
############################
#
def searchCom(M, C, P):
  choices = {c:list(np.random.permutation([x for x in C[c]])) for c in C} #very unlikely to have two the same (permutation is to account for tie breaking)
  kList = [0 for _ in C]
  R = []
  while not checkResolving(R, M):
    nextCom = getMinCom(C, P, kList)
    kList[nextCom] += 1
    col = getMinNode(choices[nextCom], M, C, P, kList)
    choices[nextCom].remove(col)
    R.append(col)
  return R

#
def getMinNode(com, M, C, P, kList):
  (minNode, minVal) = (-1, np.inf)
  for v in com:
    val = colCollisionProb(v, M, C, P, kList)
    if val < minVal: (minNode, minVal) = (v, val)
  return minNode

##########################################
### EMPIRICAL ADJACCENCY PROBABILITIES ###
##########################################
#keep track of (#1s) and (#rows) for each column in each block (ie col i has A 1s and B rows in community c)... this gives estimate of P_{i,c} (related to P for community of i and community c)
def empAdjProbs(M, C, P):
  empP = {col: {com:[sum(1 for row in C[com] if M[row][col]==1), len(C[com]) - (1 if col in C[com] else 0)] for com in C} for col in xrange(len(M))}
  pList = [[] for _ in C] #for each community a list of chosen columns
  R = []
  while not checkResolving(R, M):
    nextCom = getMinComEmp(C, P, empP, pList)
    col = getMinColEmp(C, empP, pList, nextCom)
    pList[nextCom] += [col]
    R.append(col)
    empP = updateEmp(empP, M, C, col) #Note that P does not change, that is out overall estimates for community adjacency probs stay the same but change col to col
  return R

#
def getMinComEmp(C, P, empP, pList):
  (minVal, minCom) = (np.inf, -1)
  for i in np.random.permutation(len(pList)):
    if len(pList[i])<len(C[i]):
      val = expectedCollisionsEmp(C, P, empP, pList, i)
      if val < minVal: (minVal, minCom) = (val, i)
  return minCom

#
def expectedCollisionsEmp(C, P, empP, pList, newC):
  s = 0
  counts = [len(l) for l in pList]
  counts[newC] += 1
  chosenCols = [p for com in pList for p in com]
  for (x,y) in combinations_with_replacement(xrange(len(C)), 2):
    weight = (len(C[x])-counts[x])*(len(C[y])-counts[y]) if x!=y else comb(len(C[x])-counts[x], 2)
    prod = [empiricalP(x, col, empP)*empiricalP(y, col, empP)+(1-empiricalP(x, col, empP))*(1-empiricalP(y, col, empP)) for col in chosenCols]
    prod += [P[x][newC]*P[y][newC]+(1-P[x][newC])*(1-P[y][newC])]
    prod = reduce(lambda a,b: a*b, prod)
    s += weight*prod
  return s
def empiricalP(com, col, empP):
  return float(empP[col][com][0]) / empP[col][com][1] if empP[col][com][1]!=0 else 0

#
def updateEmp(empP, M, C, col):
  com = getCom(col, C)
  for c in empP: #for every column
    #for the community that col belongs to
    if c!=col: empP[c][com][1] -= 1 #remove 1 from total (denominator) if c!=col (this was taken care of at the start)
    empP[c][com][0] -= (1 if M[col][c]==1 else 0) #if M[col][c] is 1, remove 1 from numerator also
  return empP

#
def getMinColEmp(C, empP, pList, nextCom):
  (minVal, minCol) = (np.inf, len(empP)+1) #out of bounds
  for col in [c for c in C[nextCom] if c not in pList[nextCom]]:
    val = colCollisionProbEmp(C, empP, pList, col)
    if val<minVal: (minVal, minCol) = (val, col)
  return minCol

#
def colCollisionProbEmp(C, empP, pList, newCol):
  s = 0
  newC = getCom(newCol, C)
  counts = [len(l) for l in pList]
  counts[newC] += 1
  chosenCols = [p for com in pList for p in com] + [newCol]
  for (x,y) in combinations_with_replacement(xrange(len(C)), 2):
    weight = (len(C[x])-counts[x])*(len(C[y])-counts[y]) if x!=y else comb(len(C[x])-counts[x], 2)
    prod = [empiricalP(x, col, empP)*empiricalP(y, col, empP)+(1-empiricalP(x, col, empP))*(1-empiricalP(y, col, empP)) for col in chosenCols]
    prod = reduce(lambda a,b: a*b, prod)
    s += weight*prod
  return s

########################
### GREEDY ALGORITHM ###
########################
#add one column at a time to a growing set according to the greedy algorithm wrt minimizing expected row collisions
#if the set is resolving stop, otherwise add another column
def greedyResSet(M, C, P):
  choices = {c:[x for x in C[c]] for c in C}
  kList = [0 for _ in C]
  R = []
#  order = []
  while not checkResolving(R, M):
    nextCom = getMinCom(C, P, kList)
#    if sum(kList)%100==0: print('R len', len(R), len(M[0]), kList, [len(C[c]) for c in C], [len(choices[c]) for c in choices])
    kList[nextCom] += 1
    col = np.random.choice(choices[nextCom], size=1)[0]
    choices[nextCom].remove(col)
    R.append(col)
#    order.append(nextCom)
#  print('R len', len(R), len(M[0]), kList, [len(C[c]) for c in C], [len(choices[c]) for c in choices])
#  print('greedy res set order', order)
  return R

#run the greedy algorithm to find a small value of each k_i such that expected collisions is less than alpha
def greedyBoundSet(M, C, P, alpha=-1):
  N = float(sum([len(C[c]) for c in C]))
  if alpha<=0: alpha = 1/(2.*N)
  kList = greedyKList(C, P, alpha=alpha)
#  kList = [0 for _ in C]
#  while expectedCollisions(C, P, kList) >= alpha:
#    nextCom = getMinCom(C, P, kList)
#    kList[nextCom] += 1
  return selectColumns(M, C, kList)

#randomly select columns from a matrix according to a given list
#repeat until a resolving set is found or count sets are tested
#if count is reached, increase a randomly (weighted by fraction 
#of remaining columns) chosen k by 1 and try again
def selectColumns(M, C, kList, count=10):
  pick = lambda: sum([list(np.random.choice(C[c], replace=False, size=min(kList[c], len(C[c])))) for c in xrange(len(kList))], [])
  R = pick()
  num = 0
  while not checkResolving(R, M) and num < count:
    R = pick()
    num += 1
  if checkResolving(R, M): return R
  else:
#    print('ASD', [len(C[c]) for c in xrange(len(kList))], kList, [float(len(C[c])-kList[c])/sum(len(C[d])-kList[d] for d in xrange(len(kList))) for c in xrange(len(kList))])
    index = np.random.choice(len(kList), p=[float(len(C[c])-kList[c])/sum(len(C[d])-kList[d] for d in xrange(len(kList))) for c in xrange(len(kList))])
    newKList = kList[:]
    newKList[index] += 1
    return selectColumns(M, C, newKList)
#  return R if checkResolving(R, M) else []

#find the community from which adding a single column will maximally decrease the number of expected collisions
def getMinCom(C, P, kList):
  (minVal, minCom) = (np.inf, -1)
  for i in np.random.permutation(len(kList)): #xrange(len(kList)):
    if kList[i]<len(C[i]):
      incI = kList[:]
      incI[i] += 1
      val = expectedCollisions(C, P, incI)
      if val<minVal: (minVal, minCom) = (val, i)
  return minCom

#approximate the expected number of row collisions given C, P, and a number of columns chosen from each community
def expectedCollisions(C, P, kList):
  s = 0
  for (x,y) in combinations_with_replacement(xrange(len(C)), 2):
    weight = (len(C[x])-kList[x])*(len(C[y])-kList[y]) if x!=y else comb(len(C[x])-kList[x], 2)
    prod = [np.power(P[x][a]*P[y][a]+(1-P[x][a])*(1-P[y][a]), kList[a]) for a in xrange(len(C))]
    prod = reduce(lambda a,b: a*b, prod)
    s += weight*prod
  return s

############################
### GREEDY MATRIX UPDATE ###
############################
#after choosing a community, remove the chosen column from the matrix and recompute the parameters
#repeat until a resolving set is found. 
def greedyMatrixUpdate(M, C, P):
  PCounts = getPCounts(M, C)
  choices = {c:[x for x in C[c]] for c in C}
  kList = [0 for _ in C]
  R = []
  while not checkResolving(R, M):
    pMat = [[PCounts[i][j] for j in xrange(len(choices))] for i in xrange(len(choices))]
    for i in xrange(len(choices)):
      for j in xrange(len(choices)):
        denom = float(comb(len(choices[i]), 2) if i==j else len(choices[i])*len(choices[j]))
        if denom==0: pMat[i][j] = 0.
        else: pMat[i][j] = pMat[i][j] / denom
    nextCom = getMinCom(choices, pMat, [0 for _ in C])
#    nextCom = getMinCom(choices, [[(PCounts[i][j]/float(comb(len(choices[i]), 2) if i==j else len(choices[i])*len(choices[j])) if len(choices[i])>0 and len(choices[j])>0 else 0) for j in xrange(len(choices))] for i in xrange(len(choices))], [0 for _ in C])
    kList[nextCom] += 1
    col = np.random.choice(choices[nextCom], size=1)[0]
    choices[nextCom].remove(col)
    PCounts = removeCounts(M, col, PCounts, C)
    R.append(col)
  return R

#assumes M is symmetric
def getPCounts(M, C):
  counts = [[0 for _ in C] for _ in C]
  for i in xrange(len(M)):
    for j in xrange(i, len(M[0])):
      if M[i][j]==1:
        xCom = getCom(i, C) #[index for index in C if i in C[index]][0]
        yCom = getCom(j, C) #[index for index in C if j in C[index]][0]
        counts[xCom][yCom] += 1
        if xCom!=yCom: counts[yCom][xCom] += 1
  return counts

#
def getCom(i, C):
  for c in C:
    if i in C[c]: return c
  return False

#
def removeCounts(M, col, PCounts, C):
  for i in xrange(len(M)):
    r = i if i < col else col
    c = col if i < col else i
    if M[r][c]==1:
      rCom = getCom(r, C) #[index for index in C if r in C[index]][0]
      cCom = getCom(c, C) #[index for index in C if c in C[index]][0]
      PCounts[rCom][cCom] -= 1
      if rCom!=cCom: PCounts[cCom][rCom] -= 1
  return PCounts


###################
### RANDOM SETS ###
###################
#add random columns to a set until it becomes resolving
def randomResSet(M):
  order = list(np.random.permutation(len(M[0])))
  R = [order[0]]
  order = order[1:]
  while not checkResolving(R, M):
    R.append(order[0])
    order = order[1:]
  return R

############
### ALGO ###
############
#take advantage of the monotone property in finding values of k_i such that the expected number of 
#collisions is small but so is k
def algoResSet(M, C, P, alpha=-1):
  N = float(sum([len(C[c]) for c in C]))
  if alpha<=0: alpha = 1/(2.*N)
  kList = greedyKList(C, P, alpha=alpha)
  kList = greedyDown(C, P, kList, alpha=alpha)
  check = nextComposition(-1, len(C), sum(kList)-1)
  count = 0
  temp = np.power(np.log(sum(len(C[c]) for c in C)), len(C))
  while check!=-1:
    if count%100000==0: print('   algo', count, 'k', sum(kList), 'kList', kList, 'check', check, 'size', temp)
    count += 1
    if expectedCollisions(C, P, check) <= alpha:
      kList = check[:]
      for i in xrange(len(check)):
        if check[i]>0:
          check[i] -= 1
          break
    check = nextComposition(check, len(C), sum(kList)-1) ###############
  #if any k values exceed the number of vertices in the associated community, set the k value to the community size
  kList = [min(kList[c], len(C[c])) for c in xrange(len(kList))]
  return selectColumns(M, C, kList)

#given an integer composition, a dimension, and an integer, determine the next integer composition
#the order is reverse lexigraphical
def nextComposition(last, dim, k):
  if last==-1: return [k]+[0 for _ in xrange(dim-1)]
  if last[-1]==k: return -1
  index = len(last)-2
  while last[index]==0: index -= 1
  return last[:index] + [last[index]-1, last[index+1 if last[-1]==0 else -1]+1] + [0 for _ in xrange(dim-index-2)]

#use a greedy approach to find k_i such that expected number of collisions is small
def greedyKList(C, P, alpha=-1):
  N = float(sum([len(C[c]) for c in C]))
  if alpha<=0: alpha = 1/(2.*N)
  kList = [0 for _ in C]
#  order = []
  while expectedCollisions(C, P, kList) > alpha:
    nextCom = getMinCom(C, P, kList)
#    order.append(nextCom)
    kList[nextCom] += 1
#  print('greedy k list order', order)
  return kList

#use a greedy approach to decrease k_i until decreasing any further will result in an expected 
#number of collisions greater than a threshold
def greedyDown(C, P, kList, alpha=-1):
  N = float(sum([len(C[c]) for c in C]))
  if alpha<=0: alpha = 1/(2.*N)
  downKList = kList[:]
  removeCom = 0
  while expectedCollisions(C, P, downKList) > alpha and removeCom!=-1:
    removeCom = remMinCom(C, P, downKList, alpha)
    downKList[removeCom] -= 1
  downKList[removeCom] += 1
  return downKList

#remove a column (decrement k_i) that minimally increases the expected number of collisions
def remMinCom(C, P, downKList, alpha):
  (minVal, minCom) = (np.inf, -1)
  for i in np.random.permutation(len(downKList)):
    if downKList[i]>0:
      incI = downKList[:]
      incI[i] -= 1
      val = expectedCollisions(C, P, incI)
      if val<minVal and val <= alpha: (minVal, minCom) = (val, i)
  return minCom

##############
### BOUNDS ###
##############
#determine a bound on the metric dimension of a graph generated via the sbm by 
#looking at erdos-renyi communities
def calcERBound(C, P, alpha=-1):
  N = float(sum([len(C[c]) for c in C]))
  if alpha<=0: alpha = lambda s: 1/(2.*s)
  terms = []
  for x in xrange(len(C)):
    num = np.log(2.*alpha(len(C[x])))-2*np.log(len(C[x]))
    col = np.argmin([np.abs(p-0.5) for p in P[x]])
    denom = np.log(P[x][col]*P[x][col]+(1-P[x][col])*(1-P[x][col]))
    terms.append((num/denom, col))
  return terms

#determine a bound on the metric dimension of a graph generated via the sbm assuming
#that columns are chosen randomly according to community size (the number of columns
#chosen from community i is proportional to that communities size)
def calcSBMBound(C, P, alpha=-1):
  N = float(sum([len(C[c]) for c in C]))
  if alpha<=0: alpha = 1/(2.*N)
  num = np.log(2.*alpha)-2*np.log(N)
  terms = []
  for (x,y) in combinations_with_replacement(xrange(len(C)), 2):
    T = reduce(lambda a,b: a*b, [np.power(P[x][j]*P[y][j]+(1-P[x][j])*(1-P[y][j]), len(C[j])/N) for j in xrange(len(C))])
    terms.append(T)
  denom = np.log(max(terms))
  return [(num / denom)*(len(C[c]) / N) for c in C]

#####################
### COLLECT STATS ###
#####################
#
def collectStats(network='polblogs', size=2000, repeats=1, procs=1, maxCollisions=False, ich=False):
  print('READ DATA')
  data = readDict(network+'_'+str(size)+'_stats.dict') if len(glob.glob(network+'_'+str(size)+'_stats.dict'))>0 else {}

  print('MAKE JOBS')
  jobs = []
  for fileName in glob.glob('SynthNetworks/'+network+'_'+str(size)+'_*.info'):
    if fileName not in data:
      (C, P, _, _, _, elapsed, _, _, diameter) = readInfo(fileName)
      data[fileName] = {'ICH adj res set':[], 'ICH adj time':[], \
                        'ICH dist res set':[], 'ICH dist time':[], \
                        'greedy adj res set':[], 'greedy adj time':[], \
                        'greedy dist res set':[], 'greedy dist time':[], \
                        'greedy bound adj res set':[], 'greedy bound adj time':[], \
                        'greedy bound dist res set':[], 'greedy bound dist time':[], \
                        'random adj res set':[], 'random adj time':[], \
                        'random dist res set':[], 'random dist time':[], \
                        'matrix update adj res set':[], 'matrix update adj time':[], \
                        'matrix update dist res set':[], 'matrix update dist time':[], \
                        'preorder adj res set':[], 'preorder adj time':[], \
                        'preorder dist res set':[], 'preorder dist time':[], \
                        'search com adj res set':[], 'search com adj time':[], \
                        'search com dist res set':[], 'search com dist time':[], \
                        'empirical adj res set':[], 'empirical adj time':[], \
                        'empirical dist res set':[], 'empirical dist time':[], \
                        'distance matrix time':elapsed, \
                        'diameter':diameter, \
                        'er bound':calcERBound(C, P), 'sbm bound':calcSBMBound(C, P)}
      for alpha in [0.2, 0.15, 0.1, 0.05, 0.01, 0.005]:
        data[fileName][str(alpha)+' algo adj res set'] = []
        data[fileName][str(alpha)+' algo adj time'] = []
#        data[fileName][str(alpha)+' algo dist res set'] = []
#        data[fileName][str(alpha)+' algo dist time'] = []
    #algo brute, algo, greedy bound, preorder, search com only need to be run once (res set size will not change between runs)
    if maxCollisions:
      for algo in [key for key in data[fileName] if 'time' in key and ('algo' in key or 'bound' in key or 'preorder' in key or 'search com' in key or 'empirical' in key) and ('dist' not in key or size<=5000)]:
        if len(data[fileName][algo])==0:
          jobs.append((fileName, algo))
    else:
      for algo in [key for key in data[fileName] if 'time' in key and 'algo' not in key and 'bound' not in key and 'preorder' not in key and 'search com' not in key and 'empirical' not in key and 'distance matrix' not in key and ('ICH' not in key or ich) and ('dist' not in key or size<=5000)]:
        for _ in (xrange(repeats-len(data[fileName][algo])) if 'ICH' not in algo or size<=7000 else xrange(10-len(data[fileName][algo]))): #added after first run... limit number of ich repeats... this will be 10 repeats total
          jobs.append((fileName, algo))

    #taken care of above
    #if size>5000: #do not consider the distance matrix (this will not be present in the info file)
    #  jobs = list(filter(lambda (fName, algo): 'dist' not in algo, jobs))

  print('MAKE POOL')
  pool = mp.Pool(processes=procs)
  results = pool.map_async(runJob, jobs)
  results = results.get()
  pool.close()
  pool.join()
#  results = [runJob(j) for j in jobs]

  print('UPDATE DATA')
  for (fileName, algo, R, t) in results:
    data[fileName][algo].append(t)
    data[fileName][algo.replace('time', 'res set')].append(R)
    #(_, _, _, _, _, elapsed, _, _, diameter) = readInfo(fileName)
    #data[fileName]['distance matrix time'].append(elapsed)
    #data[fileName]['diameter'].append(diameter)

  print('SAVE DATA')
  writeDict(data, network+'_'+str(size)+'_stats.dict')

#  print('PRINT DATA')
#  printData(data)

  print('DONE')

#
def runJob((fileName, algo)):
  np.random.seed()
  (C, P, E, M) = ({}, [], [], [])
  (R, t) = ([], -1)
  if 'dist' in algo: (C, P, E, _, M, _, _, _, _) = readInfo(fileName)
  else: (C, P, E, M, _, _, _, _, _) = readInfo(fileName)

  print('FILE', fileName)
  while t<=0:# or len(R)<=0:
    if algo[0]=='I':
      start = time.clock()
      R = approxMetricDim(M, randOrder=True, timeout=60*60*24) #24 hours
      if len(R)>1 and R[1]=='timeout': R = 'timeout'
      t = time.clock() - start
      print('   ich', len(R))
#      tempFile = (fileName.split('/')[-1]).split('.')[0]
#      tempFile = 'ICH_temp_'+tempFile+'.txt'
#      with open(tempFile, 'w') as o:
#        o.write(str(R)+'\n')
#        o.write(str(t))
    elif ('greedy' in algo) and ('bound' not in algo):
      start = time.clock()
      R = greedyResSet(M, C, P)
      t = time.clock() - start
      print('   greedy res set', len(R))
    elif ('greedy bound' in algo):
      start = time.clock()
      R = greedyBoundSet(M, C, P)
      t = time.clock() - start
      print('   greedy bound', len(R))
    elif algo[0]=='r':
      start = time.clock()
      R = randomResSet(M)
      t = time.clock() - start
      print('   random', len(R))
    elif algo[0]=='m':
      start = time.clock()
      R = greedyMatrixUpdate(M, C, P)
      t = time.clock() - start
      print('   matrix update', len(R))
    elif ('algo' in algo): #and ('brute' not in algo): #and (np.power(np.log(sum(len(C[c]) for c in C)), len(C)) < np.power(10,10)): #if log(n)^d is too big, dont run the algorithm (ie for polblogs with 17 communities)
      alpha = float(algo.split(' ')[0])
      start = time.clock()
      R = algoResSet(M, C, P, alpha=alpha)
      t = time.clock() - start
      print('   algo res set', alpha, len(R))
    elif 'preorder' in algo:
      start = time.clock()
      R = preorderComs(M, C, P)
      t = time.clock() - start
      print('   preorder res set', len(R))
    elif 'search com' in algo:
      start = time.clock()
      R = searchCom(M, C, P)
      t = time.clock() - start
      print('   search com res set', len(R))
    elif 'empirical' in algo:
      start = time.clock()
      R = empAdjProbs(M, C, P)
      t = time.clock() - start
      print('   empirical res set', len(R))
    else: print('   NONE', fileName, algo, (np.power(np.log(sum(len(C[c]) for c in C)), len(C)) < np.power(10,10)), (np.power(np.log(sum(len(C[c]) for c in C)), len(C)) < np.power(10, 10)))
    if len(R)==0: R = 'none found'
  return (fileName, algo, R, t)

############
### MISC ###
############
#given an edge list, make a networkx graph
def makeGraph(E):
  G = nx.Graph()
  G.add_edges_from(E)
  return G

#check whether or not a given set of columns resolves a given matrix
def checkResolving(R, M):
  tags = {}
  for row in xrange(len(M)):
    tag = ','.join([str(M[row][col]) for col in R])
    if tag in tags: return False
    tags[tag] = 1
  return True

#
def printData(data):
  for fileName in data:
    (C, P, E, A, D, elapsed, S, pVal, diameter) = readInfo(fileName)
    print('***************', fileName, '******************')
    print('Communities', [len(C[c]) for c in xrange(len(C))])
    print('Adjacency probabilities')
    for r in P: print(r)
    print('Nodes', sum(len(C[c]) for c in C), 'Edges', len(E))
    print('ER Bound', data[fileName]['er bound'])
    print('SBM Bound', data[fileName]['sbm bound'])
    for key in [k for k in data[fileName] if 'time' in k]:
      setKey = key.replace('time', 'res set')
      setLens = [len(r) for r in data[fileName][setKey]]
      print(setKey, np.mean(setLens), np.std(setLens))
      print(key, np.mean(data[fileName][key]), np.std(data[fileName][key]))

############
### MAIN ###
############
#
if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Gen Stats Options')
  parser.add_argument('--network', type=str, default='polblogs', required=False,
                      help='name of the network to consider. this is the prefix of .info files assciated with the network')
  parser.add_argument('--size', type=int, default=2000, required=False,
                      help='the approximate number of nodes in the networks to consider')
  parser.add_argument('--group', type=int, default=1, required=False,
                      help='an integer representing the group of algorithms to run')
  args = parser.parse_args()

  print('NETWORK: ', args.network, 'SIZE: ', args.size, 'GROUP: ', args.group)

  repeats = 50
  procs = 24

  if args.group==1:
    print('MAX COLLISIONS', False, 'ICH', False)
    collectStats(network=args.network, size=args.size, repeats=repeats, procs=procs, maxCollisions=False, ich=False)
  elif args.group==2:
    print('MAX COLLISIONS', True, 'ICH', False)
    collectStats(network=args.network, size=args.size, repeats=repeats, procs=procs, maxCollisions=True, ich=False)
  elif args.group==3:
    print('MAX COLLISIONS', False, 'ICH', True)
    collectStats(network=args.network, size=args.size, repeats=repeats, procs=procs, maxCollisions=False, ich=True)
  else: print('NONE RUN', 'GROUP: ', args.group)

  print('END')



