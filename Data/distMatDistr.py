#python code to count the number of entries of different values in distance (or adjacency) matrices

import numpy as np
import multiprocessing as mp
import glob

def readInfo(fileName):
  (C, P, E, A, D, elapsed, S, pVal, diameter) = ({}, [], [], [], [], -1, -1, -1, -1)
  with open(fileName, 'r') as f:
    sect = 0
    for line in f:
      if line[0]=='#': sect += 1
#      elif sect==1: #test stat info
#        l = line.strip().split(':')
#        S = float(l[1].split(',')[0])
#        pVal = float(l[1].split(',')[1])
#      elif sect==2: #get diameter
#        l = line.strip().split(':')
#        diameter = int(l[1])
#      elif sect==3: #communities
#        l = line.strip().split(':')
#        C[int(l[0])] = map(int, l[1].split(','))
#      elif sect==4: #params
#        l = eval(line.strip())
#        P.append(l)
#      elif sect==5: #edge list
#        l = line.strip().split('\t')
#        E.append((int(l[0]), int(l[1])))
#      elif sect==6: #adjacency matrix
#        l = eval(line.strip().replace('inf', 'np.inf'))
#        A.append(l)
      elif sect==7: #distance matrix
        l = eval(line.strip().replace('inf', 'np.inf'))
        D.append(l)
#      elif sect==8: #distance matrix time
#        l = float(line.strip()) if ',' not in line else tuple([float(x) for x in line.strip().split(',')])
#        elapsed = l
  return (C, P, E, A, D, elapsed, S, pVal, diameter)

def pathLengthCounts(D):
  counts = {}
  for row in xrange(len(D)):
    if row%200==0 and row>0: print(row/float(len(D)))
    for col in xrange(row, len(D[row])):
      val = D[row][col]
      counts[val] = counts.get(val, 0) + 1
  return counts

def runJob(f):
  (_, _, _, _, D, _, _, _, _) = readInfo(f)
  counts = pathLengthCounts(D)
  return (f, counts)

if __name__=='__main__':
  size = 5000
  data = {}

  jobs = [f for f in glob.glob('SynthNetworks/*.info') if int(f.split('_')[-2])==size]
  pool = mp.Pool(processes=16)
  results = pool.map_async(runJob, jobs)
  results = results.get()
  pool.close()
  pool.join()

  for (f, counts) in results:
    data[f] = counts

#  for f in glob.glob('SynthNetworks/*.info'):
#    if int(f.split('_')[-2])==size:
#      print('********', f, '**********')
#      (_, _, _, _, D, _, _, _, _) = readInfo(f)
#      data[f] = pathLengthCounts(D)

  networks = ['polbooks', 'sp_data_school_gender', 'polblogs', 'karate', 'adjnoun']
  distr = {}
  for network in networks:
    print('*********', network, '***********')
    for f in data:
      if network in f:
        for val in data[f]:
          distr[val] = distr.get(val, 0)+data[f][val]
    total = float(sum(distr.values()))
    print('total', total, np.sqrt((total*2-size)/30.))
    distr = {val:distr[val]/total for val in distr}
    print('distr', distr)
     


###########################
#size: 5000
#('*********', 'polbooks', '***********')
#('total', 375075000.0, 5000.4833099744801)
#('distr', {0: 0.0003999200159968006, 1: 0.08156445244284477, 2: 0.9168743051389722, 3: 0.0011613224021862294})
#('*********', 'sp_data_school_gender', '***********')
#('total', 375225031.0, 5001.4833199761842)
#('distr', {0: 0.0003998400639746233, 1: 0.21271149973351447, 2: 0.7868886601994158, 3: 3.0950024818206476e-12})
#('*********', 'polblogs', '***********')
#('total', 375075001.0, 5000.4833166405024)
#('distr', {0: 0.00039992001599658747, 1: 0.02249825019053056, 2: 0.7606804453141591, 3: 0.21642138447931378})
#('*********', 'karate', '***********')
#('total', 375075001.0, 5000.4833166405024)
#('distr', {0: 0.0003999200159968006, 1: 0.14028941913539647, 2: 0.8593106602715984, 3: 5.770082887483983e-10})
#('*********', 'adjnoun', '***********')
#('total', 375075000.99999994, 5000.4833166405024)
#('distr', {0: 0.0003999200159968007, 1: 0.06806792260806906, 2: 0.9315321573759342, 3: 1.5383810896754442e-18})

 
    
