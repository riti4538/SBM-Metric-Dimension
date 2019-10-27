#python code to extract distance time information from .info files

import numpy as np
import glob

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

def printData(data, networks, size=2000):
  times = {network:[] for network in networks}
  for network in networks:
    times[network].extend([data[f]['floyd'] for f in glob.glob('SynthNetworks/*.info') if network in f and int(f.split('_')[-2])==size])

  print('len', len(times), [len(times[network]) for network in networks])

  print('\\begin{table}[h]\n\\centering\n\\begin{tabular}{|'+('c|'*(len(networks)+1))+'}\n\\hline')
  print('Network & Distance Matrix Construction Time \\\\ \\hline')
  for network in networks: print(network.replace('_', '\_')+' & '+r'$'+str(round(np.mean(times[network]), 2))+' \pm '+str(round(np.std(times[network]), 2))+'$'+' \\\\ \\hline')
  print('\\end{tabular}\n\\caption[LoF entry]{Time required to generate full distance information.}\n\\label{tab:dist_mat_times}\n\\end{table}')





if __name__=='__main__':
  size = 5000
  data = {}
  for f in glob.glob('SynthNetworks/*.info'):
    if int(f.split('_')[-2])==size:
      print('**********', f, '***********')
      (_, _, _, _, _, elapsed, S, pVal, diameter) = readInfo(f)
      data[f] = {'floyd': elapsed[0], 'total': elapsed[1]}

  networks = ['polbooks', 'sp_data_school_gender', 'polblogs', 'karate', 'adjnoun'] #[(f.split('/')[-1]).split('.')[0] for f in glob.glob('RawData/*.raw') if 'football' not in f]
  printData(data, networks, size=size)


