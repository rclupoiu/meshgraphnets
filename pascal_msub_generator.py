import os
import subprocess
from subprocess import call

################# Utility functions

def WriteMsub(seq, bank, cluster, cores, time, arg_line, output_folder_name):
    seq = '{:03}'.format(seq)
    solverPath = 'python ./train.py '
    optNode = 1 # we are running on single GPU
    fileName = "Msub_" + str(seq) + '.msub'
    out_line = '> ./{}/main_{}.out'.format(output_folder_name[0], seq)
    
    with open( fileName , 'w' ) as myfile:
        myfile.write("#!/bin/csh" + '\n')
        myfile.write("#MSUB -A " + bank+ '\n')
        myfile.write("#MSUB -l nodes=" + str(optNode) +'\n')
        myfile.write('#MSUB -l partition=' + cluster + '\n')
        myfile.write('#MSUB -l walltime=' + str(time) + ':00:00' + '\n')
        myfile.write('#MSUB -q pbatch' + '\n')
        myfile.write('#MSUB -m be' + '\n')
        myfile.write('#MSUB -V' + '\n')
        myfile.write('#MSUB -o ./{}/Log_{}'.format(output_folder_name[1], seq) + '.out' + '\n')
        myfile.write('\n')
        myfile.write('##### These are shell commands' + '\n')
        myfile.write('date'+ '\n')
        myfile.write( solverPath + arg_line + out_line +'\n')
        myfile.write('date'+ '\n')
    return fileName


def GenerateArgLines(num_layer, hidden_dim, train_size, test_size):
    return '--num_layers {} --hidden_dim {} --train_size {} --test_size {} '.format(num_layer, hidden_dim,
                                                                                   train_size, test_size)
                                                                                   
                                                                                   
#env_path = '/usr/workspace/ju1/Python_tf/virtual/meshgraphnet/start_meshgraphnet.sh'
# TO DO: checking the virtual env 
#change_env_line = 'source env_path'
#call( msubLine, shell=True)

clusterName = 'pascal' # this can be changed later
coreNum = 1             # 1GPU
currentPath = os.getcwd()
bank = 'cbronze'
train_time = 2 # 2hr wall time

num_layers_list =[ 10,12,15]
hidden_dim_list = [ 10,20,30,50]
train_size_list =[45, 85, 160, 420, 500]
test_size_list =[5, 15, 40, 80, 100]

count = 0

main_out_path = os.path.join(currentPath, 'main_out')
training_log_path = os.path.join(currentPath, 'train_out')

if not os.path.isdir( main_out_path ):
    print('Creating dir {}'.format(main_out_path))
    os.mkdir(main_out_path)

if not os.path.isdir( training_log_path ):
    print('Creating dir {}'.format(training_log_path))
    os.mkdir(training_log_path)
    
output_folder_name = ['main_out', 'train_out']

for num_layer in num_layers_list:
    for hidden_dim in hidden_dim_list:
        for (train_size, test_size) in zip( train_size_list, test_size_list):
            
            args_line = GenerateArgLines(num_layer, hidden_dim,
                                         train_size, test_size)
            
            fileName = WriteMsub(count, bank, clusterName, 
                                 coreNum, train_time, args_line, output_folder_name)
            
            msubLine = 'msub ' + fileName
            call( msubLine, shell=True)
            os.chdir(currentPath)
            count += 1
# System call run msub file on cluster 
    



