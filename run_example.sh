# ----------------------------------------------------------------------
#  Parameters
# ----------------------------------------------------------------------
dname=Music    #Dataset name 
n=1000000      #Data size
d=100          #Dimension
k=10          #Top-k
training_sample=100000   #No. training samples
q=100          #No. queries

delta=0.3      #For different tradeoffs, generally smaller than 0.5 
l0=5           #Related to error rates
flag=0         #0 for approximate search and 1 for guarantees on recall rates 

dPath=./data/${dname}/${dname}.ds   #Path of dataset
qPath=./data/${dname}/${dname}.q    #Path of query set 
gtPath=./data/${dname}/${dname}.gt  #Path of ground_truth set
indexPath=./data/${dname}/${dname}.bin  #Path of index

#------------------------------------------------------------------------
# Indexing
#-----------------------------------------------------------------------
./build/main 1 ${n} ${q} ${training_sample} ${d} ${k} ${delta} ${l0} ${flag} ${dPath} ${qPath} ${gtPath} ${indexPath}


#-----------------------------------------------------------------------
# Searching
#-----------------------------------------------------------------------
./build/main 2 ${n} ${q} ${training_sample} ${d} ${k} ${delta} ${l0} ${flag} ${dPath} ${qPath} ${gtPath} ${indexPath}

