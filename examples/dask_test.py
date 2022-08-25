

# Testing DASK


import dask.dataframe as dd


df = dd.read_csv('UKIDSS_sources.csv', blocksize=1000)



for partition in df.partitions:
    print(partition.compute())