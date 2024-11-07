
import numpy as np
import pandas as pd


def convert_to_bits(data):

    return (data > 0).astype(int)

def border_crossings(data):
    T, dim_x=data.shape
    crossings=np.zeros(T-1)
    for i in range(T-1):
        crossing=0
        for j in range(dim_x):
            if data[i+1,j]!=data[i,j]:
                crossing+=1
            
        crossings[i]=crossing
        
    
    return crossings

def total_flips_per_dim(bits):
    
    T,dx=bits.shape
    flips=np.zeros(dx)
    for i in range(T-1):
        for j in range(dx):
            if bits[i+1,j]!=bits[i,j]:
                        flips[j]+=1
    return flips
    

def minimal_regions(bit_flips, total_regions):
    
    indices=[]
    flips=np.copy(bit_flips)
    for i in range(total_regions):
        index=np.argmin(flips, axis=None, out=None)
        flips[index]=100000
        indices.append(index)

    return indices

def relative_frequencies(bits):
    frequencies=np.mean(bits,axis=0)
    
    return frequencies


def unique_regions_crossed(data, dz):
    unique_regions_set = set()
    unique_regions_list = []  # Use a list to collect regions if needed later

    for step in data[:-1]:  # Exclude the last step as it does not lead to a new comparison
        region_tuple = tuple(step)
        if region_tuple not in unique_regions_set:
            unique_regions_set.add(region_tuple)
            unique_regions_list.append(step)  # Optional, in case you need the regions as a Numpy array

    regions_count = len(unique_regions_set)
   # print("Number of unique linear regions:", regions_count)
    
    # Assuming dz is defined elsewhere in your code. You might want to pass it as a function argument.
    total_regions = 2**dz
   # print("Number of total linear regions:", total_regions)

    # Convert list to Numpy array if necessary. Otherwise, just return the count.
    unique_regions_array = np.array(unique_regions_list)
    
    return regions_count, unique_regions_array


def boundary_crossings_optimized(data):
    # Detect where a change occurs between consecutive steps
    changes = np.any(data[:-1] != data[1:], axis=1)
    # Indices where changes occur
    change_indices = np.where(changes)[0]
    # Selecting the unique transitions
    crossings = data[change_indices]
    return crossings

def connectome_with_self_connections(bits, mf_regions, N):
    # Map each unique region to an index
    region_to_index = {tuple(mf_regions[i]): i for i in range(N)}

    conn = np.zeros((N, N))
    
    # Include self-connections by allowing for unchanged states
    bits = boundary_crossings_optimized(bits)
    bits_tuples = [tuple(bit) for bit in bits]

    for i in range(len(bits_tuples) - 1):
        if bits_tuples[i] in region_to_index and bits_tuples[i+1] in region_to_index:
            j = region_to_index[bits_tuples[i]]
            k = region_to_index[bits_tuples[i+1]]
            conn[j, k] += 1

    # Add self-connections for consecutive identical states
    for i in range(len(bits_tuples)):
        if bits_tuples[i] in region_to_index:
            j = region_to_index[bits_tuples[i]]
            conn[j, j] += 1  # Increment self-connection

    # Normalize the connectome
    conn /= np.sum(conn)

    return conn



def frequency_of_regions(data, regions):
    # Convert regions to a list of tuples for efficient comparison
    regions_tuples = [tuple(region) for region in regions]
    data_tuples = [tuple(point) for point in data]
    
    # Use a Pandas Series to leverage its value_counts method for frequency counting
    frequencies_series = pd.Series(data_tuples).value_counts().reindex(regions_tuples, fill_value=0)
    
    # The frequencies are aligned with the regions order because of reindex
    frequencies = frequencies_series.values
    
    return frequencies