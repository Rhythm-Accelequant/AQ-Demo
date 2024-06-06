from headers_stable import *

def hamming_distance(x, y):
    """
    Calculate the Hamming distance between two strings.

    Parameters:
        x: First string.
        y: Second string.

    Returns:
        int: Hamming distance between the two strings.
    """
    
    x = np.array(list(x))
    y = np.array(list(y))
    
    return np.sum(x != y)

def hammer(p_in,n):
    
    """
    Update a probability distribution based on Hamming distances.

    Parameters:
        p_in (dict): Input probability distribution where keys are strings and values are probabilities.
        n (int): Length of strings in the input distribution.

    Returns:
        dict: Updated probability distribution.
    """
    
    # Step-1: Create Hamming Spectrum
    """
    1) Iterate through all pairs of strings in the input probability distribution.
    2) Calculate the Hamming distance between each pair of strings.
    3) Accumulate the probabilities of strings that have a Hamming distance within a certain range (n // 2) into an array chs. 
        
    """
    chs = np.zeros(n//2)
    for x in p_in:
        for y in p_in:
            d = hamming_distance(x, y)
            if d < n//2:
                chs[d] += p_in[y]
    
    # Step-2: Compute Per-Distance Weights
    """ 
    1) Iterate through the Hamming distances computed in the previous step.
    2) If the count of strings at a particular Hamming distance is greater than zero, computes and stores(w) the weight as the reciprocal of that count.
    
    """
    w = np.zeros(n//2)
    for d in range(n//2):
        if chs[d] > 0:
            w[d] = 1 / chs[d]
    
    # Step-3: Update the Probability Distribution
    """
    1) Updates each string's probability by considering its similarity to other strings within a certain Hamming distance.
    2) It iterates through each string, adjusting its probability based on the probabilities of similar strings weighted by their Hamming distance. 
    3) Finally, it stores the updated probabilities in a dictionary
    
    """
    p_out = {}
    for x in p_in:
        score = p_in[x]
        for y in p_in:
            d = hamming_distance(x, y)
            if d < n//2 and p_in[x] > p_in[y]:
                score += w[d] * p_in[y]
        p_out[x] = score * p_in[x]
    
    # Normalize the probability distribution
    total = sum(p_out.values())
    p_out = {key: value / total for key, value in p_out.items()}
    
    return p_out