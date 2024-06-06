from headers_stable import *

def Dataset_generator(Np, Ns):
        
        random.seed(1)
        # Initializing arrays for coefficients
        C = np.zeros((Np, Ns, Ns))  # Coefficients matrix
        f1 = np.zeros((Np))         # Array for random factors
        f2 = np.zeros((Ns, Ns))     # Array for random factors

        # Generating random factors f1 for each part
        for a in range(2, Np + 1):
            f1[a - 1] = random.randrange(50, 1000) / 1000

        # Generating random factors f2 for each site pair
        for i in range(1, Ns):
            for j in range(i + 1, Ns + 1):
                f2[i - 1, j - 1] = random.randrange(100, 1000) / 100

        # Calculating coefficients based on random factors
        for a in range(2, Np + 1):
            for i in range(1, Ns):
                for j in range(i + 1, Ns + 1):
                    C[a - 1, i - 1, j - 1] = round(f1[a - 1] * f2[i - 1, j - 1], 2)

        # Writing coefficients to a CSV file
        header = ['Part','site_i','site_j','Cost']
        folder_path = "./IFolder_" + str(Np).zfill(2) + '_' + str(Ns).zfill(2)
        file_path = '/cost_' + str(Np).zfill(2) + '_' + str(Ns).zfill(2) + '.csv'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            # print('Creating folder' + folder_path)
        if not os.path.exists(folder_path + file_path):
            with open(folder_path + file_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for a in range(2, Np + 1):
                    for i in range(1, Ns):
                        for j in range(i + 1, Ns + 1):
                            data = [a - 1, i - 1, j - 1, C[a - 1, i - 1, j - 1]]
                            writer.writerow(data)

        df_cost = pd.read_csv(folder_path + file_path)
        return df_cost                    
        
        
def data_generator(nparts,nsites):
    """
    Generates coefficients representing transportation costs between parts and sites,
    then writes them to a CSV file.

    Args:
    - M (int): Number of different parts in the PBS
    - N (int): Number of different sites

    Returns:
    Dataframe consisting of coefficients representing transportation costs between parts and sites
    """
    return Dataset_generator(nparts,nsites)


# def cost(part, site_i,site_j,df_cost):
#     """
#     Generates cost.

#     """
#     # 0-INDEXED ARGUMENTS
#     # Filter the DataFrame based on values from the first three columns
#     filtered_df = df_cost[(df_cost['Part'] == part) & 
#                     (df_cost['site_i'] == site_i) & 
#                     (df_cost['site_j'] == site_j)]
    
#     # Extract the value from the fourth column
#     result = filtered_df['Cost'].iloc[0] if not filtered_df.empty else 0
#     return result

def generate_breakdown_structure(n):
    ''' Returns a dictionary of of the format {Part :[List of Sub-parts]} where the final product is 
        Product 0 
        Example : For 6 parts : {0: [1, 2, 3], 1: [4, 5]} : Final Part 0 is made up of sub-parts 1, 2 
        and 3, where sub-part 1 is made of sub-parts 4 and 5.'''
    breakdown_structure = {}
    used_parts = set()  # Keep track of parts that have been used as subparts
    for i in range(n):
        subparts = []
        for j in range((i + 1) * 2, min((i + 1) * 2 + 3, n + 1)):
            if j - 1 not in used_parts:  # Check if j has already been used as a subpart
                subparts.append(j - 1)
                used_parts.add(j - 1)
        if subparts:  # Check if the subparts list is non-empty
            breakdown_structure[i] = subparts
    return breakdown_structure

def bin2dec(lst):
    ''' 
    :lst: Input binary string 
    :type lst: list

    Returns: The decimal expansion of the binary string
    '''
    n = len(lst)
    sum = 0
    for i in range(n):
        sum += lst[i] * 2**(n-i-1)
    return int(sum)

def rand_binary(l):
    ''' Returns: A random binary string (list type) of length l'''
    return [random.choice([0,1]) for _ in range(l)]

def part2cut(part):
    ''' Returns a matrix such that (i,j)th element specifies if edge(i,j) CAN make a cut '''
    n = len(part)
    cut = np.zeros((n,n))
    for i, j in list(itertools.product(range(n), range(n))):
        cut[i,j] = part[i] ^ part[j] 

    return cut

def maxcut(x,n, sp = 0.1, seed = 18):
    ''' 
    A Membership-Query function that returns the total weight for the given partition
    :x: partition list in 0-1 basis; tells us that the i^th vertex belongs to which partition
    :sp: Sparsity of the graph : 0 <= sp <= 1: Set to 0.1
    :d: Degree (for Regular Graph) : Set to 4n/5
    '''
    d = (4*n)//5
    np.random.seed(seed)
    G = nx.random_regular_graph(d, n, seed = seed)
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = np.random.randint(1,10)
    w = nx.to_numpy_array(G)
    
    total_weight = np.sum(np.multiply(w,part2cut(x)))
    return total_weight

def generate_maxcut_QUBO(n, seed = 10):
    d = 4*((n)//5)
    np.random.seed(seed)
    G = nx.random_regular_graph(d, n, seed = seed)
    # G = nx.erdos_renyi_graph(n, 0.8, seed = seed)
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = np.random.randint(1,100)
    w = nx.to_numpy_array(G)
    Q = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            Q[i,i] += w[i,j]
            Q[i,j] += -w[i,j]

    return Q,w

def max_sol_f(f,n):
    max_cost = float('-inf')
    sol = None
    for i in [list(i) for i in list(product([0, 1], repeat=n))]:
        cost = f(i)
        if cost > max_cost:
            max_cost = cost
            sol = i
        
    return sol, max_cost

def min_sol_f(f,n):
    min_cost = float('inf')
    sol = None
    for i in [list(i) for i in list(product([0, 1], repeat=n))]:
        cost = f(i)
        if cost < min_cost:
            min_cost = cost
            sol = i
        
    return sol, min_cost

def mat_to_terms(matrix):
    ''' Converts a matrix representing coefficients of quadratic terms (and diagonal elements 
    representing coefficients of linear terms) into a dictionary of format {term : coefficient} 
    :term: eg. (1,0,0,1,0) = x0.x3
    Since Q_ij and Q_ji correspond to the same term, corresponding value in the dictionary 
    becomes Q_ij + Q_ji'''
    n = matrix.shape[0]
    result_dict = {}
    for i in range(n):
        # print(f'`mat_to_terms()` : {i+1}/{n} Completed ...')
        for j in range(i, n):  # Start from i to avoid duplicate terms
            key = [0] * n
            key[i] = 1
            key[j] = 1
            key = tuple(key)
            result_dict[key] = result_dict.get(key, 0) + matrix[i][j]  # Use get to handle missing keys
            if i != j:  # Avoid duplicate addition for diagonal elements
                result_dict[key] += matrix[j][i]

    return result_dict

#this function is to calculate number of parameters in the ansatz
#It takes arguments as "num_qubits" ---> number of qubits ; "num_layers" ----> number of layers

def param_count(num_qubits,num_layers): 
        num=num_layers%6
        
        if num_qubits%2==0:
            k=3*num_qubits -3 + 9*(num_qubits//2)
           
            if num==1:
                
                num_params = (num_layers//6)*k + num_qubits
            if num==2:
        
                num_params = (num_layers//6)*k + num_qubits + 3*(num_qubits//2)
            if num==3:
                
                num_params = (num_layers//6)*k + 2*num_qubits + 3*(num_qubits//2)
            if num==4:
                
                num_params = (num_layers//6)*k +2*num_qubits + 6*(num_qubits//2) -3
            if num==5:
                
                num_params = (num_layers//6)*k + 3*num_qubits + 6*(num_qubits//2) -3
            if num==0:
                
                num_params = (num_layers//6)*k + 3*num_qubits + 9*(num_qubits//2) -3
        
        else:
            k=3*num_qubits + 9*(num_qubits//2)
            
            if num==1:
                
                num_params = (num_layers//6)*k + num_qubits
            if num==2:
                
                num_params = (num_layers//6)*k + num_qubits + 3*(num_qubits//2)
            if num==3:
                
                num_params = (num_layers//6)*k + 2*num_qubits + 3*(num_qubits//2)
            if num==4:
                
                num_params = (num_layers//6)*k + 2*num_qubits + 6*(num_qubits//2) 
            if num==5:
                
                num_params = (num_layers//6)*k + 3*num_qubits + 6*(num_qubits//2)
            if num==0:
                
                num_params = (num_layers//6)*k + 3*num_qubits + 9*(num_qubits//2)

        return num_params

def MS_gate(theta1, theta2, theta3): #this function is to build MS gate
    # Define the gate matrix
    gate_matrix = np.array([
        [1, 0, 0, 0],
        [0, np.exp(1j*theta1), 0, 0],
        [0, 0, np.exp(1j*theta2), 0],
        [0, 0, 0, np.exp(1j*theta3)]
    ], dtype=np.complex128)
    
    # Convert the matrix to a Qiskit Operator
    ms_operator = Operator(gate_matrix)
    
    return ms_operator


def add_layer(circuit, num, r, num_qubits, params):
    if num == 0:
        for i in range(num_qubits):
            circuit.rx(params[r], i)
            r = r + 1
    if num == 1:
        for qubit in range(0, num_qubits-1, 2):
            ms_gate = MS_gate(params[r], params[r+1], params[r+2])
            circuit.unitary(ms_gate, [qubit, qubit+1], label='MS')
            #circuit.rxx(params[r], qubit, qubit+1)
            r = r + 3
    if num == 2:
        for i in range(num_qubits):
            circuit.ry(params[r], i)
            r = r + 1
    if num == 3:
        for qubit in range(1, num_qubits-1, 2):
            ms_gate = MS_gate(params[r], params[r+1], params[r+2])
            circuit.unitary(ms_gate, [qubit, qubit+1], label='MS')
            #circuit.rxx(params[r], qubit, qubit+1)
            r = r + 3
    if num == 4:
        for i in range(num_qubits):
            circuit.rz(params[r], i)
            r = r + 1
    if num == 5:
        for qubit in range(0, num_qubits-1, 2):
            ms_gate = MS_gate(params[r], params[r+1], params[r+2])
            circuit.unitary(ms_gate, [qubit, qubit+1], label='MS')
            #circuit.rxx(params[r], qubit, qubit+1)
            r = r + 3
    return r  # Return the updated value of r


def ansatz_circuit(num_qubits, num_layers, params):

    circuit = QuantumCircuit(num_qubits)
    layer = 0
    r = 0
    while layer < num_layers:
        r = add_layer(circuit, layer % 6, r,num_qubits, params)  # Assign the updated r value from add_layer
        circuit.barrier()
        layer += 1

    return circuit 



def generate_pauli_strings(v,n, k): #this function generates all possible (n choose k) pauli strings for Z containing pauli strings
    """
    Generate all possible traceless Pauli strings of n-fold tensor products of identity (I), Pauli Z,
    where only Z operates on k bits and I operates on n-k remaining bits.
    
    Parameters:
        v (int): Total number of variables in the input.
        n (int): Total number of qubits.
        k (int): Number of qubits on which Pauli Z operates.
    
    Returns:
        list: A list of all possible traceless Pauli strings.
    """
    if k > n:
        print("Value of k is greater than n")
        return []
    
    # Generate all combinations of positions where Pauli X will operate
    z_positions_combinations = combinations(range(n), k)
    
    pauli_strings = []
    
    for positions in z_positions_combinations:
        pauli_string = ['I'] * n  # Initialize Pauli string with identity (I) on all qubits
        
        # Set Pauli Z on the specified positions
        for pos in positions:
            pauli_string[pos] = 'Z'
        
        pauli_strings.append(''.join(pauli_string))
    
    p = len(pauli_strings)
    l = p
    ind = 0

    all_paulis = pauli_strings
    while l < v:
        if ind < p:
            ref = pauli_strings[ind].replace('Z','X')
        else:
            ref = pauli_strings[ind-p].replace('Z','Y')
        all_paulis.append(ref)
        ind += 1
        l += 1

    return all_paulis


def exp_val(params, all_paulis, circuit): #this function returns an array with all the expectation values on all pauli strings for a state 
    epv = [] #an array which has all expectation values on respective paulistrings
    
    est = Estimator()
    # circuit = ansatz_circuit(n, n_l, params)
    count = len(all_paulis)
    # print(count)
    # print(len(params))
    # print(len(circuit.parameters))
    expectations = est.run([circuit]*count,all_paulis,[params]*count).result().values
    # expectations = []
    return expectations

# def new_exp_val(all_paulis, circuit):
#     epv = [] #an array which has all expectation values on respective paulistrings
    
#     est = Estimator()
#     # circuit = ansatz_circuit(n, n_l, params)
#     count = len(all_paulis)
#     expectations = est.run([circuit]*count,all_paulis).result().values
#     # expectations = []
#     return expectations    


def out(epv, m): #returns the output of every variable in 0's and 1's as the obj func made from QUBO is in varibles which takes 0,1 
    xo = [] #output binary assignment after optimization
    #it is nothing but sign function of respective expectations
# m is number of varibles
#here negative is mapped to 1 because we made (1 - tanh(epv[i])) which is 1 when epv[i] is negative and 0 if epv[i] is positive
    for i in range(m):
        if epv[i] < 0:
            xo.append(1)
        if epv[i] >= 0:
            xo.append(0)
    return xo

#this function gives us the preliminary output bitstring in 0 and 1 just after optimization, but before post-processing step

def check_local(x, ans, Q):
    x = np.array(x)
    temp = x.T @ Q @ x
    if(temp < ans):
        return temp
    else:
        return ans



def post_pro(xo, Q):
    xo = np.array(xo)
    c_out = xo.copy() #this c_out is used as a temporary array while swapping and checking
    c_ans = xo.T @ Q @ xo
    for i in range(len(c_out)):

        if(c_out[i] == 1):
            if(i-1 >= 0):
                c_out[i-1], c_out[i] = c_out[i], c_out[i-1]
                
        
                if (check_local(c_out, c_ans, Q) != c_ans):  
                    temp_ans = check_local(c_out, c_ans, Q)
                    c_ans = temp_ans #this "c_ans" now carries updated answer for the question
                    xo = c_out #this "xo" will array the updated array or will remain the same which was obtained previously
                else:
                    c_out[i-1], c_out[i] = c_out[i], c_out[i-1]

            if(i+1 < len(c_out)):
                c_out[i], c_out[i+1] = c_out[i+1], c_out[i]
                
        
                if check_local(c_out, c_ans, Q) != c_ans:  
                    temp_ans = check_local(c_out, c_ans, Q)
                    c_ans = temp_ans #this "c_ans" now carries updated answer for the question
                    xo = c_out #this "xo" will array the updated array or will remain the same which was obtained previously
                else:
                    c_out[i], c_out[i+1] = c_out[i+1], c_out[i]
        

    return xo,c_ans

# def post_pro(x0, Q):
#     x0 = np.array(x0)
#     x_sol = x0.copy()
#     for i in range(len(x0)):
#         if x_sol[i] == 1:
#             if i == 0:
#                 x_temp = x_sol.copy()
#                 x_sol[i], x_sol[i+1] = x_sol[i+1], x_sol[i]
#                 if x_sol.T @ Q @ x_sol > x_temp.T @ Q @ x_temp:
#                     x_sol = x_temp

#             if i == len(x0) - 1:
#                 x_temp = x_sol.copy()
#                 x_sol[i-1], x_sol[i] = x_sol[i], x_sol[i-1]
#                 if x_sol.T @ Q @ x_sol > x_temp.T @ Q @ x_temp:
#                     x_sol = x_temp
#             else:
#                 x_temp = x_sol.copy()
#                 x_sol[i-1], x_sol[i] = x_sol[i], x_sol[i-1]
#                 if x_sol.T @ Q @ x_sol > x_temp.T @ Q @ x_temp:
#                     x_sol = x_temp
#                 x_temp = x_sol.copy()
#                 x_sol[i], x_sol[i+1] = x_sol[i+1], x_sol[i]
#                 if x_sol.T @ Q @ x_sol > x_temp.T @ Q @ x_temp:
#                     x_sol = x_temp
#     return x_sol, x_sol.T @ Q @ x_sol

def loss_func(params, pauli_strings, al, Q, circuit,offset): #this is the loss function obtained from the QUBO
    epv = exp_val(params, pauli_strings, circuit)
    x = []  # Initialize x as an empty list
    for val in epv:
        # Append values to x using the append() method
        x.append((1 - (math.tanh(val*al)))/2) # to use this in QUBO the varibles are being converted into 0 or 1 using tangent hyperbolic
    # Convert x to a numpy array for matrix multiplication
    x = np.array(x)
    x,ans = post_pro(x,Q)
    # Compute the loss function
    #return np.dot(np.dot(x.T, Q), x)
    #print(x)
    return ans+offset

# def new_loss_func(params, pauli_strings, al, Q, circuit,offset): #this is the loss function obtained from the QUBO
#     epv = exp_val(params, pauli_strings, circuit)
#     x = []  # Initialize x as an empty list
#     for val in epv:
#         # Append values to x using the append() method
#         x.append((1 - (math.tanh(val*al)))/2) # to use this in QUBO the varibles are being converted into 0 or 1 using tangent hyperbolic
#     # Convert x to a numpy array for matrix multiplication
#     x = np.array(x)
#     x,ans = post_pro(x,Q)
#     # Compute the loss function
#     #return np.dot(np.dot(x.T, Q), x)
#     #print(x)
#     return ans+offset
