# %%
from headers_stable import *
from utils_stable import *

# %% [markdown]
# # Module 1
# ## Data Generation and QUBO Definition

# %%
nparts = int(sys.argv[1]) # PARTS
nsites = int(sys.argv[2]) # SITES 
# nparts PARTS ; nsites SITES

# %%
df_cost = data_generator(nparts,nsites)

# %%
def cost(part, site_i, site_j):
    # 0-INDEXED ARGUMENTS
    # Filter the DataFrame based on values from the first three columns
    filtered_df = df_cost[(df_cost['Part'] == part) & 
                    (df_cost['site_i'] == site_i) & 
                    (df_cost['site_j'] == site_j)]
        
    # Extract the value from the fourth column
    result = filtered_df['Cost'].iloc[0] if not filtered_df.empty else 0
    return result

# %%
part_subpart = generate_breakdown_structure(nparts)

# %%
phi = []
for product, parts in part_subpart.items():
    phi.extend([(part, product) for part in parts])

psi = [(part_i, part_j) for parts in part_subpart.values() for i, part_i in enumerate(parts) for part_j in parts[i+1:]]


# %%
init_model = gp.Model('gp_Model_'+ str(nparts).zfill(2) + '_' + str(nsites).zfill(2))

x = init_model.addVars(nparts,nsites, name=[f'x_{r}_{i}' for r in range(nparts) for i in range(nsites)], vtype=GRB.BINARY)

# Objective function
objective_expr = sum(cost(r,i,j) * (x[r, i] * x[s, j] + x[r, j] * x[s, i])
                                        for i in range(nsites) for j in range(i,nsites) for (r,s) in phi)

init_model.setObjective(objective_expr, GRB.MINIMIZE)
init_model.update()
# Constraint 1
init_model.addConstrs((sum(x[r,i] for i in range(nsites)) == 1 for r in range(nparts)), name='C1')
init_model.update()
# <gurobi.Constr C1[r]> ==> Constraint 1 corresponding to Part r
# Constraint 2
init_model.addQConstr(sum(x[r, i] * x[s, i] for i in range(nsites) for (r, s) in phi) == 0, name='C2')
# Because phi is 1-indexed (just to agree with ABQC doc) but x[r,i] indices are 0-indexed
init_model.update()
# <gurobi.QConstr C2> ==> Constraint 2 as one single summation but mathematically this corresponds to 
# x_r_i * x_s_i == 0 : For all (r,s) in phi and for all Sites i
# Constraint 3
init_model.addQConstr(sum(x[r, i] * x[s, i] for i in range(nsites) for (r, s) in psi) == 0, name='C3')
# Because psi is 1-indexed (just to agree with ABQC doc) but x[r,i] indices are 0-indexed
init_model.update()
# <gurobi.QConstr C3> ==> Constraint 3 as one single summation but mathematically this corresponds to 
# x_r_i * x_s_i == 0 : For all (r,s) in psi and for all Sites i

init_model.optimize()
# print(dir(init_model))

my_model = gp.Model('my_gp_Model_'+ str(nparts).zfill(2) + '_' + str(nsites).zfill(2))

xvars = my_model.addVars(nparts,nsites, name=[f'x_{r}_{i}' for r in range(nparts) for i in range(nsites)], vtype=GRB.BINARY)

# %%
lamda = 1000*len(phi) * max(df_cost['Cost'])
#lamda = 10**5 # Large penalty factor, adjust based on the scale of the problem

# Penalties for the constraints C1, C2, and C3
penalty_C1 = sum(lamda * (sum(xvars[r, i] for i in range(nsites)) - 1)**2 for r in range(nparts))
penalty_C2 = sum(lamda * (xvars[r, i] * xvars[s, i]) for i in range(nsites) for (r, s) in phi)
penalty_C3 = sum(lamda * (xvars[r, i] * xvars[s, i]) for i in range(nsites) for (r, s) in psi)

# Updated objective function including penalties
new_objective_expr = sum(cost(r,i,j) * (xvars[r, i] * xvars[s, j] + xvars[r, j] * xvars[s, i])
                                        for i in range(nsites) for j in range(i,nsites) for (r,s) in phi) + (penalty_C1 + penalty_C2 + penalty_C3)

my_model.setObjective(new_objective_expr, GRB.MINIMIZE)
my_model.update()
my_model.optimize()
ref_val = my_model.getAttr('ObjVal')

print("Best Gurobi solution value : ", ref_val)
print("Best Gurobi assignment :")
my_model.getVars()



# %%
# QUBO matrix with constraints
Q_new = np.zeros((nparts*nsites,nparts*nsites))

# Objective function
for (r,s) in phi:
    for i in range(nsites):
        for j in range(nsites):
            if i == j:
                Q_new[nsites*(r) + i, nsites*(s) + j] += 0
            if i < j:
                Q_new[nsites*(r) + i, nsites*(s) + j] += cost(r,i, j)
            else:
                Q_new[nsites*(r) + i, nsites*(s) + j] += cost(r,j,i)

# Constraint-1
for a in range(nparts):
    for i in range(nsites):  
        Q_new[nsites*a + i, nsites*a + i] += - 2 * lamda
        for j in range(nsites):
            Q_new[nsites*a + i, nsites*a + j] += lamda

            
#Constraint-2
for (r,s) in phi:
    for i in range(nsites):
        Q_new[nsites*(r) + i, nsites*(s) + i] += lamda

#Constraint-3
for (r,s) in psi:
    for i in range(nsites):
        Q_new[nsites*(r) + i, nsites*(s) + i] += lamda
    
# print(Q_new)


# %%
# Generation of QUBO matrix for just the objective value
Q = np.zeros((nparts*nsites,nparts*nsites))

# Objective function
for (r,s) in phi:
    for i in range(nsites):
        for j in range(nsites):
            if i == j:
                Q[nsites*(r) + i, nsites*(s) + j] += 0
            if i < j:
                Q[nsites*(r) + i, nsites*(s) + j] += cost(r,i, j)
            else:
                Q[nsites*(r) + i, nsites*(s) + j] += cost(r,j,i)

# print(Q)

# %%
#define v,n,k 
# v is number of binary varibles in the objective function
# n is number of qubits used
# k is some thing we know which is understood well in the paper(tunable parameter k>1)
#for now I am taking k = 3
v = nparts*nsites
print("Number of variables : ",v)
k = 3
n = math.ceil(v**(1/k))  #here ceiling function is used to just ensure sufficient number of qubits to suffice the number of pauli strings obtained to be greater than binary varibles
#al = 10*n**(math.floor(k/2))  #al is the alpha value mentioned in the loss function
al = 1e30
while v > 3*math.comb(n,k):
    n += 1
print("Number of qubits needed : ",n)
# al = 1e10
n_l=int(v/n) # number of layers ----> in paper it is mentioned no. of layers is in the order O(v/n)
print("Number of layers for the ansatz : ", n_l) #this is for number reps of ansatz


pauli_strings = generate_pauli_strings(v,n, k)
print("Number of generated Pauli Strings:", len(pauli_strings))
# for string in pauli_strings:
#     print(string)


# %%
circuit = EfficientSU2(n,reps=n_l,entanglement='circular')
num_params = circuit.num_parameters
print(f"Selected ansatz : EfficientSU2 with {n_l} layers and {num_params} parameters")
# circuit.decompose().draw()

my_callback_dict = {"iters" : 0, 'best params' : None, 'best value' : None, 'history' : []}

def build_callback(circuit, obs, al, Q, offset, callback_dict):

    def callback(x0,*args):
        callback_dict['iters'] += 1
        val = loss_func(x0,obs,al,Q,circuit,offset)
        if callback_dict['best value']:
            if callback_dict['best value'] > val:
                callback_dict['best value'] = val
                callback_dict['best params'] = x0
        else:
            callback_dict['best value'] = val
            callback_dict['best params'] = x0
        callback_dict['history'].append(val)
        print(f"Iters done : {callback_dict['iters']}, current value : {val}, best value so far : {callback_dict['best value']}, best gurobi value : {ref_val}")
        if abs(ref_val-val) < 1e-3:
            print("Optimization is successful")
            print("Optimal value", val)
            print("Optimal params :", x0)
            epv = exp_val(x0, pauli_strings, circuit)
            xo = out(epv, v)
            xo = np.array(xo)
            f_ans, ans = post_pro(xo, Q_new) 
            print("Best part to site allocation:", f_ans)
            end_time = time.time()
            print(f'Time : {end_time - start_time}')
            #print("TIme Pass:", ans + nparts*lamda)
            sys.exit(0)
    return callback

# def plot_callback_func(loss_value,index):
#     # iteration, value = zip(*loss_value)
#     plt.plot(loss_value)
#     plt.xlabel('Iteration')
#     plt.ylabel('Loss Value')
#     plt.title('Loss Value vs Iteration')
#     plt.savefig(f"loss_curve_{index}_w.png")

def optimizer_function(params, num_iterations, pauli_strings, circuit,al,Q):
    try:
        offset = nparts*lamda
        bounds = [(None, None) for _ in range(len(params))]
        callback = build_callback(circuit,pauli_strings,al,Q,offset, callback_dict=my_callback_dict)
        result = minimize(loss_func, x0=params, method = 'COBYLA', bounds = bounds, args = (pauli_strings, al, Q,circuit,offset), callback=callback, options={'maxiter': 200})  
        print("Optimization result:")
        print("Optimal solution:", result.x)
        print("Optimal function value:", result.fun)
        return result.x, result.fun
    except Exception as e:
        print("Optimization failed:", e)
        return None

start_time = time.time()
#num_iterations = 100
seed_array = [1,42,12344321, 123,98,76,10,2,67,56,77,654,786,32,13,16,777,346,807,81]
#seed_array = [42,123,12344321, 1,98]
random.seed(v)
random.shuffle(seed_array)
num_iterations = 20
temp_result = 1e30
temp_params=[]


for i in range(len(seed_array)):
    print("Starting for seed value:", seed_array[i])
    random.seed(seed_array[i])
    init_gui = [random.random() for _ in range(num_params)]
    params_out, iter_result = optimizer_function(init_gui, num_iterations, pauli_strings, circuit, al, Q_new)
    if iter_result<temp_result:
        temp_result=iter_result
        temp_params=params_out.copy()
        seed_val = seed_array[i]
    print("Done Guess -", i+1)
    print("This is for seed value:", seed_array[i])
    print("This is the best seed value till now:", seed_val)
    


print("\n")    
print("-----------------------------------------------------------------------------------------------------------------------------")
print("All guesses are done")
print("Optimal Objective Value :",temp_result)
print("\n")
print("Optimal Parameters Values:",temp_params)

#print("Now Plotting")

#loss_value_basin_sm = []

# for i in range(len(seed_array)):
#     loss_value_basin_sm = []
#     for j in range(num_iterations+1):
#         loss_value_basin_sm.append(my_callback_dict['history'][(num_iterations+1)*i + j])
#     loss_value_basin_sm = np.array(loss_value_basin_sm)
#     plot_callback_func(loss_value_basin_sm,i+1)

# loss_value=[]

# def optimizer_function2(params, num_iterations, pauli_strings, al, m, Q, n, n_l):
#     try:
#         bounds = [(None, None) for _ in range(len(params))]
#         offset = nparts*lamda
#         result = minimize(loss_func, x0=params, method = "cobyla", bounds= bounds, args= (pauli_strings, al, m, Q, n, n_l,offset))  
#         callback_func(result.fun,loss_value)
#         print("Optimization result:")
#         print("Optimal solution:", result.x)
#         print("Optimal function value:", result.fun)
#         return result.x
#     except Exception as e:
#         print("Optimization failed:", e)
#         return None

# num_iterations = 200
# num_params = EfficientSU2(n,reps=5,entanglement='circular').num_parameters
# for _ in range(20):
#     init_gui= [random.random() for _ in range(num_params)]
#     # init_gui = np.array([1.59923537,  0.03099397,  0.67169022, -0.06055234, 0.3161772, 0.75186449, 1.18912076,  0.34170788,  0.75591285,  0.51536389, -0.16928087,  0.09657667,
#     # 0.14862277,  0.69071779,  0.63832258,  1.88287574,  0.24934395,  0.15414134,
#     # 0.7113865,   1.82154775,  0.48942748,  0.86922524,  0.53596849,  0.00782311])
#     params_out1 = optimizer_function2(init_gui, num_iterations, pauli_strings, al, m, Q_new, n, n_l)


# plot_callback_func(loss_value)
# # claerly understand this optimized value being printed is not the value of tranporation cost it is the value of Q mentioned in the function so put the obtained assignment in the C(x)

# %%
#print(params_out1)

# %%
epv1 = exp_val(temp_params, pauli_strings, circuit) #now this array contains all the final expectation values on all pauli strings after optimization
print(epv1)

# %%
xo1 = out(epv1, v) #this is the preliminary output bitstring in 0 and 1 just after optimization, but before post-processing step
print(xo1)
xo1 = np.array(xo1)

print("Assignemnt after basin_hoping optimization for all guesses:", xo1)

# %%
print("This is the value of obj_val after basin_hoping optimization for all guesses:", xo1.T @ Q @ xo1)

# %%
#this function is used to check whether the obtained bitstring after every 2-bit swap is the optimized one or not 

f_ans, ans = post_pro(xo1, Q_new) 

f_ans = np.array(f_ans)
print("This is the assignment after basin_hoping optimization for all guesses , followed by post pro at the end:",f_ans)

print("This is the value of obj_val after basin_hoping optimization for all guesses:", ans+nparts*lamda)

end_time = time.time()
print(f'Time : {end_time - start_time}')
        
      
#take output of every iteration of bit swap

# %%



# %%


# %%



