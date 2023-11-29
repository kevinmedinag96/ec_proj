import numpy as np
import evolutionary_utils
import ml_model
import pandas as pd
import tensorflow as tf
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize
import pymoo
from pymoo.termination import max_gen
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import sys

arr_1 = [1,2,3]

arr_2 = [4,5,6]

r = np.stack((arr_1,arr_2),axis=1)
"""
Class population will define the chromosome as the set of weights of a NN model. It will be a 1D vector,
every certain range will define the weights for the k layer of the NN model
"""
class EvolutionaryProblem(Problem):
    def __init__(self,N,n,model,xl,xu,bool_random_popu = True):
        
        self.N = N   #number of individuals
        self.n = n #decision varaible space
        self.model = model #ML model to guide the evolutionary process

        self.xl = xl #lower bound decision values
        self.xu = xu #upper bound

        self.fitness = np.inf * np.ones((self.N,1)) #minimization problem

        if bool_random_popu: #generate random population
            self.chromosome = evolutionary_utils.generate_random_population(self.N,self.n)
        else:
            pass
        super().__init__(n_var=self.n,n_obj=1,xl = self.xl, xu = self.xu)


    def _evaluate(self,x,out, *args, **kwargs):
        #this function is executed to compute fitness for individual's population.
        
        #set x as the chromosome since it is the numpy array of [no_indvs, no_decision variables]
        self.chromosome = x
        #decode chromosome
        self.decode_chromosome()       

        #compute catcrossentropy loss for population in iteration i
        self.compute_fitness_population()

        #return evaluated N individuals
        out["F"] = self.get_fitness()

    def get_fitness(self):
        return self.fitness

    def load_dataset(self,obs = None, features = None,num_cat_labels = 0,dataset = None,labels = None):
        """
        This function will set the training data X data and y data
        if obs = None then dataset is custom loaded
        otherwise it will be randomly generated
        """
        if obs is not None: #random
            self.X_train = np.random.random(size=(obs,features))
            y_true_data = np.random.randint(0,num_cat_labels,size=(obs,))
            self.y_train = np.array(pd.get_dummies(y_true_data).astype(np.int8))
                                    #tf.convert_to_tensor(np.array(pd.get_dummies(y_true_data).astype(np.int8)),
                            #                 dtype=tf.int8)
        else: #insert dataset
            self.X_train = dataset
            self.y_train = labels
    
    def compute_fitness_population(self):
        for indv in range(self.N):
            #processing individual's weights, biases and load them into model
            indv_weights = []
            weights = self.decoded_chromosome[0]
            for _,popu_layer in enumerate(weights):
                indv_weights.append(popu_layer[indv,:,:])

            indv_biases = []
            biases = self.decoded_chromosome[1]
            for _,popu_layer in enumerate(biases):
                indv_biases.append(popu_layer[indv,:])
            
            #update weights biases in model For indvidual i
            self.model.updateNetworkWeights(indv_weights,indv_biases)

            #predictions for the model composed of weights and biases of individual i
            y_pred_indv = self.model.compute_prediction(self.X_train)

            #compute loss fn for individual i
            loss_indv = self.model.compute_loss(self.y_train,y_pred_indv) #catcrossentropy  scratch
            #loss_indv = self.model.loss_fn(self.y_train,y_pred_indv) #catcrossentropy 
            #print(f"individual loss : {loss_indv}")

            #setting fitness (catcrossentropy loss) to individual i
            self.fitness[indv,0] = loss_indv


    def decode_chromosome(self):
        """
        This function will basically rearrange the chromosome so that the weights are aligned with the current 
        NN architecture

        encoded -> decoded
        [number_indvs, weights_layer_1 * weight_lay_2 ... * weight_lay_k] ---------------->
        list of batch_layer_1, batch_layer_2, ... batch_layer_k , i.e. we are going to ahve a list
        where each entry is the batch layers 1 , batch layer 2, ... batch layer k. THus, we will be 
        saving weights per layer in a whole population basis.

        :return: layer list with entries of the form [no_individuals, weight_unit_num_lay_prev ,weight_unit_num_lay]
        """
        units_range = self.model.input_shape
        init_range = 0
        num_w_lay_prev = units_range
        ls_popu_w_layers = []

        #decode weights for layers
        for _,num_w_lay in enumerate(self.model.weight_units_per_layer):
            units_range *= num_w_lay
            layer_i_w_popu = self.chromosome[:,init_range:init_range + units_range].reshape(self.N,num_w_lay_prev,num_w_lay)
            ls_popu_w_layers.append(layer_i_w_popu)
            init_range += units_range
            num_w_lay_prev = num_w_lay
            units_range = num_w_lay_prev

        ls_popu_bias_layers = []
        #decode biases for layers
        for _,num_w_lay in enumerate(self.model.weight_units_per_layer):
            units_range = num_w_lay
            layer_i_biases_popu = self.chromosome[:,init_range:init_range + units_range].reshape(self.N,num_w_lay)
            ls_popu_bias_layers.append(layer_i_biases_popu)
            init_range += units_range


        #tuple popu weight layer, popu biases layer
        self.decoded_chromosome = (ls_popu_w_layers,ls_popu_bias_layers)

    
if __name__ == "__main__":

    #--------------------- LOAD GENETIC MARKERS DATASET------------------------------- #
    work_path = "C:/Users/kevin/OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey/MCC/MCC-i tercer semestre/Evolutionary Computing/Hws/Project/"
    file_name = "genetic_markers_de.txt"

    cols_name = []
    for i in range(2006):
        cols_name.append("col_" + str(i))

    df2 = pd.read_csv(work_path + "data_markers_1000.ped",sep='\s+', header=None,names=cols_name)
    df2_labels = df2["col_0"]
    df2 = df2.drop(columns=['col_0', 'col_1','col_2','col_3'])

    #convert string columns into categories

    label_encoder = LabelEncoder()
    one_hot_encoder = OneHotEncoder()

    for i in range(6,2006):
        col_name = "col_" + str(i)
        df2[col_name] = label_encoder.fit_transform(df2[col_name])

    #One Hot encode output labels
    oh_df2_labels = one_hot_encoder.fit_transform(df2_labels.values.reshape(-1, 1))

    #normalize training data
    col_name = "col_5"
    df2[col_name] = (df2[col_name] - df2[col_name].mean()) / df2[col_name].std()


    #--------------------- LOAD GENETIC MARKERS DATASET------------------------------- #

    #----- SECTION 1 DESIGN AND IMPLEMENT DE ALGORITHM TO OUR GENETIC MARKERS DATASET
    sys.path.insert(0,work_path + "Modules/")
    
    N = 20 #population size
    
    input_shape = df2.columns.shape[0]
    num_cat_labels = oh_df2_labels.shape[1]
    num_weights_per_layer = [128,56,num_cat_labels]
    num_weights_per_layer_plus_input = np.array([input_shape]+ num_weights_per_layer)
    #specify decision variable space of individuals : which in this problem is the size of the total weights and biases of the
    #neural network model
    n = np.sum(num_weights_per_layer_plus_input[:-1] * num_weights_per_layer_plus_input[1:]) + np.sum(num_weights_per_layer)
    #loss = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    loss = evolutionary_utils.categorical_crossentropy_scratch_array
    nn = ml_model.Model(input_shape,num_weights_per_layer,loss)
    nn.setNetwork()

    xl = -5
    xu = 5
    problem = EvolutionaryProblem(N,n,nn,xl,xu,False)
    #problem.decode_chromosome()

    #dataset
    problem.load_dataset(dataset=df2,labels=oh_df2_labels.toarray())

    #set pymoo algorithm
    
    algorithm = DE(
        pop_size=N,
        variant="DE/rand/1/bin",
        CR=0.9,
        dither="vector",
        jitter=False
    )

    num_generations = 100
    res = minimize(problem,
                algorithm,
                seed=1,
                verbose=True,
                termination=max_gen.MaximumGenerationTermination(num_generations),
                save_history=False)

    print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

    """

    # Obtiene la historia de la optimización
    history = res.history

    best_f_min_per_generation = []
    best_f_avg_per_generation = []
    for gen, de_obj in enumerate(history):
        best_f_min_per_generation.append(de_obj.output.f_min.value)
        print(f"Generación {gen + 1}: f_min  : {best_f_min_per_generation[gen]}")
        best_f_avg_per_generation.append(de_obj.output.f_avg.value)
        print(f"Generación {gen + 1}: f_avg  : {best_f_avg_per_generation[gen]}")

    #f_min,f_avg matrix
    f_arr = np.stack((best_f_min_per_generation,best_f_avg_per_generation),axis=1)
    #guardar datos DE statistics
    file_name = "genetic_markers_de.txt"
    np.savetxt(work_path + file_name, f_arr, fmt='%f', delimiter='\t')

    """

    


