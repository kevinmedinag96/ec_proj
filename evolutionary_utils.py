import numpy as np
import pandas as pd
"""
n -> number of decision variables
N -> Number of individuals inside the population
"""

def generate_random_population(N,n):
    return np.random.random(size=(N,n))

def categorical_crossentropy_scratch(y_true,y_pred):
    """
    :param: y_true [n_indvs,true one hot label]
    :param: y_pred [n_indvs, pred one hot label]
    """
    n = len(y_true)
    ce_output = 0.0
    ce_global = 0.0
    for i in range(n):
        y_t = y_true[i]
        y_p = y_pred[i]
        for j in range(len(y_t)):
            y_tt = y_t[j]
            y_pp = y_p[j]
            ce_output += y_tt * np.log(y_pp + 1e-25) + (1 - y_tt) * np.log(1 - y_pp + 1e-25)

        ce_global +=ce_output
        #print(f"training input : {i}, ce : {ce_output}")
        ce_output = 0.0
    #print(f"batch ce : {ce_global}")
    return -ce_global / n

def categorical_crossentropy_scratch_array(y_true,y_pred):
    
    lg_pp =np.log(y_pred + 1e-25)
    log_minus_one_pp = np.log(1 - y_pred + 1e-25)

    return np.mean(-np.sum(y_true * lg_pp  + (1-y_true) * log_minus_one_pp,axis=1))


if __name__ == "__main__":
    obs = 5000
    y_true_data = np.random.randint(0,3,size=(obs,))
    y_pred_data = np.random.random(size=(obs,3))
    y_train = np.array(pd.get_dummies(y_true_data).astype(np.int8))
                                        #tf.convert_to_tensor(np.array(pd.get_dummies(y_true_data).astype(np.int8)),
                                #                 dtype=tf.int8)

    loss_1 = categorical_crossentropy_scratch_array(y_train,y_pred_data)
    loss_2 = categorical_crossentropy_scratch(y_train,y_pred_data)

    print(loss_1)
    print(loss_2)