import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import pandas as pd
y_true = [[0, 1, 0], [0, 0, 1]]
y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]

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
    



y_true = [[1, 1, 0], [0, 0, 1]]
y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
# Using 'auto'/'sum_over_batch_size' reduction type.
#cce = categorical_crossentroyp(y_true,y_pred)

cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
print(cce(y_true, y_pred))

#data to make random predictions
samples = 10
features = 5
random_predict_data = np.random.random(size=(samples,features))
tensor_random_predict_data = tf.convert_to_tensor(random_predict_data, dtype=tf.float32)
print(f"test data tensor: {tensor_random_predict_data}")

cat = 3
y_true_data = np.random.randint(0,cat,size=(tensor_random_predict_data.shape[0],))
tensor_y_true_data = tf.convert_to_tensor(y_true_data, dtype=tf.int8)
#one hot encode categorical labels
tensor_oh_test_labels = tf.convert_to_tensor(np.array(pd.get_dummies(tensor_y_true_data).astype(np.int8)),
                                             dtype=tf.int8)


print(f"test data tensor: {tensor_oh_test_labels}")

#Define Keras model----------------------------------------- #
model = Sequential()
l_num_weights = 14
L_num_weights = cat
model.add(Dense(l_num_weights, input_shape=(tensor_random_predict_data.shape[1],), activation='relu'))
model.add(Dense(L_num_weights, activation='sigmoid'))  

#test settin custom weights
l_custom_weights = np.ones((tensor_random_predict_data.shape[1],l_num_weights))
l_custom_biases = np.ones((l_num_weights,))
L_custom_weights = np.ones((l_num_weights,L_num_weights))
L_custom_biases = np.ones((L_num_weights,))
#list_weights = [l_custom_weights,l_custom_biases,L_custom_weights,L_custom_biases]
#model.set_weights(list_weights)

#model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

y_pred = model.predict(tensor_random_predict_data)

print(f"prediction : {y_pred}")

# compute categorical loss
cat_ce_loss = tf.losses.categorical_crossentropy(tensor_oh_test_labels,y_pred)
cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
cat_ce_loss_2 = cce(tensor_oh_test_labels,y_pred)

print(f"cat ce loss : {np.mean(cat_ce_loss)}")
print(f"cat ce loss 2 : {cat_ce_loss_2}")