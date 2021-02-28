import numpy as np

def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()

def y_to_onehot(y, n_class):
    data_len = len(y)

    onehot_matrix = np.zeros((n_class, data_len))
    onehot_matrix[y, np.arange(data_len)] = 1
    return onehot_matrix

def add_bias(x):
    return np.concatenate((np.ones((len(x), 1)), x), axis = 1)


class SoftmaxRegressor(object):
    def __init__(self, W = None, eta = 0.01, max_iter = 5001, history = None):
        self.W = W
        self.eta = eta
        self.max_iter = max_iter
        self.eps = 1e-7
        self.history = history
    
    def fit(self, X_train, y_train, X_valid, y_valid, patience_upper = 20):
        self.history = {}
        terminate = False
        epochs = []
        losses = []
        weights = []
        
        n_feature = X_train.shape[1]
        n_class = len(np.unique(y_train))
        train_data_len = len(X_train)
        val_data_len = len(X_valid)
        min_loss = float("inf")

        X_train_with_bias, X_valid_with_bias = add_bias(X_train), add_bias(X_valid)
        onehot_matrix = y_to_onehot(y_train, n_class)
        val_onehot_matrix = y_to_onehot(y_valid, n_class)

        # weight initalization (Xavier)
        fan_in = n_feature
        fan_out = n_class

        sigma_W = 2 / (fan_in + fan_out)
        self.W = sigma_W * np.random.randn(n_class, n_feature + 1)


        # learning algorithm
        for iteration in range(self.max_iter):
            # train loss
            class_score_matrix = np.dot(self.W, X_train_with_bias.T)
            class_probability = np.apply_along_axis(softmax, 0, class_score_matrix)
            train_loss = - (onehot_matrix * np.log(class_probability)).sum() / train_data_len

            # parameters update
            gradient = np.dot((class_probability - onehot_matrix), X_train_with_bias) / train_data_len
            self.W = self.W - self.eta * gradient

            # validation loss
            val_class_score_matrix = np.dot(self.W, X_valid_with_bias.T)
            val_class_probability = np.apply_along_axis(softmax, 0, val_class_score_matrix)
            val_loss = - (val_onehot_matrix * np.log(val_class_probability)).sum() / val_data_len

            # monitoring
            if iteration % 500 == 0:
                print("iter : {}, train_loss : {}, valid_loss : {}".format(iteration, 
                                                                           round(train_loss,4), 
                                                                           round(val_loss,4)))

            if val_loss < min_loss:
                min_loss = val_loss

                # saving history
                epochs.append(iteration)
                losses.append(min_loss)
                weights.append(self.W)

            # stopping rule
            else :
                patience += 1
                if patience == patience_upper: 
                    terminate = True
                    print('** Early Stopped **') 

            # End flow
            if iteration == self.max_iter - 1:
                terminate = True
                
            if terminate == True:
                self.history['epochs'] = epochs
                self.history['val_losses'] = losses
                self.history['weights'] = weights
                break
    
        return self.history


    def predict(self, X, return_prob = False):
        X_with_bias = add_bias(X)
            
        final_W = self.history['weights'][-1]
            
        class_score_matrix = np.dot(final_W, X_with_bias.T)
        class_probability = np.apply_along_axis(softmax, 0, class_score_matrix)
            
        if return_prob == True:
            return class_probability
            
        prediction = class_probability.argmax(axis = 0)
        return prediction