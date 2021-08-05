import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io
import math
import matplotlib.gridspec as gridspec
from plotting import newfig
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import layers, activations
from scipy.interpolate import griddata
from eager_lbfgs import lbfgs, Struct
from pyDOE import lhs
import seaborn as sns

# define size of the network
layer_sizes = [2, 128, 128, 128, 128, 1]

sizes_w = []
sizes_b = []
col_weights_vec = []
u_weights_vec = []
# for i in range(3):
for i, width in enumerate(layer_sizes):
    if i != 1:
        sizes_w.append(int(width * layer_sizes[1]))
        sizes_b.append(int(width if i != 0 else layer_sizes[1]))


# L-BFGS weight getting and setting from https://github.com/pierremtb/PINNs-TF2.0
def set_weights(model, w, sizes_w, sizes_b):
    for i, layer in enumerate(model.layers[0:]):
        start_weights = sum(sizes_w[:i]) + sum(sizes_b[:i])
        end_weights = sum(sizes_w[:i + 1]) + sum(sizes_b[:i])
        weights = w[start_weights:end_weights]
        w_div = int(sizes_w[i] / sizes_b[i])
        weights = tf.reshape(weights, [w_div, sizes_b[i]])
        biases = w[end_weights:end_weights + sizes_b[i]]
        weights_biases = [weights, biases]
        layer.set_weights(weights_biases)


def get_weights(model):
    w = []
    for layer in model.layers[0:]:
        weights_biases = layer.get_weights()
        weights = weights_biases[0].flatten()
        biases = weights_biases[1]
        w.extend(weights)
        w.extend(biases)

    w = tf.convert_to_tensor(w)
    return w


# define the neural network model
def neural_net(layer_sizes):
    model = Sequential()
    model.add(layers.InputLayer(input_shape=(layer_sizes[0],)))
    for width in layer_sizes[1:-1]:
        model.add(layers.Dense(
            width, activation=tf.nn.tanh,
            kernel_initializer="glorot_normal"))
    model.add(layers.Dense(
        layer_sizes[-1], activation=None,
        kernel_initializer="glorot_normal"))
    return model


# define the loss
def loss(x_f_batch_1, t_f_batch_1,
         x_f_batch_2, t_f_batch_2,
         x_f_batch_3, t_f_batch_3,
         x_f_batch_4, t_f_batch_4,
         x_f_int_1, t_f_int_1,
         x_f_int_2, t_f_int_2,
         x_f_int_3, t_f_int_3,
         x0, t0, u0,
         x_lb_1, t_lb_1,
         x_lb_2, t_lb_2,
         x_lb_3, t_lb_3,
         x_lb_4, t_lb_4,
         x_ub_1, t_ub_1,
         x_ub_2, t_ub_2,
         x_ub_3, t_ub_3,
         x_ub_4, t_ub_4,
         col_weights, u_weights):

    f_u_pred_1 = f_model(u_model_1, x_f_batch_1, t_f_batch_1)
    f_u_pred_2 = f_model(u_model_2, x_f_batch_2, t_f_batch_2)
    f_u_pred_3 = f_model(u_model_3, x_f_batch_3, t_f_batch_3)
    f_u_pred_4 = f_model(u_model_4, x_f_batch_4, t_f_batch_4)

    f_u_int_1_1 = f_model(u_model_1, x_f_int_1, t_f_int_1)
    f_u_int_1_2 = f_model(u_model_2, x_f_int_1, t_f_int_1)
    f_u_int_2_1 = f_model(u_model_2, x_f_int_2, t_f_int_2)
    f_u_int_2_2 = f_model(u_model_3, x_f_int_2, t_f_int_2)
    f_u_int_3_1 = f_model(u_model_3, x_f_int_3, t_f_int_3)
    f_u_int_3_2 = f_model(u_model_4, x_f_int_3, t_f_int_3)

    mse_f_int = tf.reduce_mean(tf.square(f_u_int_1_1)) + \
                tf.reduce_mean(tf.square(f_u_int_1_2)) + \
                tf.reduce_mean(tf.square(f_u_int_2_1)) + \
                tf.reduce_mean(tf.square(f_u_int_2_2)) + \
                tf.reduce_mean(tf.square(f_u_int_3_1)) + \
                tf.reduce_mean(tf.square(f_u_int_3_2))

    mse_f_u = tf.reduce_mean(tf.square(f_u_pred_1)) + \
              tf.reduce_mean(tf.square(f_u_pred_2)) + \
              tf.reduce_mean(tf.square(f_u_pred_3)) + \
              tf.reduce_mean(tf.square(f_u_pred_4))

    u0_pred = u_model_1(tf.concat([x0, t0], 1))

    mse_0_u = tf.reduce_mean(tf.square((u0 - u0_pred)))

    u_lb_pred_1, u_x_lb_pred_1 = u_x_model(u_model_1, x_lb_1, t_lb_1)
    u_lb_pred_2, u_x_lb_pred_2 = u_x_model(u_model_2, x_lb_2, t_lb_2)
    u_lb_pred_3, u_x_lb_pred_3 = u_x_model(u_model_3, x_lb_3, t_lb_3)
    u_lb_pred_4, u_x_lb_pred_4 = u_x_model(u_model_4, x_lb_4, t_lb_4)

    u_ub_pred_1, u_x_ub_pred_1 = u_x_model(u_model_1, x_ub_1, t_ub_1)
    u_ub_pred_2, u_x_ub_pred_2 = u_x_model(u_model_2, x_ub_2, t_ub_2)
    u_ub_pred_3, u_x_ub_pred_3 = u_x_model(u_model_3, x_ub_3, t_ub_3)
    u_ub_pred_4, u_x_ub_pred_4 = u_x_model(u_model_4, x_ub_4, t_ub_4)

    mse_b_u = tf.reduce_mean(tf.square(tf.math.subtract(u_lb_pred_1, u_ub_pred_1))) + \
              tf.reduce_mean(tf.square(tf.math.subtract(u_x_lb_pred_1, u_x_ub_pred_1))) + \
              tf.reduce_mean(tf.square(tf.math.subtract(u_lb_pred_2, u_ub_pred_2))) + \
              tf.reduce_mean(tf.square(tf.math.subtract(u_x_lb_pred_2, u_x_ub_pred_2))) + \
              tf.reduce_mean(tf.square(tf.math.subtract(u_lb_pred_3, u_ub_pred_3))) + \
              tf.reduce_mean(tf.square(tf.math.subtract(u_x_lb_pred_3, u_x_ub_pred_3))) + \
              tf.reduce_mean(tf.square(tf.math.subtract(u_lb_pred_4, u_ub_pred_4))) + \
              tf.reduce_mean(tf.square(tf.math.subtract(u_x_lb_pred_4, u_x_ub_pred_4)))

    int_1_pred_low, u_x_int_1_pred_low = u_x_model(u_model_1, x_f_int_1, t_f_int_1)
    int_1_pred_hi, u_x_int_1_pred_hi = u_x_model(u_model_2, x_f_int_1, t_f_int_1)
    int_2_pred_low, u_x_int_2_pred_low = u_x_model(u_model_2, x_f_int_2, t_f_int_2)
    int_2_pred_hi, u_x_int_2_pred_hi = u_x_model(u_model_3, x_f_int_2, t_f_int_2)
    int_3_pred_low, u_x_int_3_pred_low = u_x_model(u_model_3, x_f_int_3, t_f_int_3)
    int_3_pred_hi, u_x_int_3_pred_hi = u_x_model(u_model_4, x_f_int_3, t_f_int_3)

    mse_int = tf.reduce_mean(tf.square(tf.math.subtract(int_1_pred_low, int_1_pred_hi))) + \
              tf.reduce_mean(tf.square(tf.math.subtract(u_x_int_1_pred_low, u_x_int_1_pred_hi))) + \
              tf.reduce_mean(tf.square(tf.math.subtract(int_2_pred_low, int_2_pred_hi))) + \
              tf.reduce_mean(tf.square(tf.math.subtract(u_x_int_2_pred_low, u_x_int_2_pred_hi))) + \
              tf.reduce_mean(tf.square(tf.math.subtract(int_3_pred_low, int_3_pred_hi))) + \
              tf.reduce_mean(tf.square(tf.math.subtract(u_x_int_3_pred_low, u_x_int_3_pred_hi)))

    return 100*mse_0_u + mse_b_u + mse_f_u + 20*mse_int + mse_f_int, tf.reduce_mean(tf.square((u0 - u0_pred))), mse_b_u, mse_int


# define the physics-based residual, we want this to be 0

@tf.function
def f_model(u_model, x, t):
    u = u_model(tf.concat([x, t], 1))
    u_x = tf.gradients(u, x)
    u_xx = tf.gradients(u_x, x)
    u_t = tf.gradients(u, t)
    c1 = tf.constant(.0001, dtype=tf.float32)
    c2 = tf.constant(5.0, dtype=tf.float32)
    f_u = u_t - c1 * u_xx + c2 * u * u * u - c2 * u
    return f_u


@tf.function
def u_x_model(u_model, x, t):
    u = u_model(tf.concat([x, t], axis=1))
    u_x = tf.gradients(u, x)
    return u, u_x


@tf.function
def grad(x_f_batch_1, t_f_batch_1,
         x_f_batch_2, t_f_batch_2,
         x_f_batch_3, t_f_batch_3,
         x_f_batch_4, t_f_batch_4,
         x_f_int_1, t_f_int_1,
         x_f_int_2, t_f_int_2,
         x_f_int_3, t_f_int_3,
         x0, t0, u0,
         x_lb_1, t_lb_1,
         x_lb_2, t_lb_2,
         x_lb_3, t_lb_3,
         x_lb_4, t_lb_4,
         x_ub_1, t_ub_1,
         x_ub_2, t_ub_2,
         x_ub_3, t_ub_3,
         x_ub_4, t_ub_4, col_weights, u_weights):
    with tf.GradientTape(persistent=True) as tape:
        # tape.watch(col_weights)
        # tape.watch(u_weights)
        loss_value, mse_0, mse_b, mse_f = loss(x_f_batch_1, t_f_batch_1,
                                               x_f_batch_2, t_f_batch_2,
                                               x_f_batch_3, t_f_batch_3,
                                               x_f_batch_4, t_f_batch_4,
                                               x_f_int_1, t_f_int_1,
                                               x_f_int_2, t_f_int_2,
                                               x_f_int_3, t_f_int_3,
                                               x0, t0, u0,
                                               x_lb_1, t_lb_1,
                                               x_lb_2, t_lb_2,
                                               x_lb_3, t_lb_3,
                                               x_lb_4, t_lb_4,
                                               x_ub_1, t_ub_1,
                                               x_ub_2, t_ub_2,
                                               x_ub_3, t_ub_3,
                                               x_ub_4, t_ub_4,
                                               col_weights, u_weights)
        grads_1 = tape.gradient(loss_value, u_model_1.trainable_variables)
        grads_2 = tape.gradient(loss_value, u_model_2.trainable_variables)
        grads_3 = tape.gradient(loss_value, u_model_3.trainable_variables)
        grads_4 = tape.gradient(loss_value, u_model_4.trainable_variables)
        # print(grads)
        grads_col = tape.gradient(loss_value, col_weights)
        grads_u = tape.gradient(loss_value, u_weights)
        gradients_u = tape.gradient(mse_0, u_model_1.trainable_variables)
        gradients_f = tape.gradient(mse_f, u_model_1.trainable_variables)
        del tape
    return loss_value, mse_0, mse_b, mse_f, grads_1, grads_2, grads_3, grads_4, grads_col, grads_u, gradients_u, gradients_f


def fit(x_f, t_f,
        x_f_int_1, t_f_int_1,
        x_f_int_2, t_f_int_2,
        x_f_int_3, t_f_int_3,
        x0, t0, u0,
        x_lb_1, t_lb_1,
        x_lb_2, t_lb_2,
        x_lb_3, t_lb_3,
        x_lb_4, t_lb_4,
        x_ub_1, t_ub_1,
        x_ub_2, t_ub_2,
        x_ub_3, t_ub_3,
        x_ub_4, t_ub_4, col_weights, u_weights, tf_iter, newton_iter):
    # Can adjust batch size for collocation points, here we set it to N_f
    batch_sz = N_f
    n_batches = N_f // batch_sz

    start_time = time.time()
    # create optimizer s for the network weights, collocation point mask, and initial boundary mask
    tf_optimizer_1 = tf.keras.optimizers.Adam(lr=0.005, beta_1=.99)
    tf_optimizer_2 = tf.keras.optimizers.Adam(lr=0.005, beta_1=.99)
    tf_optimizer_3 = tf.keras.optimizers.Adam(lr=0.005, beta_1=.99)
    tf_optimizer_4 = tf.keras.optimizers.Adam(lr=0.005, beta_1=.99)
    tf_optimizer_weights = tf.keras.optimizers.Adam(lr=0.005, beta_1=.99)
    tf_optimizer_u = tf.keras.optimizers.Adam(lr=0.005, beta_1=.99)

    print("starting Adam training")

    # For mini-batch (if used)
    for epoch in range(tf_iter):
        for i in range(n_batches):
            x0_batch = x0
            t0_batch = t0
            u0_batch = u0

            # x_f_batch = x_f[i * batch_sz:(i * batch_sz + batch_sz), ]
            # t_f_batch = t_f[i * batch_sz:(i * batch_sz + batch_sz), ]
            x_f_batch_1 = tf.reshape(x_f[t_f<(1/4)], [-1, 1])
            t_f_batch_1 = tf.reshape(t_f[t_f<(1/4)], [-1, 1])
            x_f_batch_2 = tf.reshape(x_f[(t_f >= (1 / 4)) & (t_f <= (2 / 4))], [-1, 1])
            t_f_batch_2 = tf.reshape(t_f[(t_f >= (1 / 4)) & (t_f <= (2 / 4))], [-1, 1])
            x_f_batch_3 = tf.reshape(x_f[(t_f >= (2 / 4)) & (t_f <= (3 / 4))], [-1, 1])
            t_f_batch_3 = tf.reshape(t_f[(t_f >= (2 / 4)) & (t_f <= (3 / 4))], [-1, 1])
            x_f_batch_4 = tf.reshape(x_f[t_f>(3/4)], [-1, 1])
            t_f_batch_4 = tf.reshape(t_f[t_f>(3/4)], [-1, 1])

            # X_f = np.concatenate([x_f_batch, t_f_batch], axis=1)
            #
            # x_f_early = X_f[(X_f[:, 1] <= 1 / 3)]
            #
            # arr = np.logical_and(X_f[:, 1] >= 1 / 3, X_f[:, 1] <= 2 / 3)
            #
            # x_f_middle = X_f[np.where(arr)[0]]
            # x_f_late = X_f[(X_f[:, 1] >= 2 / 3)]

            loss_value, mse_0, mse_b, mse_f, grads_1, grads_2, grads_3, grads_4, grads_col, grads_u, g_u, g_f = grad(x_f_batch_1, t_f_batch_1,
                                                                                        x_f_batch_2, t_f_batch_2,
                                                                                        x_f_batch_3, t_f_batch_3,
                                                                                        x_f_batch_4, t_f_batch_4,
                                                                                        x_f_int_1, t_f_int_1,
                                                                                        x_f_int_2, t_f_int_2,
                                                                                        x_f_int_3, t_f_int_3,
                                                                                        x0, t0, u0,
                                                                                        x_lb_1, t_lb_1,
                                                                                        x_lb_2, t_lb_2,
                                                                                        x_lb_3, t_lb_3,
                                                                                        x_lb_4, t_lb_4,
                                                                                        x_ub_1, t_ub_1,
                                                                                        x_ub_2, t_ub_2,
                                                                                        x_ub_3, t_ub_3,
                                                                                        x_ub_4, t_ub_4,
                                                                                        col_weights, u_weights)

            tf_optimizer_1.apply_gradients(zip(grads_1, u_model_1.trainable_variables))
            tf_optimizer_2.apply_gradients(zip(grads_2, u_model_2.trainable_variables))
            tf_optimizer_3.apply_gradients(zip(grads_3, u_model_3.trainable_variables))
            tf_optimizer_4.apply_gradients(zip(grads_4, u_model_4.trainable_variables))
            # tf_optimizer_weights.apply_gradients(zip([-grads_u], [u_weights]))

        if epoch % 100 == 0:
            elapsed = time.time() - start_time
            print('It: %d, Time: %.2f' % (epoch, elapsed))
            tf.print(f"mse_0: {mse_0}  mse_b  {mse_b}  mse_f: {mse_f}   total loss: {loss_value}")
            tf.print(tf.reduce_mean(u_weights),
                     tf.reduce_mean(col_weights))
            col_weights_iter.append(tf.reduce_mean(col_weights).numpy().T)
            u_weights_iter.append(tf.reduce_mean(u_weights).numpy().T)
            start_time = time.time()

    # An "interface" to matplotlib.axes.Axes.hist() method
    # # plt.hist(x=np.concatenate([g.numpy().flatten() for g in grads_col]), bins=50, alpha=0.5, label = "Residual Gradients")
    # print(g_u[2])
    # sns.histplot(x=g_f[2].numpy().flatten(), bins=70, alpha=0.5, stat = "density", label = "Residual Gradients", element = "poly", color = "green", log_scale = True)
    # sns.histplot(x=g_u[2].numpy().flatten(), bins=70, alpha=0.5, stat = "density", label = "IC Gradients", element = "poly", color = "blue", log_scale = True)
    # plt.xlabel('Gradient Magnitude')
    # plt.ylabel('Frequency')
    # plt.title('FC1 Gradient Magnitudes, Baseline PINN, 2000 Adam Iterations')
    # plt.legend(loc='upper right')
    # plt.show()
    #
    # sns.histplot(x=g_f[4].numpy().flatten(), bins=70, alpha=0.5, stat = "density", label = "Residual Gradients", element = "poly", color = "green", log_scale = True)
    # sns.histplot(x=g_u[4].numpy().flatten(), bins=70, alpha=0.5, stat = "density", label = "IC Gradients", element = "poly", color = "blue", log_scale = True)
    # plt.xlabel('Gradient Magnitude')
    # plt.ylabel('Frequency')
    # plt.title('FC2 Gradient Magnitudes, Baseline PINN, 2000 Adam Iterations')
    # plt.legend(loc='upper right')
    # plt.show()
    #
    # sns.histplot(x=g_f[6].numpy().flatten(), bins=70, alpha=0.5, stat = "density", label = "Residual Gradients", element = "poly", color = "green", log_scale = True)
    # sns.histplot(x=g_u[6].numpy().flatten(), bins=70, alpha=0.5, stat = "density", label = "IC Gradients", element = "poly", color = "blue", log_scale = True)
    # plt.xlabel('Gradient Magnitude')
    # plt.ylabel('Frequency')
    # plt.title('FC3 Gradient Magnitudes, Baseline PINN, 2000 Adam Iterations')
    # plt.legend(loc='upper right')
    # plt.show()
    #
    # sns.histplot(x=g_f[8].numpy().flatten(), bins=70, alpha=0.5, stat = "density", label = "Residual Gradients", element = "poly", color = "green", log_scale = True)
    # sns.histplot(x=g_u[8].numpy().flatten(), bins=70, alpha=0.5, stat = "density", label = "IC Gradients", element = "poly", color = "blue", log_scale = True)
    # plt.xlabel('Gradient Magnitude')
    # plt.ylabel('Frequency')
    # plt.title('FC4 Gradient Magnitudes, Baseline PINN, 2000 Adam Iterations')
    # plt.legend(loc='upper right')
    # plt.show()
    #
    # plt.hist(x=g_f[4]/sum(g_f[4]), bins=50, alpha=0.5, label = "FC3 Gradients")
    # plt.hist(x=g_f[6], bins=50, alpha=0.5, label = "FC4 Gradients")
    # # plt.title("Gradient flow in Allen-Cahn")
    # # plt.legend(loc='upper right')
    # # plt.show()
    # plt.hist(grads_u[0], bins=50, alpha = 0.5, label = "IC Gradients")
    # plt.grid(axis='y', alpha=0.75)
    # plt.xlabel('Magnitude')
    # plt.ylabel('Frequency')
    # plt.title('Gradient magnitudes')
    # # Set a clean upper y-axis limit.
    # # plt.xlim(xmin =-1, xmax=1)
    # # plt.ylim(ymax = )
    # plt.title("Gradient flow in Allen-Cahn")
    # plt.legend(loc='upper right')
    # plt.show()

    # l-bfgs-b optimization
    # print("Starting L-BFGS training")
    #
    # loss_and_flat_grad = get_loss_and_flat_grad(x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch, x_lb, t_lb, x_ub,
    #                                             t_ub, col_weights, u_weights)
    #
    # lbfgs(loss_and_flat_grad,
    #       get_weights(u_model),
    #       Struct(), maxIter=newton_iter, learningRate=0.8)


# L-BFGS implementation from https://github.com/pierremtb/PINNs-TF2.0
def get_loss_and_flat_grad(x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch, x_lb, t_lb, x_ub, t_ub, col_weights,
                           u_weights):
    def loss_and_flat_grad(w):
        with tf.GradientTape() as tape:
            set_weights(u_model, w, sizes_w, sizes_b)
            loss_value, _, _, _ = loss(x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch, x_lb, t_lb, x_ub, t_ub,
                                       col_weights, u_weights)
        grad = tape.gradient(loss_value, u_model.trainable_variables)
        grad_flat = []
        for g in grad:
            grad_flat.append(tf.reshape(g, [-1]))
        grad_flat = tf.concat(grad_flat, 0)
        # print(loss_value, grad_flat)
        return loss_value, grad_flat

    return loss_and_flat_grad


def predict(X_star):
    X_star_1 = X_star[X_star[:, 1] < (1 / 4)]
    X_star_2 = X_star[(X_star[:, 1] >= (1 / 4)) & (X_star[:, 1] < (2 / 4))]
    X_star_3 = X_star[(X_star[:, 1] >= (2 / 4)) & (X_star[:, 1] < (3 / 4))]
    X_star_4 = X_star[X_star[:, 1] >= (3 / 4)]

    u_star_1, _ = u_x_model(u_model_1, X_star_1[:, 0:1],
                          X_star_1[:, 1:2])
    u_star_2, _ = u_x_model(u_model_2, X_star_2[:, 0:1],
                          X_star_2[:, 1:2])
    u_star_3, _ = u_x_model(u_model_3, X_star_3[:, 0:1],
                          X_star_3[:, 1:2])
    u_star_4, _ = u_x_model(u_model_4, X_star_4[:, 0:1],
                          X_star_4[:, 1:2])


    u_star = tf.concat([u_star_1, u_star_2, u_star_3, u_star_4], axis = 0)

    # f_u_star = f_model(X_star[:, 0:1],
    #                    X_star[:, 1:2])

    return u_star.numpy() #, f_u_star.numpy()


# Define constants and weight vectors

col_weights_iter = []
u_weights_iter = []
lb = np.array([-1.0])
ub = np.array([1.0])

N0 = 512
N_b = 100
N_f = 20000

col_weights = tf.Variable(tf.random.uniform([N_f, 1]))
u_weights = tf.Variable(tf.random.uniform([N0, 1]))

# initialize the NN
u_model_1 = neural_net(layer_sizes)
u_model_2 = neural_net(layer_sizes)
u_model_3 = neural_net(layer_sizes)
u_model_4 = neural_net(layer_sizes)

# view the NN
u_model_1.summary()

# Import data, same data as Raissi et al

data = scipy.io.loadmat('AC.mat')

t = data['tt'].flatten()[:, None]
x = data['x'].flatten()[:, None]
Exact = data['uu']
Exact_u = np.real(Exact)

# grab training points from domain
idx_x = np.random.choice(x.shape[0], N0, replace=False)
x0 = x[idx_x, :]
u0 = tf.cast(Exact_u[idx_x, 0:1], dtype=tf.float32)

idx_t = np.random.choice(t.shape[0], N_b, replace=False)
tb = t[idx_t, :]

# Grab collocation points using latin hpyercube sampling

X_f = lb + (ub - lb) * lhs(2, N_f)

x_f = tf.convert_to_tensor(X_f[:, 0:1], dtype=tf.float32)
t_f = tf.convert_to_tensor(np.abs(X_f[:, 1:2]), dtype=tf.float32)

X0 = np.concatenate((x0, 0 * x0), 1)  # (x0, 0)
X_lb = np.concatenate((0 * tb + lb[0], tb), 1)  # (lb[0], tb)
X_ub = np.concatenate((0 * tb + ub[0], tb), 1)  # (ub[0], tb)

x0 = tf.cast(X0[:, 0:1], dtype=tf.float32)
t0 = tf.cast(X0[:, 1:2], dtype=tf.float32)

x_lb = tf.convert_to_tensor(X_lb[:, 0:1], dtype=tf.float32)
t_lb = tf.convert_to_tensor(X_lb[:, 1:2], dtype=tf.float32)

x_lb_1 = tf.reshape(x_lb[t_lb < (1 / 4)], [-1, 1])
t_lb_1 = tf.reshape(t_lb[t_lb < (1 / 4)], [-1, 1])
x_lb_2 = tf.reshape(x_lb[(t_lb >= (1 / 4)) & (t_lb <= (2 / 4))], [-1, 1])
t_lb_2 = tf.reshape(t_lb[(t_lb >= (1 / 4)) & (t_lb <= (2 / 4))], [-1, 1])
x_lb_3 = tf.reshape(x_lb[(t_lb >= (2 / 4)) & (t_lb <= (3 / 4))], [-1, 1])
t_lb_3 = tf.reshape(t_lb[(t_lb >= (2 / 4)) & (t_lb <= (3 / 4))], [-1, 1])
x_lb_4 = tf.reshape(x_lb[t_lb > (3 / 4)], [-1, 1])
t_lb_4 = tf.reshape(t_lb[t_lb > (3 / 4)], [-1, 1])

x_ub = tf.convert_to_tensor(X_ub[:, 0:1], dtype=tf.float32)
t_ub = tf.convert_to_tensor(X_ub[:, 1:2], dtype=tf.float32)

x_ub_1 = tf.reshape(x_ub[t_ub < (1 / 4)], [-1, 1])
t_ub_1 = tf.reshape(t_ub[t_ub < (1 / 4)], [-1, 1])
x_ub_2 = tf.reshape(x_ub[(t_ub >= (1 / 4)) & (t_ub <= (2 / 4))], [-1, 1])
t_ub_2 = tf.reshape(t_ub[(t_ub >= (1 / 4)) & (t_ub <= (2 / 4))], [-1, 1])
x_ub_3 = tf.reshape(x_ub[(t_ub >= (2 / 4)) & (t_ub <= (3 / 4))], [-1, 1])
t_ub_3 = tf.reshape(t_ub[(t_ub >= (2 / 4)) & (t_ub <= (3 / 4))], [-1, 1])
x_ub_4 = tf.reshape(x_ub[t_ub > (3 / 4)], [-1, 1])
t_ub_4 = tf.reshape(t_ub[t_ub > (3 / 4)], [-1, 1])

N_int = 100



x_f_int_1 = tf.reshape(tf.random.uniform([N_int], minval =-1.0, maxval = 1.0), [-1, 1])
t_f_int_1 = tf.reshape(np.repeat(tf.cast(0.25, tf.float32), N_int), [-1, 1])
x_f_int_2 = tf.reshape(tf.random.uniform([N_int], minval =-1.0, maxval = 1.0), [-1, 1])
t_f_int_2 = tf.reshape(np.repeat(tf.cast(0.50, tf.float32), N_int), [-1, 1])
x_f_int_3 = tf.reshape(tf.random.uniform([N_int], minval =-1.0, maxval = 1.0), [-1, 1])
t_f_int_3 = tf.reshape(np.repeat(tf.cast(0.75, tf.float32), N_int), [-1, 1])

# train loop
fit(x_f, t_f,
        x_f_int_1, t_f_int_1,
        x_f_int_2, t_f_int_2,
        x_f_int_3, t_f_int_3,
        x0, t0, u0,
        x_lb_1, t_lb_1,
        x_lb_2, t_lb_2,
        x_lb_3, t_lb_3,
        x_lb_4, t_lb_4,
        x_ub_1, t_ub_1,
        x_ub_2, t_ub_2,
        x_ub_3, t_ub_3,
        x_ub_4, t_ub_4, col_weights, u_weights, tf_iter=100000, newton_iter=1)
col_weights_vec.append(col_weights_iter)
u_weights_vec.append(u_weights_iter)
print(col_weights_vec)
print(u_weights_vec)

# filename = 'plot-col-weights.txt'
# # w tells python we are opening the file to write into it
# outfile = open(filename, 'a')
# outfile.write(str(col_weights_vec))
# outfile.close()  # Close the file when we’re done!
#
# filename = 'plot-u-weights.txt'
# # w tells python we are opening the file to write into it
# outfile = open(filename, 'a')
# outfile.write(str(u_weights_vec))
# outfile.close()  # Close the file when we’re done!

ax = sns.lineplot(data=np.mean(np.array(col_weights_vec), axis=0), color="blue", label="Collocation Weight Average")
sns.lineplot(data=(np.mean(np.array(col_weights_vec), axis=0) + np.std(np.array(col_weights_vec), axis=0)),
             color="blue", linestyle="--")
sns.lineplot(data=np.mean(np.array(col_weights_vec), axis=0) - np.std(np.array(col_weights_vec), axis=0), color="blue",
             linestyle="--")
sns.lineplot(data=np.mean(np.array(u_weights_vec), axis=0), label="Initial Weight Average", color="green")
sns.lineplot(data=np.mean(np.array(u_weights_vec) + np.std(np.array(u_weights_vec), axis=0), axis=0), color="green",
             linestyle="--")
sns.lineplot(data=np.mean(np.array(u_weights_vec) - np.std(np.array(u_weights_vec), axis=0), axis=0), color="green",
             linestyle="--", )
ax.set(xlabel="Training Iteration x100", ylabel="Weight Magnitude")
plt.show()

# generate meshgrid for forward pass of u_pred

X, T = np.meshgrid(x, t)

X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

# rearrange exact soln so it matches outputs from pred()
u_star = Exact_u.T.flatten()[:, None]
u_1 = u_star[X_star[:, 1] < (1 / 4)]
u_2 = u_star[(X_star[:, 1] >= (1 / 4)) & (X_star[:, 1] < (2 / 4))]
u_3 = u_star[(X_star[:, 1] >= (2 / 4)) & (X_star[:, 1] < (3 / 4))]
u_4 = u_star[X_star[:, 1] >= (3 / 4)]

u_star = np.concatenate([u_1, u_2, u_3, u_4], axis = 0)



lb = np.array([-1.0, 0.0])
ub = np.array([1.0, 1])

u_pred= predict(X_star)

error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)

print('Error u: %e' % (error_u))

U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')

# FU_pred = griddata(X_star, f_u_pred.flatten(), (X, T), method='cubic')

######################################################################
############################# Plotting ###############################
######################################################################

X0 = np.concatenate((x0, 0 * x0), 1)  # (x0, 0)
X_lb = np.concatenate((0 * tb + lb[0], tb), 1)  # (lb[0], tb)
X_ub = np.concatenate((0 * tb + ub[0], tb), 1)  # (ub[0], tb)
X_u_train = np.vstack([X0, X_lb, X_ub])

fig, ax = newfig(1.3, 1.0)
ax.axis('off')

####### Row 0: h(t,x) ##################
gs0 = gridspec.GridSpec(1, 2)
gs0.update(top=1 - 0.06, bottom=1 - 1 / 3, left=0.15, right=0.85, wspace=0)
ax = plt.subplot(gs0[:, :])

h = ax.imshow(U_pred.T, interpolation='nearest', cmap='YlGnBu',
              extent=[lb[1], ub[1], lb[0], ub[0]],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)

line = np.linspace(x.min(), x.max(), 2)[:, None]
ax.plot(t[25] * np.ones((2, 1)), line, 'k--', linewidth=1)
ax.plot(t[50] * np.ones((2, 1)), line, 'k--', linewidth=1)
ax.plot(t[75] * np.ones((2, 1)), line, 'k--', linewidth=1)

ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
leg = ax.legend(frameon=False, loc='best')
#    plt.setp(leg.get_texts(), color='w')
ax.set_title('$u(t,x)$', fontsize=10)

####### Row 1: h(t,x) slices ##################
gs1 = gridspec.GridSpec(1, 3)
gs1.update(top=1 - 1 / 3, bottom=0, left=0.1, right=0.9, wspace=0.5)

ax = plt.subplot(gs1[0, 0])
ax.plot(x, Exact_u[:, 25], 'b-', linewidth=2, label='Exact')
ax.plot(x, U_pred[25, :], 'r--', linewidth=2, label='Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.set_title('$t = %.2f$' % (t[25]), fontsize=10)
ax.axis('square')
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])

ax = plt.subplot(gs1[0, 1])
ax.plot(x, Exact_u[:, 50], 'b-', linewidth=2, label='Exact')
ax.plot(x, U_pred[50, :], 'r--', linewidth=2, label='Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.axis('square')
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])
ax.set_title('$t = %.2f$' % (t[50]), fontsize=10)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=5, frameon=False)

ax = plt.subplot(gs1[0, 2])
ax.plot(x, Exact_u[:, 75], 'b-', linewidth=2, label='Exact')
ax.plot(x, U_pred[75, :], 'r--', linewidth=2, label='Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.axis('square')
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])
ax.set_title('$t = %.2f$' % (t[75]), fontsize=10)

# show u_pred across domain
fig, ax = plt.subplots()

fig, ax = plt.subplots()

h = plt.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
               extent=[0.0, 1.0, -1.0, 1.0],
               origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)

plt.legend(frameon=False, loc='best')

plt.show()

fig, ax = plt.subplots()

ec = plt.imshow(FU_pred.T, interpolation='nearest', cmap='rainbow',
                extent=[0.0, math.pi / 2, -5.0, 5.0],
                origin='lower', aspect='auto')

# ax.add_collection(ec)
ax.autoscale_view()
ax.set_xlabel('$x$')
ax.set_ylabel('$t$')
cbar = plt.colorbar(ec)
cbar.set_label('$\overline{f}_u$ prediction')
plt.show()

plt.scatter(t_f, x_f, c=col_weights.numpy(), s=col_weights.numpy() / 10)
plt.show()
