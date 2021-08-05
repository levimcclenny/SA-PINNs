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
from tensorflow.python.ops.parallel_for.gradients import jacobian



#define size of the network
layer_sizes = [2, 128, 128, 128, 128, 1]

sizes_w = []
sizes_b = []
col_weights_vec = []
u_weights_vec = []
for i in range(1):
    for i, width in enumerate(layer_sizes):
        if i != 1:
            sizes_w.append(int(width * layer_sizes[1]))
            sizes_b.append(int(width if i != 0 else layer_sizes[1]))


    #L-BFGS weight getting and setting from https://github.com/pierremtb/PINNs-TF2.0
    def set_weights(model, w, sizes_w, sizes_b):
            for i, layer in enumerate(model.layers[0:]):
                start_weights = sum(sizes_w[:i]) + sum(sizes_b[:i])
                end_weights = sum(sizes_w[:i+1]) + sum(sizes_b[:i])
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

    #define the neural network model
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

    # AC Loss Model Below
    #
    # #define the loss
    # def loss(x_f_batch, t_f_batch,
    #              x0, t0, u0, x_lb,
    #              t_lb, x_ub, t_ub, col_weight, u_weight, b_weight, bx_weight):
    #
    #     f_u_pred = f_model(x_f_batch, t_f_batch)
    #     u0_pred = u_model(tf.concat([x0, t0],1))
    #
    #     u_lb_pred, u_x_lb_pred, = u_x_model(x_lb, t_lb)
    #     u_ub_pred, u_x_ub_pred, = u_x_model(x_ub, t_ub)
    #
    #     mse_0_u = (u_weight/(2*N0))*tf.reduce_mean(tf.square((u0 - u0_pred)))
    #
    #     mse_b_u_weights = (b_weight/(2*N_b))*tf.reduce_mean(tf.square(tf.math.subtract(u_lb_pred,u_ub_pred))) + \
    #                       (bx_weight/(2*N_b))*tf.reduce_mean(tf.square(tf.math.subtract(u_x_lb_pred, u_x_ub_pred)))
    #
    #     mse_b_u = tf.reduce_mean(tf.square(tf.math.subtract(u_lb_pred,u_ub_pred))) + \
    #             tf.reduce_mean(tf.square(tf.math.subtract(u_x_lb_pred, u_x_ub_pred)))
    #
    #     mse_f_u = (col_weight/(2*N_f))*tf.reduce_mean(tf.square(f_u_pred))
    #
    #     return  mse_0_u + mse_b_u_weights + mse_f_u , tf.reduce_mean(tf.square((u0 - u0_pred))), mse_b_u, tf.reduce_mean(tf.square(f_u_pred))

    def loss(x_f_batch, t_f_batch,
                 x0, t0, u0, x_lb,
                 t_lb, x_ub, t_ub, col_weight, u_weight, b_weight):

        f_u_pred = f_model(x_f_batch, t_f_batch)

        u0_pred, u0_t_pred = u_t_model(x0, t0)

        u_lb_pred, _ = u_t_model(x_lb, t_lb)
        u_ub_pred, _ = u_t_model(x_ub, t_ub)

        mse_b_u = (b_weight/(2*N_b))*(tf.reduce_mean(tf.square(u_lb_pred - 0)) + tf.reduce_mean(tf.square(u_ub_pred - 0)))

        mse_0_u = (u_weight/(2*N0))*(tf.reduce_mean(tf.square((u0 - u0_pred))) + tf.reduce_mean(tf.square((u0_t_pred - 0.0))))

        mse_f_u = (col_weight/(2*N_f))*tf.reduce_mean(tf.square(f_u_pred))

        return mse_0_u + mse_b_u + mse_f_u, mse_0_u, mse_b_u, mse_f_u

    def loss_weights(x_f_batch, t_f_batch,
                 x0, t0, u0, x_lb,
                 t_lb, x_ub, t_ub):

        f_u_pred = f_model(x_f_batch, t_f_batch)

        u0_pred, u0_t_pred = u_t_model(x0, t0)

        u_lb_pred, _ = u_t_model(x_lb, t_lb)
        u_ub_pred, _ = u_t_model(x_ub, t_ub)

        mse_b_u = tf.reduce_mean(tf.square(u_lb_pred - 0)) + \
                  tf.reduce_mean(tf.square(u_ub_pred - 0))  # since ub/lb is 0

        mse_0_u = tf.reduce_mean(tf.square((u0 - u0_pred))) + \
                  tf.reduce_mean(tf.square((u0_t_pred - 0.0)))

        mse_f_u = tf.reduce_mean(tf.square(f_u_pred))

        return mse_0_u, mse_b_u, mse_f_u

    #define the physics-based residual, we want this to be 0



    @tf.function
    def f_model(x,t):
        u = u_model(tf.concat([x, t],1))
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_t = tf.gradients(u,t)[0]
        u_tt = tf.gradients(u_t,t)[0]
        f_u = u_tt - 4*u_xx
        return f_u

    @tf.function
    def u_t_model(x, t):
        u = u_model(tf.concat([x, t],1))
        u_t = tf.gradients(u, t)[0]
        return u, u_t

    @tf.function
    def grad(model, x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch, x_lb, t_lb, x_ub, t_ub, col_weight, u_weight, b_weight):
        with tf.GradientTape(persistent=True) as tape:
            loss_value, mse_0, mse_b, mse_f = loss(x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch, x_lb, t_lb, x_ub, t_ub, col_weight, u_weight, b_weight)
            grads = tape.gradient(loss_value, u_model.trainable_variables)
            del tape
        return loss_value, mse_0, mse_b, mse_f, grads

    @tf.function
    def grad_weights(model, x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch, x_lb, t_lb, x_ub, t_ub):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(u_model.trainable_variables)
            mse_0, mse_b, mse_f = loss_weights(x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch, x_lb, t_lb, x_ub, t_ub)
        gradients_u = tape.jacobian(mse_0, u_model.trainable_variables)
        gradients_f = tape.jacobian(mse_f, u_model.trainable_variables)
        gradients_b = tape.jacobian(mse_b, u_model.trainable_variables)
        return gradients_u, gradients_f, gradients_b


    def fit(x_f, t_f, x0, t0, u0, x_lb, t_lb, x_ub, t_ub, tf_iter, newton_iter):

        #Can adjust batch size for collocation points, here we set it to N_f
        batch_sz = N_f
        n_batches =  N_f // batch_sz

        start_time = time.time()
        #create optimizer s for the network weights, collocation point mask, and initial boundary mask
        tf_optimizer = tf.keras.optimizers.Adam(lr = 0.0001, beta_1=.99)

        print("starting Adam training")

        b_weight = 1.0
        bx_weight = 1.0
        u_weight = 1.0
        col_weight = 1.0


        # For mini-batch (if used)
        for epoch in range(tf_iter):
            for i in range(n_batches):

                x0_batch = x0
                t0_batch = t0
                u0_batch = u0

                x_f_batch = x_f[i*batch_sz:(i*batch_sz + batch_sz),]
                t_f_batch = t_f[i*batch_sz:(i*batch_sz + batch_sz),]

                if (epoch+1) % 100 == 0:
                    grads_u, grads_f, grads_b = grad_weights(u_model, x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch, x_lb,
                                                     t_lb, x_ub, t_ub)
                    # print(grads_f)
                    sum_u = np.sum([np.sum(l.numpy()**2) for l in grads_u])
                    print("sum_u")
                    sum_b = np.sum([np.sum(l.numpy()**2) for l in grads_b])
                    print("sum_b", sum_b)
                    sum_f = np.sum([np.sum(l.numpy()**2) for l in grads_f])
                    print("sum_f", sum_f)
                    sum = sum_u + sum_b + sum_f
                    print("sum", sum)
                    b_weight = sum/sum_b
                    print("b_weight", b_weight)
                    u_weight = sum/sum_u
                    col_weight = sum/sum_f

                loss_value, mse_0, mse_b, mse_f, grads = grad(u_model, x_f_batch, t_f_batch, x0_batch, t0_batch,  u0_batch, x_lb, t_lb, x_ub, t_ub, col_weight, u_weight, b_weight)

                tf_optimizer.apply_gradients(zip(grads, u_model.trainable_variables))



            if epoch % 100 == 0:
                elapsed = time.time() - start_time
                print('It: %d, Time: %.2f' % (epoch, elapsed))
                tf.print(f"mse_0: {mse_0}  mse_b  {mse_b}  mse_f: {mse_f}   total loss: {loss_value}")
                tf.print(b_weight, bx_weight, u_weight, col_weight)
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

        #l-bfgs-b optimization
        print("Starting L-BFGS training")

        loss_and_flat_grad = get_loss_and_flat_grad(x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch, x_lb, t_lb, x_ub, t_ub, col_weight, u_weight, b_weight, bx_weight)

        lbfgs(loss_and_flat_grad,
          get_weights(u_model),
          Struct(), maxIter=newton_iter, learningRate=0.8)


    #L-BFGS implementation from https://github.com/pierremtb/PINNs-TF2.0
    def get_loss_and_flat_grad(x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch, x_lb, t_lb, x_ub, t_ub, col_weight, u_weight, b_weight, bx_weight):
        def loss_and_flat_grad(w):
            with tf.GradientTape() as tape:
                set_weights(u_model, w, sizes_w, sizes_b)
                loss_value, _, _, _ = loss(x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch, x_lb, t_lb, x_ub, t_ub, col_weight, u_weight, b_weight, bx_weight)
            grad = tape.gradient(loss_value, u_model.trainable_variables)
            grad_flat = []
            for g in grad:
                grad_flat.append(tf.reshape(g, [-1]))
            grad_flat = tf.concat(grad_flat, 0)
            #print(loss_value, grad_flat)
            return loss_value, grad_flat

        return loss_and_flat_grad


    def predict(X_star):
        X_star = tf.convert_to_tensor(X_star, dtype=tf.float32)
        u_star, _ = u_t_model(X_star[:,0:1],
                         X_star[:,1:2])

        f_u_star = f_model(X_star[:,0:1],
                     X_star[:,1:2])

        return u_star.numpy(), f_u_star.numpy()



    # Define constants and weight vectors

    col_weights_iter = []
    u_weights_iter = []
    lb = np.array([-1.0])
    ub = np.array([1.0])

    N0 = 512
    N_b = 200
    N_f = 10000

    col_weights = tf.Variable(tf.random.uniform([N_f, 1]))
    u_weights = tf.Variable(tf.random.uniform([N0, 1]))

    #initialize the NN
    u_model = neural_net(layer_sizes)

    #view the NN
    u_model.summary()

    # Import data, same data as Raissi et al

    data = scipy.io.loadmat('AC.mat')

    t = data['tt'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = data['uu']
    Exact_u = np.real(Exact)



    #grab training points from domain
    idx_x = np.random.choice(x.shape[0], N0, replace=False)
    x0 = x[idx_x,:]
    u0 = tf.cast(Exact_u[idx_x,0:1], dtype = tf.float32)

    idx_t = np.random.choice(t.shape[0], N_b, replace=False)
    tb = t[idx_t,:]

    # Grab collocation points using latin hpyercube sampling

    X_f = lb + (ub-lb)*lhs(2, N_f)

    x_f = tf.convert_to_tensor(X_f[:,0:1], dtype=tf.float32)
    t_f = tf.convert_to_tensor(np.abs(X_f[:,1:2]), dtype=tf.float32)


    X0 = np.concatenate((x0, 0*x0), 1) # (x0, 0)
    X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
    X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)

    x0 = tf.cast(X0[:,0:1], dtype = tf.float32)
    t0 = tf.cast(X0[:,1:2], dtype = tf.float32)

    x_lb = tf.convert_to_tensor(X_lb[:,0:1], dtype=tf.float32)
    t_lb = tf.convert_to_tensor(X_lb[:,1:2], dtype=tf.float32)

    x_ub = tf.convert_to_tensor(X_ub[:,0:1], dtype=tf.float32)
    t_ub = tf.convert_to_tensor(X_ub[:,1:2], dtype=tf.float32)


    #train loop
    fit(x_f, t_f, x0, t0, u0, x_lb, t_lb, x_ub, t_ub, tf_iter = 20000, newton_iter = 1)
    col_weights_vec.append(col_weights_iter)
    u_weights_vec.append(u_weights_iter)

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

# ax = sns.lineplot(data=np.mean(np.array(col_weights_vec), axis=0), color = "blue", label="Collocation Weight Average")
# sns.lineplot(data=(np.mean(np.array(col_weights_vec), axis = 0)+np.std(np.array(col_weights_vec), axis=0)),color = "blue", linestyle = "--")
# sns.lineplot(data=np.mean(np.array(col_weights_vec), axis = 0)-np.std(np.array(col_weights_vec), axis=0),color = "blue", linestyle = "--")
# sns.lineplot(data=np.mean(np.array(u_weights_vec), axis =0), label="Initial Weight Average", color = "green")
# sns.lineplot(data=np.mean(np.array(u_weights_vec)+np.std(np.array(u_weights_vec), axis=0), axis =0), color = "green", linestyle ="--")
# sns.lineplot(data=np.mean(np.array(u_weights_vec)-np.std(np.array(u_weights_vec), axis=0), axis =0), color = "green", linestyle ="--", )
# ax.set(xlabel="Training Iteration x100", ylabel="Weight Magnitude")
# plt.show()

#generate meshgrid for forward pass of u_pred

X, T = np.meshgrid(x,t)

X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = Exact_u.T.flatten()[:,None]

lb = np.array([-1.0, 0.0])
ub = np.array([1.0, 1])

u_pred, f_u_pred = predict(X_star)

error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)

print('Error u: %e' % (error_u))


U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')

FU_pred = griddata(X_star, f_u_pred.flatten(), (X, T), method='cubic')

######################################################################
############################# Plotting ###############################
######################################################################

X0 = np.concatenate((x0, 0*x0), 1) # (x0, 0)
X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)
X_u_train = np.vstack([X0, X_lb, X_ub])

fig, ax = newfig(1.3, 1.0)
ax.axis('off')

####### Row 0: h(t,x) ##################
gs0 = gridspec.GridSpec(1, 2)
gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
ax = plt.subplot(gs0[:, :])

h = ax.imshow(U_pred.T, interpolation='nearest', cmap='YlGnBu',
              extent=[lb[1], ub[1], lb[0], ub[0]],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)


line = np.linspace(x.min(), x.max(), 2)[:,None]
ax.plot(t[25]*np.ones((2,1)), line, 'k--', linewidth = 1)
ax.plot(t[50]*np.ones((2,1)), line, 'k--', linewidth = 1)
ax.plot(t[75]*np.ones((2,1)), line, 'k--', linewidth = 1)

ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
leg = ax.legend(frameon=False, loc = 'best')
#    plt.setp(leg.get_texts(), color='w')
ax.set_title('$u(t,x)$', fontsize = 10)

####### Row 1: h(t,x) slices ##################
gs1 = gridspec.GridSpec(1, 3)
gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)

ax = plt.subplot(gs1[0, 0])
ax.plot(x,Exact_u[:,25], 'b-', linewidth = 2, label = 'Exact')
ax.plot(x,U_pred[25,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.set_title('$t = %.2f$' % (t[25]), fontsize = 10)
ax.axis('square')
ax.set_xlim([-1.1,1.1])
ax.set_ylim([-1.1,1.1])

ax = plt.subplot(gs1[0, 1])
ax.plot(x,Exact_u[:,50], 'b-', linewidth = 2, label = 'Exact')
ax.plot(x,U_pred[50,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.axis('square')
ax.set_xlim([-1.1,1.1])
ax.set_ylim([-1.1,1.1])
ax.set_title('$t = %.2f$' % (t[50]), fontsize = 10)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=5, frameon=False)

ax = plt.subplot(gs1[0, 2])
ax.plot(x,Exact_u[:,75], 'b-', linewidth = 2, label = 'Exact')
ax.plot(x,U_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.axis('square')
ax.set_xlim([-1.1,1.1])
ax.set_ylim([-1.1,1.1])
ax.set_title('$t = %.2f$' % (t[75]), fontsize = 10)

#show u_pred across domain
fig, ax = plt.subplots()

fig, ax = plt.subplots()

h = plt.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
            extent=[0.0, 1.0, -1.0, 1.0],
            origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)

plt.legend(frameon=False, loc = 'best')

plt.show()

fig, ax = plt.subplots()
192
ec = plt.imshow(FU_pred.T, interpolation='nearest', cmap='rainbow',
            extent=[0.0, math.pi/2, -5.0, 5.0],
            origin='lower', aspect='auto')

#ax.add_collection(ec)
ax.autoscale_view()
ax.set_xlabel('$x$')
ax.set_ylabel('$t$')
cbar = plt.colorbar(ec)
cbar.set_label('$\overline{f}_u$ prediction')
plt.show()

plt.scatter(t_f, x_f, c = col_weights.numpy(), s = col_weights.numpy()/10)
plt.show()
