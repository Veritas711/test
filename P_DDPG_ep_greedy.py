import os
os.chdir("C:/Users/gri4/Project")
import numpy as np
import tensorflow as tf
from tensorflow.initializers import random_uniform
import random

class OUActionNoise(object):
    def __init__(self, mu, sigma = 0.15, theta = 0.2, dt = 1e-2, x0 = None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()
        
    def __call__(self):
        x = self.x_prev + self.theta*(self.mu - self.x_prev)*self.dt + \
            self.sigma*np.sqrt(self.dt)*np.random.normal(size = self.mu.shape)
        self.x_prev = x
        return x
    
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
        
        
class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory= []

        self.new_state_memory = []

        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype = np.float32)
        
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        
        self.state_memory.append(state)
        self.new_state_memory.append(state_)

        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = 1 - int(done) # We do not want to count the rewards after the episode has ended
        self.mem_cntr += 1
        
    def sample_buffer(self, batch_size):

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        
        batch_state = []
        batch_new_state = []
        for i in batch:
            batch_state.append(self.state_memory[i])
            batch_new_state.append(self.new_state_memory[i])
                
        states_feat, states_adj, states_deg = [], [], []
        
        for i in batch_state:
            states_feat.append(i[0])
            states_adj.append(i[1])
            states_deg.append(i[2])
            
        new_states_feat, new_states_adj, new_states_deg = [], [], []

        for i in batch_new_state:
            new_states_feat.append(i[0])
            new_states_adj.append(i[1])
            new_states_deg.append(i[2])

        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]
         
        return [states_feat, states_adj, states_deg], actions, [new_states_feat, new_states_adj, new_states_deg], rewards, terminal
    

class Actor(object):
    def __init__(self, lr, n_actions, name, input_dims, sess, fc1_dims, fc2_dims, batch_size = 64, chkpt_dir = 'DDPG', use_agg = True):
        self.lr = lr
        self.n_actions =n_actions
        self.name = name
        self.fc1_dims = 45
        self.fc2_dims = 25
        self.sess = sess
        self.input_dims = input_dims
        #self.action_bound = action_bound
        self.batch_size = batch_size
        self.ckpt_dir = chkpt_dir
        self.use_aggregator = use_agg
        self.build_network()
        self.params = tf.trainable_variables(scope = self.name)
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(chkpt_dir, name+"_ddpg.ckpt")
        self.unnormalized_actor_gradients = tf.gradients(self.mu, self.params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))
        #self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(self.actor_gradients, self.params))
        self.optimize = tf.train.AdagradOptimizer(self.lr).apply_gradients(zip(self.actor_gradients, self.params))

        
    def build_network(self, use_aggregator = True):
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32, shape = [None, *self.input_dims], name = "Inputs_features")
            self.action_gradient = tf.placeholder(tf.float32, shape = [None, self.n_actions])
            
            if self.use_aggregator:
                self.adjacency = tf.placeholder(tf.float32, shape = [None, 117, 117], name = "Adjacency_matrix")
                self.degree = tf.placeholder(tf.float32, shape = [None, 117, 117], name = "Degree_matrix")
                        
                conv1 = tf.layers.Conv1D(150, 1, activation = "linear")(self.input)
                a_1 = tf.matmul(self.adjacency, conv1)
                d_1 = tf.matmul(self.degree, a_1)
                act_1 = tf.nn.relu(d_1)
            
                conv_2 = tf.layers.Conv1D(75, 1, activation = "linear")(act_1)
                a_2 = tf.matmul(self.adjacency, conv_2)
                d_2 = tf.matmul(self.degree, a_2)
                act_2 = tf.nn.relu(d_2)
            
                gmp = tf.keras.layers.GlobalMaxPooling1D()(act_2)
            
                f1 = 1/np.sqrt(self.fc1_dims)
                dense1 = tf.layers.dense(gmp, units = self.fc1_dims, kernel_initializer = random_uniform(-f1,f1), bias_initializer = random_uniform(-f1,f1))
                batch1 = tf.layers.batch_normalization(dense1)
                layer1_activation = tf.nn.relu(batch1)
            
                f2 = 1/np.sqrt(self.fc2_dims)
                
                dense2 = tf.layers.dense(layer1_activation, units = self.fc2_dims, kernel_initializer = random_uniform(-f2,f2), bias_initializer = random_uniform(-f2,f2))
                batch2 = tf.layers.batch_normalization(dense2)
                layer2_activation = tf.nn.relu(batch2)
            
                f3 = 0.003
                self.mu = tf.layers.dense(layer2_activation, units = self.n_actions, activation = 'linear', kernel_initializer = random_uniform(-f3,f3), bias_initializer = random_uniform(-f3,f3))
                #self.mu = tf.multiply(mu, self.action_bound)
                
            else:
                
                f1 = 1/np.sqrt(self.fc1_dims)
                dense1 = tf.layers.dense(self.input, units = self.fc1_dims, kernel_initializer = random_uniform(-f1,f1), bias_initializer = random_uniform(-f1,f1))
                batch1 = tf.layers.batch_normalization(dense1)
                layer1_activation = tf.nn.relu(batch1)
            
                f2 = 1/np.sqrt(self.fc2_dims)
                
                dense2 = tf.layers.dense(layer1_activation, units = self.fc2_dims, kernel_initializer = random_uniform(-f2,f2), bias_initializer = random_uniform(-f2,f2))
                batch2 = tf.layers.batch_normalization(dense2)
                layer2_activation = tf.nn.relu(batch2)
            
                f3 = 0.003
                self.mu = tf.layers.dense(layer2_activation, units = self.n_actions, activation = 'linear', kernel_initializer = random_uniform(-f3,f3), bias_initializer = random_uniform(-f3,f3))
               #self.mu = tf.multiply(mu, self.action_bound)
            
        
    def predict(self, inputs):
        if self.use_aggregator:
            inp, adj_mat, deg_mat = inputs
            return self.sess.run(self.mu, feed_dict = {self.input: inp, self.adjacency: adj_mat, self.degree: deg_mat})
        else:
            return self.sess.run(self.mu, feed_dict = {self.input: inputs})            
    
    def train(self, inputs, gradients):
        if self.use_aggregator:
            inp, adj_mat, deg_mat = inputs
            self.sess.run(self.optimize, feed_dict= {self.input: inp, self.adjacency: adj_mat, self.degree: deg_mat, self.action_gradient: gradients})
        else:
            self.sess.run(self.optimize, feed_dict= {self.input: inputs, self.action_gradient: gradients})

        
    def save_checkpoint(self):
        print("... Saving checkpoint ...")
        self.saver.save(self.sess, self.checkpoint_file)
        
    def load_checkpoint(self):
        print("... Loading checkpoint ...")
        self.saver.restore(self.sess, self.checkpoint_file)
        
class Critic(object):
    def __init__(self, lr, n_actions, name, input_dims, sess, fc1_dims, fc2_dims, chkpt_dir = 'DDPG', use_agg = True):
        self.lr = lr
        self.n_actions =n_actions
        self.name = name
        self.fc1_dims = 60
        self.fc2_dims = 30
        self.sess = sess
        self.input_dims = input_dims
        self.ckpt_dir = chkpt_dir
        self.use_aggregator = use_agg
        self.build_network()
        self.params = tf.trainable_variables(scope = self.name)
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(chkpt_dir, name+"_ddpg.ckpt")
        #self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.optimize = tf.train.AdagradOptimizer(self.lr).minimize(self.loss)
        self.action_gradients = tf.gradients(self.q, self.actions)

        
    def build_network(self):
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32, shape = [None, *self.input_dims], name = "inputs")
            self.actions = tf.placeholder(tf.float32, shape = [None, self.n_actions], name = "actions")
            self.q_target = tf.placeholder(tf.float32, shape = [None, 1], name = 'targets')
            
            if self.use_aggregator:
                self.adjacency = tf.placeholder(tf.float32, shape = [None, 117, 117], name = "Adjacency_matrix")
                self.degree = tf.placeholder(tf.float32, shape = [None, 117, 117], name = "Degree_matrix")
                        
                conv1 = tf.layers.Conv1D(200, 1, activation = "linear")(self.input)
                a_1 = tf.matmul(self.adjacency, conv1)
                d_1 = tf.matmul(self.degree, a_1)
                act_1 = tf.nn.relu(d_1)
            
                conv_2 = tf.layers.Conv1D(100, 1, activation = "linear")(act_1)
                a_2 = tf.matmul(self.adjacency, conv_2)
                d_2 = tf.matmul(self.degree, a_2)
                act_2 = tf.nn.relu(d_2)
            
                gmp = tf.keras.layers.GlobalMaxPooling1D()(act_2)
                
                f1 = 1/np.sqrt(self.fc1_dims)
                dense1 = tf.layers.dense(gmp, units = self.fc1_dims, kernel_initializer = random_uniform(-f1,f1), bias_initializer = random_uniform(-f1,f1))
                batch1 = tf.layers.batch_normalization(dense1)
                layer1_activation = tf.nn.relu(batch1)
            
                f2 = 1/np.sqrt(self.fc2_dims)
                dense2 = tf.layers.dense(layer1_activation, units = self.fc2_dims, kernel_initializer = random_uniform(-f2,f2), bias_initializer = random_uniform(-f2,f2))
                batch2 = tf.layers.batch_normalization(dense2)
            
                action_in = tf.layers.dense(self.actions, units = self.fc2_dims, activation = 'relu')
            
                state_actions = tf.add(batch2, action_in)
                state_actions = tf.nn.relu(state_actions)
            
                f3 = 0.003
                self.q = tf.layers.dense(state_actions, units = 1, kernel_initializer = random_uniform(-f3,f3), bias_initializer = random_uniform(-f3,f3), kernel_regularizer = tf.keras.regularizers.l2(0.01))
            
            else:
                f1 = 1/np.sqrt(self.fc1_dims)
                dense1 = tf.layers.dense(self.input, units = self.fc1_dims, kernel_initializer = random_uniform(-f1,f1), bias_initializer = random_uniform(-f1,f1))
                batch1 = tf.layers.batch_normalization(dense1)
                layer1_activation = tf.nn.relu(batch1)
            
                f2 = 1/np.sqrt(self.fc2_dims)
                dense2 = tf.layers.dense(layer1_activation, units = self.fc2_dims, kernel_initializer = random_uniform(-f2,f2), bias_initializer = random_uniform(-f2,f2))
                batch2 = tf.layers.batch_normalization(dense2)
            
                action_in = tf.layers.dense(self.actions, units = self.fc2_dims, activation = 'relu')
            
                state_actions = tf.add(batch2, action_in)
                state_actions = tf.nn.relu(state_actions)
            
                f3 = 0.003
                self.q = tf.layers.dense(state_actions, units = 1, kernel_initializer = random_uniform(-f3,f3), bias_initializer = random_uniform(-f3,f3), kernel_regularizer = tf.keras.regularizers.l2(0.01))
            
            self.loss = tf.losses.mean_squared_error(self.q_target, self.q)
            
    def predict(self, inputs, actions):
        if self.use_aggregator:
            inp, adj_mat, deg_mat = inputs
            return self.sess.run(self.q, feed_dict = {self.input: inp, self.adjacency: adj_mat, self.degree: deg_mat, self.actions: actions})
        else:
            return self.sess.run(self.q, feed_dict = {self.input: inputs, self.actions: actions})
    
    def train(self, inputs, actions, q_target):
        if self.use_aggregator:
            inp, adj_mat, deg_mat = inputs
            return self.sess.run(self.optimize, feed_dict = {self.input: inp, self.adjacency: adj_mat, self.degree: deg_mat, self.actions: actions, self.q_target: q_target})
        else:
            return self.sess.run(self.optimize, feed_dict = {self.input: inputs, self.actions: actions, self.q_target: q_target})
    
    def get_action_gradients(self, inputs, actions):
        if self.use_aggregator:
            inp, adj_mat, deg_mat = inputs
            return self.sess.run(self.action_gradients, feed_dict = {self.input: inp, self.adjacency: adj_mat, self.degree: deg_mat, self.actions: actions})
        else:
            return self.sess.run(self.action_gradients, feed_dict = {self.input: inputs, self.actions: actions})
    
    def save_checkpoint(self):
        print("... Saving checkpoint ...")
        self.saver.save(self.sess, self.checkpoint_file)
        
    def load_checkpoint(self):
        print("... Loading checkpoint ...")
        self.saver.restore(self.sess, self.checkpoint_file)
        

class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, env, gamma = 0.99, n_actions = 2, max_size = 100000, layer1_size = 400, layer2_size = 300, batch_size = 64, use_agg = True):
        self.gamma = gamma
        self.tau = tau
        if use_agg:
            self.memory = ReplayBuffer(max_size, [3], n_actions)
        else:
            self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.sess= tf.Session() # One session for all 4 networks.
        self.actor = Actor(alpha, n_actions, 'Actor', input_dims, self.sess, layer1_size, layer2_size, batch_size = batch_size, use_agg = use_agg)
        self.critic = Critic(beta, n_actions, 'Critic', input_dims, self.sess, layer1_size, layer2_size, use_agg = use_agg)
        self.target_actor = Actor(alpha, n_actions, 'TargetActor', input_dims, self.sess, layer1_size, layer2_size, batch_size = batch_size, use_agg = use_agg)
        self.target_critic = Critic(beta, n_actions, 'TargetCritic', input_dims, self.sess, layer1_size, layer2_size, use_agg = use_agg)
        
        self.noise = OUActionNoise(mu = np.zeros(n_actions))
        self.update_critic = [self.target_critic.params[i].assign(tf.multiply(self.critic.params[i], self.tau)\
                              + tf.multiply(self.target_critic.params[i], 1. - self.tau)) for i in range(len(self.target_critic.params))]
        self.update_actor = [self.target_actor.params[i].assign(tf.multiply(self.actor.params[i], self.tau)\
                              + tf.multiply(self.target_actor.params[i], 1. - self.tau)) for i in range(len(self.target_actor.params))]
        
        self.sess.run(tf.global_variables_initializer())

        self.update_network_parameters(first = True)
        self.epsilon = 0.4
        self.max_epsilon = 0.4
        self.min_epsilon = 0.01
        self.decay_rate = 0.0001
        
    def update_network_parameters(self, first = False):
        if first:
            old_tau = self.tau
            self.tau = 1.0
            self.target_critic.sess.run(self.update_critic)
            self.target_actor.sess.run(self.update_actor)
            self.tau = old_tau
        else:
            self.target_critic.sess.run(self.update_critic)
            self.target_actor.sess.run(self.update_actor)
    
    def remember(self, state, actions, rewards, new_state, done):
        self.memory.store_transition(state, actions, rewards, new_state, done)
    
    def choose_action(self, state, epi):
        #state = state[np.newaxis, :]
        #state = np.expand_dims(state, axis=0)
        mu = self.actor.predict(state)
        noise = self.noise()
        ep = np.random.random()
        
        if epi % 500 == 0:
            mu_prime = [20, 20, 0.3, 0.9, -4]
            noise = True
            return mu_prime, noise
        
        if ep < self.epsilon:
            P_x = random.randint(-100,100)
            P_y = random.randint(-100,100)
            t1 = random.randint(1,9)/10
            t2 = random.randint(1,9)/10
            d = random.randint(1,5)
            r = random.choice([1,-1])
            d = r*d
            mu_prime = [P_x, P_y, t1, t2, d]
            #mu_prime = [20, 20, 0.3, 0.9, -4]
            noise = True
        else:
            mu_prime1 = mu + noise
            mu_prime1 = mu_prime1[0]
            mu_prime = [mu_prime1[0]*10, mu_prime1[1]*10, mu_prime1[2], mu_prime1[3], mu_prime1[4]*10]
            
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon)*np.exp(-self.decay_rate*epi)
        
        return mu_prime, noise
    
    def test_choose_action(self, state):
        #state = state[np.newaxis, :]
        #state = np.expand_dims(state, axis=0)
        mu = self.actor.predict(state)
        
        return mu[0]
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return 
        state, action, new_state, reward, done = self.memory.sample_buffer(self.batch_size)

        critic_value_ = self.target_critic.predict(new_state, self.target_actor.predict(new_state))
        
        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j]*done[j])
        
        target = np.reshape(target, (self.batch_size,1))
        
        _ = self.critic.train(state, action, target)
        
        a_outs = self.actor.predict(state)
        grads = self.critic.get_action_gradients(state, a_outs)
        self.actor.train(state, grads[0])
        
        self.update_network_parameters()
    
    def check_update(self, new_state, reward, done):

        critic_value_ = self.target_critic.predict(new_state, self.target_actor.predict(new_state))
        target = reward + self.gamma*critic_value_*done
        return [target, critic_value_]
        
    
    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()
    
    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()