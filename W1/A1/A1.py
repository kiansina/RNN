#Building your Recurrent Neural Network - Step by Step
"""
Welcome to Course 5's first assignment, where you'll be implementing key components of a Recurrent Neural Network, or
RNN, in NumPy!

By the end of this assignment, you'll be able to:

. Define notation for building sequence models
. Describe the architecture of a basic RNN
. Identify the main components of an LSTM
. Implement backpropagation through time for a basic RNN and an LSTM
. Give examples of several types of RNN

Recurrent Neural Networks (RNN) are very effective for Natural Language Processing and other sequence tasks because they
have "memory." They can read inputs  𝑥⟨𝑡⟩ (such as words) one at a time, and remember some contextual information
through the hidden layer activations that get passed from one time step to the next. This allows a unidirectional
(one-way) RNN to take information from the past to process later inputs. A bidirectional (two-way) RNN can take context
from both the past and the future, much like Marty McFly.

Notation:

. Superscript  [𝑙] denotes an object associated with the  𝑙𝑡ℎ layer.

. Superscript  (𝑖) denotes an object associated with the  𝑖𝑡ℎ example.

. Superscript  ⟨𝑡⟩ denotes an object at the  𝑡𝑡ℎ time step.

. Subscript  𝑖 denotes the  𝑖𝑡ℎ entry of a vector.

Example:

𝑎(2)[3]<4>5 denotes the activation of the 2nd training example (2), 3rd layer [3], 4th time step <4>, and 5th entry in
the vector.
"""


#Packages

import numpy as np
from rnn_utils import *
from public_tests import *

#1 - Forward Propagation for the Basic Recurrent Neural Network
"""
Later this week, you'll get a chance to generate music using an RNN! The basic RNN that you'll implement has the
following structure:

In this example,  𝑇𝑥=𝑇𝑦.
"""

#Dimensions of input  𝑥
"""
Input with  𝑛𝑥 number of units

. For a single time step of a single input example,  𝑥(𝑖)⟨𝑡⟩ is a one-dimensional input vector
. Using language as an example, a language with a 5000-word vocabulary could be one-hot encoded into a vector that has
  5000 units. So  𝑥(𝑖)⟨𝑡⟩ would have the shape (5000,)
. The notation  𝑛𝑥 is used here to denote the number of units in a single time step of a single training example
"""


#Time steps of size  𝑇𝑥
"""
. A recurrent neural network has multiple time steps, which you'll index with  𝑡.
. In the lessons, you saw a single training example  𝑥(𝑖) consisting of multiple time steps  𝑇𝑥. In this notebook,  𝑇𝑥
  will denote the number of timesteps in the longest sequence.
"""

#Batches of size  𝑚
 """
. Let's say we have mini-batches, each with 20 training examples
. To benefit from vectorization, you'll stack 20 columns of  𝑥(𝑖) examples
. For example, this tensor has the shape (5000,20,10)
. You'll use  𝑚 to denote the number of training examples
. So, the shape of a mini-batch is  (𝑛𝑥,𝑚,𝑇𝑥)
"""


#3D Tensor of shape  (𝑛𝑥,𝑚,𝑇𝑥)
"""
. The 3-dimensional tensor  𝑥 of shape  (𝑛𝑥,𝑚,𝑇𝑥) represents the input  𝑥 that is fed into the RNN
"""


#Taking a 2D slice for each time step:  𝑥⟨𝑡⟩
"""
. At each time step, you'll use a mini-batch of training examples (not just a single example)
. So, for each time step  𝑡, you'll use a 2D slice of shape  (𝑛𝑥,𝑚)
. This 2D slice is referred to as  𝑥⟨𝑡⟩. The variable name in the code is xt.
"""


#Definition of hidden state  𝑎
"""
. The activation  𝑎⟨𝑡⟩ that is passed to the RNN from one time step to another is called a "hidden state."
"""

#Dimensions of hidden state  𝑎
"""
. Similar to the input tensor  𝑥, the hidden state for a single training example is a vector of length  𝑛𝑎

. If you include a mini-batch of  𝑚 training examples, the shape of a mini-batch is  (𝑛𝑎,𝑚)

. When you include the time step dimension, the shape of the hidden state is  (𝑛𝑎,𝑚,𝑇𝑥)

. You'll loop through the time steps with index  𝑡, and work with a 2D slice of the 3D tensor

. This 2D slice is referred to as  𝑎⟨𝑡⟩

. In the code, the variable names used are either a_prev or a_next, depending on the function being implemented

. The shape of this 2D slice is  (𝑛𝑎,𝑚)
"""

#Dimensions of prediction  𝑦̂
"""
. Similar to the inputs and hidden states,  𝑦̂ is a 3D tensor of shape  (𝑛𝑦,𝑚,𝑇𝑦)
  - 𝑛𝑦: number of units in the vector representing the prediction
  - 𝑚: number of examples in a mini-batch
  - 𝑇𝑦: number of time steps in the prediction

. For a single time step  𝑡, a 2D slice  𝑦̂ ⟨𝑡⟩ has shape  (𝑛𝑦,𝑚)
. In the code, the variable names are:
  - y_pred:  𝑦̂
  - yt_pred:  𝑦̂ ⟨𝑡⟩
"""



"""
Here's how you can implement an RNN:

Steps:
1. Implement the calculations needed for one time step of the RNN.
2. Implement a loop over  𝑇𝑥 time steps in order to process all the inputs, one at a time.
"""

#1.1 - RNN Cell
"""
You can think of the recurrent neural network as the repeated use of a single cell. First, you'll implement the
computations for a single time step. The following figure describes the operations for a single time step of an RNN
cell:

Figure 2: Basic RNN cell. Takes as input  𝑥⟨𝑡⟩ (current input) and  𝑎⟨𝑡−1⟩ (previous hidden state containing
         information from the past), and outputs 𝑎⟨𝑡⟩ which is given to the next RNN cell and also used to predict 𝑦̂⟨𝑡⟩



RNN cell versus RNN_cell_forward:

. Note that an RNN cell outputs the hidden state  𝑎⟨𝑡⟩.
  - RNN cell is shown in the figure as the inner box with solid lines

. The function that you'll implement, rnn_cell_forward, also calculates the prediction  𝑦̂ ⟨𝑡⟩
  - RNN_cell_forward is shown in the figure as the outer box with dashed lines
"""


#Exercise 1 - rnn_cell_forward
"""
Implement the RNN cell described in Figure 2.

Instructions:

1. Compute the hidden state with tanh activation:  𝑎⟨𝑡⟩=tanh(𝑊𝑎𝑎 𝑎⟨𝑡−1⟩ + 𝑊𝑎𝑥 𝑥⟨𝑡⟩ + 𝑏𝑎)
2. Using your new hidden state  𝑎⟨𝑡⟩, compute the prediction  𝑦̂ ⟨𝑡⟩=𝑠𝑜𝑓𝑡𝑚𝑎𝑥(𝑊𝑦𝑎 𝑎⟨𝑡⟩+ 𝑏𝑦). (The function softmax is
   provided)
3. Store  (𝑎⟨𝑡⟩,𝑎⟨𝑡−1⟩,𝑥⟨𝑡⟩,𝑝𝑎𝑟𝑎𝑚𝑒𝑡𝑒𝑟𝑠) in a cache
4. Return  𝑎⟨𝑡⟩,  𝑦̂ ⟨𝑡⟩ and cache


Additional Hints
. A little more information on numpy.tanh
. In this assignment, there's an existing softmax function for you to use. It's located in the file 'rnn_utils.py' and
  has already been imported.
. For matrix multiplication, use numpy.dot
"""


# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: rnn_cell_forward

def rnn_cell_forward(xt, a_prev, parameters):
    """
    Implements a single forward step of the RNN-cell as described in Figure (2)

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, a_prev, xt, parameters)
    """
    # Retrieve parameters from "parameters"
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    ### START CODE HERE ### (≈2 lines)
    # compute next activation state using the formula given above
    a_next = np.tanh(np.dot(Wax,xt)+np.dot(Waa,a_prev)+ba)
    # compute output of the current cell using the formula given above
    yt_pred = softmax(np.dot(Wya,a_next)+by)
    ### END CODE HERE ###
    # store values you need for backward propagation in cache
    cache = (a_next, a_prev, xt, parameters)
    return a_next, yt_pred, cache



#<Test>
np.random.seed(1)
xt_tmp = np.random.randn(3, 10)
a_prev_tmp = np.random.randn(5, 10)
parameters_tmp = {}
parameters_tmp['Waa'] = np.random.randn(5, 5)
parameters_tmp['Wax'] = np.random.randn(5, 3)
parameters_tmp['Wya'] = np.random.randn(2, 5)
parameters_tmp['ba'] = np.random.randn(5, 1)
parameters_tmp['by'] = np.random.randn(2, 1)

a_next_tmp, yt_pred_tmp, cache_tmp = rnn_cell_forward(xt_tmp, a_prev_tmp, parameters_tmp)
print("a_next[4] = \n", a_next_tmp[4])
print("a_next.shape = \n", a_next_tmp.shape)
print("yt_pred[1] =\n", yt_pred_tmp[1])
print("yt_pred.shape = \n", yt_pred_tmp.shape)

# UNIT TESTS
rnn_cell_forward_tests(rnn_cell_forward)
#<Test/>


#1.2 - RNN Forward Pass
"""
. A recurrent neural network (RNN) is a repetition of the RNN cell that you've just built.
    - If your input sequence of data is 10 time steps long, then you will re-use the RNN cell 10 times

. Each cell takes two inputs at each time step:
    - 𝑎⟨𝑡−1⟩: The hidden state from the previous cell
    - 𝑥⟨𝑡⟩: The current time step's input data

. It has two outputs at each time step:
    - A hidden state ( 𝑎⟨𝑡⟩)
    - A prediction ( 𝑦⟨𝑡⟩)

. The weights and biases  (𝑊𝑎𝑎,𝑏𝑎,𝑊𝑎𝑥,𝑏𝑥) are re-used each time step
    - They are maintained between calls to rnn_cell_forward in the 'parameters' dictionary
"""


#Exercise 2 - rnn_forward
"""
Implement the forward propagation of the RNN described in Figure 3.

Instructions:

. Create a 3D array of zeros,  𝑎 of shape  (𝑛𝑎,𝑚,𝑇𝑥) that will store all the hidden states computed by the RNN
. Create a 3D array of zeros,  𝑦̂, of shape  (𝑛𝑦,𝑚,𝑇𝑥) that will store the predictions
    - Note that in this case,  𝑇𝑦=𝑇𝑥 (the prediction and input have the same number of time steps)
. Initialize the 2D hidden state a_next by setting it equal to the initial hidden state,  𝑎0
. At each time step  𝑡:
    - Get  𝑥⟨𝑡⟩, which is a 2D slice of  𝑥 for a single time step  𝑡
        >  𝑥⟨𝑡⟩ has shape  (𝑛𝑥,𝑚)
        >  𝑥 has shape  (𝑛𝑥,𝑚,𝑇𝑥)
    - Update the 2D hidden state  𝑎⟨𝑡⟩ (variable name a_next), the prediction  𝑦̂ ⟨𝑡⟩ and the cache by running rnn_cell_forward
        >  𝑎⟨𝑡⟩ has shape  (𝑛𝑎,𝑚)
    - Store the 2D hidden state in the 3D tensor  𝑎, at the  𝑡𝑡ℎ position
        >  𝑎 has shape  (𝑛𝑎,𝑚,𝑇𝑥)
    - Store the 2D  𝑦̂ ⟨𝑡⟩ prediction (variable name yt_pred) in the 3D tensor  𝑦̂ 𝑝𝑟𝑒𝑑 at the  𝑡𝑡ℎ position
        > 𝑦̂ ⟨𝑡⟩ has shape  (𝑛𝑦,𝑚)
        > 𝑦̂ has shape  (𝑛𝑦,𝑚,𝑇𝑥)
    - Append the cache to the list of caches
. Return the 3D tensor  𝑎 and  𝑦̂ , as well as the list of caches


Additional Hints

. Some helpful documentation on np.zeros
. If you have a 3 dimensional numpy array and are indexing by its third dimension, you can use array slicing like this:
  var_name[:,:,i]
"""

# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: rnn_forward

def rnn_forward(x, a0, parameters):
    """
    Implement the forward propagation of the recurrent neural network described in Figure (3).

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba --  Bias numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y_pred -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of caches, x)
    """
    # Initialize "caches" which will contain the list of all caches
    caches = []
    # Retrieve dimensions from shapes of x and parameters["Wya"]
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape
    ### START CODE HERE ###
    # initialize "a" and "y_pred" with zeros (≈2 lines)
    a = np.zeros((n_a,m,T_x))
    y_pred = np.zeros((n_y,m,T_x))
    # Initialize a_next (≈1 line)
    a_next = a0
    # loop over all time-steps
    for t in range(T_x):
        # Update next hidden state, compute the prediction, get the cache (≈1 line)
        a_next, yt_pred, cache = rnn_cell_forward(x[:,:,t], a_next, parameters)
        # Save the value of the new "next" hidden state in a (≈1 line)
        a[:,:,t] = a_next
        # Save the value of the prediction in y (≈1 line)
        y_pred[:,:,t] = yt_pred
        # Append "cache" to "caches" (≈1 line)
        caches.append(cache)
    ### END CODE HERE ###
    # store values needed for backward propagation in cache
    caches = (caches, x)
    return a, y_pred, caches

#<Test>
np.random.seed(1)
x_tmp = np.random.randn(3, 10, 4)
a0_tmp = np.random.randn(5, 10)
parameters_tmp = {}
parameters_tmp['Waa'] = np.random.randn(5, 5)
parameters_tmp['Wax'] = np.random.randn(5, 3)
parameters_tmp['Wya'] = np.random.randn(2, 5)
parameters_tmp['ba'] = np.random.randn(5, 1)
parameters_tmp['by'] = np.random.randn(2, 1)

a_tmp, y_pred_tmp, caches_tmp = rnn_forward(x_tmp, a0_tmp, parameters_tmp)
print("a[4][1] = \n", a_tmp[4][1])
print("a.shape = \n", a_tmp.shape)
print("y_pred[1][3] =\n", y_pred_tmp[1][3])
print("y_pred.shape = \n", y_pred_tmp.shape)
print("caches[1][1][3] =\n", caches_tmp[1][1][3])
print("len(caches) = \n", len(caches_tmp))

#UNIT TEST
rnn_forward_test(rnn_forward)
#<Test/>

#Congratulations!
"""
You've successfully built the forward propagation of a recurrent neural network from scratch. Nice work!

Situations when this RNN will perform better:

. This will work well enough for some applications, but it suffers from vanishing gradients.
. The RNN works best when each output  𝑦̂ ⟨𝑡⟩ can be estimated using "local" context.
. "Local" context refers to information that is close to the prediction's time step  𝑡.
. More formally, local context refers to inputs  𝑥⟨𝑡′⟩ and predictions  𝑦̂ ⟨𝑡⟩ where  𝑡′ is close to  𝑡.
"""

#What you should remember:
"""
. The recurrent neural network, or RNN, is essentially the repeated use of a single cell.
. A basic RNN reads inputs one at a time, and remembers information through the hidden layer activations (hidden
  states) that are passed from one time step to the next.
     - The time step dimension determines how many times to re-use the RNN cell
. Each cell takes two inputs at each time step:
     - The hidden state from the previous cell
     - The current time step's input data
. Each cell has two outputs at each time step:
     - A hidden state
     - A prediction


In the next section, you'll build a more complex model, the LSTM, which is better at addressing vanishing gradients.
The LSTM is better able to remember a piece of information and save it for many time steps.
"""



#2 - Long Short-Term Memory (LSTM) Network
"""
The following figure shows the operations of an LSTM cell:

Figure 4: LSTM cell. This tracks and updates a "cell state," or memory variable  𝑐⟨𝑡⟩ at every time step, which can be
          different from  𝑎⟨𝑡⟩. Note, the  𝑠𝑜𝑓𝑡𝑚𝑎𝑥 includes a dense layer and softmax.


Similar to the RNN example above, you'll begin by implementing the LSTM cell for a single time step. Then, you'll
iteratively call it from inside a "for loop" to have it process an input with  𝑇𝑥 time steps.
"""





# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: lstm_cell_forward

def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    """
    Implement a single forward step of the LSTM-cell as described in Figure (4)

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    c_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo --  Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    c_next -- next memory state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)

    Note: ft/it/ot stand for the forget/update/output gates, cct stands for the candidate value (c tilde),
          c stands for the cell state (memory)
    """
    # Retrieve parameters from "parameters"
    Wf = parameters["Wf"] # forget gate weight
    bf = parameters["bf"]
    Wi = parameters["Wi"] # update gate weight (notice the variable name)
    bi = parameters["bi"] # (notice the variable name)
    Wc = parameters["Wc"] # candidate value weight
    bc = parameters["bc"]
    Wo = parameters["Wo"] # output gate weight
    bo = parameters["bo"]
    Wy = parameters["Wy"] # prediction weight
    by = parameters["by"]
    # Retrieve dimensions from shapes of xt and Wy
    n_x, m = xt.shape
    n_y, n_a = Wy.shape
    ### START CODE HERE ###
    # Concatenate a_prev and xt (≈1 line)
    concat = np.concatenate([a_prev,xt])
    # Compute values for ft, it, cct, c_next, ot, a_next using the formulas given figure (4) (≈6 lines)
    ft = sigmoid(np.dot(Wf,concat)+bf)
    it = sigmoid(np.dot(Wi,concat)+bi)
    cct = np.tanh(np.dot(Wc,concat)+bc)
    c_next = ft*c_prev+it*cct
    ot = sigmoid(np.dot(Wo,concat)+bo)
    a_next = ot*np.tanh(c_next)
    # Compute prediction of the LSTM cell (≈1 line)
    yt_pred = softmax(np.dot(Wy,a_next)+by)
    ### END CODE HERE ###
    # store values needed for backward propagation in cache
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)
    return a_next, c_next, yt_pred, cache


#<Test>
np.random.seed(1)
xt_tmp = np.random.randn(3, 10)
a_prev_tmp = np.random.randn(5, 10)
c_prev_tmp = np.random.randn(5, 10)
parameters_tmp = {}
parameters_tmp['Wf'] = np.random.randn(5, 5 + 3)
parameters_tmp['bf'] = np.random.randn(5, 1)
parameters_tmp['Wi'] = np.random.randn(5, 5 + 3)
parameters_tmp['bi'] = np.random.randn(5, 1)
parameters_tmp['Wo'] = np.random.randn(5, 5 + 3)
parameters_tmp['bo'] = np.random.randn(5, 1)
parameters_tmp['Wc'] = np.random.randn(5, 5 + 3)
parameters_tmp['bc'] = np.random.randn(5, 1)
parameters_tmp['Wy'] = np.random.randn(2, 5)
parameters_tmp['by'] = np.random.randn(2, 1)

a_next_tmp, c_next_tmp, yt_tmp, cache_tmp = lstm_cell_forward(xt_tmp, a_prev_tmp, c_prev_tmp, parameters_tmp)


print("a_next[4] = \n", a_next_tmp[4])
print("a_next.shape = ", a_next_tmp.shape)
print("c_next[2] = \n", c_next_tmp[2])
print("c_next.shape = ", c_next_tmp.shape)
print("yt[1] =", yt_tmp[1])
print("yt.shape = ", yt_tmp.shape)
print("cache[1][3] =\n", cache_tmp[1][3])
print("len(cache) = ", len(cache_tmp))

# UNIT TEST
lstm_cell_forward_test(lstm_cell_forward)
#<Test/>



# UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: lstm_forward

def lstm_forward(x, a0, parameters):
    """
    Implement the forward propagation of the recurrent neural network using an LSTM-cell described in Figure (4).

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc -- Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo -- Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    c -- The value of the cell state, numpy array of shape (n_a, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of all the caches, x)
    """
    # Initialize "caches", which will track the list of all the caches
    caches = []
    ### START CODE HERE ###
    Wy = parameters['Wy'] # saving parameters['Wy'] in a local variable in case students use Wy instead of parameters['Wy']
    # Retrieve dimensions from shapes of x and parameters['Wy'] (≈2 lines)
    n_x, m, T_x = x.shape
    n_y, n_a = Wy.shape
    # initialize "a", "c" and "y" with zeros (≈3 lines)
    a = np.zeros([n_a,m,T_x])
    c = np.zeros([n_a,m,T_x])
    y = np.zeros([n_y,m,T_x])
    # Initialize a_next and c_next (≈2 lines)
    a_next = a0
    c_next = np.zeros([n_a,m])
    # loop over all time-steps
    for t in range(T_x):
        # Get the 2D slice 'xt' from the 3D input 'x' at time step 't'
        xt = x[:,:,t]
        # Update next hidden state, next memory state, compute the prediction, get the cache (≈1 line)
        a_next, c_next, yt, cache = lstm_cell_forward(xt, a_next, c_next, parameters)
        # Save the value of the new "next" hidden state in a (≈1 line)
        a[:,:,t] = a_next
        # Save the value of the next cell state (≈1 line)
        c[:,:,t]  = c_next
        # Save the value of the prediction in y (≈1 line)
        y[:,:,t] = yt
        # Append the cache into caches (≈1 line)
        caches.append(cache)
    ### END CODE HERE ###
    # store values needed for backward propagation in cache
    caches = (caches, x)
    return a, y, c, caches



#<Test>
np.random.seed(1)
x_tmp = np.random.randn(3, 10, 7)
a0_tmp = np.random.randn(5, 10)
parameters_tmp = {}
parameters_tmp['Wf'] = np.random.randn(5, 5 + 3)
parameters_tmp['bf'] = np.random.randn(5, 1)
parameters_tmp['Wi'] = np.random.randn(5, 5 + 3)
parameters_tmp['bi']= np.random.randn(5, 1)
parameters_tmp['Wo'] = np.random.randn(5, 5 + 3)
parameters_tmp['bo'] = np.random.randn(5, 1)
parameters_tmp['Wc'] = np.random.randn(5, 5 + 3)
parameters_tmp['bc'] = np.random.randn(5, 1)
parameters_tmp['Wy'] = np.random.randn(2, 5)
parameters_tmp['by'] = np.random.randn(2, 1)

a_tmp, y_tmp, c_tmp, caches_tmp = lstm_forward(x_tmp, a0_tmp, parameters_tmp)
print("a[4][3][6] = ", a_tmp[4][3][6])
print("a.shape = ", a_tmp.shape)
print("y[1][4][3] =", y_tmp[1][4][3])
print("y.shape = ", y_tmp.shape)
print("caches[1][1][1] =\n", caches_tmp[1][1][1])
print("c[1][2][1]", c_tmp[1][2][1])
print("len(caches) = ", len(caches_tmp))

# UNIT TEST
lstm_forward_test(lstm_forward)
#<Test/>
