r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=128,
        seq_len=64,
        h_dim=512,
        n_layers=3,
        dropout=0.2,
        learn_rate=0.001,
        lr_sched_factor=0.5,
        lr_sched_patience=2,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    pass
    # ========================
    return hypers


def part1_generation_params():
    start_seq = "Ham. To be, or not to be- "
    temperature = 0.8
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    pass
    # ========================
    return start_seq, temperature


part1_q1 = r"""
We split the text because training on the entire text at once would do to things:
1) Exceed your GPU's memory capacity
2) cause vanishing gradients, making it impossible for the model to learn.
Using shorter sequences allows for Batching, which processes multiple text segments in parallel for speed,
and Truncated Backpropagation Through Time, which keeps the math stable and computationally manageable.
"""

part1_q2 = r"""
This is possible because the hidden state is passed from one batch to the next.
In a RNN the final hidden state of one sequence serves as the initial state for the next in the batch,
which allows the model to "carry" earlier context across sequence boundaries, even though the gradients are only calculated for the current.
"""

part1_q3 = r"""
We don't shuffle the batches because the model relies on their continuity to maintain long-term memory.
Since we pass the final hidden state of one batch as the initial state for the next, the sequences in the same "slot"
for adjacent batches should be consecutive segments of the original text. Shuffling would break this link, providing the model with semi-unrelated parts of the text,
which would confuse the learning process and destroy its ability to understand context.
"""

part1_q4 = r"""
1. At the default 1.0, the model might assign enough probability to unlikely characters that it occasionally picks one, 
eading to typos or nonsensical words. Lowering it concentrates the probability on the most likely next characters,
making the output feel more grammatically grounded and "sane."

2. The output becomes random gibberish. As T->inf, the probability distribution gets flattened, i.e., every character becomes
equally/similarely likely to be picked regardless of what the model actually learned. The model loses its ability to follow structure,
resulting in a chaotic string of characters and symbols.

3. The output becomes repetitive as T->0 because it becomes too confident, consistently picking only the single most probable character (argmax).
While this is very safe, it often traps the model in infinite loops—such as repeating "THE THE THE"—because it lacks the randomness needed to move onto new words.
"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=30, h_dim=5, z_dim=8, x_sigma2=0.2, learn_rate=0.01, betas=(0.2, 0.2),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers["betas"] = (0.2,0.2)
    hypers["batch_size"] = 32
    hypers["h_dim"] = 100
    hypers["z_dim"] = 100
    hypers["x_sigma2"] = 6 #0.005
    hypers["learn_rate"] = 0.001
    '''
    hypers["betas"] = (0.8,0.8)#(0.2,0.2)
    hypers["batch_size"] = 30
    hypers["h_dim"] = 90
    hypers["z_dim"] = 90
    hypers["x_sigma2"] = 0.0014
    hypers["learn_rate"] = 0.0013
    '''
    '''
    hypers["betas"] = (0.8,0.8)#(0.2,0.2)
    hypers["batch_size"] = 30
    hypers["h_dim"] = 90
    hypers["z_dim"] = 90
    hypers["x_sigma2"] = 0.0014
    hypers["learn_rate"] = 0.0015
    '''
    hypers["betas"] = (0.8,0.8)#(0.2,0.2)
    hypers["batch_size"] = 30
    hypers["h_dim"] = 100
    hypers["z_dim"] = 100
    hypers["x_sigma2"] = 0.0018
    hypers["learn_rate"] = 0.0013
    #{'batch_size': 32, 'h_dim': 100, 'z_dim': 100, 'x_sigma2': 0.002, 'learn_rate': 0.001, 'betas': 
    
    # ========================
    return hypers


part2_q1 = r"""
sigma_2 is the assumed variance to decode specific x prom latent space value z, means:  $p _{\bb{\beta}}(\bb{X} | \bb{Z}=\bb{z}) = \mathcal{N}( \Psi _{\bb{\beta}}(\bb{z}) , \sigma^2 \bb{I} )$, when $\Psi _{\bb{\beta}}(\bb{z})$ is the decoder output on z.
As much as sigma_2 is smaller, a more deterministic decoder assumed. practically, as we decrease sigma_2 we give more weigth to the data recunstruction loss over the KL-diverganze loss. the KL-diverganze loss eforces de decoder
output to distributed more simmilar to $mathcal{N}(\bb{0},\bb{I})$ , a distribution indipendent of x.
practucaly, decresing sigma will make us to create more  detailed,
but make it harder learning to create resunable inages that diffrent from the data
by sampling uniformly from the latent space.
That because we havent any constraints about the
latent space distribution, therdfore we dont know what we learned to decode from latent space values that aren't directly decoded from the data.
Regularize the latent space distribution to normal distribution alsu enshure we learned meaningful fetures.
In other words, decreasing sigma will make  it harder to learn meaningfull latent space represantation ( in sigma2 = infinity 
we would create only data compression).


for example, increasing sigma fixes error i had somtimes
of yelow noisy pixels inside the face, is price of creating more blurred images.
"""

part2_q2 = r"""
1. The purpose of the data recunstruction error is maximize the liklihode to recunstruct x, from z sampled from x decoded value.

The porpose of the KL-diverganze loss is to minimize the diffrance between the the distribution of the decoder value given x,
to a constant probabilty independendent of x. 
2. the latent-space distribution affected by the KL loss term by having distributed more simmilar to a prior distribution p(z) 
independent of x. by regularizing the latent space for the desired distribution, for example normal distribution, we can ensure it capture meaningfull information. That because the loss enable us control over the distribution shape,and smothness. 
else, the regularization of the latent space distribution will anable use to learn information about latent space values that aren't decoded from our imaged directly
or close in the original space for our data.
3. the benefit is that Sampling uniformly from the latent space and decode will anable use to create meaningfull new images, that differes from the input data. that because regularizating the latet space distrubution to a normal distribution will make shur we learned neaningful fetures,
and that the latent space values will cary resunable informantion even when not decoded directly from on of the data images.
"""

part2_q3 = r"""
Because out goal is to maximize the loklihoode to recive our data.
"""

part2_q4 = r"""
to enforce sigma to be pozitive witht vanishing or exploding gradients.
if we would enforece sigma to be pozitive by using relu or softplus on the network output,
we would have vanishing gradients when the output is near zero.
other benefit of this scheme 
for evoiding exploding or vanishing gradients is that
 we use the log of the latent-space variance in the loss computation.

"""


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 100, 
        num_heads = 5,
        num_layers = 10,
        hidden_dim = 30,
        window_size =50,
        droupout = 0.1,
        lr=0.0001,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    
    # ========================
    return hypers


part3_q1 = r"""
The a=sliding windows attention will recive attention for  contex  
with regard to a window at size size 2 * window_size + 1 tokens. 
when we stack encoder layes
on top of the other, the each encoder layer attention will regard for
a window of size size 2 * window_size + 1 on the prev network output,
wich itself containes context of size 2 * window_size + 1 for the layer before it.
Therfore,
the layer number i can recive information
at from contex at size O((2 * window_size + 1) ** num_layers) overall. 
"""

part3_q2 = r"""
My idea is in adition to the sliding window adition,
we will create compressd attention in regard to the whole image.
To create the compresset attention, I will compress the keys and values using a lernable layer.
the layer will be a  MLP that from each window of size k and stride s will
generate single token representation. 
we will choose K and s such that the new sequense will be of length w.
Than, WE will recive that the compresst K matrix, denote K', dims are [Batch, *, w, Dims]
and the compresst values matrix V' dims are [Batch, *, w, Dims].
therfor, calculating Q @ K'.T will be O(B * w * Dims *n) instead of
O(B * n * Dims * n). 
Q @ K'.T dims are [Batch, *, n, w] therfore, calculating
torch.sofmax(Q @ K'.T / sqrt(Dims), dim=-1) @ V' will be O(B * n * w * Dims)
wich is as requested. the multy layer perseptron will containe a matrix that
maps window of size k = n // w into single token, therfore
the multy layer perseptron will have O(k ) parameters, 
and won't harm the complexity.
because K'[..,i,:] containse information from
all the output, but compressed, our attention will still contain global information
but compressed. we will still have the local attention from 
the sliding window attention for having the local attention 
represented in uncompressed maner.
"""




# ==============