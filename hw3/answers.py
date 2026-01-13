r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=5,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    pass
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    pass
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**
"""

part1_q2 = r"""
**Your answer:**
"""

part1_q3 = r"""
**Your answer:**
"""

part1_q4 = r"""
**Your answer:**
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
sigma_2 is the assumed variance to decode specific x prom latent space value z, means:  $p _{\bb{\beta}}(\bb{X} | \bb{Z}=\bb{z}) = \mathcal{N}( \Psi _{\bb{\beta}}(\bb{z}) , \sigma^2 \bb{I} )$, when $\Psi _{\bb{\beta}}(\bb{z})$ is the decoder output on z. As much as sigma_2 is smaller, a more deterministic decoder assumed. practically, as we decrease sigma_2 we give more weigth to the data recunstruction loss over the KL-diverganze loss. the KL-diverganze loss eforces de decoder
output to distributed as similar as possible to $mathcal{N}(\bb{0},\bb{I})$ (KLdeverganze term), indipendent of x.
practucaly, increasing sigma make us learn more shape, detailed and acurate images in price of making it harder to generate inages diffrent from the data.
It make it harder to generate new images by sampling uniformly from the latent space, vecause we would have errors
with latent space values that arent directly decoded from original data.
In other words, it makes it harder to learn meaningfull latent space represantation ( in sigma2 = infinity 
we would create only data compression).

for example, increasing sigma fixes error i had somtimes
of yelow noisy pixels inside the face, is price of creating more blurred images.
"""

part2_q2 = r"""
1. The purpose of the data recunstruction error is maximize the liklihode to recunstruct x, from z sampled from x decoded value.
The porpose of the KL-diverganze loss is to minimize the diffrance between the the distribution of the decoder value given x,
to a constant probabilty independendent of x. 
2. the latent-space distribution affected by the KL loss term by having distributed mor simmilar to a prior distribution p(z) 
independent of x. by regularizing the latent space for the desired distribution, for example normal distribution, we can ensure it capture meaningfull information. the loss enable us control over the distribution shape,and smothness. 
3. the benefit is that Sampling uniformly from the latent space and decode will anable use to create meaningfull new images, that differes from the input data. that because regularizating the latet space distrubution to a normal distribution will make shure we learned neaningful fetures,
and that the latent space values will cary resunable informantion even when not decoded directly from on of the data images.
"""

part2_q3 = r"""
because out goal is to maximize the loklihoode to recive the data.
"""

part2_q4 = r"""
to enforce sigma to be pozitive witht vanishing of exploding gradients.
because we use  log of the latent-space variance in the loss computation, it helps with avoiding vanishing or exploding gradients.
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
The aliding windows attention will recive xtention for  contex  
with regard to a window at size size 2 * window_size + 1. 
when we stack encoder layes
on top of the other, the now encoder layer attention will regard for
a window of size size 2 * window_size + 1 on the prev network output,
wich itself containes context of size 2 * window_size + 1. therfore,
the layer number i can recive information
at from contex at size O((2 * window_size + 1) ** num_layers). 
"""

part3_q2 = r"""
My idea is in sdition to the sliding window adition,
to create compressd attention that wl regard the whoule image.
I will compress the keys and values using a lernable layer.
the layer will be a  MLP that from each window of size k and stride s will
generate single token representation. 
we will choose K and s such that the new sequense will be of length w.
than, i will recive that the compresst K matrix K' dims are [Batch, *, w, Dims]
and the compresst values matrix V' dims are [Batch, *, w, Dims].
therfor, calculating Q @ K'.T will be O(B * w * Dims *n) instead of
O(B * n * Dims * n). 
Q @ K'.T dims are [Batch, *, n, w] therfore, calculating
torch.sofmax(Q @ K'.T / sqrt(Dims), dim=-1) @ V' will be O(B * n * w * Dims)
wich is as requested. the multy layer perseptron will containe a matrix that
maps window of size k = n // w into single token, therfore
the multy layer perseptron will have O(k ) parameters, 
and wont harm the complexity.
because K'[..,i,:] containse information from
all the infurt, but compressed, aur atention will still contain global information
but compressed. we will still have the local attention from 
the sliding window attention to have the local attention 
represented in uncompressed maner.
"""




# ==============