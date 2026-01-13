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
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    pass
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**
"""

part2_q2 = r"""
**Your answer:**
"""

part2_q3 = r"""
**Your answer:**
"""

part2_q4 = r"""
**Your answer:**
"""


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 0, 
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        droupout = 0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    pass
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**
"""

part3_q2 = r"""
**Your answer:**
"""

# ==============
