import re
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def extract_board_states(buffer_text):
    """
    Scans a raw text buffer and returns a list of board strings.
    
    Args:
        buffer_text (str): The full raw text of the replay buffer.
        
    Returns:
        list of str: A list where each element is a clean board string 
                     ready for parsing.
    """
    # Regex Explanation:
    # 1. \[GAME\] Current Board: \s* -> Matches the header and potential whitespace/newlines
    # 2. (                            -> Start capturing group for the board content
    # 3.   A: \[.*?\]\s* -> Matches 'A: [...]' and trailing whitespace
    # 4.   B: \[.*?\]\s* -> Matches 'B: [...]' and trailing whitespace
    # 5.   C: \[.*?\]                 -> Matches 'C: [...]'
    # 6. )                            -> End capturing group
    # Flags: DOTALL is not strictly needed if we explicitly match newlines, 
    # but re.MULTILINE helps anchor ^ if needed. Here we keep it simple.
    
    pattern = r"\[GAME\] Current Board:\s*(A: \[.*?\]\s*B: \[.*?\]\s*C: \[.*?\])"
    
    # re.findall returns all non-overlapping matches in the string
    matches = re.findall(pattern, buffer_text, re.DOTALL)
    
    # Clean up the matches to ensure they look exactly like your example
    # (removing excess trailing newlines if regex caught them)
    cleaned_states = [m.strip() for m in matches]
    
    return cleaned_states

def parse_board_state(log_string):
    """
    Parses the game board string into a dictionary: {'A': [4,3], 'B': [2], 'C': [1]}
    Assumes lists are ordered [bottom, ..., top] based on your example.
    """
    state = {'A': [], 'B': [], 'C': []}
    
    # Regex to find lines like "A: [4, 3, 2, 1]"
    # Matches the letter, then captures the content inside brackets
    matches = re.findall(r'([ABC]): \[([\d, \-]*)\]', log_string)
    
    for peg, content in matches:
        if content.strip():
            # Convert "4, 3, 2" string to list of ints [4, 3, 2]
            # Filter out -1 if your logs sometimes use placeholders
            disks = [int(x.strip()) for x in content.split(',') if x.strip()]
            state[peg] = disks
        else:
            state[peg] = []
            
    return state

def get_disk_location(state, disk_val):
    for peg in ['A', 'B', 'C']:
        if disk_val in state[peg]:
            return peg
    return None

def calculate_hanoi_distance(state, n_disks=4, target_peg='C'):
    """
    Recursively calculates minimum moves to solve from current state.
    """
    # Base case
    if n_disks == 0:
        return 0
    
    # Find where the current largest disk (n) is
    current_peg = get_disk_location(state, n_disks)
    
    # If the disk is missing (error in log parsing?), return error or infinity
    if current_peg is None:
        return float('inf') 

    if current_peg == target_peg:
        # Disk N is already in place. We just need to solve for N-1 on top of it.
        # Target remains the same.
        return calculate_hanoi_distance(state, n_disks - 1, target_peg)
    else:
        # Disk N is on the wrong peg.
        # 1. We need to move disks 1..(N-1) to the AUX peg.
        # 2. Move Disk N (1 move).
        # 3. Move disks 1..(N-1) from AUX to Target (Known cost: 2^(N-1) - 1).
        
        # Determine Auxiliary peg (The one that isn't current or target)
        pegs = {'A', 'B', 'C'}
        aux_peg = list(pegs - {current_peg, target_peg})[0]
        
        # Recursive step: How many moves to get smaller stack to Aux?
        moves_to_stack_aux = calculate_hanoi_distance(state, n_disks - 1, aux_peg)
        
        # Total = (moves to clear way) + (move disk N) + (move stack back)
        # Simplified: moves_to_stack_aux + 2^(n-1)
        return moves_to_stack_aux + (2 ** (n_disks - 1))

def get_all_distances(buffer):

# --- Example Usage with your Log format ---
    buff_distances = []
    for run in buffer:
        replay_log = extract_board_states(run)

        distances = []
        for entry in replay_log:
            # 1. Parse
            state = parse_board_state(entry)
            # 2. Calculate Distance (Assuming target is C and 4 disks)
            dist = calculate_hanoi_distance(state, n_disks=4, target_peg='C')
            distances.append(dist)

        #print("Distances to solution:", distances)
        buff_distances.append(distances)
    return buff_distances

def plot_distance(data_matrix):
    N_steps = 32
    x_axis = np.arange(N_steps)

    # 2. Calculate the mean and standard deviation along the desired axis (axis=1 for the data above)
    # The mean and std deviation should be calculated across all walkers for each time step.
    data_mean = data_matrix.mean(axis=0)
    data_std = data_matrix.std(axis=0)

    # 3. Define the upper and lower bounds for the filled area
    # These bounds typically represent mean +/- standard deviation
    upper_bound = data_mean + data_std
    lower_bound = data_mean - data_std

    # 4. Plot the results
    fig, ax = plt.subplots()

    # Plot the mean line
    ax.plot(x_axis, data_mean, color='blue', lw=2, label='Mean Value')

    # Fill the area between the upper and lower bounds
    ax.fill_between(x_axis, lower_bound, upper_bound, color='blue', alpha=0.5, label='Mean $\\pm$ 1 Std Dev')

    # Add labels, title, and legend
    ax.set_xlabel('X-axis Label')
    ax.set_ylabel('Y-axis Label')
    ax.set_title('Line Plot with Filled Uncertainty Bars')
    ax.legend(loc='upper left')

    # Display the plot
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def plot_multiple_distances(matrices, labels= ["LLM Select", 'Empty', 'Random']):
    fig, ax = plt.subplots(figsize=(5, 4), dpi=200)

    N_steps = matrices[0].shape[1]
    x_axis = np.arange(N_steps)

    for i, data_matrix in enumerate(matrices):
        data_mean = data_matrix.mean(axis=0)
        data_std = data_matrix.std(axis=0)

        upper_bound = data_mean + data_std
        lower_bound = data_mean - data_std

        # Plot mean line
        ax.plot(
            x_axis,
            data_mean,
            lw=2,
            label=f'{labels[i]} Mean'
        )

        # Plot uncertainty band
        ax.fill_between(
            x_axis,
            lower_bound,
            upper_bound,
            alpha=0.25
        )

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Number of Moves Needed to Solve')
    ax.set_title('')
    ax.legend(loc='upper left')

    plt.show()

if __name__ == "__main__":
    with open("hanoi_caches/" + 'hanoi_4disk_empty_cache.json', 'r') as fp:
        nobuff_runs = json.load(fp)

    nobuff = get_all_distances(nobuff_runs)
    padded_vals = []
    for x in nobuff:
        if x[-1] == 0:
            x = x + [0]*(32-len(x))
        if len(x) == 32:
            padded_vals.append(x)
    #buff_full_dist = []
    #buff_full_dist = [x for x in nobuff if len(x) == 32]
    buff_full_dist = np.array(padded_vals)
    #print(buff_full_dist.shape)
    #plot_distances(buff_full_dist)

    all_LLM_sep_runs = []
    for i in range(1, 7):
        with open("hanoi_caches/" + f'hanoi_4disk_LLM_replay_sep_cache_{i}.json', 'r') as fp:
            tmp = json.load(fp)
        all_LLM_sep_runs = all_LLM_sep_runs + tmp

    buff = get_all_distances(all_LLM_sep_runs)
    padded_vals = []
    for x in buff:
        if x[-1] == 0:
            x = x + [0]*(32-len(x))
        if len(x) == 32:
            padded_vals.append(x)
    #LLMbuff_full_dist = []
    #LLMbuff_full_dist = [x for x in buff if len(x) == 32]
    LLMbuff_full_dist = np.array(padded_vals)
    #print(buff_full_dist.shape)

    random_runs = []
    for i in range(1, 7):
        with open("hanoi_caches/" + f'hanoi_4disk_random_cache_{i}.json', 'r') as fp:
            tmp = json.load(fp)
        random_runs = random_runs + tmp

    rand_buff = get_all_distances(random_runs)
    padded_vals = []
    for i, x in enumerate(rand_buff):
        if x[-1] == 0:
            x = x + [0]*(32-len(x))
        if len(x) == 32:
            padded_vals.append(x)
    print(rand_buff)
    #randbuff_full_dist = []
    #randbuff_full_dist = [x for x in rand_buff if len(x) == 32]
    randbuff_full_dist = np.array(padded_vals)

    plot_multiple_distances([LLMbuff_full_dist, buff_full_dist, randbuff_full_dist])