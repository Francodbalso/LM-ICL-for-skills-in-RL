from agent import AgentLLM
import re
import ollama
import json
import os
import random
#from textarena_interaction import ContextAgentLLM


def compare_replays(agent, replay_a, replay_b):
    game = "Tower of Hanoi"

    prompt = f"""
You need to select between 2 options for trajectories you made while attempting to beat {game}.
The one you select will be used later as a replay you will have access to while trying to beat the puzzle, the other will be discarded.
So, you should select the replay that you think will be most beneficial to you for solving the puzzle.
To select a replay please answer with either 1 or 2, 1 indicating that you would like to keep the first replay, 2 indicating that you would like to keep the second replay. Remember you can only select a single replay.

Replay 1: 
{replay_a}
End of Replay 1
Replay 2:
{replay_b}
End of Replay 2
Output strictly "1" or "2" indicating which replay you would prefer to keep.
"""
    response = agent.get_action(prompt, context=[])
    print(prompt)
    #print(response['thinking'])
    out = response['response']
    # Simple parsing logic (you may need to make this more robust for Qwen)
    if "2" in out.strip() and "1" not in out.strip().upper():
        return '2'
    return '1'

def get_LLM_selection(agent, buff_list, num_to_select):
    """
    Reduces the buffer to num_to_select items using pairwise tournament selection.
    """
    # Working copy of the buffer
    survivors = buff_list[:]
    
    # Safety check
    if len(survivors) <= num_to_select:
        return survivors

    print(f"Starting tournament: Reducing {len(survivors)} items to {num_to_select}...")

    while len(survivors) > num_to_select:
        next_round = []
        
        # Shuffle to prevent positional bias
        random.shuffle(survivors)
        
        # We need to calculate how many items we MUST eliminate this round
        # to avoid over-reducing if num_to_select is high.
        # However, a simple approach is: pair up as many as possible.
        # If we have 10 and want 5: 5 pairs -> 5 winners.
        # If we have 10 and want 8: This logic is tricky with pure knockout.
        
        # ROBUST APPROACH: 
        # Iterate through the list. Compare [i] vs [i+1]. 
        # Loser gets deleted immediately.
        # Stop deleting once we reach num_to_select.
        
        i = 0
        while i < len(survivors) - 1:
            # Check if we have already reached the target size
            if len(survivors) == num_to_select:
                break
                
            replay_a = survivors[i]
            replay_b = survivors[i+1]
            
            # Ask agent for preference
            winner = compare_replays(agent, replay_a, replay_b)
            
            if winner == '1':
                # Keep 1 (at index i), remove 2 (at index i+1)
                survivors.pop(i+1)
            else:
                # Keep 1, remove 2 (at index i)
                survivors.pop(i)
                
            # Move to next distinct pair (since we popped one, 'i' now points to a new item, 
            # but we want to move past the winner to give others a chance)
            i += 1 
            
        # If we loop through everyone and still have too many, the while loop runs again.
        
    return survivors


def buffer_selection(buff_list, method, num_to_select=3, agent=None):
    if method == 'random':
        if len(buff_list) < num_to_select:
            return buff_list
        prev_experiences = random.sample(buff_list, num_to_select)
    elif method == 'recent':
        prev_experiences = buff_list[-num_to_select:]
    elif method == 'LLM':
        if agent == None:
            print("NO AGENT PASSED")
            return -1
        else:
            prev_experiences = get_LLM_selection(agent, buff_list, num_to_select)
    else:
        print("Improper Method selected")
        return -1
    return prev_experiences

# if __name__ == "__main__":
#     fname = "hanoi_experience_4disk_1"
#     with open(f"hanoi_caches/{fname}.json", 'r') as fp:
#         buff = json.load(fp)
#     buff = buff[-7:]
#     agent = ContextAgentLLM(model_name='qwen3:4b', context_size=40000, temperature=0.5, max_tokens=5000)
#     selected_reps = buffer_selection(buff, "LLM", 3, agent)
#     print("SELECTED RUNS")
#     for i in selected_reps:
#         print(i)
        

