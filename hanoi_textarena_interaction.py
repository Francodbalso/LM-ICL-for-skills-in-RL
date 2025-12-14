import textarena as ta
from agent import AgentLLM
import re
import ollama
import json
import os
from buffer_selection import buffer_selection
import re
from optimal_agent import get_best_move
class ContextAgentLLM:
    def __init__(self, model_name, context_size, temperature, max_tokens):
        super().__init__()
        self.model_name = model_name
        self.options = {'num_ctx': context_size, # max number of tokens allowed in context
                        'temperature': temperature, # sampling temperature
                        'num_predict': max_tokens, # max number of decoded tokens before interrupt
                        'keep_alive': "3h"
                        } 
    
    def get_action(self, prompt, context):
        return ollama.generate(model=self.model_name,
                        prompt=prompt,
                        options=self.options,
                        context = context)

def extract_isolated_pair(text):
    # Define the pattern:
    # (?<![A-Z])  -> Lookbehind: Ensure the char before isn't an uppercase letter
    # ([A-Z])     -> Capture Group 1: The first letter
    # \s+         -> One or more spaces
    # ([A-Z])     -> Capture Group 2: The second letter
    # (?![A-Z])   -> Lookahead: Ensure the char after isn't an uppercase letter
    pattern = r'(?<![A-Z])([A-Z])\s+([A-Z])(?![A-Z])'
    
    matches = re.findall(pattern, text)

    # We expect exactly ONE valid pair in the string
    if len(matches) == 1:
        # matches[0] will be a tuple like ('A', 'B')
        return matches[0]
    else:
        return -1, -1
    
def extract_move_RushHour(text):
    match = re.search(r"MOVE:\s*([A-Za-z][+-])", text)
    if match:
        move = match.group(1)
        print(move)  # e.g. "+A"
    else:
        print("No move found.")
        return -1
    return move

    
def format_action(text, env_id):
    if "TowerOfHanoi-v0" in env_id:
        A, B = extract_isolated_pair(text)
        return f"[{A} {B}]"
    elif "RushHour-v0" in env_id:
        move = extract_move_RushHour(text)
        return f"[{move}]"
    else:
        return "UNKOWN ENV"

if __name__ == "__main__":
    env_id = "TowerOfHanoi-v0"
    #cache_path = "hanoi_caches/hanoi_4disk_LLM_replay_sep_cache_2.json"
    #cache_paths = ["llama_caches/hanoi_3disk_empty_cache_1.json"]
    cache_paths = [f"llama_caches/hanoi_4disk_onlysuccess5_cache_{i}.json" for i in range(1)]
    for cache_path in cache_paths:
        #cache_path = "hanoi_experience_4disk_2.json"
        # env_id = "RushHour-v0"
        # cache_path = "RushHour_experience_easy.json"
        
        append_text = "\nKeep in mind the moves you have made in the past. Submit your next move in the format: MOVE: X Y. Only answer with a pair of letters indicating the move you wish to make."
        #added_context = "The size of the disk is given by the number, so a larger number means a larger disk. Also, the bottom of the tower is the left-most disk on that tower. The order of the disks on a tower goes from left to right and the number indicates the size of the disk."
        added_context = ""
        #append_text = "\nSubmit your next move in the format: MOVE: Y+ or Y-. Y indicates the letter of the car you wish to move, + indicates forward, - indicates backward. Only answer with the move you intend to make."
    #     added_context = "\nEach cell is either “.” (empty) or a car/truck letter.\n\
    # Cars are length 2, trucks length 3.\n\
    # Horizontal vehicles can only move left/right.\n\
    # Vertical vehicles can only move up/down.\n\
    # The target car is X, always horizontal.\n\
    # The exit is to the right of row 3, column 6.\n On the game board, all letters of the same type represent a single car. So, for example, the red car is represented by two X's on the board, which indicate the two spots that the X car occupies.\
    # When you move the cars all of the letters corresponding to that car move 1 spot in the designated direction (+ or -)."
        # if os.path.exists(cache_path):
        #     print("You were about to overwrite an existing cache")
        #     break
        # else:
        #     tmp = []
        #     with open(cache_path, 'w') as fp:
        #         json.dump(tmp, fp)
        
        with open(cache_path, 'r') as fp:
            experience_cache = json.load(fp)
        #agent = ContextAgentLLM(model_name='hoangquan456/qwen3-nothink:8b', context_size=40000, temperature=0.5, max_tokens=1000)
        agent = ContextAgentLLM(model_name='Llama3.1:8b', context_size=32768, temperature=0.5, max_tokens=5000)
        thinking_chains = {}
        for episode in range(0, 10):
            thinking_chains[episode] = []
            # Create fresh environment for each episode
            env = ta.make(env_id=env_id, num_disks = 4, max_turns=30)
            #env = ta.make(env_id=env_id, difficulty="easy")
            env.reset(num_players=1)

            done = False
            counter = 0
            print(f"\n++++++++++++++++++NEW GAME {episode+1}+++++++++++++++++++\n")

            # Track the full episode trajectory for storing in cache (gameplay only, no rules)
            full_episode_text = ""

            # Track where the actual gameplay starts (after the rules)
            gameplay_start_marker = "[GAME] Current Board:"
            #gameplay_start_marker = "[GAME]"
            gameplay_offset = 0

            # Context for ollama (maintains conversation state)
            current_context = []

            # Track how much of the observation we've already processed
            prev_obs_len = 0

            # Prepare buffer text with past experiences (only for first turn)
            #selected_runs = []

            #selected_runs = []
            selected_runs = buffer_selection(experience_cache, 'LLM', 3, agent)
            #selected_runs = []
            #selected_runs = [experience_cache[0]]
            if selected_runs == -1:
                print("ERROR OCCURED")
                break

            # Slice to the last 3 runs first
            if len(selected_runs) > 3:
                selected_runs = selected_runs[-3:]

            if len(selected_runs) != 0:
                processed_runs = []
                
                for run in selected_runs:
                    run = run.strip()
                    
                    # 1. Find all punctuation indices (. ! ?)
                    # We use finditer to get the exact positions of punctuation
                    punct_matches = list(re.finditer(r'[.!?]', run))
                    
                    last_sentence = ""
                    
                    # Logic: If we have at least 2 punctuation marks, grab everything 
                    # after the second-to-last one.
                    if len(punct_matches) >= 2:
                        # Get the end index of the second-to-last punctuation mark
                        split_index = punct_matches[-2].end()
                        last_sentence = run[split_index:].strip()
                    elif len(punct_matches) == 1:
                        # If there is only one sentence, take the whole thing
                        last_sentence = run.strip()
                    else:
                        # Fallback if no punctuation is found
                        last_sentence = run.strip()[-100:] # Just take last 100 chars as fallback
                        
                    # 2. Format the specific run string
                    formatted_run = (
                        f"[Start of run]\n"
                        f"During this run: {last_sentence}\n\n"
                        f"{run}\n"
                        f"[End of run]"
                    )
                    processed_runs.append(formatted_run)

                # Join the processed runs
                buffer_text = "\n\nHere are some past attempts you made for you to draw experience from:\n" \
                    + "\n\n".join(processed_runs) \
                    + "\n\nAbove are some of your past attempts at solving this task. If the runs are unsuccessful, try new things to get to a solution."
            else:
                buffer_text = ""
            # selected_runs = buffer_selection(experience_cache, 'LLM', 3, agent)
            # if selected_runs == -1:
            #     print("ERROR OCCURED")
            #     break
            # if len(selected_runs) > 3:
            #     selected_runs = selected_runs[-3:]
            # if len(selected_runs) != 0:
            #     buffer_text = "\n\nHere are some past attempts you made for you to draw experience from:\n" \
            #         + "\n".join(selected_runs) \
            #         + "\n Above are some of your past attempts at solving this task. If the runs are unsuccessful, try new things to get to a solution."
            # else:
            #     buffer_text = ""

            buffer_len = len(buffer_text)
            print(f"Using {len(selected_runs)} past runs in buffer")

            while not done:
                player_id, full_observation = env.get_observation()
                print(full_observation)
                # On first turn, find where gameplay actually starts
                if counter == 0:
                    #insert_target = "(+ = down or right, - = up or left)."
                    insert_target = "(e.g., '[A C]')."
                    context_insert_ind = full_observation.find(insert_target)
                    full_observation = full_observation[:context_insert_ind+len(insert_target)]+added_context+full_observation[context_insert_ind+len(insert_target):]
                    gameplay_offset = full_observation.find(gameplay_start_marker)
                    if gameplay_offset == -1:
                        gameplay_offset = 0

                # Extract only the NEW part of the observation
                new_observation = full_observation[prev_obs_len:]

                print("=" * 50)
                print("NEW OBSERVATION START")

                # Only add buffer text on the first turn
                if counter == 0 and buffer_text:
                    # Insert buffer after the rules, before "At each turn, submit one move."
                    observation_to_send = new_observation.replace(
                        "At each turn, submit one move.",
                        f"{buffer_text}\n\nAt each turn, submit one move."
                    )
                    prev_obs_len = len(full_observation)
                    # DON'T add buffer_len to prev_obs_len - the buffer isn't in the actual observation
                else:
                    observation_to_send = new_observation + append_text
                    # Update prev_obs_len for next iteration
                    prev_obs_len = len(full_observation)

                #prev_obs_len += len(observation_to_send)

                print(observation_to_send)
                print("NEW OBSERVATION END")

                # Get action from agent
                print("CURRENT CONTEXT LEN", len(current_context))
                action_response = agent.get_action(observation_to_send, current_context)
                #action = get_best_move(observation_to_send)
                #print(action_response)
                # print(action_response['thinking'])
                # thinking_chains[episode].append(action_response['thinking'])
                current_context = action_response['context']

                print("RESPONSE START")
                print(action_response["response"])
                print("RESPONSE END")

                # Format action for environment
                action = format_action(action_response['response'], env_id)
                print(f"Formatted action: {action}")

                # Take step in environment
                done, step_info = env.step(action=action)

                # counter += 1
                # if counter > 50:
                #     print("Max steps reached, breaking")
                #     break

            # Get final observation if game is done
            if done:
                final_player_id, final_observation = env.get_observation()
                # Store only the gameplay part (exclude the rules)
                full_episode_text = final_observation[gameplay_offset:]
                print("\n=== GAME FINISHED ===")
                print("Final observation length:", len(final_observation))
                print("Gameplay-only length:", len(full_episode_text))
            # Store episode trajectory (gameplay only, no rules)
            
            

            # Close environment
            rewards, game_info = env.close()
            print(f"Episode {episode+1} rewards: {rewards}")
            print(f"Game info: {game_info}")
            
            # if rewards[0] != 1.0:
            #     full_episode_text = full_episode_text + "\n" + game_info[0]["reason"]
            # if rewards[0] == 1.0:
            #     full_episode_text = full_episode_text + "\n" + game_info[0]["reason"]
            full_episode_text = full_episode_text + "\n" + game_info[0]["reason"]
            experience_cache.append(full_episode_text)

        # Save updated cache
        with open(cache_path, 'w') as fp:
            json.dump(experience_cache, fp, indent=2)
        
        # for i in range(0, 100):
        #     f_name = f"{cache_path.split('.')[0]}_COT_{i}.json"
        #     if os.path.exists(f_name):
        #         continue
        #     with open(f_name, 'w') as fp:
        #         json.dump(thinking_chains, fp)
        #     break
