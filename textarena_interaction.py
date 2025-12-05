import textarena as ta
from agent import AgentLLM
import re
import ollama
import json
from enum import Enum

class Models(Enum):
    PHI = "phi3:3.8b"
    QWEN = "qwen3:4b"
    GEMMA = "gemma3:4b"

class ContextAgentLLM:
    def __init__(self, model_name, context_size, temperature, max_tokens):
        super().__init__()
        self.model_name = model_name
        self.options = {'num_ctx': context_size, # max number of tokens allowed in context
                        'temperature': temperature, # sampling temperature
                        'num_predict': max_tokens, # max number of decoded tokens before interrupt
                        'keep_alive': -1
                        } 
    
    def get_action(self, prompt, context):
        return ollama.generate(model=self.model_name,
                        prompt=prompt,
                        options=self.options,
                        context = context)
    
class TestEnv():
    def __init__(self, env_id, agent, cache_file="experience_cache.json"):
        self.env_id = env_id
        self.agent = agent
        self.cache_file = cache_file
        self.cache_path = "./cache/"
        self.get_experience_cache()

    def get_experience_cache(self):
        file_path = self.cache_path+self.cache_file
        with open(file_path, 'r') as fp:
            self.experience_cache = json.load(fp)

    def save_experience_cache(self):
        file_path = self.cache_path+self.cache_file
        with open(file_path, 'w') as fp:
            json.dump(self.experience_cache, fp, indent=2)
    
    def add_to_experience_cache(self, addition):
        self.experience_cache.append(addition)

    def run_round(self, round_id): # must pass back (save to experience boolean, experience)
        pass

    def run(self, rounds=5, update_cache=True, **kwargs):
        for episode in range(rounds):
            (save, experience) = self.run_round(episode+1, **kwargs)
            if (save): self.add_to_experience_cache(experience)
        
        if (update_cache): self.save_experience_cache()

class TowerOfHanoiTestEnv(TestEnv):
    
    def __init__(self, agent, num_disks=4, max_turns=25, cache_file="experience_cache.json"):
        self.env_id = "TowerOfHanoi-v0"
        self.num_disks = num_disks
        self.max_turns = max_turns
        super().__init__(self.env_id, agent, cache_file)

    def extract_isolated_pair(self, text):
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
    
    def format_action(self, text):
        A, B = self.extract_isolated_pair(text)
        return f"[{A} {B}]"
    
    def run_round(self, round_id): 
        env = ta.make(env_id=self.env_id, num_disks=self.num_disks, max_turns=self.max_turns)
        env.reset(num_players=1)

        done = False
        counter = 0
        print(f"\n++++++++++++++++++NEW GAME {round_id}+++++++++++++++++++\n")

        # Track the full episode trajectory for storing in cache (gameplay only, no rules)
        full_episode_text = ""

        # Track where the actual gameplay starts (after the rules)
        gameplay_start_marker = "[GAME] Current Board:"
        gameplay_offset = 0

        # Context for ollama (maintains conversation state)
        current_context = []

        # Track how much of the observation we've already processed
        prev_obs_len = 0

        # Prepare buffer text with past experiences (only for first turn)
        #selected_runs = []
        selected_runs = [run for run in self.experience_cache]
        if len(selected_runs) > 3:
            selected_runs = selected_runs[-3:]
        if len(selected_runs) != 0:
            buffer_text = "\n\nHere are some past attempts for you to draw experience from:\n" \
                + "\n".join(selected_runs) \
                + "\n Think about where you seem to get stuck in these past runs and try to explore options to solve the problem."
        else:
            buffer_text = ""

        print(f"Using {len(selected_runs)} past runs in buffer")

        while not done:
            player_id, full_observation = env.get_observation()

            # On first turn, find where gameplay actually starts
            if counter == 0:
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
                # DON'T add buffer_len to prev_obs_len - the buffer isn't in the actual observation
                prev_obs_len = len(full_observation)
            else:
                observation_to_send = new_observation + "\nSubmit your next move in the format: X Y. Only answer with a pair of letters indicating the move you wish to make."
                # Update prev_obs_len for next iteration
                prev_obs_len = len(full_observation)

            print(observation_to_send)
            print("NEW OBSERVATION END")

            # Get action from agent
            action_response = self.agent.get_action(observation_to_send, current_context)
            current_context = action_response['context']

            print("RESPONSE START")
            print(action_response["response"])
            print("RESPONSE END")

            # Format action for environment
            action = self.format_action(action_response['response'])
            print(f"Formatted action: {action}")

            # Take step in environment
            done, step_info = env.step(action=action)

            counter += 1
            if counter > 50:
                print("Max steps reached, breaking")
                break

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
        print(f"Episode {round_id+1} rewards: {rewards}")
        print(f"Game info: {game_info}")
        # if rewards[0] == 0.0:
        #     full_episode_text = full_episode_text + "\n[GAME] You attempted an invalid move. Reason: You tried to place a larger disk on a smaller disk. Please resubmit a valid move and remember to follow the game rules to avoid penalties."
        # if rewards[0] == 1.0:
        #     full_episode_text = full_episode_text + "\nYOU WIN, Good job the above attempt was succesful."
        full_episode_text = full_episode_text + "\n" + game_info[0]["reason"]
        return (True, full_episode_text)

if __name__ == "__main__":
    #agent = ContextAgentLLM(model_name='hoangquan456/qwen3-nothink:8b', context_size=40000, temperature=0.5, max_tokens=1000)
    agent = ContextAgentLLM(model_name=Models.QWEN, context_size=40000, temperature=0.5, max_tokens=8000)
    env = TowerOfHanoiTestEnv(agent, cache_file="hanoi_experience_cache.json")
    env.run()
