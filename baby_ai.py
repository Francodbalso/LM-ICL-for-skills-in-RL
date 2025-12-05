import warnings
warnings.filterwarnings("ignore") 
from agent import ContextAgentLLM
import ollama
import gym
import babyai_text
import json
import gym.utils.passive_env_checker as pec
pec.logger.deprecation = lambda *args, **kwargs: None

def format_action(response, possible_actions):
    for a in possible_actions:
        if a in response:
            return a
    return -1

if __name__ == "__main__":
    env_id = "BabyAI-MixedTrainLocal-v0"
    #env_id = "BabyAI-MixedTestLocal-v0"
    cache_path = "caches/baby_ai_experience.json"
    with open(cache_path, 'r') as fp:
        experience_cache = json.load(fp)

    agent = ContextAgentLLM(model_name='gemma3:4b', context_size=128000, temperature=0.3, max_tokens=15)
    max_steps = 5
    possible_actions = ['turn left', 'turn right', 'go forward', 'pick up', 'drop', 'toggle']

    for episode in range(0, 2):
        # Create fresh environment for each episode
        env = gym.make(env_id)
        obs, info = env.reset()
        counter = 0
        print(f"\n++++++++++++++++++NEW GAME {episode+1}+++++++++++++++++++\n")

        # Context for ollama (maintains conversation state)
        current_context = []
        # Track how much of the observation we've already processed
        prev_obs_len = 0

        # Prepare buffer text with past experiences (only for first turn)
        selected_runs = [run for run in experience_cache]
        buffer_text = ""
        if len(selected_runs) > 3:
            selected_runs = selected_runs[-3:]
        if len(selected_runs) != 0:
            buffer_text += "\nHere are some past attempts for you to draw experience from:\n"
            for i in range(len(selected_runs)):
                buffer_text +=  f"<run {i+1}>\n" + selected_runs[i] + f"\n<run {i+1}>\n"
            buffer_text += "\nThink about where you seem to get stuck in these past runs and try to explore options to solve the problem."

        print(f"Using {len(selected_runs)} past runs in buffer")

        rules = "You are in a grid world containing balls and keys of different colours. There are walls that can block your movement. " + \
                "At every step, you will receive an observation about your local surroundings, and you will then select exactly one action " + \
                "from the following options: turn left, turn right, go forward, pick up, drop, toggle. To pick up an object you must be directly in front of it."
        
        prelude = rules + buffer_text + "\nThe game begins now. "
        full_observation = prelude + "Your mission is to " + obs['mission'] + "\n" + '. '.join(info['descriptions']) + '.'
        for step in range(max_steps):
            # Extract only the NEW part of the observation
            new_observation = full_observation[prev_obs_len:]
            observation_to_send = new_observation + "\nNow select one of the following options: turn left, turn right, go forward, pick up, drop, toggle. Your selected action is: "
            print(observation_to_send)

            # Update prev_obs_len for next iteration
            prev_obs_len += len(observation_to_send)
            full_observation += "\nNow select one of the following options: turn left, turn right, go forward, pick up, drop, toggle. Your selected action is: "

            # Get action from agent
            action_response = agent.get_action(observation_to_send, current_context)
            current_context = action_response['context']

            # Format action for environment
            formatted_action = format_action(action_response['response'], possible_actions)
            if formatted_action == -1:
                raise Exception(f'invalid action selection: {action_response["response"]}')
            else:     
                action = possible_actions.index(formatted_action)
                full_observation += formatted_action

            # Take step in environment
            obs, r, done, info = env.step(action)
            
            # check for success
            if done:
                print(action_response["response"])
                print("Congratulations, you have accomplished your mission!")
                full_observation += "\nCongratulations, you have accomplished your mission!"
                break
            
            # if unsuccessful, append next observation and keep going
            full_observation += "\n" + '. '.join(info['descriptions']) + '.'

        if not done:
            print('The maximum number of steps has been reached, so you have failed!')
            full_observation += "\n" + "The maximum number of steps has been reached, so you have failed!"

        full_episode_text = full_observation[len(prelude):]
        env.close()
        experience_cache.append(full_episode_text)

    # Save updated cache
    with open(cache_path, 'w') as fp:
        json.dump(experience_cache, fp, indent=2)
