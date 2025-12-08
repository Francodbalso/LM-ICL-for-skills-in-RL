'''use this script to run a model with an already saved buffer and compute any statistics'''
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
    use_buffer = False
    # possible task choices are: ['goto', 'pickup', 'open', 'putnext', 'pick up seq go to']
    env_params = {'forced_level':'goto', 'room_size':4, 'num_dists':0}
    max_steps = 8
    n_eps = 10
    seeds = list(range(0, 0+n_eps))
    #seeds = [16]

    if use_buffer:
        cache_path = "caches/curriculum_babyai.json"
        with open(cache_path, 'r') as fp:
            experience_cache = json.load(fp)

    agent = ContextAgentLLM(model_name='gemma3:4b', context_size=128000, temperature=0.3, max_tokens=15)
    env_id = "BabyAI-MixedTrainLocal-v0"
    possible_actions = ['turn left', 'turn right', 'go forward', 'pick up', 'drop', 'toggle']
    
    rules = "You are in a grid world containing balls and keys of different colours. There are walls that can block your movement. " + \
            "At every step, you will receive an observation about your local surroundings, and you will then select exactly one action " + \
            "from the following options: turn left, turn right, go forward. When prompted to, please select the action that best fits the situation and mission you have been assigned. " + \
            "If your mission is to go to an object you must simply move towards the object until you are at its position. " + \
            "If you see a wall one step forward, you should avoid selecting the go forward action, since this will cause nothing to happen "
            # "If your mission is to pick up an object, the object must be one step in front of you before you can use the pick up action. "

    # Prepare buffer text with past experiences
    buffer_text = ""
    if use_buffer:
        selected_runs = [run for run in experience_cache]
        buffer_text += "\nHere are some past attempts for you to draw experience from:\n"
        for i in range(len(selected_runs)):
            buffer_text +=  f"<run {i+1}>\n" + selected_runs[i] + f"\n<run {i+1}>\n"
        buffer_text += "\nTry to learn from these experiences to explore options to solve the problem."

    print(rules + "\n")
    if use_buffer:
        print(f"Using {len(selected_runs)} past runs in buffer")
        print(buffer_text)

    wins = 0
    for episode in range(n_eps):
        # Create fresh environment for each episode
        env = gym.make(env_id, seed=seeds[episode], **env_params)
        obs, info = env.reset()
        counter = 0
        print(f"\n++++++++++++++++++NEW GAME {episode+1}+++++++++++++++++++\n")

        # Context for ollama (maintains conversation state)
        current_context = []
        # Track how much of the observation we've already processed
        prev_obs_len = 0

        prelude = rules + buffer_text + "\nThe game begins now. "
        full_observation = prelude + "Your mission is to " + obs['mission'] + "\n" + '. '.join(info['descriptions']) + '.'
        for step in range(max_steps):
            # Extract only the NEW part of the observation
            new_observation = full_observation[prev_obs_len:]
            observation_to_send = new_observation + "\nNow select one of the following options: turn left, turn right, go forward. Remember, you are trying to find and " + obs['mission'] + ". Your selected output action is "
            if step == 0:
                print(observation_to_send[len(prelude):])
            else:
                print(observation_to_send)

            # Update prev_obs_len for next iteration
            prev_obs_len += len(observation_to_send)
            full_observation += "\nNow select one of the following options: turn left, turn right, go forward. Remember, you are trying to find and " + obs['mission'] + ". Your selected output action is "

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
                wins += 1
                break
            
            # if unsuccessful, append next observation and keep going
            full_observation += "\n" + '. '.join(info['descriptions']) + '.'

        if not done:
            print(formatted_action)
            print('The maximum number of steps has been reached, so you have failed!')
            full_observation += "\n" + "The maximum number of steps has been reached, so you have failed!"

        full_episode_text = full_observation[len(prelude):]
        env.close()

    print(f'\nThe agent won {wins}/{n_eps} games')