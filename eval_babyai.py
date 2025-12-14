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
    use_buffer = True
    # possible task choices are: ['goto', 'pickup', 'open', 'putnext', 'pick up seq go to']
    env_params = {'forced_level':'pick up seq go to', 'room_size':5, 'num_dists':0}
    max_steps = 12
    n_eps = 25
    seeds = list(range(25, 25+n_eps))
    #seeds = [16]
    model_setting = {'model_name':'llama3.1:8b', 'context_size':7000, 'temperature':0, 'max_tokens':10}
    #model_setting = {'model_name':'gemma3:4b', 'context_size':10000, 'temperature':0, 'max_tokens':10}

    buffer_text = ""
    if use_buffer: # Prepare buffer text with past experiences
        cache_path = "caches/pickup_then_goto.json"
        #cache_path = "caches/curriculum_babyai.json"
        with open(cache_path, 'r') as fp:
            experience_cache = json.load(fp)
        selected_runs = experience_cache[:]
        buffer_text += "\nHere are some past attempts for you to draw experience from:\n"
        for i in range(len(selected_runs)):
            buffer_text +=  f"<run {i+1}>\n" + selected_runs[i] + f"\n<run {i+1}>\n"
        buffer_text += "\nTry to learn from these experiences to explore options to solve the problem."
        #buffer_text += "These experiences were for the individual pick up and go to tasks, but you will now need to compose the skills you have learned above in order to solve the new task, which requires you to first pick up an object and then go to another object. "

    agent = ContextAgentLLM(**model_setting)
    env_id = "BabyAI-MixedTrainLocal-v0"
    possible_actions = ['turn left', 'turn right', 'go forward', 'pick up', 'drop', 'toggle']
    
    rules = "You are in a grid world containing balls and keys of different colours. There are walls that can block your movement. " + \
            "At every step, you will receive an observation about your local surroundings, and you will then select exactly one action " + \
            "from the following options: turn left, turn right, go forward, pick up. When prompted to, please select the action that best fits the situation and mission you have been assigned. " + \
            "If your mission is to go to an object you must simply move towards the object until you are at its position. " + \
            "If you see a wall one step forward, you should avoid selecting the go forward action, since this will cause nothing to happen " + \
            "If you need to pick up an object, the object must be one step in front of you before you can use the pick up action successfully. "

    prelude = rules + buffer_text + "\nThe game begins now. "
    # pretokenize the rules and buffer only once so they can immediately be passed as context at every episod
    # res = agent.get_action(prompt=prelude, context=[], num_predict=0)
    # prelude_context = res['context']

    print(f'Task parameters: {env_params}')
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

        full_observation = prelude + "Your mission is to " + obs['mission'] + "\n" + '. '.join(info['descriptions']) + '.'
        for step in range(max_steps):
            # Extract only the NEW part of the observation
            new_observation = full_observation[prev_obs_len:]
            observation_to_send = new_observation + "\nNow select one of the following options: turn left, turn right, go forward, pick up. Remember, you are trying to find and " + obs['mission'] + ". Please respond with only the action. Your selected output action is: "
            if step == 0:
                print(observation_to_send[len(prelude):])
            else:
                print(observation_to_send)

            # Update prev_obs_len for next iteration
            prev_obs_len += len(observation_to_send)
            full_observation += "\nNow select one of the following options: turn left, turn right, go forward, pick up. Remember, you are trying to find and " + obs['mission'] + ". Please respond with only the action. Your selected output action is: "

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
                print(formatted_action)
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

        env.close()

    print(f'\nThe agent won {wins}/{n_eps} games')