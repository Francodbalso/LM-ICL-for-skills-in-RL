# make sure your ollama server is running
import ollama
from agent import AgentLLM
import gym
import textworld
import textworld.gym

# example game make command
# tw-make tw-simple --rewards dense    --goal detailed --seed 18 --output tw_games/simple_dense_detailed_seed18.ulx

# initialize an agent
agent = AgentLLM(model_name='gemma3:4b', context_size=128000, temperature=0, max_tokens=20)

# start game
env_id = textworld.gym.register_game('tw_games/simple_dense_brief_seed18.ulx', request_infos=textworld.EnvInfos(admissible_commands=True))
env = textworld.gym.make(env_id)
obs, infos = env.reset() # obs contains the string observation and infos contains 'admissible_commands' list of str

current_experience = '' # used to accumulate the relevant experience from this episode (to be saved later)
past_experience = '' # problem for later
prompt = 'You will take on the role of a player in a text based game. You will be given \
observations that can be found between <observation> delimiters and then you will \
be asked to provide an action in response. Your action MUST be a SINGLE item selected from \
the comma separated choices that can be found between the <possible actions> delimiters. You must not select more \
than one action, and you must not select an action that is not found among the possible actions. You might also \
be provided with some helpful past experiences, which can be found between <past experience> \
delimiters.'

prompt += '\n<past experience>\n' + past_experience + '\n<past experience>\n' + 'The new game begins now!\n'
obs = '<observation>\n' + obs[1210:] + '\n<observation>\n' + '<possible actions>\n' +  \
        ', '.join(infos['admissible_commands']) + '\n<possible actions>\n' + \
        'Select one of the comma separated actions that are presented. Your selected action is: '
current_experience += obs
prompt += obs

# start environment loop
print('\n' + prompt)
n_steps = 20
for i in range(n_steps):
    # query model for an action and verify (could add additional parsing here later)
    action = agent.get_action(prompt)['response']
    assert action in infos['admissible_commands'], f'model did not provide a possible action, instead it gave: {action}'

    # step through the env and update prompt
    obs, score, done, infos = env.step(action)
    text_increment = action + '\n\n<observation>\n' + obs + '\n<observation>\n' + '<possible actions>\n' +  \
        ', '.join(infos['admissible_commands']) + '\n<possible actions>\n' + \
        'Select one of the comma separated actions that are presented. Your selected action is: '
    print(text_increment)
    prompt += text_increment
    current_experience += text_increment

    if done:
        break


env.close()