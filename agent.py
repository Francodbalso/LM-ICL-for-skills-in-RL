import ollama

class AgentLLM:
    def __init__(self, model_name, context_size, temperature, max_tokens):
        super().__init__()
        self.model_name = model_name
        self.options = {'num_ctx': context_size, # max number of tokens allowed in context
                        'temperature': temperature, # sampling temperature
                        'num_predict': max_tokens # max number of decoded tokens before interrupt
                        } 
    
    def get_action(self, prompt):
        return ollama.generate(model=self.model_name,
                        prompt=prompt,
                        options=self.options)

class ContextAgentLLM:
    def __init__(self, model_name, context_size, temperature, max_tokens):
        super().__init__()
        self.model_name = model_name
        self.options = {'num_ctx': context_size, # max number of tokens allowed in context
                        'temperature': temperature, # sampling temperature
                        'num_predict': max_tokens, # max number of decoded tokens before interrupt
                        'keep_alive': -1
                        } 
    
    def get_action(self, prompt, context, num_predict=None):
        options_to_pass = self.options.copy()
        if num_predict is not None:
            options_to_pass['num_predict'] = num_predict

        return ollama.generate(model=self.model_name,
                        prompt=prompt,
                        options=options_to_pass,
                        context=context)