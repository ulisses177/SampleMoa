import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import json
from sentence_transformers import SentenceTransformer, util

# Define the models you have
models = {
    "llama3.1": "llama3.1",
    "qwen2": "qwen2",
    "mistral_nemo": "mistral-nemo",
    "dolphin-llama3": "dolphin-llama3"
}

# Define a function to create a chain with a given model and prompt template
def create_chain(model_name, template):
    prompt = ChatPromptTemplate.from_template(template)
    model = Ollama(model=model_name)
    chain = prompt | model
    return chain

# Define the prompt template
template = """Pergunta: {question}

Resposta: Vamos pensar passo a passo e responder em português."""

# Create chains for each model
chains = {name: create_chain(model, template) for name, model in models.items()}

# Function to invoke a chain with a question
def get_answer(chain, question):
    return chain.invoke({"question": question})

# MoA structure
class MoA:
    def __init__(self, layers):
        self.layers = layers
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def run(self, question):
        responses = [question]
        for layer in self.layers:
            new_responses = []
            for chain in layer:
                for response in responses:
                    new_responses.append(get_answer(chain, response))
            responses = new_responses
        final_response = self.aggregate_responses(responses)
        return final_response

    def aggregate_responses(self, responses):
        # Compute embeddings for responses
        embeddings = self.model.encode(responses, convert_to_tensor=True)
        
        # Compute cosine similarity matrix
        cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)
        
        # Sum the scores for each response
        scores = cosine_scores.sum(axis=1)
        
        # Find the response with the highest score
        best_response_idx = scores.argmax()
        best_response = responses[best_response_idx]
        
        return best_response

# Create MoA layers
layer1 = [chains["llama3.1"], chains["qwen2"]]
layer2 = [chains["mistral_nemo"], chains["dolphin-llama3"]]

moa = MoA([layer1, layer2])

# Test MoA with a question
question = "Faça um código que desenhe uma animação do fractal julia"
final_answer = moa.run(question)

print("Final Answer:", final_answer)
