from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
import torch

# Choose a lightweight open-source model (you can replace this later)
MODEL_NAME = "distilgpt2"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Create the pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id
)

# Wrap in LangChain LLM interface
llm = HuggingFacePipeline(pipeline=pipe)

# Define a prompt
template = """You are a compassionate mental health support assistant.
User: {question}
Assistant:"""
prompt = PromptTemplate(template=template, input_variables=["question"])

# Chain everything
chain = LLMChain(prompt=prompt, llm=llm)

def ask_question(question: str):
    """Generate a response using an open-source model."""
    response = chain.run(question)
    return response.strip()

if __name__ == "__main__":
    print(ask_question("I feel anxious about my exams."))
