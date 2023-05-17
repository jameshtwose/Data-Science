import gradio as gr
from transformers import pipeline

generator = pipeline('text-generation', model='gpt2')

def generate(text):
    result = generator(text, max_length=100, num_return_sequences=10)
    return result[0]["generated_text"]

examples = [
    ["The Moon's orbit around Earth has"],
    ["The smooth Borealis basin in the Northern Hemisphere covers 40%"],
    ["The way you speak is"]
]

demo = gr.Interface(
    fn=generate,
    inputs=gr.inputs.Textbox(lines=5, label="Input Text (prompt to be completed)"),
    outputs=gr.outputs.Textbox(label="Generated Text"),
    examples=examples,
    title="GPT-2 Text Generation Demo",
)

demo.launch(share=True)