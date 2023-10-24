from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import argparse

#model_name = "deepset/roberta-base-squad2"

parser = argparse.ArgumentParser()
parser.add_argument('--repo-id','-m', default='deepset/roberta-base-squad2', help='Path to model repo (default: deepset/roberta-base-squad2)')
parser.add_argument('--device','-d', default='cuda', help='Device to use for inference (default: cuda)')
parser.add_argument('--cache-dir', default=None, help='Directory of the folder to download models. Ex: "models" will make/use a folder named models in the same directory as this program (~\\models\\). The default directory is C:\\Users\\[username]\\.cache\\huggingface\\hub\\')
args = parser.parse_args()
import os
if args.cache_dir:
    cache_dir = (args.cache_dir+"/"+args.repo_id.split("/")[-1])
else:
    cache_dir = args.repo_id
  
  
if os.path.isdir(cache_dir):
    model_path = cache_dir
else:
    import huggingface_hub
    print("Downloading model...")
    kwargs = {}
    if cache_dir is not None:
        kwargs["local_dir"] = cache_dir
        kwargs["cache_dir"] = cache_dir
        # kwargs["local_dir"] = "C:/AI/MyThings/nimple Speech Recognition/ssd"
        # kwargs["cache_dir"] = "C:/AI/MyThings/nimple Speech Recognition/sfw"
        kwargs["local_dir_use_symlinks"] = False
    #minimum to run.
    allow_patterns = ["config.json","pytorch_model.bin","merges.txt","vocab.json"]
    #allow_patterns = ["config.json","model.bin","tokenizer.json","vocabulary.txt",]
    #model_path=huggingface_hub.snapshot_download(args.repo_id,**kwargs)#tqdm_class=disabled_tqdm,,
    model_path=huggingface_hub.snapshot_download(args.repo_id,allow_patterns=allow_patterns,**kwargs)#tqdm_class=disabled_tqdm,,
    
model = AutoModelForQuestionAnswering.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

nlp = pipeline('question-answering', model=model, tokenizer=tokenizer, device=args.device)

def findAnswerInText(question, context):
    QA_input = {
        'question': question,
        'context': context
    }
    res = nlp(QA_input)
    print(res)
    return res.get('answer'),res.get('score')

import gradio as gr
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            #autoscroll=True
            question_box= gr.Textbox(label='Question',placeholder="Why is model conversion important?")
            context_box= gr.Textbox(label='Context',placeholder="The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.")
            button = gr.Button("Do Stuff")
            #stopButton = gr.Button("Stop Stuff",variant="stop")
            
        with gr.Column():
            output_text = gr.Textbox(label='Answer',max_lines=30,interactive=True)
            confidence = gr.Textbox(label='Score',max_lines=30,interactive=True)
    button.click(findAnswerInText, 
    inputs=[
    question_box, 
    context_box, 
    ], outputs=[output_text,confidence])
    #stopButton.click(ohNo, None, None, queue=False)
demo.queue(max_size=30)
demo.launch(inbrowser=True, show_error=True, share=False)  




