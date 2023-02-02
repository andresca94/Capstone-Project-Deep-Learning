from flask import Flask, render_template, request

#Speech Recognition
import pytube
import whisper
#Language Detection
from langdetect import detect
#Tranformers
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import pipeline

app = Flask(__name__)
run_with_ngrok(app)

def get_audio(url):
    data = pytube.YouTube(url)
    # Converting and downloading as 'MP4' file
    audio = data.streams.get_audio_only()
    path = audio.download()#Get file from return of the method
    #Get Lyrics
    model = whisper.load_model('large')#large
    text = model.transcribe(path)
    print(text)
    text_output = text['text'].replace('♪','').strip()
    print(text_output)
    return text_output

activate_translation = False
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_repo = 't5-base'
LANG_TOKEN_MAPPING = {
    'en': '<en>',
    'es': '<es>'
}
model_trans = 'google/mt5-small'
tokenizer = AutoTokenizer.from_pretrained(model_trans)
model_t = AutoModelForSeq2SeqLM.from_pretrained('/content/drive/MyDrive/translation')
#model_t = model_t.cuda()



def language_dectection(text_output):
    global activate_translation
    detection = detect(text_output)
    print(detection)#Debug
    if detection != 'en':
        activate_translation = True
    else:
        activate_translation = False
    return detection


def summarization(text_output):
    # load local model
    if activate_translation == True and language_dectection(text_output) == 'es':
        model = (AutoModelForSeq2SeqLM
         .from_pretrained("/content/drive/MyDrive/Flask_model/trained_for_summarization_es")
         )
        seq2seq = pipeline("summarization", model=model, tokenizer='google/mt5-base')
        result = seq2seq(text_output, min_length=20, max_length=100)
        return result

    else:
        summarizer = pipeline("summarization", model=model_repo, tokenizer=model_repo, framework="tf")
        summary_text = summarizer(text_output, max_length=100, min_length=5, do_sample=False)[0]['summary_text']
        return summary_text



def encode_str(text, tokenizer, seq_len):  
#Tokenize,pad to max length and encode to ids Returns tensor with tocken ids
  input_ids = tokenizer.encode(
  text=text,
  return_tensors = 'pt',
  padding = 'max_length',
  truncation = True,
  max_length = seq_len)
  return input_ids[0]

def translation(summary):
    input_text = summary
    language = '<en>'
    print(language,input_text)
    input_ids = encode_str(
        text = language+input_text,
        tokenizer = tokenizer,
        seq_len = model_t.config.max_length
        )
    input_ids = input_ids.unsqueeze(0)
    print(input_ids)
    output_tokens = model_t.generate(input_ids,num_beams=10, num_return_sequences=1, length_penalty = 1, no_repeat_ngram_size=2)
    
    TRANS=tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return TRANS

@app.route('/')
def home():
    return render_template("home.html")

#@app.route('/', methods = ['POST'])
#def load():
    #return render_template("home2.html")

@app.route("/", methods = ["POST","GET"])
def predict():
# taking the input
  text = request.form['text']
# preprocessing the text
  audio = get_audio(text)
  language_dectection(audio)
  summary = summarization(audio)
  print(activate_translation)
  print(summary)
  if activate_translation:
    translate = translation(summary[0]['summary_text'])
  else:
    translate = summary
  print(translate)
  return render_template("home.html", pred=" Summary: {}".format(translate))

if __name__ == '__main__':
  app.run()