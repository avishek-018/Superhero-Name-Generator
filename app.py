from flask import Flask, render_template, jsonify, redirect, url_for, request
import tensorflow as tf 
import pickle

app = Flask(__name__)

# def generate_names(name):
#   return name+"ayan";

max_len = 33

with open("tokenizer.pkl", "rb") as f:
  tokenizer = pickle.load(f)

model = tf.keras.models.load_model("superhero_model.h5")

with open("index_to_char.pkl", "rb") as f:
  index_to_char = pickle.load(f)

def name_to_seq(name):
  return [tokenizer.texts_to_sequences(c)[0][0] for c in name]


def generate_names(seed):
  for i in range(0, 40):
    seq = name_to_seq(seed)
    padded = tf.keras.preprocessing.sequence.pad_sequences([seq], padding='pre',
                                                           maxlen=max_len-1,
                                                           truncating='pre')
    pred = model.predict(padded)[0]
    pred_char = index_to_char[tf.argmax(pred).numpy()]
    seed += pred_char

    if pred_char == '\t':
      break
  return seed

@app.route("/")
def home():
  return render_template("index.html")

@app.route("/get_name")
def get_name():
    name = request.args.get('name', 0, type=str)
    print(name)
    sup_name = generate_names(name)
    print(sup_name)
    return jsonify(result=sup_name)

if __name__ == "__main__":
  app.run()
