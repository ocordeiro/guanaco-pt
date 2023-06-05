import fasttext
import json
import openai
import re
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="facebook/fasttext-language-identification", filename="model.bin")
model = fasttext.load_model(model_path)

openai.api_key = 'YOUR_API_KEY'

def translate_text(value):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Translate the following text to Brazilian Portuguese: {value}"},
        ],
        max_tokens=1024,
        temperature=0,
    )
    return response.choices[0]["message"]["content"].strip()


def translate_item(item):
    translated_item = {}
    for key, value in item.items():
        if value:
            data = re.search(r'### Human: (.+)### Assistant: (.+)', value)
            print(data[1] + "\n" + data[2])
            print("\n")
            
            translated_value = '### Human: '+translate_text(data[1]) + "### Assistant: " + translate_text(data[2])
            translated_item[key] = translated_value
        else:
            translated_item[key] = ''
    return translated_item


with open("openassistant_best_replies_train.jsonl", "r") as fin:
    with open("openassistant_best_replies_train_translated.jsonl", "r+") as fout:

        lines_in = fin.readlines()
        lines_out = fout.readlines()
        lines_in = lines_in[len(lines_out):]

        print(f"Total de traduzidas: {len(lines_out)}")

        for line in lines_in:
            if line:
                try:
                    data = json.loads(line)
                    text = data["text"]
                    text = text.replace("\n", " ")

                    pred = model.predict(text)

                    if pred[0][0] == "__label__por_Latn":
                        fout.write(line)
                        continue

                    translated_item = translate_item(data)

                    fout.write(json.dumps(translated_item))
                    fout.write("\n")
                    fout.flush()

                except Exception as e:
                    print(f"Erro ao processar a linha: {line}. Erro: {e}")


