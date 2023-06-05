import fasttext
import json
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="facebook/fasttext-language-identification", filename="model.bin")
model = fasttext.load_model(model_path)


with open("openassistant_best_replies_train_translated.jsonl", "r") as fin:
    with open("openassistant_best_replies_train_ptbr.jsonl", "r+") as fout:

        lines_in = fin.readlines()
        lines_out = fout.readlines()
        lines_in = lines_in[len(lines_out):]
       
        for line in lines_in:
            if line:
                try:
                    data = json.loads(line)
                    text = data["text"]
                    text = text.replace("\n", " ")

                    pred = model.predict(text)

                    if pred[0][0] == "__label__por_Latn":
                        fout.write(line)
                        fout.flush()
                    else:
                        print(f"{line}")

                except Exception as e:
                    print(f"Erro ao processar a linha: {line}. Erro: {e}")


