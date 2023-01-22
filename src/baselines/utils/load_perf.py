import os
import json

for path, subdirs, files in os.walk("./tmp/exp2/CoLA/bert-base/HPO_S0"):
    for name in files:
        address = os.path.join(path, name)
        if "eval_results" in address:
            file = json.load(open(address))
            acc = file["eval_matthews_correlation"]
            print(f"{path} : {acc}")
