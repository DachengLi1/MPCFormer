import os
for path, subdirs, files in os.walk("./tmp/exp2/RTE"):
    for name in files:
        address = os.path.join(path, name)
        if "checkpoint" in address:
            print(f"removing {address}")
            os.remove(address)
