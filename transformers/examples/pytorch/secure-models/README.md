requirements:
    pip install Crypten in editable mode: https://github.com/facebookresearch/CrypTen
    
To profile 2PC transformer, use profile_transformer.py (modify master address etc.)

To profile single machine transformer, use single_transformer.py

To profile 2PC CNN, use profile_cnn.py

Currently, 2PC transformer seems to be 1000x slower than in plaintext single machine setting. While a facebook workshop paper claims it to be 12x slower.
