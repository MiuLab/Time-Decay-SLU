# Time-Decay-Learning
How Time Matters: Learning Time-Decay Attention for Contextual Spoken Language Understanding in Dialogue

## Reference
Main paper to be cited

```
@inproceedings{su2018how,
  title={How time matters: Learning Time-Decay Attention for Contextual Spoken Language Understanding in Dialogues},
    author={Shang-Yu Su, Pei-Chieh Yuan, and Yun-Nung Chen},
    booktitle={Proceedings of NAACL-HLT},
    year={2018}
}
```

## Usage
1. Put the DSTC4 data into some directory (e.g. /home/workspace/dstc4). 
   Run the code ``parse_history.py`` to preprocess the data.

2. Put the embedding files into some directory (e.g. /home/workspace/glove)
   Modify line 29 in the code ``slu_preprocess.py`` 

3. Run the code in the directory (row_*) with arguments like below:

```
    python slu.py \
    --target [ALL, Guide, Tourist]
    --level  [sentence, role]
    --attention [convex, linear, concave, universal]
```