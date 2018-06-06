#!/bin/bash
set -e

SDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WDIR=`pwd`

# Get the conala evaluation scripts
if [[ ! -e conala_eval ]]; then
  git clone git@github.com:neulab/conala_eval.git
fi

# Get the data
wget http://www.phontron.com/download/conala-corpus-v1.0.zip
unzip conala-corpus-v1.0.zip

# Extract data 
cd $WDIR/conala-corpus

python $WDIR/conala_eval/extract_slots.py

python $WDIR/conala_eval/json2seq_input.py conala-train.json.seq2seq conala-train.intent conala-train.snippet
python $WDIR/conala_eval/json2seq_input.py conala-test.json.seq2seq conala-test.intent conala-test.snippet
python $WDIR/conala_eval/json2seq_input.py conala-mined.jsonl.seq2seq conala-mined.intent conala-mined.snippet

# Split off a 400-line dev set from the training set
# Also, concatenate the first 100000 lines of mined data
for f in intent snippet; do
  head -n 400 < conala-train.$f > conala-dev.$f
  tail -n +401 < conala-train.$f > conala-trainnodev.$f
  cat conala-train.$f <(head -n 100000 conala-mined.$f) > conala-trainnodev+mined.$f
done

cd $WDIR

# Train and package seq2seq models on annotated+mined and annotated only data
for setting in annotmined annot; do

  xnmt --dynet-gpu $SDIR/$setting.yaml
  
  python conala_eval/seq_output_to_code.py results/$setting.test.hyp conala-corpus/conala-test.json.seq2seq results/$setting.test.json
  
  cd $WDIR/results
  
  cp $setting.test.json answer.txt
  zip $setting.zip answer.txt

  cd $WDIR

done

# annotmined.zip and annot.zip can be submitted to the CodaLab leaderboard:
# 
