#!/bin/bash

echo "🔥 INITIATING FULL ABLATION STUDY 🔥"

# Run Experiment 1
sh run_cross_att_no_neg.sh

# Run Experiment 2
sh run_cross_attn_with_neg.sh

# Run Experiment 3
sh run_enc_dec_no_neg.sh

# Run Experiment 4
sh run_enc_dec_with_neg.sh

echo "🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY! 🎉"
echo "Check the 'results/' folder for your models, logs, and evaluation metrics."