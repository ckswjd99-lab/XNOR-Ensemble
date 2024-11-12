NUM_MODEL=10

# > ./logs/dist_bw_naive_log.txt

for (( i=1; i<=NUM_MODEL; i++ ))
do
    echo "Running iteration $i"
    python distill_bw_naive.py >> ./logs/dist_bw_alltrain_log.txt
    python eval_ens.py >> ./logs/dist_bw_alltrain_log.txt
done