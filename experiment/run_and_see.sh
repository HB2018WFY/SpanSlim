#dataSets=("A" "B" "C" "media" "hipster-more" "sockshop" "trainticket"  "socialNetwork")

dataSets=("media" "hipster-more")

for d in ${dataSets[*]}
do
  echo "============For dataSet $d============" >>result.log
  OUTPUT_FILE="profile_$d.svg"
  TARGET_SCRIPT="./run.py"
  py-spy record -o "$OUTPUT_FILE" -- "$TARGET_SCRIPT" --dataSet "$d" >> result.log
done