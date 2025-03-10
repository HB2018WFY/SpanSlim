dataSets=("A" "B" "C" "media" "hipster-more" "sockshop" "trainticket"  "socialNetwork")

for d in ${dataSets[*]}
do
  echo "============For dataSet $d============" >result.log
  python "run.py" --dataSet "$d" > result.log
done