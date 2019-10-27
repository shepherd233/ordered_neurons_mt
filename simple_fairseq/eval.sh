for i in {0..24}
do
  echo ${i}
  python translate.py --id ${i}
done
