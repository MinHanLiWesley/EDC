DIR='0607_V2'
file_name='Heatflux.job'
tmp='tmp.sh'

for ((TOUT=4645 ; TOUT<=4670 ; TOUT=TOUT+ 1))   
do
    head -14 ${file_name} >> ${tmp}
    TOUTd=$(echo "scale=1;$TOUT/10" | bc)
    # python -u hf_modelV2.py --te $TOUTd | tee $DIR/hf1_$TOUTd.log
    echo "python -u hf_modelV2.py --te $TOUTd | tee $DIR/hf1_$TOUTd.log " >> ${tmp}
    tail -5 ${file_name} >> ${tmp}
    mv ${tmp} ${file_name}
    qsub ${file_name}
done

