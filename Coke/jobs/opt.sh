file_name='Data.job'
tmp='tmp.sh'
## mass 23 : pin 11.4~14.4
## mass 27 36 : pin 10.4~14.4
for mass in 27 36
do
  for ((  pin=104  ;  pin<=144  ;  pin=pin+10))
  do
    for ((  ccl=0 ; ccl<=2500 ; ccl=ccl+500))
    do
      for ((  chcl=0  ;  chcl<=500  ;  chcl=chcl+100))
      do
        for ((  tri=0  ;  tri<=500  ;  tri=tri+100))
        do
          for (( cp=0   ;  cp<=500  ;  cp=cp+100))
          do
            for (( tin=300  ; tin<=350  ; tin=tin+10))
            do
              head -14 ${file_name} >> ${tmp}
              pind=$(echo "scale=1;$pin/10" | bc)
              echo "python createTrainingData_coking.py  --mass $mass --tin $tin --pressure $pind  \
              --ccl4 $ccl --chcl3 $chcl --tri $tri --cp $cp " >> ${tmp}
              tail -5 ${file_name} >> ${tmp}
              mv ${tmp} ${file_name}
              qsub ${file_name}
            done
          done
        done
      done
    done
  done
done
