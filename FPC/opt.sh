file_name='DataGeneration.job'
tmp='tmp.sh'
# for ((  i=0  ;  i<=2500  ;  i=i+ 250))

# for i in 0.129 0.13 0.131 0.128
# do
#   for ((  j=0  ;  j<=500  ;  j=j+ 100))
#   do
#     for ((  k=0  ;  k<=500  ;  k=k+ 100))
#     do
#       for ((  l=0  ;  l<=500  ;  l=l+ 100))
#       do
#         head -14 ${file_name} >> ${tmp}
#         echo "python createTrainingData_testlimit.py  $i $j $k $l " >> ${tmp}
#         tail -5 ${file_name} >> ${tmp}
#         mv ${tmp} ${file_name}
#         qsub ${file_name}
#       done
#     done
#   done
# done



# for i in 23 27 32 36

# # for i in 0.129 0.13 0.131 0.128
# do
#   for ((  j=104  ;  j<=144  ;  j=j+ 10))
#   do
#     for ((  k=300  ;  k<=350  ;  k=k+ 10))
#     do
#       for ((  l=0  ;  l<=2500  ;  l=l+125))
#       do
#         head -14 ${file_name} >> ${tmp}
#         pin=$(echo "scale=1;$j/10" | bc)
#         echo "python createTrainingData_testlimit.py  $i $pin $k $l " >> ${tmp}
#         tail -5 ${file_name} >> ${tmp}
#         mv ${tmp} ${file_name}
#         qsub ${file_name}
#         done
#     done
#   done
# done
# 43 54 64 72
# for mass in 27 32 36

# for i in 0.129 0.13 0.131 0.128
# for mass in 27 32 36
# do
#   for ((  pin=104  ;  pin<=144  ;  pin=pin+ 10))
#   do
#     for ((  temp=300  ;  temp<=350  ;  temp=temp+ 10))
#     do
#       for ((  ccl=0  ;  ccl<=2500  ;  ccl=ccl+125))
#       do
#         head -14 ${file_name} >> ${tmp}
#         pind=$(echo "scale=1;$pin/10" | bc)
#         echo "python createTrainingData_prevX.py  $mass $pind $ccl $temp " >> ${tmp}
#         tail -5 ${file_name} >> ${tmp}
#         mv ${tmp} ${file_name}
#         qsub ${file_name}
#         done
#     done
#   done
# done

for mass in 22 27 36

# for i in 0.129 0.13 0.131 0.128
do
  for ((  pin=124  ;  pin<=144  ;  pin=pin+ 20))
  do
    for ((T_out=465 ; T_out <= 480 ; T_out=T_out+5))
    do
      for ((  ccl=0  ;  ccl<=1500  ;  ccl=ccl+500))
      do
        for ((  temp=330  ;  temp<=350  ;  temp=temp+ 10))
        do

          head -14 ${file_name} >> ${tmp}
          pind=$(echo "scale=1;$pin/10" | bc)
          echo "python createTrainingData_prevX.py  $mass $pind $T_out $ccl $temp " >> ${tmp}
          tail -5 ${file_name} >> ${tmp}
          mv ${tmp} ${file_name}
          qsub ${file_name}
          done
      done
    done
  done
done