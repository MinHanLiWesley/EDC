file_name='delete.job'
tmp='tmp.sh'
for ((  k=170000 ;  k<=173845 ;  k=k+ 1))

# for i in 0.129 0.13 0.131 0.128

do

  head -14 ${file_name} >> ${tmp}
  qdel $k
  tail -5 ${file_name} >> ${tmp}
  mv ${tmp} ${file_name}
  qsub ${file_name}
done