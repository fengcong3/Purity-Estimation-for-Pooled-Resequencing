
            #!/usr/bin/sh
            #-*# It has been modified by QS.py : zz.go.sh.TR4wmbX2 #-*#
            #/public/home/fengcong/.chenglab
            #change dir
            #export PATH='/public/agis/chengshifeng_group/fengcong/WGRS/software/Fc-code/shell_cmd/:$PATH'
            cd /vol3/agis/chengshifeng_group/fengcong/zz.test_for_wuliangye/Real_Test ;
            #time
            echo Running on host `hostname`
            echo PID $$
            time1=`date +"%Y-%m-%d %H:%M:%S"`
            echo Start Time is `date`
            echo Directory is `pwd`
            #stats,how to del: sed -i "/##--\/--##$/d" your_work.sh
            alias log='/public/agis/chengshifeng_group/fengcong/WGRS/software/Fc-code/shell_cmd/log'
            stats="s"
            stats_num=1
            echo "-------------------------------------------------------------------"
            #origin script content
            #########################################################################start
            
            
            
            /public/home/fengcong/anaconda2/envs/py3/bin/python real_data_pipeline.py --percentage-file percentage.txt --cram-dir /vol3/agis/chengshifeng_group/fengcong/zz.test_for_wuliangye/Real_Test --vcf chr1.snp.raw.HARD.Missing-unphasing.ID.allele2_retain.hard_retain.InbreedingCoeff_retain.ann.vcf.gz --reference /public/agis/chengshifeng_group/fengcong/WGRS/software/nginx/www/igv-webapp.2.x/data/seq/ZW6_ref.fa --chrom chr1 --standard-sample JI2174 --tool-config tool_paths.example.json
            cstat=${PIPESTATUS[@]};stats=${stats}":""${cstat}" && echo QSstats_${stats_num}:${cstat} && let stats_num+=1  ##--/--##
            
            
            
            


            #########################################################################end
            echo "-------------------------------------------------------------------"
            echo End Time is `date`
            time2=`date +"%Y-%m-%d %H:%M:%S"`
            timerun1=$(($(date +%s -d "$time2") - $(date +%s -d "$time1")))
            echo $timerun1 | awk '{print "Running time is " $1*1/3600 " hours(" $1*1/60  " mins)"}'
            echo $stats
            #qsub:qsub -V -p 0 -q test -l mem=1G,nodes=1:ppn=1 -M QYXTpjBo /vol3/agis/chengshifeng_group/fengcong/zz.test_for_wuliangye/Real_Test/zz.go.sh
	