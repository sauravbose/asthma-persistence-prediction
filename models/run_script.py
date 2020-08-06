import csv, subprocess

algo=("xgb","rf","lr","knn")

#For xgb add class_weight
cb=("no","rand_under","enn","renn","tomek", "tomek_enn","tomek_renn","class_weight")

fs=("no","chi2", "anovaF","reliefF","chi2_reliefF","anova_reliefF","multisurf","chi2_multisurf","anova_multisurf")
#Uncomment for pval 0.05
#fs=("chi2", "anovaF","chi2_reliefF","anova_reliefF","chi2_multisurf","anova_multisurf")

pv=(0.01,)

for alg in algo:
    for i in cb:
        for j in fs:
            for k in pv:
                job_name = alg+"_"+i+"_"+j+"_"+str(k)
                if alg=="nb":
                    qsub_command = "qsub -cwd -l h_vmem=80G -l m_mem_free=8G -pe smp 10 -N {} -b y python3 ../naive_bayes.py -b {} -f {} -p {} -a {}".format(job_name,i,j,k,alg)

                elif alg=="knn":
                    qsub_command = "qsub -cwd -l h_vmem=80G -l m_mem_free=8G -pe smp 10 -N {} -b y python3 ../knn.py -b {} -f {} -p {} -a {}".format(job_name,i,j,k,alg)

                elif alg=="lr":
                    qsub_command = "qsub -cwd -l h_vmem=80G -l m_mem_free=8G -pe smp 10 -N {} -b y python3 ../logistic.py -b {} -f {} -p {} -a {}".format(job_name,i,j,k,alg)

                elif alg=="rf":
                    qsub_command = "qsub -cwd -l h_vmem=80G -l m_mem_free=8G -pe smp 10 -N {} -b y python3 ../rf.py -b {} -f {} -p {} -a {}".format(job_name,i,j,k,alg)

                elif alg=="xgb":
                    qsub_command = "qsub -cwd -l h_vmem=80G -l m_mem_free=8G -pe smp 10 -N {} -b y python3 ../xgb.py -b {} -f {} -p {} -a {}".format(job_name,i,j,k,alg)

                exit_status = subprocess.call(qsub_command, shell=True)
                if exit_status is 1:  # Check to make sure the job submitted
                    print ("Job {0} failed to submit".format(qsub_command))
print ("Done submitting jobs!")
