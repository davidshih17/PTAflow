Here is the pipeline for generating reweighted posteriors starting from a normalizing flow proposal:

- For each test point {t}, use the flow to generate 100k parameter points {theta}. These are called "samples". This happens in:
sample_and_likelihoods_testpoints_10p.ipynb

- Evaluate the true likelihoods p(t|theta) on all the samples (using a CPU cluster). An exanple condor script (run_condor_ml4gw_calculate_likelihoods_flowsamples.sh) that calls calculate_likelihood_flowsamples.py is provided here. 

- Package everything (test points aka "residuals", the samples, their true likelihoods, and their flow likelihoods) into a single npy file. This 
happens in: merge_true_likelihoods.ipynb
