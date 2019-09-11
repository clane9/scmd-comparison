August 27, 2017

We use the previous clustering result to initialize the next call of "k-means" (i.e. warmstart). Please put the modified version of "kmean.m" to the statistical toolbox to replace 'kmean.m'. 

Some codes are adopted from 'SSC_ADMM_v1.1'. 

In S3LR, the best completion error (and the moderate clustering accuracy) can be selected by cross-validation on the completion error with an extra 5% held-out entries.  

To be specific, the parameters in S3LR could be determined by the following procedure: 

Step 1. perform S3LR on (p+5)% missing rate
Step 2. calculate the completion error on 5% extra missing entries
Step 3. draw a curve or surface vs. parameter(s).

Given a set of  (\lambda, \gamma, k):
1) sampling 5% entires
2) completing a matrix with 5+p%  missing
3) err. on 5% entries


Note that if the data is of low-rank, then LRMC works. 

Chunguang