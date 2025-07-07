# 70016_NLP_coursework



## Getting started

```BertModel``` folder contains all the implemented models and notebooks.

- ```Analyzer.py``` contains the training and evaluation pipeline.
- ```BaselineModels.py``` contains the code for BOW and TF-IDF baseline.
- ```pipeline.ipynb``` is the notebook for running all three tasks.
- ```PreProcessing.py``` contains the tokenizer for preprocessing the raw data/
- ```PreTrainedBert.py``` loads the pretrained model into the ```model``` class
- ```Sampling.py``` contains different data sampling and augmentation techniques.
- ```Scheduler.py``` contains different learning rate schedulers for testing.
- ```upsample_ratio_test.ipynb``` tests different upsampling mask-to-fill ratios. Training log is stored in ```mask_ratio_training_log_xlnet```.

Experimental results for mask ratio test for upsampling is captured in the log file ```mask_ratio_training_log_xlnet```.

Please run ```pip install -r requirements``` before running any of the code.

**The code for generating the dev.txt and test.txt are contained in ```pipeline.ipynb```, task 2 section.**
