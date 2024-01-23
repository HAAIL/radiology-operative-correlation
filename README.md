# Towards Automating the Quality Control in Radiology: Leveraging Large Language Models to Extract Correlative Findings in Radiology and Operative Reports




>**Abstract:**<div style="text-align: justify"> Radiology Imaging plays a pivotal role in medical diagnostics, providing clinicians with insights into patient health and guiding the next steps in treatment. The true value of a radiological image lies in the accuracy of its accompanying report. To ensure the reliability of these reports, they are often cross-referenced with operative findings. The conventional method of manually comparing radiology and operative reports is labor-intensive and demands specialized knowledge. This study explores the potential of a Large Language Model (LLM) to simplify the radiology evaluation process by automatically extracting pertinent details from these reports, focusing especially on the shoulder's primary anatomical structures. A fine-tuned LLM identifies mentions of the supraspinatus tendon, infraspinatus tendon, subscapularis tendon, biceps tendon, and glenoid labrum in lengthy radiology and operative documents. Initial findings emphasize the model's capability to pinpoint relevant data, suggesting a transformative approach to the typical evaluation methods in radiology. This method not only makes the evaluation process more efficient but also offers potential enhancements to the learning curve for radiologists, promoting ongoing advancements in radiological assessments.
 </div>

## Description
<div style="text-align: justify">

-  fine-tuning of the Falcon Large Language Model (LLM) to identify pertinent details in radiology or operative reports when queried about the structural integrity of the shoulder
- we structured sections of the radiology and operative reports into a series of question-answer (Q-A) pairs. This restructuring process was designed to pair a question about the integrity of the shoulder structure (e.g. “What is the integrity of the Supraspinatus tendon?”), with an answer sourced from the specialist-annotated reports (e.g. “There is a full-thickness tear of the distal supraspinatus tendon”).
</div>



## Setup  
The required dependencies are listed in "requirements.txt". To install the dependencies run the following command:
  ```shell script
pip install -r  requirements.txt
```




#### Training
- The main training script is placed in `FalconTrainer.py`.
Additionally, if you have wandb account, you can visualize create a project to monitor the training logs.
For this purpose `WANDB_API_KEY` needs to be specified in `Setting.py`



