ESM Secondary Structure Prediction
=== 
Jupyter notebooks containing workflow to fine-tuning the ESM model from HuggingFace and ensembling with gradient boost methods for secondary structure predictions.  
_In the process of expanding this project._

### Original Project from COM SCI C121 @ UCLA
___
Workflow 
--- 
### Data Pre-Processing
* `Pandas` : used `read_csv()` to extract 'id' and 'secondary_structure' from `train.tsv` and convert into a DataFrame.  
Since 'id' was formatted as `{pdb_id}_{amino_acid}_{index}`, those were split into separate columns. 

* `SeqIO` from `Biopython`: used to parse `sequences.fasta` to get IDs and sequences. This was then joined onto the DataFrame. 

There were some issues with the `.fasta` file provided, namely that only one sequence was provided for each id while there can be many sequences corresponding to one ID. Thus, there were indices that were longer than the length of the sequence which caused some problems, so those were simply removed. Sequences were also likely assigned to the wrong ID. Additionally, the pre-processing of the data was fairly inefficient.

(In support for an expansion to this project, I created [`ids-to-dssp`](https://github.com/tt219493/ids-to-dssp) for functions to do this pre-processing more efficiently and provide complete labeled data. )

* Creating windows: Windows of the sequence and its corresponding secondary structure label were created. Various window and step sizes were tested along with using the entire sequence as input. The smallest window size was 8 while the largest was 256.

This pre-processing was also repeated for the test set.

---

### Using ESM Model
[Pre-trained ESM Model](https://huggingface.co/facebook/esm2_t6_8M_UR50D) using 6 layers and 8M parameters was imported from `HuggingFace` which is the smallest and most lightweight version provided. 
The corresponding tokenizer was also imported and the windowed data was tokenized prior to model input.

* **Training & Fine-tuning:** 5-fold cross validation was utilized with one version having 5 different models while the other version had one model train on each fold.
  * As for hyperparameters,`Adam` was used as the optimizer, the learning rate was `2e-5`, the number of epochs was `4` and the batch size was the lowest available that was able to fit into memory. A linear warmup for 10% of the training data was also used. These were chosen after training a few models and seeing validation results.
  * Various window/step sizes were utilized as well as combining the different window sizes into one dataset. 
  * Ensembling models that were trained on different window sizes was also tested. 

* **Results:** Training one model on different folds generally performed better than ensembling 5 different models. Larger windows generally performed better, though combining them into a dataset performed best. Ensembling the predictions of the best performing models (i.e. the models trained on larger windows or trained on combined datasets) also resulted in better performance.   

Overall, prediction accuracy from the best models was 60+%

---

### Using Gradient Boosting
Instead of using the fine-tuned ESM model predictions directly, the embeddings from the final layer of the ESM model were then used as features of a gradient boosted model, such as CatBoost or XGBoost. 
Different gradient boosting models were trained using the embeddings from different ESM models.

* **Results:** The gradient boosting model that performed the best was the XGBoost Classifier. Similar to the ESM model fine-tuning, the best results came from embeddings of ESM models with larger windows or combined datasets. It also similarly improved when ensembling the results of the best XGBoost models.  

Overall, prediction accuracy increased further to 65+%.

---

Possible Improvements
---
* Since some data points were removed and some sequences were incorrectly tied to the wrong ID (since multiple sequences can correspond to an ID), improvements could come from simply having better labeled training data.
  * I believe this can result in the biggest improvement and will be easier to implement due to my new package [`ids-to-dssp`](https://github.com/tt219493/ids-to-dssp)
* Since memory and speed of training was a limitation, the smallest pre-trained ESM model was used, so using a bigger model with more layers and parameters could improve results.
* More extensive hyperparameter searching could also result in better results.
* Improving on model architecture / pipeline can also result in improvements but the first three points are easier to implement and might yield better results.

Notebook Summary
---
In `ESM.ipynb`, exploratory data analysis and combining the information from `train.tsv` and `sequences.fasta` into a Pandas DataFrame were done. Then, using that DataFrame, the pre-trained ESM model from HuggingFace was loaded and fine-tuned. Additionally, different window lengths of the sequences were also tested. Predictions on the test set after fine-tuning resulted in about 60% accuracy.

In `GradBoost.ipynb`, the embeddings from the last layer of the fine-tuned ESM model were used as inputs to various gradient boosting models, such as CatBoost (not shown in notebook) and XGBoost. Using these embeddings resulted in a slight increase in accuracy to about 65%.
