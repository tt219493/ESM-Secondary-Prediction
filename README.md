Secondary Structure Prediction (Q8) Using ESM Fine-Tuning
===
Wraps ESM Model from HuggingFace in Pytorch Lightning Module and fine-tunes for per-residue secondary structure prediction. Additionally, allows for embeddings to be outputted and used in other architectures.

**Original Project from COM SCI C121 @ UCLA**
___
Workflow 
--- 
### Data Retrieval
* [`ids-to-dssp`](https://github.com/tt219493/ids-to-dssp): Simple package created in support of this project.   
Retrieves per-residue secondary structure given a file with PDB IDs (e.g. [`train_tsv`](https://github.com/tt219493/IDs-to-DSSP/blob/main/data/train.tsv) provided in the original class project) from  [DSSP Website](https://pdb-redo.eu/dssp/) [1].
  * DSSP version 4 provides labels for left-handed Poly‐Proline II helices as "P", which is not present in other versions
  * DSSP version 4 also either does not provide a label (chain break) or provides the label "." for residues with no distinct secondary structure / loop 

* Benchmark Datasets: `CASP12` [2], `CB513` [3], `NEW364` [4], `TS115` [5] and additional training data from `NetSurfP2.0` [6] provided in [ProtTrans](https://github.com/agemagician/ProtTrans) Repo [4].

### Data Processing
* `Polars` was used to scan and store the data as a `LazyFrame`, which could then be easily queried and processed.
  * From the benchmark datasets, only the sequence strings and Q8 strings were selected.
*  The pre-trained `AutoTokenizer` provided by `HuggingFace` was used to pre-process the sequences into input id tokens.
* Any overlapping sequences to the training data were removed to prevent any data leakage.
* These pre-tokenized datasets are saved in the `data/benchmark` directory.

(Creating windows of the sequences are implemented but currently not tested since it drastically increases training time.   However, I do believe it could lead to a better model as it is augmenting the training data and could help the model better capture short-range dependencies and interactions)
___
### ESM Model
[Pre-trained ESM2 Model](https://huggingface.co/facebook/esm2_t30_150M_UR50D) with 30 layers and 150M parameters were used for main predictions. The main reason for using this pre-trained version was because it is the largest model that can be loaded into Colab / Kaggle without crashing.   
The lightest weight model was used for hyperparameter testing before scaling up.  

**Training & Fine-tuning:** Utilized 3 transfer learning steps to fine-tune the ESM model for benchmark testing
1. ESM Model with 10 labels, training on data gathered by `ids-to-dssp`. In this dataset, chain breaks and those labeled as "." by DSSP version 4 are labeled separately (10 labels).
2. Pre-trained ESM Model with 10 labels training on the same data with chain breaks and residues labled as "." to have the same label (9 labels). 
3. Pre-trained ESM Model with 10 labels training on `NetSurf` training data with 8 labels.

By utilizing transfer learning, the model can learn first the updated DSSP version 4 labels, which appears to improve accuracy on 8 label predictions.

**Hyperparameters**: Utilized default hyperparameters from HuggingFace.
* **3 epochs** -- usually overfits after 3
* **Batch size of 2** -- lower batch sizes are better but take longer; 2 appears to be the best middleground
* **Optimizer: AdamW** -- 
  * Learning Rate: 5e-5
  * Others are default Pytorch parameters
* **Scheduler: Linear Decay**

### ESM Model Accuracy (Q8):

| ESM Step    | CASP12  | CB513   | NEW364  | TS115   |  
| --------    | ------- | ------- | ------- | ------- |
| 1           | 56.9    |  51.2   | 60.6    |  62.5   |
| 2           | 66.6    |  64.0   | 70.4    |  73.5   |
| 3           | 68.3    |  66.0   | 71.8    |  74.6   |



___

### Dimensionality Reduction: Encoder
The embeddings of the inputs from the last layer of the fine-tuned ESM model will be turned into a DataFrame to be utilized as input to Gradient Boosting Models.   
However, the embedding size of 1280 was too large for Colab / Kaggle. Thus, an encoder / linear layer to reduce the embedding size from 1280 to 320 was utilized.

**Architecture**: 
* Embedding: (N, L, 1280)
* Encoder: (N, L, 1280) --> (N, L, 320)
* ReLU Activation Layer
* Classifier: (N, L, 320) --> (N, L, 10)

**Training**: 
* Utilized transfer learning similar to ESM
* 10 epochs with early stopping
* Batch size of 2
* Optimizer: Adam with default Pytorch parameters

### Encoder Accuracy (Q8):

| Encoder Step    | CASP12  | CB513   | NEW364  | TS115   |  
| --------        | ------- | ------- | ------- | ------- |
| 1               | 67.5    |  65.0   | 70.7    |  73.6   |
| 2               | 68.6    |  64.4   | 71.0    |  74.0   |
| 3               | 68.9    |  65.5   | 71.8    |  74.7   |

The main goal of the encoder was to try to retain as much information from the ESM embeddings in a lower dimensional space which it appears to do well as it has similar accuracies to the original fine-tuned ESM
___

### Gradient Boosting & Ensembling
Using the reduced size embeddings from the Encoder, XGBoost and LightGBM classifiers were trained and their predictions were ensembled.

**Training**:   
Hyperparameters are currently unoptimized.  
Trained on `NetSurf` training data only.

### Gradient Boosting & Ensemble Accuracy (Q8):

| Model           | CASP12  | CB513   | NEW364  | TS115   |  
| --------        | ------- | ------- | ------- | ------- |
| XGBM            | 67.8    |  73.8   | 71.2    |  74.3   |
| LGBM            | 67.6    |  73.8   | 71.1    |  74.2   |
| Ensemble        | 67.7    |  73.8   | 71.2    |  74.3   |
___
### Overall Results & Comparisons to State-of-the-Art Models (Q8)
Data from [4] & [9]  
| Model            | CASP12  | CB513   | NEW364  | TS115   |  
| --------         | ------- | ------- | ------- | ------- |
| ESM (Step 3)     | 68.3    |  66.0   | 71.8    |  74.6   |
| Encoder (Step 3) | 68.6    |  65.5   | 71.8    |  74.0   |
| Ensemble         | 67.7    |  73.8   | 71.2    |  74.3   |
| NetSurfP-2.0  [9]| 69.9    |  71.3   | -       |  74.0   |
| NetSurfP-3.0  [9]| 66.9    |  71.1   | -       |  74.9   |
| NetSurfP-2.0  [4]| 70.3    |  72.3   | 73.9    |  75.0   |
| ProtT5-XL-U50 [4]| 70.5    |  74.5   | 74.5    |  77.1   |

Although, my current models do not surpass the current SOTA models, they have comparable Q8 accuracy to these models.   
Additionally, one of the major improvements in NetSurfP3.0 [9] is its fast prediction times at 1000 proteins in ~100 seconds. While not fully tested, the runtimes of my models appear to match or surpass those prediction times at 1000 proteins in ~60 seconds.

___
### Future Improvements
* More extensive hyperparameter searches, especially for the Gradient Boosting Models.
* Using feature combinations with features utilized by `NetSurfP2.0` [6] and similar to the implemention using `ESM-1b` [8]  
* Implementing NetSurf architecture on top of embeddings similar to `NetSurfP3.0` [9]

___
### References
> [1] Hekkelman ML, Salmoral DÁ, Perrakis A, Joosten RP. DSSP 4: FAIR annotation of protein secondary structure. Protein Science. 2025;34(8):e70208. https://onlinelibrary.wiley.com/doi/10.1002/pro.70208. doi:10.1002/pro.70208

> [2] Abriata LA, Tamò GE, Monastyrskyy B, Kryshtafovych A, Dal Peraro M. Assessment of hard target modeling in CASP12 reveals an emerging role of alignment‐based contact prediction methods. Proteins: Structure, Function, and Bioinformatics. 2018;86(S1):97–112. https://onlinelibrary.wiley.com/doi/10.1002/prot.25423. doi:10.1002/prot.25423

> [3] Cuff JA, Barton GJ. Evaluation and improvement of multiple sequence methods for protein secondary structure prediction. Proteins: Structure, Function, and Genetics. 1999;34(4):508–519. https://onlinelibrary.wiley.com/doi/10.1002/(SICI)1097-0134(19990301)34:4<508::AID-PROT10>3.0.CO;2-4. doi:10.1002/(SICI)1097-0134(19990301)34:4<508::AID-PROT10>3.0.CO;2-4

> [4] Elnaggar A, Heinzinger M, Dallago C, Rehawi G, Wang Y, Jones L, Gibbs T, Feher T, Angerer C, Steinegger M, et al. ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning. IEEE Transactions on Pattern Analysis and Machine Intelligence. 2022;44(10):7112–7127. https://ieeexplore.ieee.org/document/9477085/. doi:10.1109/TPAMI.2021.3095381

> [5] Yang Y, Gao J, Wang J, Heffernan R, Hanson J, Paliwal K, Zhou Y. Sixty-five years of the long march in protein secondary structure prediction: the final stretch? Briefings in Bioinformatics. 2016 Dec 31:bbw129. https://academic.oup.com/bib/article-lookup/doi/10.1093/bib/bbw129. doi:10.1093/bib/bbw129

> [6] Klausen MS, Jespersen MC, Nielsen H, Jensen KK, Jurtz VI, Sønderby CK, Sommer MOA, Winther O, Nielsen M, Petersen B, et al. NetSurfP‐2.0: Improved prediction of protein structural features by integrated deep learning. Proteins: Structure, Function, and Bioinformatics. 2019;87(6):520–527. https://onlinelibrary.wiley.com/doi/10.1002/prot.25674. doi:10.1002/prot.25674

> [7] Lin Z, Akin H, Rao R, Hie B, Zhu Z, Lu W, Smetanin N, Verkuil R, Kabeli O, Shmueli Y, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. Science. 2023;379(6637):1123–1130. https://www.science.org/doi/10.1126/science.ade2574. doi:10.1126/science.ade2574

> [8] Rives A, Meier J, Sercu T, Goyal S, Lin Z, Liu J, Guo D, Ott M, Zitnick CL, Ma J, et al. Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. Proceedings of the National Academy of Sciences. 2021;118(15):e2016239118. https://pnas.org/doi/full/10.1073/pnas.2016239118. doi:10.1073/pnas.2016239118

> [9] Høie MH, Kiehl EN, Petersen B, Nielsen M, Winther O, Nielsen H, Hallgren J, Marcatili P. NetSurfP-3.0: accurate and fast prediction of protein structural features by protein language models and deep learning. Nucleic Acids Research. 2022;50(W1):W510–W515. https://academic.oup.com/nar/article/50/W1/W510/6596854. doi:10.1093/nar/gkac439


