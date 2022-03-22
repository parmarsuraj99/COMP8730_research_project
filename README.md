This repository contains code for the research project for course COMP8730. It uses HuggingFace datasets, HuggingFace Transformers and TensorFlow datasets.

The steps are divided into 
- loading, splitting and saving textual data - `load_save_lm_dataset.py`
- loading models, text data and training - `train_model.py`

You may need to install some extra libraries listed in `requirements.txt`.

The text file is relatively small so, one epoch takes around 20 minutes on Colab Pro.

After installing helping libraries, 
- run `python load_save_lm_dataset.py`
- run `python train_model.py`
- run `python train_model.py --from_scratch false`

The file `train_model.py` has two arguments
- `checkpoint`: which model/config to load from HF hub. Defaults to `ai4bharat/indic-bert`
- `from_scratch`: if `true` load from config else load pretrained. Defaults to `false`.
 
More fine grade training configuration can be done by modifying the values passed to `HFTrainer`.

--- 
After execution is done, we'll have checkpoints and training logs for both training procedures under the directories `results_scratch_False` and `results_scratch_True` respectilely.

For our experiments, we plan to integrate Weights&Biases logging.

--- 
To make the execution easy on Colab, 
we have combined the scripts in a notebook `COMP8730_proposed_solution_scripts_test.ipynb`, running this on colab would help you get started with experiment.