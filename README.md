This repository contains code for the research project for course COMP8730. It uses HuggingFace datasets, HuggingFace Transformers and TensorFlow datasets.

The steps are divided into 
- loading, splitting and saving textual data - `load_save_lm_dataset.py`
- loading models, text data and training - `train_model.py`

You may need to install some extra libraries listed in `requirements.txt`.

The text file is relatively small so, one epoch takes around 20 minutes on Colab Pro.

After installing helping libraries, 
- run `python load_save_lm_dataset.py`
- run `python train_model.py --wandb`
- run `python train_model.py --from_scratch false`

The file `train_model.py` has four arguments
- `checkpoint`: which model/config to load from HF hub. Defaults to `ai4bharat/indic-bert`
- `from_scratch`: if `true` load from config else load pretrained. Defaults to `false`.
- `wandb`: if `true` use [Weights&Biases](https://docs.wandb.ai/) for training metrics logging. Defaults to `false`.
- `chkpt_dir`: helpful to set checkpoint directory to mounted Gdrive as the runtime reset deletes files. defaults to current directory.
 
More fine grade training configuration can be done by modifying the values passed to `HFTrainer`.

--- 
After execution is done, we'll have checkpoints and training logs for both training procedures under the directories `results_scratch_False` and `results_scratch_True` respectilely.

For our experiments, we plan to integrate Weights&Biases logging.

The language model training logs on W&B can be found here:
- `from_scratch: true`: [./results_scratch_True](https://wandb.ai/parmarsuraj99/huggingface/runs/18gpi9qe?workspace=user-parmarsuraj99)
- `from_scratch: false`(loading from pretrained model): [./results_scratch_False](https://wandb.ai/parmarsuraj99/huggingface/runs/1e7ha4ti?workspace=user-parmarsuraj99)

--- 
To make the execution easy on Colab, 
we have combined the scripts in a notebook `COMP8730_proposed_solution_scripts_test.ipynb`, running this on colab would help you get started with experiment.

---
Once the models have been saved with all the checkpoints, we used Google Drive to save the checkpoints and the ease of re-loading them. We decided to use Colab for final training to make customizations easy to training. 

run `notebooks/COMP8730_proposed_solution_author_prediction.ipynb` to save and evaluate on Author prediction task
