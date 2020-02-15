To replicate my entire training process:

- The 2 folder "countryside" and "metropolitian" of the dataset is expected in the folder "data"

- The image "metropotilian/000532.gif" must be manually convert to another type such as ".jpg" or ".png" since opencv can't read ".gif"

- Run "preprocess_data.py" to run kmean clusters on the dataset. The results is saved as "data/data.pkl" and "data/label.pkl"

- Run "resample.py" to sub-sample the dataset. Result is saved as "data/sampled.pkl"

- Run "estimate_d.py" to estimate the d parameters from the sampled dataset.

- Run "train.py" to train the model

For demo, run "web.py"

For colors visualization (as explained in the report), run "visualize.py"


