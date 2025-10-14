# Image Trajectory Optimal Control

Official GitHub repository for 
*[Training-Free Reward-Guided Image Editing via Trajectory Optimal Control(2025)](https://www.arxiv.org/abs/2509.25845)*.
Please refer to the paper if you want more details.

![fig1](https://github.com/user-attachments/assets/dcfe5b56-ae1b-4f25-9181-66f081994f71)

***

## 📑 Abstract
Recent advancements in diffusion and flow-matching models have demonstrated remarkable capabilities in high-fidelity image synthesis. 
A prominent line of research involves reward-guided guidance, which steers the generation process during inference to align with specific objectives. However, leveraging this reward-guided approach to the task of image editing, which requires preserving the semantic content of the source image while enhancing a target reward, is largely unexplored. 
In this work, we introduce a novel framework for training-free, reward-guided image editing. We formulate the editing process as a trajectory optimal control problem where the reverse process of a diffusion model is treated as a controllable trajectory originating from the source image, and the adjoint states are iteratively updated to steer the editing process. 
Through extensive experiments across distinct editing tasks, we demonstrate that our approach significantly outperforms existing inversion-based training-free guidance baselines, achieving a superior balance between reward maximization and fidelity to the source image without reward hacking.

## 🛠️ Requirements
```
conda create -n itoc python=3.11
conda activate itoc
pip install -r requirements.txt
```
The model checkpoint and data are too heavy to be included in this repo and can be found in ***[here](https://drive.google.com/drive/folders/170znWA5u3nC7S1mzF7RPNP5faAn56Q45?usp=sharing).***

## 🎯 Reward-guided image editing
Check out the arguments in the script files to see more details.

### 0. Hyperparameters

* `--deterministic`: If True, the initial trajectory for the source image is generated with deterministic DDIM inversion. If False, it is generated with Markovian DDPM forward process. **(ITOC)**
* `--n_iter`: Number of iterations for the optimization loop. **(GA, ITOC)**
* `--reward_multiplier`: Multiplier for the final reward function (and its gradient). **(ITOC)**
* `--depth`: The depth of the forward noising process for inversion. **(DPS, FreeDoM, TFG, ITOC)**
* `--lr`: Learning rate for the optimization process. **(GA, ITOC)**
* `--tfg_rho`: Guidance scale multiplied on $\nabla_{\x_t}r(\hat\x_{1|t})$. **(DPS, FreeDoM, TFG)**
* `--tfg_mu`: Guidance scale multiplied on $\nabla_{\hat\x_{1|t}}r(\hat\x_{1|t})$. **(TFG)**

### 1. Human Preference(w/ ImageReward)
     
The input image `--image_path` will be edited to achieve a higher human preference alignment with a given text prompt `--reward_prompt`.
   * ITOC(ours)
       ```
       python ./src/edit_demo.py --method_name itoc --reward_name ImageReward \
     --image_path ./assets/nature.png --reward_prompt "colorful painting, river flowing grass field with flowers." \
     --deterministic True --reward_multiplier 500 --n_iter 20 --lr 5e-3 --depth 0.5
       ```
   * Baselines: Gradient Ascent
       ```
       python ./src/edit_demo.py --method_name gradient_ascent --reward_name ImageReward \
     --image_path ./assets/nature.png --reward_prompt "colorful painting, river flowing grass field with flowers." \
     --n_iter 100  --lr 2.0
       ```
   * Baselines: Inversion + Guided sampling methods(DPS, FreeDoM, TFG)

     Change `--method_name` to `inversion_dps`, `inversion_freedom`, or `inversion_tfg` to run the corresponding method.
     You can also modify `--method_name` and the corresponding hyperparameters in the other scenarios below to run the baselines for those tasks.
       ```
       python ./src/edit_demo.py --method_name inversion_tfg --reward_name ImageReward \
     --image_path ./assets/nature.png --reward_prompt "colorful painting, river flowing grass field with flowers." \
     --depth 0.7 --tfg_rho 1.0 --tfg_mu 0.5
       ```
### 2. Style Transfer(w/ Gram matrix)

The input image `--image_path` will be edited to match the style of a given style image `--style_image_path`.
   * ITOC(ours)
       ```
       python ./src/edit_demo.py --method_name itoc --reward_name Gram_Diff \
     --image_path ./assets/portrait.png --style_image_path ./assets/style_ref.png \
     --deterministic True --reward_multiplier 1000 --n_iter 20 --lr 10e-3 --depth 0.5
       ```

### 3. Counterfactual Generation(w/ classifier logit)

The input image `--image_path` will be edited to alter the decision of the classifier toward the given target class `--reward_class`.
   * ITOC(ours)
       ```
       python ./src/edit_demo.py --method_name itoc --reward_name ImageNet1k_classifier \
     --image_path ./assets/ladybug.png --reward class 306 \
     --deterministic True --reward_multiplier 250 --n_iter 20 --lr 5e-3 --depth 0.5
       ```

### 4. Text-guided Image Editing(w/ CLIP cosine similarity)
The input image `--image_path` will be edited to align with a given text prompt `--reward_prompt`.
   * ITOC(ours)
       ```
       python ./src/edit_demo.py --method_name itoc --reward_name CLIP_Score \
     --image_path ./assets/face.png --reward_prompt "a face of a smiling man." \
     --deterministic True --reward_multiplier 1000 --n_iter 20 --lr 5e-3 --depth 0.5
       ```

## Citation
If you use this code in your research, please consider citing the paper:

```bibtex
@article{chang2025training,
  title={Training-Free Reward-Guided Image Editing via Trajectory Optimal Control},
  author={Chang, Jinho and Kim, Jaemin and Ye, Jong Chul},
  journal={arXiv preprint arXiv:2509.25845},
  year={2025}
}
```

## 💡 Acknowledgement
* The code for the adjoint state calculation is based on & modified from the official code of [Adjoint Matching](https://github.com/microsoft/soc-fine-tuning-sd).
* TFG code for TFG and other baselines(DPS, FreeDoM) is based on & modified from the official code of [TFG](https://github.com/YWolfeee/Training-Free-Guidance).