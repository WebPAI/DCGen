# Automatically Generating UI Code from Screenshot: A Divide-and-Conquer-Based Approach

This is the artifact for the paper ["Divide-and-Conquer: Generating UI Code from Screenshots."](https://arxiv.org/abs/2406.16386) The paper is accepted by the ACM International Conference on the Foundations of Software Engineering (FSE'2025). This artifact supplies the DCGen toolkit and supplementary materials for the paper.


This repository contains:

1. **Code implementation of DCGen**, i.e., the Python script and instructions to run DCGen to preprocess websites, segment images, and generate UI code from screenshots with DCGen algorithm. 
2. **Sample dataset**. The sample of our experiment data is available in `/data`. The full dataset is available on :hugs:[HuggingFace](https://huggingface.co/datasets/iforgott/DCGen)
3. **A user-friendly tool based on DCGen**.


Quick links: :tv:[Demo video](#Demo-video) | :pencil:[DCGen Examples](#Examples) | 🖥️[Code usage](#Code-usage) | 🔨[Tool usage](#DCGen-tool) | :hugs:[Dataset](https://huggingface.co/datasets/iforgott/DCGen)


# Abstract

To explore automatic design-to-code solutions, we begin with a motivating study on GPT-4o, identifying three key issues in UI code generation: element omission, distortion, and misarrangement. We find that focusing on smaller visual segments helps multimodal large language models (MLLMs) mitigate these failures. In this paper, we introduce DCGen, a divide-and-conquer approach that automates the translation of webpage designs into UI code. DCGen divides screenshots into manageable segments, generates descriptions for each, and reassembles them into a complete UI code for the full design. Extensive testing on real-world websites and various MLLMs demonstrates that DCGen improves visual similarity by up to 14% compared to competing methods. Human evaluations confirm that DCGen enables developers to implement webpages faster and with greater fidelity to the original designs. To our knowledge, DCGen is the first segment-aware, MLLM-based solution for generating UI code directly from screenshots.



# Demo video

This video demonstrates how developers can use DCGen to create a webpage from a UI design through *simple copy and paste*. DCGen enables users to review and regenerate code for specific image segments, easily replacing any erroneous code with the correct version for the final webpage.

https://github.com/user-attachments/assets/93e70ae9-c119-4838-94c8-6628be5af7d5

# Examples

Here are two examples from the usefulness study. DCGen demonstrates its effectiveness by significantly reducing element omissions and distortions, leading to faster development and improved webpage quality.

<img src="./assets/case_usefulness.png" alt="case_usefulness" style="zoom: 30%;" />


# Code usage

## 0. Setup

```she
pip install -r requirements.txt
playwright install
```


```python
from utils import *
import single_file
```

## 1. Save & Process Website

```python
single_file("https://www.overleaf.com", "./test.html")
simplify_html("test.html", "test_simplified.html", pbar=True)
driver = get_driver(file="./test.html")
take_screenshot(driver, "test.png")
```

## 2. Image Segmentation

```python
img_seg = ImgSegmentation("0.png", max_depth=1)
seg.display_tree()
```

## 3. DCGen 

The demo code and prompt can be found in `scripts/experiments.py`

```python
from scripts.experiments import *
# Please refer to scripts/experiments.py for baseline prompt and DCGen prompt

bot = GPT4("path/to/you/key.txt", model="gpt-4o")
seg_params = {
    "max_depth": 2,
    "var_thresh": 50,
    "diff_thresh": 45,
    "diff_portion": 0.9,
    "window_size": 50
}
import os
os.chdir("../data")
# run single experiment. Params: bot, input image path, output html path
dcgen(bot, "../data/demo/0.png", "./test.html") 
# run dcgen for entire folder. Params: bot, input image folder, output html folder
dcgen_exp(bot, "./demo/", "dcgen_demo", multi_thread=True, seg_params=seg_params)
# baseline
single_turn_exp(bot, "./demo/", f"direct_demo", prompt_direct, multi_thread=True)
# make sure to move placeholder.png into respective dir before taking screenshots
os.system("mv placeholder.png dcgen_demo/placeholder.png")
os.system("mv placeholder.png direct_demo/placeholder.png")
take_screenshots_for_dir("./dcgen_demo/", replace=True)
take_screenshots_for_dir("./direct_demo/", replace=True)
```

## 4. Evaluate 

0. Install requirements for the metric toolkit

    ```shell
    cd scripts
    bash install.sh
    ```

1. Modify configurations in `scripts/evaluate.py`: 

   ```python
   original_reference_dir = "../data/demo"
   test_dirs = {
       "dcgen": "../data/dcgen_demo",
       "direct": "../data/direct_demo",
   }
   exp_name = "test"
   ```

   The `original_reference_dir` contains original HTML files `1.html, 2.html, ...`, their corresponding screenshots `1.png, 2.png, ...`, and the placeholder image `placeholder.png`.

   The `test_dirs` contains the generated HTML files with the same name as the original ones `1.html, 2.html, ...`, and the placeholder image `placeholder.png`.

2. Run the evaluation script

	```shell
	python evaluate.py

Note that the fine-grained metrics can only be run on linux or mac.

# DCGen tool

**Run locally**

1. Start a server

  ```shell
  cd Tool
  python app.py
  ```

2. Visit http://127.0.0.1:5000 via local browser

3. Usage:

   **Generate image for entire screenshot**
   <img src="./assets/dcgenui1.png" alt="dcgenui1" style="zoom:20%;" />

   **View the code of any image segment**
   <img src="./assets/dcgenui2.png" alt="dcgenui2" style="zoom:20%;" />

   **Generate code for a image segment**
   <img src="./assets/dcgenui3.png" alt="dcgenui3" style="zoom:20%;" />

   
