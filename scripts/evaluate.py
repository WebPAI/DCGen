import os
import sys
from rapidfuzz import fuzz

from tqdm import tqdm
import json
sys.path.append("../")

# original_reference_dir = "../data/original"
original_reference_dir = "../data/demo"
test_dirs = {
    "dcgen": "../data/dcgen_demo",
    "direct": "../data/direct_demo",
}
exp_name = "test"



def code_similarity(file1, file2):
    with open(file1, 'r', encoding='utf-8') as f:
        html1 = f.read()
    with open(file2, 'r', encoding='utf-8') as f:
        html2 = f.read()
    similarity = fuzz.ratio(html1, html2)
    return similarity

def code_sim_for_dir(ref_dir, test_dir):
    test_files = os.listdir(test_dir)
    results = {}
    for file in tqdm(test_files):
        try:
            if file.endswith(".html"):
                ref_file = os.path.join(ref_dir, file)
                test_file = os.path.join(test_dir, file)
                similarity = code_similarity(ref_file, test_file)
                results[file] = similarity
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue
    return results

def code_sim_experiment(ref_dir, test_dirs, experiment_name):
    results = {}
    for key in test_dirs:
        results[key] = code_sim_for_dir(ref_dir, test_dirs[key])
    with open(f"{experiment_name}_code_sim.json", "w") as f:
        json.dump(results, f, indent=4)

    return results
    


import os
from tqdm import tqdm
import open_clip
import torch
from PIL import Image

class CLIPScorer:
    def __init__(self, model_name='ViT-B-32-quickgelu', pretrained='openai'):
        """
        Initializes the CLIPScorer with the specified model.

        Args:
            model_name (str): The name of the CLIP model to use.
            pretrained (str): Specifies whether to load pre-trained weights.
        """
        self.device = "cuda" if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.to(self.device)

    def score(self, img1: Image.Image, img2: Image.Image) -> float:
        """
        Calculates the CLIP score (cosine similarity) between two images.

        Args:
            img1 (Image.Image): The first image as a PIL Image.
            img2 (Image.Image): The second image as a PIL Image.

        Returns:
            float: The cosine similarity score between the two images.
        """
        # Preprocess the images
        image1 = self.preprocess(img1).unsqueeze(0).to(self.device)
        image2 = self.preprocess(img2).unsqueeze(0).to(self.device)

        # Get the image features from CLIP using openclip
        with torch.no_grad():
            image1_features = self.model.encode_image(image1)
            image2_features = self.model.encode_image(image2)

        # Normalize the features to unit length
        image1_features /= image1_features.norm(dim=-1, keepdim=True)
        image2_features /= image2_features.norm(dim=-1, keepdim=True)

        # Calculate cosine similarity between the two image features
        cosine_similarity = torch.nn.functional.cosine_similarity(image1_features, image2_features)
        return cosine_similarity.item()
    

def clip_for_dir(ref_dir, test_dir):
    test_files = os.listdir(test_dir)
    results = {}
    clip_scorer = CLIPScorer()
    for file in tqdm(test_files):
        if file.endswith(".png") and not "placeholder" in file:
            ref_file = os.path.join(ref_dir, file)
            test_file = os.path.join(test_dir, file)
            ref_img = Image.open(ref_file)
            test_img = Image.open(test_file)
            similarity = clip_scorer.score(ref_img, test_img)
            results[file] = similarity
    return results

def clip_experiment(ref_dir, test_dirs, experiment_name):
    # take_screenshots_for_dir(ref_dir, replace=True)
    results = {}
    for key in test_dirs:
        results[key] = clip_for_dir(ref_dir, test_dirs[key])
    with open(f"{experiment_name}_clip.json", "w") as f:
        json.dump(results, f, indent=4)

    return results

if __name__ == "__main__":

    print(original_reference_dir)

    results = clip_experiment(original_reference_dir, test_dirs, exp_name)
    print("CLIP Results:")
    for key in results:
        print(key)
        print(sum(results[key].values()) / len(results[key]))


    results = code_sim_experiment(original_reference_dir, test_dirs, exp_name)
    print("Code Similarity Results:")
    for key in results:
        print(key)
        print(sum(results[key].values()) / len(results[key]))

    ########## For calculating fine-grained metrics ##########
    ########## Please run install.sh to install necessary dependencies ##########
    ########## This code can only be run on linux or mac ##########
    # from Design2Code.metrics.multi_processing_eval import eval
    # os.chdir("./metric/Design2Code")
    # original_reference_dir = "../../" + original_reference_dir
    # test_dirs = {key: "../../" + value for key, value in test_dirs.items()}
    # eval(original_reference_dir, test_dirs, f"../../{exp_name}_visual.json")