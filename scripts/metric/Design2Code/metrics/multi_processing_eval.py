from Design2Code.metrics.visual_score import visual_eval_v3_multi
from multiprocessing import Pool
import contextlib, joblib
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import json
import os
import shutil

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def print_multi_score(multi_score):
    _, final_size_score, final_matched_text_score, final_position_score, final_text_color_score, final_clip_score = multi_score
    print()
    print("Block-Match: ", final_size_score)
    print("Text: ", final_matched_text_score)
    print("Position: ", final_position_score)
    print("Color: ", final_text_color_score)
    print("CLIP: ", final_clip_score)
    print("--------------------------------\n")


def eval(orig_reference_dir = "../../../data/original", test_dirs = {"DCGen": "../../../data/dcgen_gpt4"}, output_path = "metrics/res_dict_testset_final.json"):
    debug = False
    multiprocessing = True
    eval_name = "testset_final"

    ## copy the original reference directory to a new directory
    ## because we will be creating new screenshots
    reference_dir = "../testset_final_" + eval_name
    os.makedirs(reference_dir, exist_ok=True)
    print(os.getcwd())
    for filename in os.listdir(orig_reference_dir):
        if filename.endswith(".html") or filename == "placeholder.png":
            shutil.copy(os.path.join(orig_reference_dir, filename), os.path.join(reference_dir, filename))
    print ("copied original reference directory to ", reference_dir)

    file_name_list = []

    ## check if the file is in all prediction directories

    for filename in os.listdir(reference_dir):
        if filename.endswith(".html"):
            if all([os.path.exists(os.path.join(test_dirs[key], filename)) for key in test_dirs]):
                file_name_list.append(filename)

    print ("total #egs: ", len(file_name_list))

    input_lists = []
    for filename in file_name_list:

        input_pred_list = [os.path.join(test_dirs[key], filename) for key in test_dirs]
        original = os.path.join(reference_dir, filename)

        input_list = [input_pred_list, original]
        input_lists.append(input_list)

    # print ("input_list: ", input_lists)
    if multiprocessing:
        with tqdm_joblib(tqdm(total=len(input_lists))) as progress_bar:
            return_score_lists = list(tqdm(Parallel(n_jobs=8)(delayed(visual_eval_v3_multi)(input_list, debug=debug) for input_list in input_lists), total=len(input_lists)))
    else:
        return_score_lists = []
        for input_list in tqdm(input_lists):
            return_score_list = visual_eval_v3_multi(input_list, debug=debug)
            return_score_lists.append(return_score_list)
        # print ("return lists: ", return_score_lists)
    
    res_dict = {}
    for key in test_dirs:
        res_dict[key] = {}

    for i, filename in enumerate(file_name_list):
        idx = 0
        return_score_list = return_score_lists[i]
        # print ("return score list: ", return_score_list)
        if return_score_list:
            for key in test_dirs:
                if multiprocessing:
                    matched, final_score, multi_score = return_score_list[idx]
                else:
                    matched = return_score_list[idx][0]
                    final_score = return_score_list[idx][1]
                    multi_score = return_score_list[idx][2]
                idx += 1
                current_score = [final_score] + [item for item in multi_score]
                res_dict[key][filename] = current_score
        else:
            print (filename + " didn't get a score")
            for key in test_dirs:
                res_dict[key][filename] = [0, 0, 0, 0, 0, 0]

    ## cache all scores 
    with open(output_path, "w") as f:
        json.dump(res_dict, f, indent=4)

    for key in test_dirs:
        print(key)
        values = list(res_dict[key].values())
        # print (values)
        current_res = np.mean(np.array(values), axis=0)
        # print(current_res)
        print_multi_score(current_res)


if __name__ == "__main__":
    import argparse
    import ast

    parser = argparse.ArgumentParser(description="Run multi-processing evaluation")
    parser.add_argument("--orig_reference_dir", type=str, 
                        default="../../../data/original", 
                        help="Path to the original reference directory")
    parser.add_argument("--test_dirs", type=str, 
                        default="{'DCGen': '../../../data/dcgen_gpt4'}", 
                        help="Dictionary of test directories (e.g., \"{'DCGen': '../../../data/dcgen_gpt4'}\")")
    parser.add_argument("--output_path", type=str, 
                        default="metrics/res_dict_testset_final.json", 
                        help="Output path for the result JSON file")
    args = parser.parse_args()

    # Convert test_dirs from string to dictionary safely using ast.literal_eval
    try:
        test_dirs = ast.literal_eval(args.test_dirs)
    except Exception as e:
        print("Error parsing test_dirs. Using default value.")
        test_dirs = {"DCGen": "../../../data/dcgen_gpt4"}

    eval(orig_reference_dir=args.orig_reference_dir, test_dirs=test_dirs, output_path=args.output_path)
    