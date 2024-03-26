# myppal: My Plug-and-Play Active Learning for Object Detection
## Exploring active learning methods that maximize sample efficiency

This project is based on the paper [Plug and Play Active Learning For Object Detection](https://arxiv.org/abs/2211.11612). It is an implementation of PPAL algorithm in Detectron2, instead of MMDetection as original paper's source code.

myppal enables you to train an Object Detection model with a specified combination of uncertainty-based sampling and diversity-based sampling. You can compare models that only use uncertainty, only diversity, half-half, or any proportion of the 2 sampling methods.

<p align="center">
<img src="Resources for README/PPAL pipeline.jpg" style="width:75%;"/>
</p>

Algorithm from the paper beat all previous similar active learning methods, but it used a constant ratio of candidate pool picked by uncertainty, so diversity sampler also had the same options to choose from. I wanted to see how do these 2 methods work together, and which should be prioritized.

More details have been included in the [thesis](myppalThesisBen.pdf) I wrote.
I also made a [video](https://youtu.be/27F1JQ2qatY) describing the whole background of this project and commenting the important thesis' parts in a more approachable way.


## Usage

1. Install Dependencies like detectron2. I had some problems first running it, but just install everything that will cause problems if lacking.
2. Setup your dataset with annotations in YOLO format ([.txt file for each image](https://docs.cogniflow.ai/en/article/how-to-create-a-dataset-for-object-detection-using-the-yolo-labeling-format-1tahk19/)), as a following structure (for default setup):
```
myppal
|
\-- data
    |
    \ full-data
      |
      |-- train
      |   |
      |   |-- imgs
      |   |
      |   \-- anns
      |
      \-- val
          |
          |-- imgs
          |
          \-- anns
```
3. Enter your class names in *class.names* file, each class name on a different line.
4. Run the main.py script, as shown in my experiment.sh examples. Check the code for all parameters or see config.py for default values. 

    The most important parameter is the UD-split that determines the whole pipeline. See the thesis document for more explanation on how this parameter works.

    Please be patient with my script, it's trying its best. It usually took on my RTX3060 laptop around 3h for the longer rounds. The more diversity, the longer the run. Especially if you do everything as me, I redirect all console outputs to a file, so it might look like nothing is happening.

5. Evaluate your new model with evaluate.py
6. Plot its loss wiht plotloss.py
7. See by yourself how well its doing with predict.py

If you have any issues or run into bugs - let me know. I did some code cleanup without even testing it because Im chill like that.

## Q&A

**How is Active Learning itself implemented?**

Due to a lack of time and knowledge, I implemented the first idea that came to my mind: Since detectron2 training was based on providing a folder of images and annotations, I simply create a copy of a whole dataset and move it to temporary subdirectories for training, and its surprisngly fast through code. It all gets cleaned up afterwards, but that means that you have 2 datasets at one time, which is not great memory-wise. Be careful. I would love to just provide one dataset and specific filenames to train on, but I could not do it.

---

**Is the algorithm deterministic?**

Nope. And it won't be. Even for the same random seeds, the outputs will slightly vary. detectron2 does not guarantee full determistic behavior, only reproducibility.

---
**Why does diversity take so long to compute? Couldn't you do it on a GPU?**

Please dont say that.

---
**Don't you feel like you should clean the code or even rewrite the whole project?**

Oh yeah I do. It's just not as important a project. I'd happily approach the whole Active Learning topic differently instead. Maybe from Reinforcement Learning side.

---
**What resources did you find helpful?**

I would actually like to thank *Computer Vision Engineer*, as without his tutorial I would probably be lost as to where to even begin. My starting point was also based on his code, even though what he did was just nicely organize official tutorial from detectron2. You can find his base repo (here)[https://github.com/computervisioneng/train-object-detector-detectron2.git].

Then the rest was up to me based on not beginner-friendly detectron2 documentation and the original algorithm paper that included equations and helpful tips.


## Citation

Oh yeah I dare you to cite me, no idea has that even works, Im not an arXiv certified guy. Full name is Beniamin Weyna, though.







