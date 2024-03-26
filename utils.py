from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.config import get_cfg
from detectron2 import model_zoo
from loss import ValidationLoss
import os
import shutil
import pickle
import math
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
import config
import torch
import random
import torch.multiprocessing as mp
from torch.multiprocessing import Manager
import time

def seedEverything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Set all the seeds")

# Base functions
def get_custom_cfg(output_dir, learning_rate, batch_size, iterations, checkpoint_period, model, device, nmr_classes, customConfigFile=None):
    cfg = get_cfg()

    # Merge the model's default configuration file with the default Detectron2 configuration file.
    if customConfigFile is None:
        print(f"Getting config for: {model}")
        cfg.merge_from_file(model_zoo.get_config_file(model))    
    else:
        print(f"Getting local config saved as: {customConfigFile}")
        with open(customConfigFile, 'rb') as file:
            cfg = pickle.load(file)
    
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.VAL = ("val",)
    cfg.DATASETS.TEST = ()

    if device in ['cpu']:
        cfg.MODEL.DEVICE = 'cpu'

    

    # Set the model weights to the ones pre-trained on the COCO dataset or to our local pth file if we provide one
    if model[-3:] == "pth":
        cfg.MODEL.WEIGHTS = model
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)

    cfg.SEED = config.SEED         # SET SEED FOR DETERMINISTIC BEHAVIOR
    cfg.DATALOADER.NUM_WORKERS = 0 # NO PARALLELISM FOR DETERMINISTIC BEHAVIOR
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.CHECKPOINT_PERIOD = checkpoint_period
    cfg.SOLVER.BASE_LR = learning_rate
    cfg.SOLVER.MAX_ITER = iterations
    cfg.SOLVER.STEPS = []
    cfg.MODEL.RETINANET.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256 # Not always using Retinanet
    cfg.MODEL.RETINANET.NUM_CLASSES = nmr_classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = nmr_classes
    cfg.OUTPUT_DIR = output_dir

    return cfg

def get_dicts(img_dir, ann_dir):
    dataset_dicts = []
    for idx, file in enumerate(os.listdir(ann_dir)):
        # annotations only in yolo format for now
        record = {}
        filename = os.path.join(img_dir, file[:-4] + config.IMG_EXTENSION)
        height, width = cv2.imread(filename).shape[:2]
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = []
        with open(os.path.join(ann_dir, file)) as r:
            temp = r.readlines()
            lines = [l.strip() for l in temp]

        for _, line in enumerate(lines):
            if len(line) > 2:
                label, cx, cy, w_, h_ = line.split(' ')
                obj = {
                    "bbox": [int((float(cx) - (float(w_) / 2)) * width),
                             int((float(cy) - (float(h_) / 2)) * height),
                             int(float(w_) * width),
                             int(float(h_) * height)],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": int(label),
                }

                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def register_datasets(root_dir, class_list_file):
    with open(class_list_file, 'r') as reader:
        classes_ = reader.read().splitlines()
    for d in ['train', 'val']:
        if d in DatasetCatalog.list():
            DatasetCatalog.remove(d)
            DatasetCatalog.register(d, lambda d=d: get_dicts(os.path.join(root_dir, d, config.F_IMAGES),
                                                            os.path.join(root_dir, d, config.F_ANNS)))
        else:
            DatasetCatalog.register(d, lambda d=d: get_dicts(os.path.join(root_dir, d, config.F_IMAGES),
                                                            os.path.join(root_dir, d, config.F_ANNS)))
        MetadataCatalog.get(d).set(thing_classes=classes_)
    return len(classes_)

def train(output_dir, data_dir, class_list_file, learning_rate, batch_size, iterations, checkpoint_period, device,
          model, customConfig=None):
    print(f"\n### Training started :) ###\n")
    nmr_classes = register_datasets(data_dir, class_list_file)
    print(f"###### NUMBER OF CLASSES: {nmr_classes} #######")

    cfg = get_custom_cfg(output_dir, learning_rate, batch_size, iterations, checkpoint_period, model, device, nmr_classes, customConfig)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    val_loss = ValidationLoss(cfg)
    trainer.register_hooks([val_loss])
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
    trainer.resume_or_load(resume=False)
    trainer.train()

    # After this training is done save training config and metadata
    # so another process can continue from that point
    with open(config.DT2_CONFIG_FILENAME, 'wb') as file:
        pickle.dump(cfg, file)

    with open(config.METADATA_FILENAME, 'wb') as file:
        pickle.dump(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), file)

# Custom functions to manage files for AL
def countFiles(directory):
    count = 0
    for listing in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, listing)):
            count += 1
    return count

def MoveSpecificFiles(source, destination, filenames):
    print(f"\n************ Moving {len(filenames)} files from {source} to {destination} ************\n")
    for name in filenames:
        shutil.move(f"{source}/{config.F_ANNS}/{name}{config.ANN_EXTENSION}", f"{destination}/{config.F_ANNS}")
        shutil.move(f"{source}/{config.F_IMAGES}/{name}{config.IMG_EXTENSION}", f"{destination}/{config.F_IMAGES}")

def makeStandardFoldersAndMovePartialData(folderName, fullDataFolder, output_dir, trainSamplesCount, valSamplesCount, round, UN_DIV_SPLIT=None, basedOn=None):
    if os.path.isdir(f"{config.DATA_DIR}/{folderName}"):
        shutil.rmtree(f"{config.DATA_DIR}/{folderName}")

    os.makedirs(f"{config.DATA_DIR}/{folderName}",                                      exist_ok=True)
    os.makedirs(f"{config.DATA_DIR}/{folderName}/{config.F_VAL}",                       exist_ok=True)
    os.makedirs(f"{config.DATA_DIR}/{folderName}/{config.F_VAL}/{config.F_IMAGES}",     exist_ok=True)
    os.makedirs(f"{config.DATA_DIR}/{folderName}/{config.F_VAL}/{config.F_ANNS}",       exist_ok=True)
    os.makedirs(f"{config.DATA_DIR}/{folderName}/{config.F_TRAIN}",                     exist_ok=True)
    os.makedirs(f"{config.DATA_DIR}/{folderName}/{config.F_TRAIN}/{config.F_IMAGES}",   exist_ok=True)
    os.makedirs(f"{config.DATA_DIR}/{folderName}/{config.F_TRAIN}/{config.F_ANNS}",     exist_ok=True)

    # Moving train dataset files
    if UN_DIV_SPLIT == None: # We pick first files if no split was provided
        # Train files
        pathToGetFilenamesFrom = f"{config.DATA_DIR}/{fullDataFolder}/{config.F_TRAIN}/{config.F_ANNS}"
        files = [f[:-4] for f in os.listdir(pathToGetFilenamesFrom) if os.path.isfile(os.path.join(pathToGetFilenamesFrom, f))]
        trainFilesToMove = sorted(files)[:trainSamplesCount]

        # Val files
        pathToGetFilenamesFrom = f"{config.DATA_DIR}/{fullDataFolder}/{config.F_VAL}/{config.F_ANNS}"
        files = [f[:-4] for f in os.listdir(pathToGetFilenamesFrom) if os.path.isfile(os.path.join(pathToGetFilenamesFrom, f))]
        valFilesToMove = sorted(files)[:valSamplesCount]
    else: 
        trainFilesToMove = getImagesForNextRound(f"{fullDataFolder}/{config.F_TRAIN}", f"{output_dir}/output-{config.F_ROUND_WORKDATA}-{round}", trainSamplesCount, UN_DIV_SPLIT)
        valFilesToMove = getImagesForNextRound(f"{fullDataFolder}/{config.F_VAL}", f"{output_dir}/output-{config.F_ROUND_WORKDATA}-{round}", valSamplesCount, UN_DIV_SPLIT)

    print(f"Will be moving {trainSamplesCount} files to train of folder {folderName}")
    MoveSpecificFiles(f"{config.DATA_DIR}/{fullDataFolder}/{config.F_TRAIN}", f"{config.DATA_DIR}/{folderName}/{config.F_TRAIN}", trainFilesToMove)
    print(f"Will be moving {valSamplesCount} files to val of folder {folderName}")
    MoveSpecificFiles(f"{config.DATA_DIR}/{fullDataFolder}/{config.F_VAL}", f"{config.DATA_DIR}/{folderName}/{config.F_VAL}", valFilesToMove)

# Uncertainty and Diversity Sampling related
def cosineSimilarity(A, B):
    #print(f"Calculating cosine similarity between A:{A} and B:{B}")
    normA = np.linalg.norm(A)
    normB = np.linalg.norm(B)
    if normA == 0 or normB == 0:
        #print(f"Yeah man no idea why are these norms 0")
        return 0

    return np.dot(A, B) / (normA * normB)

def computeCCMS(outputsA, outputsB):
    predictionsNumberA = len(outputsA['instances'].pred_boxes)
    predictionsNumberB = len(outputsB['instances'].pred_boxes)
    similarityMatrix = np.zeros((predictionsNumberA, predictionsNumberB))
    for i in range(predictionsNumberA):
        for j in range(predictionsNumberB):
            if outputsA['instances'].pred_classes[i].item() == outputsB['instances'].pred_classes[j].item():
                boxA_i = outputsA['instances'].pred_boxes[i]
                boxB_j = outputsB['instances'].pred_boxes[j]
                similarity = cosineSimilarity(np.ravel(boxA_i.tensor.cpu().numpy()), np.ravel(boxB_j.tensor.cpu().numpy())) + 1
                similarityMatrix[i, j] = similarity*outputsA['instances'].scores[i].item()
    if np.array_equal(similarityMatrix, np.zeros((predictionsNumberA, predictionsNumberB))):
        return 0
    maxSimilarityAB = similarityMatrix.max(axis=1).sum()/outputsA['instances'].scores.sum()
    maxSimilarityBA = similarityMatrix.max(axis=0).sum()/outputsB['instances'].scores.sum()
    ccms = (maxSimilarityAB + maxSimilarityBA) / 2
    return ccms

def computeCCMSmulti(allOutputs, iStart, iStop, jRange):
    mp.set_sharing_strategy('file_system')
    partialSimilarities = np.zeros((iStop-iStart+1, jRange))
    for i in range(iStart, iStop):
        for j in range(jRange):
            partialSimilarities[i-iStart, j] = computeCCMS(allOutputs[i], allOutputs[j])
    #print(f"computeCCMSpartial with start {iStart} and stop {iStop} returning {partialSimilarities}")
    return partialSimilarities

def getImagesForNextRound(imageFolder, modelFolder, targetNumber, UncertaintyDiversitySplit=0.5):
    # Necessary for multiprocessing to work here, but breaks evaluation, so I make it local
    mp.set_sharing_strategy('file_system')
    mp.set_start_method('spawn', force=True)

    #Split = 1 -> All images chosen by uncertainty, Split = 0 -> all chosen by diversity. in between -> in between xD, linearly
    print(f"Executing getUncertainties with arguments: imageFolder: {imageFolder}, modelFolder: {modelFolder}, targetNumber: {targetNumber}")
    # Inference on images to get their outputs
    with open('modelConfig.pkl', 'rb') as file:
        cfg = pickle.load(file)
    cfg.MODEL.WEIGHTS = f'./{modelFolder}/model_final.pth'
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = config.AL_OBJ_SCORE_THRESHOLD
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.AL_OBJ_SCORE_THRESHOLD
    predictor = DefaultPredictor(cfg)

    classUncertainties = {}
    uncertainties = {}
    allTheOutputs = {}
    for _classNo in range(len(MetadataCatalog.get('train').thing_classes)):   
        classUncertainties[_classNo] = 1
    pathToGetFilenamesFrom = f"{config.DATA_DIR}/{imageFolder}/{config.F_ANNS}"
    print(f"pathToGetFilenamesFrom: {pathToGetFilenamesFrom}")
    images = [f[:-4] for f in os.listdir(pathToGetFilenamesFrom) if os.path.isfile(os.path.join(pathToGetFilenamesFrom, f))]
    totalImages = len(images)
    if (totalImages-targetNumber) < totalImages*0.05:
        print(f"(totalImages-targetNumber) is lower than totalImages*0.05 so we skip")
        return images[:targetNumber]
    print(f"### CALCULATING UNCERTAINTIES FROM {imageFolder} FOLDER")
    print(f"### STARTING INFERENCE ON {len(images)} IMAGES")
    for image in images:
        outputs = predictor(cv2.imread(f"{config.DATA_DIR}/{imageFolder}/{config.F_IMAGES}/{image}{config.IMG_EXTENSION}"))
        outputs['instances'] = outputs['instances'].to("cpu")
        allTheOutputs[image] = outputs
        # Shannon Entropy
        entropy = 0
        for i in range(len(outputs['instances'].pred_boxes)):
            score = outputs['instances'].scores[i].item()
            currentUncertainty = score*math.log2(score)
            classUncertainties[outputs['instances'].pred_classes[i].item()] -= currentUncertainty*0.1
            entropy -= currentUncertainty*classUncertainties[outputs['instances'].pred_classes[i].item()]
        uncertainties[image] = entropy/max(classUncertainties.values())
    
    # scale the values down to the range of 0 to 1
    for key, value in uncertainties.items():
        uncertainties[key] = value/max(uncertainties.values())
    for key, value in classUncertainties.items():
        classUncertainties[key] = value/max(classUncertainties.values())
    print(f"Final class uncertainties for this round: {classUncertainties}")

    NrOfImagesAfterUncertaintyPass = int((totalImages-targetNumber)*(1-UncertaintyDiversitySplit)+targetNumber)
    print(f"We have {totalImages} total images, with split ratio {UncertaintyDiversitySplit} we pick {NrOfImagesAfterUncertaintyPass}\
          images based on uncertainty, and only after the diversity pass we get target {targetNumber} images for next training round")
    UncertaintyChosenImages = probabilisticSelection(uncertainties, NrOfImagesAfterUncertaintyPass)
    
    # Throw away outputs that were not chosen by uncertainty, to not even consider them during diversity to speed up execution
    imagesNotChosenByUncertainty = [image for image in list(allTheOutputs.keys()) if image not in UncertaintyChosenImages]
    print(f"UncertaintyChosenImages count {len(UncertaintyChosenImages)} and images not chosen by uncertainty: {len(imagesNotChosenByUncertainty)} when total is {totalImages}")
    for key in imagesNotChosenByUncertainty:
        del allTheOutputs[key]

    # gotta make it a list, as subsequent code works on indices and a list of outputs
    imageNames, imageOutputs = list(allTheOutputs.keys()), list(allTheOutputs.values())
    # with open("./imageOutputs.pkl", 'wb') as file:
    #     pickle.dump(imageOutputs, file)
    # print(f"Type of element of imageOutputs is: {type(imageOutputs[0])}")

    #print(f"ARE WE TRIPPIN'? \n Imag names: {imageNames}\n\nimageOutputs: {imageOutputs}")
    countOfImagesForDiversity = len(UncertaintyChosenImages)

    if countOfImagesForDiversity == targetNumber: # Meaning that Split is 1, so there is no point in calculating diversity
        return UncertaintyChosenImages
    elif (countOfImagesForDiversity-targetNumber) < countOfImagesForDiversity*0.05:
        # if we'll eliminate less than 5% of images through diversity, dont even bother. It takes so long to calculate
        print(f"(countOfImagesForDiversity-targetNumber) is lower than countOfImagesForDiversity*0.05 so we skip")
        return UncertaintyChosenImages[:targetNumber]
    else:
        numberOfProcesses = min(16, mp.cpu_count()) # Max 16 processes for now
        print(f"Using {numberOfProcesses} processes to calculate CCMS. Available number was {mp.cpu_count()}")
        # 2 times the tasks just in case one process finishes first so it doesnt have to wait
        ranges = splitRangeIntoN(0, countOfImagesForDiversity, int(numberOfProcesses))
        print(f"Created pool ranges: {ranges}")
        argsForTasks = [(imageOutputs, start, stop, countOfImagesForDiversity) for start, stop in ranges]
        print(f"Calculating similarity matrix now with multiple threads")
        startTime = time.time()
        mp.set_sharing_strategy('file_system')
        with mp.Pool(processes=numberOfProcesses) as pool:
            similarityMatrix = np.vstack(pool.starmap(computeCCMSmulti, argsForTasks))
            pool.close()
            pool.join()

        endTime = time.time()
        print(f"### Calculating similarityMatrix took {endTime - startTime} seconds")

        distanceMatrix = np.zeros((countOfImagesForDiversity, countOfImagesForDiversity))
        print(f"So we'll be executing CCMS {countOfImagesForDiversity}*{countOfImagesForDiversity} = {countOfImagesForDiversity*countOfImagesForDiversity} times")
        for i in range(countOfImagesForDiversity):
            if i % 200 == 0:
                print(f"i = {i}/{countOfImagesForDiversity}")
            for j in range(countOfImagesForDiversity):
                if i != j:
                    #similarity = computeCCMS(imageOutputs[i], imageOutputs[j])
                    similarity = similarityMatrix[i][j]
                    distanceMatrix[i, j] = 1/(similarity+0.2)
        # AAAAAAAAAAAAA TRY CHUNKING IT DOWN WITH A MANAGER TO MAKE FEWER TASKS
        #print(f"Resulting distance matrix (10 by 10 cutout): \n{distanceMatrix[:10, :10]}")
        centroids = kmeans(distanceMatrix, targetNumber)
        #print(f"Picked centroids: {centroids}")
    return [imageNames[index] for index in centroids]

# Directly copied from PPAL source code
def k_centroid_greedy(dis_matrix, K):
        N = dis_matrix.shape[0]
        centroids = []
        c = np.random.randint(0, N, (1,))[0]
        centroids.append(c)
        i = 1
        while i < K:
            centroids_diss = dis_matrix[:, centroids].copy()
            centroids_diss = centroids_diss.min(axis=1)
            centroids_diss[centroids] = -1
            new_c = np.argmax(centroids_diss)
            centroids.append(new_c)
            i += 1
        return centroids

# Directly copied from PPAL source code
def kmeans(dis_matrix, K, n_iter=100):
        N = dis_matrix.shape[0]
        centroids = k_centroid_greedy(dis_matrix, K)
        data_indices = np.arange(N)

        assign_dis_records = []
        for _ in range(n_iter):
            centroid_dis = dis_matrix[:, centroids]
            cluster_assign = np.argmin(centroid_dis, axis=1)
            assign_dis = centroid_dis.min(axis=1).sum()
            assign_dis_records.append(assign_dis)

            new_centroids = []
            for i in range(K):
                cluster_i = data_indices[cluster_assign == i]
                assert len(cluster_i) >= 1
                dis_mat_i = dis_matrix[cluster_i][:, cluster_i]
                new_centroid_i = cluster_i[np.argmin(dis_mat_i.sum(axis=1))]
                new_centroids.append(new_centroid_i)
            centroids = np.array(new_centroids)
        return centroids.tolist()

def probabilisticSelection(filenameValueDict, N):

    selectedKeys = set() # So no duplicates by design, instead of checking for duplicates in a list
    bias = 0 # Need bias, as loop was getting stuck trying to pick the last few elements in the last round,
    # when uncertainties are probably super low, so its hard to pick anything
    while len(selectedKeys) < N:
        #print(f"And we're still here bcs len(selectedKeys):{len(selectedKeys)}, N: {N}")
        for key, value in filenameValueDict.items():
            if random.random() < (value + bias*0.01):
                selectedKeys.add(key)
                if len(selectedKeys) == N:
                    break
        bias += 1
    return list(selectedKeys)

def splitRangeIntoN(start, stop, n):
    totalElements = stop - start + 1
    range_size = totalElements // n
    extra_elements = totalElements % n
    ranges = []
    current_start = start
    for i in range(n):
        current_range_size = range_size + (1 if i < extra_elements else 0)
        current_stop = current_start + current_range_size - 1
        ranges.append((current_start, current_stop))
        current_start = current_stop + 1
    return ranges