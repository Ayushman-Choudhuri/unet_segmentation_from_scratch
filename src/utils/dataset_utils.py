import os 

import pandas as pd
import numpy as np


class CityscapesLabelEncoder:
    """
    Class is built using the label mappings available here: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    
    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

                    **** In the CityscapesLabelEncoder class the trainID values of 255 and -1 are replaced by 19 in __init__***

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label

    """
    def __init__(self):
            
            self.label_map_raw = [
                (  "name"                 ,"id", "trainId", "category",     "catId",  "hasInstances","ignoreInEval",        "color"),
                (  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
                (  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
                (  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
                (  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
                (  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
                (  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
                (  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
                (  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
                (  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
                (  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
                (  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
                (  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
                (  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
                (  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
                (  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
                (  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
                (  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
                (  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
                (  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
                (  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
                (  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
                (  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
                (  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
                (  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
                (  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
                (  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
                (  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
                (  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
                (  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
                (  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
                (  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
                (  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
                (  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
                (  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
                (  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
            ]

            # create labels dataframe
            self.cityscapes_labels_df = pd.DataFrame(self.label_map_raw[1:], columns=self.label_map_raw[0])  #generate pandas dataframe 
            self.cityscapes_labels_df.loc[self.cityscapes_labels_df["trainId"].isin([255, -1]), "trainId"] = 19 #replace 255 and -1 in trainId column with 19
            
            self.category_ids = np.arange(self.cityscapes_labels_df["catId"].nunique()) #get the unique category ids
            self.class_ids = self.cityscapes_labels_df["trainId"].unique()
            self.class_ids.sort() # in-place labels ascending sort

    def label2ohe(self , labelid_img:np.ndarray , mode="catId") -> np.ndarray: 
        """
        Converts Image with labelids into a one hot encoded format
        
        labelid_img : The image with labelids after converting it to a numpy array
                      In this numpy array, every pixel would be assigned a id. These ids can be found in the self.label_map_raw list.               
        
        mode : either catID or trainID depending on the basis of which ID, the one hot encoding needs to be done. 

        """
        if not isinstance(labelid_img , np.ndarray):
             
            try: 
                labelid_img = np.array(labelid_img)
            except: 
                 raise ValueError("==> labelid_img must be converted to a np.ndarray datatype before one hot encoding")        
        
        classes = self.category_ids
        
        #get relavent classes for one hot encoding 
        if mode == "trainId": 
            
            classes = self.class_ids

        elif mode == "catId":
             
            classes = self.category_ids
        else: 
            raise ValueError("==> mode for one hot encoding needs to be either catId or trainId ")
        
        if len(classes) == 2:
            classes = [0] 

        #convert all the labels in the labelid_img to the trainId or catId based on mode selection
        for unique in np.unique(labelid_img):
            labelid_img[labelid_img == unique] = self.cityscapes_labels_df.loc[self.cityscapes_labels_df["id"] == unique, mode].values[0]
        labelid_img = labelid_img.astype(int) 

        #generate empty 3D ohe label encoding
        ohe_labels = np.zeros(labelid_img.shape[:2]+ (len(classes),))

        #Fill the ohe label encodings
        for c in classes: 
            y , x = np.where(labelid_img == c)
            ohe_labels[y,x,c]=1
        
        return ohe_labels.astype(int)
    

    def ohe2label(self, ohe_image:np.ndarray) -> np.ndarray:
        """
        This method converts one hot encoded images back to an image with label ids
        """        

        if not isinstance(ohe_image , np.ndarray):
            
            try: 
                labelid_img = np.ndarray(ohe_image)

            except: 
                raise ValueError("==> ohe_label_image must be converted to a numpy array before conversion to labelid image")

        #Make a empty labelid_img 
        labelid_img = np.zeros(ohe_image.shape[:2])

        for ch in range (ohe_image.shape[-1]):
            ys, xs = np.nonzero(ohe_image[:,:,ch])
            labelid_img[ys,xs]= ch

        return labelid_img.astype(int)
    
    def label2color(self , labelid_img:np.ndarray , mode="catId") -> np.ndarray:
        """
        This method converts labelid image to a RGB segmentation image
        
        """
        if not isinstance(labelid_img , np.ndarray):
             
            try: 
                labelid_img = np.array(labelid_img)
            except: 
                 raise ValueError("==> labelid_img must be converted to a np.ndarray datatype before one hot encoding")
                 
        
        color_img = np.zeros(labelid_img.shape[:2]+(3,)).astype(np.uint8) #construct empty RGB image 

        id_list = self.cityscapes_labels_df[mode].tolist()

        for label in id_list: 
            ys , xs = np.where(labelid_img == label)
            color_code = self.cityscapes_labels_df.loc[self.cityscapes_labels_df[mode] == label, "color"].values[0]
            color_img[ys,xs] = np.array(color_code)

        return color_img
    
    def color2label(self, color_img:np.ndarray) -> np.ndarray :
        """
        This method converts a color segmentation map (RGB) into a labelid image.
        In the labelid image, each pixel will have the respective label id corresponding to the rgb value of the color 
        """
        pass