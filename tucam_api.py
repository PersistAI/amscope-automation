#!/usr/bin/env python
# coding: utf-8
'''
Created on 2024-01-04
@author:fdy
'''

import ctypes
from ctypes import *
from TUCam import *
from enum import Enum
import time
import os
from datetime import datetime
import json

RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}


#THIS NEEDS TO BE CHANGED BASED ON THE DEVICE BEING USED
with open("config.json", "r") as f: 
    config = json.load(f)

    home = os.path.expanduser("~")
    dirty_image_path = os.path.join(home, config["image_path"])

    image_path = os.path.normpath(dirty_image_path)

save_dir = image_path #save this to image folder so it can be analyzed

class Tucam():
    def __init__(self):

        self.Path = './'
        self.TUCAMINIT = TUCAM_INIT(0, self.Path.encode('utf-8'))
        self.TUCAMOPEN = TUCAM_OPEN(0, 0)
        TUCAM_Api_Init(pointer(self.TUCAMINIT), 5000)
        print(self.TUCAMINIT.uiCamCount)
        print(self.TUCAMINIT.pstrConfigPath)
        print('Connect %d camera' %self.TUCAMINIT.uiCamCount)

    def OpenCamera(self, Idx):

        if  Idx >= self.TUCAMINIT.uiCamCount:
            return

        self.TUCAMOPEN = TUCAM_OPEN(Idx, 0)

        TUCAM_Dev_Open(pointer(self.TUCAMOPEN))

        if 0 == self.TUCAMOPEN.hIdxTUCam:
            print('Open the camera failure!')
            return
        else:
            print('Open the camera success!')

    def CloseCamera(self):
        if 0 != self.TUCAMOPEN.hIdxTUCam:
            TUCAM_Dev_Close(self.TUCAMOPEN.hIdxTUCam)
        print('Close the camera success')

    def UnInitApi(self):
        TUCAM_Api_Uninit()

    def SaveImageData(self):
        m_fs = TUCAM_FILE_SAVE()
        m_frame = TUCAM_FRAME()
        m_format = TUIMG_FORMATS
        m_frformat = TUFRM_FORMATS
        m_capmode = TUCAM_CAPTURE_MODES

        m_frame.pBuffer = 0
        m_frame.ucFormatGet = m_frformat.TUFRM_FMT_USUAl.value
        m_frame.uiRsdSize = 1

        m_fs.nSaveFmt = m_format.TUFMT_TIF.value

        TUCAM_Buf_Alloc(self.TUCAMOPEN.hIdxTUCam, pointer(m_frame))
        TUCAM_Cap_Start(self.TUCAMOPEN.hIdxTUCam, m_capmode.TUCCM_SEQUENCE.value)

        try:
            result = TUCAM_Buf_WaitForFrame(self.TUCAMOPEN.hIdxTUCam, pointer(m_frame), 1000)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            ImgName = os.path.join(save_dir, f'{timestamp}.tif')
            m_fs.pFrame = pointer(m_frame)
            m_fs.pstrSavePath = ImgName.encode('utf-8')
            TUCAM_File_SaveImage(self.TUCAMOPEN.hIdxTUCam, m_fs)
            print('Save the image data success, the path is %#s'%ImgName)
            
        except Exception:
            print('Grab the frame failure, index number is %#d')

        print("Doing TUCAM_Buf_AbortWait")
        TUCAM_Buf_AbortWait(self.TUCAMOPEN.hIdxTUCam)
        print("Doing TUCAM_Cap_Stop")
        TUCAM_Cap_Stop(self.TUCAMOPEN.hIdxTUCam) #this is what is getting caught up
        print("Doing TUCAM_Buf_Release")
        TUCAM_Buf_Release(self.TUCAMOPEN.hIdxTUCam)

        print("Done taking the pic :)")

if __name__ == '__main__':
    demo = Tucam()
    demo.OpenCamera(0)
    if demo.TUCAMOPEN.hIdxTUCam != 0:
        demo.SaveImageData()
        demo.CloseCamera()
    demo.UnInitApi()