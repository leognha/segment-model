import os
from PIL import Image
import cv2
import PIL.ImageOps
import os.path
from os.path import isfile, isdir, join

import glob
from PIL import ImageEnhance
import numpy as np


def sky2ground(pngfileroot,outdir):
    img = cv2.imread(pngfileroot, 1)
    cv2.imshow('img', img)
    img_shape = img.shape  # 图像大小(565, 650, 3)
    print(img_shape)
    h = img_shape[0]
    w = img_shape[1]
    # 彩色图像转换为灰度图像（3通道变为1通道）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(gray.shape)
    # 最大图像灰度值减去原图像，即可得到反转的图像
    dst = 255 - gray
    cv2.imshow('dst', dst)
    #cv2.imwrite(outdir, dst)
    out_dst = Image.fromarray(dst)
    out_dst.save(os.path.join(outdir, os.path.basename(pngfileroot)))
    cv2.waitKey(0)

def change_name(jpgfile,outdir):
    img = Image.open(jpgfile)
    try:
        img.save(os.path.join(outdir,"1_"+os.path.basename(jpgfile)))
    except Exception as e:
        print(e)


def convertjpg(jpgfile,outdir,width=256,height=256):

    img = Image.open(jpgfile)
    try:
        new_img=img.resize((width,height),Image.BILINEAR)
        new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))
    except Exception as e:
        print(e)

def convertjpgandlight(jpgfile,outdir,width=256,height=256):

    img = Image.open(jpgfile)
    new_image = Image.new('RGB', (512, 256))
    try:
        new_img=img.resize((width,height),Image.BILINEAR)

        enh_bri = ImageEnhance.Brightness(new_img)
        brightness = 2
        image_brightened = enh_bri.enhance(brightness)

        new_image.paste(new_img , (0,0))#函式描述：背景圖片,paste()函式四個變數分別為：起始橫軸座標，起始縱軸座標，橫軸結束座標，縱軸結束座標；
        new_image.paste(new_img, (256, 0, 512, 256))

        new_image.save(os.path.join(outdir,os.path.basename(jpgfile)))
    except Exception as e:
        print(e)

def convertjpganddark(jpgfile,outdir,width=256,height=256):

    img = Image.open(jpgfile)
    new_image = Image.new('RGB', (512, 256))
    try:
        new_img=img.resize((width,height),Image.BILINEAR)

        enh_bri = ImageEnhance.Brightness(new_img)
        brightness = 0.2
        image_brightened = enh_bri.enhance(brightness)

        new_image.paste(image_brightened , (0,0))#函式描述：背景圖片,paste()函式四個變數分別為：起始橫軸座標，起始縱軸座標，橫軸結束座標，縱軸結束座標；
        new_image.paste(image_brightened, (256, 0, 512, 256))

        new_image.save(os.path.join(outdir,os.path.basename(jpgfile)))
    except Exception as e:
        print(e)


def mergejpg_mask(jpgfile, outdir):
    img = Image.open(jpgfile)
    mask = Image.open(root_png)
    #mask = cv2.imread(root_png, 1)
    #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    main_image = Image.open(root + main_image_name)
    new_image = Image.new('RGB', (768, 256))

    try:
        new_image.paste(main_image , (0,0))#函式描述：背景圖片,paste()函式四個變數分別為：起始橫軸座標，起始縱軸座標，橫軸結束座標，縱軸結束座標；
        new_image.paste(img, (256, 0, 512, 256))
        new_image.paste(mask, (512, 0, 768, 256))
        new_image.save(os.path.join(outdir, os.path.basename(jpgfile)))
    except Exception as e:
        print(e)

def mergejpg(jpgfile, outdir):
    img = Image.open(jpgfile)
    new_img = img.resize((256, 256), Image.BILINEAR)

    new_image = Image.new('RGB', (512, 256))
    #main_image = Image.open(r"D:\users\leognha\Desktop\bicyclegan\BicycleGAN-pytorch\Advanced-BicycleGAN\data\oringdataset\204")
    try:
        new_image.paste(new_img , (0,0))#函式描述：背景圖片,paste()函式四個變數分別為：起始橫軸座標，起始縱軸座標，橫軸結束座標，縱軸結束座標；
        new_image.paste(new_img, (256, 0, 512, 256))
        new_image.save(os.path.join(outdir, os.path.basename(jpgfile)))
    except Exception as e:
        print(e)




#判定圖片黑暗程度
def dark_delete(root_jpg):

    for jpgfile in glob.glob(os.path.join(root_jpg)):
        img = cv2.imread(jpgfile, 1)


        #img = cv2.imread(r"D:\users\leognha\Desktop\bicyclegan\BicycleGAN-pytorch\Advanced-BicycleGAN\data\sky_finder\home\mihail\mypages\rpmihail\skyfinder\images\65\20130619_110303.jpg", 1)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        # 把图片转换为灰度图
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 获取灰度图矩阵的行数和列数
        r, c = gray_img.shape[:2];
        dark_sum = 0;  # 偏暗的像素 初始化为0个
        dark_prop = 0;  # 偏暗像素所占比例初始化为0
        piexs_sum = r * c;  # 整个弧度图的像素个数为r*c

        # 遍历灰度图的所有像素
        for row in gray_img:
            for colum in row:
                if colum < 40:  # 人为设置的超参数,表示0~39的灰度值为暗
                    dark_sum += 1
        dark_prop = dark_sum / (piexs_sum)
        print("dark_sum:" + str(dark_sum))
        print("piexs_sum:" + str(piexs_sum))
        print("dark_prop=dark_sum/piexs_sum:" + str(dark_prop))

        if dark_prop >= 0.43:  # 人为设置的超参数:表示若偏暗像素所占比例超过0.78,则这张图被认为整体环境黑暗的图片
            try:
                print("Dark")
                os.remove(os.path.join(jpgfile))
            except OSError as e:
                print(e)
            else:
                print("File is deleted successfully")
            #print(pic_path + " is dark!");
            #cv2.imwrite("../DarkPicDir/" + pic, img);  # 把被认为黑暗的图片保存

def dark_detect(jpgfile):
    img = cv2.imread(jpgfile, 1)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    # 把图片转换为灰度图
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 获取灰度图矩阵的行数和列数
    r, c = gray_img.shape[:2];
    dark_sum = 0;  # 偏暗的像素 初始化为0个
    dark_prop = 0;  # 偏暗像素所占比例初始化为0
    piexs_sum = r * c;  # 整个弧度图的像素个数为r*c

        # 遍历灰度图的所有像素
    for row in gray_img:
        for colum in row:
            if colum < 40:  # 人为设置的超参数,表示0~39的灰度值为暗
                dark_sum += 1
    dark_prop = dark_sum / (piexs_sum)

    return dark_prop

def datamaker(root_jpg, traindir, valdir, pngfileroot, root_baseline):



    
    print(root_jpg)
    print(traindir)
    print(valdir)
    print(pngfileroot)

    j = 0
    mask = cv2.imread(pngfileroot, 1)
    #cv2.imshow('img', mask)
    img_shape = mask.shape  # 图像大小(565, 650, 3)
    print(img_shape)
    h = img_shape[0]
    w = img_shape[1]
    # 彩色图像转换为灰度图像（3通道变为1通道）
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    print(gray.shape)
    # 最大图像灰度值减去原图像，即可得到反转的图像
    dst = 255 - gray
    #cv2.imshow('dst', dst)
    out_dst = Image.fromarray(dst)
    out_dst = out_dst.resize((256, 256), Image.BILINEAR)

    #out_dst.save(os.path.join(valdir, os.path.basename(pngfileroot)))
    baseline = dark_detect(root_baseline)
    #baseline = 1
    for jpgfile in glob.glob(os.path.join(root_jpg)):
        j = j + 1
        print(j  ,jpgfile)
        #Input image, change in every 50
        if(j%50 == 1 and  dark_detect(jpgfile) <= 0.25):

            main_image = Image.open(jpgfile)
            main_image = main_image.resize((256, 256), Image.BILINEAR)
            print("YO")
        if(j%50 == 1 and  dark_detect(jpgfile) >= 0.25):
            j = j - 1
            continue

        if(dark_detect(jpgfile) > baseline):
            j = j - 1
            continue

        #val
        if (j == 801):
            new_image = Image.new('RGB', (768, 256))
            img = Image.open(jpgfile)
            img = img.resize((256, 256), Image.BILINEAR)

            try:
                new_image.paste(main_image, (0, 0))  # 函式描述：背景圖片,paste()函式四個變數分別為：起始橫軸座標，起始縱軸座標，橫軸結束座標，縱軸結束座標；
                new_image.paste(img, (256, 0, 512, 256))
                new_image.paste(out_dst, (512, 0, 768, 256))
                new_image.save(os.path.join(valdir, os.path.basename(jpgfile)))

            except Exception as e:
                print(e)
            break

        #train
        new_image = Image.new('RGB', (768, 256))
        img = Image.open(jpgfile)
        img = img.resize((256, 256), Image.BILINEAR)
        try:
            new_image.paste(main_image, (0, 0))  # 函式描述：背景圖片,paste()函式四個變數分別為：起始橫軸座標，起始縱軸座標，橫軸結束座標，縱軸結束座標；
            new_image.paste(img, (256, 0, 512, 256))
            new_image.paste(out_dst, (512, 0, 768, 256))
            new_image.save(os.path.join(traindir, os.path.basename(jpgfile)))

        except Exception as e:
            print(e)

    print("over")

def make_mask_pair(path,traindir,maskdir):
    files = os.listdir(path)
    k=0
    sum=0

    for f in files:
        print(f)
        # 產生檔案的絕對路徑
        fullpath = join(path, f)
        # 判斷 fullpath 是檔案還是資料夾
        j = 0
        for jpgfile in glob.glob(os.path.join(fullpath + "/*.jpg")):
            try:
                img = Image.open(jpgfile)
                for pngfile in glob.glob(os.path.join(fullpath + "/*.png")):
                    mask = Image.open(pngfile)

                img=img.resize((300,300),Image.BILINEAR)
                mask = mask.resize((300, 300), Image.BILINEAR)
                img.save(os.path.join(traindir, f + "_" + os.path.basename(jpgfile)))
                mask.save(os.path.join(maskdir, f + "_" + os.path.splitext(os.path.basename(jpgfile))[0] + ".png"))
                j = j + 1
            except:
                continue

        sum=sum+j
        k=k+1
        print(k,j,sum)
        print("over")




file_number = "/65"
baseline_name = "/20140722_100242"
root = r"D:\users\leognha\Desktop\bicyclegan\BicycleGAN-pytorch\Advanced-BicycleGAN\data\sky_finder\home\mihail\mypages\rpmihail\skyfinder\images"   + file_number
root_jpg = root + "\*.jpg"
root_png = root + file_number + ".png"

root_baseline = root + baseline_name + ".jpg"

#outdur = r"D:\users\leognha\Desktop\bicyclegan\BicycleGAN-pytorch\Advanced-BicycleGAN\data\oringdataset" + file_number
#main_image_name = "/20130210_210311.jpg"
#traindir_jpg = outdur + "/*.jpg"

traindir = "/home/leognha/Desktop/seg-model/MedicalImage_Project02_Segmentation/data_test/imgs"
maskdir = "/home/leognha/Desktop/seg-model/MedicalImage_Project02_Segmentation/data_test/masks"
#valdir = "D:\users\leognha\Desktop\bicyclegan\BicycleGAN-pytorch\Advanced-BicycleGAN-mask\data\edges2shoes\val"
path = "/home/leognha/Desktop/data/sky_finder"

make_mask_pair(path, traindir, maskdir)

#files = os.listdir(r"D:\users\leognha\Desktop\bicyclegan\BicycleGAN-pytorch\Advanced-BicycleGAN\data\sky_finder\home\mihail\mypages\rpmihail\skyfinder\images")

    #print(root) #當前目錄路徑
#print(files.shape) #當前路徑下所有子目錄
    #print(files) #當前路徑下所有非目錄子檔案


#image = Image.open(r"D:\users\leognha\Desktop\bicyclegan\seg-model\MI_hw2\MedicalImage_Project02_Segmentation\origin_data\imgs\0131_06_1.png")
#image = np.array(image)
#print(image.shape)
#dark_delete(root_jpg)
#datamaker(root_jpg, traindir, valdir, root_png, root_baseline) #製作dataset

#print(dark_detect(r"D:\users\leognha\Desktop\bicyclegan\BicycleGAN-pytorch\Advanced-BicycleGAN\data\sky_finder\home\mihail\mypages\rpmihail\skyfinder\images\19306\20130311_092333.jpg"))
#print(dark_detect(r"D:\users\leognha\Desktop\bicyclegan\BicycleGAN-pytorch\Advanced-BicycleGAN\data\sky_finder\home\mihail\mypages\rpmihail\skyfinder\images\19306\20130314_165328.jpg"))

#print(os.path.splitext("0131_06_1.png")[0])
