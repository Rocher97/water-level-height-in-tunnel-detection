"""
This is the main program.
Designed by Rocher.
"""
import os

import BasicFunctionsDef as BF


def operationsSelected():
    video_file = str(input("请确认视频与代码在同一目录下，并输入视频文件的名称：")) + '.MP4'

    if not os.path.isfile(video_file):
        print('File not found!')

    else:
        mode = input("""
        请选择需要进行的操作
        a：视频帧提取
        b：原始图像裁剪
        c：运行边缘检测算法e
        d：液面高度提取
        e：退出
        输入对应的字母：
        """)

        if mode == "a":
            BF.extract_frames(video_file)
            operationsSelected()

        elif mode == "b":
            coordinates = input("输入需要保留的坐标范围，按x1，x2，y1，y2的顺序输入，以英文逗号分隔，不要留空格\n")
            coordinates = coordinates.split(',')
            coordinates = [int(coordinate) for coordinate in coordinates]
            x1, x2, y1, y2 = coordinates[0], coordinates[1], coordinates[2], coordinates[3]
            BF.crop_frames(x1, x2, y1, y2)
            operationsSelected()

        elif mode == "c":
            BF.cannyDetector()
            remove = input("是(y)否(n)去除图像下半部分的杂乱线条：")
            if remove == "y":
                remove_pixel = int(input("指定去除哪个高度以下（输入对应的像素位置）："))
                BF.removeBottom(remove_pixel)
            operationsSelected()

        elif mode == "d":
            BF.heightLocation(video_file)
            operationsSelected()

        elif mode == "e":
            pass

        else:
            return ValueError

operationsSelected()
