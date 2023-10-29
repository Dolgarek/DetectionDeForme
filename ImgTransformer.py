from pathlib import Path

from PIL import Image, ImageDraw, ImageOps, ImageFilter
import random
import os
import shutil
import uuid

from Shape import Shape


def create_img(number_of_img=1):
    #try:
    #    shutil.rmtree('dataset')
    #except Exception as e:
    #    print(f'Failed to delete directory: {e}')
    #os.mkdir('dataset')
    for x in range(number_of_img):
        new = Image.new(mode="RGBA", size=(500, 500), color="navy")
        draw = ImageDraw.Draw(new)
        grayscale = ImageOps
        figures = random.randint(1, 5)
        directorySaveA = 0
        directorySaveB = 0
        directorySaveC = 0
        for i in range(figures):
            n = random.randint(1, 3)
            x = random.randint(1, 480)
            y = random.randint(1, 480)
            if n == 1:
                draw.rectangle((x, y, x + random.randint(10, 480), y + random.randint(10, 480)), outline='teal',
                               fill='orange', width=1)
                directorySaveA = 1
            elif n == 2:
                radius = random.randint(10, 480)
                draw.ellipse((x, y, x + radius, y + radius), fill='red', outline='black')
                directorySaveB = 1
            else:
                a = random.randint(1, 480)
                b = random.randint(1, 480)
                c = random.randint(1, 480)
                draw.regular_polygon((c, b, a), 3, fill=(100, 100, 255, 255), outline="orange")
                directorySaveC = 1

            # shapes.append(Shape(n, random.randint(30,470), random.randint(30,470)))
            # draw.rectangle((x,y,x+random.randint(10,69),y+random.randint(10,69)), outline='teal', fill='orange', width=1)

        # new.show()
        # new.save('./png.png')

        subfolder = ''

        if directorySaveB == 1:
            if directorySaveA == 1:
                if directorySaveC == 1:
                    subfolder = 'circle_rectangle_triangle'
                else:
                    subfolder = 'circle_rectangle'
            elif directorySaveC == 1:
                subfolder = 'circle_triangle'
            else:
                subfolder = 'circle'
        elif directorySaveA == 1:
            if directorySaveC == 1:
                subfolder = 'rectangle_triangle'
            else:
                subfolder = 'rectangle'
        elif directorySaveC == 1:
            subfolder = 'triangle'

        directory_path = Path('dataset/' + subfolder)
        file_count = len(list(directory_path.glob('*')))


        file_count = file_count + 1

        new = grayscale.grayscale(new)
        # new.show()
        filtered = new.filter(ImageFilter.FIND_EDGES)
        #filtered.show()
        filtered.save('dataset/' + subfolder + '/' + subfolder + str(file_count) + '.png')
        print(file_count)
        print('\n')
