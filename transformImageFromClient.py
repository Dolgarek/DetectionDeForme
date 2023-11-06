from PIL import Image, ImageDraw, ImageOps, ImageFilter
import os
import uuid


def transform_image_from_client():
    grayscale = ImageOps
    new_image = Image.open("/Users/sorenmarcelino/Documents/TraitementDuDocumentAPI/uploaded_image.png")
    new = grayscale.grayscale(new_image)
    # new.show()
    filtered = new.filter(ImageFilter.FIND_EDGES)
    # filtered.show()
    filtered.save('img/' + uuid.uuid4().hex + '.png')