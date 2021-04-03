from PIL import Image,ImageFilter,ImageDraw, ImageOps

#Read image
im = Image.open('source_img/dtemkin.jpg')
im = im.convert("RGBA")
pixdata = im.load()

draw = ImageDraw.Draw(im)
draw.polygon([(20,10), (200,200), (100,20)], fill = (0,0,0,0))

width, height = im.size
for y in range(height):
    for x in range(width):
        if x < 50:
            pixdata[x, y] = (255, 255, 255, 0)

# img = open("source_img/dtemkin.jpg")
back = Image.new('RGBA', im.size)
back.paste(im)
poly = Image.new('RGBA', (512,512))
pdraw = ImageDraw.Draw(poly)
pdraw.rectangle([(10,10),(100,100)],
          fill=(255,255,255,200))
inverted_poly = ImageOps.invert(poly)
back.paste(poly, (0,0), mask=inverted_poly)
back.show()

im.save("img2.png", "PNG")

im.show()
