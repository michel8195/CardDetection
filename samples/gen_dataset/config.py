import numpy as np
import itertools
import os

# imgW,imgH: dimensions of the generated dataset images
imgW = 720
imgH = 720


cardW = 60
cardH = 114
cornerXmin = 3 * 4
cornerXmax = 9 * 4
cornerYmin = 3 * 4
cornerYmax = 19 * 4

# We convert the measures from mm to pixels: multiply by an arbitrary factor 'zoom'
zoom = 4
cardW *= zoom
cardH *= zoom
decalX = int((imgW - cardW) * 0.5)
decalY = int((imgH - cardH) * 0.5)

x1 = cornerXmin
y1 = cornerYmin
x2 = cornerXmax
y2 = cornerYmax

refCard = np.array([[0, 0], [cardW, 0], [cardW, cardH], [0, cardH]], dtype=np.float32)
refCardRot = np.array([[cardW, 0], [cardW, cardH], [0, cardH], [0, 0]], dtype=np.float32)

# Define the corners points of each 4 corners

corner1 = [[cornerXmin, cornerYmin], [cornerXmax, cornerYmin], [cornerXmin, cornerYmax], [cornerXmax, cornerYmax]]
corner2 = [[cardW - cornerXmax, cornerYmin], [cardW - cornerXmin, cornerYmin], [cardW - cornerXmax, cornerYmax],
           [cardW - cornerXmin, cornerYmax]]
corner3 = [[cornerXmin, cardH - cornerYmax], [cornerXmax, cardH - cornerYmax], [cornerXmin, cardH - cornerYmin],
           [cornerXmax, cardH - cornerYmin]]
corner4 = [[cardW - cornerXmax, cardH - cornerYmax], [cardW - cornerXmin, cardH - cornerYmax],
           [cardW - cornerXmax, cardH - cornerYmin], [cardW - cornerXmin, cardH - cornerYmin]]


card_suits = ['c']
card_values = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

list_cards = []
for suit, value in itertools.product(card_suits, card_values):
    list_cards.append('{}{}'.format(value, suit))

print(list_cards)
