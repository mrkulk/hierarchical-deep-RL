import zmq
import time
import sys
import cv2
import numpy as np
import copy
import sys

port = "5555"
if len(sys.argv) > 1:
    port =  sys.argv[1]
    int(port)

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:%s" % port)


class Recognizer:
	def __init__(self):
		self.colors = {'man': [200, 72, 72], 'skull': [236,236,236]}

	def blob_detect(self, img, id):
		mask = np.zeros(np.shape(img))
		mask[:,:,0] = self.colors[id][0];
		mask[:,:,1] = self.colors[id][1];
		mask[:,:,2] = self.colors[id][2];

		diff = img - mask
		indxs = np.where(diff == 0)
		diff[np.where(diff < 0)] = 0
		diff[np.where(diff > 0)] = 0
		diff[indxs] = 255
		mean_y = np.sum(indxs[0]) / np.shape(indxs[0])[0]
		mean_x = np.sum(indxs[1]) / np.shape(indxs[1])[0]
		return (mean_x, mean_y)

	def template_detect(self, img, id):
		template = cv2.imread('templates/' + id + '.png')
		w = np.shape(template)[1]
		h = np.shape(template)[0]
		res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
		threshold = 0.7
		loc = np.where( res >= threshold)
		return loc, w, h

	def get(self, img):
		#detect man
		man_coords = self.blob_detect(img, 'man')
		skull_coords = self.blob_detect(img, 'skull')
		ladder_coords, ladder_w, ladder_h = self.template_detect(img, 'ladder')
		key_coords, key_w, key_h = self.template_detect(img, 'key')
		door_coords, door_w, door_h = self.template_detect(img, 'door')
		return {'man': man_coords, 'skull':skull_coords, 'ladder':ladder_coords, 'key':key_coords, 'door':door_coords, 'ladder_w': ladder_w,
		'ladder_h':ladder_h	, 'key_w':key_w, 'key_h':key_h, 'door_w':door_w, 'door_h':door_h}

	def drawbbox(self, inputim, coords):
		img = copy.deepcopy(inputim)
		for id in {'ladder', 'key', 'door'}:
			for pt in zip(*coords[id][::-1]):
			  cv2.rectangle(img, pt, (pt[0] + coords[id+'_w'], pt[1] + coords[id+'_h']), (0,0,255), 2)
		cv2.rectangle(img, (coords['man'][0] - 5, coords['man'][1] - 5), (coords['man'][0] + 5, coords['man'][1] + 5),  (0,0,255), 2)
		cv2.rectangle(img, (coords['skull'][0] - 5, coords['skull'][1] - 5), (coords['skull'][0] + 5, coords['skull'][1] + 5),  (0,0,255), 2)
		return img


def show(img):
	cv2.imshow('image',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def unit_test():
    rec = Recognizer()
    try:
        img_id = str(sys.argv[1])
    except:
        print 'Using default image 1.png'
        img_id = '1'
    img_rgb = cv2.imread('dump/'+img_id+'.png')
    img_rgb = img_rgb[30:,:,:]
    coords = rec.get(img_rgb)
    img = rec.drawbbox(img_rgb, coords)
    show(img)

rec = Recognizer()
while True:
    #  Wait for next request from client
    message = socket.recv()
    # print "Received request: ", message
    img_rgb = cv2.imread('tmp.png')
    img_rgb = img_rgb[30:,:,:]
    coords = rec.get(img_rgb)
    # img = rec.drawbbox(img_rgb, coords)
    # show(img)  
    socket.send("World from %s" % str(coords))


